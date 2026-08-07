"""Robust GPU shape registration -- the occlusion-tolerant core of the tracker.

The problem
-----------
Fit a known outline (from your DXF) to each frame, allowing it to translate,
rotate, and stretch independently along its width and length axes, while some
unknown fraction of the robot is hidden behind an obstacle.

The method
----------
Chamfer matching against a distance transform, optimized with a *bounded*
robust kernel.

For each frame we build ``D(x,y)`` = distance in pixels from any point to the
nearest robot silhouette edge. The template is a set of M points; we search
over a 5-parameter pose ``(tx, ty, theta, sx, sy)`` for the placement that puts
template points closest to real edges.

The bounded kernel is what makes occlusion work. Using

    rho(d) = d^2 / (d^2 + tau^2)          (Geman-McClure)

a template point sitting on a real edge contributes ~0, and a template point
stranded in the middle of an obstacle contributes at most 1 -- and, crucially,
its *gradient goes to zero* as d grows. Occluded points therefore stop pulling
on the solution rather than dragging the whole fit off the robot, which is
exactly what a plain least-squares chamfer cost would do. The visible portion
plus the rigidity of the CAD outline determines the answer.

Why this is a good fit for a GPU
--------------------------------
The cost is a ``grid_sample`` over M points, and K independent pose hypotheses
can be evaluated as one batched tensor op. Running 64 restarts costs almost the
same wall-clock as running one, which is what makes the multi-start search that
guarantees robustness affordable in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .cad import Template, signed_distance_grid
from .gpu import Device


@dataclass
class FitConfig:
    tau_px: float = 12.0        # robust kernel scale at capture; annealed down
    tau_final_px: float = 2.5   # kernel scale at convergence, sets final precision
    inlier_px: float = 4.0      # a template point closer than this counts as matched
    n_restarts: int = 64
    # Restarts once the tracker is locked on. The multi-start search exists to
    # find the robot from a cold seed; on a warm frame every hypothesis begins a
    # fraction of a pixel from the answer, and most of them are re-deriving what
    # the first one already has. A cold frame -- the first one, and any recovery
    # after a lost stretch -- still gets the full count.
    n_restarts_warm: int = 12
    iters: int = 150
    iters_warm: int = 60        # cheaper schedule once tracking is locked on
    lr: float = 0.05
    # Stop once the best hypothesis stops improving. The annealing schedule is
    # sized for the worst case -- a cold start from a bad seed -- but a warm
    # frame in a locked-on sequence converges in a fraction of it, and the
    # remaining iterations only re-confirm the same answer.
    early_stop: bool = True
    early_stop_check: int = 10       # iterations between checks
    early_stop_tol: float = 1e-4     # relative improvement below this is done
    early_stop_patience: int = 2     # consecutive quiet checks before stopping
    max_scale_change: float = 0.60   # sx, sy constrained to [1-x, 1+x] of nominal
    allow_flip: bool = True
    recovery_conf: float = 0.55      # below this, re-seed from the mask instead of prev pose
    # Temporal prior. Muscle contraction is continuous, so a large frame-to-frame
    # scale jump is far more likely to be a fitting artifact than real biology.
    # This is what stops an occluded frame from reporting a nonsense size.
    scale_prior_weight: float = 0.35
    # Expressed as a strain *rate* rather than a per-frame fraction. A per-frame
    # limit would silently mean different physics at 30, 60 and 120 Hz -- and
    # would damp genuine contraction hardest on the slowest recording, exactly
    # where the per-frame change is largest.
    scale_prior_rate_per_s: float = 5.0
    # Asymmetric containment term. See ShapeFitter.fit for why this is the piece
    # that makes occluded fitting actually work.
    contain_weight: float = 1.0
    contain_offset_px: float = 6.0
    # Backward (coverage) term: observed robot pixels that fall outside the
    # fitted outline. Occlusion only removes pixels, so this stays valid.
    coverage_weight: float = 3.0
    coverage_points: int = 1024
    coverage_tau_mm: float = 1.5
    # Weight of the drawing's interior structure relative to its silhouette.
    #
    # The two are combined as a weighted mean of their *own* means, not by
    # pooling the points, so the balance does not drift with how many points
    # each happens to carry. At 1.0 the outer boundary and the interior each
    # contribute half.
    feature_weight: float = 1.0

    # --- anisotropy: width is a constant, length is the measurement ---------
    #
    # These two axes are not the same kind of quantity and constraining them
    # alike is a modelling error. The robot's width is a property of the mould:
    # it is what the drawing says it is, it does not participate in
    # contraction, and any frame where the fitted width departs from the
    # drawing is a frame where segmentation went wrong -- not a frame where the
    # robot got wider. Its length is the opposite: it is the thing being
    # measured, it shortens and relaxes continuously, and pinning it would
    # destroy the signal.
    #
    # So width is held near nominal by a quadratic, and length is left free
    # downward and capped upward. The cap is the useful half: relaxed length is
    # bounded above by the drawing, so an outline that has stretched past it has
    # certainly latched onto something that is not the robot, and that failure
    # is otherwise invisible -- a too-long fit still puts most of its points on
    # real edges and still reports high confidence.
    #
    # ``width_weight`` is in units of the chamfer loss, which saturates at 1, so
    # a value of 2 means "one sigma of width error costs twice a completely
    # wrong outline". That is meant to be dominant.
    width_weight: float = 2.0
    width_tol: float = 0.04          # sigma of the width hold, fraction of nominal
    width_max_change: float = 0.15   # hard clamp on sx, fraction of nominal
    # Noise allowance on the length ceiling, in pixels of the fitted outline.
    # Not zero: the edge of a real mask moves by a pixel or so frame to frame,
    # and a hard equality would clip genuine relaxation to whichever frame
    # happened to segment tightest.
    length_overshoot_px: float = 3.0
    # Frames used to learn the reference length before the ceiling is enforced.
    # Skipped entirely when the outline was placed by hand, since the placement
    # already states what the drawing's size is on this footage.
    length_ref_frames: int = 45
    length_ref_quantile: float = 0.90

    # --- work window -------------------------------------------------------
    # Building the distance field, the soft mask and the interior edge map are
    # whole-frame OpenCV operations, and at 1080p they cost more than the
    # optimizer they feed. None of them is needed more than a body length away
    # from where the robot was on the previous frame. Restricted to that window
    # they are roughly ten times cheaper, and the result inside the window is
    # identical. Only applied on warm frames -- a cold seed genuinely may be
    # anywhere in the picture.
    use_window: bool = True
    window_margin_px: float = 48.0
    window_margin_frac: float = 0.35   # of the body's own extent


@dataclass
class Pose:
    tx: float
    ty: float
    theta: float
    sx: float
    sy: float
    cost: float
    confidence: float           # silhouette inlier fraction x containment x coverage
    # Fraction of interior-feature points landing on an observed edge. Reported
    # separately rather than folded into ``confidence`` so the meaning of the
    # confidence gate does not change when a drawing happens to have features.
    feature_fit: float = float("nan")

    def as_array(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.theta, self.sx, self.sy], np.float32)


def interior_edges(signal: np.ndarray, mask: np.ndarray,
                   dilate_px: int = 9, low: int = 40, high: int = 110) -> np.ndarray:
    """Edges *inside* the robot, from the continuous image rather than the mask.

    The mask is a binary decision, so it has exactly one edge: the silhouette.
    Threshold it aggressively and the interior becomes a solid blob with no
    internal structure at all -- measured on the reference clip, zero enclosed
    holes at any morphology setting. The drawing's holes and beams then have
    nothing to match against.

    The underlying image still shows them. Taking Canny edges of the
    color-distance surface recovers the internal boundaries the threshold threw
    away, which is what makes the drawing's interior usable as extra evidence.

    Edges are restricted to the mask's neighbourhood: medium texture and the
    dish rim are strong edges too, and letting them into the distance field
    would give the template somewhere wrong to lock onto.
    """
    s = signal
    if s.dtype != np.uint8:
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        s = np.zeros_like(s, np.uint8) if hi <= lo else \
            ((s - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    s = cv2.GaussianBlur(s, (0, 0), 1.2)
    e = cv2.Canny(s, low, high)
    if dilate_px > 1:
        near = cv2.dilate((mask > 0).astype(np.uint8),
                          np.ones((dilate_px, dilate_px), np.uint8))
        e = e * near
    return e


def distance_field(mask: np.ndarray, dev: Device,
                   extra_edges: np.ndarray | None = None) -> torch.Tensor:
    """Euclidean distance to the nearest edge, as a (1,1,H,W) tensor.

    ``DIST_MASK_PRECISE`` rather than the 3x3 approximation: the 3x3 chamfer has
    up to ~2% radial error, which is a real bias when you are chasing sub-pixel
    contraction amplitudes.

    ``extra_edges`` folds interior image edges in beside the silhouette, so
    template points on holes and beams have something to match.
    """
    m = (mask > 0).astype(np.uint8)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if extra_edges is not None:
        edge = np.maximum(edge, (extra_edges > 0).astype(np.uint8))
    if edge.max() == 0:
        d = np.full(m.shape, 1e4, np.float32)
    else:
        d = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2,
                                  cv2.DIST_MASK_PRECISE).astype(np.float32)
    return torch.from_numpy(d).to(dev.torch_device)[None, None]


def _sample(D: torch.Tensor, pts: torch.Tensor,
            origin: torch.Tensor | None = None) -> torch.Tensor:
    """Bilinearly sample D at (K,M,2) pixel coordinates -> (K,M) distances.

    ``origin`` is where D's top-left corner sits in the full frame, for when D
    covers only a window around the robot. Subtracting it here rather than
    offsetting the poses keeps every pose in the image's own coordinates, which
    is what everything downstream -- the CSV, the overlay, the plots -- expects.
    """
    K, M, _ = pts.shape
    H, W = D.shape[-2:]
    if origin is not None:
        pts = pts - origin
    gx = 2.0 * pts[..., 0] / (W - 1) - 1.0
    gy = 2.0 * pts[..., 1] / (H - 1) - 1.0
    # One batch of K*M points against the single real image, rather than K
    # broadcast copies of it.
    #
    # ``D.expand(K, 1, H, W)`` is a stride-0 view: K batch entries all pointing
    # at the same memory. CUDA and CPU handle that; Metal has repeatedly not,
    # and a grid_sample that silently reads the wrong batch entry produces a
    # fit that converges confidently onto nothing in particular -- which is
    # what "tracks on Windows, wanders on the Mac" looked like. Flattening the
    # points into one batch removes the broadcast entirely, is arithmetically
    # identical, and is marginally faster everywhere because there is no
    # K-way indexing to do.
    grid = torch.stack([gx, gy], dim=-1).reshape(1, K * M, 1, 2)
    # padding_mode='border' matters: zero padding would make points that wander
    # off-image look like perfect matches and create phantom optima.
    out = F.grid_sample(D, grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.reshape(K, M)


def soft_mask(mask: np.ndarray, dev: Device, sigma: float = 2.0) -> torch.Tensor:
    """Blurred mask indicator, for the containment term. Blur gives gradients."""
    m = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), sigma)
    return torch.from_numpy(m).to(dev.torch_device)[None, None]


def _rotate(v: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    """Rotate (M,2) vectors by K angles -> (K,M,2)."""
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    x, y = v[None, :, 0], v[None, :, 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def _transform(tpl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply K poses to the M template points. tpl (M,2), p (K,5) -> (K,M,2).

    Scale is applied in the template's own frame *before* rotation, so sx and sy
    stay bound to the robot's width and length axes no matter how it is oriented
    in the image.
    """
    tx, ty, th, sx, sy = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
    x = tpl[None, :, 0] * sx[:, None]
    y = tpl[None, :, 1] * sy[:, None]
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    return torch.stack([c * x - s * y + tx[:, None],
                        s * x + c * y + ty[:, None]], dim=-1)


def seed_from_mask(mask: np.ndarray, tpl: Template) -> tuple[float, float, float, float, float]:
    """Closed-form initial guess from the mask's own moments.

    Matches centroid to centroid and the template's long axis to the mask's
    principal axis, with scales from the projected extents. Good enough that the
    optimizer usually only has to polish it.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 10:
        return 0.0, 0.0, 0.0, 1.0, 1.0
    pts = np.stack([xs, ys], 1).astype(np.float64)
    cen = pts.mean(0)
    cov = np.cov((pts - cen).T)
    w, v = np.linalg.eigh(cov)
    major = v[:, np.argmax(w)]
    theta = float(np.arctan2(major[1], major[0]) - np.pi / 2)
    proj = (pts - cen) @ v
    ext_minor, ext_major = np.ptp(proj[:, np.argmin(w)]), np.ptp(proj[:, np.argmax(w)])
    sx = ext_minor / max(tpl.width_mm, 1e-6)
    sy = ext_major / max(tpl.length_mm, 1e-6)
    return float(cen[0]), float(cen[1]), theta, float(sx), float(sy)


class ShapeFitter:
    """Fits a CAD template to a sequence of masks, tracking pose frame to frame."""

    def __init__(self, tpl: Template, cfg: FitConfig, dev: Device, dt: float = 1 / 30,
                 seed_pose: np.ndarray | None = None):
        """``seed_pose`` is an optional ``(tx, ty, theta, sx, sy)`` placed by hand.

        Automatic seeding uses the mask's own moments, which is the right answer
        when the robot is the only moving thing in frame. It is the wrong answer
        when there are two robots, a stray reflection, or a tether that segments
        as part of the body: the moments describe whatever the mask contains, and
        the optimizer then converges confidently onto the wrong object. A pose
        placed by hand in the preview says which one is the target.

        It is kept as a standing candidate rather than used only for frame zero,
        because the case that needs it most is recovery -- after a lost stretch
        the tracker re-seeds from the mask, which is exactly the moment it can
        jump to the decoy.
        """
        self.tpl, self.cfg, self.dev, self.dt = tpl, cfg, dev, dt
        self.seed_pose = None if seed_pose is None else np.asarray(seed_pose, np.float32)
        self.pts = torch.from_numpy(tpl.points).to(dev.torch_device)
        self.nrm = torch.from_numpy(tpl.normals).to(dev.torch_device)
        # Interior structure, matched against observed edges but excluded from
        # the containment term. "Just outside this edge is background" holds for
        # the silhouette and fails for an internal edge with material on both
        # sides, so applying it there would push the fit away from a correct
        # placement.
        self.feat = (torch.from_numpy(tpl.feature_points).to(dev.torch_device)
                     if getattr(tpl, "feature_points", None) is not None
                     and len(tpl.feature_points) else None)
        sdf, origin, spacing = signed_distance_grid(tpl.points)
        self.sdf = torch.from_numpy(sdf).to(dev.torch_device)[None, None]
        self.sdf_origin = torch.from_numpy(origin).to(dev.torch_device)
        self.sdf_spacing = spacing
        self.sdf_res = sdf.shape[0]
        self.prev: np.ndarray | None = None
        self._base_scale: float | None = None
        self._rng = np.random.default_rng(1)
        # Reference scales for the anisotropy terms, and the samples they are
        # learned from. A hand placement states both outright. Otherwise they
        # are measured over the first few dozen confident fits, which run
        # unconstrained -- the reference has to come from the footage, not from
        # frame zero's automatic seed. That seed is one moment estimate on one
        # mask and carries whatever bias the morphology gave it; anchoring the
        # width to it locked that bias in for the whole clip and moved the
        # derived calibration by a percent.
        self._len_ref: float | None = None
        self._wid_ref: float | None = None
        self._ref_samples: list[tuple[float, float]] = []
        if self.seed_pose is not None:
            # Anchor the scale limits to the hand-placed size. Deriving them from
            # the first automatic seed instead would bound the search around
            # whatever the mask happened to look like on frame zero -- which,
            # under occlusion, is the one frame you least want to trust.
            self._base_scale = (float(self.seed_pose[3]), float(self.seed_pose[4]))
            self._wid_ref = float(self.seed_pose[3])
            self._len_ref = float(self.seed_pose[4])

    # ---- window ----------------------------------------------------------

    def _window(self, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """Box around the previous pose to do the per-frame image work in.

        None means "the whole frame", which is what a cold or recovering frame
        gets: with no previous pose there is no reason to believe the robot is
        anywhere in particular, and guessing wrong would be a lost track rather
        than a slow one.
        """
        if not self.cfg.use_window or self.prev is None:
            return None
        h, w = shape
        tx, ty, th, sx, sy = (float(v) for v in self.prev)
        pts = self.tpl.points
        x = pts[:, 0] * sx
        y = pts[:, 1] * sy
        c, s = np.cos(th), np.sin(th)
        X = c * x - s * y + tx
        Y = s * x + c * y + ty
        extent = max(float(np.ptp(X)), float(np.ptp(Y)), 1.0)
        m = max(self.cfg.window_margin_px, self.cfg.window_margin_frac * extent)
        x0 = max(0, int(np.floor(X.min() - m)))
        y0 = max(0, int(np.floor(Y.min() - m)))
        x1 = min(w, int(np.ceil(X.max() + m)))
        y1 = min(h, int(np.ceil(Y.max() + m)))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        # Nothing gained if the window is most of the frame anyway, and a full
        # frame keeps the distance field exact at its borders.
        if (x1 - x0) * (y1 - y0) > 0.55 * w * h:
            return None
        return x0, y0, x1, y1

    def _length_ceiling(self, by: float) -> float:
        """Upper bound on the length scale, in template units.

        During the learning window this is the loose global limit; afterwards it
        is the learned relaxed length plus the pixel noise allowance.
        """
        cfg = self.cfg
        if self._len_ref is None:
            return by * (1 + cfg.max_scale_change)
        return self._len_ref + cfg.length_overshoot_px / max(self.tpl.length_mm, 1e-6)

    def _note_scales(self, sx: float, sy: float, conf: float) -> None:
        """Accumulate the learning window, and close it once it is full."""
        if self._len_ref is not None or conf < self.cfg.recovery_conf:
            return
        self._ref_samples.append((float(sx), float(sy)))
        if len(self._ref_samples) < self.cfg.length_ref_frames:
            return
        arr = np.asarray(self._ref_samples, float)
        # Median for the width, because it is one number the whole clip shares
        # and the median is the robust estimate of it. High quantile for the
        # length, because the quantity wanted is not the typical length but the
        # relaxed one -- the ceiling, not the centre.
        self._wid_ref = float(np.median(arr[:, 0]))
        self._len_ref = float(np.quantile(arr[:, 1], self.cfg.length_ref_quantile))

    def _coverage(self, obs: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Penalise observed robot pixels lying outside the fitted outline.

        Works by inverse-transforming the observed points into template space
        and reading the precomputed template SDF, which is far cheaper than
        testing point-in-polygon against a deforming outline, and is smooth
        enough to backpropagate through.
        """
        K = p.shape[0]
        tx, ty, th, sx, sy = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
        v = obs[None] - torch.stack([tx, ty], 1)[:, None, :]
        c, s = torch.cos(-th)[:, None], torch.sin(-th)[:, None]
        x = c * v[..., 0] - s * v[..., 1]
        y = s * v[..., 0] + c * v[..., 1]
        local = torch.stack([x / sx[:, None], y / sy[:, None]], dim=-1)

        g = (local - self.sdf_origin) / self.sdf_spacing
        g = 2.0 * g / (self.sdf_res - 1) - 1.0
        # Same flattening as _sample, for the same reason: no stride-0 batch.
        d = F.grid_sample(self.sdf, g.reshape(1, -1, 1, 2), mode="bilinear",
                          padding_mode="border", align_corners=True).reshape(K, -1)
        outside = d.clamp_min(0.0)
        tau2 = self.cfg.coverage_tau_mm ** 2
        return (outside * outside / (outside * outside + tau2)).mean(dim=1)

    def _candidates(self, mask: np.ndarray, k: int | None = None,
                    origin: tuple[int, int] = (0, 0)) -> torch.Tensor:
        """``mask`` may be a window; ``origin`` is where its corner sits."""
        cfg = self.cfg
        k = int(k or cfg.n_restarts)
        seed = np.array(seed_from_mask(mask, self.tpl), np.float32)
        seed[0] += origin[0]
        seed[1] += origin[1]
        cands = [seed]
        if cfg.allow_flip:
            f = seed.copy(); f[2] += np.pi; cands.append(f)
        if self.seed_pose is not None:
            # No flipped twin for the manual pose: the whole point of placing it
            # by hand is that its orientation is known, and offering the 180°
            # alternative would hand back the head/tail ambiguity it resolves.
            cands.insert(0, self.seed_pose.copy())
        if self.prev is not None:
            cands.insert(0, self.prev.copy())
            cands.insert(1, self.prev.copy())   # duplicated: gets its own jitter below

        base = np.stack(cands)
        reps = int(np.ceil(k / len(base)))
        out = np.repeat(base, reps, axis=0)[:k].copy()
        # Jitter every restart except the first, so one hypothesis is always the
        # clean warm start from the previous frame.
        rng = np.random.default_rng(0)
        n = len(out) - 1
        out[1:, 0:2] += rng.normal(0, 6.0, (n, 2))
        out[1:, 2] += rng.normal(0, 0.25, n)
        out[1:, 3:5] *= np.exp(rng.normal(0, 0.10, (n, 2)))
        return torch.from_numpy(out.astype(np.float32)).to(self.dev.torch_device)

    def fit(self, mask: np.ndarray, signal: np.ndarray | None = None) -> Pose | None:
        """Fit one frame. ``signal`` is the continuous image the mask came from;
        supplying it lets interior features match real internal edges."""
        if mask.sum() < 10:
            return None
        cfg = self.cfg
        warm = self.prev is not None

        # Everything below the window line is done on a crop around where the
        # robot was, not on the frame. The poses stay in image coordinates --
        # only the field lookups are offset -- so nothing downstream can tell.
        win = self._window(mask.shape[:2])
        if win is None:
            sub, sub_signal, off = mask, signal, None
            ox = oy = 0
        else:
            x0, y0, x1, y1 = win
            ox, oy = x0, y0
            sub = np.ascontiguousarray(mask[y0:y1, x0:x1])
            sub_signal = (None if signal is None
                          else np.ascontiguousarray(signal[y0:y1, x0:x1]))
            off = torch.tensor([float(x0), float(y0)],
                               device=self.dev.torch_device)
            if sub.sum() < 10:
                # The robot left the window -- an abrupt jump, or a lost track.
                # Fall back to the whole frame rather than reporting nothing.
                sub, sub_signal, off = mask, signal, None
                ox = oy = 0

        # Two fields on purpose. The silhouette matches mask edges only -- adding
        # interior edges there would let the outer boundary settle onto an
        # internal one. The features match both.
        D = distance_field(sub, self.dev)
        Dfeat = D
        if self.feat is not None and sub_signal is not None and cfg.feature_weight > 0:
            Dfeat = distance_field(sub, self.dev,
                                   extra_edges=interior_edges(sub_signal, sub))
        S = soft_mask(sub, self.dev)
        n_k = cfg.n_restarts_warm if warm else cfg.n_restarts
        p = self._candidates(sub, min(max(int(n_k), 1), cfg.n_restarts),
                             origin=(ox, oy)).requires_grad_(True)

        # Subsample the observed robot pixels for the coverage term.
        ys, xs = np.nonzero(sub)
        if xs.size > cfg.coverage_points:
            sel = self._rng.choice(xs.size, cfg.coverage_points, replace=False)
            xs, ys = xs[sel], ys[sel]
        obs = torch.from_numpy(np.stack([xs + ox, ys + oy], 1).astype(np.float32)
                               ).to(self.dev.torch_device)
        opt = torch.optim.Adam([p], lr=cfg.lr)

        # Scales live in log space during optimization so they cannot go
        # negative and so a 2x growth and a 2x shrink are equally reachable.
        s_lo, s_hi = 1 - cfg.max_scale_change, 1 + cfg.max_scale_change
        if self._base_scale is None:
            self._base_scale = (float(p[0, 3].item()), float(p[0, 4].item()))
        bx, by = self._base_scale
        # Until the references are learned the fit runs as it always did: the
        # loose symmetric limits and no width penalty. That learning window is
        # what the references are measured from, so constraining it would be
        # circular.
        wref = self._wid_ref
        w_weight = cfg.width_weight if wref is not None else 0.0
        if wref is None:
            sx_lo, sx_hi = bx * s_lo, bx * s_hi
        else:
            w_lim = min(cfg.width_max_change, cfg.max_scale_change)
            sx_lo, sx_hi = wref * (1 - w_lim), wref * (1 + w_lim)
        sy_hi = self._length_ceiling(by)
        sy_lo = by * s_lo

        n_iter = cfg.iters_warm if warm else cfg.iters
        prior = None
        sigma = max(cfg.scale_prior_rate_per_s * self.dt, 1e-3)
        if warm and cfg.scale_prior_weight > 0:
            prior = torch.tensor(self.prev[3:5], device=p.device)

        # Graduated non-convexity: start with a wide kernel so distant restarts
        # still feel a gradient and the basin of attraction is large, then shrink
        # it geometrically so the endgame is dominated by nearby edges only. A
        # fixed wide kernel converges reliably but imprecisely; a fixed narrow one
        # is precise but gets stuck. Annealing gets both.
        taus = np.geomspace(cfg.tau_px, cfg.tau_final_px, n_iter)

        best_seen = float("inf")
        quiet = 0
        for it in range(n_iter):
            opt.zero_grad(set_to_none=True)
            tau2 = float(taus[it]) ** 2
            q = _transform(self.pts, p)
            d = _sample(D, q, off)
            rho = (d * d) / (d * d + tau2)          # bounded, saturates at 1
            loss = rho.mean(dim=1)

            if self.feat is not None and cfg.feature_weight > 0:
                # Holes, windows, the inner edge of a frame. The silhouette
                # alone fixes pose and the two scales but says nothing about the
                # inside of the body, so a boundary that segmentation renders a
                # little wrong has nothing to correct it. Interior edges make the
                # fit over-determined, which is what lets the threshold be tight.
                qf = _transform(self.feat, p)
                df = _sample(Dfeat, qf, off)
                rf = ((df * df) / (df * df + tau2)).mean(dim=1)
                loss = (loss + cfg.feature_weight * rf) / (1.0 + cfg.feature_weight)

            # Containment term. Sample the mask just *outside* the fitted
            # outline: for a correct fit those samples are background (0).
            # If the template has collapsed onto part of the robot, its
            # boundary lies inside real tissue and the samples read 1.
            #
            # This is deliberately one-sided, and that is the whole point:
            # occlusion only ever *removes* mask pixels, so it can never
            # trigger this penalty. Chamfer distance alone cannot distinguish
            # "correctly fitted" from "shrunk onto a subset of the edges" --
            # both put template points on real edges. This term can.
            out = q + cfg.contain_offset_px * _rotate(self.nrm, p[:, 2])
            loss = loss + cfg.contain_weight * _sample(S, out, off).mean(dim=1)
            loss = loss + cfg.coverage_weight * self._coverage(obs, p)

            # Width held to the drawing. Quadratic rather than bounded on
            # purpose: this is not a robust term shrugging off outliers, it is a
            # statement that a wrong width is always wrong, and the further out
            # the fit goes the harder it should be pulled back.
            if w_weight > 0:
                wrel = (p[:, 3] - wref) / max(wref * cfg.width_tol, 1e-6)
                loss = loss + w_weight * wrel.pow(2)

            if prior is not None:
                rel = (p[:, 3:5] - prior) / prior.clamp_min(1e-6)
                loss = loss + cfg.scale_prior_weight * (rel / sigma).pow(2).mean(1)
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                p[:, 3].clamp_(sx_lo, sx_hi)
                # Length is free to shrink and capped above; see FitConfig.
                p[:, 4].clamp_(sy_lo, sy_hi)

            # Checked only every Nth iteration: reading the loss forces a
            # GPU->CPU sync, and doing that every step would cost more than the
            # iterations it saves.
            if cfg.early_stop and it >= cfg.early_stop_check and \
                    it % cfg.early_stop_check == 0 and it < n_iter - 1:
                cur = float(loss.min().detach())
                if cur > best_seen - abs(best_seen) * cfg.early_stop_tol:
                    quiet += 1
                    if quiet >= cfg.early_stop_patience:
                        # tau is annealed on a schedule tied to n_iter, so jump
                        # it to its final value rather than leaving the fit
                        # converged under a wider kernel than it would have had.
                        taus_tail = float(cfg.tau_final_px)
                        for _ in range(2):
                            opt.zero_grad(set_to_none=True)
                            q = _transform(self.pts, p)
                            d = _sample(D, q, off)
                            l2 = ((d * d) / (d * d + taus_tail ** 2)).mean(dim=1)
                            l2.sum().backward()
                            opt.step()
                        break
                else:
                    quiet = 0
                best_seen = min(best_seen, cur)

        with torch.no_grad():
            q = _transform(self.pts, p)
            d = _sample(D, q, off)
            # Selection and the reported cost use the *final* kernel and no prior,
            # so the number written to the CSV measures agreement with the image
            # alone and is comparable across frames.
            tau2 = cfg.tau_final_px ** 2
            out = q + cfg.contain_offset_px * _rotate(self.nrm, p[:, 2])
            contain = _sample(S, out, off).mean(dim=1)
            cover = self._coverage(obs, p)
            edge = ((d * d) / (d * d + tau2)).mean(dim=1)
            d_all = d
            if self.feat is not None and cfg.feature_weight > 0:
                df = _sample(Dfeat, _transform(self.feat, p), off)
                rf = ((df * df) / (df * df + tau2)).mean(dim=1)
                edge = (edge + cfg.feature_weight * rf) / (1.0 + cfg.feature_weight)
                d_all = torch.cat([d, df], dim=1)
            loss = (edge + cfg.contain_weight * contain
                    + cfg.coverage_weight * cover)
            k = int(torch.argmin(loss))
            best = p[k].detach().cpu().numpy()
            # Confidence combines "how much of the outline sits on a real edge"
            # with "is any observed robot left outside the fit", so a collapsed
            # outline scores low even though most of its points are on edges.
            conf = (float((d[k] < cfg.inlier_px).float().mean())
                    * float(1.0 - contain[k].clamp(0, 1))
                    * float(1.0 - cover[k].clamp(0, 1)))

            feat_fit = float("nan")
            if self.feat is not None and d_all is not d:
                feat_fit = float((d_all[k, d.shape[1]:] < cfg.inlier_px).float().mean())

        pose = Pose(*[float(x) for x in best], cost=float(loss[k]), confidence=conf,
                    feature_fit=feat_fit)
        # Only carry a confident pose forward. Warm-starting from a bad fit is
        # how trackers get permanently lost after a single hard frame.
        self.prev = pose.as_array() if conf >= cfg.recovery_conf else None
        self._note_scales(pose.sx, pose.sy, conf)
        return pose

    def feature_outline(self, pose: Pose) -> np.ndarray | None:
        """The interior feature points at this pose, for the overlay."""
        if self.feat is None:
            return None
        p = torch.from_numpy(pose.as_array()[None]).to(self.dev.torch_device)
        with torch.no_grad():
            return _transform(self.feat, p)[0].cpu().numpy()

    def outline(self, pose: Pose) -> np.ndarray:
        """The fitted outline in image pixels, including any occluded portion."""
        p = torch.from_numpy(pose.as_array()[None]).to(self.dev.torch_device)
        with torch.no_grad():
            return _transform(self.pts, p)[0].cpu().numpy()
