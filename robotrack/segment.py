"""Segmentation of the robot against a static background.

Your rig -- tripod, stabilization off, fixed 2x lens -- makes this much easier
than the general case. Because the camera truly does not move, a per-pixel
median over the clip is an excellent background plate, and anything that
differs from it is the robot.

This also handles your occlusion requirement for free at the segmentation
stage: a *static* obstacle is part of the median background, so when it covers
part of the robot the mask simply has a bite taken out of it. Nothing is
mistaken for robot. Recovering the hidden geometry is then the fitting stage's
job (see register.py), not segmentation's.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .decode import FrameReader
from .gpu import Device, closing, opening, otsu_threshold


@dataclass
class SegmentConfig:
    n_background_frames: int = 60   # median plate sample size
    open_px: int = 3                # speckle removal
    close_px: int = 7               # pinhole filling
    # 0.05% of the frame. The old 0.01% was tuned for a luma mask, where stray
    # foreground is rare; a color-keyed mask of a textured medium carries
    # specks everywhere and the lower floor let them inflate the body.
    min_area_frac: float = 5e-4     # ignore blobs below this fraction of frame
    manual_threshold: float | None = None
    # Region of interest, in *full-resolution* image pixels, or None for the
    # whole frame. Stored at full resolution so one region stays valid when the
    # decode scale changes, exactly like a manual placement.
    roi: tuple[int, int, int, int] | None = None   # x, y, w, h
    gap_factor: float = 1.0         # occlusion gap tolerated, in body lengths

    # --- how the robot is told apart from everything else ------------------
    # "luma"   difference from the median background plate, in brightness
    # "color" distance from the background's own color, brightness ignored
    # "auto"   measure the color separation in this clip, then pick
    mode: str = "auto"
    color_frac: float = 0.30       # cut this far along background -> robot color
    bg_chroma: tuple[float, float] | None = None      # (a, b), auto-estimated
    target_chroma: tuple[float, float] | None = None  # (a, b) of the robot
    # Below this separation in a*b* units the two colors are too close to key
    # on and "auto" falls back to luma. The reference clip measures 84.
    min_separation: float = 20.0
    # The grouped mask may not exceed this multiple of the learned body extent.
    # 1.10 leaves room for a body that stretches while still rejecting a speck
    # on the far side of the dish.
    envelope_factor: float = 1.10

    @property
    def needs_color(self) -> bool:
        return self.mode in ("color", "auto")


# ---------------------------------------------------------------------------
# Color
#
# Brightness is the unreliable channel here, which is why the original luma path
# struggles: a translucent hydrogel over a colored medium varies in luminance
# with thickness, lighting and the camera's own exposure, and a pale interior can
# read *brighter* than the part you want. Hue does not move with any of that.
#
# On the reference clip the medium sits at a*+41 b*-24 and the limbs at a*+24
# b*+58 -- 84 units apart in chroma, against 26 levels of luma. Keying on color
# turns a marginal threshold into an obvious one, and needs no background plate
# at all, which also removes the assumption that the robot moves far enough for
# a median to see through it.
# ---------------------------------------------------------------------------

def roi_rect(cfg: "SegmentConfig", width: int, height: int,
             scale: float = 1.0) -> tuple[int, int, int, int] | None:
    """The configured region in *this* frame's pixels, clipped, or None."""
    if not cfg.roi:
        return None
    try:
        x, y, w, h = (int(round(v * scale)) for v in cfg.roi)
    except (TypeError, ValueError):
        return None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1 - x0, y1 - y0


def apply_roi(mask: np.ndarray, rect) -> np.ndarray:
    """Zero everything outside ``rect``.

    Masking rather than cropping, deliberately. A crop would be marginally
    cheaper and would force every coordinate downstream -- the fit, the
    centroid, the overlay, the exported CSV -- to carry an offset, which is
    exactly the kind of bookkeeping that produces an off-by-one nobody notices
    until the numbers are already in a figure. Zeroing keeps one coordinate
    system for the whole pipeline.
    """
    if rect is None:
        return mask
    x, y, w, h = rect
    out = np.zeros_like(mask)
    out[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    return out


def crop_to_roi(frames, rect):
    """Sample frames narrowed to the region, for color estimation."""
    if rect is None:
        return frames
    x, y, w, h = rect
    return [f[y:y + h, x:x + w] for f in frames]


def chroma(frame_bgr: np.ndarray) -> np.ndarray:
    """(H,W,2) float32 of CIELAB a*, b* -- color with brightness divided out."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32)


def estimate_colors(frames_bgr, bins: int = 64
                     ) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Find the medium's color and the robot's, from sample frames.

    The medium fills most of every frame, so the densest cell of the a*b*
    histogram is the background by construction -- no assumption about which
    color it happens to be. The robot is then the far mode: the cell furthest
    from the background that still carries real mass, which rejects the sparse
    tail of specular highlights and compression noise.

    Returns ``(background_ab, target_ab, separation)``.
    """
    ab = np.concatenate([chroma(f).reshape(-1, 2) for f in frames_bgr])
    hist, xe, ye = np.histogram2d(ab[:, 0], ab[:, 1], bins=bins,
                                  range=[[0, 255], [0, 255]])
    cx, cy = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    ref = np.array([cx[i], cy[j]], np.float32)

    grid = np.stack(np.meshgrid(cx, cy, indexing="ij"), -1)
    dist = np.sqrt(((grid - ref) ** 2).sum(-1))
    mass = hist / max(float(hist.sum()), 1.0)
    # 0.2% of pixels is a few thousand at video resolution: comfortably a real
    # object, comfortably above stray noise.
    far = np.where(mass > 0.002, dist, 0.0)
    ti, tj = np.unravel_index(int(np.argmax(far)), far.shape)
    tgt = np.array([cx[ti], cy[tj]], np.float32)
    return ((float(ref[0]), float(ref[1])), (float(tgt[0]), float(tgt[1])),
            float(np.hypot(*(tgt - ref))))


def color_distance(frame_bgr: np.ndarray, bg_ab) -> np.ndarray:
    """Per-pixel distance from the background color, in a*b* units."""
    d = chroma(frame_bgr) - np.asarray(bg_ab, np.float32)
    return np.sqrt((d * d).sum(-1))


def segment_color(frame_bgr: np.ndarray, cfg: SegmentConfig,
                   bg_ab, separation: float,
                   threshold: float | None = None) -> tuple[np.ndarray, float]:
    """Color-keyed mask, with no background plate involved.

    The cut is placed a fixed fraction of the way from the medium's color to
    the robot's rather than by Otsu. Otsu assumes two comparable populations;
    here the robot is a few percent of the frame, so Otsu lands high and slices
    the body in half -- measured at 62-67% of the mask in the largest fragment,
    against 92-93% for the fractional cut.
    """
    thr = float(threshold) if threshold is not None else cfg.color_frac * separation
    d = color_distance(frame_bgr, bg_ab)
    m = (d > thr).astype(np.uint8)
    if cfg.open_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((cfg.open_px,) * 2, np.uint8))
    if cfg.close_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((cfg.close_px,) * 2, np.uint8))
    return apply_roi(m, roi_rect(cfg, m.shape[1], m.shape[0])), thr


def build_background(reader: FrameReader, cfg: SegmentConfig,
                     dev: Device) -> tuple[torch.Tensor, np.ndarray]:
    """Per-pixel median plate.

    Median rather than mean: the robot is present in every frame, and a mean
    would smear a ghost of it into the plate. As long as the robot occupies any
    given pixel for less than half the clip, the median sees straight through it.
    """
    frames = reader.sample(cfg.n_background_frames)
    if frames.size == 0:
        raise RuntimeError("Could not read any frames to build a background model.")
    stack = torch.from_numpy(frames).to(dev.torch_device).float()
    try:
        return stack.median(dim=0).values, frames
    except (NotImplementedError, RuntimeError):
        # A reduction MPS may not implement for this shape. One fallback for the
        # plate, computed once per clip, is cheaper than losing the accelerator.
        return torch.from_numpy(np.median(frames, axis=0).astype(np.float32)).to(
            dev.torch_device), frames


@dataclass
class Mask:
    index: int
    t: float
    mask: np.ndarray          # uint8 {0,1}, full frame
    area_px: float
    contour: np.ndarray | None   # (N,2) float32 outline, largest component
    frame: np.ndarray | None = None   # the decoded frame, for live overlays


def segment_frame(frame: torch.Tensor, background: torch.Tensor,
                  cfg: SegmentConfig, threshold: float | None) -> tuple[torch.Tensor, float]:
    """Absolute-difference segmentation, entirely on device."""
    diff = (frame - background).abs()
    thr = threshold if threshold is not None else otsu_threshold(diff)
    m = (diff > thr).float()
    if cfg.open_px > 1:
        m = opening(m, cfg.open_px)
    if cfg.close_px > 1:
        m = closing(m, cfg.close_px)
    rect = roi_rect(cfg, m.shape[-1], m.shape[-2])
    if rect is not None:
        x, y, w, h = rect
        keep = torch.zeros_like(m)
        keep[..., y:y + h, x:x + w] = 1.0
        m = m * keep
    return m, thr


def _bbox_gap(a, b) -> float:
    """Closest distance between two component bounding boxes (0 if touching)."""
    ax, ay, aw, ah = a[cv2.CC_STAT_LEFT], a[cv2.CC_STAT_TOP], a[cv2.CC_STAT_WIDTH], a[cv2.CC_STAT_HEIGHT]
    bx, by, bw, bh = b[cv2.CC_STAT_LEFT], b[cv2.CC_STAT_TOP], b[cv2.CC_STAT_WIDTH], b[cv2.CC_STAT_HEIGHT]
    dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return float(np.hypot(dx, dy))


def largest_component(mask_np: np.ndarray, min_area: float,
                      reach_px: float = 0.0, max_extent_px: float = 0.0,
                      envelope_factor: float = 0.0
                      ) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Keep the robot's blobs and return the outline.

    An occluder splitting the robot in two is the normal case here, so this is
    deliberately not "keep the largest blob". The grouping rule is *spatial*
    rather than a pure area ratio: a fragment is kept if it survives the noise
    floor and sits close enough to the main blob to plausibly be the same object
    across an occlusion gap.

    This matters more than it sounds. When only a sliver of the robot peeks past
    an obstacle, an area-ratio rule throws that sliver away -- and that sliver is
    precisely the evidence that pins down the far end of the body. Discarding it
    lets the fitted outline collapse onto the visible remainder.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask_np, np.uint8), 0.0, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    main = order[0]
    if areas[main] < min_area:
        return np.zeros_like(mask_np, np.uint8), 0.0, None

    # Fragments may sit up to about one body length away, since an occluder can
    # hide an arbitrary middle section. Deriving the reach from the *visible*
    # fragment is wrong precisely when it matters: under heavy occlusion the
    # main fragment is small, the reach shrinks with it, and the two halves of a
    # bisected robot are pushed apart further than the rule will bridge. The
    # caller therefore passes a reach learned from the whole clip.
    reach = max(reach_px, float(max(stats[main + 1, cv2.CC_STAT_WIDTH],
                                    stats[main + 1, cv2.CC_STAT_HEIGHT])))
    # The configured noise floor is the only size gate. Dropping below it to
    # catch smaller slivers starts admitting background texture, which is worse.
    noise_floor = min_area
    keep = [main + 1]

    # Reach alone is not enough. It was tuned for a luma mask, where stray
    # foreground is rare; a color-keyed mask of a textured medium has specks
    # scattered everywhere, and "within one body length of the main blob" then
    # sweeps up the entire frame -- observed as a mask spanning all 438x778 px
    # from 13 fragments. So fragments are accepted nearest-first and only while
    # the union still fits inside a plausible body envelope. The robot cannot be
    # bigger than the robot, and that is the constraint the reach rule was
    # missing.
    def _extent(box):
        return max(box[2] - box[0], box[3] - box[1])

    s_main = stats[main + 1]
    union = [s_main[cv2.CC_STAT_LEFT], s_main[cv2.CC_STAT_TOP],
             s_main[cv2.CC_STAT_LEFT] + s_main[cv2.CC_STAT_WIDTH],
             s_main[cv2.CC_STAT_TOP] + s_main[cv2.CC_STAT_HEIGHT]]
    if max_extent_px > 0:
        cap = max_extent_px
    elif envelope_factor > 0:
        # No clip-wide estimate available (the single-frame preview path), so the
        # main blob stands in for the body. Weaker than the learned extent, but
        # it still stops a speck across the dish from joining.
        cap = envelope_factor * _extent(union)
    else:
        cap = float("inf")
    cap = max(cap, _extent(union))     # never reject the main blob itself

    cands = [j for j in order[1:] if areas[j] >= noise_floor]
    cands.sort(key=lambda j: _bbox_gap(s_main, stats[j + 1]))
    for j in cands:
        st = stats[j + 1]
        if _bbox_gap(s_main, st) > reach:
            continue
        grown = [min(union[0], st[cv2.CC_STAT_LEFT]),
                 min(union[1], st[cv2.CC_STAT_TOP]),
                 max(union[2], st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH]),
                 max(union[3], st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT])]
        if _extent(grown) > cap:
            continue
        union = grown
        keep.append(j + 1)
    out = np.isin(labels, keep).astype(np.uint8)
    total = float(out.sum())
    if total < min_area:
        return np.zeros_like(mask_np, np.uint8), 0.0, None
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) if cnts else None
    return out, total, contour


@dataclass
class ColorModel:
    """What "auto" decided, and why -- reported so the choice is never silent."""
    mode: str                    # "luma" or "color"
    bg_ab: tuple[float, float] | None
    target_ab: tuple[float, float] | None
    separation: float
    reason: str

    def summary(self) -> str:
        if self.mode != "color":
            return f"segmentation: luma — {self.reason}"
        a, b = self.bg_ab
        ta, tb = self.target_ab
        return (f"segmentation: color — medium a{a - 128:+.0f} b{b - 128:+.0f}, "
                f"robot a{ta - 128:+.0f} b{tb - 128:+.0f}, "
                f"separation {self.separation:.0f} — {self.reason}")


def choose_color_model(reader: FrameReader, cfg: SegmentConfig,
                        n: int = 24) -> tuple[ColorModel, np.ndarray]:
    """Decide between luma and color keying, and return the sampled frames.

    Sampling is shared with the background plate so "auto" costs no extra decode.
    """
    if not cfg.needs_color:
        return ColorModel("luma", None, None, 0.0, "selected explicitly"), \
            np.empty((0,), np.uint8)

    color_reader = FrameReader(reader.info, reader.backend,
                                scale=reader.scale, color=True)
    frames = color_reader.sample(n)
    if frames.size == 0:
        return ColorModel("luma", None, None, 0.0,
                           "no frames could be sampled for color"), frames

    # Estimate the colors from inside the region, not the whole frame. The
    # estimator takes the densest a*b* cell as the medium and the farthest
    # populated cell as the robot -- so a dish rim, a lamp reflection or the
    # bench beyond the well will be chosen as "the robot" simply for being
    # far away and numerous. Narrowing to the region is what makes the
    # estimate describe the thing you are actually tracking.
    rect = roi_rect(cfg, frames.shape[2], frames.shape[1]) if frames.ndim == 4 else None
    bg, tgt, sep = estimate_colors(crop_to_roi(list(frames), rect) if rect else frames)
    if cfg.bg_chroma:
        bg = tuple(cfg.bg_chroma)
    if cfg.target_chroma:
        tgt = tuple(cfg.target_chroma)
        sep = float(np.hypot(tgt[0] - bg[0], tgt[1] - bg[1]))

    if cfg.mode == "color":
        return ColorModel("color", bg, tgt, sep, "selected explicitly"), frames
    if sep >= cfg.min_separation:
        return ColorModel("color", bg, tgt, sep,
                           "colors are well separated"), frames
    return ColorModel("luma", bg, tgt, sep,
                       f"only {sep:.0f} a*b* units apart, not enough to key on"), frames


class Segmenter:
    def __init__(self, reader: FrameReader, cfg: SegmentConfig, dev: Device):
        self.reader, self.cfg, self.dev = reader, cfg, dev
        self.min_area = cfg.min_area_frac * reader.width * reader.height
        self._thr: float | None = cfg.manual_threshold
        self._thr_history: list[float] = []
        self._extent_px: float = 0.0   # learned body size, drives fragment reach

        self.model, color_samples = choose_color_model(reader, cfg)
        self.color = self.model.mode == "color"
        self.background = None

        if self.color:
            # No background plate at all. That is not just a saving: the plate
            # assumes the robot vacates every pixel for more than half the clip,
            # and a robot that mostly sits still leaves a ghost of itself in the
            # median. Color keying has no such assumption.
            self.source = FrameReader(reader.info, reader.backend,
                                      scale=reader.scale, color=True)
            samples = color_samples
        else:
            self.source = reader
            self.background, samples = build_background(reader, cfg, dev)

        # Seed the body size from frames already decoded, so the first frames
        # are not analyzed with a fragment reach of zero.
        for f in samples[:: max(1, len(samples) // 12)] if len(samples) else []:
            mm = self._mask_of(f)
            _, _, c = largest_component(mm, self.min_area)
            if c is not None:
                self._extent_px = max(self._extent_px,
                                      float(max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))))

    def _mask_of(self, frame: np.ndarray) -> np.ndarray:
        """Raw mask for one frame, in whichever mode is active."""
        if self.color:
            m, thr = segment_color(frame, self.cfg, self.model.bg_ab,
                                    self.model.separation, self._thr)
            self._last_thr = thr
            return m
        ft = torch.from_numpy(frame.copy()).to(self.dev.torch_device).float()
        m, thr = segment_frame(ft, self.background, self.cfg, self._thr)
        self._last_thr = thr
        return m.to(torch.uint8).cpu().numpy()

    def __iter__(self):
        for i, t, frame in self.source:
            mask_np = self._mask_of(frame)
            # Freeze the threshold after a short burn-in. A per-frame Otsu drifts
            # when the robot is occluded (less foreground changes the histogram),
            # which would make the measured size depend on the occlusion. The
            # color cut is already frame-independent, so this only binds luma.
            if self.cfg.manual_threshold is None and not self.color:
                self._thr_history.append(self._last_thr)
                if len(self._thr_history) == 30:
                    self._thr = float(np.median(self._thr_history))
            mask_np, area, contour = largest_component(
                mask_np, self.min_area, self.cfg.gap_factor * self._extent_px,
                max_extent_px=self.cfg.envelope_factor * self._extent_px)
            if contour is not None:
                # Running maximum: the clip's least-occluded view is the best
                # available estimate of true body size, and it only grows.
                ext = float(max(np.ptp(contour[:, 0]), np.ptp(contour[:, 1])))
                self._extent_px = max(self._extent_px, ext)
            yield Mask(i, t, mask_np, area, contour, frame)
