"""Tracking by appearance, for footage where color keying cannot work.

Why this exists
---------------
The colour path assumes the robot can be told from the medium *one pixel at a
time*. On well-lit footage that is true and it is the fastest thing to do. On
some footage it is simply false, and no threshold rescues it. On a measured
example -- a magenta well, an orange robot, a saturated dish rim -- the robot and
the medium separate by 1.39 in b*, 0.91 in L*, 0.85 in a*, where a usable
threshold wants well above 2. Local contrast, Canny density and chroma gradient
all scored below 0.6. There is nothing there to threshold.

The information a person uses in that situation is not per-pixel at all: it is
the robot's *spatial pattern*, at a scale of a couple of hundred pixels. That
signal is strong on exactly the frame where the per-pixel one is absent --
correlating the robot's b* patch against the rest of the well beats the best
competing location by 0.34 to 0.41 even after rotating the patch 8 degrees or
rescaling it 6%.

So this module tracks the robot by aligning its *appearance* rather than by
classifying its pixels.

How
---
``cv2.findTransformECC`` maximises the enhanced correlation coefficient between
a reference patch and the current frame over an affine warp. Three properties
make it the right tool here rather than a starting point for one:

* It optimises correlation, which is invariant to any affine change of intensity
  -- the illumination drift and exposure hunting that wreck a fixed threshold do
  not move it.
* An affine warp carries translation, rotation and two independent scales, which
  is exactly the five-parameter pose the rest of this program already uses. The
  length scale it recovers *is* the measurement.
* It is in OpenCV, which is already a dependency. No model, no training data, no
  weights file to keep in step with the application.

On synthetic pose changes applied to that same failing frame it recovered the
truth to 0.00 degrees and three decimal places of scale, from seeds 12 degrees
and 25 px away.

What it does not do
-------------------
It tracks *the thing you pointed it at*. If the reference patch is placed on the
wrong object it will follow the wrong object faithfully and report a high
correlation while doing it -- which is why the reference is taken from a frame
you have inspected, and why the returned correlation is surfaced as confidence
rather than kept internal. It also cannot recover from total occlusion: with
nothing to correlate against it holds the last pose and reports a low score.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .register import Pose


@dataclass
class AppearanceConfig:
    """Settings for the appearance lock."""

    channel: str = "chroma"     # "chroma" -> b*; "luma" -> grey
    max_iters: int = 200        # from a cold seed; consecutive frames need far fewer
    eps: float = 1e-6
    # Width of the Gaussian OpenCV smooths the gradients with. This is
    # findTransformECC's seventh argument and it **must be odd** -- an even
    # value makes the call raise, which surfaces as "the tracker never
    # converges" rather than as a wrong argument, and cost an hour to find.
    gauss_filt_size: int = 5
    # Correlation below this is reported as a failed frame rather than a pose.
    # ECC returns a coefficient in [-1, 1]; a locked track sits above 0.9 and a
    # lost one collapses well below 0.5.
    min_cc: float = 0.45
    # How much context around the robot to carry in the reference patch. Some
    # background is useful -- it is what pins rotation -- but too much and the
    # medium's own texture starts to dominate the correlation.
    margin: float = 0.35


def _signal(frame: np.ndarray, channel: str) -> np.ndarray:
    """The scalar image the correlation runs on, normalised.

    b* rather than intensity by default: it is the axis that separates warm
    tissue from a magenta medium, and unlike luma it barely moves when the lamp
    or the camera's exposure does.
    """
    if frame.ndim == 2:
        g = frame.astype(np.float32)
    elif channel == "luma":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 2].astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def _decompose(warp: np.ndarray) -> tuple[float, float, float, float, float]:
    """Affine matrix -> (tx, ty, theta, sx, sy).

    The two column norms are the scales and the first column's angle is the
    rotation. Any shear the fit picked up is dropped rather than reported: this
    program's pose has no shear parameter, and a robot that appears sheared is
    a tracking artefact, not a measurement.
    """
    A = np.asarray(warp, np.float64)[:, :2]
    sx = float(np.hypot(A[0, 0], A[1, 0]))
    sy = float(np.hypot(A[0, 1], A[1, 1]))
    theta = float(np.arctan2(A[1, 0], A[0, 0]))
    return float(warp[0, 2]), float(warp[1, 2]), theta, sx, sy


class AppearanceTracker:
    """Follows a patch of the reference frame through the clip.

    ``ref_pose`` is the pose the CAD fit (or your hand placement) had on the
    reference frame. Everything this returns is composed onto it, so the poses
    coming out are in the same units and the same frame of reference as the
    ones the color path produces -- the pipeline, the plots and the exported
    table need no knowledge of which method produced them.
    """

    def __init__(self, ref_frame: np.ndarray, rect: tuple[int, int, int, int],
                 ref_pose: Pose | None = None, cfg: AppearanceConfig | None = None):
        self.cfg = cfg or AppearanceConfig()
        self.ref_pose = ref_pose
        sig = _signal(ref_frame, self.cfg.channel)

        x, y, w, h = (int(v) for v in rect)
        m = self.cfg.margin
        x0 = max(0, int(x - m * w)); y0 = max(0, int(y - m * h))
        x1 = min(sig.shape[1], int(x + w + m * w))
        y1 = min(sig.shape[0], int(y + h + m * h))
        if x1 - x0 < 24 or y1 - y0 < 24:
            raise ValueError("reference region is too small to track")
        self.origin = (x0, y0)
        self.ref = np.ascontiguousarray(sig[y0:y1, x0:x1])

        # Seed: the identity warp placing the patch back where it came from.
        self._warp = np.eye(2, 3, dtype=np.float32)
        self._warp[0, 2] = float(x0)
        self._warp[1, 2] = float(y0)
        self._base = self._warp.copy()
        self.last_cc = 1.0

        # Does the patch actually contain the robot?
        #
        # This mode measures length by watching the whole patch deform, so a
        # region that clips the robot's ends removes exactly the evidence the
        # length is read from. The failure is silent and total: correlation
        # stays high, tracking looks perfect, and the reported length simply
        # does not change. It cost an hour of chasing a real bug that turned out
        # to be a badly cropped test fixture, which is precisely what a user
        # would do to themselves by drawing the region tightly.
        self.clipped = ""
        if ref_pose is not None:
            hw = abs(ref_pose.sx) * 0.5, abs(ref_pose.sy) * 0.5
            c, s = abs(np.cos(ref_pose.theta)), abs(np.sin(ref_pose.theta))
            need_x = 2.0 * (hw[0] * c + hw[1] * s)
            need_y = 2.0 * (hw[0] * s + hw[1] * c)
            have_x, have_y = x1 - x0, y1 - y0
            if need_x > have_x * 1.02 or need_y > have_y * 1.02:
                self.clipped = (
                    f"the region is {have_x}x{have_y} px but the outline needs "
                    f"about {need_x:.0f}x{need_y:.0f} — appearance tracking "
                    f"reads length from the whole patch, so a region that cuts "
                    f"the robot off will report little or no length change")

    # ---- per frame -------------------------------------------------------

    def track(self, frame: np.ndarray) -> Pose | None:
        """Align the reference patch to this frame. None if it did not converge.

        The previous frame's warp seeds the next, which is what keeps this cheap:
        between consecutive frames the robot has moved a fraction of a pixel, so
        the optimiser starts essentially at the answer.
        """
        sig = _signal(frame, self.cfg.channel)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(self.cfg.max_iters), float(self.cfg.eps))
        warp = self._warp.copy()
        gauss = max(1, int(self.cfg.gauss_filt_size) | 1)      # force odd
        try:
            cc, warp = cv2.findTransformECC(self.ref, sig, warp, cv2.MOTION_AFFINE,
                                            crit, None, gauss)
        except cv2.error:
            # Did not converge. Hold the last pose rather than jumping: a frame
            # the optimiser could not explain is usually a blink -- a bubble, a
            # hand over the dish -- and the next one is fine.
            self.last_cc = 0.0
            return None
        if not np.isfinite(warp).all():
            self.last_cc = 0.0
            return None
        self._warp = warp.astype(np.float32)
        self.last_cc = float(cc)
        if cc < self.cfg.min_cc:
            return None
        return self._pose(warp, cc)

    def _pose(self, warp: np.ndarray, cc: float) -> Pose:
        """Compose this frame's warp onto the reference pose."""
        tx, ty, theta, sx, sy = _decompose(warp)
        bx, by, btheta, bsx, bsy = _decompose(self._base)
        # Relative to where the patch started, so an unmoved robot reads as no
        # change regardless of where in the frame it happens to sit.
        d_theta = theta - btheta
        r_sx = sx / max(bsx, 1e-9)
        r_sy = sy / max(bsy, 1e-9)

        if self.ref_pose is None:
            # No CAD reference: report the patch's own centre and scale, which
            # still gives a usable strain and trajectory even with no drawing.
            h, w = self.ref.shape
            cx = tx + 0.5 * w * warp[0, 0] + 0.5 * h * warp[0, 1]
            cy = ty + 0.5 * w * warp[1, 0] + 0.5 * h * warp[1, 1]
            return Pose(cx, cy, d_theta, r_sx, r_sy, cost=1.0 - cc, confidence=cc)

        p = self.ref_pose
        # The warp's two scales are along the *image* axes; the pose's are along
        # the *template's*. Multiplying one by the other is only correct when
        # those frames coincide -- when the robot happens to lie parallel to the
        # picture -- and the error grows with the angle between them until, at
        # 90 degrees, length and width have swapped and a contraction is
        # reported as no change at all. Measured on a synthetic 12% contraction:
        # 0.2% error with the robot upright, 5% at 40 degrees, and the entire
        # signal gone at 90. That is the discrepancy against the color path,
        # and it is worst on exactly the footage this mode exists for -- a
        # phone held over a dish, where nothing is square to anything.
        #
        # The fix is to ask the warp what it does to the template's own axes
        # rather than reading its image-axis scales and hoping. R takes template
        # directions into the reference image and A maps the reference image
        # into this frame, so the columns of A.R are where the template's x and
        # y axes ended up, and their lengths are the stretches along them.
        A = np.asarray(warp, np.float64)[:, :2]
        c0, s0 = np.cos(p.theta), np.sin(p.theta)
        M = A @ np.array([[c0, -s0], [s0, c0]])
        r_sx = float(np.hypot(M[0, 0], M[1, 0]))
        r_sy = float(np.hypot(M[0, 1], M[1, 1]))

        # Centre: apply the warp to the reference centre directly. The warp maps
        # patch coordinates into this frame's pixels, which is precisely the
        # question being asked, and needs no decomposition to be exact.
        ox, oy = self.origin
        vx, vy = p.tx - ox, p.ty - oy
        nx = A[0, 0] * vx + A[0, 1] * vy + float(warp[0, 2])
        ny = A[1, 0] * vx + A[1, 1] * vy + float(warp[1, 2])
        return Pose(float(nx), float(ny), p.theta + d_theta,
                    p.sx * r_sx, p.sy * r_sy,
                    cost=1.0 - cc, confidence=cc, feature_fit=float("nan"))

    def reset(self) -> None:
        """Return to the reference pose, for restarting a clip."""
        self._warp = self._base.copy()
        self.last_cc = 1.0
