"""Two marked points on the robot, tracked independently of the body.

Why this exists
---------------

The Cvetkovic beam model asks for one number from the video: **δ, how much
closer the two legs are than at rest.** Everything else in the force equation is
fixed geometry. Until now the app answered that question with the change in the
robot's overall length -- the long-axis extent of the mask in markerless mode,
the fitted template's length in CAD mode -- and that is not the same quantity.

It is not the same quantity for a reason that no amount of tuning fixes. Both of
the app's existing measurements are *whole-body* measurements: the mask envelope
and the 5-DOF similarity fit both describe the robot as one shape that
translates, rotates and scales. A muscle-driven robot does not deform that way.
The beam does not shorten; the legs pivot inward about their bases and the rest
of the body goes along for the ride. A similarity fit has no parameter for that,
so it spreads the leg closure over the whole outline and reports a fraction of
it. Marking the legs in the drawing and mapping them through the fitted pose
would not help either, and it is worth being explicit about why: under a pose
``(tx, ty, θ, sx, sy)`` two template-fixed points separated by ``(dx, dy)`` end
up ``hypot(sx·dx, sy·dy)`` apart, which for a pair lying along the long axis is
just ``sy·dy`` -- a *constant multiple* of the fitted length. It carries no
information the length column did not already have.

Measured against a colleague's frame-by-frame manual tracking of the same clip,
the envelope reported 0.759 mm of length change per 1 mm of real leg closure,
with a 5.76 mm intercept. The correlation was 0.993: the app was following the
motion perfectly and reporting the wrong quantity, which is the failure mode
that does not look like one.

So the two points are tracked in the image, on their own, exactly as a person
doing it by hand would. On that same clip this recovers a slope of **0.998**
against the manual trace (r = 0.995), and a median contraction amplitude of
1.2287 mm against 1.2689 -- a force of 5500 µN where the manual method gives
5589 and the envelope gives 4452.

How it tracks
-------------

Each mark owns a small square patch of the reference frame, and every subsequent
frame is aligned to *that* patch by ECC -- never to the previous frame. Chaining
frame to frame is the obvious implementation and it accumulates drift: a tenth
of a pixel of bias per frame is 90 px over a 900-frame clip, which is larger
than the signal. Matching the fixed reference costs nothing extra and cannot
drift, because there is no path for an error to accumulate along. The previous
position is still used, but only to *place the search window*, where being wrong
costs a slower solve rather than a wrong answer.

Three further choices, each of which was the difference between working and not:

**Translation only.** A leg pad rotates by a couple of degrees over a whole
contraction. Solving for rotation and scale as well means four parameters
fitted to a 69 px patch of low-texture tissue, and the extra freedom goes into
absorbing noise. The quantity wanted here is a position.

**Both patches normalised to zero mean and unit variance.** These clips brighten
and dim over thirty seconds as the medium settles and the lamp drifts. ECC's
correlation coefficient is invariant to that in principle; normalising makes it
so in practice, at the numerical precision the solver actually works to.

**A solve that reaches the edge of its search window is rejected, not used.**
That is what a lost mark looks like -- the patch has left the window and ECC has
locked onto whatever texture is nearest the boundary. Holding the last good
position and flagging the frame is recoverable; accepting the answer writes a
plausible number that is wrong, and the run gives no sign of it.

The signal tracked is the same colour-distance surface the segmenter already
built for the frame, so this costs one ECC solve per mark per frame and no
additional decoding or colour conversion. Measured at ~25 fps for two marks on
724x724 frames, against a pipeline that runs at 23-29 fps -- it is not the
bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Half-width of the patch that identifies a mark, and how far beyond it the
# search may reach. 34 px is about a leg pad on these clips at 35 px/mm -- big
# enough to hold structure that is unique in its neighbourhood, small enough
# that the patch is all robot and not half medium. The 18 px pad covers a
# contraction and a half at 1 Hz; a mark that moves further than that between
# consecutive frames has not moved, it has been lost.
PATCH_HALF = 34
SEARCH_PAD = 18

# Both patches are halved before the solve, and the result scaled back up.
#
# ECC's cost is per pixel per iteration, so a 105x105 search window at full
# resolution is four times the work of the same window at half, and the first
# version of this spent 13 ms a frame on two marks -- enough to take a 29 fps
# pipeline down to 20 and be noticed immediately as a slowdown. Halving costs
# 2.5 ms for an answer that differs from the exact solve by 0.18 px rms, on a
# signal whose contraction amplitude is 44 px. The measured force moved by 0.4%.
#
# What is *not* free is the resampling phase: INTER_AREA on a window cropped to
# an arbitrary integer origin resamples on a different half-pixel grid each
# frame, which injects jitter that has nothing to do with the robot. Cropping on
# an even boundary pins the phase and takes the error from 0.19 px to 0.18 --
# small here, and free, and the kind of thing that is much larger on a quieter
# signal.
DOWNSCALE = 2

# 30 iterations at 1e-3 rather than 50 at 1e-4: the tail of the solve is
# refining a translation far below the resolution the answer is used at.
_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)


def _prep(a: np.ndarray) -> np.ndarray:
    """Halve, then normalise to zero mean and unit variance."""
    if DOWNSCALE > 1:
        a = cv2.resize(a, None, fx=1.0 / DOWNSCALE, fy=1.0 / DOWNSCALE,
                       interpolation=cv2.INTER_AREA)
    a = a.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return np.ascontiguousarray(a, dtype=np.float32)


def _crop(img: np.ndarray, cx: float, cy: float, half: int):
    """A (2·half+1)² window near (cx, cy), with its origin.

    Returns ``(patch, x0, y0)``, or ``(None, 0, 0)`` if it would leave the
    frame. The origin is snapped down to a multiple of ``DOWNSCALE`` so every
    crop resamples on the same grid; the caller is told where the window
    actually starts rather than assuming it is centred, which is what keeps the
    sub-pixel answer honest.
    """
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    if DOWNSCALE > 1:
        x0 -= x0 % DOWNSCALE
        y0 -= y0 % DOWNSCALE
    h, w = img.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + 2 * half + 1 > w or y0 + 2 * half + 1 > h:
        return None, 0, 0
    return img[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1], x0, y0


@dataclass
class LegSample:
    """Where the two marks are in one frame."""
    ax: float
    ay: float
    bx: float
    by: float
    ok: bool                # both marks solved this frame
    # ECC's correlation coefficient for each mark, 0..1, or nan on a frame that
    # held. The tracker's own answer to "how well did this patch match what it
    # is supposed to be" -- a different question from the shape fit's
    # confidence, and the one that matters when the marks are what the force is
    # measured from.
    cc_a: float = float("nan")
    cc_b: float = float("nan")

    @property
    def separation_px(self) -> float:
        return float(np.hypot(self.ax - self.bx, self.ay - self.by))


def parse_marks(marks) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """``[[ax, ay], [bx, by]]`` from any source, or None if it is not that.

    Settings files are hand-edited and configs travel between builds, so this
    accepts a flat four-element form as well and refuses anything else rather
    than raising halfway through a run.
    """
    if marks is None:
        return None
    try:
        v = [float(x) for x in np.ravel(np.asarray(marks, dtype=float))]
    except (TypeError, ValueError):
        return None
    if len(v) != 4 or not all(np.isfinite(v)):
        return None
    if np.hypot(v[0] - v[2], v[1] - v[3]) < 2.0:
        return None                     # both marks on the same spot
    return (v[0], v[1]), (v[2], v[3])


class LegTracker:
    """Follows two marked points through a clip.

    ``marks`` are in the *full-resolution* frame's pixels, the same convention
    the region of interest uses, and are multiplied by ``scale`` here. Storing
    them at full resolution is what lets the decode scale be changed between
    runs without the marks moving off the robot.
    """

    def __init__(self, signal: np.ndarray, marks, scale: float = 1.0,
                 half: int = PATCH_HALF, pad: int = SEARCH_PAD):
        m = parse_marks(marks)
        if m is None:
            raise ValueError("leg marks must be two distinct points")
        k = float(scale) if scale and scale > 0 else 1.0
        self.half = max(8, int(round(half * k)))
        self.pad = max(4, int(round(pad * k)))
        self.pos = [[m[0][0] * k, m[0][1] * k], [m[1][0] * k, m[1][1] * k]]
        self.rest_px = float(np.hypot(self.pos[0][0] - self.pos[1][0],
                                      self.pos[0][1] - self.pos[1][1]))
        self.n_lost = 0
        self.n_widened = 0
        # The largest single-frame step either mark took. Reported so a run can
        # be checked against the window it was solved in rather than assumed to
        # have fitted comfortably inside it.
        self.max_step = 0.0

        # A mark near the frame edge shrinks its patch rather than failing the
        # run. Refusing outright was the first version and it was wrong: the
        # legs are the ends of the robot, so they are exactly the features most
        # likely to sit near the edge of a tightly framed clip, and losing the
        # whole measurement over a few pixels of margin is a worse answer than
        # matching on a smaller patch. Both marks shrink together -- they are
        # compared with each other, and two different patch sizes would give
        # them different noise floors.
        fit = self.half
        while fit >= 10 and any(_crop(signal, cx, cy, fit)[0] is None
                                for cx, cy in self.pos):
            fit -= 4
        if fit < 10:
            raise ValueError(
                "a leg mark sits on the frame edge with no room for a patch "
                "around it; mark a point further inside the frame")
        self.clipped = fit < self.half
        self.half = fit

        # Each reference keeps the mark's offset *inside its own patch*, because
        # the crop origin is snapped to the downscale grid and so the mark is
        # not exactly at the patch centre. Assuming it was is a sub-pixel error
        # that would be baked into every frame of the clip in the same
        # direction, which is the only kind that survives averaging.
        self.refs = []
        for cx, cy in self.pos:
            patch, x0, y0 = _crop(signal, cx, cy, self.half)
            self.refs.append((_prep(patch), cx - x0, cy - y0))

    def _solve(self, signal: np.ndarray, k: int, pad: int):
        """One ECC solve for mark ``k`` in a window of the given pad.

        Returns the mark's new full-resolution position, or None if the solve
        did not land inside the window.
        """
        ref, rx, ry = self.refs[k]
        cx, cy = self.pos[k]
        win, x0, y0 = _crop(signal, cx, cy, self.half + pad)
        if win is None:
            return None
        ds = DOWNSCALE
        # Where the reference patch's own origin would sit in this window if the
        # mark had not moved since the last frame. Not simply "+pad": the window
        # origin is snapped to the downscale grid, so the offset has to be
        # computed rather than assumed.
        W = np.array([[1, 0, (cx - x0 - rx) / ds],
                      [0, 1, (cy - y0 - ry) / ds]], np.float32)
        try:
            cc, W = cv2.findTransformECC(ref, _prep(win), W,
                                         cv2.MOTION_TRANSLATION, _CRITERIA,
                                         None, 1)
        except cv2.error:
            return None
        nx = x0 + rx + float(W[0, 2]) * ds
        ny = y0 + ry + float(W[1, 2]) * ds
        if not (np.isfinite(nx) and np.isfinite(ny)):
            return None
        # Touching the wall of the window is not a measurement. It is what a
        # lost mark looks like: the patch has left the window and the solver has
        # locked onto whatever texture is nearest the boundary. Rejecting it is
        # what lets the wider retry below mean something.
        if abs(nx - cx) >= pad or abs(ny - cy) >= pad:
            return None
        return nx, ny, float(cc)

    def track(self, signal: np.ndarray | None) -> LegSample:
        """Locate both marks in this frame. Never raises; holds on failure.

        A mark that hits the wall of its window gets one retry in a window
        twice as wide before being given up on. Fast contractions are why: the
        window has to cover the distance a leg travels *between frames*, and
        that distance goes as frequency times amplitude. A 1 Hz beat moves a
        leg two or three pixels per frame and a 4 Hz beat moves it four times
        as far. Sizing the window for the worst case all the time would mean
        paying for it on every frame of every clip and admitting four times as
        much rival texture into every solve; escalating only when the narrow
        window actually failed costs nothing on the frames that did not need it.
        """
        if signal is None:
            return LegSample(*self.pos[0], *self.pos[1], ok=False)
        ok = True
        cc = [float("nan"), float("nan")]
        for k in range(len(self.refs)):
            cx, cy = self.pos[k]
            d = self._solve(signal, k, self.pad)
            if d is None:
                d = self._solve(signal, k, 2 * self.pad)
                if d is not None:
                    self.n_widened += 1
            if d is None:
                ok = False
                continue
            self.max_step = max(self.max_step,
                                float(np.hypot(d[0] - cx, d[1] - cy)))
            self.pos[k] = [d[0], d[1]]
            cc[k] = d[2]
        if not ok:
            self.n_lost += 1
        return LegSample(*self.pos[0], *self.pos[1], ok=ok,
                         cc_a=cc[0], cc_b=cc[1])
