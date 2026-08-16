"""Load a 2D DXF drawing and turn it into a point-sampled tracking template.

The outline you draw in CAD becomes the shape prior the tracker fits to each
frame. Two things this buys you beyond markerless contour tracking:

* Occlusion tolerance. The template knows what the hidden part *should* look
  like, so a robot half-covered by an obstacle still yields a full outline.
* Real units for free. A DXF is drawn to scale, so if the drawing is in
  millimeters the fitted scale factors convert pixels to millimeters without a
  separate ruler calibration (though a ruler is still a good cross-check).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import ezdxf
    from ezdxf import path as ezpath
except ImportError:  # pragma: no cover
    ezdxf = None

# DXF header code 70 -> unit. Only the ones a mechanical drawing realistically uses.
_INSUNITS = {0: ("unitless", 1.0), 1: ("in", 25.4), 2: ("ft", 304.8), 4: ("mm", 1.0),
             5: ("cm", 10.0), 6: ("m", 1000.0), 11: ("angstrom", 1e-7), 13: ("um", 1e-3)}


def signed_distance_grid(pts: np.ndarray, res: int = 192, margin: float = 0.45):
    """Rasterise the template into a signed distance field in its own frame.

    Negative inside the body, positive outside, in template units (mm). This is
    what lets the fitter ask the reverse question -- "is there observed robot
    *outside* my fitted outline?" -- cheaply and differentiably, by mapping
    image pixels back into template space instead of doing point-in-polygon.

    Returns ``(sdf, origin, spacing)`` where ``origin`` is the template-space
    coordinate of grid cell (0,0) and ``spacing`` is mm per cell.
    """
    import cv2

    pts = np.asarray(pts, np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        raise ValueError("this outline has fewer than three usable points")

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = float((hi - lo).max()) * (1.0 + 2 * margin)
    # A degenerate loop -- a zero-radius circle, a zero-length line, a run of
    # identical vertices -- makes span 0, and everything below divides by it.
    # The resulting inf becomes INT_MIN under astype(int32), which numpy does
    # not define and which cv2.fillPoly does not survive: the process dies with
    # no Python traceback and no dialog, which is exactly what "it sometimes
    # crashes when opening a DXF" looked like. Refusing here turns a crash into
    # a sentence.
    if not np.isfinite(span) or span <= 1e-9:
        raise ValueError("this outline has no extent — every point is in the "
                         "same place, so there is no shape to fit")
    center = (lo + hi) / 2.0
    origin = center - span / 2.0
    spacing = span / (res - 1)

    # Clipped as well as guarded, because fillPoly indexes memory from these
    # numbers: a coordinate outside the raster is not a wrong picture, it is a
    # crash, and the guard above only covers the one cause seen so far.
    grid = np.rint((pts - origin) / spacing)
    grid_pts = np.clip(grid, -res, 2 * res).astype(np.int32)
    filled = np.zeros((res, res), np.uint8)
    cv2.fillPoly(filled, [grid_pts], 1)
    if filled.sum() == 0:
        cv2.polylines(filled, [grid_pts], True, 1, 1)

    d_out = cv2.distanceTransform(1 - filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_in = cv2.distanceTransform(filled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = (d_out - d_in).astype(np.float32) * spacing
    return sdf, origin.astype(np.float32), float(spacing)


def outward_normals(pts: np.ndarray) -> np.ndarray:
    """Unit outward normals at each point of a closed, ordered polygon.

    Sign is fixed by testing against the vector from the centroid, which is
    robust for the convex-ish bodies these robots have and avoids depending on
    the winding direction the CAD tool happened to emit.
    """
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    radial = pts - pts.mean(axis=0)
    flip = np.sign((n * radial).sum(axis=1))
    flip[flip == 0] = 1.0
    return (n * flip[:, None]).astype(np.float32)


@dataclass
class Template:
    points: np.ndarray        # (M,2) float32, centered, axis-aligned, in mm
    normals: np.ndarray       # (M,2) float32, unit outward normals
    closed_loops: list[np.ndarray]
    units: str
    width_mm: float           # extent along local x
    length_mm: float          # extent along local y
    source: str
    loop_index: int = 0       # which candidate outline was used
    n_loops: int = 1          # how many were found
    scale: float = 1.0        # user multiplier applied to the drawing
    # Interior structure -- holes, windows, the inner edge of a frame -- sampled
    # as extra points that the fit matches against observed edges. Kept separate
    # from ``points`` because only the outer boundary can carry the containment
    # term: "just outside this edge is background" is true for the silhouette and
    # false for an internal edge with material on both sides.
    feature_points: np.ndarray | None = None
    n_features: int = 0

    def summary(self) -> str:
        which = (f", outline {self.loop_index + 1} of {self.n_loops}"
                 if self.n_loops > 1 else "")
        which += f", scaled x{self.scale:g}" if self.scale != 1.0 else ""
        if self.n_features:
            which += (f", +{self.n_features} interior feature"
                      f"{'s' if self.n_features > 1 else ''} "
                      f"({len(self.feature_points)} pts)")
        return (f"{Path(self.source).name}: {len(self.points)} template points, "
                f"units={self.units}, nominal {self.width_mm:.2f} x "
                f"{self.length_mm:.2f} mm{which}")


def _resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a polygon to n points spaced evenly along its perimeter.

    Even arc-length spacing matters: raw CAD vertices cluster on curves, and
    unequal density would silently weight those regions in the fit.
    """
    p = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    target = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(target, s, p[:, 0])
    y = np.interp(target, s, p[:, 1])
    return np.stack([x, y], axis=1)


def stitch_loops(segs: list[np.ndarray], tol: float) -> list[np.ndarray]:
    """Join separate entities end-to-end into closed loops.

    A CAD outline is very often *not* one closed polyline. Drawing a part with
    the line and arc tools, or exporting a sketch, leaves the boundary as dozens
    of independent LINE and ARC entities that form a loop only geometrically.
    Treating each entity as its own candidate outline then picks the largest
    single arc as "the outer boundary" -- which is how a 58 x 20 mm drawing
    becomes a 0.7 x 3.5 mm template, and the tracker fits a fillet rather than
    the robot. Silently, and with perfectly good-looking confidence numbers.

    Chains are grown greedily from matching endpoints. That is sufficient here
    because a well-formed drawing has at most two segments meeting at a vertex;
    where three do, either choice traces the same outer boundary.
    """
    loops: list[np.ndarray] = []
    open_segs: list[np.ndarray] = []
    for s in segs:
        if len(s) >= 3 and np.linalg.norm(s[0] - s[-1]) <= tol:
            loops.append(s)
        else:
            open_segs.append(s)

    used = [False] * len(open_segs)
    for i in range(len(open_segs)):
        if used[i]:
            continue
        used[i] = True
        chain = open_segs[i].copy()
        grew = True
        while grew:
            grew = False
            if len(chain) >= 3 and np.linalg.norm(chain[0] - chain[-1]) <= tol:
                break
            for j, other in enumerate(open_segs):
                if used[j]:
                    continue
                for cand in (other, other[::-1]):
                    if np.linalg.norm(chain[-1] - cand[0]) <= tol:
                        chain = np.vstack([chain, cand[1:]])
                        used[j] = True
                        grew = True
                        break
                    if np.linalg.norm(chain[0] - cand[-1]) <= tol:
                        chain = np.vstack([cand[:-1], chain])
                        used[j] = True
                        grew = True
                        break
                if grew:
                    break
        if len(chain) >= 3:
            # An unclosed chain is still the best evidence available when nothing
            # closes -- a drawing with one hairline gap should not be unusable.
            loops.append(chain)
    return loops


def _polygon_area(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


@dataclass
class Loop:
    """One candidate outline found in a drawing."""
    points: np.ndarray
    closed: bool
    width_mm: float
    height_mm: float
    encloses: int          # how many other loops sit inside this one
    rectangular: bool
    is_frame: bool         # looks like a sheet border or title block, not a part

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm

    def label(self) -> str:
        tag = "  [sheet border]" if self.is_frame else ("" if self.closed else "  [open]")
        return f"{self.width_mm:.1f} × {self.height_mm:.1f} mm · {len(self.points)} pts{tag}"


def read_loops(path: str | Path, flatten_mm: float = 0.05,
               assume_units: str | None = None) -> tuple[list[Loop], str]:
    """Every closed outline a drawing contains, most part-like first.

    Real drawings are not bare outlines. This one is a full sheet: a 254 x 190.5
    mm page border, a title block, dimension lines, three views of the part. The
    largest closed loop in such a file is the *paper*, so picking the biggest
    loop tracks the page and reports a robot the size of a sheet of A4.

    Sheet furniture is recognized structurally rather than by size: a rectangle
    that encloses most of the other geometry is a border, whatever its
    dimensions. That leaves the part outlines, which are returned largest-first
    and can still be overridden by the caller when a drawing shows several parts.
    """
    if ezdxf is None:
        raise ImportError("ezdxf is required for DXF support:  pip install ezdxf")

    doc = ezdxf.readfile(str(path))
    code = doc.header.get("$INSUNITS", 0)
    units, to_mm = _INSUNITS.get(code, ("unitless", 1.0))
    if assume_units:
        units, to_mm = assume_units, {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}[assume_units]
    elif units == "unitless":
        units, to_mm = "mm (assumed)", 1.0

    segments: list[np.ndarray] = []
    for e in doc.modelspace():
        # Annotation is not geometry. Dimension leaders and title-block text
        # would otherwise stitch into spurious loops.
        if e.dxftype() in ("MTEXT", "TEXT", "DIMENSION", "LEADER", "MLEADER",
                           "HATCH", "POINT", "ATTDEF"):
            continue
        try:
            p = ezpath.make_path(e)
        except (TypeError, ValueError):
            continue
        v = np.array([(pt.x, pt.y) for pt in p.flattening(distance=flatten_mm)], float)
        if len(v) >= 2:          # a two-point LINE is a legitimate boundary piece
            segments.append(v * to_mm)

    if not segments:
        raise ValueError(
            f"No usable geometry in {path}. Supported: LINE, ARC, CIRCLE, ELLIPSE, "
            "LWPOLYLINE, POLYLINE, SPLINE in modelspace. If the drawing lives in a "
            "block, explode it first."
        )

    # Endpoint tolerance, scaled to the drawing: generous enough to absorb the
    # rounding in an exported DXF, far too small to bridge a real gap.
    extent = float(max(np.ptp(np.vstack(segments), axis=0)))
    tol = max(flatten_mm * 2.0, extent * 1e-4)
    raw = stitch_loops(segments, tol)
    if not raw:
        raise ValueError(f"No outline could be assembled from {path}.")

    centroids = [v.mean(axis=0) for v in raw]
    loops: list[Loop] = []
    for i, v in enumerate(raw):
        w, h = float(np.ptp(v[:, 0])), float(np.ptp(v[:, 1]))
        closed = bool(np.linalg.norm(v[0] - v[-1]) <= tol)
        lo, hi = v.min(axis=0), v.max(axis=0)
        encloses = sum(1 for j, c in enumerate(centroids)
                       if j != i and (lo <= c).all() and (c <= hi).all())
        bbox = max(w * h, 1e-12)
        rectangular = closed and _polygon_area(v) / bbox > 0.92 and len(v) <= 8
        loops.append(Loop(points=v, closed=closed, width_mm=w, height_mm=h,
                          encloses=encloses, rectangular=rectangular,
                          is_frame=rectangular and encloses >= 2))

    # Part outlines first, then by size. A closed loop outranks an open chain
    # because an open chain is a drawing defect, not a shape.
    loops.sort(key=lambda L: (not L.is_frame, L.closed, L.area_mm2), reverse=True)
    return loops, units


def describe_loops(path: str | Path, **kw) -> str:
    loops, units = read_loops(path, **kw)
    return "\n".join(f"[{i}] {L.label()}" for i, L in enumerate(loops))


def _perimeter(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([v, v[:1]]), axis=0), axis=1).sum())


def select_features(loops: list, outer_idx: int, min_frac: float = 0.01,
                    max_frac: float = 0.97, max_features: int = 6) -> list[int]:
    """Interior loops worth fitting to, largest first.

    A loop qualifies if it closes, sits inside the outer boundary, and covers a
    sensible fraction of it. The lower bound rejects fillet arcs and drafting
    marks whose position in the video is not resolvable anyway; the upper bound
    rejects a duplicate of the outer boundary itself, which would double its
    weight rather than add information.
    """
    outer = loops[outer_idx]
    lo, hi = outer.points.min(axis=0), outer.points.max(axis=0)
    area_outer = max(outer.width_mm * outer.height_mm, 1e-12)

    picks = []
    for i, L in enumerate(loops):
        if i == outer_idx or not L.closed:
            continue
        c = L.points.mean(axis=0)
        if not ((lo <= c).all() and (c <= hi).all()):
            continue
        frac = (L.width_mm * L.height_mm) / area_outer
        if min_frac <= frac <= max_frac:
            picks.append((frac, i))
    picks.sort(reverse=True)
    return [i for _, i in picks[:max_features]]


def load_dxf(path: str | Path, n_points: int = 400, flatten_mm: float = 0.05,
             assume_units: str | None = None, loop_index: int | None = None,
             scale: float = 1.0, use_features: bool = True,
             feature_points: int = 300) -> Template:
    """Read a DXF and produce a normalized template.

    ``flatten_mm`` is the chord tolerance used when converting arcs, splines and
    ellipses to polylines -- 0.05 mm is far below any measurement you will make
    from video, so curve discretisation contributes no error.

    ``scale`` multiplies the drawing's dimensions. It matters more than it looks:
    the robot's width is the calibration ruler, so a drawing at the wrong scale
    biases every micrometer in the output by the same factor. This is the dial
    for a drawing in the wrong units, at a detail scale, or of a design that was
    fabricated slightly larger or smaller than drawn.

    ``use_features`` also samples the drawing's interior structure -- holes,
    windows, the inner edge of a frame -- as additional points for the fit to
    match against observed edges. The silhouette alone constrains position,
    rotation and the two scales, but says nothing about the inside of the body,
    so a mask whose boundary is a little wrong has nothing to correct it. With
    interior features the fit is over-determined: it can be pulled back into
    place by structure that thresholding renders more reliably than the outer
    edge, which is what makes a more aggressive threshold affordable.

    ``loop_index`` selects among the candidate outlines reported by
    ``read_loops``; the default takes the best automatic guess. A drawing that
    shows several parts, or several views of one part, is genuinely ambiguous,
    and guessing silently is worse than letting the choice be made explicitly.
    """
    loops, units = read_loops(path, flatten_mm=flatten_mm, assume_units=assume_units)
    idx = 0 if loop_index is None else max(0, min(int(loop_index), len(loops) - 1))
    k = float(scale) if scale and scale > 0 else 1.0
    outer = loops[idx].points * k
    all_pts = [L.points * k for L in loops]

    feats: list[np.ndarray] = []
    if use_features and feature_points > 0:
        chosen = select_features(loops, idx)
        raw = [loops[i].points * k for i in chosen]
        total = sum(_perimeter(v) for v in raw)
        if total > 0:
            for v in raw:
                # Points shared out by perimeter, so a long edge is not sampled
                # more sparsely than a short one and no feature dominates by
                # accident of how the CAD tool happened to tessellate it.
                n = int(round(feature_points * _perimeter(v) / total))
                if n >= 8:
                    feats.append(_resample_closed(v, n))

    pts = _resample_closed(outer, n_points)
    # One origin for everything this function returns.
    #
    # This is the arc-length centroid of the outline, taken from the
    # *resampled* polygon; the interior features and the drawn loops below are
    # shifted by the same vector. They used to be centred on
    # ``outer.mean(axis=0)`` instead -- the mean of the raw DXF vertices -- and
    # that is a different point, sometimes very different. A rounded rectangle
    # exported as nine vertices has most of them bunched at one end, so their
    # mean sits nowhere near the shape's centre: on the reference drawing the
    # two origins are 0.81 mm apart, 5.6% of the part's width. The outline then
    # fitted the robot correctly while every interior feature sat off to one
    # side, always by the same amount and always in the same direction -- which
    # is exactly what it looked like.
    origin = pts.mean(axis=0)
    pts = pts - origin

    # Validate before anything downstream touches it. Everything after this --
    # the SVD, the normals, the distance grid, the fitter's scale limits --
    # divides by some property of this outline, and a drawing can perfectly
    # legally contain a zero-radius circle or a degenerate loop. Naming the
    # problem here is the difference between a message and a silent process
    # death several calls later.
    if not np.isfinite(pts).all():
        raise ValueError("this outline contains coordinates that are not "
                         "numbers — the drawing is malformed")
    w_mm, l_mm = float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))
    if max(w_mm, l_mm) <= 1e-9:
        raise ValueError("this outline has no extent. Pick a different outline "
                         "under Outline, or check the drawing's units.")

    # Rotate so the long axis is local +y. This makes the fitted scale factors
    # directly interpretable as (width, length) rather than an arbitrary pair.
    u, s, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    major = vt[0]
    ang = np.arctan2(major[1], major[0]) - np.pi / 2
    c, sn = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -sn], [sn, c]])
    pts = pts @ R.T

    feature_pts = None
    if feats:
        f = np.vstack(feats) - origin
        feature_pts = (f @ R.T).astype(np.float32)

    return Template(
        points=pts.astype(np.float32),
        normals=outward_normals(pts),
        closed_loops=[(l - origin) @ R.T for l in all_pts],
        units=units,
        width_mm=float(np.ptp(pts[:, 0])),
        length_mm=float(np.ptp(pts[:, 1])),
        source=str(path),
        loop_index=idx,
        n_loops=len(loops),
        scale=k,
        feature_points=feature_pts,
        n_features=len(feats),
    )
