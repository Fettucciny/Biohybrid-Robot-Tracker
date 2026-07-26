"""Markerless measurement straight from the mask, for when no DXF is supplied.

Faster and assumption-free, but it measures only what it can see: during an
occlusion the observed length shrinks, because the hidden part is genuinely
missing from the mask. The confidence score exposes this, and the CAD path in
register.py is the answer when you need to measure through occlusions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    cx: float
    cy: float
    width_px: float
    length_px: float
    theta: float
    area_px: float
    confidence: float


def measure_mask(mask: np.ndarray, pct: float = 1.0,
                 expected_area: float | None = None) -> Measurement | None:
    """Principal-axis extents of the mask.

    ``pct`` trims that percentage from each end of the projected extent, so a
    handful of stray pixels cannot inflate the reported length. Set it to 0 for
    strict min/max.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 10:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    cen = pts.mean(0)
    cov = np.cov((pts - cen).T)
    w, v = np.linalg.eigh(cov)
    i_major, i_minor = int(np.argmax(w)), int(np.argmin(w))
    proj = (pts - cen) @ v

    def extent(col):
        a = proj[:, col]
        return float(np.percentile(a, 100 - pct) - np.percentile(a, pct))

    major = v[:, i_major]
    area = float(xs.size)
    conf = 1.0 if expected_area is None else float(
        np.clip(1.0 - abs(area - expected_area) / max(expected_area, 1.0), 0.0, 1.0))
    return Measurement(
        cx=float(cen[0]), cy=float(cen[1]),
        width_px=extent(i_minor), length_px=extent(i_major),
        theta=float(np.arctan2(major[1], major[0]) - np.pi / 2),
        area_px=area, confidence=conf,
    )
