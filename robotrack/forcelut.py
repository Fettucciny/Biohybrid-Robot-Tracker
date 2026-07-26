"""Length -> force conversion from a measured calibration curve.

The tracker measures geometry. Turning a length into a force needs a property of
*your* actuator that no video contains, so it is supplied as a two-column CSV of
a calibration you measured on a rig:

    Length (mm),Force (mN)
    9.80,0.00
    9.60,0.42
    9.40,0.91
    ...

Units are read from the header text rather than assumed. A file that says
``Length (um)`` is treated as micrometres; one that says nothing is treated as
mm and mN, and the assumption is reported rather than made silently -- a
thousand-fold unit error is exactly the kind that survives review because every
number still looks plausible.

Between tabulated points the curve is linearly interpolated. Beyond either end it
is *clamped*, not extrapolated, and the number of frames that landed outside is
reported. A length-force curve is a material measurement with a real domain;
continuing its end slope into lengths you never tested invents stiffness data,
and does it most confidently exactly where the actuator is least linear.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Multipliers to the canonical units (mm for length, mN for force).
LENGTH_UNITS = {
    "m": 1000.0, "meter": 1000.0, "metre": 1000.0, "meters": 1000.0,
    "cm": 10.0, "centimeter": 10.0, "centimetre": 10.0,
    "mm": 1.0, "millimeter": 1.0, "millimetre": 1.0,
    "um": 1e-3, "µm": 1e-3, "μm": 1e-3, "micron": 1e-3, "micrometer": 1e-3,
    "micrometre": 1e-3, "nm": 1e-6,
}
FORCE_UNITS = {
    "n": 1000.0, "newton": 1000.0, "newtons": 1000.0,
    "mn": 1.0, "millinewton": 1.0, "millinewtons": 1.0,
    "un": 1e-3, "µn": 1e-3, "μn": 1e-3, "micronewton": 1e-3, "micronewtons": 1e-3,
    "nn": 1e-6, "kn": 1e6,
    # Mass units appear on load cells that report grams-force.
    "g": 9.80665, "gf": 9.80665, "gram": 9.80665, "grams": 9.80665,
    "mg": 9.80665e-3, "kg": 9806.65,
}


class LUTError(ValueError):
    pass


def _parse_unit(header: str, table: dict, default: str) -> tuple[float, str, bool]:
    """Pull a unit out of a header like ``Length (mm)`` or ``force_mN``.

    Returns ``(multiplier, unit_name, was_explicit)``.
    """
    h = (header or "").strip()
    # Bracketed first: "Length (mm)", "Force [mN]".
    m = re.search(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]", h)
    candidates = []
    if m:
        candidates.append(m.group(1))
    # Then a trailing token: "force_mN", "length mm", "Length-um".
    parts = re.split(r"[\s_\-/,]+", re.sub(r"[\(\[].*?[\)\]]", " ", h))
    candidates += [p for p in reversed(parts) if p]

    for c in candidates:
        key = c.strip().lower().rstrip(".")
        if key in table:
            return table[key], c.strip(), True
    return table[default], default, False


@dataclass
class ForceLUT:
    length_mm: np.ndarray          # ascending, canonical mm
    force_mn: np.ndarray           # canonical mN
    length_unit: str
    force_unit: str
    units_explicit: bool
    source: str

    @property
    def domain_mm(self) -> tuple[float, float]:
        return float(self.length_mm[0]), float(self.length_mm[-1])

    def summary(self) -> str:
        lo, hi = self.domain_mm
        note = "" if self.units_explicit else "  (units not stated in the header — assumed)"
        return (f"{Path(self.source).name}: {len(self.length_mm)} points, "
                f"length {self.length_unit} → mm, force {self.force_unit} → mN, "
                f"covering {lo:.3f}–{hi:.3f} mm{note}")

    def force(self, length_mm) -> tuple[np.ndarray, int]:
        """Interpolate force for each length. Returns ``(force_mN, n_clamped)``.

        NaN lengths -- frames that were gated out as untracked -- stay NaN rather
        than silently becoming the end-of-table force.
        """
        x = np.asarray(length_mm, dtype=float)
        out = np.full(x.shape, np.nan)
        ok = np.isfinite(x)
        if not ok.any():
            return out, 0
        lo, hi = self.domain_mm
        outside = ok & ((x < lo) | (x > hi))
        out[ok] = np.interp(x[ok], self.length_mm, self.force_mn)
        return out, int(outside.sum())


def load_lut(path: str | Path) -> ForceLUT:
    """Read a two-column Length/Force CSV."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LUTError(f"Could not read {p}:\n\n{exc}") from exc

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(text.splitlines(), dialect))
    rows = [r for r in rows if any(str(c).strip() for c in r)]
    if len(rows) < 3:
        raise LUTError(f"{p.name} has fewer than two data rows.")

    header = [str(c).strip() for c in rows[0]]
    if len(header) < 2:
        raise LUTError(f"{p.name} needs two columns: Length and Force.")

    # Locate the columns by name so the order in the file does not matter.
    def find(words):
        for i, h in enumerate(header):
            if any(w in h.lower() for w in words):
                return i
        return None

    li = find(("length", "displacement", "position", "extension", "strain"))
    fi = find(("force", "load", "tension", "stress"))
    if li is None or fi is None or li == fi:
        # Unlabelled file: fall back to positional, and say so via units_explicit.
        li, fi = 0, 1

    lmul, lunit, lexp = _parse_unit(header[li], LENGTH_UNITS, "mm")
    fmul, funit, fexp = _parse_unit(header[fi], FORCE_UNITS, "mn")

    L, F = [], []
    for r in rows[1:]:
        if max(li, fi) >= len(r):
            continue
        try:
            L.append(float(str(r[li]).strip()))
            F.append(float(str(r[fi]).strip()))
        except ValueError:
            continue          # a stray text row is skipped, not fatal
    if len(L) < 2:
        raise LUTError(f"{p.name} contains fewer than two numeric rows.")

    length = np.asarray(L, float) * lmul
    force = np.asarray(F, float) * fmul

    order = np.argsort(length)
    length, force = length[order], force[order]
    # np.interp requires strictly increasing x. Duplicate lengths are averaged
    # rather than dropped, which is what repeated measurements at one point mean.
    uniq, inverse = np.unique(length, return_inverse=True)
    if len(uniq) != len(length):
        force = np.bincount(inverse, weights=force) / np.bincount(inverse)
        length = uniq
    if len(length) < 2:
        raise LUTError(f"{p.name} has only one distinct length.")

    return ForceLUT(length_mm=length, force_mn=force,
                    length_unit=lunit if lexp else "mm (assumed)",
                    force_unit=funit if fexp else "mN (assumed)",
                    units_explicit=lexp and fexp, source=str(p))
