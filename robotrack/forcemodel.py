"""Force from beam mechanics, as an alternative to a measured lookup table.

The model
---------
This is the bio-bot force calculation used in the Bashir-lab designs: a compliant
beam spanning two legs, with a muscle ring anchored a short distance below the
beam's neutral axis. When the muscle contracts it pulls the leg tips together;
the legs rotate like rigid links, that rotation is imposed on the ends of the
beam, and the beam's bending stiffness is what resists. Read backwards, the
measured pull-in gives the force.

    I     = t^3 * w / 12                  second moment of area of the beam
    theta = asin( (delta / 2) / L_leg )   rotation of each leg
    M     = 2 * E * I * theta / L         Euler-Bernoulli, end rotation form
    F     = M / l = 2 * E * I * theta / (l * L)

``delta`` is the shortening from rest: how much closer the legs are now than at
their resting separation.

Why this form rather than the published one
-------------------------------------------
Cvetkovic et al. (PNAS 2014) write the same physics as

    P = 8 * E * I * d_max / (l * L^2)

in terms of the beam's *transverse mid-span* deflection ``d_max``. The two are
algebraically identical -- substituting ``d_max = theta * L / 4`` turns one into
the other exactly, checked numerically to 0.00% over deflections from 0.05 to
1.5 mm.

The end-rotation form is used here because it takes the quantity a tracker can
actually measure. Video from above gives the change in leg separation; it does
not give the out-of-plane bow of the beam. Converting pull-in to leg rotation
through the leg length is the step that connects the two, and it is exactly what
``SampleForce.m`` does.

Units
-----
With ``E`` in pascals and every length in millimetres, ``E*I/(l*L)`` is
Pa*mm^2 = 1e-6 N, so **force comes out in micronewtons** with no conversion
factor. That is the unit the literature reports these in -- 395 uN of active
tension and 534-1147 uN of passive tension in the PNAS paper -- and it is why
``SampleForce.m`` uses a peak prominence of 200 without further scaling.

Assumptions worth knowing
-------------------------
* Linear elastic beam, constant cross-section, symmetric bending.
* The legs are rigid relative to the beam and rotate about their base.
* The muscle pulls in the plane of the beam at a fixed moment arm ``l``.
* ``E`` is the hydrogel's modulus at the moment of measurement. It is the least
  certain number in the calculation by a wide margin: force scales linearly with
  it, and the modulus of a cast hydrogel varies batch to batch and drifts in
  culture. A force from this model is only as good as that value.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class BeamForceModel:
    """Beam-bending force model. Defaults are the values in ``SampleForce.m``."""

    E_pa: float = 293e3            # Young's modulus of the beam material
    thickness_mm: float = 1.1      # beam thickness, in the bending direction
    beam_width_mm: float = 1.925   # beam width
    L_mm: float = 8.25             # leg centre to leg centre
    l_mm: float = 1.642            # beam neutral axis to muscle line of action
    leg_long_mm: float = 4.125     # legs are tapered; both ends are averaged
    leg_short_mm: float = 3.3
    resting: str = "max"           # "max" (as in the .m file) or "median"

    @property
    def I_mm4(self) -> float:
        return self.thickness_mm ** 3 * self.beam_width_mm / 12.0

    @property
    def L_leg_mm(self) -> float:
        return (self.leg_long_mm + self.leg_short_mm) / 2.0

    @property
    def stiffness(self) -> float:
        """``2EI/(lL)`` -- force per radian of leg rotation, in uN/rad."""
        return 2.0 * self.E_pa * self.I_mm4 / max(self.l_mm * self.L_mm, 1e-12)

    def summary(self) -> str:
        return (f"beam model: E {self.E_pa / 1e3:.0f} kPa, "
                f"I {self.I_mm4:.5f} mm⁴ ({self.thickness_mm}×{self.beam_width_mm} mm), "
                f"L {self.L_mm} mm, l {self.l_mm} mm, leg {self.L_leg_mm:.3f} mm "
                f"→ {self.stiffness:.0f} µN/rad, rest = {self.resting}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(I_mm4=self.I_mm4, L_leg_mm=self.L_leg_mm,
                 stiffness_un_per_rad=self.stiffness)
        return d

    # ---- the calculation ----------------------------------------------

    def resting_length_mm(self, length_mm) -> float:
        """The separation the deflection is measured from.

        ``SampleForce.m`` uses the maximum, which is right when the muscle only
        ever shortens the robot. It is also the single most extreme sample in the
        recording, so one over-long frame sets the baseline for the whole clip
        and biases every force upward. The median of the upper quartile is
        offered as the robust alternative.
        """
        v = np.asarray(length_mm, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        if self.resting == "median":
            return float(np.median(v[v >= np.percentile(v, 75)]))
        return float(np.max(v))

    def force_un(self, length_mm, resting_mm: float | None = None):
        """Force in micronewtons for each measured length.

        Returns ``(force_uN, deflection_mm, resting_mm, n_clamped)``. Lengths
        longer than rest give a negative deflection and hence a negative force,
        which is kept rather than clipped: it is the honest reading of a robot
        that is momentarily longer than its baseline, and silently flooring it at
        zero would hide a badly chosen resting length.
        """
        v = np.asarray(length_mm, float)
        rest = self.resting_length_mm(v) if resting_mm is None else float(resting_mm)
        deflection = rest - v

        # asin's domain. A pull-in of more than twice the leg length is not
        # geometry this model describes, so it is clamped and counted rather
        # than returned as NaN.
        ratio = (deflection / 2.0) / max(self.L_leg_mm, 1e-12)
        n_clamped = int(np.sum(np.abs(ratio[np.isfinite(ratio)]) > 1.0))
        theta = np.arcsin(np.clip(ratio, -1.0, 1.0))
        force = self.stiffness * theta
        return force, deflection, rest, n_clamped


def force_from_delta_un(model: BeamForceModel, delta_mm: float) -> float:
    """Force for a single shortening, for the region-analysis readout."""
    theta = np.arcsin(np.clip((delta_mm / 2.0) / max(model.L_leg_mm, 1e-12), -1.0, 1.0))
    return float(model.stiffness * theta)
