"""Compare recovered tracking against the synthetic ground truth.

    python tests/make_synthetic.py --outdir /tmp/synth
    python -m robotrack.cli /tmp/synth/synthetic_30fps.mp4 \
        --dxf /tmp/synth/robot.dxf --outdir /tmp/synth/out
    python tests/evaluate.py /tmp/synth/out/tracking.csv /tmp/synth/truth_30.npy

Width is no longer a per-frame output -- it is the *ruler*, held fixed and used
to convert pixels to millimetres -- so it cannot be scored frame by frame the
way length and position are. What replaces that check is stricter, not weaker:
the recovered ``calibration_px_per_mm`` in run_info.json is compared against the
generator's true scale, and the width's coefficient of variation is checked to
confirm the thing being used as a ruler really is rigid. A calibration error
would bias every millimetre figure in the table, so it is the more important
number of the two.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_synthetic import FREQ_HZ, PX_PER_MM, TRAVEL_MM_S


def evaluate(csv: str, truth_npy: str, seconds: float) -> dict:
    t = pd.read_csv(csv)
    gt = np.load(truth_npy)          # frame, t, cx_mm, cy_mm, len_mm, wid_mm, occl
    n = min(len(t), len(gt))
    t, gt = t.iloc[:n], gt[:n]
    occl = gt[:, 6] > 0.5

    def err(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return (np.nanmean(np.abs(a[m] - b[m])), np.nanmax(np.abs(a[m] - b[m])),
                np.nanmean(np.abs(a[m] - b[m]) / np.abs(b[m])) * 100)

    res = {}
    quantities = [
        ("length_mm", t.length_mm.to_numpy(), gt[:, 4]),
        ("cx_mm",     t.cx_mm.to_numpy(),     gt[:, 2]),
        ("cy_mm",     t.cy_mm.to_numpy(),     gt[:, 3]),
    ]
    res["names"] = [q[0] for q in quantities]
    for name, got, want in quantities:
        mae, mx, pct = err(got, want)
        res[name] = dict(mae=mae, max=mx, pct=pct)
        mo, _, po = err(got[occl], want[occl]) if occl.any() else (np.nan,) * 3
        res[name]["mae_occluded"] = mo
        res[name]["pct_occluded"] = po

    # True path length is the arc length of the prescribed centroid trajectory,
    # not the net displacement -- the body also oscillates laterally.
    true_path = float(np.sum(np.hypot(np.diff(gt[:, 2]), np.diff(gt[:, 3]))))
    # Calibration, read from the run's own report rather than re-derived here.
    info_path = Path(csv).with_name("run_info.json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    res["px_per_mm"] = dict(got=float(info.get("calibration_px_per_mm", np.nan)),
                            want=PX_PER_MM,
                            width_cv=float(info.get("width_cv", np.nan)))
    res["path_length"] = dict(got=float(t.path_length.iloc[-1]), want=true_path,
                              net_got=float(np.hypot(t.cx_mm.iloc[-1] - t.cx_mm.iloc[0],
                                                     t.cy_mm.iloc[-1] - t.cy_mm.iloc[0])),
                              net_want=float(np.hypot(gt[-1, 2] - gt[0, 2],
                                                      gt[-1, 3] - gt[0, 3])))
    res["tracked_frac"] = float((t.confidence >= 0.5).mean())
    res["occluded_frac"] = float(occl.mean())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("truth"); ap.add_argument("--seconds", type=float, default=6)
    a = ap.parse_args()
    r = evaluate(a.csv, a.truth, a.seconds)
    print(f"{'quantity':<12} {'MAE':>9} {'MAE%':>7} {'max':>8} {'MAE(occl)':>10} {'%(occl)':>8}")
    print("-" * 60)
    for k in r["names"]:
        v = r[k]
        print(f"{k:<12} {v['mae']:>9.4f} {v['pct']:>6.2f}% {v['max']:>8.3f} "
              f"{v['mae_occluded']:>10.4f} {v['pct_occluded']:>7.2f}%")
    c = r["px_per_mm"]
    if np.isfinite(c["got"]):
        print(f"\ncalibration : {c['got']:.3f} px/mm (truth {c['want']:.3f}, "
              f"{100*(c['got']-c['want'])/c['want']:+.2f}%), "
              f"width CV {100*c['width_cv']:.2f}%")
    else:
        print("\ncalibration : run_info.json not found beside the CSV")

    p = r["path_length"]
    print(f"\npath length : {p['got']:.2f} mm (truth {p['want']:.2f} mm, "
          f"{100*(p['got']-p['want'])/p['want']:+.1f}%)")
    print(f"net displ.  : {p['net_got']:.2f} mm (truth {p['net_want']:.2f} mm, "
          f"{100*(p['net_got']-p['net_want'])/p['net_want']:+.1f}%)")
    print(f"contraction : truth {FREQ_HZ} Hz")
    print(f"tracked     : {100*r['tracked_frac']:.0f}% of frames "
          f"({100*r['occluded_frac']:.0f}% partially occluded)")
