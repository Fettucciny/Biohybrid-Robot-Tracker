"""Time-resolved analysis. Every window here is specified in milliseconds.

This is the module that makes automatic 30/60/120 Hz handling meaningful. A
"5-frame smoothing window" means 167 ms at 30 Hz and 42 ms at 120 Hz -- three
recordings of the same robot would give three different answers. Specifying
windows physically and converting through the measured frame rate makes results
comparable across recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .ingest import VideoInfo


@dataclass
class AnalysisConfig:
    smooth_ms: float = 100.0        # Savitzky-Golay window, physical time
    polyorder: int = 2
    min_confidence: float = 0.5     # below this a frame is treated as unobserved
    max_gap_ms: float = 400.0       # longer dropouts are left as NaN, not bridged
    detrend_drift: bool = True      # separate contraction from net locomotion


def _sg_window(info: VideoInfo, cfg: AnalysisConfig) -> tuple[int, int]:
    n = int(round(cfg.smooth_ms / 1000.0 * info.measured_fps))
    n = max(cfg.polyorder + 2, n)
    if n % 2 == 0:
        n += 1
    return n, cfg.polyorder


def gate_and_fill(values: np.ndarray, conf: np.ndarray, t: np.ndarray,
                  cfg: AnalysisConfig) -> tuple[np.ndarray, np.ndarray]:
    """Blank low-confidence samples, then bridge only the short gaps.

    Interpolating across a long dropout invents data. Short gaps -- a fraction
    of a contraction cycle -- are safe to bridge linearly; anything longer stays
    NaN so it is visibly missing in the output rather than quietly fabricated.
    """
    v = values.astype(float).copy()
    bad = (conf < cfg.min_confidence) | ~np.isfinite(v)
    v[bad] = np.nan
    good = ~np.isnan(v)
    if good.sum() < 2:
        return v, bad
    filled = v.copy()
    idx = np.flatnonzero(bad)
    # Group consecutive bad samples into runs and bridge only short ones.
    for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        if run.size == 0:
            continue
        a, b = run[0] - 1, run[-1] + 1
        if a < 0 or b >= len(v) or not (good[a] and good[b]):
            continue
        if (t[b] - t[a]) * 1000.0 <= cfg.max_gap_ms:
            filled[run] = np.interp(t[run], [t[a], t[b]], [v[a], v[b]])
    return filled, bad


def smooth(values: np.ndarray, info: VideoInfo, cfg: AnalysisConfig) -> np.ndarray:
    n, po = _sg_window(info, cfg)
    v = values.copy()
    ok = np.isfinite(v)
    if ok.sum() < n:
        return v
    out = np.full_like(v, np.nan)
    out[ok] = signal.savgol_filter(v[ok], min(n, (ok.sum() // 2) * 2 - 1), po)
    return out


def derivative(values: np.ndarray, t: np.ndarray, info: VideoInfo,
               cfg: AnalysisConfig) -> np.ndarray:
    """Savitzky-Golay analytic derivative, in units per second.

    Differentiating with the filter rather than after it avoids amplifying the
    per-frame noise that a raw finite difference would -- important at 120 Hz,
    where the frame-to-frame displacement is small relative to segmentation
    jitter.
    """
    n, po = _sg_window(info, cfg)
    ok = np.isfinite(values)
    out = np.full_like(values, np.nan)
    if ok.sum() < n:
        return out
    dt = float(np.median(np.diff(t[ok]))) or info.dt
    win = min(n, (ok.sum() // 2) * 2 - 1)
    out[ok] = signal.savgol_filter(values[ok], win, po, deriv=1, delta=dt)
    return out


def path_length(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Cumulative distance traveled by the centroid.

    Note this is computed on *smoothed* coordinates upstream. Path length from
    raw per-frame positions is biased upward -- tracking jitter adds length that
    the robot never traveled, and the bias grows with frame rate, so a 120 Hz
    clip would report a longer path than a 30 Hz clip of the same motion.
    """
    d = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    d = np.nan_to_num(d, nan=0.0)
    return np.concatenate([[0.0], np.cumsum(d)])


@dataclass
class Periodicity:
    dominant_hz: float
    amplitude: float
    snr: float
    resolvable: bool
    note: str


def dominant_frequency(values: np.ndarray, info: VideoInfo,
                       cfg: AnalysisConfig, band=(0.05, None)) -> Periodicity:
    """Contraction frequency via Welch PSD, with an explicit Nyquist verdict."""
    v = values[np.isfinite(values)]
    if v.size < 32:
        return Periodicity(np.nan, np.nan, np.nan, False, "too few valid samples")
    v = v - np.mean(v)
    if cfg.detrend_drift:
        v = signal.detrend(v)
    nper = min(len(v), max(64, int(info.measured_fps * 4)))
    f, pxx = signal.welch(v, fs=info.measured_fps, nperseg=nper)
    hi = band[1] or info.nyquist_hz
    sel = (f >= band[0]) & (f <= hi)
    if not sel.any():
        return Periodicity(np.nan, np.nan, np.nan, False, "empty analysis band")
    fs, ps = f[sel], pxx[sel]
    k = int(np.argmax(ps))
    peak = float(fs[k])
    noise = float(np.median(ps))
    snr = float(ps[k] / noise) if noise > 0 else np.inf
    amp = float(np.sqrt(2.0 * ps[k] * (fs[1] - fs[0]))) if len(fs) > 1 else np.nan

    if peak > info.reliable_freq_hz:
        note = (f"{peak:.2f} Hz exceeds fs/4 ({info.reliable_freq_hz:.1f} Hz). "
                f"Amplitude is likely underestimated -- re-record at a higher frame rate.")
        ok = peak < info.nyquist_hz
    else:
        note = f"well resolved at {info.nominal_fps:g} Hz"
        ok = True
    return Periodicity(peak, amp, snr, ok, note)
