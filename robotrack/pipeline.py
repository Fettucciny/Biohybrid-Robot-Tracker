"""End-to-end orchestration: video in, per-frame table plus plots and overlay out."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .cad import Template, load_dxf
from .decode import FrameReader, select_backend
from .gpu import Device, get_device, verify_device
from .ingest import VideoInfo, probe
from .kinematics import (AnalysisConfig, derivative, dominant_frequency,
                         gate_and_fill, path_length, smooth)
from .register import FitConfig, ShapeFitter
from .segment import SegmentConfig, Segmenter
from .shape import measure_mask


def _no_accelerator_hint() -> str:
    """Why there is no GPU, phrased for the platform actually in use.

    Telling a Mac user to reinstall torch from the CUDA index is worse than
    saying nothing: the CUDA index has no macOS wheel, so following the advice
    fails and the real cause -- an old macOS, or an Intel Mac -- goes unfound.
    """
    if sys.platform == "darwin":
        return ("Check 'use GPU'. Metal acceleration needs macOS 12.3 or later on "
                "Apple Silicon; an Intel Mac has no MPS backend at all.")
    return ("Check 'use GPU', and check that torch was installed from the CUDA "
            "index rather than plain PyPI.")


def _json_num(x) -> float | None:
    """NaN and infinities become null, so run_info.json stays valid JSON."""
    return float(x) if x is not None and np.isfinite(x) else None


def _json_safe(obj):
    """Recursively replace NaN/inf with null, everywhere in a nested structure.

    ``json.dump`` happily writes a bare ``NaN`` token. Python reads it back, so
    the problem is invisible from here -- but it is not valid JSON, and every
    strict parser rejects the file: JavaScript's ``JSON.parse``, R's jsonlite,
    Go, and most schema validators. A run with no calibration produces several
    such fields, so this is the normal case rather than an edge one. Scrubbing
    the whole structure at dump time is safer than remembering to wrap each new
    field as one gets added.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _nanmedian(values) -> float:
    """``np.nanmedian`` that returns NaN for an all-NaN input, quietly.

    A drawing with no interior features leaves ``feature_fit`` entirely NaN,
    which is a perfectly ordinary state -- and numpy answers it with a
    RuntimeWarning printed to stderr in the middle of the run summary. The
    answer it gives (NaN) is the right one; only the noise is wrong.
    """
    a = np.asarray(values, float)
    if a.size == 0 or not np.isfinite(a).any():
        return float("nan")
    return float(np.nanmedian(a))


@dataclass
class RunConfig:
    video: str
    dxf: str | None = None
    dxf_loop_index: int = 0             # which outline in the drawing is the robot
    dxf_scale: float = 1.0              # multiplier on the drawing's dimensions
    use_features: bool = True           # fit the drawing's interior structure too
    # "none" | "lut" | "beam". The LUT is a curve you measured; the beam model
    # computes force from the robot's own geometry and material.
    force_method: str = "none"
    force_lut: str | None = None        # CSV of Length,Force for force conversion
    beam: object = None                 # BeamForceModel when force_method == "beam"
    # The robot's true width in mm. Normally taken from the drawing, but settable
    # directly so the width ruler also works markerless, with no DXF at all.
    known_width_mm: float | None = None
    preview_every: int = 0              # emit an overlay frame every N frames (0 = off)
    # {panel: [lo, hi]} axis ranges from the interactive plot panel, recorded so
    # the exported figure matches what was on screen when Export was pressed.
    axis_ranges: dict | None = None
    outdir: str = "results"
    px_per_mm: float | None = None      # from a ruler; else self-calibrated via CAD
    scale: float = 1.0                  # decode downscale, e.g. 0.5 for 4K speedup
    write_overlay: bool = True
    # Longest side of the overlay video. An overlay is for confirming the fit
    # followed the robot, not for measuring off, and re-encoding 4K costs about
    # 3.4 minutes per 930 frames against 13 seconds at 960 px. 0 disables the cap.
    overlay_max_px: int = 960
    gpu: bool = True
    # Optional hand-placed starting pose (tx, ty, theta, sx, sy), in
    # *full-resolution* image pixels. Kept at full resolution so one placement
    # stays valid when the decode scale changes.
    manual_pose: list[float] | None = None
    # Appearance lock. When set, the pose comes from aligning a reference patch
    # rather than from fitting the drawing to a color mask -- the path to take
    # on footage where the robot and the medium are not separable per pixel.
    # (reference frame index, rect) with the rect in full-resolution pixels.
    appearance: object = None

    segment: SegmentConfig = field(default_factory=SegmentConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


@dataclass
class Result:
    info: VideoInfo
    device: Device
    table: pd.DataFrame
    template: Template | None
    calibration_px_per_mm: float | None
    calibration_source: str
    periodicity: dict
    elapsed_s: float
    fps_processed: float
    width_cv: float = float("nan")      # how constant the ruler actually was
    width_median_px: float = float("nan")
    lut: object = None
    lut_clamped: int = 0
    beam: object = None
    force_method: str = "none"
    resting_length_mm: float = float("nan")
    aspect_measured: float = float("nan")
    aspect_drawn: float = float("nan")
    stage_times: dict = field(default_factory=dict)
    # Non-fatal things the run wants to say out loud, e.g. an accelerator that
    # failed its self-check. Carried on the result rather than logged from deep
    # inside the pipeline, so the CLI and the GUI both get them.
    notes: list = field(default_factory=list)

    def _timing_lines(self) -> list[str]:
        st = self.stage_times or {}
        total = sum(st.values())
        n = max(len(self.table), 1)
        if total <= 0:
            return []
        parts = "  ".join(f"{k} {1000 * v / n:.0f} ms ({100 * v / total:.0f}%)"
                          for k, v in sorted(st.items(), key=lambda kv: -kv[1]))
        lines = [f"  time per frame   : {parts}",
                 f"  throughput       : {n / total:.1f} frames/s over {n} frames"]
        lines += [f"  NOTE             : {m}" for m in (self.notes or [])]
        if not self.device.accelerated:
            lines.append("  WARNING          : running on the CPU. The fit is the "
                         "dominant cost and is roughly 20x slower here than on an "
                         "accelerator — a clip that takes 20 s on a GPU takes about "
                         "10 min. " + _no_accelerator_hint())
        return lines

    def summary(self) -> str:
        t = self.table
        n_ok = int((t.confidence >= 0.5).sum())
        lines = [
            self.info.summary(),
            f"  device           : {self.device}",
            f"  processed        : {len(t)} frames in {self.elapsed_s:.1f} s "
            f"({self.fps_processed:.1f} fps)",
            f"  tracked          : {n_ok}/{len(t)} frames at confidence >= 0.5",
            *self._timing_lines(),
            f"  calibration      : "
            + (f"{self.calibration_px_per_mm:.3f} px/mm ({self.calibration_source})"
               if self.calibration_px_per_mm else "none -- results in px and strain"),
        ]
        for k, p in self.periodicity.items():
            if np.isfinite(p["dominant_hz"]):
                lines.append(f"  {k:<17}: {p['dominant_hz']:.3f} Hz "
                             f"(SNR {p['snr']:.1f}) -- {p['note']}")
        if np.isfinite(self.aspect_measured) and np.isfinite(self.aspect_drawn):
            err = abs(self.aspect_measured - self.aspect_drawn) / self.aspect_drawn
            if err > 0.15:
                lines.append(
                    f"  WARNING          : fitted proportions {self.aspect_measured:.2f} "
                    f"vs {self.aspect_drawn:.2f} drawn ({100 * err:.0f}% out). The fit is "
                    f"probably on part of the robot, not all of it — loosen the color "
                    f"cut and check the outline.")
            else:
                lines.append(f"  proportions      : {self.aspect_measured:.2f} vs "
                             f"{self.aspect_drawn:.2f} drawn — consistent")
        if "feature_fit" in t and np.isfinite(_nanmedian(t.feature_fit)):
            lines.append(f"  interior features: {100 * _nanmedian(t.feature_fit):.0f}% "
                         f"of interior points on an observed edge")
        if np.isfinite(self.width_cv):
            verdict = ("consistent with a rigid width"
                       if self.width_cv < 0.03 else
                       "the width is NOT constant — treat the scale as approximate")
            lines.append(f"  width (ruler)    : {self.width_median_px:.1f} px median, "
                         f"CV {100 * self.width_cv:.1f}% — {verdict}")
        if len(t):
            unit_um = self.calibration_px_per_mm is not None
            lines += [
                f"  length strain    : {t.length_strain.min():.3f} .. {t.length_strain.max():.3f}",
                f"  length           : {t.length_um.min():.0f} .. {t.length_um.max():.0f} um"
                if unit_um else
                f"  length           : {t.length_px.min():.1f} .. {t.length_px.max():.1f} px",
                f"  path traveled    : "
                + (f"{t.path_length_um.iloc[-1]:.0f} um" if unit_um
                   else f"{t.path_length.iloc[-1]:.2f} px"),
            ]
            if "force_mn" in t:
                lo, hi = np.nanmin(t.force_mn), np.nanmax(t.force_mn)
                if self.force_method == "beam":
                    lines.append(f"  force (beam)     : {1000 * lo:.0f} .. {1000 * hi:.0f} uN "
                                 f"({lo:.3f} .. {hi:.3f} mN)")
                    lines.append(f"  resting length   : {self.resting_length_mm:.4f} mm")
                    if self.beam is not None:
                        lines.append(f"  {self.beam.summary()}")
                    if self.lut_clamped:
                        lines.append(f"  NOTE             : {self.lut_clamped} frame(s) "
                                     f"pulled in further than the leg geometry allows "
                                     f"and were clamped")
                else:
                    lines.append(f"  force (LUT)      : {lo:.3f} .. {hi:.3f} mN")
                    if self.lut_clamped:
                        lines.append(f"  NOTE             : {self.lut_clamped} frame(s) fell "
                                     f"outside the LUT and were clamped to its ends")
        return "\n".join(lines)


class RunAborted(RuntimeError):
    """Raised when a run is stopped from the interface.

    A distinct type rather than a bool return: aborting is not a failure to be
    reported as a crash, and it is not a successful run with missing frames
    either. Callers can tell the three apart.
    """


def _fit_signal(m, seg):
    """The continuous surface interior edges are found in.

    In color mode that is the distance from the medium's color, which is where
    the internal boundaries actually live; in luma mode it is the frame itself.
    """
    # The segmenter already built this to cut the mask with. Recomputing it was
    # a straight duplicate of the single most expensive operation in the loop.
    if getattr(m, "signal", None) is not None:
        return m.signal
    if m.frame is None:
        return None
    if getattr(seg, "color", False) and seg.model.bg_ab is not None:
        from .segment import color_distance
        return color_distance(m.frame, seg.model.bg_ab)
    return m.frame if m.frame.ndim == 2 else cv2.cvtColor(m.frame, cv2.COLOR_BGR2GRAY)


def _overlay_frame(m, rec, fitted_outline, max_px: int = 900):
    """Compose the mask contour and fitted outline onto one frame, for the GUI.

    Shrunk here, in the worker, rather than sent full size. A 4K frame is 25 MB
    and costs about 17 ms on the UI thread just to turn into a QPixmap; at six a
    second that is a tenth of the main thread spent re-rendering a picture that
    is displayed in a few hundred pixels either way.
    """
    img = getattr(m, "frame", None)
    if img is None:
        return None
    img = (img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    if m.contour is not None and len(m.contour) > 2:
        cv2.polylines(img, [m.contour.astype(np.int32)], True, (255, 255, 255), 1,
                      cv2.LINE_AA)
    if fitted_outline is not None and len(fitted_outline) > 2:
        conf = float(rec.get("confidence", 0.0) or 0.0)
        col = (140, 220, 90) if conf >= 0.5 else (60, 190, 250)
        cv2.polylines(img, [fitted_outline.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        cx, cy = rec.get("cx"), rec.get("cy")
        if cx is not None and np.isfinite(cx):
            cv2.circle(img, (int(cx), int(cy)), 4, col, -1, cv2.LINE_AA)
    h, w = img.shape[:2]
    if max_px and max(w, h) > max_px:
        k = max_px / float(max(w, h))
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return img


def _scaled_seed(cfg: RunConfig, info: VideoInfo, reader: FrameReader) -> np.ndarray | None:
    """Convert a hand-placed pose from full-resolution pixels into decode pixels.

    The reader rounds its dimensions down to an even number, so the effective
    scale is not exactly ``cfg.scale``. Deriving it from the reader's actual
    width avoids a placement that drifts by a pixel or two at 0.25 scale --
    small, but it is the seed for everything downstream.
    """
    if not cfg.manual_pose:
        return None
    s = reader.width / float(info.width or 1)
    p = np.asarray(cfg.manual_pose, np.float32).copy()
    p[0] *= s       # tx
    p[1] *= s       # ty
    p[3] *= s       # sx  (px per template mm)
    p[4] *= s       # sy
    return p


def output_dir(outdir: str | Path, video: str | Path) -> Path:
    """Where one clip's results go: ``<outdir>/<video name>/``.

    A flat output folder works for exactly one run. The second clip overwrites
    tracking.csv, summary.png, run_info.json and overlay.mp4 without a word,
    and since the file names carry no hint of which video produced them there
    is no way to notice afterwards. Giving each clip its own folder named after
    it makes a session's worth of runs accumulate instead of collide -- and it
    is what lets the video queue mark a clip as already analyzed by asking
    whether its folder exists.
    """
    return Path(outdir) / Path(str(video)).stem


def run(cfg: RunConfig, progress=None, on_row=None, on_frame=None,
        should_abort=None) -> Result:
    t_start = time.time()
    out = output_dir(cfg.outdir, cfg.video)
    out.mkdir(parents=True, exist_ok=True)

    info = probe(cfg.video)
    dev = get_device(cfg.gpu)
    # A backend that samples images wrongly does not raise -- it returns
    # plausible numbers the optimizer then converges on. Checking once here is
    # what turns "it tracks on Windows but wanders on the Mac" into a line in
    # the log and a correct run.
    notes: list[str] = []
    trust_note = ""
    trusted, why = verify_device(dev)
    if not trusted:
        trust_note = (f"{dev} failed its self-check ({why}) — the fit ran on "
                      f"the CPU instead, which is correct but slower")
        dev = get_device(False)
    backend = select_backend(info)
    reader = FrameReader(info, backend, scale=cfg.scale)

    tpl = (load_dxf(cfg.dxf, loop_index=cfg.dxf_loop_index, scale=cfg.dxf_scale,
                    use_features=cfg.use_features)
           if cfg.dxf else None)
    seg = Segmenter(reader, cfg.segment, dev)
    fitter = ShapeFitter(tpl, cfg.fit, dev, dt=info.dt,
                         seed_pose=_scaled_seed(cfg, info, reader)) if tpl else None

    rows = []
    outlines: dict[int, np.ndarray] = {}
    every = max(int(cfg.preview_every), 0) if on_frame is not None else 0

    def wants_outline(index: int) -> bool:
        """Whether this frame's fitted outline is needed by anything.

        Two separate consumers, and they used to be one condition: the overlay
        video needs every frame's outline, and the live preview needs every
        sixth. Deriving both from ``write_overlay`` meant that turning the
        overlay video off also turned off the green outline on the picture you
        watch while the run happens -- which reads as the tracker having
        stopped working, not as a setting.
        """
        return bool(cfg.write_overlay) or bool(every and index % every == 0)
    # Where the time actually goes. A run that takes twenty seconds one day and
    # ten minutes the next is almost always one stage, not a general slowdown,
    # and without this the only way to find out is to guess.
    # The appearance lock needs one frame to learn from. It is taken before the
    # run rather than from inside the loop so a failure to build it is reported
    # immediately instead of after a minute of decoding.
    tracker = None
    if cfg.appearance:
        from .appearance import AppearanceConfig, AppearanceTracker
        ref_t, ref_rect = cfg.appearance[0], cfg.appearance[1]
        ref_pose = cfg.appearance[2] if len(cfg.appearance) > 2 else None
        ref_reader = FrameReader(info, backend, scale=cfg.scale, color=True)
        ref_frame = ref_reader.read_at(float(ref_t))
        if ref_frame is None:
            raise RuntimeError(f"could not decode the reference frame at {ref_t:.2f} s")
        # Four values only: the appearance patch is an upright crop, and a
        # rotated region carries a fifth that must not be scaled like a length.
        rect = tuple(int(round(float(v) * cfg.scale)) for v in tuple(ref_rect)[:4])
        tracker = AppearanceTracker(ref_frame, rect, ref_pose, AppearanceConfig())
        if getattr(tracker, "clipped", ""):
            notes.append(tracker.clipped)

    t_seg = t_fit = t_draw = 0.0
    t_mark = time.time()
    for m in seg:
        if should_abort is not None and should_abort():
            raise RunAborted(f"stopped after {len(rows)} frames")
        t_seg += time.time() - t_mark
        rec = {"frame": m.index, "t": m.t, "area_px": m.area_px}
        if tracker is not None:
            t0 = time.time()
            pose = tracker.track(m.frame) if m.frame is not None else None
            t_fit += time.time() - t0
            if pose is None:
                rec.update(cx=np.nan, cy=np.nan, width_px=np.nan, length_px=np.nan,
                           theta=np.nan, confidence=float(tracker.last_cc),
                           fit_cost=np.nan)
            else:
                # Scales are relative to the reference frame. With a drawing the
                # tracker has already composed them onto its pose, so width and
                # length come out in the same units as the color path; without
                # one they are ratios, and the strain is still right.
                wmm = tpl.width_mm if tpl is not None else 1.0
                lmm = tpl.length_mm if tpl is not None else 1.0
                rec.update(cx=pose.tx, cy=pose.ty,
                           width_px=pose.sx * wmm, length_px=pose.sy * lmm,
                           theta=pose.theta, confidence=pose.confidence,
                           fit_cost=pose.cost, scale_x=pose.sx, scale_y=pose.sy)
                if fitter is not None and wants_outline(m.index):
                    outlines[m.index] = fitter.outline(pose)
        elif fitter is not None:
            t0 = time.time()
            pose = fitter.fit(m.mask, signal=_fit_signal(m, seg))
            t_fit += time.time() - t0
            if pose is None:
                rec.update(cx=np.nan, cy=np.nan, width_px=np.nan, length_px=np.nan,
                           theta=np.nan, confidence=0.0, fit_cost=np.nan)
            else:
                rec.update(
                    cx=pose.tx, cy=pose.ty,
                    width_px=pose.sx * tpl.width_mm,
                    length_px=pose.sy * tpl.length_mm,
                    theta=pose.theta, confidence=pose.confidence, fit_cost=pose.cost,
                    feature_fit=pose.feature_fit,
                    scale_x=pose.sx, scale_y=pose.sy,
                )
                if wants_outline(m.index):
                    outlines[m.index] = fitter.outline(pose)
        else:
            ms = measure_mask(m.mask)
            if ms is None:
                rec.update(cx=np.nan, cy=np.nan, width_px=np.nan, length_px=np.nan,
                           theta=np.nan, confidence=0.0, fit_cost=np.nan)
            else:
                rec.update(cx=ms.cx, cy=ms.cy, width_px=ms.width_px,
                           length_px=ms.length_px, theta=ms.theta,
                           confidence=1.0, fit_cost=np.nan)
                if m.contour is not None and wants_outline(m.index):
                    outlines[m.index] = m.contour
        t0 = time.time()
        if every and m.index % every == 0:
            # Show the tracking outline while the analysis runs. Only every Nth
            # frame is composited and emitted: drawing all of them would add a
            # full-frame copy per frame to the hot loop for a picture nobody can
            # read at 36 fps anyway.
            on_frame(m.index, _overlay_frame(m, rec, outlines.get(m.index)))
            if not cfg.write_overlay:
                # Computed for the picture, not for a file. Dropping it keeps
                # the dictionary from growing across a long run for no reason.
                outlines.pop(m.index, None)
        t_draw += time.time() - t0
        rows.append(rec)
        if on_row is not None:
            # Emitted raw, before gating and smoothing. The live panel is for
            # seeing that a long run is producing sensible numbers, not for
            # reading final values off -- the finished table replaces it.
            on_row(rec)
        if progress and m.index % 25 == 0:
            progress(m.index, info.n_frames)
        t_mark = time.time()

    t = pd.DataFrame(rows)
    if t.empty:
        raise RuntimeError("No frames were processed.")

    # --- occlusion gating, then physical-time smoothing -----------------------
    conf = t.confidence.to_numpy()
    tt = t.t.to_numpy()
    for col in ("cx", "cy", "width_px", "length_px"):
        filled, bad = gate_and_fill(t[col].to_numpy(), conf, tt, cfg.analysis)
        t[col + "_raw"] = t[col]
        t[col] = smooth(filled, info, cfg.analysis)
        if col == "cx":
            t["occluded"] = bad

    # --- calibration: the robot's own width is the ruler ----------------------
    #
    # The frame is rigid across its short axis while the long axis is what
    # contracts, so the width is a constant of known length carried in every
    # frame -- a ruler that is always in the plane of motion, always in focus,
    # and cannot be forgotten at capture time. Combined with the width from the
    # drawing it gives px/mm directly, per clip.
    #
    # The assumption is checkable, so it is checked: the width's coefficient of
    # variation is computed and reported. If the width is in fact moving, that
    # number says so instead of quietly biasing every micrometer in the output.
    # An appearance lock with no reference pose reports *ratios*, not pixels --
    # it knows how much the robot changed, not how big it is. Calibrating a
    # width ruler off a ratio produces a scale that looks plausible and is
    # meaningless, so that path is refused rather than fudged: strain, frequency
    # and the trajectory are all still correct, and only absolute size is
    # withheld. Give the tracker a reference pose (a hand placement, or a good
    # fit on the reference frame) and the scales come back in real pixels.
    appearance_relative = bool(cfg.appearance) and (
        len(cfg.appearance) < 3 or cfg.appearance[2] is None)

    px_per_mm, src = cfg.px_per_mm, "user-supplied"
    width_cv = float("nan")
    width_med_px = float("nan")
    ok = t.confidence >= cfg.analysis.min_confidence
    true_width_mm = cfg.known_width_mm if cfg.known_width_mm else (
        tpl.width_mm if tpl is not None else None)
    if appearance_relative:
        true_width_mm = None
    if true_width_mm and ok.any():
        w = t.loc[ok, "width_px"].to_numpy()
        width_med_px = float(np.nanmedian(w))
        if np.isfinite(width_med_px) and width_med_px > 0:
            width_cv = float(np.nanstd(w) / width_med_px)
            if px_per_mm is None:
                px_per_mm = width_med_px / max(true_width_mm, 1e-9)
                whence = ("entered directly" if cfg.known_width_mm
                          else "from the drawing")
                src = (f"robot width — {width_med_px:.1f} px = "
                       f"{true_width_mm:.3f} mm {whence}")
    if px_per_mm is None:
        src = "none"

    unit = px_per_mm or 1.0
    um_per_px = 1000.0 / unit if px_per_mm else float("nan")

    # Length only. Width is the ruler, not a measurement, and reporting a
    # constant next to the thing that varies invites reading it as a result.
    rest_px = float(np.nanmedian(t["length_px"]))
    t["length_mm"] = t["length_px"] / unit
    t["length_um"] = t["length_mm"] * 1000.0
    t["length_strain"] = t["length_px"] / rest_px if rest_px else np.nan

    t["cx_mm"], t["cy_mm"] = t.cx / unit, t.cy / unit
    t["path_length"] = path_length(t.cx_mm.to_numpy(), t.cy_mm.to_numpy())
    t["path_length_um"] = t["path_length"] * 1000.0
    t["speed"] = np.hypot(derivative(t.cx_mm.to_numpy(), tt, info, cfg.analysis),
                          derivative(t.cy_mm.to_numpy(), tt, info, cfg.analysis))

    # --- force ---------------------------------------------------------------
    lut, beam, n_clamped = None, None, 0
    rest_mm = float("nan")
    method = cfg.force_method or ("lut" if cfg.force_lut else "none")
    if method == "lut" and cfg.force_lut:
        from .forcelut import load_lut
        lut = load_lut(cfg.force_lut)
        force, n_clamped = lut.force(t["length_mm"].to_numpy())
        t["force_mn"] = force
    elif method == "beam" and cfg.beam is not None and px_per_mm:
        beam = cfg.beam
        f_un, deflection, rest_mm, n_clamped = beam.force_un(t["length_mm"].to_numpy())
        # Stored in mN so both methods share one column, one axis and one set of
        # region statistics. The model's natural unit is uN and that is what the
        # literature quotes, so it is reported both ways in the summary.
        t["force_mn"] = f_un / 1000.0
        t["deflection_um"] = deflection * 1000.0
    elif method == "beam" and not px_per_mm:
        method = "none"

    per = {}
    for col in ("length_px", "area_px"):
        p = dominant_frequency(t[col].to_numpy(), info, cfg.analysis)
        per[col.replace("_px", "") + " frequency"] = p.__dict__

    # --- shape sanity: does the fit have the drawing's proportions? -----------
    #
    # Aggressive thresholding shrinks the mask, and past a point the fit settles
    # on a *sub-region* of the robot rather than the robot. That failure looks
    # excellent by every other measure -- on the reference clip at color cut
    # 0.50 it reported 185/185 frames tracked and a width CV of 1.5% while
    # measuring one limb. Proportions are what give it away: the ratio of the two
    # fitted scales should match the drawing, and it does not.
    aspect_ratio = float("nan")
    aspect_drawn = float("nan")
    if tpl is not None and "scale_x" in t and ok.any():
        sx = float(np.nanmedian(t.loc[ok, "scale_x"]))
        sy = float(np.nanmedian(t.loc[ok, "scale_y"]))
        if sx > 0:
            aspect_ratio = (sy * tpl.length_mm) / (sx * tpl.width_mm)
            aspect_drawn = tpl.length_mm / max(tpl.width_mm, 1e-9)

    elapsed = time.time() - t_start
    res = Result(info, dev, t, tpl, px_per_mm, src, per, elapsed,
                 len(t) / elapsed if elapsed else 0.0,
                 width_cv=width_cv, width_median_px=width_med_px,
                 lut=lut, lut_clamped=n_clamped, beam=beam,
                 force_method=method, resting_length_mm=rest_mm,
                 aspect_measured=aspect_ratio, aspect_drawn=aspect_drawn,
                 stage_times={"decode+segment": t_seg, "fit": t_fit,
                              "live preview": t_draw},
                 notes=(([trust_note] if trust_note else []) + notes))

    # --- outputs -------------------------------------------------------------
    drop = [c for c in t.columns
            if c.startswith("width") or c in ("scale_x", "scale_y")]
    t.drop(columns=drop).to_csv(out / "tracking.csv", index=False)
    meta = {"video": info.to_dict(), "device": str(dev), "decode_backend": backend.name,
            "calibration_px_per_mm": px_per_mm, "calibration_source": src,
            "calibration_um_per_px": um_per_px,
            "width_median_px": width_med_px, "width_cv": width_cv,
            "dxf_scale": cfg.dxf_scale,
            "interior_features": (tpl.n_features if tpl else 0),
            "aspect_measured": aspect_ratio, "aspect_drawn": aspect_drawn,
            # None rather than NaN: json.dump writes a bare NaN token, which is
            # not valid JSON and which strict parsers -- including anything
            # reading this file from JavaScript or R -- reject outright.
            "feature_fit_median": (_nanmedian(t["feature_fit"])
                                   if "feature_fit" in t else None),
            "known_width_mm": cfg.known_width_mm,
            "template": tpl.summary() if tpl else None, "periodicity": per,
            "manual_pose": list(cfg.manual_pose) if cfg.manual_pose else None,
            "force_method": method,
            "force_lut": (lut.summary() if lut else None),
            "force_beam_model": (beam.to_dict() if beam else None),
            "force_resting_length_mm": (rest_mm if beam else None),
            "force_frames_clamped": n_clamped,
            "axis_ranges": cfg.axis_ranges or None,
            "region_analysis": (cfg.axis_ranges or {}).get("selection_stats") or None,
            "processing_fps": res.fps_processed,
            "device_cuda": dev.cuda, "device_kind": dev.kind,
            "stage_seconds": res.stage_times,
            "fit_settings": {"restarts": cfg.fit.n_restarts, "iters": cfg.fit.iters,
                             "iters_warm": cfg.fit.iters_warm,
                             "early_stop": cfg.fit.early_stop,
                             "features": bool(tpl and tpl.n_features)},
            "roi": list(cfg.segment.roi) if cfg.segment.roi else None,
            "tracking": "appearance" if cfg.appearance else
                        ("cad-fit" if tpl is not None else "markerless"),
            "decode_scale": cfg.scale}
    (out / "run_info.json").write_text(
        json.dumps(_json_safe(meta), indent=2, default=str, allow_nan=False))
    _plots(res, out / "summary.png", cfg.axis_ranges)
    if cfg.write_overlay and outlines:
        t0 = time.time()
        _overlay(info, reader, t, outlines, out / "overlay.mp4",
                 max_px=cfg.overlay_max_px, progress=progress,
                 should_abort=should_abort)
        res.stage_times["overlay video"] = time.time() - t0
        res.elapsed_s = time.time() - t_start
        res.fps_processed = len(t) / max(res.elapsed_s, 1e-9)
        (out / "run_info.json").write_text(json.dumps(
            _json_safe({**meta, "stage_seconds": res.stage_times,
                        "elapsed_s": res.elapsed_s}),
            indent=2, default=str, allow_nan=False))
    return res


def export_subset(res: Result, outdir: str | Path, t0: float, t1: float,
                  axis_ranges: dict | None = None,
                  suffix: str = "subset") -> list[Path]:
    """Write just the selected time window as its own CSV, figure and metadata.

    A selected region is usually the part of a recording that is actually the
    experiment -- the stretch after the tissue settled and before the medium
    warmed, or one bout of contraction among several. Writing it out as a
    separate, self-describing set means the analysis of that stretch can be
    handed to someone else, or plotted, without carrying the whole clip and a
    note about which seconds mattered.

    Files land beside the full run, with ``_<suffix>`` on the stem, so the pair
    stays together and cannot be confused for two different runs of the same
    clip. Returns the paths written.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (float(t0), float(t1)) if t0 <= t1 else (float(t1), float(t0))

    t = res.table
    sub = t[(t["t"] >= lo) & (t["t"] <= hi)].copy()
    if sub.empty:
        raise ValueError(f"no frames between {lo:.3f} s and {hi:.3f} s")

    # Path length is cumulative from the start of the clip, so an unmodified
    # slice would open at whatever distance the robot had already travelled.
    # Re-zeroing makes the subset's path column mean "distance within this
    # window", which is the only reading of it that makes sense here.
    for col in ("path_length", "path_length_um"):
        if col in sub:
            sub[col] = sub[col] - sub[col].iloc[0]

    written = []
    csv_path = out / f"tracking_{suffix}.csv"
    sub.to_csv(csv_path, index=False)
    written.append(csv_path)

    # A Result carrying only the slice, so the figure builder needs no changes
    # and the subset figure is drawn by exactly the same code as the full one.
    sub_res = replace(res, table=sub)
    png_path = out / f"summary_{suffix}.png"
    _plots(sub_res, png_path, {"time": [lo, hi], "zoomed": True,
                               **{k: v for k, v in (axis_ranges or {}).items()
                                  if k not in ("time", "zoomed", "selection")}})
    written.append(png_path)

    meta = {
        "source": "region selected in the plot panel",
        "video": res.info.to_dict(),
        "window_s": [lo, hi],
        "duration_s": hi - lo,
        "frames": int(len(sub)),
        "frames_in_full_run": int(len(t)),
        "calibration_px_per_mm": res.calibration_px_per_mm,
        "calibration_source": res.calibration_source,
        "region_analysis": (axis_ranges or {}).get("selection_stats") or None,
        "note": ("Cumulative path is re-zeroed to the start of this window. "
                 "Every other column is copied unchanged from the full run."),
    }
    json_path = out / f"run_info_{suffix}.json"
    json_path.write_text(json.dumps(_json_safe(meta), indent=2, default=str,
                                    allow_nan=False))
    written.append(json_path)
    return written


def _plots(res: Result, path: Path, axis_ranges: dict | None = None) -> None:
    """The exported figure. Mirrors the on-screen panel, including its zoom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .theme import C as THEME, matplotlib_rc, series_colors
    matplotlib.rcParams.update(matplotlib_rc())
    col = series_colors()

    axis_ranges = axis_ranges or {}
    zoomed = bool(axis_ranges.get("zoomed"))

    t = res.table
    calibrated = res.calibration_px_per_mm is not None
    unit = "µm" if calibrated else "px"
    has_force = "force_mn" in t

    # Width is the ruler, not a result, so it is not plotted -- a flat line
    # beside the thing that varies reads as a measurement that did nothing.
    panels = [("length", t["length_um"] if calibrated else t["length_px"],
               f"length ({unit})", col[0])]
    if has_force:
        panels.append(("force", t["force_mn"] * 1000.0, "force (µN)", col[2]))
    panels.append(("path", t["path_length_um"] if calibrated else t["path_length"],
                   f"cumulative path ({unit})", col[1]))
    # Percent, matching the on-screen panel: the same quantity should not be
    # 0-1 in one place and 0-100 in the other.
    panels.append(("conf", t["confidence"] * 100.0, "fit conf. (%)", THEME["ok"]))

    ratios = [2.0 if k in ("length", "force") else 1.0 for k, _, _, _ in panels]
    fig, ax = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)),
                           sharex=True, gridspec_kw={"height_ratios": ratios})
    if len(panels) == 1:
        ax = [ax]

    for a, (key, y, label, color) in zip(ax, panels):
        a.plot(t.t, y, lw=1.4, color=color)
        a.set_ylabel(label)
        a.grid(alpha=0.18)
        if key == "conf":
            a.axhline(50.0, ls="--", lw=0.8, color=THEME["muted"])
            a.set_ylim(0, 102)
        # Only honour a *deliberate* zoom. Without the "zoomed" guard an
        # untouched panel reports its matplotlib defaults of (0, 1), and
        # applying those flattens every curve in the exported figure into a
        # unit band -- which is what this figure used to do, every time.
        if zoomed and key in axis_ranges:
            try:
                a.set_ylim(*axis_ranges[key])
            except (TypeError, ValueError):
                pass
    if zoomed and "time" in axis_ranges:
        try:
            ax[0].set_xlim(*axis_ranges["time"])
        except (TypeError, ValueError):
            pass
    ax[-1].set_xlabel("time (s)")

    if "occluded" in t:
        for a in ax:
            a.fill_between(t.t, *a.get_ylim(), where=t.occluded,
                           color=THEME["warn"], alpha=0.12, lw=0)

    # The analyzed region, with its numbers, so the figure carries the result
    # rather than just the picture it was read from.
    sel = (axis_ranges or {}).get("selection")
    st = (axis_ranges or {}).get("selection_stats") or {}
    if sel and len(sel) == 2:
        u = st.get("units", unit)
        label = {"length": ("length_delta", f"Δ {{:.1f}} {u} avg"),
                 "force": ("force_delta_un", "Δ {:.1f} µN avg"),
                 "path": ("speed_per_min",
                          "{:.3f} " + st.get("speed_units", "mm/min"))}
        for a, (key, _, _, _) in zip(ax, panels):
            # Edges rather than a heavy fill: the occlusion shading is already a
            # wash, and two translucent overlays on top of each other stop
            # reading as either one.
            a.axvspan(sel[0], sel[1], color=THEME["accent"], alpha=0.07, lw=0, zorder=0)
            for x in sel:
                a.axvline(x, color=THEME["accent"], lw=1.0, alpha=0.75, zorder=1)
            spec = label.get(key)
            if not spec or key == "conf":
                continue
            v = st.get(spec[0])
            if v is None or not np.isfinite(v):
                continue
            a.annotate(spec[1].format(v), xy=(0.99, 0.06), xycoords="axes fraction",
                       ha="right", va="bottom", fontsize=8, color=THEME["plot_fg"],
                       bbox=dict(boxstyle="round,pad=0.3", fc=THEME["panel"],
                                 ec=THEME["accent"], alpha=0.9, lw=0.8))

    scale_note = (f"{1000 / res.calibration_px_per_mm:.2f} µm/px from the robot's width"
                  if calibrated else "uncalibrated — pixels")
    zoom_note = "  ·  zoomed" if (axis_ranges or {}).get("zoomed") else ""
    ax[0].set_title(f"{Path(res.info.path).name} — {res.info.nominal_fps:g} Hz, "
                    f"{len(t)} frames  ·  {scale_note}{zoom_note}\n"
                    "(shaded = occluded / low confidence)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _overlay(info: VideoInfo, reader: FrameReader, t: pd.DataFrame,
             outlines: dict[int, np.ndarray], path: Path,
             max_px: int = 960, progress=None, should_abort=None) -> None:
    """Draw the fitted outline over the video and re-encode.

    This is a second full pass -- decode the whole clip again in color, composite,
    and encode -- and at full 4K it dominates everything else in the run: about
    3.4 minutes per 930 frames against 13 seconds at 960 px. Since the overlay
    exists to confirm the fit followed the robot rather than to measure from, it
    is written at a reduced size by default, and it reports progress so a long
    encode does not look like a hang.
    """
    src_w, src_h = reader.width, reader.height
    k = 1.0
    if max_px and max(src_w, src_h) > max_px:
        k = max_px / float(max(src_w, src_h))
    out_w, out_h = int(src_w * k) // 2 * 2, int(src_h * k) // 2 * 2

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         info.measured_fps, (out_w, out_h))
    color_reader = FrameReader(info, reader.backend, scale=reader.scale, color=True)
    conf = dict(zip(t.frame, t.confidence))
    # One dict lookup per frame instead of a DataFrame scan. The old version ran
    # ``t[t.frame == i]`` inside the loop, which is a full-table comparison per
    # frame and grows with clip length on top of the encode cost.
    length = dict(zip(t.frame, t.length_px))
    total = info.n_frames or len(t)
    for i, ts, frame in color_reader:
        img = frame if k == 1.0 else cv2.resize(frame, (out_w, out_h),
                                                interpolation=cv2.INTER_AREA)
        img = img.copy() if k == 1.0 else img
        if i in outlines:
            c = conf.get(i, 0.0)
            col = (0, 255, 0) if c >= 0.5 else (0, 165, 255)
            cv2.polylines(img, [(outlines[i] * k).astype(np.int32)], True, col, 2)
        L = length.get(i)
        if L is not None and np.isfinite(L):
            cv2.putText(img, f"t={ts:6.3f}s  L={L * k:6.1f}px  conf={conf.get(i, 0.0):.2f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * max(k, 0.5),
                        (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
        if progress and i % 25 == 0:
            progress(i, total)
        if should_abort is not None and should_abort():
            break          # a partial overlay still plays; the data is written
    vw.release()
