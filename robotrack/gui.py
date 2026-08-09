"""PySide6 desktop interface for robotrack.

The point of this window is not to hide the command line -- it is to let you
*see the segmentation before committing to a full run*. Getting the threshold
and morphology right on one representative frame takes seconds here and saves
reprocessing a ten-minute 4K clip because the mask was wrong.

Every control carries a (?) badge: hovering gives a one-line summary with the
valid range, clicking opens the full explanation from paramhelp.py.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from PySide6.QtCore import (QEvent, QObject, QSize, Qt, QThread, QTimer,
                            Signal)
from PySide6.QtGui import (QColor, QIcon, QImage, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QSplitter, QTextEdit, QVBoxLayout, QWidget)

from . import settings as S
from . import sound as snd
from . import update as U
from . import APP_NAME
from .cad import Template, load_dxf, read_loops
from .decode import FrameReader, select_backend
from .gpu import get_device
from .ingest import VideoInfo, probe
from .kinematics import AnalysisConfig
from .paramhelp import HELP
from .pipeline import (RunAborted, RunConfig,
                       output_dir as pipeline_output_dir, run)
from .forcelut import LUTError, load_lut
from .forcemodel import BeamForceModel
from .placement import PreviewView
from .plotpanel import PlotPanel
from .register import FitConfig, ShapeFitter
from .segment import (ColorModel, SegmentConfig, build_background,
                      choose_color_model, largest_component, segment_color_d,
                      segment_frame)
from .shape import measure_mask
from .theme import (ACCENT, C, Card, HelpBadge, apply as apply_theme, glyph,
                    set_mode as set_theme_mode, style_chip)
from .updater_ui import UpdateDialog


def _blob_extent(mask: np.ndarray) -> float:
    """Longest side of the biggest blob -- a single frame's guess at body size."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0.0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return float(max(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]))


def _hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def _bgr(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> OpenCV BGR, so overlay colors track the theme accent."""
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


OK_BGR = _bgr(C["ok"])        # confident fit
WARN_BGR = _bgr(C["warn"])    # low confidence


def _claim_taskbar_identity() -> None:
    """Tell Windows this process is its own application.

    Without an explicit AppUserModelID, Windows groups the window under the host
    executable and shows *that* file's embedded icon on the taskbar, ignoring
    setWindowIcon entirely. Setting one makes the window's own icon
    authoritative -- which is what lets an icon shipped in a code patch appear
    without rebuilding the .exe.

    Windows-only and entirely optional; any failure just leaves the previous
    behaviour.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "org.biohybridlab.robotrack")
    except Exception:
        pass


def _draw_conf_badge(img: np.ndarray, conf: float) -> None:
    """Stamp the fit confidence into the top-left of the preview frame.

    It was already computed and already shown -- as one item in a run of small
    grey text under the video. That is the wrong place for it: confidence is
    the number that decides whether the outline you are looking at means
    anything, and while you are dragging a threshold your eyes are on the frame,
    not on the status line. Putting it in the corner of the picture costs
    nothing and removes the glance.

    Colored by the same 0.5 cut the analysis uses to accept a frame, so the
    badge answers "would this frame count?" without being read as a number.
    """
    ok = conf >= 0.5
    color = OK_BGR if ok else WARN_BGR
    text = f"conf {100 * conf:.0f}%"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 9
    x0, y0 = 10, 10
    x1, y1 = x0 + tw + 2 * pad, y0 + th + base + 2 * pad
    # Darken behind the text rather than filling flat: the badge stays legible
    # over both the photograph and the bright end of the chroma colormap.
    roi = img[y0:y1, x0:x1]
    if roi.size:
        img[y0:y1, x0:x1] = (roi.astype(np.float32) * 0.35).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    cv2.putText(img, text, (x0 + pad, y1 - pad - base // 2), font, scale,
                color, thick, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------

class LoadWorker(QThread):
    """Probes the clip, decides how to key the robot, and prepares the reader."""
    done = Signal(object, object, object, object, object)   # info, reader, bg, model, err
    note = Signal(str)

    def __init__(self, video: str, scale: float, seg: SegmentConfig, gpu: bool):
        super().__init__()
        self.video, self.scale, self.seg, self.gpu = video, scale, seg, gpu

    def run(self):
        try:
            self.note.emit("probing video…")
            info = probe(self.video)
            backend = select_backend(info)
            self.note.emit(f"decoder: {backend.name} — {backend.description}")
            reader = FrameReader(info, backend, scale=self.scale)

            self.note.emit("measuring color separation…")
            model, _ = choose_color_model(reader, self.seg)
            self.note.emit(model.summary())

            bg = None
            if model.mode != "color":
                # Only luma needs the median plate, and it is the expensive part.
                self.note.emit(f"building background plate from "
                               f"{self.seg.n_background_frames} frames…")
                bg, _ = build_background(reader, self.seg, get_device(self.gpu))
            self.done.emit(info, reader, bg, model, None)
        except Exception:
            self.done.emit(None, None, None, None, traceback.format_exc())


class PreviewWorker(QThread):
    """Renders one frame through the real segmentation and fitting code."""
    done = Signal(object, str, object, object)   # image, status, error, fitted pose

    def __init__(self, reader, bg, seg_cfg, fit_cfg, tpl, t, gpu, show, seed=None,
                 model=None, cache=None, fitter=None, view="video"):
        super().__init__()
        self.reader, self.bg, self.seg_cfg, self.fit_cfg = reader, bg, seg_cfg, fit_cfg
        self.tpl, self.t, self.gpu, self.show = tpl, t, gpu, show
        self.seed = seed
        self.model = model
        self.cache = cache if cache is not None else {}
        self.fitter = fitter
        self.view = view

    def _frame(self, color: bool):
        """Decoded frame for this timestamp, from the cache when possible.

        Scrubbing revisits the same frames constantly -- nudging a parameter and
        looking again, stepping back and forth over a contraction. Each decode is
        an ffmpeg launch and a keyframe seek, so caching them is most of the
        responsiveness win.
        """
        key = (round(self.t, 4), color, self.reader.scale)
        f = self.cache.get(key)
        if f is None:
            f = self.reader.read_at(self.t)
            if f is not None:
                self.cache[key] = f
                if len(self.cache) > 48:
                    self.cache.pop(next(iter(self.cache)))
        return f

    def run(self):
        try:
            color = self.model is not None and self.model.mode == "color"
            frame = self._frame(color)
            if frame is None:
                self.done.emit(None, "", "Could not decode a frame at that position.", None)
                return
            dev = get_device(self.gpu)
            min_area = self.seg_cfg.min_area_frac * self.reader.width * self.reader.height

            dist = None
            if color:
                m, thr, dist = segment_color_d(frame, self.seg_cfg, self.model.bg_ab,
                                               self.model.separation,
                                               self.seg_cfg.manual_threshold)
                if self.view == "chroma":
                    # What the segmenter actually sees: distance from the
                    # medium's color. The threshold is a horizontal cut through
                    # this surface, so a mask that looks wrong is diagnosed here
                    # rather than guessed at from the photograph.
                    scaled = np.clip(dist / max(self.model.separation, 1e-6) * 255.0,
                                     0, 255).astype(np.uint8)
                    img = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
                else:
                    img = frame.copy()
            else:
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ft = torch.from_numpy(gray).to(dev.torch_device).float()
                mt, thr = segment_frame(ft, self.bg, self.seg_cfg,
                                        self.seg_cfg.manual_threshold)
                m = mt.to(torch.uint8).cpu().numpy()
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Same grouping rule the run uses. Without the envelope the preview
            # silently disagreed with the analysis: fragments merged unchecked and
            # the fitted length ran to the full frame height.
            mask, area, contours = largest_component(
                m, min_area, reach_px=self.seg_cfg.gap_factor * _blob_extent(m),
                envelope_factor=self.seg_cfg.envelope_factor)
            if self.show["mask"] and mask.any():
                tint = np.zeros_like(img)
                tint[mask > 0] = _bgr(ACCENT)
                img = cv2.addWeighted(img, 1.0, tint, 0.40, 0)

            bits = [("dC" if color else "thr") + f" {thr:.0f}", f"area {int(area)}px"]
            fitted = None
            if self.tpl is not None and mask.any():
                # Reuse the caller's fitter when it still matches: building one
                # rasterises a signed-distance grid and uploads the template, and
                # doing that per keystroke was pure overhead.
                fitter = self.fitter or ShapeFitter(self.tpl, self.fit_cfg, dev,
                                                    seed_pose=self.seed)
                sig = (dist if color else
                       (frame if frame.ndim == 2 else
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                pose = fitter.fit(mask, signal=sig)
                if pose is not None:
                    fitted = list(pose.as_array())
                    _draw_conf_badge(img, float(pose.confidence))
                    if self.show["fit"]:
                        out = fitter.outline(pose).astype(np.int32)
                        col = OK_BGR if pose.confidence >= 0.5 else WARN_BGR
                        cv2.polylines(img, [out], True, col, 2, cv2.LINE_AA)
                        feats = fitter.feature_outline(pose)
                        if feats is not None:
                            # Thinner and unclosed: these are matched edges, not
                            # the measured boundary, and should not be mistaken
                            # for it.
                            cv2.polylines(img, [feats.astype(np.int32)], False,
                                          col, 1, cv2.LINE_AA)
                        cv2.circle(img, (int(pose.tx), int(pose.ty)), 5, col, -1, cv2.LINE_AA)
                    bits += [f"W {pose.sx * self.tpl.width_mm:.1f}px",
                             f"L {pose.sy * self.tpl.length_mm:.1f}px",
                             f"conf {pose.confidence:.2f}"]
                    if np.isfinite(pose.feature_fit):
                        bits.append(f"interior {100 * pose.feature_fit:.0f}%")
            elif mask.any():
                ms = measure_mask(mask)
                if ms:
                    if self.show["fit"] and contours:
                        cv2.polylines(img, [c.astype(np.int32) for c in contours],
                                      True, OK_BGR, 2, cv2.LINE_AA)
                        cv2.circle(img, (int(ms.cx), int(ms.cy)), 5, OK_BGR, -1)
                    bits += [f"W {ms.width_px:.1f}px", f"L {ms.length_px:.1f}px"]
            self.done.emit(img, "    ".join(bits), None, fitted)
        except Exception:
            self.done.emit(None, "", traceback.format_exc(), None)


class PlaybackWorker(QThread):
    """Streams frames sequentially for playback.

    Playback cannot reuse the scrubbing path: that decodes one frame per request
    with a keyframe seek and an ffmpeg launch, which is fine once and hopeless
    thirty times a second. This holds a single sequential decode open instead --
    the case ffmpeg is fastest at -- and segments each frame as it arrives.

    Fitting is deliberately skipped while playing. It is the expensive stage and
    playback is for watching the clip and judging the mask, not for measuring;
    the fit returns the moment you stop.
    """
    frame = Signal(int, float, object, str)
    finished_at = Signal(int)

    def __init__(self, reader, model, bg, seg_cfg, start_index, show_mask, gpu, fps,
                 view="video"):
        super().__init__()
        self.reader, self.model, self.bg, self.seg_cfg = reader, model, bg, seg_cfg
        self.start_index, self.show_mask, self.gpu = start_index, show_mask, gpu
        self.view = view
        self.fps = max(float(fps), 1.0)
        self._stop = False
        self._last = start_index

    def stop(self):
        self._stop = True

    def run(self):
        try:
            color = self.model is not None and self.model.mode == "color"
            dev = get_device(self.gpu)
            min_area = self.seg_cfg.min_area_frac * self.reader.width * self.reader.height
            period = 1.0 / self.fps
            next_due = time.monotonic()
            for i, t, frame in self.reader:
                if self._stop:
                    break
                if i < self.start_index:
                    continue
                self._last = i
                if color:
                    m, thr, d = segment_color_d(frame, self.seg_cfg, self.model.bg_ab,
                                                self.model.separation,
                                                self.seg_cfg.manual_threshold)
                    # Playback follows the View selector rather than always
                    # showing the photograph. Watching the chroma surface move
                    # is how you see the cut about to fail -- the moment the
                    # robot's distance from the medium dips toward the
                    # threshold is visible here and invisible in the video.
                    if self.view == "chroma":
                        scaled = np.clip(d / max(self.model.separation, 1e-6) * 255.0,
                                         0, 255).astype(np.uint8)
                        img = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
                    else:
                        img = frame.copy()
                elif self.bg is not None:
                    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ft = torch.from_numpy(gray).to(dev.torch_device).float()
                    mt, thr = segment_frame(ft, self.bg, self.seg_cfg,
                                            self.seg_cfg.manual_threshold)
                    m = mt.to(torch.uint8).cpu().numpy()
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    m, thr, img = None, 0.0, (frame if frame.ndim == 3
                                              else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)).copy()
                area = 0.0
                if m is not None:
                    mask, area, contours = largest_component(
                        m, min_area, envelope_factor=self.seg_cfg.envelope_factor)
                    if self.show_mask and mask.any():
                        tint = np.zeros_like(img)
                        tint[mask > 0] = _bgr(ACCENT)
                        img = cv2.addWeighted(img, 1.0, tint, 0.40, 0)
                    # The tracked outline is drawn whether or not the mask tint
                    # is on. It is the one thing worth watching during playback:
                    # a tracker that lets go shows it here frames before the
                    # numbers do.
                    if contours:
                        # White reads on a photograph but disappears against the
                        # bright end of viridis, so the chroma view gets the
                        # accent instead.
                        line = _bgr(ACCENT) if self.view == "chroma" else (255, 255, 255)
                        cv2.polylines(img, [c.astype(np.int32) for c in contours],
                                      True, line, 2 if self.view == "chroma" else 1,
                                      cv2.LINE_AA)
                # Drop frames rather than fall behind: playing at the clip's real
                # rate matters more than showing every frame.
                next_due += period
                lag = next_due - time.monotonic()
                if lag > 0:
                    self.msleep(int(lag * 1000))
                elif lag < -period * 3:
                    next_due = time.monotonic()
                    continue
                self.frame.emit(i, t, img, f"area {int(area)}px")
        except Exception:
            pass
        finally:
            self.finished_at.emit(self._last)


class MatchWorker(QThread):
    """Scores every candidate outline against real frames from the clip."""
    done = Signal(object, object)
    note = Signal(str)

    def __init__(self, dxf, reader, seg_cfg, fit_cfg, gpu):
        super().__init__()
        self.dxf, self.reader, self.seg_cfg = dxf, reader, seg_cfg
        self.fit_cfg, self.gpu = fit_cfg, gpu

    def run(self):
        try:
            from .automatch import collect_masks, score_loops
            dev = get_device(self.gpu)
            self.note.emit("segmenting sample frames…")
            masks = collect_masks(self.reader, self.seg_cfg, dev, n_frames=4)
            if not masks:
                self.done.emit(None, "Nothing was segmented on the sampled frames.")
                return
            self.note.emit(f"fitting each candidate outline to {len(masks)} frames…")
            res = score_loops(self.dxf, masks, dev, self.fit_cfg,
                              progress=lambda a, b: self.note.emit(
                                  f"matching outline {a + 1} of {b}…"))
            self.done.emit(res, None)
        except Exception:
            self.done.emit(None, traceback.format_exc())


class RunWorker(QThread):
    progress = Signal(int, int)
    row = Signal(object)
    frame = Signal(int, object)
    done = Signal(object, object)
    aborted = Signal(str)

    def __init__(self, cfg: RunConfig):
        super().__init__()
        self.cfg = cfg
        # Checked between frames rather than by killing the thread: a torch
        # graph mid-backward and a half-written video file are not things to
        # interrupt at an arbitrary instruction.
        self._abort = threading.Event()

    def abort(self):
        self._abort.set()

    def run(self):
        try:
            res = run(self.cfg, progress=lambda i, n: self.progress.emit(i, n or 0),
                      on_row=lambda rec: self.row.emit(dict(rec)),
                      on_frame=lambda i, img: (img is not None
                                               and self.frame.emit(i, img)),
                      should_abort=self._abort.is_set)
            self.done.emit(res, None)
        except RunAborted as stop:
            self.aborted.emit(str(stop))
        except Exception:
            self.done.emit(None, traceback.format_exc())


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class NoWheel(QObject):
    """Swallows wheel events on value widgets.

    A scroll over a long parameter sidebar passes across a dozen spin boxes and
    combos on its way down, and Qt's default is for whichever one is under the
    pointer to take the wheel and change its value. The damage is quiet: a
    threshold moves by one, the preview re-renders, and there is nothing on
    screen that says a number changed. Forwarding the event to the scroll area
    instead keeps the gesture doing the one thing it was meant to do.

    Widgets stay editable in every other way -- click, type, arrow keys, and the
    wheel still works once a control has been deliberately clicked into focus.
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    def __init__(self, startup_warning: str | None = None):
        super().__init__()
        _claim_taskbar_identity()
        self.setWindowTitle(f"{APP_NAME} — muscle-driven soft robot tracking")
        self.resize(1480, 940)
        try:
            from .splash import asset
            ico = asset("robotrack.ico")
            if ico.exists():
                self.setWindowIcon(QIcon(str(ico)))
        except Exception:
            pass

        self.video: str | None = None
        self.dxf: str | None = None
        self.info: VideoInfo | None = None
        self._split = None
        self._queue: list = []
        self._sizes: dict = {}       # video path -> (frames, pixels), for the estimate
        self._sizer = None
        # Workers that were replaced while still running. Held only so Python
        # cannot destroy a live QThread; entries drop out when they finish.
        self._retired: list = []
        self._result = None          # the finished Result, for subset export
        self.reader: FrameReader | None = None
        self.background = None
        self.template: Template | None = None
        self.outdir: str | None = None
        self._preview: PreviewWorker | None = None
        self._pending = False
        self._preview_busy = False
        self._startup_warning = startup_warning
        self.model: ColorModel | None = None
        self._frame_cache: dict = {}
        self._fitter: ShapeFitter | None = None
        self._fitter_key = None
        self._player: PlaybackWorker | None = None
        self._matcher: MatchWorker | None = None
        self.lut = None
        self.lut_path: str | None = None

        # Restored state. Loaded before any widget exists so the widgets can be
        # built with the remembered values rather than built and then corrected,
        # which would fire every valueChanged signal on the way past.
        self.state = S.load_settings()
        # Known-wrong stored values, fixed before any widget reads them, and
        # announced rather than applied quietly -- see settings.CORRECTED.
        self._corrections = S.apply_corrections(self.state)
        self._loading_state = False

        # Manual placement, stored in *full-resolution* image pixels so it
        # survives a change of decode scale.
        self.manual_pose: list[float] | None = self.state.get("manual_pose")
        self.roi: list | None = self.state.get("roi") or None
        self._last_fit_pose: list[float] | None = None

        # Panel state. Read before the sidebar is built, because the sidebar
        # restores from it rather than the other way round.
        self.ui_mode = ("advanced" if str(self.state.get("ui_mode", "simple")).lower()
                        == "advanced" else "simple")
        raw = self.state.get("collapsed_sections") or {}
        self.collapsed_sections: dict[str, bool] = (
            {str(k): bool(v) for k, v in raw.items()} if isinstance(raw, dict) else {})
        self.cards: dict = {}

        # Dragging a slider fires many changes and each preview costs a decode
        # plus a fit, so coalesce them into a single render.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._render_preview)

        # Writing settings on every keystroke would mean a disk write per digit
        # typed into a spin box.
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(700)
        self._persist_timer.timeout.connect(self._persist)

        root = QWidget()
        root.setObjectName("background")      # kit hook: indigo->teal gradient
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        split = self._split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_sidebar())
        split.addWidget(self._build_viewer())
        split.addWidget(self._build_plots_column())
        split.setStretchFactor(1, 1)
        split.setSizes([400, 700, 380])
        split.setHandleWidth(10)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 12, 14, 14)
        bl.addWidget(split)
        outer.addWidget(body, 1)
        self.setCentralWidget(root)

        # Apply the wheel guard to every value widget in one pass, after the
        # whole tree exists, rather than remembering to do it at each of the
        # thirty-odd construction sites.
        self._nowheel = NoWheel(self)
        for kind in (QSpinBox, QDoubleSpinBox, QComboBox, QSlider):
            for wdg in root.findChildren(kind):
                if wdg is self.slider:
                    continue          # the scrubber is meant to take the wheel
                wdg.installEventFilter(self._nowheel)
                wdg.setFocusPolicy(Qt.StrongFocus)

        try:
            self.plots.spin_traj.setValue(float(self.state.get("traj_hz") or 0.0))
        except Exception:
            pass
        self.plots.spin_traj.valueChanged.connect(self._touch)

        self._set_enabled(False)
        dev = get_device(True)
        self.chip_gpu.setText(dev.name if dev.accelerated else "CPU only")
        style_chip(self.chip_gpu, "ok" if dev.accelerated else "warn")
        self._log(f"device: {dev}")
        for note in getattr(self, "_corrections", []):
            self._log(f"corrected a saved setting — {note}")
            self._log("  re-run any analysis whose force numbers you intend to use.")

        self._apply_state(self.state)
        self._on_force_method()
        # After _apply_state, so anything it hid for its own reasons -- the DXF
        # outline chooser, the manual-threshold box -- is already hidden when
        # the mode filter runs and does not get un-hidden by it.
        self._restore_sections()
        self._wire_persistence()
        # Which code is actually running. An update that applies but never
        # takes effect is invisible without this, and that is precisely the
        # failure that went unnoticed for three releases.
        try:
            act = U.active_overlay_path()
            self._log(f"running {U.current_version()}"
                      + (f" from patch {act.name}" if act else " as installed"))
        except Exception:
            pass
        U.mark_overlay_verified()       # this build imports; trust the overlay
        QTimer.singleShot(0, self._after_show)

    def _after_show(self):
        """Deferred startup work, once the window is actually on screen."""
        if self._startup_warning:
            QMessageBox.warning(self, "Update reverted", self._startup_warning)
            self._log(self._startup_warning.replace("\n\n", " "))

        geom = self.state.get("window_geometry")
        if geom:
            try:
                from PySide6.QtCore import QByteArray
                self.restoreGeometry(QByteArray.fromBase64(geom.encode("ascii")))
            except Exception:
                pass

        # Reopening what was open last time is the point of remembering paths,
        # but a clip that has been moved or deleted must not block the launch.
        last = self.state.get("video_path", "")
        if last and Path(last).exists():
            self.video = last
            self.lbl_video.setText(Path(last).name)
            self._log(f"reopening {Path(last).name}")
            self._reload()
        elif last:
            self._log(f"last video is no longer at {last}")

        # The launch check no longer opens anything. It used to call the update
        # dialog with quiet=True, which still ran ``dlg.exec()`` -- so every
        # launch after an update reopened the very window you had just finished
        # with, reporting that you were up to date. Now it asks in the
        # background and, if there is something to install, makes the Update
        # button breathe. Noticing is left to you.
        if (self.state.get("check_updates_on_start")
                and self.state.get("update_channel")
                and U.read_marker() is None):
            # Bound to `self`, so Qt drops the pending call if the window is
            # destroyed first. Without that the timer can still fire on a
            # window that is already closing, starting a thread nothing will
            # ever wait for -- which is a crash on exit rather than a leak.
            QTimer.singleShot(1500, self, self._check_updates_quietly)

    # ---- chrome ----------------------------------------------------------

    def _build_header(self) -> QWidget:
        h = QFrame(); h.setObjectName("Header"); h.setFixedHeight(58)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(18, 0, 18, 0)
        lay.setSpacing(10)

        dot = QLabel("❮❯"); dot.setObjectName("AppMark")
        name = QLabel(APP_NAME); name.setObjectName("AppName")
        tag = QLabel("muscle-driven soft robot kinematics"); tag.setObjectName("Tagline")
        lay.addWidget(dot); lay.addWidget(name); lay.addSpacing(4); lay.addWidget(tag)
        lay.addStretch(1)

        self.chip_rate = QLabel("no video"); self.chip_rate.setObjectName("Chip")
        self.chip_mode = QLabel("markerless"); self.chip_mode.setObjectName("Chip")
        self.chip_gpu = QLabel("…"); self.chip_gpu.setObjectName("Chip")
        self.chip_key = QLabel("—"); self.chip_key.setObjectName("Chip")
        self.chip_key.setToolTip("How the robot is told apart from the medium")
        self.chip_version = QLabel(f"v{U.current_version()}")
        self.chip_version.setObjectName("Chip")
        self.chip_version.setToolTip("Installed version. Click Update to check the channel.")
        for c in (self.chip_rate, self.chip_mode, self.chip_key, self.chip_gpu,
                  self.chip_version):
            lay.addWidget(c)

        # Three icon buttons. Square, same size, no labels: the header carries
        # five status chips already and three words of chrome beside them read
        # as more information to parse. Each keeps its tooltip, and the tooltip
        # is a sentence rather than a repeat of the removed word -- an icon
        # needs to be explainable, not merely named.
        self.btn_mode = QPushButton()
        self.btn_mode.setObjectName("Ghost")
        self.btn_mode.setFixedSize(34, 30)
        self.btn_mode.setIconSize(QSize(17, 17))
        self.btn_mode.clicked.connect(self._toggle_ui_mode)
        lay.addWidget(self.btn_mode)

        self.btn_theme = QPushButton()
        self.btn_theme.setObjectName("Ghost")
        self.btn_theme.setFixedSize(34, 30)
        self.btn_theme.setIconSize(QSize(17, 17))
        self.btn_theme.clicked.connect(self._toggle_theme)
        self._sync_theme_button()
        lay.addWidget(self.btn_theme)

        self.btn_update = QPushButton()
        self.btn_update.setObjectName("Ghost")
        # Height fixed, width free: this one grows a version number beside its
        # icon when a release is waiting.
        self.btn_update.setFixedHeight(30)
        self.btn_update.setMinimumWidth(34)
        self.btn_update.setIconSize(QSize(17, 17))
        self.btn_update.setToolTip("Check for and install a new version")
        self.btn_update.clicked.connect(lambda: self._check_updates(quiet=False))
        lay.addWidget(self.btn_update)
        self._sync_header_icons()
        return h

    def _sync_header_icons(self):
        """Repaint the header icons in the current theme's text colour.

        Icons are pixmaps, so unlike everything else in the window they are not
        reached by a stylesheet change. A switch to light mode would otherwise
        leave three near-white glyphs on a white strip.
        """
        light = self.state.get("theme_mode", "dark") == "light"
        col = C["text"]
        # Each icon shows the state you would be *in* after pressing it, which
        # is the convention every OS uses for these two toggles.
        self.btn_theme.setIcon(glyph("moon" if light else "sun", col))
        self.btn_mode.setIcon(glyph("sliders" if self.ui_mode == "simple" else "list", col))
        self.btn_update.setIcon(glyph("download", col))

    # ---- sidebar ---------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 8, 0)
        v.setSpacing(11)

        # Cards are built here in whatever order is convenient and *added* in
        # SIDEBAR_ORDER at the end, so the order a reader sees on screen is
        # stated in one place instead of being an accident of which control was
        # written first.
        cards: dict[str, Card] = {}

        # -------------------------------------------------- input
        c = cards["input"] = Card("Input", key="input")
        self.btn_video = QPushButton("Choose video…")
        self.btn_video.clicked.connect(self._pick_video)
        self.lbl_video = QLabel("none selected"); self.lbl_video.setObjectName("Hint")
        self.lbl_video.setWordWrap(True)
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(7)
        rl.addWidget(QLabel("Video")); rl.addWidget(HelpBadge(HELP["video"]))
        rl.addStretch(1); rl.addWidget(self.btn_video)
        c.add_widget(row); c.add_widget(self.lbl_video)

        self.btn_dxf = QPushButton("Choose DXF…")
        self.btn_dxf.clicked.connect(self._pick_dxf)
        self.btn_dxf_clear = QPushButton("Clear"); self.btn_dxf_clear.setFixedWidth(56)
        self.btn_dxf_clear.clicked.connect(self._clear_dxf)
        self.lbl_dxf = QLabel("none — markerless mode"); self.lbl_dxf.setObjectName("Hint")
        self.lbl_dxf.setWordWrap(True)
        row2 = QWidget(); rl2 = QHBoxLayout(row2); rl2.setContentsMargins(0, 0, 0, 0); rl2.setSpacing(7)
        rl2.addWidget(QLabel("CAD outline")); rl2.addWidget(HelpBadge(HELP["dxf"]))
        rl2.addStretch(1); rl2.addWidget(self.btn_dxf); rl2.addWidget(self.btn_dxf_clear)
        c.add_widget(row2); c.add_widget(self.lbl_dxf)

        # Which outline in the drawing is the robot. Hidden unless the file
        # actually contains more than one candidate, which most do not.
        self.cmb_loop = QComboBox()
        self.cmb_loop.currentIndexChanged.connect(self._on_loop_changed)
        self.row_loop = c.add_row("Outline", self.cmb_loop, HELP["dxf_outline"])
        self.row_loop.setVisible(False)

        self.spin_dxfscale = QDoubleSpinBox()
        self.spin_dxfscale.setRange(0.0001, 1000.0)
        self.spin_dxfscale.setDecimals(4)
        self.spin_dxfscale.setSingleStep(0.05)
        self.spin_dxfscale.setValue(1.0)
        self.spin_dxfscale.setPrefix("× ")
        self.spin_dxfscale.valueChanged.connect(self._on_dxf_scale)
        c.add_row("Drawing scale", self.spin_dxfscale, HELP["dxf_scale"])

        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(0.0, 10000.0)
        self.spin_width.setDecimals(3)
        self.spin_width.setSingleStep(0.1)
        self.spin_width.setSuffix(" mm")
        self.spin_width.setSpecialValueText("from drawing")
        self.spin_width.valueChanged.connect(self._on_width_changed)
        c.add_row("True width", self.spin_width, HELP["known_width"])

        self.btn_fit_scale = QPushButton("Set scale from true width")
        self.btn_fit_scale.setObjectName("Ghost")
        self.btn_fit_scale.setToolTip(
            "Work out the drawing scale from the true width you entered")
        self.btn_fit_scale.clicked.connect(self._scale_from_true_width)
        c.add_widget(self.btn_fit_scale)

        self.lbl_dims = QLabel(""); self.lbl_dims.setObjectName("Readout")
        self.lbl_dims.setWordWrap(True)
        c.add_widget(self.lbl_dims)


        self.lbl_info = QLabel(""); self.lbl_info.setObjectName("Readout")
        self.lbl_info.setWordWrap(True)
        c.add_widget(self.lbl_info)

        # -------------------------------------------------- segmentation
        c = cards["segmentation"] = Card("Segmentation", key="segmentation")
        self.cmb_keying = QComboBox()
        self.cmb_keying.addItems(["Auto", "Color (a*b*)", "Brightness"])
        self.cmb_keying.currentIndexChanged.connect(self._reload_needed)
        c.add_row("Keying", self.cmb_keying, HELP["keying"])

        self.spin_cfrac = QDoubleSpinBox(); self.spin_cfrac.setRange(0.05, 0.90)
        self.spin_cfrac.setSingleStep(0.05); self.spin_cfrac.setValue(0.30)
        self.spin_cfrac.setDecimals(2)
        self.spin_cfrac.valueChanged.connect(self._on_param)
        c.add_row("Color cut", self.spin_cfrac, HELP["color_frac"])

        self.spin_env = QDoubleSpinBox(); self.spin_env.setRange(1.0, 4.0)
        self.spin_env.setSingleStep(0.05); self.spin_env.setValue(1.10)
        self.spin_env.valueChanged.connect(self._on_param)
        c.add_row("Body envelope", self.spin_env, HELP["envelope"], advanced=True)

        self.cmb_thr = QComboBox(); self.cmb_thr.addItems(["Auto (Otsu)", "Manual"])
        self.cmb_thr.currentIndexChanged.connect(self._on_thr_mode)
        c.add_row("Threshold", self.cmb_thr, HELP["threshold_mode"])

        self.spin_thr = QSpinBox(); self.spin_thr.setRange(1, 254); self.spin_thr.setValue(30)
        self.spin_thr.setSuffix(" levels"); self.spin_thr.setEnabled(False)
        self.spin_thr.valueChanged.connect(self._on_param)
        c.add_row("Manual value", self.spin_thr, HELP["threshold"])

        self.spin_open = QSpinBox(); self.spin_open.setRange(1, 31)
        self.spin_open.setSingleStep(2); self.spin_open.setValue(3); self.spin_open.setSuffix(" px")
        self.spin_open.valueChanged.connect(self._on_param)
        c.add_row("Despeckle", self.spin_open, HELP["despeckle"], advanced=True)

        self.spin_close = QSpinBox(); self.spin_close.setRange(1, 41)
        self.spin_close.setSingleStep(2); self.spin_close.setValue(7); self.spin_close.setSuffix(" px")
        self.spin_close.valueChanged.connect(self._on_param)
        c.add_row("Fill holes", self.spin_close, HELP["fill_holes"], advanced=True)

        self.spin_minarea = QDoubleSpinBox(); self.spin_minarea.setDecimals(4)
        self.spin_minarea.setRange(0.0010, 1.0000); self.spin_minarea.setSingleStep(0.005)
        self.spin_minarea.setValue(0.0500); self.spin_minarea.setSuffix(" %")
        self.spin_minarea.valueChanged.connect(self._on_param)
        c.add_row("Min blob size", self.spin_minarea, HELP["min_area"], advanced=True)

        self.spin_gap = QDoubleSpinBox(); self.spin_gap.setRange(0.0, 3.0)
        self.spin_gap.setSingleStep(0.1); self.spin_gap.setValue(1.0)
        self.spin_gap.setSuffix(" × body")
        self.spin_gap.valueChanged.connect(self._on_param)
        c.add_row("Occlusion gap", self.spin_gap, HELP["gap_factor"], advanced=True)

        self.spin_bg = QSpinBox(); self.spin_bg.setRange(10, 400); self.spin_bg.setValue(60)
        self.spin_bg.setSuffix(" frames")
        self.spin_bg.valueChanged.connect(self._reload_needed)
        c.add_row("Background", self.spin_bg, HELP["background_frames"], advanced=True)

        self.btn_rebuild = QPushButton("Rebuild background")
        self.btn_rebuild.setObjectName("Ghost")
        self.btn_rebuild.clicked.connect(self._reload)
        self.btn_rebuild.setVisible(False)
        c.add_widget(self.btn_rebuild)

        # -------------------------------------------------- fitting
        # The whole card is advanced. Nothing in it needs changing to analyse a
        # clip -- it is the solver's own dials, and the defaults are the ones
        # every number in the README was measured with.
        c = cards["fitting"] = Card("Shape fitting", key="shape_fitting",
                                     advanced=True)
        self.spin_tau = QDoubleSpinBox(); self.spin_tau.setRange(1, 60); self.spin_tau.setValue(12)
        self.spin_tau.setSuffix(" px"); self.spin_tau.valueChanged.connect(self._on_param)
        c.add_row("Kernel τ start", self.spin_tau, HELP["tau"])

        self.spin_tauf = QDoubleSpinBox(); self.spin_tauf.setRange(0.5, 10)
        self.spin_tauf.setSingleStep(0.5); self.spin_tauf.setValue(2.5); self.spin_tauf.setSuffix(" px")
        self.spin_tauf.valueChanged.connect(self._on_param)
        c.add_row("Kernel τ final", self.spin_tauf, HELP["tau_final"])

        self.spin_restarts = QSpinBox(); self.spin_restarts.setRange(1, 256)
        self.spin_restarts.setValue(64)
        c.add_row("Restarts", self.spin_restarts, HELP["restarts"])

        self.chk_earlystop = QCheckBox("stop each fit once it converges")
        self.chk_earlystop.setChecked(True)
        rowes = QWidget(); res_ = QHBoxLayout(rowes)
        res_.setContentsMargins(0, 0, 0, 0); res_.setSpacing(7)
        res_.addWidget(self.chk_earlystop); res_.addWidget(HelpBadge(HELP["early_stop"]))
        res_.addStretch(1)
        c.add_widget(rowes)

        self.spin_cover = QDoubleSpinBox(); self.spin_cover.setRange(0, 20)
        self.spin_cover.setSingleStep(0.5); self.spin_cover.setValue(3.0)
        self.spin_cover.valueChanged.connect(self._on_param)
        c.add_row("Coverage weight", self.spin_cover, HELP["coverage"])

        self.spin_prior = QDoubleSpinBox(); self.spin_prior.setRange(0, 2)
        self.spin_prior.setSingleStep(0.05); self.spin_prior.setValue(0.35)
        c.add_row("Smoothness prior", self.spin_prior, HELP["scale_prior"])

        self.chk_features = QCheckBox("fit interior features (holes, beams)")
        self.chk_features.setChecked(True)
        self.chk_features.stateChanged.connect(self._on_features_toggled)
        rowf = QWidget(); rf = QHBoxLayout(rowf); rf.setContentsMargins(0, 0, 0, 0); rf.setSpacing(7)
        rf.addWidget(self.chk_features); rf.addWidget(HelpBadge(HELP["features"]))
        rf.addStretch(1)
        c.add_widget(rowf)

        self.spin_featw = QDoubleSpinBox(); self.spin_featw.setRange(0.0, 5.0)
        self.spin_featw.setSingleStep(0.1); self.spin_featw.setValue(1.0)
        self.spin_featw.valueChanged.connect(self._on_param)
        c.add_row("Feature weight", self.spin_featw, HELP["feature_weight"])

        self.spin_maxscale = QDoubleSpinBox(); self.spin_maxscale.setRange(0.10, 0.95)
        self.spin_maxscale.setSingleStep(0.05); self.spin_maxscale.setValue(0.60)
        c.add_row("Max size change", self.spin_maxscale, HELP["max_scale"])

        rule_a = QFrame(); rule_a.setObjectName("cardRule"); rule_a.setFixedHeight(1)
        c.add_widget(rule_a)

        self.spin_widthw = QDoubleSpinBox(); self.spin_widthw.setRange(0.0, 20.0)
        self.spin_widthw.setSingleStep(0.5); self.spin_widthw.setValue(2.0)
        self.spin_widthw.setSpecialValueText("off")
        self.spin_widthw.valueChanged.connect(self._on_param)
        c.add_row("Width hold", self.spin_widthw, HELP["width_weight"])

        self.spin_widthtol = QDoubleSpinBox(); self.spin_widthtol.setRange(0.5, 30.0)
        self.spin_widthtol.setSingleStep(0.5); self.spin_widthtol.setValue(4.0)
        self.spin_widthtol.setSuffix(" %")
        self.spin_widthtol.valueChanged.connect(self._on_param)
        c.add_row("Width tolerance", self.spin_widthtol, HELP["width_tol"])

        self.spin_lenover = QDoubleSpinBox(); self.spin_lenover.setRange(0.0, 60.0)
        self.spin_lenover.setSingleStep(0.5); self.spin_lenover.setValue(3.0)
        self.spin_lenover.setSuffix(" px")
        self.spin_lenover.valueChanged.connect(self._on_param)
        c.add_row("Length overshoot", self.spin_lenover, HELP["length_overshoot"])

        # -------------------------------------------------- force
        c = cards["force"] = Card("Force", key="force")
        self.cmb_force = QComboBox()
        self.cmb_force.addItems(["Simulated LUT (COMSOL)", "Cvetkovic Model"])
        self.cmb_force.setCurrentIndex(1)
        self.cmb_force.currentIndexChanged.connect(self._on_force_method)
        c.add_row("Method", self.cmb_force, HELP["force_method"])

        self.btn_lut = QPushButton("Choose simulation…")
        self.btn_lut.clicked.connect(self._pick_lut)
        self.btn_lut_clear = QPushButton("Clear"); self.btn_lut_clear.setFixedWidth(56)
        self.btn_lut_clear.clicked.connect(self._clear_lut)
        self.row_lut = QWidget()
        rl3 = QHBoxLayout(self.row_lut); rl3.setContentsMargins(0, 0, 0, 0); rl3.setSpacing(7)
        rl3.addWidget(QLabel("Curve")); rl3.addWidget(HelpBadge(HELP["force_lut"]))
        rl3.addStretch(1); rl3.addWidget(self.btn_lut); rl3.addWidget(self.btn_lut_clear)
        c.add_widget(self.row_lut)
        self.lbl_lut = QLabel("no simulated curve loaded"); self.lbl_lut.setObjectName("Hint")
        self.lbl_lut.setWordWrap(True)
        c.add_widget(self.lbl_lut)

        # Beam-model constants. Defaults are the values from SampleForce.m.
        self.beam_rows = []

        def beam_row(label, lo, hi, val, dec, suffix, help_key):
            w = QDoubleSpinBox()
            w.setRange(lo, hi); w.setDecimals(dec); w.setValue(val)
            w.setSuffix(suffix)
            w.setSingleStep(10 ** -max(dec - 2, 0))
            w.valueChanged.connect(self._on_beam_changed)
            row = c.add_row(label, w, HELP[help_key])
            self.beam_rows.append(row)
            return w

        self.spin_E = beam_row("Young's modulus", 0.1, 1e6, 293.0, 2, " kPa", "beam_E")
        self.spin_t = beam_row("Beam thickness", 0.001, 100.0, 1.100, 3, " mm", "beam_geom")
        self.spin_bw = beam_row("Beam width", 0.001, 100.0, 5.060, 3, " mm", "beam_geom")
        self.spin_slot = beam_row("Beam slot width", 0.0, 100.0, 0.9625, 4, " mm", "beam_slot")
        self.spin_Lleg2leg = beam_row("Leg to leg", 0.001, 1000.0, 8.030, 3, " mm", "beam_geom")
        self.spin_arm = beam_row("Muscle offset", 0.001, 1000.0, 1.238, 3, " mm", "beam_geom")
        self.spin_leg_long = beam_row("Leg length (long)", 0.001, 1000.0, 4.125, 3, " mm", "beam_geom")
        self.spin_leg_short = beam_row("Leg length (short)", 0.001, 1000.0, 3.300, 3, " mm", "beam_geom")

        self.cmb_rest = QComboBox()
        self.cmb_rest.addItems(["Maximum (Cvetkovic Model)", "Robust (upper quartile)"])
        self.cmb_rest.currentIndexChanged.connect(self._on_beam_changed)
        self.beam_rows.append(c.add_row("Resting length", self.cmb_rest, HELP["beam_rest"],
                                advanced=True))

        self.lbl_beam = QLabel(""); self.lbl_beam.setObjectName("Readout")
        self.lbl_beam.setWordWrap(True)
        c.add_widget(self.lbl_beam)
        self.beam_rows.append(self.lbl_beam)

        # -------------------------------------------------- placement
        c = cards["placement"] = Card("Target placement", key="target_placement")
        self.chk_manual = QCheckBox("Place the outline by hand")
        self.chk_manual.stateChanged.connect(self._on_manual_toggled)
        rowm = QWidget(); rm = QHBoxLayout(rowm); rm.setContentsMargins(0, 0, 0, 0); rm.setSpacing(7)
        rm.addWidget(self.chk_manual); rm.addWidget(HelpBadge(HELP["manual_placement"]))
        rm.addStretch(1)
        c.add_widget(rowm)

        self.lbl_place = QLabel("Load a DXF to place it.")
        self.lbl_place.setObjectName("Hint")
        self.lbl_place.setWordWrap(True)
        c.add_widget(self.lbl_place)

        rowb = QWidget(); rb = QHBoxLayout(rowb); rb.setContentsMargins(0, 0, 0, 0); rb.setSpacing(7)
        self.btn_snap = QPushButton("Snap to current fit")
        self.btn_snap.setObjectName("Ghost")
        self.btn_snap.setToolTip("Adopt the automatic fit on this frame as the starting placement")
        self.btn_snap.clicked.connect(self._snap_to_fit)
        self.btn_place_clear = QPushButton("Clear")
        self.btn_place_clear.setObjectName("Ghost")
        self.btn_place_clear.setFixedWidth(60)
        self.btn_place_clear.clicked.connect(self._clear_placement)
        rb.addWidget(self.btn_snap, 1); rb.addWidget(self.btn_place_clear)
        c.add_widget(rowb)

        self.btn_automatch = QPushButton("Match drawing to video")
        self.btn_automatch.setToolTip(
            "Fit every candidate outline to real frames and keep the one that agrees best")
        self.btn_automatch.clicked.connect(self._automatch)
        rowa = QWidget(); ra = QHBoxLayout(rowa); ra.setContentsMargins(0, 0, 0, 0); ra.setSpacing(7)
        ra.addWidget(self.btn_automatch, 1); ra.addWidget(HelpBadge(HELP["automatch"]))
        c.add_widget(rowa)

        self.lbl_pose = QLabel("—"); self.lbl_pose.setObjectName("Readout")
        self.lbl_pose.setWordWrap(True)
        c.add_widget(self.lbl_pose)

        # ---- region of interest ------------------------------------------
        rule = QFrame(); rule.setObjectName("cardRule"); rule.setFixedHeight(1)
        c.add_widget(rule)

        self.chk_roi = QCheckBox("Limit tracking to a region")
        self.chk_roi.stateChanged.connect(self._on_roi_toggle)
        rowr = QWidget(); rr = QHBoxLayout(rowr); rr.setContentsMargins(0, 0, 0, 0); rr.setSpacing(7)
        rr.addWidget(self.chk_roi); rr.addWidget(HelpBadge(HELP["roi"]))
        rr.addStretch(1)
        c.add_widget(rowr)

        self.lbl_roi = QLabel("Whole frame.")
        self.lbl_roi.setObjectName("Hint"); self.lbl_roi.setWordWrap(True)
        c.add_widget(self.lbl_roi)

        self.chk_appearance = QCheckBox("Lock onto appearance instead of color")
        rowa2 = QWidget(); ra2 = QHBoxLayout(rowa2)
        ra2.setContentsMargins(0, 0, 0, 0); ra2.setSpacing(7)
        ra2.addWidget(self.chk_appearance); ra2.addWidget(HelpBadge(HELP["appearance"]))
        ra2.addStretch(1)
        c.add_widget(rowa2)

        self.btn_roi_clear = QPushButton("Clear region")
        self.btn_roi_clear.setObjectName("Ghost")
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        c.add_widget(self.btn_roi_clear)

        # -------------------------------------------------- analysis
        # Advanced as a whole: everything in it is post-processing of a
        # finished series rather than a decision about how to measure, and the
        # defaults are right for any clip the tracker can follow.
        c = cards["analysis"] = Card("Analysis", key="analysis", advanced=True)
        self.spin_smooth = QDoubleSpinBox(); self.spin_smooth.setRange(0, 2000)
        self.spin_smooth.setValue(100); self.spin_smooth.setSuffix(" ms")
        c.add_row("Smoothing", self.spin_smooth, HELP["smoothing"])

        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0, 1)
        self.spin_conf.setSingleStep(0.05); self.spin_conf.setValue(0.50)
        c.add_row("Min confidence", self.spin_conf, HELP["min_confidence"])

        self.spin_gapms = QDoubleSpinBox(); self.spin_gapms.setRange(0, 5000)
        self.spin_gapms.setValue(400); self.spin_gapms.setSuffix(" ms")
        c.add_row("Max bridged gap", self.spin_gapms, HELP["max_gap"], advanced=True)

        self.spin_ppm = QDoubleSpinBox(); self.spin_ppm.setRange(0, 10000)
        self.spin_ppm.setDecimals(3); self.spin_ppm.setSpecialValueText("auto")
        self.spin_ppm.setSuffix(" px/mm")
        c.add_row("Calibration", self.spin_ppm, HELP["px_per_mm"])

        # -------------------------------------------------- output
        c = cards["output"] = Card("Output", key="output")
        self.cmb_scale = QComboBox(); self.cmb_scale.addItems(["1.0 — full", "0.5 — half", "0.25 — quarter"])
        self.cmb_scale.currentIndexChanged.connect(self._reload_needed)
        c.add_row("Decode scale", self.cmb_scale, HELP["decode_scale"])

        self.chk_overlay = QCheckBox("write overlay.mp4"); self.chk_overlay.setChecked(True)
        r = QWidget(); rh = QHBoxLayout(r); rh.setContentsMargins(0, 0, 0, 0); rh.setSpacing(7)
        rh.addWidget(self.chk_overlay); rh.addWidget(HelpBadge(HELP["overlay"])); rh.addStretch(1)
        c.add_widget(r)

        self.chk_gpu = QCheckBox("use GPU"); self.chk_gpu.setChecked(True)
        self.chk_gpu.stateChanged.connect(self._on_param)
        r2 = QWidget(); rh2 = QHBoxLayout(r2); rh2.setContentsMargins(0, 0, 0, 0); rh2.setSpacing(7)
        rh2.addWidget(self.chk_gpu); rh2.addWidget(HelpBadge(HELP["gpu"])); rh2.addStretch(1)
        c.add_widget(r2)

        # -------------------------------------------------- configuration
        # Pinned, never collapsible, and deliberately the most compact card in
        # the column: it is the only one that is about the program rather than
        # about the experiment, so it should be reachable at all times and take
        # as little room as possible while being so. The paragraph that used to
        # explain it is a tooltip now -- it was three lines of text explaining
        # two buttons whose labels already say what they do.
        c = cards["configuration"] = Card("Configuration", key="configuration",
                                          collapsible=False, compact=True)
        rowc = QWidget(); rc = QHBoxLayout(rowc); rc.setContentsMargins(0, 0, 0, 0); rc.setSpacing(6)
        self.btn_cfg_save = QPushButton("Save…"); self.btn_cfg_save.setObjectName("Ghost")
        self.btn_cfg_load = QPushButton("Load…"); self.btn_cfg_load.setObjectName("Ghost")
        self.btn_cfg_reset = QPushButton("Reset"); self.btn_cfg_reset.setObjectName("Ghost")
        self.btn_cfg_save.setToolTip("Write every current setting to a .rtcfg file, "
                                     "so this run can be reproduced or shared")
        self.btn_cfg_load.setToolTip("Load settings from a .rtcfg file")
        self.btn_cfg_reset.setToolTip("Return every setting to the value the app ships with")
        self.btn_cfg_save.clicked.connect(self._save_config)
        self.btn_cfg_load.clicked.connect(self._load_config)
        self.btn_cfg_reset.clicked.connect(self._reset_config)
        for b in (self.btn_cfg_save, self.btn_cfg_load, self.btn_cfg_reset):
            rc.addWidget(b, 1)
        c.add_widget(rowc)

        self.chk_update_start = QCheckBox("check for updates at launch")
        rowu = QWidget(); ru = QHBoxLayout(rowu); ru.setContentsMargins(0, 0, 0, 0); ru.setSpacing(7)
        ru.addWidget(self.chk_update_start); ru.addWidget(HelpBadge(HELP["update_channel"]))
        ru.addStretch(1)
        c.add_widget(rowu)

        # ---- assemble, in the order work actually happens in ----------------
        self.cards = cards
        for key in self.SIDEBAR_ORDER:
            card = cards.get(key)
            if card is None:
                continue
            card.toggled.connect(self._on_card_toggled)
            v.addWidget(card)

        # Run, the progress bar and the log all live in the viewer column now.
        # The button belongs next to the thing it acts on, and the log belongs
        # under the picture it describes -- at the bottom of a long scrolling
        # sidebar both were routinely off-screen at the moment they mattered.
        v.addStretch(1)

        scroll.setWidget(inner)
        scroll.setMinimumWidth(400)
        return scroll

    # Configuration first because it is not part of the workflow and should not
    # move; the rest in the order a clip passes through the program -- what you
    # loaded, where the robot is, how it is told apart from the medium, how the
    # outline is fitted to it, what that becomes in newtons, how the series is
    # cleaned up, and what gets written out.
    SIDEBAR_ORDER = ("configuration", "input", "placement", "segmentation",
                     "fitting", "force", "analysis", "output")

    # Sections that start closed for someone who has never opened the program.
    # Only two: a column of collapsed boxes is as unhelpful as an endless one,
    # and these are the two nobody needs on a first run.
    DEFAULT_COLLAPSED = {"shape_fitting", "output"}

    # ---- panel state -----------------------------------------------------

    def _on_card_toggled(self, key: str, collapsed: bool) -> None:
        self.collapsed_sections[key] = bool(collapsed)
        self._touch()

    def _restore_sections(self) -> None:
        """Apply the remembered collapsed state, then the simple/advanced mode."""
        for key, card in self.cards.items():
            want = self.collapsed_sections.get(key, key in self.DEFAULT_COLLAPSED)
            card.set_collapsed(bool(want))
        self._apply_ui_mode()

    def _toggle_ui_mode(self) -> None:
        self.ui_mode = "simple" if self.ui_mode == "advanced" else "advanced"
        self._apply_ui_mode()
        self._touch()

    def _apply_ui_mode(self) -> None:
        advanced = self.ui_mode == "advanced"
        for card in self.cards.values():
            card.setVisible(card.set_show_advanced(advanced))
        if hasattr(self, "btn_update"):
            self._sync_header_icons()
        self.btn_mode.setToolTip(
            "Showing every control, including the solver's own settings. "
            "Click for the short list." if advanced else
            "Showing the controls a run needs. Click to reveal the solver "
            "settings and the morphology parameters as well.")

    # ---- viewer ----------------------------------------------------------

    # ---- plots column ----------------------------------------------------

    VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".mts", ".m2ts"}

    def _build_plots_column(self) -> QWidget:
        """Queue on top, plots in the middle, Next video at the bottom.

        The queue exists because a session is a folder of clips, not one clip.
        Re-opening a file dialog for each of twenty recordings -- and having to
        remember which of them you already did -- is the real friction in a
        day's analysis, and neither problem is visible from inside a single run.
        """
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        head = QWidget(); hh = QHBoxLayout(head)
        hh.setContentsMargins(2, 0, 2, 0); hh.setSpacing(7)
        hh.addWidget(QLabel("Videos in this folder"))
        hh.addWidget(HelpBadge(HELP["video_queue"]))
        hh.addStretch(1)
        self.lbl_queue_count = QLabel(""); self.lbl_queue_count.setObjectName("Readout")
        hh.addWidget(self.lbl_queue_count)
        v.addWidget(head)

        self.lbl_eta = QLabel(""); self.lbl_eta.setObjectName("Hint")
        self.lbl_eta.setWordWrap(True)
        v.addWidget(self.lbl_eta)

        self.list_videos = QListWidget()
        self.list_videos.setMaximumHeight(146)
        self.list_videos.itemActivated.connect(self._on_queue_pick)
        self.list_videos.itemClicked.connect(self._on_queue_pick)
        v.addWidget(self.list_videos)

        self.plots = PlotPanel()
        self.plots.selectionAnalyzed.connect(self._on_region)
        v.addWidget(self.plots, 1)

        row = QWidget(); rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(7)

        self.btn_export_sel = QPushButton("Export selected region")
        self.btn_export_sel.setObjectName("Ghost")
        self.btn_export_sel.setToolTip(
            "Write the highlighted time window as its own CSV, figure and "
            "metadata, beside the full run and named with a _subset suffix.")
        self.btn_export_sel.setEnabled(False)
        self.btn_export_sel.clicked.connect(self._export_selection)
        rl.addWidget(self.btn_export_sel, 1)

        self.btn_next_video = QPushButton("Next video  →")
        self.btn_next_video.setObjectName("Ghost")
        self.btn_next_video.setToolTip(
            "Load the next clip in this folder. Parameters and the CAD outline "
            "are kept; the fit and the plots are cleared.")
        self.btn_next_video.setEnabled(False)
        self.btn_next_video.clicked.connect(self._next_video)
        rl.addWidget(self.btn_next_video, 1)
        v.addWidget(row)
        return w

    # ---- how long the rest of the folder will take -----------------------

    MAX_HISTORY = 40

    def _record_throughput(self, res):
        """Remember what this run cost, as frames x pixels per second.

        Neither frame count nor resolution predicts runtime on its own -- a
        short 4K clip and a long postage-stamp clip can take the same time --
        but their product tracks it closely, because the per-frame cost is
        dominated by work that scales with area. One number per run, and the
        median of them is the machine's rate.
        """
        try:
            px = int(res.info.width) * int(res.info.height)
            n = int(len(res.table))
            secs = float(res.elapsed_s)
            if n <= 0 or px <= 0 or secs <= 0:
                return
            hist = list(self.state.get("run_history") or [])
            hist.append([n, px, secs])
            self.state["run_history"] = hist[-self.MAX_HISTORY:]
            S.save_settings(self.state)
        except Exception:
            pass

    def _seconds_per_unit(self) -> float | None:
        """Median seconds per frame-pixel, or None if nothing has been timed.

        Median, not mean: one run that was aborted late, or one where the
        machine was busy, would drag an average badly and there are only ever a
        handful of samples.
        """
        hist = self.state.get("run_history") or []
        rates = []
        for entry in hist:
            try:
                n, px, secs = float(entry[0]), float(entry[1]), float(entry[2])
                if n > 0 and px > 0 and secs > 0:
                    rates.append(secs / (n * px))
            except (TypeError, ValueError, IndexError):
                continue
        if not rates:
            return None
        rates.sort()
        mid = len(rates) // 2
        return (rates[mid] if len(rates) % 2 else
                0.5 * (rates[mid - 1] + rates[mid]))

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 90:
            return f"{seconds:.0f} s"
        if seconds < 5400:
            return f"{seconds / 60:.0f} min"
        return f"{seconds / 3600:.1f} h"

    def _update_eta(self):
        """Estimate the remaining work in this folder, from measured rates."""
        rate = self._seconds_per_unit()
        pending = [q for q in getattr(self, "_queue", []) if not self._is_analyzed(q)]
        if not pending:
            self.lbl_eta.setText("")
            return
        if rate is None:
            self.lbl_eta.setText(f"{len(pending)} to do · time unknown until "
                                 f"one run has been timed")
            return
        sized = [self._sizes.get(str(q)) for q in pending]
        known = [sz for sz in sized if sz]
        if not known:
            self.lbl_eta.setText(f"{len(pending)} to do · measuring…")
            return
        total = sum(n * px for n, px in known)
        # Unsized clips are charged the average of the ones we do know, so the
        # figure does not silently drop the files it has not measured yet.
        if len(known) < len(sized):
            total += (len(sized) - len(known)) * (total / max(len(known), 1))
        est = total * rate
        approx = "~" if len(known) == len(sized) else "≥~"
        self.lbl_eta.setText(f"{len(pending)} to do · {approx}"
                             f"{self._fmt_duration(est)} remaining")

    def _measure_queue_sizes(self):
        """Read frame count and resolution for each clip, off the GUI thread.

        One light ffprobe per file, header only -- a few milliseconds each, and
        cached by path so a folder is measured once per session. It still has no
        business happening on the GUI thread: twenty files on a network share is
        a visible stall, and this is only ever decorating a label.
        """
        want = [str(q) for q in getattr(self, "_queue", [])
                if str(q) not in self._sizes]
        if not want:
            self._update_eta()
            return
        if getattr(self, "_sizer", None) is not None and self._sizer.isRunning():
            return

        class _Sizer(QThread):
            measured = Signal(dict)

            def run(self):
                from .ingest import quick_probe
                out = {}
                for path in want:
                    if self.isInterruptionRequested():
                        break
                    got = quick_probe(path)
                    if got:
                        n, w, h = got
                        out[path] = (n, w * h)
                self.measured.emit(out)

        self._retire("_sizer")
        self._sizer = _Sizer(self)
        self._sizer.measured.connect(self._on_sizes)
        self._sizer.start()

    def _on_sizes(self, sizes: dict):
        if getattr(self, "_closing", False):
            return
        self._sizes.update(sizes)
        self._update_eta()

    # ---- exporting a selected region -------------------------------------

    def _export_selection(self):
        """Write the highlighted window as its own set of files."""
        res = getattr(self, "_result", None)
        if res is None:
            QMessageBox.information(
                self, "Nothing to export",
                "Run an analysis first — a region can only be exported from a "
                "finished run, not from the live trace.")
            return
        ranges = self.plots.axis_ranges()
        sel = ranges.get("selection")
        if not sel:
            QMessageBox.information(
                self, "No region selected",
                "Drag left-to-right across any plot to select a time window, "
                "then press this again.")
            return
        try:
            from .pipeline import export_subset
            dest = pipeline_output_dir(self.outdir, self.video)
            paths = export_subset(res, dest, sel[0], sel[1], ranges)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            self._log(f"subset export failed: {exc}")
            return
        self._log(f"exported {sel[0]:.2f}–{sel[1]:.2f} s to "
                  + ", ".join(p.name for p in paths))
        QMessageBox.information(
            self, "Region exported",
            f"Wrote {len(paths)} files to\n{dest}\n\n"
            + "\n".join(p.name for p in paths))

    # ---- the video queue -------------------------------------------------

    @staticmethod
    def _natural_key(name: str):
        """Sort IMG_2 before IMG_10, the way a file manager does.

        A plain string sort puts IMG_10 first because "1" sorts before "2",
        which scrambles exactly the numbered sequences a camera produces.
        Splitting digit runs out and comparing them numerically is what people
        mean by alphabetical here.
        """
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", name)]

    def _sibling_videos(self):
        if not self.video:
            return []
        folder = Path(self.video).parent
        try:
            files = [q for q in folder.iterdir()
                     if q.is_file() and q.suffix.lower() in self.VIDEO_SUFFIXES]
        except OSError:
            return []
        return sorted(files, key=lambda q: self._natural_key(q.name))

    def _is_analyzed(self, video) -> bool:
        """Has this clip been run before? Answered by the output folder.

        Cheap and honest: pipeline.run writes into <outdir>/<clip name>/, so the
        folder's existence *is* the fact "there are results". No index to keep
        in sync, and it stays correct when you delete a folder to redo one.
        """
        if not self.outdir:
            return False
        try:
            return (Path(self.outdir) / video.stem).is_dir()
        except OSError:
            return False

    @staticmethod
    def _dot_icon(color):
        """A small filled or hollow dot, drawn rather than shipped as a file."""
        pm = QPixmap(14, 14)
        pm.fill(Qt.transparent)
        pnt = QPainter(pm)
        pnt.setRenderHint(QPainter.Antialiasing, True)
        if color:
            pnt.setBrush(QColor(color)); pnt.setPen(Qt.NoPen)
            pnt.drawEllipse(3, 3, 8, 8)
        else:
            pen = QPen(QColor(C["line"])); pen.setWidth(1)
            pnt.setPen(pen); pnt.setBrush(Qt.NoBrush)
            pnt.drawEllipse(3, 3, 8, 8)
        pnt.end()
        return QIcon(pm)

    def _refresh_queue(self):
        vids = self._sibling_videos()
        self._queue = vids
        self.list_videos.blockSignals(True)
        self.list_videos.clear()
        try:
            current = Path(self.video).resolve() if self.video else None
        except OSError:
            current = None
        done = 0
        for pth in vids:
            item = QListWidgetItem(pth.name)
            item.setData(Qt.UserRole, str(pth))
            if self._is_analyzed(pth):
                done += 1
                item.setIcon(self._dot_icon(C["ok"]))
                item.setToolTip("analyzed — results exist in the output folder")
            else:
                item.setIcon(self._dot_icon(None))
                item.setToolTip("not analyzed yet")
            self.list_videos.addItem(item)
            try:
                if current is not None and pth.resolve() == current:
                    self.list_videos.setCurrentItem(item)
            except OSError:
                pass
        self.list_videos.blockSignals(False)
        self.lbl_queue_count.setText(f"{done}/{len(vids)} analyzed" if vids else "")
        self.btn_next_video.setEnabled(self._next_index() is not None)
        self._update_eta()
        self._measure_queue_sizes()

    def _next_index(self):
        vids = getattr(self, "_queue", [])
        if not vids or not self.video:
            return None
        try:
            here = Path(self.video).resolve()
            idx = next(i for i, q in enumerate(vids) if q.resolve() == here)
        except (StopIteration, OSError):
            return None
        return idx + 1 if idx + 1 < len(vids) else None

    def _on_queue_pick(self, item):
        path = item.data(Qt.UserRole)
        if not path:
            return
        try:
            same = Path(path).resolve() == Path(self.video or "").resolve()
        except OSError:
            same = False
        if not same:
            self._switch_video(path)

    def _next_video(self):
        i = self._next_index()
        if i is None:
            self._log("this is the last video in the folder.")
            return
        self._switch_video(str(self._queue[i]))

    def _switch_video(self, path: str):
        """Load another clip, keeping everything that is not about this clip.

        Thresholds, the drawing, the force model and the output folder describe
        the *experiment* and carry over; the fit, the manual placement and the
        plotted results describe the *clip* and do not. Carrying a hand
        placement across would seed the next fit at a position measured in a
        different frame, which is worse than starting with no seed at all.
        """
        if getattr(self, "_running", False):
            QMessageBox.information(
                self, "Analysis running",
                "Stop the current analysis before loading another video.")
            return
        self._stop_play()
        self.video = path
        self.state["last_video_dir"] = str(Path(path).parent)
        self.lbl_video.setText(Path(path).name)
        self._forget_clip_state()
        self._touch()
        self._refresh_queue()
        self._reload()

    def _build_viewer(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(9)

        # Run sits above the picture, where it is always visible: an analysis is
        # started and aborted while watching the frame, not while scrolled to
        # the bottom of the parameter list.
        runrow = QWidget(); rr = QHBoxLayout(runrow)
        rr.setContentsMargins(2, 0, 2, 0); rr.setSpacing(9)
        self.btn_run = QPushButton("Run analysis")
        self.btn_run.setProperty("primary", True)
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setMinimumHeight(34)
        rr.addWidget(self.btn_run, 1)
        self.bar = QProgressBar(); self.bar.setVisible(False); self.bar.setTextVisible(True)
        self.bar.setMinimumHeight(34)
        rr.addWidget(self.bar, 2)
        v.addWidget(runrow)

        topbar = QWidget(); tb = QHBoxLayout(topbar)
        tb.setContentsMargins(2, 0, 2, 0); tb.setSpacing(8)
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Video", "Color distance (b*)"])
        self.cmb_view.setFixedWidth(190)
        self.cmb_view.currentIndexChanged.connect(self._on_param)
        tb.addWidget(QLabel("View")); tb.addWidget(HelpBadge(HELP["view_mode"]))
        tb.addWidget(self.cmb_view)
        tb.addStretch(1)
        tb.addWidget(HelpBadge(HELP["plot_region"]))
        self.btn_reset_zoom = QPushButton("Reset plots")
        self.btn_reset_zoom.setObjectName("Ghost")
        self.btn_reset_zoom.setToolTip(
            "Clear the selected region and return to the full clip")
        self.btn_reset_zoom.clicked.connect(
            lambda: (self.plots.clear_selection(), self.plots.reset_zoom()))
        tb.addWidget(self.btn_reset_zoom)
        v.addWidget(topbar)

        self.view = PreviewView()
        self.view.setText("Choose a video to begin")
        self.view.set_accent(ACCENT)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setMinimumSize(440, 320)
        self.view.roiChanged.connect(self._on_roi_drawn)
        self.view.poseChanged.connect(self._on_pose_dragged)
        self.view.poseCommitted.connect(self._on_pose_committed)
        v.addWidget(self.view, 1)

        bar = QWidget(); h = QHBoxLayout(bar); h.setContentsMargins(2, 0, 2, 0); h.setSpacing(10)
        self.btn_play = QPushButton("▶"); self.btn_play.setFixedWidth(38)
        self.btn_play.setToolTip("Play / pause  (Space)")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        h.addWidget(self.btn_play)
        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.cmb_speed.setCurrentIndex(2)
        self.cmb_speed.setFixedWidth(76)
        h.addWidget(self.cmb_speed)
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_scrub)
        self.lbl_t = QLabel("—"); self.lbl_t.setObjectName("Readout"); self.lbl_t.setMinimumWidth(170)
        self.lbl_t.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self.slider, 1); h.addWidget(self.lbl_t)
        v.addWidget(bar)

        bar2 = QWidget(); h2 = QHBoxLayout(bar2); h2.setContentsMargins(2, 0, 2, 0); h2.setSpacing(14)
        self.chk_mask = QCheckBox("mask"); self.chk_mask.setChecked(True)
        self.chk_mask.stateChanged.connect(self._on_param)
        self.chk_fit = QCheckBox("outline"); self.chk_fit.setChecked(True)
        self.chk_fit.stateChanged.connect(self._on_param)
        self.lbl_status = QLabel(""); self.lbl_status.setObjectName("Readout")
        h2.addWidget(self.chk_mask); h2.addWidget(self.chk_fit)
        h2.addStretch(1); h2.addWidget(self.lbl_status)
        v.addWidget(bar2)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMinimumHeight(110); self.log.setMaximumHeight(190)
        v.addWidget(self.log)
        return w

    # ---- config assembly -------------------------------------------------

    def _seg_cfg(self) -> SegmentConfig:
        return SegmentConfig(
            mode=["auto", "color", "luma"][self.cmb_keying.currentIndex()],
            color_frac=self.spin_cfrac.value(),
            envelope_factor=self.spin_env.value(),
            n_background_frames=self.spin_bg.value(),
            open_px=self.spin_open.value() | 1,
            close_px=self.spin_close.value() | 1,
            min_area_frac=self.spin_minarea.value() / 100.0,
            gap_factor=self.spin_gap.value(),
            manual_threshold=None if self.cmb_thr.currentIndex() == 0
            else float(self.spin_thr.value()),
            roi=tuple(self._active_roi() or ()) or None,
        )

    def _fit_cfg(self) -> FitConfig:
        return FitConfig(
            tau_px=self.spin_tau.value(),
            tau_final_px=min(self.spin_tauf.value(), self.spin_tau.value() - 0.1),
            n_restarts=self.spin_restarts.value(),
            early_stop=self.chk_earlystop.isChecked(),
            coverage_weight=self.spin_cover.value(),
            feature_weight=(self.spin_featw.value()
                            if self.chk_features.isChecked() else 0.0),
            scale_prior_weight=self.spin_prior.value(),
            max_scale_change=self.spin_maxscale.value(),
            width_weight=self.spin_widthw.value(),
            width_tol=max(self.spin_widthtol.value(), 0.1) / 100.0,
            length_overshoot_px=self.spin_lenover.value(),
        )

    def _ana_cfg(self) -> AnalysisConfig:
        return AnalysisConfig(smooth_ms=self.spin_smooth.value(),
                              min_confidence=self.spin_conf.value(),
                              max_gap_ms=self.spin_gapms.value())

    def _scale(self) -> float:
        return [1.0, 0.5, 0.25][self.cmb_scale.currentIndex()]

    # ---- persisted state -------------------------------------------------
    #
    # One table maps every persisted key to its widget and the pair of accessors
    # for it. Keeping it in one place is what stops the three things that must
    # agree -- defaults, save, restore -- from drifting apart as controls are
    # added, which is the usual way a "remembered" setting quietly stops being
    # remembered.

    def _bindings(self) -> dict:
        def spin(w):
            return (w.value, w.setValue)

        def check(w):
            return (w.isChecked, w.setChecked)

        def combo(w):
            return (w.currentIndex, w.setCurrentIndex)

        return {
            "threshold_mode": combo(self.cmb_thr),
            "threshold": spin(self.spin_thr),
            "despeckle_px": spin(self.spin_open),
            "fill_holes_px": spin(self.spin_close),
            "min_area_pct": spin(self.spin_minarea),
            "gap_factor": spin(self.spin_gap),
            "envelope_factor": spin(self.spin_env),
            "seg_mode_index": combo(self.cmb_keying),
            "color_frac": spin(self.spin_cfrac),
            "dxf_scale": spin(self.spin_dxfscale),
            "known_width_mm": spin(self.spin_width),
            "use_features": check(self.chk_features),
            "early_stop": check(self.chk_earlystop),
            "force_method_index": combo(self.cmb_force),
            "beam_E_kpa": spin(self.spin_E),
            "beam_thickness_mm": spin(self.spin_t),
            "beam_width_mm": spin(self.spin_bw),
            "beam_slot_mm": spin(self.spin_slot),
            "beam_leg_to_leg_mm": spin(self.spin_Lleg2leg),
            "beam_muscle_offset_mm": spin(self.spin_arm),
            "beam_leg_long_mm": spin(self.spin_leg_long),
            "beam_leg_short_mm": spin(self.spin_leg_short),
            "beam_resting_index": combo(self.cmb_rest),
            "feature_weight": spin(self.spin_featw),
            "view_mode_index": combo(self.cmb_view),
            "background_frames": spin(self.spin_bg),
            "tau_px": spin(self.spin_tau),
            "tau_final_px": spin(self.spin_tauf),
            "restarts": spin(self.spin_restarts),
            "coverage_weight": spin(self.spin_cover),
            "scale_prior_weight": spin(self.spin_prior),
            "max_scale_change": spin(self.spin_maxscale),
            "width_weight": spin(self.spin_widthw),
            "width_tol_pct": spin(self.spin_widthtol),
            "length_overshoot_px": spin(self.spin_lenover),
            "smooth_ms": spin(self.spin_smooth),
            "min_confidence": spin(self.spin_conf),
            "max_gap_ms": spin(self.spin_gapms),
            "px_per_mm": spin(self.spin_ppm),
            "decode_scale_index": combo(self.cmb_scale),
            "write_overlay": check(self.chk_overlay),
            "use_gpu": check(self.chk_gpu),
            "show_mask": check(self.chk_mask),
            "show_outline": check(self.chk_fit),
            "manual_placement": check(self.chk_manual),
            "roi_enabled": check(self.chk_roi),
            "check_updates_on_start": check(self.chk_update_start),
        }

    def _collect_state(self) -> dict:
        st = dict(self.state)
        for key, (getter, _) in self._bindings().items():
            st[key] = getter()
        st["video_path"] = self.video or ""
        st["dxf_path"] = self.dxf or ""
        st["output_dir"] = self.outdir or ""
        st["roi"] = list(self.roi) if self.roi else None
        st["ui_mode"] = self.ui_mode
        st["collapsed_sections"] = dict(self.collapsed_sections)
        st["traj_hz"] = float(self.plots.spin_traj.value())
        st["manual_pose"] = list(self.manual_pose) if self.manual_pose else None
        st["force_lut_path"] = self.lut_path or ""
        try:
            st["window_geometry"] = bytes(self.saveGeometry().toBase64()).decode("ascii")
        except Exception:
            pass
        return st

    def _apply_state(self, st: dict, paths: bool = False) -> None:
        """Push a state dict into the widgets.

        ``_loading_state`` suppresses the change handlers throughout: without it,
        restoring twenty controls queues twenty preview renders and twenty
        settings writes, and the last one wins by accident rather than by design.
        """
        self._loading_state = True
        try:
            for key, (_, setter) in self._bindings().items():
                if key in st:
                    try:
                        setter(st[key])
                    except (TypeError, ValueError):
                        pass
            self.state.update(st)
            self.manual_pose = st.get("manual_pose") or None
            lut_path = st.get("force_lut_path", "")
            if lut_path and lut_path != (self.lut_path or ""):
                self._load_lut(lut_path, quiet=True)
            if paths:
                dxf = st.get("dxf_path", "")
                if dxf and Path(dxf).exists():
                    self._load_template(dxf, quiet=True)
                elif dxf:
                    self._log(f"config referenced a DXF that is not there: {dxf}")
                self.outdir = st.get("output_dir") or None
        finally:
            self._loading_state = False
        self.spin_thr.setEnabled(self.cmb_thr.currentIndex() == 1)
        self._sync_placement_ui()

    def _wire_persistence(self) -> None:
        """Every control writes settings when it changes. Connected after the
        initial restore so the restore itself does not trigger a save."""
        widgets = [self.chk_earlystop, self.cmb_force, self.spin_E, self.spin_t, self.spin_bw,
                   self.spin_Lleg2leg, self.spin_arm, self.spin_leg_long,
                   self.spin_leg_short, self.cmb_rest,
                   self.spin_dxfscale, self.spin_width, self.cmb_view,
                   self.chk_features, self.spin_featw,
                   self.cmb_keying, self.spin_cfrac, self.spin_env,
                   self.cmb_thr, self.spin_thr, self.spin_open, self.spin_close,
                   self.spin_minarea, self.spin_gap, self.spin_bg, self.spin_tau,
                   self.spin_tauf, self.spin_restarts, self.spin_cover,
                   self.spin_prior, self.spin_maxscale,
                   self.spin_widthw, self.spin_widthtol, self.spin_lenover,
                   self.spin_smooth,
                   self.spin_conf, self.spin_gapms, self.spin_ppm,
                   self.cmb_scale, self.chk_overlay, self.chk_gpu,
                   self.chk_mask, self.chk_fit, self.chk_manual,
                   self.chk_update_start]
        for w in widgets:
            for signal_name in ("valueChanged", "currentIndexChanged", "stateChanged"):
                sig = getattr(w, signal_name, None)
                if sig is not None:
                    sig.connect(self._touch)
                    break

    def _touch(self, *_):
        if not self._loading_state:
            self._persist_timer.start()

    def _persist(self):
        self.state = self._collect_state()
        S.save_settings(self.state)

    # Every QThread this window can have in flight. Qt calls abort() on the
    # process if one of these is still running when it is destroyed, so they all
    # have to be accounted for on the way out -- not just the ones that are
    # obviously long-lived.
    _WORKER_ATTRS = ("_player", "_matcher", "_update_check", "_sizer",
                     "_loader", "_preview", "_runner")

    def _retire(self, attr: str) -> None:
        """Let go of a worker without destroying it if it is still running.

        Qt aborts the process when a running QThread is destroyed -- not an
        exception, not a dialog, ``SIGABRT`` and no traceback. Python decides to
        destroy one the moment the last reference goes, and every worker here is
        held by exactly one attribute, so ``self._loader = LoadWorker(...)``
        while a load was in flight was a hard crash. Reproduced in four lines:
        start a QThread that sleeps, rebind the attribute, collect, abort.

        That is what "it crashes while building the video background" was. The
        loader is the slowest worker and the only one whose message is on screen
        the whole time it runs, so any second load request during one -- picking
        another clip, pressing Rebuild background, stepping through the queue --
        landed on it.

        The old worker is therefore parked in a list until it finishes on its
        own, and its signals are cut first so a stale result cannot arrive after
        something newer has replaced it.
        """
        w = getattr(self, attr, None)
        if not isinstance(w, QThread):
            return
        for name in ("done", "note", "progress", "row", "frame", "aborted",
                     "finished_at"):
            sig = getattr(w, name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass            # nothing was connected; fine either way
        if w.isRunning():
            self._retired.append(w)
            w.finished.connect(lambda w=w: self._retired.remove(w)
                               if w in self._retired else None)
        setattr(self, attr, None)

    def _shutdown_workers(self, ms: int = 2500):
        """Stop every worker thread before Qt tears the window down.

        This fixes a real crash on exit, and the path to it is ordinary: the app
        reopens your last video at launch, which starts a loader thread that
        probes decode backends by running ffmpeg. Close the window during those
        few seconds -- which is exactly what you do if you opened it by mistake,
        or realise you want a different folder -- and Qt destroys a running
        QThread and aborts. The process dies with SIGABRT after the window has
        already gone, so it reads as a crash with no cause.

        Each worker is asked nicely first, waited for, and only then terminated.
        Terminating mid-ffmpeg is not clean, but the alternative here is not a
        clean shutdown -- it is a crash.
        """
        workers = [getattr(self, n, None) for n in self._WORKER_ATTRS]
        workers += list(getattr(self, "_retired", []))
        for t in workers:
            if not isinstance(t, QThread) or not t.isRunning():
                continue
            for stopper in ("abort", "stop"):
                fn = getattr(t, stopper, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
            t.requestInterruption()
            t.quit()
            if not t.wait(ms):
                t.terminate()
                t.wait(500)

    def closeEvent(self, ev):
        self._closing = True
        self._stop_play()
        if getattr(self, "_pulse_timer", None) is not None:
            self._pulse_timer.stop()
        self._debounce.stop()
        self._shutdown_workers()
        self._persist()
        super().closeEvent(ev)

    # ---- config files ----------------------------------------------------

    def _save_config(self):
        start = self.state.get("last_config_dir") or str(Path.home())
        f, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", str(Path(start) / f"robotrack{S.CONFIG_SUFFIX}"),
            S.CONFIG_FILTER)
        if not f:
            return
        try:
            self.state = self._collect_state()
            self.state["last_config_dir"] = str(Path(f).parent)
            p = S.write_config(f, self.state)
            S.save_settings(self.state)
            self._log(f"saved configuration to {p.name}")
        except OSError as exc:
            QMessageBox.warning(self, "Save configuration", str(exc))

    def _load_config(self):
        start = self.state.get("last_config_dir") or str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, "Load configuration", start,
                                           S.CONFIG_FILTER)
        if not f:
            return
        try:
            st = S.read_config(f)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load configuration",
                                f"Could not read that configuration:\n\n{exc}")
            return
        st["last_config_dir"] = str(Path(f).parent)
        # Paths come along: a config is a description of a run, and half a run is
        # not reproducible. The video is offered rather than opened, because
        # rebuilding the background plate is slow and may not be what was wanted.
        self._apply_state(st, paths=True)
        origin = S.config_origin(f)
        self._log(f"loaded {Path(f).name}" + (f" — {origin}" if origin else ""))
        self._persist()

        vid = st.get("video_path", "")
        if vid and vid != (self.video or "") and Path(vid).exists():
            if QMessageBox.question(
                    self, "Load configuration",
                    f"Also open the video this configuration was saved with?\n\n"
                    f"{Path(vid).name}") == QMessageBox.Yes:
                self.video = vid
                self.lbl_video.setText(Path(vid).name)
                self._reload()
                return
        if self.reader is not None:
            self._render_preview()

    def _reset_config(self):
        if QMessageBox.question(
                self, "Reset", "Return every setting to its default?\n\n"
                "The current video and DXF stay open.") != QMessageBox.Yes:
            return
        st = dict(S.DEFAULTS)
        st["video_path"] = self.video or ""
        st["dxf_path"] = self.dxf or ""
        st["update_channel"] = self.state.get("update_channel", "")
        self.manual_pose = None
        self._apply_state(st)
        self._persist()
        self._log("settings reset to defaults")
        if self.reader is not None:
            self._render_preview()

    # ---- drawing scale and force LUT -------------------------------------

    def _on_dxf_scale(self, *_):
        """Re-read the drawing at the new scale.

        The scale is not cosmetic: the robot's width is the calibration ruler,
        so a drawing at 5:1 detail scale makes every micrometer in the output
        five times too large. Reloading immediately means the millimeter readout
        below confirms the number against what you measured on the bench.
        """
        self._touch()
        if self.dxf and not self._loading_state:
            self._load_template(self.dxf, loop_index=self.state.get("dxf_loop_index", 0))
        self._update_dims()

    def _true_width_mm(self) -> float | None:
        """The width the ruler uses: typed if given, else from the drawing."""
        v = self.spin_width.value()
        if v > 0:
            return float(v)
        return self.template.width_mm if self.template is not None else None

    def _on_features_toggled(self, *_):
        """Interior features change the template itself, so reload the drawing."""
        self.spin_featw.setEnabled(self.chk_features.isChecked())
        self._touch()
        if self.dxf and not self._loading_state:
            self._load_template(self.dxf, loop_index=self.state.get("dxf_loop_index", 0))

    def _scale_from_true_width(self):
        """Derive the drawing scale from the width you measured.

        A drawing at 5:1 and one in centimetres look identical from inside the
        file, so the scale cannot be inferred from the DXF alone. One measured
        width settles it, and dividing is more reliable than guessing round
        numbers -- Legs.DXF comes out at exactly 0.2000.
        """
        w_true = self.spin_width.value()
        if w_true <= 0:
            QMessageBox.information(
                self, "Drawing scale",
                "Enter the robot's measured width in True width first, then press "
                "this to work out the drawing scale from it.")
            return
        if self.template is None:
            QMessageBox.information(self, "Drawing scale", "Load a DXF first.")
            return
        drawn = self.template.width_mm / max(self.template.scale, 1e-9)
        if drawn <= 0:
            return
        k = w_true / drawn
        self._log(f"drawing scale {k:.4f} = {w_true:.3f} mm measured ÷ "
                  f"{drawn:.3f} mm drawn")
        self.spin_dxfscale.setValue(k)

    def _on_width_changed(self, *_):
        self._touch()
        self._update_dims()

    def _update_dims(self):
        """State the scale, or state plainly that there isn't one.

        A run with no DXF and no typed width produces pixels, and the only clue
        used to be an axis label. Saying so next to the controls that fix it is
        the difference between a puzzling plot and a two-second correction.
        """
        w_true = self._true_width_mm()
        if self.template is not None:
            t = self.template
            msg = f"outline is {t.width_mm:.3f} × {t.length_mm:.3f} mm"
            if self.spin_width.value() > 0:
                msg += f"; ruler overridden to {w_true:.3f} mm"
        elif w_true:
            msg = f"markerless, ruler = {w_true:.3f} mm wide"
        else:
            self.lbl_dims.setText("no scale — results will be in pixels. "
                                  "Load a DXF or type the true width.")
            return

        w_px = self._measured_width_px()
        if w_px and w_true:
            msg += f"   →  {1000.0 * w_true / w_px:.2f} µm/px"
        self.lbl_dims.setText(msg)

    def _measured_width_px(self) -> float | None:
        """The robot's width in full-resolution pixels, from the last preview."""
        if not self._last_fit_pose or self.template is None:
            return None
        w = self._last_fit_pose[3] * self.template.width_mm / max(self._preview_scale(), 1e-9)
        return w if w > 0 else None

    def _beam_model(self) -> BeamForceModel:
        return BeamForceModel(
            E_pa=self.spin_E.value() * 1000.0,          # kPa in the UI, Pa in the model
            thickness_mm=self.spin_t.value(),
            beam_width_mm=self.spin_bw.value(),
            slot_width_mm=self.spin_slot.value(),
            L_mm=self.spin_Lleg2leg.value(),
            l_mm=self.spin_arm.value(),
            leg_long_mm=self.spin_leg_long.value(),
            leg_short_mm=self.spin_leg_short.value(),
            resting="median" if self.cmb_rest.currentIndex() == 1 else "max",
        )

    def force_method(self) -> str:
        return ["lut", "beam"][self.cmb_force.currentIndex()]

    def _on_force_method(self, *_):
        m = self.force_method()
        self.row_lut.setVisible(m == "lut")
        self.lbl_lut.setVisible(m == "lut")
        for w in self.beam_rows:
            w.setVisible(m == "beam")
        self._on_beam_changed()
        self._touch()
        has = (m == "lut" and self.lut is not None) or (m == "beam")
        self.plots.configure(self.plots.um_per_px, has)

    def _on_beam_changed(self, *_):
        """Keep the derived constants visible as the inputs are typed.

        Force scales linearly with E and with I, and I with the cube of
        thickness — a 10% error in a thickness measured with callipers is a 33%
        error in force. Showing the derived numbers makes that visible instead of
        leaving it inside the arithmetic.
        """
        if self.force_method() != "beam":
            self.lbl_beam.setText("")
            return
        b = self._beam_model()
        self.lbl_beam.setText(
            f"I = {b.I_mm4:.5f} mm⁴   ·   leg {b.L_leg_mm:.3f} mm   ·   "
            f"{b.stiffness:.0f} µN per radian\n"
            f"100 µm pull-in → {b.stiffness * np.arcsin(0.05 / b.L_leg_mm):.0f} µN")
        self._touch()

    def _pick_lut(self):
        start = self.state.get("last_lut_dir") or ""
        f, _ = QFileDialog.getOpenFileName(
            self, "Choose a force calibration CSV", start,
            "CSV (*.csv *.txt *.tsv);;All files (*)")
        if f:
            self.state["last_lut_dir"] = str(Path(f).parent)
            self._load_lut(f)

    def _load_lut(self, path: str, quiet: bool = False) -> bool:
        try:
            lut = load_lut(path)
        except LUTError as exc:
            if quiet:
                self._log(f"force LUT: {exc}")
            else:
                QMessageBox.warning(self, "Force LUT", str(exc))
            return False
        self.lut, self.lut_path = lut, path
        self.lbl_lut.setText(lut.summary())
        if self.cmb_force.currentIndex() != 0:
            self.cmb_force.setCurrentIndex(0)      # choosing a curve means using it
        self._log(f"force LUT: {lut.summary()}")
        if not lut.units_explicit:
            self._log("  units were not stated in the header — mm and mN assumed")
        self.plots.configure(self.plots.um_per_px, True)
        self._touch()
        return True

    def _clear_lut(self):
        self.lut, self.lut_path = None, None
        self.lbl_lut.setText("no simulated curve loaded")
        self.plots.configure(self.plots.um_per_px, False)
        self._touch()

    def _um_per_px(self) -> float | None:
        """Provisional scale for the live panel, from the last preview fit."""
        w_px, w_mm = self._measured_width_px(), self._true_width_mm()
        if not w_px or not w_mm:
            return None
        return 1000.0 * w_mm / w_px

    # ---- manual placement ------------------------------------------------

    def _preview_scale(self) -> float:
        """Ratio between decoded preview pixels and full-resolution pixels."""
        if self.reader is None or not self.info or not self.info.width:
            return 1.0
        return self.reader.width / float(self.info.width)

    def _pose_to_preview(self, pose):
        if pose is None:
            return None
        s = self._preview_scale()
        p = list(pose)
        return [p[0] * s, p[1] * s, p[2], p[3] * s, p[4] * s]

    def _pose_to_full(self, pose):
        if pose is None:
            return None
        s = self._preview_scale() or 1.0
        p = list(pose)
        return [p[0] / s, p[1] / s, p[2], p[3] / s, p[4] / s]

    def _sync_placement_ui(self):
        """Keep the placement card, the overlay and the enabled states in step."""
        has_tpl = self.template is not None
        editable = has_tpl and self.chk_manual.isChecked()
        self.chk_manual.setEnabled(has_tpl)
        self.btn_snap.setEnabled(editable and self._last_fit_pose is not None)
        self.btn_place_clear.setEnabled(editable and self.manual_pose is not None)

        if not has_tpl:
            self.lbl_place.setText("Load a DXF to place it.")
            self.lbl_pose.setText("—")
            self.view.set_template(None)
            self.view.set_pose(None, editable=False)
            return

        self.view.set_template(self.template.points,
                               self.template.width_mm, self.template.length_mm)
        if editable:
            self.lbl_place.setText(
                "Drag the outline onto the robot. The round handle sets the long "
                "axis and rotation, the square handle sets width, the wheel scales "
                "both. This is a starting guess — the fit is free to move from it.")
        else:
            self.lbl_place.setText(
                "The starting pose is found automatically from the mask. Turn this "
                "on when something else in frame is being picked up instead.")

        if editable and self.manual_pose is None and self.reader is not None:
            self.manual_pose = self._pose_to_full(
                self._last_fit_pose or self.view.default_pose())
        self.view.set_pose(self._pose_to_preview(self.manual_pose) if editable
                           else None, editable=editable)
        self._update_pose_readout()

    def _update_pose_readout(self):
        p = self.manual_pose
        if not p or self.template is None:
            self.lbl_pose.setText("—")
            return
        w = p[3] * self.template.width_mm
        h = p[4] * self.template.length_mm
        self.lbl_pose.setText(
            f"x {p[0]:.0f}  y {p[1]:.0f}  θ {np.degrees(p[2]):+.1f}°   "
            f"W {w:.0f}px  L {h:.0f}px")

    def _on_manual_toggled(self, *_):
        self._sync_placement_ui()
        if not self._loading_state:
            self._touch()
            if self.reader is not None:
                self._on_param()

    def _on_pose_dragged(self, pose):
        # Live: update only the cheap readout. Re-fitting on every mouse move
        # would queue a decode and an optimization per pixel of travel.
        self.manual_pose = self._pose_to_full(pose)
        self._update_pose_readout()
        self.btn_place_clear.setEnabled(True)

    def _on_pose_committed(self, pose):
        self.manual_pose = self._pose_to_full(pose)
        self._update_pose_readout()
        self._touch()
        self._on_param()        # now re-fit, seeded from where it was dropped

    def _snap_to_fit(self):
        if self._last_fit_pose is None:
            return
        self.manual_pose = self._pose_to_full(self._last_fit_pose)
        self.view.set_pose(self._pose_to_preview(self.manual_pose), editable=True)
        self._update_pose_readout()
        self._touch()
        self._log("placement taken from the automatic fit on this frame")

    def _clear_placement(self):
        self.manual_pose = None
        self._sync_placement_ui()
        self._touch()
        self._on_param()

    def _manual_seed(self):
        """The seed handed to the fitter, in preview pixels, or None."""
        if not (self.template is not None and self.chk_manual.isChecked()
                and self.manual_pose):
            return None
        return np.array(self._pose_to_preview(self.manual_pose), np.float32)

    def _on_region(self, st: dict):
        """Report what the selected stretch measured.

        Only when it actually changed. The panel re-measures on every redraw --
        a theme switch, a resize, a zoom -- and each one logged the identical
        four lines, so a session's log was mostly the same selection repeated a
        dozen times with the real events buried between them.
        """
        if not st:
            return
        if st.get("note"):
            self._log(f"selection {st['t0']:.2f}–{st['t1']:.2f} s: {st['note']}")
            return
        u = st.get("units", "px")
        bits = [f"selection {st['t0']:.2f}–{st['t1']:.2f} s "
                f"({st['duration_s']:.2f} s, {st['n_frames']} frames)"]
        if np.isfinite(st.get("length_delta", np.nan)):
            bits.append(f"  Δlength {st['length_delta']:.1f} {u} average over "
                        f"{st['length_cycles']} cycles")
        if np.isfinite(st.get("force_delta_un", np.nan)):
            bits.append(f"  Δforce  {st['force_delta_un']:.1f} µN average over "
                        f"{st.get('force_cycles', 0)} cycles")
        if np.isfinite(st.get("speed_per_min", np.nan)):
            u2 = st.get("speed_units", "mm/min")
            bits.append(f"  speed   {st['speed_per_min']:.3f} {u2} along the path, "
                        f"{st.get('net_speed_per_min', float('nan')):.3f} {u2} net")
            r = st.get("wander_ratio", float("nan"))
            if np.isfinite(r) and r > 2.0:
                bits.append(f"          path is {r:.1f}x the net rate — the centroid "
                            f"is wandering more than the robot is traveling")
        text = "\n".join(bits)
        if text != getattr(self, "_last_region_log", None):
            self._last_region_log = text
            self._log(text)

    def _on_run_frame(self, index: int, img):
        """Show the frame the analysis is on, with its outline drawn."""
        if img is None:
            return
        h, w = img.shape[:2]
        qi = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self.view.set_frame(QPixmap.fromImage(qi), (w, h))
        if self.info:
            self.slider.blockSignals(True)
            self.slider.setValue(min(index, self.slider.maximum()))
            self.slider.blockSignals(False)
            self.lbl_t.setText(f"frame {index} / {self.info.n_frames}   "
                               f"{index / max(self.info.measured_fps, 1e-6):7.3f} s")

    # ---- playback --------------------------------------------------------

    def _speed(self) -> float:
        return [0.25, 0.5, 1.0, 2.0, 4.0][self.cmb_speed.currentIndex()]

    def _toggle_play(self):
        if self._player is not None and self._player.isRunning():
            self._stop_play()
            return
        if self.reader is None or self.info is None:
            return
        # Placement handles would be dragged against a moving picture, and the
        # preview fit is meaningless mid-playback, so both stand down.
        self.view.set_pose(None, editable=False)
        self._debounce.stop()
        self.btn_play.setText("❚❚")
        self._retire("_player")
        self._player = PlaybackWorker(
            self.reader, self.model, self.background, self._seg_cfg(),
            self.slider.value(), self.chk_mask.isChecked(),
            self.chk_gpu.isChecked(), self.info.measured_fps * self._speed(),
            view=self._view_mode())
        self._player.frame.connect(self._on_play_frame)
        self._player.finished_at.connect(self._on_play_end)
        self._player.start()

    def _stop_play(self):
        if self._player is not None:
            self._player.stop()
            self._player.wait(2000)

    def _on_play_frame(self, i, t, img, status):
        h, w = img.shape[:2]
        qi = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self.view.set_frame(QPixmap.fromImage(qi), (w, h))
        self.lbl_status.setText(status)
        self.slider.blockSignals(True)      # do not queue a preview per frame
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.lbl_t.setText(f"frame {i} / {self.info.n_frames}   {t:7.3f} s")

    def _on_play_end(self, last):
        self.btn_play.setText("▶")
        self._player = None
        self.slider.setValue(last)
        self._sync_placement_ui()
        self._on_param()                     # bring the fit back on this frame

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Space and self.btn_play.isEnabled():
            self._toggle_play()
            ev.accept()
            return
        super().keyPressEvent(ev)

    # ---- automatic outline matching --------------------------------------

    def _automatch(self):
        if not self.dxf or self.reader is None:
            QMessageBox.information(self, "Match drawing",
                                    "Open a video and a DXF first.")
            return
        self._stop_play()
        self.btn_automatch.setEnabled(False)
        self.btn_automatch.setText("Matching…")
        self._retire("_matcher")
        self._matcher = MatchWorker(self.dxf, self.reader, self._seg_cfg(),
                                    self._fit_cfg(), self.chk_gpu.isChecked())
        self._matcher.note.connect(self._log)
        self._matcher.done.connect(self._matched)
        self._matcher.start()

    def _matched(self, res, err):
        self.btn_automatch.setEnabled(True)
        self.btn_automatch.setText("Match drawing to video")
        if err or res is None or res.best is None:
            QMessageBox.warning(self, "Match drawing",
                                err or "No outline could be matched to this clip.")
            return
        self._log(res.summary())
        best = res.best
        if best.confidence < 0.35:
            QMessageBox.warning(
                self, "Match drawing",
                f"The best candidate only reached confidence {best.confidence:.2f}.\n\n"
                "That usually means the segmentation is wrong rather than the outline — "
                "check the mask on a clear frame first.")
        self._load_template(self.dxf, loop_index=best.index)
        if best.pose:
            self.manual_pose = self._pose_to_full(best.pose)
            self.chk_manual.setChecked(True)
            self._sync_placement_ui()
            self.view.set_pose(self._pose_to_preview(self.manual_pose), editable=True)
            self._update_pose_readout()
        self._touch()
        self._on_param()

    # ---- updates ---------------------------------------------------------

    def _check_updates(self, quiet: bool = False):
        self._stop_update_pulse()
        channel = self.state.get("update_channel", "")
        if quiet and not channel:
            return
        dlg = UpdateDialog(channel, self,
                           on_channel_change=self._set_update_channel)
        if quiet:
            # Launched by the "check at launch" preference: look first, and only
            # interrupt if there is actually something to install.
            dlg.check()
        dlg.exec()
        # Read back rather than listened for. A signal emitted from a dialog
        # that is in the middle of closing has to survive too much to be
        # trusted with the one step the update depends on, and the symptom when
        # it did not survive was total silence.
        if getattr(dlg, "installed_version", ""):
            self._relaunch()

    def _current_time(self) -> float:
        info = self.info
        if info is None:
            return 0.0
        i = int(self.slider.value())
        ts = getattr(info, "timestamps", None)
        if ts is not None and getattr(ts, "size", 0) > i:
            return float(ts[i] - ts[0])
        return i / max(getattr(info, "measured_fps", 30.0), 1e-6)

    def _last_pose_obj(self):
        """The preview fit on the current frame, as a Pose, or None.

        Supplying it is what turns the appearance lock from a relative
        measurement into an absolute one -- without it the tracker knows how
        much the robot changed but not how big it is, and the run refuses to
        calibrate rather than inventing a scale.
        """
        p = getattr(self, "_last_fit_pose", None)
        if not p or self.template is None:
            return None
        try:
            from .register import Pose
            return Pose(float(p[0]), float(p[1]), float(p[2]),
                        float(p[3]), float(p[4]), cost=0.0, confidence=1.0)
        except Exception:
            return None

    # ---- region of interest ----------------------------------------------

    def _active_roi(self):
        """The region the analysis should actually use, or None.

        The checkbox governs *use*, not merely drawing. It used to control only
        whether the region could be edited, so unticking it left the region
        silently applied to every run -- the one state where the control said
        one thing and the program did another. The drawn shape is kept while it
        is off, so unticking is a way to compare with and without rather than a
        way to throw away a careful placement.
        """
        if not getattr(self, "chk_roi", None) or not self.chk_roi.isChecked():
            return None
        return list(self.roi) if self.roi else None

    def _on_roi_toggle(self, *_):
        on = self.chk_roi.isChecked()
        # Hidden entirely when off, rather than shown-but-frozen. A region that
        # still dimmed the frame while doing nothing was most of why "off" did
        # not look like off.
        self.view.set_roi(self.roi if on else None, edit=on)
        if on and not self.roi:
            self._log("drag on the video to draw the region to track inside; "
                      "then drag its edges to resize, or the grip above it to rotate.")
        if self.reader is not None:
            # Whether the region applies changes what the color model should be
            # estimated from, exactly as moving it does.
            self._reload_needed()
        self._sync_roi_label()
        self._touch()
        self._on_param()

    def _on_roi_drawn(self, roi):
        was = list(self.roi) if self.roi else None
        self.roi = list(roi) if roi else None
        self._sync_roi_label()
        self._touch()
        # The region is not just a clip: the medium's color and the robot's are
        # estimated from inside it. That estimate is made once, when the clip is
        # opened, so after moving the region the preview kept segmenting with
        # the *old* region's color model and looked unchanged no matter what you
        # did -- the region appeared to be stuck on whichever one you drew
        # first. The analysis itself always re-estimates, so this only ever
        # affected the picture; it still made the control feel broken.
        if was != self.roi and self.reader is not None:
            self._reload_needed()
            self.btn_rebuild.setText("Re-estimate colors in this region")
            self._log("region changed — press \u201cRe-estimate colors in this "
                      "region\u201d to update the preview's color model. "
                      "A run always re-estimates.")
        self._render_preview()

    def _clear_roi(self):
        self.roi = None
        self.chk_roi.setChecked(False)
        self.view.set_roi(None, edit=False)
        self._sync_roi_label()
        self._touch()
        self._render_preview()

    def _sync_roi_label(self):
        on = bool(getattr(self, "chk_roi", None) and self.chk_roi.isChecked())
        if not self.roi:
            self.lbl_roi.setText(
                "Whole frame. Use a region when something outside the dish is "
                "brighter or more saturated than the robot — the color model is "
                "estimated inside the region, not just clipped to it.")
            return
        if not on:
            x, y, w, h = self.roi[:4]
            self.lbl_roi.setText(
                f"Whole frame. A {w} × {h} px region is saved but not in use — "
                "tick the box above to apply it again.")
            return
        x, y, w, h = self.roi[:4]
        ang = float(self.roi[4]) if len(self.roi) > 4 else 0.0
        frac = ""
        if self.info is not None and self.info.width and self.info.height:
            frac = f" — {100.0 * w * h / (self.info.width * self.info.height):.0f}% of the frame"
        turned = f", turned {ang:+.1f}°" if abs(ang) > 0.05 else ""
        self.lbl_roi.setText(
            f"Tracking inside {w} × {h} px at ({x}, {y}){turned}{frac}. "
            "Drag an edge or corner to resize, the round grip to rotate "
            "(hold Shift for 15° steps), or inside it to move.")

    # ---- what belongs to the clip, and what belongs to the experiment ------

    def _forget_clip_state(self):
        """Drop everything that describes the previous *clip*.

        Thresholds, the drawing, the force model and the output folder describe
        the experiment and carry over. The fit, the hand placement, the plots
        and the region describe one recording: a region is a box in that
        recording's own pixels, and carrying it to the next clip masks a
        different scene -- silently, because a plausible region produces a
        plausible-looking mask somewhere else entirely.
        """
        self._last_fit_pose = None
        self.manual_pose = None
        self._result = None
        self.btn_export_sel.setEnabled(False)
        if hasattr(self, "chk_manual"):
            self.chk_manual.setChecked(False)
        if self.roi:
            self._log("region cleared — it was drawn on the previous clip")
        self.roi = None
        if hasattr(self, "chk_roi"):
            self.chk_roi.setChecked(False)
        self.view.set_roi(None, edit=False)
        self._sync_roi_label()
        self.plots.reset()

    # ---- appearance ------------------------------------------------------

    def _sync_theme_button(self):
        light = self.state.get("theme_mode", "dark") == "light"
        self.btn_theme.setToolTip("Switch to dark mode" if light else
                                  "Switch to light mode")
        if hasattr(self, "btn_update"):
            self._sync_header_icons()

    def _toggle_theme(self):
        mode = "light" if self.state.get("theme_mode", "dark") == "dark" else "dark"
        self.state["theme_mode"] = mode
        self._apply_theme_mode(mode)
        S.save_settings(self.state)

    def _apply_theme_mode(self, mode: str):
        """Restyle everything that cached a color at construction time.

        A Qt stylesheet reaches every widget, but three things here do not go
        through it: the OpenCV overlays are drawn with BGR tuples read from the
        token table at import, matplotlib bakes rcParams into artists when they
        are created, and the placement view holds its own accent. All three have
        to be told, or a switch leaves dark-theme marks on a light window.
        """
        global OK_BGR, WARN_BGR
        set_theme_mode(mode)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        OK_BGR, WARN_BGR = _bgr(C["ok"]), _bgr(C["warn"])
        # Chips carry a per-widget stylesheet set at runtime, so they need
        # repainting explicitly with whatever state they are currently in.
        dev = get_device(self.chk_gpu.isChecked() if hasattr(self, "chk_gpu") else True)
        style_chip(self.chip_gpu, "ok" if dev.accelerated else "warn")
        style_chip(self.chip_rate, "warn" if (self.info and self.info.is_vfr) else
                   ("ok" if self.info else ""))
        style_chip(self.chip_key, "ok" if (self.model and self.model.mode == "color") else "")
        style_chip(self.chip_mode, "")
        style_chip(self.chip_version, "")
        try:
            self.view.set_accent(ACCENT)
        except Exception:
            pass
        try:
            self.plots.retheme()
        except Exception:
            pass
        self._sync_theme_button()
        # Redraw the frame so its overlay picks up the new state colors.
        if self.reader is not None:
            self._render_preview()

    def _check_updates_quietly(self):
        """Ask the channel whether anything is newer, without blocking or asking.

        Does nothing once the window is on its way out: the answer would have
        nowhere to go, and the thread carrying it would outlive the window.

        Runs on a worker thread: a channel can be a GitHub API call, a network
        share or a cloud-synced folder, and any of those can take seconds or
        hang. Doing it on the GUI thread would freeze the window during launch,
        which is the worst possible moment.
        """
        if getattr(self, "_closing", False):
            return
        spec = self.state.get("update_channel", "")

        def work():
            try:
                return U.check(spec)
            except Exception:
                return None            # an unreachable channel is not an event

        class _Check(QThread):
            found = Signal(object)

            def run(self):
                self.found.emit(work())

        self._update_check = _Check(self)
        self._update_check.found.connect(self._on_update_found)
        self._update_check.start()

    def _on_update_found(self, rel):
        if rel is None or getattr(self, "_closing", False):
            return
        self._pending_release = rel
        self._log(f"update available: {rel.version} — press Update to install it.")
        self.btn_update.setText(f" {rel.version}")
        self.btn_update.setToolTip(
            f"Version {rel.version} is available. Click to review and install it.")
        self._start_update_pulse()

    def _start_update_pulse(self):
        """Fade the Update button between its normal look and the accent.

        A badge would be easy to miss on a window this dense, and a dialog is
        what this feature exists to avoid. Motion in the corner is noticeable
        without being modal, and it stops the moment the button is pressed.
        """
        self._pulse_phase = 0.0
        if getattr(self, "_pulse_timer", None) is None:
            self._pulse_timer = QTimer(self)
            self._pulse_timer.timeout.connect(self._pulse_step)
        self._pulse_timer.start(40)          # 25 fps is plenty for a slow fade

    def _stop_update_pulse(self):
        if getattr(self, "_pulse_timer", None) is not None:
            self._pulse_timer.stop()
        self.btn_update.setStyleSheet("")

    def _pulse_step(self):
        import math
        self._pulse_phase = (self._pulse_phase + 0.045) % 1.0
        # A raised cosine, so it eases at both ends instead of bouncing between
        # two colors -- "breathing" rather than "blinking", which is the
        # difference between noticeable and irritating.
        t = 0.5 - 0.5 * math.cos(2 * math.pi * self._pulse_phase)
        base, hot = QColor(C["line"]), QColor(ACCENT)
        mix = QColor(
            int(base.red() + (hot.red() - base.red()) * t),
            int(base.green() + (hot.green() - base.green()) * t),
            int(base.blue() + (hot.blue() - base.blue()) * t))
        txt = QColor(
            int(QColor(C["text"]).red() + (hot.red() - QColor(C["text"]).red()) * t),
            int(QColor(C["text"]).green() + (hot.green() - QColor(C["text"]).green()) * t),
            int(QColor(C["text"]).blue() + (hot.blue() - QColor(C["text"]).blue()) * t))
        self.btn_update.setStyleSheet(
            f"QPushButton#Ghost {{ border-color: {mix.name()}; color: {txt.name()}; }}")

    def _set_update_channel(self, channel: str):
        self.state["update_channel"] = channel
        S.save_settings(self.state)
        self._log(f"update channel: {U.describe_channel(channel)}")

    def _relaunch(self):
        """Start the updated copy, and only leave once it has proved it started."""
        try:
            self._relaunch_inner()
        except Exception:
            # Nothing in here may fail silently. An exception raised inside a
            # Qt slot is printed to a stderr that a windowed build does not
            # have, which is indistinguishable from the restart simply not
            # happening -- and that is exactly what this whole area kept
            # looking like.
            tb = traceback.format_exc()
            self._log(tb.strip().splitlines()[-1])
            self._offer_restart()
            QMessageBox.warning(
                self, "Restart to finish",
                f"The update is installed and will be used next time you open "
                f"{APP_NAME}.\n\nSomething went wrong while restarting:\n\n{tb}")

    def _relaunch_inner(self):
        """Start the updated copy, and only leave once it has proved it started.

        The previous versions of this launched a child and quit on the strength
        of ``Popen`` having returned. The updater's own state file showed what
        that was worth: patch after patch applied correctly with
        ``{"attempts": 0}`` beside it -- meaning no new process ever ran this
        package's startup code -- while the program reported success and carried
        on as the old version.

        So the child now has to announce itself. It bumps the pending marker at
        the very top of its own launch, and this waits for that before closing
        anything. If it never arrives the working copy stays on screen with a
        standing instruction, which is a far better failure than a self-restart
        that quietly did not happen -- or, worse, one that closed the only
        window and put nothing back.
        """
        self._persist()
        pending = U.read_marker()
        version = pending[0] if pending else ""
        # Logged before the attempt, not after: if the restart takes the window
        # with it, this is the line that survives.
        self._log("restart: " + U.launch_diagnostics())
        try:
            proc = U.relaunch()
        except Exception as exc:
            # The update is on disk and will be used at the next launch. Say
            # that first: the previous wording led with the failure, and the
            # one thing the user needed to know -- that closing and reopening
            # is all that is left to do -- came last.
            QMessageBox.information(
                self, "Restart to finish",
                f"The update is installed and will be used the next time you "
                f"open {APP_NAME}.\n\nIt could not restart itself just now:\n\n"
                f"{exc}\n\nClose the window and open it again.")
            self._log("update installed; restart could not be started "
                      "automatically — close and reopen to use it")
            self._offer_restart()
            return
        how = "via the shell" if proc is None else f"as pid {proc.pid}"
        self._log(f"started the updated copy {how}; waiting for it to report in")
        self._await_relaunch(proc, version, tries=80)

    def _await_relaunch(self, proc, version: str, tries: int):
        """Poll for the child's own startup, then hand over to it."""
        if version and U.child_started(version):
            self._log("the updated copy is running — closing this one")
            self._closing = True
            self._shutdown_workers()
            QApplication.instance().closeAllWindows()
            QApplication.instance().quit()
            # Backstop. quit() only unwinds the loop it is called from, and a
            # stray modal loop would leave this process alive beside the copy
            # that just replaced it -- two windows, one of them stale.
            QTimer.singleShot(2500, lambda: os._exit(0))
            return
        dead = proc is not None and proc.poll() not in (None, 0)
        if tries > 0 and not dead:
            # A breadcrumb every five seconds, so a slow start and a dead one
            # look different in the log.
            if tries % 20 == 0:
                self._log(f"  still waiting for the updated copy "
                          f"({(80 - tries) // 4} s)")
            QTimer.singleShot(250, lambda: self._await_relaunch(proc, version, tries - 1))
            return
        why = (f"it exited with code {proc.returncode}" if dead
               else "it did not finish starting within 20 seconds")
        self._log(f"the updated copy did not start ({why}) — this window is "
                  f"still the old version; close and reopen to use the update")
        self._offer_restart()
        QMessageBox.information(
            self, "Restart to finish",
            f"The update is installed and will be used the next time you open "
            f"{APP_NAME}.\n\nA new copy was started but {why}, so this window "
            f"has been left open rather than closing it and leaving you with "
            f"nothing.\n\nClose it and open it again when you are ready.")

    def _offer_restart(self):
        """Leave a standing, unmissable instruction in the header.

        A dialog can be dismissed without being read, and this one carries the
        only thing the user still has to do.
        """
        self.btn_update.setText(" Restart to finish")
        self.btn_update.setToolTip(
            "The update is installed. Close and reopen the program to use it.")
        self._start_update_pulse()
        return
        # Close the windows before quitting. quit() alone leaves any nested
        # modal loop running, and the process then stays up beside the fresh
        # copy it just started.
        self._closing = True
        self._shutdown_workers()
        QApplication.instance().closeAllWindows()
        QApplication.instance().quit()

    # ---- actions ---------------------------------------------------------

    def _log(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _set_enabled(self, on: bool):
        for x in (self.btn_run, self.slider, self.chk_mask, self.chk_fit,
                  self.btn_play):
            x.setEnabled(on)

    def _on_thr_mode(self, i: int):
        self.spin_thr.setEnabled(i == 1)
        self._on_param()

    def _reload_needed(self, *_):
        # Background frames and decode scale change the plate itself, so the
        # preview cannot be updated without rebuilding it. Make that explicit
        # rather than silently rebuilding on every keystroke.
        if self.reader is not None:
            self.btn_rebuild.setVisible(True)

    def _pick_video(self):
        start = self.state.get("last_video_dir") or ""
        f, _ = QFileDialog.getOpenFileName(
            self, "Choose video", start,
            "Video (*.mov *.MOV *.mp4 *.MP4 *.m4v *.avi *.mkv);;All files (*)")
        if f:
            # The same rule Next video already applied to the hand placement:
            # both belong to the clip being left behind.
            if f != (self.video or ""):
                self._forget_clip_state()
            self.video = f
            self.state["last_video_dir"] = str(Path(f).parent)
            self.lbl_video.setText(Path(f).name)
            self._touch()
            self._refresh_queue()
            self._reload()

    def _pick_dxf(self):
        start = self.state.get("last_dxf_dir") or ""
        f, _ = QFileDialog.getOpenFileName(self, "Choose DXF", start,
                                           "DXF (*.dxf *.DXF);;All files (*)")
        if not f:
            return
        self.state["last_dxf_dir"] = str(Path(f).parent)
        self._load_template(f)

    def _load_template(self, path: str, quiet: bool = False,
                       loop_index: int | None = None) -> bool:
        """Load a DXF and bring the placement UI in line with it.

        A new drawing, or a different outline within one, invalidates any
        existing placement: the scale factors are expressed per template
        millimeter, so carrying them across would silently mean a different size.
        """
        same_file = path == (self.dxf or "")
        if loop_index is None:
            # The remembered choice belongs to the remembered *file*, so compare
            # against the saved path too -- on a fresh launch self.dxf is empty
            # and the restore would otherwise silently fall back to outline 1.
            remembered = path == (self.state.get("dxf_path") or "")
            loop_index = self.state.get("dxf_loop_index", 0) if (same_file or remembered) else 0
        try:
            tpl = load_dxf(path, loop_index=loop_index,
                           scale=self.spin_dxfscale.value(),
                           use_features=self.chk_features.isChecked())
        except Exception as exc:
            if not quiet:
                QMessageBox.warning(self, "DXF", f"Could not load that drawing:\n\n{exc}")
            else:
                self._log(f"could not load {Path(path).name}: {exc}")
            return False
        changed = (not same_file) or tpl.loop_index != self.state.get("dxf_loop_index", 0)
        self.template, self.dxf = tpl, path
        self.state["dxf_loop_index"] = tpl.loop_index
        self.lbl_dxf.setText(tpl.summary())
        self.chip_mode.setText("CAD template")
        style_chip(self.chip_mode, "ok")
        self._log(f"template: {tpl.summary()}")
        if changed:
            self.manual_pose = None
        self._populate_loops(path, tpl.loop_index)
        self._update_dims()
        self._sync_placement_ui()
        self._touch()
        if not quiet:
            self._on_param()
        return True

    def _populate_loops(self, path: str, current: int) -> None:
        """List the drawing's candidate outlines, if there is a choice to make.

        A production drawing is a sheet, not a bare outline: a page border, a
        title block, dimensions and several views. Which closed curve is the
        robot is genuinely ambiguous, and picking silently is how the tracker
        ends up measuring a fillet or the paper.
        """
        try:
            loops, _ = read_loops(path)
        except Exception:
            self.row_loop.setVisible(False)
            return
        # Labeled at the drawing scale, not as drawn. Listing the raw numbers
        # meant looking for "5.25 x 12.60" in a list that said "26.25 x 63.00"
        # and concluding the right outline was not there.
        k = self.spin_dxfscale.value() or 1.0
        self._loading_state = True
        try:
            self.cmb_loop.clear()
            for i, L in enumerate(loops):
                tag = ("  [sheet border]" if L.is_frame
                       else ("" if L.closed else "  [open]"))
                self.cmb_loop.addItem(
                    f"{i + 1}.  {L.width_mm * k:.2f} × {L.height_mm * k:.2f} mm · "
                    f"{len(L.points)} pts{tag}", i)
            self.cmb_loop.setCurrentIndex(min(current, max(len(loops) - 1, 0)))
        finally:
            self._loading_state = False
        self.row_loop.setVisible(len(loops) > 1)
        if len(loops) > 1:
            self._log(f"{len(loops)} candidate outlines in this drawing — "
                      f"using #{current + 1}; change it under Outline if that is wrong")

    def _on_loop_changed(self, idx: int):
        if self._loading_state or idx < 0 or not self.dxf:
            return
        self._load_template(self.dxf, loop_index=idx)

    def _clear_dxf(self):
        self.dxf, self.template = None, None
        self.manual_pose = None
        self.lbl_dxf.setText("none — markerless mode")
        self.row_loop.setVisible(False)
        self.chip_mode.setText("markerless")
        style_chip(self.chip_mode, "")
        self._sync_placement_ui()
        self._touch()
        self._on_param()

    def _reload(self):
        if not self.video:
            return
        self._stop_play()
        self.btn_rebuild.setVisible(False)
        self.btn_rebuild.setText("Rebuild background")
        self._set_enabled(False)
        self.view.set_frame(None, (0, 0))
        self.view.set_pose(None, editable=False)
        self.view.setText("building background model…")
        self._retire("_loader")
        self._loader = LoadWorker(self.video, self._scale(), self._seg_cfg(),
                                  self.chk_gpu.isChecked())
        self._loader.note.connect(self._log)
        self._loader.done.connect(self._loaded)
        self._loader.start()

    def _loaded(self, info, reader, bg, model, err):
        if err:
            self.view.set_frame(None, (0, 0))
            self.view.setText("failed to load")
            QMessageBox.critical(self, "Load failed", err)
            self._log(err.strip().splitlines()[-1])
            return
        self.info, self.reader, self.background, self.model = info, reader, bg, model
        self._frame_cache.clear()
        self._fitter = None
        # Color keying reads BGR; luma reads gray. Decode what is actually used
        # rather than decoding both.
        if model is not None and model.mode == "color":
            self.reader = FrameReader(info, reader.backend, scale=reader.scale, color=True)
        self.chip_key.setText("color" if (model and model.mode == "color") else "luma")
        style_chip(self.chip_key, "ok" if (model and model.mode == "color") else "")
        self.lbl_info.setText(info.summary())
        self._log(info.summary())
        self.chip_rate.setText(f"{info.nominal_fps:g} Hz · {info.n_frames} frames")
        style_chip(self.chip_rate, "warn" if info.is_vfr else "ok")
        self.slider.setRange(0, max(info.n_frames - 1, 0))
        self.slider.setValue(0)
        self._set_enabled(True)
        self._fit_viewer_to_video()
        self._refresh_queue()
        self._sync_placement_ui()
        self._touch()
        self._render_preview()

    def _on_param(self, *_):
        if self.reader is not None:
            self._debounce.start()

    def _on_scrub(self, i: int):
        if self.info:
            self.lbl_t.setText(f"frame {i} / {self.info.n_frames}   "
                               f"{i / self.info.measured_fps:7.3f} s")
        self._on_param()

    def _render_preview(self):
        if self.reader is None:
            return
        # Gated on our own flag, not on QThread.isRunning().
        #
        # `done` is emitted from inside the worker's run(), so by the time the
        # queued slot executes on the GUI thread the worker has usually -- but
        # only usually -- finished. When it has not, the re-render that
        # _preview_done fires lands here, sees isRunning() still True, sets
        # _pending and returns... and no further `done` is coming, so the
        # preview never updates again for the rest of the session. Whether that
        # happens is scheduler timing, which is why it showed up as "the
        # preview does not live-update on the Mac" while the identical code was
        # fine on a faster Windows box. This flag is cleared by the slot itself,
        # so it cannot race with thread teardown.
        if self._preview_busy:
            self._pending = True          # coalesce; re-render when this one lands
            return
        t = self.slider.value() / (self.info.measured_fps or 30.0)
        seed = self._manual_seed()
        # A cheaper schedule than the run. The preview exists to judge whether the
        # mask and the outline are right, and 24 restarts land within a pixel of
        # 64 on a frame that is going to be re-measured properly anyway. The run
        # itself always uses the full settings.
        full = self._fit_cfg()
        fit_cfg = FitConfig(**{**full.__dict__,
                               "n_restarts": min(full.n_restarts, 24),
                               "iters": min(full.iters, 80),
                               "iters_warm": min(full.iters_warm, 40)})
        # The fitter is rebuilt only when something it actually depends on moves.
        key = (id(self.template), tuple(sorted(fit_cfg.__dict__.items())),
               None if seed is None else tuple(np.round(seed, 3)),
               self.chk_gpu.isChecked())
        if key != self._fitter_key or self._fitter is None:
            self._fitter = (ShapeFitter(self.template, fit_cfg,
                                        get_device(self.chk_gpu.isChecked()),
                                        seed_pose=seed)
                            if self.template is not None else None)
            self._fitter_key = key
        elif self._fitter is not None:
            # A fresh preview is a standalone measurement of this one frame, so
            # it must not warm-start from whatever frame was shown last.
            self._fitter.prev = None

        self._retire("_preview")
        self._preview = PreviewWorker(
            self.reader, self.background, self._seg_cfg(), fit_cfg,
            self.template, t, self.chk_gpu.isChecked(),
            {"mask": self.chk_mask.isChecked(), "fit": self.chk_fit.isChecked()},
            seed=seed, model=self.model, cache=self._frame_cache,
            fitter=self._fitter,
            view=self._view_mode())
        self._preview.done.connect(self._preview_done)
        self._preview_busy = True
        self._preview.start()

    def _preview_done(self, img, status, err, fitted):
        self._preview_busy = False
        if err:
            self._log(err.strip().splitlines()[-1])
        elif img is not None:
            h, w = img.shape[:2]
            qi = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
            # Handed over at full size: the view scales it while painting, so a
            # window resize no longer costs a re-decode and a re-fit.
            self.view.set_frame(QPixmap.fromImage(qi), (w, h))
            self.lbl_status.setText(status)
            self._last_fit_pose = fitted
            self._update_dims()
            self.btn_snap.setEnabled(bool(fitted) and self.chk_manual.isChecked()
                                     and self.template is not None)
        if self._pending:
            self._pending = False
            self._render_preview()

    def _fit_viewer_to_video(self):
        """Give the viewer column the width the frame actually wants.

        The three panes start at a fixed 400/700/380 split, which is right for
        nothing in particular: a portrait phone clip gets a wide pane with grey
        margins either side, and a 4K landscape clip gets a letterboxed strip.
        Sizing the middle pane to the frame's aspect ratio removes the wasted
        space without taking any from the sidebar -- the plots give it up and
        get it back, and they reflow happily.
        """
        if self.info is None or self._split is None:
            return
        h = max(self.view.height(), 200)
        want = int(round(h * (self.info.width / max(self.info.height, 1))))
        sizes = self._split.sizes()
        if len(sizes) != 3:
            return
        total = sum(sizes)
        # Never starve the other two: the sidebar keeps its width and the plots
        # keep a floor, so a very wide clip stops growing rather than pushing
        # the plot panel off the window.
        want = max(420, min(want, total - sizes[0] - 300))
        if abs(want - sizes[1]) < 24:
            return
        self._split.setSizes([sizes[0], want, total - sizes[0] - want])

    def _view_mode(self) -> str:
        """"video" or "chroma", from the View selector.

        One place, because the preview worker and the playback worker both need
        it and they used to derive it separately -- which is how playback ended
        up ignoring the selector entirely.
        """
        return "chroma" if self.cmb_view.currentIndex() == 1 else "video"

    def _reset_run_button(self):
        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run analysis")
        self.btn_run.setProperty("primary", True)
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)

    def _run_aborted(self, why: str):
        self.bar.setVisible(False)
        self.bar.setFormat("")
        self.plots.stop_live()
        self._reset_run_button()
        snd.stopped(self.state.get("sound_enabled", True))
        self._log(f"aborted — {why}. No results were written.")

    def _run(self):
        # The same button aborts. Two buttons would mean one of them is disabled
        # and useless at any given moment, and the one you want is whichever the
        # run is not currently doing.
        if getattr(self, "_running", False):
            if self._runner is not None:
                self.btn_run.setEnabled(False)
                self.btn_run.setText("Stopping…")
                self._log("abort requested — finishing the current frame")
                self._runner.abort()
            return
        if not self.video:
            return
        out = QFileDialog.getExistingDirectory(self, "Choose an output folder",
                                               self.state.get("output_dir") or "")
        if not out:
            return
        self.outdir = out
        self._touch()
        # The green dots are "a results folder exists under here", so they are
        # only meaningful once "here" is known -- and they change wholesale when
        # it changes.
        self._refresh_queue()
        manual = (list(self.manual_pose)
                  if (self.manual_pose and self.chk_manual.isChecked()
                      and self.template is not None) else None)
        cfg = RunConfig(
            video=self.video, dxf=self.dxf,
            dxf_loop_index=int(self.state.get("dxf_loop_index", 0)),
            dxf_scale=self.spin_dxfscale.value(),
            use_features=self.chk_features.isChecked(),
            force_method=self.force_method(),
            force_lut=self.lut_path,
            beam=self._beam_model() if self.force_method() == "beam" else None,
            # Whatever is framed in the panel right now is what the exported
            # figure will show, and it is recorded in run_info.json so the
            # figure and the numbers cannot disagree about their range.
            axis_ranges=self.plots.axis_ranges(),
            # The appearance lock learns from the frame you are looking at, the
            # region you drew, and whatever pose is currently fitted there --
            # which is why it is worth scrubbing to a clean frame first.
            # The appearance patch is an upright crop: any rotation on the
            # region is deliberately dropped here rather than silently baked in,
            # because the tracker recovers rotation itself and a pre-rotated
            # reference would double-count it.
            appearance=((self._current_time(), tuple(self._active_roi()[:4]),
                         self._last_pose_obj())
                        if (self.chk_appearance.isChecked() and self._active_roi())
                        else None),
            known_width_mm=(self.spin_width.value() or None),
            # Roughly six updates a second at 36 fps: enough to watch tracking
            # hold, cheap enough not to slow the run.
            preview_every=6,
            outdir=out,
            px_per_mm=self.spin_ppm.value() or None,
            scale=self._scale(), write_overlay=self.chk_overlay.isChecked(),
            gpu=self.chk_gpu.isChecked(), manual_pose=manual,
            segment=self._seg_cfg(), fit=self._fit_cfg(), analysis=self._ana_cfg(),
        )
        if manual:
            self._log("seeding the fit from the hand-placed outline")
        # Say out loud which region this run used. A region is invisible in the
        # output but decides what was measured, and "did it use the one I just
        # drew?" is otherwise unanswerable without opening run_info.json.
        if cfg.segment.roi:
            r = list(cfg.segment.roi)
            turn = f", turned {r[4]:+.1f}\u00b0" if len(r) > 4 and abs(r[4]) > 0.05 else ""
            self._log(f"region: {r[2]} \u00d7 {r[3]} px at ({r[0]}, {r[1]}){turn}")
        else:
            self._log("region: whole frame")
        self.btn_run.setText("Abort")
        self.btn_run.setProperty("primary", False)
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)
        self._running = True
        self.bar.setVisible(True); self.bar.setRange(0, 0)
        self.bar.setTextVisible(True)
        self.bar.setFormat("starting…")
        self._run_started = time.time()
        dev = get_device(self.chk_gpu.isChecked())
        self._log(f"running → {out}")
        self._log(f"  device {dev}  ·  decode scale {self._scale()}  ·  "
                  f"{self._fit_cfg().n_restarts} restarts  ·  "
                  f"features {'on' if self.chk_features.isChecked() else 'off'}")
        if not dev.accelerated:
            # A 20x slowdown deserves more than a chip in the corner.
            self._log("  WARNING: no GPU acceleration — this will run roughly 20x slower.")
            import sys as _sys
            why = ("Metal acceleration needs macOS 12.3 or later on Apple Silicon; "
                   "an Intel Mac has no MPS backend."
                   if _sys.platform == "darwin" else
                   "Check your NVIDIA driver, and that torch came from the CUDA "
                   "index rather than plain PyPI.")
            if QMessageBox.question(
                    self, "Running on the CPU",
                    "No GPU acceleration is available, so the shape fit will run on "
                    f"the CPU.\n\n{why}\n\n"
                    "The fit is the dominant cost: a clip that takes 20 seconds on the "
                    "GPU takes about 10 minutes here.\n\nRun anyway?") != QMessageBox.Yes:
                self._reset_run_button()
                self.bar.setVisible(False)
                return
        self.view.set_pose(None, editable=False)
        # Force needs a *calibration*, not just a method: it is computed from
        # length in millimetres, and with no ruler there are no millimetres.
        # Claiming a force panel here and then losing the column at the end made
        # the panel appear during the run and vanish the moment it finished,
        # which reads as the plot breaking rather than as a missing input.
        can_calibrate = bool(self._true_width_mm() or self.spin_ppm.value())
        will_have_force = can_calibrate and (
            self.force_method() == "beam"
            or (self.force_method() == "lut" and self.lut is not None))
        if self.force_method() != "none" and not can_calibrate:
            self._log("no force this run: it needs a calibration, and neither a "
                      "drawing, a true width nor a px/mm figure is set")
        self.plots.start_live(um_per_px=self._um_per_px(),
                              has_force=will_have_force,
                              width_mm=self._true_width_mm())
        self._retire("_runner")
        self._runner = RunWorker(cfg)
        self._runner.progress.connect(self._on_progress)
        self._runner.row.connect(self.plots.add_row)
        self._runner.frame.connect(self._on_run_frame)
        self._runner.aborted.connect(self._run_aborted)
        self._runner.done.connect(self._run_done)
        self._runner.start()

    def _on_progress(self, i, n):
        if n:
            self.bar.setRange(0, n)
            self.bar.setValue(i)
            self.bar.setTextVisible(True)
            elapsed = max(time.time() - self._run_started, 1e-3)
            rate = i / elapsed
            eta = (n - i) / rate if rate > 0 else 0
            self.bar.setFormat(f"frame {i} / {n}   %p%   {rate:.0f} fps"
                               + (f"   about {_hms(eta)} left" if i > 20 else ""))
        else:
            self.bar.setRange(0, 0)
            self.bar.setFormat("")

    def _run_done(self, res, err):
        self.bar.setVisible(False)
        self.bar.setFormat("")
        self.plots.stop_live()
        self._reset_run_button()
        if err:
            snd.stopped(self.state.get("sound_enabled", True))
            QMessageBox.critical(self, "Analysis failed", err)
            self._log(err.strip().splitlines()[-1])
            return
        # Rising chime before the dialog, not after: the dialog blocks, and the
        # whole point is to be heard by someone who has walked away from it.
        snd.finished(self.state.get("sound_enabled", True))
        self._log(res.summary())
        # Notes are things the run wants read, not skimmed past in a summary
        # block -- an accelerator that failed its self-check, a region that
        # clips the robot. Repeated on their own lines for that reason.
        for note in getattr(res, "notes", []) or []:
            self._log(f"note: {note}")
        self._result = res
        self.btn_export_sel.setEnabled(True)
        self._record_throughput(res)
        # Swap the live raw trace for the gated, smoothed, calibrated table.
        um = (1000.0 / res.calibration_px_per_mm) if res.calibration_px_per_mm else None
        self.plots.set_table(res.table, um, "force_mn" in res.table)
        if um is None:
            self._log("no calibration — plots are in pixels. Load a DXF, or type "
                      "the robot's true width in the Input card, to get micrometers.")
        self._sync_placement_ui()
        self._persist()
        # Refresh the dots: the clip that just finished now has a folder.
        self._refresh_queue()
        box = QMessageBox(self)
        box.setWindowTitle("Analysis complete")
        box.setText(f"Written to {pipeline_output_dir(self.outdir, self.video)}")
        box.setDetailedText(res.summary())
        box.exec()


def main(argv=None, startup_warning: str | None = None, splash: bool = True) -> int:
    """Entry point for ``robotrack-gui``.

    Reuses an existing QApplication when one is already up -- the frozen launcher
    creates it before the splash, and creating a second one aborts Qt.
    """
    if QApplication.instance() is None and splash:
        from .splash import run_with_splash
        return run_with_splash(argv if argv is not None else sys.argv,
                               startup_warning=startup_warning)
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    # Read the saved mode before any widget exists: restyling afterwards works,
    # but the window would visibly flash from dark to light on every launch.
    try:
        apply_theme(app, S.load_settings().get("theme_mode", "dark"))
    except Exception:
        apply_theme(app)
    w = MainWindow(startup_warning=startup_warning)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
