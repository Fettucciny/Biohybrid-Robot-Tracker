"""PySide6 desktop interface for robotrack.

The point of this window is not to hide the command line -- it is to let you
*see the segmentation before committing to a full run*. Getting the threshold
and morphology right on one representative frame takes seconds here and saves
reprocessing a ten-minute 4K clip because the mask was wrong.

Every control carries a (?) badge: hovering gives a one-line summary with the
valid range, clicking opens the full explanation from paramhelp.py.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSpinBox, QSplitter, QTextEdit, QVBoxLayout,
                               QWidget)

from . import settings as S
from . import update as U
from .cad import Template, load_dxf, read_loops
from .decode import FrameReader, select_backend
from .gpu import get_device
from .ingest import VideoInfo, probe
from .kinematics import AnalysisConfig
from .paramhelp import HELP
from .pipeline import RunAborted, RunConfig, run
from .forcelut import LUTError, load_lut
from .forcemodel import BeamForceModel
from .placement import PreviewView
from .plotpanel import PlotPanel
from .register import FitConfig, ShapeFitter
from .segment import (ColourModel, SegmentConfig, choose_colour_model, colour_distance,
                      build_background, largest_component, segment_colour,
                      segment_frame)
from .shape import measure_mask
from .theme import ACCENT, C, Card, HelpBadge, apply as apply_theme, style_chip
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


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> OpenCV BGR, so overlay colours track the theme accent."""
    h = hex_colour.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


OK_BGR = _bgr(C["ok"])        # confident fit
WARN_BGR = _bgr(C["warn"])    # low confidence


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

            self.note.emit("measuring colour separation…")
            model, _ = choose_colour_model(reader, self.seg)
            self.note.emit(model.summary())

            bg = None
            if model.mode != "colour":
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

    def _frame(self, colour: bool):
        """Decoded frame for this timestamp, from the cache when possible.

        Scrubbing revisits the same frames constantly -- nudging a parameter and
        looking again, stepping back and forth over a contraction. Each decode is
        an ffmpeg launch and a keyframe seek, so caching them is most of the
        responsiveness win.
        """
        key = (round(self.t, 4), colour, self.reader.scale)
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
            colour = self.model is not None and self.model.mode == "colour"
            frame = self._frame(colour)
            if frame is None:
                self.done.emit(None, "", "Could not decode a frame at that position.", None)
                return
            dev = get_device(self.gpu)
            min_area = self.seg_cfg.min_area_frac * self.reader.width * self.reader.height

            if colour:
                m, thr = segment_colour(frame, self.seg_cfg, self.model.bg_ab,
                                        self.model.separation,
                                        self.seg_cfg.manual_threshold)
                if self.view == "chroma":
                    # What the segmenter actually sees: distance from the
                    # medium's colour. The threshold is a horizontal cut through
                    # this surface, so a mask that looks wrong is diagnosed here
                    # rather than guessed at from the photograph.
                    d = colour_distance(frame, self.model.bg_ab)
                    scaled = np.clip(d / max(self.model.separation, 1e-6) * 255.0,
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
            mask, area, contour = largest_component(
                m, min_area, reach_px=self.seg_cfg.gap_factor * _blob_extent(m),
                envelope_factor=self.seg_cfg.envelope_factor)
            if self.show["mask"] and mask.any():
                tint = np.zeros_like(img)
                tint[mask > 0] = _bgr(ACCENT)
                img = cv2.addWeighted(img, 1.0, tint, 0.40, 0)

            bits = [("dC" if colour else "thr") + f" {thr:.0f}", f"area {int(area)}px"]
            fitted = None
            if self.tpl is not None and mask.any():
                # Reuse the caller's fitter when it still matches: building one
                # rasterises a signed-distance grid and uploads the template, and
                # doing that per keystroke was pure overhead.
                fitter = self.fitter or ShapeFitter(self.tpl, self.fit_cfg, dev,
                                                    seed_pose=self.seed)
                sig = (colour_distance(frame, self.model.bg_ab) if colour else
                       (frame if frame.ndim == 2 else
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                pose = fitter.fit(mask, signal=sig)
                if pose is not None:
                    fitted = list(pose.as_array())
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
                    if self.show["fit"] and contour is not None:
                        cv2.polylines(img, [contour.astype(np.int32)], True,
                                      OK_BGR, 2, cv2.LINE_AA)
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

    def __init__(self, reader, model, bg, seg_cfg, start_index, show_mask, gpu, fps):
        super().__init__()
        self.reader, self.model, self.bg, self.seg_cfg = reader, model, bg, seg_cfg
        self.start_index, self.show_mask, self.gpu = start_index, show_mask, gpu
        self.fps = max(float(fps), 1.0)
        self._stop = False
        self._last = start_index

    def stop(self):
        self._stop = True

    def run(self):
        try:
            colour = self.model is not None and self.model.mode == "colour"
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
                if colour:
                    m, thr = segment_colour(frame, self.seg_cfg, self.model.bg_ab,
                                            self.model.separation,
                                            self.seg_cfg.manual_threshold)
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
                    mask, area, contour = largest_component(
                        m, min_area, envelope_factor=self.seg_cfg.envelope_factor)
                    if self.show_mask and mask.any():
                        tint = np.zeros_like(img)
                        tint[mask > 0] = _bgr(ACCENT)
                        img = cv2.addWeighted(img, 1.0, tint, 0.40, 0)
                    # The tracked outline is drawn whether or not the mask tint
                    # is on. It is the one thing worth watching during playback:
                    # a tracker that lets go shows it here frames before the
                    # numbers do.
                    if contour is not None and len(contour) > 2:
                        cv2.polylines(img, [contour.astype(np.int32)], True,
                                      (255, 255, 255), 1, cv2.LINE_AA)
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

class MainWindow(QMainWindow):
    def __init__(self, startup_warning: str | None = None):
        super().__init__()
        self.setWindowTitle("robotrack — biohybrid robot tracking")
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
        self.reader: FrameReader | None = None
        self.background = None
        self.template: Template | None = None
        self.outdir: str | None = None
        self._preview: PreviewWorker | None = None
        self._pending = False
        self._startup_warning = startup_warning
        self.model: ColourModel | None = None
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
        self._loading_state = False

        # Manual placement, stored in *full-resolution* image pixels so it
        # survives a change of decode scale.
        self.manual_pose: list[float] | None = self.state.get("manual_pose")
        self._last_fit_pose: list[float] | None = None

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

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_sidebar())
        split.addWidget(self._build_viewer())
        self.plots = PlotPanel()
        self.plots.selectionAnalysed.connect(self._on_region)
        split.addWidget(self.plots)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 700, 380])
        split.setHandleWidth(10)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 12, 14, 14)
        bl.addWidget(split)
        outer.addWidget(body, 1)
        self.setCentralWidget(root)

        self._set_enabled(False)
        dev = get_device(True)
        self.chip_gpu.setText(dev.name if dev.accelerated else "CPU only")
        style_chip(self.chip_gpu, "ok" if dev.accelerated else "warn")
        self._log(f"device: {dev}")

        self._apply_state(self.state)
        self._on_force_method()
        self._wire_persistence()
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

        if self.state.get("check_updates_on_start") and self.state.get("update_channel"):
            QTimer.singleShot(1200, lambda: self._check_updates(quiet=True))

    # ---- chrome ----------------------------------------------------------

    def _build_header(self) -> QWidget:
        h = QFrame(); h.setObjectName("Header"); h.setFixedHeight(58)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(18, 0, 18, 0)
        lay.setSpacing(10)

        dot = QLabel("❮❯"); dot.setObjectName("AppMark")
        name = QLabel("robotrack"); name.setObjectName("AppName")
        tag = QLabel("muscle-driven robot kinematics"); tag.setObjectName("Tagline")
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

        self.btn_update = QPushButton("Update")
        self.btn_update.setObjectName("Ghost")
        self.btn_update.setToolTip("Check for and install a new version")
        self.btn_update.clicked.connect(lambda: self._check_updates(quiet=False))
        lay.addWidget(self.btn_update)
        return h

    # ---- sidebar ---------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 8, 0)
        v.setSpacing(11)

        # -------------------------------------------------- input
        c = Card("Input")
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
        v.addWidget(c)

        # -------------------------------------------------- segmentation
        c = Card("Segmentation")
        self.cmb_keying = QComboBox()
        self.cmb_keying.addItems(["Auto", "Colour (a*b*)", "Brightness"])
        self.cmb_keying.currentIndexChanged.connect(self._reload_needed)
        c.add_row("Keying", self.cmb_keying, HELP["keying"])

        self.spin_cfrac = QDoubleSpinBox(); self.spin_cfrac.setRange(0.05, 0.90)
        self.spin_cfrac.setSingleStep(0.05); self.spin_cfrac.setValue(0.30)
        self.spin_cfrac.setDecimals(2)
        self.spin_cfrac.valueChanged.connect(self._on_param)
        c.add_row("Colour cut", self.spin_cfrac, HELP["colour_frac"])

        self.spin_env = QDoubleSpinBox(); self.spin_env.setRange(1.0, 4.0)
        self.spin_env.setSingleStep(0.05); self.spin_env.setValue(1.10)
        self.spin_env.valueChanged.connect(self._on_param)
        c.add_row("Body envelope", self.spin_env, HELP["envelope"])

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
        c.add_row("Despeckle", self.spin_open, HELP["despeckle"])

        self.spin_close = QSpinBox(); self.spin_close.setRange(1, 41)
        self.spin_close.setSingleStep(2); self.spin_close.setValue(7); self.spin_close.setSuffix(" px")
        self.spin_close.valueChanged.connect(self._on_param)
        c.add_row("Fill holes", self.spin_close, HELP["fill_holes"])

        self.spin_minarea = QDoubleSpinBox(); self.spin_minarea.setDecimals(4)
        self.spin_minarea.setRange(0.0010, 1.0000); self.spin_minarea.setSingleStep(0.005)
        self.spin_minarea.setValue(0.0500); self.spin_minarea.setSuffix(" %")
        self.spin_minarea.valueChanged.connect(self._on_param)
        c.add_row("Min blob size", self.spin_minarea, HELP["min_area"])

        self.spin_gap = QDoubleSpinBox(); self.spin_gap.setRange(0.0, 3.0)
        self.spin_gap.setSingleStep(0.1); self.spin_gap.setValue(1.0)
        self.spin_gap.setSuffix(" × body")
        self.spin_gap.valueChanged.connect(self._on_param)
        c.add_row("Occlusion gap", self.spin_gap, HELP["gap_factor"])

        self.spin_bg = QSpinBox(); self.spin_bg.setRange(10, 400); self.spin_bg.setValue(60)
        self.spin_bg.setSuffix(" frames")
        self.spin_bg.valueChanged.connect(self._reload_needed)
        c.add_row("Background", self.spin_bg, HELP["background_frames"])

        self.btn_rebuild = QPushButton("Rebuild background")
        self.btn_rebuild.setObjectName("Ghost")
        self.btn_rebuild.clicked.connect(self._reload)
        self.btn_rebuild.setVisible(False)
        c.add_widget(self.btn_rebuild)
        v.addWidget(c)

        # -------------------------------------------------- fitting
        c = Card("Shape fitting")
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
        v.addWidget(c)

        # -------------------------------------------------- force
        c = Card("Force")
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

        self.spin_E = beam_row("Young's modulus", 0.1, 1e6, 293.0, 1, " kPa", "beam_E")
        self.spin_t = beam_row("Beam thickness", 0.001, 100.0, 1.100, 3, " mm", "beam_geom")
        self.spin_bw = beam_row("Beam width", 0.001, 100.0, 1.925, 3, " mm", "beam_geom")
        self.spin_Lleg2leg = beam_row("Leg to leg", 0.001, 1000.0, 8.250, 3, " mm", "beam_geom")
        self.spin_arm = beam_row("Muscle offset", 0.001, 1000.0, 1.642, 3, " mm", "beam_geom")
        self.spin_leg_long = beam_row("Leg length (long)", 0.001, 1000.0, 4.125, 3, " mm", "beam_geom")
        self.spin_leg_short = beam_row("Leg length (short)", 0.001, 1000.0, 3.300, 3, " mm", "beam_geom")

        self.cmb_rest = QComboBox()
        self.cmb_rest.addItems(["Maximum (Cvetkovic Model)", "Robust (upper quartile)"])
        self.cmb_rest.currentIndexChanged.connect(self._on_beam_changed)
        self.beam_rows.append(c.add_row("Resting length", self.cmb_rest, HELP["beam_rest"]))

        self.lbl_beam = QLabel(""); self.lbl_beam.setObjectName("Readout")
        self.lbl_beam.setWordWrap(True)
        c.add_widget(self.lbl_beam)
        self.beam_rows.append(self.lbl_beam)
        v.addWidget(c)

        # -------------------------------------------------- placement
        c = Card("Target placement")
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
        v.addWidget(c)

        # -------------------------------------------------- analysis
        c = Card("Analysis")
        self.spin_smooth = QDoubleSpinBox(); self.spin_smooth.setRange(0, 2000)
        self.spin_smooth.setValue(100); self.spin_smooth.setSuffix(" ms")
        c.add_row("Smoothing", self.spin_smooth, HELP["smoothing"])

        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0, 1)
        self.spin_conf.setSingleStep(0.05); self.spin_conf.setValue(0.50)
        c.add_row("Min confidence", self.spin_conf, HELP["min_confidence"])

        self.spin_gapms = QDoubleSpinBox(); self.spin_gapms.setRange(0, 5000)
        self.spin_gapms.setValue(400); self.spin_gapms.setSuffix(" ms")
        c.add_row("Max bridged gap", self.spin_gapms, HELP["max_gap"])

        self.spin_ppm = QDoubleSpinBox(); self.spin_ppm.setRange(0, 10000)
        self.spin_ppm.setDecimals(3); self.spin_ppm.setSpecialValueText("auto")
        self.spin_ppm.setSuffix(" px/mm")
        c.add_row("Calibration", self.spin_ppm, HELP["px_per_mm"])
        v.addWidget(c)

        # -------------------------------------------------- output
        c = Card("Output")
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
        v.addWidget(c)

        # -------------------------------------------------- configuration
        c = Card("Configuration")
        note = QLabel("Every setting here is remembered between sessions. "
                      "Save one to a file to reproduce a run, or to share it.")
        note.setObjectName("Hint"); note.setWordWrap(True)
        c.add_widget(note)

        rowc = QWidget(); rc = QHBoxLayout(rowc); rc.setContentsMargins(0, 0, 0, 0); rc.setSpacing(7)
        self.btn_cfg_save = QPushButton("Save config…"); self.btn_cfg_save.setObjectName("Ghost")
        self.btn_cfg_load = QPushButton("Load config…"); self.btn_cfg_load.setObjectName("Ghost")
        self.btn_cfg_save.clicked.connect(self._save_config)
        self.btn_cfg_load.clicked.connect(self._load_config)
        rc.addWidget(self.btn_cfg_save, 1); rc.addWidget(self.btn_cfg_load, 1)
        c.add_widget(rowc)

        self.btn_cfg_reset = QPushButton("Reset to defaults")
        self.btn_cfg_reset.setObjectName("Ghost")
        self.btn_cfg_reset.clicked.connect(self._reset_config)
        c.add_widget(self.btn_cfg_reset)

        self.chk_update_start = QCheckBox("check for updates at launch")
        rowu = QWidget(); ru = QHBoxLayout(rowu); ru.setContentsMargins(0, 0, 0, 0); ru.setSpacing(7)
        ru.addWidget(self.chk_update_start); ru.addWidget(HelpBadge(HELP["update_channel"]))
        ru.addStretch(1)
        c.add_widget(rowu)
        v.addWidget(c)

        self.btn_run = QPushButton("Run analysis")
        self.btn_run.setProperty("primary", True)
        self.btn_run.clicked.connect(self._run)
        v.addWidget(self.btn_run)

        self.bar = QProgressBar(); self.bar.setVisible(False); self.bar.setTextVisible(False)
        v.addWidget(self.bar)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(120)
        v.addWidget(self.log)
        v.addStretch(1)

        scroll.setWidget(inner)
        scroll.setMinimumWidth(400)
        return scroll

    # ---- viewer ----------------------------------------------------------

    def _build_viewer(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(9)

        topbar = QWidget(); tb = QHBoxLayout(topbar)
        tb.setContentsMargins(2, 0, 2, 0); tb.setSpacing(8)
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Video", "Colour distance (b*)"])
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
        return w

    # ---- config assembly -------------------------------------------------

    def _seg_cfg(self) -> SegmentConfig:
        return SegmentConfig(
            mode=["auto", "colour", "luma"][self.cmb_keying.currentIndex()],
            colour_frac=self.spin_cfrac.value(),
            envelope_factor=self.spin_env.value(),
            n_background_frames=self.spin_bg.value(),
            open_px=self.spin_open.value() | 1,
            close_px=self.spin_close.value() | 1,
            min_area_frac=self.spin_minarea.value() / 100.0,
            gap_factor=self.spin_gap.value(),
            manual_threshold=None if self.cmb_thr.currentIndex() == 0
            else float(self.spin_thr.value()),
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
            "colour_frac": spin(self.spin_cfrac),
            "dxf_scale": spin(self.spin_dxfscale),
            "known_width_mm": spin(self.spin_width),
            "use_features": check(self.chk_features),
            "early_stop": check(self.chk_earlystop),
            "force_method_index": combo(self.cmb_force),
            "beam_E_kpa": spin(self.spin_E),
            "beam_thickness_mm": spin(self.spin_t),
            "beam_width_mm": spin(self.spin_bw),
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
            "check_updates_on_start": check(self.chk_update_start),
        }

    def _collect_state(self) -> dict:
        st = dict(self.state)
        for key, (getter, _) in self._bindings().items():
            st[key] = getter()
        st["video_path"] = self.video or ""
        st["dxf_path"] = self.dxf or ""
        st["output_dir"] = self.outdir or ""
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
                   self.spin_prior, self.spin_maxscale, self.spin_smooth,
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

    def closeEvent(self, ev):
        self._stop_play()
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
        so a drawing at 5:1 detail scale makes every micrometre in the output
        five times too large. Reloading immediately means the millimetre readout
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
        # would queue a decode and an optimisation per pixel of travel.
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
        """Report what the selected stretch measured."""
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
                            f"is wandering more than the robot is travelling")
        self._log("\n".join(bits))

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
        self._player = PlaybackWorker(
            self.reader, self.model, self.background, self._seg_cfg(),
            self.slider.value(), self.chk_mask.isChecked(),
            self.chk_gpu.isChecked(), self.info.measured_fps * self._speed())
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
        channel = self.state.get("update_channel", "")
        if quiet and not channel:
            return
        dlg = UpdateDialog(channel, self,
                           on_channel_change=self._set_update_channel)
        dlg.relaunchRequested.connect(self._relaunch)
        if quiet:
            # Launched by the "check at launch" preference: look first, and only
            # interrupt if there is actually something to install.
            dlg.check()
        dlg.exec()

    def _set_update_channel(self, channel: str):
        self.state["update_channel"] = channel
        S.save_settings(self.state)
        self._log(f"update channel: {U.describe_channel(channel)}")

    def _relaunch(self):
        self._persist()
        try:
            U.relaunch()
        except Exception as exc:
            QMessageBox.warning(
                self, "Restart",
                f"The update is installed but robotrack could not restart itself:\n\n"
                f"{exc}\n\nClose and reopen it.")
            return
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
            self.video = f
            self.state["last_video_dir"] = str(Path(f).parent)
            self.lbl_video.setText(Path(f).name)
            self._touch()
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
        millimetre, so carrying them across would silently mean a different size.
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
        # Labelled at the drawing scale, not as drawn. Listing the raw numbers
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
        self._set_enabled(False)
        self.view.set_frame(None, (0, 0))
        self.view.set_pose(None, editable=False)
        self.view.setText("building background model…")
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
        # Colour keying reads BGR; luma reads gray. Decode what is actually used
        # rather than decoding both.
        if model is not None and model.mode == "colour":
            self.reader = FrameReader(info, reader.backend, scale=reader.scale, color=True)
        self.chip_key.setText("colour" if (model and model.mode == "colour") else "luma")
        style_chip(self.chip_key, "ok" if (model and model.mode == "colour") else "")
        self.lbl_info.setText(info.summary())
        self._log(info.summary())
        self.chip_rate.setText(f"{info.nominal_fps:g} Hz · {info.n_frames} frames")
        style_chip(self.chip_rate, "warn" if info.is_vfr else "ok")
        self.slider.setRange(0, max(info.n_frames - 1, 0))
        self.slider.setValue(0)
        self._set_enabled(True)
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
        if self._preview is not None and self._preview.isRunning():
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

        self._preview = PreviewWorker(
            self.reader, self.background, self._seg_cfg(), fit_cfg,
            self.template, t, self.chk_gpu.isChecked(),
            {"mask": self.chk_mask.isChecked(), "fit": self.chk_fit.isChecked()},
            seed=seed, model=self.model, cache=self._frame_cache,
            fitter=self._fitter,
            view="chroma" if self.cmb_view.currentIndex() == 1 else "video")
        self._preview.done.connect(self._preview_done)
        self._preview.start()

    def _preview_done(self, img, status, err, fitted):
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
        self.plots.start_live(um_per_px=self._um_per_px(),
                              has_force=(self.force_method() == "beam"
                                         or (self.force_method() == "lut"
                                             and self.lut is not None)),
                              width_mm=self._true_width_mm())
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
            QMessageBox.critical(self, "Analysis failed", err)
            self._log(err.strip().splitlines()[-1])
            return
        self._log(res.summary())
        # Swap the live raw trace for the gated, smoothed, calibrated table.
        um = (1000.0 / res.calibration_px_per_mm) if res.calibration_px_per_mm else None
        self.plots.set_table(res.table, um, "force_mn" in res.table)
        if um is None:
            self._log("no calibration — plots are in pixels. Load a DXF, or type "
                      "the robot's true width in the Input card, to get micrometres.")
        self._sync_placement_ui()
        self._persist()
        box = QMessageBox(self)
        box.setWindowTitle("Analysis complete")
        box.setText(f"Written to {self.outdir}")
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
    apply_theme(app)
    w = MainWindow(startup_warning=startup_warning)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
