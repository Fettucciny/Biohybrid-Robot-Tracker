"""Live, interactive plots beside the video.

Two jobs that are usually two different tools:

**While a run is going**, this is the only way to see that a ten-minute clip is
producing sensible numbers before it finishes. Rows arrive frame by frame and are
drawn at a fixed refresh rate rather than per row -- redrawing a matplotlib figure
1800 times would take longer than the analysis.

**After it finishes**, the same panel becomes the reading surface: scroll to zoom
time, shift-scroll to zoom the value axis, and whatever range is on screen when
Export is pressed is the range written into the exported figure and recorded in
run_info.json. A figure you cropped to the interesting three seconds is worth
keeping; one that silently reverted to the full clip is not.

Units follow the calibration. With the robot's width as the ruler the vertical
axis is micrometers; without a calibration it stays in pixels and says so.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel,
                               QSizePolicy, QVBoxLayout, QWidget)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .theme import C as THEME, matplotlib_rc, series_colors


def average_delta(t: np.ndarray, y: np.ndarray,
                  prominence_frac: float = 0.15) -> tuple[float, int]:
    """Mean peak-to-trough swing of an oscillating trace, and the cycle count.

    Not ``max - min``: that reports the single largest excursion in the window,
    which is the noisiest sample available and grows with window length. Pairing
    successive turning points instead gives the average contraction amplitude,
    which is what a "delta length" is meant to mean.

    Turning points need a prominence floor or ordinary measurement noise
    registers as thousands of tiny cycles. The floor scales with the trace's own
    range, so it needs no unit-specific tuning and works on strain and
    micrometers alike.
    """
    from scipy.signal import find_peaks

    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if y.size < 5:
        return float("nan"), 0
    rng = float(np.nanmax(y) - np.nanmin(y))
    if rng <= 0:
        return 0.0, 0
    prom = rng * prominence_frac
    peaks, _ = find_peaks(y, prominence=prom)
    troughs, _ = find_peaks(-y, prominence=prom)
    if peaks.size == 0 and troughs.size == 0:
        return rng, 0                     # monotonic or a single half-swing

    marked = sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in troughs])
    # Enforce alternation: two peaks in a row would otherwise contribute the
    # small difference between them and drag the average down.
    clean: list[tuple[int, int]] = []
    for idx, kind in marked:
        if clean and clean[-1][1] == kind:
            keep = idx if (y[idx] > y[clean[-1][0]]) == (kind == 1) else clean[-1][0]
            clean[-1] = (keep, kind)
        else:
            clean.append((idx, kind))
    if len(clean) < 2:
        return rng, 0
    deltas = [abs(float(y[clean[i + 1][0]] - y[clean[i][0]]))
              for i in range(len(clean) - 1)]
    # Cycles, not turning points. ``deltas`` counts peak-to-trough *half*
    # swings, so returning its length reported very close to twice the real
    # cycle count -- 89 for the 45 contractions in a 45.5 s clip at 1.0000 Hz,
    # which is exactly 2n-1. The detection was never the problem; only the
    # label was. One cycle is one peak and one trough, so it is half the
    # alternating sequence.
    return float(np.mean(deltas)), len(clean) // 2


def decimate(t: np.ndarray, ys: list, max_points: int = 1400):
    """Thin a long trace for *drawing*, keeping its envelope.

    Matplotlib redraws every vertex on every pan and every scroll, so a 20,000
    frame clip across five panels is 100,000+ vertices per mouse-move and the
    interaction becomes unusable. Plotting fewer points fixes that, but plain
    subsampling drops peaks -- and peaks are the signal here.

    Min/max decimation instead: each bucket contributes its lowest and highest
    sample, in time order. The drawn envelope is identical to the full trace at
    screen resolution while the vertex count is bounded.

    Only the picture is thinned. Region statistics are always computed on the
    full series, so a measured delta never depends on how much was drawn.
    """
    n = t.size
    if n <= max_points:
        return t, ys
    buckets = max(max_points // 2, 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    idx = []
    ref = ys[0]
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = ref[a:b]
        finite = np.isfinite(seg)
        if not finite.any():
            idx.append(a)
            continue
        lo = a + int(np.nanargmin(np.where(finite, seg, np.nan)))
        hi = a + int(np.nanargmax(np.where(finite, seg, np.nan)))
        idx.extend((lo, hi) if lo <= hi else (hi, lo))
    idx = np.unique(np.asarray(idx, int))
    return t[idx], [None if y is None else y[idx] for y in ys]


def resampled_path(t: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                   hz: float | None) -> np.ndarray:
    """Cumulative path length measured at ``hz``, not at the frame rate.

    Path length is a sum of step distances, and every step is the true motion
    plus the tracker's own noise. Noise never cancels in that sum: two positions
    that are really identical still contribute ``|jitter|`` to the total, so the
    path grows roughly with the *number of samples* even when the robot is
    stationary. At 60 fps that is 60 spurious contributions a second, and it is
    why a path-slope speed reads high while net displacement does not.

    Taking the centroid at a lower rate keeps the real trajectory -- a robot
    that moves millimetres per minute is not doing anything interesting between
    consecutive 60 Hz frames -- while cutting the number of noise contributions
    in proportion. Halving the rate roughly halves the jitter contribution and
    leaves genuine displacement untouched.

    Passing ``None`` keeps every frame, which is the old behaviour.
    """
    ok = np.isfinite(t) & np.isfinite(cx) & np.isfinite(cy)
    if ok.sum() < 2:
        return np.full(t.shape, np.nan)
    ts, xs, ys = t[ok], cx[ok], cy[ok]
    if hz and hz > 0:
        # Nearest real sample at each grid time, rather than interpolation:
        # interpolating between two noisy points invents a position that was
        # never measured, and averages the noise in a way that flatters the
        # result. Nearest keeps every reported point an actual observation.
        span = float(ts[-1] - ts[0])
        n = int(np.floor(span * hz)) + 1
        if n >= 2 and n < ts.size:
            grid = ts[0] + np.arange(n) / hz
            idx = np.unique(np.searchsorted(ts, grid).clip(0, ts.size - 1))
            ts, xs, ys = ts[idx], xs[idx], ys[idx]
    step = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(step)])
    # Back onto the original time base so it can be plotted against the rest.
    return np.interp(t, ts, cum, left=np.nan, right=cum[-1])


def slope_per_min(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope in units per minute."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 2 or np.ptp(t) <= 0:
        return float("nan")
    return float(np.polyfit(t, y, 1)[0] * 60.0)


class PlotPanel(QWidget):
    """Stacked time series with scroll-zoom, region analysis and panning."""

    REFRESH_MS = 250
    DRAW_THROTTLE_MS = 30        # ~33 fps ceiling for pan and zoom
    selectionAnalyzed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        matplotlib.rcParams.update(matplotlib_rc())
        self.colors = series_colors()

        self.fig = Figure(figsize=(4.2, 8.0), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(300)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.canvas, 1)

        row = QWidget(); rl = QHBoxLayout(row)
        rl.setContentsMargins(2, 0, 2, 0); rl.setSpacing(7)
        lab = QLabel("Trajectory sampling")
        rl.addWidget(lab)
        self.spin_traj = QDoubleSpinBox()
        self.spin_traj.setRange(0.0, 240.0)
        self.spin_traj.setDecimals(2)
        self.spin_traj.setSingleStep(1.0)
        self.spin_traj.setValue(0.0)
        self.spin_traj.setSuffix(" Hz")
        self.spin_traj.setFixedWidth(104)
        self.spin_traj.setToolTip(
            "Measure the centroid this many times a second when adding up path "
            "length. 0 uses every frame.\n\nPath length sums step distances, and "
            "tracker noise never cancels in that sum — it accumulates with the "
            "number of samples, so a stationary robot still gains distance at "
            "60 samples a second. Sampling lower cuts the noise in proportion "
            "and leaves real displacement alone.\n\nThis is not the Smoothing "
            "setting under Analysis. Smoothing keeps every frame and pulls each "
            "one toward its neighbours, which is what length wants; this throws "
            "frames away, which is the only thing that helps a sum.")
        self.spin_traj.valueChanged.connect(self._on_traj_hz)
        rl.addWidget(self.spin_traj)
        self.lbl_traj = QLabel("every frame"); self.lbl_traj.setObjectName("Readout")
        rl.addWidget(self.lbl_traj, 1)
        lay.addWidget(row)

        self.traj_hz: float | None = None

        self.um_per_px: float | None = None
        self.has_force = False
        # True width of the robot in mm. With it, the panel derives its own
        # scale from the widths arriving in the live rows, so the axis is in
        # micrometers from the first cycle instead of waiting for the run to
        # finish -- and without depending on a preview fit having happened.
        self.width_mm: float | None = None
        self._drag = None
        self._sel = None            # (t0, t1) of the analyzed region
        self._sel_drag = None
        self._sel_artists: list = []
        self.stats: dict = {}
        self._rows: list[dict] = []
        self._dirty = False
        self._axes: dict[str, object] = {}
        self._home: dict[str, tuple] = {}

        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._maybe_redraw)

        # Interaction redraws are throttled rather than queued. Mouse moves
        # arrive far faster than a figure can render, and issuing one draw per
        # event builds a backlog that makes the plot lag behind the cursor.
        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(self.DRAW_THROTTLE_MS)
        self._draw_timer.timeout.connect(self._flush_draw)
        self._last_draw = 0.0
        self._draw_queued = False

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._build()

    def _throttled_draw(self):
        """Draw now if enough time has passed, otherwise once, shortly."""
        now = time.monotonic() * 1000.0
        if now - self._last_draw >= self.DRAW_THROTTLE_MS:
            self._flush_draw()
        elif not self._draw_queued:
            self._draw_queued = True
            self._draw_timer.start()

    def _flush_draw(self):
        self._draw_queued = False
        self._last_draw = time.monotonic() * 1000.0
        self.canvas.draw_idle()

    # ---- construction ----------------------------------------------------

    def _panels(self) -> list[tuple[str, str]]:
        unit = "µm" if self.um_per_px else "px"
        panels = [("length", f"length ({unit})")]
        if self.has_force:
            panels.append(("force", "force (µN)"))
        panels += [("path", f"movement ({unit})"), ("conf", "fit conf. (%)")]
        return panels

    def _build(self):
        self.fig.clear()
        self._axes.clear()
        panels = self._panels()
        ratios = [2.0 if k in ("length", "force") else 1.0 for k, _ in panels]
        axs = self.fig.subplots(len(panels), 1, sharex=True,
                                gridspec_kw={"height_ratios": ratios})
        if len(panels) == 1:
            axs = [axs]
        for ax, (key, label) in zip(axs, panels):
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=7)
            self._axes[key] = ax
        axs[-1].set_xlabel("time (s)", fontsize=8)
        self._axes["conf"].axhline(50.0, ls="--", lw=0.8, color=THEME["muted"])
        # Solve the layout once, then switch the engine off. constrained_layout
        # re-runs its solver on *every* draw, which measured 82 ms of a 135 ms
        # redraw -- paid again on every pan and every scroll, for a layout that
        # has not changed since the figure was built.
        try:
            self.canvas.draw()
            self.fig.set_layout_engine("none")
        except Exception:
            pass

    def configure(self, um_per_px: float | None, has_force: bool):
        """Set units and panel set. Rebuilds only when the layout must change."""
        rebuild = (bool(self.um_per_px) != bool(um_per_px)) or (has_force != self.has_force)
        self.um_per_px, self.has_force = um_per_px, has_force
        if rebuild:
            self._build()
        self._dirty = True

    # ---- data ------------------------------------------------------------

    def start_live(self, um_per_px: float | None = None, has_force: bool = False,
                   width_mm: float | None = None):
        self._rows.clear()
        self.width_mm = width_mm if width_mm and width_mm > 0 else None
        self.configure(um_per_px, has_force)
        self._home.clear()
        self._timer.start()

    def _live_um_per_px(self) -> float | None:
        """Scale from the widths seen so far -- the same ruler the run uses."""
        if not self.width_mm or not self._rows:
            return None
        w = np.array([r.get("width_px", np.nan) for r in self._rows], float)
        w = w[np.isfinite(w) & (w > 0)]
        if w.size < 3:
            return None
        return 1000.0 * self.width_mm / float(np.median(w))

    def add_row(self, rec: dict):
        self._rows.append(rec)
        self._dirty = True

    def stop_live(self):
        self._timer.stop()
        self._redraw()

    def _on_traj_hz(self, value: float):
        """Live-update the trajectory and its speed for a new sampling rate."""
        self.traj_hz = float(value) if value and value > 0 else None
        if self.traj_hz:
            self.lbl_traj.setText(f"every {1000.0 / self.traj_hz:.0f} ms")
        else:
            self.lbl_traj.setText("every frame")
        self._redraw()
        if self._sel is not None:
            # Re-run the region analysis so the speed readout follows the rate
            # rather than going stale against the curve it is drawn on.
            self.selectionAnalyzed.emit(self.analyze_selection())
            self._draw_selection()

    def retheme(self):
        """Rebuild the figure under the current theme mode.

        matplotlib bakes rcParams into artists when they are created, so a
        already-drawn figure keeps its old face, tick and label colours no
        matter what the stylesheet does. The cheapest correct answer is to set
        the new rcParams and rebuild, then redraw the data.
        """
        matplotlib.rcParams.update(matplotlib_rc())
        self.fig.set_facecolor(THEME["plot_bg"])
        self.canvas.setStyleSheet(f"background-color: {THEME['plot_bg']};")
        self._build()
        self._redraw()
        self._draw_selection()

    def reset(self):
        """Empty the panel, for when a different clip is loaded.

        Leaving the previous clip's curves on screen while a new video loads is
        worse than an empty panel: the axes still carry the old time range and
        the old micrometre scale, so the plots look like data about the video
        now in the viewer, and they are not.
        """
        self._timer.stop()
        self._rows = []
        self._home.clear()
        self.clear_selection()
        self._redraw()

    def set_table(self, df, um_per_px: float | None, has_force: bool):
        """Replace live data with the finished, gated and smoothed table."""
        self._timer.stop()
        self.configure(um_per_px, has_force)
        cols = [c for c in ("t", "length_px", "length_strain", "force_mn",
                            "path_length", "confidence", "cx", "cy",
                            "width_px") if c in df.columns]
        self._rows = df[cols].to_dict("records")
        self._home.clear()
        self._redraw()

    # ---- drawing ---------------------------------------------------------

    def _maybe_redraw(self):
        if self._dirty:
            self._redraw()

    def _series(self):
        if not self._rows:
            return None
        t = np.array([r.get("t", np.nan) for r in self._rows], float)
        length_px = np.array([r.get("length_px", np.nan) for r in self._rows], float)
        conf = np.array([r.get("confidence", np.nan) for r in self._rows], float)

        live_k = self._live_um_per_px()
        if live_k and not self.um_per_px:
            # Only a rebuild changes the axis label, so adopt the derived scale
            # through configure() rather than assigning it behind the label.
            self.configure(live_k, self.has_force)
        elif live_k:
            self.um_per_px = live_k
        k = self.um_per_px or 1.0
        length = length_px * k

        if "length_strain" in self._rows[0]:
            strain = np.array([r.get("length_strain", np.nan) for r in self._rows], float)
        else:
            # Live: the resting length is not known yet, so the running median
            # stands in. It converges within a cycle or two and never pretends
            # to be the final number -- the finished table replaces it.
            med = np.nanmedian(length_px) if np.isfinite(length_px).any() else np.nan
            strain = length_px / med if med else np.full_like(length_px, np.nan)

        # With a sampling rate set, the path is rebuilt from the centroids at
        # that rate rather than taken from the pipeline's per-frame cumulative
        # column -- that column was summed at the frame rate and its jitter is
        # already baked in.
        cxr = np.array([r.get("cx", np.nan) for r in self._rows], float) * k
        cyr = np.array([r.get("cy", np.nan) for r in self._rows], float) * k
        if self.traj_hz:
            path = resampled_path(t, cxr, cyr, self.traj_hz)
        elif "path_length" in self._rows[0]:
            path = np.array([r.get("path_length", np.nan) for r in self._rows], float) * (
                1000.0 if self.um_per_px else 1.0)
        else:
            d = np.nan_to_num(np.hypot(np.diff(cxr), np.diff(cyr)), nan=0.0)
            path = np.concatenate([[0.0], np.cumsum(d)])

        # Displacement along each axis, relative to the first tracked position.
        # Net x and y say *where* the robot went; the cumulative path only says
        # how far it traveled, and the two differ whenever it doubles back.
        cx = np.array([r.get("cx", np.nan) for r in self._rows], float) * k
        cy = np.array([r.get("cy", np.nan) for r in self._rows], float) * k
        first = np.flatnonzero(np.isfinite(cx))
        if first.size:
            dx, dy = cx - cx[first[0]], cy - cy[first[0]]
        else:
            dx = dy = np.full_like(cx, np.nan)

        force = None
        if self.has_force:
            # Stored in mN so one column serves both methods; shown in µN, which
            # is the unit this literature quotes and the unit the Cvetkovic model
            # produces natively.
            force = np.array([r.get("force_mn", np.nan)
                              for r in self._rows], float) * 1000.0
        return t, length, strain, force, path, conf * 100.0, dx, dy

    def _redraw(self):
        self._dirty = False
        data = self._series()
        if data is None:
            return
        t, length, strain, force, path, conf, dx, dy = data
        t, (length, strain, force, path, conf, dx, dy) = decimate(
            t, [length, strain, force, path, conf, dx, dy])
        # Zoom is user state: preserve it across a live refresh, or the view
        # would snap back to full range every quarter second while zoomed in.
        keep = {k: (ax.get_xlim(), ax.get_ylim()) for k, ax in self._axes.items()
                if k in self._home}

        col = self.colors
        for key, y, c in (("length", length, col[0]),
                          ("force", force, col[2]),
                          ("path", path, col[0]),
                          ("conf", conf, THEME["ok"])):
            ax = self._axes.get(key)
            if ax is None or y is None:
                continue
            ax.clear()
            if key == "path":
                ax.plot(t, y, lw=1.4, color=c, label="path")
                ax.plot(t, dx, lw=1.1, color=col[1], label="x")
                ax.plot(t, dy, lw=1.1, color=col[2], label="y")
                ax.axhline(0.0, lw=0.7, color=THEME["muted"], alpha=0.6)
                ax.legend(loc="upper left", fontsize=6, framealpha=0.25, ncol=3)
            else:
                ax.plot(t, y, lw=1.3, color=c)
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=7)
            if key == "conf":
                ax.axhline(50.0, ls="--", lw=0.8, color=THEME["muted"])
                ax.set_ylim(0, 102)

        unit = "µm" if self.um_per_px else "px"
        labels = {"length": f"length ({unit})", "force": "force (µN)",
                  "path": f"movement ({unit})", "conf": "fit conf. (%)"}
        for key, ax in self._axes.items():
            ax.set_ylabel(labels.get(key, key), fontsize=8)
        list(self._axes.values())[-1].set_xlabel("time (s)", fontsize=8)

        for key, (xl, yl) in keep.items():
            self._axes[key].set_xlim(xl)
            self._axes[key].set_ylim(yl)
        if self._sel is not None:
            # ax.clear() dropped the span and labels; recompute against the data
            # that is now on screen rather than redrawing stale numbers.
            self._sel_artists = []
            self.analyze_selection()
        else:
            self.canvas.draw_idle()

    # ---- interaction -----------------------------------------------------

    def _remember_home(self, key, ax):
        if key not in self._home:
            self._home[key] = (ax.get_xlim(), ax.get_ylim())

    def _on_scroll(self, event):
        """Scroll zooms time; shift-scroll zooms the value axis.

        Time is shared across panels, so an x zoom applies to all of them --
        comparing force against strain at different time ranges would be
        actively misleading. A y zoom applies only to the panel under the cursor.
        """
        ax = event.inaxes
        if ax is None:
            return
        factor = 0.85 if event.button == "up" else 1 / 0.85
        shift = bool(event.guiEvent and (event.guiEvent.modifiers() & Qt.ShiftModifier))

        if shift:
            key = next((k for k, a in self._axes.items() if a is ax), None)
            if key is None or event.ydata is None:
                return
            self._remember_home(key, ax)
            lo, hi = ax.get_ylim()
            c = event.ydata
            ax.set_ylim(c - (c - lo) * factor, c + (hi - c) * factor)
        else:
            if event.xdata is None:
                return
            for key, a in self._axes.items():
                self._remember_home(key, a)
            lo, hi = ax.get_xlim()
            c = event.xdata
            new = (c - (c - lo) * factor, c + (hi - c) * factor)
            for a in self._axes.values():
                a.set_xlim(new)
        self._throttled_draw()

    # ---- panning and region selection ------------------------------------
    #
    # Right-drag pans. Left-drag marks a stretch of time and measures it. The
    # split matters because the two want opposite defaults: panning should feel
    # free and reversible, while selecting is a deliberate act whose result you
    # then read off.

    def _on_press(self, event):
        if event.dblclick:
            self.clear_selection()
            self.reset_zoom()
            return
        if event.inaxes is None:
            return
        key = next((k for k, a in self._axes.items() if a is event.inaxes), None)
        if key is None:
            return

        if event.button == 3:                       # right: pan
            for k, a in self._axes.items():
                self._remember_home(k, a)
            ax = event.inaxes
            bbox = ax.get_window_extent()
            self._drag = {"key": key, "px": event.x, "py": event.y,
                          "xlim": ax.get_xlim(), "ylim": ax.get_ylim(),
                          "w": max(bbox.width, 1.0), "h": max(bbox.height, 1.0)}
            self.canvas.setCursor(Qt.ClosedHandCursor)
        elif event.button == 1 and event.xdata is not None:
            self._sel_drag = [float(event.xdata), float(event.xdata)]
            self.canvas.setCursor(Qt.SplitHCursor)

    def _on_motion(self, event):
        if self._drag is not None and event.x is not None:
            d = self._drag
            lo0, hi0 = d["xlim"]
            dx = (d["px"] - event.x) * (hi0 - lo0) / d["w"]
            new = (lo0 + dx, hi0 + dx)
            for a in self._axes.values():    # sharex: an absolute set is idempotent
                a.set_xlim(new)
            ax = self._axes[d["key"]]
            if event.inaxes is ax:
                ylo0, yhi0 = d["ylim"]
                dy = (d["py"] - event.y) * (yhi0 - ylo0) / d["h"]
                ax.set_ylim(ylo0 + dy, yhi0 + dy)
            self._throttled_draw()
            return

        if self._sel_drag is not None and event.xdata is not None:
            self._sel_drag[1] = float(event.xdata)
            self._draw_selection(preview=True)

    def _on_release(self, event):
        if self._drag is not None:
            self._drag = None
            self.canvas.setCursor(Qt.ArrowCursor)
            return
        if self._sel_drag is not None:
            a, b = sorted(self._sel_drag)
            self._sel_drag = None
            self.canvas.setCursor(Qt.ArrowCursor)
            # A click is not a zero-width selection; treat it as clearing one.
            span = b - a
            full = abs(np.diff(next(iter(self._axes.values())).get_xlim())[0])
            if span <= full * 0.005:
                self.clear_selection()
                return
            self._sel = (a, b)
            self.analyze_selection()

    # ---- region analysis -------------------------------------------------

    def analyze_selection(self) -> dict:
        """Measure the selected stretch and label each panel with the result."""
        self.stats = {}
        if self._sel is None:
            return self.stats
        data = self._series()
        if data is None:
            return self.stats
        t, length, strain, force, path, conf, dx, dy = data
        t0, t1 = self._sel
        m = np.isfinite(t) & (t >= t0) & (t <= t1)
        n = int(m.sum())
        unit = "µm" if self.um_per_px else "px"
        out: dict = {"t0": float(t0), "t1": float(t1),
                     "duration_s": float(t1 - t0), "n_frames": n,
                     "units": unit}
        if n < 5:
            out["note"] = "too few frames in the selection to measure"
            self.stats = out
            self._draw_selection()
            self.selectionAnalyzed.emit(out)
            return out

        d_len, c_len = average_delta(t[m], length[m])
        out["length_delta"] = d_len
        out["length_cycles"] = c_len

        # Speed from the cumulative path only. x and y are drawn for direction,
        # but a regression through either would report a component, not a speed.
        #
        # Reported in mm/min. These robots cover millimeters over a clip, so
        # micrometers per minute runs to five figures for an ordinary walk.
        div = 1000.0 if self.um_per_px else 1.0
        out["speed_units"] = "mm/min" if self.um_per_px else "px/min"
        out["speed_per_min"] = slope_per_min(t[m], path[m]) / div

        # Net displacement over the same window, as a check on the path slope.
        #
        # Cumulative path only ever increases, so every wobble of the centroid
        # adds distance the robot never traveled and its slope reads as
        # locomotion plus jitter. Net displacement cannot do that -- it is
        # start to finish, in a straight line. On the reference clip the path
        # slope runs 1.2-1.3x the net rate; a much larger ratio means the
        # centroid is wandering rather than the robot walking.
        dxm, dym = dx[m], dy[m]
        good = np.isfinite(dxm) & np.isfinite(dym)
        if good.sum() >= 2:
            xs, ys, ts = dxm[good], dym[good], t[m][good]
            span = float(ts[-1] - ts[0])
            net = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
            out["net_displacement"] = net / div
            out["net_speed_per_min"] = (net / span * 60.0 / div) if span > 0 else float("nan")
            out["wander_ratio"] = (out["speed_per_min"] / out["net_speed_per_min"]
                                   if out["net_speed_per_min"] else float("nan"))

        if force is not None:
            d_f, c_f = average_delta(t[m], force[m])
            out["force_delta_un"] = d_f
            out["force_cycles"] = c_f

        self.stats = out
        self._draw_selection()
        self.selectionAnalyzed.emit(out)
        return out

    def _annotation(self, key: str) -> str:
        st = self.stats
        if not st or "note" in st:
            return ""
        u = st.get("units", "px")
        if key == "length" and np.isfinite(st.get("length_delta", np.nan)):
            return (f"Δ {st['length_delta']:.1f} {u} avg"
                    f"  ({st['length_cycles']} cycles)")
        if key == "force" and np.isfinite(st.get("force_delta_un", np.nan)):
            return (f"Δ {st['force_delta_un']:.1f} µN avg"
                    f"  ({st.get('force_cycles', 0)} cycles)")
        if key == "path" and np.isfinite(st.get("speed_per_min", np.nan)):
            u2 = st.get("speed_units", "mm/min")
            txt = f"path {st['speed_per_min']:.3f} {u2}"
            if np.isfinite(st.get("net_speed_per_min", np.nan)):
                txt += f"   ·   net {st['net_speed_per_min']:.3f} {u2}"
            return txt
        return ""

    def _draw_selection(self, preview: bool = False):
        for a in self._sel_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._sel_artists = []

        span = self._sel_drag if preview else self._sel
        if span is None:
            self._throttled_draw()
            return
        a, b = sorted(span)
        for key, ax in self._axes.items():
            self._sel_artists.append(
                ax.axvspan(a, b, color=THEME["accent"], alpha=0.10, lw=0, zorder=0))
            for x in (a, b):
                self._sel_artists.append(
                    ax.axvline(x, color=THEME["accent"], lw=1.0, alpha=0.75, zorder=1))
            if preview or key == "conf":
                continue                      # confidence is not measured
            text = self._annotation(key)
            if text:
                # These carry the actual result of a region selection -- the
                # average delta and the speed -- so they are the one thing on
                # the panel worth reading from a step back, and 7.5 pt was
                # sized for a caption rather than for a number you act on.
                self._sel_artists.append(ax.annotate(
                    text, xy=(0.985, 0.07), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=11.5, fontweight="bold",
                    color=THEME["text"],
                    bbox=dict(boxstyle="round,pad=0.42", fc=THEME["panel"],
                              ec=THEME["accent"], alpha=0.94, lw=1.2)))
        self._throttled_draw()

    def clear_selection(self):
        self._sel = None
        self.stats = {}
        self._draw_selection()

    def reset_zoom(self):
        self._home.clear()
        self._redraw()

    # ---- export ----------------------------------------------------------

    def axis_ranges(self) -> dict:
        """What is on screen now, as ``{panel: [lo, hi]}`` plus the shared x.

        Reports nothing when the panel holds no data. An empty matplotlib axis
        sits at its default (0, 1), and those defaults are indistinguishable
        from a deliberate zoom once they have been written to a dict -- which is
        exactly how every exported figure ended up squashed to a 0-1 y range.
        The ranges were captured before the run, when there was nothing plotted
        yet, and then faithfully applied to the finished figure.
        """
        if not self._rows:
            return {"zoomed": False}
        out: dict[str, list] = {}
        for key, ax in self._axes.items():
            lo, hi = ax.get_ylim()
            out[key] = [float(lo), float(hi)]
        any_ax = next(iter(self._axes.values()), None)
        if any_ax is not None:
            xl = any_ax.get_xlim()
            out["time"] = [float(xl[0]), float(xl[1])]
        out["zoomed"] = bool(self._home)
        if self._sel is not None:
            out["selection"] = [float(self._sel[0]), float(self._sel[1])]
            out["selection_stats"] = dict(self.stats)
        return out
