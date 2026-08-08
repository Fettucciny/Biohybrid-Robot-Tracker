"""The application's visual identity, built on a vendored theme kit.

The kit's rule for a new program is that the neutrals, gradient, radii,
typography and plot colormap stay identical across the family, and **only the
accent changes**. This module follows that: it delegates to ``mea_theme`` for
everything shared and adds only what robotrack needs that the kit does not
already define -- the (?) help affordance, status chips, and the preview surface.

The accent
----------
``#FF5470`` -- crimson-rose. Chosen deliberately rather than picked by eye:

* It is the furthest available hue from the four existing accents (53 degrees to
  the nearest, ``solar`` amber; 67 degrees to ``pattern`` magenta), so the
  program is identifiable at a glance in a crowded taskbar.
* Contrast against the on-accent near-black is 6.21:1, above the 4.5:1 AA
  threshold the kit's dark-text-on-accent rule requires, and comfortably above
  ``analyzer`` blue's 5.25:1.
* Thematically it is oxygenated muscle rather than the blues and teals that
  electrophysiology tooling has standardised on -- this program measures
  contraction, not spikes.

One caution that comes with a red accent: red reads as "danger" in most
interfaces, and the primary action button is filled with it. Warnings and errors
therefore use amber and a desaturated state color, never another red, so the
accent never has to compete with an alarm.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

from PySide6.QtCore import Qt, QPoint, QPointF, Signal
from PySide6.QtGui import (QColor, QCursor, QFont, QIcon, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from . import mea_theme as M

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

ACCENT_NAME = "myo"
ACCENT = "#FF5470"          # crimson-rose

# --------------------------------------------------------------------------
# Light mode
#
# The kit is a dark theme and its constants are module-level, which is fine
# until someone wants to work next to a window. Rather than fork the kit or
# string-replace hex values out of generated QSS -- which silently misses every
# derived tint, and there are several -- the light palette is applied by
# swapping those constants for the duration of each call that reads them. The
# kit then generates a correct light stylesheet, palette and rcParams using its
# own logic, including the hover and pressed tints it computes on the fly.
#
# The accent does not change between modes. #FF5470 has enough chroma to read
# on both surfaces, and an app that changes its identity colour when you flip a
# switch looks like two different programs.
# --------------------------------------------------------------------------

LIGHT_SURFACES = {
    "INDIGO":     "#EEF3F8",
    "TEAL":       "#E2ECEF",
    "WINDOW_TOP": "#EEF3F8",
    "WINDOW_BOT": "#E2ECEF",
    "PANEL":      "#FFFFFF",
    "PANEL_HI":   "#F1F5FA",
    "FIELD":      "#FFFFFF",
    "LINE":       "#C4D0DD",
    "LINE_SOFT":  "#E3EAF2",
    "TEXT":       "#16202C",
    "TEXT_MUTED": "#4C5D70",
    "TEXT_DIM":   "#7B8B9C",
    "PLOT_BG":    "#FFFFFF",
    "PLOT_FG":    "#33445A",
}

# Status colors need their own light steps. Emerald #34D399 and amber #FBBF24
# are chosen to sit on a near-black surface; on white they fall to roughly 1.9:1
# and 1.7:1, well under the 3:1 a non-text mark needs, so a "confident fit"
# chip would be a pale smear. These are the darker steps of the same hues.
LIGHT_STATUS = {"ok": "#0E9F6E", "warn": "#B45309"}

MODE = "dark"


@contextmanager
def _kit_palette(mode: str):
    """Run a block with the kit's constants set to ``mode``."""
    if mode != "light":
        yield
        return
    saved = {k: getattr(M, k) for k in LIGHT_SURFACES}
    for k, v in LIGHT_SURFACES.items():
        setattr(M, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(M, k, v)


def set_mode(mode: str) -> str:
    """Switch the token table. Returns the mode actually set."""
    global MODE
    MODE = "light" if str(mode).lower() == "light" else "dark"
    light = MODE == "light"
    # Mutated in place, never rebound: every module did ``from .theme import C``
    # at import time and holds this exact dict.
    C.update({
        "panel":     LIGHT_SURFACES["PANEL"]      if light else M.PANEL,
        "panel_hi":  LIGHT_SURFACES["PANEL_HI"]   if light else M.PANEL_HI,
        "field":     LIGHT_SURFACES["FIELD"]      if light else M.FIELD,
        "line":      LIGHT_SURFACES["LINE"]       if light else M.LINE,
        "line_soft": LIGHT_SURFACES["LINE_SOFT"]  if light else M.LINE_SOFT,
        "text":      LIGHT_SURFACES["TEXT"]       if light else M.TEXT,
        "muted":     LIGHT_SURFACES["TEXT_MUTED"] if light else M.TEXT_MUTED,
        "dim":       LIGHT_SURFACES["TEXT_DIM"]   if light else M.TEXT_DIM,
        "plot_bg":   LIGHT_SURFACES["PLOT_BG"]    if light else M.PLOT_BG,
        "plot_fg":   LIGHT_SURFACES["PLOT_FG"]    if light else M.PLOT_FG,
        "ok":        LIGHT_STATUS["ok"]   if light else M.ACCENTS["explorer"],
        "warn":      LIGHT_STATUS["warn"] if light else M.ACCENTS["solar"],
    })
    return MODE


# Shared tokens, re-exported so app code never hardcodes a hex value.
C = {
    "accent":    ACCENT,
    "accent_hi": M.tint(ACCENT, 0.18),
    "accent_lo": M.tint(ACCENT, -0.18),
    "on_accent": "#0A0E16",
    "panel":     M.PANEL,
    "panel_hi":  M.PANEL_HI,
    "field":     M.FIELD,
    "line":      M.LINE,
    "line_soft": M.LINE_SOFT,
    "text":      M.TEXT,
    "muted":     M.TEXT_MUTED,
    "dim":       M.TEXT_DIM,
    "plot_bg":   M.PLOT_BG,
    "plot_fg":   M.PLOT_FG,
    # State colors. Deliberately not red -- see the module docstring.
    "ok":        M.ACCENTS["explorer"],   # emerald
    "warn":      M.ACCENTS["solar"],      # amber
}


def _extra_qss() -> str:
    """robotrack-only components, in the kit's visual grammar."""
    return f"""
/* ---- header strip ---- */
QFrame#Header {{
    background: {M.PANEL};
    border: none;
    border-bottom: 1px solid {M.LINE};
}}
QLabel#AppName {{
    font-size: 13pt; font-weight: 700; letter-spacing: 0.3px; color: {M.TEXT};
}}
QLabel#AppMark {{ color: {ACCENT}; font-size: 13pt; font-weight: 700; }}
QLabel#Tagline {{ color: {M.TEXT_DIM}; font-size: 9pt; }}

/* ---- status chips ---- */
QLabel#Chip {{
    background: {M.FIELD};
    border: 1px solid {M.LINE};
    border-radius: 10px;
    padding: 3px 10px;
    color: {M.TEXT_MUTED};
    font-size: 9pt;
}}

/* ---- message boxes ----
   Styled explicitly, on every platform, because the generic QDialog rule does
   not reliably reach them on macOS: the dialog keeps a system-drawn light
   background while the global `color` from the kit stays near-white, and the
   message renders white-on-white. What the user sees is an empty box with a
   warning icon in it -- which is exactly the bug this fixes, and it hid the
   text of every error the app tried to report. */
QMessageBox {{
    background: {M.PANEL};
}}
QMessageBox QLabel {{
    color: {M.TEXT};
    background: transparent;
    font-size: 10pt;
}}
QMessageBox QTextEdit {{
    color: {M.TEXT};
    background: {M.FIELD};
    border: 1px solid {M.LINE};
}}
QMessageBox QPushButton {{ min-width: 88px; }}

/* ---- cards ---- */
QFrame#card {{
    background: {M.PANEL};
    border: 1px solid {M.LINE};
    border-radius: 12px;
    margin-top: 0; padding: 0;
}}
QFrame#helpCard {{
    background: {M.PANEL};
    border: 1px solid {ACCENT};
    border-radius: 12px;
    margin-top: 0; padding: 0;
}}
QFrame#cardRule {{ background: {M.LINE}; border: none; }}

/* ---- card titles (we use QFrame#card, not QGroupBox, for tighter control) ---- */
QLabel#CardTitle {{
    color: {M.TEXT_MUTED};
    font-size: 8pt;
    font-weight: 600;
    letter-spacing: 1.3px;
}}
QLabel#CardChevron {{
    color: {M.TEXT_DIM};
    font-size: 9pt;
    min-width: 11px;
}}
QLabel#FieldLabel {{ color: {M.TEXT_MUTED}; }}
QLabel#Hint {{ color: {M.TEXT_DIM}; font-size: 9pt; }}
QLabel#Readout {{
    color: {M.TEXT_MUTED};
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 9pt;
}}

/* ---- the (?) affordance ---- */
QToolButton#Help {{
    background: transparent;
    border: 1px solid {M.LINE};
    border-radius: 7px;
    color: {M.TEXT_DIM};
    font-size: 8pt;
    font-weight: 700;
    padding: 0;
}}
QToolButton#Help:hover {{
    color: {C['on_accent']};
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ---- ghost button (secondary, non-committal actions) ---- */
QPushButton#Ghost {{
    background: transparent;
    border: 1px dashed {M.LINE};
    color: {M.TEXT_MUTED};
}}
QPushButton#Ghost:hover {{ color: {M.TEXT}; border-color: {ACCENT}; }}

/* ---- preview surface: the loudest thing on screen should be the data ---- */
QLabel#Viewer {{
    background: {M.PLOT_BG};
    border: 1px solid {M.LINE};
    border-radius: 12px;
    color: {M.TEXT_DIM};
}}

QTextEdit {{
    background: {M.PLOT_BG};
    border: 1px solid {M.LINE};
    border-radius: 8px;
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 9pt;
    color: {M.TEXT_MUTED};
}}

QSplitter::handle {{ background: transparent; width: 10px; }}
QScrollArea {{ background: transparent; border: none; }}
"""


def stylesheet() -> str:
    with _kit_palette(MODE):
        return M.qss(ACCENT) + _extra_qss()


def apply(app, mode: str | None = None) -> None:
    """Apply palette, fonts and stylesheet. Safe to call again to switch mode."""
    if mode is not None:
        set_mode(mode)
    app.setStyle("Fusion")
    with _kit_palette(MODE):
        M.apply_qt(app, ACCENT)      # shared palette + base stylesheet
    app.setStyleSheet(stylesheet())  # plus robotrack's own components


def matplotlib_rc() -> dict:
    """Themed rcParams for figures, following the current mode."""
    with _kit_palette(MODE):
        return M.matplotlib_rc()


def series_colors() -> list[str]:
    """Line colors for plots — accent, blue, amber, then a muted grey.

    The previous set took two stops off the shared parula colormap, which looked
    of a piece with the kit and was, on measurement, close to unusable: the
    accent against the parula green separated by ΔE 2.0 under deuteranopia, so
    the two curves on the movement panel were the same colour for a red-green
    colourblind reader. Blue and amber against the accent measure 20.7, and the
    same three work on both surfaces, so the modes do not diverge.

    The accent sits slightly outside the validator's lightness band. That is a
    deliberate exception: it is the product's identity colour, fixed elsewhere,
    and every separation check passes with it in place.
    """
    return [ACCENT, "#3B82F6", "#D97706",
            LIGHT_SURFACES["TEXT_MUTED"] if MODE == "light" else M.TEXT_MUTED]


# --------------------------------------------------------------------------
# Help affordance
# --------------------------------------------------------------------------

class HelpPopup(QFrame):
    """Detail panel shown when a (?) badge is clicked.

    A popup rather than an always-visible hint line: guidance for twenty-odd
    parameters would triple the height of the control column and bury the
    controls themselves, which are what people came for.
    """

    MARGIN = 16
    WIDTH = 400

    def __init__(self, spec: dict, parent: QWidget | None = None):
        super().__init__(parent, Qt.Popup)
        self.setObjectName("helpCard")      # accent-bordered variant of a card

        width = spec.get("width", self.WIDTH)
        inner = width - 2 * self.MARGIN

        lay = QVBoxLayout(self)
        lay.setContentsMargins(self.MARGIN, 14, self.MARGIN, 14)
        lay.setSpacing(9)

        def wrapped(html: str, color: str, bold: bool = False) -> QLabel:
            """A word-wrapped label that reports its true height.

            A QLabel with wordWrap does not know its height until it knows its
            width, and calling adjustSize() on the parent asks for height first
            -- which is exactly how these popups ended up clipped. Fixing the
            label width up front makes heightForWidth answerable, so the layout
            can size the popup correctly.
            """
            lab = QLabel(html)
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.RichText)
            lab.setFixedWidth(inner)
            if bold:
                f = QFont(M.FONT_FAMILY, 11); f.setBold(True)
                lab.setFont(f)
            lab.setStyleSheet(f"color: {color}; background: transparent;")
            lab.setMinimumHeight(lab.heightForWidth(inner))
            return lab

        lay.addWidget(wrapped(spec["title"], M.TEXT, bold=True))
        lay.addWidget(wrapped(spec["what"], M.TEXT_MUTED))
        # Entries that are an explanation rather than a dial omit the range and
        # default line: quoting a "default" for a physical constant invites
        # reading it as a recommended value, which it is not.
        if spec.get("range") or spec.get("default"):
            lay.addWidget(wrapped(
                f"<span style='color:{M.TEXT_DIM}'>Range</span> "
                f"<span style='color:{M.TEXT}'>{spec.get('range', '—')}</span>"
                f"<span style='color:{M.TEXT_DIM}'> &nbsp;·&nbsp; Default </span>"
                f"<span style='color:{M.TEXT}'>{spec.get('default', '—')}</span>",
                M.TEXT_MUTED))

        if spec.get("guidance"):
            rule = QFrame()
            rule.setFixedHeight(1)
            rule.setFixedWidth(inner)
            rule.setStyleSheet(f"background: {M.LINE}; border: none;")
            lay.addWidget(rule)
            lay.addWidget(wrapped(
                "".join(f"<div style='margin-bottom:5px'>&bull;&nbsp; {t}</div>"
                        for t in spec["guidance"]), M.TEXT_MUTED))

        self.setFixedWidth(width)
        lay.activate()
        self.setFixedHeight(lay.sizeHint().height())

    def popup_at(self, global_pos: QPoint) -> None:
        self.move(global_pos + QPoint(14, 6))
        self.show()


class HelpBadge(QToolButton):
    """Small circular (?) next to a control. Hover summarises, click explains."""

    def __init__(self, spec: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("Help")
        self.setText("?")
        self.setFixedSize(16, 16)
        self.setCursor(Qt.WhatsThisCursor)
        meta = (f"<br><span style='color:{M.TEXT_DIM}'>Range: {spec['range']} · "
                f"default {spec['default']} · click for detail</span>"
                if (spec.get("range") or spec.get("default")) else
                f"<br><span style='color:{M.TEXT_DIM}'>click for detail</span>")
        self.setToolTip(f"<b>{spec['title']}</b><br>{spec['short']}{meta}")
        self.clicked.connect(self._show)

    def _show(self):
        HelpPopup(self.spec, self).popup_at(QCursor.pos())


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------

def glyph(name: str, color: str | None = None, size: int = 18) -> QIcon:
    """A small vector icon, painted rather than looked up in a font.

    Unicode symbols were the obvious route and are the wrong one here: a sun, a
    moon and a pair of sliders are all outside the coverage of several stock
    Windows UI fonts, and the fallback is a box or -- worse -- a colour emoji at
    the wrong weight. These are twenty lines of QPainter, look identical on both
    platforms, and take the theme's own colour so they stay legible when the
    window switches between light and dark.

    Drawn on a 100x100 canvas and scaled, so the same code gives a crisp icon at
    any device pixel ratio.
    """
    col = QColor(color or M.TEXT)
    px = QPixmap(size * 4, size * 4)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(px.width() / 100.0, px.height() / 100.0)
    pen = QPen(col, 8.5)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    if name == "sun":                       # switch to light
        p.drawEllipse(QPointF(50, 50), 20, 20)
        for k in range(8):
            a = math.radians(k * 45)
            c, s = math.cos(a), math.sin(a)
            p.drawLine(QPointF(50 + 30 * c, 50 + 30 * s),
                       QPointF(50 + 40 * c, 50 + 40 * s))
    elif name == "moon":                    # switch to dark
        path = QPainterPath()
        path.addEllipse(QPointF(50, 50), 32, 32)
        bite = QPainterPath()
        bite.addEllipse(QPointF(70, 34), 30, 30)
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawPath(path.subtracted(bite))
    elif name == "sliders":                 # advanced: every control exposed
        for y, knob in ((28, 66), (50, 38), (72, 58)):
            p.drawLine(QPointF(14, y), QPointF(86, y))
            p.setBrush(col)
            p.drawEllipse(QPointF(knob, y), 7.5, 7.5)
            p.setBrush(Qt.NoBrush)
    elif name == "list":                    # simple: the short list
        for y in (30, 50, 70):
            p.drawLine(QPointF(22, y), QPointF(84, y))
            p.setBrush(col)
            p.drawEllipse(QPointF(14, y), 3.6, 3.6)
            p.setBrush(Qt.NoBrush)
    elif name == "download":                # update
        p.drawLine(QPointF(50, 14), QPointF(50, 60))
        p.drawPolyline(QPolygonF([QPointF(30, 42), QPointF(50, 62), QPointF(70, 42)]))
        p.drawPolyline(QPolygonF([QPointF(18, 74), QPointF(18, 86), QPointF(82, 86),
                                  QPointF(82, 74)]))
    p.end()
    return QIcon(px)


class Card(QFrame):
    """Titled, collapsible panel. ``body`` is a QVBoxLayout callers add rows to.

    Two independent ways a row can be hidden, which are deliberately not the
    same mechanism:

    **Collapsed** is the user's choice about *this* card, taken by clicking its
    title, and it is remembered between sessions. It hides everything.

    **Advanced** is a property of the row itself, declared where the row is
    built. A row marked advanced is one you would change while developing a
    protocol and never touch again while running it. Simple mode hides those
    everywhere at once, so someone handed the program can see the six controls
    that matter rather than the forty that exist.

    A card whose rows are *all* advanced disappears entirely in simple mode --
    an empty titled box is worse than no box.
    """

    toggled = Signal(str, bool)         # (key, collapsed)

    def __init__(self, title: str, parent: QWidget | None = None,
                 key: str | None = None, collapsible: bool = True,
                 compact: bool = False, advanced: bool = False):
        super().__init__(parent)
        # Styled from the application stylesheet, deliberately. A per-widget
        # setStyleSheet wins over the app's, so cards that carried their own
        # background stayed dark navy forever -- a light-mode window with a
        # column of dark cards in it, which is exactly what happened.
        self.setObjectName("card")
        self.key = key or title.lower().replace(" ", "_")
        self.title = title
        self.card_advanced = bool(advanced)
        self._collapsible = bool(collapsible)
        self._collapsed = False
        self._advanced_rows: list[QWidget] = []
        self._rows: list[QWidget] = []

        outer = QVBoxLayout(self)
        pad = 9 if compact else 14
        outer.setContentsMargins(pad + 5, pad - 2, pad + 5, pad - 1)
        outer.setSpacing(6 if compact else 9)

        head = QWidget()
        hh = QHBoxLayout(head)
        hh.setContentsMargins(0, 0, 0, 0)
        hh.setSpacing(6)
        # The chevron is a label rather than a button so the whole header strip
        # is one click target. A 9x9 arrow is a miserable thing to have to hit.
        self._chevron = QLabel("▾")
        self._chevron.setObjectName("CardChevron")
        self._label = QLabel(title.upper())
        self._label.setObjectName("CardTitle")
        hh.addWidget(self._chevron)
        hh.addWidget(self._label)
        hh.addStretch(1)
        self._head = head
        outer.addWidget(head)
        if self._collapsible:
            head.setCursor(QCursor(Qt.PointingHandCursor))
            head.mouseReleaseEvent = self._on_head_click
            head.setToolTip("Click to collapse or expand this section")
        else:
            self._chevron.setVisible(False)

        self._rule = QFrame()
        self._rule.setObjectName("cardRule")
        self._rule.setFixedHeight(1)
        outer.addWidget(self._rule)

        # The rows live in a host widget rather than straight in the layout, so
        # collapsing is one setVisible instead of a walk over every child --
        # and so a hidden row's own visibility is preserved across a collapse.
        self._host = QWidget()
        self.body = QVBoxLayout(self._host)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(6 if compact else 8)
        outer.addWidget(self._host)

    # ---- building --------------------------------------------------------

    def add_row(self, label: str, widget: QWidget, spec: dict | None = None,
                advanced: bool = False) -> QWidget:
        """One labeled control with an optional (?) badge."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(7)

        lab = QLabel(label)
        lab.setObjectName("FieldLabel")
        h.addWidget(lab)
        if spec:
            h.addWidget(HelpBadge(spec))
        h.addStretch(1)
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        widget.setMinimumWidth(128)
        h.addWidget(widget)
        self.body.addWidget(row)
        self._register(row, advanced)
        return row

    def add_widget(self, w: QWidget, advanced: bool = False) -> None:
        self.body.addWidget(w)
        self._register(w, advanced)

    def _register(self, w: QWidget, advanced: bool) -> None:
        self._rows.append(w)
        if advanced:
            self._advanced_rows.append(w)

    # ---- state -----------------------------------------------------------

    def _on_head_click(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self.set_collapsed(not self._collapsed)
            self.toggled.emit(self.key, self._collapsed)

    def set_collapsed(self, on: bool) -> None:
        if not self._collapsible:
            on = False
        self._collapsed = bool(on)
        self._host.setVisible(not self._collapsed)
        self._rule.setVisible(not self._collapsed)
        self._chevron.setText("▸" if self._collapsed else "▾")

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_show_advanced(self, on: bool) -> bool:
        """Apply simple/advanced mode. Returns whether the card has anything left.

        Rows the caller has hidden for its own reasons -- a DXF outline chooser
        with one candidate, a manual-threshold box in auto mode -- must stay
        hidden, so switching to advanced only *un*hides rows this method hid.

        ``isHidden`` rather than ``not isVisible``: the two differ exactly while
        the window has not been shown yet, which is when this first runs. Every
        widget is invisible at that point, so the ``isVisible`` spelling
        concluded there was nothing to hide and simple mode came up showing
        everything until the user toggled it twice.
        """
        for w in self._advanced_rows:
            if on:
                if w.property("_hidden_by_mode"):
                    w.setProperty("_hidden_by_mode", False)
                    w.setVisible(True)
            elif not w.isHidden():
                w.setProperty("_hidden_by_mode", True)
                w.setVisible(False)
        if self.card_advanced and not on:
            return False
        # Every row advanced and advanced is off: nothing to show.
        return on or len(self._advanced_rows) < len(self._rows) or not self._rows


def style_chip(lab: QLabel, kind: str = "") -> None:
    """Recolor a status chip. ``kind`` is '', 'ok' or 'warn' -- never red.

    Reads its surfaces from ``C`` rather than from the kit's module constants,
    because this is called at runtime -- on every device probe and every clip
    load -- rather than while the stylesheet is being generated. Outside that
    context the kit's constants are always the dark ones, which is how a light
    window ended up with black status pills in its header.
    """
    col = {"ok": C["ok"], "warn": C["warn"]}.get(kind, C["muted"])
    border = col if kind else C["line"]
    lab.setStyleSheet(
        f"background:{C['field']}; border:1px solid {border}; border-radius:10px;"
        f" padding:3px 10px; font-size:9pt; color:{col};")
