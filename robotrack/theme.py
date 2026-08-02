"""robotrack's visual identity, built on the shared MEA Suite theme kit.

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
  program is identifiable at a glance across a taskbar of suite windows.
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

from contextlib import contextmanager

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QCursor, QFont
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
    of a piece with the suite and was, on measurement, close to unusable: the
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

class Card(QFrame):
    """Titled panel. ``body`` is a QVBoxLayout callers add rows to."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        # Styled from the application stylesheet, deliberately. A per-widget
        # setStyleSheet wins over the app's, so cards that carried their own
        # background stayed dark navy forever -- a light-mode window with a
        # column of dark cards in it, which is exactly what happened.
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 13)
        outer.setSpacing(9)

        head = QLabel(title.upper())
        head.setObjectName("CardTitle")
        outer.addWidget(head)

        rule = QFrame()
        rule.setObjectName("cardRule")
        rule.setFixedHeight(1)
        outer.addWidget(rule)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add_row(self, label: str, widget: QWidget, spec: dict | None = None) -> QWidget:
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
        return row

    def add_widget(self, w: QWidget) -> None:
        self.body.addWidget(w)


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
