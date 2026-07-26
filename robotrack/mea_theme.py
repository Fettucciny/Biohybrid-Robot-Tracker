"""
mea_theme.py  --  Shared UI theme for the MEA Suite programs
============================================================

One small, dependency-light module that gives every MEA program the same look
as the launchers, splash screens, and the MEA Suite selector: the indigo->teal
dark palette, a per-program accent colour, clean fonts, rounded "card" panels,
and matching plot colours.

It supports both GUI toolkits used across the suite:
  * PyQt5 / pyqtgraph  (MEA Explorer, MEA Pattern Creator, MEA Solar Cell)
  * Tkinter / ttk      (MEA Analyzer / MEA-NAP)

Nothing here imports a GUI toolkit at module load; each toolkit is imported
lazily inside the function that needs it, so importing mea_theme is always safe.

--------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------
PyQt app (do this once, right after you create QApplication):

    import mea_theme
    app = QApplication(sys.argv)
    mea_theme.apply_qt(app, "explorer")        # accent by program name
    # for plots:
    mea_theme.style_pyqtgraph()                # dark background + antialias
    cmap = mea_theme.parula()                  # 256-stop parula colormap (hex list)

Tkinter app (after you create the root window):

    import mea_theme
    root = tk.Tk()
    mea_theme.apply_tk(root, "analyzer")

Accent names: "analyzer" (blue), "explorer" (emerald),
              "pattern" (magenta), "solar" (amber).
You can also pass any "#RRGGBB" string as the accent.
--------------------------------------------------------------------------
"""

# ----------------------------------------------------------------------
# Palette  (matches base_tile / splash / selector in the launcher assets)
# ----------------------------------------------------------------------
INDIGO      = "#16213E"   # gradient top
TEAL        = "#0D4A56"   # gradient bottom
WINDOW_TOP  = "#16213E"
WINDOW_BOT  = "#0E2A31"
PANEL       = "#0E1626"   # card / group background
PANEL_HI    = "#141F33"   # hovered / raised panel
FIELD       = "#0B1220"   # input field background
LINE        = "#26374A"   # borders / dividers
LINE_SOFT   = "#1C2A3A"
TEXT        = "#E8EEF4"   # primary text
TEXT_MUTED  = "#9FB4C8"   # secondary text
TEXT_DIM    = "#6B8299"
PLOT_BG     = "#0C121E"
PLOT_FG     = "#C7D2DE"

# Per-program accent colours (identical to the icons / splashes / selector)
ACCENTS = {
    "analyzer": "#3B82F6",   # royal blue
    "explorer": "#34D399",   # emerald
    "pattern":  "#DB7FFF",   # magenta
    "solar":    "#FBBF24",   # amber
    "suite":    "#3B82F6",
}

FONT_FAMILY = "Segoe UI"     # clean Windows default; falls back gracefully
FONT_SIZE   = 10             # points

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def accent(name_or_hex="analyzer"):
    """Resolve an accent name ('explorer') or pass through a '#RRGGBB'."""
    if isinstance(name_or_hex, str) and name_or_hex.startswith("#"):
        return name_or_hex
    return ACCENTS.get(str(name_or_hex).lower(), ACCENTS["analyzer"])

def _rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def tint(h, f):
    """Lighten (f>0) or darken (f<0) a hex colour toward white/black."""
    r, g, b = _rgb(h)
    if f >= 0:
        r, g, b = (int(r+(255-r)*f), int(g+(255-g)*f), int(b+(255-b)*f))
    else:
        r, g, b = (int(r*(1+f)), int(g*(1+f)), int(b*(1+f)))
    return "#%02X%02X%02X" % (r, g, b)

def alpha(h, a):
    """Return an 'rgba(r,g,b,a)' string for Qt stylesheets (a in 0..1)."""
    r, g, b = _rgb(h); return "rgba(%d,%d,%d,%.3f)" % (r, g, b, a)

# ----------------------------------------------------------------------
# PyQt5 / PySide
# ----------------------------------------------------------------------
def qss(acc="analyzer"):
    """Return a full Qt stylesheet string themed with the given accent."""
    A   = accent(acc)
    Ah  = tint(A, 0.18)      # accent hover
    Ad  = tint(A, -0.18)     # accent pressed
    return f"""
* {{
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE}pt;
    color: {TEXT};
}}
QMainWindow, QDialog, QWidget#background {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 {WINDOW_TOP}, stop:1 {WINDOW_BOT});
}}
QWidget {{ background: transparent; }}
QToolTip {{
    background: {PANEL_HI}; color: {TEXT};
    border: 1px solid {LINE}; padding: 4px 6px; border-radius: 4px;
}}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {TEXT_MUTED}; }}

/* --- cards / group boxes --- */
QGroupBox, QFrame#card {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 12px;
    margin-top: 14px;
    padding: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: {TEXT_MUTED}; font-weight: 600;
}}

/* --- buttons --- */
QPushButton {{
    background: {PANEL_HI};
    border: 1px solid {LINE};
    border-radius: 8px;
    padding: 6px 14px;
}}
QPushButton:hover  {{ border-color: {A}; background: {tint(PANEL_HI, 0.06)}; }}
QPushButton:pressed{{ background: {tint(PANEL_HI, -0.15)}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {LINE_SOFT}; }}
QPushButton[primary="true"] {{
    background: {A}; border: 1px solid {A}; color: #0A0E16; font-weight: 600;
}}
QPushButton[primary="true"]:hover   {{ background: {Ah}; border-color: {Ah}; }}
QPushButton[primary="true"]:pressed {{ background: {Ad}; border-color: {Ad}; }}

/* --- fields --- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QAbstractSpinBox {{
    background: {FIELD}; border: 1px solid {LINE}; border-radius: 6px;
    padding: 4px 6px; selection-background-color: {A}; selection-color: #0A0E16;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {A}; }}
QComboBox QAbstractItemView {{
    background: {PANEL_HI}; border: 1px solid {LINE};
    selection-background-color: {A}; selection-color: #0A0E16; outline: none;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}

/* --- checkboxes / radios --- */
QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px; border: 1px solid {LINE};
    background: {FIELD}; border-radius: 4px;
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {A}; border-color: {A};
}}

/* --- tabs --- */
QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_MUTED};
    padding: 7px 14px; border: none; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {A}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

/* --- sliders --- */
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {A}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {tint(A,0.2)}; width: 14px; margin: -6px 0; border-radius: 7px;
}}

/* --- progress --- */
QProgressBar {{
    background: {FIELD}; border: 1px solid {LINE}; border-radius: 6px;
    text-align: center; color: {TEXT};
}}
QProgressBar::chunk {{ background: {A}; border-radius: 6px; }}

/* --- scrollbars --- */
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; min-height: 24px; border-radius: 6px; }}
QScrollBar::handle:vertical:hover {{ background: {tint(LINE,0.25)}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {LINE}; min-width: 24px; border-radius: 6px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

/* --- menus / headers / tables --- */
QMenuBar, QMenu {{ background: {PANEL}; }}
QMenu::item:selected, QMenuBar::item:selected {{ background: {A}; color: #0A0E16; }}
QHeaderView::section {{
    background: {PANEL_HI}; color: {TEXT_MUTED};
    border: none; border-right: 1px solid {LINE}; padding: 4px 6px;
}}
QTableView, QTreeView, QListView {{
    background: {FIELD}; border: 1px solid {LINE};
    gridline-color: {LINE_SOFT}; selection-background-color: {A}; selection-color: #0A0E16;
}}
QStatusBar {{ background: {PANEL}; color: {TEXT_MUTED}; }}
"""

def apply_qt(app, acc="analyzer"):
    """Apply the theme to a QApplication (palette + fonts + stylesheet).

    Binding-agnostic: works with PyQt6 (MEA Analyzer / MEA-NAP), PyQt5 (the
    other suite programs, pre-migration), or the PySide equivalents. PyQt6
    scopes its enums (QPalette.ColorRole.Window) while PyQt5 does not
    (QPalette.Window), so the colour roles are resolved by name."""
    A = accent(acc)
    QtGui = None
    for binding in ("PyQt6", "PyQt5", "PySide6", "PySide2"):
        try:
            mod = __import__(binding, fromlist=["QtGui"])
            QtGui = mod.QtGui
            break
        except Exception:
            continue
    if QtGui is None:                                        # no Qt available
        return A
    # base palette (helps native widgets that ignore the stylesheet)
    pal = QtGui.QPalette()
    C = QtGui.QColor
    _roles = getattr(QtGui.QPalette, "ColorRole", QtGui.QPalette)  # scoped on Qt6

    def role(name):
        return getattr(_roles, name)

    for name, colour in (
            ("Window", WINDOW_TOP), ("Base", FIELD), ("AlternateBase", PANEL),
            ("Text", TEXT), ("WindowText", TEXT), ("Button", PANEL_HI),
            ("ButtonText", TEXT), ("Highlight", A), ("HighlightedText", "#0A0E16"),
            ("ToolTipBase", PANEL_HI), ("ToolTipText", TEXT)):
        try:
            pal.setColor(role(name), C(colour))
        except Exception:
            pass
    app.setPalette(pal)
    try:
        app.setFont(QtGui.QFont(FONT_FAMILY, FONT_SIZE))
    except Exception:
        pass
    app.setStyleSheet(qss(A))
    return A

def style_pyqtgraph(acc="analyzer"):
    """Set global pyqtgraph options to match the theme (call once)."""
    import pyqtgraph as pg
    pg.setConfigOptions(background=PLOT_BG, foreground=PLOT_FG,
                        antialias=True, imageAxisOrder="row-major")
    return accent(acc)

# ----------------------------------------------------------------------
# parula colormap (matches the SpKit / Phase-Lock heatmaps)
# ----------------------------------------------------------------------
_PARULA_ANCHORS = [
    "#352A87", "#2058B0", "#1B7EB4", "#22A784", "#7DD34F", "#D9DA4B", "#F9FB0E",
]
def parula(n=256):
    """Return a list of n hex colours approximating MATLAB 'parula'."""
    a = [_rgb(c) for c in _PARULA_ANCHORS]; m = len(a) - 1
    out = []
    for i in range(n):
        t = i / (n - 1) * m; k = int(t); f = t - k
        if k >= m: k, f = m - 1, 1.0
        r = a[k][0] + (a[k+1][0]-a[k][0])*f
        g = a[k][1] + (a[k+1][1]-a[k][1])*f
        b = a[k][2] + (a[k+1][2]-a[k][2])*f
        out.append("#%02X%02X%02X" % (int(r), int(g), int(b)))
    return out

def parula_lut(n=256):
    """parula as an (n,3) uint8 numpy array for pyqtgraph ColorMap / LUT."""
    import numpy as np
    return np.array([_rgb(c) for c in parula(n)], dtype="uint8")

# ----------------------------------------------------------------------
# Tkinter / ttk
# ----------------------------------------------------------------------
def apply_tk(root, acc="analyzer"):
    """Apply the theme to a Tk root via ttk styles + option database."""
    import tkinter as tk
    from tkinter import ttk
    A = accent(acc)
    root.configure(bg=WINDOW_TOP)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")   # 'clam' is the most stylable built-in theme
    except Exception:
        pass
    style.configure(".", background=WINDOW_TOP, foreground=TEXT,
                    fieldbackground=FIELD, bordercolor=LINE,
                    font=(FONT_FAMILY, FONT_SIZE))
    style.configure("TFrame", background=WINDOW_TOP)
    style.configure("Card.TFrame", background=PANEL, relief="flat")
    style.configure("TLabel", background=WINDOW_TOP, foreground=TEXT)
    style.configure("Muted.TLabel", background=WINDOW_TOP, foreground=TEXT_MUTED)
    style.configure("TButton", background=PANEL_HI, foreground=TEXT,
                    bordercolor=LINE, focuscolor=A, padding=(12, 6), relief="flat")
    style.map("TButton",
              background=[("active", tint(PANEL_HI, 0.06)), ("pressed", tint(PANEL_HI, -0.15))],
              bordercolor=[("active", A)])
    style.configure("Accent.TButton", background=A, foreground="#0A0E16", relief="flat")
    style.map("Accent.TButton", background=[("active", tint(A, 0.18)), ("pressed", tint(A, -0.18))])
    style.configure("TCheckbutton", background=WINDOW_TOP, foreground=TEXT)
    style.map("TCheckbutton", foreground=[("selected", TEXT)])
    style.configure("TRadiobutton", background=WINDOW_TOP, foreground=TEXT)
    style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT,
                    bordercolor=LINE, insertcolor=TEXT)
    style.map("TEntry", bordercolor=[("focus", A)])
    style.configure("TCombobox", fieldbackground=FIELD, foreground=TEXT,
                    background=PANEL_HI, arrowcolor=TEXT_MUTED, bordercolor=LINE)
    style.map("TCombobox", fieldbackground=[("readonly", FIELD)],
              foreground=[("readonly", TEXT)], bordercolor=[("focus", A)])
    # dropdown list colours (Tk listbox behind the Combobox popdown)
    root.option_add("*TCombobox*Listbox.background", PANEL_HI)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", A)
    root.option_add("*TCombobox*Listbox.selectForeground", "#0A0E16")
    style.configure("TNotebook", background=WINDOW_TOP, bordercolor=LINE)
    style.configure("TNotebook.Tab", background=WINDOW_TOP, foreground=TEXT_MUTED, padding=(14, 7))
    style.map("TNotebook.Tab", foreground=[("selected", TEXT)],
              background=[("selected", PANEL)])
    style.configure("Horizontal.TProgressbar", background=A, troughcolor=FIELD, bordercolor=LINE)
    style.configure("TScrollbar", background=PANEL_HI, troughcolor=WINDOW_TOP,
                    bordercolor=WINDOW_TOP, arrowcolor=TEXT_MUTED)
    # plain tk widgets (Menu, Text, Listbox, Canvas) that ttk doesn't cover
    root.option_add("*Menu.background", PANEL)
    root.option_add("*Menu.foreground", TEXT)
    root.option_add("*Menu.activeBackground", A)
    root.option_add("*Menu.activeForeground", "#0A0E16")
    root.option_add("*Text.background", FIELD)
    root.option_add("*Text.foreground", TEXT)
    root.option_add("*Text.insertBackground", TEXT)
    root.option_add("*Listbox.background", FIELD)
    root.option_add("*Listbox.foreground", TEXT)
    return A

# matplotlib rcParams (for the MEA-NAP figures) --------------------------
def matplotlib_rc():
    """Return an rcParams dict for dark, theme-matched matplotlib figures."""
    return {
        "figure.facecolor": PLOT_BG, "axes.facecolor": PLOT_BG,
        "savefig.facecolor": PLOT_BG, "axes.edgecolor": LINE,
        "axes.labelcolor": PLOT_FG, "text.color": TEXT,
        "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
        "grid.color": LINE_SOFT, "font.family": "DejaVu Sans",
    }
