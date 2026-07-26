"""Launch splash, showing what start-up is actually doing.

Why a splash is warranted here at all: the frozen bundle carries PyTorch with the
CUDA runtime, Qt and OpenCV, and importing that stack takes several seconds on a
cold filesystem cache. Double-clicking an icon and getting nothing for five
seconds reads as a failed launch, and the usual response is to double-click
again -- which starts a second copy competing for the same GPU.

So the splash reports real stages rather than running a decorative timer. Each
heavy import is a step, and the caption names it: if a launch ever hangs, the
last caption drawn says exactly which dependency it hung on, which is far more
useful than a progress bar that was always going to take 1.5 seconds.

This module imports nothing but PySide6 -- it has to be on screen *before* the
expensive imports begin, so it cannot depend on them.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

# The suite colormap, sampled. Duplicated here rather than imported from
# mea_theme because that module is not free to import at this point.
PARULA = ["#352A87", "#1D6BB2", "#22A784", "#ABD64D", "#F9FB0E"]
TEXT_DIM = "#6B8299"
TEXT_MUTED = "#9FB4C8"


def asset(name: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / name


def _parula_at(t: float) -> QColor:
    t = min(max(t, 0.0), 1.0) * (len(PARULA) - 1)
    i = min(int(t), len(PARULA) - 2)
    f = t - i
    a, b = QColor(PARULA[i]), QColor(PARULA[i + 1])
    return QColor(int(a.red() + (b.red() - a.red()) * f),
                  int(a.green() + (b.green() - a.green()) * f),
                  int(a.blue() + (b.blue() - a.blue()) * f))


class Splash(QSplashScreen):
    """Splash with a captioned parula progress rule."""

    # Geometry of the rule, as fractions of the image, so it stays aligned with
    # the artwork if the artwork is re-rendered at another size.
    RULE = (0.36, 0.855, 0.86, 0.875)
    CAPTION_Y = 0.79

    def __init__(self, total_steps: int = 8):
        pix = QPixmap(str(asset("splash.png")))
        if pix.isNull():
            # Never let a missing asset stop the program opening.
            pix = QPixmap(640, 380)
            pix.fill(QColor("#0E1626"))
        super().__init__(pix, Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._total = max(total_steps, 1)
        self._done = 0
        self._caption = "starting…"

    def step(self, caption: str) -> None:
        self._done = min(self._done + 1, self._total)
        self._caption = caption
        self.repaint()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def drawContents(self, painter: QPainter) -> None:
        w, h = self.pixmap().width(), self.pixmap().height()
        x0, y0, x1, y1 = (self.RULE[0] * w, self.RULE[1] * h,
                          self.RULE[2] * w, self.RULE[3] * h)

        f = QFont()
        f.setPointSizeF(8.5)
        painter.setFont(f)
        painter.setPen(QColor(TEXT_MUTED))
        painter.drawText(QRectF(x0, self.CAPTION_Y * h, (x1 - x0), 16),
                         Qt.AlignLeft | Qt.AlignVCenter, self._caption)

        painter.setPen(Qt.NoPen)
        painter.fillRect(QRectF(x0, y0, x1 - x0, y1 - y0), QColor("#1C2A3A"))
        frac = self._done / self._total
        span = (x1 - x0) * frac
        # Drawn column by column so the bar carries the colormap itself rather
        # than a flat fill: the same visual language as the exported figures.
        steps = max(int(span), 1)
        for i in range(steps):
            painter.fillRect(QRectF(x0 + i, y0, 1.2, y1 - y0),
                             _parula_at(i / max((x1 - x0) - 1, 1)))


def run_with_splash(argv, startup_warning: str | None = None) -> int:
    """Create the application, show the splash, then load everything behind it.

    The QApplication is created here rather than in ``gui.main`` because nothing
    can be drawn before it exists, and drawing early is the entire point.
    """
    app = QApplication(argv)

    from .theme import apply as apply_theme
    apply_theme(app)

    splash = Splash(total_steps=7)
    splash.show()
    app.processEvents()

    splash.step("loading numerical libraries…")
    import numpy  # noqa: F401

    splash.step("loading image processing…")
    import cv2  # noqa: F401

    splash.step("loading PyTorch…")
    import torch  # noqa: F401

    splash.step("checking for a GPU…")
    from .gpu import get_device
    dev = get_device(True)

    splash.step("loading CAD and analysis…")
    import ezdxf  # noqa: F401
    import pandas  # noqa: F401
    import scipy.signal  # noqa: F401

    splash.step(f"ready on {dev.name[:34]}" if dev.cuda else "ready (CPU only)")
    from .gui import MainWindow

    splash.step("building the window…")
    win = MainWindow(startup_warning=startup_warning)
    win.show()
    splash.finish(win)
    return app.exec()
