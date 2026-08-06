"""Interactive placement of the CAD outline over the video frame.

Why this is worth a custom widget
---------------------------------
Automatic seeding reads the mask's own moments. That is correct when the robot
is the only moving thing in frame, and wrong the moment it is not: a second
robot in shot, a reflection off the dish, a tether or a bubble that segments as
foreground. The moments then describe the union of all of it, the optimizer
converges confidently onto that, and the numbers look plausible while measuring
the wrong object.

Dragging the real outline onto the real robot removes that ambiguity in about
five seconds, and the placement is a *seed*, not a constraint -- the fit is free
to move away from it. It also resolves the head/tail ambiguity that a
symmetric-ish body has, which matters for the sign of the orientation angle.

Interaction
-----------
The outline is drawn from the same template points the fitter uses, so what is
placed is exactly what will be fitted -- not a bounding box standing in for it.

    drag the body          move
    drag the round handle  rotate and set length (the long axis)
    drag the square handle set width (the short axis)
    wheel                  scale both axes together
    Ctrl + drag            rotate without changing size

Pose convention matches ``register._transform``: scale in the template's own
frame, then rotate, then translate, with local +y as the long axis. So the pose
this widget produces can be handed to the fitter unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QLabel

HANDLE_R = 7.0          # handle radius, widget pixels
GRAB_SLOP = 11.0        # how close the cursor must be to grab one


def transform_points(pts: np.ndarray, pose) -> np.ndarray:
    """Apply (tx, ty, theta, sx, sy) to (M,2) template points -> image pixels."""
    tx, ty, th, sx, sy = [float(v) for v in pose]
    x = pts[:, 0] * sx
    y = pts[:, 1] * sy
    c, s = math.cos(th), math.sin(th)
    return np.stack([c * x - s * y + tx, s * x + c * y + ty], axis=1)


def local_to_image(pose, lx: float, ly: float) -> tuple[float, float]:
    """One point given in template units -> image pixels."""
    tx, ty, th, sx, sy = [float(v) for v in pose]
    x, y = lx * sx, ly * sy
    c, s = math.cos(th), math.sin(th)
    return c * x - s * y + tx, s * x + c * y + ty


class PreviewView(QLabel):
    """The preview surface, with an optional draggable template overlay.

    The overlay is painted by Qt rather than composited into the frame by
    OpenCV. That is what keeps dragging responsive: a redraw is a repaint of a
    cached pixmap, not a decode plus a segmentation plus a fit.
    """

    poseChanged = Signal(object)        # emitted live while dragging
    poseCommitted = Signal(object)      # emitted once, on mouse release
    roiChanged = Signal(object)         # (x, y, w, h) in image px, or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Viewer")
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

        self._frame: QPixmap | None = None
        self._image_size = (0, 0)       # (w, h) of the frame in image pixels

        self.template: np.ndarray | None = None   # (M,2) template units
        self.width_units = 1.0
        self.length_units = 1.0
        self.pose: list[float] | None = None
        self.editable = False

        self._drag: str | None = None   # None | "move" | "length" | "width" | "rotate"
        self._grab_offset = (0.0, 0.0)
        self._color = QColor("#FF5470")

        # Region of interest, in *image* pixels. Kept in image space rather than
        # widget space for the same reason the pose is: the widget resizes and
        # the decode scale changes, and neither should move the region.
        self.roi: tuple[int, int, int, int] | None = None
        self.roi_edit = False
        self._roi_anchor = None

    # ---- content ---------------------------------------------------------

    def set_accent(self, hex_color: str) -> None:
        self._color = QColor(hex_color)

    def set_roi(self, roi, edit: bool | None = None) -> None:
        self.roi = tuple(int(v) for v in roi) if roi else None
        if edit is not None:
            self.roi_edit = bool(edit)
        self.setCursor(Qt.CrossCursor if self.roi_edit else Qt.ArrowCursor)
        self.update()

    def set_frame(self, pixmap: QPixmap | None, image_size: tuple[int, int]) -> None:
        self._frame = pixmap
        self._image_size = image_size
        self.update()

    def set_template(self, points: np.ndarray | None,
                     width_units: float = 1.0, length_units: float = 1.0) -> None:
        self.template = None if points is None else np.asarray(points, float)
        self.width_units = max(float(width_units), 1e-6)
        self.length_units = max(float(length_units), 1e-6)
        self.update()

    def set_pose(self, pose, editable: bool | None = None) -> None:
        self.pose = None if pose is None else [float(v) for v in pose]
        if editable is not None:
            self.editable = bool(editable)
        self.update()

    def default_pose(self) -> list[float] | None:
        """A sensible starting placement: centered, upright, filling ~60% of frame."""
        w, h = self._image_size
        if not (w and h) or self.template is None:
            return None
        s = (0.60 * h) / self.length_units
        return [w / 2.0, h / 2.0, 0.0, s, s]

    # ---- coordinate mapping ---------------------------------------------

    def _fit(self) -> tuple[float, float, float]:
        """(zoom, offset_x, offset_y) mapping image pixels to widget pixels."""
        w, h = self._image_size
        if not (w and h):
            return 1.0, 0.0, 0.0
        z = min(self.width() / w, self.height() / h)
        return z, (self.width() - w * z) / 2.0, (self.height() - h * z) / 2.0

    def _to_widget(self, x: float, y: float) -> QPointF:
        z, ox, oy = self._fit()
        return QPointF(x * z + ox, y * z + oy)

    def _to_image(self, p: QPointF) -> tuple[float, float]:
        z, ox, oy = self._fit()
        if z <= 0:
            return 0.0, 0.0
        return (p.x() - ox) / z, (p.y() - oy) / z

    def _handles(self) -> dict[str, QPointF]:
        if self.pose is None or self.template is None:
            return {}
        return {
            "length": self._to_widget(*local_to_image(self.pose, 0.0, self.length_units / 2)),
            "width": self._to_widget(*local_to_image(self.pose, self.width_units / 2, 0.0)),
        }

    # ---- painting --------------------------------------------------------

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._frame is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        z, ox, oy = self._fit()
        w, h = self._image_size
        p.drawPixmap(QRectF(ox, oy, w * z, h * z), self._frame,
                     QRectF(self._frame.rect()))

        # The region is drawn under the outline and over the frame: everything
        # outside it is dimmed rather than hidden, so you can still see what you
        # excluded and why -- a region that has clipped the robot is obvious at
        # a glance instead of looking like a tracking failure.
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            tl = self._to_widget(rx, ry)
            br = self._to_widget(rx + rw, ry + rh)
            rect = QRectF(tl, br)
            shade = QColor(0, 0, 0, 110)
            full = QRectF(ox, oy, w * z, h * z)
            p.setPen(Qt.NoPen); p.setBrush(shade)
            p.drawRect(QRectF(full.left(), full.top(), full.width(), rect.top() - full.top()))
            p.drawRect(QRectF(full.left(), rect.bottom(), full.width(), full.bottom() - rect.bottom()))
            p.drawRect(QRectF(full.left(), rect.top(), rect.left() - full.left(), rect.height()))
            p.drawRect(QRectF(rect.right(), rect.top(), full.right() - rect.right(), rect.height()))
            pen = QPen(self._color, 2.0, Qt.DashLine)
            p.setPen(pen); p.setBrush(Qt.NoBrush)
            p.drawRect(rect)
        elif self.roi_edit and self._roi_anchor is not None:
            pass

        if self.pose is None or self.template is None:
            return

        pts = transform_points(self.template, self.pose)
        poly = QPolygonF([self._to_widget(x, y) for x, y in pts])

        # Dashed, so a placed outline never reads as a measured result. The
        # fitted outline the analysis produces is drawn solid.
        pen = QPen(self._color, 2.0, Qt.DashLine if self.editable else Qt.SolidLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)

        center = self._to_widget(self.pose[0], self.pose[1])
        p.setBrush(self._color)
        p.setPen(QPen(QColor("#0A0E16"), 1.0))
        p.drawEllipse(center, 3.5, 3.5)

        if not self.editable:
            return

        hs = self._handles()
        p.setPen(QPen(self._color, 1.4))
        p.setBrush(Qt.NoBrush)
        for name, pt in hs.items():
            p.drawLine(center, pt)
        p.setBrush(self._color)
        p.setPen(QPen(QColor("#0A0E16"), 1.2))
        if "length" in hs:
            p.drawEllipse(hs["length"], HANDLE_R, HANDLE_R)
        if "width" in hs:
            r = HANDLE_R - 0.5
            p.drawRect(QRectF(hs["width"].x() - r, hs["width"].y() - r, 2 * r, 2 * r))

    # ---- interaction -----------------------------------------------------

    def _hit(self, pos: QPointF) -> str | None:
        for name, pt in self._handles().items():
            if (pt - pos).manhattanLength() <= GRAB_SLOP * 2:
                d = math.hypot(pt.x() - pos.x(), pt.y() - pos.y())
                if d <= GRAB_SLOP:
                    return name
        if self.pose is not None and self.template is not None:
            pts = transform_points(self.template, self.pose)
            poly = QPolygonF([self._to_widget(x, y) for x, y in pts])
            if poly.containsPoint(pos, Qt.OddEvenFill):
                return "move"
        return None

    def mousePressEvent(self, ev):
        if self.roi_edit and ev.button() == Qt.LeftButton and self._frame is not None:
            self._roi_anchor = self._to_image(ev.position())
            self.roi = None
            self.update()
            ev.accept()
            return
        if not self.editable or self.pose is None or ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        pos = ev.position()
        hit = self._hit(pos)
        if hit is None:
            return super().mousePressEvent(ev)
        if hit == "move" and (ev.modifiers() & Qt.ControlModifier):
            hit = "rotate"
        self._drag = hit
        ix, iy = self._to_image(pos)
        self._grab_offset = (self.pose[0] - ix, self.pose[1] - iy)
        self.setCursor(Qt.ClosedHandCursor if hit == "move" else Qt.CrossCursor)
        ev.accept()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self.roi_edit and self._roi_anchor is not None:
            self.roi = self._rect_from(self._roi_anchor, self._to_image(pos))
            self.update()
            ev.accept()
            return
        if self._drag is None:
            if self.editable and self.pose is not None:
                hit = self._hit(pos)
                self.setCursor({"move": Qt.OpenHandCursor,
                                "length": Qt.CrossCursor,
                                "width": Qt.CrossCursor}.get(hit, Qt.ArrowCursor))
            return super().mouseMoveEvent(ev)

        ix, iy = self._to_image(pos)
        tx, ty, th, sx, sy = self.pose

        if self._drag == "move":
            tx, ty = ix + self._grab_offset[0], iy + self._grab_offset[1]
        else:
            vx, vy = ix - tx, iy - ty
            r = math.hypot(vx, vy)
            if r < 1e-6:
                return
            if self._drag == "length":
                # Local +y maps to (-sin th, cos th); point it along the cursor.
                th = math.atan2(-vx, vy)
                sy = max(r / (self.length_units / 2), 1e-3)
            elif self._drag == "width":
                # Width only: project onto the current local +x axis so grabbing
                # this handle cannot quietly rotate the body as well.
                proj = vx * math.cos(th) + vy * math.sin(th)
                sx = max(abs(proj) / (self.width_units / 2), 1e-3)
            elif self._drag == "rotate":
                th = math.atan2(-vx, vy)

        self.pose = [tx, ty, th, sx, sy]
        self.poseChanged.emit(list(self.pose))
        self.update()
        ev.accept()

    def _rect_from(self, a, b):
        w_img, h_img = self._image_size
        x0, y0 = sorted((a[0], b[0])), sorted((a[1], b[1]))
        x, X = int(max(0, x0[0])), int(min(w_img, x0[1]))
        y, Y = int(max(0, y0[0])), int(min(h_img, y0[1]))
        # A region smaller than this is a mis-click, not an intent.
        if X - x < 16 or Y - y < 16:
            return None
        return (x, y, X - x, Y - y)

    def mouseReleaseEvent(self, ev):
        if self.roi_edit and self._roi_anchor is not None:
            self._roi_anchor = None
            self.roiChanged.emit(self.roi)
            ev.accept()
            return
        if self._drag is not None:
            self._drag = None
            self.setCursor(Qt.ArrowCursor)
            # Committing only on release is what keeps the expensive listener --
            # a full re-fit of the frame -- from firing on every mouse move.
            self.poseCommitted.emit(list(self.pose) if self.pose else None)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        if not self.editable or self.pose is None:
            return super().wheelEvent(ev)
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            return super().wheelEvent(ev)
        f = math.exp(0.06 * steps)
        self.pose[3] *= f
        self.pose[4] *= f
        self.poseChanged.emit(list(self.pose))
        self.poseCommitted.emit(list(self.pose))
        self.update()
        ev.accept()
