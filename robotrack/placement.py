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
from PySide6.QtGui import (QColor, QCursor, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import QLabel

HANDLE_R = 7.0          # handle radius, widget pixels
GRAB_SLOP = 11.0        # how close the cursor must be to grab one
ROI_HANDLE = 4.5        # half-width of a region handle square, widget pixels
ROI_MIN_PX = 16.0       # a region smaller than this is a mis-click, not an intent
ROI_ROT_ARM = 26.0      # how far the rotate grip sits above the top edge

# Which edges each region handle moves, in the region's own frame: (u, v) with
# -1 meaning the low edge, +1 the high edge and 0 "leave this axis alone".
ROI_HANDLES = {
    "nw": (-1, -1), "n": (0, -1), "ne": (1, -1), "e": (1, 0),
    "se": (1, 1), "s": (0, 1), "sw": (-1, 1), "w": (-1, 0),
}
ROI_CURSORS = {
    "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
    "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
    "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
    "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
    "roi_rot": Qt.CrossCursor, "roi_body": Qt.SizeAllCursor,
}
# Every grabbable thing on the picture, region and outline alike, so hovering
# answers "what would a click here do" in one lookup.
HOVER_CURSORS = dict(ROI_CURSORS)
HOVER_CURSORS.update({"move": Qt.OpenHandCursor, "length": Qt.CrossCursor,
                      "width": Qt.CrossCursor, "roi_new": Qt.CrossCursor})


def transform_points(pts: np.ndarray, pose) -> np.ndarray:
    """Apply (tx, ty, theta, sx, sy) to (M,2) template points -> image pixels."""
    tx, ty, th, sx, sy = [float(v) for v in pose]
    x = pts[:, 0] * sx
    y = pts[:, 1] * sy
    c, s = math.cos(th), math.sin(th)
    return np.stack([c * x - s * y + tx, s * x + c * y + ty], axis=1)


def _near(a: QPointF, b: QPointF, slop: float) -> bool:
    return math.hypot(a.x() - b.x(), a.y() - b.y()) <= slop


def _norm_roi(roi) -> tuple | None:
    """Any accepted region spelling -> (x, y, w, h, angle_deg), or None.

    Four values means upright, which is what every build before 0.17 wrote and
    what drawing a fresh region still produces. Widening the tuple rather than
    introducing a second field keeps one region concept in the settings file,
    the config file, the run metadata and the preview.
    """
    if not roi:
        return None
    v = [float(t) for t in roi]
    if len(v) < 4 or v[2] < 1 or v[3] < 1:
        return None
    ang = v[4] if len(v) > 4 else 0.0
    return (int(round(v[0])), int(round(v[1])), int(round(v[2])), int(round(v[3])),
            round(float(ang), 2))


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

        # Region of interest, in *image* pixels, as (x, y, w, h, angle_deg).
        # Kept in image space rather than widget space for the same reason the
        # pose is: the widget resizes and the decode scale changes, and neither
        # should move the region. x, y are the corner of the *unrotated* box;
        # the angle turns it about its own centre.
        self.roi: tuple | None = None
        self.roi_edit = False
        self._roi_anchor = None
        self._roi_drag: str | None = None
        self._roi_prev: tuple | None = None
        self._hover: str | None = None   # "roi" | "outline" | None
        self._roi_grab = (0.0, 0.0)     # cursor offset from centre, at grab time
        self._roi_start: tuple | None = None

    # ---- content ---------------------------------------------------------

    def set_accent(self, hex_color: str) -> None:
        self._color = QColor(hex_color)

    def set_roi(self, roi, edit: bool | None = None) -> None:
        self.roi = _norm_roi(roi)
        if edit is not None:
            self.roi_edit = bool(edit)
        self.setCursor(Qt.CrossCursor if (self.roi_edit and self.roi is None)
                       else Qt.ArrowCursor)
        self.update()

    # ---- region geometry -------------------------------------------------

    def _roi_center(self) -> tuple[float, float, float, float, float]:
        """(cx, cy, half-width, half-height, angle in radians) in image px."""
        x, y, w, h, a = self.roi
        return x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0, math.radians(a)

    def _roi_point(self, u: float, v: float) -> QPointF:
        """A point on the region, in widget pixels. (u, v) are in [-1, 1]."""
        cx, cy, hw, hh, r = self._roi_center()
        dx, dy = u * hw, v * hh
        c, s = math.cos(r), math.sin(r)
        return self._to_widget(cx + c * dx - s * dy, cy + s * dx + c * dy)

    def _roi_polygon(self) -> QPolygonF:
        return QPolygonF([self._roi_point(u, v)
                          for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def _roi_rot_grip(self) -> QPointF:
        """Where the rotation grip sits: out along the region's own -y axis.

        Placed in *widget* pixels beyond the top edge rather than in image
        pixels, so it stays a comfortable distance from the edge whatever the
        zoom -- an image-space offset vanishes into the border on a small
        region and floats absurdly far away on a large one.
        """
        top = self._roi_point(0, -1)
        cen = self._roi_point(0, 0)
        vx, vy = top.x() - cen.x(), top.y() - cen.y()
        n = math.hypot(vx, vy)
        if n < 1e-6:
            return QPointF(top.x(), top.y() - ROI_ROT_ARM)
        return QPointF(top.x() + vx / n * ROI_ROT_ARM, top.y() + vy / n * ROI_ROT_ARM)

    def _set_hover(self, hit: str | None) -> None:
        """Remember which overlay the cursor is over, and repaint if it moved.

        This is the whole answer to "which one am I about to edit?". With two
        draggable overlays on one picture, a cursor shape alone is not enough:
        a move cursor over the region and a move cursor over the outline look
        the same. Brightening the one under the pointer and dimming the other
        makes it unambiguous before the click, which is when it matters.
        """
        which = None
        if hit in ("move", "length", "width", "rotate"):
            which = "outline"
        elif hit is not None and hit != "roi_new":
            which = "roi"
        if which != self._hover:
            self._hover = which
            self.update()

    def _roi_hit(self, pos: QPointF) -> str | None:
        """Which part of the region is under the cursor, if any."""
        if self.roi is None:
            return None
        if _near(self._roi_rot_grip(), pos, GRAB_SLOP):
            return "roi_rot"
        for name, (u, v) in ROI_HANDLES.items():
            if _near(self._roi_point(u, v), pos, GRAB_SLOP):
                return name
        if self._roi_polygon().containsPoint(pos, Qt.OddEvenFill):
            return "roi_body"
        return None

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
        # When both overlays are editable, the one under the cursor is drawn at
        # full strength and the other is faded. Two identical-looking dashed
        # outlines with no way to tell which responds to a drag was the whole
        # complaint; this answers it before the click rather than after.
        both = self.roi_edit and self.editable and self.pose is not None
        roi_col = QColor(self._color)
        tpl_col = QColor(self._color)
        if both and self._hover == "outline":
            roi_col.setAlpha(90)
        elif both and self._hover == "roi":
            tpl_col.setAlpha(90)

        if self.roi is not None:
            poly = self._roi_polygon()
            full = QRectF(ox, oy, w * z, h * z)
            # Subtracting the polygon from the frame handles the rotated case in
            # one fill, where the old four-rectangle border only ever worked for
            # an upright box.
            outside = QPainterPath()
            outside.addRect(full)
            inside = QPainterPath()
            inside.addPolygon(poly)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 110))
            p.drawPath(outside.subtracted(inside))

            p.setPen(QPen(roi_col, 2.0, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(poly)

            if self.roi_edit:
                grip = self._roi_rot_grip()
                top = self._roi_point(0, -1)
                p.setPen(QPen(roi_col, 1.4))
                p.drawLine(top, grip)
                p.setBrush(roi_col)
                p.setPen(QPen(QColor("#0A0E16"), 1.2))
                p.drawEllipse(grip, HANDLE_R - 1.0, HANDLE_R - 1.0)
                for u, v in ROI_HANDLES.values():
                    c = self._roi_point(u, v)
                    p.drawRect(QRectF(c.x() - ROI_HANDLE, c.y() - ROI_HANDLE,
                                      2 * ROI_HANDLE, 2 * ROI_HANDLE))

        if self.pose is None or self.template is None:
            return

        pts = transform_points(self.template, self.pose)
        poly = QPolygonF([self._to_widget(x, y) for x, y in pts])

        # Dashed, so a placed outline never reads as a measured result. The
        # fitted outline the analysis produces is drawn solid.
        pen = QPen(tpl_col, 2.0, Qt.DashLine if self.editable else Qt.SolidLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)

        center = self._to_widget(self.pose[0], self.pose[1])
        p.setBrush(tpl_col)
        p.setPen(QPen(QColor("#0A0E16"), 1.0))
        p.drawEllipse(center, 3.5, 3.5)

        if not self.editable:
            return

        hs = self._handles()
        p.setPen(QPen(tpl_col, 1.4))
        p.setBrush(Qt.NoBrush)
        for name, pt in hs.items():
            p.drawLine(center, pt)
        p.setBrush(tpl_col)
        p.setPen(QPen(QColor("#0A0E16"), 1.2))
        if "length" in hs:
            p.drawEllipse(hs["length"], HANDLE_R, HANDLE_R)
        if "width" in hs:
            r = HANDLE_R - 0.5
            p.drawRect(QRectF(hs["width"].x() - r, hs["width"].y() - r, 2 * r, 2 * r))

    # ---- interaction -----------------------------------------------------

    def _hit(self, pos: QPointF) -> str | None:
        """Which part of the *outline* is under the cursor."""
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

    def hit_test(self, pos: QPointF) -> str | None:
        """What a click here would grab, with both overlays live.

        The region and the hand-placed outline occupy the same picture and the
        outline is normally *inside* the region, so with both editable every
        click landed on the region and the outline could not be touched at all.

        Priority is by specificity, not by layer: handles before bodies,
        because a handle is a small deliberate target and a body is most of the
        frame; then the outline's body before the region's, because the outline
        is the smaller and more precisely placed of the two and the region is
        still reachable everywhere around it. Drawing a fresh region is last,
        so it can never pre-empt an adjustment to something already there.
        """
        outline_live = self.editable and self.pose is not None
        hit = self._hit(pos) if outline_live else None
        if hit is not None and hit != "move":
            return hit                                   # outline handle
        roi_hit = self._roi_hit(pos) if self.roi_edit else None
        if roi_hit is not None and roi_hit != "roi_body":
            return roi_hit                               # region handle or grip
        if hit == "move":
            return hit                                   # inside the outline
        if roi_hit is not None:
            return roi_hit                               # inside the region
        return "roi_new" if (self.roi_edit and self._frame is not None) else None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        hit = self.hit_test(ev.position())
        if hit is None:
            return super().mousePressEvent(ev)

        if hit in ROI_CURSORS or hit in ROI_HANDLES:
            self._roi_drag = hit
            self._roi_start = self.roi
            cx, cy, _, _, _ = self._roi_center()
            ix, iy = self._to_image(ev.position())
            self._roi_grab = (cx - ix, cy - iy)
            self.setCursor(QCursor(ROI_CURSORS.get(hit, Qt.CrossCursor)))
            ev.accept()
            return

        if hit == "roi_new":
            self._roi_anchor = self._to_image(ev.position())
            # Keep the old region until a new one is actually big enough to be
            # a region. Clearing on press meant a single stray click anywhere on
            # the frame silently destroyed a carefully placed region -- and a
            # stray click is exactly what happens when you are reaching for the
            # outline underneath it.
            self._roi_prev = self.roi
            self.roi = None
            self.update()
            ev.accept()
            return

        pos = ev.position()
        if hit == "move" and (ev.modifiers() & Qt.ControlModifier):
            hit = "rotate"
        self._drag = hit
        ix, iy = self._to_image(pos)
        self._grab_offset = (self.pose[0] - ix, self.pose[1] - iy)
        self.setCursor(Qt.ClosedHandCursor if hit == "move" else Qt.CrossCursor)
        ev.accept()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._roi_drag is not None:
            self._drag_roi(pos, ev.modifiers())
            self.update()
            ev.accept()
            return
        if self.roi_edit and self._roi_anchor is not None:
            self.roi = self._rect_from(self._roi_anchor, self._to_image(pos))
            self.update()
            ev.accept()
            return
        if self._drag is None:
            # One hover query for both overlays, so the cursor and the
            # highlight always agree with what a click would actually grab.
            hit = self.hit_test(pos)
            self._set_hover(hit)
            self.setCursor(QCursor(HOVER_CURSORS.get(hit, Qt.ArrowCursor)))
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

    def _drag_roi(self, pos: QPointF, mods) -> None:
        """Move, resize or rotate the region, in the region's own frame."""
        if self._roi_start is None:
            return
        x, y, w, h, a = self._roi_start
        cx, cy = x + w / 2.0, y + h / 2.0
        hw, hh = w / 2.0, h / 2.0
        r = math.radians(a)
        c, s = math.cos(r), math.sin(r)
        ix, iy = self._to_image(pos)

        if self._roi_drag == "roi_body":
            cx, cy = ix + self._roi_grab[0], iy + self._roi_grab[1]

        elif self._roi_drag == "roi_rot":
            # The grip starts on the region's -y axis, so the pointer angle
            # measured from there is the new rotation directly.
            ang = math.degrees(math.atan2(iy - cy, ix - cx)) + 90.0
            if mods & Qt.ShiftModifier:
                ang = round(ang / 15.0) * 15.0
            a = round((ang + 180.0) % 360.0 - 180.0, 2)

        else:
            u, v = ROI_HANDLES[self._roi_drag]
            # Cursor in the region's own axes, relative to its centre.
            dx, dy = ix - cx, iy - cy
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            mcx = mcy = 0.0
            if u:
                opp = -u * hw                      # the edge that stays put
                hw = max(abs(lx - opp) / 2.0, ROI_MIN_PX / 2.0)
                mcx = (lx + opp) / 2.0
            if v:
                opp = -v * hh
                hh = max(abs(ly - opp) / 2.0, ROI_MIN_PX / 2.0)
                mcy = (ly + opp) / 2.0
            # The centre moves in image space by the local shift, rotated back.
            cx += c * mcx - s * mcy
            cy += s * mcx + c * mcy
            w, h = 2 * hw, 2 * hh

        self.roi = _norm_roi((cx - w / 2.0, cy - h / 2.0, w, h, a))

    def _rect_from(self, a, b):
        w_img, h_img = self._image_size
        x0, y0 = sorted((a[0], b[0])), sorted((a[1], b[1]))
        x, X = int(max(0, x0[0])), int(min(w_img, x0[1]))
        y, Y = int(max(0, y0[0])), int(min(h_img, y0[1]))
        # A region smaller than this is a mis-click, not an intent.
        if X - x < ROI_MIN_PX or Y - y < ROI_MIN_PX:
            return None
        return (x, y, X - x, Y - y, 0.0)

    def mouseReleaseEvent(self, ev):
        if self._roi_drag is not None:
            self._roi_drag = None
            self._roi_start = None
            self.setCursor(Qt.CrossCursor)
            self.roiChanged.emit(self.roi)
            ev.accept()
            return
        if self.roi_edit and self._roi_anchor is not None:
            self._roi_anchor = None
            if self.roi is None and self._roi_prev is not None:
                self.roi = self._roi_prev      # too small to be a drag: a click
                self.update()
            self._roi_prev = None
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
