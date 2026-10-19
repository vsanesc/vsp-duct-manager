# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The 2D duct editors: SIDE (x-z) and TOP (x-y) views of the centrelines, and the area law A(s).

Like the section editor, the mouse logic is plain methods (``press`` / ``move`` / ``release`` / ``double_click`` /
``zoom``) taking a position, so it can be driven without synthesising Qt events. Every edit goes through
``explore_model.InletModel`` (one shared undo history); the canvases hold only view state.

Centrelines: drag the junction (side: x and z, top: x only, it stays on y = 0) and the CAD-style tangent handles:
the inlet handle slides along the inlet normal (length only), the junction handles keep the tangent in the XZ
plane (side: angle and length, top: length only), the face handle slides along the engine axis. Double-click a
handle to return it to its default.

Area law: drag the interior control points of the branch and trunk curves and the junction area (the right end of
the branch curve and the left end of the trunk curve are the same value); double-click a curve to add a point,
right-click a point to remove it.
"""
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ductmgr import areacurve
from ductmgr.explore_model import InletModel
from ductmgr.viz import ACCENT, ERROR_COLOR

CANVAS_BACKGROUND = "#101316"
GRID_COLOR, AXIS_COLOR, TEXT_COLOR = "#1E2329", "#2C333B", "#9AA4AD"
FUSELAGE_FILL, FUSELAGE_LINE = "#1B2A36", "#6f9bb8"
CONTEXT_LINE, ENGINE_LINE, FACE_LINE = "#4A525A", "#aab4bd", "#e5534b"
OPENING_COLOR, CENTRE_COLOR, HANDLE_COLOR = "#f0a030", "#3fb950", "#58a6ff"
MIRROR_COLOR = "#2B6B5A"
KEY_COLOR, POINT_COLOR, BAD_COLOR = "#d29922", "#E6EDF3", "#e5534b"
HIT_PX = 10  # hit radius of a handle / control point
MARKER_PX = 5
MIN_SCALE, MAX_SCALE = 5.0, 50000.0
ZOOM_STEP = 1.15
FIT_MARGIN = 0.12
ENVELOPE_BINS = 160
GRID_MIN_PX = 60
CURVE_SAMPLES = 120
PLOT_MARGIN = (46, 14, 14, 30)  # left, top, right, bottom in pixels
PLOT_GAP = 26
Y_PADDING = 0.25  # of the data range, above and below the curves


def nice_step(minimum: float) -> float:
    exp = math.floor(math.log10(minimum))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * 10.0**exp >= minimum:
            return m * 10.0**exp
    return 10.0 ** (exp + 1)


def envelope(points: np.ndarray, across: int, bins: int = ENVELOPE_BINS) -> np.ndarray:
    """Closed outline of the projection of ``points`` onto (x, axis ``across``): per x bin the min and max."""
    if len(points) == 0:
        return np.zeros((0, 2))
    x, v = points[:, 0], points[:, across]
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    upper, lower = [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            xc = 0.5 * (edges[b] + edges[b + 1])
            upper.append((xc, v[m].max()))
            lower.append((xc, v[m].min()))
    if not upper:
        return np.zeros((0, 2))
    return np.vstack([np.array(upper), np.array(lower)[::-1]])


class Canvas2D(QtWidgets.QWidget):
    """Pan / zoom plumbing shared by the 2D tabs."""

    def __init__(self, model: InletModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.scale = 100.0
        self.centre = np.zeros(2)
        self._pan: tuple[QtCore.QPointF, np.ndarray] | None = None
        self._drag = None
        self.setMinimumSize(300, 150)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # view transform (world h to the right, v up)
    def to_screen(self, hv) -> QtCore.QPointF:
        return QtCore.QPointF(self.width() / 2 + (hv[0] - self.centre[0]) * self.scale,
                              self.height() / 2 - (hv[1] - self.centre[1]) * self.scale)

    def to_world(self, pos) -> np.ndarray:
        return np.array([(pos.x() - self.width() / 2) / self.scale + self.centre[0],
                         -(pos.y() - self.height() / 2) / self.scale + self.centre[1]])

    def zoom(self, pos, notches: float) -> None:
        before = self.to_world(pos)
        self.scale = float(np.clip(self.scale * ZOOM_STEP**notches, MIN_SCALE, MAX_SCALE))
        self.centre = self.centre + (before - self.to_world(pos))
        self.update()

    def pan_start(self, pos) -> None:
        self._pan = (pos, self.centre.copy())

    def pan_move(self, pos) -> bool:
        if self._pan is None:
            return False
        start, centre0 = self._pan
        self.centre = centre0 - np.array([pos.x() - start.x(), -(pos.y() - start.y())]) / self.scale
        self.update()
        return True

    @staticmethod
    def _pen(color, width=1.0, style=QtCore.Qt.PenStyle.SolidLine):
        pen = QtGui.QPen(QtGui.QColor(color))
        pen.setWidthF(width)
        pen.setStyle(style)
        pen.setCosmetic(True)
        return pen

    def _polyline(self, painter, pts, close=False):
        path = QtGui.QPainterPath()
        for k, p in enumerate(pts):
            q = self.to_screen(p)
            path.moveTo(q) if k == 0 else path.lineTo(q)
        if close:
            path.closeSubpath()
        painter.drawPath(path)

    def wheelEvent(self, e):
        self.zoom(e.position(), e.angleDelta().y() / 120.0)

    def mouseMoveEvent(self, e):
        self.move(e.position())

    def mouseReleaseEvent(self, e):
        self.release(e.position())

    def mousePressEvent(self, e):
        self.setFocus()
        self.press(e.position(), e.button())

    def mouseDoubleClickEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self.double_click(e.position())

    def resizeEvent(self, e):
        super().resizeEvent(e)


class PathCanvas(Canvas2D):
    """SIDE (x-z) or TOP (x-y) view of the fuselage, engine, opening and centrelines."""

    def __init__(self, model: InletModel, view: str, parent=None):
        super().__init__(model, parent)
        if view not in ("side", "top"):
            raise ValueError("view must be 'side' or 'top'")
        self.view = view
        self.across = 2 if view == "side" else 1  # the world axis shown vertically
        scene = model.scene
        self._fuselage = envelope(np.asarray(scene.fuselage.points), self.across)
        self._engine = envelope(np.asarray(scene.engine_mesh.points), self.across) \
            if getattr(scene, "engine_mesh", None) is not None and scene.engine_mesh.n_points else np.zeros((0, 2))
        self._context = [envelope(np.asarray(m.points), self.across)
                         for _, m in (getattr(scene, "context_parts", None) or []) if m.n_points]
        self._fitted = False
        model.add_listener(self.update)
        self.fit_view()

    # ---- projection ------------------------------------------------------------------------------------------
    def project(self, p) -> np.ndarray:
        p = np.atleast_2d(p)
        return np.c_[p[:, 0], p[:, self.across]]

    def fit_view(self) -> None:
        pts = [self._fuselage] if len(self._fuselage) else [np.zeros((1, 2))]
        duct = self.model.duct if self.model.placed else None
        if duct is not None:
            pts.append(self.project(duct.centreline.branch.ctrl))
            pts.append(self.project(duct.centreline.trunk.ctrl))
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        size = np.maximum(hi - lo, 1e-3) * (1.0 + 2.0 * FIT_MARGIN)
        self.centre = (lo + hi) / 2.0
        w, h = max(self.width(), 300), max(self.height(), 150)
        self.scale = float(np.clip(min(w / size[0], h / size[1]), MIN_SCALE, MAX_SCALE))
        self.update()

    # ---- handles ---------------------------------------------------------------------------------------------------
    def handles(self) -> dict[str, np.ndarray]:
        """World 2D position of each draggable item: the junction and the four tangent handles."""
        duct = self.model.duct if self.model.placed else None
        if duct is None:
            return {}
        pts = duct.centreline.handle_points()
        return {name: self.project(p)[0] for name, p in pts.items()}

    def anchors(self) -> dict[str, np.ndarray]:
        cl = self.model.duct.centreline
        a = {"start": cl.start, "junction_in": cl.branch.ctrl[3], "junction_out": cl.junction, "end": cl.face_centre}
        return {k: self.project(v)[0] for k, v in a.items()}

    def hit(self, pos) -> str | None:
        best, best_d = None, HIT_PX
        for name, hv in self.handles().items():
            q = self.to_screen(hv)
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d <= best_d:
                best, best_d = name, d
        return best

    # ---- mouse logic -------------------------------------------------------------------------------------------------
    def press(self, pos, button=QtCore.Qt.MouseButton.LeftButton) -> bool:
        if button == QtCore.Qt.MouseButton.LeftButton:
            self._drag = self.hit(pos)
            return self._drag is not None
        if button in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            self.pan_start(pos)
            return True
        return False

    def move(self, pos) -> None:
        if self._drag is not None:
            self.drag_to(self._drag, self.to_world(pos))
        elif not self.pan_move(pos):
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor if self.hit(pos) else QtCore.Qt.CursorShape.ArrowCursor)

    def release(self, pos=None) -> None:
        self._drag = None
        self._pan = None

    def double_click(self, pos) -> bool:
        """Return the handle under the cursor to its default."""
        name = self.hit(pos)
        fields = {"start": "start_handle", "junction_in": "junction_in_handle", "junction_out": "junction_out_handle",
                  "end": "end_handle"}
        if name in fields:
            changes = {fields[name]: None}
            if name in ("junction_in", "junction_out"):
                changes["junction_pitch_deg"] = None
            return self.model.set_centreline(f"cl:{name}", **changes)
        return False

    def drag_to(self, name: str, q: np.ndarray) -> bool:
        """Move the handle ``name`` towards the world point ``q`` (h, v) of this view, within its constraint."""
        m = self.model
        cl = m.duct.centreline if m.duct is not None else None
        if cl is None:
            return False
        r = cl.resolved
        if name == "junction":
            h, v = float(q[0]), float(q[1])
            x, z = (h, v) if self.view == "side" else (h, r.junction[1])
            return m.drag_junction(x, z)
        if name == "start":
            a = self.project(cl.start)[0]
            d = self.project(cl.start_dir)[0]
            return self._slide(m, "start_handle", "cl:start", q, a, d, 1.0)
        if name == "end":
            a = self.project(cl.face_centre)[0]
            d = self.project(cl.face_axis)[0]
            return self._slide(m, "end_handle", "cl:end", q, a, d, -1.0)
        # junction handles: the tangent lies in the XZ plane
        anchor = cl.branch.ctrl[3] if name == "junction_in" else cl.junction  # the branch ends at its half's centroid
        j = np.array([anchor[0], anchor[self.across]])
        sign = -1.0 if name == "junction_in" else 1.0
        w = (q - j) * sign
        key = "junction_in_handle" if name == "junction_in" else "junction_out_handle"
        if self.view == "side":
            length = float(np.hypot(*w))
            if length < 1e-9 or w[0] <= 0.0:
                return False
            phi = math.degrees(math.atan2(w[1], w[0]))
            return m.set_centreline(f"cl:{name}", **{key: length, "junction_pitch_deg": phi})
        cos_phi = math.cos(math.radians(r.junction_pitch_deg))
        if abs(cos_phi) < 1e-6:
            return False
        return m.set_centreline(f"cl:{name}", **{key: max(float(w[0]) / cos_phi, 0.0)})

    @staticmethod
    def _slide(m, field, key, q, anchor, direction, sign) -> bool:
        dd = float(np.dot(direction, direction))
        if dd < 1e-12:
            return False  # the direction is perpendicular to this view: it cannot be dragged here
        length = float(np.dot((q - anchor) * sign, direction)) / dd
        return m.set_centreline(key, **{field: max(length, 0.0)})

    # ---- drawing ----------------------------------------------------------------------------------------------------------
    def _grid(self, p):
        step = nice_step(GRID_MIN_PX / self.scale)
        h0, v0 = self.to_world(QtCore.QPointF(0, self.height()))
        h1, v1 = self.to_world(QtCore.QPointF(self.width(), 0))
        p.setFont(QtGui.QFont("Consolas", 7))
        for axis, lo, hi in (("h", h0, h1), ("v", v0, v1)):
            k = math.ceil(lo / step)
            while k * step <= hi:
                val = k * step
                p.setPen(self._pen(AXIS_COLOR if abs(val) < step / 2 else GRID_COLOR))
                if axis == "h":
                    x = self.to_screen((val, 0)).x()
                    p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, self.height()))
                else:
                    y = self.to_screen((0, val)).y()
                    p.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(self.width(), y))
                k += 1
        length = nice_step(100.0 / self.scale)
        p.setPen(self._pen(TEXT_COLOR, 2.0))
        y0 = self.height() - 12.0
        p.drawLine(QtCore.QPointF(10, y0), QtCore.QPointF(10 + length * self.scale, y0))
        p.setPen(self._pen(TEXT_COLOR))
        p.drawText(QtCore.QPointF(10, y0 - 4), f"{length:g} m")
        p.drawText(QtCore.QPointF(self.width() - 170, self.height() - 8),
                   "x aft ->, z up" if self.view == "side" else "x aft ->, y right up")

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(CANVAS_BACKGROUND))
        self._grid(p)
        if len(self._fuselage):
            p.setPen(self._pen(FUSELAGE_LINE, 1.5))
            p.setBrush(QtGui.QColor(FUSELAGE_FILL))
            self._polyline(p, self._fuselage, close=True)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(self._pen(CONTEXT_LINE, 1.0))
        for poly in self._context:
            self._polyline(p, poly, close=True)
        if len(self._engine):
            p.setPen(self._pen(ENGINE_LINE, 1.2))
            self._polyline(p, self._engine, close=True)
        m = self.model
        duct = m.duct if m.placed else None
        if duct is None:
            if m.placed and m.duct_error:
                p.setPen(self._pen(BAD_COLOR))
                p.drawText(QtCore.QPointF(10, 18), m.duct_error)
            return
        cl = duct.centreline
        # engine face
        r = 0.5 * duct.spec.face_diameter
        ax = duct.spec.face_axis / np.linalg.norm(duct.spec.face_axis)
        perp = np.cross(ax, [0.0, 1.0, 0.0] if self.view == "side" else [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) > 1e-9:
            perp = perp / np.linalg.norm(perp)
            a, b = cl.face_centre + r * perp, cl.face_centre - r * perp
            p.setPen(self._pen(FACE_LINE, 3.0))
            self._polyline(p, self.project(np.array([a, b])))
        # opening
        st0 = duct.branch[0]
        ring = duct.place(st0, st0.ring)
        p.setPen(self._pen(OPENING_COLOR, 2.5))
        self._polyline(p, self.project(ring), close=True)
        if self.view == "top":
            p.setPen(self._pen(MIRROR_COLOR, 1.5))
            self._polyline(p, self.project(ring * np.array([1.0, -1.0, 1.0])), close=True)
        # centrelines
        t = np.linspace(0.0, 1.0, CURVE_SAMPLES)
        branch, trunk = cl.branch.points(t), cl.trunk.points(t)
        if self.view == "top":
            p.setPen(self._pen(MIRROR_COLOR, 2.0))
            self._polyline(p, self.project(branch * np.array([1.0, -1.0, 1.0])))
        p.setPen(self._pen(CENTRE_COLOR, 2.5))
        self._polyline(p, self.project(branch))
        self._polyline(p, self.project(trunk))
        # key section positions along the path
        p.setPen(self._pen(KEY_COLOR, 1.5))
        for ks in m.state.sections[1:]:
            c = self.project(cl.point_at(ks.s))[0]
            q = self.to_screen(c)
            p.drawEllipse(q, 4, 4)
        # tangent handles, CAD style: a dashed line to a square, a larger circle for the junction
        anchors, handles = self.anchors(), self.handles()
        for name, hv in handles.items():
            q = self.to_screen(hv)
            if name == "junction":
                p.setPen(self._pen(ACCENT, 2.0))
                p.setBrush(QtGui.QColor(ACCENT))
                p.drawEllipse(q, MARKER_PX + 2, MARKER_PX + 2)
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                continue
            a = self.to_screen(anchors[name])
            p.setPen(self._pen(HANDLE_COLOR, 1.0, QtCore.Qt.PenStyle.DashLine))
            p.drawLine(a, q)
            p.setPen(self._pen(HANDLE_COLOR, 1.5))
            p.setBrush(QtGui.QColor(HANDLE_COLOR))
            p.drawRect(QtCore.QRectF(q.x() - MARKER_PX, q.y() - MARKER_PX, 2 * MARKER_PX, 2 * MARKER_PX))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(self._pen(TEXT_COLOR))
        p.setFont(QtGui.QFont("Consolas", 8))
        p.drawText(QtCore.QPointF(10, 16), "SIDE (x-z)" if self.view == "side" else "TOP (x-y)")


class AreaCanvas(Canvas2D):
    """A(s) for the branch and the trunk, two plots side by side."""

    def __init__(self, model: InletModel, parent=None):
        super().__init__(model, parent)
        self._hover = None
        model.add_listener(self.update)

    # ---- layout ------------------------------------------------------------------------------------------------------------
    def plot_rect(self, which: str) -> QtCore.QRectF:
        l, t, r, b = PLOT_MARGIN
        w = (self.width() - l - r - PLOT_GAP) / 2.0
        x0 = l if which == "branch" else l + w + PLOT_GAP
        return QtCore.QRectF(x0, t, max(w, 10.0), max(self.height() - t - b, 10.0))

    def y_range(self, which: str) -> tuple[float, float]:
        """Each plot has its own vertical range (the trunk carries n times the area of a branch)."""
        a = self.model.areas
        values = [k[1] for k in (a.branch_knots if which == "branch" else a.trunk_knots)]
        lo, hi = min(values), max(values)
        pad = max(hi - lo, 0.05 * hi) * Y_PADDING
        return max(lo - pad, 0.0), hi + pad

    def to_px(self, which: str, s: float, area: float) -> QtCore.QPointF:
        r = self.plot_rect(which)
        lo, hi = self.y_range(which)
        return QtCore.QPointF(r.left() + s * r.width(), r.bottom() - (area - lo) / (hi - lo) * r.height())

    def from_px(self, which: str, pos) -> tuple[float, float]:
        r = self.plot_rect(which)
        lo, hi = self.y_range(which)
        return ((pos.x() - r.left()) / r.width(), lo + (r.bottom() - pos.y()) / r.height() * (hi - lo))

    def which_plot(self, pos) -> str | None:
        for w in ("branch", "trunk"):
            if self.plot_rect(w).adjusted(-8, -8, 8, 8).contains(pos):
                return w
        return None

    # ---- items -----------------------------------------------------------------------------------------------------------------
    def items(self) -> list[tuple[str, str, int, float, float]]:
        """(plot, kind, index, s, area) of every draggable point: interior points and the junction ends."""
        a, out = self.model.areas, []
        for k, (s, v) in enumerate(a.branch_interior):
            out.append(("branch", "point", k, s, v))
        for k, (s, v) in enumerate(a.trunk_interior):
            out.append(("trunk", "point", k, s, v))
        out.append(("branch", "junction", -1, 1.0, a.junction_area / a.n))
        out.append(("trunk", "junction", -1, 0.0, a.junction_area))
        return out

    def hit(self, pos):
        best, best_d = None, HIT_PX
        for item in self.items():
            q = self.to_px(item[0], item[3], item[4])
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d <= best_d:
                best, best_d = item, d
        return best

    # ---- mouse logic ------------------------------------------------------------------------------------------------------------
    def press(self, pos, button=QtCore.Qt.MouseButton.LeftButton) -> bool:
        if not self.model.placed:
            return False
        item = self.hit(pos)
        if button == QtCore.Qt.MouseButton.LeftButton:
            self._drag = item
            return item is not None
        if button == QtCore.Qt.MouseButton.RightButton and item is not None and item[1] == "point":
            return self.model.area_remove_point(item[0], item[2])
        return False

    def move(self, pos) -> None:
        if self._drag is None:
            return
        plot, kind, index = self._drag[0], self._drag[1], self._drag[2]
        s, area = self.from_px(plot, pos)
        if kind == "junction":
            self.model.area_set_junction(area * (self.model.areas.n if plot == "branch" else 1.0))
        else:
            self.model.area_move_point(plot, index, s, area)

    def release(self, pos=None) -> None:
        self._drag = None

    def double_click(self, pos) -> bool:
        """Add a control point on the curve under the cursor."""
        plot = self.which_plot(pos)
        if plot is None or self.hit(pos) is not None or not self.model.placed:
            return False
        s, _ = self.from_px(plot, pos)
        return self.model.area_add_point(plot, s)

    # ---- drawing ------------------------------------------------------------------------------------------------------------------
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(CANVAS_BACKGROUND))
        m = self.model
        if not m.placed:
            p.setPen(self._pen(TEXT_COLOR))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Place the inlet (P) to edit the area law")
            return
        a = m.areas
        duct = m.duct
        limit = m.settings.checks.max_half_angle_deg
        p.setFont(QtGui.QFont("Consolas", 8))
        for which, fn, knots in (("branch", a.branch_area, a.branch_knots), ("trunk", a.trunk_area, a.trunk_knots)):
            r = self.plot_rect(which)
            lo, hi = self.y_range(which)
            p.setPen(self._pen(AXIS_COLOR))
            p.setBrush(QtGui.QColor("#13171B"))
            p.drawRect(r)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            step = nice_step((hi - lo) / 5.0)
            k = math.ceil(lo / step)
            while k * step <= hi:
                y = self.to_px(which, 0.0, k * step).y()
                p.setPen(self._pen(GRID_COLOR))
                p.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))
                p.setPen(self._pen(TEXT_COLOR))
                p.drawText(QtCore.QPointF(2, y + 4), f"{k * step:.3f}")
                k += 1
            p.setPen(self._pen(TEXT_COLOR))
            title = "branch  A(s)" if which == "branch" else "trunk  A(s)"
            p.drawText(QtCore.QPointF(r.left() + 4, r.top() + 12), title)
            p.drawText(QtCore.QPointF(r.left(), r.bottom() + 14), "0")
            p.drawText(QtCore.QPointF(r.right() - 8, r.bottom() + 14), "1")
            p.drawText(QtCore.QPointF(r.center().x() - 30, r.bottom() + 26), "local arc length s")
            # the curve, red where it decreases or its cone half-angle is over the limit
            s = np.linspace(0.0, 1.0, CURVE_SAMPLES)
            v = fn(s)
            length = 1.0
            if duct is not None:
                length = duct.centreline.length_branch if which == "branch" else duct.centreline.length_trunk
            radius = np.sqrt(v / math.pi)
            slope = np.diff(radius) / (np.diff(s) * max(length, 1e-9))
            bad = (np.diff(v) < -1e-12) | (np.degrees(np.arctan(slope)) > limit)
            for i in range(len(s) - 1):
                p.setPen(self._pen(BAD_COLOR if bad[i] else ACCENT, 2.5))
                p.drawLine(self.to_px(which, s[i], v[i]), self.to_px(which, s[i + 1], v[i + 1]))
            # key sections of this part
            if duct is not None:
                sj = duct.s_junction
                for ks in m.state.sections:
                    in_branch = ks.s <= sj
                    if (which == "branch") != in_branch:
                        continue
                    sl = ks.s / sj if in_branch else (ks.s - sj) / (1.0 - sj)
                    x = self.to_px(which, sl, 0.0).x()
                    p.setPen(self._pen(KEY_COLOR, 1.0, QtCore.Qt.PenStyle.DashLine))
                    p.drawLine(QtCore.QPointF(x, r.top()), QtCore.QPointF(x, r.bottom()))
            # control points
            for plot, kind, idx, sc, ar in self.items():
                if plot != which:
                    continue
                q = self.to_px(which, sc, ar)
                p.setPen(self._pen(POINT_COLOR, 1.5))
                p.setBrush(QtGui.QColor(POINT_COLOR if kind == "point" else ACCENT))
                p.drawEllipse(q, MARKER_PX, MARKER_PX)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            # the fixed end points
            for sc, ar in ((0.0, a.a_in),) if which == "branch" else ((1.0, a.a_face),):
                q = self.to_px(which, sc, ar)
                p.setPen(self._pen(TEXT_COLOR, 1.5))
                p.drawRect(QtCore.QRectF(q.x() - 4, q.y() - 4, 8, 8))
