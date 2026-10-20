# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The section editor: the active key section in its plane, with any number of draggable vertices, straight or
curved (cubic Bézier) edges and per-vertex continuity.

A custom-painted widget. Its mouse logic is plain methods (``press`` / ``move`` / ``release`` / ``double_click`` /
``zoom``) taking a position and the Alt flag, so it can be driven without synthesising Qt events. All coordinates
are the plane's (u, v) in metres; the view maps them to pixels (u to the right, v up, as seen from the front).

Mouse: drag a vertex or a Bézier handle; double-click an edge to insert a vertex; right-click a vertex (or press Del
with it selected) to remove it; wheel zooms, middle- or right-drag on empty space pans.
"""
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ductmgr import outline as ol
from ductmgr.explore_model import InletModel
from ductmgr.viz import ACCENT, BACKGROUND, ERROR_COLOR

HANDLE_RADIUS_PX = 9  # hit radius of a vertex handle
VERTEX_DRAW_PX = 5.5  # half size of a vertex marker
BEZIER_HANDLE_PX = 4  # half size of a Bézier handle marker
EDGE_HIT_PX = 8  # a double-click this close to an edge inserts a vertex there
SNAP_PIXELS = 12  # a vertex dropped within this many pixels of the section curve snaps onto it
MIN_SCALE_PX_PER_M, MAX_SCALE_PX_PER_M = 20.0, 20000.0
ZOOM_STEP = 1.15  # per wheel notch
FIT_MARGIN = 0.25  # fraction of the content size added around it by "Fit view"
GRID_MIN_PX = 48  # the grid step is the smallest 1-2-5 value that spans at least this many pixels
CHINE_MARKER_PX = 5
CANVAS_BACKGROUND = "#101316"
GRID_COLOR, AXIS_COLOR = "#1E2329", "#2C333B"
SECTION_COLOR, CHINE_COLOR = "#8B949E", "#D29922"
FREE_EDGE_COLOR = "#E6EDF3"
FIXED_COLOR = "#7D8590"  # the engine face: not editable
GHOST_COLOR = "#6E7681"  # the mirrored left inlet, greyed
SYMMETRY_COLOR = "#BC8CFF"
HANDLE_COLOR = "#58A6FF"  # Bézier handles and their tangent lines
SELECT_COLOR = "#F2CC60"
TEXT_COLOR = "#9AA4AD"
SCALE_BAR_PX = 100

VERTEX, HANDLE = "vertex", "handle"
LEFT, MIDDLE, RIGHT = (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton,
                       QtCore.Qt.MouseButton.RightButton)


def nice_step(minimum: float) -> float:
    """The smallest 1-2-5 x 10^k that is >= ``minimum``."""
    exp = math.floor(math.log10(minimum))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * 10.0**exp >= minimum:
            return m * 10.0**exp
    return 10.0 ** (exp + 1)


class SketchCanvas(QtWidgets.QWidget):
    def __init__(self, model: InletModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.scale = 600.0  # pixels per metre
        self.centre = np.array([0.0, 0.0])  # (u, v) at the middle of the widget
        self.selected_vertex: int | None = None
        self.selected_edge: int | None = None
        self._drag: tuple | None = None  # (VERTEX, i) or (HANDLE, edge, which)
        self._pan_from: tuple[QtCore.QPointF, np.ndarray] | None = None
        self._view_key: tuple | None = None
        self._n_vertices = 0
        self._selection_listeners: list = []
        self.setMinimumSize(360, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        model.add_listener(self.on_model_changed)
        self.on_model_changed()

    # ---- the shape being edited --------------------------------------------------------------------------------
    @property
    def editable(self) -> bool:
        m = self.model
        return m.placed and not m.active_is_fixed

    @property
    def shape(self):
        return self.model.active_shape

    # ---- view transform ------------------------------------------------------------------------------------
    def to_screen(self, uv) -> QtCore.QPointF:
        u, v = uv
        return QtCore.QPointF(self.width() / 2 + (u - self.centre[0]) * self.scale,
                              self.height() / 2 - (v - self.centre[1]) * self.scale)

    def to_world(self, pos) -> np.ndarray:
        return np.array([(pos.x() - self.width() / 2) / self.scale + self.centre[0],
                         -(pos.y() - self.height() / 2) / self.scale + self.centre[1]])

    def fit_view(self) -> None:
        """Frame the active section (for the inlet also the skin near it) and the origin."""
        op = self.model.active_opening
        pts = [np.zeros((1, 2))]
        if op is not None:
            pts.append(op.outline)
            sec = self.model.active_curve
            if sec is not None:  # only the part of the curve close to the opening
                near = np.linalg.norm(sec.points - np.array(op.centroid_uv), axis=1)
                radius = 2.0 * float(np.ptp(op.outline, axis=0).max())
                pts.append(sec.points[near < radius] if (near < radius).any() else sec.points)
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        size = np.maximum(hi - lo, 1e-3) * (1.0 + 2.0 * FIT_MARGIN)
        self.centre = (lo + hi) / 2.0
        self.scale = float(np.clip(min(self.width() / size[0], self.height() / size[1]),
                                   MIN_SCALE_PX_PER_M, MAX_SCALE_PX_PER_M))
        self.update()

    def on_model_changed(self) -> None:
        m = self.model
        key = (m.placed, m.trunk_selected, m.active if m.placed else None)
        if key != self._view_key:
            self._view_key = key
            self.selected_vertex = self.selected_edge = None
            self._drag = None
            if m.placed:
                self.fit_view()  # another section (or the first opening): frame it
        shape = self.shape
        n = 0 if shape is None else shape.n
        if n != self._n_vertices:  # a vertex was inserted / removed: a stale selection would point at the wrong one
            self._n_vertices = n
            self.selected_vertex = self.selected_edge = None
        self.update()

    # ---- hit testing -----------------------------------------------------------------------------------------------
    def vertex_positions(self) -> list[QtCore.QPointF]:
        shape = self.shape
        return [] if shape is None else [self.to_screen(v) for v in shape.vertices]

    def handle_positions(self) -> dict[tuple[int, int], QtCore.QPointF]:
        """Screen position of every Bézier handle of a free curved edge, keyed by (edge, which)."""
        shape, out = self.shape, {}
        if shape is None or not self.editable:
            return out
        for e in range(shape.n):
            if shape.edge_is_curved(e):
                ctrl = shape.control_points(e)
                out[(e, 0)] = self.to_screen(ctrl[1])
                out[(e, 1)] = self.to_screen(ctrl[2])
        return out

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a.x() - b.x(), a.y() - b.y())

    def hit_vertex(self, pos) -> int | None:
        best, best_d = None, HANDLE_RADIUS_PX
        for i, p in enumerate(self.vertex_positions()):
            d = self._dist(p, pos)
            if d <= best_d:
                best, best_d = i, d
        return best

    def hit_handle(self, pos) -> tuple[int, int] | None:
        best, best_d = None, HANDLE_RADIUS_PX
        for key, p in self.handle_positions().items():
            d = self._dist(p, pos)
            if d <= best_d:
                best, best_d = key, d
        return best

    def hit_edge(self, pos) -> tuple[int, np.ndarray] | None:
        """(edge index, nearest point on it in (u, v)) of the edge within EDGE_HIT_PX of ``pos``, else None."""
        op = self.model.active_opening
        if op is None:
            return None
        best, best_d = None, EDGE_HIT_PX
        for e, path in enumerate(op.edge_paths):
            pts = np.array([[self.to_screen(p).x(), self.to_screen(p).y()] for p in path])
            a, b = pts[:-1], pts[1:]
            ab = b - a
            t = np.clip(((np.array([pos.x(), pos.y()]) - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12),
                        0.0, 1.0)
            near = a + t[:, None] * ab
            d = np.hypot(near[:, 0] - pos.x(), near[:, 1] - pos.y())
            k = int(np.argmin(d))
            if d[k] <= best_d:
                best_d = d[k]
                best = (e, self.to_world(QtCore.QPointF(near[k, 0], near[k, 1])))
        return best

    def snap_tolerance(self, alt: bool) -> float | None:
        """Snap distance in metres (SNAP_PIXELS at the current zoom); None while Alt is held."""
        return None if alt else SNAP_PIXELS / self.scale

    # ---- mouse logic (testable without Qt events) ------------------------------------------------------------
    def press(self, pos, alt: bool = False, button=LEFT) -> bool:
        if button == LEFT:
            if self.editable:
                handle = self.hit_handle(pos)
                if handle is not None:
                    self._drag = (HANDLE, *handle)
                    self.selected_edge, self.selected_vertex = handle[0], None
                    self.model_notify()
                    return True
                i = self.hit_vertex(pos)
                if i is not None:
                    self._drag = (VERTEX, i)
                    self.selected_vertex, self.selected_edge = i, None
                    self.model_notify()
                    return True
                edge = self.hit_edge(pos)
                self.selected_edge = None if edge is None else edge[0]
                self.selected_vertex = None
                self.model_notify()
                return edge is not None
            return False
        if button == RIGHT and self.editable:
            i = self.hit_vertex(pos)
            if i is not None:  # right-click on a vertex removes it
                self.selected_vertex = i
                return self.delete_selected()
        if button in (MIDDLE, RIGHT):
            self._pan_from = (pos, self.centre.copy())
            return True
        return False

    def move(self, pos, alt: bool = False) -> None:
        if self._drag is not None:
            uv = tuple(self.to_world(pos))
            if self._drag[0] == VERTEX:
                self.model.drag_vertex(self._drag[1], uv, self.snap_tolerance(alt))
            else:
                self.model.drag_handle(self._drag[1], self._drag[2], uv)
        elif self._pan_from is not None:
            start, centre0 = self._pan_from
            self.centre = centre0 - np.array([pos.x() - start.x(), -(pos.y() - start.y())]) / self.scale
            self.update()
        elif self.editable:
            over = self.hit_vertex(pos) is not None or self.hit_handle(pos) is not None
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor if over else QtCore.Qt.CursorShape.ArrowCursor)

    def release(self, pos=None) -> None:
        if self._drag is not None:
            self._drag = None
            self.model.finish_edit()  # a key section goes back to its area A(s) when the drag ends
        self._pan_from = None

    def double_click(self, pos) -> int | None:
        """Insert a vertex on the edge under ``pos``; the new vertex index, or None."""
        if not self.editable or self.hit_vertex(pos) is not None:
            return None
        hit = self.hit_edge(pos)
        if hit is None:
            return None
        at = self.model.insert_vertex(hit[0], tuple(hit[1]))
        if at is not None:
            self.selected_vertex, self.selected_edge = at, None
            self.update()
        return at

    def delete_selected(self) -> bool:
        """Remove the selected vertex (Del / right-click)."""
        if self.selected_vertex is None or not self.editable:
            return False
        i, self.selected_vertex = self.selected_vertex, None
        return self.model.delete_vertex(i)

    def model_notify(self) -> None:
        """Selection changes are not model state, but the dock's buttons follow them."""
        self.update()
        for callback in list(self._selection_listeners):
            callback()

    def add_selection_listener(self, callback) -> None:
        self._selection_listeners.append(callback)

    def zoom(self, pos, notches: float) -> None:
        """Zoom about the cursor: the world point under it stays put."""
        before = self.to_world(pos)
        self.scale = float(np.clip(self.scale * ZOOM_STEP**notches, MIN_SCALE_PX_PER_M, MAX_SCALE_PX_PER_M))
        after = self.to_world(pos)
        self.centre = self.centre + (before - after)
        self.update()

    # Qt event handlers -> the methods above
    def mousePressEvent(self, e):
        self.setFocus()
        self.press(e.position(), bool(e.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier), e.button())

    def mouseMoveEvent(self, e):
        self.move(e.position(), bool(e.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier))

    def mouseReleaseEvent(self, e):
        self.release(e.position())

    def mouseDoubleClickEvent(self, e):
        if e.button() == LEFT:
            self.double_click(e.position())

    def wheelEvent(self, e):
        self.zoom(e.position(), e.angleDelta().y() / 120.0)

    def keyPressEvent(self, e):
        if e.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace) and self.delete_selected():
            return
        super().keyPressEvent(e)

    # ---- drawing ----------------------------------------------------------------------------------------------
    def _pen(self, color, width=1.0, style=QtCore.Qt.PenStyle.SolidLine):
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

    def _draw_grid(self, painter):
        step = nice_step(GRID_MIN_PX / self.scale)
        u0, v0 = self.to_world(QtCore.QPointF(0, self.height()))
        u1, v1 = self.to_world(QtCore.QPointF(self.width(), 0))
        painter.setFont(QtGui.QFont("Consolas", 7))
        for axis, lo, hi in (("u", u0, u1), ("v", v0, v1)):
            k = math.ceil(lo / step)
            while k * step <= hi:
                val = k * step
                painter.setPen(self._pen(AXIS_COLOR if abs(val) < step / 2 else GRID_COLOR))
                if axis == "u":
                    x = self.to_screen((val, 0)).x()
                    painter.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, self.height()))
                else:
                    y = self.to_screen((0, val)).y()
                    painter.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(self.width(), y))
                k += 1
        # scale bar: a round length near SCALE_BAR_PX pixels
        length = nice_step(SCALE_BAR_PX / self.scale)
        x0, y0 = 12.0, self.height() - 14.0
        painter.setPen(self._pen(TEXT_COLOR, 2.0))
        painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x0 + length * self.scale, y0))
        painter.setPen(self._pen(TEXT_COLOR))
        label = f"{length * 1000:g} mm" if length < 0.1 else f"{length:g} m"
        painter.drawText(QtCore.QPointF(x0, y0 - 5), label)
        painter.drawText(QtCore.QPointF(self.width() - 205, self.height() - 8), "u right, v up (view from the front)")

    def _draw_symmetry_plane(self, painter):
        m = self.model
        if m.trunk_selected:  # the trunk frame is symmetric: the plane y = 0 is the vertical axis u = 0
            if m.state.symmetry != "xz":
                return
            x = self.to_screen((0.0, 0.0)).x()
            painter.setPen(self._pen(SYMMETRY_COLOR, 1.0, QtCore.Qt.PenStyle.DashLine))
            painter.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, self.height()))
            painter.setPen(self._pen(SYMMETRY_COLOR))
            painter.drawText(QtCore.QPointF(min(max(x + 6, 4), self.width() - 130), 28), "symmetry plane y = 0")
            return
        plane = m.state.plane
        if plane is None:
            return
        u, v, _ = plane.frame()
        oy = plane.origin[1]
        w, h = self.width(), self.height()
        painter.setPen(self._pen(SYMMETRY_COLOR, 1.0, QtCore.Qt.PenStyle.DashLine))
        # the plane y = 0:  oy + u * u_y + v * v_y = 0, a line in (u, v)
        if abs(v[1]) > 1e-9:
            ua, ub = self.to_world(QtCore.QPointF(0, 0))[0], self.to_world(QtCore.QPointF(w, 0))[0]
            pts = [(x, -(oy + x * u[1]) / v[1]) for x in (ua, ub)]
        elif abs(u[1]) > 1e-9:
            x = -oy / u[1]
            va, vb = self.to_world(QtCore.QPointF(0, h))[1], self.to_world(QtCore.QPointF(0, 0))[1]
            pts = [(x, va), (x, vb)]
        else:
            return
        self._polyline(painter, pts)
        painter.setPen(self._pen(SYMMETRY_COLOR))
        anchor = self.to_screen(pts[0])
        painter.drawText(QtCore.QPointF(min(max(anchor.x() + 6, 4), w - 130), min(max(anchor.y() - 6, 12), h - 4)),
                         "symmetry plane y = 0")

    def _draw_vertex(self, p, centre, mode: str, colour: str, filled: bool, selected: bool):
        """Marker by continuity: corner = square, smooth = circle, symmetric = circle with an inner ring."""
        r = VERTEX_DRAW_PX
        p.setPen(self._pen(SELECT_COLOR if selected else colour, 2.4 if selected else 2.0))
        p.setBrush(QtGui.QColor(colour) if filled else QtGui.QColor(CANVAS_BACKGROUND))
        if mode == ol.CORNER:
            p.drawRect(QtCore.QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r))
        else:
            p.drawEllipse(centre, r, r)
            if mode == ol.SYMMETRIC:
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(centre, r * 0.45, r * 0.45)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(CANVAS_BACKGROUND))
        self._draw_grid(p)
        m = self.model
        if m.state.plane is None:
            p.setPen(self._pen(TEXT_COLOR))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Place the inlet (P) to start sketching")
            return
        self._draw_symmetry_plane(p)
        sec = m.active_curve
        if sec is not None:
            p.setPen(self._pen(SECTION_COLOR, 1.5))
            self._polyline(p, sec.points, close=sec.closed)
            p.setPen(self._pen(CHINE_COLOR, 1.5))
            p.setBrush(QtGui.QColor(CHINE_COLOR))
            for i in sec.chines:  # chine corners: diamonds
                c = self.to_screen(sec.points[i])
                r = CHINE_MARKER_PX
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(c.x(), c.y() - r), QtCore.QPointF(c.x() + r, c.y()),
                                               QtCore.QPointF(c.x(), c.y() + r), QtCore.QPointF(c.x() - r, c.y())]))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        op = m.active_opening
        shape = m.active_shape
        if op is None or shape is None:
            return
        plane = m.state.plane
        if m.active_is_inlet and m.has_twin(m.selected):  # the mirrored twin, projected into this plane and greyed
            ghost = plane.to_uv(plane.to_3d(op.outline) * np.array([1.0, -1.0, 1.0]))
            p.setPen(self._pen(GHOST_COLOR, 1.0, QtCore.Qt.PenStyle.DashLine))
            self._polyline(p, ghost, close=True)
        # the outline: skin-following edges in the accent colour, free edges (straight or curved) in white
        for e, (path, snapped) in enumerate(zip(op.edge_paths, op.snapped_edges)):
            if not op.valid:
                p.setPen(self._pen(ERROR_COLOR, 3.0))
            elif m.active_is_fixed:
                p.setPen(self._pen(FIXED_COLOR, 2.0))
            elif snapped:
                p.setPen(self._pen(ACCENT, 3.0))
            elif e == self.selected_edge:
                p.setPen(self._pen(SELECT_COLOR, 2.6))
            else:
                p.setPen(self._pen(FREE_EDGE_COLOR, 1.6))
            self._polyline(p, path)
        if not self.editable:
            p.setPen(self._pen(TEXT_COLOR))
            p.setFont(QtGui.QFont("Consolas", 9))
            p.drawText(QtCore.QPointF(10, 18), "Engine face (s = 1): a fixed circle" if m.active_is_face
                       else "Junction: this branch's half of the trunk start section (derived, fixed)")
            return
        # Bézier handles of free curved edges: tangent line and a small square
        for (e, which), hp in self.handle_positions().items():
            anchor = self.to_screen(shape.vertices[(e + which) % shape.n])
            p.setPen(self._pen(HANDLE_COLOR, 1.0, QtCore.Qt.PenStyle.DashLine))
            p.drawLine(anchor, hp)
            p.setPen(self._pen(HANDLE_COLOR, 1.5))
            p.setBrush(QtGui.QColor(HANDLE_COLOR))
            r = BEZIER_HANDLE_PX
            p.drawRect(QtCore.QRectF(hp.x() - r, hp.y() - r, 2 * r, 2 * r))
        # vertex markers: filled = snapped to the skin
        p.setFont(QtGui.QFont("Consolas", 7))
        for i, (c, flag, mode) in enumerate(zip(self.vertex_positions(), shape.snapped, shape.continuity)):
            colour = ERROR_COLOR if not op.valid else (ACCENT if flag else FREE_EDGE_COLOR)
            self._draw_vertex(p, c, mode, colour, flag, i == self.selected_vertex)
            p.setPen(self._pen(TEXT_COLOR))
            p.drawText(QtCore.QPointF(c.x() + 9, c.y() - 7), str(i + 1))
        if not op.valid:
            p.setPen(self._pen(ERROR_COLOR))
            p.setFont(QtGui.QFont("Consolas", 9, QtGui.QFont.Weight.Bold))
            p.drawText(QtCore.QPointF(10, 18), "ERROR: " + op.errors[0])


class SketchDock(QtWidgets.QWidget):
    """The canvas with its tool buttons and live readouts."""

    def __init__(self, model: InletModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.canvas = SketchCanvas(model)
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-weight: bold;")
        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-family: Consolas; font-size: 8pt;")

        self.btn_fit_area = QtWidgets.QPushButton("Fit area")
        self.btn_fit_area.setToolTip("Inlet: scale only the free vertices about the snapped edges until the projected "
                                     "area matches the capture-area target")
        self.btn_fit_view = QtWidgets.QPushButton("Fit view")
        self.btn_fit_area.clicked.connect(lambda *_: self.model.fit_area())
        self.btn_fit_view.clicked.connect(lambda *_: self.canvas.fit_view())

        self.btn_straight = QtWidgets.QPushButton("Straight")
        self.btn_curved = QtWidgets.QPushButton("Curved")
        self.btn_straight.setToolTip("Make the selected edge straight")
        self.btn_curved.setToolTip("Make the selected edge a cubic Bézier with two draggable handles")
        self.btn_straight.clicked.connect(lambda *_: self._set_edge("straight"))
        self.btn_curved.clicked.connect(lambda *_: self._set_edge("curved"))
        self.btn_corner = QtWidgets.QPushButton("Corner")
        self.btn_smooth = QtWidgets.QPushButton("Smooth")
        self.btn_symmetric = QtWidgets.QPushButton("Symmetric")
        self.btn_corner.setToolTip("The two edges meet at an angle")
        self.btn_smooth.setToolTip("G1: the tangent is continuous, the handle lengths are independent")
        self.btn_symmetric.setToolTip("G1 with equal handle lengths")
        self.btn_corner.clicked.connect(lambda *_: self._set_continuity("corner"))
        self.btn_smooth.clicked.connect(lambda *_: self._set_continuity("smooth"))
        self.btn_symmetric.clicked.connect(lambda *_: self._set_continuity("symmetric"))
        self.btn_delete = QtWidgets.QPushButton("Delete vertex")
        self.btn_delete.setToolTip("Remove the selected vertex (Del or right-click); a section keeps at least 3")
        self.btn_delete.clicked.connect(lambda *_: self.canvas.delete_selected())

        hint = QtWidgets.QLabel("Drag vertices and Bézier handles. Double-click an edge to insert a vertex; "
                                "right-click a vertex (or Del) to remove it. Near the section curve a vertex snaps "
                                "(filled); Alt disables snapping. Wheel zooms, middle-drag pans.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9AA4AD; font-size: 8pt;")

        def row(*widgets):
            r = QtWidgets.QHBoxLayout()
            for w in widgets:
                r.addWidget(w)
            r.addStretch(1)
            return r

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.title)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(row(QtWidgets.QLabel("Edge"), self.btn_straight, self.btn_curved,
                             QtWidgets.QLabel("Vertex"), self.btn_corner, self.btn_smooth, self.btn_symmetric))
        layout.addLayout(row(self.btn_delete, self.btn_fit_area, self.btn_fit_view))
        layout.addWidget(self.info)
        layout.addWidget(hint)
        model.add_listener(self.refresh)
        self.canvas.add_selection_listener(self.refresh)
        self.refresh()

    def _set_title(self, text: str) -> None:
        """The section title, led by the inlet's name when there are several (or "Trunk"), and noting a symmetric
        editor."""
        m = self.model
        if m.trunk_selected:
            text = f"Trunk  |  {text}"
        elif len(m.state.inlets) > 1:
            text = f"{m.inlet.name}  |  {text}"
        if m.symmetric_editing:
            text += "  [mirror symmetric about u = 0: edit one side]"
        self.title.setText(text)

    # ---- buttons -> model ------------------------------------------------------------------------------------------
    def _set_edge(self, kind: str) -> None:
        if self.canvas.selected_edge is not None:
            self.model.set_edge_type(self.canvas.selected_edge, kind)

    def _set_continuity(self, mode: str) -> None:
        if self.canvas.selected_vertex is not None:
            self.model.set_continuity(self.canvas.selected_vertex, mode)

    # ---- model -> widgets ---------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        m, c = self.model, self.canvas
        op = m.active_opening
        shape = m.active_shape
        editable = c.editable
        e, v = c.selected_edge, c.selected_vertex
        free_edge = editable and e is not None and e < shape.n and not shape.edge_is_snapped(e)
        has_vertex = editable and v is not None and v < shape.n
        self.btn_straight.setEnabled(free_edge and shape.edges[e] != "straight")
        self.btn_curved.setEnabled(free_edge and shape.edges[e] != "curved")
        for btn, mode in ((self.btn_corner, "corner"), (self.btn_smooth, "smooth"), (self.btn_symmetric, "symmetric")):
            btn.setEnabled(has_vertex and shape.continuity[v] != mode)
        self.btn_delete.setEnabled(has_vertex and shape.n > ol.MIN_VERTICES)
        self.btn_fit_area.setEnabled(editable)
        if op is None:
            self.title.setText("")
            self.info.setText("")
            return
        if m.active_is_junction:
            self._set_title(f"Section s = {m.active_s:.3f}  -  junction: the branch end, exactly its half of the "
                               "trunk start section (derived, fixed)")
        elif m.active_is_face:
            self._set_title("Section s = 1.000  -  engine face (fixed circle)")
        elif m.active_is_trunk_start:
            self._set_title("Section s = 0.000  -  trunk start: every branch end is cut from this section "
                            f"(size held at the junction area {op.target_area:.5f} m²)")
        elif m.active_is_inlet:
            self._set_title("Section s = 0.000  -  inlet (in the inlet plane, snaps to the skin)")
        else:
            self._set_title(f"Section s = {m.active_s:.3f}  -  key section "
                               f"(shape is free, size is held at A(s) = {op.target_area:.5f} m²)")
        green, amber, red = "#4CAF50", "#E0A030", "#F85149"
        colour = green if abs(op.delta) <= 0.005 else amber
        if m.active_is_inlet:
            clearance = "n/a" if op.clearance is None else f"{op.clearance:.4f} m"
            lines = [
                f'<span style="color:{colour}">Area {op.area_projected:.5f} m² projected vs target '
                f"{op.target_area:.5f} ({100 * op.delta:+.2f} %)</span>  in-plane {op.area_plane:.5f} m²",
                f"Snapped edge length {op.snapped_length:.4f} m   Free-edge clearance {clearance}",
            ]
        else:
            lines = [f'<span style="color:{colour}">Area {op.area_plane:.5f} m² vs A(s) {op.target_area:.5f} '
                     f"({100 * op.delta:+.2f} %)</span>"]
        lines.append(f"{shape.n} vertices, {sum(1 for k in range(shape.n) if shape.edge_is_curved(k))} curved edges")
        lines += [f'<span style="color:{red}">ERROR: {err}</span>' for err in op.errors]
        self.info.setText("<br>".join(lines))
