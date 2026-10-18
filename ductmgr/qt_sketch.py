# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The section sketcher: the fuselage section in the inlet plane with 5 draggable opening vertices.

A custom-painted widget. Its mouse logic is plain methods (``press`` / ``move`` / ``release`` / ``zoom``) taking
a position and the Alt flag, so it can be driven without synthesising Qt events. All coordinates are the plane's
(u, v) in metres; the view maps them to pixels (u to the right, v up, as seen from the front).
"""
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ductmgr.explore_model import InletModel
from ductmgr.viz import ACCENT, BACKGROUND, ERROR_COLOR

HANDLE_RADIUS_PX = 9  # hit radius and drawn size of a vertex handle
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
GHOST_COLOR = "#6E7681"  # the mirrored left inlet, greyed
SYMMETRY_COLOR = "#BC8CFF"
TEXT_COLOR = "#9AA4AD"
SCALE_BAR_PX = 100


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
        self._drag: int | None = None
        self._pan_from: tuple[QtCore.QPointF, np.ndarray] | None = None
        self._had_opening = False
        self.setMinimumSize(360, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        model.add_listener(self.on_model_changed)
        self.on_model_changed()

    # ---- view transform ------------------------------------------------------------------------------------
    def to_screen(self, uv) -> QtCore.QPointF:
        u, v = uv
        return QtCore.QPointF(self.width() / 2 + (u - self.centre[0]) * self.scale,
                              self.height() / 2 - (v - self.centre[1]) * self.scale)

    def to_world(self, pos) -> np.ndarray:
        return np.array([(pos.x() - self.width() / 2) / self.scale + self.centre[0],
                         -(pos.y() - self.height() / 2) / self.scale + self.centre[1]])

    def fit_view(self) -> None:
        """Frame the opening, the section near it and the origin."""
        op = self.model.opening
        pts = [np.zeros((1, 2))]
        if op is not None:
            pts.append(op.outline)
            sec = self.model.section
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
        has = self.model.opening is not None
        if has and not self._had_opening:
            self.fit_view()  # first time an opening exists: frame it
        self._had_opening = has
        self.update()

    # ---- mouse logic (testable without Qt events) ------------------------------------------------------------
    def vertex_positions(self) -> list[QtCore.QPointF]:
        verts = self.model.state.vertices
        return [] if verts is None else [self.to_screen(v) for v in verts]

    def hit_vertex(self, pos) -> int | None:
        best, best_d = None, HANDLE_RADIUS_PX
        for i, p in enumerate(self.vertex_positions()):
            d = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            if d <= best_d:
                best, best_d = i, d
        return best

    def snap_tolerance(self, alt: bool) -> float | None:
        """Snap distance in metres (SNAP_PIXELS at the current zoom); None while Alt is held."""
        return None if alt else SNAP_PIXELS / self.scale

    def press(self, pos, alt: bool = False, button=QtCore.Qt.MouseButton.LeftButton) -> bool:
        if button == QtCore.Qt.MouseButton.LeftButton:
            self._drag = self.hit_vertex(pos)
            return self._drag is not None
        if button in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            self._pan_from = (pos, self.centre.copy())
            return True
        return False

    def move(self, pos, alt: bool = False) -> None:
        if self._drag is not None:
            self.model.drag_vertex(self._drag, tuple(self.to_world(pos)), self.snap_tolerance(alt))
        elif self._pan_from is not None:
            start, centre0 = self._pan_from
            self.centre = centre0 - np.array([pos.x() - start.x(), -(pos.y() - start.y())]) / self.scale
            self.update()
        else:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor if self.hit_vertex(pos) is not None
                           else QtCore.Qt.CursorShape.ArrowCursor)

    def release(self, pos=None) -> None:
        self._drag = None
        self._pan_from = None

    def zoom(self, pos, notches: float) -> None:
        """Zoom about the cursor: the world point under it stays put."""
        before = self.to_world(pos)
        self.scale = float(np.clip(self.scale * ZOOM_STEP**notches, MIN_SCALE_PX_PER_M, MAX_SCALE_PX_PER_M))
        after = self.to_world(pos)
        self.centre = self.centre + (before - after)
        self.update()

    # Qt event handlers -> the methods above
    def mousePressEvent(self, e):
        self.press(e.position(), bool(e.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier), e.button())

    def mouseMoveEvent(self, e):
        self.move(e.position(), bool(e.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier))

    def mouseReleaseEvent(self, e):
        self.release(e.position())

    def wheelEvent(self, e):
        self.zoom(e.position(), e.angleDelta().y() / 120.0)

    def resizeEvent(self, e):
        super().resizeEvent(e)

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
        plane = self.model.state.plane
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
        sec = m.section
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
        op = m.opening
        if op is None:
            return
        plane = m.state.plane
        # the left inlet (mirror image), projected into this plane and greyed
        ghost = plane.to_uv(plane.to_3d(op.outline) * np.array([1.0, -1.0, 1.0]))
        p.setPen(self._pen(GHOST_COLOR, 1.0, QtCore.Qt.PenStyle.DashLine))
        self._polyline(p, ghost, close=True)
        # the right opening: snapped edges in the accent colour (following the curve), free edges straight
        for path, snapped in zip(op.edge_paths, op.snapped_edges):
            if not op.valid:
                p.setPen(self._pen(ERROR_COLOR, 3.0))
            elif snapped:
                p.setPen(self._pen(ACCENT, 3.0))
            else:
                p.setPen(self._pen(FREE_EDGE_COLOR, 1.6))
            self._polyline(p, path)
        # vertex handles: snapped = filled, free = hollow
        p.setFont(QtGui.QFont("Consolas", 7))
        for i, (c, flag) in enumerate(zip(self.vertex_positions(), m.state.snapped)):
            colour = ERROR_COLOR if not op.valid else (ACCENT if flag else FREE_EDGE_COLOR)
            p.setPen(self._pen(colour, 2.0))
            p.setBrush(QtGui.QColor(colour) if flag else QtGui.QColor(CANVAS_BACKGROUND))
            p.drawEllipse(c, HANDLE_RADIUS_PX * 0.6, HANDLE_RADIUS_PX * 0.6)
            p.setPen(self._pen(TEXT_COLOR))
            p.drawText(QtCore.QPointF(c.x() + 9, c.y() - 7), str(i + 1))
        if not op.valid:
            p.setPen(self._pen(ERROR_COLOR))
            p.setFont(QtGui.QFont("Consolas", 9, QtGui.QFont.Weight.Bold))
            p.drawText(QtCore.QPointF(10, 18), "ERROR: " + op.errors[0])


class SketchDock(QtWidgets.QWidget):
    """The canvas with its live readouts and the Fit area / Fit view buttons."""

    def __init__(self, model: InletModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.canvas = SketchCanvas(model)
        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-family: Consolas; font-size: 8pt;")
        self.btn_fit_area = QtWidgets.QPushButton("Fit area")
        self.btn_fit_area.setToolTip("Scale only the free vertices about the snapped edge until the projected area "
                                     "matches the capture-area target")
        self.btn_fit_view = QtWidgets.QPushButton("Fit view")
        self.btn_fit_area.clicked.connect(lambda *_: self.model.fit_area())
        self.btn_fit_view.clicked.connect(lambda *_: self.canvas.fit_view())
        hint = QtWidgets.QLabel("Drag the vertices. Near the section curve they snap (filled); Alt disables "
                                "snapping. Wheel zooms, middle-drag pans.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9AA4AD; font-size: 8pt;")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_fit_area)
        row.addWidget(self.btn_fit_view)
        row.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(row)
        layout.addWidget(self.info)
        layout.addWidget(hint)
        model.add_listener(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        op = self.model.opening
        self.btn_fit_area.setEnabled(op is not None)
        if op is None:
            self.info.setText("")
            return
        green, amber, red = "#4CAF50", "#E0A030", "#F85149"
        colour = green if abs(op.delta) <= 0.005 else amber
        clearance = "n/a" if op.clearance is None else f"{op.clearance:.4f} m"
        lines = [
            f'<span style="color:{colour}">Area {op.area_projected:.5f} m² projected vs target {op.target_area:.5f} '
            f"({100 * op.delta:+.2f} %)</span>  in-plane {op.area_plane:.5f} m²",
            f"Snapped edge length {op.snapped_length:.4f} m   Free-edge clearance {clearance}",
        ]
        lines += [f'<span style="color:{red}">ERROR: {e}</span>' for e in op.errors]
        self.info.setText("<br>".join(lines))
