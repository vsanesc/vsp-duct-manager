# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The section sketcher widget: view transform, hit testing, dragging with snapping, painting."""
import os

if not os.environ.get("DUCTMGR_GUI_TESTS"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import bodies  # noqa: E402
from ductmgr import qt_sketch as qs  # noqa: E402
from ductmgr import sketch as sk  # noqa: E402
from ductmgr.config import EXAMPLE_CONFIG  # noqa: E402
from ductmgr.explore_model import InletModel  # noqa: E402
from ductmgr.section import distance_to_section, nearest_on_section  # noqa: E402
from ductmgr.viz import ACCENT, ERROR_COLOR, load_scene  # noqa: E402

LEFT = QtCore.Qt.MouseButton.LeftButton
MIDDLE = QtCore.Qt.MouseButton.MiddleButton
SIZE = (640, 480)


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    return app


@pytest.fixture(scope="module")
def scene(example_env):
    cfg, path, _ = example_env
    return load_scene(cfg, path)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def model(scene, example_env, clock):
    _, path, _ = example_env
    cfg = copy.deepcopy(example_env[0])
    return InletModel(scene, cfg, path, None, clock=clock)


@pytest.fixture()
def canvas(qapp, model):
    c = qs.SketchCanvas(model)
    c.resize(*SIZE)
    c.fit_view()
    return c


def at(canvas, uv):
    return canvas.to_screen(uv)


def pixels(widget):
    """RGBA array of what the widget paints."""
    img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3]
    return arr.copy()  # the view's memory belongs to ``img``, which dies when this function returns


def count_colour(arr, hex_colour, tol=24):
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)])
    return int((np.abs(arr.astype(int) - rgb).max(axis=2) <= tol).sum())


# ---- view transform ---------------------------------------------------------------------------------------
def test_screen_and_world_coordinates_round_trip_with_v_up(canvas):
    for uv in ((0.0, 0.0), (0.12, -0.05), (-0.3, 0.4)):
        assert canvas.to_world(canvas.to_screen(uv)) == pytest.approx(uv, abs=1e-12)
    low, high = canvas.to_screen((0, 0)), canvas.to_screen((0, 0.1))
    assert high.y() < low.y() and high.x() == low.x()  # larger v is higher on screen (smaller y)
    right = canvas.to_screen((0.1, 0))
    assert right.x() > low.x()  # larger u is to the right


def test_fit_view_frames_the_opening_and_the_origin(canvas, model):
    rect = QtCore.QRectF(0, 0, *SIZE)
    for p in canvas.vertex_positions():
        assert rect.contains(p)
    assert rect.contains(canvas.to_screen((0.0, 0.0)))  # the plane origin
    for p in map(canvas.to_screen, model.opening.outline):
        assert rect.contains(p)
    assert qs.MIN_SCALE_PX_PER_M <= canvas.scale <= qs.MAX_SCALE_PX_PER_M


# ---- hit testing ------------------------------------------------------------------------------------------------
def test_hit_testing_picks_the_nearest_handle_within_its_radius(canvas, model):
    for i, v in enumerate(model.state.vertices):
        p = at(canvas, v)
        assert canvas.hit_vertex(p) == i
        assert canvas.hit_vertex(QtCore.QPointF(p.x() + qs.HANDLE_RADIUS_PX - 1, p.y())) == i
        assert canvas.hit_vertex(QtCore.QPointF(p.x() + qs.HANDLE_RADIUS_PX + 3, p.y())) is None
    assert canvas.hit_vertex(QtCore.QPointF(2, 2)) is None


# ---- dragging and snapping ---------------------------------------------------------------------------------------------
def test_dragging_a_vertex_moves_it_to_the_cursor(canvas, model, clock):
    start = model.state
    idx = 3  # the apex: free
    assert canvas.press(at(canvas, model.state.vertices[idx]), alt=False)
    target = np.array(model.state.vertices[idx]) + [0.02, 0.05]  # well above the skin
    canvas.move(at(canvas, target), alt=False)
    assert np.array(model.state.vertices[idx]) == pytest.approx(target, abs=1e-9)
    assert model.state.snapped[idx] is False
    canvas.release()
    canvas.move(at(canvas, target + [0.1, 0.1]), alt=False)  # released: moving does nothing
    assert np.array(model.state.vertices[idx]) == pytest.approx(target, abs=1e-9)
    assert model.undo() and model.state == start  # the whole drag is one step on the shared undo stack


def test_pressing_on_empty_space_drags_nothing(canvas, model):
    before = model.state
    assert not canvas.press(QtCore.QPointF(3, 3))
    canvas.move(QtCore.QPointF(100, 100))
    assert model.state == before


def test_a_vertex_dropped_near_the_curve_snaps_on_and_marks_itself(canvas, model):
    idx = 3
    _, _, _ = nearest_on_section(model.section, model.state.vertices[idx])
    foot = nearest_on_section(model.section, (0.0, 0.2))[0]  # a skin point under the apex
    canvas.press(at(canvas, model.state.vertices[idx]))
    # 5 pixels from the curve: inside the 12 px snap radius
    offset = 5.0 / canvas.scale
    canvas.move(at(canvas, foot + [0.0, offset]), alt=False)
    canvas.release()
    assert model.state.snapped[idx] is True
    assert distance_to_section(model.section, np.array(model.state.vertices[idx]))[0] < 1e-6  # exactly on the curve


def test_the_snap_distance_is_a_pixel_tolerance(canvas, model):
    assert canvas.snap_tolerance(False) == pytest.approx(qs.SNAP_PIXELS / canvas.scale)
    assert canvas.snap_tolerance(True) is None  # Alt
    s0 = canvas.snap_tolerance(False)
    canvas.zoom(QtCore.QPointF(300, 200), 3)  # zoomed in: the same pixels are fewer metres
    assert canvas.snap_tolerance(False) < s0


def test_dropping_further_than_the_tolerance_does_not_snap_and_unsnaps(canvas, model, clock):
    snapped_idx = 0
    assert model.state.snapped[snapped_idx]
    canvas.press(at(canvas, model.state.vertices[snapped_idx]))
    away = np.array(model.state.vertices[snapped_idx]) + [0.0, (qs.SNAP_PIXELS + 25) / canvas.scale]  # drag it away
    canvas.move(at(canvas, away), alt=False)
    canvas.release()
    assert model.state.snapped[snapped_idx] is False  # unsnapped
    assert np.array(model.state.vertices[snapped_idx]) == pytest.approx(away, abs=1e-9)
    clock.advance(5.0)
    # and dropping a free vertex far from the curve stays free
    canvas.press(at(canvas, model.state.vertices[3]))
    canvas.move(at(canvas, np.array(model.state.vertices[3]) + [0.0, 40 / canvas.scale]))
    canvas.release()
    assert model.state.snapped[3] is False


def test_alt_disables_snapping(canvas, model):
    idx = 3
    foot = nearest_on_section(model.section, (0.0, 0.2))[0]
    spot = foot + [0.0, 3.0 / canvas.scale]  # 3 px from the curve: would snap without Alt
    canvas.press(at(canvas, model.state.vertices[idx]))
    canvas.move(at(canvas, spot), alt=True)
    canvas.release()
    assert model.state.snapped[idx] is False
    assert np.array(model.state.vertices[idx]) == pytest.approx(spot, abs=1e-9)


def test_snapping_two_neighbours_makes_their_edge_follow_the_curve(canvas, model):
    # vertices 0 and 1 are snapped, so edge 0 follows the curve; snap vertex 2 too and edge 1 (1 -> 2) follows it
    assert model.opening.snapped_edges == (True, False, False, False, False)
    foot = nearest_on_section(model.section, (0.2, 0.0))[0]
    canvas.press(at(canvas, model.state.vertices[2]))
    canvas.move(at(canvas, foot + [0.0, 2.0 / canvas.scale]), alt=False)
    canvas.release()
    assert model.state.snapped[2] is True
    assert model.opening.snapped_edges == (True, True, False, False, False)
    assert model.opening.snapped_length > 0


def test_dragging_a_vertex_across_the_body_invalidates_the_opening(canvas, model):
    canvas.press(at(canvas, model.state.vertices[3]))
    canvas.move(at(canvas, (0.0, -0.8)), alt=True)  # inside the fuselage
    canvas.release()
    assert not model.valid and model.opening.errors


# ---- zoom and pan ------------------------------------------------------------------------------------------------------
def test_zoom_keeps_the_point_under_the_cursor_fixed(canvas):
    cursor = QtCore.QPointF(420, 160)
    before = canvas.to_world(cursor)
    scale0 = canvas.scale
    canvas.zoom(cursor, 4)
    assert canvas.to_world(cursor) == pytest.approx(before, abs=1e-12)
    assert canvas.scale == pytest.approx(scale0 * qs.ZOOM_STEP**4, rel=1e-12)  # zoomed in by 4 notches
    canvas.zoom(cursor, -4)
    assert canvas.scale == pytest.approx(scale0, rel=1e-12)
    assert canvas.to_world(cursor) == pytest.approx(before, abs=1e-12)  # and back, the point still under the cursor


def test_zoom_is_limited(canvas):
    canvas.zoom(QtCore.QPointF(10, 10), 200)
    assert canvas.scale == qs.MAX_SCALE_PX_PER_M
    canvas.zoom(QtCore.QPointF(10, 10), -400)
    assert canvas.scale == qs.MIN_SCALE_PX_PER_M


def test_middle_drag_pans_without_touching_the_sketch(canvas, model):
    before_state, before_centre = model.state, canvas.centre.copy()
    assert canvas.press(QtCore.QPointF(300, 200), button=MIDDLE)
    canvas.move(QtCore.QPointF(340, 230))
    canvas.release()
    assert model.state == before_state
    shift = canvas.centre - before_centre
    assert shift == pytest.approx([-40 / canvas.scale, 30 / canvas.scale], abs=1e-12)  # the content follows the mouse


# ---- painting ---------------------------------------------------------------------------------------------------------------
def test_the_snapped_edge_is_painted_in_the_accent_colour_and_the_curve_is_drawn(canvas):
    arr = pixels(canvas)
    assert count_colour(arr, ACCENT) > 60  # the snapped edge following the curve
    assert count_colour(arr, qs.SECTION_COLOR, 70) > 40  # the section curve (thin and antialiased)
    assert count_colour(arr, qs.CANVAS_BACKGROUND, 4) > arr.shape[0] * arr.shape[1] * 0.5  # mostly background


def test_an_invalid_opening_is_painted_red_with_an_error_message(canvas, model):
    before = pixels(canvas)
    assert count_colour(before, ERROR_COLOR) == 0
    canvas.press(at(canvas, model.state.vertices[3]))
    canvas.move(at(canvas, (0.0, -0.8)), alt=True)
    canvas.release()
    after = pixels(canvas)
    assert count_colour(after, ERROR_COLOR) > 100
    assert count_colour(after, ACCENT) == 0  # the outline is red, not accent


def test_the_symmetry_plane_and_a_scale_bar_are_painted(canvas):
    arr = pixels(canvas)
    assert count_colour(arr, qs.SYMMETRY_COLOR, 20) > 20  # the dashed y = 0 line
    assert count_colour(arr, qs.TEXT_COLOR, 12) > 20  # the scale bar and labels


def test_the_painter_copes_with_a_swept_pitched_plane_and_no_inlet(qapp, model, scene, example_env):
    model.set_param("sweep_deg", 30.0)
    model.set_param("pitch_deg", 10.0)
    c = qs.SketchCanvas(model)
    c.resize(*SIZE)
    assert pixels(c).size > 0
    cfg = copy.deepcopy(example_env[0])
    cfg["ducts"]["inlet"]["origin"] = cfg["ducts"]["inlet"]["vertices"] = cfg["ducts"]["inlet"]["snapped"] = None
    empty = qs.SketchCanvas(InletModel(scene, cfg, EXAMPLE_CONFIG, None))
    empty.resize(*SIZE)
    assert pixels(empty).size > 0 and empty.vertex_positions() == [] and empty.hit_vertex(QtCore.QPointF(5, 5)) is None


def test_chine_corners_are_marked(qapp):
    prism = bodies.chined_prism()
    scene = SimpleNamespace(fuselage=prism, sizing=SimpleNamespace(capture_area_per_inlet=0.05), engine_check=None,
                            warnings=[], face_center=None)
    cfg = {"ducts": {"inlet": {"origin": [4.0, 0.0, 0.4]}}}
    m = InletModel(scene, cfg, EXAMPLE_CONFIG, None)
    assert len(m.section.chines) == 6
    c = qs.SketchCanvas(m)
    c.resize(*SIZE)
    c.fit_view()
    assert count_colour(pixels(c), qs.CHINE_COLOR, 12) > 20


def test_nice_grid_steps():
    assert qs.nice_step(0.07) == pytest.approx(0.1) and qs.nice_step(0.12) == pytest.approx(0.2)
    assert qs.nice_step(0.3) == pytest.approx(0.5) and qs.nice_step(0.6) == pytest.approx(1.0)
    assert qs.nice_step(1.0) == pytest.approx(1.0)


# ---- the dock ----------------------------------------------------------------------------------------------------------------------
def test_the_dock_shows_the_readouts_and_fit_area_restores_the_target(qapp, model):
    dock = qs.SketchDock(model)
    assert dock.btn_fit_area.isEnabled()
    assert "projected vs target" in dock.info.text() and "Snapped edge length" in dock.info.text()
    model.drag_vertex(3, (0.0, 0.9), snap_tolerance=None)  # the area is off now
    assert abs(model.opening.delta) > sk.AREA_TOLERANCE
    assert "#E0A030" in dock.info.text() or "E0A030".lower() in dock.info.text().lower()  # amber: not within 0.5 %
    snapped_before = [model.state.vertices[i] for i in (0, 1)]
    dock.btn_fit_area.click()
    assert abs(model.opening.delta) < sk.AREA_TOLERANCE and "#4CAF50".lower() in dock.info.text().lower()
    assert [model.state.vertices[i] for i in (0, 1)] == snapped_before


def test_the_dock_reports_errors_in_red(qapp, model):
    dock = qs.SketchDock(model)
    model.drag_vertex(0, (model.state.vertices[0][0], -0.8), snap_tolerance=None)
    assert "ERROR" in dock.info.text() and "#F85149".lower() in dock.info.text().lower()


def test_fit_view_button_reframes(qapp, model):
    dock = qs.SketchDock(model)
    dock.canvas.resize(*SIZE)
    dock.canvas.zoom(QtCore.QPointF(5, 5), 30)
    zoomed = dock.canvas.scale
    dock.btn_fit_view.click()
    assert dock.canvas.scale != zoomed


def test_fit_area_is_disabled_before_an_inlet_exists(qapp, scene, example_env):
    cfg = copy.deepcopy(example_env[0])
    cfg["ducts"]["inlet"]["origin"] = cfg["ducts"]["inlet"]["vertices"] = cfg["ducts"]["inlet"]["snapped"] = None
    dock = qs.SketchDock(InletModel(scene, cfg, EXAMPLE_CONFIG, None))
    assert not dock.btn_fit_area.isEnabled() and dock.info.text() == ""
