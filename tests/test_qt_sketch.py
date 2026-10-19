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
    for i, v in enumerate(model.state.sections[0].shape.vertices):
        p = at(canvas, v)
        assert canvas.hit_vertex(p) == i
        assert canvas.hit_vertex(QtCore.QPointF(p.x() + qs.HANDLE_RADIUS_PX - 1, p.y())) == i
        assert canvas.hit_vertex(QtCore.QPointF(p.x() + qs.HANDLE_RADIUS_PX + 3, p.y())) is None
    assert canvas.hit_vertex(QtCore.QPointF(2, 2)) is None


# ---- dragging and snapping ---------------------------------------------------------------------------------------------
def test_dragging_a_vertex_moves_it_to_the_cursor(canvas, model, clock):
    start = model.state
    idx = 3  # the apex: free
    assert canvas.press(at(canvas, model.state.sections[0].shape.vertices[idx]), alt=False)
    target = np.array(model.state.sections[0].shape.vertices[idx]) + [0.02, 0.05]  # well above the skin
    canvas.move(at(canvas, target), alt=False)
    assert np.array(model.state.sections[0].shape.vertices[idx]) == pytest.approx(target, abs=1e-9)
    assert model.state.sections[0].shape.snapped[idx] is False
    canvas.release()
    canvas.move(at(canvas, target + [0.1, 0.1]), alt=False)  # released: moving does nothing
    assert np.array(model.state.sections[0].shape.vertices[idx]) == pytest.approx(target, abs=1e-9)
    assert model.undo() and model.state == start  # the whole drag is one step on the shared undo stack


def test_pressing_on_empty_space_drags_nothing(canvas, model):
    before = model.state
    assert not canvas.press(QtCore.QPointF(3, 3))
    canvas.move(QtCore.QPointF(100, 100))
    assert model.state == before


def test_a_vertex_dropped_near_the_curve_snaps_on_and_marks_itself(canvas, model):
    idx = 3
    _, _, _ = nearest_on_section(model.section, model.state.sections[0].shape.vertices[idx])
    foot = nearest_on_section(model.section, (0.0, 0.2))[0]  # a skin point under the apex
    canvas.press(at(canvas, model.state.sections[0].shape.vertices[idx]))
    # 5 pixels from the curve: inside the 12 px snap radius
    offset = 5.0 / canvas.scale
    canvas.move(at(canvas, foot + [0.0, offset]), alt=False)
    canvas.release()
    assert model.state.sections[0].shape.snapped[idx] is True
    assert distance_to_section(model.section, np.array(model.state.sections[0].shape.vertices[idx]))[0] < 1e-6  # exactly on the curve


def test_the_snap_distance_is_a_pixel_tolerance(canvas, model):
    assert canvas.snap_tolerance(False) == pytest.approx(qs.SNAP_PIXELS / canvas.scale)
    assert canvas.snap_tolerance(True) is None  # Alt
    s0 = canvas.snap_tolerance(False)
    canvas.zoom(QtCore.QPointF(300, 200), 3)  # zoomed in: the same pixels are fewer metres
    assert canvas.snap_tolerance(False) < s0


def test_dropping_further_than_the_tolerance_does_not_snap_and_unsnaps(canvas, model, clock):
    snapped_idx = 0
    assert model.state.sections[0].shape.snapped[snapped_idx]
    canvas.press(at(canvas, model.state.sections[0].shape.vertices[snapped_idx]))
    away = np.array(model.state.sections[0].shape.vertices[snapped_idx]) + [0.0, (qs.SNAP_PIXELS + 25) / canvas.scale]  # drag it away
    canvas.move(at(canvas, away), alt=False)
    canvas.release()
    assert model.state.sections[0].shape.snapped[snapped_idx] is False  # unsnapped
    assert np.array(model.state.sections[0].shape.vertices[snapped_idx]) == pytest.approx(away, abs=1e-9)
    clock.advance(5.0)
    # and dropping a free vertex far from the curve stays free
    canvas.press(at(canvas, model.state.sections[0].shape.vertices[3]))
    canvas.move(at(canvas, np.array(model.state.sections[0].shape.vertices[3]) + [0.0, 40 / canvas.scale]))
    canvas.release()
    assert model.state.sections[0].shape.snapped[3] is False


def test_alt_disables_snapping(canvas, model):
    idx = 3
    foot = nearest_on_section(model.section, (0.0, 0.2))[0]
    spot = foot + [0.0, 3.0 / canvas.scale]  # 3 px from the curve: would snap without Alt
    canvas.press(at(canvas, model.state.sections[0].shape.vertices[idx]))
    canvas.move(at(canvas, spot), alt=True)
    canvas.release()
    assert model.state.sections[0].shape.snapped[idx] is False
    assert np.array(model.state.sections[0].shape.vertices[idx]) == pytest.approx(spot, abs=1e-9)


def test_snapping_two_neighbours_makes_their_edge_follow_the_curve(canvas, model):
    # vertices 0 and 1 are snapped, so edge 0 follows the curve; snap vertex 2 too and edge 1 (1 -> 2) follows it
    assert model.opening.snapped_edges == (True, False, False, False, False)
    foot = nearest_on_section(model.section, (0.2, 0.0))[0]
    canvas.press(at(canvas, model.state.sections[0].shape.vertices[2]))
    canvas.move(at(canvas, foot + [0.0, 2.0 / canvas.scale]), alt=False)
    canvas.release()
    assert model.state.sections[0].shape.snapped[2] is True
    assert model.opening.snapped_edges == (True, True, False, False, False)
    assert model.opening.snapped_length > 0


def test_dragging_a_vertex_across_the_body_invalidates_the_opening(canvas, model):
    canvas.press(at(canvas, model.state.sections[0].shape.vertices[3]))
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
    canvas.press(at(canvas, model.state.sections[0].shape.vertices[3]))
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
    scene = SimpleNamespace(fuselage=prism, sizing=SimpleNamespace(capture_area_per_inlet=0.05, n_inlets=2),
                            engine_check=None, warnings=[], face_center=None, face_diameter=0.5)
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
    snapped_before = [model.state.sections[0].shape.vertices[i] for i in (0, 1)]
    dock.btn_fit_area.click()
    assert abs(model.opening.delta) < sk.AREA_TOLERANCE and "#4CAF50".lower() in dock.info.text().lower()
    assert [model.state.sections[0].shape.vertices[i] for i in (0, 1)] == snapped_before


def test_the_dock_reports_errors_in_red(qapp, model):
    dock = qs.SketchDock(model)
    model.drag_vertex(0, (model.state.sections[0].shape.vertices[0][0], -0.8), snap_tolerance=None)
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


# ---- Phase 3a: general section editor -----------------------------------------------------------------------
RIGHT = QtCore.Qt.MouseButton.RightButton


def shape_of(model):
    return model.active_shape


def mid_of_edge(canvas, edge):
    """Screen position in the middle of an edge's drawn path."""
    path = canvas.model.active_opening.edge_paths[edge]
    if len(path) == 2:  # a straight edge is just its two end points
        return canvas.to_screen((path[0] + path[1]) / 2)
    return canvas.to_screen(path[len(path) // 2])


def test_double_clicking_a_free_edge_inserts_a_vertex_there(canvas, model, clock):
    before = shape_of(model)
    assert not before.edge_is_snapped(2)
    clock.advance(10.0)
    at_edge = mid_of_edge(canvas, 2)
    index = canvas.double_click(at_edge)
    assert index == 3 and shape_of(model).n == before.n + 1
    assert canvas.selected_vertex == 3
    new = canvas.to_screen(shape_of(model).vertices[3])
    assert abs(new.x() - at_edge.x()) < 2.0 and abs(new.y() - at_edge.y()) < 2.0  # where it was clicked (on the edge)
    assert model.valid


def test_double_clicking_empty_space_or_a_vertex_inserts_nothing(canvas, model):
    n = shape_of(model).n
    assert canvas.double_click(QtCore.QPointF(5, 5)) is None
    assert canvas.double_click(at(canvas, shape_of(model).vertices[3])) is None
    assert shape_of(model).n == n and not model.can_undo


def test_double_clicking_a_snapped_edge_inserts_a_snapped_vertex_on_the_skin(canvas, model):
    assert shape_of(model).edge_is_snapped(0)
    index = canvas.double_click(mid_of_edge(canvas, 0))
    assert index == 1 and shape_of(model).snapped[1] is True
    assert distance_to_section(model.section, np.array(shape_of(model).vertices[1]))[0] < 1e-6


def test_right_click_on_a_vertex_removes_it_and_on_empty_space_pans(canvas, model, clock):
    n = shape_of(model).n
    clock.advance(10.0)
    assert canvas.press(at(canvas, shape_of(model).vertices[4]), button=RIGHT)
    assert shape_of(model).n == n - 1 and canvas.selected_vertex is None
    centre = canvas.centre.copy()
    canvas.press(QtCore.QPointF(5, 5), button=RIGHT)
    canvas.move(QtCore.QPointF(45, 25))
    canvas.release()
    assert not np.allclose(canvas.centre, centre) and shape_of(model).n == n - 1  # a pan, not an edit


def test_the_delete_key_removes_the_selected_vertex_but_never_below_three(canvas, model, clock):
    canvas.press(at(canvas, shape_of(model).vertices[3]))
    canvas.release()
    assert canvas.selected_vertex == 3
    clock.advance(10.0)
    assert canvas.delete_selected() and shape_of(model).n == 4
    canvas.selected_vertex = 2
    clock.advance(10.0)
    assert canvas.delete_selected() and shape_of(model).n == 3
    canvas.selected_vertex = 0
    assert not canvas.delete_selected() and shape_of(model).n == 3  # the model refuses the 3rd -> 2nd


def test_a_selected_edge_can_be_made_curved_and_its_handles_dragged(canvas, model, clock):
    canvas.press(mid_of_edge(canvas, 3))
    canvas.release()
    assert canvas.selected_edge == 3 and canvas.selected_vertex is None
    clock.advance(10.0)
    model.set_edge_type(3, "curved")
    handles = canvas.handle_positions()
    assert set(handles) == {(3, 0), (3, 1)}  # a free curved edge shows two handles
    target = at(canvas, np.array(shape_of(model).control_points(3)[1]) + [0.03, 0.05])
    clock.advance(10.0)
    assert canvas.press(handles[(3, 0)])
    canvas.move(target)
    canvas.release()
    ctrl = shape_of(model).control_points(3)
    assert canvas.to_screen(ctrl[1]).x() == pytest.approx(target.x(), abs=1e-6)
    assert canvas.to_screen(ctrl[1]).y() == pytest.approx(target.y(), abs=1e-6)


def test_snapped_and_straight_edges_have_no_handles(canvas, model):
    assert canvas.handle_positions() == {}
    model.set_edge_type(3, "curved")
    assert all(e == 3 for e, _ in canvas.handle_positions())  # the snapped edge 0 shows none


def test_the_continuity_buttons_follow_the_selection(qapp, model, clock):
    dock = qs.SketchDock(model)
    c = dock.canvas
    c.resize(*SIZE)
    c.fit_view()
    assert not dock.btn_smooth.isEnabled() and not dock.btn_curved.isEnabled()  # nothing selected
    c.press(at(c, shape_of(model).vertices[3]))
    c.release()
    assert dock.btn_smooth.isEnabled() and dock.btn_symmetric.isEnabled() and not dock.btn_corner.isEnabled()
    clock.advance(10.0)
    dock.btn_smooth.click()
    assert shape_of(model).continuity[3] == "smooth" and not dock.btn_smooth.isEnabled()
    c.press(mid_of_edge(c, 2))
    c.release()
    assert dock.btn_curved.isEnabled() and not dock.btn_straight.isEnabled()
    clock.advance(10.0)
    dock.btn_curved.click()
    assert shape_of(model).edge_is_curved(2) and dock.btn_straight.isEnabled()
    c.press(mid_of_edge(c, 0))  # a snapped edge cannot be edited
    c.release()
    assert not dock.btn_curved.isEnabled() and not dock.btn_straight.isEnabled()


def test_delete_button_is_off_at_three_vertices(qapp, model, clock):
    dock = qs.SketchDock(model)
    for _ in range(2):
        clock.advance(10.0)
        model.delete_vertex(4 if model.active_shape.n == 5 else 3)
    assert model.active_shape.n == 3
    dock.canvas.selected_vertex = 1
    dock.refresh()
    assert not dock.btn_delete.isEnabled()


def test_the_selection_is_cleared_when_the_vertex_count_changes(canvas, model, clock):
    canvas.selected_vertex = 4
    clock.advance(10.0)
    model.insert_vertex(2, np.array(shape_of(model).vertices[2]) * 0.5 + np.array(shape_of(model).vertices[3]) * 0.5)
    assert canvas.selected_vertex is None


def test_markers_differ_by_continuity_and_a_curved_outline_is_painted(canvas, model, clock):
    canvas.resize(*SIZE)
    plain = count_colour(pixels(canvas), "#E6EDF3")
    clock.advance(10.0)
    model.set_edge_type(3, "curved")
    clock.advance(10.0)
    model.drag_handle(3, 0, np.array(shape_of(model).control_points(3)[1]) + [0.05, 0.12])
    model.set_continuity(3, "symmetric")
    arr = pixels(canvas)
    assert count_colour(arr, "#58A6FF") > 20  # the Bézier handles and tangent lines
    assert count_colour(arr, "#E6EDF3") != plain


def test_a_key_section_has_no_skin_and_no_snapping(qapp, model, clock):
    model.add_key_section(0.4)
    c = qs.SketchCanvas(model)
    c.resize(*SIZE)
    assert model.active_curve is None and c.editable
    v = np.array(shape_of(model).vertices[1])
    clock.advance(10.0)
    c.press(c.to_screen(v))
    c.move(c.to_screen(v + [0.002, 0.003]))
    c.release()
    assert not any(shape_of(model).snapped)  # nothing to snap to
    assert model.active_opening.area_plane == pytest.approx(model.area_at(0.4), rel=1e-9)  # resized at drag end


def test_the_engine_face_is_shown_fixed_and_cannot_be_edited(qapp, model):
    model.select_section(model.n_sections)
    c = qs.SketchCanvas(model)
    c.resize(*SIZE)
    assert not c.editable and c.handle_positions() == {}
    before = model.state
    assert not c.press(at(c, shape_of(model).vertices[0])) and not c.double_click(mid_of_edge(c, 0))
    assert model.state == before
    assert count_colour(pixels(c), qs.FIXED_COLOR) > 100  # the circle is drawn


def test_switching_sections_reframes_the_view_and_clears_the_selection(qapp, model):
    c = qs.SketchCanvas(model)
    c.resize(*SIZE)
    c.selected_vertex = 2
    model.add_key_section(0.5)
    assert c.selected_vertex is None
    model.select_section(model.n_sections)
    assert c.selected_vertex is None and c.selected_edge is None


def test_the_dock_title_names_the_active_section(qapp, model):
    dock = qs.SketchDock(model)
    assert "s = 0.000" in dock.title.text() and "inlet" in dock.title.text()
    model.add_key_section(0.4)
    assert "s = 0.400" in dock.title.text() and "A(s)" in dock.title.text()
    assert "A(s)" in dock.info.text()
    model.select_section(model.n_sections)  # the fixed junction section
    assert "junction" in dock.title.text() and not dock.btn_fit_area.isEnabled()
    model.select_section(model.n_sections + 1)
    assert "engine face" in dock.title.text() and not dock.btn_fit_area.isEnabled()
