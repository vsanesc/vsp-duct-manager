# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The 2D duct editors: side / top centreline views and the area-law plot."""
import os

if not os.environ.get("DUCTMGR_GUI_TESTS"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import math

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import cfgutil  # noqa: E402
from ductmgr import qt_paths as qp  # noqa: E402
from ductmgr.explore_model import InletModel  # noqa: E402
from ductmgr.viz import ACCENT, load_scene  # noqa: E402

LEFT, RIGHT = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton
MIDDLE = QtCore.Qt.MouseButton.MiddleButton
SIZE = (700, 340)


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
def model(qapp, scene, example_env, clock):
    _, path, _ = example_env
    return InletModel(scene, copy.deepcopy(example_env[0]), path, None, clock=clock)


def canvas_for(model, view):
    c = qp.PathCanvas(model, view)
    c.resize(*SIZE)
    c.fit_view()
    return c


def pixels(widget):
    img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    return np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3].copy()


def count_colour(arr, hex_colour, tol=24):
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)])
    return int((np.abs(arr.astype(int) - rgb).max(axis=2) <= tol).sum())


def drag(canvas, name, target_hv):
    """Press on a handle, move to the world point ``target_hv`` and release."""
    start = canvas.to_screen(canvas.handles()[name])
    assert canvas.press(start)
    canvas.move(canvas.to_screen(target_hv))
    canvas.release()


# ---- the side and top views --------------------------------------------------------------------------------------------------
def test_the_envelope_of_a_mesh_is_a_closed_outline_of_its_projection():
    pts = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 0], [1, 0, -1.0], [1, 2, 0]], dtype=float)
    env = qp.envelope(pts, 2, bins=4)
    assert env[:, 1].max() == pytest.approx(1.0) and env[:, 1].min() == pytest.approx(-1.0)
    assert env[:, 0].min() < 0.5 and env[:, 0].max() > 1.5


def test_the_views_project_x_z_and_x_y(model):
    side, top = canvas_for(model, "side"), canvas_for(model, "top")
    p = np.array([[1.0, 2.0, 3.0]])
    assert side.project(p)[0] == pytest.approx([1.0, 3.0]) and top.project(p)[0] == pytest.approx([1.0, 2.0])
    with pytest.raises(ValueError):
        qp.PathCanvas(model, "front")


def test_the_handles_are_the_junction_and_four_tangent_handles(model):
    c = canvas_for(model, "side")
    assert set(c.handles()) == {"start", "junction_in", "junction", "junction_out", "end"}
    cl = model.centreline
    assert c.handles()["junction"] == pytest.approx([cl.junction[0], cl.junction[2]])
    assert c.handles()["start"] == pytest.approx([cl.branch.ctrl[1][0], cl.branch.ctrl[1][2]])


def test_dragging_the_junction_in_the_side_view_moves_x_and_z(model, clock):
    c = canvas_for(model, "side")
    clock.advance(10.0)
    drag(c, "junction", (3.6, 0.55))
    assert model.state.duct.centreline.junction == pytest.approx((3.6, 0.55), abs=1e-9)
    assert model.dirty


def test_dragging_the_junction_in_the_top_view_moves_x_only(model, clock):
    c = canvas_for(model, "top")
    z0 = model.centreline.junction[2]
    clock.advance(10.0)
    drag(c, "junction", (3.3, 0.9))  # y is ignored: the junction stays on y = 0
    assert model.state.duct.centreline.junction == pytest.approx((3.3, z0), abs=1e-9)
    assert model.centreline.junction[1] == 0.0


def test_the_inlet_handle_only_slides_along_the_inward_normal(model, clock):
    c = canvas_for(model, "side")
    cl = model.centreline
    d2 = np.array([cl.start_dir[0], cl.start_dir[2]])
    a2 = np.array([cl.start[0], cl.start[2]])
    target = a2 + 0.7 * d2 + np.array([-d2[1], d2[0]]) * 0.2  # 0.7 along, 0.2 off the line
    clock.advance(10.0)
    drag(c, "start", target)
    assert model.state.inlet.start_handle == pytest.approx(float(np.dot(target - a2, d2) / np.dot(d2, d2)))
    after = model.centreline
    assert after.branch.ctrl[1] == pytest.approx(after.start + model.state.inlet.start_handle * after.start_dir)
    assert after.branch.tangents([0.0])[0] == pytest.approx(after.start_dir, abs=1e-12)  # the direction is fixed


def test_the_face_handle_slides_along_the_engine_axis_and_cannot_go_negative(model, clock):
    c = canvas_for(model, "side")
    clock.advance(10.0)
    drag(c, "end", (4.2, 0.0))
    face = model.scene.face_center
    assert model.state.duct.centreline.end_handle == pytest.approx(face[0] - 4.2)
    clock.advance(10.0)
    drag(c, "end", (face[0] + 1.0, 0.0))  # beyond the face: clamped to zero
    assert model.state.duct.centreline.end_handle == 0.0


def test_the_junction_handles_set_angle_and_length_in_the_side_view(model, clock):
    c = canvas_for(model, "side")
    j = np.array([model.centreline.junction[0], model.centreline.junction[2]])
    clock.advance(10.0)
    drag(c, "junction_out", j + np.array([0.3, 0.3]))
    cl = model.state.duct.centreline
    assert cl.junction_pitch_deg == pytest.approx(45.0, abs=1e-6) and cl.junction_out_handle == pytest.approx(0.3 * math.sqrt(2))
    clock.advance(10.0)
    drag(c, "junction_in", j - np.array([0.2, 0.0]))  # the in-handle lies behind the junction, collinear
    cl = model.state.duct.centreline
    assert cl.junction_pitch_deg == pytest.approx(0.0, abs=1e-6) and model.state.inlet.junction_in_handle == pytest.approx(0.2)
    t = model.centreline
    assert t.branch.tangents([1.0])[0] == pytest.approx(t.trunk.tangents([0.0])[0], abs=1e-12)  # still G1


def test_in_the_top_view_the_junction_handles_change_the_length_only(model, clock):
    c = canvas_for(model, "top")
    pitch = model.centreline.resolved.junction_pitch_deg
    clock.advance(10.0)
    drag(c, "junction_out", (model.centreline.junction[0] + 0.5, 0.3))
    cl = model.state.duct.centreline
    assert cl.junction_pitch_deg is None  # the tangent stays in the XZ plane: no angle from the top view
    assert cl.junction_out_handle == pytest.approx(0.5 / math.cos(math.radians(pitch)))


def test_a_handle_cannot_be_dragged_in_a_view_where_its_direction_is_edge_on(model, clock):
    before = model.state
    clock.advance(10.0)
    # a direction that projects to nothing in this view: the handle cannot be placed by a 2D drag
    assert not qp.PathCanvas._slide(model, "start_handle", "k", np.array([1.0, 1.0]), np.zeros(2), np.zeros(2), 1.0)
    assert model.state == before


def test_double_clicking_a_handle_returns_it_to_its_default(model, clock):
    c = canvas_for(model, "side")
    clock.advance(10.0)
    drag(c, "start", (2.5, 1.0))
    clock.advance(10.0)
    drag(c, "junction_out", (4.6, 0.5))
    assert model.state.inlet.start_handle is not None
    assert c.double_click(c.to_screen(c.handles()["start"])) and model.state.inlet.start_handle is None
    assert c.double_click(c.to_screen(c.handles()["junction_out"]))
    cl = model.state.duct.centreline
    assert cl.junction_out_handle is None and cl.junction_pitch_deg is None
    assert not c.double_click(QtCore.QPointF(2, 2))


def test_drags_share_the_model_undo_history(model, clock):
    c = canvas_for(model, "side")
    top = canvas_for(model, "top")
    start = model.state
    for k in range(4):
        clock.advance(0.01)
        c.drag_to("junction", np.array([3.4 + 0.01 * k, 0.4]))
    clock.advance(10.0)
    top.drag_to("junction", np.array([3.9, 0.0]))
    assert model.undo() and model.state.duct.centreline.junction[0] == pytest.approx(3.43)
    assert model.undo() and model.state == start


def test_clicking_empty_space_pans_and_the_wheel_zooms_about_the_cursor(model):
    c = canvas_for(model, "side")
    pos = QtCore.QPointF(200, 120)
    before = c.to_world(pos)
    c.zoom(pos, 3)
    assert c.to_world(pos) == pytest.approx(before) and c.scale > 0
    centre = c.centre.copy()
    assert not c.press(QtCore.QPointF(5, 5))  # nothing there
    assert c.press(QtCore.QPointF(5, 5), MIDDLE)
    c.move(QtCore.QPointF(60, 30))
    c.release()
    assert not np.allclose(c.centre, centre)


def test_the_views_paint_the_centrelines_the_opening_and_the_handles(model):
    for view in ("side", "top"):
        c = canvas_for(model, view)
        arr = pixels(c)
        assert count_colour(arr, qp.CENTRE_COLOR) > 100
        assert count_colour(arr, qp.HANDLE_COLOR) > 20
        assert count_colour(arr, qp.OPENING_COLOR) > 20
    assert count_colour(pixels(canvas_for(model, "top")), qp.MIRROR_COLOR) > 50  # the left branch in the top view


def test_an_unplaced_model_paints_the_scene_without_ducts(qapp, scene, example_env):
    cfg = copy.deepcopy(example_env[0])
    cfgutil.unplace(cfg)
    m = InletModel(scene, cfg, example_env[1], None)
    c = canvas_for(m, "side")
    assert c.handles() == {} and count_colour(pixels(c), qp.CENTRE_COLOR) == 0


# ---- the area law ----------------------------------------------------------------------------------------------------------------
def area_canvas(model):
    c = qp.AreaCanvas(model)
    c.resize(*SIZE)
    return c


def test_the_area_plot_lists_the_interior_points_and_the_two_junction_ends(model):
    c = area_canvas(model)
    a = model.areas
    items = c.items()
    assert len([i for i in items if i[1] == "point"]) == len(a.branch_interior) + len(a.trunk_interior)
    ends = [i for i in items if i[1] == "junction"]
    assert {(e[0], e[3]) for e in ends} == {("branch", 1.0), ("trunk", 0.0)}
    assert next(e for e in ends if e[0] == "trunk")[4] == pytest.approx(next(e for e in ends if e[0] == "branch")[4] * a.n)


def test_dragging_an_interior_point_edits_the_curve_through_the_model(model, clock):
    c = area_canvas(model)
    plot, kind, idx, s, area = next(i for i in c.items() if i[0] == "branch" and i[1] == "point" and i[2] == 1)
    clock.advance(10.0)
    target = c.to_px("branch", 0.55, area * 1.05)
    assert c.press(c.to_px(plot, s, area))
    c.move(target)
    c.release()
    new_s, new_area = model.areas.branch_interior[1]
    assert new_s == pytest.approx(0.55, abs=1e-6) and new_area == pytest.approx(area * 1.05, rel=1e-6)
    assert model.state.inlet.area_branch is not None and model.dirty


def test_dragging_the_junction_end_of_either_curve_sets_the_junction_area(model, clock):
    c = area_canvas(model)
    clock.advance(10.0)
    branch_end = next(i for i in c.items() if i[0] == "branch" and i[1] == "junction")
    c.press(c.to_px("branch", 1.0, branch_end[4]))
    c.move(c.to_px("branch", 1.0, 0.1))  # the branch ends with half the junction area
    c.release()
    assert model.areas.junction_area == pytest.approx(0.2, rel=1e-6)
    clock.advance(10.0)
    trunk_start = next(i for i in c.items() if i[0] == "trunk" and i[1] == "junction")
    c.press(c.to_px("trunk", 0.0, trunk_start[4]))
    c.move(c.to_px("trunk", 0.0, 0.18))
    c.release()
    assert model.areas.junction_area == pytest.approx(0.18, rel=1e-6)
    assert model.areas.branch_area(1.0) == pytest.approx(0.09)


def test_double_click_adds_a_point_and_right_click_removes_one(model, clock):
    c = area_canvas(model)
    n = len(model.areas.trunk_interior)
    clock.advance(10.0)
    pos = c.to_px("trunk", 0.375, float(model.areas.trunk_area(0.375)))
    assert c.double_click(pos) and len(model.areas.trunk_interior) == n + 1
    new = next(p for p in model.areas.trunk_interior if abs(p[0] - 0.375) < 0.02)
    clock.advance(10.0)
    assert c.press(c.to_px("trunk", *new), RIGHT) and len(model.areas.trunk_interior) == n
    assert not c.press(QtCore.QPointF(2, 2), RIGHT)


def test_the_area_plot_marks_a_shrinking_curve_in_red(model, clock):
    c = area_canvas(model)
    good = pixels(c)
    clock.advance(10.0)
    model.area_move_point("branch", 1, 0.5, 0.03)  # a deep dip
    bad = pixels(c)
    assert count_colour(bad, qp.BAD_COLOR) > count_colour(good, qp.BAD_COLOR)
    assert count_colour(good, ACCENT) > 100


def test_the_area_plot_asks_to_place_the_inlet_first(qapp, scene, example_env):
    cfg = copy.deepcopy(example_env[0])
    cfgutil.unplace(cfg)
    m = InletModel(scene, cfg, example_env[1], None)
    c = area_canvas(m)
    assert not c.press(QtCore.QPointF(100, 100)) and not c.double_click(QtCore.QPointF(100, 100))
    pixels(c)  # paints without error
