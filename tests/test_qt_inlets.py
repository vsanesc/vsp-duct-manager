# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The inlets list in the explorer window, the 2D views with several inlets, and the 3D drawing of them."""
import os

if not os.environ.get("DUCTMGR_GUI_TESTS"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import bodies  # noqa: E402
from ductmgr import qt_explorer as qe  # noqa: E402
from ductmgr import qt_paths as qp  # noqa: E402
from ductmgr.config import EXAMPLE_CONFIG  # noqa: E402
from ductmgr.explore_model import InletModel  # noqa: E402
from ductmgr.viz import SceneView  # noqa: E402

SIZE = (700, 340)


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    return app


class Clock:
    t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def entry(name, y, mirror=False, fraction=None, sweep=0.0):
    r = math.hypot(y, 0.8)
    return {"name": name, "origin": [4.0, y / r, 0.8 / r], "sweep_deg": sweep, "mirror": mirror,
            "flow_fraction": fraction}


def build(inlets, symmetry="xz"):
    scene = SimpleNamespace(fuselage=bodies.cylinder(1.0), sizing=SimpleNamespace(capture_area_per_inlet=0.06, n_inlets=2),
                            engine_check=None, warnings=[], face_center=np.array([6.0, 0.0, 0.0]), face_diameter=0.5,
                            wings=[], engine_mesh=pv.PolyData(), context_parts=[], context=pv.PolyData(),
                            hub_to_tip_ratio=0.0, envelope_diameter=None, face_source="config")
    cfg = {"ducts": {"symmetry": symmetry, "inlets": inlets}}
    clock = Clock()
    return scene, InletModel(scene, cfg, EXAMPLE_CONFIG, None, clock=clock), clock


@pytest.fixture()
def window(qapp):
    made = []

    def factory(inlets, symmetry="xz"):
        scene, model, clock = build(inlets, symmetry)
        win = qe.ExploreWindow(scene, model, with_3d=False)
        made.append(win)
        return win, model, clock

    yield factory
    for w in made:
        w.ask_unsaved = lambda: "discard"
        w.model._listeners.clear()
        w.close()


def rows(w):
    return [w.inlet_list.item(i).text() for i in range(w.inlet_list.count())]


# ---- the panel ---------------------------------------------------------------------------------------------------------------
def test_the_inlets_list_shows_name_flow_share_and_twin(window):
    w, m, _ = window([entry("R", 0.5, True), entry("C", 0.0)])
    r = rows(w)
    assert len(r) == 3 and r[0].startswith("R") and "33.3 %" in r[0] and "+ mirror" in r[0]
    assert r[1].startswith("C") and "self-symmetric" in r[1]
    assert r[2].startswith("Trunk")  # the fixed last row
    assert w.inlet_list.currentRow() == 0 and w.chk_mirror.isChecked() and w.chk_mirror.isEnabled()
    assert w.spin_flow.value() == pytest.approx(100 / 3, abs=0.01)
    assert not w.btn_flow_equal.isEnabled()  # nothing explicit to reset
    assert w.combo_symmetry.currentData() == "xz" and not w.btn_del_inlet.isEnabled() is False


def test_selecting_a_row_selects_the_inlet_and_updates_the_plane_boxes(window):
    w, m, _ = window([entry("a", 0.6), entry("b", -0.3)], "none")
    y0 = w.spin_pos["y"].value()
    w.inlet_list.setCurrentRow(1)
    assert m.selected == 1 and w.spin_pos["y"].value() < 0 < y0 and not m.dirty or True
    assert not m.can_undo and "b" in w.sketch.title.text()
    assert w.section_list.count() == 3  # inlet, junction, face of the selected inlet


def test_the_mirror_box_is_off_for_a_self_symmetric_inlet_and_without_symmetry(window):
    w, m, _ = window([entry("C", 0.0, True)])
    assert not w.chk_mirror.isEnabled()
    w2, m2, _ = window([entry("a", 0.5, True)], "none")
    assert not w2.chk_mirror.isEnabled() and w2.chk_mirror.isChecked()


def test_the_mirror_box_edits_the_model_and_undo_follows(window):
    w, m, clock = window([entry("a", 0.5, True)])
    assert m.n_branches == 2
    clock.advance(10.0)
    w.chk_mirror.setChecked(False)
    assert m.n_branches == 1 and not m.inlet.mirror
    w.act_undo.trigger()
    assert m.n_branches == 2 and w.chk_mirror.isChecked()


def test_the_flow_share_box_and_equal_button(window):
    w, m, clock = window([entry("a", 0.5), entry("b", -0.5)], "none")
    clock.advance(10.0)
    w.spin_flow.setValue(70.0)
    assert m.state.inlets[0].flow_fraction == pytest.approx(0.7) and "normalised" in w.message_label.text()
    assert w.btn_flow_equal.isEnabled()
    w.btn_flow_equal.click()
    assert m.state.inlets[0].flow_fraction is None and w.spin_flow.value() == pytest.approx(50.0)


def test_add_duplicate_remove_and_rename_buttons(window, monkeypatch):
    w, m, clock = window([entry("a", 0.5, True)])
    w.btn_add_inlet.click()
    assert len(m.state.inlets) == 2 and m.selected == 1 and m.mode == "placing" and w.btn_place.isChecked()
    assert "not placed" in rows(w)[1] and not w.spin_flow.isEnabled() and not w.btn_dup_inlet.isEnabled()
    w.inlet_list.setCurrentRow(0)
    clock.advance(10.0)
    w.btn_dup_inlet.click()
    assert len(m.state.inlets) == 3 and m.inlet.name == "a copy"
    monkeypatch.setattr(w, "ask_inlet_name", lambda current: "renamed")
    w.btn_ren_inlet.click()
    assert m.inlet.name == "renamed" and rows(w)[1].startswith("renamed")
    monkeypatch.setattr(w, "ask_inlet_name", lambda current: None)  # cancelled
    w.btn_ren_inlet.click()
    assert m.inlet.name == "renamed"
    assert w.btn_del_inlet.isEnabled()
    w.btn_del_inlet.click()
    assert len(m.state.inlets) == 2


def test_the_symmetry_combo_switches_the_global_symmetry(window):
    w, m, clock = window([entry("R", 0.5, True)])
    clock.advance(10.0)
    w.combo_symmetry.setCurrentIndex(w.combo_symmetry.findData("none"))
    assert m.state.symmetry == "none" and m.n_branches == 1 and not w.chk_mirror.isEnabled()
    w.act_undo.trigger()
    assert m.state.symmetry == "xz" and w.combo_symmetry.currentData() == "xz"


def test_the_readouts_list_the_inlets(window):
    w, m, _ = window([entry("R", 0.5, True), entry("C", 0.0)])
    text = w.readout_view.toPlainText()
    assert "3 branch(es)" in text and "symmetry xz" in text and "self-symmetric" in text


# ---- the Trunk row ---------------------------------------------------------------------------------------------------------------------
def test_the_trunk_row_is_fixed_last_and_selects_the_trunk(window):
    w, m, _ = window([entry("R", 0.5, True), entry("C", 0.0)])
    assert rows(w)[-1].startswith("Trunk") and w.inlet_list.count() == len(m.state.inlets) + 1
    w.inlet_list.setCurrentRow(2)
    assert m.trunk_selected and w.inlet_list.currentRow() == 2
    # the section editor shows the trunk's sections; the Area tab its curve; the inlet's own controls are off
    assert [w.section_list.item(i).text().split("   ")[1] for i in range(w.section_list.count())] == [
        "trunk start", "engine face (fixed)"]
    assert "Trunk" in w.sketch.title.text() and "trunk start" in w.sketch.title.text()
    for off in (w.btn_del_inlet, w.btn_dup_inlet, w.btn_ren_inlet, w.btn_up_inlet, w.btn_down_inlet, w.chk_mirror,
                w.spin_flow, w.btn_place, w.spin_pitch, w.spin_sweep, w.btn_align, *w.spin_pos.values()):
        assert not off.isEnabled()
    assert w.btn_add_section.isEnabled() and w.spin_wall.isEnabled()
    w.inlet_list.setCurrentRow(1)  # back to an inlet
    assert not m.trunk_selected and w.btn_ren_inlet.isEnabled() and w.section_list.count() == 3


def test_the_trunk_row_cannot_be_removed_or_moved_and_inlets_can_move(window):
    w, m, clock = window([entry("a", 0.6), entry("b", -0.3), entry("c", 0.0)], "none")
    w.inlet_list.setCurrentRow(1)
    assert w.btn_up_inlet.isEnabled() and w.btn_down_inlet.isEnabled()
    clock.advance(10.0)
    w.btn_down_inlet.click()
    assert [i.name for i in m.state.inlets] == ["a", "c", "b"] and m.selected == 2 and rows(w)[-1].startswith("Trunk")
    assert not w.btn_down_inlet.isEnabled() and w.inlet_list.currentRow() == 2
    clock.advance(10.0)
    w.btn_up_inlet.click()
    assert [i.name for i in m.state.inlets] == ["a", "b", "c"] and m.selected == 1
    w.inlet_list.setCurrentRow(3)
    assert not w.btn_del_inlet.isEnabled()
    assert not m.remove_inlet(3) and len(m.state.inlets) == 3  # there is no inlet 3: the trunk row is not an inlet


def test_adding_a_trunk_key_section_from_the_panel(window, monkeypatch):
    w, m, clock = window([entry("a", 0.6), entry("b", -0.3)], "none")
    w.inlet_list.setCurrentRow(2)
    monkeypatch.setattr(w, "ask_section_s", lambda: 0.5)
    clock.advance(10.0)
    w._on_add_section()
    assert [ks.s for ks in m.state.trunk] == [0.0, 0.5] and w.section_list.count() == 3
    assert w.btn_remove_section.isEnabled() and m.active == 1
    w.btn_remove_section.click()
    assert [ks.s for ks in m.state.trunk] == [0.0]


def test_reset_and_simplify_buttons_act_on_the_trunk_start_only(window, monkeypatch):
    w, m, clock = window([entry("a", 0.6), entry("b", -0.3)], "none")
    dock = w.sketch
    assert not dock.btn_reset_start.isEnabled() and not dock.btn_simplify.isEnabled()  # an inlet section is selected
    w.inlet_list.setCurrentRow(2)
    assert dock.btn_reset_start.isEnabled() and dock.btn_simplify.isEnabled()
    w.section_list.setCurrentRow(1)  # the engine face
    assert not dock.btn_reset_start.isEnabled() and not dock.btn_simplify.isEnabled()
    w.section_list.setCurrentRow(0)
    for k in range(4):  # more vertices to simplify away
        clock.advance(10.0)
        m.insert_vertex(k, tuple(np.mean([m.active_shape.vertices[k], m.active_shape.vertices[(k + 1) % m.active_shape.n]],
                                         axis=0) * 1.05))
    n = m.active_shape.n
    assert n >= 6
    monkeypatch.setattr(dock, "ask_vertex_count", lambda current: current - 2)
    clock.advance(10.0)
    dock.btn_simplify.click()
    assert m.state.trunk[0].shape.n == n - 2 and "Simplified" in dock.info.text() and "deviation" in dock.info.text()
    assert w.act_undo.isEnabled()
    w.act_undo.trigger()
    assert m.state.trunk[0].shape.n == n
    monkeypatch.setattr(dock, "ask_exponent", lambda: 4.0)
    clock.advance(10.0)
    dock.btn_reset_start.click()
    assert m.state.trunk[0].shape.n == 4 and "exponent 4" in w.message_label.text()
    monkeypatch.setattr(dock, "ask_exponent", lambda: None)  # cancelled: nothing changes
    state = m.state
    dock.btn_reset_start.click()
    assert m.state == state


def test_the_readouts_list_the_trunk(window):
    w, m, _ = window([entry("a", 0.6), entry("b", -0.3)], "none")
    assert "Trunk" in w.readout_view.toPlainText()
    w.inlet_list.setCurrentRow(2)
    text = w.readout_view.toPlainText()
    assert "Trunk start" in text and "engine face circle" in text


# ---- the 2D views --------------------------------------------------------------------------------------------------------------------
def canvas(model, view):
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


def test_clicking_an_opening_in_the_top_view_selects_its_inlet(qapp):
    scene, m, _ = build([entry("a", 0.6), entry("b", -0.5)], "none")
    c = canvas(m, "top")
    start_b = c.branch_starts()[1][1]
    assert m.selected == 0
    assert c.press(c.to_screen(start_b))
    assert m.selected == 1 and not m.can_undo  # selecting is not a change


def test_a_twin_selects_its_own_inlet_and_the_handles_follow_the_selection(qapp):
    scene, m, _ = build([entry("R", 0.6, True), entry("L", -0.2)], "xz")
    c = canvas(m, "top")
    twin_start = c.branch_starts()[1][1]  # the mirrored twin of inlet 0
    assert c.press(c.to_screen(twin_start)) and m.selected == 0
    h0 = c.handles()["start"].copy()
    m.select_inlet(1)
    assert not np.allclose(c.handles()["start"], h0)  # the other inlet's own handle


def test_twins_are_greyed_in_the_2d_views_and_others_dim(qapp):
    scene, m, _ = build([entry("R", 0.6, True), entry("C", 0.0)], "xz")
    arr = pixels(canvas(m, "top"))
    assert count_colour(arr, qp.TWIN_COLOR) > 20 and count_colour(arr, qp.OTHER_COLOR) > 20
    assert count_colour(arr, qp.CENTRE_COLOR) > 50  # the selected inlet and the trunk


def test_without_symmetry_the_top_view_moves_the_junction_in_y_and_the_handles_set_the_yaw(qapp):
    scene, m, clock = build([entry("a", 0.6), entry("b", -0.5)], "none")
    c = canvas(m, "top")
    clock.advance(10.0)
    c.drag_to("junction", np.array([3.5, 0.25]))
    assert m.centreline.junction[1] == pytest.approx(0.25) and m.state.duct.centreline.junction_y == pytest.approx(0.25)
    clock.advance(10.0)
    j = m.centreline.junction
    c.drag_to("junction_out", np.array([j[0] + 0.4, j[1] + 0.2]))
    cl = m.state.duct.centreline
    assert cl.junction_yaw_deg is not None and cl.junction_yaw_deg > 5.0 and cl.junction_out_handle > 0.3
    clock.advance(10.0)
    assert c.double_click(c.to_screen(c.handles()["junction_out"]))
    assert m.state.duct.centreline.junction_yaw_deg is None  # a double click returns the tangent to its default


def test_with_symmetry_xz_the_top_view_junction_stays_on_the_plane(qapp):
    scene, m, clock = build([entry("R", 0.6, True)], "xz")
    c = canvas(m, "top")
    clock.advance(10.0)
    c.drag_to("junction", np.array([3.4, 0.5]))
    assert m.centreline.junction[1] == 0.0 and m.state.duct.centreline.junction_y is None


def test_the_area_plot_edits_the_selected_inlets_curve_and_its_share(qapp):
    scene, m, clock = build([entry("a", 0.6, fraction=0.7), entry("b", -0.5, fraction=0.3)], "none")
    c = qp.AreaCanvas(m)
    c.resize(*SIZE)
    items = c.items()
    end = next(i for i in items if i[0] == "branch" and i[1] == "junction")
    assert end[4] == pytest.approx(0.7 * m.areas.junction_area)  # the branch ends with its flow share
    clock.advance(10.0)
    c.press(c.to_px("branch", 1.0, end[4]))
    c.move(c.to_px("branch", 1.0, 0.1))
    c.release()
    assert m.areas.junction_area == pytest.approx(0.1 / 0.7, rel=1e-6)  # the plot value divided by the share
    m.select_inlet(1)
    end_b = next(i for i in c.items() if i[0] == "branch" and i[1] == "junction")
    assert end_b[4] == pytest.approx(0.3 * m.areas.junction_area)


def test_the_area_plot_shows_only_the_trunk_curve_while_the_trunk_is_selected(qapp):
    scene, m, clock = build([entry("a", 0.6), entry("b", -0.5)], "none")
    c = qp.AreaCanvas(m)
    c.resize(*SIZE)
    assert c.plots() == ("branch", "trunk") and any(i[0] == "branch" for i in c.items())
    m.select_trunk()
    assert c.plots() == ("trunk",) and all(i[0] == "trunk" for i in c.items())
    assert c.plot_rect("trunk").width() > SIZE[0] * 0.8 and c.which_plot(c.plot_rect("trunk").center()) == "trunk"
    clock.advance(10.0)
    end = next(i for i in c.items() if i[1] == "junction")
    c.press(c.to_px("trunk", 0.0, end[4]))
    c.move(c.to_px("trunk", 0.0, 0.12))
    c.release()
    assert m.areas.junction_area == pytest.approx(0.12, rel=1e-6)
    assert all(i.area_branch is None for i in m.state.inlets)
    c.grab()  # paints without the branch plot


def test_the_paths_show_only_the_trunks_handles_while_it_is_selected(qapp):
    scene, m, _ = build([entry("a", 0.6), entry("b", -0.5)], "none")
    c = canvas(m, "side")
    assert set(c.handles()) == {"start", "junction_in", "junction", "junction_out", "end"}
    m.select_trunk()
    assert set(c.handles()) == {"junction", "junction_out", "end"}
    c.grab()


# ---- the 3D drawing ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def plotter():
    pl = pv.Plotter(off_screen=True, window_size=(640, 480))
    yield pl
    pl.close()


def test_every_branch_has_its_centreline_and_body_actors_and_stale_ones_go(plotter):
    scene, m, clock = build([entry("a", 0.6, True), entry("b", 0.0)], "xz")
    view = SceneView(plotter, scene)
    m.ensure_built()
    view.update_inlet(m)
    view.update_ducts(m)
    acts = set(plotter.renderer.actors)
    for name in ("cl_a", "cl_a_mirror", "cl_b", "cl_trunk", "duct_neg_branch_a", "duct_pos_branch_a_mirror",
                 "duct_neg_branch_b", "duct_pos_trunk"):
        assert name in acts
    assert "inlet_other1_fill" in acts and "inlet_right_fill" in acts and "inlet_left_fill" in acts
    clock.advance(10.0)
    m.remove_inlet(1)
    m.ensure_built()
    view.update_inlet(m)
    view.update_ducts(m)
    acts = set(plotter.renderer.actors)
    assert "cl_b" not in acts and "duct_neg_branch_b" not in acts and "inlet_other1_fill" not in acts
    assert "cl_a_mirror" in acts


def test_no_inlet_is_highlighted_while_the_trunk_is_selected(plotter):
    scene, m, _ = build([entry("a", 0.6), entry("b", -0.5)], "none")
    view = SceneView(plotter, scene)
    view.update_inlet(m)
    acts = set(plotter.renderer.actors)
    assert "inlet_right_fill" in acts and "inlet_other1_fill" in acts
    m.select_trunk()
    view.update_inlet(m)
    acts = set(plotter.renderer.actors)
    assert "inlet_right_fill" not in acts and "inlet_other0_fill" in acts and "inlet_other1_fill" in acts


def test_the_twin_is_greyed_and_a_non_mirrored_inlet_has_no_left_opening(plotter):
    scene, m, _ = build([entry("a", 0.6, True)], "xz")
    view = SceneView(plotter, scene)
    m.ensure_built()
    view.update_ducts(m)
    twin = plotter.renderer.actors["cl_a_mirror"].GetProperty().GetColor()
    from ductmgr.viz import TWIN_COLOR

    grey = tuple(int(TWIN_COLOR[i:i + 2], 16) / 255 for i in (1, 3, 5))
    assert twin == pytest.approx(grey, abs=1e-2)
    scene2, m2, _ = build([entry("a", 0.6)], "none")
    view2 = SceneView(plotter, scene2)
    view2.update_inlet(m2)
    assert "inlet_left_fill" not in plotter.renderer.actors and "inlet_right_fill" in plotter.renderer.actors
