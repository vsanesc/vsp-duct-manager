# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The Qt window without its 3D view, on Qt's offscreen platform (the 3D view needs a real display; see
test_qt_gui_3d.py). Everything that is a rule lives in the model; these tests check that the window shows
it and forwards to it correctly."""
import os

if not os.environ.get("DUCTMGR_GUI_TESTS"):  # the opt-in GUI tests need the real platform
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import bodies  # noqa: E402
from ductmgr import qt_explorer as qe  # noqa: E402
from ductmgr import qt_sketch as qs  # noqa: E402
from ductmgr.config import EXAMPLE_CONFIG, load_config  # noqa: E402
from ductmgr.explore_model import InletModel, Readout  # noqa: E402
from ductmgr.section import sweep_for_direction  # noqa: E402
from ductmgr.viz import ACCENT, BACKGROUND, load_scene  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    return app


@pytest.fixture(scope="module")
def scene(example_env):
    cfg, path, _ = example_env
    return load_scene(cfg, path)


@pytest.fixture()
def writable(example_env, tmp_path):
    _, path, _ = example_env
    target = tmp_path / "duct_config.yaml"
    shutil.copyfile(path, target)
    return target


@pytest.fixture()
def make_window(qapp, scene, example_env):
    windows = []

    def factory(save_target="none", cfg=None, config_path=None, the_scene=None):
        _, path, _ = example_env
        path = config_path or path
        cfg = cfg if cfg is not None else load_config(path)[0]
        target = None if save_target == "none" else save_target
        sc = the_scene or scene
        model = InletModel(sc, cfg, path, target)
        win = qe.ExploreWindow(sc, model, with_3d=False)
        windows.append(win)
        return win

    yield factory
    for w in windows:
        w.ask_unsaved = lambda: "discard"  # never let a teardown block on the modal dialog
        w.model._listeners.clear()
        w.close()


def skin_point(scene, x=3.0, y=0.4):
    z = np.array(scene.fuselage.ray_trace([x, y, 5.0], [x, y, -5.0])[0])[:, 2].max()
    return [x, y, float(z)]


def unplaced(example_env):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    inlet = cfg["ducts"]["inlet"]
    inlet["origin"] = inlet["vertices"] = inlet["snapped"] = None
    return cfg


# ---- building ---------------------------------------------------------------------------------------
def test_the_window_opens_in_view_mode_without_a_star(make_window):
    w = make_window()
    assert w.mode_label.text() == "View"
    assert w.windowTitle().startswith("VSP Duct Manager") and not w.windowTitle().endswith("*")
    assert not w.btn_place.isChecked() and not w.chk_lock.isChecked()
    assert w.dock.width() == qe.PANEL_WIDTH or w.dock.maximumWidth() == qe.PANEL_WIDTH  # compact side panel


def test_the_dark_theme_is_applied(make_window):
    w = make_window()
    css = w.styleSheet()
    assert BACKGROUND.lower() == "#15181c" and ACCENT.lower() == "#2bb3c0"
    assert "#15181C" in css and "#2BB3C0" in css


def test_the_panel_has_the_plane_controls_and_no_shape_or_skin_fit_widgets(make_window):
    w = make_window()
    assert set(w.spin_pos) == {"x", "y", "z"}
    assert w.spin_pitch.suffix() == " deg" and w.spin_sweep.suffix() == " deg"
    assert [w.combo_step.itemData(i) for i in range(w.combo_step.count())] == list(qe.NUDGE_STEPS)
    for box in (*w.spin_pos.values(), w.spin_pitch, w.spin_sweep):
        assert isinstance(box, qe.FocusSpinBox)
    for axis, box in w.spin_pos.items():
        assert box.decimals() == 3 and box.suffix() == " m"  # 3 decimals, in metres
    assert w.btn_align.text() == "Align to chine"
    for gone in ("spin_shape", "spin_tilt", "combo_fit"):  # the parametric shape / skin fit controls are removed
        assert not hasattr(w, gone)
    for grp in (w.grp_place, w.grp_position, w.grp_readouts):
        assert isinstance(grp, qe.CollapsibleGroup)
    assert w.readout_view.isReadOnly()


def test_the_sketch_is_docked_next_to_the_3d_view(make_window):
    w = make_window()
    assert isinstance(w.sketch, qs.SketchDock) and w.sketch_dock.widget() is w.sketch
    assert w.dockWidgetArea(w.sketch_dock) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
    assert w.dockWidgetArea(w.dock) == QtCore.Qt.DockWidgetArea.RightDockWidgetArea
    assert w.sketch_dock.minimumWidth() >= qe.SKETCH_WIDTH
    w.show()
    assert w.act_sketch.isChecked() and not w.sketch_dock.isHidden()
    w.act_sketch.trigger()  # the toolbar toggle hides / shows the dock
    assert w.sketch_dock.isHidden()
    w.act_sketch.trigger()
    assert not w.sketch_dock.isHidden()


def test_toolbar_has_undo_redo_views_toggles_and_the_sketch(make_window):
    w = make_window()
    names = {a.text() for a in w.findChildren(QtGui.QAction)}
    assert {"Undo", "Redo", "Side", "Top", "Front", "Iso", "Context", "Envelope", "Sketch"} <= names
    assert not w.act_context.isEnabled() and not w.act_envelope.isEnabled()  # no 3D view in this window


def test_shortcuts(make_window):
    w = make_window()
    keys = lambda act: [s.toString() for s in act.shortcuts()]  # noqa: E731
    assert keys(w.act_place) == ["P"] and keys(w.act_cancel) == ["Esc"]
    assert keys(w.act_undo) == ["Ctrl+Z"] and "Ctrl+Y" in keys(w.act_redo)
    assert keys(w.act_save) == ["Ctrl+S"]


# ---- place mode in the window ---------------------------------------------------------------------------
def test_place_button_and_status_bar_show_the_mode(make_window):
    w = make_window()
    w.btn_place.click()
    assert w.model.mode == "placing" and w.btn_place.isChecked() and w.act_place.isChecked()
    assert w.mode_label.text() == "Placing inlet - click on fuselage, Esc to cancel"
    w.btn_place.click()
    assert w.model.mode == "view" and w.mode_label.text() == "View" and not w.btn_place.isChecked()


def test_the_p_action_and_escape_drive_the_same_mode(make_window):
    w = make_window()
    w.act_place.trigger()
    assert w.model.mode == "placing" and w.mode_label.text().startswith("Placing inlet")
    w.act_cancel.trigger()  # Esc
    assert w.model.mode == "view" and not w.btn_place.isChecked() and w.mode_label.text() == "View"
    w.act_cancel.trigger()  # Esc in View mode does nothing
    assert w.model.mode == "view"


def test_a_click_places_once_and_the_window_follows(make_window, example_env, scene):
    w = make_window(cfg=unplaced(example_env))
    assert not any(b.isEnabled() for b in w.spin_pos.values())  # nothing to edit before placing
    assert not w.spin_sweep.isEnabled() and not w.btn_align.isEnabled() and not w.sketch.btn_fit_area.isEnabled()
    p = skin_point(scene)
    assert not w.model.handle_click(p)  # a click in View mode is ignored by the model
    assert not w.windowTitle().endswith("*")
    w.btn_place.click()
    assert w.model.handle_click(p)
    assert w.mode_label.text() == "View" and not w.btn_place.isChecked()
    assert w.windowTitle().endswith("*")
    assert all(b.isEnabled() for b in w.spin_pos.values()) and w.spin_sweep.isEnabled() and w.btn_align.isEnabled()
    assert w.spin_pos["x"].value() == pytest.approx(p[0], abs=1e-3) and w.spin_pos["z"].value() == pytest.approx(p[2], abs=1e-3)
    assert "Origin x y z" in w.readout_view.toPlainText()
    assert w.sketch.canvas.vertex_positions()  # the sketch appeared with its 5 vertices


def test_lock_disables_placement_and_nudges_but_not_the_angles(make_window):
    w = make_window()
    w.chk_lock.setChecked(True)
    assert w.model.locked and not w.btn_place.isEnabled()
    assert not any(b.isEnabled() for b in w.spin_pos.values()) and not w.combo_step.isEnabled()
    assert w.spin_pitch.isEnabled() and w.spin_sweep.isEnabled() and w.sketch.btn_fit_area.isEnabled()
    w.act_place.trigger()  # the shortcut respects the lock
    assert w.model.mode == "view" and w.mode_label.text() == "View"
    w.chk_lock.setChecked(False)
    assert w.btn_place.isEnabled() and all(b.isEnabled() for b in w.spin_pos.values())


def test_locking_while_placing_returns_to_view(make_window):
    w = make_window()
    w.btn_place.click()
    w.chk_lock.setChecked(True)
    assert w.mode_label.text() == "View" and not w.btn_place.isChecked()


# ---- editing ---------------------------------------------------------------------------------------------
def test_spin_boxes_edit_the_model_and_the_title_gets_a_star(make_window):
    w = make_window()
    w.spin_sweep.setValue(12.0)
    assert w.model.state.plane.sweep_deg == 12.0 and w.windowTitle().endswith("*")
    w.spin_pitch.setValue(-6.0)
    assert w.model.state.plane.pitch_deg == -6.0
    w.spin_pos["x"].setValue(3.3)
    assert w.model.state.plane.origin[0] == pytest.approx(3.3)


def test_the_panel_follows_model_changes_without_feedback_loops(make_window):
    w = make_window()
    before = w.model.can_undo
    w.model.set_param("sweep_deg", 20.0)  # changed from outside the panel
    assert w.spin_sweep.value() == 20.0
    w.model.undo()
    assert w.spin_sweep.value() == 0.0
    assert w.model.can_undo == before  # syncing the widgets did not create history


def test_undo_and_redo_actions_also_cover_sketch_drags(make_window):
    w = make_window()
    assert not w.act_undo.isEnabled() and not w.act_redo.isEnabled()
    w.spin_sweep.setValue(20.0)
    assert w.act_undo.isEnabled() and not w.act_redo.isEnabled()
    start = w.model.state.vertices
    c = w.sketch.canvas
    c.resize(*(640, 480))
    c.fit_view()
    c.press(c.to_screen(w.model.state.vertices[3]))
    c.move(c.to_screen((0.0, 0.8)), alt=True)
    c.release()
    assert w.model.state.vertices != start and w.windowTitle().endswith("*")
    w.act_undo.trigger()  # the drag
    assert w.model.state.vertices == start and w.spin_sweep.value() == 20.0
    w.act_undo.trigger()  # then the sweep
    assert w.spin_sweep.value() == 0.0 and not w.windowTitle().endswith("*") and w.act_redo.isEnabled()
    w.act_redo.trigger()
    assert w.spin_sweep.value() == 20.0 and w.windowTitle().endswith("*")


def test_nudge_step_sets_the_spin_step(make_window):
    w = make_window()
    w.combo_step.setCurrentIndex(qe.NUDGE_STEPS.index(0.05))
    assert all(b.singleStep() == pytest.approx(0.05) for b in w.spin_pos.values())
    w.combo_step.setCurrentIndex(qe.NUDGE_STEPS.index(0.001))
    assert all(b.singleStep() == pytest.approx(0.001) for b in w.spin_pos.values())


def test_the_wheel_only_acts_on_a_focused_spin_box(make_window, qapp):
    w = make_window()
    w.show()
    qapp.processEvents()
    box = w.spin_sweep
    before = box.value()

    def wheel():
        ev = QtGui.QWheelEvent(QtCore.QPointF(5, 5), QtCore.QPointF(5, 5), QtCore.QPoint(0, 0),
                               QtCore.QPoint(0, 120), QtCore.Qt.MouseButton.NoButton,
                               QtCore.Qt.KeyboardModifier.NoModifier, QtCore.Qt.ScrollPhase.NoScrollPhase, False)
        QtWidgets.QApplication.sendEvent(box, ev)
        return ev

    box.clearFocus()
    qapp.processEvents()
    assert not box.hasFocus()
    ev = wheel()
    assert box.value() == before and not ev.isAccepted()  # ignored, so the panel scrolls instead
    w.activateWindow()
    box.setFocus()
    qapp.processEvents()
    if box.hasFocus():  # an offscreen window may refuse focus; the unfocused half above is the safety property
        wheel()
        assert box.value() > before


def test_groups_collapse(make_window):
    w = make_window()
    grp = w.grp_position
    assert grp.expanded and not grp.body.isHidden()
    grp.header.click()
    assert not grp.expanded and grp.body.isHidden()
    grp.header.click()
    assert grp.expanded and not grp.body.isHidden()


# ---- fit area and align to chine -------------------------------------------------------------------------------
def test_the_fit_area_button_restores_the_target(make_window):
    w = make_window()
    m = w.model
    m.drag_vertex(3, (0.0, 0.9), snap_tolerance=None)
    assert abs(m.opening.delta) > 0.005
    w.sketch.btn_fit_area.click()
    assert abs(m.opening.delta) < 0.005 and "Area fitted" in w.message_label.text()


def test_align_to_chine_sets_the_sweep_spin_box():
    taper = bodies.tapered_chined_body()
    scene = SimpleNamespace(fuselage=taper, sizing=SimpleNamespace(capture_area_per_inlet=0.05), engine_check=None,
                            warnings=[], face_center=None)
    cfg = {"ducts": {"inlet": {"origin": [4.0, 1.0, 0.6]}}}
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = qe.ExploreWindow(scene, InletModel(scene, cfg, EXAMPLE_CONFIG, None), with_3d=False)
    try:
        win.btn_align.click()
        expected = sweep_for_direction(1.0, 0.125)
        assert win.spin_sweep.value() == pytest.approx(expected, abs=0.01)  # the resulting sweep is shown...
        assert "Aligned to the chine" in win.message_label.text()
        win.spin_sweep.setValue(10.0)  # ...and stays editable
        assert win.model.state.plane.sweep_deg == 10.0
    finally:
        win.ask_unsaved = lambda: "discard"
        win.model._listeners.clear()
        win.close()


def test_align_without_a_chine_says_so_in_the_status_bar(make_window):
    w = make_window()  # the default example fuselage has no sharp chine near the dorsal origin
    w.btn_align.click()
    text = w.message_label.text()
    assert "Aligned" in text or "No chine" in text  # either outcome is reported, never silent


# ---- readouts ---------------------------------------------------------------------------------------------------
def test_readout_html_colours_pass_green_warn_amber_and_error_red():
    html = qe.readouts_html([Readout("A", "fine", "pass"), Readout("B", "careful", "warn"),
                             Readout("C", "plain"), Readout("D", "bad", "error"), Readout("", "WARN <x> & y", "warn")])
    assert qe.COLORS["pass"] == "#4CAF50" and qe.COLORS["warn"] == "#E0A030" and qe.COLORS["error"] == "#F85149"
    assert f'color:{qe.COLORS["pass"]}">A' in html and f'color:{qe.COLORS["warn"]}">B' in html
    assert f'color:{qe.COLORS["info"]}">C' in html and f'color:{qe.COLORS["error"]}">D' in html
    assert "WARN &lt;x&gt; &amp; y" in html  # escaped
    assert "pre-wrap" in html  # long lines wrap instead of scrolling sideways


def test_the_readout_panel_shows_the_model_lines(make_window):
    w = make_window()
    text = w.readout_view.toPlainText()
    for label in ("Face / envelope", "Area projected", "Area in plane", "Snapped edge length", "Free-edge clearance"):
        assert label in text
    assert qe.COLORS["pass"].lower() in w.readout_view.toHtml().lower()


def test_an_invalid_opening_shows_a_red_error_and_disables_save(make_window, writable):
    w = make_window(save_target=writable, config_path=writable)
    assert w.btn_save.isEnabled()
    w.model.drag_vertex(0, (w.model.state.vertices[0][0], -0.8), snap_tolerance=None)  # inside the fuselage
    assert not w.model.valid
    assert not w.btn_save.isEnabled() and not w.act_save.isEnabled()  # Save is disabled while invalid
    assert "not valid" in w.btn_save.toolTip()
    assert qe.COLORS["error"].lower() in w.readout_view.toHtml().lower()
    assert "ERROR" in w.readout_view.toPlainText()
    before = writable.read_bytes()
    w.act_save.trigger()  # the disabled action (Ctrl+S) does nothing...
    w.btn_save.click()  # ...and neither does the disabled button
    assert writable.read_bytes() == before
    assert not w.model.save() and w.message_label.text().startswith("ERROR: not saved")  # the model refuses too
    assert writable.read_bytes() == before
    w.act_undo.trigger()
    assert w.btn_save.isEnabled()


# ---- save / revert / close ------------------------------------------------------------------------------------------
def test_save_button_and_shortcut_write_the_config_and_clear_the_star(make_window, writable):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_sweep.setValue(9.0)
    assert w.windowTitle().endswith("*")
    w.btn_save.click()
    assert not w.windowTitle().endswith("*")
    assert yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlet"]["sweep_deg"] == 9.0
    w.spin_pitch.setValue(5.0)
    w.act_save.trigger()  # Ctrl+S
    assert yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlet"]["pitch_deg"] == 5.0


def test_revert_button_restores_the_saved_inlet(make_window, writable):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_sweep.setValue(30.0)
    w.btn_revert.click()
    assert w.spin_sweep.value() == 0.0 and not w.windowTitle().endswith("*")
    w.act_undo.trigger()
    assert w.spin_sweep.value() == 30.0  # revert is undoable


def test_the_example_config_window_cannot_save(make_window):
    w = make_window()  # no save target
    assert not w.btn_save.isEnabled() and not w.act_save.isEnabled()
    assert "read-only" in w.btn_save.toolTip()


def test_closing_with_unsaved_changes_asks_first(make_window, writable, monkeypatch):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_sweep.setValue(10.0)
    asked = []
    for answer, closes in (("cancel", False), ("discard", True)):
        monkeypatch.setattr(w, "ask_unsaved", lambda a=answer: asked.append(a) or a)
        assert w.close() is closes
    assert asked == ["cancel", "discard"]


def test_closing_with_save_writes_the_file(make_window, writable, monkeypatch):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_sweep.setValue(10.0)
    monkeypatch.setattr(w, "ask_unsaved", lambda: "save")
    assert w.close()
    assert yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlet"]["sweep_deg"] == 10.0


def test_a_failed_save_keeps_the_window_open(make_window, writable, monkeypatch):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_sweep.setValue(10.0)
    monkeypatch.setattr(w, "ask_unsaved", lambda: "save")
    monkeypatch.setattr(w.model, "save", lambda: False)
    assert not w.close()


def test_a_clean_window_closes_without_asking(make_window, monkeypatch):
    w = make_window()
    monkeypatch.setattr(w, "ask_unsaved", lambda: pytest.fail("must not ask when nothing changed"))
    assert w.close()


def fake_message_box(monkeypatch, captured):
    class FakeBox:
        class Icon:
            Warning = 0

        class ButtonRole:
            AcceptRole = 0
            DestructiveRole = 1
            RejectRole = 2

        def __init__(self, parent):
            self.buttons = []
            captured["box"] = self

        def setIcon(self, *_): pass
        def setWindowTitle(self, *_): pass
        def setText(self, *_): pass
        def setInformativeText(self, text): captured["info"] = text
        def setDefaultButton(self, *_): pass

        def addButton(self, text, role):
            self.buttons.append(text)
            return text

        def exec(self): self.clicked = "Discard"
        def clickedButton(self): return self.clicked

    monkeypatch.setattr(qe.QtWidgets, "QMessageBox", FakeBox)


def test_the_unsaved_dialog_offers_no_save_for_the_read_only_example(make_window, monkeypatch):
    w = make_window()  # read-only: Save would be a lie
    w.spin_sweep.setValue(10.0)
    captured = {}
    fake_message_box(monkeypatch, captured)
    assert w.ask_unsaved() == "discard"
    assert captured["box"].buttons == ["Discard", "Cancel"] and "read-only" in captured["info"]


def test_the_unsaved_dialog_offers_no_save_for_an_invalid_opening(make_window, writable, monkeypatch):
    w = make_window(save_target=writable, config_path=writable)
    w.model.drag_vertex(0, (w.model.state.vertices[0][0], -0.8), snap_tolerance=None)
    captured = {}
    fake_message_box(monkeypatch, captured)
    assert w.ask_unsaved() == "discard"
    assert captured["box"].buttons == ["Discard", "Cancel"] and "not valid" in captured["info"]
