# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The Qt window without its 3D view, on Qt's offscreen platform (the 3D view needs a real display; see
test_qt_gui_3d.py). Everything that is a rule lives in the model; these tests check that the window shows
it and forwards to it correctly."""
import os

if not os.environ.get("DUCTMGR_GUI_TESTS"):  # the opt-in GUI tests need the real platform
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import shutil

import pytest
import yaml

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from ductmgr import qt_explorer as qe  # noqa: E402
from ductmgr.config import load_config  # noqa: E402
from ductmgr.explore_model import InletModel, Readout  # noqa: E402
from ductmgr.viz import ACCENT, BACKGROUND, load_scene  # noqa: E402

TOP = [3.0, 0.4, 1.4]


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

    def factory(save_target="none", cfg=None, config_path=None):
        _, path, _ = example_env
        path = config_path or path
        cfg = cfg if cfg is not None else load_config(path)[0]
        target = None if save_target == "none" else save_target
        model = InletModel(scene, cfg, path, target)
        win = qe.ExploreWindow(scene, model, with_3d=False)
        windows.append(win)
        return win

    yield factory
    for w in windows:
        w.ask_unsaved = lambda: "discard"  # never let a teardown block on the modal dialog
        w.model._listeners.clear()
        w.close()


def unplaced(example_env):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"]["centroid"] = None
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


def test_the_panel_has_every_requested_control(make_window):
    w = make_window()
    assert set(w.spin_pos) == {"x", "y", "z"}
    assert set(w.spin_shape) == {"width_bottom", "width_top", "height", "apex_height", "skew"}
    assert w.spin_tilt.suffix() == " deg"
    assert [w.combo_fit.itemText(i) for i in range(w.combo_fit.count())] == ["none", "bottom_edge", "flush"]
    assert [w.combo_step.itemData(i) for i in range(w.combo_step.count())] == list(qe.NUDGE_STEPS)
    for box in (*w.spin_pos.values(), *w.spin_shape.values(), w.spin_tilt):
        assert isinstance(box, qe.FocusSpinBox)
    for axis, box in w.spin_pos.items():
        assert box.decimals() == 3 and box.suffix() == " m"  # 3 decimals, in metres
    for key, box in w.spin_shape.items():
        assert box.suffix().strip() and box.minimum() < box.maximum()  # units shown, sensible range
    for key in ("width_bottom", "width_top", "height"):
        assert w.spin_shape[key].minimum() > 0.0
    for label in (w.grp_place, w.grp_position, w.grp_shape, w.grp_orient, w.grp_readouts):
        assert isinstance(label, qe.CollapsibleGroup)
    assert w.readout_view.isReadOnly()
    assert "Consolas" in w.readout_view.font().family() or w.readout_view.font().styleHint() is not None


def test_toolbar_has_undo_redo_views_and_toggles(make_window):
    w = make_window()
    names = {a.text() for a in w.findChildren(QtGui.QAction)}
    assert {"Undo", "Redo", "Side", "Top", "Front", "Iso", "Context", "Envelope"} <= names
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


def test_a_click_places_once_and_the_window_follows(make_window, example_env):
    w = make_window(cfg=unplaced(example_env))
    assert not any(b.isEnabled() for b in w.spin_pos.values())  # nothing to edit before placing
    assert not w.model.handle_click(TOP)  # a click in View mode is ignored by the model
    assert not w.windowTitle().endswith("*")
    w.btn_place.click()
    assert w.model.handle_click(TOP)
    assert w.mode_label.text() == "View" and not w.btn_place.isChecked()
    assert w.windowTitle().endswith("*")
    assert all(b.isEnabled() for b in w.spin_pos.values())
    assert w.spin_pos["x"].value() == pytest.approx(3.0, abs=1e-3) and w.spin_pos["y"].value() == pytest.approx(0.4, abs=1e-3)
    assert "Centroid" in w.readout_view.toPlainText()


def test_lock_disables_placement_and_position_but_not_the_rest(make_window):
    w = make_window()
    w.chk_lock.setChecked(True)
    assert w.model.locked and not w.btn_place.isEnabled()
    assert not any(b.isEnabled() for b in w.spin_pos.values()) and not w.combo_step.isEnabled()
    assert all(b.isEnabled() for b in w.spin_shape.values()) and w.spin_tilt.isEnabled()
    w.act_place.trigger()  # the shortcut respects the lock
    assert w.model.mode == "view" and w.mode_label.text() == "View"
    w.chk_lock.setChecked(False)
    assert w.btn_place.isEnabled() and all(b.isEnabled() for b in w.spin_pos.values())


def test_locking_while_placing_returns_to_view(make_window):
    w = make_window()
    w.btn_place.click()
    w.chk_lock.setChecked(True)
    assert w.mode_label.text() == "View" and not w.btn_place.isChecked()


# ---- editing -----------------------------------------------------------------------------------------------
def test_spin_boxes_edit_the_model_and_the_title_gets_a_star(make_window):
    w = make_window()
    w.spin_shape["width_top"].setValue(0.9)
    assert w.model.state.shape.width_top == 0.9 and w.windowTitle().endswith("*")
    w.spin_tilt.setValue(12.0)
    assert w.model.state.tilt_deg == 12.0
    w.combo_fit.setCurrentText("flush")
    assert w.model.state.skin_fit == "flush"
    w.spin_pos["x"].setValue(3.3)
    assert w.model.centroid[0] == pytest.approx(3.3)


def test_the_panel_follows_model_changes_without_feedback_loops(make_window):
    w = make_window()
    before = w.model.can_undo
    w.model.set_param("skin_fit", "none")  # changed from outside the panel
    assert w.combo_fit.currentText() == "none"
    w.model.undo()
    assert w.combo_fit.currentText() == "bottom_edge"
    assert w.model.can_undo == before  # syncing the widgets did not create history


def test_undo_and_redo_actions(make_window):
    w = make_window()
    assert not w.act_undo.isEnabled() and not w.act_redo.isEnabled()
    w.spin_tilt.setValue(20.0)
    assert w.act_undo.isEnabled() and not w.act_redo.isEnabled()
    w.act_undo.trigger()
    assert w.spin_tilt.value() == 0.0 and w.act_redo.isEnabled() and not w.windowTitle().endswith("*")
    w.act_redo.trigger()
    assert w.spin_tilt.value() == 20.0 and w.windowTitle().endswith("*")


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
    box = w.spin_shape["height"]
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
    grp = w.grp_shape
    assert grp.expanded and not grp.body.isHidden()
    grp.header.click()
    assert not grp.expanded and grp.body.isHidden()
    grp.header.click()
    assert grp.expanded and not grp.body.isHidden()


# ---- readouts -----------------------------------------------------------------------------------------------------
def test_readout_html_colours_warn_amber_and_pass_green():
    html = qe.readouts_html([Readout("A", "fine", "pass"), Readout("B", "careful", "warn"),
                             Readout("C", "plain"), Readout("", "WARN <x> & y", "warn")])
    assert qe.COLORS["pass"] == "#4CAF50" and qe.COLORS["warn"] == "#E0A030"
    assert f'color:{qe.COLORS["pass"]}">A' in html and f'color:{qe.COLORS["warn"]}">B' in html
    assert f'color:{qe.COLORS["info"]}">C' in html
    assert "WARN &lt;x&gt; &amp; y" in html  # escaped
    assert "pre-wrap" in html  # long lines wrap instead of scrolling sideways


def test_the_readout_panel_shows_the_model_lines(make_window):
    w = make_window()
    text = w.readout_view.toPlainText()
    for label in ("Face / envelope", "Capture area target", "Capture area actual", "Max skin gap", "Opening height"):
        assert label in text
    assert qe.COLORS["pass"].lower() in w.readout_view.toHtml().lower()


# ---- save / revert / close ---------------------------------------------------------------------------------------------
def test_save_button_and_shortcut_write_the_config_and_clear_the_star(make_window, writable):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_shape["width_top"].setValue(0.9)
    assert w.windowTitle().endswith("*")
    w.btn_save.click()
    assert not w.windowTitle().endswith("*")
    assert yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlet"]["shape"]["width_top"] == 0.9
    w.spin_tilt.setValue(5.0)
    w.act_save.trigger()  # Ctrl+S
    assert yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlet"]["tilt_deg"] == 5.0


def test_revert_button_restores_the_saved_inlet(make_window, writable):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_tilt.setValue(30.0)
    w.btn_revert.click()
    assert w.spin_tilt.value() == 0.0 and not w.windowTitle().endswith("*")
    w.act_undo.trigger()
    assert w.spin_tilt.value() == 30.0  # revert is undoable


def test_the_example_config_window_cannot_save(make_window):
    w = make_window()  # no save target
    assert not w.btn_save.isEnabled() and not w.act_save.isEnabled()


def test_closing_with_unsaved_changes_asks_first(make_window, writable, monkeypatch):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_tilt.setValue(10.0)
    asked = []
    for answer, closes in (("cancel", False), ("discard", True)):
        monkeypatch.setattr(w, "ask_unsaved", lambda a=answer: asked.append(a) or a)
        assert w.close() is closes
    assert asked == ["cancel", "discard"]


def test_closing_with_save_writes_the_file(make_window, writable, monkeypatch):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_tilt.setValue(10.0)
    monkeypatch.setattr(w, "ask_unsaved", lambda: "save")
    assert w.close()
    assert yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlet"]["tilt_deg"] == 10.0


def test_a_failed_save_keeps_the_window_open(make_window, writable, monkeypatch):
    w = make_window(save_target=writable, config_path=writable)
    w.spin_tilt.setValue(10.0)
    monkeypatch.setattr(w, "ask_unsaved", lambda: "save")
    monkeypatch.setattr(w.model, "save", lambda: False)
    assert not w.close()


def test_a_clean_window_closes_without_asking(make_window, monkeypatch):
    w = make_window()
    monkeypatch.setattr(w, "ask_unsaved", lambda: pytest.fail("must not ask when nothing changed"))
    assert w.close()


def test_the_unsaved_dialog_offers_no_save_for_the_read_only_example(make_window, monkeypatch):
    w = make_window()  # read-only: Save would be a lie
    w.spin_tilt.setValue(10.0)
    captured = {}

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
    assert w.ask_unsaved() == "discard"
    assert captured["box"].buttons == ["Discard", "Cancel"] and "read-only" in captured["info"]
