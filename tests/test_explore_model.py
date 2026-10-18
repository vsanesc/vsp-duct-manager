# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 VÃ­ctor SÃ¡nchez Escalante

"""Explorer logic without any GUI: undo/redo, place-mode state machine, lock, dirty flag, save/revert, readouts."""
import copy
import shutil

import numpy as np
import pytest
import yaml

from ductmgr import explore_model as em
from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.explore_model import InletModel
from ductmgr.geometry import InletShape, inlet_frame
from ductmgr.viz import load_scene

TOP = [3.0, 0.4, 1.4]  # a point above the dorsal skin of the example fuselage


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture(scope="module")
def scene(example_env):
    cfg, path, _ = example_env
    return load_scene(cfg, path)


@pytest.fixture()
def writable(example_env, tmp_path):
    """A private-style config copy (absolute model path) that the model may save into."""
    _, path, _ = example_env
    target = tmp_path / "duct_config.yaml"
    shutil.copyfile(path, target)
    return target


@pytest.fixture()
def clock():
    return Clock()


def make(scene, config_path, save_target=None, clock=None, cfg=None):
    cfg = cfg if cfg is not None else load_config(config_path)[0]
    return InletModel(scene, cfg, config_path, save_target, **({"clock": clock} if clock else {}))


def place(model, point=TOP):
    assert model.begin_place() and model.handle_click(point)


def no_inlet_cfg(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"]["centroid"] = None
    return cfg, path


# ---- initial state -------------------------------------------------------------------------------------------
def test_initial_state_comes_from_the_config(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    assert m.mode == "view" and not m.locked and not m.dirty and not m.can_undo and not m.can_redo
    assert m.state.skin_fit == "bottom_edge" and m.state.tilt_deg == 0.0
    assert m.centroid is not None and m.on_surface and m.fit.mode == "bottom_edge"
    assert m.centroid[:2] == pytest.approx([2.0, 0.5])
    assert m.mode_text == "View"


# ---- undo / redo ---------------------------------------------------------------------------------------------------
def test_undo_redo_round_trip_over_every_kind_of_change(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    start = m.state
    steps = []
    for change in (
        lambda: m.begin_place() and m.handle_click(TOP),
        lambda: m.set_param("width_top", 0.9),
        lambda: m.set_param("tilt_deg", 12.0),
        lambda: m.set_param("skin_fit", "none"),
        lambda: m.set_position(3.4, 0.3, 1.4),
    ):
        clock.advance(10.0)  # outside the merge window: every change is its own step
        change()
        steps.append(m.state)
    assert len(set(steps)) == 5 and m.can_undo and not m.can_redo

    for expected in reversed(steps[:-1]):
        assert m.undo() and m.state == expected
    assert m.undo() and m.state == start and not m.can_undo
    assert not m.undo()
    for expected in steps:
        assert m.redo() and m.state == expected
    assert not m.redo() and not m.can_redo


def test_undo_restores_the_derived_geometry_too(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    c0, a0 = m.centroid.copy(), m.fit.area
    place(m)
    c1 = m.centroid.copy()
    assert c1 != pytest.approx(c0)
    m.undo()
    assert m.centroid == pytest.approx(c0) and m.fit.area == pytest.approx(a0)
    m.redo()
    assert m.centroid == pytest.approx(c1)


def test_history_holds_at_least_50_changes(scene, example_env, clock):
    _, path, _ = example_env
    assert em.UNDO_LIMIT >= 50
    m = make(scene, path, clock=clock)
    for i in range(1, 61):
        clock.advance(10.0)
        assert m.set_param("tilt_deg", float(i))
    undone = 0
    while m.undo():
        undone += 1
    assert undone == 60 and m.state.tilt_deg == 0.0


def test_history_is_bounded(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    for i in range(1, em.UNDO_LIMIT + 50):
        clock.advance(10.0)
        m.set_param("tilt_deg", i * 0.1)
    undone = 0
    while m.undo():
        undone += 1
    assert undone == em.UNDO_LIMIT


def test_rapid_edits_of_one_parameter_share_one_undo_step(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):  # a drag on a spin box
        clock.advance(0.1)
        m.set_param("tilt_deg", v)
    assert m.state.tilt_deg == 5.0
    assert m.undo() and m.state.tilt_deg == 0.0 and not m.can_undo  # one step back to the start

    m.redo()
    clock.advance(10.0)  # a pause ends the merge
    m.set_param("tilt_deg", 9.0)
    assert m.undo() and m.state.tilt_deg == 5.0
    m.redo()
    clock.advance(0.1)
    m.set_param("width_top", 0.9)  # another parameter always starts a new step
    clock.advance(0.1)
    m.set_param("tilt_deg", 10.0)
    assert m.undo() and m.state.tilt_deg == 9.0 and m.state.shape.width_top == 0.9


def test_a_new_change_clears_the_redo_history(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    clock.advance(10.0)
    m.set_param("tilt_deg", 5.0)
    m.undo()
    assert m.can_redo
    clock.advance(10.0)
    m.set_param("tilt_deg", 7.0)
    assert not m.can_redo and not m.redo()


def test_unchanged_values_are_not_history(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert not m.set_param("tilt_deg", m.state.tilt_deg)
    assert not m.set_param("skin_fit", m.state.skin_fit)
    assert not m.can_undo and not m.dirty


def test_undo_and_redo_notify_listeners(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    calls = []
    m.add_listener(lambda: calls.append(1))
    m.set_param("tilt_deg", 3.0)
    m.undo()
    m.redo()
    assert len(calls) == 3
    m.set_param("tilt_deg", 3.0)  # no-op: nothing to tell
    assert len(calls) == 3


def test_invalid_parameters_are_rejected_without_touching_the_history(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    with pytest.raises(ValueError):
        m.set_param("width_top", 0.0)
    with pytest.raises(ValueError, match="skin_fit"):
        m.set_param("skin_fit", "wrap")
    with pytest.raises(KeyError):
        m.set_param("bogus", 1.0)
    assert not m.can_undo and not m.dirty


# ---- place mode --------------------------------------------------------------------------------------------------------
def test_clicks_do_nothing_by_default(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    before = m.state
    assert m.mode == "view" and not m.handle_click(TOP)
    assert m.state == before and not m.dirty and not m.can_undo


def test_place_mode_places_once_then_switches_itself_off(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    assert m.begin_place() and m.mode == "placing"
    assert m.mode_text == "Placing inlet - click on fuselage, Esc to cancel"
    assert m.handle_click(TOP)
    assert m.mode == "view" and m.mode_text == "View"
    assert m.centroid[:2] == pytest.approx([3.0, 0.4]) and m.dirty
    placed = m.state
    assert not m.handle_click([4.0, 0.2, 1.4])  # the mode ended: further clicks are ignored
    assert m.state == placed


def test_escape_cancels_placement(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    before = m.state
    m.begin_place()
    assert m.cancel_place() and m.mode == "view"
    assert not m.handle_click(TOP) and m.state == before and not m.can_undo
    assert not m.cancel_place()  # nothing to cancel in View mode


def test_the_place_button_toggles(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    assert m.toggle_place() and m.mode == "placing"
    m.toggle_place()
    assert m.mode == "view"


def test_a_left_side_click_is_mirrored_onto_the_right_inlet(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    place(m, [3.0, -0.4, 1.4])
    assert m.state.base_point[1] == pytest.approx(0.4) and m.centroid[1] == pytest.approx(0.4)
    assert m.centroid[2] > 1.0  # snapped to the dorsal skin


def test_a_placement_is_one_undo_step(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    start = m.state
    place(m)
    assert m.undo() and m.state == start and not m.can_undo


def test_lock_blocks_placement_and_nudges_but_not_the_shape(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    m.set_locked(True)
    assert m.locked and not m.begin_place() and m.mode == "view" and "locked" in m.message.lower()
    assert not m.handle_click(TOP)
    before = m.state
    assert not m.set_position(3.5, 0.3, 1.4) and m.state == before  # nudges are disabled
    clock.advance(10.0)
    assert m.set_param("width_top", 0.9)  # the lock covers position, not the shape
    m.set_locked(False)
    assert m.begin_place()


def test_locking_while_placing_ends_place_mode(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    m.begin_place()
    m.set_locked(True)
    assert m.mode == "view"
    assert not m.handle_click(TOP)


def test_the_lock_is_neither_history_nor_a_change(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    m.set_locked(True)
    m.set_locked(False)
    assert not m.dirty and not m.can_undo


# ---- position ----------------------------------------------------------------------------------------------------------------
def test_position_edits_snap_to_the_skin(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert m.set_position(3.3, 0.25, 1.4)
    top = np.array(scene.fuselage.ray_trace([3.3, 0.25, 5.0], [3.3, 0.25, -5.0])[0])[:, 2].max()
    assert m.centroid == pytest.approx([3.3, 0.25, top]) and m.on_surface
    clock.advance(10.0)
    m.set_position(3.3, -0.25, 1.4)  # y is kept >= 0: this is the right inlet
    assert m.centroid[1] == pytest.approx(0.25)
    clock.advance(10.0)
    m.set_position(3.3, 0.25, -1.4)  # a height near the underside selects the lower skin
    assert m.centroid[2] < 0.0


def test_a_position_off_the_fuselage_is_flagged(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    m.set_position(3.0, 5.0, 1.4)
    assert not m.on_surface
    assert any(r.label == "Centroid" and r.level == "warn" for r in m.readouts())


# ---- dirty flag, save, revert --------------------------------------------------------------------------------------------------
def test_dirty_follows_the_state_and_undo_back_to_saved_clears_it(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert not m.dirty
    clock.advance(10.0)
    m.set_param("tilt_deg", 15.0)
    assert m.dirty
    m.undo()
    assert not m.dirty  # back at the saved state
    m.redo()
    assert m.dirty


def test_save_writes_the_inlet_and_clears_dirty(scene, writable, clock):
    before = yaml.safe_load(writable.read_text(encoding="utf-8"))
    m = make(scene, writable, writable, clock=clock)
    place(m)
    m.set_param("width_top", 0.9)
    m.set_param("tilt_deg", 12.0)
    assert m.dirty and m.save() and not m.dirty
    assert m.message == f"Saved inlet to {writable.name}."
    after = yaml.safe_load(writable.read_text(encoding="utf-8"))
    inlet = after["ducts"]["inlet"]
    assert inlet["centroid"] == pytest.approx([round(float(v), 6) for v in m.centroid])
    assert inlet["shape"]["width_top"] == 0.9 and inlet["tilt_deg"] == 12.0 and inlet["skin_fit"] == "bottom_edge"
    assert inlet["normal"] == pytest.approx([round(float(v), 6) for v in inlet_frame(12.0)[2]])
    for section in ("model", "engine", "design_point", "checks"):
        assert after[section] == before[section]


def test_saving_is_not_an_edit_so_undo_still_works_and_makes_it_dirty_again(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    clock.advance(10.0)
    m.set_param("tilt_deg", 12.0)
    m.save()
    assert m.can_undo and not m.dirty
    m.undo()
    assert m.dirty  # differs from what was saved


def test_save_is_refused_for_the_read_only_example_config(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path, None)
    place(m)
    before = EXAMPLE_CONFIG.read_bytes()
    assert not m.save() and "read-only" in m.message and m.dirty
    assert EXAMPLE_CONFIG.read_bytes() == before


def test_save_before_an_inlet_is_placed_does_nothing(scene, example_env, writable):
    cfg, path = no_inlet_cfg(example_env)
    before = writable.read_bytes()
    m = InletModel(scene, cfg, writable, writable)
    assert m.centroid is None and m.inlet_definition() is None
    assert not m.save() and m.message.startswith("Nothing to save") and writable.read_bytes() == before


def test_revert_returns_to_the_saved_inlet_and_can_be_undone(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    saved = m.state
    clock.advance(10.0)
    m.set_param("skin_fit", "none")
    clock.advance(10.0)
    m.set_param("tilt_deg", 20.0)
    assert m.dirty
    assert m.revert() and not m.dirty
    assert m.state == saved and m.message == f"Reverted to {writable.name}."
    assert m.undo() and m.state.tilt_deg == 20.0 and m.dirty  # revert is an ordinary, undoable step


def test_revert_reads_the_file_again_including_the_capture_area(scene, writable):
    m = make(scene, writable, writable)
    area = m.area
    cfg = yaml.safe_load(writable.read_text(encoding="utf-8"))
    cfg["design_point"]["mach"] = 0.5
    cfg["ducts"]["inlet"]["tilt_deg"] = 7.0
    writable.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    assert m.revert()
    assert m.area != pytest.approx(area) and m.state.tilt_deg == 7.0 and not m.dirty
    assert m.fit.target_area == pytest.approx(m.area)


def test_revert_of_a_config_without_an_inlet_removes_the_placed_one(scene, example_env, writable):
    cfg, _ = no_inlet_cfg(example_env)
    writable.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    m = make(scene, writable, writable)
    place(m)
    assert m.centroid is not None
    assert m.revert() and m.centroid is None and not m.dirty


def test_revert_reports_a_broken_file_instead_of_crashing(scene, writable):
    m = make(scene, writable, writable)
    state = m.state
    writable.write_text("engine: [unclosed\n", encoding="utf-8")
    assert not m.revert()
    assert m.message.startswith("Revert failed") and m.state == state


# ---- readouts ------------------------------------------------------------------------------------------------------------------------
def levels(model):
    return {r.label or r.value: r.level for r in model.readouts()}


def test_readouts_before_placement(scene, example_env):
    cfg, path = no_inlet_cfg(example_env)
    m = InletModel(scene, cfg, path, None)
    rows = m.readouts()
    assert rows[0].label == "Face / envelope" and rows[0].level == "pass"
    assert "0.870" in rows[0].value and "PASS" in rows[0].value
    assert any(r.label == "Inlet" and "not placed" in r.value for r in rows)


def test_readouts_after_placement(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    place(m)
    rows = {r.label: r for r in m.readouts() if r.label}
    for label in ("Face / envelope", "Centroid", "Capture area target", "Capture area actual", "Skin fit",
                  "Max skin gap", "Opening height", "Aspect ratio (w/h)", "Distance to face", "Offset dy, dz",
                  "Offset / length"):
        assert label in rows
    assert rows["Capture area actual"].level == "pass" and rows["Max skin gap"].level == "pass"
    assert rows["Skin fit"].value == "bottom_edge"
    c = m.centroid
    assert rows["Distance to face"].value == f"{np.linalg.norm(np.array([5.0, 0.0, 0.0]) - c):.4f} m"
    assert rows["Offset dy, dz"].value == f"{-c[1]:+.4f}, {-c[2]:+.4f} m"
    assert rows["Offset / length"].value == f"{np.hypot(c[1], c[2]) / (5.0 - c[0]):.3f}"
    assert rows["Capture area target"].value == f"{m.area:.5f} m^2"


def test_the_engine_check_shows_amber_when_it_warns_and_is_not_listed_twice(scene, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["face_diameter"] = 0.4
    bad = load_scene(cfg, path)
    m = InletModel(bad, cfg, path, None)
    rows = m.readouts()
    assert rows[0].label == "Face / envelope" and rows[0].level == "warn" and "WARN" in rows[0].value
    assert sum("envelope diameter" in r.value or "Face / envelope" == r.label for r in rows) == 1


def test_flush_shows_its_warning_in_amber(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    place(m)
    clock.advance(10.0)
    m.set_param("skin_fit", "flush")
    warns = [r for r in m.readouts() if r.level == "warn"]
    assert any("flush inlet: low pressure recovery and boundary-layer ingestion" in r.value for r in warns)
    assert {r.label: r.value for r in m.readouts()}["Opening height"] == "n/a (flush)"


def test_the_none_mode_has_neutral_gap_and_area_levels(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    clock.advance(10.0)
    m.set_param("skin_fit", "none")
    rows = {r.label: r for r in m.readouts() if r.label}
    assert rows["Capture area actual"].level == "info" and rows["Max skin gap"].level == "info"


def test_without_an_engine_face_the_readouts_say_so(scene, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["geom_name"] = None
    cfg["ducts"]["engine_face"]["centre"] = None
    s = load_scene(cfg, path)
    assert s.face_center is None
    m = InletModel(s, cfg, path, None)
    rows = {r.label: r for r in m.readouts() if r.label}
    assert rows["Engine face"].level == "warn" and "Distance to face" not in rows

