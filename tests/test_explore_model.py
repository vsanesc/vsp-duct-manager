# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Explorer logic without any GUI: sketch editing, one shared undo stack, place-mode state machine, lock,
dirty flag, save/revert and the config round trip."""
import copy
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import bodies
from ductmgr import explore_model as em
from ductmgr import sketch as sk
from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.explore_model import InletModel, legacy_notice, state_from_config
from ductmgr.geometry import InletPlane
from ductmgr.section import distance_to_section, sweep_for_direction
from ductmgr.viz import load_scene


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


def skin_point(scene, x, y):
    """The dorsal skin point above (x, y) of the example fuselage."""
    z = np.array(scene.fuselage.ray_trace([x, y, 5.0], [x, y, -5.0])[0])[:, 2].max()
    return [x, y, float(z)]


def place(model, scene, x=3.0, y=0.4):
    assert model.begin_place() and model.handle_click(skin_point(scene, x, y))


def unplaced_cfg(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    inlet = cfg["ducts"]["inlet"]
    inlet["origin"] = inlet["vertices"] = inlet["snapped"] = None
    return cfg, path


def body_scene(body, area=0.1):
    """A stand-in scene around an analytic body (no OpenVSP)."""
    return SimpleNamespace(fuselage=body, sizing=SimpleNamespace(capture_area_per_inlet=area), engine_check=None,
                           warnings=[], face_center=np.array([6.0, 0.0, 0.0]))


def body_model(body, origin, area=0.1, pitch=0.0, sweep=0.0, clock=None):
    cfg = {"ducts": {"inlet": {"origin": list(origin), "pitch_deg": pitch, "sweep_deg": sweep}}}
    return InletModel(body_scene(body, area), cfg, EXAMPLE_CONFIG, None, **({"clock": clock} if clock else {}))


# ---- loading ---------------------------------------------------------------------------------------------------
def test_the_inlet_is_loaded_from_the_config(scene, example_env):
    cfg, path, _ = example_env
    m = make(scene, path)
    ref = cfg["ducts"]["inlet"]
    assert m.state.plane == InletPlane(tuple(ref["origin"]), 0.0, 0.0)
    assert list(m.state.snapped) == ref["snapped"]
    assert np.array(m.state.vertices) == pytest.approx(np.array(ref["vertices"]), abs=1e-5)
    assert m.section is not None and m.opening is not None and m.valid
    assert not m.dirty and not m.can_undo and m.mode == "view" and m.mode_text == "View"
    for i in (0, 1):  # snapped vertices are exactly on the current section
        assert distance_to_section(m.section, np.array(m.state.vertices[i]))[0] < 1e-6


def test_the_loaded_sketch_matches_the_phase_1_capture_area(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    assert abs(m.opening.delta) < sk.AREA_TOLERANCE and m.opening.target_area == pytest.approx(m.area)


def test_old_keys_are_ignored_with_a_one_line_notice(scene, example_env):
    cfg, path = unplaced_cfg(example_env)
    cfg["ducts"]["inlet"].update({"shape": {"width_top": 0.7}, "skin_fit": "flush", "centroid": [2, 0.5, 1.4],
                                  "tilt_deg": 5.0})
    notice = legacy_notice(cfg)
    assert "\n" not in notice and "ducts.inlet" in notice
    for key in ("shape", "skin_fit", "centroid", "tilt_deg"):
        assert key in notice
    m = InletModel(scene, cfg, path, None)
    assert not m.placed and m.notice == notice  # the old keys do not place anything
    assert any(r.value == notice and r.level == "warn" for r in m.readouts())


def test_null_old_keys_give_no_notice(example_env):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"].update({"centroid": None, "normal": None, "polygon": None, "tilt_deg": None})
    assert legacy_notice(cfg) == ""
    assert legacy_notice({}) == ""


@pytest.mark.parametrize("patch, message", [
    ({"origin": [1.0, 2.0]}, "origin"),
    ({"vertices": [[0, 0]] * 4}, "5"),
    ({"vertices": [[0, 0, 0]] * 5}, "pairs"),
    ({"snapped": [True, False]}, "snapped"),
    ({"snapped": [1, 0, 0, 0, 0]}, "snapped"),
    ({"pitch_deg": 95.0}, "pitch"),
])
def test_malformed_inlets_are_rejected_clearly(example_env, patch, message):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"].update(patch)
    with pytest.raises(ValueError, match=message):
        state_from_config(cfg)


def test_an_origin_without_vertices_gets_a_default_sketch_and_is_dirty(scene, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"]["vertices"] = cfg["ducts"]["inlet"]["snapped"] = None
    m = InletModel(scene, cfg, path, None)
    assert m.state.vertices is not None and m.state.snapped[:2] == (True, True) and m.valid
    assert m.dirty  # the sketch was created, not read from the file


# ---- editing the sketch ----------------------------------------------------------------------------------------------
def test_dragging_a_vertex_near_the_curve_snaps_it_onto_the_curve(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert m.state.snapped[2] is False
    target = np.array(m.state.vertices[2])
    p, _, _ = __import__("ductmgr.section", fromlist=["nearest_on_section"]).nearest_on_section(m.section, target)
    near = tuple(p + np.array([0.0, 0.004]))  # 4 mm off the curve
    assert m.drag_vertex(2, near, snap_tolerance=0.01)
    assert m.state.snapped[2] is True
    assert distance_to_section(m.section, np.array(m.state.vertices[2]))[0] < 1e-6  # on the curve
    assert m.dirty


def test_dragging_away_unsnaps_and_alt_disables_snapping(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert m.state.snapped[0] is True
    far = (m.state.vertices[0][0], m.state.vertices[0][1] + 0.2)
    clock.advance(5.0)
    assert m.drag_vertex(0, far, snap_tolerance=0.01) and m.state.snapped[0] is False  # dropped far: unsnapped
    assert m.state.vertices[0] == far
    clock.advance(5.0)
    on_curve = tuple(np.array(m.state.vertices[1]))  # exactly on the curve, snapped vertex 1
    m.drag_vertex(1, (on_curve[0], on_curve[1] + 0.0005), snap_tolerance=None)  # Alt: no snapping even when close
    assert m.state.snapped[1] is False and m.state.vertices[1] == (on_curve[0], on_curve[1] + 0.0005)


def test_a_vertex_can_slide_along_the_curve_staying_snapped(scene, example_env, clock):
    _, path, _ = example_env
    from ductmgr.section import nearest_on_section, point_at

    m = make(scene, path, clock=clock)
    _, s0, _ = nearest_on_section(m.section, m.state.vertices[0])
    ahead = point_at(m.section, s0 - 0.03)  # a point 3 cm further along the curve
    m.drag_vertex(0, tuple(ahead + [0.0, 0.001]), snap_tolerance=0.05)  # dropped 1 mm off it
    assert m.state.snapped[0]
    assert distance_to_section(m.section, np.array(m.state.vertices[0]))[0] < 1e-6
    _, s1, _ = nearest_on_section(m.section, m.state.vertices[0])
    assert s1 == pytest.approx(s0 - 0.03, abs=0.002)  # it moved 3 cm along the curve, staying on it


def test_rapid_drag_events_are_one_undo_step_shared_with_everything_else(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    start = m.state
    for k in range(10):  # a drag: many events a few ms apart
        clock.advance(0.01)
        m.drag_vertex(3, (0.0 + 0.002 * k, 0.5), snap_tolerance=0.01)
    clock.advance(10.0)
    m.set_param("pitch_deg", 4.0)  # a plane change goes on the same stack
    assert m.undo() and m.state.plane.pitch_deg == 0.0
    assert m.undo() and m.state == start and not m.can_undo  # the whole drag undone at once
    assert m.redo() and m.redo() and m.state.plane.pitch_deg == 4.0


def test_moving_the_plane_keeps_the_sketch_and_resnaps_the_snapped_vertices(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    before = m.state
    x, y, z = before.plane.origin
    new_x = x + 0.3
    z_new = skin_point(scene, new_x, y)[2]
    clock.advance(10.0)
    assert m.set_origin(new_x, y, z_new)
    after = m.state
    assert after.snapped == before.snapped
    for i in (2, 3, 4):
        assert after.vertices[i] == before.vertices[i]  # free vertices keep their (u, v)
    for i in (0, 1):
        assert distance_to_section(m.section, np.array(after.vertices[i]))[0] < 1e-6  # on the NEW curve
    assert m.undo() and m.state == before


def test_pitch_and_sweep_edits(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    m.set_param("sweep_deg", 25.0)
    clock.advance(10.0)
    m.set_param("pitch_deg", -10.0)
    assert (m.state.plane.sweep_deg, m.state.plane.pitch_deg) == (25.0, -10.0)
    assert m.opening.projection_factor == pytest.approx(math.cos(math.radians(25.0)) * math.cos(math.radians(10.0)))
    with pytest.raises(ValueError, match="pitch"):
        m.set_param("pitch_deg", 90.0)
    with pytest.raises(KeyError):
        m.set_param("bogus", 1.0)
    assert (m.state.plane.sweep_deg, m.state.plane.pitch_deg) == (25.0, -10.0)  # rejected edits change nothing


def test_unchanged_values_are_not_history(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    assert not m.set_param("sweep_deg", 0.0) and not m.can_undo and not m.dirty


def test_fit_area_through_the_model(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    clock.advance(10.0)
    m.drag_vertex(3, (0.0, 0.9), snap_tolerance=None)  # push the apex up: the area is off now
    assert abs(m.opening.delta) > sk.AREA_TOLERANCE
    clock.advance(10.0)
    snapped_before = [m.state.vertices[i] for i in (0, 1)]
    assert m.fit_area() and "Area fitted" in m.message
    assert abs(m.opening.delta) < sk.AREA_TOLERANCE
    assert [m.state.vertices[i] for i in (0, 1)] == snapped_before  # the snapped vertices did not move
    assert m.undo() and abs(m.opening.delta) > sk.AREA_TOLERANCE  # fitting is an ordinary undo step


def test_history_holds_at_least_50_changes_and_is_bounded(scene, example_env, clock):
    _, path, _ = example_env
    assert em.UNDO_LIMIT >= 50
    m = make(scene, path, clock=clock)
    for i in range(1, em.UNDO_LIMIT + 40):
        clock.advance(10.0)
        m.set_param("sweep_deg", i * 0.1)
    undone = 0
    while m.undo():
        undone += 1
    assert undone == em.UNDO_LIMIT >= 50


def test_a_new_change_clears_the_redo_history_and_listeners_hear_everything(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    calls = []
    m.add_listener(lambda: calls.append(1))
    clock.advance(10.0)
    m.set_param("sweep_deg", 5.0)
    m.undo()
    assert m.can_redo
    clock.advance(10.0)
    m.set_param("sweep_deg", 7.0)
    assert not m.can_redo and not m.redo()
    m.set_param("sweep_deg", 7.0)  # no-op: nothing to tell
    assert len(calls) == 3


# ---- chine alignment -----------------------------------------------------------------------------------------------------
def test_align_to_chine_sets_the_sweep_and_keeps_the_sketch_valid():
    taper = bodies.tapered_chined_body()
    m = body_model(taper, (4.0, 1.0, 0.6), area=0.05)
    assert m.align_to_chine()
    expected = sweep_for_direction(1.0, 0.125)
    assert m.state.plane.sweep_deg == pytest.approx(expected, abs=1e-3)
    assert "Aligned to the chine" in m.message and f"{m.state.plane.sweep_deg:.2f}" in m.message
    assert m.opening is not None
    m.set_param("sweep_deg", 10.0)  # it remains editable
    assert m.state.plane.sweep_deg == 10.0
    assert m.undo() and m.undo() and m.state.plane.sweep_deg == 0.0  # and every step is undoable


def test_align_to_chine_without_a_chine_says_so():
    m = body_model(bodies.cylinder(1.0), (4.0, 0.0, 1.0))
    assert not m.align_to_chine() and "No chine" in m.message and m.state.plane.sweep_deg == 0.0


# ---- place mode --------------------------------------------------------------------------------------------------------------------
def test_clicks_do_nothing_by_default(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    before = m.state
    assert m.mode == "view" and not m.handle_click(skin_point(scene, 3.0, 0.4))
    assert m.state == before and not m.dirty and not m.can_undo


def test_place_mode_places_once_then_switches_itself_off(scene, example_env):
    cfg, path = unplaced_cfg(example_env)
    m = InletModel(scene, cfg, path, None)
    assert not m.placed and m.opening is None and m.readouts()[-1].label in ("Inlet", "")
    assert m.begin_place() and m.mode == "placing" and m.mode_text == "Placing inlet - click on fuselage, Esc to cancel"
    p = skin_point(scene, 3.0, 0.4)
    assert m.handle_click(p)
    assert m.mode == "view" and m.mode_text == "View" and m.placed and m.dirty
    assert m.state.plane.origin == pytest.approx(tuple(p)) and m.valid  # a default sketch appears on the skin
    assert m.state.snapped[:2] == (True, True)
    placed = m.state
    assert not m.handle_click(skin_point(scene, 4.0, 0.2)) and m.state == placed  # the mode ended


def test_placing_again_moves_the_origin_and_keeps_the_sketch_and_angles(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    m.set_param("sweep_deg", 15.0)
    verts = m.state.vertices
    free = [verts[i] for i in (2, 3, 4)]
    assert m.begin_place() and m.handle_click(skin_point(scene, 3.4, 0.5))
    assert m.state.plane.sweep_deg == 15.0 and m.state.plane.origin[0] == pytest.approx(3.4)
    assert [m.state.vertices[i] for i in (2, 3, 4)] == free


def test_escape_cancels_placement(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    before = m.state
    m.begin_place()
    assert m.cancel_place() and m.mode == "view"
    assert not m.handle_click(skin_point(scene, 3.0, 0.4)) and m.state == before and not m.can_undo
    assert not m.cancel_place()
    assert m.toggle_place() and m.mode == "placing"
    m.toggle_place()
    assert m.mode == "view"


def test_a_left_side_click_is_mirrored_onto_the_right_inlet(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    p = skin_point(scene, 3.0, 0.4)
    m.begin_place()
    m.handle_click([p[0], -p[1], p[2]])
    assert m.state.plane.origin[1] == pytest.approx(0.4)


def test_a_placement_is_one_undo_step(scene, example_env):
    cfg, path = unplaced_cfg(example_env)
    m = InletModel(scene, cfg, path, None)
    start = m.state
    m.begin_place()
    m.handle_click(skin_point(scene, 3.0, 0.4))
    assert m.undo() and m.state == start and not m.placed and not m.can_undo


def test_lock_blocks_placement_and_nudges_but_not_the_sketch_or_angles(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    m.set_locked(True)
    assert m.locked and not m.begin_place() and m.mode == "view" and "locked" in m.message.lower()
    assert not m.handle_click(skin_point(scene, 3.0, 0.4))
    before = m.state
    x, y, z = before.plane.origin
    assert not m.set_origin(x + 0.1, y, z) and m.state == before  # nudges are disabled
    clock.advance(10.0)
    assert m.set_param("sweep_deg", 5.0)  # the lock covers placement and nudges
    clock.advance(10.0)
    assert m.drag_vertex(3, (0.0, 0.5), snap_tolerance=0.01)
    m.set_locked(False)
    assert m.begin_place()


def test_locking_while_placing_ends_place_mode(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    m.begin_place()
    m.set_locked(True)
    assert m.mode == "view" and not m.handle_click(skin_point(scene, 3.0, 0.4))


def test_the_lock_is_neither_history_nor_a_change(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    m.set_locked(True)
    m.set_locked(False)
    assert not m.dirty and not m.can_undo


# ---- validity and saving ---------------------------------------------------------------------------------------------------------
def invalidate(m, clock):
    clock.advance(10.0)
    m.drag_vertex(0, (m.state.vertices[0][0], -0.8), snap_tolerance=None)  # a free vertex deep inside the fuselage


def test_an_opening_inside_the_fuselage_is_invalid_and_cannot_be_saved(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    assert m.valid and m.can_save
    invalidate(m, clock)
    assert not m.valid and not m.can_save
    before = writable.read_bytes()
    assert not m.save()
    assert m.message.startswith("ERROR: not saved") and "inside the fuselage" in m.message
    assert writable.read_bytes() == before and m.dirty
    m.undo()
    assert m.valid and m.can_save  # valid again: Save is possible


def test_invalid_openings_show_red_error_readouts(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    invalidate(m, clock)
    errors = [r for r in m.readouts() if r.level == "error"]
    assert any("inside the fuselage" in r.value and r.value.startswith("ERROR") for r in errors)
    rows = {r.label: r for r in m.readouts() if r.label}
    assert rows["Free-edge clearance"].level == "error" and float(rows["Free-edge clearance"].value.split()[0]) < 0


def test_save_writes_the_inlet_and_round_trips_uv_vertices(scene, writable, clock):
    before = yaml.safe_load(writable.read_text(encoding="utf-8"))
    m = make(scene, writable, writable, clock=clock)
    clock.advance(10.0)
    m.set_param("sweep_deg", 12.0)
    clock.advance(10.0)
    m.set_param("pitch_deg", -6.0)
    clock.advance(10.0)
    m.drag_vertex(3, (0.01, 0.42), snap_tolerance=None)
    assert m.dirty and m.save() and not m.dirty
    assert m.message == f"Saved inlet to {writable.name}."
    after = yaml.safe_load(writable.read_text(encoding="utf-8"))
    inlet = after["ducts"]["inlet"]
    assert inlet["origin"] == pytest.approx([round(c, 6) for c in m.state.plane.origin])
    assert inlet["pitch_deg"] == -6.0 and inlet["sweep_deg"] == 12.0
    assert np.array(inlet["vertices"]) == pytest.approx(np.array([[round(u, 6), round(v, 6)] for u, v in m.state.vertices]))
    assert inlet["snapped"] == list(m.state.snapped) and all(isinstance(f, bool) for f in inlet["snapped"])
    for section in ("model", "engine", "design_point", "checks"):
        assert after[section] == before[section]
    # a fresh model reads the same inlet back, in (u, v)
    again = make(scene, writable, writable)
    assert again.state.plane == InletPlane(tuple(inlet["origin"]), -6.0, 12.0)
    assert np.array(again.state.vertices) == pytest.approx(np.array(m.state.vertices), abs=1e-5)
    assert again.state.snapped == m.state.snapped and not again.dirty


def test_saving_is_not_an_edit_so_undo_still_works(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    clock.advance(10.0)
    m.set_param("sweep_deg", 12.0)
    m.save()
    assert m.can_undo and not m.dirty
    m.undo()
    assert m.dirty


def test_save_is_refused_for_the_read_only_example_config(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path, None)
    before = EXAMPLE_CONFIG.read_bytes()
    assert not m.can_save and not m.save() and "read-only" in m.message
    assert EXAMPLE_CONFIG.read_bytes() == before


def test_save_before_an_inlet_is_placed_does_nothing(scene, example_env, writable):
    cfg, path = unplaced_cfg(example_env)
    before = writable.read_bytes()
    m = InletModel(scene, cfg, writable, writable)
    assert m.inlet_definition() is None
    assert not m.save() and m.message.startswith("Nothing to save") and writable.read_bytes() == before


# ---- dirty flag and revert -------------------------------------------------------------------------------------------------------------
def test_dirty_follows_the_state_and_undo_back_to_saved_clears_it(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert not m.dirty
    clock.advance(10.0)
    m.set_param("sweep_deg", 15.0)
    assert m.dirty
    m.undo()
    assert not m.dirty
    m.redo()
    assert m.dirty


def test_revert_returns_to_the_saved_inlet_and_can_be_undone(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    saved = m.state
    clock.advance(10.0)
    m.set_param("sweep_deg", 20.0)
    clock.advance(10.0)
    m.drag_vertex(3, (0.0, 0.7), snap_tolerance=None)
    assert m.dirty and m.revert() and not m.dirty
    assert m.state == saved and m.message == f"Reverted to {writable.name}."
    assert m.undo() and m.dirty  # revert is an ordinary, undoable step


def test_revert_reads_the_file_again_including_the_capture_area(scene, writable):
    m = make(scene, writable, writable)
    area = m.area
    cfg = yaml.safe_load(writable.read_text(encoding="utf-8"))
    cfg["design_point"]["mach"] = 0.5
    cfg["ducts"]["inlet"]["sweep_deg"] = 7.0
    writable.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    assert m.revert()
    assert m.area != pytest.approx(area) and m.state.plane.sweep_deg == 7.0 and not m.dirty
    assert m.opening.target_area == pytest.approx(m.area)


def test_revert_of_a_config_without_an_inlet_removes_the_placed_one(scene, example_env, writable):
    cfg, _ = unplaced_cfg(example_env)
    writable.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    m = make(scene, writable, writable)
    m.begin_place()
    m.handle_click(skin_point(scene, 3.0, 0.4))
    assert m.placed
    assert m.revert() and not m.placed and not m.dirty


def test_revert_reports_a_broken_file_instead_of_crashing(scene, writable):
    m = make(scene, writable, writable)
    state = m.state
    writable.write_text("engine: [unclosed\n", encoding="utf-8")
    assert not m.revert()
    assert m.message.startswith("Revert failed") and m.state == state


# ---- readouts -------------------------------------------------------------------------------------------------------------------------------
def test_readouts_before_placement(scene, example_env):
    cfg, path = unplaced_cfg(example_env)
    m = InletModel(scene, cfg, path, None)
    rows = m.readouts()
    assert rows[0].label == "Face / envelope" and rows[0].level == "pass" and "0.870" in rows[0].value
    assert any(r.label == "Inlet" and "not placed" in r.value for r in rows)


def test_readouts_after_placement(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    rows = {r.label: r for r in m.readouts() if r.label}
    for label in ("Face / envelope", "Origin x y z", "Pitch / sweep", "|n.X|", "Area in plane", "Area projected",
                  "Area target", "Snapped edge length", "Free-edge clearance", "Aspect ratio (w/h)",
                  "Distance to face", "Offset dy, dz", "Offset / length"):
        assert label in rows
    assert rows["Area projected"].level == "pass" and rows["Free-edge clearance"].level == "pass"
    op = m.opening
    assert rows["Area in plane"].value == f"{op.area_plane:.5f} m^2"
    assert rows["Area projected"].value.startswith(f"{op.area_projected:.5f} m^2")
    assert rows["Area target"].value == f"{m.area:.5f} m^2"
    assert rows["Snapped edge length"].value == f"{op.snapped_length:.4f} m"
    c = m.opening_centroid
    assert rows["Distance to face"].value == f"{np.linalg.norm(np.array([5.0, 0.0, 0.0]) - c):.4f} m"
    assert rows["Offset dy, dz"].value == f"{-c[1]:+.4f}, {-c[2]:+.4f} m"


def test_a_steep_sweep_warns_about_the_flow_angle(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    m.set_param("sweep_deg", 70.0)
    rows = {r.label: r for r in m.readouts() if r.label}
    assert rows["|n.X|"].level == "warn" and rows["|n.X|"].value == f"{math.cos(math.radians(70)):.3f}"
    assert any("off the flow" in r.value and r.level == "warn" for r in m.readouts())


def test_the_engine_check_amber_when_it_warns_and_listed_once(scene, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["face_diameter"] = 0.4
    bad = load_scene(cfg, path)
    m = InletModel(bad, cfg, path, None)
    rows = m.readouts()
    assert rows[0].label == "Face / envelope" and rows[0].level == "warn" and "WARN" in rows[0].value
    assert sum("envelope diameter" in r.value or r.label == "Face / envelope" for r in rows) == 1


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
