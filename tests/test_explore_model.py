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
from ductmgr.section import distance_to_section
from ductmgr.viz import load_scene

import cfgutil


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


def vs(state):
    """Vertices of the inlet section (s = 0)."""
    return state.sections[0].shape.vertices


def sn(state):
    return state.sections[0].shape.snapped


def unplaced_cfg(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfgutil.unplace(cfg)
    return cfg, path


def body_scene(body, area=0.1, wings=(), face_diameter=0.5):
    """A stand-in scene around an analytic body (no OpenVSP)."""
    return SimpleNamespace(fuselage=body, sizing=SimpleNamespace(capture_area_per_inlet=area, n_inlets=2),
                           engine_check=None, warnings=[], face_center=np.array([6.0, 0.0, 0.0]),
                           face_diameter=face_diameter, wings=list(wings))


def body_model(body, origin, area=0.1, pitch=0.0, sweep=0.0, clock=None):
    cfg = {"ducts": {"inlet": {"origin": list(origin), "pitch_deg": pitch, "sweep_deg": sweep}}}
    return InletModel(body_scene(body, area), cfg, EXAMPLE_CONFIG, None, **({"clock": clock} if clock else {}))


# ---- loading ---------------------------------------------------------------------------------------------------
def test_the_inlet_is_loaded_from_the_config(scene, example_env):
    cfg, path, _ = example_env
    m = make(scene, path)
    ref = cfgutil.first(cfg)
    section = ref["key_sections"][0]
    assert m.state.plane == InletPlane(tuple(ref["origin"]), 0.0, 0.0)
    assert list(sn(m.state)) == section["snapped"] and len(m.state.sections) == 1
    assert np.array(vs(m.state)) == pytest.approx(np.array(section["vertices"]), abs=1e-5)
    assert m.state.sections[0].s == 0.0 and not m.notice
    assert m.section is not None and m.opening is not None and m.valid
    assert not m.dirty and not m.can_undo and m.mode == "view" and m.mode_text == "View"
    for i in (0, 1):  # snapped vertices are exactly on the current section
        assert distance_to_section(m.section, np.array(vs(m.state)[i]))[0] < 1e-6


def test_the_loaded_sketch_matches_the_phase_1_capture_area(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    assert abs(m.opening.delta) < sk.AREA_TOLERANCE and m.opening.target_area == pytest.approx(m.area)


def test_old_keys_are_ignored_with_a_one_line_notice(scene, example_env):
    cfg, path = unplaced_cfg(example_env)
    cfgutil.to_legacy(cfg)
    cfg["ducts"]["inlet"].update({"shape": {"width_top": 0.7}, "skin_fit": "flush", "centroid": [2, 0.5, 1.4],
                                  "tilt_deg": 5.0, "origin": None})
    notice = legacy_notice(cfg)
    assert "\n" not in notice and "ducts" in notice
    for key in ("shape", "skin_fit", "centroid", "tilt_deg"):
        assert key in notice
    m = InletModel(scene, cfg, path, None)
    assert not m.placed and m.notice == notice  # the old keys do not place anything
    assert any(r.value == notice and r.level == "warn" for r in m.readouts())


def test_null_old_keys_give_no_notice(example_env):
    cfg, _, _ = example_env
    cfg = cfgutil.to_legacy(copy.deepcopy(cfg))
    cfg["ducts"]["inlet"].update({"centroid": None, "normal": None, "polygon": None, "tilt_deg": None})
    assert "migrated" in legacy_notice(cfg)  # the old single-inlet format is migrated (only that is reported)
    assert "ignoring" not in legacy_notice(cfg)
    assert legacy_notice(copy.deepcopy(example_env[0])) == ""  # the new format has nothing to say
    assert legacy_notice({}) == ""


@pytest.mark.parametrize("patch, message", [
    ({"origin": [1.0, 2.0]}, "origin"),
    ({"pitch_deg": 95.0}, "pitch"),
])
def test_malformed_inlet_planes_are_rejected_clearly(example_env, patch, message):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfgutil.first(cfg).update(patch)
    with pytest.raises(ValueError, match=message):
        state_from_config(cfg)


@pytest.mark.parametrize("patch, message", [
    ({"vertices": [[0, 0]] * 2}, "3"),
    ({"vertices": [[0, 0, 0]] * 5}, "pairs"),
    ({"snapped": [True, False]}, "snapped"),
    ({"edges": ["straight"] * 4}, "edges"),
    ({"edges": ["wavy"] * 5}, "edges"),
    ({"s": 1.0}, "s"),
    ({"s": -0.1}, "s"),
    ({"s": None}, "'s'"),
])
def test_malformed_key_sections_are_rejected_clearly(example_env, patch, message):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfgutil.first(cfg)["key_sections"][0].update(patch)
    with pytest.raises(ValueError, match=message):
        state_from_config(cfg)


def test_two_key_sections_too_close_in_s_are_rejected(example_env):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    second = copy.deepcopy(cfgutil.first(cfg)["key_sections"][0])
    second["s"] = 0.005
    cfgutil.first(cfg)["key_sections"].append(second)
    with pytest.raises(ValueError, match="closer"):
        state_from_config(cfg)


def test_an_origin_without_key_sections_gets_a_default_sketch_and_is_dirty(scene, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfgutil.first(cfg)["key_sections"] = None
    m = InletModel(scene, cfg, path, None)
    assert sn(m.state)[:2] == (True, True) and m.valid
    assert m.dirty  # the sketch was created, not read from the file


# ---- editing the sketch ----------------------------------------------------------------------------------------------
def test_dragging_a_vertex_near_the_curve_snaps_it_onto_the_curve(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert sn(m.state)[2] is False
    target = np.array(vs(m.state)[2])
    p, _, _ = __import__("ductmgr.section", fromlist=["nearest_on_section"]).nearest_on_section(m.section, target)
    near = tuple(p + np.array([0.0, 0.004]))  # 4 mm off the curve
    assert m.drag_vertex(2, near, snap_tolerance=0.01)
    assert sn(m.state)[2] is True
    assert distance_to_section(m.section, np.array(vs(m.state)[2]))[0] < 1e-6  # on the curve
    assert m.dirty


def test_dragging_away_unsnaps_and_alt_disables_snapping(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert sn(m.state)[0] is True
    far = (vs(m.state)[0][0], vs(m.state)[0][1] + 0.2)
    clock.advance(5.0)
    assert m.drag_vertex(0, far, snap_tolerance=0.01) and sn(m.state)[0] is False  # dropped far: unsnapped
    assert vs(m.state)[0] == far
    clock.advance(5.0)
    on_curve = tuple(np.array(vs(m.state)[1]))  # exactly on the curve, snapped vertex 1
    m.drag_vertex(1, (on_curve[0], on_curve[1] + 0.0005), snap_tolerance=None)  # Alt: no snapping even when close
    assert sn(m.state)[1] is False and vs(m.state)[1] == (on_curve[0], on_curve[1] + 0.0005)


def test_a_vertex_can_slide_along_the_curve_staying_snapped(scene, example_env, clock):
    _, path, _ = example_env
    from ductmgr.section import nearest_on_section, point_at

    m = make(scene, path, clock=clock)
    _, s0, _ = nearest_on_section(m.section, vs(m.state)[0])
    ahead = point_at(m.section, s0 - 0.03)  # a point 3 cm further along the curve
    m.drag_vertex(0, tuple(ahead + [0.0, 0.001]), snap_tolerance=0.05)  # dropped 1 mm off it
    assert sn(m.state)[0]
    assert distance_to_section(m.section, np.array(vs(m.state)[0]))[0] < 1e-6
    _, s1, _ = nearest_on_section(m.section, vs(m.state)[0])
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
    assert sn(after) == sn(before)
    for i in (2, 3, 4):
        assert vs(after)[i] == vs(before)[i]  # free vertices keep their (u, v)
    for i in (0, 1):
        assert distance_to_section(m.section, np.array(vs(after)[i]))[0] < 1e-6  # on the NEW curve
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
    snapped_before = [vs(m.state)[i] for i in (0, 1)]
    assert m.fit_area() and "Area fitted" in m.message
    assert abs(m.opening.delta) < sk.AREA_TOLERANCE
    assert [vs(m.state)[i] for i in (0, 1)] == snapped_before  # the snapped vertices did not move
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


# ---- planform edge alignment ----------------------------------------------------------------------------------------------------
def sweep_text(m):
    return {r.label: r.value for r in m.readouts() if r.label}["Sweep source"]


def test_aligning_to_an_angle_sets_the_sweep_to_the_negative_edge_angle(scene, example_env, clock):
    from ductmgr.geometry import planform_trace_direction

    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    assert sweep_text(m) == "custom"  # nothing aligned yet
    clock.advance(10.0)
    assert m.align_to_edge(30.0, "manual")
    assert m.state.plane.sweep_deg == -30.0 and m.dirty
    assert "Aligned to manual (Λ = 30.0°)" in m.message
    dx, dy = planform_trace_direction(m.state.plane)
    assert dy > 0 and dx > 0 and math.degrees(math.atan2(dx, dy)) == pytest.approx(30.0, abs=1e-9)  # trails aft
    assert sweep_text(m) == "aligned to: manual (Λ = 30.0°)"
    assert m.opening.projection_factor == pytest.approx(math.cos(math.radians(30.0)))


def test_editing_the_sweep_afterwards_shows_custom_and_undo_restores_the_alignment_text(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    clock.advance(10.0)
    m.align_to_edge(30.0, "Wing LE")
    assert sweep_text(m) == "aligned to: Wing LE (Λ = 30.0°)"
    clock.advance(10.0)
    m.set_param("sweep_deg", -29.0)  # not persistent: any edit is "custom"
    assert sweep_text(m) == "custom"
    m.undo()
    assert sweep_text(m) == "aligned to: Wing LE (Λ = 30.0°)"  # the sweep is the aligned one again
    m.undo()
    assert m.state.plane.sweep_deg == 0.0 and sweep_text(m) == "custom"


def test_the_alignment_is_not_saved_and_is_forgotten_on_revert(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    clock.advance(10.0)
    m.align_to_edge(25.0, "Wing LE")
    assert m.save()
    saved = yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlets"][0]
    assert saved["sweep_deg"] == -25.0 and not any("align" in k for k in saved)  # only the sweep is stored
    fresh = make(scene, writable, writable)
    assert sweep_text(fresh) == "custom"  # a reloaded sweep is just a number
    clock.advance(10.0)
    m.set_param("sweep_deg", 5.0)
    m.align_to_edge(40.0, "Wing LE")
    m.revert()
    assert sweep_text(m) == "custom"


@pytest.mark.parametrize("pitch, edge, blocked", [
    (0.0, 70.0, False),  # |n.X| = 0.34: allowed (the < 0.5 warning stays)
    (0.0, 78.0, False),  # 0.208
    (0.0, 79.0, True),   # 0.191
    (0.0, 80.0, True),
    (60.0, 60.0, False),  # cos 60 cos 60 = 0.25
    (60.0, 70.0, True),   # 0.171
    (-60.0, 70.0, True),
])
def test_an_alignment_that_would_leave_n_dot_x_below_0_2_is_blocked(scene, example_env, clock, pitch, edge, blocked):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    clock.advance(10.0)
    m.set_param("pitch_deg", pitch)
    before, history = m.state, m.can_undo
    clock.advance(10.0)
    ok = m.align_to_edge(edge, "manual")
    factor = math.cos(math.radians(pitch)) * math.cos(math.radians(edge))
    if blocked:
        assert not ok and m.state == before and m.can_undo == history  # nothing changed, no history entry
        assert f"|n.X| would be {factor:.2f}" in m.message and "< 0.2" in m.message and "off the flow" in m.message
        assert sweep_text(m) == "custom"
    else:
        assert ok and m.state.plane.sweep_deg == -edge


def test_the_flow_angle_warning_stays_for_an_allowed_but_steep_alignment(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    assert m.align_to_edge(65.0, "manual")  # |n.X| = 0.42: allowed, but still below 0.5
    assert any("off the flow" in r.value and r.level == "warn" for r in m.readouts())
    rows = {r.label: r for r in m.readouts() if r.label}
    assert rows["|n.X|"].level == "warn"


def test_the_wing_edge_is_read_in_the_section_spanning_the_inlet_y(scene, example_env, clock):
    wing = vspio_wing()
    cfg = {"ducts": {"inlet": {"origin": [4.0, 0.3, 1.0]}}}
    m = InletModel(body_scene(bodies.cylinder(1.0), wings=[wing]), cfg, EXAMPLE_CONFIG, None, clock=clock)
    inner = m.wing_edge(0, "LE")
    assert (inner.section_index, inner.angle_deg) == (0, 35.0)
    assert m.wing_edge(0, "TE").angle_deg == 12.0
    clock.advance(10.0)
    m.set_origin(4.0, 1.2, 1.0)  # an inlet further out: the second section applies
    assert m.wing_edge(0, "LE").angle_deg == 15.0 and m.wing_edge(0, "LE").section_index == 1
    assert m.align_to_edge(m.wing_edge(0, "LE").angle_deg, "Wing LE")
    assert m.state.plane.sweep_deg == -15.0 and sweep_text(m) == "aligned to: Wing LE (Λ = 15.0°)"


def test_aligning_needs_a_placed_inlet(scene, example_env):
    cfg, path = unplaced_cfg(example_env)
    m = InletModel(scene, cfg, path, None)
    assert not m.align_to_edge(30.0, "manual") and not m.can_undo


def test_the_left_inlet_follows_the_mirrored_edge_after_an_alignment(scene, example_env):
    from ductmgr.geometry import planform_trace_direction

    _, path, _ = example_env
    m = make(scene, path)
    m.align_to_edge(35.0, "manual")
    left = m.state.plane.mirrored()
    assert left.sweep_deg == 35.0 and left.origin[1] == -m.state.plane.origin[1]
    dx, dy = planform_trace_direction(left)  # oriented outboard (+y), so compare as lines
    right_dx, right_dy = planform_trace_direction(m.state.plane)
    assert (dx, dy) == pytest.approx((-right_dx, right_dy), abs=1e-12) or (dx, dy) == pytest.approx((right_dx, -right_dy), abs=1e-12)


def vspio_wing():
    """A two-section planform: LE 35 / TE 12 degrees out to y = 0.9, then LE 15 / TE 15 degrees."""
    from ductmgr import vspio

    return vspio.WingPlanform("id", "Wing", (vspio.WingSection(0.0, 0.9, 35.0, 12.0),
                                             vspio.WingSection(0.9, 2.5, 15.0, 15.0)), False)


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
    assert sn(m.state)[:2] == (True, True)
    placed = m.state
    assert not m.handle_click(skin_point(scene, 4.0, 0.2)) and m.state == placed  # the mode ended


def test_placing_again_moves_the_origin_and_keeps_the_sketch_and_angles(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    m.set_param("sweep_deg", 15.0)
    verts = vs(m.state)
    free = [verts[i] for i in (2, 3, 4)]
    assert m.begin_place() and m.handle_click(skin_point(scene, 3.4, 0.5))
    assert m.state.plane.sweep_deg == 15.0 and m.state.plane.origin[0] == pytest.approx(3.4)
    assert [vs(m.state)[i] for i in (2, 3, 4)] == free


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
    m.drag_vertex(0, (vs(m.state)[0][0], -0.8), snap_tolerance=None)  # a free vertex deep inside the fuselage


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
    inlet = after["ducts"]["inlets"][0]
    assert inlet["origin"] == pytest.approx([round(c, 6) for c in m.state.plane.origin])
    assert inlet["pitch_deg"] == -6.0 and inlet["sweep_deg"] == 12.0
    saved = inlet["key_sections"][0]
    assert np.array(saved["vertices"]) == pytest.approx(np.array([[round(u, 6), round(v, 6)] for u, v in vs(m.state)]))
    assert saved["snapped"] == list(sn(m.state)) and all(isinstance(f, bool) for f in saved["snapped"])
    assert "vertices" not in inlet and "snapped" not in inlet
    for section in ("engine", "design_point", "checks"):
        assert after[section] == before[section]
    assert {k: v for k, v in after["model"].items() if k != "skin_bodies"} == before["model"]  # only the skin is added
    assert after["model"]["skin_bodies"] == [before["model"]["fuselage_name"]]
    # a fresh model reads the same inlet back, in (u, v)
    again = make(scene, writable, writable)
    assert again.state.plane == InletPlane(tuple(inlet["origin"]), -6.0, 12.0)
    assert np.array(vs(again.state)) == pytest.approx(np.array(vs(m.state)), abs=1e-5)
    assert sn(again.state) == sn(m.state) and not again.dirty


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
    assert m.inlets_definition() == []
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
    cfg["ducts"]["inlets"][0]["sweep_deg"] = 7.0
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


# ---- Phase 3a: the general section editor --------------------------------------------------------------------------------------------------
def legacy_cfg(example_env):
    """The oldest format: ducts.inlet.vertices / snapped and no key_sections (no ducts.inlets either)."""
    cfg, path, _ = example_env
    section = cfgutil.first(cfg)["key_sections"][0]
    return cfgutil.to_five_vertex(copy.deepcopy(cfg)), path, section


def test_an_old_five_vertex_config_loads_as_a_straight_inlet_section_with_a_notice(scene, example_env):
    cfg, path, section = legacy_cfg(example_env)
    m = InletModel(scene, cfg, path, None)
    shape = m.state.sections[0].shape
    assert len(m.state.sections) == 1 and m.state.sections[0].s == 0.0
    assert shape.n == 5 and set(shape.edges) == {"straight"} and set(shape.continuity) == {"corner"}
    assert list(shape.snapped) == section["snapped"]
    assert np.array(shape.vertices) == pytest.approx(np.array(section["vertices"]), abs=1e-5)
    assert "\n" not in m.notice and "5-vertex" in m.notice and "straight-edged" in m.notice
    assert any(r.value == m.notice and r.level == "warn" for r in m.readouts())
    assert m.valid


def test_a_single_inlet_config_migrates_to_a_one_item_list_with_mirror_and_symmetry_xz(scene, example_env):
    cfg, _, _ = example_env
    legacy = cfgutil.to_legacy(copy.deepcopy(cfg))
    state = state_from_config(legacy)
    assert len(state.inlets) == 1 and state.symmetry == "xz" and state.inlets[0].mirror is True
    assert state.inlets[0].plane == state_from_config(cfg).inlets[0].plane
    notice = legacy_notice(legacy)
    assert "\n" not in notice and "migrated to ducts.inlets" in notice and "mirror true" in notice and "xz" in notice
    legacy["ducts"]["n_inlets"] = 1  # a single inlet that is not mirrored
    assert state_from_config(legacy).inlets[0].mirror is False and "mirror false" in legacy_notice(legacy)


def test_the_migrated_config_gives_the_same_ducts_as_the_new_format(scene, example_env):
    cfg, path, _ = example_env
    old = InletModel(scene, cfgutil.to_legacy(copy.deepcopy(cfg)), path, None)
    new = InletModel(scene, copy.deepcopy(cfg), path, None)
    assert old.state.inlets[0].sections == new.state.inlets[0].sections and old.state.symmetry == new.state.symmetry
    assert old.n_branches == new.n_branches == 2 and old.duct.n_branches == 2
    assert old.centreline.length == pytest.approx(new.centreline.length)


def test_saving_a_legacy_config_migrates_it_to_the_inlets_list(scene, example_env, writable):
    cfg, _, _ = example_env
    legacy = cfgutil.to_five_vertex(copy.deepcopy(cfg))
    writable.write_text(yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8")
    m = make(scene, writable, writable)
    assert m.notice and m.save() and m.notice == ""
    saved = yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]
    assert "inlet" not in saved and "key_sections" not in saved
    assert saved["symmetry"] == "xz" and len(saved["inlets"]) == 1 and saved["inlets"][0]["mirror"] is True
    section = saved["inlets"][0]["key_sections"][0]
    assert section["s"] == 0.0 and section["edges"] == ["straight"] * 5
    assert make(scene, writable, writable).notice == ""


def test_old_vertices_next_to_the_new_inlets_list_are_ignored_with_a_notice(example_env):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"] = {"vertices": [[0, 0]] * 5}
    assert "superseded" in legacy_notice(cfg)


def test_inserting_a_vertex_on_an_edge(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    before = m.state
    a, b = np.array(vs(before)[2]), np.array(vs(before)[3])
    mid = tuple((a + b) / 2 + [0.002, 0.0])  # the edge from vertex 2 to 3 is free and straight
    assert not before.sections[0].shape.edge_is_snapped(2)
    clock.advance(10.0)
    at = m.insert_vertex(2, mid)
    assert at == 3 and len(vs(m.state)) == 6 and len(sn(m.state)) == 6 and sn(m.state)[3] is False
    assert m.valid
    # the new vertex lies on the old straight edge (the click was 2 mm off it)
    p = np.array(vs(m.state)[3])
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    assert 0.0 < t < 1.0 and np.linalg.norm(a + t * (b - a) - p) < 1e-9
    assert m.undo() and m.state == before


def test_a_vertex_inserted_on_a_snapped_edge_lands_on_the_skin_and_stays_snapped(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    before = m.state
    assert before.sections[0].shape.edge_is_snapped(0)
    a, b = np.array(vs(before)[0]), np.array(vs(before)[1])
    click = tuple((a + b) / 2 + [0.0, 0.003])  # a little off the curve
    at = m.insert_vertex(0, click)
    assert at == 1 and sn(m.state)[1] is True and len(vs(m.state)) == 6
    assert distance_to_section(m.section, np.array(vs(m.state)[1]))[0] < 1e-6
    assert m.state.sections[0].shape.edge_is_snapped(0) and m.state.sections[0].shape.edge_is_snapped(1)
    assert m.valid and m.undo() and m.state == before


def test_deleting_vertices_keeps_a_valid_outline_down_to_three(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    clock.advance(10.0)
    assert m.delete_vertex(4) and len(vs(m.state)) == 4 and m.valid
    clock.advance(10.0)
    assert m.delete_vertex(3) and len(vs(m.state)) == 3
    before = m.state
    clock.advance(10.0)
    assert not m.delete_vertex(2) and m.state == before  # a section needs at least 3 vertices
    assert m.undo() and len(vs(m.state)) == 4


def test_curved_edges_and_continuity_through_the_model(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    base_area = m.opening.area_plane
    clock.advance(10.0)
    assert m.set_edge_type(3, "curved")
    shape = m.state.sections[0].shape
    assert shape.edge_is_curved(3) and shape.handles[3] is not None
    clock.advance(10.0)
    assert m.drag_handle(3, 0, (vs(m.state)[3][0] + 0.03, vs(m.state)[3][1] + 0.06))
    assert m.opening.area_plane != pytest.approx(base_area, abs=1e-9)  # the area is that of the curved outline
    clock.advance(10.0)
    assert m.set_continuity(3, "smooth") and m.state.sections[0].shape.continuity[3] == "smooth"
    assert m.valid
    before = m.state
    assert not m.set_edge_type(0, "curved") and m.state == before  # a snapped edge is not editable


def test_a_free_curved_edge_changes_the_projected_area_the_fit_works_on(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    clock.advance(10.0)
    m.set_edge_type(3, "curved")
    clock.advance(10.0)
    m.drag_handle(3, 0, (vs(m.state)[3][0] + 0.05, vs(m.state)[3][1] + 0.1))
    assert abs(m.opening.delta) > sk.AREA_TOLERANCE
    snapped = [vs(m.state)[i] for i in (0, 1)]
    clock.advance(10.0)
    assert m.fit_area() and abs(m.opening.delta) < sk.AREA_TOLERANCE
    assert [vs(m.state)[i] for i in (0, 1)] == snapped


# ---- key sections --------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def cyl_model(clock):
    return body_model(bodies.cylinder(1.0), (4.0, 0.3, 0.9), area=0.12, clock=clock)


def test_the_junction_and_the_engine_face_are_listed_last_and_are_fixed(cyl_model):
    m = cyl_model
    rows = m.section_list()
    assert [r[1:] for r in rows] == [("inlet", True), ("junction: branch end (fixed)", False), ("engine face (fixed)", False)]
    assert rows[1][0] == pytest.approx(m.junction_s) and rows[2][0] == 1.0
    assert m.select_section(len(rows) - 1) and m.active_is_face and m.active_s == 1.0 and m.active_is_fixed
    assert m.active_shape.n >= 3 and all(e == "curved" for e in m.active_shape.edges)  # a circle
    face = m.active_opening
    assert face.area_plane == pytest.approx(m.face_area, rel=2e-3)
    before = m.state
    assert not m.drag_vertex(0, (0.0, 0.0), None) and not m.delete_vertex(0) and not m.fit_area()
    assert m.state == before and not m.select_section(len(rows))


def test_the_junction_section_is_the_derived_half_of_the_trunk_start_and_cannot_be_edited(cyl_model, clock):
    m = cyl_model
    assert m.select_section(m.n_sections) and m.active_is_junction and not m.active_is_face and m.active_is_fixed
    assert m.active_s == pytest.approx(m.junction_s)
    op = m.active_opening
    assert op.area_plane == pytest.approx(0.5 * m.areas.junction_area, rel=2e-3)  # half the trunk area
    shape = m.active_shape
    pts = np.array(shape.vertices)
    cut = pts[:, 0].max() if (np.abs(pts[:, 0] - pts[:, 0].max()) < 1e-9).sum() > 3 else pts[:, 0].min()
    assert (np.abs(pts[:, 0] - cut) < 1e-9).sum() > 3  # one side is the straight cut on the symmetry plane
    before = m.state
    clock.advance(10.0)
    assert not m.drag_vertex(0, (0.0, 0.0), None) and not m.insert_vertex(0, (0.0, 0.0)) and not m.fit_area()
    assert m.state == before
    # it follows the centreline / area law and is identical to the branch end of the built duct
    m.set_area_points(m.areas.with_junction_area(0.3), None)
    assert m.active_opening.area_plane == pytest.approx(0.15, rel=2e-3)
    ring = m.duct.branch[-1].ring
    assert np.array(m.active_shape.vertices) == pytest.approx(ring, abs=1e-12)


def test_the_junction_row_survives_undo_and_the_selection_follows_it(cyl_model, clock):
    m = cyl_model
    clock.advance(10.0)
    m.add_key_section(0.3)
    assert m.select_section(2) and m.active_is_junction
    clock.advance(10.0)
    assert m.set_wall_thickness(0.01) and m.active_is_junction  # an edit keeps the fixed row selected
    m.undo()
    m.undo()  # the added key section goes away: the junction row moves up one place
    assert m.n_sections == 1 and m.active_is_junction


def test_a_single_inlet_has_no_junction_row(clock):
    cfg = {"ducts": {"n_inlets": 1, "inlet": {"origin": [4.0, 0.3, 0.9]}}}  # one inlet, not mirrored
    scene = body_scene(bodies.cylinder(1.0), 0.12)
    m = InletModel(scene, cfg, EXAMPLE_CONFIG, None, clock=clock)
    assert [r[1] for r in m.section_list()] == ["inlet", "engine face (fixed)"] and not m.junction_row


def test_adding_a_key_section_copies_the_previous_shape_at_the_area_law_size(cyl_model, clock):
    m = cyl_model
    clock.advance(10.0)
    assert m.add_key_section(0.3)
    assert [ks.s for ks in m.state.sections] == [0.0, 0.3] and m.active == 1 and not m.active_is_inlet
    ks = m.state.sections[1]
    assert ks.shape.n == m.state.sections[0].shape.n and not any(ks.shape.snapped)
    assert m.active_opening.area_plane == pytest.approx(m.area_at(0.3), rel=1e-9)
    assert abs(m.active_opening.delta) < 1e-9 and m.valid
    assert m.undo() and len(m.state.sections) == 1


def test_key_section_areas_follow_the_branch_curve_and_the_trunk_curve(cyl_model, clock):
    m = cyl_model
    clock.advance(10.0)
    assert m.add_key_section(round(m.junction_s * 0.5, 4))  # the inlet's branch: the branch curve
    [(ks, ev)] = list(zip(m.state.sections[1:], m.evaluations[1:]))
    assert ev.area_plane == pytest.approx(float(m.areas.branch_area(ks.s / m.junction_s)), rel=1e-9)
    clock.advance(10.0)
    assert m.select_trunk() and m.add_key_section(0.5)  # the trunk: its own curve, s along the trunk
    [(kt, evt)] = list(zip(m.state.sections[1:], m.evaluations[1:]))
    assert kt.s == 0.5 and evt.area_plane == pytest.approx(float(m.areas.trunk_area(0.5)), rel=1e-9)
    assert m.evaluations[0].area_plane == pytest.approx(m.areas.junction_area, rel=1e-9)  # the trunk start


def test_a_key_section_keeps_its_shape_when_edited_but_not_its_size(cyl_model, clock):
    m = cyl_model
    m.add_key_section(0.4)
    target = m.area_at(0.4)
    clock.advance(10.0)
    n = m.active_shape.n
    start = m.active_shape.vertices[2]
    for k in range(5):  # a drag, then the end of the drag
        clock.advance(0.01)
        m.drag_vertex(2, (start[0] + 0.01 * (k + 1), start[1] + 0.02 * (k + 1)), None)
    m.finish_edit()
    assert m.active_shape.n == n
    target = m.area_at(0.4)  # (the junction, hence A(0.4), follows the edited shape)
    assert m.active_opening.area_plane == pytest.approx(target, rel=1e-9)  # resized back to A(s)
    assert m.active_shape.vertices[2] != start  # but the shape did change


def test_a_drag_of_a_key_section_and_its_rescale_are_one_undo_step(cyl_model, clock):
    m = cyl_model
    clock.advance(10.0)
    m.add_key_section(0.4)
    start_state = m.state
    clock.advance(10.0)
    v = m.active_shape.vertices[1]
    for k in range(4):
        clock.advance(0.01)
        m.drag_vertex(1, (v[0] + 0.01 * (k + 1), v[1]), None)
    m.finish_edit()
    assert m.undo() and m.state == start_state


def test_rescaling_a_section_that_already_has_its_area_is_a_no_op(cyl_model):
    m = cyl_model
    m.add_key_section(0.4)
    state = m.state
    assert not m.finish_edit() and m.state == state  # nothing to do: no spurious history


def test_removing_key_sections(cyl_model, clock):
    m = cyl_model
    clock.advance(10.0)
    m.add_key_section(0.2)
    clock.advance(10.0)
    m.add_key_section(0.4)
    assert [ks.s for ks in m.state.sections] == [0.0, 0.2, 0.4]
    assert m.remove_key_section(1) and [ks.s for ks in m.state.sections] == [0.0, 0.4]
    assert not m.remove_key_section(0) and not m.remove_key_section(5)  # the inlet and bad indexes stay
    assert m.undo() and len(m.state.sections) == 3


def test_bad_key_section_positions_are_refused_with_a_message(cyl_model):
    m = cyl_model
    for s in (0.0, 1.0, -0.2, 1.5):
        assert not m.add_key_section(s) and m.message
    assert m.add_key_section(0.3) and not m.add_key_section(0.305)  # too close to an existing one
    assert len(m.state.sections) == 2


def test_selecting_a_section_shows_it_in_the_editor_without_touching_history(cyl_model):
    m = cyl_model
    m.add_key_section(0.3)
    history = len(m._past)
    assert m.select_section(0) and m.active_is_inlet and m.active_curve is m.section
    assert m.select_section(1) and m.active_curve is None  # only the inlet section has a skin to snap to
    assert not m.select_section(7) and len(m._past) == history


def test_key_sections_survive_a_save_and_a_reload(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    clock.advance(10.0)
    m.add_key_section(0.45)
    clock.advance(10.0)
    m.set_edge_type(0, "curved")
    m.finish_edit()
    clock.advance(10.0)
    m.select_section(0)
    m.insert_vertex(2, np.array(vs(m.state)[2]) * 0.5 + np.array(vs(m.state)[3]) * 0.5)
    assert m.save()
    again = make(scene, writable, writable)
    assert [ks.s for ks in again.state.sections] == [0.0, 0.45] and not again.dirty
    for a, b in zip(again.state.sections, m.state.sections):
        assert a.shape.n == b.shape.n and a.shape.edges == b.shape.edges and a.shape.continuity == b.shape.continuity
        assert np.array(a.shape.vertices) == pytest.approx(np.array(b.shape.vertices), abs=2e-6)
    assert again.valid


def test_an_invalid_intermediate_section_blocks_saving(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    clock.advance(10.0)
    m.add_key_section(0.45)
    v = m.active_shape.vertices
    clock.advance(10.0)
    m.drag_vertex(3, tuple(1.3 * np.array(v[0])), None)  # the apex dragged to the opposite corner: a bow tie
    m.finish_edit()
    assert not m.valid and not m.save() and m.message.startswith("ERROR: not saved")


def test_readouts_list_the_intermediate_sections_and_the_face(cyl_model):
    m = cyl_model
    m.add_key_section(0.3)
    rows = {r.label: r for r in m.readouts() if r.label}
    assert "Section s=0.300" in rows and "Section s=1.000" in rows
    assert rows["Section s=0.300"].level == "pass" and "A(s)" in rows["Section s=0.300"].value
    assert "fixed" in rows["Section s=1.000"].value


def test_moving_the_plane_rescales_the_other_sections_to_the_new_area_law(cyl_model, clock):
    m = cyl_model
    m.add_key_section(0.3)
    clock.advance(10.0)
    m.set_param("sweep_deg", 40.0)  # the in-plane inlet area grows by 1 / cos(40)
    assert m.evaluations[1].area_plane == pytest.approx(m.area_at(0.3), rel=1e-9)
    assert m.inlet_plane_area == pytest.approx(0.12 / math.cos(math.radians(40.0)))


# ---- Phase 3b: centrelines, area law, bodies, checks ---------------------------------------------------------------------------------
from ductmgr import areacurve  # noqa: E402
from ductmgr.areacurve import AreaPoints  # noqa: E402
from ductmgr.centreline import CentrelineParams  # noqa: E402


def test_the_duct_parameters_are_loaded_from_the_config(scene, example_env):
    cfg, path, _ = example_env
    m = make(scene, path)
    d = m.state.duct
    assert d.centreline.junction == (cfg["ducts"]["junction"]["point"][0], cfg["ducts"]["junction"]["point"][2])
    assert d.centreline.junction_pitch_deg is None and d.centreline.start_handle is None
    assert d.area == AreaPoints()
    assert d.wall_thickness == cfg["ducts"]["wall_thickness"] and d.lip_extension == cfg["ducts"]["lip_extension"]
    assert d.lip_mode == "blunt"
    assert m.settings.checks.lip_length == cfg["checks"]["lip_length"]
    assert not m.dirty


def test_lip_extension_is_still_read_from_its_old_place_and_defaults_apply(scene, example_env):
    cfg, path, _ = example_env
    cfg = cfgutil.to_legacy(copy.deepcopy(cfg))
    ducts = cfg["ducts"]
    ducts.pop("lip_extension")
    ducts["inlet"]["lip_extension"] = 0.123
    ducts["wall_thickness"] = None
    d = state_from_config(cfg).duct
    assert d.lip_extension == 0.123 and d.wall_thickness == em.DEFAULT_WALL_THICKNESS
    ducts["inlet"].pop("lip_extension")
    assert state_from_config(cfg).duct.lip_extension == em.DEFAULT_LIP_EXTENSION


@pytest.mark.parametrize("patch, message", [
    ({"wall_thickness": -0.001}, "wall_thickness"),
    ({"lip_extension": -0.1}, "lip_extension"),
    ({"lip_mode": "rounded"}, "lip_mode"),
    ({"junction": {"point": [1.0, 2.0]}}, "junction.point"),
    ({"centreline": [1]}, "centreline"),
    ({"area_points": {"branch": [[1.5, 0.1]]}}, "between 0 and 1"),
    ({"area_law": "parabolic"}, "area_law"),
    ({"n_stations_branch": 1}, "n_stations_branch"),
])
def test_malformed_duct_settings_are_rejected_clearly(example_env, patch, message):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"].update(patch)
    with pytest.raises(ValueError, match=message):
        state_from_config(cfg)
        em.duct_settings(cfg)


def test_the_centreline_is_built_from_the_inlet_centroid_the_junction_and_the_face(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    cl = m.centreline
    assert cl is not None
    assert cl.start == pytest.approx(m.opening_centroid, abs=2e-3)  # the centroid of the sketched opening
    assert cl.face_centre == pytest.approx(scene.face_center)
    assert cl.junction[0] == m.state.duct.centreline.junction[0] and cl.junction[1] == 0.0
    assert m.junction_s == pytest.approx(cl.s_junction)
    assert cl.branch.tangents([0.0])[0] == pytest.approx(-m.state.plane.normal, abs=1e-12)


def test_dragging_the_junction_is_one_undo_step_and_keeps_g1(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    start = m.state
    for k in range(5):
        clock.advance(0.01)
        assert m.drag_junction(3.5 + 0.01 * k, 0.1 + 0.01 * k)
    assert m.state.duct.centreline.junction == pytest.approx((3.54, 0.14))
    cl = m.centreline
    assert cl.branch.tangents([1.0])[0] == pytest.approx(cl.trunk.tangents([0.0])[0], abs=1e-12)
    clock.advance(10.0)
    assert m.undo() and m.state == start and not m.can_undo


def test_handle_edits_and_their_defaults(scene, example_env, clock):
    _, path, _ = example_env
    m = make(scene, path, clock=clock)
    default = m.centreline.resolved
    clock.advance(10.0)
    assert m.set_centreline("h", start_handle=0.9, junction_pitch_deg=20.0, junction_out_handle=0.3)
    r = m.centreline.resolved
    assert r.start_handle == 0.9 and r.junction_pitch_deg == 20.0 and r.junction_out_handle == 0.3
    assert r.end_handle == pytest.approx(default.end_handle)  # untouched ones stay default
    t = m.centreline.junction_tangent
    assert math.degrees(math.atan2(t[2], t[0])) == pytest.approx(20.0)
    clock.advance(10.0)
    assert not m.set_centreline("h", end_handle=-1.0) and "negative" in m.message
    with pytest.raises(KeyError):
        m.set_centreline("h", bogus=1.0)
    clock.advance(10.0)
    assert m.reset_centreline() and m.state.inlet.start_handle is None
    assert m.centreline.resolved.start_handle == pytest.approx(default.start_handle)


def test_the_lock_blocks_junction_drags(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    m.set_locked(True)
    before = m.state
    assert not m.drag_junction(3.0, 0.5) and m.state == before


def test_key_sections_follow_the_centreline_and_the_area_law(cyl_model, clock):
    m = cyl_model
    clock.advance(10.0)
    m.add_key_section(0.3)
    clock.advance(10.0)
    assert m.drag_junction(3.0, 0.3) or m.set_centreline("c", junction=(3.0, 0.3))
    assert m.evaluations[1].area_plane == pytest.approx(m.area_at(0.3), rel=1e-9)
    clock.advance(10.0)
    assert m.area_set_junction(0.4)
    assert m.areas.junction_area == 0.4 and m.evaluations[1].area_plane == pytest.approx(m.area_at(0.3), rel=1e-9)
    assert m.undo() and m.areas.junction_area != 0.4


def test_area_points_can_be_moved_added_and_removed(cyl_model, clock):
    m = cyl_model
    default = m.areas.branch_interior
    clock.advance(10.0)
    assert m.area_move_point("branch", 1, 0.55, 0.115) and m.state.inlet.area_branch is not None
    assert m.areas.branch_interior[1] == (0.55, 0.115) and m.areas.trunk_interior == m.areas.trunk_interior
    clock.advance(10.0)
    m.area_move_point("branch", 1, 5.0, 0.115)  # dragged far right: kept between its neighbours
    assert m.areas.branch_interior[1][0] < m.areas.branch_interior[2][0]
    clock.advance(10.0)
    assert m.area_add_point("trunk", 0.6) and len(m.areas.trunk_interior) == len(default) + 1
    assert not m.area_add_point("trunk", 0.6)  # too close to the point just added
    n = len(m.areas.trunk_interior)
    assert m.area_remove_point("trunk", 0) and len(m.areas.trunk_interior) == n - 1
    assert not m.area_remove_point("trunk", 99)
    assert m.reset_area() and m.state.duct.area == AreaPoints() and m.state.inlet.area_branch is None


def test_dragging_an_area_point_is_one_undo_step(cyl_model, clock):
    m = cyl_model
    start = m.state
    for k in range(6):
        clock.advance(0.01)
        m.area_move_point("branch", 0, 0.25, 0.11 + 0.001 * k)
    clock.advance(10.0)
    assert m.undo() and m.state == start


def test_invalid_area_points_are_refused_with_a_message(cyl_model):
    m = cyl_model
    before = m.state
    assert not m.set_area_points(AreaPoints(None, ((1.5, 0.1),), None)) and m.state == before and m.message


def test_wall_thickness_and_lip_extension(cyl_model, clock):
    m = cyl_model
    assert m.set_wall_thickness(0.012) and m.state.duct.wall_thickness == 0.012
    assert m.set_lip_extension(0.08) and m.state.duct.lip_extension == 0.08
    assert not m.set_wall_thickness(0.0) and not m.set_lip_extension(-0.1)
    assert m.duct.spec.wall_thickness == 0.012 and m.duct.spec.lip_extension == 0.08


def test_the_duct_spec_uses_the_sketched_sections_and_the_scene(cyl_model):
    m = cyl_model
    m.add_key_section(0.3)
    spec = m.duct_spec()
    assert [k.s for k in spec.inlets[0].keys] == [0.0, 0.3] and m.duct.n_branches == 2 and spec.symmetry == "xz"
    assert [k.s for k in spec.trunk] == [0.0]  # the trunk start: its own key section
    assert spec.face_centre == pytest.approx([6.0, 0.0, 0.0]) and spec.face_diameter == 0.5
    assert spec.inlets[0].inlet_area == pytest.approx(m.opening.area_plane)
    assert m.duct.areas.a_in == pytest.approx(m.opening.area_plane)


def test_bodies_and_checks_are_built_on_demand_and_cached_per_state(cyl_model, clock):
    m = cyl_model
    assert not m.checks_current
    assert m.ensure_built() and m.checks_current and set(m.bodies) == set(dm_names())
    first = m.checks
    assert m.checks is first  # cached
    clock.advance(10.0)
    m.set_wall_thickness(0.02)
    assert not m.checks_current  # a new state invalidates it
    assert m.checks is not first and m.checks_current


def dm_names(name="inlet 1"):
    return [f"{k}_{p}" for k in ("neg", "pos") for p in (f"branch_{name}", f"branch_{name}_mirror", "trunk")]


def test_there_is_no_duct_without_an_engine_face(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["geom_name"] = None
    cfg["ducts"]["engine_face"]["centre"] = None
    s = load_scene(cfg, path)
    m = InletModel(s, cfg, path, None)
    assert m.duct is None and m.centreline is None and not m.ensure_built() and not m.can_build
    assert m.junction_s == m.settings.junction_fraction  # the old proxy
    assert all("Junction" != r.label for r in m.readouts())


def test_the_readouts_list_the_duct_numbers_and_the_checks(cyl_model):
    m = cyl_model
    rows = {r.label: r for r in m.readouts() if r.label}
    for label in ("Junction", "Path length", "Area junction", "Offset / length", "Centreline curvature",
                  "Cone half-angle branch", "Cone half-angle trunk"):
        assert label in rows
    assert "External region" not in rows  # the heavy checks are not computed behind the user's back
    m.ensure_built()
    rows = {r.label: r for r in m.readouts() if r.label}
    for label in ("External region", "Wall left (internal)", "Other Geoms", "Line of sight"):
        assert label in rows
    m.live_checks = True
    m.set_wall_thickness(0.02)
    assert any(r.label == "Body checks" and "updating" in r.value for r in m.readouts())


def test_duct_parameters_survive_a_save_and_a_reload(scene, writable, clock):
    before = yaml.safe_load(writable.read_text(encoding="utf-8"))
    m = make(scene, writable, writable, clock=clock)
    for step in (lambda: m.drag_junction(3.4, 0.35), lambda: m.set_centreline("h", start_handle=0.7, junction_pitch_deg=12.0),
                 lambda: m.area_move_point("branch", 1, 0.5, 0.1), lambda: m.area_set_junction(0.19),
                 lambda: m.set_wall_thickness(0.03), lambda: m.set_lip_extension(0.06)):
        clock.advance(10.0)
        assert step()
    assert m.save()
    ducts = yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]
    assert ducts["junction"]["point"] == [3.4, 0.0, 0.35]
    assert ducts["inlets"][0]["start_handle"] == 0.7 and ducts["centreline"]["junction_pitch_deg"] == 12.0
    assert "start_handle" not in ducts["centreline"]  # the inlet handles belong to their inlet
    assert "end_handle" not in ducts["centreline"] and ducts["trunk"]["end_handle"] is None  # the trunk's; null = default
    assert "area_points" not in ducts  # moved into ducts.trunk
    assert ducts["trunk"]["junction_area"] == 0.19 and ducts["inlets"][0]["area_points"][1] == [0.5, 0.1]
    assert ducts["wall_thickness"] == 0.03 and ducts["lip_extension"] == 0.06 and ducts["lip_mode"] == "blunt"
    after = yaml.safe_load(writable.read_text(encoding="utf-8"))
    for section in ("engine", "design_point", "checks"):
        assert after[section] == before[section]
    assert {k: v for k, v in after["model"].items() if k != "skin_bodies"} == before["model"]  # only the skin is added
    assert after["model"]["skin_bodies"] == [before["model"]["fuselage_name"]]
    again = make(scene, writable, writable)
    assert not again.dirty and again.state.duct.centreline == m.state.duct.centreline
    assert again.state.inlet.start_handle == 0.7
    assert again.state.duct.wall_thickness == 0.03 and again.state.duct.lip_extension == 0.06
    for got, want in zip(again.areas.branch_interior + again.areas.trunk_interior,
                         m.areas.branch_interior + m.areas.trunk_interior):
        assert got == pytest.approx(want, abs=1e-6)


def test_saving_moves_a_legacy_lip_extension_out_of_the_inlet_block(scene, example_env, writable):
    text = cfgutil.to_legacy(yaml.safe_load(writable.read_text(encoding="utf-8")))
    text["ducts"].pop("lip_extension")
    text["ducts"]["inlet"]["lip_extension"] = 0.09
    writable.write_text(yaml.safe_dump(text, sort_keys=False), encoding="utf-8")
    m = make(scene, writable, writable)
    assert m.state.duct.lip_extension == 0.09
    assert m.save()
    saved = yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]
    assert "inlet" not in saved and saved["lip_extension"] == 0.09


def test_revert_restores_the_duct_parameters(scene, writable, clock):
    m = make(scene, writable, writable, clock=clock)
    saved = m.state.duct
    clock.advance(10.0)
    m.drag_junction(3.2, 0.5)
    clock.advance(10.0)
    m.set_wall_thickness(0.04)
    assert m.dirty and m.revert() and not m.dirty and m.state.duct == saved


# ---- the Y junction settings ----------------------------------------------------------------------------------------------------
def test_the_junction_settings_come_from_the_config_with_defaults(example_env):
    cfg, _, _ = example_env
    s = em.duct_settings(cfg)
    assert s.splitter_length == 0.0 and s.overlap == cfg["ducts"]["junction"]["overlap"]
    bare = copy.deepcopy(cfg)
    bare["ducts"]["junction"].pop("overlap")
    bare["ducts"]["junction"].pop("splitter_length")
    s = em.duct_settings(bare)
    assert s.splitter_length == 0.0 and s.overlap == em.dm.DEFAULT_OVERLAP
    bare["ducts"]["junction"]["overlap"] = -0.01
    with pytest.raises(ValueError, match="overlap"):
        em.duct_settings(bare)
    bare["ducts"]["junction"].update(overlap=0.02, splitter_length=-1.0)
    with pytest.raises(ValueError, match="splitter_length"):
        em.duct_settings(bare)


def test_the_settings_reach_the_duct_spec_and_a_splitter_that_is_too_long_is_reported(scene, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["junction"].update(splitter_length=0.1, overlap=0.03)
    m = InletModel(scene, cfg, path, None)
    assert m.duct_spec().splitter_length == 0.1 and m.duct_spec().overlap == 0.03
    assert m.duct.bodies()["neg_trunk"].dist[0] == pytest.approx(0.1, abs=1e-9)  # trunk distances start at the junction
    cfg["ducts"]["junction"]["splitter_length"] = 50.0
    bad = InletModel(scene, cfg, path, None)
    assert bad.duct is None and "shorter than the trunk" in bad.duct_error
    assert bad.centreline is None  # no centreline without a valid junction; the model still loads


# ---- export (Phase 4) -------------------------------------------------------------------------------------------------
def test_export_blocker_names_why_the_design_cannot_be_exported(scene, example_env):
    cfg, path, _ = example_env
    ready = make(scene, path)
    assert ready.export_blocker() is None
    unplaced = make(scene, path, cfg=cfgutil.unplace(copy.deepcopy(cfg)))
    assert unplaced.export_blocker() == "place an inlet first"
    ready.readouts = lambda: [em.Readout("", "ERROR the wall is too thin", "error")]
    assert ready.export_blocker() == "the wall is too thin"
    assert ready.export_blocker([em.Readout("Wall", "0.01 m", "pass")]) is None  # readouts passed in are used instead


def test_export_inputs_hold_what_the_exporter_needs(scene, example_env):
    _, path, _ = example_env
    m = make(scene, path)
    inputs = m.export_inputs()
    assert set(inputs) == {"design", "cfg", "model_path", "skin", "skin_names", "bodies", "viewer_los"}
    assert inputs["design"] is m.duct and inputs["skin_names"] == list(m.state.skin)
    assert inputs["model_path"].name == "model.vsp3" and inputs["model_path"].is_file()
    assert set(inputs["viewer_los"]) == {"x", "n"} and all(0.0 <= v <= 1.0 for v in inputs["viewer_los"].values())
    assert make(scene, path, cfg=cfgutil.unplace(copy.deepcopy(load_config(path)[0]))).export_inputs() is None
