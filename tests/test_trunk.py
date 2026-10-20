# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The trunk as its own entity: key sections, area curve and handles that do not depend on the inlets; every branch end is
cut from the trunk start section; configs that kept the trunk in the first inlet are migrated."""
import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import bodies
from ductmgr import duct as dm
from ductmgr import explore_model as em
from ductmgr import outline as ol
from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.explore_model import InletModel
from ductmgr.geometry import polygon_area
from ductmgr.section import points_in_polygon

JUNCTION = [3.6, 0.0, 0.3]
JUNCTION_AREA = 0.16


class Clock:
    t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture()
def clock():
    return Clock()


def scene_for(area=0.06):
    body = bodies.cylinder(1.0)
    return SimpleNamespace(fuselage=body, sizing=SimpleNamespace(capture_area_per_inlet=area, n_inlets=2),
                           engine_check=None, warnings=[], face_center=np.array([6.0, 0.0, 0.0]), face_diameter=0.5,
                           wings=[], parts=[("Test Fuselage", body)])


def inlet_entry(name, y, z, mirror=False, fraction=None, x=4.0):
    r = math.hypot(y, z)  # the origin goes onto the unit cylinder's skin, in the direction given
    return {"name": name, "origin": [x, y / r, z / r], "pitch_deg": 0.0, "sweep_deg": 0.0, "flow_fraction": fraction,
            "mirror": mirror}


def config(inlets, symmetry="xz", explicit=True, **trunk):
    ducts = {"symmetry": symmetry, "inlets": inlets}
    if explicit:  # the junction point and area given: nothing about the trunk is derived from the inlets
        ducts["junction"] = {"point": list(JUNCTION)}
        trunk = {"junction_area": JUNCTION_AREA, **trunk}
    if trunk:
        ducts["trunk"] = trunk
    return {"ducts": ducts}


def make(cfg, clock=None, path=EXAMPLE_CONFIG, target=None):
    return InletModel(scene_for(), cfg, path, target, **({"clock": clock} if clock else {}))


MIRRORED_PAIR_AND_CENTRE = [inlet_entry("R", 0.45, 0.6, True), inlet_entry("C", 0.0, 0.95), inlet_entry("S", 0.2, 0.9, True)]
FREE_THREE = [inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.4, 0.5), inlet_entry("c", 0.0, 0.95)]


# ---- the trunk geometry depends on nothing about the inlets ------------------------------------------------------------------------
def trunk_signature(m):
    """Everything of the trunk that is built: stations, start section, centreline, negative and positive bodies."""
    d = m.duct
    b = d.bodies()
    stations = np.array([[*s.centre, *s.T, *s.U, *s.V, s.area, s.s, s.dist] for s in d.trunk])
    return [stations, np.array([s.ring for s in d.trunk]), np.array(d.trunk_start),
            np.array(d.branches[0].centreline.trunk.ctrl), b["neg_trunk"].rings, b["neg_trunk"].dist,
            b["pos_trunk"].rings, b["pos_trunk"].dist, b["neg_trunk"].mesh.points, b["pos_trunk"].mesh.points]


def assert_identical(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape and np.array_equal(x, y)  # bit for bit


@pytest.mark.parametrize("explicit", [True, False])
def test_reordering_and_renaming_inlets_leave_the_trunk_bit_identical(clock, explicit):
    m = make(config(MIRRORED_PAIR_AND_CENTRE, "xz", explicit), clock)
    before = trunk_signature(m)
    clock.advance(10.0)
    assert m.move_inlet(0, 1) and [i.name for i in m.state.inlets] == ["C", "R", "S"]
    assert_identical(before, trunk_signature(m))
    clock.advance(10.0)
    assert m.move_inlet(2, -1) and m.move_inlet(0, 1) and [i.name for i in m.state.inlets] == ["S", "C", "R"]
    assert_identical(before, trunk_signature(m))
    clock.advance(10.0)
    assert m.rename_inlet(1, "zzz first") and m.rename_inlet(2, "another name")
    assert_identical(before, trunk_signature(m))


def test_removing_an_inlet_leaves_the_trunk_bit_identical_when_its_own_values_are_given(clock):
    m = make(config(MIRRORED_PAIR_AND_CENTRE, "xz", True), clock)
    before = trunk_signature(m)
    for index in (0, 0):  # the first one twice: R, then C goes first
        clock.advance(10.0)
        assert m.remove_inlet(index)
    assert [i.name for i in m.state.inlets] == ["S"] and m.n_branches == 2
    assert_identical(before, trunk_signature(m))


def test_adding_a_third_branch_layout_leaves_the_trunk_alone(clock):
    m = make(config(FREE_THREE[:2], "none", True), clock)
    before = trunk_signature(m)
    m3 = make(config(FREE_THREE, "none", True), clock)
    assert_identical(before, trunk_signature(m3))  # two or three branches: the same trunk


def test_the_trunk_does_not_move_with_an_inlet_edit_either(clock):
    m = make(config(MIRRORED_PAIR_AND_CENTRE, "xz", True), clock)
    before = trunk_signature(m)
    clock.advance(10.0)
    v = m.state.inlets[0].sections[0].shape.vertices[2]
    assert m.drag_vertex(2, (v[0] + 0.01, v[1]), None)
    m.finish_edit()
    assert_identical(before, trunk_signature(m))  # the branch changes, the trunk start and the trunk do not


# ---- editing the trunk start section reshapes every branch end ----------------------------------------------------------------------------
def end_rings(d):
    """Each branch's end section in the trunk frame (a, b), placed where the branch ends."""
    _, U, V = d.trunk_frame0
    out = []
    for bg in d.branches:
        st = bg.stations[-1]
        rel = d.place(st, st.ring) - d.trunk[0].centre
        out.append(np.c_[rel @ U, rel @ V])
    return out


def assert_tiling(d):
    rings = end_rings(d)
    _, U, V = d.trunk_frame0
    rel = d.place(d.trunk[0], d.trunk[0].ring) - d.trunk[0].centre
    start = np.c_[rel @ U, rel @ V]
    total = abs(polygon_area(start))
    assert sum(abs(polygon_area(r)) for r in rings) == pytest.approx(total, rel=1e-3)
    lo, hi = start.min(axis=0), start.max(axis=0)
    mid = lambda a, b: np.linspace(a, b, 201)[:-1] + (b - a) / 400  # noqa: E731
    g = np.stack(np.meshgrid(mid(lo[0], hi[0]), mid(lo[1], hi[1])), -1).reshape(-1, 2)
    count = np.sum([points_in_polygon(r, g) for r in rings], axis=0)
    in_s = points_in_polygon(start, g)
    assert (count <= 1).all()  # no overlap
    assert not (in_s & (count == 0)).any()  # no gap
    assert not (~in_s & (count > 0)).any()  # nothing outside the trunk start section


@pytest.mark.parametrize("inlets, symmetry, drag", [
    (MIRRORED_PAIR_AND_CENTRE, "xz", 3),  # halves... and a centre inlet: sectors; symmetric editing moves vertex 1 too
    (FREE_THREE, "none", 3),  # three free branches: sectors
    (FREE_THREE[:2], "none", 0),  # two free branches: halves
])
def test_editing_the_trunk_start_reshapes_every_branch_end_and_keeps_exact_tiling(clock, inlets, symmetry, drag):
    m = make(config(inlets, symmetry), clock)
    d0 = m.duct
    assert_tiling(d0)
    ends0 = end_rings(d0)
    assert m.select_trunk() and m.active_is_trunk_start and not m.active_is_fixed
    v = m.active_shape.vertices[drag]
    clock.advance(10.0)
    assert m.drag_vertex(drag, (v[0] * 1.2, v[1] * 1.2), None)
    assert m.finish_edit()
    d1 = m.duct
    assert d1 is not d0
    assert d1.trunk[0].area == pytest.approx(JUNCTION_AREA, rel=1e-9)  # still held at the junction area
    assert np.abs(d1.trunk[0].ring - d0.trunk[0].ring).max() > 1e-3  # the trunk start changed
    ends1 = end_rings(d1)
    for a, b in zip(ends0, ends1):  # every branch end changed with it
        assert a.shape == b.shape and np.abs(a - b).max() > 1e-4
    assert_tiling(d1)
    bodies_ = d1.bodies()
    for body in bodies_.values():
        assert body.mesh.n_open_edges == 0


def test_the_default_trunk_start_is_a_superellipse_held_at_the_junction_area(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    [start] = m.state.trunk
    assert start.s == 0.0 and start.shape.n == 4 and all(e == "curved" for e in start.shape.edges)
    assert ol.outline_area(start.shape) == pytest.approx(JUNCTION_AREA, rel=1e-6)
    assert m.duct.trunk_start.shape == (m.duct.n, 2) and polygon_area(m.duct.trunk_start) == pytest.approx(JUNCTION_AREA, rel=1e-9)
    assert m.duct.areas.junction_area == JUNCTION_AREA


def test_trunk_key_sections_loft_between_the_start_the_inner_sections_and_the_face(clock):
    cfg = config(FREE_THREE[:2], "none")
    m = make(cfg, clock)
    assert m.select_trunk() and m.add_key_section(0.5) and m.active == 1
    [start, mid] = m.state.trunk
    assert mid.s == 0.5 and m.evaluations[1].area_plane == pytest.approx(float(m.areas.trunk_area(0.5)), rel=1e-9)
    clock.advance(10.0)
    shape = ol.circle_shape(0.2)  # make the middle section a circle: the stations near it are close to circular
    m._commit(m._normalised(m.state.with_selected(sections=(start, em.KeySection(0.5, shape)))))
    ring = min(m.duct.trunk, key=lambda s: abs(s.s - 0.5)).ring
    radii = np.linalg.norm(ring, axis=1)
    assert radii.max() / radii.min() < 1.02
    assert m.remove_key_section(1) and len(m.state.trunk) == 1 and not m.remove_key_section(0)


def test_a_symmetric_layout_keeps_the_trunk_start_mirror_symmetric_while_editing(clock):
    m = make(config(MIRRORED_PAIR_AND_CENTRE, "xz"), clock)
    assert m.select_trunk() and m.symmetric_editing
    v = m.active_shape.vertices[3]
    clock.advance(10.0)
    assert m.drag_vertex(3, (v[0] * 1.1, v[1] + 0.01), None)
    assert ol.is_mirror_symmetric(m.active_shape)
    free = make(config(FREE_THREE, "none"), clock)
    assert free.select_trunk() and not free.symmetric_editing


# ---- selecting the trunk -----------------------------------------------------------------------------------------------------------------------------
def test_selecting_the_trunk_shows_its_sections_and_does_not_touch_the_history(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    state, count = m.state, len(m._past)
    assert not m.trunk_selected and m.select_trunk() and m.trunk_selected
    assert m.state == state and len(m._past) == count  # a selection is not an edit
    rows = m.section_list()
    assert [r[1:] for r in rows] == [("trunk start", True), ("engine face (fixed)", False)]
    assert not m.junction_row and m.active_s == 0.0
    assert m.select_section(1) and m.active_is_face and m.active_is_fixed and m.active_s == 1.0
    assert m.active_shape.n >= 3 and m.active_opening.area_plane == pytest.approx(m.face_area, rel=2e-3)
    assert not m.drag_vertex(0, (0.0, 0.0), None)  # the face is fixed
    assert m.select_inlet(1) and not m.trunk_selected and m.selected == 1 and m.active == 0
    assert [r[1] for r in m.section_list()][:1] == ["inlet"]


def test_the_trunk_selection_is_not_part_of_the_state_equality_and_survives_undo(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    m.select_trunk()
    clock.advance(10.0)
    assert m.set_wall_thickness(0.012)
    assert m.trunk_selected
    assert m.undo() and m.trunk_selected and m.state.duct.wall_thickness != 0.012
    assert m.redo() and m.trunk_selected
    m.select_inlet(0)
    assert replace(m.state, trunk_selected=True) == m.state  # equality ignores the selection


def test_the_trunk_edits_are_undoable_with_the_shared_history(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    m.select_trunk()
    before = m.state
    v = m.active_shape.vertices[0]
    clock.advance(10.0)
    assert m.drag_vertex(0, (v[0], v[1] * 1.3), None)
    m.finish_edit()
    assert m.state != before and m.dirty
    assert m.undo() and m.state == before


def test_the_trunk_area_curve_is_edited_without_touching_the_inlets(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    m.select_trunk()
    clock.advance(10.0)
    assert m.area_move_point("trunk", 1, 0.5, 0.18)
    assert m.state.duct.area.trunk is not None and m.state.duct.area.trunk[1] == pytest.approx((0.5, 0.18))
    assert all(i.area_branch is None for i in m.state.inlets)  # the branch curves stay on their inlets
    clock.advance(10.0)
    assert m.area_set_junction(0.15) and m.state.duct.area.junction_area == 0.15
    assert m.state.trunk[0].shape != make(config(FREE_THREE[:2], "none"), clock).state.trunk[0].shape  # rescaled to it
    assert ol.outline_area(m.state.trunk[0].shape) == pytest.approx(0.15, rel=1e-6)
    clock.advance(10.0)
    assert m.reset_area() and m.state.duct.area.trunk is None


def test_the_trunk_handles_are_the_trunks_and_the_inlet_handles_are_not(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    m.select_trunk()
    clock.advance(10.0)
    assert m.set_centreline("h", junction_out_handle=0.9, end_handle=0.7, start_handle=0.4)
    assert m.state.duct.centreline.junction_out_handle == 0.9 and m.state.duct.centreline.end_handle == 0.7
    assert all(i.start_handle is None for i in m.state.inlets)  # an inlet handle is not set while the trunk is selected


# ---- the config: ducts.trunk ------------------------------------------------------------------------------------------------------------------------
def write_cfg(tmp_path, cfg):
    path = tmp_path / "duct_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_the_trunk_survives_a_save_and_a_reload(clock, tmp_path):
    path = write_cfg(tmp_path, config(FREE_THREE[:2], "none"))
    m = make(load_config(path)[0], clock, path, path)
    m.select_trunk()
    clock.advance(10.0)
    assert m.add_key_section(0.4)
    clock.advance(10.0)
    v = m.active_shape.vertices[1]
    assert m.drag_vertex(1, (v[0] * 1.1, v[1]), None)
    m.finish_edit()
    clock.advance(10.0)
    assert m.set_centreline("h", junction_out_handle=0.8, end_handle=0.6) and m.area_move_point("trunk", 0, 0.3, 0.17)
    assert m.save()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]
    t = saved["trunk"]
    assert [k["s"] for k in t["key_sections"]] == [0.0, 0.4] and t["junction_area"] == JUNCTION_AREA
    assert t["junction_out_handle"] == 0.8 and t["end_handle"] == 0.6 and t["area_points"][0] == [0.3, 0.17]
    assert "end_handle" not in saved.get("centreline", {}) and "area_points" not in saved
    again = make(load_config(path)[0], clock, path)
    assert not again.dirty and not again.notice
    assert [ks.s for ks in again.state.trunk] == [0.0, 0.4]
    for a, b in zip(again.state.trunk, m.state.trunk):
        assert np.array(a.shape.vertices) == pytest.approx(np.array(b.shape.vertices), abs=2e-6)
    assert again.state.duct.centreline.end_handle == 0.6
    assert again.state.duct.area.junction_area == m.state.duct.area.junction_area
    assert np.array(again.state.duct.area.trunk) == pytest.approx(np.array(m.state.duct.area.trunk), abs=2e-6)  # 6 decimals


def test_the_old_locations_of_the_trunk_values_are_still_read_and_moved_on_save(clock, tmp_path):
    cfg = config(FREE_THREE[:2], "none", explicit=False)
    cfg["ducts"]["area_points"] = {"junction_area": JUNCTION_AREA, "trunk": [[0.5, 0.18]]}
    cfg["ducts"]["centreline"] = {"junction_out_handle": 0.8, "end_handle": 0.6, "junction_pitch_deg": 5.0}
    path = write_cfg(tmp_path, cfg)
    m = make(load_config(path)[0], clock, path, path)
    assert m.state.duct.area.junction_area == JUNCTION_AREA and m.state.duct.area.trunk == ((0.5, 0.18),)
    assert m.state.duct.centreline.junction_out_handle == 0.8 and m.state.duct.centreline.end_handle == 0.6
    assert m.save()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]
    assert "area_points" not in saved and saved["centreline"].keys() >= {"junction_pitch_deg"}
    assert "junction_out_handle" not in saved["centreline"] and "end_handle" not in saved["centreline"]
    assert saved["trunk"]["junction_area"] == JUNCTION_AREA and saved["trunk"]["end_handle"] == 0.6


def test_a_bad_trunk_block_is_refused():
    with pytest.raises(ValueError, match="ducts.trunk"):
        em.state_from_config({"ducts": {"inlets": [{"name": "a"}], "trunk": [1]}})
    with pytest.raises(ValueError, match="junction_area"):
        em.duct_params_from_config({"ducts": {"trunk": {"junction_area": -1.0}}})
    with pytest.raises(ValueError, match="key_sections"):
        em.state_from_config({"ducts": {"inlets": [{"name": "a"}], "trunk": {"key_sections": [{"s": 1.5}]}}})


# ---- migration of configs that kept the trunk on the first inlet ---------------------------------------------------------------------------
def legacy_cfg(symmetry="xz", late_sections=1):
    """A config in the previous format: no ducts.trunk, trunk-side key sections on the first inlet."""
    base = make(config(MIRRORED_PAIR_AND_CENTRE if symmetry == "xz" else FREE_THREE, symmetry, explicit=False))
    entries = base.inlets_definition()
    cfg = {"ducts": {"symmetry": symmetry, "inlets": entries}}
    sj = base.junction_s
    shape = ol.circle_shape(0.2)
    keys = entries[0]["key_sections"]
    keys.append({"s": round(sj / 2, 4), **ol.shape_to_config(shape)})  # a key section of the branch: stays
    for k in range(late_sections):  # trunk-side ones
        keys.append({"s": round(sj + (1 - sj) * (k + 1) / (late_sections + 1), 4), **ol.shape_to_config(shape)})
    return cfg, sj


@pytest.mark.parametrize("symmetry", ["xz", "none"])
def test_a_config_with_trunk_key_sections_on_the_first_inlet_is_migrated(clock, symmetry, tmp_path):
    cfg, sj = legacy_cfg(symmetry, late_sections=2)
    keys_before = [k["s"] for k in cfg["ducts"]["inlets"][0]["key_sections"]]
    m = make(cfg, clock)
    assert "ducts.trunk" in m.notice and "migrated" in m.notice and "\n" not in m.notice
    assert "2 trunk-side key section(s)" in m.notice
    first = m.state.inlets[0]
    assert [ks.s for ks in first.sections] == keys_before[:2]  # the inlet and its branch section stay
    assert all(ks.s < m.junction_s - 0.005 for ks in first.sections[1:])
    trunk = m.state.trunk
    assert [ks.s for ks in trunk][0] == 0.0 and len(trunk) == 3
    expect = [(s - sj) / (1 - sj) for s in keys_before[2:]]
    assert [ks.s for ks in trunk[1:]] == pytest.approx(expect, abs=0.03)  # global s -> s along the trunk
    start = trunk[0].shape  # the derived start section, now an editable section of its own
    assert start.n == m.duct.n and not any(start.snapped)
    assert ol.outline_area(start) == pytest.approx(m.duct.areas.junction_area, rel=1e-6)
    if symmetry == "xz":
        assert ol.is_mirror_symmetric(start)
    assert not m.dirty and m.valid
    assert_tiling(m.duct)
    for body in m.duct.bodies().values():
        assert body.mesh.n_open_edges == 0
    # saving writes the new format; the reload has nothing to migrate
    path = write_cfg(tmp_path, cfg)
    saver = make(load_config(path)[0], clock, path, path)
    assert saver.save()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]
    assert len(saved["trunk"]["key_sections"]) == 3
    assert [k["s"] for k in saved["inlets"][0]["key_sections"]] == keys_before[:2]
    again = make(load_config(path)[0], clock, path)
    assert not again.notice and not again.dirty and len(again.state.trunk) == 3
    assert again.duct.trunk_start == pytest.approx(saver.duct.trunk_start, abs=1e-4)


def test_migration_keeps_the_trunk_start_the_junction_section_the_old_code_derived(clock):
    """The migrated start section is the blend of the first inlet's sections at the junction (symmetrised)."""
    cfg, sj = legacy_cfg("xz", late_sections=0)
    m = make(cfg, clock)
    d = m.duct
    first = d.branches[0]
    legacy = dm.symmetric_ring(d.ring_at(first.s_junction, first.areas.junction_area, first.rings), d.n)
    angles = np.linspace(0.0, 2.0 * math.pi, 90, endpoint=False)
    got, want = dm.ray_radii(d.trunk_start, angles), dm.ray_radii(legacy, angles)
    assert np.abs(got - want).max() / want.max() < 0.02  # the same shape (up to the small shift of the junction s)


def test_trunk_key_sections_beyond_the_junction_on_other_inlets_are_dropped_with_the_notice(clock):
    cfg, sj = legacy_cfg("none", late_sections=1)
    second = cfg["ducts"]["inlets"][1]["key_sections"]
    second.append({"s": round(sj + (1 - sj) / 2, 4), **ol.shape_to_config(ol.circle_shape(0.2))})
    m = make(cfg, clock)
    assert "dropped" in m.notice
    assert [ks.s for ks in m.state.inlets[1].sections] == [0.0]
    assert len(m.state.trunk) == 2


def test_a_config_with_a_trunk_is_not_migrated(clock):
    cfg, _ = legacy_cfg("xz", late_sections=1)
    m0 = make(cfg, clock)
    cfg2 = copy.deepcopy(cfg)
    cfg2["ducts"]["inlets"][0]["key_sections"] = m0.inlets_definition()[0]["key_sections"]
    cfg2["ducts"]["trunk"] = m0.trunk_definition()
    m = make(cfg2, clock)
    assert "migrated" not in m.notice
    assert [ks.s for ks in m.state.trunk] == [ks.s for ks in m0.state.trunk]


def test_a_design_without_inlet_sections_gets_the_default_trunk_without_a_notice(clock):
    m = make(config(FREE_THREE[:2], "none", explicit=False), clock)
    assert not m.notice and len(m.state.trunk) == 1 and m.state.trunk[0].shape.n == 4


def test_placing_or_moving_an_inlet_leaves_the_trunk_row(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    m.select_trunk()
    assert m.begin_place() and not m.trunk_selected and m.selected == 0
    m.cancel_place()
    m.select_trunk()
    clock.advance(10.0)
    assert m.set_param("pitch_deg", 5.0) and not m.trunk_selected and m.state.inlets[0].plane.pitch_deg == 5.0


# ---- reset to default and simplify the trunk start ---------------------------------------------------------------------------------------
def test_reset_to_default_restores_a_superellipse_held_at_the_junction_area_and_is_undoable(clock):
    m = make(config(MIRRORED_PAIR_AND_CENTRE, "xz"), clock)
    default = m.state.trunk[0].shape
    m.select_trunk()
    clock.advance(10.0)
    v = m.active_shape.vertices[3]
    assert m.drag_vertex(3, (v[0] * 1.3, v[1]), None)
    m.finish_edit()
    edited = m.state
    assert m.state.trunk[0].shape != default
    clock.advance(10.0)
    assert m.reset_trunk_start(4.0)
    shape = m.state.trunk[0].shape
    assert shape.n == 4 and shape != default  # exponent 4: squarer than the default 2.5
    assert ol.outline_area(shape) == pytest.approx(JUNCTION_AREA, rel=1e-6) and ol.is_mirror_symmetric(shape)
    assert "exponent 4" in m.message and m.dirty
    assert m.undo() and m.state == edited
    clock.advance(10.0)
    assert m.reset_trunk_start() and m.state.trunk[0].shape == default  # the default exponent
    assert_tiling(m.duct)


def test_reset_to_default_keeps_the_other_trunk_sections_and_refuses_a_bad_exponent(clock):
    m = make(config(FREE_THREE[:2], "none"), clock)
    m.select_trunk()
    assert m.add_key_section(0.5)
    others = m.state.trunk[1:]
    for bad in (0.5, 9.0):
        assert not m.reset_trunk_start(bad) and "exponent" in m.message
    assert m.reset_trunk_start(3.0) and m.state.trunk[1:] == others and len(m.state.trunk) == 2


def migrated_model(clock, symmetry="xz"):
    cfg, _ = legacy_cfg(symmetry, late_sections=0)
    return make(cfg, clock)


@pytest.mark.parametrize("symmetry, n", [("xz", 8), ("xz", 9), ("none", 6)])
def test_simplify_reduces_the_migrated_polygon_keeping_area_symmetry_and_tiling(clock, symmetry, n):
    m = migrated_model(clock, symmetry)
    before = m.state
    assert m.state.trunk[0].shape.n == m.duct.n  # the migrated start is a polygon of the section samples
    m.select_trunk()
    clock.advance(10.0)
    assert m.simplify_trunk_start(n)
    shape = m.state.trunk[0].shape
    assert shape.n == n and any(e == "curved" for e in shape.edges)
    assert ol.outline_area(shape) == pytest.approx(m.areas.junction_area, rel=1e-6)
    if symmetry == "xz":
        assert ol.is_mirror_symmetric(shape)
    assert "deviation max" in m.simplify_note and f"{before.trunk[0].shape.n} -> {n} vertices" in m.simplify_note
    assert "deviation" in m.message and any(r.label == "Simplify" for r in m.readouts())
    assert_tiling(m.duct)
    for body in m.duct.bodies().values():
        assert body.mesh.n_open_edges == 0
    assert m.undo() and m.state == before and not m.simplify_note  # one undo step; the note goes with it
    assert m.redo() and m.state.trunk[0].shape.n == n


def test_the_simplify_note_goes_away_with_the_next_edit(clock):
    m = migrated_model(clock)
    m.select_trunk()
    assert m.simplify_trunk_start(8) and m.simplify_note
    clock.advance(10.0)
    assert m.set_wall_thickness(0.012) and not m.simplify_note


def test_simplify_refuses_what_it_cannot_do_and_changes_nothing(clock):
    m = migrated_model(clock)
    state = m.state
    for bad in (2, 64, 500):
        assert not m.simplify_trunk_start(bad) and m.message.startswith("Not simplified") and m.state == state
    default = make(config(FREE_THREE[:2], "none"), clock)
    assert not default.simplify_trunk_start(4) and "fewer" in default.message  # the default has 4 vertices already
