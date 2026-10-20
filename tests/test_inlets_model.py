# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""1..N inlets in the explorer model: the inlets list, flow fractions, symmetry, self-symmetric editing, config."""
import copy
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import bodies
import cfgutil
from ductmgr import explore_model as em
from ductmgr import outline as ol
from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.explore_model import InletModel, legacy_notice, state_from_config

AREA = 0.06  # the stand-in scene's capture area per inlet; the total is twice that


class Clock:
    t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture()
def clock():
    return Clock()


def scene_for(body, area=AREA, face_diameter=0.5):
    return SimpleNamespace(fuselage=body, sizing=SimpleNamespace(capture_area_per_inlet=area, n_inlets=2),
                           engine_check=None, warnings=[], face_center=np.array([6.0, 0.0, 0.0]),
                           face_diameter=face_diameter, wings=[], parts=[("Test Fuselage", body)])


def inlet_entry(name, y, z, fraction=None, mirror=False, x=4.0, sweep=0.0, pitch=0.0):
    r = math.hypot(y, z)  # the origin goes onto the unit cylinder's skin, in the direction given
    y, z = (y / r, z / r) if r > 0 else (y, z)
    return {"name": name, "origin": [x, y, z], "pitch_deg": pitch, "sweep_deg": sweep, "flow_fraction": fraction,
            "mirror": mirror}


def model_for(inlets, symmetry="xz", clock=None, **ducts):
    cfg = {"ducts": {"symmetry": symmetry, "inlets": inlets, **ducts}}
    return InletModel(scene_for(bodies.cylinder(1.0)), cfg, EXAMPLE_CONFIG, None, **({"clock": clock} if clock else {}))


def verts(m, k=None):
    return m.state.inlets[m.selected if k is None else k].sections[0].shape.vertices


# ---- loading --------------------------------------------------------------------------------------------------------------------
def test_the_inlets_list_is_loaded_with_its_options(clock):
    m = model_for([inlet_entry("R", 0.3, 0.6, 0.25, True), inlet_entry("top", 0.0, 0.9), inlet_entry("L", -0.5, 0.2)],
                  "none", clock)
    assert [i.name for i in m.state.inlets] == ["R", "top", "L"] and m.state.symmetry == "none"
    assert [i.mirror for i in m.state.inlets] == [True, False, False]
    assert m.state.inlets[0].flow_fraction == 0.25 and m.state.inlets[1].flow_fraction is None
    assert all(i.placed and i.sections for i in m.state.inlets) and m.valid and m.selected == 0


def test_names_must_be_unique_and_the_symmetry_known(clock):
    with pytest.raises(ValueError, match="unique"):
        model_for([inlet_entry("a", 0.3, 0.6), inlet_entry("a", -0.3, 0.6)])
    with pytest.raises(ValueError, match="symmetry"):
        model_for([inlet_entry("a", 0.3, 0.6)], "yz")
    with pytest.raises(ValueError, match="flow_fraction"):
        model_for([inlet_entry("a", 0.3, 0.6, -1.0)])
    with pytest.raises(ValueError, match="non-empty"):
        state_from_config({"ducts": {"inlets": []}})
    with pytest.raises(ValueError, match="mirror"):
        state_from_config({"ducts": {"inlets": [{"name": "a", "mirror": "yes"}]}})


def test_the_symmetry_defaults_to_xz_and_an_inlet_without_an_origin_is_not_placed():
    state = state_from_config({"ducts": {"inlets": [{"name": "a"}]}})
    assert state.symmetry == "xz" and not state.inlets[0].placed and state.inlets[0].mirror is False


def test_flow_fractions_that_do_not_sum_to_one_are_normalised_with_a_notice(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5, 0.5), inlet_entry("b", -0.4, 0.5, 0.2)], "none", clock)
    assert "normalised" in m.notice and "\n" not in m.notice and m.fractions_normalised
    assert m.flow_fraction(0) == pytest.approx(0.5 / 0.7) and m.flow_fraction(1) == pytest.approx(0.2 / 0.7)
    assert any("do not sum to 1" in r.value for r in m.readouts())
    ok = model_for([inlet_entry("a", 0.4, 0.5, 0.6), inlet_entry("b", -0.4, 0.5, 0.4)], "none", clock)
    assert not ok.notice and not ok.fractions_normalised


def test_the_default_split_is_equal_over_all_physical_branches(clock):
    m = model_for([inlet_entry("R", 0.4, 0.5, mirror=True), inlet_entry("C", 0.0, 0.9)], "xz", clock)
    assert m.n_branches == 3 and m.flow_fraction(0) == pytest.approx(1 / 3) and m.flow_fraction(1) == pytest.approx(1 / 3)


# ---- capture area per inlet ------------------------------------------------------------------------------------------------------------
def test_the_capture_area_of_an_inlet_is_its_flow_fraction_of_the_total(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5, 0.7), inlet_entry("b", -0.4, 0.5, 0.3)], "none", clock)
    total = m.total_capture()
    assert total == pytest.approx(2 * AREA)
    assert m.capture_target(0) == pytest.approx(0.7 * total) and m.capture_target(1) == pytest.approx(0.3 * total)
    assert m.state.inlets[0].sections[0].shape is not None
    assert abs(m.evaluations[0].delta) < 0.006 and m.opening.target_area == pytest.approx(0.7 * total)
    m.select_inlet(1)
    assert abs(m.opening.delta) < 0.006 and m.opening.target_area == pytest.approx(0.3 * total)


def test_a_mirrored_pair_splits_its_share_between_the_twins(clock):
    m = model_for([inlet_entry("R", 0.4, 0.5, mirror=True)], "xz", clock)
    assert m.n_branches == 2 and m.capture_target(0) == pytest.approx(m.total_capture() / 2)  # per physical branch


def test_changing_the_flow_share_changes_the_area_and_normalises(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.4, 0.5)], "none", clock)
    before = m.capture_target(0)
    clock.advance(10.0)
    assert m.set_flow_fraction(0, 0.8) and m.fractions_normalised
    assert m.flow_fraction(0) == pytest.approx(0.8 / 1.3) and m.flow_fraction(1) == pytest.approx(0.5 / 1.3)
    assert m.capture_target(0) > before and "normalised" in m.message
    assert m.capture_target(0) + m.capture_target(1) == pytest.approx(m.total_capture())
    assert m.set_flow_fraction(0, None) and m.state.inlets[0].flow_fraction is None
    assert not m.set_flow_fraction(0, 0.0) and not m.set_flow_fraction(9, 0.3)


# ---- the inlets list: select, add, duplicate, remove, rename -----------------------------------------------------------------------------------
def test_selecting_an_inlet_is_not_a_change_and_edits_go_to_the_selected_inlet(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.4, 0.5)], "none", clock)
    dirty = m.dirty  # (the default sketches were created, so a fresh model without sections is dirty)
    assert not m.can_undo
    assert m.select_inlet(1) and m.selected == 1 and m.dirty == dirty and not m.can_undo and m.plane.origin[1] < 0
    clock.advance(10.0)
    assert m.set_param("sweep_deg", 12.0)
    assert m.state.inlets[1].plane.sweep_deg == 12.0 and m.state.inlets[0].plane.sweep_deg == 0.0
    assert m.undo() and m.selected == 1  # undo keeps the selection
    assert not m.select_inlet(5) and not m.select_inlet(-1)


def test_add_inlet_selects_an_unplaced_inlet_and_enters_place_mode(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5)], "xz", clock)
    start = m.state
    k = m.add_inlet("second")
    assert k == 1 and m.selected == 1 and m.inlet.name == "second" and not m.placed and m.mode == "placing"
    assert m.state.inlets[1].mirror is True and m.dirty
    z = float(np.array(m.scene.fuselage.ray_trace([4.0, 0.0, 5.0], [4.0, 0.0, -5.0])[0])[:, 2].max())
    assert m.handle_click([4.0, -0.6, z]) and m.placed and m.mode == "view"
    assert m.plane.origin[1] == pytest.approx(0.6)  # a mirrored inlet is kept on the right
    assert m.undo() and m.undo() and m.state == start


def test_default_inlet_names_are_unique(clock):
    m = model_for([inlet_entry("inlet 2", 0.4, 0.5)], "xz", clock)
    m.add_inlet(place=False)
    m.add_inlet(place=False)
    names = [i.name for i in m.state.inlets]
    assert len(set(names)) == 3 and m.mode == "view"


def test_an_unplaced_inlet_is_not_part_of_the_duct_and_the_others_still_build(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5, mirror=True)], "xz", clock)
    m.add_inlet("b", place=False)
    assert not m.placed and m.n_branches == 2 and m.valid  # only "a" and its twin
    m.select_inlet(0)
    assert m.ensure_built() and m.duct.n_branches == 2
    assert not any("branch_b" in n for n in m.bodies)
    assert m.flow_fraction(1) is None  # not placed: no share yet


def test_duplicate_copies_the_inlet_under_a_new_name_and_selects_it(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5, mirror=True)], "xz", clock)
    clock.advance(10.0)
    k = m.duplicate_inlet()
    assert k == 1 and m.selected == 1 and m.inlet.name == "a copy" and m.inlet.plane == m.state.inlets[0].plane
    assert m.state.inlets[1].sections == m.state.inlets[0].sections and m.n_branches == 4
    clock.advance(10.0)
    original = m.state.inlets[0].plane
    m.set_origin(4.0, 0.7, 0.7)  # move the copy
    assert m.state.inlets[0].plane == original and m.state.inlets[1].plane.origin[1] == 0.7
    assert m.duplicate_inlet(9) is None


def test_remove_inlet(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.4, 0.5), inlet_entry("c", 0.0, 0.9)], "none", clock)
    m.select_inlet(2)
    clock.advance(10.0)
    assert m.remove_inlet(2) and [i.name for i in m.state.inlets] == ["a", "b"] and m.selected == 1
    assert m.remove_inlet(0) and [i.name for i in m.state.inlets] == ["b"]
    assert not m.remove_inlet(0) and "last inlet" in m.message  # at least one inlet stays
    assert m.undo() and m.undo() and len(m.state.inlets) == 3


def test_rename_inlet_validates_the_name(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.4, 0.5)], "none", clock)
    assert m.rename_inlet(0, "  nose  ") and m.state.inlets[0].name == "nose"
    assert not m.rename_inlet(0, "") and "needs a name" in m.message
    assert not m.rename_inlet(0, "b") and "already" in m.message
    assert not m.rename_inlet(0, "nose") and m.state.inlets[0].name == "nose"  # unchanged: nothing happens
    assert m.undo() and m.state.inlets[0].name == "a"


# ---- symmetry ----------------------------------------------------------------------------------------------------------------------------------
def test_symmetry_xz_adds_twins_and_none_removes_them(clock):
    m = model_for([inlet_entry("R", 0.4, 0.5, mirror=True)], "xz", clock)
    assert m.n_branches == 2 and m.has_twin(0) and m.duct.n_branches == 2
    clock.advance(10.0)
    assert m.set_symmetry("none") and m.n_branches == 1 and not m.has_twin(0) and m.state.symmetry == "none"
    assert m.undo() and m.state.symmetry == "xz" and m.n_branches == 2
    with pytest.raises(ValueError):
        m.set_symmetry("yz")


def test_the_mirror_flag_has_no_effect_without_symmetry_or_for_a_self_symmetric_inlet(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5)], "none", clock)
    m.set_mirror(0, True)
    assert m.state.inlets[0].mirror and m.n_branches == 1 and "symmetry xz" in m.message
    c = model_for([inlet_entry("C", 0.0, 0.9, mirror=True)], "xz", clock)
    assert c.n_branches == 1 and c.inlet.self_symmetric and not c.has_twin(0)
    c.set_mirror(0, True)
    assert "own mirror image" in c.message


def test_with_symmetry_xz_the_junction_is_held_on_y_equals_0_and_without_it_is_free(clock):
    m = model_for([inlet_entry("R", 0.4, 0.5, mirror=True)], "xz", clock)
    clock.advance(10.0)
    assert m.drag_junction(3.0, 0.3, y=0.4)  # y is ignored
    assert m.centreline.junction[1] == 0.0 and m.centreline.junction_tangent[1] == 0.0
    assert not m.set_centreline("c", junction_y=0.5, junction_yaw_deg=10.0) or m.centreline.junction[1] == 0.0
    f = model_for([inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.3, 0.2)], "none", clock)
    clock.advance(10.0)
    assert f.drag_junction(3.0, 0.3, y=0.35)
    assert f.centreline.junction == pytest.approx([3.0, 0.35, 0.3])
    clock.advance(10.0)
    assert f.set_centreline("c", junction_yaw_deg=12.0)
    assert f.centreline.junction_tangent[1] == pytest.approx(math.sin(math.radians(12.0)))


def test_the_junction_row_and_area_curve_belong_to_the_selected_inlet(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5, 0.7), inlet_entry("b", -0.4, 0.5, 0.3)], "none", clock)
    sj0, sj1 = m.junction_s, m.select_inlet(1) and m.junction_s
    assert m.section_list()[1][1].startswith("junction") and m.section_list()[1][0] == pytest.approx(sj1)
    m.select_inlet(0)
    assert m.areas.share == pytest.approx(0.7) and m.areas.branch_area(1.0) == pytest.approx(0.7 * m.areas.junction_area)
    m.select_inlet(1)
    assert m.areas.share == pytest.approx(0.3)
    assert m.areas.junction_area == pytest.approx(m.areas.junction_area)


def test_per_inlet_area_points_and_handles(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.4, 0.5)], "none", clock)
    clock.advance(10.0)
    assert m.area_move_point("branch", 1, 0.5, 0.05) and m.state.inlets[0].area_branch is not None
    assert m.state.inlets[1].area_branch is None  # the other inlet keeps its default curve
    clock.advance(10.0)
    assert m.set_centreline("h", start_handle=0.45, junction_in_handle=0.3)
    assert m.state.inlets[0].start_handle == 0.45 and m.state.inlets[1].start_handle is None
    assert m.state.duct.centreline.start_handle is None  # not a shared value
    m.select_inlet(1)
    assert m.centreline.resolved.start_handle != 0.45
    m.select_inlet(0)
    clock.advance(10.0)
    assert m.reset_centreline() and m.state.inlets[0].start_handle is None and m.state.inlets[1].start_handle is None


def test_inlets_take_no_key_sections_beyond_the_junction_the_trunk_does(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5), inlet_entry("b", -0.4, 0.5)], "none", clock)
    late = (m.junction_s + 1.0) / 2
    for k in (0, 1):  # no inlet, the first included
        m.select_inlet(k)
        assert not m.add_key_section(late) and "Trunk" in m.message
        assert m.add_key_section(m.junction_s / 2)
    assert m.select_trunk() and m.add_key_section(0.5)  # the trunk's s is along the trunk
    assert [ks.s for ks in m.state.sections] == [0.0, 0.5]


def test_sections_are_sized_per_inlet(clock):
    m = model_for([inlet_entry("a", 0.4, 0.5, 0.7), inlet_entry("b", -0.4, 0.5, 0.3)], "none", clock)
    for k in (0, 1):
        m.select_inlet(k)
        s = m.junction_s / 2
        m.add_key_section(s)
        assert m.evaluations[1].area_plane == pytest.approx(m.area_at(s), rel=1e-9)
    assert m.state.inlets[0].sections[1].shape != m.state.inlets[1].sections[1].shape


# ---- self-symmetric inlets: one side is edited, the other follows ------------------------------------------------------------------------------------
def centre_model(clock):
    return model_for([inlet_entry("C", 0.0, 0.7)], "xz", clock)


def symmetric(m):
    return ol.is_mirror_symmetric(m.state.sections[0].shape)


def test_a_self_symmetric_inlet_starts_symmetric_and_says_it_is_edited_on_one_side(clock):
    m = centre_model(clock)
    assert m.symmetric_editing and symmetric(m)


def test_dragging_a_vertex_of_a_self_symmetric_inlet_moves_its_mirror_partner(clock):
    m = centre_model(clock)
    shape = m.state.sections[0].shape
    i = next(k for k, (u, v) in enumerate(shape.vertices) if u > 0.01 and not shape.snapped[k])
    partner = ol.vertex_partners(shape)[i]
    u, v = shape.vertices[i]
    clock.advance(10.0)
    assert m.drag_vertex(i, (u + 0.02, v + 0.03), snap_tolerance=None)
    new = m.state.sections[0].shape
    assert new.vertices[i] == pytest.approx((u + 0.02, v + 0.03)) and new.vertices[partner] == pytest.approx((-(u + 0.02), v + 0.03))
    assert symmetric(m)


def test_an_on_axis_vertex_stays_on_the_axis(clock):
    m = centre_model(clock)
    shape = m.state.sections[0].shape
    apex = next(k for k, (u, v) in enumerate(shape.vertices) if abs(u) < 1e-9)
    clock.advance(10.0)
    assert m.drag_vertex(apex, (0.05, shape.vertices[apex][1] + 0.02), snap_tolerance=None)
    assert m.state.sections[0].shape.vertices[apex][0] == 0.0 and symmetric(m)


def test_inserting_and_deleting_vertices_keep_a_self_symmetric_section_symmetric(clock):
    m = centre_model(clock)
    shape = m.state.sections[0].shape
    n = shape.n
    edge = next(e for e in range(n) if not shape.edge_is_snapped(e)
                and shape.vertices[e][0] > 0.01 and shape.vertices[(e + 1) % n][0] > 0.01)  # an edge on the right side
    a, b = np.array(shape.vertices[edge]), np.array(shape.vertices[(edge + 1) % n])
    clock.advance(10.0)
    assert m.insert_vertex(edge, tuple((a + b) / 2)) is not None
    assert m.state.sections[0].shape.n == n + 2 and symmetric(m)  # the vertex and its mirror image
    clock.advance(10.0)
    new = m.state.sections[0].shape
    i = next(k for k, (u, v) in enumerate(new.vertices) if u > 0.01 and not new.snapped[k])
    assert m.delete_vertex(i) and m.state.sections[0].shape.n == n and symmetric(m)


def test_curved_edges_continuity_and_handles_are_mirrored_too(clock):
    m = centre_model(clock)
    shape = m.state.sections[0].shape
    n = shape.n
    edge = next(e for e in range(n) if not shape.edge_is_snapped(e) and shape.vertices[e][0] > 0.01
                and shape.vertices[(e + 1) % n][0] > 0.01)
    clock.advance(10.0)
    assert m.set_edge_type(edge, "curved")
    sh = m.state.sections[0].shape
    partner = ol.edge_partner(sh, ol.vertex_partners(sh), edge)
    assert sh.edge_is_curved(edge) and sh.edge_is_curved(partner)
    clock.advance(10.0)
    ctrl = sh.control_points(edge)
    assert m.drag_handle(edge, 0, tuple(ctrl[1] + [0.02, 0.03]))
    sh = m.state.sections[0].shape
    a, b = sh.control_points(edge), sh.control_points(partner)
    assert b[1] == pytest.approx(a[2] * [-1.0, 1.0], abs=1e-12) and b[2] == pytest.approx(a[1] * [-1.0, 1.0], abs=1e-12)
    clock.advance(10.0)
    assert m.set_continuity(edge, "smooth")
    sh = m.state.sections[0].shape
    assert sh.continuity[edge] == "smooth" and sh.continuity[ol.vertex_partners(sh)[edge]] == "smooth" and symmetric(m)


def test_an_inlet_off_the_symmetry_plane_is_not_forced_symmetric(clock):
    m = model_for([inlet_entry("R", 0.4, 0.7, mirror=True)], "xz", clock)
    assert not m.symmetric_editing
    shape = m.state.sections[0].shape
    clock.advance(10.0)
    assert m.drag_vertex(3, (shape.vertices[3][0] + 0.02, shape.vertices[3][1]), None)
    assert len(m.state.sections[0].shape.vertices) == shape.n and not symmetric(m)  # no partner was moved


# ---- config round trip ----------------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def writable(tmp_path):
    target = tmp_path / "duct_config.yaml"
    shutil.copyfile(EXAMPLE_CONFIG, target)
    return target


def example_model(writable, clock=None):
    cfg, _ = load_config(writable)
    scene = scene_for(bodies.cylinder(1.0))
    scene.face_center = np.array([5.0, 0.0, 0.0])
    return InletModel(scene, cfg, writable, writable, **({"clock": clock} if clock else {}))


def test_several_inlets_survive_a_save_and_a_reload(writable, clock):
    m = example_model(writable, clock)
    for step in (lambda: m.set_symmetry("none"), lambda: m.set_flow_fraction(0, 0.6),
                 lambda: m.set_mirror(0, False)):
        clock.advance(10.0)
        assert step()
    m.add_inlet("second", place=False)
    m.select_inlet(1)
    z = float(np.array(m.scene.fuselage.ray_trace([3.0, 0.5, 5.0], [3.0, 0.5, -5.0])[0])[:, 2].max())
    m.begin_place()
    m.handle_click([3.0, -0.5, z])
    clock.advance(10.0)
    m.set_flow_fraction(1, 0.4)
    clock.advance(10.0)
    m.set_centreline("c", junction_yaw_deg=8.0, start_handle=0.3)
    clock.advance(10.0)
    m.drag_junction(3.5, 0.3, y=0.1)
    clock.advance(10.0)
    m.area_move_point("branch", 0, 0.3, 0.05)
    assert m.save()
    saved = yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]
    assert saved["symmetry"] == "none" and [i["name"] for i in saved["inlets"]] == ["right", "second"]
    assert saved["inlets"][0]["flow_fraction"] == pytest.approx(0.6) and saved["inlets"][1]["flow_fraction"] == pytest.approx(0.4)
    assert saved["inlets"][0]["mirror"] is False and saved["inlets"][1]["start_handle"] == 0.3
    assert saved["inlets"][1]["area_points"][0][0] == pytest.approx(0.3)
    assert saved["junction"]["point"] == [3.5, 0.1, 0.3] and saved["centreline"]["junction_yaw_deg"] == 8.0
    again = example_model(writable)
    assert not again.dirty and again.state.symmetry == "none" and [i.name for i in again.state.inlets] == ["right", "second"]
    assert again.state.inlets[1].plane.origin == pytest.approx(m.state.inlets[1].plane.origin, abs=1e-6)
    assert again.state.inlets[1].start_handle == 0.3
    assert again.state.duct.centreline.junction_y == pytest.approx(0.1)
    assert again.state.duct.centreline.junction_yaw_deg == pytest.approx(8.0)
    for a, b in zip(again.state.inlets, m.state.inlets):
        assert a.sections[0].shape.n == b.sections[0].shape.n and a.mirror == b.mirror


def test_unplaced_inlets_are_not_saved_and_revert_restores_the_list(writable, clock):
    m = example_model(writable, clock)
    clock.advance(10.0)
    m.add_inlet("ghost", place=False)
    assert m.save()
    assert [i["name"] for i in yaml.safe_load(writable.read_text(encoding="utf-8"))["ducts"]["inlets"]] == ["right"]
    clock.advance(10.0)
    m.set_flow_fraction(0, 0.9)
    assert m.revert() and len(m.state.inlets) == 1 and m.state.inlets[0].flow_fraction is None
    assert m.undo() and m.dirty  # revert is an ordinary, undoable step


def test_saving_comes_with_symmetry_and_keeps_other_comments(writable, clock):
    m = example_model(writable, clock)
    clock.advance(10.0)
    m.set_wall_thickness(0.01)
    assert m.save()
    text = writable.read_text(encoding="utf-8")
    assert "symmetry: xz" in text
    assert "# m, the positive body is the inner section offset outward" in text
    assert yaml.safe_load(text)["checks"] == load_config(EXAMPLE_CONFIG)[0]["checks"]
