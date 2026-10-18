# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Wing planform edge sweeps read from the VSP parameters, checked against the exported surface."""
import math

import numpy as np
import openvsp as vsp
import pytest

from ductmgr import vspio
from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.viz import load_scene
from make_example_model import build_example_model

# ---- the formula, written out independently ------------------------------------------------------------------
def le_te(sweep, loc, cr, ct, span, dih):
    tan_le = math.tan(math.radians(sweep)) + loc * (cr - ct) / span
    tan_te = math.tan(math.radians(sweep)) + (loc * (cr - ct) + (ct - cr)) / span
    c = math.cos(math.radians(dih))
    return math.degrees(math.atan(tan_le / c)), math.degrees(math.atan(tan_te / c))


def test_a_section_swept_at_the_leading_edge_gives_that_sweep():
    le, te = vspio.section_edge_sweeps(30.0, 0.0, 4.0, 1.0, 9.0, 0.0)
    assert le == pytest.approx(30.0, abs=1e-12)
    assert te == pytest.approx(math.degrees(math.atan(math.tan(math.radians(30)) - 3.0 / 9.0)), abs=1e-12)


@pytest.mark.parametrize("sweep, loc, cr, ct, span, dih", [
    (20.0, 0.25, 4.0, 1.0, 9.0, 0.0),
    (20.0, 0.5, 4.0, 1.0, 9.0, 0.0),
    (25.0, 0.25, 4.0, 2.0, 9.0, 15.0),
    (-10.0, 0.0, 3.0, 2.0, 5.0, 0.0),
    (35.0, 1.0, 2.0, 2.0, 3.0, 8.0),
])
def test_edge_sweeps_match_the_independent_formula(sweep, loc, cr, ct, span, dih):
    assert vspio.section_edge_sweeps(sweep, loc, cr, ct, span, dih) == pytest.approx(
        le_te(sweep, loc, cr, ct, span, dih), abs=1e-12)


def test_a_degenerate_span_is_rejected():
    with pytest.raises(ValueError, match="span"):
        vspio.section_edge_sweeps(30.0, 0.0, 4.0, 1.0, 0.0, 0.0)


def test_the_trailing_edge_of_an_untapered_wing_has_the_same_sweep():
    le, te = vspio.section_edge_sweeps(25.0, 0.3, 2.0, 2.0, 6.0, 0.0)
    assert le == pytest.approx(25.0) and te == pytest.approx(25.0)


# ---- wings built through the API --------------------------------------------------------------------------------
def build_wing(sections, name="Wing", y_rel=0.0, rotation=None):
    """A VSP wing with the given per-section parameters; returns its geom id."""
    vsp.ClearVSPModel()
    wing = vsp.AddGeom("WING")
    vsp.SetGeomName(wing, name)
    vsp.SetParmVal(wing, "Sym_Planar_Flag", "Sym", 0)
    surf = vsp.GetXSecSurf(wing, 0)
    for _ in range(len(sections) - 1):
        vsp.InsertXSec(wing, 1, vsp.XS_FOUR_SERIES)
    vsp.Update()
    for i, sec in enumerate(sections, start=1):
        xs = vsp.GetXSec(surf, i)
        for key, value in sec.items():
            vsp.SetParmVal(vsp.GetXSecParm(xs, key), value)
            vsp.Update()
    vsp.SetParmVal(wing, "Y_Rel_Location", "XForm", y_rel)
    for axis, value in (rotation or {}).items():
        vsp.SetParmVal(wing, axis, "XForm", value)
    vsp.Update()
    return wing


def measured_edges(geom_name, y):
    """(LE, TE) x at spanwise station y, from the exported surface: the independent reference."""
    mesh = vspio.export_mesh([vspio.find_geom(geom_name)])
    pts = np.asarray(mesh.points)
    slab = pts[np.abs(pts[:, 1] - y) < 0.015]
    return slab[:, 0].min(), slab[:, 0].max()


def measured_sweeps(geom_name, y0, y1):
    (le0, te0), (le1, te1) = measured_edges(geom_name, y0), measured_edges(geom_name, y1)
    dy = y1 - y0
    return math.degrees(math.atan2(le1 - le0, dy)), math.degrees(math.atan2(te1 - te0, dy))


CASES = {
    "default (LE swept 30)": [dict(Sweep=30)],
    "swept at quarter chord": [dict(Sweep=20, Sweep_Location=0.25)],
    "swept at mid chord, tapered": [dict(Sweep=20, Sweep_Location=0.5)],
    "dihedral and taper": [dict(Sweep=25, Sweep_Location=0.25, Dihedral=15, Tip_Chord=2)],
    "forward swept": [dict(Sweep=-10)],
    "two sections": [dict(Sweep=35, Sweep_Location=0.0), dict(Sweep=15, Sweep_Location=0.3, Dihedral=8)],
}


@pytest.mark.parametrize("label", list(CASES))
def test_edge_sweeps_agree_with_the_surface_of_the_wing(label):
    build_wing(CASES[label])
    (wing,) = vspio.read_wings()
    assert wing.name == "Wing" and len(wing.sections) == len(CASES[label])
    for sec in wing.sections:
        ys = (sec.y0 + 0.2 * (sec.y1 - sec.y0), sec.y0 + 0.8 * (sec.y1 - sec.y0))
        le, te = measured_sweeps("Wing", *ys)
        assert sec.le_deg == pytest.approx(le, abs=0.02)  # the leading edge, within 0.02 deg of the surface
        assert sec.te_deg == pytest.approx(te, abs=0.02)  # and the trailing edge


def test_a_wing_with_a_known_leading_edge_sweep_gives_that_sweep():
    build_wing([dict(Sweep=30)])  # sweep reference at the leading edge (Sweep_Location 0)
    (wing,) = vspio.read_wings()
    assert vspio.edge_sweep_at(wing, 1.0, "LE").angle_deg == pytest.approx(30.0, abs=1e-9)


def test_the_sweeps_come_from_the_parameters_not_the_mesh(monkeypatch):
    build_wing(CASES["two sections"])
    monkeypatch.setattr(vspio, "export_mesh", lambda *_: pytest.fail("the mesh must not be used"))
    monkeypatch.setattr(vspio.pv, "read", lambda *_: pytest.fail("the mesh must not be used"))
    (wing,) = vspio.read_wings()
    assert wing.sections[0].le_deg == pytest.approx(35.0, abs=1e-9)


def test_the_section_spanning_the_inlet_y_is_used():
    build_wing(CASES["two sections"])
    (wing,) = vspio.read_wings()
    first, second = wing.sections
    assert first.y0 == pytest.approx(0.0) and first.y1 == pytest.approx(second.y0) and second.y1 > second.y0
    a = vspio.edge_sweep_at(wing, first.y1 * 0.5, "LE")
    b = vspio.edge_sweep_at(wing, second.y0 + 0.5 * (second.y1 - second.y0), "LE")
    assert (a.section_index, a.n_sections, a.angle_deg) == (0, 2, pytest.approx(first.le_deg))
    assert (b.section_index, b.angle_deg) == (1, pytest.approx(second.le_deg))
    assert a.y0 == first.y0 and a.y1 == first.y1 and not a.outside and not b.outside
    assert vspio.edge_sweep_at(wing, -first.y1 * 0.5, "LE").angle_deg == a.angle_deg  # the left side: |y|
    te = vspio.edge_sweep_at(wing, first.y1 * 0.5, "TE")
    assert te.angle_deg == pytest.approx(first.te_deg) and te.angle_deg != a.angle_deg


def test_a_y_outside_the_wing_uses_the_nearest_section_and_says_so():
    build_wing(CASES["two sections"])
    (wing,) = vspio.read_wings()
    far = vspio.edge_sweep_at(wing, wing.sections[-1].y1 + 3.0, "LE")
    assert far.outside and far.section_index == 1
    near = vspio.edge_sweep_at(wing, 0.0, "LE")
    assert not near.outside and near.section_index == 0  # the root plane itself belongs to the first section


def test_the_wing_root_offset_shifts_the_spanwise_ranges():
    build_wing([dict(Sweep=30)], y_rel=0.5)
    (wing,) = vspio.read_wings()
    assert wing.sections[0].y0 == pytest.approx(0.5)
    assert not vspio.edge_sweep_at(wing, 0.7, "LE").outside
    assert vspio.edge_sweep_at(wing, 0.2, "LE").outside


def test_a_rotated_wing_is_reported():
    build_wing([dict(Sweep=30)])
    assert not vspio.read_wings()[0].rotated
    build_wing([dict(Sweep=30)], rotation={"Y_Rel_Rotation": 4.0})
    assert vspio.read_wings()[0].rotated


def test_only_wing_type_geoms_are_listed_and_names_are_kept():
    vsp.ClearVSPModel()
    for kind, name in (("FUSELAGE", "Body"), ("WING", "Main"), ("WING", "Fins"), ("WING", "Fins"), ("POD", "Pod")):
        vsp.SetGeomName(vsp.AddGeom(kind), name)
    vsp.Update()
    wings = vspio.read_wings()
    assert [w.name for w in wings] == ["Main", "Fins", "Fins"]
    assert len({w.geom_id for w in wings}) == 3
    with pytest.raises(ValueError, match="edge"):
        vspio.edge_sweep_at(wings[0], 0.1, "mid")


def test_a_model_without_wings_has_none(example_env):
    _, _, model = example_env
    vspio.open_model(model)
    assert vspio.read_wings() == []


def test_load_scene_reads_the_wings_of_the_model(tmp_path):
    cfg, _ = load_config(EXAMPLE_CONFIG)
    path = tmp_path / "with_wing.vsp3"
    build_example_model(path, cfg)
    vspio.open_model(path)
    wing = vsp.AddGeom("WING")
    vsp.SetGeomName(wing, "Main wing")
    vsp.SetParmVal(vsp.GetXSecParm(vsp.GetXSec(vsp.GetXSecSurf(wing, 0), 1), "Sweep"), 33.0)
    vsp.Update()
    vsp.WriteVSPFile(str(path), vsp.SET_ALL)
    cfg["model"]["vsp3_path"] = str(path)
    scene = load_scene(cfg, EXAMPLE_CONFIG)
    assert [w.name for w in scene.wings] == ["Main wing"]
    assert vspio.edge_sweep_at(scene.wings[0], 0.6, "LE").angle_deg == pytest.approx(33.0, abs=1e-9)
