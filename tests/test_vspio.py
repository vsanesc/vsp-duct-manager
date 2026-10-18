# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Reading the model through the OpenVSP API: geoms, meshes, engine face, surface queries."""
import tempfile
from pathlib import Path

import numpy as np
import openvsp as vsp
import pytest

from ductmgr import vspio


@pytest.fixture()
def model(example_env):
    _, _, path = example_env
    vspio.open_model(path)
    return path


def _set_state():
    ids = vsp.FindGeoms()
    return (
        [vsp.GetSetName(i) for i in range(vsp.GetNumSets())],
        {(g, i): vsp.GetSetFlag(g, i) for g in ids for i in range(vsp.GetNumSets())},
    )


def _assert_session_unchanged(before):
    names, flags = _set_state()
    assert names == before[0]
    changed = {k: (before[1].get(k), flags.get(k)) for k in set(flags) | set(before[1]) if before[1].get(k) != flags.get(k)}
    assert not changed, f"set flags changed (geom id, set index): (before, after) = {changed}"


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        vspio.open_model(tmp_path / "nope.vsp3")


def test_list_and_find_geoms(model):
    geoms = vspio.list_geoms()
    assert {g.name: g.type_name for g in geoms} == {"Test Fuselage": "Fuselage", "Test Engine": "Pod"}
    assert vspio.find_geom("Test Engine").bbox_min[0] == pytest.approx(5.0)
    with pytest.raises(LookupError, match="no geom named"):
        vspio.find_geom("Missing")
    with pytest.raises(LookupError, match="unique"):
        vspio.find_geom(geoms[0].name, geoms=[geoms[0], geoms[0]])


def test_fuselage_mesh_matches_the_api_bounding_box(model):
    fus = vspio.find_geom("Test Fuselage")
    mesh = vspio.fuselage_mesh("Test Fuselage")
    assert mesh.n_cells > 100
    b = np.array(mesh.bounds).reshape(3, 2)
    assert b[:, 0] == pytest.approx(fus.bbox_min, abs=1e-3)
    assert b[:, 1] == pytest.approx(fus.bbox_max, abs=1e-3)
    assert vspio.check_mesh_bounds(mesh, [fus]) is None


def test_only_the_requested_geom_is_exported(model):
    engine = vspio.export_mesh([vspio.find_geom("Test Engine")])  # the pod only: not the 8 m x 2.5 m fuselage
    assert engine.bounds[0] == pytest.approx(5.0, abs=1e-3) and engine.bounds[1] == pytest.approx(6.5, abs=1e-3)
    assert engine.bounds[3] - engine.bounds[2] == pytest.approx(0.5, abs=1e-3)
    fuselage = vspio.fuselage_mesh("Test Fuselage")
    assert fuselage.bounds[3] - fuselage.bounds[2] == pytest.approx(2.5, abs=1e-3)


def test_export_leaves_no_temp_files_and_restores_the_session(model):
    tmp_root = Path(tempfile.gettempdir())
    before_dirs = {p.name for p in tmp_root.glob("ductmgr_*")}
    before = _set_state()
    vspio.fuselage_mesh("Test Fuselage")
    vspio.export_mesh(vspio.list_geoms())
    assert {p.name for p in tmp_root.glob("ductmgr_*")} == before_dirs
    _assert_session_unchanged(before)  # set names and every geom's set flags as found
    assert vspio.TEMP_SET_NAME not in before[0]


def test_export_leaves_no_extra_geoms_behind(model):
    names_before = sorted(g.name for g in vspio.list_geoms())
    vspio.fuselage_mesh("Test Fuselage")
    vspio.export_mesh(vspio.list_geoms())
    assert sorted(g.name for g in vspio.list_geoms()) == names_before  # no leftover "MeshGeom"


def test_empty_export():
    assert vspio.export_mesh([]).n_points == 0


def test_temp_files_are_removed_even_when_the_export_fails(model, monkeypatch):
    tmp_root = Path(tempfile.gettempdir())
    before_dirs = {p.name for p in tmp_root.glob("ductmgr_*")}
    before = _set_state()
    monkeypatch.setattr(vspio.pv, "read", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        vspio.fuselage_mesh("Test Fuselage")
    assert {p.name for p in tmp_root.glob("ductmgr_*")} == before_dirs
    _assert_session_unchanged(before)


def test_engine_face_from_the_bounding_box(model):
    face = vspio.engine_face("Test Engine")
    assert face.center == pytest.approx([5.0, 0.0, 0.0], abs=1e-6)  # centre of the front (min-x) end
    assert face.diameter == pytest.approx(0.5, abs=1e-6)  # y/z extent


def test_engine_diameter_warning(model):
    face = vspio.engine_face("Test Engine")  # 0.5 m
    assert vspio.engine_diameter_warning(face, 0.5, 0.02) is None
    assert vspio.engine_diameter_warning(face, 0.509, 0.02) is None  # -1.8 %: inside the tolerance
    msg = vspio.engine_diameter_warning(face, 0.45, 0.02)  # +11.1 %
    assert msg.startswith("WARN") and "0.5000" in msg and "0.4500" in msg and "+11.1" in msg
    assert "neither value was changed" in msg
    assert vspio.engine_diameter_warning(face, 0.45, 0.2) is None  # tolerance is configurable


def test_unit_mismatch_is_flagged(model):
    fus = vspio.find_geom("Test Fuselage")
    mesh = vspio.fuselage_mesh("Test Fuselage")
    assert vspio.check_mesh_bounds(mesh.scale(0.001, inplace=False), [fus]).startswith("WARN")


def test_context_geoms_exclude_by_name(model):
    assert [g.name for g in vspio.context_geoms(["Test Fuselage", "Test Engine"])] == []
    assert [g.name for g in vspio.context_geoms(["Test Fuselage"])] == ["Test Engine"]


def test_snap_to_surface_picks_the_skin_nearest_the_reference_height(model):
    mesh = vspio.fuselage_mesh("Test Fuselage")
    top, n_top = vspio.snap_to_surface(mesh, 4.0, 0.0, z_ref=+1.0)
    bottom, n_bottom = vspio.snap_to_surface(mesh, 4.0, 0.0, z_ref=-1.0)
    assert top[2] > 1.0 > -1.0 > bottom[2]
    assert top[:2] == pytest.approx([4.0, 0.0]) and bottom[:2] == pytest.approx([4.0, 0.0])
    assert n_top[2] > 0.9 and n_bottom[2] < -0.9  # outward-pointing
    assert np.linalg.norm(n_top) == pytest.approx(1.0)
    assert mesh.find_closest_point(top) is not None
    assert np.linalg.norm(mesh.points[mesh.find_closest_point(top)] - top) < 0.5  # lies on the mesh


def test_snap_misses_return_none(model):
    mesh = vspio.fuselage_mesh("Test Fuselage")
    assert vspio.snap_to_surface(mesh, 4.0, 5.0, z_ref=0.0) is None
    assert vspio.snap_to_surface(mesh, 20.0, 0.0, z_ref=0.0) is None
