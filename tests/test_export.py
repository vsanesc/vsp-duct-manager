# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Phase 4 against the real OpenVSP: what an export writes (geoms, symmetry flags, XSecs), that it never touches anything
else, and the automatic verification of the saved file."""
import hashlib

import numpy as np
import openvsp as vsp
import pytest

import exportenv as env
from ductmgr import cedit, exportrun, vspio
from ductmgr import sectionfit as sf

PREFIX = "DUCT_"


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(model, out, verify=False, model_path=None):
    model.ensure_built()
    assert model.export_blocker() is None
    inputs = model.export_inputs()
    if model_path is not None:
        inputs["model_path"] = model_path
    return exportrun.export_design(**inputs, out_path=out, verify=verify)


def saved(path):
    """Open an exported file: {name: geom id} of the DUCT_ geoms."""
    vspio.open_model(path)
    return {vsp.GetGeomName(g): g for g in vspio.geoms_with_prefix(PREFIX)}


def flag(gid, name, group):
    return vsp.GetParmVal(gid, name, group)


# ---- the environments ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def s_env(tmp_path_factory):
    """The example model: a dorsal inlet and its mirrored twin, an S-duct to the engine."""
    return env.build_env(tmp_path_factory.mktemp("s_env"))


@pytest.fixture(scope="module")
def straight_env(tmp_path_factory):
    """The engine raised to the height of the dorsal skin: a nearly straight duct from an inlet on y = 0."""
    return env.build_env(tmp_path_factory.mktemp("straight_env"), engine_z=1.2)


@pytest.fixture(scope="module")
def s_export(s_env, tmp_path_factory):
    cfg, path, model_path = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "right", 2.0, 0.5, True)], junction=[4.0, 0.0, 0.2])
    out = tmp_path_factory.mktemp("s_out") / "out.vsp3"
    return run(m, out, verify=True), out, model_path


@pytest.fixture(scope="module")
def straight_export(straight_env, tmp_path_factory):
    cfg, path, model_path = straight_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "top", 2.0, 0.0)], junction=[3.8, 0.0, 1.3])
    out = tmp_path_factory.mktemp("straight_out") / "out.vsp3"
    return run(m, out, verify=True), out, model_path


# ---- the geoms ---------------------------------------------------------------------------------------------------------------------
def check_geom_structure(gid, negative: bool, mirrored: bool, plan):
    assert vsp.GetGeomTypeName(gid) == "Stack"
    assert flag(gid, "Negative_Volume_Flag", "Negative_Volume_Props") == (1.0 if negative else 0.0)
    assert flag(gid, "Sym_Planar_Flag", "Sym") == (float(vsp.SYM_XZ) if mirrored else 0.0)
    surf = vsp.GetXSecSurf(gid, 0)
    n = vsp.GetNumXSec(surf)
    assert n == len(plan.sections) >= 5  # at least 3 curves and the two cap points
    shapes = [vsp.GetXSecShape(vsp.GetXSec(surf, i)) for i in range(n)]
    assert shapes[0] == shapes[-1] == vsp.XS_POINT  # the flat end caps
    assert all(s == vsp.XS_EDIT_CURVE for s in shapes[1:-1])  # the user can move the points in the GUI


def test_one_straight_inlet_exports_a_plain_pair_and_a_trunk_pair(straight_export):
    result, out, _ = straight_export
    geoms = saved(out)
    assert set(geoms) == {"DUCT_BRANCH_top_POS", "DUCT_BRANCH_top_NEG", "DUCT_TRUNK_POS", "DUCT_TRUNK_NEG"}
    for name, gid in geoms.items():
        check_geom_structure(gid, name.endswith("_NEG"), False, result.plan.geom(name))  # no symmetry flag anywhere


def test_a_mirrored_inlet_is_one_pair_with_planar_xz_symmetry_not_two(s_export):
    result, out, _ = s_export
    geoms = saved(out)
    assert set(geoms) == {"DUCT_BRANCH_right_POS", "DUCT_BRANCH_right_NEG", "DUCT_TRUNK_POS", "DUCT_TRUNK_NEG"}
    for name, gid in geoms.items():
        check_geom_structure(gid, name.endswith("_NEG"), "BRANCH" in name, result.plan.geom(name))
    # the mirrored twin really is there: the exported mesh has both sides
    mesh = vspio.export_mesh([vspio.find_geom("DUCT_BRANCH_right_NEG")])
    assert mesh.bounds[2] < -0.3 and mesh.bounds[3] > 0.3


def test_three_inlets_with_xz_symmetry_a_mirrored_one_and_one_on_the_axis(s_env, tmp_path):
    cfg, path, _ = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "a", 2.0, 0.6, True),
                                       lambda s: env.inlet(s, "b", 3.0, 0.0)], junction=[4.2, 0.0, 0.3])
    assert m.duct.n_branches == 3
    result = run(m, tmp_path / "o.vsp3")
    geoms = saved(tmp_path / "o.vsp3")
    assert set(geoms) == {"DUCT_BRANCH_a_POS", "DUCT_BRANCH_a_NEG", "DUCT_BRANCH_b_POS", "DUCT_BRANCH_b_NEG",
                          "DUCT_TRUNK_POS", "DUCT_TRUNK_NEG"}
    for name, gid in geoms.items():
        check_geom_structure(gid, name.endswith("_NEG"), "_a_" in name, result.plan.geom(name))


def test_three_independent_inlets_without_symmetry_get_no_symmetry_flags(s_env, tmp_path):
    cfg, path, _ = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "one", 2.0, 0.6), lambda s: env.inlet(s, "two", 2.0, -0.6),
                                       lambda s: env.inlet(s, "three", 3.0, 0.0)], symmetry="none",
                           junction=[4.2, 0.0, 0.3])
    result = run(m, tmp_path / "o.vsp3")
    geoms = saved(tmp_path / "o.vsp3")
    assert len(geoms) == 8  # three pairs and the trunk pair
    for name, gid in geoms.items():
        check_geom_structure(gid, name.endswith("_NEG"), False, result.plan.geom(name))


def test_the_xsec_count_is_adaptive_and_reported_per_geom(s_export):
    result, _, _ = s_export
    counts = {g.name: len(g.real_sections) for g in result.plan.geoms}
    assert all(3 <= c < 40 for c in counts.values())
    assert counts["DUCT_BRANCH_right_NEG"] > counts["DUCT_TRUNK_NEG"]  # the branch bends and changes shape more
    for g in result.plan.geoms:
        assert any(g.name in line and f"{len(g.real_sections)} XSecs" in line for line in result.lines)


def test_the_negative_starts_upstream_of_the_opening_and_has_a_prismatic_segment(s_export):
    result, _, _ = s_export
    neg = result.plan.geom("DUCT_BRANCH_right_NEG")
    real = neg.real_sections
    assert [s.role for s in real[:3]] == ["lip", "station", "prism"]
    assert real[0].dist == pytest.approx(-0.05) and real[1].dist == pytest.approx(0.0) and real[2].dist > 0.0
    # the prismatic ring is the opening section moved along the inlet axis: same curve, same frame
    assert np.allclose(real[1].R, real[2].R) and np.allclose(real[1].ring, real[2].ring)
    assert np.allclose(real[2].origin - real[1].origin, real[2].dist * real[1].R[:, 0])
    pos = result.plan.geom("DUCT_BRANCH_right_POS").real_sections
    assert pos[0].dist == pytest.approx(0.0)  # the positive starts exactly at the opening plane


def test_the_branch_negative_overlaps_into_the_trunk_and_the_trunk_negative_starts_upstream(s_export):
    result, _, _ = s_export
    cfg_overlap = 0.02
    branch = result.plan.geom("DUCT_BRANCH_right_NEG").real_sections
    trunk = result.plan.geom("DUCT_TRUNK_NEG").real_sections
    assert branch[-1].role == "into" and branch[-1].dist - [s for s in branch if s.role == "station"][-1].dist == \
        pytest.approx(cfg_overlap, abs=1e-9)
    assert trunk[0].role == "overlap" and trunk[0].dist == pytest.approx(-cfg_overlap)


def test_the_positive_sections_are_within_the_tolerance_of_the_sampled_offset(s_export):
    result, out, _ = s_export
    geoms = saved(out)
    for name in ("DUCT_BRANCH_right_POS", "DUCT_TRUNK_POS"):
        plan = result.plan.geom(name)
        assert plan.fit.linear or plan.fit.max_dev < 0.005
        real = plan.real_sections
        surf = vsp.GetXSecSurf(geoms[name], 0)
        for k, s in enumerate(real):
            curve = cedit.sample_cedit(s.cedit)
            assert sf.relative_deviation(s.ring, curve) < 0.005 or plan.fit.linear
            got = vsp.GetEditXSecCtrlVec(vsp.GetXSec(surf, k + 1), True)  # what is in the file is what was planned
            assert max(max(abs(g.x() - p[0]), abs(g.y() - p[1])) for g, p in zip(got, s.cedit.points)) < 1e-9


def test_every_section_closes_on_itself_in_openvsp(s_export):
    """A closed Edit Curve must give a closed surface all along the Stack (OpenVSP's own flat-cap option loses a segment of
    some Edit Curves, which is why the caps are point XSecs)."""
    _, out, _ = s_export
    for name, gid in saved(out).items():
        us = np.linspace(0.05, 0.95, 9)
        a = vsp.CompVecPnt01(gid, 0, list(us), [0.0] * 9)
        b = vsp.CompVecPnt01(gid, 0, list(us), [1.0] * 9)
        assert max(np.hypot(p.y() - q.y(), p.z() - q.z()) for p, q in zip(a, b)) < 1e-6, name


def test_the_exported_bodies_are_watertight_meshes(s_export):
    _, out, _ = s_export
    saved(out)
    for name in ("DUCT_BRANCH_right_NEG", "DUCT_TRUNK_NEG", "DUCT_TRUNK_POS"):
        mesh = vspio.export_mesh([vspio.find_geom(name)])
        mesh = mesh.clean()
        assert mesh.n_open_edges < 0.01 * mesh.n_cells, name  # at most a few cracks of the cap tessellation


# ---- never touching anything else ---------------------------------------------------------------------------------------
def test_the_input_file_is_never_written_and_the_output_is_a_different_file(s_export):
    result, out, model_path = s_export
    assert result.out_path == out.resolve() and out.resolve() != model_path.resolve()
    assert any("was not modified" in line for line in result.lines)


def test_refusing_to_write_over_the_input(s_env, tmp_path):
    cfg, path, model_path = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "right", 2.0, 0.5, True)], junction=[4.0, 0.0, 0.2])
    before = sha(model_path)
    with pytest.raises(ValueError, match="refusing"):
        run(m, model_path)
    assert sha(model_path) == before


def test_an_empty_prefix_is_refused_because_it_would_match_every_geom(s_env, tmp_path):
    cfg, path, _ = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "right", 2.0, 0.5, True)], junction=[4.0, 0.0, 0.2])
    m.cfg["model"]["duct_prefix"] = "  "
    with pytest.raises(ValueError, match="duct_prefix"):
        run(m, tmp_path / "o.vsp3")


def test_re_export_replaces_the_duct_geoms_and_leaves_everything_else_identical(s_env, s_export, tmp_path):
    cfg, path, model_path = s_env
    _, first_out, _ = s_export
    vspio.open_model(model_path)
    original = vspio.param_hashes(PREFIX)
    names_original = {g.name for g in vspio.list_geoms()}
    # export again, from the file that already holds a duct, with a different design (one inlet on the axis)
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "top", 3.0, 0.0)], junction=[4.2, 0.0, 0.3])
    second = tmp_path / "second.vsp3"
    result = run(m, second, model_path=first_out)
    assert sorted(result.removed) == sorted(["DUCT_BRANCH_right_POS", "DUCT_BRANCH_right_NEG", "DUCT_TRUNK_POS",
                                             "DUCT_TRUNK_NEG"])
    geoms = saved(second)
    assert set(geoms) == {"DUCT_BRANCH_top_POS", "DUCT_BRANCH_top_NEG", "DUCT_TRUNK_POS", "DUCT_TRUNK_NEG"}  # none left over
    assert {g.name for g in vspio.list_geoms()} - set(geoms) == names_original
    assert vspio.param_hashes(PREFIX) == original  # every other geom: all parameter values identical


def test_only_geoms_with_the_prefix_are_ever_deleted(s_env, tmp_path):
    cfg, path, model_path = s_env
    vspio.open_model(model_path)
    decoy = vsp.AddGeom("POD")
    vsp.SetGeomName(decoy, "MY_DUCT_NOT_OURS")  # contains the prefix but does not start with it
    vsp.Update()
    with_decoy = tmp_path / "with_decoy.vsp3"
    vsp.WriteVSPFile(str(with_decoy), vsp.SET_ALL)
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "right", 2.0, 0.5, True)], junction=[4.0, 0.0, 0.2])
    result = run(m, tmp_path / "o.vsp3", model_path=with_decoy)
    assert result.removed == []
    assert "MY_DUCT_NOT_OURS" in {g.name for g in (saved(tmp_path / "o.vsp3") and vspio.list_geoms())}


# ---- the automatic verification ---------------------------------------------------------------------------------------------
def levels(result):
    return {label: level for label, _, level in result.verification.findings if label}


@pytest.mark.parametrize("which", ["s_export", "straight_export"])
def test_verification_passes_for_the_s_duct_and_the_straight_duct(which, request):
    result, _, _ = request.getfixturevalue(which)
    v = result.verification
    assert v is not None and not v.failed, "\n".join(result.lines)
    lv = levels(result)
    for label in ("Untouched Geoms", "File contents", "CompGeom", "Volume", "Area"):
        assert lv[label] == "pass", (label, "\n".join(v.lines()))
    assert result.lines[-1].endswith("PASS") or "warnings" in result.lines[-1]


@pytest.mark.parametrize("which", ["s_export", "straight_export"])
def test_the_exported_negatives_match_the_area_law_within_three_percent(which, request):
    result, _, _ = request.getfixturevalue(which)
    assert abs(result.verification.numbers["area_error"]) < 0.03


@pytest.mark.parametrize("which", ["s_export", "straight_export"])
def test_compgeom_gives_a_non_empty_result_and_the_volumes_agree(which, request):
    result, _, _ = request.getfixturevalue(which)
    n = result.verification.numbers
    assert n["volume_compgeom"] > 0.0
    assert abs(n["volume_compgeom"] - n["volume_ours"]) / n["volume_ours"] < 0.03
    assert abs(n["removed_compgeom"] - n["removed_ours"]) / abs(n["removed_ours"]) < 0.03


def test_the_line_of_sight_on_the_exported_mesh_matches_the_viewer(s_export, straight_export):
    for result, _, _ in (s_export, straight_export):
        n = result.verification.numbers
        assert "los_x" in n and "los_n" in n
        assert levels(result)["Line of sight"] == "pass"


def test_verification_notices_a_modified_geom(s_export, tmp_path):
    """Change a parameter of the fuselage in the saved file: the untouched-geoms check must fail."""
    from ductmgr import exportcheck

    result, out, model_path = s_export
    vspio.open_model(model_path)
    before = vspio.param_hashes(PREFIX)
    names = {g.id: g.name for g in vspio.list_geoms()}
    vspio.open_model(out)
    fus = next(g for g in vsp.FindGeoms() if vsp.GetGeomName(g) == "Test Fuselage")
    vsp.SetParmVal(fus, "Length", "Design", 8.5)
    vsp.Update()
    v = exportcheck.Verification()
    exportcheck.check_untouched(v, before, names, PREFIX, [g.name for g in result.plan.geoms])
    assert v.failed and "Test Fuselage" in v.findings[0][1]


def test_no_compgeom_files_or_temp_sets_are_left_next_to_the_output(s_export):
    _, out, _ = s_export
    assert not list(out.parent.glob("*_CompGeom.*"))
    vspio.open_model(out)
    assert all(vsp.GetSetName(i) != vspio.TEMP_SET_NAME for i in range(vsp.GetNumSets()))
    assert {g.name for g in vspio.list_geoms()} == {"Test Fuselage", "Test Engine", "DUCT_BRANCH_right_POS",
                                                    "DUCT_BRANCH_right_NEG", "DUCT_TRUNK_POS", "DUCT_TRUNK_NEG"}


# ---- a crash of OpenVSP is reported where it happened ------------------------------------------------------------------
def fake_children(monkeypatch, crash_first_write_of=None, retry_ok=True):
    """Replace the child processes: 'prepare' and 'write' succeed (empty files), except that the first 'write' of the
    named Geom dies in its XSec 3 (and, with ``retry_ok`` False, so does every later one)."""
    seen = {"n": 0}

    def run(task, args, tmp, say):
        if task == "prepare":
            args["out_file"].write_bytes(b"")
            return "ok", {"before": {}, "names": {}, "removed": []}, [], ""
        g = args["geom"]
        if g.name == crash_first_write_of and (seen["n"] == 0 or not retry_ok):
            seen["n"] += 1
            where = f"{g.name}: Edit Curve of XSec 3 of {len(g.sections) - 1} (exit code 0xC0000374)"
            return "crash", None, [], where
        args["out_file"].write_bytes(b"")
        return "ok", True, [], f"Writing {g.name}"

    monkeypatch.setattr(exportrun, "_run_task", run)


def test_a_crash_is_an_error_naming_the_geom_and_the_xsec_when_retrying_is_off(s_env, tmp_path, monkeypatch):
    cfg, path, _ = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "right", 2.0, 0.5, True)], junction=[4.0, 0.0, 0.2])
    m.cfg.setdefault("export", {})["retry_on_crash"] = False
    fake_children(monkeypatch, crash_first_write_of="DUCT_BRANCH_right_NEG")
    with pytest.raises(exportrun.OpenVSPCrash) as info:
        run(m, tmp_path / "o.vsp3")
    text = str(info.value)
    assert text.startswith("ERROR") and "DUCT_BRANCH_right_NEG" in text and "Edit Curve of XSec 3" in text
    assert "0xC0000374" in text and "retry_on_crash" in text


def test_with_retrying_on_the_crash_and_the_changed_sections_are_reported_as_warnings(s_env, tmp_path, monkeypatch):
    cfg, path, _ = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "right", 2.0, 0.5, True)], junction=[4.0, 0.0, 0.2])
    fake_children(monkeypatch, crash_first_write_of="DUCT_TRUNK_NEG")
    result = run(m, tmp_path / "o.vsp3")
    lines = [ln for ln in result.lines if "WARN" in ln]
    assert any("OpenVSP crashed at: DUCT_TRUNK_NEG: Edit Curve of XSec 3" in ln for ln in lines)
    assert any("differ from the first plan" in ln and "Bezier segments" in ln for ln in lines)


def test_every_version_crashing_is_an_error_too(s_env, tmp_path, monkeypatch):
    cfg, path, _ = s_env
    m = env.explorer_model(cfg, path, [lambda s: env.inlet(s, "right", 2.0, 0.5, True)], junction=[4.0, 0.0, 0.2])
    fake_children(monkeypatch, crash_first_write_of="DUCT_TRUNK_POS", retry_ok=False)
    with pytest.raises(exportrun.OpenVSPCrash, match="every version of the Geom crashed it"):
        run(m, tmp_path / "o.vsp3")


def test_the_verification_checks_that_every_edit_curve_is_stored_as_written(s_export):
    result, _, _ = s_export
    assert levels(result)["Edit Curve XSecs"] == "pass"
    assert result.verification.numbers["xsec_deviation"] < 1e-3  # 0.1 % of the section size
