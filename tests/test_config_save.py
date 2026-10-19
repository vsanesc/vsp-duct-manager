# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Saving the inlet plane and key sections into a config: comments and other keys survive; the example is never
written."""
import shutil

import pytest
import yaml

from ductmgr.config import EXAMPLE_CONFIG, load_config, save_inlet_definition

PLANE = {"origin": [3.123456789, 0.4, 1.5], "pitch_deg": 15.0, "sweep_deg": -22.5}
KEY_SECTIONS = [
    {"s": 0.0, "vertices": [[-0.1, -0.05], [0.12, 0.02], [0.11, 0.3], [0.0123456789, 0.38], [-0.11, 0.3]],
     "snapped": [True, True, False, False, False], "edges": ["straight", "curved", "straight", "straight", "straight"],
     "handles": [None, [[0.01, 0.02], [-0.03, 0.04]], None, None, None],
     "continuity": ["corner", "smooth", "symmetric", "corner", "corner"]},
    {"s": 0.35, "vertices": [[0.0, 0.0], [0.2, 0.0], [0.1, 0.2]], "snapped": [False] * 3,
     "edges": ["straight"] * 3, "handles": [None] * 3, "continuity": ["corner"] * 3},
]
OLD_FORMAT = {**PLANE, "vertices": [[-0.1, -0.05], [0.12, 0.02], [0.11, 0.3]], "snapped": [True, False, False]}


@pytest.fixture()
def target(tmp_path):
    path = tmp_path / "duct_config.yaml"
    shutil.copyfile(EXAMPLE_CONFIG, path)
    return path


def test_the_plane_and_key_sections_are_written_and_read_back(target):
    save_inlet_definition(PLANE, target, KEY_SECTIONS, remove_inlet_keys=("vertices", "snapped"))
    cfg, _ = load_config(target)
    inlet = cfg["ducts"]["inlet"]
    assert inlet["origin"] == [3.123457, 0.4, 1.5]  # rounded to 6 decimals
    assert inlet["pitch_deg"] == 15.0 and inlet["sweep_deg"] == -22.5
    got = cfg["ducts"]["key_sections"]
    assert [e["s"] for e in got] == [0.0, 0.35]
    assert got[0]["vertices"][3] == [0.012346, 0.38]
    assert got[0]["handles"][0] is None and got[0]["handles"][1] == [[0.01, 0.02], [-0.03, 0.04]]
    assert got[0]["edges"][1] == "curved" and got[0]["continuity"][2] == "symmetric"
    assert got[0]["snapped"] == [True, True, False, False, False] and all(isinstance(f, bool) for f in got[0]["snapped"])
    assert len(got[1]["vertices"]) == 3


def test_everything_else_is_untouched(target):
    before = yaml.safe_load(target.read_text(encoding="utf-8"))
    save_inlet_definition(PLANE, target, KEY_SECTIONS)
    after = yaml.safe_load(target.read_text(encoding="utf-8"))
    for section in ("model", "engine", "design_point", "checks"):
        assert after[section] == before[section]
    assert after["ducts"]["lip_extension"] == before["ducts"]["lip_extension"]
    assert after["ducts"]["junction"] == before["ducts"]["junction"]


def test_comments_outside_the_key_sections_are_preserved_line_for_line(target):
    original = target.read_text(encoding="utf-8").splitlines()
    save_inlet_definition(PLANE, target, KEY_SECTIONS)
    written = target.read_text(encoding="utf-8").splitlines()

    def outside(lines):  # everything except the plane values and the key_sections block
        a = next(i for i, x in enumerate(lines) if x.startswith("  key_sections:"))
        b = next(i for i, x in enumerate(lines) if x.startswith("  junction:"))
        return lines[:a], lines[b:]

    (o_head, o_tail), (w_head, w_tail) = outside(original), outside(written)
    def plain(lines):  # the round trip writes a null as an empty value; nothing else may change
        return [" ".join(x.replace("null", "").split()) for x in lines]

    assert plain(o_tail) == plain(w_tail)  # junction onwards, comments included
    touched = ("origin:", "pitch_deg:", "sweep_deg:")
    for old, new in zip(o_head, w_head):
        if not any(old.strip().startswith(k) for k in touched):
            assert new == old
        else:
            assert ("#" in new) == ("#" in old)  # trailing comments on edited lines survive, none are invented
    assert len(o_head) == len(w_head)


def test_the_old_vertices_and_snapped_are_removed_on_save(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("ducts:\n  inlet:\n    origin: [1.0, 0.5, 1.0]\n    vertices:\n      - [0.0, 0.0]\n"
                    "    snapped: [false]\n    lip_extension: 0.05\n", encoding="utf-8")
    save_inlet_definition(PLANE, path, KEY_SECTIONS, remove_inlet_keys=("vertices", "snapped"))
    inlet = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]["inlet"]
    assert "vertices" not in inlet and "snapped" not in inlet
    assert inlet["lip_extension"] == 0.05 and inlet["pitch_deg"] == 15.0


def test_key_sections_are_one_vertex_per_line_in_flow_style(target):
    save_inlet_definition(PLANE, target, KEY_SECTIONS)
    text = target.read_text(encoding="utf-8")
    assert "- [-0.1, -0.05]" in text and "edges: [straight, curved, straight, straight, straight]" in text
    assert "handles: [null, [[0.01, 0.02], [-0.03, 0.04]], null, null, null]" in text


def test_saving_the_same_values_twice_is_stable(target):
    save_inlet_definition(PLANE, target, KEY_SECTIONS)
    once = target.read_bytes()
    save_inlet_definition(PLANE, target, KEY_SECTIONS)
    assert target.read_bytes() == once
    cfg, _ = load_config(target)  # and saving what was read back changes nothing
    save_inlet_definition({k: cfg["ducts"]["inlet"][k] for k in PLANE}, target, cfg["ducts"]["key_sections"])
    assert target.read_bytes() == once


def test_fewer_key_sections_replace_the_old_list(target):
    save_inlet_definition(PLANE, target, KEY_SECTIONS)
    save_inlet_definition(PLANE, target, KEY_SECTIONS[:1])
    assert len(load_config(target)[0]["ducts"]["key_sections"]) == 1


def test_key_sections_are_left_alone_when_not_given(target):
    save_inlet_definition(PLANE, target, KEY_SECTIONS)
    save_inlet_definition({**PLANE, "pitch_deg": 3.0}, target)
    cfg, _ = load_config(target)
    assert cfg["ducts"]["inlet"]["pitch_deg"] == 3.0 and len(cfg["ducts"]["key_sections"]) == 2


def test_a_plain_old_format_definition_is_still_writable(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("ducts:\n  inlet:\n", encoding="utf-8")
    save_inlet_definition(OLD_FORMAT, path)
    inlet = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]["inlet"]
    assert inlet["vertices"][2] == [0.11, 0.3] and inlet["snapped"] == [True, False, False]


def test_the_example_config_is_never_written():
    before = EXAMPLE_CONFIG.read_bytes()
    with pytest.raises(PermissionError, match="example"):
        save_inlet_definition(PLANE, EXAMPLE_CONFIG, KEY_SECTIONS)
    assert EXAMPLE_CONFIG.read_bytes() == before


def test_missing_target_is_created_from_the_example(tmp_path):
    path = tmp_path / "new_config.yaml"
    save_inlet_definition(PLANE, path, KEY_SECTIONS)
    cfg, _ = load_config(path)
    assert cfg["ducts"]["inlet"]["sweep_deg"] == -22.5
    assert cfg["engine"] == load_config(EXAMPLE_CONFIG)[0]["engine"]


def test_inlet_section_is_created_when_absent_or_null(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("# my comment\nmodel:\n  vsp3_path: a.vsp3\nducts:\n  inlet:\n", encoding="utf-8")
    save_inlet_definition(PLANE, path, KEY_SECTIONS)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# my comment\n")
    assert yaml.safe_load(text)["ducts"]["inlet"]["pitch_deg"] == 15.0

    path.write_text("model:\n  vsp3_path: a.vsp3\n", encoding="utf-8")  # no `ducts` at all
    save_inlet_definition(PLANE, path, KEY_SECTIONS)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["ducts"]["inlet"]["origin"][2] == 1.5 and len(cfg["ducts"]["key_sections"]) == 2


def test_old_keys_in_a_config_are_left_alone(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("ducts:\n  inlet:\n    centroid: [1.0, 2.0, 3.0]   # old\n    shape: {width_top: 0.7}\n",
                    encoding="utf-8")
    save_inlet_definition(PLANE, path, KEY_SECTIONS)
    inlet = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]["inlet"]
    assert inlet["centroid"] == [1.0, 2.0, 3.0] and inlet["shape"] == {"width_top": 0.7}  # never deleted
    assert "# old" in path.read_text(encoding="utf-8")


def test_crlf_line_endings_are_preserved(tmp_path):
    path = tmp_path / "crlf.yaml"
    path.write_bytes(EXAMPLE_CONFIG.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    save_inlet_definition(PLANE, path, KEY_SECTIONS)
    raw = path.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")


# ---- Phase 3b: junction, centreline handles, area points, wall ---------------------------------------------------
DUCTS = {
    "wall_thickness": 0.012, "lip_extension": 0.07, "lip_mode": "blunt",
    "junction": {"point": [3.5, 0.0, 0.25]},
    "centreline": {"junction_pitch_deg": 14.0, "start_handle": 0.6, "junction_in_handle": None,
                   "junction_out_handle": None, "end_handle": None, "handle_fraction": 0.3333},
    "area_points": {"junction_area": 0.2, "branch": [[0.3, 0.11], [0.7, 0.13]], "trunk": None},
}


def test_ducts_values_are_merged_into_the_ducts_block(target):
    save_inlet_definition(PLANE, target, KEY_SECTIONS, ducts_values=DUCTS)
    d = load_config(target)[0]["ducts"]
    assert d["wall_thickness"] == 0.012 and d["lip_extension"] == 0.07 and d["lip_mode"] == "blunt"
    assert d["junction"]["point"] == [3.5, 0.0, 0.25]
    assert d["centreline"]["start_handle"] == 0.6 and d["centreline"]["junction_in_handle"] is None
    assert d["area_points"]["branch"] == [[0.3, 0.11], [0.7, 0.13]] and d["area_points"]["trunk"] is None
    assert d["area_points"]["junction_area"] == 0.2


def test_existing_comments_survive_the_ducts_values(target):
    save_inlet_definition(PLANE, target, KEY_SECTIONS, ducts_values=DUCTS)
    text = target.read_text(encoding="utf-8")
    assert "# [x, 0, z] where the branches merge" in text
    assert "# m, the positive body is the inner section offset outward" in text
    assert "# tangent direction at the junction in the XZ plane" in text


def test_ducts_values_are_stable_and_a_missing_block_is_created(target, tmp_path):
    save_inlet_definition(PLANE, target, KEY_SECTIONS, ducts_values=DUCTS)
    once = target.read_bytes()
    save_inlet_definition(PLANE, target, KEY_SECTIONS, ducts_values=DUCTS)
    assert target.read_bytes() == once
    bare = tmp_path / "bare.yaml"
    bare.write_text("ducts:\n  n_inlets: 2\n", encoding="utf-8")
    save_inlet_definition(PLANE, bare, ducts_values=DUCTS)
    d = yaml.safe_load(bare.read_text(encoding="utf-8"))["ducts"]
    assert d["centreline"]["junction_pitch_deg"] == 14.0 and d["area_points"]["branch"][0] == [0.3, 0.11]
    assert d["n_inlets"] == 2


def test_the_old_inlet_lip_extension_can_be_removed_on_save(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("ducts:\n  inlet:\n    lip_extension: 0.05\n    origin: [1.0, 0.5, 1.0]\n", encoding="utf-8")
    save_inlet_definition(PLANE, path, remove_inlet_keys=("lip_extension",), ducts_values={"lip_extension": 0.07})
    d = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]
    assert "lip_extension" not in d["inlet"] and d["lip_extension"] == 0.07
