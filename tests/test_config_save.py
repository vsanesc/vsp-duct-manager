# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Saving the inlet definition into a config: comments and other keys survive; the example is never written."""
import shutil

import numpy as np
import pytest
import yaml

from ductmgr.config import EXAMPLE_CONFIG, load_config, save_inlet_definition

DEFINITION = {
    "origin": [3.123456789, 0.4, 1.5],
    "pitch_deg": 15.0,
    "sweep_deg": -22.5,
    "vertices": [[-0.1, -0.05], [0.12, 0.02], [0.11, 0.3], [0.0123456789, 0.38], [-0.11, 0.3]],
    "snapped": [True, True, False, False, False],
}
TOUCHED = ("origin:", "pitch_deg:", "sweep_deg:", "snapped:", "- [")  # lines whose values may change


@pytest.fixture()
def target(tmp_path):
    path = tmp_path / "duct_config.yaml"
    shutil.copyfile(EXAMPLE_CONFIG, path)
    return path


def test_values_are_written_and_everything_else_is_untouched(target):
    before = yaml.safe_load(target.read_text(encoding="utf-8"))
    save_inlet_definition(DEFINITION, target)
    after = yaml.safe_load(target.read_text(encoding="utf-8"))

    inlet = after["ducts"]["inlet"]
    assert inlet["origin"] == [3.123457, 0.4, 1.5]  # rounded to 6 decimals
    assert inlet["pitch_deg"] == 15.0 and inlet["sweep_deg"] == -22.5
    assert inlet["vertices"] == [[-0.1, -0.05], [0.12, 0.02], [0.11, 0.3], [0.012346, 0.38], [-0.11, 0.3]]
    assert inlet["snapped"] == [True, True, False, False, False]
    assert all(isinstance(f, bool) for f in inlet["snapped"])

    for section in ("model", "engine", "design_point", "checks"):
        assert after[section] == before[section]
    assert inlet["lip_extension"] == before["ducts"]["inlet"]["lip_extension"]
    assert after["ducts"]["junction"] == before["ducts"]["junction"]


def test_comments_and_layout_are_preserved(target):
    original = target.read_text(encoding="utf-8").splitlines()
    save_inlet_definition(DEFINITION, target)
    written = target.read_text(encoding="utf-8").splitlines()
    assert len(written) == len(original)
    for old, new in zip(original, written):
        if not any(old.strip().startswith(k) for k in TOUCHED):
            assert new == old  # every untouched line, comments included, is byte-identical
        else:
            assert ("#" in new) == ("#" in old)  # trailing comments on edited lines survive, none are invented
    text = target.read_text(encoding="utf-8")
    assert "# the 5 sketch vertices [u, v] in metres relative to O" in text  # comments on the block list


def test_the_vertices_are_one_pair_per_line_in_flow_style(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("ducts:\n  inlet:\n", encoding="utf-8")  # no vertices yet: the list is created
    save_inlet_definition(DEFINITION, path)
    text = path.read_text(encoding="utf-8")
    assert "vertices:\n" in text and text.count("- [") == 5
    assert "[-0.1, -0.05]" in text and "snapped: [true, true, false, false, false]" in text
    assert yaml.safe_load(text)["ducts"]["inlet"]["vertices"][3] == [0.012346, 0.38]


def test_vertices_round_trip_in_u_v(target):
    save_inlet_definition(DEFINITION, target)
    cfg, _ = load_config(target)
    inlet = cfg["ducts"]["inlet"]
    assert np.array(inlet["vertices"]) == pytest.approx(np.array(DEFINITION["vertices"]), abs=5e-7)
    assert inlet["origin"] == pytest.approx(DEFINITION["origin"], abs=5e-7)
    # saving what was read back changes nothing: the (u, v) pairs are stable
    once = target.read_bytes()
    save_inlet_definition({k: inlet[k] for k in ("origin", "pitch_deg", "sweep_deg", "vertices", "snapped")}, target)
    assert target.read_bytes() == once


def test_saving_the_same_values_twice_is_stable(target):
    save_inlet_definition(DEFINITION, target)
    once = target.read_bytes()
    save_inlet_definition(DEFINITION, target)
    assert target.read_bytes() == once


def test_the_example_config_is_never_written():
    before = EXAMPLE_CONFIG.read_bytes()
    with pytest.raises(PermissionError, match="example"):
        save_inlet_definition(DEFINITION, EXAMPLE_CONFIG)
    assert EXAMPLE_CONFIG.read_bytes() == before


def test_missing_target_is_created_from_the_example(tmp_path):
    path = tmp_path / "new_config.yaml"
    save_inlet_definition(DEFINITION, path)
    cfg, _ = load_config(path)
    assert cfg["ducts"]["inlet"]["sweep_deg"] == -22.5
    assert cfg["engine"] == load_config(EXAMPLE_CONFIG)[0]["engine"]


def test_inlet_section_is_created_when_absent_or_null(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("# my comment\nmodel:\n  vsp3_path: a.vsp3\nducts:\n  inlet:\n", encoding="utf-8")
    save_inlet_definition(DEFINITION, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# my comment\n")
    assert yaml.safe_load(text)["ducts"]["inlet"]["pitch_deg"] == 15.0

    path.write_text("model:\n  vsp3_path: a.vsp3\n", encoding="utf-8")  # no `ducts` at all
    save_inlet_definition(DEFINITION, path)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]["inlet"]["origin"][2] == 1.5


def test_old_keys_in_a_config_are_left_alone(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("ducts:\n  inlet:\n    centroid: [1.0, 2.0, 3.0]   # old\n    shape: {width_top: 0.7}\n",
                    encoding="utf-8")
    save_inlet_definition(DEFINITION, path)
    inlet = yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]["inlet"]
    assert inlet["centroid"] == [1.0, 2.0, 3.0] and inlet["shape"] == {"width_top": 0.7}  # never deleted
    assert "# old" in path.read_text(encoding="utf-8")


def test_crlf_line_endings_are_preserved(tmp_path):
    path = tmp_path / "crlf.yaml"
    path.write_bytes(EXAMPLE_CONFIG.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    save_inlet_definition(DEFINITION, path)
    raw = path.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
