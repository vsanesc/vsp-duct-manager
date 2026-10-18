# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Saving the inlet definition into a config: comments and other keys survive; the example is never written."""
import shutil

import pytest
import yaml

from ductmgr.config import EXAMPLE_CONFIG, load_config, save_inlet_definition

DEFINITION = {
    "centroid": [3.123456789, 0.4, 1.5],
    "normal": [-0.9659258263, 0.0, 0.2588190451],
    "tilt_deg": 15.0,
    "shape": {"width_bottom": 1.1, "width_top": 0.6, "height": 1.0, "apex_height": 0.0, "skew": -0.2},
}


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
    assert inlet["centroid"] == [3.123457, 0.4, 1.5]  # rounded to 6 decimals
    assert inlet["normal"] == [-0.965926, 0.0, 0.258819]
    assert inlet["tilt_deg"] == 15.0
    assert inlet["shape"] == DEFINITION["shape"]

    for section in ("model", "engine", "design_point", "checks"):
        assert after[section] == before[section]
    assert inlet["lip_extension"] == before["ducts"]["inlet"]["lip_extension"]
    assert after["ducts"]["junction"] == before["ducts"]["junction"]


def test_comments_and_layout_are_preserved(target):
    original = target.read_text(encoding="utf-8").splitlines()
    save_inlet_definition(DEFINITION, target)
    written = target.read_text(encoding="utf-8").splitlines()
    assert len(written) == len(original)
    touched = ("centroid", "normal", "tilt_deg", "width_bottom", "width_top", "height", "apex_height", "skew")
    for old, new in zip(original, written):
        if not any(old.strip().startswith(k + ":") for k in touched):
            assert new == old  # every untouched line, comments included, is byte-identical
        else:
            assert "#" in new or "#" not in old  # trailing comments on edited lines survive


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
    assert cfg["ducts"]["inlet"]["tilt_deg"] == 15.0
    assert cfg["engine"] == load_config(EXAMPLE_CONFIG)[0]["engine"]


def test_inlet_section_is_created_when_absent_or_null(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("# my comment\nmodel:\n  vsp3_path: a.vsp3\nducts:\n  inlet:\n", encoding="utf-8")
    save_inlet_definition(DEFINITION, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# my comment\n")
    assert yaml.safe_load(text)["ducts"]["inlet"]["shape"]["skew"] == -0.2

    path.write_text("model:\n  vsp3_path: a.vsp3\n", encoding="utf-8")  # no `ducts` at all
    save_inlet_definition(DEFINITION, path)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["ducts"]["inlet"]["centroid"][2] == 1.5


def test_existing_shape_comments_survive_a_partial_update(target):
    save_inlet_definition({"shape": {"skew": 0.4}}, target)
    text = target.read_text(encoding="utf-8")
    assert "skew: 0.4" in text and "# lateral apex offset, fraction of width_top" in text
    assert yaml.safe_load(text)["ducts"]["inlet"]["shape"]["width_top"] == 0.7  # untouched sibling


def test_crlf_line_endings_are_preserved(tmp_path):
    path = tmp_path / "crlf.yaml"
    path.write_bytes(EXAMPLE_CONFIG.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    save_inlet_definition(DEFINITION, path)
    raw = path.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
