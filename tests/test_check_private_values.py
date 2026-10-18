# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Tests for the private-value pre-commit hook. All values are invented; the real config is never read."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import check_private_values as cpv

SCRIPT = Path(cpv.__file__).resolve()

PRIVATE = {
    "model": {"vsp3_path": "Secret_Jet-9.vsp3", "fuselage_name": "Zorro Alpha", "duct_prefix": "DUCT_"},
    "engine": {"face_diameter": 7.77, "hub_to_tip_ratio": 0.0, "corrected_flow_per_area": 190.0, "limit": 31415},
    "design_point": {"mach": 0.123, "pressure_recovery": 0.97, "flag": True, "range": [111, 222]},
    "ducts": {"n_inlets": 2, "offset": -4.56, "tag": "ab", "label": "cosine"},
}
EXAMPLE = {
    "model": {"duct_prefix": "DUCT_"},
    "engine": {"corrected_flow_per_area": 190.0},
    "design_point": {"pressure_recovery": 0.97},
    "ducts": {"label": "cosine"},
}


def keys(pairs):
    return {key for _, key in pairs}


# ---- collection ------------------------------------------------------------------------------
def test_trivial_and_public_values_are_skipped():
    numbers, strings = cpv.collect_private_values(PRIVATE, EXAMPLE)
    assert keys(numbers) == {
        "engine.face_diameter",
        "engine.limit",
        "design_point.mach",
        "design_point.range[0]",
        "design_point.range[1]",
        "ducts.offset",
    }
    # skipped: 0.0, 2, True, shared 190.0 / 0.97, short string "ab", shared strings "DUCT_" / "cosine"
    assert keys(strings) == {"model.vsp3_path", "model.fuselage_name"}


def test_file_name_also_contributes_its_stem():
    _, strings = cpv.collect_private_values(PRIVATE, EXAMPLE)
    assert ("secret_jet-9", "model.vsp3_path") in strings
    assert ("secret_jet-9.vsp3", "model.vsp3_path") in strings


def test_no_example_means_nothing_is_public():
    numbers, _ = cpv.collect_private_values(PRIVATE, None)
    assert "engine.corrected_flow_per_area" in keys(numbers)


# ---- matching ----------------------------------------------------------------------------------
@pytest.fixture()
def values():
    return cpv.collect_private_values(PRIVATE, EXAMPLE)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("face = 7.77", {"engine.face_diameter"}),
        ("face = 7.770  # trailing zero", {"engine.face_diameter"}),
        ("x = [1.0, 7.77, 3]", {"engine.face_diameter"}),
        ("sentence ends with 7.77.", {"engine.face_diameter"}),
        ("shift = -4.56", {"ducts.offset"}),
        ("shift = 4.56", {"ducts.offset"}),  # sign is ignored
        ("limit 31,415 units", {"engine.limit"}),
        ("limit = 31415.0", {"engine.limit"}),
        ("limit = 3.1415e4", {"engine.limit"}),
        ("band [111,222]", {"design_point.range[0]", "design_point.range[1]"}),
        ("mach: 0.123", {"design_point.mach"}),
        ("open Secret_Jet-9.vsp3 now", {"model.vsp3_path"}),
        ("see SECRET_JET-9 notes", {"model.vsp3_path"}),
        ("fuselage 'zorro alpha'", {"model.fuselage_name"}),
    ],
)
def test_hits(values, line, expected):
    assert cpv.find_hits(line, *values) == expected


@pytest.mark.parametrize(
    "line",
    [
        "version v7.77.1",  # glued to a letter / longer dotted number
        "id = a7.77",
        "x = 17.77",
        "x = 7.775",
        "x = 77.7",
        "name_7_77",
        "ratio 190.0 and 0.97",  # public (shared with the example)
        "zero 0.0, one 1, two 2",  # trivial
        "plain text with nothing",
    ],
)
def test_non_hits(values, line):
    assert cpv.find_hits(line, *values) == set()


# ---- diff parsing ----------------------------------------------------------------------------------
def test_added_lines_tracks_paths_and_line_numbers():
    diff = "\n".join(
        [
            "diff --git a/a.py b/a.py",
            "--- a/a.py",
            "+++ b/a.py",
            "@@ -1,0 +5,2 @@",
            "+first",
            "+second",
            "@@ -10 +20 @@",
            "-old",
            "+new",
            "diff --git a/gone.py b/gone.py",
            "--- a/gone.py",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-bye",
            "diff --git a/dir/b.txt b/dir/b.txt",
            "--- /dev/null",
            "+++ b/dir/b.txt",
            "@@ -0,0 +1 @@",
            "+hello",
        ]
    )
    assert list(cpv.added_lines(diff)) == [
        ("a.py", 5, "first"),
        ("a.py", 6, "second"),
        ("a.py", 20, "new"),
        ("dir/b.txt", 1, "hello"),
    ]


# ---- end to end in a throw-away git repository -------------------------------------------------------
def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def run_hook(repo):
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=repo, capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "duct_config.yaml").write_text(yaml.safe_dump(PRIVATE), encoding="utf-8")
    (tmp_path / "duct_config.example.yaml").write_text(yaml.safe_dump(EXAMPLE), encoding="utf-8")
    return tmp_path


def test_blocks_staged_private_value_without_echoing_it(repo):
    (repo / "notes.py").write_text("ok = 1\nface = 7.77\nmodel = 'Zorro Alpha'\n", encoding="utf-8")
    git(repo, "add", "notes.py")
    result = run_hook(repo)
    assert result.returncode == 1
    assert "notes.py:2:" in result.stdout and "notes.py:3:" in result.stdout
    assert "engine.face_diameter" in result.stdout and "model.fuselage_name" in result.stdout
    for secret in ("7.77", "Zorro", "zorro"):
        assert secret not in result.stdout + result.stderr


def test_clean_staged_file_passes(repo):
    (repo / "notes.py").write_text("answer = 42\nname = 'neutral'\nratio = 190.0\n", encoding="utf-8")
    git(repo, "add", "notes.py")
    result = run_hook(repo)
    assert result.returncode == 0 and result.stdout == ""


def test_only_added_lines_are_scanned(repo):
    (repo / "notes.py").write_text("face = 7.77\n", encoding="utf-8")
    git(repo, "add", "notes.py")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-q", "--no-verify", "-m", "seed")
    (repo / "notes.py").write_text("", encoding="utf-8")  # removing the value is allowed
    git(repo, "add", "notes.py")
    assert run_hook(repo).returncode == 0


def test_staging_the_private_config_itself_is_blocked(repo):
    git(repo, "add", "-f", "duct_config.yaml")
    result = run_hook(repo)
    assert result.returncode == 1
    assert "private config itself is staged" in result.stdout
    assert "7.77" not in result.stdout


def test_passes_silently_without_private_config(repo):
    (repo / "duct_config.yaml").unlink()
    (repo / "notes.py").write_text("face = 7.77\n", encoding="utf-8")
    git(repo, "add", "notes.py")
    result = run_hook(repo)
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""


def test_unparseable_config_fails_without_echoing_content(repo):
    (repo / "duct_config.yaml").write_text("face: 7.77\n  bad: [unclosed\n", encoding="utf-8")
    result = run_hook(repo)
    assert result.returncode == 1
    assert "7.77" not in result.stdout + result.stderr
