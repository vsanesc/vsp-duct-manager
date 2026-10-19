# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The area law A(s) along a branch."""
import math

import numpy as np
import pytest

from ductmgr import area_law as al

A_IN, A_FACE = 0.11, 0.264  # m^2: a branch inlet and the engine face
SJ = 0.4


@pytest.mark.parametrize("law", al.LAWS)
def test_the_weight_goes_from_zero_to_one(law):
    assert al.weight(0.0, law) == 0.0 and al.weight(1.0, law) == pytest.approx(1.0)
    ws = [al.weight(s, law) for s in np.linspace(0.0, 1.0, 21)]
    assert all(b >= a for a, b in zip(ws, ws[1:]))  # monotonic


def test_linear_and_cosine_weights():
    assert al.weight(0.3, "linear") == 0.3
    assert al.weight(0.5, "cosine") == pytest.approx(0.5)
    assert al.weight(0.25, "cosine") == pytest.approx(0.5 * (1 - math.cos(math.pi / 4)))
    assert al.weight(0.25, "cosine") < 0.25  # the cosine law starts slowly and ends slowly
    assert al.weight(0.75, "cosine") > 0.75


@pytest.mark.parametrize("law", al.LAWS)
def test_the_end_values_are_the_inlet_and_the_face_area(law):
    assert al.section_area(0.0, A_IN, A_FACE, SJ, law) == pytest.approx(A_IN)
    assert al.section_area(1.0, A_IN, A_FACE, SJ, law) == pytest.approx(A_FACE)  # the trunk carries both branches


@pytest.mark.parametrize("law", al.LAWS)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_junction_area_is_n_times_the_branch_area_there(law, n):
    just_before = al.section_area(SJ, A_IN, A_FACE, SJ, law, n)  # the junction itself is still a branch section
    just_after = al.section_area(SJ + 1e-9, A_IN, A_FACE, SJ, law, n)
    assert just_after == pytest.approx(n * just_before, rel=1e-6)
    assert al.section_area(1.0, A_IN, A_FACE, SJ, law, n) == pytest.approx(A_FACE)


def test_the_law_by_hand():
    s = 0.2
    branch = A_IN + (A_FACE / 2 - A_IN) * 0.5 * (1 - math.cos(math.pi * s))
    assert al.section_area(s, A_IN, A_FACE, SJ, "cosine") == pytest.approx(branch)
    s = 0.7  # in the trunk
    branch = A_IN + (A_FACE / 2 - A_IN) * s
    assert al.section_area(s, A_IN, A_FACE, SJ, "linear") == pytest.approx(2 * branch)


def test_branch_and_trunk_areas_are_each_monotonic_when_the_face_is_larger():
    for law in al.LAWS:
        branch = [al.section_area(s, A_IN, A_FACE, SJ, law) for s in np.linspace(0.0, SJ, 15)]
        trunk = [al.section_area(s, A_IN, A_FACE, SJ, law) for s in np.linspace(SJ + 1e-6, 1.0, 15)]
        assert all(b >= a for a, b in zip(branch, branch[1:])) and all(b >= a for a, b in zip(trunk, trunk[1:]))


def test_invalid_arguments():
    with pytest.raises(ValueError, match="within"):
        al.weight(1.2)
    with pytest.raises(ValueError, match="area law"):
        al.weight(0.5, "parabolic")
    with pytest.raises(ValueError, match="s_junction"):
        al.section_area(0.5, A_IN, A_FACE, 1.5)


def test_the_junction_position_from_a_point_or_a_fraction():
    o, f = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    assert al.junction_s(o, f, junction_point=(3.0, 0.0, 0.0)) == pytest.approx(0.3)
    assert al.junction_s(o, f, junction_point=(4.0, 3.0, 0.0)) == pytest.approx(5.0 / (5.0 + math.hypot(6.0, 3.0)))
    assert al.junction_s(o, f) == al.DEFAULT_JUNCTION_FRACTION
    assert al.junction_s(o, f, fraction=0.35) == 0.35
    assert al.junction_s(o, None, junction_point=(3.0, 0.0, 0.0), fraction=0.6) == 0.6  # no face: the fraction
    with pytest.raises(ValueError, match="fraction"):
        al.junction_s(o, f, fraction=1.5)
