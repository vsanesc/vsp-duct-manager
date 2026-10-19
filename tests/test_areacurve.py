# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The area law as two monotone PCHIP curves (branch, trunk)."""
import math

import numpy as np
import pytest

from ductmgr import areacurve as ac
from ductmgr.areacurve import AreaCurves, AreaPoints

A_IN, A_FACE = 0.11, 0.264  # m^2


def test_the_end_values_and_the_junction_continuity():
    a = AreaCurves(A_IN, A_FACE)
    assert a.branch_area(0.0) == pytest.approx(A_IN) and a.trunk_area(1.0) == pytest.approx(A_FACE)
    # trunk A(0) = n x branch A(1)
    assert a.trunk_area(0.0) == pytest.approx(2 * a.branch_area(1.0))
    a3 = AreaCurves(A_IN, A_FACE, n_branches=3)
    assert a3.trunk_area(0.0) == pytest.approx(3 * a3.branch_area(1.0))


def test_the_default_junction_area_is_the_geometric_mean_of_inlets_and_face():
    a = AreaCurves(A_IN, A_FACE)
    assert a.junction_area == pytest.approx(math.sqrt(2 * A_IN * A_FACE))
    assert a.branch_area(1.0) == pytest.approx(a.junction_area / 2)
    # equal diffusion ratio in both stages
    assert a.junction_area / (2 * A_IN) == pytest.approx(A_FACE / a.junction_area)


def test_global_area_uses_the_branch_before_and_the_trunk_after_the_junction():
    a = AreaCurves(A_IN, A_FACE)
    sj = 0.4
    s = np.array([0.0, 0.2, 0.4, 0.4000001, 0.7, 1.0])
    got = a.area(s, sj)
    assert got[0] == pytest.approx(A_IN) and got[2] == pytest.approx(a.junction_area / 2)  # the junction is the branch's end
    assert got[3] == pytest.approx(a.junction_area, rel=1e-3)  # just beyond it: the trunk's start
    assert got[1] == pytest.approx(a.branch_area(0.5)) and got[4] == pytest.approx(a.trunk_area(0.5))
    assert got[5] == pytest.approx(A_FACE)


@pytest.mark.parametrize("law", ["linear", "cosine"])
def test_the_default_interior_points_follow_the_law_and_the_curve_is_monotone(law):
    a = AreaCurves(A_IN, A_FACE, law=law)
    for curve in (a.branch_area, a.trunk_area):
        v = curve(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-12)
    if law == "linear":
        s = np.linspace(0, 1, 11)
        want = A_IN + (a.branch_area(1.0) - A_IN) * s
        assert a.branch_area(s) == pytest.approx(want)
    else:
        assert a.branch_area(0.25) < A_IN + 0.25 * (a.branch_area(1.0) - A_IN)  # slow start


def test_pchip_never_overshoots_the_data():
    pts = AreaPoints(None, ((0.3, 0.12), (0.5, 0.125), (0.7, 0.20)), None)
    a = AreaCurves(A_IN, A_FACE, pts)
    s = np.linspace(0, 1, 400)
    v = a.branch_area(s)
    knots = np.array([k[1] for k in a.branch_knots])
    assert v.min() >= knots.min() - 1e-12 and v.max() <= knots.max() + 1e-12


def test_a_dip_in_the_control_points_is_reported_as_decreasing():
    pts = AreaPoints(None, ((0.3, 0.2), (0.6, 0.09)), None)
    a = AreaCurves(A_IN, A_FACE, pts)
    dec = a.decreasing("branch")
    assert dec and all(0.0 <= lo < hi <= 1.0 for lo, hi in dec)
    assert not AreaCurves(A_IN, A_FACE).decreasing("branch")
    assert not AreaCurves(A_IN, A_FACE).decreasing("trunk")


def test_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        AreaCurves(A_IN, A_FACE, AreaPoints(None, ((1.2, 0.1),), None))
    with pytest.raises(ValueError, match="increase"):
        AreaCurves(A_IN, A_FACE, AreaPoints(None, ((0.5, 0.1), (0.4, 0.1)), None))
    with pytest.raises(ValueError, match="positive"):
        AreaCurves(A_IN, A_FACE, AreaPoints(None, ((0.5, -0.1),), None))
    with pytest.raises(ValueError, match="positive"):
        AreaCurves(0.0, A_FACE)
    with pytest.raises(ValueError, match="area law"):
        AreaCurves(A_IN, A_FACE, law="parabolic")


def test_editing_helpers_make_everything_explicit():
    a = AreaCurves(A_IN, A_FACE)
    assert not a.explicit
    pts = a.with_interior("branch", [(0.5, 0.115)])
    b = AreaCurves(A_IN, A_FACE, pts)
    assert b.explicit and b.branch_interior == ((0.5, 0.115),) and b.trunk_interior == a.trunk_interior
    c = AreaCurves(A_IN, A_FACE, a.with_junction_area(0.2))
    assert c.junction_area == 0.2 and c.branch_area(1.0) == pytest.approx(0.1)


def test_config_round_trip():
    pts = AreaPoints(0.19, ((0.4, 0.12),), None)
    data = ac.points_to_config(pts)
    assert data == {"junction_area": 0.19, "branch": [[0.4, 0.12]], "trunk": None}
    assert ac.points_from_config(data) == pts
    assert ac.points_from_config(None) == AreaPoints()
    with pytest.raises(ValueError, match="pair"):
        ac.points_from_config({"branch": [[0.4]]})
    with pytest.raises(ValueError, match="mapping"):
        ac.points_from_config([1])


def test_a_branch_ends_with_its_flow_share_of_the_junction_area():
    a = AreaCurves(A_IN, A_FACE, share=0.7, n_branches=3, a_in_total=0.4)
    assert a.junction_area == pytest.approx(math.sqrt(0.4 * A_FACE))  # from the summed inlet area of all branches
    assert a.branch_area(1.0) == pytest.approx(0.7 * a.junction_area) and a.trunk_area(0.0) == pytest.approx(a.junction_area)
    shares = [0.5, 0.3, 0.2]
    curves = [AreaCurves(A_IN, A_FACE, share=s, n_branches=3, a_in_total=0.4) for s in shares]
    assert sum(c.branch_area(1.0) for c in curves) == pytest.approx(curves[0].trunk_area(0.0))  # the ends tile the trunk
    with pytest.raises(ValueError, match="share"):
        AreaCurves(A_IN, A_FACE, share=1.5)
