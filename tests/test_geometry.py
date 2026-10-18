# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""2D polygon helpers and the inlet readout (the plane frame itself is tested in test_section.py)."""
import math

import numpy as np
import pytest

from ductmgr import geometry as g


def test_polygon_area_is_signed_and_exact():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    assert g.polygon_area(square) == pytest.approx(4.0)  # counter-clockwise: positive
    assert g.polygon_area(square[::-1]) == pytest.approx(-4.0)
    assert g.polygon_area(square + [10.0, -3.0]) == pytest.approx(4.0)  # translation invariant


def test_polygon_centroid_of_a_pentagon_by_hand():
    # trapezoid (bottom 2, top 1, height 1) plus a triangle on top (base 1, height 0.5): area 1.75
    poly = np.array([[-1, 0], [1, 0], [0.5, 1], [0, 1.5], [-0.5, 1]], dtype=float)
    assert g.polygon_area(poly) == pytest.approx(1.75)
    cy = (1.5 * (4 / 9) + 0.25 * (1 + 0.5 / 3)) / 1.75
    assert g.polygon_centroid(poly) == pytest.approx([0.0, cy], abs=1e-12)


def test_aspect_ratio_of_a_rectangle():
    assert g.aspect_ratio(np.array([[0, 0], [3, 0], [3, 1.5], [0, 1.5]], dtype=float)) == pytest.approx(2.0)


def test_readout_hand_calculation():
    poly = np.array([[-1.0, 0.0], [1.0, 0.0], [1.0, 2.0], [-1.0, 2.0]])  # 2 wide, 2 high
    r = g.inlet_readout([2.0, 1.0, 3.0], [5.0, 0.0, 1.0], poly, 0.11)
    assert r.dy == pytest.approx(-1.0) and r.dz == pytest.approx(-2.0)
    assert r.length == pytest.approx(3.0)
    assert r.offset == pytest.approx(math.sqrt(5.0))
    assert r.distance_to_face == pytest.approx(math.sqrt(14.0))
    assert r.offset_over_length == pytest.approx(math.sqrt(5.0) / 3.0)
    assert r.aspect_ratio == pytest.approx(1.0)
    assert r.capture_area == 0.11 and r.centroid == (2.0, 1.0, 3.0)


def test_readout_when_face_is_not_aft_of_the_inlet():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    r = g.inlet_readout([5.0, 0.5, 0.5], [3.0, 0.0, 0.0], poly, 0.1)
    assert math.isnan(r.offset_over_length) and r.length == pytest.approx(-2.0)
