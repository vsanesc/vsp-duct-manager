# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Inlet shape tests: exact area, quadrilateral case, mirror symmetry, placement, readout."""
import math

import numpy as np
import pytest

from ductmgr import geometry as g
from ductmgr.geometry import InletShape

SHAPES = [
    InletShape(),
    InletShape(width_bottom=2.0, width_top=1.0, height=1.0, apex_height=0.5, skew=0.0),
    InletShape(width_bottom=0.6, width_top=1.4, height=2.0, apex_height=0.7, skew=0.3),
    InletShape(width_bottom=1.0, width_top=0.3, height=0.5, apex_height=0.1, skew=-0.4),
    InletShape(width_bottom=1.2, width_top=1.0, height=0.9, apex_height=0.0),
]


# ---- area, vertex count, orientation ----------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("area", [0.01, 0.11, 2.5])
def test_area_is_exact_and_polygon_is_ccw_and_centred(shape, area):
    poly = g.inlet_polygon(shape, area)
    assert g.polygon_area(poly) == pytest.approx(area, rel=1e-12)  # positive => counter-clockwise
    assert g.polygon_centroid(poly) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_apex_height_zero_gives_quadrilateral():
    quad = g.inlet_polygon(InletShape(apex_height=0.0), 0.1)
    pent = g.inlet_polygon(InletShape(apex_height=0.3), 0.1)
    assert quad.shape == (4, 2) and InletShape(apex_height=0.0).is_quadrilateral
    assert pent.shape == (5, 2) and not InletShape(apex_height=0.3).is_quadrilateral


def test_square_hand_calculation():
    poly = g.inlet_polygon(InletShape(1.0, 1.0, 1.0, 0.0, 0.0), 4.0)  # unit square scaled to side 2
    expected = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    assert poly == pytest.approx(expected, abs=1e-12)


def test_pentagon_hand_calculation():
    # raw vertices (-1,0) (1,0) (.5,1) (0,1.5) (-.5,1): area 1.75, centroid y = 0.547619...
    poly = g.inlet_polygon(InletShape(2.0, 1.0, 1.0, 0.5, 0.0), 1.75)  # scale factor exactly 1
    cy = (1.5 * (4 / 9) + 0.25 * (1 + 0.5 / 3)) / 1.75
    expected = np.array([[-1, 0], [1, 0], [0.5, 1], [0, 1.5], [-0.5, 1]]) - [0.0, cy]
    assert poly == pytest.approx(expected, abs=1e-12)


def test_scaling_is_uniform_so_shape_ratios_are_preserved():
    a = g.inlet_polygon(InletShape(2.0, 1.0, 1.0, 0.5, 0.0), 0.1)
    b = g.inlet_polygon(InletShape(2.0, 1.0, 1.0, 0.5, 0.0), 0.4)
    assert b == pytest.approx(2.0 * a, abs=1e-12)  # area x4 -> lengths x2
    assert (a[1, 0] - a[0, 0]) / (a[2, 0] - a[4, 0]) == pytest.approx(2.0)  # bottom / top width


def test_skew_moves_apex_laterally():
    left = g.inlet_polygon(InletShape(1.0, 0.9, 1.0, 0.4, -0.25), 0.1)
    right = g.inlet_polygon(InletShape(1.0, 0.9, 1.0, 0.4, +0.25), 0.1)
    assert left[3, 0] < 0.0 < right[3, 0]
    assert left[3, 1] == pytest.approx(right[3, 1])


def test_shape_validation_and_dict_round_trip():
    for bad in ({"width_bottom": 0.0}, {"width_top": -1.0}, {"height": 0.0}, {"apex_height": -0.1}):
        with pytest.raises(ValueError):
            InletShape(**bad)
    with pytest.raises(ValueError):
        g.inlet_polygon(InletShape(), 0.0)
    with pytest.raises(ValueError, match="unknown"):
        InletShape.from_dict({"nonsense": 1.0})
    s = SHAPES[2]
    assert InletShape.from_dict(s.to_dict()) == s
    assert InletShape.from_dict(None) == InletShape()
    assert InletShape.from_dict({"width_top": None, "skew": 0.2}).skew == 0.2


# ---- frame and placement ---------------------------------------------------------------------------
def test_frame_at_zero_tilt_faces_the_freestream():
    u, v, n = g.inlet_frame(0.0)
    assert n == pytest.approx([-1, 0, 0]) and u == pytest.approx([0, 1, 0]) and v == pytest.approx([0, 0, 1])


@pytest.mark.parametrize("tilt", [-40.0, 0.0, 12.5, 45.0])
def test_frame_is_orthonormal_and_tilt_pitches_the_opening_up(tilt):
    u, v, n = g.inlet_frame(tilt)
    m = np.array([u, v, n])
    assert m @ m.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.cross(u, v) == pytest.approx(-n)  # (u, v, flow) is right-handed, flow = -n
    assert n[2] == pytest.approx(math.sin(math.radians(tilt)))
    assert n[1] == 0.0


@pytest.mark.parametrize("tilt", [0.0, 20.0, -30.0])
def test_placed_polygon_lies_in_the_inlet_plane_around_the_centroid(tilt):
    centroid = np.array([2.5, 0.45, 0.31])
    poly = g.inlet_polygon(SHAPES[2], 0.11)
    pts, n = g.place_polygon(poly, centroid, tilt)
    assert (pts - centroid) @ n == pytest.approx(np.zeros(len(pts)), abs=1e-12)
    assert pts.mean(axis=0) != pytest.approx(centroid)  # vertex mean is not the area centroid...
    u, v, _ = g.inlet_frame(tilt)
    local = np.c_[(pts - centroid) @ u, (pts - centroid) @ v]
    assert g.polygon_centroid(local) == pytest.approx([0.0, 0.0], abs=1e-12)  # ...the area centroid is
    assert g.polygon_area(local) == pytest.approx(0.11, rel=1e-12)


# ---- mirror ------------------------------------------------------------------------------------------
def test_mirror_is_exact_and_involutive():
    right, n = g.place_polygon(g.inlet_polygon(SHAPES[2], 0.11), [2.5, 0.45, 0.31], 17.0)
    left = g.mirror_xz(right, reverse=True)
    assert np.array_equal(left[::-1, 0], right[:, 0])  # x unchanged
    assert np.array_equal(left[::-1, 1], -right[:, 1])  # y negated exactly
    assert np.array_equal(left[::-1, 2], right[:, 2])  # z unchanged
    assert np.array_equal(g.mirror_xz(g.mirror_xz(right)), right)
    assert np.array_equal(g.mirror_xz(left, reverse=True), right)
    assert g.mirror_normal(n)[0] == n[0] and g.mirror_normal(n)[1] == -n[1] and g.mirror_normal(n)[2] == n[2]


def test_mirrored_polygon_keeps_its_area_and_orientation():
    right, _ = g.place_polygon(g.inlet_polygon(SHAPES[1], 0.2), [2.0, 0.5, 0.3], 0.0)
    left = g.mirror_xz(right, reverse=True)
    u, v, _ = g.inlet_frame(0.0)
    # mirroring flips handedness; reversing the vertex order restores counter-clockwise in the same
    # global (+Y, +Z) axes that the right inlet is counter-clockwise in
    for pts in (right, left):
        c = pts.mean(axis=0)
        assert g.polygon_area(np.c_[(pts - c) @ u, (pts - c) @ v]) == pytest.approx(0.2, rel=1e-12)


# ---- readout ---------------------------------------------------------------------------------------------
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
    poly = g.inlet_polygon(InletShape(), 0.1)
    r = g.inlet_readout([5.0, 0.5, 0.5], [3.0, 0.0, 0.0], poly, 0.1)
    assert math.isnan(r.offset_over_length) and r.length == pytest.approx(-2.0)


def test_aspect_ratio_of_a_rectangle():
    assert g.aspect_ratio(np.array([[0, 0], [3, 0], [3, 1.5], [0, 1.5]], dtype=float)) == pytest.approx(2.0)
