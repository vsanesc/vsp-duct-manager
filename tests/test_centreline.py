# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Centrelines: cubic Béziers branch + trunk, G1 at the junction, arc-length mapping."""
import math

import numpy as np
import pytest

from ductmgr.centreline import Bezier3, Centreline, CentrelineParams

START = np.array([1.0, 0.5, 1.2])
START_DIR = np.array([1.0, 0.0, -0.2]) / np.linalg.norm([1.0, 0.0, -0.2])
FACE = np.array([6.0, 0.0, 0.0])
AXIS = np.array([1.0, 0.0, 0.0])


def make(**params):
    return Centreline(START, START_DIR, FACE, AXIS, CentrelineParams(**params))


def test_a_straight_bezier_has_the_chord_length_and_arc_length_inverts():
    b = Bezier3([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    assert b.length == pytest.approx(3.0, rel=1e-6)
    assert b.points(b.t_at_length(np.array([0.0, 1.5, 3.0]))) == pytest.approx(np.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0]]), abs=1e-4)
    assert b.max_curvature() == pytest.approx(0.0, abs=1e-9)


def test_a_circle_like_bezier_has_the_right_curvature():
    k = 4.0 / 3.0 * math.tan(math.pi / 8.0)
    b = Bezier3([[1, 0, 0], [1, k, 0], [k, 1, 0], [0, 1, 0]])  # a quarter circle of radius 1
    assert b.length == pytest.approx(math.pi / 2, rel=1e-3)
    assert b.curvature(np.array([0.5]))[0] == pytest.approx(1.0, rel=1e-2)


def test_the_ends_and_tangents():
    c = make()
    assert c.branch.points([0.0])[0] == pytest.approx(START) and c.trunk.points([1.0])[0] == pytest.approx(FACE)
    assert c.branch.tangents([0.0])[0] == pytest.approx(START_DIR, abs=1e-12)  # inward along the plane normal
    assert c.trunk.tangents([1.0])[0] == pytest.approx(AXIS, abs=1e-12)  # along the engine axis
    assert c.point_at([0.0])[0] == pytest.approx(START) and c.point_at([1.0])[0] == pytest.approx(FACE)
    assert c.point_at([c.s_junction])[0] == pytest.approx(c.junction, abs=1e-6)


def test_g1_at_the_junction_and_it_lies_on_the_symmetry_plane():
    for params in ({}, {"junction": (3.0, 0.7), "junction_pitch_deg": 25.0, "junction_in_handle": 0.4,
                        "junction_out_handle": 1.9}):
        c = make(**params)
        t_in = c.branch.tangents([1.0])[0]
        t_out = c.trunk.tangents([0.0])[0]
        assert t_in == pytest.approx(t_out, abs=1e-12)  # tangent continuous with different handle lengths
        assert c.junction[1] == 0.0 and t_in[1] == 0.0  # so the mirrored branch meets the trunk G1 as well
        # the mirrored right branch ends with the mirrored tangent: the same one, because t_y = 0
        mirrored = t_in * np.array([1.0, -1.0, 1.0])
        assert mirrored == pytest.approx(t_in)


def test_defaults_and_explicit_handles():
    c = make()
    chord_b = np.linalg.norm(c.junction - START)
    chord_t = np.linalg.norm(FACE - c.junction)
    r = c.resolved
    assert r.start_handle == pytest.approx(chord_b / 3) and r.junction_out_handle == pytest.approx(chord_t / 3)
    assert r.junction == pytest.approx(((START + 0.5 * (FACE - START))[0], (START + 0.5 * (FACE - START))[2]))
    c2 = make(start_handle=0.8, junction=(3.0, 0.4), end_handle=0.3)
    assert c2.branch.ctrl[1] == pytest.approx(START + 0.8 * START_DIR)
    assert c2.trunk.ctrl[2] == pytest.approx(FACE - 0.3 * AXIS)
    assert c2.resolved.junction == (3.0, 0.4)


def test_the_default_junction_tangent_points_from_the_junction_to_the_face():
    c = make(junction=(3.0, 0.4))
    d = FACE - np.array([3.0, 0.0, 0.4])
    assert math.degrees(math.atan2(d[2], d[0])) == pytest.approx(c.resolved.junction_pitch_deg)


def test_s_is_the_arc_length_fraction_of_the_whole_path():
    c = make(junction=(3.0, 0.4))
    assert c.length == pytest.approx(c.length_branch + c.length_trunk)
    assert c.s_junction == pytest.approx(c.length_branch / c.length)
    s = np.linspace(0, 1, 11)
    pts = c.point_at(s)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.sum() == pytest.approx(c.length, rel=5e-3)
    assert steps.max() / steps.min() < 1.3  # equal arc length steps give nearly equal chords


def test_negative_handles_are_rejected_and_zero_direction_too():
    with pytest.raises(ValueError, match="negative"):
        make(start_handle=-0.1)
    with pytest.raises(ValueError, match="zero-length"):
        Centreline(START, [0, 0, 0], FACE, AXIS, CentrelineParams())


def test_offset_over_length_and_curvature():
    c = make()
    offset, length, ratio = c.offset_over_length()
    rel = START - FACE
    assert offset == pytest.approx(math.hypot(rel[1], rel[2])) and ratio == pytest.approx(offset / length)
    assert c.max_curvature() > 0.0


def test_handle_points_are_the_inner_control_points():
    c = make(junction=(3.0, 0.4))
    h = c.handle_points()
    assert h["start"] == pytest.approx(c.branch.ctrl[1]) and h["junction_in"] == pytest.approx(c.branch.ctrl[2])
    assert h["junction_out"] == pytest.approx(c.trunk.ctrl[1]) and h["end"] == pytest.approx(c.trunk.ctrl[2])
    # the three junction points are collinear (G1)
    a, b = h["junction"] - h["junction_in"], h["junction_out"] - h["junction"]
    assert np.cross(a, b) == pytest.approx(np.zeros(3), abs=1e-12)


def test_a_branch_end_offset_moves_the_end_but_keeps_the_junction_tangent():
    off = np.array([0.0, 0.12, 0.03])
    c = Centreline(START, START_DIR, FACE, AXIS, CentrelineParams(junction=(3.0, 0.4)), branch_end_offset=off)
    assert c.branch.ctrl[3] == pytest.approx(c.junction + off) and c.trunk.ctrl[0] == pytest.approx(c.junction)
    assert c.branch.tangents([1.0])[0] == pytest.approx(c.trunk.tangents([0.0])[0], abs=1e-12)  # still G1
    assert c.s_junction == pytest.approx(c.length_branch / c.length)
