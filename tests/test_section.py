# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The inlet plane frame and the section curve (plane ∩ skin) on analytic bodies."""
import math

import numpy as np
import pytest

import bodies
from ductmgr import section as sec
from ductmgr.geometry import InletPlane, mirror_vertices, plane_frame

R = 1.0


@pytest.fixture(scope="module")
def cyl():
    return bodies.cylinder(R)


@pytest.fixture(scope="module")
def prism():
    return bodies.chined_prism()


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---- the plane frame ---------------------------------------------------------------------------------
def test_zero_angles_are_the_forward_facing_x_plane():
    u, v, n = plane_frame(0.0, 0.0)
    assert n == pytest.approx([-1, 0, 0]) and v == pytest.approx([0, 0, 1]) and u == pytest.approx([0, -1, 0])
    assert np.cross(u, v) == pytest.approx(n)  # (u, v, n) is right-handed: u right, v up, n towards the viewer


@pytest.mark.parametrize("pitch, sweep", [(0, 0), (20, 0), (0, 30), (15, -25), (-40, 60), (70, -80)])
def test_frame_matches_the_rotation_definition(pitch, sweep):
    u, v, n = plane_frame(pitch, sweep)
    n_ref = rot_z(math.radians(sweep)) @ rot_y(math.radians(pitch)) @ np.array([-1.0, 0.0, 0.0])
    assert n == pytest.approx(n_ref, abs=1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v_ref = z - np.dot(z, n_ref) * n_ref
    assert v == pytest.approx(v_ref / np.linalg.norm(v_ref), abs=1e-12)
    assert u == pytest.approx(np.cross(v, n), abs=1e-12)
    m = np.array([u, v, n])
    assert m @ m.T == pytest.approx(np.eye(3), abs=1e-12)
    assert n[2] == pytest.approx(math.sin(math.radians(pitch)))  # positive pitch tilts the opening up


def test_pitch_at_90_degrees_is_rejected():
    with pytest.raises(ValueError, match="pitch"):
        plane_frame(90.0, 0.0)


@pytest.mark.parametrize("pitch, sweep", [(0, 0), (20, 30), (-15, -45)])
def test_projection_factor_and_round_trip(pitch, sweep):
    plane = InletPlane((3.0, 0.5, 0.2), pitch, sweep)
    assert plane.projection_factor == pytest.approx(math.cos(math.radians(pitch)) * math.cos(math.radians(sweep)))
    uv = np.array([[0.1, 0.2], [-0.3, 0.05], [0.0, 0.0]])
    assert plane.to_uv(plane.to_3d(uv)) == pytest.approx(uv, abs=1e-12)
    assert plane.to_3d([[0.0, 0.0]])[0] == pytest.approx([3.0, 0.5, 0.2])
    assert (plane.to_3d(uv) - plane.origin) @ plane.normal == pytest.approx(np.zeros(3), abs=1e-12)


@pytest.mark.parametrize("pitch, sweep", [(0, 0), (12, 28), (-20, -35)])
def test_the_left_plane_is_the_exact_mirror_for_non_zero_sweep(pitch, sweep):
    right = InletPlane((3.0, 0.6, 0.3), pitch, sweep)
    left = right.mirrored()
    assert left.origin == (3.0, -0.6, 0.3) and left.sweep_deg == -sweep and left.pitch_deg == pitch
    verts = [(-0.2, 0.0), (0.2, 0.0), (0.18, 0.25), (0.0, 0.34), (-0.18, 0.25)]
    r3 = right.to_3d(verts)
    mirrored_uv = mirror_vertices(verts)
    l3 = left.to_3d(mirrored_uv)
    expected = r3[::-1] * np.array([1.0, -1.0, 1.0])  # mirror about XZ, in the reversed (CCW-preserving) order
    assert l3 == pytest.approx(expected, abs=1e-12)
    ln, rn = left.normal, right.normal
    assert ln == pytest.approx(rn * np.array([1.0, -1.0, 1.0]), abs=1e-12)


def test_mirror_vertices_negates_u_and_reverses_the_order():
    verts, flags = mirror_vertices([(1, 0), (2, 1), (3, 2)], [True, False, False])
    assert verts == [(-3.0, 2.0), (-2.0, 1.0), (-1.0, 0.0)] and flags == [False, False, True]


# ---- the section ------------------------------------------------------------------------------------------
def test_zero_angles_reproduce_the_x_const_section_exactly(cyl):
    plane = InletPlane((4.0, 0.0, 1.0), 0.0, 0.0)
    s = sec.compute_section(cyl, plane)
    pts3 = plane.to_3d(s.points)
    assert pts3[:, 0] == pytest.approx(np.full(len(pts3), 4.0), abs=1e-12)  # all on x = 4
    ref = cyl.slice(normal=(1, 0, 0), origin=(4.0, 0.0, 0.0)).clean().points  # the direct x = const slice
    # same points as the direct slice (the mesh itself is float32, hence the 1e-5 comparison)
    a = np.array(sorted(map(tuple, np.round(pts3, 5))))
    b = np.array(sorted(map(tuple, np.round(np.asarray(ref, dtype=float), 5))))
    assert a.shape == b.shape and np.abs(a - b).max() < 2e-5
    assert s.closed and s.length == pytest.approx(2 * math.pi * R, rel=1e-3)


@pytest.mark.parametrize("sweep", [0.0, 30.0, -45.0])
def test_swept_plane_gives_the_analytic_ellipse_on_a_cylinder(cyl, sweep):
    o = np.array([4.0, 0.0, 1.0])
    plane = InletPlane(tuple(o), 0.0, sweep)
    s = sec.compute_section(cyl, plane)
    u_hat, v_hat, n = plane.frame()
    c = math.cos(math.radians(sweep))
    # points on the plane and on the cylinder: y = o_y - u cos(sweep), z = o_z + v  =>  y^2 + z^2 = 1
    p3 = plane.to_3d(s.points)
    assert (p3 - o) @ n == pytest.approx(np.zeros(len(p3)), abs=1e-9)
    assert np.hypot(p3[:, 1], p3[:, 2]) == pytest.approx(np.full(len(p3), R), abs=5e-4)
    # analytic upper branch v(u) = sqrt(1 - (o_y - u c)^2) - o_z
    top = s.points[s.points[:, 1] > 0.0]
    analytic = np.sqrt(1.0 - (o[1] - top[:, 0] * c) ** 2) - o[2]
    assert top[:, 1] == pytest.approx(analytic, abs=5e-4)
    assert s.points[:, 0].max() == pytest.approx(1.0 / c, rel=2e-3)  # semi-axis 1 / cos(sweep) along u


def test_pitched_plane_section_lies_on_plane_and_body(cyl):
    plane = InletPlane((4.0, 0.0, 1.0), 20.0, 0.0)
    s = sec.compute_section(cyl, plane)
    p3 = plane.to_3d(s.points)
    assert (p3 - plane.origin) @ plane.normal == pytest.approx(np.zeros(len(p3)), abs=1e-9)
    assert np.hypot(p3[:, 1], p3[:, 2]) == pytest.approx(np.full(len(p3), R), abs=5e-4)


def test_a_plane_that_misses_the_body_has_no_section(cyl):
    assert sec.compute_section(cyl, InletPlane((20.0, 0.0, 0.0))) is None


def test_the_loop_is_counter_clockwise_and_closed(cyl):
    s = sec.compute_section(cyl, InletPlane((4.0, 0.0, 1.0)))
    from ductmgr.geometry import polygon_area

    assert polygon_area(s.points) > 0 and s.closed


# ---- chines -----------------------------------------------------------------------------------------------------
def test_chine_corners_are_detected_on_the_prism(prism):
    plane = InletPlane((4.0, 0.0, 0.4), 0.0, 0.0)
    s = sec.compute_section(prism, plane)
    found = np.round(plane.to_3d(s.points[s.chines])[:, 1:], 6)  # (y, z) of the marked corners
    expected = np.round(bodies.CHINED, 6)
    assert sorted(map(tuple, found)) == sorted(map(tuple, expected))


def test_a_smooth_section_has_no_chines(cyl):
    s = sec.compute_section(cyl, InletPlane((4.0, 0.0, 1.0)))
    assert len(s.chines) == 0


def test_chine_threshold_is_30_degrees():
    assert sec.CHINE_ANGLE_DEG == 30.0
    sharp = sec.make_section(np.array([[0, 0], [1, 0], [2, math.tan(math.radians(31))]]), closed=False)
    soft = sec.make_section(np.array([[0, 0], [1, 0], [2, math.tan(math.radians(29))]]), closed=False)
    assert list(sharp.chines) == [1] and len(soft.chines) == 0


# ---- queries on the curve -----------------------------------------------------------------------------------------
def test_nearest_point_is_exact_on_the_curve_and_a_foot_point_off_it():
    s = sec.make_section(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]]), closed=False)
    p, pos, d = sec.nearest_on_section(s, (1.0, 0.0))
    assert p == pytest.approx([1.0, 0.0], abs=1e-15) and pos == pytest.approx(1.0) and d == pytest.approx(0.0, abs=1e-15)
    p, pos, d = sec.nearest_on_section(s, (1.0, 0.3))
    assert p == pytest.approx([1.0, 0.0]) and d == pytest.approx(0.3)
    p, pos, d = sec.nearest_on_section(s, (2.4, 0.5))
    assert p == pytest.approx([2.0, 0.5]) and pos == pytest.approx(2.5)


def test_the_shorter_arc_of_a_closed_curve_includes_the_chine_corners(prism):
    plane = InletPlane((4.0, 0.0, 0.4), 0.0, 0.0)
    s = sec.compute_section(prism, plane)
    uv = lambda y, z: (-y, z - 0.4)  # noqa: E731  (u = o_y - y, v = z - o_z at zero sweep)
    # two points on the sloped top sides: the short way is over the flat top, through both top chines
    a, b = uv(-0.75, 0.2), uv(0.75, 0.2)
    pa, sa, _ = sec.nearest_on_section(s, a)
    pb, sb, _ = sec.nearest_on_section(s, b)
    path, length = sec.arc_path(s, sa, sb)
    analytic = np.array([uv(-0.75, 0.2), uv(-0.5, 0.4), uv(0.5, 0.4), uv(0.75, 0.2)])
    tol = 1e-6  # the mesh is float32
    assert path[0] == pytest.approx(a, abs=tol) and path[-1] == pytest.approx(b, abs=tol)
    assert bodies.polyline_distance(path, analytic).max() < tol  # every path point is on the analytic arc
    for corner in analytic[1:3]:  # both chine corners are on the path
        assert np.linalg.norm(path - corner, axis=1).min() < tol
    seg = np.linalg.norm(np.diff(analytic, axis=0), axis=1).sum()
    assert length == pytest.approx(seg, abs=tol)
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() == pytest.approx(seg, abs=tol)
    # the reverse direction is the same arc, reversed
    back, blen = sec.arc_path(s, sb, sa)
    assert blen == pytest.approx(length) and back == pytest.approx(path[::-1], abs=1e-9)


def test_the_arc_goes_the_short_way_round_the_bottom_too(prism):
    plane = InletPlane((4.0, 0.0, 0.4), 0.0, 0.0)
    s = sec.compute_section(prism, plane)
    uv = lambda y, z: (-y, z - 0.4)  # noqa: E731
    a, b = uv(-1.0, -0.4), uv(1.0, -0.4)  # on the two vertical sides, lower half
    _, sa, _ = sec.nearest_on_section(s, a)
    _, sb, _ = sec.nearest_on_section(s, b)
    path, length = sec.arc_path(s, sa, sb)
    expected = 0.2 + 2.0 + 0.2  # down the side (0.2), across the flat bottom (2.0), up the other side (0.2)
    assert length == pytest.approx(expected, abs=1e-6)
    assert path[:, 1].min() == pytest.approx(-0.6 - 0.4, abs=1e-6)  # it passes the bottom, not the top


def test_arcs_on_an_open_curve_have_one_way():
    s = sec.make_section(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), closed=False)
    path, length = sec.arc_path(s, 0.5, 2.5)
    assert length == pytest.approx(2.0) and path[0] == pytest.approx([0.5, 0.0]) and path[-1] == pytest.approx([0.5, 1.0])
    assert [tuple(p) for p in path[1:-1]] == [(1.0, 0.0), (1.0, 1.0)]


def test_signed_distance_is_negative_inside_and_positive_outside(cyl):
    plane = InletPlane((4.0, 0.0, 1.0))
    s = sec.compute_section(cyl, plane)
    # o_z = 1: the circle is y = -u, z = 1 + v, so the centre of the body is at (u, v) = (0, -1)
    sd = sec.signed_distance(s, np.array([[0.0, -1.0], [0.0, 0.2], [0.0, -0.9], [2.0, 0.0]]))
    # the centre (-1.0 deep), 0.2 above the top, 0.1 from the centre (0.9 deep), and a point far to the side
    assert sd == pytest.approx([-R, 0.2, -0.9, math.hypot(2.0, 1.0) - R], abs=2e-3)
    assert sd[0] < 0 and sd[1] > 0 and sd[2] < 0 and sd[3] > 0


# ---- planform edge alignment -------------------------------------------------------------------------------------------
def trace_by_intersection(plane):
    """Direction (dx, dy) of plane ∩ horizontal plane through the origin. A point of the trace has p_z = o_z and
    (p - o) . n = 0, i.e. n_x dx + n_y dy = 0: the direction spans the null space of the row (n_x, n_y), found by
    SVD (independent of the n x Z shortcut used by the module, and valid when n_y = 0)."""
    n = plane.normal
    _, _, vt = np.linalg.svd(np.array([[n[0], n[1]]]))
    d = vt[1]
    assert abs(d[0] * n[0] + d[1] * n[1]) < 1e-12  # it lies in the plane
    return d / np.linalg.norm(d)


def angle_between_lines(a, b):
    """Angle in degrees between two lines (directions up to sign)."""
    c = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(min(1.0, c)))


EDGE_SWEEPS = [0.0, 5.0, 15.0, 35.0, 60.0, -20.0]
PITCHES = [0.0, 20.0, -30.0]


@pytest.mark.parametrize("pitch", PITCHES)
@pytest.mark.parametrize("edge", EDGE_SWEEPS)
def test_the_planform_trace_is_parallel_to_the_edge_within_0_01_degrees(edge, pitch):
    from ductmgr.geometry import planform_trace_direction, sweep_for_edge_angle

    plane = InletPlane((3.0, 0.6, 0.3), pitch, sweep_for_edge_angle(edge))
    edge_dir = np.array([math.sin(math.radians(edge)), math.cos(math.radians(edge))])  # swept back: x grows with y
    assert angle_between_lines(trace_by_intersection(plane), edge_dir) < 0.01
    assert angle_between_lines(np.array(planform_trace_direction(plane)), edge_dir) < 0.01
    assert plane.projection_factor == pytest.approx(math.cos(math.radians(pitch)) * math.cos(math.radians(edge)))


def test_the_sweep_is_the_negative_edge_angle():
    from ductmgr.geometry import sweep_for_edge_angle

    assert sweep_for_edge_angle(35.0) == -35.0 and sweep_for_edge_angle(-12.5) == 12.5


@pytest.mark.parametrize("pitch", PITCHES)
def test_the_right_inlets_outer_edge_trails_aft_for_a_swept_back_edge(pitch):
    from ductmgr.geometry import planform_trace_direction, sweep_for_edge_angle

    for edge in (10.0, 35.0, 60.0):
        dx, dy = planform_trace_direction(InletPlane((3.0, 0.6, 0.3), pitch, sweep_for_edge_angle(edge)))
        assert dy > 0 and dx > 0  # going outboard (+Y) the trace goes aft (+X)
        assert math.degrees(math.atan2(dx, dy)) == pytest.approx(edge, abs=1e-9)
    dx, dy = planform_trace_direction(InletPlane((3.0, 0.6, 0.3), pitch, sweep_for_edge_angle(-25.0)))
    assert dy > 0 and dx < 0  # a forward-swept edge: outboard goes forward


@pytest.mark.parametrize("pitch", PITCHES)
@pytest.mark.parametrize("edge", [10.0, 35.0, 60.0])
def test_the_left_inlet_is_aligned_with_the_mirrored_edge_exactly(edge, pitch):
    from ductmgr.geometry import sweep_for_edge_angle

    right = InletPlane((3.0, 0.6, 0.3), pitch, sweep_for_edge_angle(edge))
    left = right.mirrored()
    assert left.sweep_deg == -right.sweep_deg and left.origin == (3.0, -0.6, 0.3)
    r, l_ = trace_by_intersection(right), trace_by_intersection(left)
    # mirrored about XZ: x unchanged, y negated (directions are lines, so compare up to sign)
    assert abs(l_[0] * r[1] + l_[1] * r[0]) < 1e-12 or abs(l_[0] * r[1] - l_[1] * r[0]) < 1e-12
    mirrored_r = np.array([r[0], -r[1]])
    assert angle_between_lines(l_, mirrored_r) < 1e-9
    # the left wing's leading edge, swept back, goes outboard (-Y) and aft: direction (sin, -cos)
    left_edge = np.array([math.sin(math.radians(edge)), -math.cos(math.radians(edge))])
    assert angle_between_lines(l_, left_edge) < 0.01
    assert left.projection_factor == pytest.approx(right.projection_factor, abs=1e-15)


def test_a_steeply_pitched_plane_still_has_a_unit_planform_trace():
    from ductmgr.geometry import MAX_PITCH_DEG, planform_trace_direction

    plane = InletPlane((0.0, 0.0, 0.0), MAX_PITCH_DEG, 0.0)  # almost horizontal: the trace still exists
    dx, dy = planform_trace_direction(plane)
    assert math.hypot(dx, dy) == pytest.approx(1.0) and (dx, dy) == pytest.approx((0.0, 1.0))  # along Y at sweep 0
