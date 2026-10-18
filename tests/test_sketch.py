# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The 5-vertex opening: snapping, snapped edges, areas, Fit area, validity, defaults, mirroring."""
import math

import numpy as np
import pytest

import bodies
from ductmgr import section as sec
from ductmgr import sketch as sk
from ductmgr.geometry import InletPlane, mirror_vertices, polygon_area

TOL = 1e-6  # the test meshes are float32


@pytest.fixture(scope="module")
def cyl():
    return bodies.cylinder(1.0)


@pytest.fixture(scope="module")
def prism():
    return bodies.chined_prism()


@pytest.fixture(scope="module")
def taper():
    return bodies.tapered_chined_body()


def shoelace_yz(points3d):
    """Frontal (projected normal to the freestream) area of a closed 3D outline, written out independently."""
    y, z = points3d[:, 1], points3d[:, 2]
    return 0.5 * abs(np.dot(y, np.roll(z, -1)) - np.dot(np.roll(y, -1), z))


def prism_uv(y, z):
    """(u, v) of a prism (y, z) point in the plane x = 4 through (4, 0, 0.4) at zero angles: u = -y, v = z - 0.4."""
    return (-y, z - 0.4)


PRISM_PLANE = InletPlane((4.0, 0.0, 0.4))
CYL_PLANE = InletPlane((4.0, 0.0, 1.0))


# ---- snapping ----------------------------------------------------------------------------------------------
def test_a_dropped_vertex_snaps_onto_the_curve_within_1e_6(cyl):
    s = sec.compute_section(cyl, CYL_PLANE)
    rng = np.random.default_rng(1)
    for _ in range(30):
        raw = (rng.uniform(-0.7, 0.7), rng.uniform(-0.25, 0.05))
        near = sec.nearest_on_section(s, raw)[2]
        uv, snapped = sk.snap_point(s, raw, tolerance=0.3)
        assert snapped == (near <= 0.3)
        if snapped:
            assert sec.distance_to_section(s, uv)[0] < 1e-6  # on the polyline


def test_snapping_respects_the_tolerance_and_can_be_disabled(cyl):
    s = sec.compute_section(cyl, CYL_PLANE)
    on_top = (0.0, 0.01)  # 1 cm above the skin at the top of the circle
    assert sk.snap_point(s, on_top, 0.02)[1] and not sk.snap_point(s, on_top, 0.005)[1]
    uv, snapped = sk.snap_point(s, on_top, None)  # Alt held: snapping off
    assert not snapped and uv == on_top
    assert sk.snap_point(None, on_top, 1.0) == (on_top, False)  # no curve, nothing to snap to


def test_snapped_vertices_follow_a_changed_section(taper):
    p1, p2 = InletPlane((3.0, 0.0, 0.6)), InletPlane((5.0, 0.0, 0.6))
    s1, s2 = sec.compute_section(taper, p1), sec.compute_section(taper, p2)
    a = sk.snap_point(s1, (-0.2, 0.0), 1.0)[0]
    b = sk.snap_point(s1, (0.2, 0.0), 1.0)[0]
    free = (0.0, 0.5)
    verts, flags = sk.resnap([a, b, free], [True, True, False], s2)
    assert flags == (True, True, False) and verts[2] == free  # the free vertex stays where it is
    for p in verts[:2]:
        assert sec.distance_to_section(s2, p)[0] < 1e-6
    verts, flags = sk.resnap([a, b, free], [True, True, False], None)  # no curve: nothing can stay snapped
    assert flags == (False, False, False)


# ---- snapped edges follow the section ---------------------------------------------------------------------------
def test_a_snapped_edge_equals_the_section_arc_between_its_vertices_including_chines(prism):
    s = sec.compute_section(prism, PRISM_PLANE)
    A, B = prism_uv(0.75, 0.2), prism_uv(-0.75, 0.2)  # on the two sloped top sides
    C, D, E = prism_uv(-0.3, 0.9), prism_uv(0.0, 1.1), prism_uv(0.3, 0.9)
    pa, pb = sk.snap_point(s, A, 1.0)[0], sk.snap_point(s, B, 1.0)[0]
    verts = (pa, pb, C, D, E)
    paths, flags, lengths = sk.edge_paths(verts, (True, True, False, False, False), s)
    assert flags == [True, False, False, False, False]
    analytic = np.array([A, prism_uv(0.5, 0.4), prism_uv(-0.5, 0.4), B])  # over the top, through both chines
    path = paths[0]
    assert path[0] == pytest.approx(pa) and path[-1] == pytest.approx(pb)
    assert bodies.polyline_distance(path, analytic).max() < TOL
    for corner in analytic[1:3]:
        assert np.linalg.norm(path - corner, axis=1).min() < TOL  # the chine corners are on the edge
    expected = np.linalg.norm(np.diff(analytic, axis=0), axis=1).sum()
    assert lengths[0] == pytest.approx(expected, abs=TOL)


def test_free_edges_are_straight_lines(prism):
    s = sec.compute_section(prism, PRISM_PLANE)
    verts = (prism_uv(0.75, 0.2), prism_uv(-0.75, 0.2), (0.3, 0.5), (0.0, 0.7), (-0.3, 0.5))
    paths, flags, _ = sk.edge_paths(verts, (True, True, False, False, False), s)
    for i in (1, 2, 3, 4):
        assert len(paths[i]) == 2 and not flags[i]
        assert paths[i][0] == pytest.approx(verts[i]) and paths[i][1] == pytest.approx(verts[(i + 1) % 5])
    # an edge with only ONE snapped end is free as well
    paths, flags, _ = sk.edge_paths(verts, (True, False, False, False, False), s)
    assert flags == [False] * 5


def test_the_outline_is_the_concatenation_of_the_edge_paths(prism):
    s = sec.compute_section(prism, PRISM_PLANE)
    verts = (prism_uv(0.75, 0.2), prism_uv(-0.75, 0.2), (0.3, 0.5), (0.0, 0.7), (-0.3, 0.5))
    paths, _, _ = sk.edge_paths(verts, (True, True, False, False, False), s)
    outline = sk.build_outline(paths)
    assert len(outline) == sum(len(p) - 1 for p in paths)
    assert outline[0] == pytest.approx(verts[0])
    assert abs(polygon_area(outline)) > 0


# ---- areas ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pitch, sweep", [(0.0, 0.0), (10.0, 20.0), (-15.0, 35.0)])
def test_projected_area_is_the_in_plane_area_times_n_dot_x(cyl, pitch, sweep):
    plane = InletPlane((4.0, 0.0, 1.0), pitch, sweep)
    s = sec.compute_section(cyl, plane)
    verts, snapped = sk.default_sketch(s, plane, 0.1)
    op = sk.evaluate(verts, snapped, s, plane, 0.1)
    assert op.area_plane == pytest.approx(abs(polygon_area(op.outline)), rel=1e-12)
    assert op.projection_factor == pytest.approx(math.cos(math.radians(pitch)) * math.cos(math.radians(sweep)))
    assert op.area_projected == pytest.approx(op.area_plane * op.projection_factor, rel=1e-12)
    # independent check: the frontal (y, z) shoelace of the 3D outline
    assert shoelace_yz(plane.to_3d(op.outline)) == pytest.approx(op.area_projected, rel=1e-9)
    assert op.delta == pytest.approx(op.area_projected / 0.1 - 1.0)


def test_a_steeply_swept_opening_is_flagged_as_far_off_the_flow(cyl):
    plane = InletPlane((4.0, 0.0, 1.0), 0.0, 70.0)  # |n.X| = cos 70 = 0.34 < 0.5
    s = sec.compute_section(cyl, plane)
    verts, snapped = sk.default_sketch(s, plane, 0.05)
    op = sk.evaluate(verts, snapped, s, plane, 0.05)
    assert any("off the flow" in w and "0.34" in w for w in op.warnings)
    good = sk.evaluate(*sk.default_sketch(sec.compute_section(cyl, CYL_PLANE), CYL_PLANE, 0.05),
                       sec.compute_section(cyl, CYL_PLANE), CYL_PLANE, 0.05)
    assert not any("off the flow" in w for w in good.warnings)


def test_snapped_edge_length_and_clearance_readouts(prism):
    s = sec.compute_section(prism, PRISM_PLANE)
    A, B = prism_uv(0.75, 0.2), prism_uv(-0.75, 0.2)
    pa, pb = sk.snap_point(s, A, 1.0)[0], sk.snap_point(s, B, 1.0)[0]
    verts = (pa, pb, (0.6, 0.5), (0.0, 0.8), (-0.6, 0.5))
    op = sk.evaluate(verts, (True, True, False, False, False), s, PRISM_PLANE, 0.2)
    analytic = np.array([A, prism_uv(0.5, 0.4), prism_uv(-0.5, 0.4), B])
    assert op.snapped_length == pytest.approx(np.linalg.norm(np.diff(analytic, axis=0), axis=1).sum(), abs=TOL)
    # the free side edges leave the snapped vertices on the slopes and stay just above the skin near the chines
    assert op.clearance is not None and 0.0 < op.clearance < 0.1
    assert op.snapped_edges == (True, False, False, False, False)


# ---- Fit area ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pitch, sweep", [(0.0, 0.0), (15.0, 25.0)])
@pytest.mark.parametrize("target", [0.05, 0.12, 0.3])
def test_fit_area_scales_only_the_free_vertices_and_hits_the_target(cyl, pitch, sweep, target):
    plane = InletPlane((4.0, 0.0, 1.0), pitch, sweep)
    s = sec.compute_section(cyl, plane)
    base, snapped = sk.default_sketch(s, plane, 0.08)
    verts = [tuple(v) for v in base]
    for i in (2, 3, 4):  # push the free vertices around so the start is not already fitted
        verts[i] = (verts[i][0] * 1.4, verts[i][1] * 1.4 + 0.02)
    verts = tuple(verts)
    new, ok, message = sk.fit_area(verts, snapped, s, plane, target)
    assert ok and "Area fitted" in message
    for i in (0, 1):  # snapped vertices are untouched, bit for bit
        assert new[i] == verts[i]
    centre = sk.snapped_edge_centroid(verts, snapped, s)
    factors = [np.dot(np.array(new[i]) - centre, np.array(verts[i]) - centre) / np.dot(np.array(verts[i]) - centre, np.array(verts[i]) - centre)
               for i in (2, 3, 4)]
    assert factors == pytest.approx([factors[0]] * 3, rel=1e-9)  # one common scale about the snapped-edge centre
    for i in (2, 3, 4):
        assert np.array(new[i]) == pytest.approx(centre + factors[0] * (np.array(verts[i]) - centre), abs=1e-12)
    op = sk.evaluate(new, snapped, s, plane, target)
    assert abs(op.delta) < sk.AREA_TOLERANCE and abs(shoelace_yz(plane.to_3d(op.outline)) / target - 1) < sk.AREA_TOLERANCE


def test_the_fit_centre_is_the_centroid_of_the_snapped_edge(prism):
    s = sec.compute_section(prism, PRISM_PLANE)
    pa, pb = sk.snap_point(s, prism_uv(0.75, 0.2), 1.0)[0], sk.snap_point(s, prism_uv(-0.75, 0.2), 1.0)[0]
    verts = (pa, pb, (0.3, 0.5), (0.0, 0.7), (-0.3, 0.5))
    c = sk.snapped_edge_centroid(verts, (True, True, False, False, False), s)
    # the arc A -> top chines -> B is symmetric about u = 0 and has its length-weighted centre on it
    # the arc runs from the sloped sides (v = -0.2) up to the flat top (v = 0): its centre is just below the top
    assert c[0] == pytest.approx(0.0, abs=1e-6) and c[1] == pytest.approx(-0.1 * 2 * math.hypot(0.25, 0.2) / (2 * math.hypot(0.25, 0.2) + 1.0), abs=1e-6)
    free_only = sk.snapped_edge_centroid(verts, (False,) * 5, s)
    assert free_only == pytest.approx(np.mean(verts, axis=0))


def test_fit_area_with_nothing_free_or_an_unreachable_target(cyl):
    plane = CYL_PLANE
    s = sec.compute_section(cyl, plane)
    verts, snapped = sk.default_sketch(s, plane, 0.1)
    same, ok, msg = sk.fit_area(verts, (True,) * 5, s, plane, 0.1)
    assert not ok and "nothing to scale" in msg and same == tuple(verts)
    _, ok, msg = sk.fit_area(verts, snapped, s, plane, 500.0)  # far beyond the scale search range
    assert not ok and "could not reach" in msg


def test_fit_area_works_with_no_snapped_vertex_at_all(cyl):
    plane = CYL_PLANE
    s = sec.compute_section(cyl, plane)
    verts = ((-0.3, 0.1), (0.3, 0.1), (0.3, 0.5), (0.0, 0.7), (-0.3, 0.5))
    new, ok, _ = sk.fit_area(verts, (False,) * 5, s, plane, 0.5)
    assert ok
    assert sk.evaluate(new, (False,) * 5, s, plane, 0.5).delta == pytest.approx(0.0, abs=sk.AREA_TOLERANCE)


# ---- validity -----------------------------------------------------------------------------------------------------------------------
def test_an_opening_on_the_skin_is_valid(cyl, prism):
    for body, plane in ((cyl, CYL_PLANE), (prism, PRISM_PLANE)):
        s = sec.compute_section(body, plane)
        verts, snapped = sk.default_sketch(s, plane, 0.06)
        op = sk.evaluate(verts, snapped, s, plane, 0.06)
        assert op.valid and not op.errors and op.penetration <= sk.INSIDE_TOL
        assert op.clearance is not None and op.clearance > 0.0


def test_an_opening_that_crosses_the_body_is_invalid(cyl):
    s = sec.compute_section(cyl, CYL_PLANE)
    # the free bottom edge at v = -0.3 lies inside the circle (whose top is at v = 0)
    verts = ((-0.5, -0.3), (0.5, -0.3), (0.5, 0.3), (0.0, 0.5), (-0.5, 0.3))
    op = sk.evaluate(verts, (False,) * 5, s, CYL_PLANE, 0.2)
    assert not op.valid and any("inside the fuselage" in e for e in op.errors)
    assert op.clearance < 0 and op.penetration > 0.1
    # the same pentagon lifted above the skin is fine
    lifted = tuple((u, v + 0.5) for u, v in verts)
    assert sk.evaluate(lifted, (False,) * 5, s, CYL_PLANE, 0.2).valid


def test_a_pentagon_surrounding_the_body_is_invalid_even_though_its_edges_are_outside(cyl):
    s = sec.compute_section(cyl, CYL_PLANE)
    # centre of the body is at (0, -1) in (u, v); a big pentagon around it touches the body nowhere but encloses it
    verts = ((-3.0, -4.0), (3.0, -4.0), (3.0, 2.0), (0.0, 3.0), (-3.0, 2.0))
    op = sk.evaluate(verts, (False,) * 5, s, CYL_PLANE, 1.0)
    assert not op.valid and op.penetration > 0.5


def test_a_self_crossing_outline_is_invalid(cyl):
    s = sec.compute_section(cyl, CYL_PLANE)
    bow_tie = ((-0.3, 0.2), (0.3, 0.6), (0.3, 0.2), (-0.3, 0.6), (0.0, 0.8))
    op = sk.evaluate(bow_tie, (False,) * 5, s, CYL_PLANE, 0.1)
    assert not op.valid and any("crosses itself" in e for e in op.errors)


def test_a_plane_that_misses_the_fuselage_is_invalid(cyl):
    plane = InletPlane((20.0, 0.0, 0.0))
    assert sec.compute_section(cyl, plane) is None
    verts, snapped = sk.default_sketch(None, plane, 0.1)
    assert snapped == (False,) * 5
    op = sk.evaluate(verts, snapped, None, plane, 0.1)
    assert not op.valid and any("does not cut" in e for e in op.errors)
    assert op.clearance is None


# ---- defaults ---------------------------------------------------------------------------------------------------------------------------
def test_the_default_sketch_has_a_snapped_bottom_edge_and_the_target_area(prism, cyl):
    for body, plane in ((prism, PRISM_PLANE), (cyl, CYL_PLANE)):
        s = sec.compute_section(body, plane)
        verts, snapped = sk.default_sketch(s, plane, 0.09)
        assert snapped == (True, True, False, False, False) and len(verts) == sk.N_VERTICES
        for p in verts[:2]:
            assert sec.distance_to_section(s, p)[0] < 1e-6
        op = sk.evaluate(verts, snapped, s, plane, 0.09)
        assert abs(op.delta) < sk.AREA_TOLERANCE and op.valid
        assert polygon_area(op.outline) > 0  # counter-clockwise in (u, v)
        assert verts[0][0] < 0 < verts[1][0]  # bottom-left, bottom-right
        assert all(v[1] > max(verts[0][1], verts[1][1]) for v in verts[2:])  # the rest is above the bottom edge


def test_the_default_sketch_without_a_curve_is_unsnapped_but_sized():
    plane = InletPlane((0.0, 0.0, 0.0))
    verts, snapped = sk.default_sketch(None, plane, 0.1)
    assert snapped == (False,) * 5
    area = abs(polygon_area(np.array(verts)))
    assert area * plane.projection_factor == pytest.approx(0.1, rel=sk.AREA_TOLERANCE)


# ---- mirror symmetry ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pitch, sweep", [(0.0, 0.0), (12.0, 28.0), (-10.0, -35.0)])
def test_the_left_opening_is_the_exact_mirror_image_for_free_vertices(pitch, sweep):
    right = InletPlane((3.0, 0.6, 0.3), pitch, sweep)
    verts = ((-0.2, 0.0), (0.2, 0.0), (0.18, 0.25), (0.0, 0.34), (-0.18, 0.25))
    mv = mirror_vertices(verts)
    r_out = sk.build_outline(sk.edge_paths(verts, (False,) * 5, None)[0])
    l_out = sk.build_outline(sk.edge_paths(tuple(mv), (False,) * 5, None)[0])
    r3 = right.to_3d(r_out)
    l3 = right.mirrored().to_3d(l_out)
    mirrored_r = r3 * np.array([1.0, -1.0, 1.0])
    # same closed outline, traversed in the opposite (CCW-preserving) direction
    a = sorted(map(tuple, np.round(l3, 12)))
    b = sorted(map(tuple, np.round(mirrored_r, 12)))
    assert np.abs(np.array(a) - np.array(b)).max() < 1e-12
    assert shoelace_yz(l3) == pytest.approx(shoelace_yz(r3), rel=1e-12)


def test_the_mirrored_sketch_snaps_to_the_mirrored_section(prism):
    right = InletPlane((4.0, 0.0, 0.4), 0.0, 0.0)
    s = sec.compute_section(prism, right)
    verts, snapped = sk.default_sketch(s, right, 0.09)
    mv, mflags = mirror_vertices(verts, snapped)
    left = right.mirrored()  # origin y = 0 on the symmetry plane: the same plane, negated sweep = 0
    sl = sec.compute_section(prism, left)
    op_r = sk.evaluate(verts, snapped, s, right, 0.09)
    op_l = sk.evaluate(tuple(mv), tuple(mflags), sl, left, 0.09)
    assert op_l.area_projected == pytest.approx(op_r.area_projected, abs=1e-6)
    assert op_l.snapped_length == pytest.approx(op_r.snapped_length, abs=1e-6)
    assert op_l.valid
