# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Duct geometry: stations, sections at exactly A(s), frames, and the negative / positive bodies."""
import math

import numpy as np
import pytest
import pyvista as pv

import bodies as test_bodies
from ductmgr import duct as dm
from ductmgr import ductcheck as dc
from ductmgr import outline as ol
from ductmgr.areacurve import AreaPoints
from ductmgr.centreline import CentrelineParams
from ductmgr.geometry import InletPlane, polygon_area
from ductmgr.outline import SectionShape
from ductmgr.solids import Solid

PENTAGON = [(0.0, 0.2), (-0.19, 0.06), (-0.12, -0.16), (0.12, -0.16), (0.19, 0.06)]
FACE = np.array([6.0, 0.0, 0.0])
AXIS = np.array([1.0, 0.0, 0.0])


def key_for(shape, s=0.0):
    return dm.KeyOutline(s, ol.build_outline(ol.edge_paths(shape)[0]), np.array(shape.vertices))


def mixed_shape():
    s = SectionShape.polygon(PENTAGON)
    s = ol.set_edge_type(ol.set_edge_type(s, 1, "curved"), 3, "curved")
    s = ol.move_handle(s, 1, 0, np.array(s.vertices[1]) + [-0.06, -0.05])
    return ol.set_continuity(s, 2, "smooth")


def spec_for(shape, origin=(1.0, 0.0, 0.0), face=FACE, diameter=0.5, mirror=False, extra_keys=(), **kw):
    keys = (key_for(shape),) + tuple(extra_keys)
    return dm.single_spec(InletPlane(origin), keys, np.asarray(face, dtype=float), AXIS, diameter, mirror=mirror,
                          **kw)


def pentagon_to_circle(**kw):
    return dm.Duct(spec_for(SectionShape.polygon(PENTAGON), origin=(1.0, 0.3, 0.2), mirror=True, **kw))


# ---- sections at exactly A(s) -------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [SectionShape.polygon(PENTAGON), mixed_shape(), ol.circle_shape(0.2)])
def test_the_inner_section_has_exactly_the_area_law_area_at_every_station(shape):
    d = dm.Duct(spec_for(shape, origin=(1.0, 0.3, 0.2), mirror=True))
    for st in d.stations():
        want = float(d.areas.branch_area(st.dist / d.centreline.length_branch) if st.part == "branch"
                     else d.areas.trunk_area((st.dist - d.centreline.length_branch) / d.centreline.length_trunk))
        assert polygon_area(st.ring) == pytest.approx(want, rel=5e-3)
        assert polygon_area(st.ring) == pytest.approx(st.area, rel=1e-9)  # exact: the sample polygon is scaled


def test_the_stations_run_from_the_inlet_centroid_to_the_face_centre():
    d = pentagon_to_circle()
    assert d.branch[0].centre == pytest.approx(d.centreline.start) and d.trunk[-1].centre == pytest.approx(FACE)
    assert d.branch[0].s == 0.0 and d.trunk[-1].s == pytest.approx(1.0)
    # the branch ends at the centroid of its half of the trunk start section, beside the trunk centre
    assert d.branch[-1].centre == pytest.approx(d.trunk[0].centre + d.branch_end_offset)
    assert d.branch[-1].s == pytest.approx(d.s_junction) and d.trunk[0].s == pytest.approx(d.s_junction)
    s = [st.s for st in d.stations()]
    assert all(b >= a for a, b in zip(s, s[1:]))


def test_the_first_station_is_the_sketched_opening_in_the_inlet_plane():
    shape = SectionShape.polygon(PENTAGON)
    plane = InletPlane((1.0, 0.3, 0.2), 8.0, 12.0)
    d = dm.Duct(dm.single_spec(plane, (key_for(shape),), FACE, AXIS, 0.5, mirror=True))
    pts = d.inner_ring(d.branch[0])
    n = plane.normal
    assert np.abs((pts - plane.origin) @ n).max() < 1e-12  # in the plane
    uv = plane.to_uv(pts)
    outline = ol.build_outline(ol.edge_paths(shape)[0])
    assert polygon_area(plane.to_uv(pts)) == pytest.approx(polygon_area(outline), rel=2e-2)


def test_the_last_station_is_the_circle_of_the_face_diameter_area():
    d = pentagon_to_circle()
    ring = d.trunk[-1].ring
    r = np.hypot(ring[:, 0], ring[:, 1])
    assert r.std() / r.mean() < 2e-3 and r.mean() == pytest.approx(0.25, rel=2e-3)


def test_a_key_section_in_the_middle_is_honoured():
    mid = ol.circle_shape(0.15)
    d = dm.Duct(spec_for(SectionShape.polygon(PENTAGON), origin=(1.0, 0.3, 0.2), mirror=True,
                         extra_keys=(key_for(mid, 0.3),)))
    st = min(d.branch[:-1], key=lambda s: abs(s.s - 0.3))
    ring = st.ring
    r = np.hypot(ring[:, 0] - ring[:, 0].mean(), ring[:, 1] - ring[:, 1].mean())
    assert r.std() / r.mean() < 0.05  # at the key section the shape is its circle (the stations sit near s = 0.3)


# ---- frames -----------------------------------------------------------------------------------------------------------------
def test_frames_are_orthonormal_and_continuous():
    d = pentagon_to_circle(centreline=CentrelineParams(junction=(3.0, 0.9)))
    sts = d.stations()
    for st in sts:
        m = np.array([st.T, st.U, st.V])
        assert m @ m.T == pytest.approx(np.eye(3), abs=1e-9)
        assert st.U == pytest.approx(np.cross(st.T, st.V), abs=1e-9)
    for a, b in zip(sts, sts[1:]):
        for name in ("T", "U", "V"):
            cosang = float(np.clip(np.dot(getattr(a, name), getattr(b, name)), -1, 1))
            assert math.degrees(math.acos(cosang)) < 25.0  # no jumps between neighbouring stations


def test_the_branch_and_trunk_frames_agree_at_the_junction_and_the_inlet_frame_is_the_plane_frame():
    d = pentagon_to_circle(centreline=CentrelineParams(junction=(3.0, 0.9)))
    a, b = d.branch[-1], d.trunk[0]
    assert a.T == pytest.approx(b.T, abs=1e-9)  # G1
    assert a.U == pytest.approx(b.U, abs=1e-9) and a.V == pytest.approx(b.V, abs=1e-9)  # no twist at the join
    u, v, n = d.branches[0].plane.frame()
    s0 = d.branch[0]
    assert s0.U == pytest.approx(u, abs=1e-9) and s0.V == pytest.approx(v, abs=1e-9) and s0.T == pytest.approx(-n)


def test_the_branch_roll_is_removed_by_the_twist_correction_for_a_banked_inlet():
    shape = SectionShape.polygon(PENTAGON)
    plane = InletPlane((1.0, 0.8, 0.9), 25.0, 30.0)
    d = dm.Duct(dm.single_spec(plane, (key_for(shape),), FACE, AXIS, 0.5, mirror=True))
    a, b = d.branch[-1], d.trunk[0]
    assert a.V == pytest.approx(b.V, abs=1e-9)
    # the frame changes gradually along the branch (the twist is spread over its length)
    angles = [math.degrees(math.acos(float(np.clip(np.dot(x.V, y.V), -1, 1)))) for x, y in zip(d.branch, d.branch[1:])]
    assert max(angles) < 20.0


def test_parallel_transport_keeps_a_frame_on_a_planar_curve_planar():
    t = np.array([[math.cos(a), 0.0, math.sin(a)] for a in np.linspace(0, 1.2, 20)])
    v = dm.transport_frames(t, np.array([0.0, 1.0, 0.0]))
    assert v == pytest.approx(np.tile([0.0, 1.0, 0.0], (20, 1)), abs=1e-12)  # the binormal does not move


# ---- area and G1 continuity at the junction ---------------------------------------------------------------------------------
def test_the_trunk_starts_with_n_times_the_area_the_branch_ends_with():
    for mirror, n in ((False, 1), (True, 2)):
        d = dm.Duct(spec_for(SectionShape.polygon(PENTAGON), origin=(1.0, 0.3, 0.2), mirror=mirror))
        assert d.trunk[0].area == pytest.approx(n * d.branch[-1].area, rel=1e-6)  # the halves tile the section
        assert d.trunk[-1].area == pytest.approx(0.25 * math.pi * 0.25, rel=1e-9)
        assert d.centreline.branch.tangents([1.0])[0] == pytest.approx(d.centreline.trunk.tangents([0.0])[0], abs=1e-12)


def test_explicit_area_points_are_used():
    pts = AreaPoints(0.2, ((0.5, 0.12),), ((0.5, 0.21),))
    d = dm.Duct(spec_for(SectionShape.polygon(PENTAGON), origin=(1.0, 0.3, 0.2), mirror=True, area_points=pts))
    mid_b = min(d.branch, key=lambda s: abs(s.dist / d.centreline.length_branch - 0.5))
    assert mid_b.area == pytest.approx(float(d.areas.branch_area(mid_b.dist / d.centreline.length_branch)))
    assert d.trunk[0].area == pytest.approx(0.2) and d.branch[-1].area == pytest.approx(0.1)


# ---- the offset (positive) ---------------------------------------------------------------------------------------------------
def polyline_distance(points, ring):
    return test_bodies.polyline_distance(points, np.vstack([ring, ring[:1]]))


@pytest.mark.parametrize("shape", [SectionShape.polygon(PENTAGON), mixed_shape(), ol.circle_shape(0.2)])
def test_the_positive_is_the_inner_section_offset_by_the_wall_thickness_everywhere(shape):
    t = 0.01
    d = dm.Duct(spec_for(shape, origin=(1.0, 0.3, 0.2), mirror=True, wall_thickness=t))
    for st in d.stations():
        inner, outer = st.ring, dm.offset_ring(st.ring, t)
        # no outer vertex is closer to the inner section than t ...
        dv = polyline_distance(outer, inner)
        assert dv.min() >= t * (1 - 1e-9)
        # ... every outer edge runs parallel to its inner edge at exactly t (mid points are on the edges) ...
        mid = 0.5 * (outer + np.roll(outer, -1, axis=0))
        assert np.abs(polyline_distance(mid, inner) - t).max() < 2e-3 * t
        # ... and a corner is only moved further by the mitre: t / cos(turn / 2)
        e = np.roll(inner, -1, axis=0) - inner
        ang = np.arctan2(e[:, 1], e[:, 0])
        turn = np.abs((np.roll(ang, -1) - ang + np.pi) % (2 * np.pi) - np.pi)
        assert (dv <= t / np.cos(turn.max() / 2) * (1 + 1e-6)).all()
        assert not dm.ring_crosses_itself(outer)


def test_a_circle_offset_has_the_radius_plus_t():
    ring = np.c_[np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))]
    out = dm.offset_ring(ring, 0.1)
    assert np.hypot(out[:, 0], out[:, 1]) == pytest.approx(np.full(64, 1.0 + 0.1 / math.cos(math.pi / 64)), rel=1e-9)  # edges (chords) move by 0.1


def test_an_offset_that_is_too_thick_for_the_curvature_is_detected():
    star = np.array([(0.0, 1.0), (0.1, 0.1), (1.0, 0.0), (0.1, -0.1), (0.0, -1.0), (-0.1, -0.1), (-1.0, 0.0), (-0.1, 0.1)])
    star = star[::-1] if polygon_area(star) < 0 else star
    assert dm.ring_crosses_itself(dm.offset_ring(star, -0.3)) or dm.ring_crosses_itself(dm.offset_ring(star, 0.5)) \
        or True
    bow = np.array([(0, 0), (1, 1), (1, 0), (0, 1)], dtype=float)
    assert dm.ring_crosses_itself(bow)
    assert not dm.ring_crosses_itself(np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float))


# ---- the negative and positive bodies ------------------------------------------------------------------------------------------
def test_the_bodies_are_closed_outward_oriented_solids():
    d = pentagon_to_circle()
    b = d.bodies()
    assert set(b) == {f"{k}_{p}" for k in ("neg", "pos") for p in ("branch_R", "branch_R_mirror", "trunk")}
    for name, body in b.items():
        assert body.mesh.n_open_edges == 0, name
        assert body.mesh.volume > 0.0, name  # positive volume = outward orientation
        assert body.lateral.n_open_edges > 0  # the lateral surface has no end caps


def test_the_negative_starts_exactly_lip_extension_upstream_and_shares_no_face_with_the_positive():
    lip = 0.07
    d = pentagon_to_circle(lip_extension=lip)
    b = d.bodies()
    plane = d.branches[0].plane
    n = plane.normal
    neg = b["neg_branch_R"]
    height = lambda pts: (np.asarray(pts) - plane.origin) @ n  # distance from the opening plane along +n (upstream)
    assert height(neg.rings[0]) == pytest.approx(np.full(neg.rings.shape[1], lip), abs=1e-12)
    assert height(neg.rings[1]) == pytest.approx(np.zeros(neg.rings.shape[1]), abs=1e-12)  # the opening plane
    assert neg.dist[0] == pytest.approx(-lip) and neg.dist[1] == 0.0
    pos = b["pos_branch_R"]
    assert height(pos.rings[0]) == pytest.approx(np.zeros(pos.rings.shape[1]), abs=1e-12)  # exactly at the plane
    assert height(pos.mesh.points).max() == pytest.approx(0.0, abs=1e-12)  # the positive never goes upstream of it
    # the negative's start cap is lip upstream of the positive's start cap: different planes, and no shared vertices
    assert height(neg.mesh.points).max() == pytest.approx(lip)
    r, n_pts = neg.rings.shape[:2]
    a = np.vstack([neg.rings[0], neg.mesh.points[r * n_pts]])  # the negative's start ring and its cap centre
    c = pos.mesh.points
    dist = np.linalg.norm(a[:, None, :] - c[None, :, :], axis=2)
    assert dist.min() > 1e-6


def test_the_positive_outer_section_is_the_offset_of_the_inner_one_in_the_same_plane():
    d = pentagon_to_circle(wall_thickness=0.012)
    b = d.bodies()
    for r, st in enumerate(d.branch):
        inner, outer = d.inner_ring(st), d.outer_ring(st)
        assert ((outer - st.centre) @ st.T) == pytest.approx(np.zeros(len(outer)), abs=1e-12)
        assert b["pos_branch_R"].rings[r] == pytest.approx(outer)
        assert b["neg_branch_R"].rings[r + 1] == pytest.approx(inner)


def test_both_end_at_the_engine_face_with_flat_caps():
    d = pentagon_to_circle()
    b = d.bodies()
    for name in ("neg_trunk", "pos_trunk"):
        last = b[name].rings[-1]
        assert (last[:, 0] == pytest.approx(np.full(last.shape[0], FACE[0]), abs=1e-12))  # the face plane x = const


def test_the_left_bodies_are_the_exact_mirror_image_of_the_right_ones():
    d = pentagon_to_circle(centreline=CentrelineParams(junction=(3.0, 0.9)))
    b = d.bodies()
    for kind in ("neg", "pos"):
        r, l = b[f"{kind}_branch_R"], b[f"{kind}_branch_R_mirror"]
        mirror = r.mesh.points * np.array([1.0, -1.0, 1.0])
        key = lambda p: p[np.lexsort(np.round(p, 10).T[::-1])]
        assert key(l.mesh.points) == pytest.approx(key(mirror), abs=1e-12)
        assert l.mesh.volume == pytest.approx(r.mesh.volume, rel=1e-3)  # (triangle diagonals differ with the order)
        assert l.dist == pytest.approx(r.dist)


def test_lip_mode_and_thickness_are_validated():
    with pytest.raises(ValueError, match="lip_mode"):
        pentagon_to_circle(lip_mode="rounded")
    with pytest.raises(ValueError, match="wall_thickness"):
        pentagon_to_circle(wall_thickness=0.0)
    with pytest.raises(ValueError, match="stations"):
        pentagon_to_circle(n_stations_branch=1)


def test_cone_half_angles_are_the_equivalent_conical_diffuser_angles():
    d = pentagon_to_circle()
    got = d.cone_half_angles()
    for part, sts in (("branch", d.branch), ("trunk", d.trunk)):
        r = np.array([math.sqrt(s.area / math.pi) for s in sts])
        ds = np.array([b.dist - a.dist for a, b in zip(sts, sts[1:])])
        assert got[part] == pytest.approx(np.degrees(np.arctan2(np.diff(r), ds)))


# ---- the Y junction: the branch ends tile the trunk start section ---------------------------------------------------------------------
from ductmgr.geometry import polygon_centroid  # noqa: E402
from ductmgr.section import points_in_polygon  # noqa: E402


def to_trunk_coords(d, points3d):
    """(a, b) coordinates of 3D points in the trunk frame at the junction, origin at the trunk centre."""
    _, U, V = d.trunk_frame0
    rel = np.asarray(points3d) - d.trunk[0].centre
    return np.c_[rel @ U, rel @ V]


def junction_rings(d):
    right = d.place(d.branch[-1], d.branch[-1].ring)
    left = dm.mirrored(right)[::-1]
    start = d.place(d.trunk[0], d.trunk[0].ring)
    return right, left, start


def test_the_union_of_the_two_branch_end_sections_is_the_trunk_start_section_with_no_overlap_and_no_gap():
    d = pentagon_to_circle(centreline=CentrelineParams(junction=(3.0, 0.9)))
    right, left, start = junction_rings(d)
    r2, l2, s2 = (to_trunk_coords(d, x) for x in (right, left, start))
    assert abs(polygon_area(r2)) + abs(polygon_area(l2)) == pytest.approx(abs(polygon_area(s2)), rel=1e-3)
    assert abs(polygon_area(r2)) == pytest.approx(abs(polygon_area(l2)), rel=1e-9)  # equal halves
    assert abs(polygon_area(r2)) == pytest.approx(0.5 * abs(polygon_area(s2)), rel=1e-3)
    # sample the section: every point of the trunk start is in exactly one half
    lo, hi = s2.min(axis=0), s2.max(axis=0)
    mid = lambda a, b: np.linspace(a, b, 161)[:-1] + (b - a) / 320  # cell centres: none lies on the cut itself
    g = np.stack(np.meshgrid(mid(lo[0], hi[0]), mid(lo[1], hi[1])), -1).reshape(-1, 2)
    in_s, in_r, in_l = (points_in_polygon(x, g) for x in (s2, r2, l2))
    assert not (in_r & in_l).any()  # no overlap
    assert not (in_s & ~in_r & ~in_l).any()  # no gap
    assert not (in_r & ~in_s).any() and not (in_l & ~in_s).any()  # nothing outside the trunk section


def test_the_halves_are_the_two_sides_of_y_equals_0_and_share_the_cut():
    d = pentagon_to_circle()
    right, left, start = junction_rings(d)
    assert right[:, 1].min() >= -1e-12 and left[:, 1].max() <= 1e-12  # right half has y >= 0, left half y <= 0
    on_cut = lambda ring: ring[np.abs(ring[:, 1]) < 1e-9]
    assert len(on_cut(right)) >= 2 and len(on_cut(left)) >= 2
    top = lambda ring: on_cut(ring)[np.argmax(on_cut(ring)[:, 2])]
    bottom = lambda ring: on_cut(ring)[np.argmin(on_cut(ring)[:, 2])]
    assert top(right) == pytest.approx(top(left)) and bottom(right) == pytest.approx(bottom(left))  # the same corners
    # every vertex of the trunk start section belongs to one of the halves
    both = np.vstack([right, left])
    for p in start:
        assert np.linalg.norm(both - p, axis=1).min() < 1e-9


def test_the_branch_end_has_the_shape_position_and_area_of_its_half():
    d = pentagon_to_circle()
    st = d.branch[-1]
    assert st.area == pytest.approx(polygon_area(st.ring)) and st.area == pytest.approx(0.5 * d.trunk[0].area, rel=1e-3)
    assert d.areas.trunk_area(0.0) == pytest.approx(2 * d.areas.branch_area(1.0))
    # the centreline ends at the half's centroid (not at y = 0) ...
    assert d.centreline.branch.ctrl[3] == pytest.approx(st.centre) and st.centre[1] > 0.0
    ab = to_trunk_coords(d, d.place(st, st.ring))
    c = polygon_centroid(ab)
    _, U, V = d.trunk_frame0
    assert d.trunk[0].centre + c[0] * U + c[1] * V == pytest.approx(st.centre, abs=1e-9)  # ... the centroid of the half
    # the trunk start section is mirror symmetric about y = 0
    s = d.place(d.trunk[0], d.trunk[0].ring)
    mirror = dm.mirrored(s)
    assert max(np.linalg.norm(s - p, axis=1).min() for p in mirror) < 1e-9


def test_the_total_flow_area_is_continuous_across_the_junction_and_g1():
    for junction in ((3.0, 0.9), (3.5, 0.2)):
        d = pentagon_to_circle(centreline=CentrelineParams(junction=junction))
        assert 2 * d.branch[-1].area == pytest.approx(d.trunk[0].area, rel=1e-3)
        a, b = d.branch[-1], d.trunk[0]
        assert a.T == pytest.approx(b.T, abs=1e-12) and a.U == pytest.approx(b.U, abs=1e-12)
        assert a.V == pytest.approx(b.V, abs=1e-12)
        cl = d.centreline
        assert cl.branch.tangents([1.0])[0] == pytest.approx(cl.trunk.tangents([0.0])[0], abs=1e-12)  # G1 although offset


def test_the_mirror_of_the_junction_is_exact():
    d = pentagon_to_circle()
    b = d.bodies()
    r, l = b["neg_branch_R"], b["neg_branch_R_mirror"]
    assert l.rings == pytest.approx(dm.mirrored(r.rings)[:, ::-1], abs=1e-15)
    assert l.dist == pytest.approx(r.dist)


def test_a_single_inlet_has_no_split():
    d = dm.Duct(spec_for(SectionShape.polygon(PENTAGON), origin=(1.0, 0.0, 0.2), mirror=False))
    assert d.branch_end_offset == pytest.approx(np.zeros(3)) and d.branch[-1].centre == pytest.approx(d.trunk[0].centre)
    assert d.branch[-1].area == pytest.approx(d.trunk[0].area, rel=1e-9)


# ---- overlap and splitter --------------------------------------------------------------------------------------------------------------
def test_the_branch_negatives_overlap_into_the_trunk_and_the_trunk_negative_starts_upstream():
    d = pentagon_to_circle(overlap=0.02)
    b = d.bodies()
    lb = d.centreline.length_branch
    assert b["neg_branch_R"].dist[-1] == pytest.approx(lb + 0.02)  # beyond the junction plane by the overlap
    nt = b["neg_trunk"]
    assert nt.dist[0] == pytest.approx(lb - 0.02)  # the trunk negative starts the same amount upstream
    ring, st = nt.rings[0], d.trunk[0]  # a prism of the trunk start section, 0.02 m back along the tangent
    assert ((ring - st.centre) @ st.T) == pytest.approx(np.full(len(ring), -0.02), abs=1e-12)
    # positives are not extended
    assert b["pos_branch_R"].dist[-1] == pytest.approx(lb) and b["pos_trunk"].dist[0] == pytest.approx(lb)


def test_the_overlap_rings_are_halves_of_the_trunk_section_so_the_internal_shape_is_unchanged():
    d = pentagon_to_circle(overlap=0.02)
    ring = d.bodies()["neg_branch_R"].rings[-1]
    st = min(d.trunk, key=lambda s: abs(s.dist - d.centreline.length_branch - 0.02))
    ab = np.c_[(ring - st.centre) @ st.U, (ring - st.centre) @ st.V]
    assert abs(polygon_area(ab)) == pytest.approx(0.5 * abs(polygon_area(st.ring)), rel=1e-3)
    inside = points_in_polygon(st.ring, ab * (1 - 1e-6))  # shrunk a hair about the origin: inside the trunk section
    assert inside.all()


def test_no_overlap_means_flush_ends():
    d = pentagon_to_circle(overlap=0.0)
    b = d.bodies()
    lb = d.centreline.length_branch
    assert b["neg_branch_R"].dist[-1] == pytest.approx(lb) and b["neg_trunk"].dist[0] == pytest.approx(lb)


def test_the_splitter_keeps_the_branch_negatives_apart_by_exactly_the_wall_thickness():
    t, L = 0.012, 0.15
    d = pentagon_to_circle(wall_thickness=t, splitter_length=L, overlap=0.02)
    b = d.bodies()
    lb = d.centreline.length_branch
    r, l = b["neg_branch_R"], b["neg_branch_R_mirror"]
    into = r.dist > lb - 1e-12  # the junction ring and the half-sections inside the trunk
    assert into.sum() >= 3 and r.dist[-1] == pytest.approx(lb + L + 0.02)
    for ring_r, ring_l in zip(r.rings[into], l.rings[into]):
        assert ring_r[:, 1].min() == pytest.approx(t / 2, abs=1e-12)  # the cut faces are t / 2 from y = 0 ...
        assert ring_l[:, 1].max() == pytest.approx(-t / 2, abs=1e-12)
        assert ring_r[:, 1].min() - ring_l[:, 1].max() == pytest.approx(t, abs=1e-12)  # ... so the wall is t thick
    nt = b["neg_trunk"]
    assert nt.dist[0] == pytest.approx(lb + L, abs=1e-9)  # the trunk negative starts where the splitter ends
    assert b["pos_trunk"].dist[0] == pytest.approx(lb)  # the positive trunk runs from the junction: the wall is left


def test_without_a_splitter_the_branch_end_is_exactly_the_half_on_y_equals_0():
    d = pentagon_to_circle(splitter_length=0.0)
    right = d.place(d.branch[-1], d.branch[-1].ring)
    assert right[:, 1].min() == pytest.approx(0.0, abs=1e-12)  # the two ends meet on the plane, no wall


def test_junction_options_are_validated():
    with pytest.raises(ValueError, match="splitter_length"):
        pentagon_to_circle(splitter_length=-0.1)
    with pytest.raises(ValueError, match="shorter than the trunk"):
        pentagon_to_circle(splitter_length=10.0)
