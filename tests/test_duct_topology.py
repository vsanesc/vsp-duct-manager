# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""General inlet topology: 1..N inlets, optional symmetry, the sector junction."""
import math

import numpy as np
import pytest

from ductmgr import duct as dm
from ductmgr import outline as ol
from ductmgr.areacurve import AreaPoints
from ductmgr.centreline import CentrelineParams
from ductmgr.geometry import InletPlane, polygon_area
from ductmgr.section import points_in_polygon

FACE = np.array([6.0, 0.0, 0.0])
AXIS = np.array([1.0, 0.0, 0.0])
FACE_D = 0.5


def key(r=0.1, s=0.0):
    shape = ol.circle_shape(r)
    return dm.KeyOutline(s, ol.build_outline(ol.edge_paths(shape)[0]), np.array(shape.vertices))


def pentagon_key():
    shape = ol.SectionShape.polygon([(0.0, 0.12), (-0.11, 0.04), (-0.07, -0.1), (0.07, -0.1), (0.11, 0.04)])
    return dm.KeyOutline(0.0, ol.build_outline(ol.edge_paths(shape)[0]), np.array(shape.vertices))


def inlet(name, y, z, fraction=None, mirror=False, r=0.1, sweep=0.0, k=None):
    return dm.InletSpec(name, InletPlane((1.0, y, z), 0.0, sweep), (k or key(r),), fraction, mirror)


def duct(inlets, symmetry="none", **kw):
    spec = dm.DuctSpec(tuple(inlets), FACE, AXIS, FACE_D, symmetry=symmetry, **kw)
    return dm.Duct(spec)


def trunk_coords(d, pts3d):
    _, U, V = d.trunk_frame0
    rel = np.asarray(pts3d) - d.trunk[0].centre
    return np.c_[rel @ U, rel @ V]


def end_rings(d):
    """Each branch's end section in the trunk frame (a, b), placed where the branch ends."""
    out = []
    for bg in d.branches:
        st = bg.stations[-1]
        out.append(trunk_coords(d, d.place(st, st.ring)))
    return out


def grid_over(poly, cells=200):
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    mid = lambda a, b: np.linspace(a, b, cells + 1)[:-1] + (b - a) / (2 * cells)
    return np.stack(np.meshgrid(mid(lo[0], hi[0]), mid(lo[1], hi[1])), -1).reshape(-1, 2)


def assert_tiling(d, fractions=None):
    rings = end_rings(d)
    s2 = trunk_coords(d, d.place(d.trunk[0], d.trunk[0].ring))
    total = abs(polygon_area(s2))
    assert sum(abs(polygon_area(r)) for r in rings) == pytest.approx(total, rel=1e-3)  # union area = trunk area
    g = grid_over(s2)
    inside = [points_in_polygon(r, g) for r in rings]
    in_s = points_in_polygon(s2, g)
    count = np.sum(inside, axis=0)
    assert (count <= 1).all()  # pairwise overlap 0
    assert not (in_s & (count == 0)).any()  # no gap
    assert not (~in_s & (count > 0)).any()  # nothing outside the trunk section
    if fractions is not None:
        for r, f in zip(rings, fractions):
            assert abs(polygon_area(r)) == pytest.approx(f * total, rel=1e-3)


# ---- flow fractions -------------------------------------------------------------------------------------------------------------
def test_default_fractions_are_an_equal_split_over_all_physical_branches():
    f, changed = dm.flow_fractions([None, None, None], [False, False, False])
    assert f == pytest.approx([1 / 3] * 3) and not changed
    f, changed = dm.flow_fractions([None, None], [True, False])  # three branches: a mirrored pair and one more
    assert f == pytest.approx([1 / 3, 1 / 3]) and not changed


def test_fractions_that_do_not_sum_to_one_are_normalised_and_reported():
    f, changed = dm.flow_fractions([0.5, 0.2], [False, False])
    assert changed and f == pytest.approx([0.5 / 0.7, 0.2 / 0.7]) and sum(f) == pytest.approx(1.0)
    f, changed = dm.flow_fractions([0.3, 0.4], [True, False])  # 2 x 0.3 + 0.4 = 1: already right
    assert not changed and f == pytest.approx([0.3, 0.4])
    f, changed = dm.flow_fractions([0.6, 0.4], [True, False])  # 2 x 0.6 + 0.4 = 1.6
    assert changed and 2 * f[0] + f[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        dm.flow_fractions([0.0, 1.0], [False, False])


def test_the_duct_uses_the_normalised_fractions_and_says_so():
    d = duct([inlet("a", 0.4, 0.2, 0.5), inlet("b", -0.4, 0.2, 0.2)])
    assert d.fractions_normalised and d.fractions == pytest.approx([0.5 / 0.7, 0.2 / 0.7])
    assert [bg.flow for bg in d.branches] == pytest.approx(d.fractions)


# ---- 1..N inlets ----------------------------------------------------------------------------------------------------------------------
LAYOUTS = {
    1: [(0.0, 0.3)],
    2: [(0.4, 0.2), (-0.4, 0.3)],
    3: [(0.45, 0.1), (-0.45, 0.1), (0.0, 0.6)],
    4: [(0.4, 0.4), (-0.4, 0.4), (0.4, -0.2), (-0.4, -0.2)],
}


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_one_to_four_inlets_build_valid_closed_ducts(count):
    d = duct([inlet(f"i{k}", y, z) for k, (y, z) in enumerate(LAYOUTS[count])])
    assert len(d.branches) == count
    b = d.bodies()
    assert set(b) == {f"{t}_branch_i{k}" for t in ("neg", "pos") for k in range(count)} | {"neg_trunk", "pos_trunk"}
    for body in b.values():
        assert body.mesh.n_open_edges == 0 and body.mesh.volume > 0.0
    for bg in d.branches:  # every branch is G1 with the trunk and starts at its own centroid
        cl = bg.centreline
        assert cl.branch.tangents([1.0])[0] == pytest.approx(cl.trunk.tangents([0.0])[0], abs=1e-12)
        assert cl.branch.points([0.0])[0] == pytest.approx(bg.stations[0].centre)
    assert [t.area for t in (d.trunk[0],)] == pytest.approx([d.branches[0].areas.junction_area])


@pytest.mark.parametrize("count", [2, 3, 4])
def test_the_branch_ends_tile_the_trunk_start_section_with_equal_fractions(count):
    d = duct([inlet(f"i{k}", y, z) for k, (y, z) in enumerate(LAYOUTS[count])])
    assert_tiling(d, [1.0 / count] * count)


@pytest.mark.parametrize("fractions", [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.6, 0.25, 0.15]])
def test_sector_tiling_is_exact_with_unequal_fractions(fractions):
    d = duct([inlet(f"i{k}", y, z, f) for k, ((y, z), f) in enumerate(zip(LAYOUTS[3], fractions))])
    assert_tiling(d, fractions)
    for bg in d.branches:  # the flow area of every branch end is its fraction of the junction area
        assert bg.stations[-1].area == pytest.approx(bg.flow * d.branches[0].areas.junction_area, rel=1e-3)


def test_four_sectors_with_unequal_fractions_and_the_sectors_follow_the_branch_order():
    fr = [0.4, 0.3, 0.2, 0.1]
    d = duct([inlet(f"i{k}", y, z, f) for k, ((y, z), f) in enumerate(zip(LAYOUTS[4], fr))])
    assert_tiling(d, fr)
    angles = []  # the angular position of each branch around the trunk axis matches that of its sector centroid
    _, U, V = d.trunk_frame0
    for bg, ring in zip(d.branches, end_rings(d)):
        rel = bg.stations[0].centre - d.centreline.junction
        pos = math.atan2(rel @ V, rel @ U)
        c = ring.mean(axis=0)
        angles.append(abs((math.atan2(c[1], c[0]) - pos + math.pi) % (2 * math.pi) - math.pi))
    assert max(angles) < math.radians(75)  # each sector faces its branch (not the opposite side)


def test_two_branches_are_split_by_the_plane_equidistant_to_both_with_unequal_fractions():
    d = duct([inlet("a", 0.4, 0.2, 0.7), inlet("b", -0.4, 0.2, 0.3)])
    assert_tiling(d, [0.7, 0.3])
    right, left = end_rings(d)
    assert right[:, 0].min() < left[:, 0].min() or right[:, 0].max() > left[:, 0].max()  # side by side in a
    # the cut is a straight line, so the two ends share the corner points
    shared = [p for p in right if np.linalg.norm(left - p, axis=1).min() < 1e-9]
    assert len(shared) >= 2


def test_the_split_plane_of_two_branches_is_perpendicular_to_the_line_joining_them():
    d = duct([inlet("a", 0.5, 0.5), inlet("b", -0.3, -0.1)])
    _, U, V = d.trunk_frame0
    n = np.array(d.branches[0].piece.normal)
    cl = d.centreline
    rel = [bg.stations[0].centre - cl.junction for bg in d.branches]
    line = np.array([rel[0] @ U - rel[1] @ U, rel[0] @ V - rel[1] @ V])
    assert n == pytest.approx(line / np.linalg.norm(line))


# ---- symmetry xz ----------------------------------------------------------------------------------------------------------------------------
def test_a_mirrored_inlet_gets_an_exact_twin_and_the_junction_lies_on_y_equals_0():
    d = duct([inlet("R", 0.45, 0.3, mirror=True)], "xz")
    assert [bg.name for bg in d.branches] == ["R", "R_mirror"] and d.branches[1].twin
    b = d.bodies()
    r, l = b["neg_branch_R"], b["neg_branch_R_mirror"]
    assert l.rings == pytest.approx(dm.mirrored(r.rings)[:, ::-1], abs=1e-15)
    cl = d.centreline
    assert cl.junction[1] == 0.0 and cl.junction_tangent[1] == 0.0
    twin_cl = d.branches[1].centreline
    assert twin_cl.branch.ctrl == pytest.approx(dm.mirrored(cl.branch.ctrl), abs=1e-12)
    assert twin_cl.trunk.ctrl == pytest.approx(cl.trunk.ctrl)
    assert_tiling(d, [0.5, 0.5])
    assert d.branches[0].piece.normal in ((1.0, 0.0), (-1.0, 0.0)) and d.branches[0].piece.offset == 0.0  # y = 0


def test_symmetry_xz_forces_the_junction_onto_the_symmetry_plane_even_if_asked_otherwise():
    d = duct([inlet("R", 0.45, 0.3, mirror=True)], "xz",
             centreline=CentrelineParams(junction=(3.0, 0.2), junction_y=0.4, junction_yaw_deg=20.0))
    assert d.centreline.junction[1] == 0.0 and d.centreline.junction_tangent[1] == 0.0


def test_a_self_symmetric_inlet_on_y_equals_0_is_never_duplicated():
    assert dm.self_symmetric(InletPlane((1.0, 0.0, 0.3), 12.0, 0.0))
    assert not dm.self_symmetric(InletPlane((1.0, 0.0, 0.3), 0.0, 5.0)) and not dm.self_symmetric(InletPlane((1.0, 0.1, 0.3)))
    d = duct([inlet("C", 0.0, 0.5, mirror=True)], "xz")
    assert len(d.branches) == 1 and not d.branches[0].twin
    assert not any(n.endswith("_mirror") for n in d.bodies() if "branch" in n)


def test_a_mirrored_pair_plus_a_self_symmetric_inlet_is_symmetric_and_tiles_exactly():
    d = duct([inlet("R", 0.45, 0.1, mirror=True), inlet("C", 0.0, 0.6)], "xz")
    assert [bg.name for bg in d.branches] == ["R", "R_mirror", "C"]
    assert_tiling(d, [1 / 3] * 3)
    rings = end_rings(d)
    c_end = rings[2]  # the on-axis branch ends on a sector that is symmetric about a = 0
    mirror = c_end * np.array([-1.0, 1.0])
    assert max(np.linalg.norm(c_end - p, axis=1).min() for p in mirror) < 1e-6
    r, l = d.bodies()["neg_branch_R"], d.bodies()["neg_branch_R_mirror"]
    assert l.rings == pytest.approx(dm.mirrored(r.rings)[:, ::-1], abs=1e-15)
    assert d.branches[1].branch_end_offset == pytest.approx(dm.mirrored(d.branches[0].branch_end_offset))


def test_unequal_fractions_with_a_mirrored_pair_keep_the_layout_symmetric():
    d = duct([inlet("R", 0.45, 0.1, 0.3, mirror=True), inlet("C", 0.0, 0.6, 0.4)], "xz")
    assert d.fractions == pytest.approx([0.3, 0.4])
    assert_tiling(d, [0.3, 0.3, 0.4])


# ---- symmetry none ----------------------------------------------------------------------------------------------------------------------------
def test_without_symmetry_a_mirror_request_is_ignored_and_the_junction_is_free():
    d = duct([inlet("a", 0.45, 0.3, mirror=True)], "none")
    assert len(d.branches) == 1  # no twin
    free = duct([inlet("a", 0.6, 0.3), inlet("b", -0.2, 0.5)], "none",
                centreline=CentrelineParams(junction=(3.0, 0.4), junction_y=0.25, junction_yaw_deg=10.0))
    cl = free.centreline
    assert cl.junction == pytest.approx([3.0, 0.25, 0.4]) and cl.junction_tangent[1] == pytest.approx(math.sin(math.radians(10.0)))
    assert free.branches[1].centreline.junction == pytest.approx(cl.junction)  # one shared junction
    assert_tiling(free)


def test_an_asymmetric_layout_without_symmetry_builds_and_its_default_junction_is_off_the_plane():
    d = duct([inlet("a", 0.7, 0.4), inlet("b", 0.2, -0.1)], "none")
    assert d.centreline.junction[1] != 0.0  # the default junction follows the inlets, not y = 0
    for body in d.bodies().values():
        assert body.mesh.n_open_edges == 0
    assert_tiling(d, [0.5, 0.5])


def test_the_trunk_start_section_is_only_symmetrised_when_the_layout_is_symmetric():
    skew = ol.SectionShape.polygon([(0.0, 0.12), (-0.04, 0.05), (-0.11, -0.05), (0.07, -0.1), (0.12, 0.0)])
    skew_key = dm.KeyOutline(0.0, ol.build_outline(ol.edge_paths(skew)[0]), np.array(skew.vertices))
    sym = duct([inlet("R", 0.45, 0.3, mirror=True, k=skew_key)], "xz")
    s = sym.trunk[0].ring
    assert max(np.linalg.norm(s - p * np.array([-1.0, 1.0]), axis=1).min() for p in s) < 1e-9
    skew = ol.SectionShape.polygon([(0.0, 0.12), (-0.04, 0.05), (-0.11, -0.05), (0.07, -0.1), (0.12, 0.0)])
    skew_key = dm.KeyOutline(0.0, ol.build_outline(ol.edge_paths(skew)[0]), np.array(skew.vertices))
    free = duct([inlet("a", 0.45, 0.3, k=skew_key), inlet("b", -0.45, 0.3, k=skew_key)], "none")
    s = free.trunk[0].ring
    assert max(np.linalg.norm(s - p * np.array([-1.0, 1.0]), axis=1).min() for p in s) > 1e-3


# ---- splitter walls and overlap with sectors -------------------------------------------------------------------------------------------
def test_splitter_walls_along_the_rays_have_exactly_the_wall_thickness():
    t, L = 0.012, 0.15
    d = duct([inlet(f"i{k}", y, z) for k, (y, z) in enumerate(LAYOUTS[3])], wall_thickness=t, splitter_length=L)
    b = d.bodies()
    lb = d.branches[0].centreline.length_branch
    for bg, other in ((d.branches[0], d.branches[1]),):
        r = b[f"neg_branch_{bg.name}"]
        into = r.dist > lb - 1e-12
        assert into.sum() >= 3
    # the distance between two neighbouring sectors at the junction plane is the wall thickness: compare their
    # end sections, which are pulled back by t / 2 from the shared ray
    rings = end_rings(d)
    gaps = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, c = rings[i], rings[j]
            dist = np.linalg.norm(a[:, None] - c[None, :], axis=2).min()
            gaps.append(dist)
    assert min(gaps) == pytest.approx(t, abs=1e-9)  # neighbours are exactly one wall thickness apart
    assert b["neg_trunk"].dist[0] == pytest.approx(lb + L, abs=1e-9)


def test_branch_negatives_overlap_into_the_trunk_for_sectors_too():
    d = duct([inlet(f"i{k}", y, z) for k, (y, z) in enumerate(LAYOUTS[3])], overlap=0.03)
    b = d.bodies()
    for bg in d.branches:
        lb = bg.centreline.length_branch
        assert b[f"neg_branch_{bg.name}"].dist[-1] == pytest.approx(lb + 0.03)
    assert b["neg_trunk"].dist[0] == pytest.approx(d.branches[0].centreline.length_branch - 0.03)


# ---- one continuous duct -------------------------------------------------------------------------------------------------------------------
def test_a_single_branch_is_one_continuous_duct_with_a_g1_intermediate_point():
    d = duct([inlet("a", 0.0, 0.3)], "none", centreline=CentrelineParams(junction=(3.0, 0.0)))
    assert d.n_branches == 1
    bg = d.branches[0]
    assert bg.branch_end_offset == pytest.approx(np.zeros(3))
    assert bg.stations[-1].centre == pytest.approx(d.trunk[0].centre) and bg.stations[-1].area == pytest.approx(d.trunk[0].area)
    assert bg.centreline.branch.tangents([1.0])[0] == pytest.approx(bg.centreline.trunk.tangents([0.0])[0], abs=1e-12)
    b = d.bodies()
    assert b["neg_trunk"].dist[0] == pytest.approx(d.trunk[0].dist)  # no overlap with itself


# ---- validation ----------------------------------------------------------------------------------------------------------------------------------
def test_validation():
    with pytest.raises(ValueError, match="symmetry"):
        duct([inlet("a", 0.4, 0.3)], "yz")
    with pytest.raises(ValueError, match="at least one inlet"):
        dm.Duct(dm.DuctSpec((), FACE, AXIS, FACE_D))
