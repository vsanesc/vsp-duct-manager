# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The export mapping: key sections -> OpenVSP Edit Curve (CEDIT) XSecs, checked against the real API."""
import math

import numpy as np
import openvsp as vsp
import pytest

import bodies
from ductmgr import cedit
from ductmgr import outline as ol
from ductmgr import section as sec
from ductmgr import sketch as sk
from ductmgr.geometry import InletPlane, polygon_area
from ductmgr.outline import SectionShape

PENTAGON = [(0.0, 1.0), (-0.95, 0.31), (-0.59, -0.81), (0.59, -0.81), (0.95, 0.31)]
PRISM_PLANE = InletPlane((4.0, 0.0, 0.4))
DEVIATION = 1e-3  # of the section size: the round-trip limit


def outline_of(shape, section=None):
    return ol.build_outline(ol.edge_paths(shape, section)[0])


def size_of(points):
    return float(np.ptp(points, axis=0).max())


def polyline_deviation(a, b):
    """Symmetric Hausdorff distance between two closed sampled curves (each as a polyline)."""
    def one_way(p, q):
        return sec.distance_to_section(sec.make_section(q, True), p).max()

    return max(one_way(a, b), one_way(b, a))


# ---- shapes to test ---------------------------------------------------------------------------------------------
def mixed_shape():
    """A pentagon with a curved edge, a smooth vertex, a symmetric vertex and corners."""
    s = SectionShape.polygon(PENTAGON)
    s = ol.set_edge_type(ol.set_edge_type(s, 1, "curved"), 2, "curved")
    s = ol.move_handle(s, 1, 0, np.array(s.vertices[1]) + [-0.2, -0.3])
    s = ol.move_handle(s, 2, 1, np.array(s.vertices[3]) + [-0.25, 0.2])
    s = ol.set_continuity(s, 2, "smooth")
    return ol.set_continuity(s, 3, "symmetric")


def inlet_shape():
    """The default inlet sketch on a chined body: a snapped bottom edge following the skin through two chines."""
    body = bodies.chined_prism()
    s = sec.compute_section(body, PRISM_PLANE)
    shape = sk.default_sketch(s, PRISM_PLANE, 0.12)
    return shape, s


# ---- the pure mapping -------------------------------------------------------------------------------------------
def test_a_circle_is_four_segments_thirteen_points():
    c = cedit.section_to_cedit(ol.circle_shape(1.0))
    assert c.n_segments == 4 and len(c.points) == 13 and len(c.u) == 13 and len(c.g1) == 13
    assert c.u[0] == 0.0 and c.u[-1] == 1.0 and c.u[3] == pytest.approx(0.25)  # anchors at k / m, as VSP's default
    assert c.points[0] == pytest.approx(c.points[-1])  # closed
    assert c.width == pytest.approx(2.0, rel=2e-3) and c.height == pytest.approx(2.0, rel=2e-3)
    assert max(max(abs(x), abs(y)) for x, y in c.points) <= 0.5 + 1e-9  # normalised: the curve fits +-0.5


def test_straight_edges_are_degenerate_beziers_with_handles_at_one_third_and_two_thirds():
    shape = SectionShape.polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)])
    c = cedit.section_to_cedit(shape)
    pts = np.array(c.points) * np.array([c.width, c.height]) + np.array(c.centre)
    for k in range(c.n_segments):
        a, h1, h2, b = pts[3 * k: 3 * k + 4]
        assert h1 == pytest.approx(a + (b - a) / 3, abs=1e-12) and h2 == pytest.approx(a + 2 * (b - a) / 3, abs=1e-12)
    assert c.n_segments == 4 and not any(c.g1)  # all corners: no G1 flag anywhere
    assert cedit.sample_cedit(c)[0] == pytest.approx(cedit.sample_cedit(c)[-1])  # a closed curve


def test_a_straight_edged_polygon_is_reproduced_exactly_by_the_samples():
    shape = SectionShape.polygon(PENTAGON)
    c = cedit.section_to_cedit(shape)
    assert polyline_deviation(cedit.sample_cedit(c)[:-1], outline_of(shape)) < 1e-9


def test_the_curve_starts_at_the_reference_vertex_and_runs_clockwise_like_vsp():
    c = cedit.section_to_cedit(SectionShape.polygon(PENTAGON))
    start = np.array(c.points[0]) * np.array([c.width, c.height]) + np.array(c.centre)
    assert start == pytest.approx([0.0, 1.0], abs=1e-12)  # the apex, closest to +v
    assert polygon_area(cedit.sample_cedit(c)[:-1]) < 0  # clockwise in the section plane
    ccw = cedit.section_to_cedit(SectionShape.polygon(PENTAGON), clockwise=False)
    assert polygon_area(cedit.sample_cedit(ccw)[:-1]) > 0
    other = cedit.section_to_cedit(SectionShape.polygon(PENTAGON), start_vertex=2)
    assert np.array(other.points[0]) * [other.width, other.height] + other.centre == pytest.approx([-0.59, -0.81], abs=1e-12)


def test_the_mapping_is_the_same_curve_in_either_direction():
    shape = mixed_shape()
    a = cedit.sample_cedit(cedit.section_to_cedit(shape, clockwise=True))
    b = cedit.sample_cedit(cedit.section_to_cedit(shape, clockwise=False))
    assert polyline_deviation(a[:-1], b[:-1]) < 1e-9
    assert polyline_deviation(a[:-1], outline_of(shape)) < DEVIATION * size_of(outline_of(shape))


def test_g1_flags_only_where_the_handles_really_are_collinear():
    c = cedit.section_to_cedit(ol.circle_shape(1.0))
    assert [c.g1[i] for i in (0, 3, 6, 9)] == [True] * 4  # the 4 smooth anchors
    assert not any(c.g1[i] for i in range(13) if i % 3) and not c.g1[12]  # handles and the closing copy: no flag
    # a 'smooth' vertex between two straight edges at an angle is NOT collinear: it must stay a corner
    fake = ol.set_continuity(SectionShape.polygon(PENTAGON), 1, "smooth")
    assert not any(cedit.section_to_cedit(fake).g1)
    mixed = cedit.section_to_cedit(mixed_shape())
    assert 1 <= sum(mixed.g1) <= 4 and not mixed.g1[-1]  # the closing point repeats the first anchor: no flag


def test_corners_are_kept_as_corners():
    shape = mixed_shape()
    c = cedit.section_to_cedit(shape)
    segs, g1 = cedit.segments_of(shape)
    corner_vertices = [i for i, m in enumerate(shape.continuity) if m == "corner"]
    assert corner_vertices  # the shape has corners
    assert sum(g1) < len(g1)  # not everything is flagged smooth


def test_a_skin_following_edge_becomes_a_chain_of_degenerate_beziers_through_the_chines():
    shape, s = inlet_shape()
    segs, g1 = cedit.segments_of(shape, s)
    arc_pieces = [seg for seg in segs if np.linalg.norm(seg[1] - (seg[0] + (seg[3] - seg[0]) / 3)) < 1e-12]
    assert len(segs) > shape.n  # the arc along the skin contributes several segments (chine corners)
    assert len(arc_pieces) >= 3
    c = cedit.section_to_cedit(shape, s)
    assert c.n_segments == len(segs)
    assert polyline_deviation(cedit.sample_cedit(c)[:-1], outline_of(shape, s)) < DEVIATION * size_of(outline_of(shape, s))


def test_a_degenerate_section_cannot_be_normalised():
    flat = SectionShape.polygon([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    with pytest.raises(ValueError, match="too small"):
        cedit.section_to_cedit(flat)


# ---- round trip through the OpenVSP API ---------------------------------------------------------------------------
@pytest.fixture()
def xsec():
    """An Edit Curve XSec of a Stack, ready to receive a section."""
    vsp.ClearVSPModel()
    stack = vsp.AddGeom("STACK")
    vsp.Update()
    surf = vsp.GetXSecSurf(stack, 0)
    xs = cedit.make_edit_curve(surf, 1)
    return xs


CASES = {
    "circle": lambda: (ol.circle_shape(0.28, centre=(0.0, 0.15)), None),
    "straight pentagon": lambda: (SectionShape.polygon(PENTAGON), None),
    "mixed curves": lambda: (mixed_shape(), None),
    "inlet with a snapped edge": inlet_shape,
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_xsec_built_through_the_api_matches_our_outline(xsec, name):
    shape, s = CASES[name]()
    c = cedit.section_to_cedit(shape, s)
    cedit.apply_cedit(xsec, c)
    ours = outline_of(shape, s)
    # the XSec is centred on its own origin: add back the centre we removed
    theirs = cedit.read_cedit_samples(xsec, c.n_segments) + np.array(c.centre)
    deviation = polyline_deviation(ours, theirs)
    assert deviation < DEVIATION * size_of(ours), f"{name}: {deviation:.2e} m on a {size_of(ours):.3f} m section"


def test_the_control_points_round_trip_exactly(xsec):
    shape, s = inlet_shape()
    c = cedit.section_to_cedit(shape, s)
    cedit.apply_cedit(xsec, c)
    got = vsp.GetEditXSecCtrlVec(xsec, True)
    assert len(got) == len(c.points)
    assert max(max(abs(g.x() - p[0]), abs(g.y() - p[1])) for g, p in zip(got, c.points)) < 1e-12
    assert list(vsp.GetEditXSecUVec(xsec)) == pytest.approx(list(c.u), abs=1e-12)
    assert vsp.GetXSecWidth(xsec) == pytest.approx(c.width) and vsp.GetXSecHeight(xsec) == pytest.approx(c.height)
    assert vsp.GetParmVal(vsp.GetXSecParm(xsec, "CurveType")) == vsp.CEDIT
    for sym in ("SymType", "TBSym", "RLSym", "AllSym"):  # the XSec must not mirror the edit
        assert vsp.GetParmVal(vsp.GetXSecParm(xsec, sym)) == 0.0


def test_the_g1_flags_arrive_in_the_xsec(xsec):
    c = cedit.section_to_cedit(ol.circle_shape(1.0))
    cedit.apply_cedit(xsec, c)
    flags = [bool(vsp.GetParmVal(vsp.GetXSecParm(xsec, f"G1_{i}"))) for i in range(len(c.points))]
    assert flags == list(c.g1) and sum(flags) == 4


def test_corners_stay_sharp_in_the_xsec(xsec):
    square = SectionShape.polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
    c = cedit.section_to_cedit(square)
    cedit.apply_cedit(xsec, c)
    pts = cedit.read_cedit_samples(xsec, c.n_segments)
    on_boundary = np.minimum(np.abs(np.abs(pts[:, 0]) - 1.0), np.abs(np.abs(pts[:, 1]) - 1.0))
    assert on_boundary.max() < 1e-9  # a rounded corner would cut inside the square
    assert np.abs(pts).max() == pytest.approx(1.0, abs=1e-9)


def test_applying_twice_replaces_the_curve(xsec):
    cedit.apply_cedit(xsec, cedit.section_to_cedit(ol.circle_shape(1.0)))
    shape = SectionShape.polygon(PENTAGON)
    cedit.apply_cedit(xsec, cedit.section_to_cedit(shape))
    c = cedit.section_to_cedit(shape)
    got = cedit.read_cedit_samples(xsec, c.n_segments) + np.array(c.centre)
    assert polyline_deviation(outline_of(shape), got) < DEVIATION * size_of(outline_of(shape))
    assert len(vsp.GetEditXSecCtrlVec(xsec, True)) == len(c.points)  # 16 points now, not the circle's 13
