# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Simplify a section to N vertices with fitted curved edges; reset the trunk start to a superellipse."""
import math

import numpy as np
import pytest

from ductmgr import duct as dm
from ductmgr import outline as ol
from ductmgr import simplify as sp
from ductmgr.geometry import polygon_area


def dense_polygon(a=1.0, b=0.8, exponent=2.5, n=64):
    se = ol.superellipse_shape(a, b, exponent)
    ring = dm.symmetric_ring(np.asarray(ol.build_outline(ol.edge_paths(se)[0])), n)
    return ol.SectionShape.polygon(ring)


def area(shape):
    return ol.outline_area(shape)


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_the_result_has_n_vertices_curved_edges_the_same_area_and_a_small_deviation(n):
    poly = dense_polygon()
    r = sp.simplify(poly, n, symmetric=True)
    assert r.shape.n == r.n_after == n and r.n_before == 64
    assert area(r.shape) == pytest.approx(area(poly), rel=1e-9) and r.area == pytest.approx(area(poly), rel=1e-9)
    assert sum(e == ol.CURVED for e in r.shape.edges) >= n - 1 and not any(r.shape.snapped)
    assert r.max_dev >= r.rms_dev > 0.0
    assert r.rel_max < {4: 0.05, 6: 0.02, 8: 0.01, 12: 0.01}[n]  # a few percent of the radius at worst
    assert f"{r.n_before} -> {n} vertices" in r.text() and "mm" in r.text()


def test_more_vertices_deviate_less_overall():
    poly = dense_polygon()
    dev = [sp.simplify(poly, n, True).max_dev for n in (4, 8, 12)]  # (the greedy choice is not strictly monotone)
    assert dev[0] > dev[1] > 0.5 * dev[2] and dev[0] > 2 * dev[2]


def test_a_symmetric_section_stays_exactly_mirror_symmetric_with_collinear_smooth_handles():
    r = sp.simplify(dense_polygon(), 8, symmetric=True)
    assert ol.is_mirror_symmetric(r.shape)
    partners = ol.vertex_partners(r.shape)
    for e in range(r.shape.n):
        ep = ol.edge_partner(r.shape, partners, e)
        assert ep is not None
        if r.shape.edges[e] == ol.CURVED:
            (a0, a1), (b0, b1) = r.shape.handles[e], r.shape.handles[ep]
            assert np.array(b0) == pytest.approx([-a1[0], a1[1]], abs=1e-12)
            assert np.array(b1) == pytest.approx([-a0[0], a0[1]], abs=1e-12)
    for i in range(r.shape.n):  # G1 at smooth vertices: the two handles point opposite ways
        if r.shape.continuity[i] == ol.SMOOTH:
            out_h, in_h = np.array(r.shape.handles[i][0]), np.array(r.shape.handles[i - 1][1])
            cross = out_h[0] * in_h[1] - out_h[1] * in_h[0]
            assert abs(cross) < 1e-9 * np.linalg.norm(out_h) * np.linalg.norm(in_h)
            assert out_h @ in_h < 0.0


def test_the_vertices_of_the_result_are_vertices_of_the_original():
    poly = dense_polygon()
    r = sp.simplify(poly, 8, symmetric=True)
    original = np.array(poly.vertices)
    assert all(np.linalg.norm(original - np.array(v), axis=1).min() < 0.02 for v in r.shape.vertices)  # (re-centred a hair)


def test_corners_are_kept_as_corners_and_straight_edges_stay_straight():
    square = ol.SectionShape.polygon([(-1, -1), (-0.5, -1), (0, -1), (0.5, -1), (1, -1), (1, -0.5), (1, 0), (1, 0.5),
                                      (1, 1), (0.5, 1), (0, 1), (-0.5, 1), (-1, 1), (-1, 0.5), (-1, 0), (-1, -0.5)])
    r = sp.simplify(square, 4, symmetric=False)
    assert r.shape.n == 4 and all(e == ol.STRAIGHT for e in r.shape.edges) and all(c == ol.CORNER for c in r.shape.continuity)
    assert r.max_dev < 1e-9 and r.rms_dev < 1e-9 and area(r.shape) == pytest.approx(4.0)


def test_a_free_shape_is_simplified_without_symmetry_and_keeps_its_centroid():
    pent = ol.SectionShape.polygon([(0.0, 0.2), (-0.19, 0.06), (-0.12, -0.16), (0.12, -0.16), (0.19, 0.06)])
    r = sp.simplify(pent, 4)
    assert r.shape.n == 4 and area(r.shape) == pytest.approx(area(pent), rel=1e-9)
    assert ol.outline_centroid(r.shape) == pytest.approx(ol.outline_centroid(pent), abs=1e-9)


def test_impossible_requests_are_refused_with_a_reason():
    poly = dense_polygon()
    with pytest.raises(ValueError, match="fewer"):
        sp.simplify(poly, 64)
    with pytest.raises(ValueError, match="at least 3"):
        sp.simplify(poly, 2)
    with pytest.raises(ValueError, match="not mirror symmetric"):
        sp.simplify(ol.SectionShape.polygon([(-1, 0), (0.3, -1), (1, 0.2), (0, 1), (-0.5, 0.7)]), 3, symmetric=True)
    # a symmetric section without a vertex on the axis (4 vertices in two pairs) cannot give an odd count
    pairs = ol.SectionShape.polygon([(-0.5, 1), (-1, -1), (1, -1), (0.5, 1)] + [(0.5, 1.5), (-0.5, 1.5)])
    assert ol.is_mirror_symmetric(pairs) and all(p != v for v, p in enumerate(ol.vertex_partners(pairs)))
    with pytest.raises(ValueError, match="mirror"):
        sp.simplify(pairs, 5, symmetric=True)


def test_an_odd_count_removes_one_axis_vertex():
    poly = dense_polygon()  # 64 samples: the top and the bottom vertex are on the axis
    for n in (7, 9, 11):
        r = sp.simplify(poly, n, symmetric=True)
        assert r.shape.n == n and ol.is_mirror_symmetric(r.shape) and area(r.shape) == pytest.approx(area(poly), rel=1e-9)


# ---- the superellipse ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1.0, 2.0, 2.5, 4.0])
def test_the_superellipse_passes_through_its_diagonal_point(p):
    shape = ol.superellipse_shape(1.0, 1.0, p)
    paths = ol.edge_paths(shape)[0][0]  # the first quarter, from the top vertex to the left one
    mid = paths[len(paths) // 2]
    assert abs(mid[0]) ** p + abs(mid[1]) ** p == pytest.approx(1.0, abs=2e-3)
    assert ol.is_mirror_symmetric(shape) and shape.n == 4


def test_the_superellipse_with_exponent_2_is_the_circle_of_the_circle_shape():
    se = ol.superellipse_shape(0.3, 0.3, 2.0)
    c = ol.circle_shape(0.3)
    assert np.array(se.handles[0]) == pytest.approx(np.array(c.handles[0]), abs=2e-4)
    assert area(se) == pytest.approx(math.pi * 0.09, rel=1e-3)


def test_a_bad_superellipse_is_refused():
    with pytest.raises(ValueError):
        ol.superellipse_shape(0.0, 1.0)
    with pytest.raises(ValueError):
        ol.superellipse_shape(1.0, 1.0, -1.0)
