# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Resampling and correspondence between key sections: arc-length parametrisation from a reference vertex."""
import math

import numpy as np
import pytest

from ductmgr import interp
from ductmgr import outline as ol
from ductmgr.geometry import polygon_area, polygon_centroid
from ductmgr.outline import SectionShape

N = 60


def regular_pentagon(apex_up=True, start=0, radius=1.0):
    """A regular pentagon, counter-clockwise; ``start`` rotates which vertex is listed first."""
    offset = math.pi / 2 if apex_up else -math.pi / 2
    verts = [(radius * math.cos(offset + 2 * math.pi * k / 5), radius * math.sin(offset + 2 * math.pi * k / 5))
             for k in range(5)]
    verts = verts[start:] + verts[:start]
    return SectionShape.polygon(verts)


def pair(shape):
    return ol.build_outline(ol.edge_paths(shape)[0]), np.array(shape.vertices)


def angles(points):
    c = polygon_centroid(points)
    return np.degrees(np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0]))


def angular_gap(a, b):
    """Smallest signed angle difference (degrees) between corresponding directions."""
    return (a - b + 180.0) % 360.0 - 180.0


# ---- the reference vertex --------------------------------------------------------------------------------------
def test_the_reference_vertex_is_the_one_closest_to_plus_v():
    verts = [(0.0, -1.0), (1.0, 0.0), (0.4, 1.0), (-1.0, 0.1)]
    assert interp.reference_vertex(verts) == 2  # (0.4, 1.0) is nearest to straight up
    assert interp.reference_vertex([(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)]) == 3


def test_ties_between_symmetric_vertices_go_to_the_smaller_u():
    square = [(1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)]  # two top vertices, equally close to +v
    assert interp.reference_vertex(square) == 2  # (-1, 1)
    assert interp.reference_vertex(square[::-1]) == 1


def test_the_reference_is_independent_of_which_vertex_is_listed_first():
    ref = {tuple(np.round(regular_pentagon(start=s).vertices[interp.reference_vertex(regular_pentagon(start=s).vertices)], 9))
           for s in range(5)}
    assert ref == {(0.0, 1.0)}  # always the apex


# ---- resampling ----------------------------------------------------------------------------------------------------
def test_the_samples_are_counter_clockwise_equally_spaced_in_arc_length_and_start_at_the_reference():
    outline, verts = pair(regular_pentagon())
    r = interp.resample(outline, verts, N)
    assert r.shape == (N, 2) and r[0] == pytest.approx([0.0, 1.0], abs=1e-12)  # the apex vertex
    assert polygon_area(r) > 0
    seg = np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1)
    # equal spacing along the boundary (the chord of a sample pair across a corner is a bit shorter)
    perimeter = 5 * 2 * math.sin(math.pi / 5)
    assert seg.sum() == pytest.approx(perimeter, rel=5e-3) and seg.max() / seg.min() < 1.2


def test_a_clockwise_outline_gives_the_same_samples():
    outline, verts = pair(regular_pentagon())
    a = interp.resample(outline, verts, N)
    b = interp.resample(outline[::-1].copy(), verts[::-1].copy(), N)
    assert b == pytest.approx(a, abs=1e-9)


@pytest.mark.parametrize("start", range(5))
def test_the_listing_order_of_the_vertices_does_not_matter(start):
    a = interp.resample(*pair(regular_pentagon(start=0)), N)
    b = interp.resample(*pair(regular_pentagon(start=start)), N)
    assert b == pytest.approx(a, abs=1e-9)


def test_resampling_keeps_the_shape():
    shape = ol.circle_shape(1.0)
    r = interp.resample(*pair(shape), 96)
    assert np.abs(np.hypot(r[:, 0], r[:, 1]) - 1.0).max() < 5e-4
    assert abs(polygon_area(r)) == pytest.approx(math.pi, rel=2e-3)
    with pytest.raises(ValueError):
        interp.resample(*pair(shape), 2)


# ---- no twist ---------------------------------------------------------------------------------------------------------
def test_a_pentagon_and_a_circle_correspond_without_twist():
    pent = regular_pentagon(apex_up=True)
    circle = ol.circle_shape(1.0)
    a, b = interp.correspond([pair(pent), pair(circle)], N)
    gap = angular_gap(angles(a), angles(b))
    # no systematic rotation (mean signed offset ~ 0); what remains (about 2 degrees at most) is the difference
    # between arc length and angle along a pentagon edge
    assert abs(gap.mean()) < 0.05 and np.abs(gap).max() < 3.0
    # ...so blending the two keeps every point close to its direction
    mid = interp.blend(a, b, 0.5)
    assert np.abs(angular_gap(angles(mid), angles(a))).max() < 2.0
    assert abs(polygon_area(mid)) > 0 and not self_crossing(mid)


def self_crossing(points):
    from ductmgr import sketch as sk

    return sk._crossing(points)


def test_pairing_by_index_would_twist_but_the_reference_vertex_does_not():
    # an apex-DOWN pentagon listed from its bottom vertex, against the circle whose first point is its top
    pent, circle = regular_pentagon(apex_up=False), ol.circle_shape(1.0)
    (out_p, _), (out_c, _) = pair(pent), pair(circle)
    assert pent.vertices[0][1] < -0.9  # the first listed vertex is the bottom one
    naive = abs(angular_gap(angles(out_p)[0], angles(out_c)[0]))
    assert naive == pytest.approx(180.0, abs=1.0)  # pairing index 0 with index 0 matches opposite sides: a half turn
    a, b = interp.correspond([pair(pent), pair(circle)], N)
    ours = np.abs(angular_gap(angles(a), angles(b)))
    assert ours.max() < 40.0  # at worst half a vertex spacing (36 degrees): this pentagon has no vertex at the top


@pytest.mark.parametrize("n_vertices", [3, 4, 6, 8])
def test_regular_polygons_correspond_to_the_circle_without_twist(n_vertices):
    verts = [(math.cos(math.pi / 2 + 2 * math.pi * k / n_vertices), math.sin(math.pi / 2 + 2 * math.pi * k / n_vertices))
             for k in range(n_vertices)]
    poly = SectionShape.polygon(verts)
    a, b = interp.correspond([pair(poly), pair(ol.circle_shape(1.0))], N)
    # a polygon with a vertex at the top against the symmetric circle: no net rotation, and the offset stays
    # below the arc-length-versus-angle difference of the coarsest polygon (a triangle: about 11 degrees)
    gap = angular_gap(angles(a), angles(b))
    assert abs(gap.mean()) < 0.5 and np.abs(gap).max() < 12.0


def test_a_translated_or_scaled_section_resamples_consistently():
    shape = regular_pentagon()
    base = interp.resample(*pair(shape), N)
    moved = interp.resample(*pair(ol.translate(shape, (3.0, -2.0))), N)
    assert moved == pytest.approx(base + [3.0, -2.0], abs=1e-9)
    big = interp.resample(*pair(ol.scale_all(shape, (0.0, 0.0), 2.5)), N)
    assert big == pytest.approx(2.5 * base, abs=1e-9)


def test_blend_is_linear_per_point():
    a, b = np.arange(12.0).reshape(6, 2), np.arange(12.0).reshape(6, 2) + 4
    assert interp.blend(a, b, 0.0) == pytest.approx(a) and interp.blend(a, b, 1.0) == pytest.approx(b)
    assert interp.blend(a, b, 0.25) == pytest.approx(a + 1.0)


def test_an_irregular_inlet_like_pentagon_morphs_into_the_circle_without_folding():
    pent = SectionShape.polygon([(-0.092, -0.075), (0.143, 0.042), (0.134, 0.319), (-0.002, 0.400), (-0.138, 0.319)])
    circle = ol.circle_shape(0.28, centre=(0.0, 0.15))
    a, b = interp.correspond([pair(pent), pair(circle)], N)
    for t in np.linspace(0.0, 1.0, 11):
        m = interp.blend(a, b, t)
        assert polygon_area(m) > 0 and not self_crossing(m)  # a valid simple outline all the way
