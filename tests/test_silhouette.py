# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The exact projected outline of a mesh (union of its projected triangles)."""
import numpy as np
import pytest
import pyvista as pv
import shapely

import bodies
from ductmgr import silhouette as sil

STATIONS = np.array([0.0, 0.4, 2.0, 5.0, 8.0])  # a coarse nose: one long facet from x = 0 to 0.4
RADII = np.array([0.0, 0.5, 0.9, 1.0, 0.7])


def revolve(resolution=64):
    """A body of revolution about X with the coarse profile (STATIONS, RADII); the nose is a point."""
    t = 2.0 * np.pi * np.arange(resolution) / resolution
    pts, faces = [], []
    for x, r in zip(STATIONS, RADII):
        if r == 0.0:
            pts.append((x, 0.0, 0.0))
        else:
            pts += [(x, r * np.cos(a), r * np.sin(a)) for a in t]
    ring0 = 1
    faces += [[3, 0, ring0 + k, ring0 + (k + 1) % resolution] for k in range(resolution)]  # the nose fan
    for s in range(1, len(STATIONS) - 1):
        a, b = 1 + (s - 1) * resolution, 1 + s * resolution
        for k in range(resolution):
            k1 = (k + 1) % resolution
            faces += [[3, a + k, b + k, b + k1], [3, a + k, b + k1, a + k1]]
    last = 1 + (len(STATIONS) - 2) * resolution
    cap = len(pts)
    pts.append((STATIONS[-1], 0.0, 0.0))
    faces += [[3, last + k, cap, last + (k + 1) % resolution] for k in range(resolution)]
    return pv.PolyData(np.array(pts), np.hstack(faces))


def upper_edge(ring, x):
    """The upper boundary y of a ring at abscissa x (intersection with a vertical line)."""
    line = shapely.LineString([(x, -10.0), (x, 10.0)])
    hit = shapely.Polygon(ring).intersection(line)
    return hit.bounds[3]


@pytest.mark.parametrize("across", [sil.SIDE, sil.TOP])
def test_a_coarse_nose_gives_the_analytic_silhouette_with_no_spikes(across):
    mesh = revolve()
    rings = sil.outline_rings(mesh, across)
    assert len(rings) == 1
    ring = rings[0]
    width = 2.0 * RADII.max()
    xs = np.linspace(0.01, 7.99, 400)
    analytic = np.interp(xs, STATIONS, RADII)
    got = np.array([upper_edge(ring, x) for x in xs])
    assert np.abs(got - analytic).max() < 0.005 * width  # within 0.5 % of the body width
    poly = shapely.Polygon(ring)
    assert poly.is_valid and poly.exterior.is_simple  # no self-crossing, no reversals


def test_the_nose_has_no_vertices_beyond_the_analytic_profile():
    ring = sil.outline_rings(revolve(), sil.SIDE)[0]
    x, z = ring[:, 0], ring[:, 1]
    r = np.interp(x, STATIONS, RADII)
    assert (np.abs(z) <= r + 1e-9).all()  # nothing pokes out of the profile (the old binned outline spiked here)


def test_the_ring_is_counter_clockwise_and_not_closed_by_repeating_its_first_point():
    ring = sil.outline_rings(bodies.cylinder(1.0), sil.SIDE)[0]
    x, z = ring[:, 0], ring[:, 1]
    assert 0.5 * (np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) > 0
    assert not np.allclose(ring[0], ring[-1])
    assert ring[:, 0].min() == pytest.approx(0.0) and ring[:, 0].max() == pytest.approx(8.0)
    assert ring[:, 1].max() == pytest.approx(1.0, abs=1e-3) and ring[:, 1].min() == pytest.approx(-1.0, abs=1e-3)


def test_a_chined_prism_top_view_is_its_exact_planform():
    ring = sil.outline_rings(bodies.chined_prism(), sil.TOP)[0]
    assert ring[:, 1].max() == pytest.approx(1.0) and ring[:, 1].min() == pytest.approx(-1.0)
    assert shapely.Polygon(ring).area == pytest.approx(8.0 * 2.0, rel=1e-9)  # a rectangle 8 x 2


def test_two_separate_pieces_give_two_rings_and_a_hole_is_ignored():
    a = pv.Box(bounds=(0, 1, -1, 1, -1, 1))
    b = pv.Box(bounds=(3, 4, -1, 1, -0.5, 0.5))
    merged = a.triangulate().merge(b.triangulate())
    rings = sil.outline_rings(merged, sil.SIDE)
    assert len(rings) == 2 and shapely.Polygon(rings[0]).area > shapely.Polygon(rings[1]).area  # largest first


def test_an_empty_mesh_has_no_outline():
    assert sil.outline_rings(pv.PolyData(), sil.SIDE) == []


def test_edge_on_triangles_do_not_break_the_union():
    plate = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=2, j_size=1).triangulate()  # in the x-z plane
    rings = sil.outline_rings(plate, sil.SIDE)
    assert len(rings) == 1 and shapely.Polygon(rings[0]).area == pytest.approx(2.0, rel=1e-6)
    assert sil.outline_rings(plate, sil.TOP) == []  # edge-on in the top view: a line has no area
