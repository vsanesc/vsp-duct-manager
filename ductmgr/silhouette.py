# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The exact projected outline of a triangle mesh in the 2D Side (x-z) and Top (x-y) views.

Every triangle is projected onto the view plane and the outline is the boundary of the union of all of them, so a
coarsely tessellated nose gives the true silhouette (no binning of vertices, no spikes)."""
import numpy as np
import pyvista as pv
import shapely
from shapely.geometry.polygon import orient

SNAP_FRACTION = 1e-9  # of the body size: projected points closer than this are the same point
MIN_AREA_FRACTION = 1e-14  # projected triangles with less area than this (of the body's size squared) are edge-on
SIMPLIFY_FRACTION = 1e-7  # of the body size: collinear / redundant outline points are dropped

SIDE, TOP = 2, 1  # the world axis shown vertically: z in the side view, y in the top view


def projected_polygon(mesh: pv.PolyData, across: int) -> shapely.Geometry:
    """The union of the mesh triangles projected onto (x, axis ``across``), as a shapely geometry."""
    tri = mesh.triangulate()
    if tri.n_cells == 0:
        return shapely.Polygon()
    pts = np.asarray(tri.points, dtype=float)[:, [0, across]]
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    size = float(np.ptp(pts, axis=0).max()) or 1.0
    grid = size * SNAP_FRACTION
    pts = np.round(pts / grid) * grid  # snap: coincident edges of neighbouring triangles match exactly
    c = pts[faces]  # (n, 3, 2)
    area2 = (c[:, 1, 0] - c[:, 0, 0]) * (c[:, 2, 1] - c[:, 0, 1]) - (c[:, 2, 0] - c[:, 0, 0]) * (c[:, 1, 1] - c[:, 0, 1])
    keep = np.abs(area2) > 2.0 * MIN_AREA_FRACTION * size * size  # edge-on triangles project to a line
    if not keep.any():
        return shapely.Polygon()
    c = c[keep]
    rings = np.concatenate([c, c[:, :1]], axis=1)  # closed rings (n, 4, 2)
    polys = shapely.polygons(rings)
    return shapely.union_all(polys)


def outline_rings(mesh: pv.PolyData, across: int) -> list[np.ndarray]:
    """The outline of the projected mesh: one counter-clockwise ring (closed by its first point, not repeated) per
    separate piece; holes are ignored (the silhouette of a body has none that matter)."""
    geom = projected_polygon(mesh, across)
    if geom.is_empty:
        return []
    size = float(np.ptp(np.asarray(mesh.points)[:, [0, across]], axis=0).max()) or 1.0
    parts = [geom] if geom.geom_type == "Polygon" else [g for g in geom.geoms if g.geom_type == "Polygon"]
    rings = []
    for poly in parts:
        poly = orient(shapely.Polygon(poly.exterior).simplify(size * SIMPLIFY_FRACTION), 1.0)
        if poly.is_empty:
            continue
        rings.append(np.asarray(poly.exterior.coords)[:-1])
    rings.sort(key=lambda r: -abs(_area(r)))
    return rings


def _area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
