# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Evaluating a key section: areas, clearance to the skin, validity, Fit area and the default inlet sketch.

A section is a ``outline.SectionShape`` in (u, v). The *inlet* section (s = 0) lies in the inlet plane: some
vertices may be snapped onto the section curve (plane ∩ skin), and an edge between two snapped vertices follows
it. Its capture area is the in-plane area times |n . X|. Other key sections have no skin and no plane (``plane``
is None): the area is the plain in-plane area and only the outline itself is checked.

The outline is the sampled boundary, so curved edges count with their true shape.
"""
import math
from dataclasses import dataclass

import numpy as np

from ductmgr import outline as ol
from ductmgr.geometry import InletPlane, polygon_area, polygon_centroid
from ductmgr.outline import SectionShape
from ductmgr.section import (
    Section,
    distance_to_section,
    make_section,
    nearest_on_section,
    points_in_polygon,
    signed_distance,
)

N_VERTICES = 5  # the default inlet sketch is a pentagon
AREA_TOLERANCE = 0.005  # relative: Fit area must land within this, and a larger miss is flagged
AREA_SOLVE_TOLERANCE = 1e-4  # relative: the bisection inside Fit area stops here
INSIDE_TOL = 1e-6  # m: points deeper than this inside the fuselage section make the opening invalid
RESNAP_KEEP_TOL = 1e-9  # m: a snapped vertex this close to the curve is left exactly where it is
RESCALE_KEEP_TOL = 1e-11  # relative: a section this close to its target area is not rescaled
MIN_PROJECTION = 0.5  # |n . X| below this: the opening is more than 60 deg off the flow
MIN_OUTLINE_AREA = 1e-12  # m^2: below this the outline is degenerate
STRAIGHT_EDGE_SAMPLES = 24  # a straight free edge is checked at this many segments (curves use BEZIER_SAMPLES)
SCALE_RANGE = (0.05, 20.0)  # Fit area scale search range
SCALE_SCAN_POINTS = 241
DEFAULT_WIDTH_OVER_SQRT_AREA = 1.0  # default pentagon: bottom width relative to sqrt(in-plane area)
DEFAULT_HEIGHT_OVER_WIDTH = 0.8
DEFAULT_TOP_WIDTH_FRACTION = 0.8  # top corners' spacing relative to the bottom width
DEFAULT_APEX_FRACTION = 0.3  # apex height above the top edge, relative to the body height


@dataclass(frozen=True)
class Opening:
    shape: SectionShape
    outline: np.ndarray  # (M, 2) closed polyline in (u, v), curves sampled finely
    edge_paths: tuple[np.ndarray, ...]  # the path of each edge
    snapped_edges: tuple[bool, ...]
    area_plane: float  # in-plane area
    area_projected: float  # projected normal to the freestream (equals the in-plane area without a plane)
    target_area: float
    projection_factor: float  # |n . X| (1 without a plane)
    snapped_length: float  # total length of the skin-following edges
    clearance: float | None  # min distance of the free edges above the skin (negative = inside); None if unknown
    penetration: float  # depth by which the opening reaches into the fuselage (0 if none)
    centroid_uv: tuple[float, float]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def delta(self) -> float:
        """Relative difference of the projected area from the target."""
        return self.area_projected / self.target_area - 1.0


# ---- snapping --------------------------------------------------------------------------------------------
def snap_point(section: Section | None, uv, tolerance: float | None) -> tuple[tuple[float, float], bool]:
    """Snap ``uv`` onto the section curve when it is within ``tolerance`` (metres); None disables snapping."""
    uv = (float(uv[0]), float(uv[1]))
    if section is None or tolerance is None:
        return uv, False
    point, _, dist = nearest_on_section(section, uv)
    if dist <= tolerance:
        return (float(point[0]), float(point[1])), True
    return uv, False


def resnap(shape: SectionShape, section: Section | None) -> SectionShape:
    """After the section changed: put every snapped vertex on the new curve (nearest point). Without a curve
    nothing can be snapped, so all flags clear. Vertices already on the curve are left bit for bit."""
    verts, flags = [], []
    for (u, v), flag in zip(shape.vertices, shape.snapped):
        if flag and section is not None:
            point, _, dist = nearest_on_section(section, (u, v))
            verts.append((float(u), float(v)) if dist <= RESNAP_KEEP_TOL else (float(point[0]), float(point[1])))
            flags.append(True)
        else:
            verts.append((float(u), float(v)))
            flags.append(False)
    if tuple(verts) == shape.vertices and tuple(flags) == shape.snapped:
        return shape
    return shape.with_(vertices=tuple(verts), snapped=tuple(flags))


# ---- outline helpers -----------------------------------------------------------------------------------------
def _crossing(outline: np.ndarray) -> bool:
    """True if the closed polyline crosses itself (a proper crossing between non-adjacent segments)."""
    p, q = outline, np.roll(outline, -1, axis=0)
    m = len(p)
    if m < 4:
        return False

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    p1, q1 = p[:, None, :], q[:, None, :]
    p2, q2 = p[None, :, :], q[None, :, :]
    d1, d2 = orient(p1, q1, p2), orient(p1, q1, q2)
    d3, d4 = orient(p2, q2, p1), orient(p2, q2, q1)
    eps = 1e-12
    cross = (d1 * d2 < -eps) & (d3 * d4 < -eps)
    idx = np.arange(m)
    adjacent = (np.abs(idx[:, None] - idx[None, :]) <= 1) | (np.abs(idx[:, None] - idx[None, :]) == m - 1)
    return bool((cross & ~adjacent).any())


def _free_edge_samples(shape: SectionShape, paths) -> np.ndarray:
    """Points along the free edges (curves sampled finely); a snapped end vertex is on the skin by construction."""
    pts = []
    n = shape.n
    for i in range(n):
        if shape.edge_is_snapped(i):
            continue
        path = paths[i]
        if len(path) == 2:  # a straight edge: subdivide it so the minimum along it is found
            t = np.linspace(0.0, 1.0, STRAIGHT_EDGE_SAMPLES + 1)[:, None]
            path = path[0] + t * (path[1] - path[0])
        lo = 1 if shape.snapped[i] else 0
        hi = len(path) - 1 if shape.snapped[(i + 1) % n] else len(path)
        pts.append(path[lo:hi])
    return np.vstack(pts) if pts else np.empty((0, 2))


def distance_to_outline(outline: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance of points to the closed outline polyline."""
    return distance_to_section(make_section(outline, True), pts)


def evaluate(shape: SectionShape, section: Section | None, plane: InletPlane | None, target_area: float) -> Opening:
    """Areas, snapped-edge length, clearance and validity of a section.

    With a ``plane`` this is the inlet section (the skin checks apply and the area is projected normal to the
    freestream); with ``plane`` None it is a plain key section (no skin, no projection).
    """
    paths, flags, lengths = ol.edge_paths(shape, section if plane is not None else None)
    outline = ol.build_outline(paths)
    area = abs(polygon_area(outline))
    factor = plane.projection_factor if plane is not None else 1.0
    errors: list[str] = []
    warnings: list[str] = []

    clearance = None
    penetration = 0.0
    if plane is not None:
        if section is None:
            errors.append("the inlet plane does not cut the fuselage")
        else:
            samples = _free_edge_samples(shape, paths)
            if len(samples) and section.closed:
                clearance = float(signed_distance(section, samples).min())
                penetration = max(penetration, -clearance)
            if section.closed:
                inside = points_in_polygon(outline, section.points)
                if inside.any():  # skin vertices inside the opening: the body pokes through it
                    depth = distance_to_outline(outline, section.points[inside])
                    deep = depth > INSIDE_TOL
                    if deep.any():
                        penetration = max(penetration, float(depth[deep].max()))
            if penetration > INSIDE_TOL:
                errors.append(f"the opening lies inside the fuselage by up to {1000 * penetration:.1f} mm")
    if area < MIN_OUTLINE_AREA:
        errors.append("the outline has no area")
    if _crossing(outline):
        errors.append("the outline crosses itself")
    if plane is not None and factor < MIN_PROJECTION:
        warnings.append(
            f"|n.X| = {factor:.2f}: the opening is more than {math.degrees(math.acos(MIN_PROJECTION)):.0f} deg "
            "off the flow"
        )
    delta = area * factor / target_area - 1.0
    if abs(delta) > AREA_TOLERANCE:
        warnings.append(f"projected area is {100 * delta:+.2f} % from the target: use Fit area")

    centroid = tuple(float(c) for c in polygon_centroid(outline)) if area > MIN_OUTLINE_AREA else (0.0, 0.0)
    return Opening(
        shape=shape,
        outline=outline,
        edge_paths=tuple(paths),
        snapped_edges=tuple(flags),
        area_plane=float(area),
        area_projected=float(area * factor),
        target_area=float(target_area),
        projection_factor=float(factor),
        snapped_length=float(sum(l for l, f in zip(lengths, flags) if f)),
        clearance=clearance,
        penetration=float(penetration),
        centroid_uv=centroid,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


# ---- Fit area -------------------------------------------------------------------------------------------------
def snapped_edge_centroid(shape: SectionShape, section: Section | None) -> np.ndarray:
    """Centre about which Fit area scales: the (length-weighted) centroid of the skin-following edges; failing
    that the snapped vertices; failing that the vertex centroid."""
    paths, flags, _ = ol.edge_paths(shape, section)
    weights, mids = [], []
    for path, flag in zip(paths, flags):
        if flag and len(path) > 1:
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            total = seg.sum()
            if total > 0:
                weights.append(total)
                mids.append((((path[:-1] + path[1:]) / 2) * seg[:, None]).sum(axis=0) / total)
    if weights:
        return np.average(np.array(mids), axis=0, weights=np.array(weights))
    verts = np.array(shape.vertices, dtype=float)
    if any(shape.snapped):
        return verts[list(shape.snapped)].mean(axis=0)
    return verts.mean(axis=0)


def _projected_area(shape: SectionShape, section, plane: InletPlane | None) -> float:
    paths, _, _ = ol.edge_paths(shape, section if plane is not None else None)
    return abs(polygon_area(ol.build_outline(paths))) * (plane.projection_factor if plane is not None else 1.0)


def fit_area(shape: SectionShape, section: Section | None, plane: InletPlane | None, target_area: float):
    """Scale ONLY the free vertices (and their handles) about the centre of the snapped edge so the projected
    area hits the target. Returns (shape, ok, message). Snapped vertices never move."""
    if all(shape.snapped):
        return shape, False, "all vertices are snapped: nothing to scale"
    centre = snapped_edge_centroid(shape, section)

    def area_at(s: float) -> float:
        return _projected_area(ol.scale_free(shape, centre, s), section, plane)

    grid = np.geomspace(SCALE_RANGE[0], SCALE_RANGE[1], SCALE_SCAN_POINTS)
    f = np.array([area_at(s) - target_area for s in grid])
    brackets = [i for i in range(len(grid) - 1) if f[i] == 0.0 or f[i] * f[i + 1] < 0.0]
    if brackets:
        i = min(brackets, key=lambda k: abs(math.log(math.sqrt(grid[k] * grid[k + 1]))))
        lo, hi = grid[i], grid[i + 1]
        flo = f[i]
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            fm = area_at(mid) - target_area
            if abs(fm) / target_area < AREA_SOLVE_TOLERANCE:
                lo = hi = mid
                break
            if fm * flo < 0.0:
                hi = mid
            else:
                lo, flo = mid, fm
        s = math.sqrt(lo * hi)
    else:
        s = float(grid[int(np.argmin(np.abs(f)))])
    result = ol.scale_free(shape, centre, s)
    err = area_at(s) / target_area - 1.0
    if abs(err) <= AREA_TOLERANCE:
        return result, True, f"Area fitted: x{s:.3f} about the snapped edge ({100 * err:+.3f} %)"
    return result, False, f"Fit area could not reach the target (closest {100 * err:+.1f} %)"


def rescale_to_area(shape: SectionShape, target_area: float) -> SectionShape:
    """Uniformly scale a free (key) section about its area centroid so its area equals ``target_area`` exactly
    (the area scales with the square of the factor). Used to hold a section at the area law A(s)."""
    area = ol.outline_area(shape)
    if area < MIN_OUTLINE_AREA or abs(target_area / area - 1.0) < RESCALE_KEEP_TOL:
        return shape  # degenerate, or already at the target: leave it bit for bit (so rescaling is idempotent)
    centre = ol.outline_centroid(shape)
    return ol.scale_all(shape, centre, math.sqrt(target_area / area))


# ---- defaults -------------------------------------------------------------------------------------------------------
def default_sketch(section: Section | None, plane: InletPlane, target_area: float) -> SectionShape:
    """A first pentagon: bottom edge snapped on the section curve around the origin, three free vertices above,
    scaled to the target area."""
    a_plane = target_area / max(plane.projection_factor, 1e-6)
    width = DEFAULT_WIDTH_OVER_SQRT_AREA * math.sqrt(a_plane)
    height = DEFAULT_HEIGHT_OVER_WIDTH * width
    bl, br = (-width / 2.0, 0.0), (width / 2.0, 0.0)
    snapped = [False] * N_VERTICES
    if section is not None:
        (bl, _), (br, _) = snap_point(section, bl, math.inf), snap_point(section, br, math.inf)
        snapped[0] = snapped[1] = True
    v_top = max(bl[1], br[1]) + height
    top = DEFAULT_TOP_WIDTH_FRACTION * width / 2.0
    vertices = (bl, br, (top, v_top), (0.0, v_top + DEFAULT_APEX_FRACTION * height), (-top, v_top))
    shape = SectionShape.polygon(vertices, snapped)
    fitted, _, _ = fit_area(shape, section, plane, target_area)
    return fitted
