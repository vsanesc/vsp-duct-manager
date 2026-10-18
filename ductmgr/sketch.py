# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The inlet opening as a 5-vertex sketch in the inlet plane's (u, v) coordinates.

Vertices may be *snapped* onto the section curve (plane ∩ skin). An edge whose two end vertices are both
snapped follows the section curve between them (the shorter arc, chine corners included); every other edge is
a straight line. The capture area is the opening area projected normal to the freestream: the in-plane area
times |n . X|.
"""
import math
from dataclasses import dataclass

import numpy as np

from ductmgr.geometry import InletPlane, polygon_area, polygon_centroid
from ductmgr.section import (
    Section,
    arc_path,
    nearest_on_section,
    points_in_polygon,
    distance_to_section,
    signed_distance,
)

N_VERTICES = 5
AREA_TOLERANCE = 0.005  # relative: Fit area must land within this, and a larger miss is flagged
AREA_SOLVE_TOLERANCE = 1e-4  # relative: the bisection inside Fit area stops here
INSIDE_TOL = 1e-6  # m: points deeper than this inside the fuselage section make the opening invalid
RESNAP_KEEP_TOL = 1e-9  # m: a snapped vertex this close to the curve is left exactly where it is
EDGE_SAMPLES = 24  # samples per free edge for clearance and validity
MIN_PROJECTION = 0.5  # |n . X| below this: the opening is more than 60 deg off the flow
SCALE_RANGE = (0.05, 20.0)  # Fit area scale search range
SCALE_SCAN_POINTS = 241
DEFAULT_WIDTH_OVER_SQRT_AREA = 1.0  # default pentagon: bottom width relative to sqrt(in-plane area)
DEFAULT_HEIGHT_OVER_WIDTH = 0.8
DEFAULT_TOP_WIDTH_FRACTION = 0.8  # top corners' spacing relative to the bottom width
DEFAULT_APEX_FRACTION = 0.3  # apex height above the top edge, relative to the body height


@dataclass(frozen=True)
class Opening:
    outline: np.ndarray  # (M, 2) closed polyline in (u, v)
    edge_paths: tuple[np.ndarray, ...]  # the path of each of the 5 edges
    snapped_edges: tuple[bool, ...]
    area_plane: float  # in-plane area
    area_projected: float  # projected normal to the freestream
    target_area: float
    projection_factor: float  # |n . X|
    snapped_length: float  # total length of the snapped edges
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


def resnap(vertices, snapped, section: Section | None):
    """After the section changed: put every snapped vertex on the new curve (nearest point). Without a curve
    nothing can be snapped, so all flags clear."""
    out, flags = [], []
    for (u, v), flag in zip(vertices, snapped):
        if flag and section is not None:
            point, _, dist = nearest_on_section(section, (u, v))
            # already on the curve: leave it bit for bit (so re-snapping is idempotent and creates no change)
            out.append((float(u), float(v)) if dist <= RESNAP_KEEP_TOL else (float(point[0]), float(point[1])))
            flags.append(True)
        else:
            out.append((float(u), float(v)))
            flags.append(False)
    return tuple(out), tuple(flags)


# ---- outline ------------------------------------------------------------------------------------------------
def edge_paths(vertices, snapped, section: Section | None):
    """(paths, snapped_edge flags, lengths): the polyline of each edge and its length."""
    paths, flags, lengths = [], [], []
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        a, b = np.array(vertices[i], dtype=float), np.array(vertices[j], dtype=float)
        if section is not None and snapped[i] and snapped[j]:
            _, s0, _ = nearest_on_section(section, a)
            _, s1, _ = nearest_on_section(section, b)
            path, length = arc_path(section, s0, s1)
            path = path.copy()
            path[0], path[-1] = a, b  # exactly the vertices (they are on the curve to rounding)
            paths.append(path)
            flags.append(True)
            lengths.append(length)
        else:
            paths.append(np.array([a, b]))
            flags.append(False)
            lengths.append(float(np.linalg.norm(b - a)))
    return paths, flags, lengths


def build_outline(paths) -> np.ndarray:
    """The closed outline: every edge path without its last point (it is the next edge's first)."""
    return np.vstack([p[:-1] for p in paths])


def _outline_area(vertices, snapped, section, plane: InletPlane) -> tuple[float, float]:
    paths, _, _ = edge_paths(vertices, snapped, section)
    area = abs(polygon_area(build_outline(paths)))
    return area, area * plane.projection_factor


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


def _free_edge_samples(vertices, snapped) -> np.ndarray:
    """Points along the free edges; endpoints only where the vertex is itself free."""
    pts = []
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        if snapped[i] and snapped[j]:
            continue
        a, b = np.array(vertices[i], dtype=float), np.array(vertices[j], dtype=float)
        t = np.linspace(0.0, 1.0, EDGE_SAMPLES)
        for k, tk in enumerate(t):
            if (k == 0 and snapped[i]) or (k == len(t) - 1 and snapped[j]):
                continue  # a snapped end is on the skin by construction
            pts.append(a + tk * (b - a))
    return np.array(pts).reshape(-1, 2)


def evaluate(vertices, snapped, section: Section | None, plane: InletPlane, target_area: float) -> Opening:
    """Areas, snapped-edge length, clearance and validity of the sketch."""
    paths, flags, lengths = edge_paths(vertices, snapped, section)
    outline = build_outline(paths)
    area = abs(polygon_area(outline))
    factor = plane.projection_factor
    errors: list[str] = []
    warnings: list[str] = []

    clearance = None
    penetration = 0.0
    if section is None:
        errors.append("the inlet plane does not cut the fuselage")
    else:
        samples = _free_edge_samples(vertices, snapped)
        if len(samples) and section.closed:
            clearance = float(signed_distance(section, samples).min())
            penetration = max(penetration, -clearance)
        if section.closed:
            inside = points_in_polygon(outline, section.points)
            if inside.any():  # skin vertices inside the opening: the body pokes through it
                depth = distance_to_section_outline(outline, section.points[inside])
                deep = depth > INSIDE_TOL
                if deep.any():
                    penetration = max(penetration, float(depth[deep].max()))
        if penetration > INSIDE_TOL:
            errors.append(f"the opening lies inside the fuselage by up to {1000 * penetration:.1f} mm")
    if _crossing(outline):
        errors.append("the outline crosses itself")
    if factor < MIN_PROJECTION:
        warnings.append(
            f"|n.X| = {factor:.2f}: the opening is more than {math.degrees(math.acos(MIN_PROJECTION)):.0f} deg "
            "off the flow"
        )
    delta = area * factor / target_area - 1.0
    if abs(delta) > AREA_TOLERANCE:
        warnings.append(f"projected area is {100 * delta:+.2f} % from the target: use Fit area")

    centroid = tuple(float(c) for c in polygon_centroid(outline)) if area > 0 else (0.0, 0.0)
    return Opening(
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


def distance_to_section_outline(outline: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance of points to the closed outline polyline."""
    from ductmgr.section import make_section

    return distance_to_section(make_section(outline, True), pts)


# ---- Fit area -------------------------------------------------------------------------------------------------
def snapped_edge_centroid(vertices, snapped, section: Section | None) -> np.ndarray:
    """Centre about which Fit area scales: the (length-weighted) centroid of the snapped edges; failing that
    the snapped vertices; failing that the vertex centroid."""
    paths, flags, lengths = edge_paths(vertices, snapped, section)
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
    verts = np.array(vertices, dtype=float)
    if any(snapped):
        return verts[list(snapped)].mean(axis=0)
    return verts.mean(axis=0)


def fit_area(vertices, snapped, section: Section | None, plane: InletPlane, target_area: float):
    """Scale ONLY the free vertices about the centre of the snapped edge so the projected area hits the target.

    Returns (vertices, ok, message). Snapped vertices never move.
    """
    verts = np.array(vertices, dtype=float)
    free = [i for i, flag in enumerate(snapped) if not flag]
    if not free:
        return tuple(map(tuple, verts)), False, "all vertices are snapped: nothing to scale"
    centre = snapped_edge_centroid(vertices, snapped, section)

    def scaled(s: float) -> np.ndarray:
        out = verts.copy()
        out[free] = centre + s * (verts[free] - centre)
        return out

    def projected(s: float) -> float:
        return _outline_area(tuple(map(tuple, scaled(s))), snapped, section, plane)[1]

    grid = np.geomspace(SCALE_RANGE[0], SCALE_RANGE[1], SCALE_SCAN_POINTS)
    f = np.array([projected(s) - target_area for s in grid])
    brackets = [i for i in range(len(grid) - 1) if f[i] == 0.0 or f[i] * f[i + 1] < 0.0]
    if brackets:
        i = min(brackets, key=lambda k: abs(math.log(math.sqrt(grid[k] * grid[k + 1]))))
        lo, hi = grid[i], grid[i + 1]
        flo = f[i]
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            fm = projected(mid) - target_area
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
    result = tuple(map(tuple, scaled(s)))
    final = projected(s)
    err = final / target_area - 1.0
    if abs(err) <= AREA_TOLERANCE:
        return result, True, f"Area fitted: x{s:.3f} about the snapped edge ({100 * err:+.3f} %)"
    return result, False, f"Fit area could not reach the target (closest {100 * err:+.1f} %)"


# ---- defaults -------------------------------------------------------------------------------------------------------
def default_sketch(section: Section | None, plane: InletPlane, target_area: float):
    """A first pentagon: bottom edge snapped on the section curve around the origin, three free vertices above,
    scaled to the target area. Returns (vertices, snapped)."""
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
    vertices = tuple((float(u), float(v)) for u, v in vertices)
    fitted, _, _ = fit_area(vertices, tuple(snapped), section, plane, target_area)
    return fitted, tuple(snapped)
