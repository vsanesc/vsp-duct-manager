# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Fitting the inlet outline to the fuselage skin.

Modes (``ducts.inlet.skin_fit``):

none         the planar polygon of ``geometry.inlet_polygon``, centred on the anchor; its in-plane area is the
             capture area (unchanged behaviour).
bottom_edge  planar outline whose bottom edge follows the curve (inlet plane) ∩ (skin), between the lateral
             positions of the two lower vertices; the other vertices keep the shape ratios above the skin.
flush        the planar polygon projected onto the skin along the inlet axis (non-planar outline).

For bottom_edge and flush the capture area is the outline area projected normal to the freestream (+X), and
the shape is scaled iteratively until it matches the target. The anchor is the clicked skin point: the centre
of the bottom edge for bottom_edge, the centre of the polygon for none and flush.

Units: metres. VSP axes: +X aft, +Y right, +Z up.
"""
import math
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from ductmgr.geometry import InletShape, inlet_frame, inlet_vertices, polygon_area, polygon_centroid

SKIN_FIT_MODES = ("none", "bottom_edge", "flush")
DEFAULT_SKIN_FIT = "none"  # used when the config has no ducts.inlet.skin_fit
FREESTREAM = np.array([1.0, 0.0, 0.0])  # air moves aft (+X) relative to the aircraft

FLUSH_WARNING = (
    "flush inlet: low pressure recovery and boundary-layer ingestion; not recommended for a main engine inlet"
)

AREA_TOLERANCE = 0.005  # relative: a fit worse than this is reported
AREA_ITER_TOLERANCE = 1e-3  # relative: the iteration stops below this
MAX_ITERATIONS = 40
STEP_LIMIT = 3.0  # largest scale change per iteration
STALL_LIMIT = 6  # stop after this many iterations without a (STALL_IMPROVEMENT) relative improvement
STALL_IMPROVEMENT = 0.01
PROFILE_HALF_POINTS = 60  # grid points each side of the anchor for the skin curve
PROFILE_HALF_RANGE_FACTOR = 4.0  # profile half-width, in units of sqrt(target area)
SKIN_MAX_SLOPE = 8.0  # dv/du above which the skin curve is considered to turn away (side wall)
EDGE_SAMPLES = 16  # samples per polygon edge for contact checks and the flush projection
GAP_WARN_FRACTION = 0.02  # of sqrt(target area): contact further than this from the skin is reported


@dataclass(frozen=True)
class SkinProfile:
    """Skin height v(u) in the inlet plane around the anchor (contiguous branch through the anchor)."""

    u: np.ndarray
    v: np.ndarray
    valid: bool

    def v_at(self, u):
        return np.interp(u, self.u, self.v) if self.valid else np.zeros_like(np.asarray(u, dtype=float))

    def covers(self, u_min: float, u_max: float) -> bool:
        return self.valid and self.u[0] <= u_min and u_max <= self.u[-1]


@dataclass(frozen=True)
class InletOutline:
    mode: str
    anchor: np.ndarray
    normal: np.ndarray  # inlet plane normal (facing the freestream)
    points: np.ndarray  # (N, 3) closed outline (the first point is not repeated)
    uv: np.ndarray  # (M, 2) planar outline in the inlet plane (before projection, for flush)
    planar: bool
    scale: float
    target_area: float
    area: float  # capture area as defined for the mode (see module docstring)
    planar_area: float  # in-plane area of ``uv``
    max_gap: float  # max distance of the skin-contact part of the outline from the skin
    opening_height: float  # max height of the outline above the skin, in the inlet plane (0 for flush)
    warnings: tuple[str, ...]

    @property
    def area_error(self) -> float:
        return self.area / self.target_area - 1.0


# ---- helpers ---------------------------------------------------------------------------------------
def validate_mode(mode: str) -> str:
    if mode not in SKIN_FIT_MODES:
        raise ValueError(f"skin_fit must be one of {', '.join(SKIN_FIT_MODES)} (got {mode!r})")
    return mode


def projected_area(points: np.ndarray) -> float:
    """Area of a closed outline projected normal to the freestream (onto the YZ plane)."""
    return abs(polygon_area(np.asarray(points)[:, 1:3]))


def _to_3d(uv: np.ndarray, anchor, u_hat, v_hat) -> np.ndarray:
    return np.asarray(anchor) + np.outer(uv[:, 0], u_hat) + np.outer(uv[:, 1], v_hat)


def _segments(lines: np.ndarray) -> np.ndarray:
    """(S, 2) point-index pairs from a VTK ``lines`` array (polylines are split into segments)."""
    pairs = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1 : i + 1 + n]
        pairs.extend(zip(ids[:-1], ids[1:]))
        i += n + 1
    return np.array(pairs, dtype=int).reshape(-1, 2)


def skin_profile(mesh: pv.PolyData, anchor, tilt_deg: float, half_range: float) -> SkinProfile:
    """Skin height v(u) along the inlet plane's lateral axis, following the skin branch through the anchor.

    The section of the mesh by the inlet plane is intersected with vertical (v-direction) lines on a lateral
    grid; starting at the anchor the intersection nearest the previous one is followed outward on both sides,
    stopping where the skin turns away (side wall) or ends.
    """
    u_hat, v_hat, normal = inlet_frame(tilt_deg)
    section = mesh.slice(normal=normal, origin=anchor)
    empty = SkinProfile(np.array([0.0]), np.array([0.0]), False)
    if section.n_points == 0 or section.n_lines == 0:
        return empty
    rel = np.asarray(section.points) - np.asarray(anchor)
    U, V = rel @ u_hat, rel @ v_hat
    seg = _segments(np.asarray(section.lines))
    ua, ub, va, vb = U[seg[:, 0]], U[seg[:, 1]], V[seg[:, 0]], V[seg[:, 1]]
    lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
    usable = ua != ub

    def candidates(u: float) -> np.ndarray:
        m = usable & (lo <= u) & (u <= hi)
        return va[m] + (u - ua[m]) * (vb[m] - va[m]) / (ub[m] - ua[m])

    grid = np.linspace(-half_range, half_range, 2 * PROFILE_HALF_POINTS + 1)
    du = grid[1] - grid[0]
    k0 = PROFILE_HALF_POINTS
    c0 = candidates(0.0)
    if len(c0) == 0:
        return empty
    v0 = float(c0[np.argmin(np.abs(c0))])
    values = {k0: v0}
    for step in (1, -1):
        last = v0
        k = k0 + step
        while 0 <= k < len(grid):
            c = candidates(grid[k])
            if len(c) == 0:
                break
            v = float(c[np.argmin(np.abs(c - last))])
            if abs(v - last) > SKIN_MAX_SLOPE * du:
                break
            values[k] = last = v
            k += step
    ks = sorted(values)
    return SkinProfile(grid[ks], np.array([values[k] for k in ks]), True)


def _project_along_axis(mesh: pv.PolyData, points: np.ndarray, axis: np.ndarray, reach: float):
    """Move each point along ±axis to the nearest skin intersection. Returns (points, number missed)."""
    out = np.array(points, dtype=float, copy=True)
    missed = 0
    for i, p in enumerate(points):
        hits, _ = mesh.ray_trace(p - axis * reach, p + axis * reach)
        if len(hits) == 0:
            missed += 1
            continue
        out[i] = hits[int(np.argmin(np.linalg.norm(hits - p, axis=1)))]
    return out, missed


def distance_to_skin(mesh: pv.PolyData, points: np.ndarray) -> np.ndarray:
    _, closest = mesh.find_closest_cell(np.asarray(points), return_closest_point=True)
    return np.linalg.norm(np.asarray(points) - closest, axis=1)


def _edge_samples(corners: np.ndarray, closed: bool = True) -> np.ndarray:
    """Points along the edges of a polygon (corners included), EDGE_SAMPLES per edge."""
    n = len(corners)
    edges = n if closed else n - 1
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES, endpoint=False)[:, None]
    parts = [corners[i] + t * (corners[(i + 1) % n] - corners[i]) for i in range(edges)]
    if not closed:
        parts.append(corners[-1:])
    return np.vstack(parts)


def _solve_scale(area_of, s0: float, target: float) -> tuple[float, float]:
    """Fixed-point iteration s <- s*sqrt(target/area(s)) until the area is within AREA_ITER_TOLERANCE.

    Each step is limited to a factor STEP_LIMIT so a degenerate (near-zero area) outline cannot run away; the
    best scale found is returned.
    """
    s, area = s0, area_of(s0)
    best = (abs(area / target - 1.0) if area > 0.0 else math.inf, s, area)
    stalled = 0
    for _ in range(MAX_ITERATIONS):
        if area > 0.0 and abs(area / target - 1.0) < AREA_ITER_TOLERANCE:
            break
        factor = math.sqrt(target / area) if area > 0.0 else STEP_LIMIT
        s *= min(max(factor, 1.0 / STEP_LIMIT), STEP_LIMIT)
        area = area_of(s)
        err = abs(area / target - 1.0) if area > 0.0 else math.inf
        if err < best[0] * (1.0 - STALL_IMPROVEMENT):
            stalled = 0
        else:
            stalled += 1
            if stalled >= STALL_LIMIT:  # no progress: give up early, the caller reports the miss
                break
        if err < best[0]:
            best = (err, s, area)
    return best[1], best[2]


def _bottom_edge_uv(shape: InletShape, s: float, profile: SkinProfile) -> tuple[np.ndarray, int]:
    """Outline (u, v) with the bottom edge on the skin curve; also the number of bottom-curve points."""
    wb, wt, h = shape.width_bottom, shape.width_top, shape.height
    ul, ur = -s * wb / 2.0, s * wb / 2.0
    inside = profile.u[(profile.u > ul) & (profile.u < ur)] if profile.valid else np.array([])
    ub = np.concatenate([[ul], inside, [ur]])
    bottom = np.c_[ub, profile.v_at(ub)]
    v_top = float(profile.v_at(0.0)) + s * h
    top = [(s * wt / 2.0, v_top)]
    if not shape.is_quadrilateral:
        top.append((s * shape.skew * wt, v_top + s * h * shape.apex_height))
    top.append((-s * wt / 2.0, v_top))
    return np.vstack([bottom, np.array(top)]), len(bottom)


# ---- main entry point --------------------------------------------------------------------------------
def fit_inlet(
    mesh: pv.PolyData,
    anchor,
    tilt_deg: float,
    shape: InletShape,
    target_area: float,
    mode: str = DEFAULT_SKIN_FIT,
) -> InletOutline:
    """Build the inlet outline for ``mode`` at ``anchor`` (a point on the skin) and match ``target_area``.

    ``scale`` in the result multiplies the unscaled shape vertices (``geometry.inlet_vertices``) in every mode.
    """
    validate_mode(mode)
    if not target_area > 0.0:
        raise ValueError("target capture area must be positive")
    anchor = np.asarray(anchor, dtype=float)
    u_hat, v_hat, normal = inlet_frame(tilt_deg)
    size = math.sqrt(target_area)
    warns: list[str] = []
    profile = skin_profile(mesh, anchor, tilt_deg, PROFILE_HALF_RANGE_FACTOR * size)
    reach = float(np.linalg.norm(np.array(mesh.bounds[1::2]) - np.array(mesh.bounds[0::2])))

    vertices = inlet_vertices(shape)
    s0 = math.sqrt(target_area / polygon_area(vertices))  # scale at which the planar polygon has the target area
    centred = vertices - polygon_centroid(vertices)

    if mode == "none":
        scale = s0
        uv = centred * scale
        points = _to_3d(uv, anchor, u_hat, v_hat)
        area = planar = abs(polygon_area(uv))
        contact = _edge_samples(points[:2], closed=False)  # the bottom edge
        outline_pts, planar_flag = points, True
        top_uv = uv[2:]
    elif mode == "bottom_edge":

        def area_of(s: float) -> float:
            uv, _ = _bottom_edge_uv(shape, s, profile)
            return projected_area(_to_3d(uv, anchor, u_hat, v_hat))

        scale, area = _solve_scale(area_of, s0, target_area)
        uv, n_bottom = _bottom_edge_uv(shape, scale, profile)
        points = _to_3d(uv, anchor, u_hat, v_hat)
        planar = abs(polygon_area(uv))
        contact = points[:n_bottom]
        outline_pts, planar_flag = points, True
        top_uv = uv[n_bottom:]
        if not profile.valid:
            warns.append("skin curve not found in the inlet plane: the bottom edge is straight")
        elif not profile.covers(-scale * shape.width_bottom / 2.0, scale * shape.width_bottom / 2.0):
            warns.append("the skin curve does not cover the inlet width: the bottom edge is extrapolated")
    else:  # flush

        def build(k: float):
            corners = _to_3d(centred * k, anchor, u_hat, v_hat)
            projected, missed = _project_along_axis(mesh, _edge_samples(corners), normal, reach)
            return projected, missed

        def area_of(k: float) -> float:
            return projected_area(build(k)[0])

        scale, area = _solve_scale(area_of, s0, target_area)
        projected, missed = build(scale)
        uv = centred * scale
        planar = abs(polygon_area(uv))
        outline_pts, planar_flag = projected, False
        contact = projected
        top_uv = uv
        warns.append(FLUSH_WARNING)
        if missed:
            warns.append(
                f"flush: {missed} of {len(projected)} outline points do not reach the skin along the inlet "
                "axis (increase the tilt so the plane is inclined to the skin)"
            )

    gap = float(distance_to_skin(mesh, contact).max())
    if mode == "flush":
        height = 0.0
    else:
        height = float(np.max(top_uv[:, 1] - profile.v_at(top_uv[:, 0])))
    if mode != "none" and abs(area / target_area - 1.0) > AREA_TOLERANCE:
        warns.append(
            f"capture area {area:.5f} m^2 is {100 * (area / target_area - 1.0):+.2f} % from the target "
            f"{target_area:.5f} m^2 (limit {100 * AREA_TOLERANCE:.1f} %): the fit did not converge"
            + (
                " (a flush outline on skin that is nearly parallel to the freestream has almost no frontal area)"
                if mode == "flush"
                else ""
            )
        )
    if mode != "none" and gap > GAP_WARN_FRACTION * size:
        warns.append(f"the outline is up to {gap:.4f} m from the skin")

    return InletOutline(
        mode=mode,
        anchor=anchor,
        normal=normal,
        points=outline_pts,
        uv=uv,
        planar=planar_flag,
        scale=float(scale),
        target_area=float(target_area),
        area=float(area),
        planar_area=float(planar),
        max_gap=gap,
        opening_height=height,
        warnings=tuple(warns),
    )
