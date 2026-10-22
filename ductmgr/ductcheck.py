# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Live checks of the duct bodies against the fuselage and the other Geoms.

The model intended in OpenVSP is (fuselage ∪ positives) − negatives. Everything here is computed from meshes:
signed distances (negative inside) of body vertices to the closed fuselage / Geom meshes, and straight rays
through the negative bores.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ductmgr import duct as dm
from ductmgr.areacurve import AreaCurves
from ductmgr.section import points_in_polygon
from ductmgr.solids import RayCaster, Solid

DEFAULT_MIN_CLEARANCE = 0.02  # m, when checks.min_clearance is not configured
DEFAULT_MAX_HALF_ANGLE_DEG = 6.0  # when checks.max_diffuser_half_angle_deg is not configured
DEFAULT_LIP_LENGTH = 0.15  # m, when checks.lip_length is not configured
COVERED_TOL = 1e-6  # m: a skin point deeper than this inside the positive is not on the outer boundary
OUTSIDE_TOL = 1e-6  # m: a vertex further outside than this is outside the skin
INSIDE_TOL = 1e-6  # m: a vertex deeper than this is inside another Geom
LOS_GRID = 15  # face sample points per side (those inside the disc are used)
LOS_FACE_FRACTION = 0.98  # sampled disc radius / face radius (the face polygon is slightly inside its circle)
LOS_END_TOL = 1e-6  # m: rays stop this far from their end points, so touching the wall at the face is not a hit


@dataclass(frozen=True)
class CheckSettings:
    min_clearance: float = DEFAULT_MIN_CLEARANCE
    max_half_angle_deg: float = DEFAULT_MAX_HALF_ANGLE_DEG
    lip_length: float = DEFAULT_LIP_LENGTH

    @classmethod
    def from_config(cls, cfg: dict) -> "CheckSettings":
        c = (cfg or {}).get("checks") or {}

        def pick(key, default):
            v = c.get(key)
            return default if v is None else float(v)

        return cls(pick("min_clearance", DEFAULT_MIN_CLEARANCE),
                   pick("max_diffuser_half_angle_deg", DEFAULT_MAX_HALF_ANGLE_DEG),
                   pick("lip_length", DEFAULT_LIP_LENGTH))


Finding = tuple[str, str, str]  # (label, value, level): level is "info", "pass", "warn" or "error"


@dataclass
class DuctChecks:
    findings: list[Finding] = field(default_factory=list)
    external_length: float = 0.0
    external_area: float = 0.0
    external_extent: float = 0.0  # distance from the opening to the last external point
    external_unplanned: bool = False
    min_wall: float | None = None
    wall_ok: bool | None = None
    external_wall: float | None = None  # wall thickness measured where the positive is outside the skin
    intersections: list[str] = field(default_factory=list)
    los_x: float | None = None  # fraction of the face seen along +X through the opening
    los_n: float | None = None  # ... along the inlet normal
    los_any: float | None = None
    max_curvature: float = 0.0
    half_angle_max: float = 0.0
    half_angle_exceeded: list[str] = field(default_factory=list)
    area_decreasing: list[str] = field(default_factory=list)
    offset: float = 0.0
    offset_over_length: float = 0.0
    external_cells: dict = field(default_factory=dict)  # body name -> bool per cell (outside the fuselage)
    wall_points: dict = field(default_factory=dict)  # negative body name -> wall left per mesh point
    intersecting: set = field(default_factory=set)  # names of bodies that intersect another Geom

    def add(self, label: str, value: str, level: str = "info") -> None:
        self.findings.append((label, value, level))


class BodySolids(dict):
    """Distance queries for the bodies, built once per check run and only for the bodies that are asked for."""

    def __init__(self, bodies: dict):
        super().__init__()
        self.bodies = bodies

    def __missing__(self, key):
        name, lateral = key if isinstance(key, tuple) else (key, False)
        body = self.bodies[name]
        self[key] = Solid(body.lateral if lateral else body.mesh)
        return self[key]


def branch_names(bodies: dict) -> list[str]:
    """The ``<name>`` of every ``neg_branch_<name>`` body (mirrored twins included)."""
    return [n[len("neg_branch_"):] for n in bodies if n.startswith("neg_branch_")]


def refine_rings(rings: np.ndarray, k: int = dm.REFINE_SUBDIVISIONS) -> tuple[np.ndarray, np.ndarray]:
    """Rings with ``k - 1`` linearly interpolated rings between each pair; also the index map (fractional ring
    positions in the original ring list)."""
    r = len(rings)
    pos = np.linspace(0.0, r - 1.0, (r - 1) * k + 1)
    lo = np.minimum(pos.astype(int), r - 2)
    f = (pos - lo)[:, None, None]
    return rings[lo] * (1 - f) + rings[lo + 1] * f, pos


def _interp_dist(dist: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return np.interp(pos, np.arange(len(dist)), dist)


def _tri_areas(mesh) -> np.ndarray:
    return mesh.compute_cell_sizes(length=False, area=True, volume=False)["Area"]


def external_region(duct: dm.Duct, bodies: dict, fuselage: Solid, c: DuctChecks, s: CheckSettings) -> None:
    """Where the positive lies outside the fuselage skin: length, wetted area, extent, unplanned bump.

    The coarse station rings are tested first; only the segments that come within one station spacing of the skin
    are refined and tested again."""
    total_len = total_area = 0.0
    extent = 0.0
    for name in [f"pos_branch_{b}" for b in branch_names(bodies)] + ["pos_trunk"]:
        body = bodies[name]
        rings, dist = body.rings, body.dist
        if name == "pos_trunk":  # the trunk measures from the junction: add the way from the nearest opening
            dist = dist + duct.nearest_opening_length()
        n_r, n_p, _ = rings.shape
        sd = fuselage.signed(rings.reshape(-1, 3)).reshape(n_r, n_p)
        margin = float(np.diff(dist).max()) if n_r > 1 else 0.0
        near = sd.max(axis=1) > -margin
        out_coarse = (sd > OUTSIDE_TOL).any(axis=1)
        if not near.any():
            continue
        seg_near = near[:-1] | near[1:]
        for i in np.flatnonzero(seg_near):  # refine only the segments close to the skin
            sub, pos = refine_rings(rings[i:i + 2])
            d = _interp_dist(dist[i:i + 2], pos)
            sdf = fuselage.signed(sub.reshape(-1, 3)).reshape(len(sub), n_p)
            out = (sdf > OUTSIDE_TOL).any(axis=1)
            total_len += float(np.diff(d)[(out[:-1] | out[1:])].sum())
            if out.any():
                extent = max(extent, float(d[out].max()))
            lat = dm.tube_mesh(sub, False)
            flag = fuselage.signed(lat.cell_centers().points) > OUTSIDE_TOL
            total_area += float(_tri_areas(lat)[flag].sum())
        del out_coarse
    for name, body in bodies.items():
        if name.startswith("pos_"):
            mesh = body.mesh
            vsd = fuselage.signed(mesh.points)
            tri = mesh.faces.reshape(-1, 4)[:, 1:]
            maybe = vsd[tri].max(axis=1) > -float(np.diff(body.dist).max())
            flags = np.zeros(mesh.n_cells, dtype=bool)
            if maybe.any():
                flags[maybe] = fuselage.signed(mesh.cell_centers().points[maybe]) > OUTSIDE_TOL
            c.external_cells[name] = flags
    c.external_length, c.external_area, c.external_extent = total_len, total_area, extent
    c.external_unplanned = extent > s.lip_length + 1e-9
    level = "warn" if c.external_unplanned else "pass"
    if total_len == 0.0 and total_area == 0.0:
        c.add("External region", "none: the positive is inside the skin", "info")
    else:
        c.add("External region", f"length {total_len:.3f} m, wetted area {total_area:.5f} m^2, reaches "
              f"{extent:.3f} m from the opening (lip_length {s.lip_length:g} m)", level)
    if c.external_unplanned:
        c.add("", f"WARN the positive leaves the skin {extent:.3f} m aft of the opening, beyond checks.lip_length "
              f"({s.lip_length:g} m): unplanned bump", "warn")


def wall_left(duct: dm.Duct, bodies: dict, fuselage: Solid, c: DuctChecks, s: CheckSettings,
              solids: BodySolids | None = None) -> None:
    """The negative must lie inside (fuselage ∪ positive), and ``checks.min_clearance`` applies to the INTERNAL
    part only: the distance from the negative (the inner wall) to the skin where the positive is inside the skin.
    Where the positive is outside the skin (the external region) the wall is ``wall_thickness`` by definition; it
    is reported for information and never compared with the clearance."""
    internal_min = math.inf
    union_min = math.inf  # the wall of the union fuselage ∪ positive around the negative (negative = it leaves)
    solids = solids if solids is not None else BodySolids(bodies)
    external_wall = math.inf
    pairs = [(f"neg_branch_{b}", f"pos_branch_{b}") for b in branch_names(bodies)] + [("neg_trunk", "pos_trunk")]
    for neg_name, pos_name in pairs:
        neg, pos = bodies[neg_name], bodies[pos_name]
        solid_p = solids[pos_name]
        lateral_p = solids[(pos_name, True)]
        rings = neg.rings
        r, n, _ = rings.shape
        pts = rings.reshape(-1, 3)
        d_f = fuselage.signed(pts)
        d_p = solid_p.signed(pts)
        wall_f = -d_f
        lat = lateral_p.distance(pts)
        wall_p = np.where(d_p < 0.0, lat, -d_p)
        union = np.maximum(wall_f, wall_p).reshape(r, n)
        # A negative vertex is EXTERNAL when the nearest boundary of (fuselage ∪ positive) is the part of the
        # positive outside the skin (the wall there is wall_thickness), INTERNAL when it is the skin itself.
        flags = c.external_cells.get(pos_name)
        if flags is not None and flags.any():
            ext_surface = Solid(pos.mesh.extract_cells(np.flatnonzero(flags)).extract_surface(algorithm="dataset_surface"))
            d_ext = ext_surface.distance(pts)
        else:
            d_ext = np.full(len(pts), np.inf)
        # the skin only bounds the wall where it is not buried inside the positive
        skin_pts = fuselage.closest_points(pts)
        covered = solid_p.signed(skin_pts) < -COVERED_TOL
        ext_vertex = ((d_f > 0.0) | covered | (d_ext <= wall_f)).reshape(r, n)
        # rings upstream of / in the plane of the positive's start cap, and in its end cap (the face, or the junction
        # plane of a branch), are not part of the wall; the branch overlap and splitter rings beyond the junction are
        # covered by the trunk's own rings
        keep = (neg.dist > pos.dist[0] + 1e-9) & (neg.dist < pos.dist[-1] - 1e-9)
        internal = keep[:, None] & ~ext_vertex
        external = keep[:, None] & ext_vertex
        if internal.any():
            internal_min = min(internal_min, float((wall_f.reshape(r, n))[internal].min()))
        if keep.any():
            union_min = min(union_min, float(union[keep].min()))
        if external.any():
            external_wall = min(external_wall, float(d_ext.reshape(r, n)[external].min()))
        scalar = np.full(neg.mesh.n_points, np.inf)
        scalar[: r * n] = np.where(internal, wall_f.reshape(r, n), np.inf).ravel()
        c.wall_points[neg_name] = scalar
    c.external_wall = None if math.isinf(external_wall) else external_wall
    c.min_wall = None if math.isinf(internal_min) else internal_min
    if not math.isinf(union_min) and union_min < 0.0:
        c.add("Wall left (min)", f"{union_min:.4f} m: the negative leaves (fuselage ∪ positive)", "error")
    if c.min_wall is not None:
        c.wall_ok = c.min_wall >= s.min_clearance
        c.add("Wall left (internal)", f"{c.min_wall:.4f} m to the skin (limit {s.min_clearance:g} m)",
              "pass" if c.wall_ok else "warn")
        if not c.wall_ok:
            c.add("", f"WARN wall left between the negative and the skin {c.min_wall:.4f} m < checks.min_clearance "
                  f"({s.min_clearance:g} m)", "warn")
    if c.external_wall is not None:
        c.add("Wall (external)", f"{c.external_wall:.4f} m = wall_thickness (not compared with min_clearance)")


def other_geoms(bodies: dict, parts: list[Solid], c: DuctChecks, solids: BodySolids | None = None) -> None:
    """Positive / negative bodies must not intersect the other Geoms (the engine is not among them)."""
    solids = solids if solids is not None else BodySolids(bodies)
    for solid in parts:
        lo, hi = np.array(solid.bounds[0::2]), np.array(solid.bounds[1::2])
        hit_bodies = []
        for name in bodies:
            body = bodies[name]
            pts = body.mesh.points
            near = np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9), axis=1)
            inside_duct_pts = bool(near.any()) and solid.closed and bool((solid.signed(pts[near]) < -INSIDE_TOL).any())
            # the other Geom's vertices inside the body
            gp = solid.mesh.points
            blo, bhi = np.array(body.mesh.bounds[0::2]), np.array(body.mesh.bounds[1::2])
            gnear = np.all((gp >= blo - 1e-9) & (gp <= bhi + 1e-9), axis=1)
            geom_inside = False
            if gnear.any():
                body_solid = solids[name]
                geom_inside = body_solid.closed and bool((body_solid.signed(gp[gnear]) < -INSIDE_TOL).any())
            if inside_duct_pts or geom_inside:
                hit_bodies.append(name)
                c.intersecting.add(name)
        if hit_bodies:
            c.intersections.append(f"{solid.name or 'Geom'}: {', '.join(hit_bodies)}")
    if c.intersections:
        for text in c.intersections:
            c.add("", f"WARN intersects {text}", "warn")
    else:
        c.add("Other Geoms", f"no intersection ({len(parts)} checked)" if parts else "none to check", "pass")


def _face_grid(duct: dm.Duct) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on the engine face disc: (G, G, 3) points and the (G, G) inside-disc mask."""
    spec = duct.spec
    axis = np.asarray(spec.face_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    ea = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], axis) * axis
    ea /= np.linalg.norm(ea)
    eb = np.cross(axis, ea)
    r = 0.5 * spec.face_diameter * LOS_FACE_FRACTION
    g = np.linspace(-r, r, LOS_GRID)
    a, b = np.meshgrid(g, g, indexing="ij")
    pts = spec.face_centre + a[..., None] * ea + b[..., None] * eb
    return pts, (a**2 + b**2) <= r**2 * (1 + 1e-9)


def los_fractions(duct: dm.Duct, casters: list) -> tuple[float, float]:
    """Fraction of the engine face seen straight through an opening, along +X and along the inlet normal.

    Rays run from face sample points back towards the opening; a point counts as visible when the ray reaches the
    opening plane inside the opening outline without crossing the bore wall. ``casters`` has one ``RayCaster`` per
    branch of ``duct.branches`` (the wall that branch's rays must not cross). The union over every opening (mirrored
    twins included) is reported."""
    pts, mask = _face_grid(duct)
    flat, m = pts.reshape(-1, 3), mask.ravel()
    sides = []
    for bg, caster in zip(duct.branches, casters):  # every opening, mirrored twins included
        st0 = bg.stations[0]
        sides.append((st0.centre, -st0.T, st0.U, st0.V, st0.ring, caster))
    dirs = {"x": np.array([1.0, 0.0, 0.0]), "n": None}
    result = {}
    for key in ("x", "n"):
        vis_any = np.zeros(len(flat), dtype=bool)
        for centre, n, U, V, ring_uv, caster in sides:
            d = dirs["x"] if key == "x" else -n  # the direction the air travels
            denom = float(np.dot(n, d))
            if abs(denom) < 1e-9:
                continue
            # P - tau d lies in the opening plane: n . (P - tau d - centre) = 0
            tau = np.dot(flat - centre, n) / denom
            q = flat - tau[:, None] * d
            uv = np.c_[np.dot(q - centre, U), np.dot(q - centre, V)]
            ok = m & (tau > 0.0) & points_in_polygon(ring_uv, uv)
            for i in np.flatnonzero(ok):
                a = flat[i] - LOS_END_TOL * d
                b = q[i] + LOS_END_TOL * d
                if caster.blocked(a, b):
                    ok[i] = False
            vis_any |= ok
        result[key] = float(vis_any[m].mean()) if m.any() else 0.0
    return result["x"], result["n"]


def line_of_sight(duct: dm.Duct, bodies: dict, c: DuctChecks) -> None:
    """Line of sight through the negative bores (branch + trunk lateral surfaces, without end caps)."""
    casters = [RayCaster(bodies[f"neg_branch_{bg.name}"].lateral, bodies["neg_trunk"].lateral) for bg in duct.branches]
    c.los_x, c.los_n = los_fractions(duct, casters)
    c.los_any = max(c.los_x, c.los_n)
    level = "pass" if c.los_any > 0.0 else "warn"
    c.add("Line of sight", f"{100 * c.los_x:.0f} % of the face along +X, {100 * c.los_n:.0f} % along the inlet "
          f"normal" + ("  (hidden)" if c.los_any == 0.0 else ""), level)


def numbers(duct: dm.Duct, c: DuctChecks, s: CheckSettings) -> None:
    primaries = [i for i, bg in enumerate(duct.branches) if not bg.twin]
    many = len(primaries) > 1
    for i in primaries:
        bg = duct.branches[i]
        tag = f" {bg.name}" if many else ""
        cl = bg.centreline
        offset, _, ratio = cl.offset_over_length()
        if i == primaries[0]:
            c.offset, c.offset_over_length = offset, ratio
        c.add(f"Offset / length{tag}", f"{offset:.4f} m / {cl.length:.4f} m = {ratio:.3f}")
        curvature = cl.max_curvature()
        c.max_curvature = max(c.max_curvature, curvature)
        if curvature < 1e-9:
            c.add(f"Centreline curvature{tag}", "straight")
        else:
            c.add(f"Centreline curvature{tag}", f"max {curvature:.3f} 1/m (min radius {1.0 / curvature:.3f} m)")
        for part, a in duct.cone_half_angles(i).items():
            if part == "trunk" and i != primaries[0]:
                continue  # one trunk
            if not a.size:
                continue
            worst = float(a.max())
            c.half_angle_max = max(c.half_angle_max, worst)
            over = np.flatnonzero(a > s.max_half_angle_deg)
            label = f"{part}{tag}" if part == "branch" else part
            c.add(f"Cone half-angle {label}", f"max {worst:.2f} deg (limit {s.max_half_angle_deg:g})",
                  "pass" if not len(over) else "warn")
            if len(over):
                c.half_angle_exceeded.append(label)
                c.add("", f"WARN equivalent cone half-angle exceeds {s.max_half_angle_deg:g} deg in {len(over)} "
                      f"segment(s) of the {label}", "warn")
        for curve in ("branch", "trunk"):
            if curve == "trunk" and i != primaries[0]:
                continue
            dec = bg.areas.decreasing(curve)
            if dec:
                label = f"{curve}{tag}" if curve == "branch" else curve
                c.area_decreasing.append(label)
                spans = ", ".join(f"{x:.2f}-{y:.2f}" for x, y in dec)
                c.add("", f"WARN A(s) decreases in the {label} (s = {spans})", "warn")


def run_checks(duct: dm.Duct, fuselage: Solid | None, parts: list[Solid], settings: CheckSettings,
               bodies: dict | None = None) -> tuple[DuctChecks, dict]:
    bodies = bodies if bodies is not None else duct.bodies()
    c = DuctChecks()
    solids = BodySolids(bodies)
    numbers(duct, c, settings)
    if duct.spec.wall_thickness and any(dm.ring_crosses_itself(dm.offset_ring(st.ring, duct.spec.wall_thickness))
                                        for st in duct.stations()):
        c.add("", "ERROR wall_thickness is too large for the local curvature: the offset section crosses itself",
              "error")
    if fuselage is None or not fuselage.closed:
        c.add("Fuselage", "mesh missing or not watertight: skin checks skipped", "warn")
    else:
        external_region(duct, bodies, fuselage, c, settings)
        wall_left(duct, bodies, fuselage, c, settings, solids)
    other_geoms(bodies, parts, c, solids)
    line_of_sight(duct, bodies, c)
    return c, bodies
