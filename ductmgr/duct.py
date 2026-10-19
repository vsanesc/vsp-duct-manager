# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Duct geometry: stations along the centrelines, sections morphed between key sections, and the NEGATIVE and
POSITIVE bodies built from the same stations.

The duct has 1..N inlets (physical branches), optionally with mirrored twins about y = 0, one trunk and the engine
face. For each branch:

- stations lie along the branch Bézier (inlet section centroid -> its end at the junction) and the shared trunk
  Bézier (junction -> face centre), equally spaced in arc length; ``s`` is the arc-length fraction of the whole path
  of THAT branch;
- the section at a station is the blend of the two neighbouring key sections (resampled with the arc-length
  correspondence of ``interp``), then scaled about its centroid so its area is exactly A(s) (the INNER section);
- frames are parallel-transported (minimal rotation) and the branch's accumulated roll is distributed over its
  length so it meets the trunk's frame at the junction.

The junction depends on the number of branches B (twins counted):

- B = 1: no junction. One continuous duct; the junction point is only an intermediate point with a G1 tangent;
- B = 2: the trunk start section is cut into two pieces by the plane through the trunk axis that is equidistant to
  both branches (y = 0 when symmetric), at the offset that gives each its flow fraction of the area;
- B >= 3: the trunk start section is cut into B sectors by rays from its centroid, ordered by the angular position of
  each branch around the trunk axis; the ray angles are solved so each sector has its flow fraction of the area.

Each branch ends exactly on its piece (same shape, position and area), so the pieces tile the trunk start section;
the branch centrelines end at the pieces' centroids and keep the junction tangent (G1).

- NEGATIVE body = inner sections, preceded by a prismatic extension of ``lip_extension`` upstream of the opening
  plane (along the inlet normal), ending in a flat cap at the engine face. Each branch negative continues into
  the trunk as piece-sections by ``overlap`` (and by the ``splitter_length``, separated by walls of
  ``wall_thickness`` along the cuts); with no splitter the trunk negative starts ``overlap`` upstream of the
  junction;
- POSITIVE body = the inner sections offset outward in their plane by ``wall_thickness``, starting exactly at
  the opening plane (a blunt lip) and running to the face.

The intended model in OpenVSP is (fuselage ∪ positives) − negatives. A mirrored twin is the exact mirror image about
y = 0 of its branch.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from ductmgr import interp
from ductmgr import outline as ol
from ductmgr.areacurve import AreaCurves, AreaPoints
from ductmgr.centreline import Centreline, CentrelineParams, default_junction
from ductmgr.geometry import InletPlane, polygon_area, polygon_centroid

LIP_MODES = ("blunt",)  # extensible: other lip shapes later
DEFAULT_LIP_MODE = "blunt"
DEFAULT_OVERLAP = 0.02  # m, ducts.junction.overlap when not configured
SYMMETRIES = ("xz", "none")
MITRE_LIMIT = 4.0  # an offset corner is never moved further than this many wall thicknesses
UP = np.array([0.0, 0.0, 1.0])
REFINE_SUBDIVISIONS = 3  # extra rings between stations when the external region is measured
SELF_SYMMETRIC_TOL = 1e-9  # m / deg: an inlet this close to y = 0 with no sweep is its own mirror image
SECTOR_GRID = 8192  # angular samples of the radial function when the sector rays are solved
SECTOR_SEARCH = 360  # trial rotations of the first ray


@dataclass(frozen=True)
class KeyOutline:
    """A key section for lofting: its closed outline and corner vertices, in its own (u, v)."""

    s: float
    outline: np.ndarray
    vertices: np.ndarray


@dataclass(frozen=True)
class InletSpec:
    """One inlet: its plane and key sections, and what it asks of the junction."""

    name: str
    plane: InletPlane
    keys: tuple[KeyOutline, ...]  # sorted by s, keys[0].s == 0 (the inlet); the face circle is added
    flow_fraction: float | None = None  # per physical branch; None = an equal share
    mirror: bool = False  # ask for a mirrored twin (only with symmetry xz; a self-symmetric inlet gets none)
    start_handle: float | None = None
    junction_in_handle: float | None = None
    area_points: tuple | None = None  # interior (s_local, area) points of this branch's area curve; None = default
    inlet_area: float | None = None  # A(0); the area of the sketched outline when not given


def self_symmetric(plane: InletPlane) -> bool:
    """True when the plane is its own mirror image about y = 0 (centred on it, no sweep): such an inlet is never
    duplicated and its section is symmetric about u = 0."""
    return abs(plane.origin[1]) < SELF_SYMMETRIC_TOL and abs(plane.sweep_deg) < SELF_SYMMETRIC_TOL


def has_twin(inlet: InletSpec, symmetry: str) -> bool:
    return symmetry == "xz" and inlet.mirror and not self_symmetric(inlet.plane)


def flow_fractions(fractions, twins) -> tuple[list[float], bool]:
    """The per-branch flow fraction of every inlet, normalised so that all physical branches (a twin counts as one
    more) sum to 1. ``None`` is an equal share. Returns (fractions, normalised): ``normalised`` is True when given
    values had to be rescaled."""
    total_branches = sum(1 + int(t) for t in twins)
    raw = [1.0 / total_branches if f is None else float(f) for f in fractions]
    if any(r <= 0.0 for r in raw):
        raise ValueError("flow fractions must be positive")
    total = sum(r * (1 + int(t)) for r, t in zip(raw, twins))
    changed = abs(total - 1.0) > 1e-9 and any(f is not None for f in fractions)
    return [r / total for r in raw], changed


@dataclass(frozen=True)
class DuctSpec:
    inlets: tuple[InletSpec, ...]
    face_centre: np.ndarray
    face_axis: np.ndarray
    face_diameter: float
    centreline: CentrelineParams = field(default_factory=CentrelineParams)
    area_points: AreaPoints = field(default_factory=AreaPoints)  # junction area and the trunk interior points
    area_law: str = "cosine"
    symmetry: str = "xz"
    n_stations_branch: int = 12
    n_stations_trunk: int = 12
    n_points: int = interp.DEFAULT_SAMPLES
    wall_thickness: float = 0.004
    lip_extension: float = 0.05
    lip_mode: str = DEFAULT_LIP_MODE
    junction_fraction: float = 0.5
    splitter_length: float = 0.0  # m: the branch negatives stay separate (pieces) this far into the trunk
    overlap: float = DEFAULT_OVERLAP  # m: each branch negative reaches this far beyond the junction plane


def single_spec(plane, keys, face_centre, face_axis, face_diameter, mirror=True, name="R", inlet_area=None,
                centreline=None, area_points=None, flow_fraction=None, **kw) -> DuctSpec:
    """A spec with one inlet (and, with ``mirror``, its twin): the common case."""
    centreline = centreline or CentrelineParams()
    area_points = area_points or AreaPoints()
    inlet = InletSpec(name, plane, tuple(keys), flow_fraction, mirror, centreline.start_handle,
                      centreline.junction_in_handle, area_points.branch, inlet_area)
    return DuctSpec((inlet,), np.asarray(face_centre, dtype=float), np.asarray(face_axis, dtype=float), face_diameter,
                    centreline, AreaPoints(area_points.junction_area, None, area_points.trunk), **kw)


@dataclass
class Station:
    part: str  # "branch" or "trunk"
    s: float  # global arc-length fraction
    dist: float  # arc length from the opening plane along the path (m)
    centre: np.ndarray
    T: np.ndarray
    U: np.ndarray
    V: np.ndarray
    area: float
    ring: np.ndarray  # (N, 2) inner section in (u, v), centred on its centroid, counter-clockwise


@dataclass
class Body:
    name: str  # e.g. "neg_branch_R", "pos_branch_R_mirror", "neg_trunk"
    kind: str  # "negative" or "positive"
    part: str  # "branch" or "trunk"
    side: str  # the branch name, "<name>_mirror" for a twin, "C" for the trunk
    rings: np.ndarray  # (R, N, 3)
    dist: np.ndarray  # (R,) arc length of each ring from the opening plane (negative upstream of it)
    mesh: pv.PolyData  # closed, outward-oriented, triangulated
    lateral: pv.PolyData  # the open tube without end caps


@dataclass(frozen=True)
class Piece:
    """Which part of the trunk start section a branch ends on."""

    kind: str = "whole"  # "whole" (B = 1), "half" (B = 2) or "sector" (B >= 3)
    normal: tuple = (1.0, 0.0)  # half: keep points with normal . p >= offset
    offset: float = 0.0
    a0: float = 0.0  # sector: between the rays at these angles (counter-clockwise from a0 to a1)
    a1: float = 0.0


@dataclass
class BranchGeom:
    name: str  # the body name part: the inlet name, "<name>_mirror" for a twin
    inlet: int  # index into DuctSpec.inlets
    twin: bool  # the mirror image of the inlet's primary branch
    plane: InletPlane
    flow: float  # the fraction of the flow this branch carries
    centreline: Centreline | None = None
    areas: AreaCurves | None = None
    s_junction: float = 0.0
    piece: Piece = field(default_factory=Piece)
    branch_end: np.ndarray | None = None  # the end section (its piece), centred on its centroid
    branch_end_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    stations: list = field(default_factory=list)
    rings: list = field(default_factory=list)  # key rings (s, ring) of the inlet, face included


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    return v * math.cos(angle) + np.cross(k, v) * math.sin(angle) + k * np.dot(k, v) * (1.0 - math.cos(angle))


def _minimal_rotation(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-14:
        return v if c > 0 else -v
    return _rotate(v, axis, math.atan2(s, c))


def transport_frames(T: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """The V vector at every tangent of ``T`` (n, 3), parallel-transported from ``v0`` (made orthogonal to T[0])."""
    v = v0 - np.dot(v0, T[0]) * T[0]
    out = [v / np.linalg.norm(v)]
    for i in range(len(T) - 1):
        v = _minimal_rotation(out[-1], T[i], T[i + 1])
        v = v - np.dot(v, T[i + 1]) * T[i + 1]
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def symmetric_v(T: np.ndarray) -> np.ndarray:
    """The V of a frame that does not depend on the side: world +Z made orthogonal to T."""
    ref = UP if abs(float(np.dot(UP, T))) < 0.999 else np.array([0.0, 1.0, 0.0])
    v = ref - np.dot(ref, T) * T
    return v / np.linalg.norm(v)


def offset_ring(ring: np.ndarray, t: float) -> np.ndarray:
    """The counter-clockwise ``ring`` moved outward by ``t`` perpendicular to its edges (mitred corners).

    Every edge of the result is parallel to the corresponding edge of ``ring`` at exactly distance ``t``; a
    corner is moved along its bisector so that both adjacent edges keep that distance (never further than
    MITRE_LIMIT * t)."""
    p = np.asarray(ring, dtype=float)
    prev_e = p - np.roll(p, 1, axis=0)
    next_e = np.roll(p, -1, axis=0) - p

    def normals(e):
        n = np.c_[e[:, 1], -e[:, 0]]  # right of the travel direction = outward for a counter-clockwise ring
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(length > 0, length, 1.0)

    n1, n2 = normals(prev_e), normals(next_e)
    denom = 1.0 + (n1 * n2).sum(axis=1)
    m = (n1 + n2) / np.maximum(denom, 1e-12)[:, None]
    length = np.linalg.norm(m, axis=1)
    over = length > MITRE_LIMIT
    m[over] *= (MITRE_LIMIT / length[over])[:, None]
    return p + t * m


def ring_crosses_itself(ring: np.ndarray) -> bool:
    """True when the closed polyline has two crossing (non-adjacent) edges."""
    p = np.asarray(ring, dtype=float)
    a, b = p, np.roll(p, -1, axis=0)
    n = len(p)

    def side(o, d, q):  # o, d: (n, 2) edge ends; q: (m, 2) points -> (n, m)
        return ((d[:, 0] - o[:, 0])[:, None] * (q[None, :, 1] - o[:, 1][:, None])
                - (d[:, 1] - o[:, 1])[:, None] * (q[None, :, 0] - o[:, 0][:, None]))

    d1, d2 = side(a, b, a), side(a, b, b)  # edge i against the ends of edge j
    prod = d1 * d2
    cross = (prod < 0) & (prod.T < 0)  # each edge separates the other's end points
    idx = np.arange(n)
    far = np.abs(idx[:, None] - idx[None, :])
    far = (far > 1) & (far < n - 1)  # non-adjacent edges
    return bool((cross & far).any())


def _tube_faces(n_rings: int, n: int) -> np.ndarray:
    faces = []
    for r in range(n_rings - 1):
        for j in range(n):
            j1 = (j + 1) % n
            a, b, c, d = r * n + j, r * n + j1, (r + 1) * n + j1, (r + 1) * n + j
            faces += [3, a, b, c, 3, a, c, d]
    return np.array(faces)


def tube_mesh(rings: np.ndarray, caps: bool) -> pv.PolyData:
    """Triangulated surface through the rings (R, N, 3); with ``caps`` closed by a fan at each end ring and
    oriented outward."""
    r, n, _ = rings.shape
    pts = rings.reshape(-1, 3)
    faces = _tube_faces(r, n)
    if caps:
        c0, c1 = len(pts), len(pts) + 1
        pts = np.vstack([pts, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
        extra = []
        for j in range(n):
            j1 = (j + 1) % n
            extra += [3, c0, j1, j, 3, c1, (r - 1) * n + j, (r - 1) * n + j1]
        faces = np.r_[faces, extra]
    mesh = pv.PolyData(pts, faces)
    if caps:
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True,
                                    consistent_normals=True, split_vertices=False, inplace=False)
        for name in ("Normals",):
            if name in mesh.cell_data:
                del mesh.cell_data[name]
    return mesh


def effective_points(n: int) -> int:
    """Samples per section actually used: a multiple of 4, so the junction section can be split on a symmetric
    sample (a vertex at the top and at the bottom of the cut)."""
    return 4 * math.ceil(max(int(n), 8) / 4)


def ray_radii(poly: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Distance from the origin to the outer boundary of ``poly`` along each direction of ``angles``."""
    p = np.asarray(poly, dtype=float)
    e = np.roll(p, -1, axis=0) - p
    d = np.c_[np.cos(angles), np.sin(angles)]

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    den = cross(d[:, None, :], e[None, :, :])  # (m, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cross(p[None, :, :], e[None, :, :]) / den
        u = cross(p[None, :, :], d[:, None, :]) / den
    ok = (np.abs(den) > 1e-300) & (u >= -1e-12) & (u <= 1 + 1e-12) & (r > 0)
    return np.where(ok, r, 0.0).max(axis=1)


def symmetric_ring(ring: np.ndarray, n: int) -> np.ndarray:
    """``ring`` made exactly symmetric about the line a = 0: the radial function about the origin is averaged
    with its mirror image, resampled by arc length from the top (sample k and n - k are mirror images and the
    bottom is sample n / 2), centred on its centroid and scaled back to the area of ``ring``."""
    area = polygon_area(ring)
    m = 4 * n
    ang = math.pi / 2 + 2.0 * math.pi * np.arange(m) / m
    r = ray_radii(ring, ang)
    r = 0.5 * (r + r[(-np.arange(m)) % m])
    pts = np.c_[r * np.cos(ang), r * np.sin(ang)]
    out = interp.resample(pts, pts[:1], n)
    mirror = (-np.arange(n)) % n
    out = np.c_[0.5 * (out[:, 0] - out[mirror, 0]), 0.5 * (out[:, 1] + out[mirror, 1])]  # exact symmetry
    out[:, 1] -= polygon_centroid(out)[1]
    return out * math.sqrt(area / polygon_area(out))


def clip_ring(poly: np.ndarray, normal, offset: float) -> np.ndarray:
    """The part of the closed ``poly`` with ``normal . p >= offset`` (Sutherland-Hodgman)."""
    nrm = np.asarray(normal, dtype=float)
    pts = np.asarray(poly, dtype=float)
    out = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        da, db = float(nrm @ a) - offset, float(nrm @ b) - offset
        if da >= 0:
            out.append(a)
        if (da > 0 > db) or (da < 0 < db):
            out.append(a + (b - a) * (da / (da - db)))
    res = []
    for p in out:  # drop repeated points (a vertex exactly on the cut)
        if not res or np.linalg.norm(p - res[-1]) > 1e-12:
            res.append(p)
    if len(res) > 1 and np.linalg.norm(res[0] - res[-1]) <= 1e-12:
        res.pop()
    return np.array(res)


def fill_ring(poly: np.ndarray, cut_edges, n: int) -> np.ndarray:
    """``poly`` as an ``n``-point ring: every vertex is kept (the shape does not change) and the extra points are
    spread along the straight ``cut_edges`` (pairs of consecutive vertex indices), proportionally to their length.
    The ring starts at the highest corner of the cuts."""
    pts = np.asarray(poly, dtype=float)
    m = len(pts)
    corners = sorted({i for e in cut_edges for i in e})
    if n < m:
        return interp.resample_keep(pts, corners, n)
    lengths = np.array([np.linalg.norm(pts[j] - pts[i]) for i, j in cut_edges])
    if lengths.sum() <= 0.0:
        raise ValueError("the junction section could not be split")
    extra = n - m
    counts = np.floor(extra * lengths / lengths.sum()).astype(int)
    for k in np.argsort(-(extra * lengths / lengths.sum() - counts))[: extra - int(counts.sum())]:
        counts[k] += 1
    by_start = {i: (j, int(c)) for (i, j), c in zip(cut_edges, counts)}
    out, where = [], {}
    for i in range(m):
        where[i] = len(out)
        out.append(pts[i])
        if i in by_start:
            j, c = by_start[i]
            out.extend(pts[i] + (pts[j] - pts[i]) * (k + 1) / (c + 1) for k in range(c))
    ring = np.array(out)
    start = max((where[i] for i in corners), key=lambda k: (ring[k, 1], -ring[k, 0]))
    return np.roll(ring, -start, axis=0)


def _cut_edges(poly: np.ndarray, lines) -> list[tuple[int, int]]:
    """Edges of ``poly`` lying on one of the ``lines`` (normal, offset): both end points satisfy normal . p = offset."""
    m = len(poly)
    edges = []
    for nrm, off in lines:
        on = np.abs(poly @ np.asarray(nrm, dtype=float) - off) < 1e-9
        edges += [(i, (i + 1) % m) for i in range(m) if on[i] and on[(i + 1) % m]]
    return list(dict.fromkeys(edges))  # (two coincident lines give the same edge once)


def half_ring(ring: np.ndarray, normal, offset: float, n: int) -> np.ndarray:
    """The part of ``ring`` with ``normal . p >= offset`` as an ``n``-point ring (the cut is a straight side that
    carries the extra points, so the part is geometrically exactly that part of ``ring``)."""
    half = clip_ring(ring, normal, offset)
    edges = _cut_edges(half, [(normal, offset)])
    if not edges:
        raise ValueError("the junction section could not be split")
    return fill_ring(half, edges[:1], n)


def sector_polygon(ring: np.ndarray, a0: float, a1: float) -> np.ndarray:
    """The sector of the star-shaped ``ring`` (about the origin) between the rays at angles a0 and a1 (counter-
    clockwise from a0): the origin, the point on ray a0, the ring vertices in between, the point on ray a1."""
    span = (a1 - a0) % (2.0 * math.pi) or 2.0 * math.pi
    rel = (np.arctan2(ring[:, 1], ring[:, 0]) - a0) % (2.0 * math.pi)
    inside = np.flatnonzero((rel > 1e-12) & (rel < span - 1e-12))
    arc = ring[inside[np.argsort(rel[inside])]]
    r0, r1 = ray_radii(ring, np.array([a0, a1]))
    p0 = r0 * np.array([math.cos(a0), math.sin(a0)])
    p1 = r1 * np.array([math.cos(a1), math.sin(a1)])
    return np.vstack([[0.0, 0.0], p0, arc, p1])


def sector_ring(ring: np.ndarray, a0: float, a1: float, wall_half: float, n: int) -> np.ndarray:
    """An ``n``-point ring of the sector between the rays (a0, a1), pulled back by ``wall_half`` from each ray."""
    poly = sector_polygon(ring, a0, a1)
    lines = [((-math.sin(a0), math.cos(a0)), wall_half), ((math.sin(a1), -math.cos(a1)), wall_half)]
    if wall_half > 0.0:
        if (a1 - a0) % (2.0 * math.pi) > math.pi:
            raise ValueError("a splitter wall needs sectors narrower than 180 degrees")
        for nrm, off in lines:
            poly = clip_ring(poly, nrm, off)
    edges = _cut_edges(poly, lines)
    if len(edges) < 1:
        raise ValueError("the junction section could not be split")
    return fill_ring(poly, edges, n)


def _area_of_half(ring, normal, offset) -> float:
    part = clip_ring(ring, normal, offset)
    return polygon_area(part) if len(part) >= 3 else 0.0


def half_offset(ring: np.ndarray, normal, fraction: float) -> float:
    """The offset c for which the part of ``ring`` with ``normal . p >= c`` has ``fraction`` of its area."""
    nrm = np.asarray(normal, dtype=float)
    proj = ring @ nrm
    lo, hi = float(proj.min()), float(proj.max())
    target = fraction * polygon_area(ring)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _area_of_half(ring, nrm, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sector_angles(ring: np.ndarray, positions, fractions, symmetric: bool = False) -> list[tuple[float, float]]:
    """Ray angles (a0, a1) of one sector per branch, in the order given: sector k is the branch's share
    ``fractions[k]`` of the area of the star-shaped ``ring`` (about the origin), the sectors follow each other
    counter-clockwise in the order of the branches' angular ``positions``, and the one free rotation is chosen so
    the sectors sit as close as possible (in angle) to their branches."""
    b = len(positions)
    order = np.argsort(np.asarray(positions, dtype=float), kind="stable")
    m = SECTOR_GRID
    th = 2.0 * math.pi * np.arange(m + 1) / m
    r = ray_radii(ring, th[:-1])
    cum = np.r_[0.0, np.cumsum(0.5 * r**2 * (2.0 * math.pi / m))]
    total = float(cum[-1])

    def area_at(theta):
        k = math.floor(theta / (2.0 * math.pi))
        return k * total + float(np.interp(theta - 2.0 * math.pi * k, th, cum))

    def angle_at(value):
        k = math.floor(value / total)
        return 2.0 * math.pi * k + float(np.interp(value - k * total, cum, th))

    share = np.cumsum([0.0] + [float(fractions[j]) for j in order])
    pos = np.asarray(positions, dtype=float)[order]

    def bounds(a0):
        c0 = area_at(a0)
        return [angle_at(c0 + share[k] * total) for k in range(b)] + [a0 + 2.0 * math.pi]

    def cost(a0):
        bd = bounds(a0)
        mid = 0.5 * (np.array(bd[:-1]) + np.array(bd[1:]))
        d = (mid - pos + math.pi) % (2.0 * math.pi) - math.pi
        return float((d**2).sum())

    grid = np.linspace(0.0, 2.0 * math.pi, SECTOR_SEARCH, endpoint=False)
    best = float(grid[int(np.argmin([cost(a) for a in grid]))])
    step = 2.0 * math.pi / SECTOR_SEARCH
    lo, hi = best - step, best + step
    for _ in range(60):  # golden-section refinement
        c1, c2 = hi - 0.618033988749895 * (hi - lo), lo + 0.618033988749895 * (hi - lo)
        if cost(c1) < cost(c2):
            hi = c2
        else:
            lo = c1
    bd = bounds(0.5 * (lo + hi))
    if symmetric:  # the layout is mirror symmetric (angle -> pi - angle): make the rays exactly so
        rays = np.array(bd[:-1])
        reflected = math.pi - rays  # each ray mirrored
        fixed = rays.copy()
        for k in range(b):
            j = int(np.argmin(np.abs((rays[k] - reflected + math.pi) % (2.0 * math.pi) - math.pi)))
            partner = reflected[j] + 2.0 * math.pi * round((rays[k] - reflected[j]) / (2.0 * math.pi))
            fixed[k] = 0.5 * (rays[k] + partner)
        bd = list(fixed) + [fixed[0] + 2.0 * math.pi]
    out = [(0.0, 0.0)] * b
    for k, j in enumerate(order):
        out[int(j)] = (bd[k], bd[k + 1])
    return out


def piece_ring(ring: np.ndarray, piece: Piece, wall_half: float, n: int, scale: float = 1.0) -> np.ndarray:
    """A branch's piece of a trunk section ``ring`` (centred on its own centroid), as an ``n``-point ring.
    ``scale`` is the linear size of ``ring`` relative to the trunk start section the piece was laid out on."""
    if piece.kind == "whole":
        return ring
    if piece.kind == "half":
        return half_ring(ring, piece.normal, piece.offset * scale + wall_half, n)
    return sector_ring(ring, piece.a0, piece.a1, wall_half, n)


def mirrored(points: np.ndarray) -> np.ndarray:
    return points * np.array([1.0, -1.0, 1.0])


class Duct:
    """Everything built from a ``DuctSpec``."""

    def __init__(self, spec: DuctSpec, plan_only: bool = False):
        """``plan_only`` stops after the centrelines, the area curves and the junction sections (cheap: what the
        explorer needs to size the key sections), without stations or bodies."""
        if spec.lip_mode not in LIP_MODES:
            raise ValueError(f"ducts.lip_mode must be one of {', '.join(LIP_MODES)} (got {spec.lip_mode!r}); "
                             "other lip shapes are not implemented yet")
        if spec.symmetry not in SYMMETRIES:
            raise ValueError(f"ducts.symmetry must be one of {', '.join(SYMMETRIES)} (got {spec.symmetry!r})")
        if spec.wall_thickness <= 0.0 or spec.lip_extension < 0.0:
            raise ValueError("ducts.wall_thickness must be positive and ducts.lip_extension not negative")
        if spec.n_stations_branch < 2 or spec.n_stations_trunk < 2:
            raise ValueError("at least 2 stations per branch / trunk are needed")
        if spec.splitter_length < 0.0 or spec.overlap < 0.0:
            raise ValueError("ducts.junction.splitter_length and ducts.junction.overlap must not be negative")
        if not spec.inlets:
            raise ValueError("at least one inlet is needed")
        self.spec = spec
        n = self.n = effective_points(spec.n_points)
        self.symmetric = spec.symmetry == "xz"
        twins = [has_twin(i, spec.symmetry) for i in spec.inlets]
        fractions, self.fractions_normalised = flow_fractions([i.flow_fraction for i in spec.inlets], twins)
        self.fractions = fractions
        self.branches: list[BranchGeom] = []
        for k, ins in enumerate(spec.inlets):
            self.branches.append(BranchGeom(ins.name, k, False, ins.plane, fractions[k]))
            if twins[k]:
                self.branches.append(BranchGeom(f"{ins.name}_mirror", k, True, ins.plane.mirrored(), fractions[k]))
        n_branches = len(self.branches)
        self.n_branches = n_branches
        self._inlet_rings = [self._key_rings(ins, n) for ins in spec.inlets]
        a_in = [ins.inlet_area if ins.inlet_area is not None else polygon_area(rings[0][1])
                for ins, rings in zip(spec.inlets, self._inlet_rings)]
        a_total = sum(a_in[bg.inlet] for bg in self.branches)
        a_face = 0.25 * math.pi * spec.face_diameter**2
        starts = {}
        for bg in self.branches:
            ins = spec.inlets[bg.inlet]
            ring0 = self._inlet_rings[bg.inlet][0][1]
            u0, v0, n0 = ins.plane.frame()
            start = ins.plane.to_3d([polygon_centroid(ring0)])[0]
            direction = -n0
            if bg.twin:
                start, direction = mirrored(start), mirrored(direction)
            starts[id(bg)] = (start, direction)
            bg.rings = self._inlet_rings[bg.inlet]
            points = AreaPoints(spec.area_points.junction_area, ins.area_points, spec.area_points.trunk)
            bg.areas = AreaCurves(a_in[bg.inlet], a_face, points, spec.area_law, n_branches, bg.flow, a_total)
        # the junction: common to all branches
        mean_start = np.mean([starts[id(bg)][0] for bg in self.branches], axis=0)
        dj = default_junction(mean_start, spec.face_centre, spec.junction_fraction)
        p = spec.centreline
        jx, jz = p.junction if p.junction is not None else (dj[0], dj[2])
        jy = 0.0 if self.symmetric else (dj[1] if p.junction_y is None else float(p.junction_y))
        self._common = p.with_(junction=(jx, jz), junction_y=jy)
        self._starts = starts
        for bg in self.branches:
            bg.centreline = self._centreline(bg, None)
            bg.s_junction = bg.centreline.s_junction
        bg0 = self.branches[0]
        self.start_dir = starts[id(bg0)][1]
        self.inlet_centre_uv = polygon_centroid(self._inlet_rings[0][0][1])
        self._junction_sections()
        for bg in self.branches:
            bg.centreline = self._centreline(bg, bg.branch_end_offset)
            bg.s_junction = bg.centreline.s_junction
        length_trunk = bg0.centreline.length_trunk
        if n_branches > 1 and spec.splitter_length + spec.overlap >= length_trunk:
            raise ValueError("ducts.junction.splitter_length + overlap must be shorter than the trunk")
        if plan_only:
            return
        self._build_stations()

    # ---- the first branch, for the common single-inlet case ----------------------------------------------------------
    @property
    def centreline(self) -> Centreline:
        return self.branches[0].centreline

    @property
    def areas(self) -> AreaCurves:
        return self.branches[0].areas

    @property
    def s_junction(self) -> float:
        return self.branches[0].s_junction

    @property
    def branch(self) -> list[Station]:
        return self.branches[0].stations

    @property
    def branch_end(self) -> np.ndarray:
        return self.branches[0].branch_end

    @property
    def branch_end_offset(self) -> np.ndarray:
        return self.branches[0].branch_end_offset

    @property
    def rings(self):
        return self.branches[0].rings

    def _centreline(self, bg: BranchGeom, offset) -> Centreline:
        ins = self.spec.inlets[bg.inlet]
        start, direction = self._starts[id(bg)]
        params = self._common
        if ins.start_handle is not None:
            params = params.with_(start_handle=ins.start_handle)
        if ins.junction_in_handle is not None:
            params = params.with_(junction_in_handle=ins.junction_in_handle)
        return Centreline(start, direction, self.spec.face_centre, self.spec.face_axis, params,
                          self.spec.junction_fraction, offset, self.symmetric)

    # ---- sections ----------------------------------------------------------------------------------------------------
    def _key_rings(self, ins: InletSpec, n: int) -> list[tuple[float, np.ndarray]]:
        keys = list(ins.keys)
        if not keys or keys[0].s != 0.0:
            raise ValueError("the first key section must be the inlet (s = 0)")
        face = ol.circle_shape(0.5 * self.spec.face_diameter)
        face_outline = ol.build_outline(ol.edge_paths(face)[0])
        pairs = [(k.s, k.outline, k.vertices) for k in keys]
        pairs.append((1.0, face_outline, np.array(face.vertices)))
        rings = interp.correspond([(o, v) for _, o, v in pairs], n)
        return [(s, r) for (s, _, _), r in zip(pairs, rings)]

    def ring_at(self, s: float, area: float, keys=None) -> np.ndarray:
        """The inner section (N, 2) at global ``s``: blended key sections, centred and scaled to ``area``."""
        keys = self.rings if keys is None else keys
        i = 0
        while i < len(keys) - 2 and s > keys[i + 1][0]:
            i += 1
        (s0, r0), (s1, r1) = keys[i], keys[i + 1]
        t = 0.0 if s1 <= s0 else min(max((s - s0) / (s1 - s0), 0.0), 1.0)
        unit = lambda r: (r - polygon_centroid(r)) / math.sqrt(polygon_area(r - polygon_centroid(r)))  # shape only
        r = interp.blend(unit(r0), unit(r1), t)
        r = r - polygon_centroid(r)
        return r * math.sqrt(area / polygon_area(r))

    # ---- the junction ------------------------------------------------------------------------------------------------------
    def _junction_sections(self) -> None:
        """The trunk start section S, each branch's piece of it, and where the pieces' centroids are."""
        spec, n = self.spec, self.n
        bg0 = self.branches[0]
        cl0 = bg0.centreline
        aj = bg0.areas.junction_area
        ring = self.ring_at(cl0.s_junction, aj, bg0.rings)
        T = cl0.junction_tangent
        V = symmetric_v(T)
        U = np.cross(T, V)
        self.trunk_frame0 = (T, U, V)
        count = len(self.branches)
        if count == 1:
            self.trunk_start = ring
            bg0.piece = Piece("whole")
            bg0.branch_end, bg0.branch_end_offset = ring.copy(), np.zeros(3)
            return
        s_ring = symmetric_ring(ring, n) if self.symmetric else ring
        self.trunk_start = s_ring
        wall_half = 0.5 * spec.wall_thickness if spec.splitter_length > 0 else 0.0
        self._wall_half = wall_half
        pos = []
        for bg in self.branches:
            rel = self._starts[id(bg)][0] - cl0.junction
            pos.append((float(rel @ U), float(rel @ V)))
        flows = [bg.flow for bg in self.branches]
        if count == 2:
            d = np.array(pos[0]) - np.array(pos[1])
            nrm = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-12 else np.array([1.0, 0.0])
            if self.symmetric and self.branches[1].twin and self.branches[1].inlet == self.branches[0].inlet:
                nrm = np.array([np.sign(pos[0][0]) or 1.0, 0.0])  # a mirrored pair: exactly the plane y = 0
                c = 0.0
            else:
                c = half_offset(s_ring, nrm, flows[0])
            pieces = [Piece("half", tuple(nrm), c), Piece("half", tuple(-nrm), -c)]
        else:
            angles = [math.atan2(q[1], q[0]) for q in pos]
            ab = sector_angles(s_ring, angles, flows, self.symmetric)
            pieces = [Piece("sector", a0=a0, a1=a1) for a0, a1 in ab]
        for bg, piece in zip(self.branches, pieces):
            bg.piece = piece
        for bg in self.branches:
            if bg.twin:
                continue
            end = piece_ring(s_ring, bg.piece, wall_half, n)
            c = polygon_centroid(end)
            bg.branch_end = end - c
            bg.branch_end_offset = c[0] * U + c[1] * V
        for bg in self.branches:
            if bg.twin:  # the exact mirror image of its primary (a -> -a in the trunk frame, order reversed)
                src = next(b for b in self.branches if not b.twin and b.inlet == bg.inlet)
                r = src.branch_end * np.array([-1.0, 1.0])
                bg.branch_end = np.roll(r[::-1], 1, axis=0)
                bg.branch_end_offset = mirrored(src.branch_end_offset)

    # ---- stations -----------------------------------------------------------------------------------------------------------
    def _build_stations(self) -> None:
        spec = self.spec
        bg0 = self.branches[0]
        cl0 = bg0.centreline
        lb0, lt = cl0.length_branch, cl0.length_trunk
        total0 = cl0.length

        def along(curve, count, extra=()):
            dist = np.linspace(0.0, curve.length, count)
            if len(extra):  # planes that must carry a station: merge those closer than 1e-6 m to a regular one
                dist = np.unique(np.round(np.r_[dist, [d for d in extra if np.abs(dist - d).min() > 1e-6]], 12))
            t = curve.t_at_length(dist)
            return dist, curve.points(t), curve.tangents(t)

        extra = [d for d in (spec.overlap, spec.splitter_length, spec.splitter_length + spec.overlap)
                 if 1e-9 < d < lt - 1e-9] if self.n_branches > 1 else []
        dt, pt, tt = along(cl0.trunk, spec.n_stations_trunk, extra)
        vt = transport_frames(tt, symmetric_v(tt[0]))
        sj0 = bg0.s_junction
        eps = 1e-6
        trunk_keys = [(sj0, self.trunk_start)] + [k for k in bg0.rings if sj0 + eps < k[0] < 1.0] + [bg0.rings[-1]]

        def build(part, dist, pts, tans, vs, s, offset, area_fn, keys, last_ring=None):
            out = []
            for i in range(len(dist)):
                T, V = tans[i], vs[i]
                U = np.cross(T, V)
                if last_ring is not None and i == len(dist) - 1:
                    ring = last_ring  # the branch end is exactly its piece of the trunk start section
                    area = polygon_area(ring)
                else:
                    area = float(area_fn(i))
                    ring = self.ring_at(float(s[i]), area, keys)
                out.append(Station(part, float(s[i]), float(dist[i] + offset), pts[i], T, U, V, area, ring))
            return out

        areas0 = bg0.areas
        self.trunk = build("trunk", dt, pt, tt, vt, (dt + lb0) / total0, lb0,
                           lambda i: areas0.trunk_area(dt[i] / lt if lt > 0 else 1.0), trunk_keys)
        for bg in self.branches:
            if bg.twin:
                continue
            cl = bg.centreline
            lb = cl.length_branch
            db, pb, tb = along(cl.branch, spec.n_stations_branch)
            vb = transport_frames(tb, bg.plane.frame()[1])
            end_b, end_t = vb[-1], vt[0]
            twist = math.atan2(float(np.dot(np.cross(end_b, end_t), tb[-1])), float(np.dot(end_b, end_t)))
            frac = db / lb if lb > 0 else np.zeros_like(db)
            vb = np.array([_rotate(v, t, twist * f) for v, t, f in zip(vb, tb, frac)])
            sj = bg.s_junction
            keys = [k for k in bg.rings if k[0] < sj - eps and k[0] < 1.0] + [(sj, bg.branch_end)]
            bg.stations = build("branch", db, pb, tb, vb, db / cl.length, 0.0,
                                lambda i, db=db, lb=lb, bg=bg: bg.areas.branch_area(db[i] / lb if lb > 0 else 1.0),
                                keys, bg.branch_end)
        for bg in self.branches:
            if bg.twin:  # mirrored copies of the primary's stations
                src = next(b for b in self.branches if not b.twin and b.inlet == bg.inlet)
                bg.stations = [Station(s.part, s.s, s.dist, mirrored(s.centre), mirrored(s.T), mirrored(s.U),
                                       mirrored(s.V), s.area, s.ring) for s in src.stations]

    def stations(self) -> list[Station]:
        return self.branches[0].stations + self.trunk

    # ---- rings -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def place(st: Station, ring: np.ndarray) -> np.ndarray:
        return st.centre + ring[:, :1] * st.U + ring[:, 1:] * st.V

    def inner_ring(self, st: Station) -> np.ndarray:
        return self.place(st, st.ring)

    def outer_ring(self, st: Station) -> np.ndarray:
        return self.place(st, offset_ring(st.ring, self.spec.wall_thickness))

    def lip_ring(self, bg: BranchGeom | None = None) -> np.ndarray:
        """The inlet inner section moved ``lip_extension`` upstream (along +n, out of the opening)."""
        bg = bg or self.branches[0]
        st = bg.stations[0]
        return self.inner_ring(st) - self.spec.lip_extension * st.T

    # ---- bodies --------------------------------------------------------------------------------------------------------------
    def _trunk_piece(self, st: Station, bg: BranchGeom) -> np.ndarray:
        """A branch's piece of a trunk station, for the branch negative continuing into the trunk (placed in 3D)."""
        scale = math.sqrt(st.area / polygon_area(self.trunk_start))
        return self.place(st, piece_ring(st.ring, bg.piece, self._wall_half if self.n_branches > 1 else 0.0, self.n,
                                         scale))

    def bodies(self) -> dict[str, Body]:
        out: dict[str, Body] = {}
        spec = self.spec
        bg0 = self.branches[0]
        lb0 = bg0.centreline.length_branch
        multi = self.n_branches > 1
        split = multi and spec.splitter_length > 0.0
        reach = (spec.splitter_length if split else 0.0) + spec.overlap  # how far the branches go into the trunk
        into = [s for s in self.trunk if 1e-9 < s.dist - lb0 <= reach + 1e-9]
        pos_t = np.array([self.outer_ring(s) for s in self.trunk])
        dist_t = np.array([s.dist for s in self.trunk])
        if split:
            keep = [s for s in self.trunk if s.dist - lb0 >= spec.splitter_length - 1e-9]
            neg_t = np.array([self.inner_ring(s) for s in keep])
            dist_nt = np.array([s.dist for s in keep])
        else:
            first = self.trunk[0]
            rings = [self.inner_ring(first) - spec.overlap * first.T] if (multi and spec.overlap > 0.0) else []
            neg_t = np.array(rings + [self.inner_ring(s) for s in self.trunk])
            dist_nt = np.r_[[first.dist - spec.overlap] if (multi and spec.overlap > 0.0) else [], dist_t]
        for bg in self.branches:
            if bg.twin:
                continue
            lb = bg.centreline.length_branch
            neg_b = np.array([self.lip_ring(bg)] + [self.inner_ring(s) for s in bg.stations]
                             + [self._trunk_piece(s, bg) for s in into])
            dist_b = np.r_[-spec.lip_extension, [s.dist for s in bg.stations], [lb + s.dist - lb0 for s in into]]
            pos_b = np.array([self.outer_ring(s) for s in bg.stations])
            dist_pb = np.array([s.dist for s in bg.stations])
            for kind, tag, rb, db in (("negative", "neg", neg_b, dist_b), ("positive", "pos", pos_b, dist_pb)):
                out[f"{tag}_branch_{bg.name}"] = self._body(f"{tag}_branch_{bg.name}", kind, "branch", bg.name, rb, db)
                if any(b.twin and b.inlet == bg.inlet for b in self.branches):
                    twin = f"{bg.name}_mirror"
                    out[f"{tag}_branch_{twin}"] = self._body(f"{tag}_branch_{twin}", kind, "branch", twin,
                                                             mirrored(rb)[:, ::-1], db)
        out["neg_trunk"] = self._body("neg_trunk", "negative", "trunk", "C", neg_t, dist_nt)
        out["pos_trunk"] = self._body("pos_trunk", "positive", "trunk", "C", pos_t, dist_t)
        return out

    @staticmethod
    def _body(name, kind, part, side, rings, dist) -> Body:
        return Body(name, kind, part, side, rings, np.asarray(dist), tube_mesh(rings, True), tube_mesh(rings, False))

    # ---- numbers --------------------------------------------------------------------------------------------------------------
    def cone_half_angles(self, branch: int = 0) -> dict[str, np.ndarray]:
        """Equivalent conical diffuser half-angle (degrees) of each segment between stations, per part."""
        out = {}
        for part, sts in (("branch", self.branches[branch].stations), ("trunk", self.trunk)):
            r = np.sqrt(np.array([s.area for s in sts]) / math.pi)
            ds = np.diff([s.dist for s in sts])
            out[part] = np.degrees(np.arctan2(np.diff(r), np.maximum(ds, 1e-12)))
        return out
