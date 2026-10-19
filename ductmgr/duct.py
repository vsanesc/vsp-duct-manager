# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Duct geometry: stations along the centrelines, sections morphed between key sections, and the NEGATIVE and
POSITIVE bodies built from the same stations.

For one (right) inlet:

- stations lie along the branch Bézier (inlet section centroid -> junction) and the trunk Bézier (junction ->
  face centre), equally spaced in arc length; ``s`` is the arc-length fraction of the whole path;
- the section at a station is the blend of the two neighbouring key sections (resampled with the arc-length
  correspondence of ``interp``), then scaled about its centroid so its area is exactly A(s) (the INNER section);
- frames are parallel-transported (minimal rotation) and the branch's accumulated roll is distributed over its
  length so it meets the trunk's symmetric frame at the junction;
- at the Y junction the trunk start section is symmetrised about y = 0 and split by that plane into two halves;
  each branch ends with exactly its half (same shape, position, half the area), so the two branch ends tile the
  trunk start section; the branch centrelines end at the halves' centroids and keep the junction tangent (G1);
- NEGATIVE body = inner sections, preceded by a prismatic extension of ``lip_extension`` upstream of the opening
  plane (along the inlet normal), ending in a flat cap at the engine face. Each branch negative continues into
  the trunk as half-sections by ``overlap`` (and by the ``splitter_length``, separated by a wall of
  ``wall_thickness`` on y = 0); with no splitter the trunk negative starts ``overlap`` upstream of the junction;
- POSITIVE body = the inner sections offset outward in their plane by ``wall_thickness``, starting exactly at
  the opening plane (a blunt lip) and running to the face.

The intended model in OpenVSP is (fuselage ∪ positives) − negatives. The left inlet is the exact mirror image
about y = 0 of everything built here.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from ductmgr import interp
from ductmgr import outline as ol
from ductmgr.areacurve import AreaCurves, AreaPoints
from ductmgr.centreline import Centreline, CentrelineParams
from ductmgr.geometry import InletPlane, polygon_area, polygon_centroid

LIP_MODES = ("blunt",)  # extensible: other lip shapes later
DEFAULT_LIP_MODE = "blunt"
DEFAULT_OVERLAP = 0.02  # m, ducts.junction.overlap when not configured
MITRE_LIMIT = 4.0  # an offset corner is never moved further than this many wall thicknesses
UP = np.array([0.0, 0.0, 1.0])
REFINE_SUBDIVISIONS = 3  # extra rings between stations when the external region is measured


@dataclass(frozen=True)
class KeyOutline:
    """A key section for lofting: its closed outline and corner vertices, in its own (u, v)."""

    s: float
    outline: np.ndarray
    vertices: np.ndarray


@dataclass(frozen=True)
class DuctSpec:
    plane: InletPlane
    keys: tuple[KeyOutline, ...]  # sorted by s, keys[0].s == 0 (the inlet); the face circle is added here
    face_centre: np.ndarray
    face_axis: np.ndarray
    face_diameter: float
    centreline: CentrelineParams = field(default_factory=CentrelineParams)
    area_points: AreaPoints = field(default_factory=AreaPoints)
    area_law: str = "cosine"
    n_branches: int = 2
    n_stations_branch: int = 12
    n_stations_trunk: int = 12
    n_points: int = interp.DEFAULT_SAMPLES
    wall_thickness: float = 0.004
    lip_extension: float = 0.05
    lip_mode: str = DEFAULT_LIP_MODE
    junction_fraction: float = 0.5
    inlet_area: float | None = None  # A(0); the area of the sketched inlet outline when not given
    splitter_length: float = 0.0  # m: the branch negatives stay separate (half-sections) this far into the trunk
    overlap: float = DEFAULT_OVERLAP  # m: each branch negative reaches this far beyond the junction plane (inside the duct)


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
    name: str  # e.g. "neg_branch_R"
    kind: str  # "negative" or "positive"
    part: str  # "branch" or "trunk"
    side: str  # "R", "L" or "C" (the trunk is one body)
    rings: np.ndarray  # (R, N, 3)
    dist: np.ndarray  # (R,) arc length of each ring from the opening plane (negative upstream of it)
    mesh: pv.PolyData  # closed, outward-oriented, triangulated
    lateral: pv.PolyData  # the open tube without end caps


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


def half_ring(ring: np.ndarray, normal, offset: float, n: int) -> np.ndarray:
    """The half of ``ring`` on the side ``normal . p >= offset`` as an ``n``-point ring, starting at the upper
    corner on the cut. Every vertex of the half is kept (so the half is geometrically exactly half of ``ring``) and
    the remaining points are spread along the straight cut."""
    half = clip_ring(ring, normal, offset)
    nrm = np.asarray(normal, dtype=float)
    on_cut = np.flatnonzero(np.abs(half @ nrm - offset) < 1e-9)
    if len(on_cut) < 2:
        raise ValueError("the junction section could not be split")
    up = int(on_cut[np.argmax(half[on_cut, 1])])
    low = int(on_cut[np.argmin(half[on_cut, 1])])
    m = len(half)
    if m > n:
        return interp.resample_keep(half, [up, low], n)
    seq = np.roll(half, -up, axis=0)  # starts at the upper corner
    k = (low - up) % m  # the lower corner's position in seq
    if k == 1:  # the cut comes first: u -> l is the straight side
        pad = np.linspace(seq[0], seq[1], n - m + 2)[1:-1]
        return np.vstack([seq[:1], pad, seq[1:]])
    pad = np.linspace(seq[-1], seq[0], n - m + 2)[1:-1]  # the cut closes the loop: l ... -> u
    return np.vstack([seq, pad]) if k == m - 1 else interp.resample_keep(half, [up, low], n)


def mirrored(points: np.ndarray) -> np.ndarray:
    return points * np.array([1.0, -1.0, 1.0])


class Duct:
    """Everything built from a ``DuctSpec``."""

    def __init__(self, spec: DuctSpec, plan_only: bool = False):
        """``plan_only`` stops after the centrelines and the area curves (cheap: what the explorer needs to size
        the key sections), without stations or bodies."""
        if spec.lip_mode not in LIP_MODES:
            raise ValueError(f"ducts.lip_mode must be one of {', '.join(LIP_MODES)} (got {spec.lip_mode!r}); "
                             "other lip shapes are not implemented yet")
        if spec.wall_thickness <= 0.0 or spec.lip_extension < 0.0:
            raise ValueError("ducts.wall_thickness must be positive and ducts.lip_extension not negative")
        if spec.n_stations_branch < 2 or spec.n_stations_trunk < 2:
            raise ValueError("at least 2 stations per branch / trunk are needed")
        if spec.splitter_length < 0.0 or spec.overlap < 0.0:
            raise ValueError("ducts.junction.splitter_length and ducts.junction.overlap must not be negative")
        self.spec = spec
        n = self.n = effective_points(spec.n_points)
        self.rings = self._key_rings(n)
        ring0 = self.rings[0][1]
        self.inlet_centre_uv = polygon_centroid(ring0)
        start = spec.plane.to_3d([self.inlet_centre_uv])[0]
        u0, v0, n0 = spec.plane.frame()
        self.start_dir = -n0
        a_in = spec.inlet_area if spec.inlet_area is not None else polygon_area(ring0)
        a_face = 0.25 * math.pi * spec.face_diameter**2
        self.areas = AreaCurves(a_in, a_face, spec.area_points, spec.area_law, spec.n_branches)
        # first pass: where the junction is, to fix the trunk start section and its halves; their centroids then
        # move the branch ends off y = 0 (the junction s changes very slightly: the section shape is not redone)
        pre = Centreline(start, self.start_dir, spec.face_centre, spec.face_axis, spec.centreline,
                         spec.junction_fraction)
        self._junction_sections(pre)
        self.centreline = Centreline(start, self.start_dir, spec.face_centre, spec.face_axis, spec.centreline,
                                     spec.junction_fraction, self.branch_end_offset)
        self.s_junction = self.centreline.s_junction
        if spec.n_branches == 2 and spec.splitter_length + spec.overlap >= self.centreline.length_trunk:
            raise ValueError("ducts.junction.splitter_length + overlap must be shorter than the trunk")
        if plan_only:
            return
        self.branch, self.trunk = self._stations(v0)

    # ---- the Y junction ---------------------------------------------------------------------------------------------------
    def _junction_sections(self, cl: Centreline) -> None:
        """Trunk start section S, the branch end section (its half) and the offset of the half's centroid."""
        spec, n = self.spec, self.n
        aj = self.areas.junction_area
        ring = self.ring_at(cl.s_junction, aj, self.rings)
        T = cl.junction_tangent
        V = symmetric_v(T)
        U = np.cross(T, V)
        self.trunk_frame0 = (T, U, V)
        if spec.n_branches == 2:
            s_ring = symmetric_ring(ring, n)
            self.split_normal = np.array([1.0 if U[1] > 0 else -1.0, 0.0])  # the right branch has y > 0
            self.split_offset = 0.5 * spec.wall_thickness if spec.splitter_length > 0 else 0.0
            half = half_ring(s_ring, self.split_normal, self.split_offset, n)
            c = polygon_centroid(half)
            self.branch_end = half - c
            self.branch_end_offset = c[0] * U + c[1] * V
        else:
            s_ring = ring
            self.split_normal, self.split_offset = None, 0.0
            self.branch_end = ring * math.sqrt(1.0 / spec.n_branches)
            self.branch_end_offset = np.zeros(3)
        self.trunk_start = s_ring

    # ---- sections ----------------------------------------------------------------------------------------------------
    def _key_rings(self, n: int) -> list[tuple[float, np.ndarray]]:
        keys = list(self.spec.keys)
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

    # ---- stations -----------------------------------------------------------------------------------------------------------
    def _stations(self, v0: np.ndarray) -> tuple[list[Station], list[Station]]:
        spec, cl = self.spec, self.centreline
        total = cl.length
        lb, lt = cl.length_branch, cl.length_trunk

        def along(curve, count, offset, extra=()):
            dist = np.linspace(0.0, curve.length, count)
            if len(extra):  # planes that must carry a station: merge those closer than 1e-6 m to a regular one
                dist = np.unique(np.round(np.r_[dist, [d for d in extra if np.abs(dist - d).min() > 1e-6]], 12))
            t = curve.t_at_length(dist)
            return dist, curve.points(t), curve.tangents(t), (dist + offset) / total

        db, pb, tb, sb = along(cl.branch, spec.n_stations_branch, 0.0)
        extra = [d for d in (spec.overlap, spec.splitter_length, spec.splitter_length + spec.overlap)
                 if 1e-9 < d < lt - 1e-9]
        dt, pt, tt, st = along(cl.trunk, spec.n_stations_trunk, lb, extra)
        # trunk frames start from the side-independent frame; the branch roll is spread so it ends there
        vt = transport_frames(tt, symmetric_v(tt[0]))
        vb = transport_frames(tb, v0)
        end_b, end_t = vb[-1], vt[0]
        twist = math.atan2(float(np.dot(np.cross(end_b, end_t), tb[-1])), float(np.dot(end_b, end_t)))
        frac = db / lb if lb > 0 else np.zeros_like(db)
        vb = np.array([_rotate(v, t, twist * f) for v, t, f in zip(vb, tb, frac)])

        sj = self.s_junction
        eps = 1e-6
        branch_keys = [k for k in self.rings if k[0] < sj - eps and k[0] < 1.0] + [(sj, self.branch_end)]
        trunk_keys = [(sj, self.trunk_start)] + [k for k in self.rings if sj + eps < k[0] < 1.0] + [self.rings[-1]]

        def build(part, dist, pts, tans, vs, s, offset, area_fn, keys, last_ring=None):
            out = []
            for i in range(len(dist)):
                T, V = tans[i], vs[i]
                U = np.cross(T, V)
                if last_ring is not None and i == len(dist) - 1:
                    ring = last_ring  # the branch end is exactly its half of the trunk start section
                    area = polygon_area(ring)
                else:
                    area = float(area_fn(i))
                    ring = self.ring_at(float(s[i]), area, keys)
                out.append(Station(part, float(s[i]), float(dist[i] + offset), pts[i], T, U, V, area, ring))
            return out

        branch = build("branch", db, pb, tb, vb, sb, 0.0,
                       lambda i: self.areas.branch_area(db[i] / lb if lb > 0 else 1.0), branch_keys, self.branch_end)
        trunk = build("trunk", dt, pt, tt, vt, st, lb,
                      lambda i: self.areas.trunk_area(dt[i] / lt if lt > 0 else 1.0), trunk_keys)
        return branch, trunk

    def stations(self) -> list[Station]:
        return self.branch + self.trunk

    # ---- rings -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def place(st: Station, ring: np.ndarray) -> np.ndarray:
        return st.centre + ring[:, :1] * st.U + ring[:, 1:] * st.V

    def inner_ring(self, st: Station) -> np.ndarray:
        return self.place(st, st.ring)

    def outer_ring(self, st: Station) -> np.ndarray:
        return self.place(st, offset_ring(st.ring, self.spec.wall_thickness))

    def lip_ring(self) -> np.ndarray:
        """The inlet inner section moved ``lip_extension`` upstream (along +n, out of the opening)."""
        st = self.branch[0]
        return self.inner_ring(st) - self.spec.lip_extension * st.T

    # ---- bodies --------------------------------------------------------------------------------------------------------------
    def _trunk_half(self, st: Station) -> np.ndarray:
        """Half-section of a trunk station for a branch negative continuing into the trunk (placed in 3D)."""
        spec = self.spec
        if spec.n_branches == 2:
            ring = half_ring(st.ring, self.split_normal, self.split_offset, self.n)
        else:
            ring = st.ring
        return self.place(st, ring)

    def bodies(self) -> dict[str, Body]:
        out: dict[str, Body] = {}
        spec = self.spec
        lb = self.centreline.length_branch
        split = spec.n_branches == 2 and spec.splitter_length > 0.0
        reach = (spec.splitter_length if split else 0.0) + spec.overlap  # how far the branches go into the trunk
        into = [s for s in self.trunk if 1e-9 < s.dist - lb <= reach + 1e-9]
        neg_b = np.array([self.lip_ring()] + [self.inner_ring(s) for s in self.branch] + [self._trunk_half(s) for s in into])
        dist_b = np.r_[-spec.lip_extension, [s.dist for s in self.branch], [s.dist for s in into]]
        pos_b = np.array([self.outer_ring(s) for s in self.branch])
        dist_pb = np.array([s.dist for s in self.branch])
        pos_t = np.array([self.outer_ring(s) for s in self.trunk])
        dist_t = np.array([s.dist for s in self.trunk])
        # the trunk negative: from the splitter's end, or from `overlap` upstream of the junction (a prism)
        if split:
            keep = [s for s in self.trunk if s.dist - lb >= spec.splitter_length - 1e-9]
            neg_t = np.array([self.inner_ring(s) for s in keep])
            dist_nt = np.array([s.dist for s in keep])
        else:
            first = self.trunk[0]
            rings = [self.inner_ring(first) - spec.overlap * first.T] if spec.overlap > 0.0 else []
            neg_t = np.array(rings + [self.inner_ring(s) for s in self.trunk])
            dist_nt = np.r_[[first.dist - spec.overlap] if spec.overlap > 0.0 else [], dist_t]
        for kind, tag, rb, db, rt, dt_ in (("negative", "neg", neg_b, dist_b, neg_t, dist_nt),
                                           ("positive", "pos", pos_b, dist_pb, pos_t, dist_t)):
            out[f"{tag}_branch_R"] = self._body(f"{tag}_branch_R", kind, "branch", "R", rb, db)
            out[f"{tag}_branch_L"] = self._body(f"{tag}_branch_L", kind, "branch", "L", mirrored(rb)[:, ::-1], db)
            out[f"{tag}_trunk"] = self._body(f"{tag}_trunk", kind, "trunk", "C", rt, dt_)
        return out

    @staticmethod
    def _body(name, kind, part, side, rings, dist) -> Body:
        return Body(name, kind, part, side, rings, np.asarray(dist), tube_mesh(rings, True), tube_mesh(rings, False))

    # ---- numbers --------------------------------------------------------------------------------------------------------------
    def cone_half_angles(self) -> dict[str, np.ndarray]:
        """Equivalent conical diffuser half-angle (degrees) of each segment between stations, per part."""
        out = {}
        for part, sts in (("branch", self.branch), ("trunk", self.trunk)):
            r = np.sqrt(np.array([s.area for s in sts]) / math.pi)
            ds = np.diff([s.dist for s in sts])
            out[part] = np.degrees(np.arctan2(np.diff(r), np.maximum(ds, 1e-12)))
        return out
