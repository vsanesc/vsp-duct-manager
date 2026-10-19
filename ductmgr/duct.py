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
- NEGATIVE body = inner sections, preceded by a prismatic extension of ``lip_extension`` upstream of the opening
  plane (along the inlet normal), ending in a flat cap at the engine face;
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


def mirrored(points: np.ndarray) -> np.ndarray:
    return points * np.array([1.0, -1.0, 1.0])


class Duct:
    """Everything built from a ``DuctSpec``."""

    def __init__(self, spec: DuctSpec):
        if spec.lip_mode not in LIP_MODES:
            raise ValueError(f"ducts.lip_mode must be one of {', '.join(LIP_MODES)} (got {spec.lip_mode!r}); "
                             "other lip shapes are not implemented yet")
        if spec.wall_thickness <= 0.0 or spec.lip_extension < 0.0:
            raise ValueError("ducts.wall_thickness must be positive and ducts.lip_extension not negative")
        if spec.n_stations_branch < 2 or spec.n_stations_trunk < 2:
            raise ValueError("at least 2 stations per branch / trunk are needed")
        self.spec = spec
        n = spec.n_points
        self.rings = self._key_rings(n)
        ring0 = self.rings[0][1]
        self.inlet_centre_uv = polygon_centroid(ring0)
        start = spec.plane.to_3d([self.inlet_centre_uv])[0]
        u0, v0, n0 = spec.plane.frame()
        self.start_dir = -n0
        self.centreline = Centreline(start, self.start_dir, spec.face_centre, spec.face_axis, spec.centreline,
                                     spec.junction_fraction)
        cl = self.centreline
        self.s_junction = cl.s_junction
        a_in = spec.inlet_area if spec.inlet_area is not None else polygon_area(ring0)
        a_face = 0.25 * math.pi * spec.face_diameter**2
        self.areas = AreaCurves(a_in, a_face, spec.area_points, spec.area_law, spec.n_branches)
        self.branch, self.trunk = self._stations(v0)

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

    def ring_at(self, s: float, area: float) -> np.ndarray:
        """The inner section (N, 2) at global ``s``: blended key sections, centred and scaled to ``area``."""
        keys = self.rings
        i = 0
        while i < len(keys) - 2 and s > keys[i + 1][0]:
            i += 1
        (s0, r0), (s1, r1) = keys[i], keys[i + 1]
        t = 0.0 if s1 <= s0 else min(max((s - s0) / (s1 - s0), 0.0), 1.0)
        r = interp.blend(r0 - polygon_centroid(r0), r1 - polygon_centroid(r1), t)
        r = r - polygon_centroid(r)
        return r * math.sqrt(area / polygon_area(r))

    # ---- stations -----------------------------------------------------------------------------------------------------------
    def _stations(self, v0: np.ndarray) -> tuple[list[Station], list[Station]]:
        spec, cl = self.spec, self.centreline
        total = cl.length
        lb, lt = cl.length_branch, cl.length_trunk

        def along(curve, count, offset):
            dist = np.linspace(0.0, curve.length, count)
            t = curve.t_at_length(dist)
            return dist, curve.points(t), curve.tangents(t), (dist + offset) / total

        db, pb, tb, sb = along(cl.branch, spec.n_stations_branch, 0.0)
        dt, pt, tt, st = along(cl.trunk, spec.n_stations_trunk, lb)
        # trunk frames start from the side-independent frame; the branch roll is spread so it ends there
        vt = transport_frames(tt, symmetric_v(tt[0]))
        vb = transport_frames(tb, v0)
        end_b, end_t = vb[-1], vt[0]
        twist = math.atan2(float(np.dot(np.cross(end_b, end_t), tb[-1])), float(np.dot(end_b, end_t)))
        frac = db / lb if lb > 0 else np.zeros_like(db)
        vb = np.array([_rotate(v, t, twist * f) for v, t, f in zip(vb, tb, frac)])

        def build(part, dist, pts, tans, vs, s, offset, area_fn):
            out = []
            for i in range(len(dist)):
                T, V = tans[i], vs[i]
                U = np.cross(T, V)
                area = float(area_fn(i))
                out.append(Station(part, float(s[i]), float(dist[i] + offset), pts[i], T, U, V, area,
                                   self.ring_at(float(s[i]), area)))
            return out

        branch = build("branch", db, pb, tb, vb, sb, 0.0,
                       lambda i: self.areas.branch_area(db[i] / lb if lb > 0 else 1.0))
        trunk = build("trunk", dt, pt, tt, vt, st, lb,
                      lambda i: self.areas.trunk_area(dt[i] / lt if lt > 0 else 1.0))
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
    def bodies(self) -> dict[str, Body]:
        out: dict[str, Body] = {}
        spec = self.spec
        neg_b = np.array([self.lip_ring()] + [self.inner_ring(s) for s in self.branch])
        dist_b = np.r_[-spec.lip_extension, [s.dist for s in self.branch]]
        pos_b = np.array([self.outer_ring(s) for s in self.branch])
        neg_t = np.array([self.inner_ring(s) for s in self.trunk])
        pos_t = np.array([self.outer_ring(s) for s in self.trunk])
        dist_t = np.array([s.dist for s in self.trunk])
        dist_pb = np.array([s.dist for s in self.branch])
        for kind, tag, rb, db, rt in (("negative", "neg", neg_b, dist_b, neg_t), ("positive", "pos", pos_b, dist_pb, pos_t)):
            out[f"{tag}_branch_R"] = self._body(f"{tag}_branch_R", kind, "branch", "R", rb, db)
            out[f"{tag}_branch_L"] = self._body(f"{tag}_branch_L", kind, "branch", "L", mirrored(rb)[:, ::-1], db)
            out[f"{tag}_trunk"] = self._body(f"{tag}_trunk", kind, "trunk", "C", rt, dist_t)
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
