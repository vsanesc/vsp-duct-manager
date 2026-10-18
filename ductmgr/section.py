# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The fuselage section in the inlet plane: plane ∩ skin as a 2D polyline in (u, v), chine corners, nearest-point
snapping, arcs between two points on the curve, and signed distance to the skin.

Everything works in the plane's (u, v) coordinates (see ``geometry.InletPlane``), never in (y, z).
"""
import math
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from ductmgr.geometry import InletPlane, polygon_area

CHINE_ANGLE_DEG = 30.0  # a vertex whose turning angle exceeds this is a chine corner
MERGE_TOL = 1e-9  # m: consecutive section points closer than this are merged
ARC_EPS = 1e-12  # m: tolerance when deciding which polyline vertices lie strictly inside an arc


@dataclass(frozen=True)
class Section:
    points: np.ndarray  # (N, 2) ordered polyline; for a closed loop the first point is not repeated
    closed: bool
    chines: np.ndarray  # indices into ``points`` of chine corners
    cum: np.ndarray  # (N,) arclength of each vertex along the polyline, cum[0] = 0
    length: float  # total length (including the closing segment of a loop)

    @property
    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Start and end points of every segment, each (S, 2)."""
        a = self.points
        b = np.roll(self.points, -1, axis=0)
        return (a, b) if self.closed else (a[:-1], b[:-1])


# ---- construction ------------------------------------------------------------------------------------
def turning_angles(points: np.ndarray, closed: bool) -> np.ndarray:
    """Absolute turning angle in degrees at every vertex (0 at the ends of an open polyline)."""
    d_in = points - np.roll(points, 1, axis=0)
    d_out = np.roll(points, -1, axis=0) - points
    cross = d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0]
    dot = np.einsum("ij,ij->i", d_in, d_out)
    angle = np.degrees(np.abs(np.arctan2(cross, dot)))
    if not closed:
        angle[0] = angle[-1] = 0.0
    return angle


def make_section(points: np.ndarray, closed: bool) -> Section:
    pts = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > MERGE_TOL]
    pts = pts[keep]
    if closed and len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= MERGE_TOL:
        pts = pts[:-1]
    if closed and polygon_area(pts) < 0.0:
        pts = pts[::-1].copy()  # counter-clockwise
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    if not closed:
        seg[-1] = 0.0
    cum = np.r_[0.0, np.cumsum(seg[:-1])]
    chines = np.flatnonzero(turning_angles(pts, closed) > CHINE_ANGLE_DEG)
    return Section(pts, closed, chines, cum, float(seg.sum()))


def _segments_from_lines(lines: np.ndarray) -> np.ndarray:
    pairs = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1 : i + 1 + n]
        pairs.extend(zip(ids[:-1], ids[1:]))
        i += n + 1
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _trace_chains(n_points: int, seg: np.ndarray) -> list[tuple[list[int], bool]]:
    """Connected components of a segment soup as ordered index chains: (indices, is_closed_loop)."""
    adj: dict[int, list[int]] = {i: [] for i in range(n_points)}
    for a, b in seg:
        if a != b:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
    seen: set[int] = set()
    chains = []
    # open chains first so they are walked from an end; loops afterwards
    order = sorted((i for i in adj if adj[i]), key=lambda i: len(adj[i]) != 1)
    for start in order:
        if start in seen:
            continue
        chain, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [j for j in adj[cur] if j != prev and j not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            seen.add(cur)
        closed = len(chain) > 2 and chain[0] in adj[chain[-1]]
        chains.append((chain, closed))
    return chains


def compute_section(mesh: pv.PolyData, plane: InletPlane) -> Section | None:
    """The curve where the inlet plane cuts the skin, in the plane's (u, v); the component nearest the origin.

    None when the plane misses the mesh.
    """
    cut = mesh.slice(normal=plane.normal, origin=plane.origin).clean()
    if cut.n_points == 0 or cut.n_lines == 0:
        return None
    seg = _segments_from_lines(np.asarray(cut.lines))
    uv_all = plane.to_uv(np.asarray(cut.points))
    chains = _trace_chains(len(uv_all), seg)
    if not chains:
        return None
    best = min(chains, key=lambda c: float(np.min(np.linalg.norm(uv_all[c[0]], axis=1))))
    return make_section(uv_all[best[0]], best[1])


# ---- queries on the curve -------------------------------------------------------------------------------
def nearest_on_section(section: Section, uv) -> tuple[np.ndarray, float, float]:
    """Nearest point on the section curve to ``uv``: (point, arclength position, distance)."""
    p = np.asarray(uv, dtype=float)
    a, b = section.segments
    d = b - a
    seg_len2 = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", p - a, d) / np.where(seg_len2 > 0, seg_len2, 1.0), 0.0, 1.0)
    foot = a + t[:, None] * d
    dist = np.linalg.norm(foot - p, axis=1)
    k = int(np.argmin(dist))
    s = float(section.cum[k] + t[k] * math.sqrt(seg_len2[k]))
    return foot[k], s, float(dist[k])


def point_at(section: Section, s: float) -> np.ndarray:
    """The curve point at arclength position ``s`` (wrapped for a closed loop)."""
    pts, cum, L = section.points, section.cum, section.length
    if section.closed:
        s = s % L
        ext_pts = np.vstack([pts, pts[:1]])
        ext_cum = np.r_[cum, L]
    else:
        s = min(max(s, 0.0), L)
        ext_pts, ext_cum = pts, cum
    return np.array([np.interp(s, ext_cum, ext_pts[:, 0]), np.interp(s, ext_cum, ext_pts[:, 1])])


def _forward_path(section: Section, sa: float, sb: float) -> np.ndarray:
    """Points from position ``sa`` to ``sb`` travelling in the direction of increasing arclength."""
    L, pts, cum = section.length, section.points, section.cum
    if section.closed:
        sb = sb if sb >= sa else sb + L
        ext_cum = np.r_[cum, cum + L]
        ext_pts = np.vstack([pts, pts])
    else:
        ext_cum, ext_pts = cum, pts
    inner = ext_pts[(ext_cum > sa + ARC_EPS) & (ext_cum < sb - ARC_EPS)]
    return np.vstack([point_at(section, sa), inner, point_at(section, sb)])


def arc_path(section: Section, s0: float, s1: float) -> tuple[np.ndarray, float]:
    """The shorter arc of the curve between two arclength positions: (points incl. both ends, length).

    Chine corners (polyline vertices) between the ends are included. An open curve has a single arc.
    """
    if section.closed:
        L = section.length
        forward = (s1 - s0) % L
        if forward <= L - forward:
            path, length = _forward_path(section, s0, s1), forward
        else:
            path, length = _forward_path(section, s1, s0)[::-1], L - forward
    elif s1 >= s0:
        path, length = _forward_path(section, s0, s1), s1 - s0
    else:
        path, length = _forward_path(section, s1, s0)[::-1], s0 - s1
    return path, float(length)


def points_in_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd test: True where a point lies inside the closed polygon ``poly`` (N, 2)."""
    poly = np.asarray(poly, dtype=float)
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    a, b = poly, np.roll(poly, -1, axis=0)
    px, py = pts[:, 0][:, None], pts[:, 1][:, None]
    straddles = (a[:, 1][None, :] > py) != (b[:, 1][None, :] > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = a[:, 0][None, :] + (py - a[:, 1][None, :]) * (b[:, 0] - a[:, 0])[None, :] / (
            b[:, 1] - a[:, 1]
        )[None, :]
    return (straddles & (x_cross > px)).sum(axis=1) % 2 == 1


def inside_mask(section: Section, pts: np.ndarray) -> np.ndarray:
    """True where a point is inside the (closed) section; always False for an open curve."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    if not section.closed:
        return np.zeros(len(pts), dtype=bool)
    return points_in_polygon(section.points, pts)


def distance_to_section(section: Section, pts: np.ndarray) -> np.ndarray:
    """Unsigned distance from each point to the curve."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    a, b = section.segments
    d = b - a
    seg_len2 = np.einsum("ij,ij->i", d, d)
    rel = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("mij,ij->mi", rel, d) / np.where(seg_len2 > 0, seg_len2, 1.0)[None, :], 0.0, 1.0)
    foot = a[None, :, :] + t[:, :, None] * d[None, :, :]
    return np.linalg.norm(foot - pts[:, None, :], axis=2).min(axis=1)


def signed_distance(section: Section, pts: np.ndarray) -> np.ndarray:
    """Distance to the skin: positive outside the fuselage section, negative inside (unsigned if open)."""
    dist = distance_to_section(section, pts)
    return np.where(inside_mask(section, pts), -dist, dist)
