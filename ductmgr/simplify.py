# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Reduce a section to N vertices with Bézier edges fitted to the original outline, preserving its area.

The vertices of the result are a subset of the original ones: the ones whose removal moves the outline least are
dropped first (greedy, the error of a removal being the worst distance of the original outline from the single
cubic Bézier that replaces the edges around the vertex). Each remaining edge is a cubic Bézier fitted to the
original outline between its two vertices (least squares on the handle lengths, tangents taken from the original
outline, chord-length parametrisation refined a few times). A vertex where the original outline turns by more
than ``CORNER_DEG`` stays a corner; the others are smooth (G1). With ``symmetric`` the vertex set is kept mirror
symmetric about u = 0 (a vertex is removed together with its mirror image). The result is moved onto the original
centroid and scaled to the original area; the deviation is measured after that, in both directions.
"""
import math
from dataclasses import dataclass

import numpy as np

from ductmgr import outline as ol
from ductmgr.geometry import polygon_area, polygon_centroid

CORNER_DEG = 25.0  # the original outline turns by more than this at a vertex: it stays a corner
STRAIGHT_TOL = 2e-4  # of the section size: an edge this flat is made straight
FIT_ITERATIONS = 4  # reparametrisations of the least-squares fit
MAX_HANDLE = 1.5  # a fitted handle never exceeds this many chords
CURVE_SAMPLES = 64  # per fitted edge when the deviation is measured
CHUNK = 400


@dataclass(frozen=True)
class Simplified:
    shape: ol.SectionShape
    n_before: int
    n_after: int
    max_dev: float  # m, the worst distance between the two outlines (either way)
    rms_dev: float  # m
    rel_max: float  # max_dev over the radius of the circle of the same area
    area: float  # m^2, the same as before

    def text(self) -> str:
        return (f"{self.n_before} -> {self.n_after} vertices, curved edges fitted, area kept at {self.area:.5f} m^2; "
                f"deviation max {1000 * self.max_dev:.2f} mm ({100 * self.rel_max:.2f} % of the equivalent radius), "
                f"rms {1000 * self.rms_dev:.2f} mm")


def _distances(points: np.ndarray, poly: np.ndarray, closed: bool) -> np.ndarray:
    """Distance of every point to the polyline ``poly``."""
    a = poly
    b = np.roll(poly, -1, axis=0) if closed else poly[1:]
    a = a if closed else poly[:-1]
    edge = b - a
    length2 = np.maximum((edge**2).sum(axis=1), 1e-300)
    out = np.empty(len(points))
    for i in range(0, len(points), CHUNK):
        q = points[i:i + CHUNK]
        t = np.clip(((q[:, None, :] - a[None]) * edge[None]).sum(axis=2) / length2[None], 0.0, 1.0)
        nearest = a[None] + t[:, :, None] * edge[None]
        out[i:i + CHUNK] = np.sqrt(((q[:, None, :] - nearest) ** 2).sum(axis=2)).min(axis=1)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


class _Outline:
    """The original outline as a dense closed polyline with the position of every original vertex in it."""

    def __init__(self, shape: ol.SectionShape):
        paths, _, _ = ol.edge_paths(shape)
        self.pts = ol.build_outline(paths)
        self.m = len(self.pts)
        self.vidx = np.cumsum([0] + [len(p) - 1 for p in paths[:-1]])
        self.scale = max(float(np.ptp(self.pts, axis=0).max()), 1e-12)

    def span(self, i0: int, i1: int) -> np.ndarray:
        """Dense points from vertex-position ``i0`` to ``i1`` (inclusive, going forward around the outline)."""
        a, b = int(self.vidx[i0]), int(self.vidx[i1])
        idx = np.arange(a, b + 1) if b > a else np.r_[np.arange(a, self.m), np.arange(0, b + 1)]
        return self.pts[idx]

    def tangents(self, vertex: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """(arriving direction, leaving direction, is a corner) of the original outline at an original vertex."""
        k = int(self.vidx[vertex])
        tin = _unit(self.pts[k] - self.pts[(k - 1) % self.m])
        tout = _unit(self.pts[(k + 1) % self.m] - self.pts[k])
        angle = math.degrees(math.acos(float(np.clip(tin @ tout, -1.0, 1.0))))
        return tin, tout, angle > CORNER_DEG


def _bernstein(t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return np.c_[u**3, 3 * u**2 * t, 3 * u * t**2, t**3]


def _fit_edge(d: np.ndarray, ta: np.ndarray, tb: np.ndarray, scale: float):
    """(handles or None for a straight edge, max error) of the cubic Bézier through the end points of the dense
    points ``d`` with end tangents ``ta`` (leaving the first) and ``tb`` (arriving at the last)."""
    a, b = d[0], d[-1]
    chord = float(np.linalg.norm(b - a))
    inner = d[1:-1]
    if len(inner) == 0 or chord <= 0.0:
        return None, 0.0
    off = np.abs((b - a)[0] * (inner[:, 1] - a[1]) - (b - a)[1] * (inner[:, 0] - a[0])) / chord
    if off.max() <= STRAIGHT_TOL * scale:
        return None, float(off.max())
    seg = np.linalg.norm(np.diff(d, axis=0), axis=1)
    t = np.r_[0.0, np.cumsum(seg)] / max(float(seg.sum()), 1e-300)
    grid = np.linspace(0.0, 1.0, 200)
    alpha = beta = chord / 3.0
    for _ in range(FIT_ITERATIONS):
        bern = _bernstein(t)
        x1, x2 = bern[:, 1:2] * ta, -bern[:, 2:3] * tb
        r = d - (bern[:, 0] + bern[:, 1])[:, None] * a - (bern[:, 2] + bern[:, 3])[:, None] * b
        m = np.array([[(x1 * x1).sum(), (x1 * x2).sum()], [(x1 * x2).sum(), (x2 * x2).sum()]])
        rhs = np.array([(x1 * r).sum(), (x2 * r).sum()])
        if abs(np.linalg.det(m)) > 1e-18 * max(chord, 1e-12) ** 4:
            alpha, beta = np.linalg.solve(m, rhs)
        if not (0.0 < alpha <= MAX_HANDLE * chord and 0.0 < beta <= MAX_HANDLE * chord):
            alpha = beta = chord / 3.0
        ctrl = np.array([a, a + alpha * ta, b - beta * tb, b])
        curve = _bernstein(grid) @ ctrl
        nearest = np.argmin(((d[:, None, :] - curve[None]) ** 2).sum(axis=2), axis=1)
        t = grid[nearest]
        t[0], t[-1] = 0.0, 1.0
    ctrl = np.array([a, a + alpha * ta, b - beta * tb, b])
    err = float(_distances(d, _bernstein(np.linspace(0, 1, CURVE_SAMPLES)) @ ctrl, closed=False).max())
    return (alpha * ta, -beta * tb), err


def simplify(shape: ol.SectionShape, n: int, symmetric: bool = False) -> Simplified:
    """``shape`` reduced to ``n`` vertices (a subset of its own) with fitted curved edges, at the same area.
    ``symmetric``: keep the vertex set mirror symmetric about u = 0 (the shape must be)."""
    count = shape.n
    if n < ol.MIN_VERTICES:
        raise ValueError(f"a section needs at least {ol.MIN_VERTICES} vertices")
    if n >= count:
        raise ValueError(f"the section has {count} vertices: ask for fewer than that")
    dense = _Outline(shape)
    partners = ol.vertex_partners(shape) if symmetric else [None] * count
    if symmetric and any(p is None for p in partners):
        raise ValueError("the section is not mirror symmetric")
    kept = list(range(count))
    cache: dict = {}

    def fit(a: int, b: int):
        """Fit of the span from original vertex ``a`` to ``b`` (their ends' tangents are the original outline's)."""
        key = (a, b)
        if key not in cache:
            ta = dense.tangents(a)
            tb = dense.tangents(b)
            cache[key] = _fit_edge(dense.span(a, b), ta[1], tb[0], dense.scale)
        return cache[key]

    def removal_cost(v: int) -> float:
        p = kept.index(v)
        return fit(kept[p - 1], kept[(p + 1) % len(kept)])[1]

    # mirror pairs go together, a vertex on the axis alone: one axis vertex must go when the number to remove is odd
    axis_left = (count - n) % 2 if symmetric else 0
    if symmetric and axis_left and not any(p == v for v, p in enumerate(partners)):
        raise ValueError(f"cannot reach {n} vertices keeping the section mirror symmetric: its vertices come in mirror "
                         f"pairs; ask for {n - 1} or {n + 1}")
    while len(kept) > n:
        options = []
        for v in kept:
            if symmetric and partners[v] == v and not axis_left:
                continue  # the axis vertices that stay: only the odd one is removed
            group = {v, partners[v]} if symmetric else {v}
            if group <= set(kept) and len(kept) - len(group) >= n:
                options.append((max(removal_cost(g) for g in group), v, group))
        if not options:
            raise ValueError(f"cannot reach {n} vertices keeping the section mirror symmetric; ask for {n - 1} or {n + 1}")
        _, _, group = min(options, key=lambda o: (o[0], o[1]))
        kept = [v for v in kept if v not in group]
        if symmetric and len(group) == 1 and partners[next(iter(group))] == next(iter(group)):
            axis_left = 0

    k = len(kept)
    verts, edges, handles, snapped = [], [], [], []
    turning = [dense.tangents(v) for v in kept]
    fits = [fit(kept[i], kept[(i + 1) % k]) for i in range(k)]
    for i in range(k):
        verts.append(tuple(float(c) for c in shape.vertices[kept[i]]))
        (hs, _) = fits[i]
        edges.append(ol.STRAIGHT if hs is None else ol.CURVED)
        handles.append(None if hs is None else (tuple(float(c) for c in hs[0]), tuple(float(c) for c in hs[1])))
        snapped.append(False)
    continuity = []
    for i in range(k):
        both_curved = edges[i] == ol.CURVED and edges[i - 1] == ol.CURVED
        continuity.append(ol.SMOOTH if both_curved and not turning[i][2] else ol.CORNER)
    out = ol.SectionShape(tuple(verts), tuple(snapped), tuple(edges), tuple(handles), tuple(continuity))
    if symmetric:  # the two sides were fitted separately: make the edges exact mirror images
        out = _mirror_edges(out)
    # smooth vertices share one tangent direction (the mean of the two fitted ones), keeping each handle's length
    out = _align_smooth(out)

    a_old, c_old = polygon_area(dense.pts), polygon_centroid(dense.pts)
    new_pts = ol.build_outline(ol.edge_paths(out)[0])
    a_new = abs(polygon_area(new_pts))
    shift = c_old - polygon_centroid(new_pts)
    if symmetric:
        shift[0] = 0.0  # stay on the axis
    out = ol.translate(out, shift)
    out = ol.scale_all(out, c_old, math.sqrt(abs(a_old) / a_new))
    new_pts = ol.build_outline(ol.edge_paths(out)[0])
    d1 = _distances(dense.pts, new_pts, closed=True)
    d2 = _distances(new_pts, dense.pts, closed=True)
    both = np.r_[d1, d2]
    radius = math.sqrt(abs(a_old) / math.pi)
    return Simplified(out, count, k, float(both.max()), float(np.sqrt((both**2).mean())), float(both.max()) / radius,
                      abs(polygon_area(new_pts)))


def _align_smooth(shape: ol.SectionShape) -> ol.SectionShape:
    """Make the two handles at every smooth vertex collinear (G1), keeping their lengths."""
    hs = [None if h is None else [np.array(h[0], dtype=float), np.array(h[1], dtype=float)] for h in shape.handles]
    n = shape.n
    for i in range(n):
        if shape.continuity[i] != ol.SMOOTH:
            continue
        out_h, in_h = hs[i][0], hs[i - 1][1]  # the leaving handle of edge i, the arriving one of edge i - 1
        d = _unit(_unit(out_h) - _unit(in_h))
        out_h[:] = d * float(np.linalg.norm(out_h))
        in_h[:] = -d * float(np.linalg.norm(in_h))
    handles = tuple(None if h is None else (tuple(h[0]), tuple(h[1])) for h in hs)
    return shape.with_(handles=handles)


def _mirror_edges(shape: ol.SectionShape) -> ol.SectionShape:
    """Make every edge the exact mirror image (about u = 0) of its partner edge; a self-partner edge is symmetrised."""
    partners = ol.vertex_partners(shape)
    n = shape.n
    mirror = lambda h: (-h[0], h[1])  # noqa: E731
    edges, handles = list(shape.edges), list(shape.handles)
    done = set()
    for e in range(n):
        ep = ol.edge_partner(shape, partners, e)
        if ep is None or e in done:
            continue
        done |= {e, ep}
        if edges[e] == ol.STRAIGHT:
            edges[ep], handles[ep] = ol.STRAIGHT, None
            continue
        h0, h1 = handles[e]
        if ep == e:  # the edge crosses the axis: its two handles are mirror images
            avg = (0.5 * (h0[0] - h1[0]), 0.5 * (h0[1] + h1[1]))
            handles[e] = (avg, mirror(avg))
        else:
            edges[ep], handles[ep] = ol.CURVED, (mirror(h1), mirror(h0))
    return shape.with_(edges=tuple(edges), handles=tuple(handles))
