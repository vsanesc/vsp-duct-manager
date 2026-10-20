# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""A key section's closed outline: any number (>= 3) of vertices joined by straight or curved edges.

Curved edges are cubic Béziers. Edge i runs from vertex i to vertex i+1 (cyclic); its two control handles are
stored as offsets from the vertices they belong to (``h0`` from vertex i, ``h1`` from vertex i+1), so moving a
vertex carries its handles with it. A vertex has a continuity mode that constrains its two handles:

corner     the handles are independent;
smooth     G1: the two handles are collinear (opposite directions), their lengths independent;
symmetric  collinear and of equal length.

Next to a straight edge a smooth vertex aligns its curved handle with that edge. An edge whose two end vertices
are both *snapped* (lies on the fuselage section) follows the skin instead and is not editable.

Coordinates are the section's (u, v) in metres.
"""
import math
from dataclasses import dataclass

import numpy as np

from ductmgr.geometry import polygon_area, polygon_centroid
from ductmgr.section import Section, arc_path, nearest_on_section

STRAIGHT, CURVED = "straight", "curved"
CORNER, SMOOTH, SYMMETRIC = "corner", "smooth", "symmetric"
EDGE_TYPES = (STRAIGHT, CURVED)
CONTINUITY_MODES = (CORNER, SMOOTH, SYMMETRIC)
MIN_VERTICES = 3
BEZIER_SAMPLES = 48  # segments per curved edge when the outline is sampled
HANDLE_FRACTION = 1.0 / 3.0  # a new curved edge starts as the straight chord: handles at 1/3 and 2/3
CIRCLE_KAPPA = 4.0 / 3.0 * math.tan(math.pi / 8.0)  # handle length / radius for a quarter circle
NEAREST_T_SAMPLES = 256  # resolution of the nearest-point search on a Bézier
NEAREST_T_REFINE = 40  # golden-section refinement steps


def _v(p) -> tuple[float, float]:
    return (float(p[0]), float(p[1]))


@dataclass(frozen=True)
class SectionShape:
    vertices: tuple[tuple[float, float], ...]
    snapped: tuple[bool, ...]
    edges: tuple[str, ...]  # per edge: STRAIGHT or CURVED
    handles: tuple  # per edge: None (straight) or ((h0u, h0v), (h1u, h1v)) offsets from its two vertices
    continuity: tuple[str, ...]  # per vertex

    def __post_init__(self):
        n = len(self.vertices)
        if n < MIN_VERTICES:
            raise ValueError(f"a section needs at least {MIN_VERTICES} vertices (got {n})")
        for name in ("snapped", "edges", "handles", "continuity"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have one entry per vertex ({n}), got {len(getattr(self, name))}")
        for e, h in zip(self.edges, self.handles):
            if e not in EDGE_TYPES:
                raise ValueError(f"edge type must be one of {', '.join(EDGE_TYPES)} (got {e!r})")
            if (e == CURVED) != (h is not None):
                raise ValueError("a curved edge needs two handles and a straight edge has none")
        for c in self.continuity:
            if c not in CONTINUITY_MODES:
                raise ValueError(f"continuity must be one of {', '.join(CONTINUITY_MODES)} (got {c!r})")

    @property
    def n(self) -> int:
        return len(self.vertices)

    def edge_is_snapped(self, i: int) -> bool:
        """True when both end vertices are snapped: the edge follows the skin."""
        return bool(self.snapped[i] and self.snapped[(i + 1) % self.n])

    def edge_is_curved(self, i: int) -> bool:
        """A free (not skin-following) curved edge."""
        return self.edges[i] == CURVED and not self.edge_is_snapped(i)

    def control_points(self, i: int) -> np.ndarray:
        """The 4 Bézier control points (absolute) of a curved edge, (4, 2)."""
        if self.edges[i] != CURVED:
            raise ValueError("only a curved edge has Bézier control points")
        a, b = np.array(self.vertices[i]), np.array(self.vertices[(i + 1) % self.n])
        h0, h1 = self.handles[i]
        return np.array([a, a + np.array(h0), b + np.array(h1), b])

    @classmethod
    def polygon(cls, vertices, snapped=None) -> "SectionShape":
        """A straight-edged shape."""
        verts = tuple(_v(p) for p in vertices)
        n = len(verts)
        return cls(verts, tuple(bool(f) for f in (snapped if snapped is not None else [False] * n)),
                   (STRAIGHT,) * n, (None,) * n, (CORNER,) * n)

    def with_(self, **changes) -> "SectionShape":
        data = dict(vertices=self.vertices, snapped=self.snapped, edges=self.edges, handles=self.handles,
                    continuity=self.continuity)
        data.update(changes)
        return SectionShape(**data)


# ---- Bézier helpers -----------------------------------------------------------------------------------------
def bezier_points(ctrl: np.ndarray, samples: int = BEZIER_SAMPLES) -> np.ndarray:
    """Points of the cubic Bézier with control points ``ctrl`` (4, 2), ``samples`` segments, ends included."""
    t = np.linspace(0.0, 1.0, samples + 1)[:, None]
    p0, p1, p2, p3 = ctrl
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def bezier_at(ctrl: np.ndarray, t: float) -> np.ndarray:
    return bezier_points_t(ctrl, np.array([t]))[0]


def bezier_points_t(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)[:, None]
    p0, p1, p2, p3 = ctrl
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def split_bezier(ctrl: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """De Casteljau split at ``t``: the control points of the two halves, each (4, 2)."""
    p0, p1, p2, p3 = (np.asarray(c, dtype=float) for c in ctrl)
    a, b, c = p0 + t * (p1 - p0), p1 + t * (p2 - p1), p2 + t * (p3 - p2)
    d, e = a + t * (b - a), b + t * (c - b)
    f = d + t * (e - d)
    return np.array([p0, a, d, f]), np.array([f, e, c, p3])


def nearest_t_on_bezier(ctrl: np.ndarray, point) -> float:
    """Parameter of the point on the Bézier nearest to ``point`` (dense search, then golden-section refinement)."""
    p = np.asarray(point, dtype=float)
    ts = np.linspace(0.0, 1.0, NEAREST_T_SAMPLES + 1)
    d = np.linalg.norm(bezier_points_t(ctrl, ts) - p, axis=1)
    k = int(np.argmin(d))
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, NEAREST_T_SAMPLES)]
    g = (math.sqrt(5.0) - 1.0) / 2.0

    def dist(t):
        return float(np.linalg.norm(bezier_at(ctrl, t) - p))

    a, b = lo, hi
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = dist(c), dist(e)
    for _ in range(NEAREST_T_REFINE):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a)
            fc = dist(c)
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a)
            fe = dist(e)
    return float((a + b) / 2.0)


# ---- working copy for edits -------------------------------------------------------------------------------------
class _Work:
    """Mutable copy of a SectionShape used inside the edit functions."""

    def __init__(self, shape: SectionShape):
        self.v = [np.array(p, dtype=float) for p in shape.vertices]
        self.snapped = list(shape.snapped)
        self.edges = list(shape.edges)
        self.handles = [None if h is None else [np.array(h[0], dtype=float), np.array(h[1], dtype=float)]
                        for h in shape.handles]
        self.cont = list(shape.continuity)

    @property
    def n(self) -> int:
        return len(self.v)

    def curved(self, e: int) -> bool:
        return self.edges[e] == CURVED and not (self.snapped[e] and self.snapped[(e + 1) % self.n])

    def skin(self, e: int) -> bool:
        return bool(self.snapped[e] and self.snapped[(e + 1) % self.n])

    def freeze(self) -> SectionShape:
        return SectionShape(
            tuple(_v(p) for p in self.v), tuple(self.snapped), tuple(self.edges),
            tuple(None if h is None else (_v(h[0]), _v(h[1])) for h in self.handles), tuple(self.cont))


def _unit(x: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(x))
    return x / n if n > 0.0 else None


def _enforce(w: _Work, i: int, dominant: str = "out") -> None:
    """Make the handles at vertex ``i`` satisfy its continuity mode; ``dominant`` ("out" = the handle of the edge
    leaving the vertex, "in" = of the edge arriving) is the one kept, the other one follows."""
    mode = w.cont[i]
    if mode == CORNER:
        return
    n = w.n
    e_out, e_in = i, (i - 1) % n
    out_c, in_c = w.curved(e_out), w.curved(e_in)
    if out_c and in_c:
        ho, hi = w.handles[e_out][0], w.handles[e_in][1]
        src, tgt = (hi, ho) if dominant == "in" else (ho, hi)
        d = _unit(src)
        if d is None:
            return
        new = -src.copy() if mode == SYMMETRIC else -d * (float(np.linalg.norm(tgt)) or float(np.linalg.norm(src)))
        tgt[:] = new
    elif out_c and not w.skin(e_in):  # a straight edge arrives: leave along its direction
        t = _unit(w.v[i] - w.v[(i - 1) % n])
        if t is not None:
            h = w.handles[e_out][0]
            h[:] = t * float(np.linalg.norm(h))
    elif in_c and not w.skin(e_out):  # a straight edge leaves: arrive along its direction
        t = _unit(w.v[(i + 1) % n] - w.v[i])
        if t is not None:
            h = w.handles[e_in][1]
            h[:] = -t * float(np.linalg.norm(h))


def _enforce_all(w: _Work, indices) -> None:
    for i in indices:
        out_c = w.curved(i % w.n)
        _enforce(w, i % w.n, "out" if out_c else "in")


# ---- editing operations ------------------------------------------------------------------------------------------
def move_vertex(shape: SectionShape, i: int, uv) -> SectionShape:
    """Move vertex ``i``; its handles go with it. Continuity at it and its neighbours is re-applied."""
    w = _Work(shape)
    w.v[i] = np.array(uv, dtype=float)
    _enforce_all(w, [i - 1, i, i + 1])
    return w.freeze()


def move_handle(shape: SectionShape, edge: int, which: int, uv) -> SectionShape:
    """Drag handle ``which`` (0 = at the edge's first vertex, 1 = at its second) to the absolute point ``uv``."""
    w = _Work(shape)
    if not w.curved(edge):
        raise ValueError("only a free curved edge has handles to move")
    anchor = (edge + which) % w.n
    w.handles[edge][which] = np.array(uv, dtype=float) - w.v[anchor]
    _enforce(w, anchor, "out" if which == 0 else "in")
    return w.freeze()


def set_edge_type(shape: SectionShape, edge: int, kind: str) -> SectionShape:
    """Make an edge straight or curved. A new curved edge starts as the straight chord (handles at 1/3, 2/3)."""
    if kind not in EDGE_TYPES:
        raise ValueError(f"edge type must be one of {', '.join(EDGE_TYPES)}")
    w = _Work(shape)
    if w.skin(edge):
        raise ValueError("an edge between two snapped vertices follows the skin and cannot be edited")
    n = w.n
    a, b = w.v[edge], w.v[(edge + 1) % n]
    if kind == CURVED and w.edges[edge] != CURVED:
        w.edges[edge] = CURVED
        w.handles[edge] = [(b - a) * HANDLE_FRACTION, -(b - a) * HANDLE_FRACTION]
        _enforce(w, edge, "in")  # the new edge adapts to the tangent already there
        _enforce(w, (edge + 1) % n, "out")
    elif kind == STRAIGHT and w.edges[edge] != STRAIGHT:
        w.edges[edge] = STRAIGHT
        w.handles[edge] = None
        _enforce_all(w, [edge, edge + 1])
    return w.freeze()


def set_continuity(shape: SectionShape, i: int, mode: str) -> SectionShape:
    if mode not in CONTINUITY_MODES:
        raise ValueError(f"continuity must be one of {', '.join(CONTINUITY_MODES)}")
    w = _Work(shape)
    w.cont[i] = mode
    _enforce(w, i, "out" if w.curved(i) else "in")
    return w.freeze()


def insert_vertex(shape: SectionShape, edge: int, point) -> tuple[SectionShape, int]:
    """Insert a vertex on ``edge`` at the point of the edge nearest ``point``. Returns (shape, new vertex index).

    A curved edge is split exactly (de Casteljau), so the outline does not change, and the new vertex is smooth.
    A straight edge gets a corner vertex. On a skin-following edge the new vertex is snapped (``point`` is taken
    to be on the skin).
    """
    w = _Work(shape)
    n = w.n
    j = (edge + 1) % n
    a, b = w.v[edge], w.v[j]
    new_flag, new_cont = False, CORNER
    left = right = None
    if w.skin(edge):
        pos = np.array(point, dtype=float)
        new_flag = True
        kinds = (STRAIGHT, STRAIGHT)  # both halves follow the skin; their own definition is only a fallback
    elif w.curved(edge):
        ctrl = shape.control_points(edge)
        t = nearest_t_on_bezier(ctrl, point)
        c1, c2 = split_bezier(ctrl, t)
        pos = c1[3]
        left = [c1[1] - c1[0], c1[2] - c1[3]]
        right = [c2[1] - c2[0], c2[2] - c2[3]]
        kinds = (CURVED, CURVED)
        new_cont = SMOOTH
    else:
        d = b - a
        t = float(np.clip(np.dot(np.array(point, dtype=float) - a, d) / max(float(np.dot(d, d)), 1e-300), 0.0, 1.0))
        pos = a + t * d
        kinds = (STRAIGHT, STRAIGHT)
    at = edge + 1  # index of the new vertex
    w.v.insert(at, pos)
    w.snapped.insert(at, new_flag)
    w.cont.insert(at, new_cont)
    # edge ``edge`` becomes (edge -> new), and a second edge (new -> old end) follows it
    w.edges[edge] = kinds[0]
    w.handles[edge] = None if kinds[0] == STRAIGHT else [np.array(left[0]), np.array(left[1])]
    w.edges.insert(at, kinds[1])
    w.handles.insert(at, None if kinds[1] == STRAIGHT else [np.array(right[0]), np.array(right[1])])
    return w.freeze(), at


def delete_vertex(shape: SectionShape, i: int) -> SectionShape:
    """Remove vertex ``i``; its two edges merge into one. The outline stays closed."""
    n = shape.n
    if n <= MIN_VERTICES:
        raise ValueError(f"a section needs at least {MIN_VERTICES} vertices")
    w = _Work(shape)
    old = [k for k in range(n) if k != i]
    m = len(old)
    verts = [w.v[k] for k in old]
    snapped = [w.snapped[k] for k in old]
    cont = [w.cont[k] for k in old]
    edges, handles = [], []
    for k in range(m):
        a, b = old[k], old[(k + 1) % m]
        if b == (a + 1) % n:  # an untouched edge
            edges.append(w.edges[a])
            handles.append(w.handles[a])
        else:  # the merged edge: a -> i -> b
            e_in, e_out = a, i
            if w.edges[e_in] == STRAIGHT and w.edges[e_out] == STRAIGHT:
                edges.append(STRAIGHT)
                handles.append(None)
            else:
                chord = (w.v[b] - w.v[a]) * HANDLE_FRACTION
                h0 = w.handles[e_in][0].copy() if w.edges[e_in] == CURVED else chord
                h1 = w.handles[e_out][1].copy() if w.edges[e_out] == CURVED else -chord
                edges.append(CURVED)
                handles.append([h0, h1])
    w.v, w.snapped, w.cont, w.edges, w.handles = verts, snapped, cont, edges, handles
    # the vertices around the removed one keep their mode but must re-satisfy it
    _enforce_all(w, [old.index((i - 1) % n), old.index((i + 1) % n)])
    return w.freeze()


def scale_free(shape: SectionShape, centre, factor: float) -> SectionShape:
    """Scale the free (not snapped) vertices and the handles attached to them about ``centre``."""
    c = np.asarray(centre, dtype=float)
    w = _Work(shape)
    n = w.n
    for k in range(n):
        if not w.snapped[k]:
            w.v[k] = c + factor * (w.v[k] - c)
    for e in range(n):
        h = w.handles[e]
        if h is None:
            continue
        if not w.snapped[e]:
            h[0] = h[0] * factor
        if not w.snapped[(e + 1) % n]:
            h[1] = h[1] * factor
    return w.freeze()


def scale_all(shape: SectionShape, centre, factor: float) -> SectionShape:
    """Scale every vertex and handle (snapped or not) about ``centre``."""
    return scale_free(shape.with_(snapped=(False,) * shape.n), centre, factor).with_(snapped=shape.snapped)


def translate(shape: SectionShape, delta) -> SectionShape:
    d = np.asarray(delta, dtype=float)
    return shape.with_(vertices=tuple(_v(np.array(p) + d) for p in shape.vertices))


# ---- sampling ------------------------------------------------------------------------------------------------------
def edge_paths(shape: SectionShape, section: Section | None = None, samples: int = BEZIER_SAMPLES):
    """(paths, snapped-edge flags, lengths): the polyline of every edge (curves sampled finely) and its length.

    An edge between two snapped vertices follows the shorter arc of ``section``; without a section it falls back
    to its own (straight or curved) definition.
    """
    paths, flags, lengths = [], [], []
    n = shape.n
    for i in range(n):
        j = (i + 1) % n
        a, b = np.array(shape.vertices[i]), np.array(shape.vertices[j])
        if section is not None and shape.edge_is_snapped(i):
            _, s0, _ = nearest_on_section(section, a)
            _, s1, _ = nearest_on_section(section, b)
            path, length = arc_path(section, s0, s1)
            path = path.copy()
            path[0], path[-1] = a, b  # exactly the vertices (they are on the curve to rounding)
            flags.append(True)
        elif shape.edges[i] == CURVED:
            path = bezier_points(shape.control_points(i), samples)
            path[0], path[-1] = a, b
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            flags.append(False)
        else:
            path = np.array([a, b])
            length = float(np.linalg.norm(b - a))
            flags.append(False)
        paths.append(path)
        lengths.append(float(length))
    return paths, flags, lengths


def build_outline(paths) -> np.ndarray:
    """The closed outline: every edge path without its last point (it is the next edge's first)."""
    return np.vstack([p[:-1] for p in paths])


def outline_area(shape: SectionShape, section: Section | None = None) -> float:
    paths, _, _ = edge_paths(shape, section)
    return abs(polygon_area(build_outline(paths)))


def outline_centroid(shape: SectionShape, section: Section | None = None) -> np.ndarray:
    paths, _, _ = edge_paths(shape, section)
    return polygon_centroid(build_outline(paths))


# ---- standard shapes --------------------------------------------------------------------------------------------------
def circle_shape(radius: float, centre=(0.0, 0.0)) -> SectionShape:
    """A circle as 4 symmetric Bézier arcs, counter-clockwise from the top vertex (0, r)."""
    r, k = float(radius), float(radius) * CIRCLE_KAPPA
    cu, cv = float(centre[0]), float(centre[1])
    verts = ((cu, cv + r), (cu - r, cv), (cu, cv - r), (cu + r, cv))
    handles = (((-k, 0.0), (0.0, k)), ((0.0, -k), (-k, 0.0)), ((k, 0.0), (0.0, -k)), ((0.0, k), (k, 0.0)))
    return SectionShape(verts, (False,) * 4, (CURVED,) * 4, handles, (SMOOTH,) * 4)


def superellipse_shape(a: float, b: float, exponent: float = 2.5, centre=(0.0, 0.0)) -> SectionShape:
    """|u/a|^p + |v/b|^p = 1 as 4 smooth Bézier arcs (vertices at the top, left, bottom and right, counter-clockwise from
    the top). The handle length is chosen so each arc passes through the superellipse's diagonal point; exponent 2
    is the ellipse, larger values are squarer."""
    if a <= 0.0 or b <= 0.0 or exponent <= 0.0:
        raise ValueError("a superellipse needs positive semi-axes and exponent")
    k = (8.0 * 2.0 ** (-1.0 / float(exponent)) - 4.0) / 3.0
    ka, kb = k * float(a), k * float(b)
    cu, cv = float(centre[0]), float(centre[1])
    verts = ((cu, cv + b), (cu - a, cv), (cu, cv - b), (cu + a, cv))
    handles = (((-ka, 0.0), (0.0, kb)), ((0.0, -kb), (-ka, 0.0)), ((ka, 0.0), (0.0, -kb)), ((0.0, kb), (ka, 0.0)))
    return SectionShape(verts, (False,) * 4, (CURVED,) * 4, handles, (SMOOTH,) * 4)


# ---- config round trip --------------------------------------------------------------------------------------------------
def shape_to_config(shape: SectionShape) -> dict:
    """``vertices``, ``snapped``, ``edges``, ``handles`` and ``continuity`` as plain lists for the YAML config."""
    return {
        "vertices": [list(p) for p in shape.vertices],
        "snapped": list(shape.snapped),
        "edges": list(shape.edges),
        "handles": [None if h is None else [list(h[0]), list(h[1])] for h in shape.handles],
        "continuity": list(shape.continuity),
    }


def shape_from_config(data: dict, where: str = "section") -> SectionShape:
    """Parse a config mapping. ``snapped``, ``edges``, ``handles`` and ``continuity`` are optional (all straight,
    unsnapped corners by default). Raises ValueError with a readable message."""
    raw = data.get("vertices")
    if not (isinstance(raw, (list, tuple)) and len(raw) >= MIN_VERTICES
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw)):
        raise ValueError(f"{where}: vertices must be at least {MIN_VERTICES} [u, v] pairs")
    n = len(raw)

    def per(name, default, check):
        value = data.get(name)
        if value is None:
            return [default] * n
        if not (isinstance(value, (list, tuple)) and len(value) == n and all(check(x) for x in value)):
            raise ValueError(f"{where}: {name} must have {n} valid entries")
        return list(value)

    snapped = per("snapped", False, lambda x: isinstance(x, bool))
    edges = per("edges", STRAIGHT, lambda x: x in EDGE_TYPES)
    cont = per("continuity", CORNER, lambda x: x in CONTINUITY_MODES)
    raw_h = data.get("handles")
    if raw_h is not None and not (isinstance(raw_h, (list, tuple)) and len(raw_h) == n):
        raise ValueError(f"{where}: handles must have {n} entries")
    handles = []
    for i in range(n):
        h = raw_h[i] if raw_h is not None else None
        if edges[i] == CURVED:
            if h is None:  # a curved edge without stored handles starts as the chord
                a, b = np.array(raw[i], dtype=float), np.array(raw[(i + 1) % n], dtype=float)
                h = [list((b - a) * HANDLE_FRACTION), list(-(b - a) * HANDLE_FRACTION)]
            elif not (isinstance(h, (list, tuple)) and len(h) == 2 and all(len(p) == 2 for p in h)):
                raise ValueError(f"{where}: handles[{i}] must be [[du, dv], [du, dv]]")
            handles.append((_v(h[0]), _v(h[1])))
        else:
            handles.append(None)
    return SectionShape(tuple(_v(p) for p in raw), tuple(snapped), tuple(edges), tuple(handles), tuple(cont))


# ---- mirror symmetry about u = 0 (a self-symmetric inlet keeps its sections symmetric) -----------------------------------
SYMMETRY_TOL = 1e-6  # of the section size: vertices this close to each other's mirror image are partners


def vertex_partners(shape: SectionShape, tol: float = SYMMETRY_TOL) -> list[int | None]:
    """For every vertex the index of its mirror image about u = 0 (itself for a vertex on the axis), or None."""
    pts = np.array(shape.vertices)
    scale = max(float(np.ptp(pts, axis=0).max()), 1e-12)
    out: list[int | None] = []
    for i, (u, v) in enumerate(pts):
        d = np.linalg.norm(pts - np.array([-u, v]), axis=1)
        j = int(np.argmin(d))
        out.append(j if d[j] <= tol * scale else None)
    return out


def is_mirror_symmetric(shape: SectionShape, tol: float = SYMMETRY_TOL) -> bool:
    return all(p is not None for p in vertex_partners(shape, tol))


def edge_partner(shape: SectionShape, partners: list[int | None], e: int) -> int | None:
    """The edge that is the mirror image of edge ``e`` (it runs from the partner of e's end to the partner of its
    start, so a mirror-symmetric edge is its own partner)."""
    n = shape.n
    a, b = partners[e], partners[(e + 1) % n]
    if a is None or b is None or (b + 1) % n != a:
        return None
    return b


def move_vertex_symmetric(shape: SectionShape, i: int, uv, flag: bool | None = None) -> SectionShape:
    """Move vertex ``i`` of a symmetric shape and its mirror partner with it (a vertex on the axis stays on it)."""
    partners = vertex_partners(shape)
    j = partners[i]
    u, v = float(uv[0]), float(uv[1])
    if j is None:
        out = move_vertex(shape, i, (u, v))
    elif j == i:
        out = move_vertex(shape, i, (0.0, v))
    else:
        out = move_vertex(move_vertex(shape, i, (u, v)), j, (-u, v))
    if flag is not None:
        snapped = list(out.snapped)
        for k in {i, j} - {None}:
            snapped[k] = bool(flag)
        out = out.with_(snapped=tuple(snapped))
    return out


def move_handle_symmetric(shape: SectionShape, edge: int, which: int, uv) -> SectionShape:
    """Drag a Bézier handle of a symmetric shape and the mirrored handle of its partner edge."""
    ep = edge_partner(shape, vertex_partners(shape), edge)
    out = move_handle(shape, edge, which, uv)
    if ep is not None:
        out = move_handle(out, ep, 1 - which, (-float(uv[0]), float(uv[1])))
    return out


def insert_vertex_symmetric(shape: SectionShape, edge: int, point) -> tuple[SectionShape, int]:
    """Insert a vertex and its mirror image on the partner edge (on the axis when the edge is its own partner)."""
    ep = edge_partner(shape, vertex_partners(shape), edge)
    if ep is None:
        return insert_vertex(shape, edge, point)
    if ep == edge:
        return insert_vertex(shape, edge, (0.0, float(point[1])))
    out, at = insert_vertex(shape, edge, point)
    pu, pv = out.vertices[at]
    out, _ = insert_vertex(out, ep + 1 if ep > edge else ep, (-pu, pv))
    return out, at


def delete_vertex_symmetric(shape: SectionShape, i: int) -> SectionShape:
    j = vertex_partners(shape)[i]
    if j is None or j == i:
        return delete_vertex(shape, i)
    out = delete_vertex(shape, max(i, j))
    return delete_vertex(out, min(i, j))


def set_edge_type_symmetric(shape: SectionShape, edge: int, kind: str) -> SectionShape:
    ep = edge_partner(shape, vertex_partners(shape), edge)
    out = set_edge_type(shape, edge, kind)
    return out if ep is None or ep == edge else set_edge_type(out, ep, kind)


def set_continuity_symmetric(shape: SectionShape, i: int, mode: str) -> SectionShape:
    j = vertex_partners(shape)[i]
    out = set_continuity(shape, i, mode)
    return out if j is None or j == i else set_continuity(out, j, mode)
