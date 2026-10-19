# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Resampling key sections to a common point count with consistent correspondence, so that blending between
sections does not twist.

Every outline is oriented counter-clockwise and parametrised by arc length starting at a *reference vertex*:
the vertex whose direction from the section's centroid is closest to the section's local +v. The N samples are
equally spaced in arc length from there, so sample k of one section corresponds to sample k of another.
"""
import math

import numpy as np

from ductmgr.geometry import polygon_area, polygon_centroid

DEFAULT_SAMPLES = 64  # points per resampled section (ducts.n_section_points overrides)
TIE_DECIMALS = 9  # reference vertices closer in angle than this are tied; the smaller u wins


def reference_vertex(vertices, outline: np.ndarray | None = None) -> int:
    """Index of the vertex closest in direction to the local +v, seen from the centroid of the ``outline``
    (of the vertices when no outline is given). Ties (symmetric sections) go to the smaller u."""
    verts = np.asarray(vertices, dtype=float)
    centre = polygon_centroid(outline if outline is not None else verts)
    rel = verts - centre
    norm = np.linalg.norm(rel, axis=1)
    cos = np.where(norm > 0, rel[:, 1] / np.where(norm > 0, norm, 1.0), -np.inf)
    order = np.lexsort((verts[:, 0], np.round(-cos, TIE_DECIMALS)))
    return int(order[0])


def resample(outline: np.ndarray, vertices, n: int = DEFAULT_SAMPLES) -> np.ndarray:
    """``n`` points equally spaced in arc length along the closed ``outline``, counter-clockwise, starting at the
    reference vertex. ``vertices`` are the section's corner / anchor points (they lie on the outline)."""
    pts = np.asarray(outline, dtype=float)
    if n < 3:
        raise ValueError("at least 3 samples are needed")
    if polygon_area(pts) < 0.0:
        pts = pts[::-1].copy()
    ref = reference_vertex(vertices, pts)
    start = np.asarray(vertices, dtype=float)[ref]
    k = int(np.argmin(np.linalg.norm(pts - start, axis=1)))
    loop = np.vstack([np.roll(pts, -k, axis=0), np.roll(pts, -k, axis=0)[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    targets = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.c_[np.interp(targets, cum, loop[:, 0]), np.interp(targets, cum, loop[:, 1])]


def correspond(sections, n: int = DEFAULT_SAMPLES) -> list[np.ndarray]:
    """Resample every ``(outline, vertices)`` pair to ``n`` corresponding points."""
    return [resample(outline, vertices, n) for outline, vertices in sections]


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Point-by-point linear blend of two corresponding resampled sections."""
    return (1.0 - t) * np.asarray(a) + t * np.asarray(b)


def resample_keep(poly: np.ndarray, corners, n: int) -> np.ndarray:
    """``n`` points along the closed counter-clockwise polygon ``poly`` that include every corner in ``corners``
    (indices into ``poly``). The points are equally spaced in arc length within each side between corners, the
    count per side being proportional to its length. The first sample is the corner with the largest v."""
    pts = np.asarray(poly, dtype=float)
    idx = sorted(set(int(c) for c in corners))
    if len(idx) < 2 or n < len(idx):
        raise ValueError("at least two corners and as many samples as corners are needed")
    start = max(idx, key=lambda i: (pts[i, 1], -pts[i, 0]))
    order = idx[idx.index(start):] + idx[:idx.index(start)]
    m = len(pts)
    sides = []
    for a, b in zip(order, order[1:] + order[:1]):
        path = [pts[a]]
        k = (a + 1) % m
        while k != b:
            path.append(pts[k])
            k = (k + 1) % m
        path.append(pts[b])
        path = np.array(path)
        sides.append((path, float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())))
    total = sum(length for _, length in sides)
    counts = [max(1, int(round(n * length / total))) for _, length in sides]
    while sum(counts) != n:  # settle the rounding on the longest sides
        j = int(np.argmax([length / c for (_, length), c in zip(sides, counts)])) if sum(counts) < n \
            else int(np.argmin([length / c if c > 1 else math.inf for (_, length), c in zip(sides, counts)]))
        counts[j] += 1 if sum(counts) < n else -1
    out = []
    for (path, length), c in zip(sides, counts):
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        t = np.linspace(0.0, cum[-1], c, endpoint=False)
        out.append(np.c_[np.interp(t, cum, path[:, 0]), np.interp(t, cum, path[:, 1])])
    return np.vstack(out)
