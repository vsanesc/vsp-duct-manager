# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Resampling key sections to a common point count with consistent correspondence, so that blending between
sections does not twist.

Every outline is oriented counter-clockwise and parametrised by arc length starting at a *reference vertex*:
the vertex whose direction from the section's centroid is closest to the section's local +v. The N samples are
equally spaced in arc length from there, so sample k of one section corresponds to sample k of another.
"""
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
