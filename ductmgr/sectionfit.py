# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Edit Curve (CEDIT) sections for a set of corresponding rings.

The duct's sections are ``N``-point rings that correspond index by index (the same sample on every ring is the same
place on the wall). The export turns each ring into a short chain of cubic Béziers that the user can move point by
point in the OpenVSP GUI, instead of ``N`` linear segments:

 * the anchors are chosen once for the whole set, at the same indices on every ring, so every XSec has the same number
   of segments with the same correspondence and OpenVSP lofts matching anchors. Anchors are put where any ring turns
   sharply (corners, both ends of a cut corner) and the rest are spread where the rings bend most; while some ring is
   still off by more than the tolerance an anchor is added in the segment that fits worst;
 * between two anchors a cubic is fitted to the ring points (``simplify.fit_edge``: least squares on the handle
   lengths, tangents from the ring), a nearly flat run stays a straight (degenerate) segment;
 * the fitted section is moved onto the ring's centroid and scaled to its area, so A(s) is exact;
 * the deviation from the ring (both ways) is measured; when it exceeds the tolerance the number of anchors grows, and
   when even that fails the sections are written as linear CEDITs through the ring points.

Rings are in the section coordinates of the OpenVSP XSec (clockwise, the direction of VSP's own curves).
"""
import math
from dataclasses import dataclass

import numpy as np

from ductmgr import cedit
from ductmgr.geometry import polygon_area, polygon_centroid
from ductmgr.outline import bezier_points
from ductmgr.simplify import distances, fit_edge

DEFAULT_ANCHORS = 8  # at least this many segments per section (corners may need more)
DEFAULT_TOLERANCE = 0.005  # of the section size: the largest allowed deviation from the ring
DEFAULT_CORNER_DEG = 20.0  # a ring that turns by more than this at a sample has a corner there
CURVE_SAMPLES = 24  # per fitted segment, for the area and the deviation
TIGHTEN = 0.7  # factor on the per-segment tolerance when the whole curve is off although every segment fits
FILL_FLOOR = 0.3  # of the mean turning per sample: anchors are also spread along the arc, not only to the bends


@dataclass(frozen=True)
class FitResult:
    sections: tuple  # one cedit.CeditSection per ring
    anchors: tuple[int, ...]  # the indices of the ring points that are anchors
    linear: bool  # True: the tolerance could not be met and the sections are linear CEDITs through the ring points
    max_dev: float  # worst deviation over all rings, as a fraction of the section size
    devs: tuple[float, ...]  # per ring

    @property
    def n_segments(self) -> int:
        return len(self.anchors)


def turning_angles(ring: np.ndarray) -> np.ndarray:
    """Unsigned turning angle (rad) of the closed polyline at each of its points."""
    a = ring - np.roll(ring, 1, axis=0)
    b = np.roll(ring, -1, axis=0) - ring
    return np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1)))


def _allocate(extra: int, mass: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """``extra`` anchors over the gaps in proportion to their mass (largest remainders), never above a gap's room."""
    out = np.zeros(len(mass), dtype=int)
    left = extra
    while left > 0 and (out < capacity).any():
        room = out < capacity
        share = np.where(room, mass, 0.0)
        ideal = left * share / share.sum() if share.sum() > 0 else np.where(room, 1.0, 0.0) * left / room.sum()
        take = np.minimum(np.floor(ideal).astype(int), capacity - out)
        if take.sum() == 0:  # only fractions left: give one to the biggest remainder that has room
            k = int(np.argmax(np.where(room, ideal, -1.0)))
            take[k] = 1
        out += take
        left = extra - int(out.sum())
    return out


def choose_anchors(rings, count: int = DEFAULT_ANCHORS, corner_deg: float = DEFAULT_CORNER_DEG) -> list[int]:
    """Indices of the anchors shared by every ring: index 0, every point where some ring turns by more than
    ``corner_deg`` and, up to ``count`` in all, further points where the rings bend most."""
    n = len(rings[0])
    turn = np.max([turning_angles(np.asarray(r, dtype=float)) for r in rings], axis=0)
    forced = sorted({0} | {int(j) for j in np.flatnonzero(turn > math.radians(corner_deg))})
    extra = min(count, n) - len(forced)
    if extra <= 0:
        return forced
    w = turn + FILL_FLOOR * float(turn.mean())
    gaps = []  # (first anchor, last anchor, indices between them)
    for k, a in enumerate(forced):
        b = forced[k + 1] if k + 1 < len(forced) else forced[0] + n
        gaps.append((a, b))
    cum = []
    for a, b in gaps:
        idx = np.arange(a, b + 1)
        wi = w[idx % n]
        cum.append(np.r_[0.0, np.cumsum(0.5 * (wi[:-1] + wi[1:]))])
    mass = np.array([c[-1] for c in cum])
    capacity = np.array([b - a - 1 for a, b in gaps])
    alloc = _allocate(extra, mass, capacity)
    anchors = set(forced)
    for (a, b), c, cm in zip(gaps, alloc, cum):
        taken = a
        for i in range(int(c)):
            target = (i + 1) / (int(c) + 1) * cm[-1]
            j = a + int(np.argmin(np.abs(cm - target)))
            j = min(max(j, taken + 1), b - (int(c) - i))  # strictly increasing, leaving room for the rest
            anchors.add(j % n)
            taken = j
    return sorted(anchors)


def _tangents(ring: np.ndarray):
    """(arriving, leaving, central) unit tangents at every point, in the direction of increasing index."""
    def unit(v):
        length = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(length > 0, length, 1.0)

    tin = unit(ring - np.roll(ring, 1, axis=0))
    tout = unit(np.roll(ring, -1, axis=0) - ring)
    return tin, tout, unit(np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0))


def _degenerate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0, b])


class RingFitter:
    """Fits the Bézier segment between two anchors of one ring (cached: inserting an anchor only changes two segments)."""

    def __init__(self, ring: np.ndarray, corner_deg: float = DEFAULT_CORNER_DEG):
        self.ring = np.asarray(ring, dtype=float)
        self.n = len(self.ring)
        self.tin, self.tout, self.tcen = _tangents(self.ring)
        self.corner = np.degrees(np.arccos(np.clip((self.tin * self.tout).sum(axis=1), -1.0, 1.0))) > corner_deg
        self.scale = max(float(np.ptp(self.ring, axis=0).max()), 1e-12)
        self._cache: dict = {}

    def segment(self, a: int, b: int):
        """(the cubic from anchor a to anchor b, its error over the ring size, whether it is a curve)."""
        key = (a, b)
        if key not in self._cache:
            n = self.n
            span = np.arange(a, b + 1) if b > a else np.r_[np.arange(a, n), np.arange(0, b + 1)]
            d = self.ring[span % n]
            ta = self.tout[a] if self.corner[a] else self.tcen[a]
            tb = self.tin[b] if self.corner[b] else self.tcen[b]
            handles, err = fit_edge(d, ta, tb, self.scale)
            seg = _degenerate(d[0], d[-1]) if handles is None else np.array(
                [d[0], d[0] + handles[0], d[-1] + handles[1], d[-1]])
            self._cache[key] = (seg, err / self.scale, handles is not None)
        return self._cache[key]

    def assemble(self, anchors):
        """The segments and G1 flags with the anchors at the given indices, moved onto the ring's centroid and scaled to
        its area."""
        parts = [self.segment(a, anchors[(k + 1) % len(anchors)]) for k, a in enumerate(anchors)]
        segs = [p[0] for p in parts]
        flags = [bool((not self.corner[a]) and parts[k][2] and parts[k - 1][2]) for k, a in enumerate(anchors)]
        curve = np.vstack([bezier_points(sg, CURVE_SAMPLES)[:-1] for sg in segs])
        c_fit, c_ring = polygon_centroid(curve), polygon_centroid(self.ring)
        factor = math.sqrt(abs(polygon_area(self.ring)) / abs(polygon_area(curve)))
        return [(sg - c_fit) * factor + c_ring for sg in segs], flags

    def errors(self, anchors) -> list[float]:
        return [self.segment(a, anchors[(k + 1) % len(anchors)])[1] for k, a in enumerate(anchors)]


def fit_ring(ring: np.ndarray, anchors, corner_deg: float = DEFAULT_CORNER_DEG):
    """The Bézier segments and G1 flags (see ``cedit.segments_to_cedit``) of ``ring`` with its anchors at the given
    indices, moved onto the ring's centroid and scaled to its area."""
    return RingFitter(ring, corner_deg).assemble(list(anchors))


def curve_points(segs) -> np.ndarray:
    return np.vstack([bezier_points(s, CURVE_SAMPLES)[:-1] for s in segs])


def relative_deviation(ring: np.ndarray, curve: np.ndarray) -> float:
    """Largest distance between the closed polyline ``ring`` and the sampled ``curve``, either way, over the size of
    the ring."""
    d = max(float(distances(ring, curve, closed=True).max()), float(distances(curve, ring, closed=True).max()))
    return d / max(float(np.ptp(ring, axis=0).max()), 1e-12)


def linear_cedit(ring: np.ndarray, start_shift: int = 0) -> cedit.CeditSection:
    """The ring as a linear CEDIT: one degenerate Bézier per edge, starting at ring point ``start_shift``."""
    ring = np.asarray(ring, dtype=float)
    n = len(ring)
    segs = [_degenerate(ring[k], ring[(k + 1) % n]) for k in range(n)]
    return cedit.segments_to_cedit(segs, [False] * n, ring[start_shift % n], clockwise=True)


def fit_rings(rings, anchors: int = DEFAULT_ANCHORS, tolerance: float = DEFAULT_TOLERANCE,
              corner_deg: float = DEFAULT_CORNER_DEG, start_shift: int = 0) -> FitResult:
    """CEDIT sections for the corresponding ``rings`` with common anchors: a start of ``anchors`` (more where the rings
    have corners), then one more in the worst-fitting segment until every ring is within ``tolerance`` (a fraction of
    its size); linear CEDITs when that is not reached. ``start_shift`` makes the curve start at the anchor that many places
    after ring point 0 (the same anchor on every ring). OpenVSP 3.55 corrupts its heap for some Stacks whose curves start at ring
    point 0 (the ring's reference vertex, usually a top corner) and never for the same curves started one anchor later."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    n = len(rings[0])
    fitters = [RingFitter(r, corner_deg) for r in rings]
    idx = choose_anchors(rings, max(int(anchors), 3), corner_deg)
    target = tolerance
    while len(idx) < n:
        errs = np.array([f.errors(idx) for f in fitters])  # (rings, segments)
        if errs.max() <= target:
            fits = [f.assemble(idx) for f in fitters]
            devs = [relative_deviation(r, curve_points(segs)) for r, (segs, _) in zip(rings, fits)]
            if max(devs) <= tolerance:
                sections = tuple(cedit.segments_to_cedit(segs, g1, r[idx[start_shift % len(idx)]], clockwise=True)
                                 for r, (segs, g1) in zip(rings, fits))
                return FitResult(sections, tuple(idx), False, float(max(devs)), tuple(float(d) for d in devs))
            target *= TIGHTEN  # the segments look fine one by one but the curve as a whole is not
            continue
        order = np.argsort(-errs.max(axis=0))  # segments, worst first
        for k in order:
            a, b = idx[int(k)], idx[(int(k) + 1) % len(idx)]
            gap = (b - a) % n or n
            if gap >= 2:  # room for an anchor in between
                idx = sorted(set(idx) | {(a + gap // 2) % n})
                break
        else:
            break  # every segment is already a single edge of the ring
    return FitResult(tuple(linear_cedit(r, start_shift) for r in rings), tuple(range(n)), True, 0.0,
                     tuple(0.0 for _ in rings))
