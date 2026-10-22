# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Export mapping: a key section -> an OpenVSP Edit Curve XSec of type CEDIT (cubic Béziers).

Every edge becomes one or more cubic Bézier segments:
 * a straight edge is a degenerate Bézier with its handles at 1/3 and 2/3 of the chord;
 * a curved edge is its own Bézier;
 * an edge that follows the skin is a chain of degenerate Béziers along the section polyline, so chine corners stay
   corners.
Corners stay corners: the G1 flag of an anchor is set only for a smooth / symmetric vertex whose two handles are
really collinear. The result stays editable point by point in the VSP GUI.

A CEDIT curve is ``3 m + 1`` control points (anchor, handle, handle, anchor, ...) for ``m`` segments, closed (the
last point equals the first), in coordinates normalised by the XSec's Width and Height: a control point
``(x, y)`` is at ``(y, z) = (x * Width, y * Height)`` in the section plane. The points here are in the section's
own ``(a, b)`` coordinates (``a`` to VSP's y, ``b`` to z); VSP's default curve is clockwise in that plane, so by
default the order is reversed from the counter-clockwise (u, v) outline. Mapping (u, v) onto the aircraft is the
caller's business (Phase 4).
"""
from dataclasses import dataclass

import numpy as np

from ductmgr.geometry import polygon_area
from ductmgr.interp import reference_vertex
from ductmgr.outline import CURVED, SMOOTH, SYMMETRIC, SectionShape, bezier_points
from ductmgr.section import Section, arc_path, nearest_on_section

G1_COLLINEAR_TOL = 1e-6  # |sin| of the angle between the two handles at an anchor for a G1 flag
MIN_EXTENT = 1e-9  # m: a section narrower than this cannot be normalised
SAMPLES_PER_SEGMENT = 64


@dataclass(frozen=True)
class CeditSection:
    width: float  # the XSec's Width / Height: the extents of the outline
    height: float
    centre: tuple[float, float]  # centre of the outline's bounding box (section coordinates)
    u: tuple[float, ...]  # one parameter value per control point
    points: tuple[tuple[float, float], ...]  # normalised control points, 3 m + 1
    g1: tuple[bool, ...]  # the G1 flag of every control point (only anchors can be True; the closing point repeats the first anchor and is False)

    @property
    def n_segments(self) -> int:
        return (len(self.points) - 1) // 3


# ---- the mapping --------------------------------------------------------------------------------------------
def _degenerate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0, b])


def segments_of(shape: SectionShape, section: Section | None = None):
    """(segments, anchor G1 flags): the closed outline as cubic Bézier segments (each (4, 2)), in order from
    vertex 0, with the G1 flag of the anchor that starts each segment."""
    segs, flags = [], []
    n = shape.n
    for i in range(n):
        a, b = np.array(shape.vertices[i], dtype=float), np.array(shape.vertices[(i + 1) % n], dtype=float)
        if section is not None and shape.edge_is_snapped(i):
            _, s0, _ = nearest_on_section(section, a)
            _, s1, _ = nearest_on_section(section, b)
            path, _ = arc_path(section, s0, s1)
            path = path.copy()
            path[0], path[-1] = a, b
            pieces = [_degenerate(path[k], path[k + 1]) for k in range(len(path) - 1)]
        elif shape.edges[i] == CURVED:
            pieces = [shape.control_points(i)]
        else:
            pieces = [_degenerate(a, b)]
        for k, p in enumerate(pieces):
            segs.append(p)
            flags.append(k == 0 and shape.continuity[i] in (SMOOTH, SYMMETRIC))  # a vertex anchor; skin corners no
    # a G1 flag only where the handles really are collinear
    m = len(segs)
    out = []
    for k in range(m):
        prev_in = segs[(k - 1) % m][2] - segs[(k - 1) % m][3]  # handle at the anchor, from the previous segment
        nxt_out = segs[k][1] - segs[k][0]
        n1, n2 = np.linalg.norm(prev_in), np.linalg.norm(nxt_out)
        collinear = (n1 > 0 and n2 > 0 and abs(prev_in[0] * nxt_out[1] - prev_in[1] * nxt_out[0]) / (n1 * n2)
                     < G1_COLLINEAR_TOL and float(np.dot(prev_in, nxt_out)) < 0.0)
        out.append(bool(flags[k] and collinear))
    return segs, out


def section_to_cedit(shape: SectionShape, section: Section | None = None, clockwise: bool = True,
                     start_vertex: int | None = None) -> CeditSection:
    """The CEDIT control data of a section. The curve starts at ``start_vertex`` (default: the reference vertex,
    the one closest to +v) and runs clockwise (VSP's own direction) unless ``clockwise`` is False."""
    segs, g1 = segments_of(shape, section)
    if start_vertex is None:
        start_vertex = reference_vertex(np.array(shape.vertices))
    return segments_to_cedit(segs, g1, np.array(shape.vertices[start_vertex], dtype=float), clockwise)


def segments_to_cedit(segs: list, g1: list, start_anchor, clockwise: bool = True) -> CeditSection:
    """The CEDIT control data of a closed outline given as cubic Bézier segments (each (4, 2), in order, the end of
    one being the start of the next) with the G1 flag of the anchor that starts each. The curve starts at the anchor
    nearest ``start_anchor`` and runs clockwise (VSP's own direction) unless ``clockwise`` is False."""
    segs, g1 = list(segs), list(g1)
    m = len(segs)
    extent = np.ptp(np.vstack([bezier_points(s, 8) for s in segs]), axis=0)
    if extent[0] < MIN_EXTENT or extent[1] < MIN_EXTENT:
        raise ValueError("the section is too small to normalise")
    ccw = polygon_area(np.vstack([bezier_points(s, 8)[:-1] for s in segs])) > 0.0
    if ccw == clockwise:  # reverse the traversal: each segment is flipped and the anchors' flags follow
        segs = [s[::-1].copy() for s in reversed(segs)]
        g1 = [g1[(m - j) % m] for j in range(m)]
    # rotate so the curve begins at the start vertex
    k0 = min(range(m), key=lambda k: float(np.linalg.norm(segs[k][0] - np.asarray(start_anchor, dtype=float))))
    segs, g1 = segs[k0:] + segs[:k0], g1[k0:] + g1[:k0]

    pts = np.vstack([s[:3] for s in segs] + [segs[-1][3:4]])
    outline = np.vstack([bezier_points(s, 16)[:-1] for s in segs])
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    width, height = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if width < MIN_EXTENT or height < MIN_EXTENT:
        raise ValueError("the section is too small to normalise")
    centre = (lo + hi) / 2.0
    norm = (pts - centre) / np.array([width, height])
    u = [(k + j / 3.0) / m for k in range(m) for j in range(3)] + [1.0]
    flags = [False] * len(pts)
    for k in range(m):
        flags[3 * k] = g1[k]
    # the closing point repeats the first anchor and carries no flag of its own (VSP reads it back as False)
    return CeditSection(width, height, (float(centre[0]), float(centre[1])), tuple(u),
                        tuple((float(x), float(y)) for x, y in norm), tuple(flags))


def sample_cedit(c: CeditSection, per_segment: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """The curve described by ``c`` in section coordinates (centre added back), (K, 2), closed (last = first)."""
    pts = np.array(c.points) * np.array([c.width, c.height]) + np.array(c.centre)
    out = [bezier_points(pts[3 * k: 3 * k + 4], per_segment)[:-1] for k in range(c.n_segments)]
    return np.vstack(out + [pts[-1:]])


# ---- applying it through the OpenVSP API ----------------------------------------------------------------------------
def make_edit_curve(xsec_surf_id: str, index: int) -> str:
    """Turn XSec ``index`` of a surface into an Edit Curve XSec and return its id."""
    import openvsp as vsp

    vsp.ChangeXSecShape(xsec_surf_id, index, vsp.XS_EDIT_CURVE)
    vsp.Update()
    return vsp.GetXSec(xsec_surf_id, index)


def apply_cedit(xsec_id: str, c: CeditSection, g1: bool = True) -> None:
    """Write ``c`` into an Edit Curve XSec: Width / Height, the control points and (with ``g1``) the G1 flags.

    The XSec's built-in symmetry is switched off first (it would mirror the edit), and the curve type is CEDIT.
    ``g1=False`` leaves the per-point G1 parameters at OpenVSP's defaults (they are cheap to set; the geometry does not
    depend on them).
    """
    import openvsp as vsp

    def parm(name):
        return vsp.GetXSecParm(xsec_id, name)

    for name in ("SymType", "TBSym", "RLSym", "AllSym"):
        vsp.SetParmVal(parm(name), 0.0)
    vsp.SetParmVal(parm("CurveType"), float(vsp.CEDIT))
    vsp.Update()
    vsp.SetXSecWidthHeight(xsec_id, c.width, c.height)
    vsp.Update()
    ctrl = [vsp.vec3d(x, y, 0.0) for x, y in c.points]
    vsp.SetEditXSecPnts(xsec_id, list(c.u), ctrl, [0.0] * len(ctrl))
    vsp.Update()
    if g1:
        for i, flag in enumerate(c.g1):  # the per-point parameters exist only now that the points do
            vsp.SetParmVal(parm(f"G1_{i}"), 1.0 if flag else 0.0)
        vsp.Update()


def read_cedit_samples(xsec_id: str, n_segments: int, per_segment: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """Points of the XSec's actual curve (``ComputeXSecPnt``) as (y, z) section coordinates.

    Each of the ``n_segments`` Bézier segments is sampled from its anchor (the anchors sit at ``k / n_segments``),
    so corners are hit exactly. The curve is closed; the last point is not repeated.
    """
    import openvsp as vsp

    fractions = [(k + j / per_segment) / n_segments for k in range(n_segments) for j in range(per_segment)]
    pts = [vsp.ComputeXSecPnt(xsec_id, f) for f in fractions]
    return np.array([[p.y(), p.z()] for p in pts])
