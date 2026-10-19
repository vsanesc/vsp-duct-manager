# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The duct centrelines: a cubic Bézier per branch (inlet section centroid -> junction) and one for the trunk
(junction -> engine-face centre).

Constraints:
- the branch starts at the inlet section centroid, tangent along the inlet plane normal pointing inward (-n);
- the trunk ends at the face centre, tangent along the engine axis;
- the junction lies on y = 0 and the tangent there lies in the XZ plane (so the right branch and its mirror image
  join the trunk tangent-continuously, G1), the in-handle of the branch and the out-handle of the trunk being
  collinear with independent lengths.

``s`` everywhere else is the normalised arc length along the whole path (branch, then trunk): 0 at the inlet
section, ``s_junction`` at the junction and 1 at the engine face.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

DEFAULT_HANDLE_FRACTION = 1.0 / 3.0  # handle length / chord length when a handle length is not configured
TABLE_SAMPLES = 400  # arc-length table resolution per Bézier
CURVATURE_SAMPLES = 200


@dataclass(frozen=True)
class CentrelineParams:
    """The editable part of the centrelines. ``None`` means "default" (derived from the other inputs)."""

    junction: tuple[float, float] | None = None  # (x, z) of the junction on y = 0
    junction_pitch_deg: float | None = None  # tangent direction at the junction in the XZ plane (0 = +X, + up)
    start_handle: float | None = None  # length of the handle at the inlet (along -n)
    junction_in_handle: float | None = None  # branch side of the junction
    junction_out_handle: float | None = None  # trunk side of the junction
    end_handle: float | None = None  # at the face, along the engine axis
    handle_fraction: float = DEFAULT_HANDLE_FRACTION

    def with_(self, **changes) -> "CentrelineParams":
        return replace(self, **changes)


class Bezier3:
    """A cubic Bézier in 3D with an arc-length table."""

    def __init__(self, ctrl):
        self.ctrl = np.asarray(ctrl, dtype=float).reshape(4, 3)
        t = np.linspace(0.0, 1.0, TABLE_SAMPLES + 1)
        self._t = t
        self._cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points(t), axis=0), axis=1))]

    @property
    def length(self) -> float:
        return float(self._cum[-1])

    def points(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
        p0, p1, p2, p3 = self.ctrl
        u = 1.0 - t
        return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3

    def derivative(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
        p0, p1, p2, p3 = self.ctrl
        u = 1.0 - t
        return 3 * u**2 * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * t**2 * (p3 - p2)

    def second_derivative(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
        p0, p1, p2, p3 = self.ctrl
        return 6 * (1.0 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)

    def tangents(self, t) -> np.ndarray:
        d = self.derivative(t)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.where(n > 0, n, 1.0)

    def curvature(self, t) -> np.ndarray:
        d, dd = self.derivative(t), self.second_derivative(t)
        speed = np.linalg.norm(d, axis=1)
        return np.linalg.norm(np.cross(d, dd), axis=1) / np.maximum(speed**3, 1e-300)

    def t_at_length(self, length) -> np.ndarray:
        """Bézier parameter at the given arc length(s) from the start."""
        return np.interp(np.asarray(length, dtype=float), self._cum, self._t)

    def max_curvature(self) -> float:
        return float(self.curvature(np.linspace(0.0, 1.0, CURVATURE_SAMPLES)).max())


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("a zero-length direction")
    return v / n


@dataclass(frozen=True)
class Resolved:
    """The centreline parameters with every default filled in (what is shown and saved as handles)."""

    junction: tuple[float, float]
    junction_pitch_deg: float
    start_handle: float
    junction_in_handle: float
    junction_out_handle: float
    end_handle: float


class Centreline:
    """Branch + trunk Béziers for one inlet (the other inlet is the mirror image about y = 0)."""

    def __init__(self, start, start_dir, face_centre, face_axis, params: CentrelineParams,
                 default_junction_fraction: float = 0.5, branch_end_offset=None):
        """``branch_end_offset``: the branch ends at the junction plus this vector (the centroid of its half of the
        trunk start section) instead of at the junction itself; its tangent there is still the junction tangent."""
        self.start = np.asarray(start, dtype=float)
        self.start_dir = _unit(start_dir)
        self.face_centre = np.asarray(face_centre, dtype=float)
        self.face_axis = _unit(face_axis)
        self.params = params
        p = params
        jx, jz = p.junction if p.junction is not None else self._default_junction(default_junction_fraction)
        junction = np.array([jx, 0.0, jz])
        if p.junction_pitch_deg is not None:
            phi = math.radians(p.junction_pitch_deg)
        else:
            d = self.face_centre - junction
            phi = math.atan2(d[2], d[0]) if (d[0] or d[2]) else 0.0
        t = np.array([math.cos(phi), 0.0, math.sin(phi)])
        self.branch_end_offset = np.zeros(3) if branch_end_offset is None else np.asarray(branch_end_offset, dtype=float)
        branch_end = junction + self.branch_end_offset
        chord_b = float(np.linalg.norm(branch_end - self.start))
        chord_t = float(np.linalg.norm(self.face_centre - junction))
        f = p.handle_fraction
        h0 = f * chord_b if p.start_handle is None else p.start_handle
        h_in = f * chord_b if p.junction_in_handle is None else p.junction_in_handle
        h_out = f * chord_t if p.junction_out_handle is None else p.junction_out_handle
        h_end = f * chord_t if p.end_handle is None else p.end_handle
        for name, h in (("start", h0), ("junction in", h_in), ("junction out", h_out), ("end", h_end)):
            if h < 0.0:
                raise ValueError(f"the {name} handle length must not be negative")
        self.resolved = Resolved((float(jx), float(jz)), math.degrees(phi), float(h0), float(h_in), float(h_out),
                                 float(h_end))
        self.junction = junction
        self.junction_tangent = t
        self.branch = Bezier3([self.start, self.start + h0 * self.start_dir, branch_end - h_in * t, branch_end])
        self.trunk = Bezier3([junction, junction + h_out * t, self.face_centre - h_end * self.face_axis,
                              self.face_centre])
        lb, lt = self.branch.length, self.trunk.length
        self.length_branch, self.length_trunk = lb, lt
        self.s_junction = lb / (lb + lt) if lb + lt > 0 else 0.5

    def _default_junction(self, fraction: float) -> tuple[float, float]:
        p = self.start + fraction * (self.face_centre - self.start)
        return float(p[0]), float(p[2])

    # ---- by global s ---------------------------------------------------------------------------------------------------
    def locate(self, s) -> tuple[np.ndarray, np.ndarray]:
        """(is_trunk flag, Bézier parameter) for global arc-length fractions ``s`` (array)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        total = self.length_branch + self.length_trunk
        dist = s * total
        trunk = dist > self.length_branch
        t = np.where(trunk, self.trunk.t_at_length(np.maximum(dist - self.length_branch, 0.0)),
                     self.branch.t_at_length(np.minimum(dist, self.length_branch)))
        return trunk, t

    def point_at(self, s) -> np.ndarray:
        trunk, t = self.locate(s)
        out = np.empty((len(t), 3))
        for flag, curve in ((False, self.branch), (True, self.trunk)):
            m = trunk == flag
            if m.any():
                out[m] = curve.points(t[m])
        return out

    def tangent_at(self, s) -> np.ndarray:
        trunk, t = self.locate(s)
        out = np.empty((len(t), 3))
        for flag, curve in ((False, self.branch), (True, self.trunk)):
            m = trunk == flag
            if m.any():
                out[m] = curve.tangents(t[m])
        return out

    # ---- derived numbers ----------------------------------------------------------------------------------------------------
    @property
    def length(self) -> float:
        return self.length_branch + self.length_trunk

    def max_curvature(self) -> float:
        return max(self.branch.max_curvature(), self.trunk.max_curvature())

    def offset_over_length(self) -> tuple[float, float, float]:
        """(offset, length, offset / length): the lateral distance of the inlet centroid from the engine axis
        over the centreline length."""
        rel = self.start - self.face_centre
        offset = float(np.linalg.norm(rel - np.dot(rel, self.face_axis) * self.face_axis))
        return offset, self.length, offset / self.length if self.length > 0 else math.inf

    def handle_points(self) -> dict[str, np.ndarray]:
        """The 3D positions of the editable handles (the inner control points) and the junction."""
        b, t = self.branch.ctrl, self.trunk.ctrl
        return {"start": b[1], "junction_in": b[2], "junction": self.junction, "junction_out": t[1], "end": t[2]}
