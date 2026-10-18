# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The inlet plane and 2D polygon helpers.

The inlet plane is defined by an origin O and two angles. With ``pitch`` (rotation about Y, opening tilted up
for positive values) and ``sweep`` (rotation about Z), the outward normal, facing the freestream, is

    n = R_z(sweep) . R_y(pitch) . (-X)

so pitch = sweep = 0 is the plane x = const facing forward. The in-plane basis is v = the component of +Z
orthogonal to n (normalised) and u = v x n. Seen from the front, looking along -n towards the opening, u is to
the right and v is up (so the aircraft's +Y side appears on the left). Every sketch vertex is stored as (u, v)
in metres relative to O; (y, z) is never stored.

The left inlet is the mirror image of the right one about the XZ plane: mirror O, negate sweep, and negate u
of every vertex (mirroring flips handedness, so the vertex order is reversed to stay counter-clockwise).

Units: metres. VSP axes: +X aft, +Y right, +Z up.
"""
import math
from dataclasses import dataclass

import numpy as np

FREESTREAM = np.array([1.0, 0.0, 0.0])  # air moves aft (+X) relative to the aircraft
MAX_PITCH_DEG = 89.0  # the frame is singular at +-90 deg (v is undefined)


def plane_frame(pitch_deg: float, sweep_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal in-plane axes (u, v) and the outward normal n (facing the freestream)."""
    if abs(pitch_deg) > MAX_PITCH_DEG:
        raise ValueError(f"|pitch| must be at most {MAX_PITCH_DEG:g} deg (the in-plane frame is singular at 90)")
    t, s = math.radians(pitch_deg), math.radians(sweep_deg)
    n = np.array([-math.cos(t) * math.cos(s), -math.cos(t) * math.sin(s), math.sin(t)])
    z = np.array([0.0, 0.0, 1.0])
    v = z - np.dot(z, n) * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    return u, v, n


@dataclass(frozen=True)
class InletPlane:
    origin: tuple[float, float, float]
    pitch_deg: float = 0.0
    sweep_deg: float = 0.0

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return plane_frame(self.pitch_deg, self.sweep_deg)

    @property
    def normal(self) -> np.ndarray:
        return self.frame()[2]

    @property
    def projection_factor(self) -> float:
        """|n . X|: in-plane area times this is the area projected normal to the freestream."""
        return abs(float(np.dot(self.normal, FREESTREAM)))

    def to_3d(self, uv) -> np.ndarray:
        """3D points of (u, v) coordinates, shape (N, 3)."""
        u, v, _ = self.frame()
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return np.asarray(self.origin) + np.outer(uv[:, 0], u) + np.outer(uv[:, 1], v)

    def to_uv(self, points) -> np.ndarray:
        """(u, v) of 3D points (orthogonal projection onto the plane), shape (N, 2)."""
        u, v, _ = self.frame()
        rel = np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(self.origin)
        return np.c_[rel @ u, rel @ v]

    def mirrored(self) -> "InletPlane":
        """The left inlet's plane: origin mirrored about XZ, sweep negated."""
        x, y, z = self.origin
        return InletPlane((x, -y, z), self.pitch_deg, -self.sweep_deg)


def planform_trace_direction(plane: "InletPlane") -> tuple[float, float]:
    """Unit planform direction (dx, dy), outboard (dy >= 0), of the trace of the plane on the horizontal plane
    through its origin. The trace is n x Z; it does not depend on the pitch."""
    n = plane.normal
    dx, dy = float(n[1]), float(-n[0])  # n x Z = (n_y, -n_x, 0)
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        raise ValueError("the plane is horizontal: it has no planform trace")
    dx, dy = dx / norm, dy / norm
    return (-dx, -dy) if dy < 0.0 else (dx, dy)


def sweep_for_edge_angle(edge_sweep_deg: float) -> float:
    """The plane sweep that makes the planform trace parallel to an edge swept back by ``edge_sweep_deg``.

    The right inlet's outer (+Y) edge then trails aft. The trace direction is (-sin(sweep), cos(sweep)), the edge
    direction (sin(edge), cos(edge)), hence sweep = -edge.
    """
    return -float(edge_sweep_deg)


def mirror_vertices(vertices, flags=None):
    """The left inlet's (u, v) vertices (and optional per-vertex flags): u negated, order reversed."""
    mirrored = [(-float(u), float(v)) for u, v in vertices][::-1]
    return mirrored if flags is None else (mirrored, list(flags)[::-1])


# ---- 2D polygon helpers ---------------------------------------------------------------------------
def polygon_area(poly: np.ndarray) -> float:
    """Signed area of a 2D polygon (positive when counter-clockwise), shoelace formula."""
    poly = np.asarray(poly, dtype=float)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area centroid of a 2D polygon."""
    poly = np.asarray(poly, dtype=float)
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a6 = 3.0 * float(cross.sum())
    return np.array([np.dot(x + xn, cross) / a6, np.dot(y + yn, cross) / a6])


def aspect_ratio(poly: np.ndarray) -> float:
    """Width over height of the polygon's bounding box."""
    extent = np.asarray(poly).max(axis=0) - np.asarray(poly).min(axis=0)
    return float(extent[0] / extent[1])


@dataclass(frozen=True)
class InletReadout:
    centroid: tuple[float, float, float]
    capture_area: float
    aspect_ratio: float
    distance_to_face: float  # straight-line distance, opening centroid -> engine face centre
    dy: float  # lateral offset, inlet -> face
    dz: float  # vertical offset, inlet -> face
    offset: float  # hypot(dy, dz): distance of the inlet from the engine axis line
    length: float  # axial distance (face x - inlet x); the engine axis is along X
    offset_over_length: float  # S-duct severity; nan when the face is not aft of the inlet


def inlet_readout(centroid, face_center, poly: np.ndarray, capture_area: float) -> InletReadout:
    c = np.asarray(centroid, dtype=float)
    f = np.asarray(face_center, dtype=float)
    dx, dy, dz = f - c
    offset = math.hypot(dy, dz)
    return InletReadout(
        centroid=tuple(float(x) for x in c),
        capture_area=float(capture_area),
        aspect_ratio=aspect_ratio(poly),
        distance_to_face=float(np.linalg.norm(f - c)),
        dy=float(dy),
        dz=float(dz),
        offset=offset,
        length=float(dx),
        offset_over_length=offset / dx if dx > 0.0 else math.nan,
    )
