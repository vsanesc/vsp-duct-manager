# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Inlet shape and placement (Phase 2: shape only, no ducts yet).

The inlet is an irregular pentagon (or a quadrilateral) described by ratios, scaled so its area equals
the per-inlet capture area. 2D coordinates are (u, v): u is lateral (+Y, outboard for the right inlet) and
v is "up" in the inlet plane. With flow direction f = -n (n is the outward plane normal facing the
freestream), (u, v, f) is right-handed, so the vertex order is counter-clockwise seen looking upstream.

Units: metres. VSP axes: +X aft, +Y right, +Z up.
"""
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

# Shape used when the config has no ``ducts.inlet.shape`` yet. Only the ratios matter.
DEFAULT_SHAPE = {
    "width_bottom": 1.0,
    "width_top": 0.7,
    "height": 1.0,
    "apex_height": 0.3,
    "skew": 0.0,
}
DEFAULT_TILT_DEG = 0.0


@dataclass(frozen=True)
class InletShape:
    """Shape ratios of the inlet polygon (absolute size comes from the capture area).

    width_bottom, width_top, height : relative sizes of the bottom edge, top edge and body height.
    apex_height : height of the 5th vertex above the top edge, as a fraction of ``height``;
                  0 gives a quadrilateral.
    skew        : lateral offset of the apex, as a fraction of ``width_top`` (+ = toward +u).
    """

    width_bottom: float = DEFAULT_SHAPE["width_bottom"]
    width_top: float = DEFAULT_SHAPE["width_top"]
    height: float = DEFAULT_SHAPE["height"]
    apex_height: float = DEFAULT_SHAPE["apex_height"]
    skew: float = DEFAULT_SHAPE["skew"]

    def __post_init__(self):
        for name in ("width_bottom", "width_top", "height"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"inlet shape '{name}' must be positive")
        if self.apex_height < 0.0:
            raise ValueError("inlet shape 'apex_height' must be >= 0")

    @classmethod
    def from_dict(cls, data: dict | None) -> "InletShape":
        """Build from a config mapping; missing or null keys take the defaults."""
        data = data or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown inlet shape key(s): {', '.join(sorted(unknown))}")
        return cls(**{k: float(v) for k, v in data.items() if v is not None})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_quadrilateral(self) -> bool:
        return self.apex_height == 0.0


def polygon_area(poly: np.ndarray) -> float:
    """Signed area of a 2D polygon (positive when counter-clockwise), shoelace formula."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area centroid of a 2D polygon."""
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a6 = 3.0 * float(cross.sum())
    return np.array([np.dot(x + xn, cross) / a6, np.dot(y + yn, cross) / a6])


def inlet_polygon(shape: InletShape, area: float) -> np.ndarray:
    """(N, 2) counter-clockwise vertices, centred on the area centroid, with area exactly ``area``.

    N is 4 for a quadrilateral (apex_height = 0) and 5 otherwise. Vertex order: bottom-left, bottom-right,
    top-right, [apex], top-left.
    """
    if not area > 0.0:
        raise ValueError("inlet area must be positive")
    wb, wt, h = shape.width_bottom, shape.width_top, shape.height
    pts = [(-wb / 2, 0.0), (wb / 2, 0.0), (wt / 2, h)]
    if not shape.is_quadrilateral:
        pts.append((shape.skew * wt, h * (1.0 + shape.apex_height)))
    pts.append((-wt / 2, h))
    poly = np.array(pts, dtype=float)
    poly -= polygon_centroid(poly)
    return poly * math.sqrt(area / polygon_area(poly))


def inlet_frame(tilt_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal in-plane axes (u, v) and outward plane normal n for a tilt about the Y axis.

    tilt = 0 gives n = -X (facing the freestream) with v = +Z. Positive tilt pitches the opening upward
    (n gains a +Z component) while the plane stays perpendicular to the XZ plane.
    """
    t = math.radians(tilt_deg)
    n = np.array([-math.cos(t), 0.0, math.sin(t)])
    u = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, n)
    return u, v, n


def place_polygon(poly: np.ndarray, centroid, tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """3D vertices of the polygon in the inlet plane through ``centroid``, and the plane normal."""
    u, v, n = inlet_frame(tilt_deg)
    c = np.asarray(centroid, dtype=float)
    return c + np.outer(poly[:, 0], u) + np.outer(poly[:, 1], v), n


def mirror_xz(points: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Mirror points about the XZ plane (y -> -y). ``reverse`` also reverses the vertex order."""
    out = np.array(points, dtype=float, copy=True)
    out[..., 1] *= -1.0
    return out[::-1].copy() if reverse else out


def mirror_normal(normal) -> np.ndarray:
    n = np.array(normal, dtype=float, copy=True)
    n[1] *= -1.0
    return n


def aspect_ratio(poly: np.ndarray) -> float:
    """Lateral width over total height of the polygon's bounding box."""
    extent = poly.max(axis=0) - poly.min(axis=0)
    return float(extent[0] / extent[1])


@dataclass(frozen=True)
class InletReadout:
    centroid: tuple[float, float, float]
    capture_area: float
    aspect_ratio: float
    distance_to_face: float  # straight-line distance, inlet centroid -> engine face centre
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
