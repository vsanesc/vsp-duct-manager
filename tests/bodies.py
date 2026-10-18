# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Analytic test bodies for the section / sketch tests (axes: +X aft, +Y right, +Z up)."""
import numpy as np
import pyvista as pv

# Chined cross-section (y, z): sharp corners at the top chines (y = +-0.5, z = 0.4), the sides and the bottom.
CHINED = np.array([(-1.0, 0.0), (-0.5, 0.4), (0.5, 0.4), (1.0, 0.0), (1.0, -0.6), (-1.0, -0.6)])
LENGTH = 8.0


def cylinder(radius: float = 1.0, resolution: int = 360) -> pv.PolyData:
    """Closed cylinder along X, centred on the X axis, x = 0 .. 8."""
    return pv.Cylinder(center=(LENGTH / 2, 0.0, 0.0), direction=(1.0, 0.0, 0.0), radius=radius, height=LENGTH,
                       resolution=resolution).triangulate().clean()


def chined_prism() -> pv.PolyData:
    """The CHINED cross-section extruded along X, x = 0 .. 8."""
    n = len(CHINED)
    face = pv.PolyData(np.c_[np.zeros(n), CHINED[:, 0], CHINED[:, 1]], faces=np.r_[n, np.arange(n)])
    return face.extrude((LENGTH, 0.0, 0.0), capping=True).triangulate().clean()


# Section of the tapered body: a steeper side than CHINED keeps the top chine sharp (> 30 deg) while it widens.
TAPER_RING = np.array([(-1.0, -0.2), (-0.5, 0.6), (0.5, 0.6), (1.0, -0.2), (1.0, -0.8), (-1.0, -0.8)])


def tapered_chined_body(taper: float = 0.25, stations: int = 9) -> pv.PolyData:
    """TAPER_RING with its width growing linearly with x: y_scale(x) = 1 + taper * x.

    The top chine therefore runs along the planform line y = 0.5 * (1 + taper * x), z = 0.6.
    """
    xs = np.linspace(0.0, LENGTH, stations)
    ring = np.vstack([TAPER_RING, TAPER_RING[:1]])  # closed ring of points
    X = np.tile(xs, (len(ring), 1))
    Y = ring[:, 0][:, None] * (1.0 + taper * xs)[None, :]
    Z = np.tile(ring[:, 1][:, None], (1, stations))
    grid = pv.StructuredGrid(X[:, :, None], Y[:, :, None], Z[:, :, None])
    return grid.extract_surface(algorithm=None).triangulate().clean()


def polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance of points to an open polyline (analytic reference, written independently of the module)."""
    a, b = polyline[:-1], polyline[1:]
    d = b - a
    out = []
    for p in np.atleast_2d(points):
        t = np.clip(((p - a) * d).sum(axis=1) / (d * d).sum(axis=1), 0.0, 1.0)
        out.append(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())
    return np.array(out)
