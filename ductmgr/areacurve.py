# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The area law A(s) of the duct as two editable monotone (PCHIP) curves, one for the branch and one for the trunk.

Each curve is a function of its own local arc-length fraction ``s_local`` in [0, 1]:

- branch: A(0) = the inlet section area (fixed), A(1) = this branch's share of the junction area;
- trunk:  A(0) = the junction area, A(1) = the engine-face area (fixed).

So the area is continuous in the sense the geometry needs: the trunk starts with exactly the sum of the areas its
branches end with (their shares add up to 1). The junction area is a free value (default: the geometric mean of the combined inlet area
and the face area, so both stages diffuse by the same ratio); the interior control points default to the
configured law (linear or cosine) and become explicit once edited.
"""
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator

from ductmgr.area_law import DEFAULT_LAW, LAWS, weight

DEFAULT_INTERIOR_S = (0.25, 0.5, 0.75)  # where the default (law-derived) interior control points sit
MIN_S_GAP = 0.02  # interior control points keep at least this local-s distance from each other and the ends
CURVE_SAMPLES = 200


@dataclass(frozen=True)
class AreaPoints:
    """The editable inputs. ``None`` fields are derived from the law."""

    junction_area: float | None = None
    branch: tuple[tuple[float, float], ...] | None = None  # interior (s_local, area) points
    trunk: tuple[tuple[float, float], ...] | None = None

    def with_(self, **changes) -> "AreaPoints":
        return replace(self, **changes)


def parse_points(raw, where: str) -> tuple[tuple[float, float], ...] | None:
    """Interior control points from a config list of [s, A] pairs (``None`` stays None)."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where} must be a list of [s, area] pairs")
    pts = []
    for k, item in enumerate(raw):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"{where}[{k}] must be an [s, area] pair")
        pts.append((float(item[0]), float(item[1])))
    validate_interior(pts, where)
    return tuple(pts)


def validate_interior(points, where: str = "area points") -> None:
    prev = 0.0
    for s, a in points:
        if not 0.0 < s < 1.0:
            raise ValueError(f"{where}: s must be strictly between 0 and 1 (got {s:g})")
        if s <= prev:
            raise ValueError(f"{where}: s values must increase (got {s:g} after {prev:g})")
        if a <= 0.0:
            raise ValueError(f"{where}: areas must be positive (got {a:g})")
        prev = s


class AreaCurves:
    def __init__(self, a_in: float, a_face: float, points: AreaPoints | None = None, law: str = DEFAULT_LAW,
                 n_branches: int = 2, share: float | None = None, a_in_total: float | None = None):
        """One branch's curve and the trunk's. ``share`` is the fraction of the trunk flow this branch carries
        (default 1 / n_branches) and ``a_in_total`` the summed inlet area of all branches (default n_branches x
        ``a_in``): the junction area defaults to sqrt(a_in_total x a_face)."""
        if law not in LAWS:
            raise ValueError(f"area law must be one of {', '.join(LAWS)} (got {law!r})")
        if a_in <= 0.0 or a_face <= 0.0:
            raise ValueError("the inlet and face areas must be positive")
        points = points or AreaPoints()
        self.a_in, self.a_face, self.law, self.n = float(a_in), float(a_face), law, int(n_branches)
        self.share = 1.0 / self.n if share is None else float(share)
        if not 0.0 < self.share <= 1.0:
            raise ValueError("a branch's flow share must be in (0, 1]")
        self.points = points
        self.junction_area = (float(points.junction_area) if points.junction_area is not None
                              else math.sqrt((self.n * self.a_in if a_in_total is None else a_in_total) * self.a_face))
        if self.junction_area <= 0.0:
            raise ValueError("the junction area must be positive")
        a_branch_end = self.junction_area * self.share
        if points.branch is None:
            self.branch_interior = tuple((s, self.a_in + (a_branch_end - self.a_in) * weight(s, law))
                                         for s in DEFAULT_INTERIOR_S)
        else:
            validate_interior(points.branch, "branch area points")
            self.branch_interior = tuple(points.branch)
        if points.trunk is None:
            self.trunk_interior = tuple((s, self.junction_area + (self.a_face - self.junction_area) * weight(s, law))
                                        for s in DEFAULT_INTERIOR_S)
        else:
            validate_interior(points.trunk, "trunk area points")
            self.trunk_interior = tuple(points.trunk)
        self.branch_knots = ((0.0, self.a_in), *self.branch_interior, (1.0, a_branch_end))
        self.trunk_knots = ((0.0, self.junction_area), *self.trunk_interior, (1.0, self.a_face))
        self._branch = PchipInterpolator(*zip(*self.branch_knots))
        self._trunk = PchipInterpolator(*zip(*self.trunk_knots))

    @property
    def explicit(self) -> bool:
        p = self.points
        return p.junction_area is not None or p.branch is not None or p.trunk is not None

    def branch_area(self, s_local) -> np.ndarray:
        return self._branch(np.clip(s_local, 0.0, 1.0))

    def trunk_area(self, s_local) -> np.ndarray:
        return self._trunk(np.clip(s_local, 0.0, 1.0))

    def area(self, s, s_junction: float) -> np.ndarray:
        """A at global arc-length fractions ``s`` (the junction itself counts as the end of the branch)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        sj = max(float(s_junction), 1e-12)
        branch = self.branch_area(s / sj)
        trunk = self.trunk_area((s - s_junction) / max(1.0 - s_junction, 1e-12))
        return np.where(s <= s_junction + 1e-12, branch, trunk)

    def decreasing(self, curve: str) -> list[tuple[float, float]]:
        """Local-s intervals where the curve decreases (the diffuser must not shrink)."""
        fn = self._branch if curve == "branch" else self._trunk
        s = np.linspace(0.0, 1.0, CURVE_SAMPLES)
        down = np.diff(fn(s)) < -1e-12 * self.a_face
        out, start = [], None
        for i, flag in enumerate(down):
            if flag and start is None:
                start = s[i]
            if not flag and start is not None:
                out.append((float(start), float(s[i])))
                start = None
        if start is not None:
            out.append((float(start), 1.0))
        return out

    # ---- editing helpers (used by the 2D tab) ---------------------------------------------------------------------------
    def with_interior(self, curve: str, interior) -> AreaPoints:
        """The area points with ``curve``'s interior replaced (and the other curves made explicit too)."""
        interior = tuple(sorted((float(s), float(a)) for s, a in interior))
        validate_interior(interior, f"{curve} area points")
        return AreaPoints(self.junction_area, interior if curve == "branch" else self.branch_interior,
                          interior if curve == "trunk" else self.trunk_interior)

    def with_junction_area(self, area: float) -> AreaPoints:
        return AreaPoints(float(area), self.branch_interior, self.trunk_interior)


def points_to_config(points: AreaPoints) -> dict:
    """The ``ducts.area_points`` mapping (None values stay None = derived)."""
    def listed(p):
        return None if p is None else [[float(s), float(a)] for s, a in p]

    return {"junction_area": None if points.junction_area is None else float(points.junction_area),
            "branch": listed(points.branch), "trunk": listed(points.trunk)}


def points_from_config(data) -> AreaPoints:
    if data is None:
        return AreaPoints()
    if not isinstance(data, dict):
        raise ValueError("ducts.area_points must be a mapping")
    junction = data.get("junction_area")
    if junction is not None and float(junction) <= 0.0:
        raise ValueError("ducts.area_points.junction_area must be positive")
    return AreaPoints(None if junction is None else float(junction),
                      parse_points(data.get("branch"), "ducts.area_points.branch"),
                      parse_points(data.get("trunk"), "ducts.area_points.trunk"))
