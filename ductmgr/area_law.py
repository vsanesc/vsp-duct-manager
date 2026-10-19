# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The area law A(s) along a branch.

``s`` is the normalised arc length along one branch: 0 at the inlet, 1 at the end of the trunk (the engine face).
The two branches meet at the junction ``s_junction``. A smooth weight w(s) (linear or cosine) takes the
per-branch area from the inlet section area ``a_in`` to ``a_face / n_branches`` (the face area shared out among the
branches); beyond the junction the trunk carries all branches, so its area is n_branches times the branch value:

    branch(s) = a_in + (a_face / n - a_in) * w(s)
    A(s)      = branch(s)       for s <= s_junction
                n * branch(s)   for s >  s_junction     (so the junction area is 2 x the branch area, n = 2)

and A(1) = a_face.
"""
import math

import numpy as np

LAWS = ("linear", "cosine")
DEFAULT_LAW = "cosine"
DEFAULT_JUNCTION_FRACTION = 0.5  # of the inlet -> face distance, when no junction point is configured


def weight(s: float, law: str = DEFAULT_LAW) -> float:
    """The blending weight w(s) in [0, 1]: 0 at the inlet, 1 at the face."""
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"s must be within [0, 1] (got {s})")
    if law == "linear":
        return float(s)
    if law == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * s))
    raise ValueError(f"area law must be one of {', '.join(LAWS)} (got {law!r})")


def branch_area(s: float, a_in: float, a_face: float, law: str = DEFAULT_LAW, n_branches: int = 2) -> float:
    """The area of one branch at ``s`` ignoring the merge at the junction."""
    return a_in + (a_face / n_branches - a_in) * weight(s, law)


def section_area(s: float, a_in: float, a_face: float, s_junction: float, law: str = DEFAULT_LAW,
                 n_branches: int = 2) -> float:
    """A(s): a branch section up to the junction, the merged trunk section after it."""
    if not 0.0 <= s_junction <= 1.0:
        raise ValueError("s_junction must be within [0, 1]")
    a = branch_area(s, a_in, a_face, law, n_branches)
    return a * n_branches if s > s_junction else a


def junction_s(origin, face_centre, junction_point=None, fraction: float = DEFAULT_JUNCTION_FRACTION) -> float:
    """Where the junction sits along the path (straight-line proxy of the arc length).

    With a ``junction_point`` it is |origin -> junction| over |origin -> junction| + |junction -> face|;
    otherwise ``fraction`` (the configured fraction of the inlet -> face distance) is used.
    """
    if junction_point is not None and face_centre is not None:
        d1 = float(np.linalg.norm(np.asarray(junction_point, dtype=float) - np.asarray(origin, dtype=float)))
        d2 = float(np.linalg.norm(np.asarray(face_centre, dtype=float) - np.asarray(junction_point, dtype=float)))
        if d1 + d2 > 0.0:
            return d1 / (d1 + d2)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("the junction fraction must be within [0, 1]")
    return float(fraction)
