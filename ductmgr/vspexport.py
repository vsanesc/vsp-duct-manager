# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Phase 4: the duct as OpenVSP Stack geoms. This module plans the export (pure geometry, no OpenVSP calls); ``vspio``
writes the plan into the model and ``exportcheck`` verifies the saved file.

Geoms, with ``<P>`` the configured prefix (``model.duct_prefix``, default ``DUCT_``):

 * per independent inlet branch ``<P>BRANCH_<name>_POS`` (the positive: the inner section offset outward by the wall
   thickness, from the opening plane to the junction) and ``<P>BRANCH_<name>_NEG`` (``Negative_Volume_Flag`` = 1: the
   inner sections, from ``lip_extension`` upstream of the opening, through a prismatic segment along the inlet axis,
   to the junction and ``overlap`` beyond it). A mirrored twin is not a second Geom: the pair gets OpenVSP's planar XZ
   symmetry; an inlet centred on y = 0 gets a plain pair;
 * ``<P>TRUNK_POS`` and ``<P>TRUNK_NEG`` for the trunk (the negative starts ``overlap`` upstream of the junction and ends
   at the engine face).

Every Stack is closed by flat end caps and has one Edit Curve XSec per station. The caps are not OpenVSP's own cap option
(which loses a segment of the section for some Edit Curves) but a POINT XSec ``cap_thickness`` beyond each end with its
tangent in the section plane: a disc of that thickness, watertight in OpenVSP's mesh. The stations are chosen adaptively from a fine
resampling of the same duct (denser where the centreline bends or the section changes), each section is a short chain of
Beziers (``sectionfit``) and the station frames become the Stack's XDelta / YDelta / ZDelta and XRotate / YRotate /
ZRotate. In OpenVSP a section's frame is (x = the flow direction T, y = -U, z = V): for a duct flowing along +X with
V up that is the identity, so the sections look natural in the GUI; the (u, v) ring is mapped to (y, z) = (-u, v).

The Stack starts with a prismatic segment: the section of the opening plane repeated along the inlet axis down to below
the skin, so the negative volume cuts the skin exactly along the sketched outline; the transition to the other sections
happens after that, inside the fuselage.
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np

from ductmgr import duct as dm
from ductmgr import sectionfit
from ductmgr.geometry import polygon_area

DEFAULT_PREFIX = "DUCT_"
MIRROR_UV = np.array([-1.0, 1.0])  # (u, v) of a ring -> (y, z) of the OpenVSP XSec


@dataclass(frozen=True)
class ExportSettings:
    """The ``export:`` block of the config (every value has a default)."""

    fine_stations: int = 48  # stations per part of the fine resampling the adaptive selection starts from
    station_tolerance: float = 0.005  # of the section size: how far the interpolated sections may be from the design
    min_stations: int = 3  # XSecs per Geom at least
    cap_thickness: float = 0.001  # m, how far beyond each end section the point XSec of the flat cap lies
    cap_strength: float = 0.01  # skinning tangent strength next to a cap (small: the cap is flat, it does not bulge)
    cedit_anchors: int = sectionfit.DEFAULT_ANCHORS  # segments per section at least (more where the sections have corners)
    cedit_tolerance: float = sectionfit.DEFAULT_TOLERANCE  # of the section size: else a linear CEDIT through the points
    cedit_corner_deg: float = sectionfit.DEFAULT_CORNER_DEG
    cedit_start: int = 1  # the Edit Curves start at the anchor this many places after ring point 0 (see sectionfit)
    prism_min: float = 0.03  # m, the shortest prismatic segment after the opening plane
    prism_clearance: float = 0.01  # m, how far inside the skin its end must be
    skin_tolerance: float = 0.003  # m, a section point this close to the skin counts as on it
    tess_w: int = 65  # OpenVSP tessellation around the section (an odd number; one more than the points)
    tess_u: int = 9  # ... along each Stack interval
    verify: bool = True
    retry_on_crash: bool = True  # a Geom whose OpenVSP process dies is made again with other sections (reported); else an ERROR
    area_tolerance: float = 0.03  # verification: WARN above this relative error
    volume_tolerance: float = 0.03
    volume_samples: int = 20000  # sample points per body for the volume estimate

    @classmethod
    def from_config(cls, cfg: dict) -> "ExportSettings":
        c = (cfg or {}).get("export") or {}
        if not isinstance(c, dict):
            raise ValueError("export must be a mapping")
        base = cls()
        values = {}
        for f in cls.__dataclass_fields__:
            v = c.get(f)
            if v is None:
                continue
            default = getattr(base, f)
            values[f] = bool(v) if isinstance(default, bool) else type(default)(v)
        out = cls(**values)
        positive = ("fine_stations", "min_stations", "cedit_anchors", "cedit_tolerance", "cedit_corner_deg",
                    "station_tolerance", "cap_thickness", "cap_strength", "tess_w", "tess_u", "area_tolerance", "volume_tolerance", "volume_samples")
        for f in positive:
            if getattr(out, f) <= 0:
                raise ValueError(f"export.{f} must be positive")
        if out.min_stations < 2 or out.fine_stations < 4:
            raise ValueError("export.min_stations must be at least 2 and export.fine_stations at least 4")
        if out.prism_min < 0.0 or out.prism_clearance < 0.0 or out.skin_tolerance < 0.0:
            raise ValueError("export.prism_min, prism_clearance and skin_tolerance must not be negative")
        return out


@dataclass
class SectionPlan:
    """One XSec of a Stack: where it is, how it is oriented and its Edit Curve."""

    origin: np.ndarray  # world position of the XSec origin (the centre of the section's bounding box)
    R: np.ndarray  # 3 x 3, columns = the XSec's x (flow), y, z axes in world coordinates
    cedit: object  # cedit.CeditSection; None for the point XSec of an end cap
    dist: float  # arc length from the opening plane
    role: str  # "lip", "station", "prism", "into", "overlap" or "cap"
    index: int | None  # design station index (branch or trunk) when this is a design station
    ring: np.ndarray  # the ring in XSec coordinates (y, z), as given to the fit


@dataclass
class GeomPlan:
    name: str
    kind: str  # "positive" or "negative"
    part: str  # "branch" or "trunk"
    inlet: str | None  # the inlet's name (branches)
    mirrored: bool  # OpenVSP planar XZ symmetry: the mirrored twin comes from the Geom itself
    sections: list = field(default_factory=list)
    n_design_rings: int = 0  # the rings the adaptive selection started from
    fit: sectionfit.FitResult | None = None
    notes: list = field(default_factory=list)  # WARN / info lines for the report
    prism_depth: float | None = None
    source: list = field(default_factory=list)  # the (fine) rings the plan was made from, to make it again with other settings
    base_notes: list = field(default_factory=list)  # the notes given when it was made (the fit adds its own to ``notes``)

    @property
    def negative(self) -> bool:
        return self.kind == "negative"

    @property
    def real_sections(self) -> list:
        """The sections that carry a curve (everything but the two cap points)."""
        return [s for s in self.sections if s.cedit is not None]

    @property
    def summary(self) -> str:
        fit = self.fit
        shape = "linear CEDIT through the ring points" if fit.linear else \
            f"{fit.n_segments} Bezier segments per section, fit deviation max {100 * fit.max_dev:.2f} % of the size"
        sym = ", planar XZ symmetry" if self.mirrored else ""
        return (f"{len(self.real_sections)} XSecs + 2 cap points (adaptive, from {self.n_design_rings} rings), "
                f"{shape}{sym}")


@dataclass
class ExportPlan:
    geoms: list
    settings: ExportSettings
    prefix: str

    def geom(self, name: str) -> GeomPlan:
        return next(g for g in self.geoms if g.name == name)


def sanitize(name: str) -> str:
    """A Geom-name-safe version of an inlet name (spaces and punctuation become underscores)."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "inlet"


# ---- OpenVSP's frame conventions -------------------------------------------------------------------------------------
def frame_matrix(T: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """The XSec frame of a station: columns x = T (flow), y = -U, z = V (right-handed since U = T x V)."""
    return np.column_stack([T, -U, V])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_xyz(M: np.ndarray) -> tuple[float, float, float]:
    """Angles (degrees) with ``M = Rx(a) Ry(b) Rz(c)``, OpenVSP's XRotate / YRotate / ZRotate of an XSec."""
    sb = float(np.clip(M[0, 2], -1.0, 1.0))
    b = math.asin(sb)
    if abs(sb) < 1.0 - 1e-12:
        a = math.atan2(-M[1, 2], M[2, 2])
        c = math.atan2(-M[0, 1], M[0, 0])
    else:  # gimbal lock: only a +- c is determined; put it all on a
        c = 0.0
        a = math.atan2(M[2, 1], M[1, 1])
    return math.degrees(a), math.degrees(b), math.degrees(c)


def stack_parameters(sections: list) -> list[dict]:
    """The Stack parameters of every XSec for the given absolute positions and frames. In OpenVSP (relative mode) the
    position of XSec i is the one of i - 1 plus the previous frame applied to (XDelta, YDelta, ZDelta), and its frame is
    the previous one times Rx(XRotate) Ry(YRotate) Rz(ZRotate); the first XSec is relative to the Geom (the identity)."""
    out = []
    prev_R, prev_p = np.eye(3), np.zeros(3)
    for s in sections:
        delta = prev_R.T @ (np.asarray(s.origin) - prev_p)
        rx, ry, rz = euler_xyz(prev_R.T @ s.R)
        out.append({"XDelta": float(delta[0]), "YDelta": float(delta[1]), "ZDelta": float(delta[2]),
                    "XRotate": rx, "YRotate": ry, "ZRotate": rz})
        prev_R, prev_p = s.R, np.asarray(s.origin, dtype=float)
    return out


def replay_stack(params: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(position, frame) of every XSec that OpenVSP derives from ``params`` (the inverse of ``stack_parameters``)."""
    out = []
    R, p = np.eye(3), np.zeros(3)
    for q in params:
        p = p + R @ np.array([q["XDelta"], q["YDelta"], q["ZDelta"]])
        R = R @ rot_x(math.radians(q["XRotate"])) @ rot_y(math.radians(q["YRotate"])) @ rot_z(math.radians(q["ZRotate"]))
        out.append((p.copy(), R.copy()))
    return out


# ---- adaptive station selection --------------------------------------------------------------------------------------
def _hermite_errors(points: np.ndarray, dist: np.ndarray, kept: list[int]) -> np.ndarray:
    """Distance (max over the points of a ring) between every ring and the cubic through the kept rings (Hermite with
    finite-difference tangents): what OpenVSP's smooth loft would do to a ring that is left out."""
    k = np.array(kept)
    pk, dk = points[k], dist[k]
    m = np.empty_like(pk)
    if len(k) == 2:
        m[:] = ((pk[1] - pk[0]) / (dk[1] - dk[0]))[None]
    else:
        m[1:-1] = (pk[2:] - pk[:-2]) / (dk[2:] - dk[:-2])[:, None, None]
        m[0] = (pk[1] - pk[0]) / (dk[1] - dk[0])
        m[-1] = (pk[-1] - pk[-2]) / (dk[-1] - dk[-2])
    err = np.zeros(len(points))
    for i in range(len(k) - 1):
        a, b = k[i], k[i + 1]
        h = dk[i + 1] - dk[i]
        for r in range(a + 1, b):
            u = (dist[r] - dk[i]) / h
            h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
            p = h00 * pk[i] + h10 * h * m[i] + h01 * pk[i + 1] + h11 * h * m[i + 1]
            err[r] = float(np.linalg.norm(p - points[r], axis=1).max())
    return err


def select_rings(points: np.ndarray, dist: np.ndarray, tolerance: float, mandatory=(), min_count: int = 3) -> list[int]:
    """Indices of the rings to keep: the ends and the ``mandatory`` ones, then, one at a time, the ring that the
    interpolation through the kept ones misses most, until every other ring is within ``tolerance`` (metres) and at
    least ``min_count`` are kept."""
    r = len(points)
    kept = sorted({0, r - 1} | {int(i) for i in mandatory})
    while len(kept) < r:
        err = _hermite_errors(points, np.asarray(dist, dtype=float), kept)
        worst = int(np.argmax(err))
        if err[worst] <= tolerance:
            if len(kept) >= min_count:
                break
            gaps = [(kept[i + 1] - kept[i], kept[i]) for i in range(len(kept) - 1)]
            gap, start = max(gaps)
            if gap < 2:
                break
            worst = start + gap // 2  # too few XSecs: halve the longest interval
        kept = sorted(kept + [worst])
    return kept


# ---- the prismatic start ---------------------------------------------------------------------------------------------
def prism_depth(opening: dm.RingSection, outer: np.ndarray, skin, settings: ExportSettings, max_depth: float):
    """How far along the inlet axis the opening section is repeated: the smallest depth at which the points of the
    section that lie on the skin at the opening plane are ``prism_clearance`` inside it (at least ``prism_min``, at
    most ``max_depth``). Returns (depth, note)."""
    floor = min(settings.prism_min, max_depth)
    if skin is None:
        return max(floor, 0.25 * max_depth), ""
    pts = np.vstack([opening.points(), outer])
    on_skin = np.abs(skin.signed(pts)) < settings.skin_tolerance
    if not on_skin.any():
        return floor, ""
    base = pts[on_skin]
    depths = np.linspace(0.0, max_depth, 61)
    sd = np.array([skin.signed(base + d * opening.T).max() for d in depths])
    ok = np.flatnonzero(sd <= -settings.prism_clearance)
    if len(ok):
        return max(float(depths[ok[0]]), floor), ""
    return max_depth, (f"WARN the skin does not close over the opening within {max_depth:.3f} m along the inlet axis: the "
                       f"prismatic segment ({max_depth:.3f} m) ends only {max(0.0, -sd.min()) * 1000:.1f} mm inside it "
                       f"(wanted {settings.prism_clearance * 1000:.0f} mm), so the transition may graze the skin outline")


# ---- the plan --------------------------------------------------------------------------------------------------------
def _equivalent_diameter(sections: list) -> float:
    return max(2.0 * math.sqrt(abs(polygon_area(s.ring)) / math.pi) for s in sections)


def _with_prism(rings: list, depth: float) -> list:
    """The rings with the prismatic ring (the opening section ``depth`` further along the flow) after the opening and
    the design rings it replaces left out."""
    k = next(i for i, s in enumerate(rings) if s.role == "station" and s.index == 0)
    o = rings[k]
    prism = dm.RingSection(depth, o.origin + depth * o.T, o.T, o.U, o.V, o.ring, None, "prism")
    return rings[:k + 1] + [prism] + [s for s in rings[k + 1:] if s.dist > depth + 1e-9]


def _mandatory(rings: list) -> set[int]:
    """The rings the interpolation must not drop: the prismatic and lip rings, the opening, the junction and everything
    beyond it (overlap, splitter), and the first ring of a trunk."""
    keep = {i for i, s in enumerate(rings) if s.role in ("lip", "prism", "into", "overlap")}
    for i, s in enumerate(rings):
        if s.role == "station" and s.index == 0:
            keep.add(i)
    first_into = next((i for i, s in enumerate(rings) if s.role == "into"), None)
    if first_into is not None:
        keep.add(first_into - 1)  # the junction plane
    return keep


def _sections_plan(rings: list, kept: list[int], fit: sectionfit.FitResult) -> list[SectionPlan]:
    out = []
    for i, c in zip(kept, fit.sections):
        s = rings[i]
        R = frame_matrix(s.T, s.U, s.V)
        origin = np.asarray(s.origin) + R @ np.array([0.0, c.centre[0], c.centre[1]])
        out.append(SectionPlan(origin, R, c, s.dist, s.role, s.index, s.ring * MIRROR_UV))
    return out


def _with_caps(sections: list[SectionPlan], thickness: float) -> list[SectionPlan]:
    """The sections with a point XSec ``thickness`` beyond each end, in the frame of that end."""
    first, last = sections[0], sections[-1]
    cap0 = SectionPlan(first.origin - thickness * first.R[:, 0], first.R, None, first.dist - thickness, "cap", None,
                       np.zeros((0, 2)))
    cap1 = SectionPlan(last.origin + thickness * last.R[:, 0], last.R, None, last.dist + thickness, "cap", None,
                       np.zeros((0, 2)))
    return [cap0] + sections + [cap1]


def build_geom(name: str, kind: str, part: str, inlet: str | None, mirrored: bool, rings: list,
               settings: ExportSettings, notes=()) -> GeomPlan:
    """The Stack plan of one body from its (fine) rings: adaptive XSec selection, then the common-anchor CEDIT fit."""
    points = np.array([s.points() for s in rings])
    dist = np.array([s.dist for s in rings])
    tol = settings.station_tolerance * _equivalent_diameter(rings)
    kept = select_rings(points, dist, tol, _mandatory(rings), settings.min_stations)
    fit = sectionfit.fit_rings([rings[i].ring * MIRROR_UV for i in kept], settings.cedit_anchors,
                               settings.cedit_tolerance, settings.cedit_corner_deg, settings.cedit_start)
    plan = GeomPlan(name, kind, part, inlet, mirrored, _with_caps(_sections_plan(rings, kept, fit), settings.cap_thickness),
                    len(rings), fit, list(notes), source=rings, base_notes=list(notes))
    if fit.linear:
        plan.notes.append(f"WARN {name}: the Bezier fit is not within {100 * settings.cedit_tolerance:g} % of the "
                          "section size; the sections are linear CEDITs through the ring points")
    return plan


def rebuild_geom(g: GeomPlan, settings: ExportSettings) -> GeomPlan:
    """``g`` made again from its rings with other ``settings`` (another number of segments, another station tolerance):
    different data for OpenVSP, whose heap corruption depends on the exact curves it is given."""
    out = build_geom(g.name, g.kind, g.part, g.inlet, g.mirrored, g.source, settings, g.base_notes)
    out.prism_depth = g.prism_depth
    return out


def plan_export(fine: dm.Duct, design: dm.Duct, settings: ExportSettings, skin=None,
                prefix: str = DEFAULT_PREFIX) -> ExportPlan:
    """The geoms of the export. ``fine`` is the duct resampled with many stations (the adaptive selection works on its
    rings), ``design`` the duct as the viewer shows it (its first station limits the prismatic segment), ``skin`` the
    fuselage solid (None: a default depth)."""
    sections = fine.body_sections(single_overlap=True)
    geoms = []
    names = set()
    for bg in fine.branches:
        if bg.twin:
            continue
        label = sanitize(bg.name)
        if label in names:
            raise ValueError(f"inlet names must stay distinct after cleaning them for Geom names ({bg.name!r})")
        names.add(label)
        twin = any(b.twin and b.inlet == bg.inlet for b in fine.branches)
        neg, pos = sections[f"neg_branch_{bg.name}"], sections[f"pos_branch_{bg.name}"]
        design_bg = next(b for b in design.branches if not b.twin and b.inlet == bg.inlet)
        first = design_bg.stations[1].dist if len(design_bg.stations) > 1 else bg.stations[1].dist
        opening = next(s for s in neg if s.role == "station" and s.index == 0)
        outer = next(s for s in pos if s.index == 0).points()
        depth, note = prism_depth(opening, outer, skin, settings, 0.5 * first)
        notes = [note] if note else []
        for tag, kind, rings in (("POS", "positive", pos), ("NEG", "negative", neg)):
            g = build_geom(f"{prefix}BRANCH_{label}_{tag}", kind, "branch", bg.name, twin, _with_prism(rings, depth),
                           settings, notes if tag == "NEG" else [])
            g.prism_depth = depth
            geoms.append(g)
    for tag, kind in (("POS", "positive"), ("NEG", "negative")):
        geoms.append(build_geom(f"{prefix}TRUNK_{tag}", kind, "trunk", None, False, sections[f"{tag.lower()}_trunk"],
                                settings))
    return ExportPlan(geoms, settings, prefix)


def fine_duct(design: dm.Duct, settings: ExportSettings) -> dm.Duct:
    """The design duct resampled with ``settings.fine_stations`` stations per part (never fewer than the design's)."""
    from dataclasses import replace

    spec = design.spec
    return dm.Duct(replace(spec, n_stations_branch=max(spec.n_stations_branch, settings.fine_stations),
                           n_stations_trunk=max(spec.n_stations_trunk, settings.fine_stations)))
