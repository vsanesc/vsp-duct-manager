# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""State of the inlet / duct explorer, independent of any GUI toolkit.

The design has 1..N inlets. Each has its own plane (origin, pitch, sweep), *key sections* along its branch (the
inlet section at s = 0, in the inlet plane with skin snapping, and any intermediate sections; the fixed junction and
engine face sections close the list), a flow fraction and a mirror flag. A global symmetry (``xz`` or ``none``) says
whether mirrored twins are generated and whether the junction is constrained to y = 0. One inlet is *selected*: the
section editor, the 2D views and the panel edit it. Every key section has its own shape (any number of vertices,
straight or curved edges) but its size follows the area law A(s). One undo / redo history is shared by the 3D view,
the panel and the editors; there are also the place-mode state machine, the lock, the dirty flag and save / revert.
The Qt window and the headless screenshot both drive this class, so everything that matters is testable without a
display.
"""
import math
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ductmgr import area_law, areacurve, ductcheck, sketch, vspio
from ductmgr import duct as dm
from ductmgr import interp
from ductmgr import outline as ol
from ductmgr.areacurve import AreaCurves, AreaPoints
from ductmgr.centreline import Centreline, CentrelineParams
from ductmgr.config import PRIVATE_CONFIG, load_config, save_inlet_definition
from ductmgr.engine import EngineSizingWarning, size_from_config
from ductmgr.geometry import InletPlane, inlet_readout, polygon_centroid, sweep_for_edge_angle
from ductmgr.outline import SectionShape
from ductmgr.section import Section, compute_section
from ductmgr.solids import Solid

MIN_ALIGN_PROJECTION = 0.2  # |n.X| below this blocks an edge alignment
MIN_S_SEPARATION = 0.01  # key sections closer than this in s are refused
UNDO_LIMIT = 200  # history entries kept (the requirement is at least 50)
MERGE_WINDOW_S = 0.8  # consecutive edits of one parameter / vertex within this window share one undo step
PLANE_KEYS = ("pitch_deg", "sweep_deg")

MODE_VIEW, MODE_PLACING = "view", "placing"
MODE_TEXT = {MODE_VIEW: "View", MODE_PLACING: "Placing inlet - click on fuselage, Esc to cancel"}

LEGACY_INLET_KEYS = ("shape", "skin_fit", "attach_point", "centroid", "normal", "tilt_deg", "station_x", "polygon")
SYMMETRIES = dm.SYMMETRIES
DEFAULT_SYMMETRY = "xz"
DEFAULT_INLET_NAME = "inlet"


@dataclass(frozen=True)
class KeySection:
    s: float  # normalised arc length along the branch: 0 inlet, 1 engine face
    shape: SectionShape


DEFAULT_WALL_THICKNESS = 0.005  # m, when ducts.wall_thickness is not configured
DEFAULT_LIP_EXTENSION = 0.05  # m, when ducts.lip_extension (or the old ducts.inlet.lip_extension) is not configured
HEAVY_KEYS = ("wall_thickness", "lip_extension")


@dataclass(frozen=True)
class DuctParams:
    """The editable duct inputs besides the inlets (all part of the undo history): the shared junction / trunk
    centreline, the junction and trunk area points, the walls."""

    centreline: CentrelineParams = CentrelineParams()
    area: AreaPoints = AreaPoints()  # junction area and trunk points (the branch points belong to each inlet)
    wall_thickness: float = DEFAULT_WALL_THICKNESS
    lip_extension: float = DEFAULT_LIP_EXTENSION
    lip_mode: str = dm.DEFAULT_LIP_MODE


@dataclass(frozen=True)
class Inlet:
    name: str
    plane: InletPlane | None = None  # None until the inlet has been placed
    sections: tuple[KeySection, ...] = ()  # sorted by s; sections[0] is the inlet (s = 0) once placed
    flow_fraction: float | None = None  # per physical branch; None = an equal share
    mirror: bool = True  # a mirrored twin (only with symmetry xz; an inlet centred on y = 0 never gets one)
    start_handle: float | None = None
    junction_in_handle: float | None = None
    area_branch: tuple | None = None  # interior points of this branch's area curve; None = from the area law

    @property
    def placed(self) -> bool:
        return self.plane is not None

    @property
    def self_symmetric(self) -> bool:
        return self.plane is not None and dm.self_symmetric(self.plane)


@dataclass(frozen=True)
class InletState:
    """Everything the undo history tracks. ``selected`` is only the selection: it is not part of equality."""

    inlets: tuple[Inlet, ...]
    symmetry: str = DEFAULT_SYMMETRY
    duct: DuctParams = DuctParams()
    selected: int = field(default=0, compare=False)

    @property
    def inlet(self) -> Inlet:
        return self.inlets[min(self.selected, len(self.inlets) - 1)]

    @property
    def plane(self) -> InletPlane | None:
        return self.inlet.plane

    @property
    def sections(self) -> tuple[KeySection, ...]:
        return self.inlet.sections

    def with_inlet(self, index: int, **changes) -> "InletState":
        inlets = list(self.inlets)
        inlets[index] = replace(inlets[index], **changes)
        return replace(self, inlets=tuple(inlets))

    def with_selected(self, **changes) -> "InletState":
        return self.with_inlet(min(self.selected, len(self.inlets) - 1), **changes)


@dataclass(frozen=True)
class Readout:
    label: str
    value: str
    level: str = "info"  # "info", "pass", "warn" or "error"


@dataclass(frozen=True)
class DuctsSettings:
    """The ``ducts:`` / ``checks:`` values that are not edited in the explorer."""

    law: str = area_law.DEFAULT_LAW
    junction_fraction: float = area_law.DEFAULT_JUNCTION_FRACTION
    face_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    n_stations_branch: int = 12
    n_stations_trunk: int = 12
    n_points: int = interp.DEFAULT_SAMPLES
    checks: ductcheck.CheckSettings = ductcheck.CheckSettings()
    splitter_length: float = 0.0  # ducts.junction.splitter_length
    overlap: float = dm.DEFAULT_OVERLAP  # ducts.junction.overlap


# ---- config <-> state -----------------------------------------------------------------------------------------
def legacy_notice(cfg: dict) -> str:
    """One line naming old formats that are migrated or ignored; empty when there are none."""
    ducts = cfg.get("ducts") or {}
    inlet = ducts.get("inlet") or {}
    notes = []
    old = [k for k in LEGACY_INLET_KEYS if inlet.get(k) is not None]
    if old:
        notes.append(f"ignoring old key(s) {', '.join(old)}")
    if ducts.get("inlets") is None and inlet.get("origin") is not None:
        mirror = ducts.get("n_inlets") != 1
        what = "ducts.inlet / ducts.key_sections"
        if inlet.get("vertices") is not None and ducts.get("key_sections") is None:
            what = "the 5-vertex ducts.inlet.vertices / snapped format (a straight-edged inlet section)"
        notes.append(f"{what} migrated to ducts.inlets (1 inlet, mirror {'true' if mirror else 'false'}, symmetry "
                     "xz); Save writes the new format")
    elif inlet.get("vertices") is not None:
        notes.append("ducts.inlet.vertices / snapped are superseded by ducts.inlets and ignored")
    return "ducts: " + "; ".join(notes) if notes else ""


def duct_settings(cfg: dict) -> DuctsSettings:
    ducts = cfg.get("ducts") or {}
    law = ducts.get("area_law") or area_law.DEFAULT_LAW
    if law not in area_law.LAWS:
        raise ValueError(f"ducts.area_law must be one of {', '.join(area_law.LAWS)} (got {law!r})")
    fraction = ducts.get("junction_fraction")
    axis = (ducts.get("engine_face") or {}).get("normal")
    if axis is not None and not (isinstance(axis, (list, tuple)) and len(axis) == 3 and any(axis)):
        raise ValueError("ducts.engine_face.normal must be a non-zero [x, y, z]")

    def count(key, default, minimum):
        v = ducts.get(key)
        v = default if v is None else int(v)
        if v < minimum:
            raise ValueError(f"ducts.{key} must be at least {minimum}")
        return v

    def junction_length(key, default):
        v = (ducts.get("junction") or {}).get(key)
        v = default if v is None else float(v)
        if v < 0.0:
            raise ValueError(f"ducts.junction.{key} must not be negative")
        return v

    return DuctsSettings(
        law,
        area_law.DEFAULT_JUNCTION_FRACTION if fraction is None else float(fraction),
        tuple(float(c) for c in axis) if axis is not None else (1.0, 0.0, 0.0),
        count("n_stations_branch", 12, 2),
        count("n_stations_trunk", 12, 2),
        count("n_section_points", interp.DEFAULT_SAMPLES, 8),
        ductcheck.CheckSettings.from_config(cfg),
        junction_length("splitter_length", 0.0),
        junction_length("overlap", dm.DEFAULT_OVERLAP),
    )


def symmetry_from_config(cfg: dict) -> str:
    value = (cfg.get("ducts") or {}).get("symmetry")
    if value is None:
        return DEFAULT_SYMMETRY
    if value not in SYMMETRIES:
        raise ValueError(f"ducts.symmetry must be one of {', '.join(SYMMETRIES)} (got {value!r})")
    return value


def duct_params_from_config(cfg: dict, symmetry: str | None = None) -> DuctParams:
    ducts = cfg.get("ducts") or {}
    symmetry = symmetry or symmetry_from_config(cfg)
    c = ducts.get("centreline") or {}
    if not isinstance(c, dict):
        raise ValueError("ducts.centreline must be a mapping")

    def opt(key):
        v = c.get(key)
        return None if v is None else float(v)

    point = (ducts.get("junction") or {}).get("point")
    junction = junction_y = None
    if point is not None:
        if not (isinstance(point, (list, tuple)) and len(point) == 3):
            raise ValueError("ducts.junction.point must be [x, y, z]")
        junction = (float(point[0]), float(point[2]))
        if symmetry == "none":
            junction_y = float(point[1])  # else the junction lies on y = 0
    fraction = c.get("handle_fraction")
    centreline = CentrelineParams(
        junction, opt("junction_pitch_deg"), junction_y, opt("junction_yaw_deg") if symmetry == "none" else None,
        None, None,  # the inlet handles belong to the inlets
        opt("junction_out_handle"), opt("end_handle"),
        CentrelineParams().handle_fraction if fraction is None else float(fraction),
    )
    wall = ducts.get("wall_thickness")
    lip = ducts.get("lip_extension")
    if lip is None:
        lip = (ducts.get("inlet") or {}).get("lip_extension")  # the old location
    area = areacurve.points_from_config(ducts.get("area_points"))
    params = DuctParams(
        centreline, AreaPoints(area.junction_area, None, area.trunk),
        DEFAULT_WALL_THICKNESS if wall is None else float(wall),
        DEFAULT_LIP_EXTENSION if lip is None else float(lip),
        ducts.get("lip_mode") or dm.DEFAULT_LIP_MODE,
    )
    if params.wall_thickness <= 0.0:
        raise ValueError("ducts.wall_thickness must be positive")
    if params.lip_extension < 0.0:
        raise ValueError("ducts.lip_extension must not be negative")
    if params.lip_mode not in dm.LIP_MODES:
        raise ValueError(f"ducts.lip_mode must be one of {', '.join(dm.LIP_MODES)} (got {params.lip_mode!r})")
    return params


def parse_key_sections(raw, where: str = "ducts.key_sections") -> tuple[KeySection, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where} must be a list")
    sections = []
    for k, entry in enumerate(raw):
        here = f"{where}[{k}]"
        if not isinstance(entry, dict) or entry.get("s") is None:
            raise ValueError(f"{here}: needs an 's' (0 = inlet, between 0 and 1 for the others)")
        s = float(entry["s"])
        if not 0.0 <= s < 1.0:
            raise ValueError(f"{here}: s must be 0 (the inlet) or between 0 and 1, exclusive (the engine face "
                             "at s = 1 is fixed)")
        sections.append(KeySection(s, ol.shape_from_config(entry, here)))
    sections.sort(key=lambda ks: ks.s)
    for a, b in zip(sections, sections[1:]):
        if b.s - a.s < MIN_S_SEPARATION:
            raise ValueError(f"{where}: s = {a.s:g} and {b.s:g} are closer than {MIN_S_SEPARATION:g}")
    return tuple(sections)


def _plane_from(entry: dict, where: str) -> InletPlane | None:
    origin = entry.get("origin")
    if origin is None:
        return None
    if not (isinstance(origin, (list, tuple)) and len(origin) == 3):
        raise ValueError(f"{where}.origin must be [x, y, z]")
    plane = InletPlane(tuple(float(c) for c in origin), float(entry.get("pitch_deg") or 0.0),
                       float(entry.get("sweep_deg") or 0.0))
    plane.frame()  # validates the angles
    return plane


def _inlet_from(entry: dict, k: int, where: str) -> Inlet:
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a mapping")
    name = str(entry.get("name") or f"{DEFAULT_INLET_NAME} {k + 1}")
    plane = _plane_from(entry, where)
    raw = entry.get("key_sections")
    sections = parse_key_sections(raw, f"{where}.key_sections") if raw is not None else ()
    fraction = entry.get("flow_fraction")
    if fraction is not None and float(fraction) <= 0.0:
        raise ValueError(f"{where}.flow_fraction must be positive")
    mirror = entry.get("mirror")
    if mirror is not None and not isinstance(mirror, bool):
        raise ValueError(f"{where}.mirror must be true or false")
    handle = lambda key: None if entry.get(key) is None else float(entry[key])  # noqa: E731
    return Inlet(name, plane, sections, None if fraction is None else float(fraction), bool(mirror),
                 handle("start_handle"), handle("junction_in_handle"),
                 areacurve.parse_points(entry.get("area_points"), f"{where}.area_points"))


def state_from_config(cfg: dict) -> InletState:
    ducts = cfg.get("ducts") or {}
    symmetry = symmetry_from_config(cfg)
    params = duct_params_from_config(cfg, symmetry)
    raw = ducts.get("inlets")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("ducts.inlets must be a non-empty list")
        inlets = tuple(_inlet_from(e, k, f"ducts.inlets[{k}]") for k, e in enumerate(raw))
        names = [i.name for i in inlets]
        if len(set(names)) != len(names):
            raise ValueError("ducts.inlets: the names must be unique")
        return InletState(inlets, symmetry, params)
    # the old single-inlet format: one inlet, mirrored, symmetry xz
    inlet = ducts.get("inlet") or {}
    plane = _plane_from(inlet, "ducts.inlet")
    keys = ducts.get("key_sections")
    sections: tuple[KeySection, ...] = ()
    if plane is not None:
        if keys is not None:
            sections = parse_key_sections(keys)
        elif inlet.get("vertices") is not None:  # the 5-vertex format: a straight-edged inlet section
            data = {"vertices": inlet["vertices"], "snapped": inlet.get("snapped")}
            sections = (KeySection(0.0, ol.shape_from_config(data, "ducts.inlet")),)
    c = ducts.get("centreline") or {}
    area = areacurve.points_from_config(ducts.get("area_points"))
    handle = lambda key: None if c.get(key) is None else float(c[key])  # noqa: E731
    first = Inlet(f"{DEFAULT_INLET_NAME} 1", plane, sections, None, ducts.get("n_inlets") != 1,
                  handle("start_handle"), handle("junction_in_handle"), area.branch)
    return InletState((first,), symmetry, params)


class _Plan:
    """The planning result for a state: the plan-only Duct (None when it cannot be built), where each inlet's
    primary branch is in it, and fallback area curves without a duct."""

    def __init__(self, duct, branch_of: dict, fallback: dict, error: str, s_fallback: float):
        self.duct, self.branch_of, self.fallback, self.error, self.s_fallback = duct, branch_of, fallback, error, s_fallback

    def areas(self, k: int) -> AreaCurves:
        return self.duct.branches[self.branch_of[k]].areas if self.duct is not None else self.fallback[k]

    def centreline(self, k: int):
        return self.duct.branches[self.branch_of[k]].centreline if self.duct is not None else None

    def s_junction(self, k: int) -> float:
        return self.duct.branches[self.branch_of[k]].s_junction if self.duct is not None else self.s_fallback

    def junction_ring(self, k: int):
        if self.duct is None or self.duct.n_branches < 2:
            return None
        return self.duct.branches[self.branch_of[k]].branch_end

    @property
    def n_branches(self) -> int:
        return self.duct.n_branches if self.duct is not None else len(self.fallback)


PLAN_CACHE_SIZE = 16
MIN_AREA_FRACTION = 0.01  # a dragged area control point never goes below this fraction of the face area


class InletModel:
    def __init__(self, scene, cfg: dict, config_path: Path, save_target: Path | None, clock=time.monotonic):
        """``save_target`` is the config Save writes to; None makes the model read-only (example config)."""
        self.scene = scene
        self.config_path = Path(config_path)
        self.save_target = Path(save_target) if save_target is not None else None
        self._clock = clock
        self._listeners: list = []
        self._past: list[InletState] = []
        self._future: list[InletState] = []
        self._last_key: str | None = None
        self._last_time = -1e9
        self._skin_cache: dict = {}
        self._plan_cache: dict = {}
        self._built: dict = {}  # state-keyed cache of the Duct, its bodies and checks
        self._solids: dict = {}
        self._skins: list = []
        self._evals: list = []
        self.mode = MODE_VIEW
        self.live_checks = False  # the window turns this on to show a pending note for the heavy checks
        self.locked = False
        self.message = ""
        self.settings = duct_settings(cfg)
        self.alignment: tuple[str, float, float] | None = None  # (edge label, edge sweep, applied plane sweep)
        self.active = 0  # index of the section the editor shows (of the selected inlet)
        loaded = state_from_config(cfg)
        self.state = self._normalised(loaded)
        # a state completed here (resnapped, default sketch added) differs from the file only by rounding: the
        # file is the saved state unless a sketch had to be created
        created = any(i.placed and not i.sections for i in loaded.inlets)
        self.saved_state = loaded if created else self.state
        self.notice = self._notice_for(cfg)
        self._recompute()

    def _notice_for(self, cfg: dict) -> str:
        notes = [legacy_notice(cfg)] if legacy_notice(cfg) else []
        _, changed = self._fractions(self.state)
        if changed:
            notes.append("ducts.inlets: the flow fractions do not sum to 1 and are normalised")
        return "; ".join(notes)

    # ---- observers ---------------------------------------------------------------------------------
    def add_listener(self, callback) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            callback()

    def _say(self, text: str) -> None:
        self.message = text

    # ---- the inlets ---------------------------------------------------------------------------------------------
    @property
    def selected(self) -> int:
        return min(self.state.selected, len(self.state.inlets) - 1)

    @property
    def inlet(self) -> Inlet:
        return self.state.inlet

    @property
    def symmetry(self) -> str:
        return self.state.symmetry

    def _placed(self, state: InletState | None = None) -> list[int]:
        st = state or self.state
        return [k for k, i in enumerate(st.inlets) if i.placed and i.sections]

    def has_twin(self, k: int, state: InletState | None = None) -> bool:
        st = state or self.state
        i = st.inlets[k]
        return st.symmetry == "xz" and i.mirror and i.placed and not i.self_symmetric

    def _fractions(self, state: InletState) -> tuple[dict, bool]:
        """(flow fraction per physical branch of every placed inlet, whether given values were normalised)."""
        placed = [k for k, i in enumerate(state.inlets) if i.placed]
        if not placed:
            return {}, False
        f, changed = dm.flow_fractions([state.inlets[k].flow_fraction for k in placed],
                                       [self.has_twin(k, state) for k in placed])
        return dict(zip(placed, f)), changed

    def flow_fraction(self, k: int | None = None) -> float | None:
        """The normalised flow fraction of an inlet (per physical branch); None if it is not placed."""
        k = self.selected if k is None else k
        return self._fractions(self.state)[0].get(k)

    @property
    def fractions_normalised(self) -> bool:
        return self._fractions(self.state)[1]

    def total_capture(self) -> float:
        """The total capture area (Phase 1), projected normal to the freestream, all inlets together."""
        sizing = self.scene.sizing
        total = getattr(sizing, "capture_area_total", None)
        return float(total) if total is not None else float(sizing.capture_area_per_inlet * sizing.n_inlets)

    def capture_target(self, k: int, state: InletState | None = None) -> float:
        """The projected capture area of inlet ``k``: its flow fraction of the total."""
        st = state or self.state
        f = self._fractions(st)[0].get(k)
        return self.total_capture() * (f if f is not None else 1.0 / max(len(st.inlets), 1))

    @property
    def area(self) -> float:
        """The selected inlet's capture area (projected normal to the freestream): its inlet section's target."""
        return self.capture_target(self.selected)

    @property
    def n_branches(self) -> int:
        """The number of physical branches (mirrored twins counted)."""
        return max(self._plan().n_branches, 1)

    @property
    def plane(self) -> InletPlane | None:
        return self.state.plane

    @property
    def placed(self) -> bool:
        return self.state.inlet.placed and bool(self.state.inlet.sections)

    @property
    def section(self) -> Section | None:
        """The skin curve in the selected inlet's plane."""
        return self._skins[self.selected] if self.selected < len(self._skins) else None

    @property
    def evaluations(self) -> list[sketch.Opening]:
        """The evaluated key sections of the selected inlet."""
        return self._evals[self.selected] if self.selected < len(self._evals) else []

    @property
    def opening(self) -> sketch.Opening | None:
        """The evaluated inlet section (s = 0) of the selected inlet."""
        ev = self.evaluations
        return ev[0] if ev else None

    @property
    def valid(self) -> bool:
        placed = [e for k, e in enumerate(self._evals) if self.state.inlets[k].placed and e]
        return bool(placed) and all(o.valid for evs in placed for o in evs)

    @property
    def dirty(self) -> bool:
        return self.state != self.saved_state

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    @property
    def can_save(self) -> bool:
        return self.save_target is not None and self.valid

    @property
    def mode_text(self) -> str:
        return MODE_TEXT[self.mode]

    @property
    def opening_centroid(self) -> np.ndarray | None:
        """3D centroid of the selected inlet's opening."""
        if self.opening is None:
            return None
        return self.state.plane.to_3d([self.opening.centroid_uv])[0]

    # ---- the area law -------------------------------------------------------------------------------------
    @property
    def inlet_plane_area(self) -> float:
        """A(0): the inlet section's in-plane area, the capture area divided by |n . X|."""
        plane = self.state.plane
        return self.area / max(plane.projection_factor if plane else 1.0, 1e-6)

    @property
    def face_area(self) -> float:
        """A(1): the duct's end area, the full circle of the face diameter (the hub is the engine's, not the duct's)."""
        return 0.25 * math.pi * self.scene.face_diameter**2

    @property
    def junction_s(self) -> float:
        """Where the branches merge along the selected inlet's path (arc-length fraction)."""
        return self._plan().s_junction(self.selected)

    def area_at(self, s: float, inlet: int | None = None) -> float:
        """A(s) of an inlet's branch (default: the selected one), the in-plane area its key sections are held to."""
        k = self.selected if inlet is None else inlet
        plan = self._plan()
        return float(plan.areas(k).area(s, plan.s_junction(k))[0])

    @property
    def areas(self) -> AreaCurves:
        return self._plan().areas(self.selected)

    @property
    def centreline(self) -> Centreline | None:
        return self._plan().centreline(self.selected)

    def _evaluate_inlet(self, k: int, state: InletState):
        ins = state.inlets[k]
        if not ins.placed or not ins.sections:
            return None
        return sketch.evaluate(ins.sections[0].shape, self._section_for(ins.plane), ins.plane,
                               self.capture_target(k, state))

    def _build_spec(self, st: InletState, ks_outlines: dict, op_areas: dict) -> dm.DuctSpec:
        s = self.settings
        specs = []
        for k in self._placed(st):
            ins = st.inlets[k]
            keys = tuple(dm.KeyOutline(ks.s, o, np.array(ks.shape.vertices))
                         for ks, o in zip(ins.sections, ks_outlines[k]))
            specs.append(dm.InletSpec(ins.name, ins.plane, keys, ins.flow_fraction, ins.mirror, ins.start_handle,
                                      ins.junction_in_handle, ins.area_branch, op_areas[k]))
        return dm.DuctSpec(
            tuple(specs), np.asarray(self.scene.face_center, dtype=float), np.asarray(s.face_axis),
            self.scene.face_diameter, st.duct.centreline, st.duct.area, s.law, st.symmetry, s.n_stations_branch,
            s.n_stations_trunk, s.n_points, st.duct.wall_thickness, st.duct.lip_extension, st.duct.lip_mode,
            s.junction_fraction, s.splitter_length, s.overlap)

    def _plan(self, state: InletState | None = None) -> _Plan:
        """The centrelines, area curves and junction sections for ``state``. They depend on the planes, the key
        section SHAPES (the junction section is derived from them; the sizes do not matter), the flow fractions and
        the duct parameters."""
        st = state or self.state
        placed = self._placed(st)
        face = self.scene.face_center
        if not placed:
            return _Plan(None, {}, {k: AreaCurves(1.0, 1.0, st.duct.area, self.settings.law, 1) for k in range(len(st.inlets))},
                         "", self.settings.junction_fraction)
        key = (st.symmetry, st.duct, tuple(
            (st.inlets[k].name, st.inlets[k].plane, tuple(ks.shape for ks in st.inlets[k].sections),
             tuple(ks.s for ks in st.inlets[k].sections), st.inlets[k].flow_fraction, st.inlets[k].mirror,
             st.inlets[k].start_handle, st.inlets[k].junction_in_handle, st.inlets[k].area_branch) for k in placed),
               self.settings, self.scene.face_diameter, None if face is None else tuple(np.asarray(face).tolist()))
        hit = self._plan_cache.get(key)
        if hit is not None:
            return hit
        ops = {k: self._evaluate_inlet(k, st) for k in placed}
        areas_in = {k: ops[k].area_plane for k in placed}
        duct, branch_of, error = None, {}, ""
        if face is not None:
            outlines = {k: [ops[k].outline] + [ol.build_outline(ol.edge_paths(ks.shape)[0])
                                               for ks in st.inlets[k].sections[1:]] for k in placed}
            try:
                duct = dm.Duct(self._build_spec(st, outlines, areas_in), plan_only=True)
                branch_of = {k: next(i for i, bg in enumerate(duct.branches) if bg.inlet == j and not bg.twin)
                             for j, k in enumerate(placed)}
            except ValueError as exc:
                error = str(exc)
        fallback = {}
        if duct is None:
            fr, _ = self._fractions(st)
            total = sum(areas_in[k] * (2 if self.has_twin(k, st) else 1) for k in placed)
            count = sum(2 if self.has_twin(k, st) else 1 for k in placed)
            for k in placed:
                pts = AreaPoints(st.duct.area.junction_area, st.inlets[k].area_branch, st.duct.area.trunk)
                fallback[k] = AreaCurves(areas_in[k], self.face_area, pts, self.settings.law, count, fr[k], total)
        hit = _Plan(duct, branch_of, fallback, error, self.settings.junction_fraction)
        if len(self._plan_cache) > PLAN_CACHE_SIZE:
            self._plan_cache.clear()
        self._plan_cache[key] = hit
        return hit

    def face_shape(self) -> SectionShape:
        """The engine face (s = 1): a circle of the face diameter, fixed."""
        return ol.circle_shape(0.5 * self.scene.face_diameter)

    # ---- the active section ------------------------------------------------------------------------------------
    @property
    def n_sections(self) -> int:
        return len(self.state.sections)

    @property
    def junction_row(self) -> bool:
        """The fixed junction section (this branch's piece of the trunk start section) is listed when two or more
        branches meet."""
        return self.placed and self._plan().junction_ring(self.selected) is not None

    @property
    def active_is_junction(self) -> bool:
        return self.junction_row and self.active == self.n_sections

    @property
    def active_is_face(self) -> bool:
        return self.placed and self.active == self.n_sections + (1 if self.junction_row else 0)

    @property
    def active_is_fixed(self) -> bool:
        """The junction and the engine face sections are derived / fixed and cannot be edited."""
        return self.placed and self.active >= self.n_sections

    @property
    def active_is_inlet(self) -> bool:
        return self.placed and self.active == 0

    @property
    def active_s(self) -> float | None:
        if not self.placed:
            return None
        if self.active_is_face:
            return 1.0
        if self.active_is_junction:
            return self.junction_s
        return self.state.sections[self.active].s

    def junction_shape(self) -> SectionShape | None:
        """This branch's end section: its piece of the trunk start section (derived, fixed)."""
        ring = self._plan().junction_ring(self.selected)
        return None if ring is None else SectionShape.polygon(ring)

    @property
    def active_shape(self) -> SectionShape | None:
        if not self.placed:
            return None
        if self.active_is_face:
            return self.face_shape()
        if self.active_is_junction:
            return self.junction_shape()
        return self.state.sections[self.active].shape

    @property
    def active_opening(self) -> sketch.Opening | None:
        """The evaluated active section; for the engine face and the junction a plain evaluation of the shape."""
        if not self.placed:
            return None
        if self.active_is_face:
            return sketch.evaluate(self.face_shape(), None, None, self.face_area)
        if self.active_is_junction:
            shape = self.junction_shape()
            return sketch.evaluate(shape, None, None, ol.outline_area(shape))
        return self.evaluations[self.active]

    @property
    def active_curve(self) -> Section | None:
        """The skin curve the active section can snap to (only the inlet section has one)."""
        return self.section if self.active_is_inlet else None

    @property
    def symmetric_editing(self) -> bool:
        """The selected inlet is centred on y = 0: its sections are kept mirror symmetric about u = 0."""
        return self.placed and self.inlet.self_symmetric and not self.active_is_fixed

    def section_list(self) -> list[tuple[float, str, bool]]:
        """(s, label, editable) of every key section plus the fixed junction and engine face sections."""
        rows = []
        for k, ks in enumerate(self.state.sections):
            rows.append((ks.s, "inlet" if k == 0 else "key section", True))
        if self.junction_row:
            rows.append((self.junction_s, "junction: branch end (fixed)", False))
        if self.placed:
            rows.append((1.0, "engine face (fixed)", False))
        return rows

    def select_section(self, index: int) -> bool:
        """Show another section (0 = inlet, then the key sections, the fixed junction, the engine face). Not part
        of the undo history."""
        if not self.placed or not 0 <= index < len(self.section_list()):
            return False
        if index != self.active:
            self.active = index
            self._notify()
        return True

    def _section_for(self, plane: InletPlane) -> Section | None:
        if plane not in self._skin_cache:
            if len(self._skin_cache) > 16:
                self._skin_cache.clear()
            self._skin_cache[plane] = compute_section(self.scene.fuselage, plane)
        return self._skin_cache[plane]

    def _normalised(self, state: InletState) -> InletState:
        """Every placed inlet snapped exactly onto its current section (a default sketch if it has none yet), and
        every other key section held at its area A(s)."""
        inlets = list(state.inlets)
        for k, ins in enumerate(inlets):
            if not ins.placed:
                continue
            section = self._section_for(ins.plane)
            secs = list(ins.sections)
            if not secs or secs[0].s != 0.0:
                secs.insert(0, KeySection(0.0, sketch.default_sketch(section, ins.plane, self.capture_target(k, state))))
            else:
                secs[0] = KeySection(0.0, sketch.resnap(secs[0].shape, section))
            inlets[k] = replace(ins, sections=tuple(secs))
        base = replace(state, inlets=tuple(inlets))
        plan = self._plan(base)
        for k in self._placed(base):
            secs = list(inlets[k].sections)
            for j in range(1, len(secs)):
                target = float(plan.areas(k).area(secs[j].s, plan.s_junction(k))[0])
                secs[j] = KeySection(secs[j].s, sketch.rescale_to_area(secs[j].shape, target))
            inlets[k] = replace(inlets[k], sections=tuple(secs))
        return replace(state, inlets=tuple(inlets))

    def _recompute(self) -> None:
        st = self.state
        plan = self._plan()
        self._skins, self._evals = [], []
        for k, ins in enumerate(st.inlets):
            if not ins.placed or not ins.sections:
                self._skins.append(None)
                self._evals.append([])
                continue
            skin = self._section_for(ins.plane)
            evs = [sketch.evaluate(ins.sections[0].shape, skin, ins.plane, self.capture_target(k, st))]
            for ks in ins.sections[1:]:
                evs.append(sketch.evaluate(ks.shape, None, None,
                                           float(plan.areas(k).area(ks.s, plan.s_junction(k))[0])))
            self._skins.append(skin)
            self._evals.append(evs)
        if not self.placed:
            self.active = 0

    # ---- history ----------------------------------------------------------------------------------------
    def _commit(self, new: InletState, key: str | None = None) -> bool:
        """Apply a change as one undo step; consecutive edits of the same ``key`` are merged."""
        if new == self.state:
            return False
        now = self._clock()
        merge = key is not None and key == self._last_key and now - self._last_time < MERGE_WINDOW_S and self._past
        if not merge:
            self._past.append(self.state)
            del self._past[:-UNDO_LIMIT]
        self._future.clear()
        self._last_key, self._last_time = key, now
        self._set_state(new)
        return True

    def _set_state(self, new: InletState, keep_selection: bool = True) -> None:
        """Make ``new`` the state; the selection stays on its inlet, and the editor on its section (or on a fixed
        one), clamped to what exists."""
        was_fixed = self.active_is_fixed
        offset = self.active - self.n_sections  # 0: the junction row (or the face without one), 1: the face
        selected = self.state.selected if keep_selection else new.selected
        self.state = replace(new, selected=min(max(selected, 0), len(new.inlets) - 1))
        self.active = len(self.state.sections) + offset if was_fixed else min(self.active, max(len(self.state.sections) - 1, 0))
        self._recompute()
        self.active = min(self.active, len(self.section_list()) - 1) if self.placed else 0

    def undo(self) -> bool:
        if not self._past:
            return False
        self._future.append(self.state)
        self._last_key = None
        self._set_state(self._past.pop())
        self._say("Undo")
        self._notify()
        return True

    def redo(self) -> bool:
        if not self._future:
            return False
        self._past.append(self.state)
        self._last_key = None
        self._set_state(self._future.pop())
        self._say("Redo")
        self._notify()
        return True

    def _changed(self, changed: bool, text: str = "") -> bool:
        if changed:
            self._say(text)
            self._notify()
        return changed

    # ---- inlets: select, add, duplicate, remove, rename, mirror, flow, symmetry ------------------------------------------------------
    def _unique_name(self, base: str) -> str:
        names = {i.name for i in self.state.inlets}
        if base not in names:
            return base
        k = 2
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"

    def select_inlet(self, index: int) -> bool:
        """Edit another inlet. Not part of the undo history."""
        if not 0 <= index < len(self.state.inlets):
            return False
        if index != self.selected:
            self.state = replace(self.state, selected=index)
            self.active = 0
            self.alignment = None
            self._notify()
        return True

    def _commit_state(self, new: InletState, key: str | None = None, select: int | None = None) -> bool:
        changed = self._commit(self._normalised(new), key)
        if select is not None:
            self.state = replace(self.state, selected=select)
            self.active = 0
        return changed

    def add_inlet(self, name: str | None = None, place: bool = True) -> int:
        """Add an inlet (not placed yet) and select it; with ``place`` the explorer goes into place mode."""
        name = self._unique_name(name or f"{DEFAULT_INLET_NAME} {len(self.state.inlets) + 1}")
        new = Inlet(name, None, (), None, self.state.symmetry == "xz")
        index = len(self.state.inlets)
        self._commit_state(replace(self.state, inlets=self.state.inlets + (new,)), None, select=index)
        if place and not self.locked:
            self.mode = MODE_PLACING
        self._say(f"Added inlet '{name}'" + (": click on the fuselage to place it" if place else ""))
        self._notify()
        return index

    def duplicate_inlet(self, index: int | None = None) -> int | None:
        """Copy an inlet (same plane and sections) under a new name and select the copy."""
        index = self.selected if index is None else index
        if not 0 <= index < len(self.state.inlets):
            return None
        src = self.state.inlets[index]
        copy = replace(src, name=self._unique_name(f"{src.name} copy"))
        new_index = index + 1
        inlets = self.state.inlets[:new_index] + (copy,) + self.state.inlets[new_index:]
        self._commit_state(replace(self.state, inlets=inlets), None, select=new_index)
        self._say(f"Duplicated '{src.name}' as '{copy.name}'; move the copy")
        self._notify()
        return new_index

    def remove_inlet(self, index: int | None = None) -> bool:
        index = self.selected if index is None else index
        if len(self.state.inlets) <= 1 or not 0 <= index < len(self.state.inlets):
            self._say("The last inlet cannot be removed")
            self._notify()
            return False
        name = self.state.inlets[index].name
        inlets = self.state.inlets[:index] + self.state.inlets[index + 1:]
        keep = self.selected - 1 if self.selected > index or self.selected == len(inlets) else self.selected
        self._commit_state(replace(self.state, inlets=inlets), None, select=max(keep, 0))
        self._say(f"Removed inlet '{name}'")
        self._notify()
        return True

    def rename_inlet(self, index: int, name: str) -> bool:
        name = str(name).strip()
        if not name:
            self._say("An inlet needs a name")
            self._notify()
            return False
        if any(i.name == name for k, i in enumerate(self.state.inlets) if k != index):
            self._say(f"There is already an inlet called '{name}'")
            self._notify()
            return False
        return self._changed(self._commit_state(self.state.with_inlet(index, name=name), f"name{index}"))

    def set_mirror(self, index: int, flag: bool) -> bool:
        """Ask for (or drop) the mirrored twin of an inlet. Only meaningful with symmetry xz."""
        if not 0 <= index < len(self.state.inlets):
            return False
        text = ""
        ins = self.state.inlets[index]
        if flag and ins.self_symmetric:
            text = f"'{ins.name}' is centred on y = 0: it is its own mirror image and gets no twin"
        elif flag and self.state.symmetry != "xz":
            text = "The mirror flag only has an effect with symmetry xz"
        changed = self._commit_state(self.state.with_inlet(index, mirror=bool(flag)))
        self._say(text)
        self._notify()
        return changed

    def set_flow_fraction(self, index: int, value: float | None) -> bool:
        """The flow fraction of an inlet's branch (None = equal share); all fractions are normalised to sum to 1."""
        if not 0 <= index < len(self.state.inlets):
            return False
        if value is not None and value <= 0.0:
            self._say("A flow fraction must be positive")
            self._notify()
            return False
        changed = self._changed(self._commit_state(
            self.state.with_inlet(index, flow_fraction=None if value is None else float(value)), f"flow{index}"))
        if changed and self.fractions_normalised:
            self._say("The flow fractions were normalised to sum to 1")
            self._notify()
        return changed

    def set_symmetry(self, symmetry: str) -> bool:
        if symmetry not in SYMMETRIES:
            raise ValueError(f"symmetry must be one of {', '.join(SYMMETRIES)}")
        return self._changed(self._commit_state(replace(self.state, symmetry=symmetry)), f"Symmetry: {symmetry}")

    # ---- the plane --------------------------------------------------------------------------------------------
    def _with_plane(self, plane: InletPlane, key: str | None) -> bool:
        """Move / rotate the selected inlet's plane: the sketch keeps its (u, v), snapped vertices follow the new
        section and the other key sections are re-held at their (changed) area A(s)."""
        plane.frame()  # validates the angles before anything changes
        if plane == self.state.plane:
            return False
        return self._commit(self._normalised(self.state.with_selected(plane=plane)), key)

    def set_param(self, key: str, value) -> bool:
        """Change the pitch or the sweep (degrees)."""
        if key not in PLANE_KEYS:
            raise KeyError(key)
        if self.state.plane is None:
            raise ValueError("place the inlet first")
        return self._changed(self._with_plane(replace(self.state.plane, **{key: float(value)}), key))

    def _origin_y(self, y: float) -> float:
        """A mirrored inlet is kept on the right (y >= 0): its twin is the left one."""
        return abs(y) if self.state.symmetry == "xz" and self.inlet.mirror else y

    def set_origin(self, x: float, y: float, z: float) -> bool:
        """Move the plane origin. Blocked while locked."""
        if self.locked:
            self._say("Inlet is locked")
            self._notify()
            return False
        old = self.state.plane
        plane = InletPlane((float(x), self._origin_y(float(y)), float(z)), old.pitch_deg if old else 0.0,
                           old.sweep_deg if old else 0.0)
        return self._changed(self._with_plane(plane, "origin"))

    # ---- editing the active section ------------------------------------------------------------------------------
    def _replace_active(self, shape: SectionShape) -> InletState:
        secs = list(self.state.sections)
        secs[self.active] = KeySection(secs[self.active].s, shape)
        return self.state.with_selected(sections=tuple(secs))

    def _edit(self, fn, key: str | None = None, rescale: bool = True) -> bool:
        """Apply ``fn(shape) -> shape`` to the active section as one undo step. Every other section is brought back
        to its area A(s) in the same step (``rescale`` False defers that, for the events of a drag)."""
        if not self.placed or self.active_is_fixed:
            self._say("This section is fixed" if self.placed else "Place the inlet first")
            self._notify()
            return False
        try:
            shape = fn(self.state.sections[self.active].shape)
        except ValueError as exc:
            self._say(str(exc))
            self._notify()
            return False
        new = self._replace_active(shape)
        if rescale:  # the junction depends on the key section shapes: hold every section at its new A(s)
            new = self._normalised(new)
        return self._changed(self._commit(new, key))

    def finish_edit(self) -> bool:
        """End of a drag: hold the sections at their area A(s) (merged into the drag's undo step)."""
        if not self.placed or self.active_is_fixed:
            return False
        return self._changed(self._commit(self._normalised(self.state), self._last_key))

    def drag_vertex(self, index: int, uv, snap_tolerance: float | None) -> bool:
        """Move vertex ``index`` of the active section to ``uv``. The inlet vertex snaps onto the skin curve
        within ``snap_tolerance`` metres (None disables snapping) and unsnaps when dropped further away. On a
        self-symmetric inlet the mirror partner moves with it."""
        if not self.placed or self.active_is_fixed:
            return False

        def move(shape):
            point, flag = ((float(uv[0]), float(uv[1])), False)
            if self.active_is_inlet:
                point, flag = sketch.snap_point(self.section, uv, snap_tolerance)
            if self.symmetric_editing:
                return ol.move_vertex_symmetric(shape, index, point, flag)
            moved = ol.move_vertex(shape, index, point)
            snapped = list(moved.snapped)
            snapped[index] = flag
            return moved.with_(snapped=tuple(snapped))

        return self._edit(move, f"vertex{index}@{self.active}", rescale=False)

    def drag_handle(self, edge: int, which: int, uv) -> bool:
        """Drag a Bézier handle of a free curved edge of the active section."""
        fn = ol.move_handle_symmetric if self.symmetric_editing else ol.move_handle
        return self._edit(lambda sh: fn(sh, edge, which, uv), f"handle{edge}.{which}@{self.active}", rescale=False)

    def insert_vertex(self, edge: int, uv) -> int | None:
        """Insert a vertex on ``edge`` at the point of the edge nearest ``uv``. Returns its index, or None."""
        made = []

        def insert(shape):
            point = uv
            if self.active_is_inlet and shape.edge_is_snapped(edge):  # a vertex on the skin edge stays on the skin
                point = tuple(sketch.snap_point(self.section, uv, float("inf"))[0])
            fn = ol.insert_vertex_symmetric if self.symmetric_editing else ol.insert_vertex
            new, at = fn(shape, edge, point)
            made.append(at)
            return new

        return made[0] if self._edit(insert) and made else None

    def delete_vertex(self, index: int) -> bool:
        fn = ol.delete_vertex_symmetric if self.symmetric_editing else ol.delete_vertex
        return self._edit(lambda sh: fn(sh, index))

    def set_edge_type(self, edge: int, kind: str) -> bool:
        fn = ol.set_edge_type_symmetric if self.symmetric_editing else ol.set_edge_type
        return self._edit(lambda sh: fn(sh, edge, kind))

    def set_continuity(self, index: int, mode: str) -> bool:
        fn = ol.set_continuity_symmetric if self.symmetric_editing else ol.set_continuity
        return self._edit(lambda sh: fn(sh, index, mode))

    def fit_area(self) -> bool:
        """Active inlet section: scale only the free vertices so the projected area matches the capture area.
        Another key section: rescale uniformly to A(s)."""
        if not self.placed or self.active_is_fixed:
            return False
        if self.active > 0:
            self._edit(lambda sh: sh)  # already holds A(s) after every edit; nothing to fit
            self.finish_edit()
            self._say(f"Section s = {self.state.sections[self.active].s:g} is held at A(s) = "
                      f"{self.area_at(self.state.sections[self.active].s):.5f} m^2")
            self._notify()
            return True
        shape, ok, message = sketch.fit_area(self.state.sections[0].shape, self.section, self.state.plane, self.area)
        self._commit(self._normalised(self._replace_active(shape)))
        self._say(message)
        self._notify()
        return ok

    # ---- key sections ------------------------------------------------------------------------------------------------
    def add_key_section(self, s: float) -> bool:
        """Add an intermediate key section at ``s`` (between 0 and 1, exclusive). It starts as a copy of the shape
        of the section before it, centred and held at A(s), and becomes the active one. Sections beyond the
        junction shape the trunk: they are only taken from the first inlet."""
        if not self.placed:
            self._say("Place the inlet first")
            self._notify()
            return False
        s = float(s)
        if not 0.0 < s < 1.0:
            self._say("A key section needs s between 0 and 1 (0 is the inlet, 1 the fixed engine face)")
            self._notify()
            return False
        existing = [ks.s for ks in self.state.sections] + [1.0]
        if any(abs(s - e) < MIN_S_SEPARATION for e in existing):
            self._say(f"There is already a section within {MIN_S_SEPARATION:g} of s = {s:g}")
            self._notify()
            return False
        if self.selected != 0 and s >= self.junction_s - MIN_S_SEPARATION and self.junction_row:
            self._say("Sections beyond the junction shape the trunk and are taken from the first inlet")
            self._notify()
            return False
        before = max((ks for ks in self.state.sections if ks.s < s), key=lambda ks: ks.s)
        shape = before.shape.with_(snapped=(False,) * before.shape.n)
        shape = ol.translate(shape, -ol.outline_centroid(before.shape, self.section if before.s == 0.0 else None))
        shape = sketch.rescale_to_area(shape, self.area_at(s))
        secs = sorted(list(self.state.sections) + [KeySection(s, shape)], key=lambda ks: ks.s)
        index = [ks.s for ks in secs].index(s)
        self._commit(self._normalised(self.state.with_selected(sections=tuple(secs))))
        self.active = index
        self._say(f"Added a key section at s = {s:g}")
        self._notify()
        return True

    def remove_key_section(self, index: int) -> bool:
        """Remove an intermediate key section (the inlet and the engine face cannot be removed)."""
        if not self.placed or not 0 < index < self.n_sections:
            self._say("Only the intermediate key sections can be removed")
            self._notify()
            return False
        secs = [ks for k, ks in enumerate(self.state.sections) if k != index]
        self._commit(self._normalised(self.state.with_selected(sections=tuple(secs))))
        self.active = min(self.active, index - 1) if self.active >= index else self.active
        self._say("Removed the key section")
        self._notify()
        return True

    # ---- centreline, area law and wall: the duct parameters ---------------------------------------------------------------------
    def _apply(self, new: InletState, key: str | None = None, text: str = "") -> bool:
        """Commit a changed state (the sections are held at the new area law in the same step)."""
        if not self.placed:
            self._say("Place the inlet first")
            self._notify()
            return False
        return self._changed(self._commit(self._normalised(new), key), text)

    def _set_duct(self, duct: DuctParams, key: str | None, text: str = "") -> bool:
        return self._apply(replace(self.state, duct=duct), key, text)

    CENTRELINE_KEYS = ("junction", "junction_pitch_deg", "junction_y", "junction_yaw_deg", "start_handle",
                       "junction_in_handle", "junction_out_handle", "end_handle")
    INLET_HANDLE_KEYS = ("start_handle", "junction_in_handle")  # belong to the selected inlet; the rest is shared

    def set_centreline(self, key: str | None = "centreline", **changes) -> bool:
        """Change centreline parameters (junction, junction_pitch_deg, junction_y, junction_yaw_deg, start_handle,
        junction_in_handle, junction_out_handle, end_handle); ``None`` returns a value to its default. The inlet
        handles belong to the selected inlet, the rest is shared. With symmetry xz the junction is held on y = 0."""
        for name, value in changes.items():
            if name not in self.CENTRELINE_KEYS:
                raise KeyError(name)
            if name.endswith("_handle") and value is not None and value < 0.0:
                self._say("A handle length cannot be negative")
                self._notify()
                return False
        own = {k: changes.pop(k) for k in self.INLET_HANDLE_KEYS if k in changes}
        if self.state.symmetry == "xz":
            changes.pop("junction_y", None)
            changes.pop("junction_yaw_deg", None)
        state = replace(self.state, duct=replace(self.state.duct, centreline=self.state.duct.centreline.with_(**changes)))
        if own:
            state = state.with_selected(**own)
        return self._apply(state, key)

    def reset_centreline(self) -> bool:
        """Handles and tangent back to their defaults (1/3 of the chord, aimed at the face); the junction stays."""
        old = self.state.duct.centreline
        cl = CentrelineParams(old.junction, junction_y=old.junction_y, handle_fraction=old.handle_fraction)
        state = replace(self.state, duct=replace(self.state.duct, centreline=cl)).with_selected(
            start_handle=None, junction_in_handle=None)
        return self._apply(state, None, "Centreline reset")

    def drag_junction(self, x: float, z: float, y: float | None = None) -> bool:
        if self.locked:
            self._say("Inlet is locked")
            self._notify()
            return False
        changes = {"junction": (float(x), float(z))}
        if y is not None and self.state.symmetry != "xz":
            changes["junction_y"] = float(y)
        return self.set_centreline("junction", **changes)

    def set_area_points(self, points: AreaPoints, key: str | None = "area") -> bool:
        """The junction / trunk points are shared, the branch points belong to the selected inlet."""
        try:
            AreaCurves(self.areas.a_in, self.face_area, points, self.settings.law, 1)
        except ValueError as exc:
            self._say(str(exc))
            self._notify()
            return False
        state = replace(self.state, duct=replace(self.state.duct, area=AreaPoints(points.junction_area, None, points.trunk)))
        return self._apply(state.with_selected(area_branch=points.branch), key)

    def _interior(self, curve: str) -> list:
        return list(self.areas.branch_interior if curve == "branch" else self.areas.trunk_interior)

    def area_move_point(self, curve: str, index: int, s: float, area: float) -> bool:
        """Drag interior control point ``index`` of ``curve`` ("branch" or "trunk"), kept between its neighbours."""
        interior = self._interior(curve)
        gap = areacurve.MIN_S_GAP
        lo = interior[index - 1][0] + gap if index > 0 else gap
        hi = interior[index + 1][0] - gap if index + 1 < len(interior) else 1.0 - gap
        interior[index] = (min(max(float(s), lo), hi), max(float(area), MIN_AREA_FRACTION * self.face_area))
        return self.set_area_points(self.areas.with_interior(curve, interior), f"area:{curve}:{index}")

    def area_add_point(self, curve: str, s: float, area: float | None = None) -> bool:
        interior = self._interior(curve)
        fn = self.areas.branch_area if curve == "branch" else self.areas.trunk_area
        gap = areacurve.MIN_S_GAP
        if not gap <= s <= 1.0 - gap or any(abs(s - p[0]) < gap for p in interior):
            self._say("Not added: too close to another control point or to an end")
            self._notify()
            return False
        interior.append((float(s), float(fn(s)) if area is None else float(area)))
        return self.set_area_points(self.areas.with_interior(curve, interior), None)

    def area_remove_point(self, curve: str, index: int) -> bool:
        interior = self._interior(curve)
        if not 0 <= index < len(interior):
            return False
        del interior[index]
        return self.set_area_points(self.areas.with_interior(curve, interior), None)

    def area_set_junction(self, area: float) -> bool:
        floor = MIN_AREA_FRACTION * self.face_area
        return self.set_area_points(self.areas.with_junction_area(max(float(area), floor)), "area:junction")

    def reset_area(self) -> bool:
        state = replace(self.state, duct=replace(self.state.duct, area=AreaPoints())).with_selected(area_branch=None)
        return self._apply(state, None)

    def set_wall_thickness(self, value: float) -> bool:
        if value <= 0.0:
            self._say("The wall thickness must be positive")
            self._notify()
            return False
        return self._set_duct(replace(self.state.duct, wall_thickness=float(value)), "wall_thickness")

    def set_lip_extension(self, value: float) -> bool:
        if value < 0.0:
            self._say("The lip extension cannot be negative")
            self._notify()
            return False
        return self._set_duct(replace(self.state.duct, lip_extension=float(value)), "lip_extension")

    # ---- bodies and checks (heavy: built on demand and cached per state) -----------------------------------------------------------
    @property
    def can_build(self) -> bool:
        return bool(self._placed()) and self.scene.face_center is not None and self.valid

    def duct_spec(self) -> dm.DuctSpec | None:
        if not self.can_build:
            return None
        placed = self._placed()
        outlines = {k: [ev.outline for ev in self._evals[k]] for k in placed}
        return self._build_spec(self.state, outlines, {k: self._evals[k][0].area_plane for k in placed})

    @property
    def duct(self) -> dm.Duct | None:
        """The stations (cheap): built for the current state on first use."""
        if self._built.get("state") != self.state:
            self._built = {"state": self.state}
        if "duct" not in self._built:
            spec = self.duct_spec()
            try:
                self._built["duct"] = dm.Duct(spec) if spec is not None else None
                self._built["error"] = ""
            except ValueError as exc:
                self._built["duct"], self._built["error"] = None, str(exc)
        return self._built["duct"]

    @property
    def duct_error(self) -> str:
        _ = self.duct
        return self._built.get("error", "") or (self._plan().error if self._placed() else "")

    def branch_index(self, k: int | None = None) -> int | None:
        """Index in ``duct.branches`` of an inlet's primary branch (default: the selected inlet)."""
        k = self.selected if k is None else k
        return self._plan().branch_of.get(k)

    def _solid(self, key, make):
        if key not in self._solids:
            self._solids[key] = make()
        return self._solids[key]

    def fuselage_solid(self) -> Solid:
        return self._solid("fuselage", lambda: Solid(self.scene.fuselage, "fuselage"))

    def other_solids(self) -> list[Solid]:
        parts = getattr(self.scene, "context_parts", None) or []
        return [self._solid(("part", k), lambda m=m, n=n: Solid(m, n)) for k, (n, m) in enumerate(parts)]

    @property
    def checks_current(self) -> bool:
        return self._built.get("state") == self.state and "checks" in self._built

    def ensure_built(self) -> bool:
        """Build the bodies and run the checks for the current state (cached). False when there is no duct."""
        duct = self.duct
        if duct is None:
            return False
        if "checks" not in self._built:
            checks, bodies = ductcheck.run_checks(duct, self.fuselage_solid(), self.other_solids(),
                                                  self.settings.checks)
            self._built["checks"], self._built["bodies"] = checks, bodies
        return True

    @property
    def bodies(self) -> dict | None:
        return self._built["bodies"] if self.ensure_built() else None

    @property
    def checks(self) -> "ductcheck.DuctChecks | None":
        return self._built["checks"] if self.ensure_built() else None

    # ---- aligning the plane to an edge ---------------------------------------------------------------------------------
    def wing_edge(self, wing_index: int, edge: str) -> vspio.EdgeSweep:
        """The planform edge sweep of a wing in the section that spans the inlet's y."""
        return vspio.edge_sweep_at(self.scene.wings[wing_index], self.state.plane.origin[1], edge)

    def align_to_edge(self, edge_sweep_deg: float, label: str) -> bool:
        """Set the sweep so the opening's planform trace (plane ∩ horizontal plane through O) is parallel to an
        edge swept back by ``edge_sweep_deg``; the right inlet's outer edge then trails aft.

        Refused when the resulting |n.X| would be below MIN_ALIGN_PROJECTION. The alignment is not persistent:
        editing the sweep afterwards shows "custom".
        """
        if self.state.plane is None:
            return False
        sweep = sweep_for_edge_angle(edge_sweep_deg)
        candidate = replace(self.state.plane, sweep_deg=sweep)
        factor = candidate.projection_factor
        if factor < MIN_ALIGN_PROJECTION:
            self._say(f"Not aligned: |n.X| would be {factor:.2f} (< {MIN_ALIGN_PROJECTION:g}), the opening would be "
                      f"more than {math.degrees(math.acos(MIN_ALIGN_PROJECTION)):.0f} deg off the flow")
            self._notify()
            return False
        self._with_plane(candidate, None)
        self.alignment = (label, float(edge_sweep_deg), sweep)
        self._say(f"Aligned to {label} (Λ = {edge_sweep_deg:.1f}°): sweep {sweep:.2f} deg")
        self._notify()
        return True

    @property
    def alignment_text(self) -> str:
        """"aligned to: <edge> (Λ = xx.x°)" while the sweep is still the aligned one, else "custom"."""
        a = self.alignment
        if a is not None and self.state.plane is not None and abs(self.state.plane.sweep_deg - a[2]) < 1e-9:
            return f"aligned to: {a[0]} (Λ = {a[1]:.1f}°)"
        return "custom"

    # ---- place mode -----------------------------------------------------------------------------------------------
    def begin_place(self) -> bool:
        if self.locked:
            self._say("Inlet is locked: unlock it to place")
            self._notify()
            return False
        self.mode = MODE_PLACING
        self._say("")
        self._notify()
        return True

    def cancel_place(self) -> bool:
        if self.mode != MODE_PLACING:
            return False
        self.mode = MODE_VIEW
        self._say("Placement cancelled")
        self._notify()
        return True

    def toggle_place(self) -> bool:
        return self.cancel_place() if self.mode == MODE_PLACING else self.begin_place()

    def handle_click(self, point) -> bool:
        """A click on the fuselage. Ignored unless in place mode; places the selected inlet's plane origin there
        (creating the default sketch the first time) and leaves place mode."""
        if self.mode != MODE_PLACING:
            return False
        p = np.asarray(point, dtype=float)
        old = self.state.plane
        plane = InletPlane((float(p[0]), self._origin_y(float(p[1])), float(p[2])), old.pitch_deg if old else 0.0,
                           old.sweep_deg if old else 0.0)
        self._with_plane(plane, None)
        self.mode = MODE_VIEW
        self._say("Inlet placed")
        self._notify()
        return True

    def set_locked(self, locked: bool) -> None:
        self.locked = bool(locked)
        if self.locked and self.mode == MODE_PLACING:
            self.mode = MODE_VIEW
        self._say("Inlet locked" if self.locked else "")
        self._notify()

    # ---- save / revert -----------------------------------------------------------------------------------------------
    def inlets_definition(self) -> list[dict]:
        """What Save writes to ``ducts.inlets``: every placed inlet with its key sections (the inlet section first).
        Flow fractions are written normalised when they were given."""
        fractions, _ = self._fractions(self.state)
        out = []
        for k in self._placed():
            ins = self.state.inlets[k]
            out.append({
                "name": ins.name,
                "origin": [float(c) for c in ins.plane.origin],
                "pitch_deg": float(ins.plane.pitch_deg),
                "sweep_deg": float(ins.plane.sweep_deg),
                "flow_fraction": None if ins.flow_fraction is None else float(fractions[k]),
                "mirror": bool(ins.mirror),
                "start_handle": ins.start_handle,
                "junction_in_handle": ins.junction_in_handle,
                "area_points": None if ins.area_branch is None else [[float(s), float(a)] for s, a in ins.area_branch],
                "key_sections": [{"s": float(ks.s), **ol.shape_to_config(ks.shape)} for ks in ins.sections],
            })
        return out

    def ducts_definition(self) -> dict:
        """What Save writes into ``ducts``: symmetry, the inlets, junction, centreline handles, area points, wall
        thickness and lip."""
        d, cl = self.state.duct, self.state.duct.centreline
        out = {
            "symmetry": self.state.symmetry,
            "inlets": self.inlets_definition(),
            "wall_thickness": float(d.wall_thickness),
            "lip_extension": float(d.lip_extension),
            "lip_mode": d.lip_mode,
            "centreline": {
                "junction_pitch_deg": cl.junction_pitch_deg, "junction_yaw_deg": cl.junction_yaw_deg,
                "junction_out_handle": cl.junction_out_handle, "end_handle": cl.end_handle,
                "handle_fraction": float(cl.handle_fraction),
            },
            "area_points": {"junction_area": None if d.area.junction_area is None else float(d.area.junction_area),
                            "trunk": None if d.area.trunk is None else [[float(s), float(a)] for s, a in d.area.trunk]},
        }
        if cl.junction is not None:
            y = 0.0 if self.state.symmetry == "xz" or cl.junction_y is None else float(cl.junction_y)
            out["junction"] = {"point": [float(cl.junction[0]), y, float(cl.junction[1])]}
        return out

    def save(self) -> bool:
        if self.save_target is None:
            self._say(f"Not saved: the example config is read-only. Copy it to {PRIVATE_CONFIG.name}.")
        elif not self._placed():
            self._say("Nothing to save: place an inlet first.")
        elif not self.valid:
            first = next(e for evs in self._evals for ev in evs for e in ev.errors)
            self._say(f"ERROR: not saved: {first}")
        else:
            try:
                path = save_inlet_definition(None, self.save_target, None, remove_ducts_keys=("inlet", "key_sections"),
                                             ducts_values=self.ducts_definition())
            except OSError as exc:
                self._say(f"Save failed: {exc}")
            else:
                self.saved_state = self.state
                self.notice = ""  # the old keys were migrated
                self._say(f"Saved inlet to {path.name}.")
                self._notify()
                return True
        self._notify()
        return False

    def revert(self) -> bool:
        """Go back to the design stored in the config file (undoable)."""
        try:
            cfg, _ = load_config(self.config_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineSizingWarning)
                sizing = size_from_config(cfg)
            stored = state_from_config(cfg)
            settings = duct_settings(cfg)
        except Exception as exc:  # a broken file must not crash the window; say what is wrong
            self._say(f"Revert failed: {type(exc).__name__}: {exc}".splitlines()[0])
            self._notify()
            return False
        self.scene.sizing = sizing
        self.settings = settings
        self.alignment = None
        self._skin_cache.clear()  # the capture area (hence the default sketch) may have changed
        self._plan_cache.clear()
        self._built = {}
        stored = replace(stored, selected=min(self.state.selected, len(stored.inlets) - 1))
        normalised = self._normalised(stored)
        self._commit(normalised)
        self._recompute()
        created = any(i.placed and not i.sections for i in stored.inlets)
        self.saved_state = stored if created else normalised
        self.notice = self._notice_for(cfg)
        self.mode = MODE_VIEW
        self._say(f"Reverted to {self.config_path.name}.")
        self._notify()
        return True

    # ---- readouts -------------------------------------------------------------------------------------------------------
    def inlet_readouts(self) -> list[Readout]:
        """The inlets list: how many, which are mirrored, the flow fractions."""
        st = self.state
        fractions, changed = self._fractions(st)
        twins = sum(1 for k in range(len(st.inlets)) if self.has_twin(k))
        count = len(st.inlets)
        out = [Readout("Inlets", f"{count} inlet(s), {len([i for i in st.inlets if i.placed]) + twins} branch(es), "
                       f"symmetry {st.symmetry}")]
        if count > 1 or twins:
            for k, ins in enumerate(st.inlets):
                f = fractions.get(k)
                tag = ("self-symmetric" if ins.self_symmetric else "+ mirrored twin" if self.has_twin(k) else "")
                mark = ">" if k == self.selected else " "
                out.append(Readout(f"{mark} {ins.name}"[:20], "not placed" if not ins.placed else
                                   f"flow {100 * f:.1f} %  {tag}".rstrip()))
        if changed:
            out.append(Readout("", "WARN the flow fractions do not sum to 1: normalised", "warn"))
        return out

    def duct_readouts(self) -> list[Readout]:
        """The duct lines: cheap numbers always, the heavy checks once they have been computed for this state."""
        out: list[Readout] = []
        if self.scene.face_center is None:
            return out
        if self.duct_error:
            return [Readout("Duct", f"ERROR {self.duct_error}", "error")]
        duct = self.duct
        if duct is None:
            return out
        k = self.branch_index() or 0
        bg = duct.branches[k]
        cl, areas = bg.centreline, bg.areas
        label = "Junction" if duct.n_branches > 1 else "Via point"
        out += [
            Readout(label, f"x {cl.junction[0]:.4f}  y {cl.junction[1]:.4f}  z {cl.junction[2]:.4f} m  "
                    f"(s = {cl.s_junction:.3f})"),
            Readout("Path length", f"branch {cl.length_branch:.4f} m + trunk {cl.length_trunk:.4f} m"),
            Readout("Area junction", f"{areas.junction_area:.5f} m^2, this branch {areas.share:.3f} of it = "
                    f"{areas.junction_area * areas.share:.5f}"),
        ]
        checks = self._built["checks"] if self.checks_current else None
        if checks is None:
            cheap = ductcheck.DuctChecks()
            ductcheck.numbers(duct, cheap, self.settings.checks)
            findings = cheap.findings
        else:
            findings = checks.findings
        out += [Readout(label, value, level) for label, value, level in findings]
        if checks is None and self.live_checks:
            out.append(Readout("Body checks", "updating...", "info"))
        return out

    def readouts(self) -> list[Readout]:
        """Lines for the read-only panel; ``level`` drives the colour (PASS green, WARN amber, ERROR red)."""
        out: list[Readout] = []
        check = self.scene.engine_check
        # the engine check has its own line; keep it out of the generic warning list
        notes = [w for w in self.scene.warnings if check is None or w != f"WARN {check.message}"]
        if check is not None:
            out.append(Readout("Face / envelope",
                               f"{check.face_diameter:.4f} / {check.envelope_diameter:.4f} m = {check.ratio:.3f} "
                               f"[{check.status}: {check.ratio_range[0]:g}-{check.ratio_range[1]:g}]",
                               "pass" if check.status == "PASS" else "warn"))
        tail = [Readout("", w, "warn") for w in notes]
        if self.notice:
            tail.append(Readout("", self.notice, "warn"))
        out += self.inlet_readouts()
        if self.opening is None:
            out.append(Readout("Inlet", "not placed - use Place inlet (P)"))
            return out + tail

        op, plane = self.opening, self.state.plane
        x, y, z = plane.origin
        within = abs(op.delta) <= sketch.AREA_TOLERANCE
        out += [
            Readout("Origin x y z", f"{x:.4f} {y:.4f} {z:.4f} m"),
            Readout("Pitch / sweep", f"{plane.pitch_deg:.2f} / {plane.sweep_deg:.2f} deg"),
            Readout("Sweep source", self.alignment_text),
            Readout("|n.X|", f"{op.projection_factor:.3f}",
                    "warn" if op.projection_factor < sketch.MIN_PROJECTION else "info"),
            Readout("Area in plane", f"{op.area_plane:.5f} m^2"),
            Readout("Area projected", f"{op.area_projected:.5f} m^2 ({100 * op.delta:+.2f} %)",
                    "pass" if within else "warn"),
            Readout("Area target", f"{op.target_area:.5f} m^2"),
            Readout("Snapped edge length", f"{op.snapped_length:.4f} m"),
            Readout("Free-edge clearance",
                    "n/a" if op.clearance is None else f"{op.clearance:.4f} m",
                    "info" if op.clearance is None else ("pass" if op.clearance >= 0 else "error")),
        ]
        face = self.scene.face_center
        if face is None:
            out.append(Readout("Engine face", "unknown: no distance / offset readout", "warn"))
        else:
            r = inlet_readout(self.opening_centroid, face, op.outline, op.area_projected)
            out += [
                Readout("Aspect ratio (w/h)", f"{r.aspect_ratio:.3f}"),
                Readout("Distance to face", f"{r.distance_to_face:.4f} m"),
                Readout("Offset dy, dz", f"{r.dy:+.4f}, {r.dz:+.4f} m"),
                Readout("Offset / length",
                        f"{r.offset_over_length:.3f}" if r.length > 0 else "n/a (face not aft)",
                        "info" if r.length > 0 else "warn"),
            ]
        # the intermediate key sections and the face, each at its area law value
        for ks, ev in list(zip(self.state.sections, self.evaluations))[1:]:
            out.append(Readout(f"Section s={ks.s:.3f}", f"{ev.area_plane:.5f} m^2 (A(s) {ev.target_area:.5f}, "
                               f"{100 * ev.delta:+.2f} %), {ks.shape.n} vertices",
                               "pass" if ev.valid and abs(ev.delta) <= sketch.AREA_TOLERANCE else "error"))
        out.append(Readout("Section s=1.000", f"engine face circle, {self.face_area:.5f} m^2 (fixed)"))
        for k, evs in enumerate(self._evals):
            name = self.state.inlets[k].name
            for ks, ev in zip(self.state.inlets[k].sections, evs):
                out += [Readout("", f"ERROR: {name}, s = {ks.s:g}: {e}", "error") for e in ev.errors]
        out += [Readout("", f"WARN {w}", "warn") for w in op.warnings]
        return out + self.duct_readouts() + tail
