# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""State of the section editor, independent of any GUI toolkit.

Holds the inlet plane and the *key sections* along a branch: the inlet section (s = 0, in the inlet plane, with
skin snapping) and any intermediate sections at chosen s; the engine face (s = 1) is a fixed circle. One of them is
*active*, the one the editor shows. Every key section has its own shape (any number of vertices, straight or curved
edges) but its size follows the area law A(s). One undo / redo history is shared by the 3D view, the panel and the
editor; there are also the place-mode state machine, the lock, the dirty flag and save / revert. The Qt window and
the headless screenshot both drive this class, so everything that matters is testable without a display.
"""
import math
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ductmgr import area_law, sketch, vspio
from ductmgr import outline as ol
from ductmgr.config import PRIVATE_CONFIG, load_config, save_inlet_definition
from ductmgr.engine import EngineSizingWarning, size_from_config
from ductmgr.geometry import InletPlane, inlet_readout, sweep_for_edge_angle
from ductmgr.outline import SectionShape
from ductmgr.section import Section, compute_section

MIN_ALIGN_PROJECTION = 0.2  # |n.X| below this blocks an edge alignment
MIN_S_SEPARATION = 0.01  # key sections closer than this in s are refused
UNDO_LIMIT = 200  # history entries kept (the requirement is at least 50)
MERGE_WINDOW_S = 0.8  # consecutive edits of one parameter / vertex within this window share one undo step
PLANE_KEYS = ("pitch_deg", "sweep_deg")

MODE_VIEW, MODE_PLACING = "view", "placing"
MODE_TEXT = {MODE_VIEW: "View", MODE_PLACING: "Placing inlet - click on fuselage, Esc to cancel"}

LEGACY_INLET_KEYS = ("shape", "skin_fit", "attach_point", "centroid", "normal", "tilt_deg", "station_x", "polygon")


@dataclass(frozen=True)
class KeySection:
    s: float  # normalised arc length along the branch: 0 inlet, 1 engine face
    shape: SectionShape


@dataclass(frozen=True)
class InletState:
    """Everything the undo history tracks."""

    plane: InletPlane | None  # None until an inlet has been placed
    sections: tuple[KeySection, ...]  # sorted by s; sections[0] is the inlet (s = 0) once placed


@dataclass(frozen=True)
class Readout:
    label: str
    value: str
    level: str = "info"  # "info", "pass", "warn" or "error"


@dataclass(frozen=True)
class DuctsSettings:
    """The ``ducts:`` values the area law needs."""

    law: str = area_law.DEFAULT_LAW
    junction_point: tuple[float, float, float] | None = None
    junction_fraction: float = area_law.DEFAULT_JUNCTION_FRACTION


# ---- config <-> state -----------------------------------------------------------------------------------------
def legacy_notice(cfg: dict) -> str:
    """One line naming old ducts.inlet keys / formats that are ignored or migrated; empty when there are none."""
    ducts = cfg.get("ducts") or {}
    inlet = ducts.get("inlet") or {}
    notes = []
    old = [k for k in LEGACY_INLET_KEYS if inlet.get(k) is not None]
    if old:
        notes.append(f"ignoring old key(s) {', '.join(old)}")
    if inlet.get("vertices") is not None:
        if ducts.get("key_sections") is None:
            notes.append("the 5-vertex ducts.inlet.vertices / snapped format is loaded as a straight-edged inlet "
                         "section; Save writes ducts.key_sections")
        else:
            notes.append("ducts.inlet.vertices / snapped are superseded by ducts.key_sections and ignored")
    return "ducts.inlet: " + "; ".join(notes) if notes else ""


def duct_settings(cfg: dict) -> DuctsSettings:
    ducts = cfg.get("ducts") or {}
    point = (ducts.get("junction") or {}).get("point")
    law = ducts.get("area_law") or area_law.DEFAULT_LAW
    if law not in area_law.LAWS:
        raise ValueError(f"ducts.area_law must be one of {', '.join(area_law.LAWS)} (got {law!r})")
    fraction = ducts.get("junction_fraction")
    return DuctsSettings(
        law,
        tuple(float(c) for c in point) if point is not None else None,
        area_law.DEFAULT_JUNCTION_FRACTION if fraction is None else float(fraction),
    )


def parse_key_sections(raw) -> tuple[KeySection, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("ducts.key_sections must be a list")
    sections = []
    for k, entry in enumerate(raw):
        where = f"ducts.key_sections[{k}]"
        if not isinstance(entry, dict) or entry.get("s") is None:
            raise ValueError(f"{where}: needs an 's' (0 = inlet, between 0 and 1 for the others)")
        s = float(entry["s"])
        if not 0.0 <= s < 1.0:
            raise ValueError(f"{where}: s must be 0 (the inlet) or between 0 and 1, exclusive (the engine face "
                             "at s = 1 is fixed)")
        sections.append(KeySection(s, ol.shape_from_config(entry, where)))
    sections.sort(key=lambda ks: ks.s)
    for a, b in zip(sections, sections[1:]):
        if b.s - a.s < MIN_S_SEPARATION:
            raise ValueError(f"ducts.key_sections: s = {a.s:g} and {b.s:g} are closer than {MIN_S_SEPARATION:g}")
    return tuple(sections)


def state_from_config(cfg: dict) -> InletState:
    ducts = cfg.get("ducts") or {}
    inlet = ducts.get("inlet") or {}
    origin = inlet.get("origin")
    if origin is None:
        return InletState(None, ())
    if not (isinstance(origin, (list, tuple)) and len(origin) == 3):
        raise ValueError("ducts.inlet.origin must be [x, y, z]")
    plane = InletPlane(
        tuple(float(c) for c in origin), float(inlet.get("pitch_deg") or 0.0), float(inlet.get("sweep_deg") or 0.0)
    )
    plane.frame()  # validates the angles
    raw = ducts.get("key_sections")
    if raw is not None:
        return InletState(plane, parse_key_sections(raw))
    if inlet.get("vertices") is not None:  # the old 5-vertex format: a straight-edged inlet section
        flags = inlet.get("snapped")
        data = {"vertices": inlet["vertices"], "snapped": flags}
        return InletState(plane, (KeySection(0.0, ol.shape_from_config(data, "ducts.inlet")),))
    return InletState(plane, ())


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
        self._section_cache: tuple[InletPlane, Section | None] | None = None
        self.mode = MODE_VIEW
        self.locked = False
        self.message = ""
        self.notice = legacy_notice(cfg)
        self.settings = duct_settings(cfg)
        self.alignment: tuple[str, float, float] | None = None  # (edge label, edge sweep, applied plane sweep)
        self.section: Section | None = None  # the skin curve in the inlet plane
        self.evaluations: list[sketch.Opening] = []  # one per key section
        self.active = 0  # index of the section the editor shows; len(sections) is the engine face
        loaded = state_from_config(cfg)
        self.state = self._normalised(loaded)
        # a state completed here (resnapped, default sketch added) differs from the file only by rounding: the
        # file is the saved state unless a sketch had to be created
        self.saved_state = self.state if loaded.sections or loaded.plane is None else loaded
        self._recompute()

    # ---- observers ---------------------------------------------------------------------------------
    def add_listener(self, callback) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            callback()

    def _say(self, text: str) -> None:
        self.message = text

    # ---- derived state --------------------------------------------------------------------------------
    @property
    def area(self) -> float:
        """The per-inlet capture area (projected normal to the freestream): the inlet section's target."""
        return self.scene.sizing.capture_area_per_inlet

    @property
    def plane(self) -> InletPlane | None:
        return self.state.plane

    @property
    def placed(self) -> bool:
        return self.state.plane is not None

    @property
    def opening(self) -> sketch.Opening | None:
        """The evaluated inlet section (s = 0)."""
        return self.evaluations[0] if self.evaluations else None

    @property
    def valid(self) -> bool:
        return bool(self.evaluations) and all(e.valid for e in self.evaluations)

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
        """3D centroid of the inlet opening."""
        if self.opening is None:
            return None
        return self.state.plane.to_3d([self.opening.centroid_uv])[0]

    # ---- the area law -------------------------------------------------------------------------------------
    @property
    def n_branches(self) -> int:
        return self.scene.sizing.n_inlets

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
        plane = self.state.plane
        origin = plane.origin if plane else (0.0, 0.0, 0.0)
        return area_law.junction_s(origin, self.scene.face_center, self.settings.junction_point,
                                   self.settings.junction_fraction)

    def area_at(self, s: float) -> float:
        """A(s), the in-plane area every key section is held to."""
        return area_law.section_area(s, self.inlet_plane_area, self.face_area, self.junction_s, self.settings.law,
                                     self.n_branches)

    def face_shape(self) -> SectionShape:
        """The engine face (s = 1): a circle of the face diameter, fixed."""
        return ol.circle_shape(0.5 * self.scene.face_diameter)

    # ---- the active section ------------------------------------------------------------------------------------
    @property
    def n_sections(self) -> int:
        return len(self.state.sections)

    @property
    def active_is_face(self) -> bool:
        return self.placed and self.active >= self.n_sections

    @property
    def active_is_inlet(self) -> bool:
        return self.placed and self.active == 0

    @property
    def active_s(self) -> float | None:
        if not self.placed:
            return None
        return 1.0 if self.active_is_face else self.state.sections[self.active].s

    @property
    def active_shape(self) -> SectionShape | None:
        if not self.placed:
            return None
        return self.face_shape() if self.active_is_face else self.state.sections[self.active].shape

    @property
    def active_opening(self) -> sketch.Opening | None:
        """The evaluated active section; for the engine face a plain evaluation of the circle."""
        if not self.placed:
            return None
        if self.active_is_face:
            return sketch.evaluate(self.face_shape(), None, None, self.face_area)
        return self.evaluations[self.active]

    @property
    def active_curve(self) -> Section | None:
        """The skin curve the active section can snap to (only the inlet section has one)."""
        return self.section if self.active_is_inlet else None

    def section_list(self) -> list[tuple[float, str, bool]]:
        """(s, label, editable) of every key section plus the fixed engine face."""
        rows = []
        for k, ks in enumerate(self.state.sections):
            rows.append((ks.s, "inlet" if k == 0 else "key section", True))
        if self.placed:
            rows.append((1.0, "engine face (fixed)", False))
        return rows

    def select_section(self, index: int) -> bool:
        """Show another key section (0 = inlet, n_sections = the engine face). Not part of the undo history."""
        if not self.placed or not 0 <= index <= self.n_sections:
            return False
        if index != self.active:
            self.active = index
            self._notify()
        return True

    def _section_for(self, plane: InletPlane) -> Section | None:
        if self._section_cache is None or self._section_cache[0] != plane:
            self._section_cache = (plane, compute_section(self.scene.fuselage, plane))
        return self._section_cache[1]

    def _normalised(self, state: InletState) -> InletState:
        """The inlet snapped exactly onto the current section (a default sketch if there is none yet), and every
        other key section held at its area A(s)."""
        if state.plane is None:
            return state
        section = self._section_for(state.plane)
        secs = list(state.sections)
        if not secs or secs[0].s != 0.0:
            secs.insert(0, KeySection(0.0, sketch.default_sketch(section, state.plane, self.area)))
        else:
            secs[0] = KeySection(0.0, sketch.resnap(secs[0].shape, section))
        a_in = self.area / max(state.plane.projection_factor, 1e-6)
        s_j = area_law.junction_s(state.plane.origin, self.scene.face_center, self.settings.junction_point,
                                  self.settings.junction_fraction)
        for k in range(1, len(secs)):
            target = area_law.section_area(secs[k].s, a_in, self.face_area, s_j, self.settings.law, self.n_branches)
            secs[k] = KeySection(secs[k].s, sketch.rescale_to_area(secs[k].shape, target))
        return InletState(state.plane, tuple(secs))

    def _recompute(self) -> None:
        st = self.state
        if st.plane is None:
            self.section = None
            self.evaluations = []
            self.active = 0
            return
        self.section = self._section_for(st.plane)
        evals = [sketch.evaluate(st.sections[0].shape, self.section, st.plane, self.area)]
        for ks in st.sections[1:]:
            evals.append(sketch.evaluate(ks.shape, None, None, self.area_at(ks.s)))
        self.evaluations = evals

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

    def _set_state(self, new: InletState) -> None:
        """Make ``new`` the state; the editor stays on its section (or on the engine face), clamped to what exists."""
        was_face = self.active_is_face
        self.state = new
        self.active = len(new.sections) if was_face else min(self.active, max(len(new.sections) - 1, 0))
        self._recompute()

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

    # ---- the plane --------------------------------------------------------------------------------------------
    def _changed(self, changed: bool, text: str = "") -> bool:
        if changed:
            self._say(text)
            self._notify()
        return changed

    def _with_plane(self, plane: InletPlane, key: str | None) -> bool:
        """Move / rotate the plane: the sketch keeps its (u, v), snapped vertices follow the new section and the
        other key sections are re-held at their (changed) area A(s)."""
        plane.frame()  # validates the angles before anything changes
        if plane == self.state.plane:
            return False
        base = replace(self.state, plane=plane)
        return self._commit(self._normalised(base), key)

    def set_param(self, key: str, value) -> bool:
        """Change the pitch or the sweep (degrees)."""
        if key not in PLANE_KEYS:
            raise KeyError(key)
        if self.state.plane is None:
            raise ValueError("place the inlet first")
        return self._changed(self._with_plane(replace(self.state.plane, **{key: float(value)}), key))

    def set_origin(self, x: float, y: float, z: float) -> bool:
        """Move the plane origin (the right inlet keeps y >= 0). Blocked while locked."""
        if self.locked:
            self._say("Inlet is locked")
            self._notify()
            return False
        old = self.state.plane
        plane = InletPlane((float(x), abs(float(y)), float(z)), old.pitch_deg if old else 0.0,
                           old.sweep_deg if old else 0.0)
        return self._changed(self._with_plane(plane, "origin"))

    # ---- editing the active section ------------------------------------------------------------------------------
    def _replace_active(self, shape: SectionShape) -> InletState:
        secs = list(self.state.sections)
        secs[self.active] = KeySection(secs[self.active].s, shape)
        return replace(self.state, sections=tuple(secs))

    def _edit(self, fn, key: str | None = None, rescale: bool = True) -> bool:
        """Apply ``fn(shape) -> shape`` to the active section as one undo step. A non-inlet section is brought
        back to its area A(s) in the same step (``rescale`` False defers that, for the events of a drag)."""
        if not self.placed or self.active_is_face:
            self._say("The engine face section is fixed" if self.placed else "Place the inlet first")
            self._notify()
            return False
        try:
            shape = fn(self.state.sections[self.active].shape)
        except ValueError as exc:
            self._say(str(exc))
            self._notify()
            return False
        if rescale and self.active > 0:
            shape = sketch.rescale_to_area(shape, self.area_at(self.state.sections[self.active].s))
        return self._changed(self._commit(self._replace_active(shape), key))

    def finish_edit(self) -> bool:
        """End of a drag: hold a non-inlet section at its area A(s) (merged into the drag's undo step)."""
        if not self.placed or self.active_is_face or self.active == 0:
            return False
        ks = self.state.sections[self.active]
        shape = sketch.rescale_to_area(ks.shape, self.area_at(ks.s))
        return self._changed(self._commit(self._replace_active(shape), self._last_key))

    def drag_vertex(self, index: int, uv, snap_tolerance: float | None) -> bool:
        """Move vertex ``index`` of the active section to ``uv``. The inlet vertex snaps onto the skin curve
        within ``snap_tolerance`` metres (None disables snapping) and unsnaps when dropped further away."""
        if not self.placed or self.active_is_face:
            return False

        def move(shape):
            point, flag = ((float(uv[0]), float(uv[1])), False)
            if self.active_is_inlet:
                point, flag = sketch.snap_point(self.section, uv, snap_tolerance)
            moved = ol.move_vertex(shape, index, point)
            snapped = list(moved.snapped)
            snapped[index] = flag
            return moved.with_(snapped=tuple(snapped))

        return self._edit(move, f"vertex{index}@{self.active}", rescale=False)

    def drag_handle(self, edge: int, which: int, uv) -> bool:
        """Drag a Bézier handle of a free curved edge of the active section."""
        return self._edit(lambda sh: ol.move_handle(sh, edge, which, uv), f"handle{edge}.{which}@{self.active}",
                          rescale=False)

    def insert_vertex(self, edge: int, uv) -> int | None:
        """Insert a vertex on ``edge`` at the point of the edge nearest ``uv``. Returns its index, or None."""
        made = []

        def insert(shape):
            point = uv
            if self.active_is_inlet and shape.edge_is_snapped(edge):  # a vertex on the skin edge stays on the skin
                point = tuple(sketch.snap_point(self.section, uv, float("inf"))[0])
            new, at = ol.insert_vertex(shape, edge, point)
            made.append(at)
            return new

        return made[0] if self._edit(insert) and made else None

    def delete_vertex(self, index: int) -> bool:
        return self._edit(lambda sh: ol.delete_vertex(sh, index))

    def set_edge_type(self, edge: int, kind: str) -> bool:
        return self._edit(lambda sh: ol.set_edge_type(sh, edge, kind))

    def set_continuity(self, index: int, mode: str) -> bool:
        return self._edit(lambda sh: ol.set_continuity(sh, index, mode))

    def fit_area(self) -> bool:
        """Active inlet section: scale only the free vertices so the projected area matches the capture area.
        Another key section: rescale uniformly to A(s)."""
        if not self.placed or self.active_is_face:
            return False
        if self.active > 0:
            self._edit(lambda sh: sh)  # already holds A(s) after every edit; nothing to fit
            self.finish_edit()
            self._say(f"Section s = {self.state.sections[self.active].s:g} is held at A(s) = "
                      f"{self.area_at(self.state.sections[self.active].s):.5f} m^2")
            self._notify()
            return True
        shape, ok, message = sketch.fit_area(self.state.sections[0].shape, self.section, self.state.plane, self.area)
        self._commit(self._replace_active(shape))
        self._say(message)
        self._notify()
        return ok

    # ---- key sections ------------------------------------------------------------------------------------------------
    def add_key_section(self, s: float) -> bool:
        """Add an intermediate key section at ``s`` (between 0 and 1, exclusive). It starts as a copy of the shape
        of the section before it, centred and held at A(s), and becomes the active one."""
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
        before = max((ks for ks in self.state.sections if ks.s < s), key=lambda ks: ks.s)
        shape = before.shape.with_(snapped=(False,) * before.shape.n)
        shape = ol.translate(shape, -ol.outline_centroid(before.shape, self.section if before.s == 0.0 else None))
        shape = sketch.rescale_to_area(shape, self.area_at(s))
        secs = sorted(list(self.state.sections) + [KeySection(s, shape)], key=lambda ks: ks.s)
        index = [ks.s for ks in secs].index(s)
        self._commit(replace(self.state, sections=tuple(secs)))
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
        self._commit(replace(self.state, sections=tuple(secs)))
        self.active = min(self.active, index - 1) if self.active >= index else self.active
        self._say("Removed the key section")
        self._notify()
        return True

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
        """A click on the fuselage. Ignored unless in place mode; places the plane origin there (creating the
        default sketch the first time) and leaves place mode."""
        if self.mode != MODE_PLACING:
            return False
        p = np.asarray(point, dtype=float)
        old = self.state.plane
        plane = InletPlane((float(p[0]), abs(float(p[1])), float(p[2])), old.pitch_deg if old else 0.0,
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
    def inlet_definition(self) -> dict | None:
        """What Save writes to ``ducts.inlet`` (the plane); None until an inlet has been placed."""
        st = self.state
        if st.plane is None or not st.sections:
            return None
        return {
            "origin": [float(c) for c in st.plane.origin],
            "pitch_deg": float(st.plane.pitch_deg),
            "sweep_deg": float(st.plane.sweep_deg),
        }

    def key_sections_definition(self) -> list[dict]:
        """What Save writes to ``ducts.key_sections``: every key section, the inlet first."""
        return [{"s": float(ks.s), **ol.shape_to_config(ks.shape)} for ks in self.state.sections]

    def save(self) -> bool:
        definition = self.inlet_definition()
        if self.save_target is None:
            self._say(f"Not saved: the example config is read-only. Copy it to {PRIVATE_CONFIG.name}.")
        elif definition is None:
            self._say("Nothing to save: place an inlet first.")
        elif not self.valid:
            first = next(e for ev in self.evaluations for e in ev.errors)
            self._say(f"ERROR: not saved: {first}")
        else:
            try:
                path = save_inlet_definition(definition, self.save_target, self.key_sections_definition(),
                                             remove_inlet_keys=("vertices", "snapped"))
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
        """Go back to the inlet stored in the config file (undoable)."""
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
        self.notice = legacy_notice(cfg)
        self.alignment = None
        self._section_cache = None  # the capture area (hence the default sketch) may have changed
        normalised = self._normalised(stored)
        self._commit(normalised)
        self._recompute()
        self.saved_state = normalised if stored.sections or stored.plane is None else stored
        self.mode = MODE_VIEW
        self._say(f"Reverted to {self.config_path.name}.")
        self._notify()
        return True

    # ---- readouts -------------------------------------------------------------------------------------------------------
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
        for ks, ev in zip(self.state.sections, self.evaluations):
            out += [Readout("", f"ERROR: s = {ks.s:g}: {e}", "error") for e in ev.errors]
        out += [Readout("", f"WARN {w}", "warn") for w in op.warnings]
        return out + tail
