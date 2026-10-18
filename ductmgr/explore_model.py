# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""State of the inlet explorer, independent of any GUI toolkit.

Holds the inlet plane and the 5-vertex section sketch (vertices in the plane's (u, v), snapped flags), the
section curve and the evaluated opening, one undo/redo history shared by the 3D view, the panel and the
sketcher, the place-mode state machine, the lock, the dirty flag and save/revert. The Qt window and the headless
screenshot both drive this class, so everything that matters is testable without a display.
"""
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import math

from ductmgr import sketch, vspio
from ductmgr.config import PRIVATE_CONFIG, load_config, save_inlet_definition
from ductmgr.engine import EngineSizingWarning, size_from_config
from ductmgr.geometry import InletPlane, inlet_readout, sweep_for_edge_angle
from ductmgr.section import Section, compute_section

MIN_ALIGN_PROJECTION = 0.2  # |n.X| below this blocks an edge alignment
UNDO_LIMIT = 200  # history entries kept (the requirement is at least 50)
MERGE_WINDOW_S = 0.8  # consecutive edits of one parameter / vertex within this window share one undo step
PLANE_KEYS = ("pitch_deg", "sweep_deg")

MODE_VIEW, MODE_PLACING = "view", "placing"
MODE_TEXT = {MODE_VIEW: "View", MODE_PLACING: "Placing inlet - click on fuselage, Esc to cancel"}

LEGACY_INLET_KEYS = ("shape", "skin_fit", "attach_point", "centroid", "normal", "tilt_deg", "station_x", "polygon")


@dataclass(frozen=True)
class InletState:
    """Everything the undo history tracks."""

    plane: InletPlane | None  # None until an inlet has been placed
    vertices: tuple[tuple[float, float], ...] | None  # 5 (u, v) pairs, right inlet
    snapped: tuple[bool, ...]


@dataclass(frozen=True)
class Readout:
    label: str
    value: str
    level: str = "info"  # "info", "pass", "warn" or "error"


# ---- config <-> state -----------------------------------------------------------------------------------------
def legacy_notice(cfg: dict) -> str:
    """One line naming old ducts.inlet keys that are ignored; empty when there are none."""
    inlet = ((cfg.get("ducts") or {}).get("inlet")) or {}
    old = [k for k in LEGACY_INLET_KEYS if inlet.get(k) is not None]
    if not old:
        return ""
    return (f"ducts.inlet: ignoring old key(s) {', '.join(old)}; the inlet is now origin, pitch_deg, sweep_deg, "
            "vertices and snapped")


def state_from_config(cfg: dict) -> InletState:
    inlet = ((cfg.get("ducts") or {}).get("inlet")) or {}
    origin = inlet.get("origin")
    if origin is None:
        return InletState(None, None, (False,) * sketch.N_VERTICES)
    if not (isinstance(origin, (list, tuple)) and len(origin) == 3):
        raise ValueError("ducts.inlet.origin must be [x, y, z]")
    plane = InletPlane(
        tuple(float(c) for c in origin), float(inlet.get("pitch_deg") or 0.0), float(inlet.get("sweep_deg") or 0.0)
    )
    plane.frame()  # validates the angles
    raw = inlet.get("vertices")
    if raw is None:
        return InletState(plane, None, (False,) * sketch.N_VERTICES)
    if not (isinstance(raw, (list, tuple)) and len(raw) == sketch.N_VERTICES
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw)):
        raise ValueError(f"ducts.inlet.vertices must be {sketch.N_VERTICES} [u, v] pairs")
    flags = inlet.get("snapped")
    if flags is None:
        flags = [False] * sketch.N_VERTICES
    if not (isinstance(flags, (list, tuple)) and len(flags) == sketch.N_VERTICES
            and all(isinstance(f, bool) for f in flags)):
        raise ValueError(f"ducts.inlet.snapped must be {sketch.N_VERTICES} true/false values")
    return InletState(plane, tuple((float(u), float(v)) for u, v in raw), tuple(flags))


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
        self.alignment: tuple[str, float, float] | None = None  # (edge label, edge sweep, applied plane sweep)
        self.section: Section | None = None
        self.opening: sketch.Opening | None = None
        loaded = state_from_config(cfg)
        self.state = self._normalised(loaded)
        # a state completed here (resnapped to the section, default sketch added) differs from the file only
        # by rounding: the file is the saved state unless a sketch had to be created
        self.saved_state = self.state if loaded.vertices is not None or loaded.plane is None else loaded
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
        return self.scene.sizing.capture_area_per_inlet

    @property
    def plane(self) -> InletPlane | None:
        return self.state.plane

    @property
    def placed(self) -> bool:
        return self.state.plane is not None

    @property
    def valid(self) -> bool:
        return self.opening is not None and self.opening.valid

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
        """3D centroid of the opening."""
        if self.opening is None:
            return None
        return self.state.plane.to_3d([self.opening.centroid_uv])[0]

    def left_vertices(self):
        """The left inlet's vertices and flags (mirror image of the right one)."""
        from ductmgr.geometry import mirror_vertices

        return mirror_vertices(self.state.vertices, self.state.snapped)

    def _section_for(self, plane: InletPlane) -> Section | None:
        if self._section_cache is None or self._section_cache[0] != plane:
            self._section_cache = (plane, compute_section(self.scene.fuselage, plane))
        return self._section_cache[1]

    def _normalised(self, state: InletState) -> InletState:
        """Snapped vertices put exactly on the current section; a default sketch if there is none yet."""
        if state.plane is None:
            return state
        section = self._section_for(state.plane)
        if state.vertices is None:
            verts, flags = sketch.default_sketch(section, state.plane, self.area)
        else:
            verts, flags = sketch.resnap(state.vertices, state.snapped, section)
        return replace(state, vertices=verts, snapped=flags)

    def _recompute(self) -> None:
        st = self.state
        if st.plane is None:
            self.section = self.opening = None
            return
        self.section = self._section_for(st.plane)
        self.opening = (sketch.evaluate(st.vertices, st.snapped, self.section, st.plane, self.area)
                        if st.vertices is not None else None)

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
        self.state = new
        self._recompute()
        return True

    def undo(self) -> bool:
        if not self._past:
            return False
        self._future.append(self.state)
        self.state = self._past.pop()
        self._last_key = None
        self._recompute()
        self._say("Undo")
        self._notify()
        return True

    def redo(self) -> bool:
        if not self._future:
            return False
        self._past.append(self.state)
        self.state = self._future.pop()
        self._last_key = None
        self._recompute()
        self._say("Redo")
        self._notify()
        return True

    # ---- editing ----------------------------------------------------------------------------------------------
    def _changed(self, changed: bool, text: str = "") -> bool:
        if changed:
            self._say(text)
            self._notify()
        return changed

    def _with_plane(self, plane: InletPlane, key: str | None) -> bool:
        """Move / rotate the plane: the sketch keeps its (u, v) and snapped vertices follow the new section."""
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

    def drag_vertex(self, index: int, uv, snap_tolerance: float | None) -> bool:
        """Move vertex ``index`` to ``uv``; it snaps onto the section curve within ``snap_tolerance`` metres
        (None disables snapping) and unsnaps when dropped further away."""
        if self.state.vertices is None:
            return False
        point, flag = sketch.snap_point(self.section, uv, snap_tolerance)
        verts = list(self.state.vertices)
        flags = list(self.state.snapped)
        verts[index], flags[index] = point, flag
        new = replace(self.state, vertices=tuple(verts), snapped=tuple(flags))
        return self._changed(self._commit(new, f"vertex{index}"))

    def fit_area(self) -> bool:
        """Scale only the free vertices so the projected area matches the capture area target."""
        if self.state.vertices is None:
            return False
        verts, ok, message = sketch.fit_area(self.state.vertices, self.state.snapped, self.section,
                                             self.state.plane, self.area)
        self._commit(replace(self.state, vertices=verts))
        self._say(message)
        self._notify()
        return ok

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
        """What Save writes to ``ducts.inlet``; None until an inlet has been placed."""
        st = self.state
        if st.plane is None or st.vertices is None:
            return None
        return {
            "origin": [float(c) for c in st.plane.origin],
            "pitch_deg": float(st.plane.pitch_deg),
            "sweep_deg": float(st.plane.sweep_deg),
            "vertices": [[float(u), float(v)] for u, v in st.vertices],
            "snapped": [bool(f) for f in st.snapped],
        }

    def save(self) -> bool:
        definition = self.inlet_definition()
        if self.save_target is None:
            self._say(f"Not saved: the example config is read-only. Copy it to {PRIVATE_CONFIG.name}.")
        elif definition is None:
            self._say("Nothing to save: place an inlet first.")
        elif not self.valid:
            self._say(f"ERROR: not saved: {self.opening.errors[0]}")
        else:
            try:
                path = save_inlet_definition(definition, self.save_target)
            except OSError as exc:
                self._say(f"Save failed: {exc}")
            else:
                self.saved_state = self.state
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
        except Exception as exc:  # a broken file must not crash the window; say what is wrong
            self._say(f"Revert failed: {type(exc).__name__}: {exc}".splitlines()[0])
            self._notify()
            return False
        self.scene.sizing = sizing
        self.notice = legacy_notice(cfg)
        self.alignment = None
        self._section_cache = None  # the capture area (hence the default sketch) may have changed
        normalised = self._normalised(stored)
        self._commit(normalised)
        self._recompute()
        self.saved_state = normalised if stored.vertices is not None or stored.plane is None else stored
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
        out += [Readout("", f"ERROR: {e}", "error") for e in op.errors]
        out += [Readout("", f"WARN {w}", "warn") for w in op.warnings if "from the target" not in w]
        return out + tail
