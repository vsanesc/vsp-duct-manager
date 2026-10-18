# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""State of the inlet explorer, independent of any GUI toolkit.

Holds the inlet definition (picked point, shape, tilt, skin fit), the fitted outline, the undo/redo history,
the place-mode state machine, the lock, the dirty flag and save/revert. The Qt window and the headless
screenshot both drive this class, so everything that matters is testable without a display.
"""
import time
import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from ductmgr import vspio
from ductmgr.config import PRIVATE_CONFIG, load_config, save_inlet_definition
from ductmgr.engine import EngineSizingWarning, size_from_config
from ductmgr.geometry import DEFAULT_TILT_DEG, InletShape, inlet_frame, inlet_readout
from ductmgr.skinfit import AREA_TOLERANCE, DEFAULT_SKIN_FIT, InletOutline, fit_inlet, validate_mode

UNDO_LIMIT = 200  # history entries kept (the requirement is at least 50)
MERGE_WINDOW_S = 0.8  # consecutive edits of one parameter within this window share one undo step
GAP_PASS = 1e-3  # m: skin gap below which the skin contact is shown as PASS

SHAPE_KEYS = tuple(f.name for f in fields(InletShape))
PARAM_KEYS = SHAPE_KEYS + ("tilt_deg", "skin_fit")

MODE_VIEW, MODE_PLACING = "view", "placing"
MODE_TEXT = {MODE_VIEW: "View", MODE_PLACING: "Placing inlet - click on fuselage, Esc to cancel"}


@dataclass(frozen=True)
class InletState:
    """Everything the undo history tracks."""

    base_point: tuple[float, float, float] | None  # the picked / typed point; the centroid is its skin snap
    shape: InletShape
    tilt_deg: float
    skin_fit: str


@dataclass(frozen=True)
class Readout:
    label: str
    value: str
    level: str = "info"  # "info", "pass" or "warn"


def state_from_config(cfg: dict) -> InletState:
    inlet = ((cfg.get("ducts") or {}).get("inlet")) or {}
    centroid = inlet.get("centroid")
    tilt = inlet.get("tilt_deg")
    return InletState(
        base_point=tuple(float(v) for v in centroid) if centroid is not None else None,
        shape=InletShape.from_dict(inlet.get("shape")),
        tilt_deg=DEFAULT_TILT_DEG if tilt is None else float(tilt),
        skin_fit=validate_mode(inlet.get("skin_fit") or DEFAULT_SKIN_FIT),
    )


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
        self.mode = MODE_VIEW
        self.locked = False
        self.message = ""
        self.centroid: np.ndarray | None = None
        self.surface_normal: np.ndarray | None = None
        self.on_surface = True
        self.fit: InletOutline | None = None
        self.state = state_from_config(cfg)
        self.saved_state = self.state
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
    def dirty(self) -> bool:
        return self.state != self.saved_state

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    @property
    def mode_text(self) -> str:
        return MODE_TEXT[self.mode]

    def _recompute(self) -> None:
        st = self.state
        if st.base_point is None:
            self.centroid = self.surface_normal = self.fit = None
            self.on_surface = True
            return
        x, y, z = st.base_point
        snapped = vspio.snap_to_surface(self.scene.fuselage, x, abs(y), z)
        self.on_surface = snapped is not None
        if snapped is None:  # the vertical line through (x, y) misses the fuselage: keep the given height
            self.centroid = np.array([x, abs(y), z])
            self.surface_normal = vspio.surface_normal(self.scene.fuselage, self.centroid)
        else:
            self.centroid, self.surface_normal = snapped
        self.fit = fit_inlet(self.scene.fuselage, self.centroid, st.tilt_deg, st.shape, self.area, st.skin_fit)

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
    def set_param(self, key: str, value) -> bool:
        """Change a shape ratio, the tilt or the skin-fit mode. Returns True when something changed."""
        st = self.state
        if key in SHAPE_KEYS:
            new = replace(st, shape=replace(st.shape, **{key: float(value)}))
        elif key == "tilt_deg":
            new = replace(st, tilt_deg=float(value))
        elif key == "skin_fit":
            new = replace(st, skin_fit=validate_mode(value))
        else:
            raise KeyError(key)
        changed = self._commit(new, key)
        if changed:
            self._say("")
            self._notify()
        return changed

    def set_position(self, x: float, y: float, z: float) -> bool:
        """Move the inlet to (x, y, z); it snaps to the skin nearest that height. Blocked while locked."""
        if self.locked:
            self._say("Inlet is locked")
            self._notify()
            return False
        changed = self._commit(replace(self.state, base_point=(float(x), abs(float(y)), float(z))), "position")
        if changed:
            self._say("")
            self._notify()
        return changed

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
        """A click on the fuselage. Ignored unless in place mode; places the inlet and leaves place mode."""
        if self.mode != MODE_PLACING:
            return False
        p = np.asarray(point, dtype=float)
        self._commit(replace(self.state, base_point=(float(p[0]), abs(float(p[1])), float(p[2]))))
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
        if self.centroid is None:
            return None
        _, _, normal = inlet_frame(self.state.tilt_deg)
        return {
            "centroid": [float(v) for v in self.centroid],
            "normal": [float(v) for v in normal],
            "tilt_deg": float(self.state.tilt_deg),
            "skin_fit": self.state.skin_fit,
            "shape": self.state.shape.to_dict(),
        }

    def save(self) -> bool:
        definition = self.inlet_definition()
        if self.save_target is None:
            self._say(f"Not saved: the example config is read-only. Copy it to {PRIVATE_CONFIG.name}.")
        elif definition is None:
            self._say("Nothing to save: place an inlet first.")
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
        self._commit(stored)
        self._recompute()  # the capture area may have changed even if the inlet did not
        self.saved_state = stored
        self.mode = MODE_VIEW
        self._say(f"Reverted to {self.config_path.name}.")
        self._notify()
        return True

    # ---- readouts -------------------------------------------------------------------------------------------------------
    def readouts(self) -> list[Readout]:
        """Lines for the read-only panel; ``level`` drives the colour (WARN amber, PASS green)."""
        out: list[Readout] = []
        check = self.scene.engine_check
        # the engine check has its own line; keep it out of the generic warning list
        notes = [w for w in self.scene.warnings if check is None or w != f"WARN {check.message}"]
        if check is not None:
            out.append(Readout("Face / envelope",
                               f"{check.face_diameter:.4f} / {check.envelope_diameter:.4f} m = {check.ratio:.3f} "
                               f"[{check.status}: {check.ratio_range[0]:g}-{check.ratio_range[1]:g}]",
                               "pass" if check.status == "PASS" else "warn"))
        if self.fit is None:
            out.append(Readout("Inlet", "not placed - use Place inlet (P)"))
            out += [Readout("", w, "warn") for w in notes]
            return out

        fit, c = self.fit, self.centroid
        err = fit.area_error
        area_level = "info" if fit.mode == "none" else ("pass" if abs(err) <= AREA_TOLERANCE else "warn")
        out += [
            Readout("Centroid", f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} m", "info" if self.on_surface else "warn"),
            Readout("Capture area target", f"{fit.target_area:.5f} m^2"),
            Readout("Capture area actual", f"{fit.area:.5f} m^2 ({100 * err:+.2f} %)", area_level),
            Readout("Skin fit", fit.mode),
            Readout("Max skin gap", f"{fit.max_gap:.4f} m",
                    "info" if fit.mode == "none" else ("pass" if fit.max_gap <= GAP_PASS else "warn")),
            Readout("Opening height", "n/a (flush)" if fit.mode == "flush" else f"{fit.opening_height:.4f} m"),
        ]
        face = self.scene.face_center
        if face is None:
            out.append(Readout("Engine face", "unknown: no distance / offset readout", "warn"))
        else:
            r = inlet_readout(c, face, fit.uv, fit.area)
            out += [
                Readout("Aspect ratio (w/h)", f"{r.aspect_ratio:.3f}"),
                Readout("Distance to face", f"{r.distance_to_face:.4f} m"),
                Readout("Offset dy, dz", f"{r.dy:+.4f}, {r.dz:+.4f} m"),
                Readout("Offset / length",
                        f"{r.offset_over_length:.3f}" if r.length > 0 else "n/a (face not aft)",
                        "info" if r.length > 0 else "warn"),
            ]
        out += [Readout("", w, "warn") for w in notes]
        out += [Readout("", f"WARN {w}", "warn") for w in fit.warnings]
        return out
