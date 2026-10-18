# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""PyVista scene and interactive inlet placement (Phase 2).

``load_scene`` reads the model and the Phase 1 sizing; ``ExploreApp`` owns the interactive state (picked
point, nudges, tilt, shape ratios), draws the right inlet and its mirror image, shows a live readout, and
saves/reloads the inlet definition in the private config.
"""
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml

from ductmgr import vspio
from ductmgr.config import PRIVATE_CONFIG, load_config, resolve_model_path, save_inlet_definition
from ductmgr.engine import EngineSizingWarning, SizingResult, size_from_config
from ductmgr.geometry import (
    DEFAULT_TILT_DEG,
    InletShape,
    aspect_ratio,
    inlet_frame,
    inlet_readout,
    mirror_xz,
)
from ductmgr.skinfit import DEFAULT_SKIN_FIT, SKIN_FIT_MODES, InletOutline, fit_inlet, validate_mode

# ---- appearance ----------------------------------------------------------------------------
FUSELAGE_COLOR, FUSELAGE_OPACITY = "#7fb3d5", 0.35
CONTEXT_COLOR, CONTEXT_OPACITY = "#c8c8c8", 0.7
ENGINE_COLOR, ENGINE_OPACITY = "#6b6b6b", 0.6
FACE_COLOR = "#d62728"
RIGHT_COLOR, LEFT_COLOR = "#ff7f0e", "#17becf"
INLET_OPACITY = 0.65
BACKGROUND = "white"
DISC_RESOLUTION = (4, 64)  # radial, circumferential
ENVELOPE_COLOR, ENVELOPE_OPACITY, ENVELOPE_SEGMENTS = "#d62728", 0.35, 96
TANGENT_PATCH_SCALE = 1.6  # tangent-plane square, in units of the inlet's largest dimension

# ---- UI ranges (not design inputs) ------------------------------------------------------------
SHAPE_SLIDERS = {  # name: (min, max)
    "width_bottom": (0.2, 3.0),
    "width_top": (0.2, 3.0),
    "height": (0.2, 3.0),
    "apex_height": (0.0, 1.0),
    "skew": (-0.5, 0.5),
}
TILT_RANGE_DEG = (-45.0, 45.0)
NUDGE_RANGE_FRACTION = 0.1  # of the fuselage length, each way
READOUT_POS = (0.56, 0.98)  # normalised viewport, top-left corner of the readout block
READOUT_FONT_SIZE = 9
SLIDER_TITLE_HEIGHT, SLIDER_WIDTH, SLIDER_TUBE_WIDTH = 0.018, 0.018, 0.004
STATUS_POS = (0.005, 0.005)  # bottom-left corner of the help / warnings block
SLIDER_X = (0.02, 0.20)
SLIDER_TOP, SLIDER_STEP = 0.94, 0.095

HELP = (  # no '|' here: VTK renders it as a wide blank
    "Click the fuselage (or hover + p) to place the right inlet.  Sliders: shape, tilt, nudge.  "
    "s: save   r: reload   f: skin fit   1 side   2 top   3 front   4 iso"
)


@dataclass
class Scene:
    fuselage: pv.PolyData
    context: pv.PolyData
    engine_mesh: pv.PolyData
    face_center: np.ndarray | None
    face_diameter: float  # the fan, always engine.face_diameter from the config
    face_source: str  # where the face centre came from: "model", "config" or "none"
    hub_to_tip_ratio: float
    sizing: SizingResult
    envelope_diameter: float | None = None  # engine Geom bounding box (casing), when the Geom is known
    engine_check: vspio.FaceEnvelopeCheck | None = None
    warnings: list[str] = field(default_factory=list)


def load_scene(cfg: dict, config_path: Path) -> Scene:
    """Read the fuselage, engine and context geoms from the model and size the capture area."""
    eng = cfg.get("engine") or {}
    model_path = resolve_model_path(cfg, config_path)
    msgs: list[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineSizingWarning)  # surfaced through result.warnings
        sizing = size_from_config(cfg)
    msgs.extend(f"WARN {w}" for w in sizing.warnings)

    vspio.open_model(model_path)
    fuselage_name = cfg["model"]["fuselage_name"]
    geoms = vspio.list_geoms()
    fus_geom = vspio.find_geom(fuselage_name, geoms)
    fuselage = vspio.export_mesh([fus_geom])
    note = vspio.check_mesh_bounds(fuselage, [fus_geom])
    if note:
        msgs.append(note)

    engine_name = eng.get("geom_name")
    engine_geom = None
    face_center = None
    face_diameter = float(eng["face_diameter"])  # the fan: from the config, never from the bounding box
    face_source = "none"
    envelope_diameter = None
    engine_check = None
    if engine_name:
        face = vspio.engine_face(engine_name)
        engine_geom = face.geom
        face_center, face_source = face.center, "model"
        envelope_diameter = face.envelope_diameter
        engine_check = vspio.check_face_to_envelope(
            face, face_diameter, tuple(eng.get("face_to_envelope_range") or vspio.DEFAULT_FACE_TO_ENVELOPE_RANGE)
        )
        if engine_check.status == "WARN":
            msgs.append(f"WARN {engine_check.message}")
    else:
        centre = ((cfg.get("ducts") or {}).get("engine_face") or {}).get("centre")
        if centre is not None:
            face_center, face_source = np.array(centre, dtype=float), "config"
        msgs.append(
            "WARN engine.geom_name is not set: the engine face is "
            + ("taken from ducts.engine_face.centre" if centre is not None else "unknown")
        )

    skip = {fus_geom.id} | ({engine_geom.id} if engine_geom else set())
    context = vspio.export_mesh([g for g in geoms if g.id not in skip])
    engine_mesh = vspio.export_mesh([engine_geom]) if engine_geom else pv.PolyData()

    return Scene(
        fuselage=fuselage,
        context=context,
        engine_mesh=engine_mesh,
        face_center=face_center,
        face_diameter=face_diameter,
        face_source=face_source,
        hub_to_tip_ratio=float(eng.get("hub_to_tip_ratio") or 0.0),
        sizing=sizing,
        envelope_diameter=envelope_diameter,
        engine_check=engine_check,
        warnings=msgs,
    )


def _polygon_mesh(points: np.ndarray, planar: bool = True) -> tuple[pv.PolyData, pv.PolyData]:
    """Filled polygon and its closed outline. A non-planar outline is filled as a fan about its mean point."""
    n = len(points)
    if planar:
        fill = pv.PolyData(points, faces=np.hstack([[n], np.arange(n)])).triangulate()
    else:
        pts = np.vstack([points, points.mean(axis=0)])
        tris = np.array([[3, i, (i + 1) % n, n] for i in range(n)]).ravel()
        fill = pv.PolyData(pts, faces=tris)
    outline = pv.lines_from_points(np.vstack([points, points[:1]]))
    return fill, outline


class ExploreApp:
    """Interactive inlet placement on a PyVista plotter."""

    def __init__(self, plotter: pv.Plotter, scene: Scene, cfg: dict, config_path: Path, save_target: Path | None):
        """``save_target`` is the config ``s`` writes to; None disables saving (example config)."""
        self.plotter = plotter
        self.scene = scene
        self.config_path = Path(config_path)
        self.save_target = Path(save_target) if save_target is not None else None
        self.status = ""
        self.actors: dict[str, object] = {}
        self.sliders: dict[str, object] = {}
        self.base_point: np.ndarray | None = None
        self.nudge = [0.0, 0.0]
        self.centroid: np.ndarray | None = None
        self.surface_normal: np.ndarray | None = None
        self.on_surface = True
        self.readout = None
        self.skin_fit = DEFAULT_SKIN_FIT
        self.fit: InletOutline | None = None
        self._apply_config(cfg)
        self._draw_static()
        self.update()

    # ---- state ------------------------------------------------------------------------------
    @property
    def area(self) -> float:
        return self.scene.sizing.capture_area_per_inlet

    def _apply_config(self, cfg: dict) -> None:
        inlet = ((cfg.get("ducts") or {}).get("inlet")) or {}
        self.shape = InletShape.from_dict(inlet.get("shape"))
        tilt = inlet.get("tilt_deg")
        self.tilt_deg = DEFAULT_TILT_DEG if tilt is None else float(tilt)
        self.skin_fit = validate_mode(inlet.get("skin_fit") or DEFAULT_SKIN_FIT)
        self.nudge = [0.0, 0.0]
        centroid = inlet.get("centroid")
        self.base_point = np.array(centroid, dtype=float) if centroid is not None else None
        self._recompute_centroid()

    def _recompute_centroid(self) -> None:
        if self.base_point is None:
            self.centroid = self.surface_normal = None
            return
        bx, by, bz = self.base_point
        x, y = bx + self.nudge[0], abs(by + self.nudge[1])  # the right inlet stays at y >= 0
        snapped = vspio.snap_to_surface(self.scene.fuselage, x, y, bz)
        self.on_surface = snapped is not None
        if snapped is None:  # line through (x, y) misses the fuselage: keep the picked height
            self.centroid = np.array([x, y, bz])
            self.surface_normal = vspio.surface_normal(self.scene.fuselage, self.centroid)
        else:
            self.centroid, self.surface_normal = snapped

    def pick(self, point) -> None:
        """Place the right inlet at a picked surface point (a point on the left side is mirrored)."""
        p = np.array(point, dtype=float)
        p[1] = abs(p[1])
        self.base_point = p
        self.nudge = [0.0, 0.0]
        for key in ("nudge_x", "nudge_y"):
            self._set_slider(key, 0.0)
        self._recompute_centroid()
        self.status = ""
        self.update()

    def set_param(self, key: str, value: float) -> None:
        if key in SHAPE_SLIDERS:
            self.shape = replace(self.shape, **{key: float(value)})
        elif key == "tilt_deg":
            self.tilt_deg = float(value)
        elif key == "nudge_x":
            self.nudge[0] = float(value)
            self._recompute_centroid()
        elif key == "nudge_y":
            self.nudge[1] = float(value)
            self._recompute_centroid()
        else:
            raise KeyError(key)
        self.update()

    def set_skin_fit(self, mode: str) -> None:
        self.skin_fit = validate_mode(mode)
        self.update()

    def cycle_skin_fit(self) -> None:
        self.set_skin_fit(SKIN_FIT_MODES[(SKIN_FIT_MODES.index(self.skin_fit) + 1) % len(SKIN_FIT_MODES)])

    def inlet_definition(self) -> dict | None:
        """What ``s`` writes to ``ducts.inlet``; None until an inlet has been placed."""
        if self.centroid is None:
            return None
        _, _, normal = inlet_frame(self.tilt_deg)
        return {
            "centroid": [float(v) for v in self.centroid],
            "normal": [float(v) for v in normal],
            "tilt_deg": float(self.tilt_deg),
            "skin_fit": self.skin_fit,
            "shape": self.shape.to_dict(),
        }

    # ---- actions ------------------------------------------------------------------------------
    def save(self) -> None:
        definition = self.inlet_definition()
        if self.save_target is None:
            self.status = f"Not saved: the example config is read-only. Copy it to {PRIVATE_CONFIG.name}."
        elif definition is None:
            self.status = "Nothing to save: place an inlet first."
        else:
            try:
                path = save_inlet_definition(definition, self.save_target)
                self.status = f"Saved inlet to {path.name}."
            except (OSError, PermissionError) as exc:
                self.status = f"Save failed: {exc}"
        self._draw_text()
        self.plotter.render()

    def reload(self) -> None:
        try:
            cfg, _ = load_config(self.config_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineSizingWarning)
                self.scene.sizing = size_from_config(cfg)
            self._apply_config(cfg)
            for key, value in {**self.shape.to_dict(), "tilt_deg": self.tilt_deg, "nudge_x": 0.0, "nudge_y": 0.0}.items():
                self._set_slider(key, value)
            self.status = f"Reloaded {self.config_path.name}."
        except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
            self.status = f"Reload failed: {type(exc).__name__}: {exc}".splitlines()[0]
        self.update()

    # ---- drawing --------------------------------------------------------------------------------
    def _draw_static(self) -> None:
        p, s = self.plotter, self.scene
        p.set_background(BACKGROUND)
        self.actors["fuselage"] = p.add_mesh(
            s.fuselage, color=FUSELAGE_COLOR, opacity=FUSELAGE_OPACITY, smooth_shading=True,
            name="fuselage", pickable=True,
        )
        if s.context.n_points:
            self.actors["context"] = p.add_mesh(
                s.context, color=CONTEXT_COLOR, opacity=CONTEXT_OPACITY, name="context", pickable=False
            )
        if s.engine_mesh.n_points:
            self.actors["engine"] = p.add_mesh(
                s.engine_mesh, color=ENGINE_COLOR, opacity=ENGINE_OPACITY, name="engine", pickable=False
            )
        if s.face_center is not None:
            radius = 0.5 * s.face_diameter
            disc = pv.Disc(
                center=s.face_center, inner=s.hub_to_tip_ratio * radius, outer=radius,
                normal=(1.0, 0.0, 0.0), r_res=DISC_RESOLUTION[0], c_res=DISC_RESOLUTION[1],
            )
            self.actors["face"] = p.add_mesh(disc, color=FACE_COLOR, opacity=0.9, name="face", pickable=False)
            if s.envelope_diameter is not None:  # the casing, as a faint outline around the fan disc
                t = np.linspace(0.0, 2.0 * np.pi, ENVELOPE_SEGMENTS + 1)
                r = 0.5 * s.envelope_diameter
                ring = s.face_center + np.c_[np.zeros_like(t), r * np.cos(t), r * np.sin(t)]
                self.actors["envelope"] = p.add_mesh(
                    pv.lines_from_points(ring), color=ENVELOPE_COLOR, opacity=ENVELOPE_OPACITY,
                    line_width=1, name="envelope", pickable=False,
                )

        everything = s.fuselage.bounds
        if s.context.n_points:
            b = np.array([everything, s.context.bounds]).reshape(2, 3, 2)
            everything = tuple(np.stack([b[:, :, 0].min(0), b[:, :, 1].max(0)], axis=1).ravel())
        p.show_bounds(
            bounds=everything, xtitle="X aft (m)", ytitle="Y right (m)", ztitle="Z up (m)",
            ticks="both", minor_ticks=True, grid=False, location="outer", color="black", font_size=9,
            fmt="%.1f", n_xlabels=5, n_ylabels=4, n_zlabels=3,
        )
        p.add_axes()
        p.camera_position = "iso"

    def _draw_inlets(self) -> None:
        names = ("inlet_right_fill", "inlet_right_edge", "inlet_left_fill", "inlet_left_edge", "tangent",
                 "to_face")
        if self.centroid is None:
            for name in names:
                self.plotter.remove_actor(name, render=False)
            return
        right = self.fit.points
        for side, pts, color in (("right", right, RIGHT_COLOR), ("left", mirror_xz(right, reverse=True), LEFT_COLOR)):
            fill, outline = _polygon_mesh(pts, self.fit.planar)
            self.plotter.add_mesh(fill, color=color, opacity=INLET_OPACITY, name=f"inlet_{side}_fill",
                                  pickable=False, show_edges=False, render=False)
            self.plotter.add_mesh(outline, color="black", line_width=3, name=f"inlet_{side}_edge",
                                  pickable=False, render=False)

        size = TANGENT_PATCH_SCALE * float(np.ptp(self.fit.uv, axis=0).max())
        patch = pv.Plane(center=self.centroid, direction=self.surface_normal, i_size=size, j_size=size)
        self.plotter.add_mesh(patch, color="green", opacity=0.25, name="tangent", pickable=False, render=False)

        if self.scene.face_center is not None:
            line = pv.Line(self.centroid, self.scene.face_center)
            self.plotter.add_mesh(line, color="black", line_width=1, name="to_face", pickable=False, render=False)
        else:
            self.plotter.remove_actor("to_face", render=False)

    def readout_text(self) -> str:
        a = self.area
        lines = [f"Capture area target  : {a:.5f} m^2", f"Shape                : {self.shape_label()}"]
        if self.centroid is None:
            lines.append("")
            lines.append("No inlet placed: click on the fuselage.")
            return "\n".join(lines)
        fit = self.fit
        face = self.scene.face_center
        c = self.centroid
        lines.append(f"Centroid  x y z      : {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} m"
                     + ("" if self.on_surface else "  (off surface!)"))
        lines.append(f"Tilt                 : {self.tilt_deg:.1f} deg")
        lines += [
            f"Skin fit             : {fit.mode}",
            f"Capture area actual  : {fit.area:.5f} m^2 ({100 * fit.area_error:+.2f} %)",
            f"Max skin gap         : {fit.max_gap:.4f} m",
            "Opening height       : " + ("n/a (flush)" if fit.mode == "flush" else f"{fit.opening_height:.4f} m"),
        ]
        if face is None:
            self.readout = None
            lines.append(f"Aspect ratio (w/h)   : {aspect_ratio(fit.uv):.3f}")
            lines.append("Engine face unknown: no distance / offset readout.")
            return "\n".join(lines)
        r = inlet_readout(c, face, fit.uv, fit.area)
        self.readout = r
        lines += [
            f"Aspect ratio (w/h)   : {r.aspect_ratio:.3f}",
            f"Distance to face     : {r.distance_to_face:.4f} m",
            f"Offset dy, dz        : {r.dy:+.4f}, {r.dz:+.4f} m",
            f"Offset magnitude     : {r.offset:.4f} m",
            f"Axial length         : {r.length:.4f} m",
            "Offset / length      : " + (f"{r.offset_over_length:.3f}" if r.length > 0 else "n/a (face not aft)"),
        ]
        return "\n".join(lines)

    def shape_label(self) -> str:
        return "quadrilateral" if self.shape.is_quadrilateral else "pentagon"

    def _text(self, name: str, text: str, x: float, y: float, size: int, top: bool, font: str = "arial") -> None:
        """Left-justified text at a normalised viewport position (so it survives window resizing)."""
        actor = self.plotter.add_text(text, position=(0, 0), font_size=size, font=font, color="black",
                                      name=name, render=False)
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(x, y)
        prop = actor.GetTextProperty()
        prop.SetJustificationToLeft()
        prop.SetVerticalJustification(2 if top else 0)  # 2 = top, 0 = bottom

    def _draw_text(self) -> None:
        self._text("readout", self.readout_text(), READOUT_POS[0], READOUT_POS[1], READOUT_FONT_SIZE, top=True,
                   font="courier")
        check = self.scene.engine_check
        passed = [check.message] if check is not None and check.status == "PASS" else []
        fit_warnings = [f"WARN {w}" for w in self.fit.warnings] if self.fit is not None else []
        notes = passed + self.scene.warnings + fit_warnings + ([self.status] if self.status else [])
        self._text("status", "\n".join([HELP] + notes), STATUS_POS[0], STATUS_POS[1], 8, top=False)

    def _refit(self) -> None:
        if self.centroid is None:
            self.fit = None
        else:
            self.fit = fit_inlet(self.scene.fuselage, self.centroid, self.tilt_deg, self.shape, self.area,
                                 self.skin_fit)

    def update(self) -> None:
        self._refit()
        self._draw_inlets()
        self._draw_text()
        self.plotter.render()

    # ---- controls (need an interactive window) ------------------------------------------------------
    def _set_slider(self, key: str, value: float) -> None:
        widget = self.sliders.get(key)
        if widget is not None:
            widget.GetSliderRepresentation().SetValue(float(value))

    def add_controls(self) -> None:
        p = self.plotter
        length = self.scene.fuselage.bounds[1] - self.scene.fuselage.bounds[0]
        nudge = NUDGE_RANGE_FRACTION * length
        rows = [
            *((k, SHAPE_SLIDERS[k], getattr(self.shape, k)) for k in SHAPE_SLIDERS),
            ("tilt_deg", TILT_RANGE_DEG, self.tilt_deg),
            ("nudge_x", (-nudge, nudge), 0.0),
            ("nudge_y", (-nudge, nudge), 0.0),
        ]
        for i, (key, rng, value) in enumerate(rows):
            y = SLIDER_TOP - i * SLIDER_STEP
            self.sliders[key] = p.add_slider_widget(
                lambda v, k=key: self.set_param(k, v), rng=rng, value=value, title=key,
                pointa=(SLIDER_X[0], y), pointb=(SLIDER_X[1], y), interaction_event="always",
                style="modern", title_height=SLIDER_TITLE_HEIGHT, slider_width=SLIDER_WIDTH,
                tube_width=SLIDER_TUBE_WIDTH, fmt="%.3f", color="black",
            )

        toggles = [("context", "context"), ("engine", "engine"), ("fuselage", "fuselage")]
        for i, (key, label) in enumerate(toggles):
            if key not in self.actors:
                continue
            p.add_checkbox_button_widget(
                lambda on, k=key: (self.actors[k].SetVisibility(bool(on)), p.render()),
                value=True, position=(10, 90 + 40 * i), size=25, color_on="gray", color_off="white",
            )
            p.add_text(label, position=(45, 93 + 40 * i), font_size=8, color="black", name=f"toggle_{key}")

        p.enable_surface_point_picking(
            callback=lambda point, *_: self.pick(point), show_message=False, left_clicking=True,
            show_point=False, clear_on_no_selection=False,
        )
        p.add_key_event("s", self.save)
        p.add_key_event("r", self.reload)
        p.add_key_event("f", self.cycle_skin_fit)
        views = {"1": "xz", "2": "xy", "3": "yz", "4": "iso"}
        for key, view in views.items():
            p.add_key_event(key, lambda v=view: self._set_view(v))

    def _set_view(self, view: str) -> None:
        self.plotter.camera_position = view
        self.plotter.reset_camera()
        self.plotter.render()
