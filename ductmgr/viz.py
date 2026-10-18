# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Scene loading and 3D drawing for the inlet explorer (Phase 2).

``load_scene`` reads the model and the Phase 1 sizing. ``SceneView`` draws the scene on any PyVista plotter
(an embedded Qt one, or an off-screen one for screenshots); it has no controls of its own and draws no text:
the panel and status bar belong to the window. Inlet state lives in ``explore_model.InletModel``.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker

from ductmgr import vspio
from ductmgr.config import resolve_model_path
from ductmgr.engine import EngineSizingWarning, SizingResult, size_from_config
from ductmgr.geometry import mirror_xz

# ---- appearance (dark theme) ---------------------------------------------------------------
BACKGROUND = "#15181C"
ACCENT = "#2BB3C0"
FUSELAGE_COLOR, FUSELAGE_OPACITY = "#6f9bb8", 0.32
CONTEXT_COLOR, CONTEXT_OPACITY = "#8b949e", 0.55
ENGINE_COLOR, ENGINE_OPACITY = "#aab4bd", 0.45
FACE_COLOR = "#e5534b"
RIGHT_COLOR, LEFT_COLOR = "#f0a030", ACCENT
INLET_OPACITY = 0.7
OUTLINE_COLOR = "#e6edf3"
RULER_COLOR = "#9aa4ad"
TANGENT_COLOR, TANGENT_OPACITY = "#3fb950", 0.18
GUIDE_COLOR = "#9aa4ad"
DISC_RESOLUTION = (4, 64)  # radial, circumferential
ENVELOPE_COLOR, ENVELOPE_OPACITY, ENVELOPE_SEGMENTS = "#e5534b", 0.4, 96
TANGENT_PATCH_SCALE = 1.6  # tangent-plane square, in units of the inlet's largest dimension
PICK_TOLERANCE = 0.0005  # vtkCellPicker tolerance (fraction of the render window diagonal)

VIEWS = {"side": "xz", "top": "xy", "front": "yz", "iso": "iso"}


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


class SceneView:
    """Draws the scene and the inlets on a PyVista plotter. Holds no inlet state and adds no widgets."""

    INLET_ACTORS = ("inlet_right_fill", "inlet_right_edge", "inlet_left_fill", "inlet_left_edge", "tangent",
                    "to_face")

    def __init__(self, plotter, scene: Scene):
        self.plotter = plotter
        self.scene = scene
        self.actors: dict[str, object] = {}
        self._draw_static()

    # ---- static scene ----------------------------------------------------------------------------------
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
            ticks="both", minor_ticks=True, grid=False, location="outer", color=RULER_COLOR, font_size=9,
            fmt="%.1f", n_xlabels=5, n_ylabels=4, n_zlabels=3,
        )
        p.add_axes()
        p.camera_position = "iso"

    # ---- inlets ------------------------------------------------------------------------------------------
    def update_inlet(self, model) -> None:
        """Redraw the right inlet, its mirror image, the tangent patch and the guide line from the model."""
        p = self.plotter
        fit = model.fit
        if fit is None:
            for name in self.INLET_ACTORS:
                p.remove_actor(name, render=False)
            return
        right = fit.points
        for side, pts, color in (("right", right, RIGHT_COLOR), ("left", mirror_xz(right, reverse=True), LEFT_COLOR)):
            fill, outline = _polygon_mesh(pts, fit.planar)
            p.add_mesh(fill, color=color, opacity=INLET_OPACITY, name=f"inlet_{side}_fill", pickable=False,
                       show_edges=False, render=False)
            p.add_mesh(outline, color=OUTLINE_COLOR, line_width=2, name=f"inlet_{side}_edge", pickable=False,
                       render=False)

        size = TANGENT_PATCH_SCALE * float(np.ptp(fit.uv, axis=0).max())
        patch = pv.Plane(center=model.centroid, direction=model.surface_normal, i_size=size, j_size=size)
        p.add_mesh(patch, color=TANGENT_COLOR, opacity=TANGENT_OPACITY, name="tangent", pickable=False,
                   render=False)
        face = self.scene.face_center
        if face is not None:
            p.add_mesh(pv.Line(model.centroid, face), color=GUIDE_COLOR, line_width=1, name="to_face",
                       pickable=False, render=False)
        else:
            p.remove_actor("to_face", render=False)

    # ---- view controls --------------------------------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.actors

    def set_visible(self, name: str, visible: bool) -> None:
        if name in self.actors:
            self.actors[name].SetVisibility(bool(visible))

    def is_visible(self, name: str) -> bool:
        return name in self.actors and bool(self.actors[name].GetVisibility())

    def set_view(self, name: str) -> None:
        self.plotter.camera_position = VIEWS[name]
        self.plotter.reset_camera()

    # ---- picking ------------------------------------------------------------------------------------------------
    def pick_fuselage(self, x: int, y: int) -> np.ndarray | None:
        """Point on the fuselage under the display position (x, y) in render-window pixels, else None.

        Only the fuselage can be hit: the context, engine and inlet actors are not in the pick list.
        """
        picker = vtkCellPicker()
        picker.SetTolerance(PICK_TOLERANCE)
        picker.InitializePickList()
        picker.AddPickList(self.actors["fuselage"])
        picker.PickFromListOn()
        if not picker.Pick(int(x), int(y), 0, self.plotter.renderer) or picker.GetCellId() < 0:
            return None
        return np.array(picker.GetPickPosition())
