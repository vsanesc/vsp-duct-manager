# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Scene loading and the 3D drawing/picking layer, driven headless on an off-screen plotter."""
import copy

import numpy as np
import pytest
import pyvista as pv

from ductmgr.explore_model import InletModel
from ductmgr.config import load_config
from ductmgr.viz import BACKGROUND, ERROR_COLOR, VIEWS, SceneView, load_scene

WINDOW = (640, 480)


@pytest.fixture(scope="module")
def scene(example_env):
    cfg, path, _ = example_env
    return load_scene(cfg, path)


@pytest.fixture()
def plotter():
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    yield pl
    pl.close()


def model_for(scene, example_env, placed=True):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    if not placed:
        cfg["ducts"]["inlet"]["origin"] = cfg["ducts"]["key_sections"] = None
    return InletModel(scene, cfg, path, None)


# ---- scene ----------------------------------------------------------------------------------------
def test_scene_reads_fuselage_engine_and_sizing(scene):
    assert scene.fuselage.n_cells > 100
    assert scene.engine_mesh.n_cells > 0
    assert scene.face_source == "model"
    assert scene.face_center == pytest.approx([5.0, 0.0, 0.0], abs=1e-6)  # front end of the engine Geom
    assert scene.face_diameter == 0.5  # the fan: engine.face_diameter from the config
    assert scene.envelope_diameter == pytest.approx(0.575, abs=1e-6)  # the Geom's bounding box (casing)
    assert scene.sizing.capture_area_per_inlet == pytest.approx(scene.sizing.capture_area_total / 2)
    assert scene.warnings == []  # example config is internally consistent


def test_face_to_envelope_check_passes_for_the_example(scene):
    c = scene.engine_check
    assert c.status == "PASS" and c.ratio == pytest.approx(0.5 / 0.575)
    assert c.face_diameter == 0.5 and c.envelope_diameter == pytest.approx(0.575, abs=1e-6)
    assert c.ratio_range == (0.80, 0.98)
    assert "0.5000 m (config)" in c.message and "0.5750 m (model)" in c.message and "PASS" in c.message


@pytest.mark.parametrize("face_diameter, ratio", [(0.4, 0.4 / 0.575), (0.57, 0.57 / 0.575)])
def test_face_diameter_outside_the_band_warns_and_is_never_replaced(example_env, face_diameter, ratio):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["face_diameter"] = face_diameter
    s = load_scene(cfg, path)
    assert s.engine_check.status == "WARN" and s.engine_check.ratio == pytest.approx(ratio)
    assert any(w.startswith("WARN") and "Test Engine" in w and f"{face_diameter:.4f} m (config)" in w
               and "0.5750 m (model)" in w for w in s.warnings)
    assert s.face_diameter == face_diameter  # the config value is used as given; the bbox never replaces it
    assert cfg["engine"]["face_diameter"] == face_diameter


def test_face_to_envelope_range_comes_from_the_config_with_a_default(example_env):
    cfg, path, _ = example_env
    narrow = copy.deepcopy(cfg)
    narrow["engine"]["face_to_envelope_range"] = [0.90, 0.98]  # the example's 0.870 now falls outside
    assert load_scene(narrow, path).engine_check.status == "WARN"
    default = copy.deepcopy(cfg)
    del default["engine"]["face_to_envelope_range"]
    s = load_scene(default, path)
    assert s.engine_check.ratio_range == (0.80, 0.98) and s.engine_check.status == "PASS"


def test_without_engine_geom_name_the_config_face_is_used(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["geom_name"] = None
    s = load_scene(cfg, path)
    assert s.face_source == "config"
    assert s.face_center == pytest.approx(cfg["ducts"]["engine_face"]["centre"])
    assert s.face_diameter == 0.5
    assert s.engine_mesh.n_points == 0
    assert s.engine_check is None and s.envelope_diameter is None  # no Geom, so no envelope to compare
    assert any("geom_name is not set" in w for w in s.warnings)


def test_missing_fuselage_geom_is_a_clear_error(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["model"]["fuselage_name"] = "Nope"
    with pytest.raises(LookupError, match="Nope"):
        load_scene(cfg, path)


# ---- drawing ----------------------------------------------------------------------------------------
def _extent(actor, axis):
    b = actor.GetMapper().GetInput().GetBounds()
    return b[2 * axis + 1] - b[2 * axis]


def test_face_disc_uses_the_config_diameter_and_the_envelope_is_an_outline(plotter, scene):
    view = SceneView(plotter, scene)
    face, envelope = view.actors["face"], view.actors["envelope"]
    assert _extent(face, 1) == pytest.approx(0.5, abs=1e-6) and _extent(face, 2) == pytest.approx(0.5, abs=1e-6)
    assert _extent(envelope, 1) == pytest.approx(0.575, abs=1e-3)
    assert _extent(envelope, 2) == pytest.approx(0.575, abs=1e-3)
    assert envelope.GetProperty().GetOpacity() < 0.5  # faint
    assert envelope.GetMapper().GetInput().n_lines > 0 and envelope.GetMapper().GetInput().n_faces == 0
    for actor in (face, envelope):  # both sit in the front-end plane of the engine Geom
        b = actor.GetMapper().GetInput().GetBounds()
        assert b[0] == pytest.approx(5.0, abs=1e-6) and b[1] == pytest.approx(5.0, abs=1e-6)


def test_no_envelope_outline_without_an_engine_geom(plotter, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["geom_name"] = None
    view = SceneView(plotter, load_scene(cfg, path))
    assert "face" in view.actors and "envelope" not in view.actors


def test_the_3d_view_holds_no_widgets_and_no_text(plotter, scene, example_env):
    view = SceneView(plotter, scene)
    view.update_inlet(model_for(scene, example_env))
    assert not plotter.widgets.slider_widgets and not plotter.widgets.button_widgets
    assert not any("TextActor" in type(a).__name__ for a in plotter.renderer.actors.values())


def test_inlets_are_drawn_and_removed_from_the_model(plotter, scene, example_env):
    view = SceneView(plotter, scene)
    names = set(plotter.renderer.actors)
    assert not set(SceneView.INLET_ACTORS) & names  # nothing before an inlet exists
    view.update_inlet(model_for(scene, example_env))
    assert set(SceneView.INLET_ACTORS) <= set(plotter.renderer.actors)
    view.update_inlet(model_for(scene, example_env, placed=False))
    assert not set(SceneView.INLET_ACTORS) & set(plotter.renderer.actors)


def test_the_left_inlet_mirrors_the_right_one(plotter, scene, example_env):
    view = SceneView(plotter, scene)
    view.update_inlet(model_for(scene, example_env))
    # each outline is closed by repeating its first point; the left one is the mirrored vertices in reverse order
    right = np.asarray(plotter.renderer.actors["inlet_right_edge"].GetMapper().GetInput().points)[:-1]
    left = np.asarray(plotter.renderer.actors["inlet_left_edge"].GetMapper().GetInput().points)[:-1]
    assert left[::-1][:, 1] == pytest.approx(-right[:, 1], abs=1e-6)  # y negated (float32 actor points)
    assert left[::-1][:, [0, 2]] == pytest.approx(right[:, [0, 2]], abs=1e-6)


def test_the_opening_is_drawn_on_the_inlet_plane(plotter, scene, example_env):
    view = SceneView(plotter, scene)
    model = model_for(scene, example_env)
    view.update_inlet(model)
    plane = model.state.plane
    right = np.asarray(plotter.renderer.actors["inlet_right_edge"].GetMapper().GetInput().points)[:-1]
    assert (right - plane.origin) @ plane.normal == pytest.approx(np.zeros(len(right)), abs=1e-5)  # in the plane
    assert right == pytest.approx(plane.to_3d(model.opening.outline), abs=1e-5)  # at the sketched (u, v)


def test_the_section_curve_and_the_plane_frame_are_drawn(plotter, scene, example_env):
    view = SceneView(plotter, scene)
    model = model_for(scene, example_env)
    view.update_inlet(model)
    acts = plotter.renderer.actors
    for name in ("section_curve", "frame_u", "frame_v", "frame_n"):
        assert name in acts
    section = np.asarray(acts["section_curve"].GetMapper().GetInput().points)
    plane = model.state.plane
    assert (section - plane.origin) @ plane.normal == pytest.approx(np.zeros(len(section)), abs=1e-5)
    assert len(section) == len(model.section.points) + 1  # a closed loop
    for name, axis in zip(("frame_u", "frame_v", "frame_n"), plane.frame()):  # arrows start at O along u, v, n
        pts = np.asarray(acts[name].GetMapper().GetInput().points)
        along = (pts - plane.origin) @ axis
        assert along.min() == pytest.approx(0.0, abs=1e-5)  # the arrow starts at the origin...
        assert along.max() > 0.05  # ...and points along its axis
        across = np.linalg.norm((pts - plane.origin) - np.outer(along, axis), axis=1)
        assert across.max() < 0.05  # a thin arrow: nothing strays sideways


def test_an_invalid_opening_is_drawn_red(plotter, scene, example_env):
    view = SceneView(plotter, scene)
    model = model_for(scene, example_env)
    view.update_inlet(model)
    ok_colour = plotter.renderer.actors["inlet_right_edge"].GetProperty().GetColor()
    model.drag_vertex(0, (model.state.sections[0].shape.vertices[0][0], -0.8), snap_tolerance=None)  # inside the fuselage
    assert not model.valid
    view.update_inlet(model)
    red = tuple(int(ERROR_COLOR[i:i + 2], 16) / 255 for i in (1, 3, 5))
    for name in ("inlet_right_edge", "inlet_right_fill", "inlet_left_edge", "inlet_left_fill"):
        assert plotter.renderer.actors[name].GetProperty().GetColor() == pytest.approx(red, abs=1e-3)
    assert ok_colour != pytest.approx(red, abs=1e-3)


def test_visibility_toggles(plotter, scene):
    view = SceneView(plotter, scene)
    assert view.has("envelope") and view.is_visible("envelope")
    view.set_visible("envelope", False)
    assert not view.is_visible("envelope")
    view.set_visible("envelope", True)
    assert view.is_visible("envelope")
    assert not view.has("context") and not view.is_visible("context")  # the example model has no other geoms
    view.set_visible("context", False)  # unknown names are harmless


@pytest.mark.parametrize("name", list(VIEWS))
def test_standard_views(plotter, scene, name):
    view = SceneView(plotter, scene)
    view.set_view(name)
    pos = np.array(plotter.camera_position[0]) - np.array(plotter.camera_position[1])
    axis = {"side": 1, "top": 2, "front": 0}.get(name)
    if axis is not None:  # looking along one axis: the camera offset is aligned with it
        assert np.abs(pos).argmax() == axis


# ---- picking and rendering ------------------------------------------------------------------------------
def _display(plotter, point):
    r = plotter.renderer
    r.SetWorldPoint(*point, 1.0)
    r.WorldToDisplay()
    d = r.GetDisplayPoint()
    return int(round(d[0])), int(round(d[1]))


def test_only_the_fuselage_can_be_picked(plotter, scene):
    view = SceneView(plotter, scene)
    view.set_view("top")
    plotter.render()
    p = np.array([3.0, 0.4, 1.4])  # near the dorsal skin of the example fuselage
    top = np.array(scene.fuselage.ray_trace([p[0], p[1], 5.0], [p[0], p[1], -5.0])[0])[:, 2].max()
    hit = view.pick_fuselage(*_display(plotter, [p[0], p[1], top]))
    assert hit is not None and hit[:2] == pytest.approx([3.0, 0.4], abs=0.05) and hit[2] == pytest.approx(top, abs=0.05)
    assert view.pick_fuselage(1, 1) is None  # background
    # the engine pod is inside the translucent fuselage; picking its front face must still return the fuselage skin
    engine_xy = _display(plotter, [5.0, 0.0, 0.0])
    h = view.pick_fuselage(*engine_xy)
    assert h is not None and abs(h[2]) > 0.5  # the skin, not the engine at z = 0


def test_the_scene_renders_on_the_dark_background(plotter, scene, example_env):
    view = SceneView(plotter, scene)
    view.update_inlet(model_for(scene, example_env))
    image = plotter.screenshot(return_img=True)
    assert image.shape[:2] == (WINDOW[1], WINDOW[0])
    bg = tuple(int(BACKGROUND[i:i + 2], 16) for i in (1, 3, 5))
    assert tuple(image[2, 2]) == bg  # an untouched corner is the theme background
    assert (image != np.array(bg)).any()  # and something was drawn
