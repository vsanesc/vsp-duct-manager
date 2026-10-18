# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Viewer logic, driven headless: scene loading, picking, sliders, nudges, save and reload."""
import copy
import shutil

import numpy as np
import pytest
import pyvista as pv
import yaml

from ductmgr.config import EXAMPLE_CONFIG, load_config
from ductmgr.geometry import InletShape, inlet_frame
from ductmgr.viz import ExploreApp, load_scene


@pytest.fixture(scope="module")
def scene(example_env):
    cfg, path, _ = example_env
    return load_scene(cfg, path)


@pytest.fixture()
def plotter():
    pl = pv.Plotter(off_screen=True, window_size=(640, 480))
    yield pl
    pl.close()


@pytest.fixture()
def writable(example_env, tmp_path):
    """A private-style config copy (absolute model path) that the viewer may save into."""
    _, path, _ = example_env
    target = tmp_path / "duct_config.yaml"
    shutil.copyfile(path, target)
    return target


def make_app(plotter, scene, config_path, save_target):
    cfg, _ = load_config(config_path)
    return ExploreApp(plotter, scene, cfg, config_path, save_target)


# ---- scene ----------------------------------------------------------------------------------------
def test_scene_reads_fuselage_engine_and_sizing(scene):
    assert scene.fuselage.n_cells > 100
    assert scene.engine_mesh.n_cells > 0
    assert scene.face_source == "model"
    assert scene.face_center == pytest.approx([5.0, 0.0, 0.0], abs=1e-6)
    assert scene.face_diameter == pytest.approx(0.5, abs=1e-6)
    assert scene.sizing.capture_area_per_inlet == pytest.approx(scene.sizing.capture_area_total / 2)
    assert scene.warnings == []  # example config is internally consistent


def test_engine_diameter_mismatch_warns_without_changing_anything(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["face_diameter"] = 0.6
    s = load_scene(cfg, path)
    assert any("Test Engine" in w and "0.5000" in w and "0.6000" in w for w in s.warnings)
    assert s.face_diameter == pytest.approx(0.5, abs=1e-6)  # the model value is shown, nothing is rewritten
    assert cfg["engine"]["face_diameter"] == 0.6


def test_without_engine_geom_name_the_config_face_is_used(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["engine"]["geom_name"] = None
    s = load_scene(cfg, path)
    assert s.face_source == "config"
    assert s.face_center == pytest.approx(cfg["ducts"]["engine_face"]["centre"])
    assert s.engine_mesh.n_points == 0
    assert any("geom_name is not set" in w for w in s.warnings)


def test_missing_fuselage_geom_is_a_clear_error(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["model"]["fuselage_name"] = "Nope"
    with pytest.raises(LookupError, match="Nope"):
        load_scene(cfg, path)


# ---- placement and sliders ----------------------------------------------------------------------------
def test_initial_inlet_comes_from_the_config_and_is_snapped_to_the_skin(plotter, scene, example_env):
    _, path, _ = example_env
    app = make_app(plotter, scene, path, None)
    assert app.centroid is not None and app.on_surface
    assert app.centroid[:2] == pytest.approx([2.0, 0.5])  # x, y as configured
    assert app.readout is not None and app.readout.capture_area == pytest.approx(app.area)


def test_picking_a_left_side_point_is_mirrored_to_the_right(plotter, scene, example_env):
    _, path, _ = example_env
    app = make_app(plotter, scene, path, None)
    app.pick([3.0, -0.4, 1.4])
    assert app.base_point[1] == pytest.approx(0.4)
    assert app.centroid[0] == pytest.approx(3.0) and app.centroid[1] == pytest.approx(0.4)
    assert app.centroid[2] > 1.0  # on the dorsal skin
    assert app.nudge == [0.0, 0.0]


def test_nudges_move_the_centroid_and_re_snap_to_the_surface(plotter, scene, example_env):
    _, path, _ = example_env
    app = make_app(plotter, scene, path, None)
    app.pick([3.0, 0.4, 1.4])
    z0 = app.centroid[2]
    app.set_param("nudge_x", 0.3)
    assert app.centroid[0] == pytest.approx(3.3) and app.centroid[1] == pytest.approx(0.4)
    app.set_param("nudge_y", -0.15)
    assert app.centroid[1] == pytest.approx(0.25)
    top = np.array(scene.fuselage.ray_trace([3.3, 0.25, 5.0], [3.3, 0.25, -5.0])[0])[:, 2].max()
    assert app.centroid[2] == pytest.approx(top)  # still on the skin
    assert abs(app.centroid[2] - z0) < 0.5
    app.set_param("nudge_y", -1.5)  # would cross the symmetry plane: the right inlet stays at y >= 0
    assert app.centroid[1] >= 0.0


def test_shape_and_tilt_sliders_update_the_state_and_readout(plotter, scene, example_env):
    _, path, _ = example_env
    app = make_app(plotter, scene, path, None)
    app.set_param("width_top", 1.2)
    app.set_param("apex_height", 0.0)
    app.set_param("skew", 0.2)
    app.set_param("tilt_deg", 15.0)
    assert app.shape == InletShape(width_bottom=1.0, width_top=1.2, height=1.0, apex_height=0.0, skew=0.2)
    assert app.shape_label() == "quadrilateral" and "quadrilateral" in app.readout_text()
    assert app.tilt_deg == 15.0
    app.set_param("apex_height", 0.3)
    assert app.shape_label() == "pentagon"
    with pytest.raises(KeyError):
        app.set_param("bogus", 1.0)


def test_readout_matches_hand_calculation(plotter, scene, example_env):
    _, path, _ = example_env
    app = make_app(plotter, scene, path, None)
    app.pick([2.0, 0.5, 1.4])
    c = app.centroid
    r = app.readout
    face = np.array([5.0, 0.0, 0.0])
    assert r.length == pytest.approx(3.0)
    assert r.dy == pytest.approx(0.0 - c[1]) and r.dz == pytest.approx(0.0 - c[2])
    assert r.offset == pytest.approx(np.hypot(c[1], c[2]))
    assert r.distance_to_face == pytest.approx(np.linalg.norm(face - c))
    assert r.offset_over_length == pytest.approx(np.hypot(c[1], c[2]) / 3.0)
    text = app.readout_text()
    for label in ("Capture area", "Centroid", "Aspect ratio", "Distance to face", "Offset dy, dz", "Offset / length"):
        assert label in text


def test_no_inlet_until_one_is_placed(plotter, scene, example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"]["centroid"] = None
    app = ExploreApp(plotter, scene, cfg, path, None)
    assert app.centroid is None and app.inlet_definition() is None
    assert "click on the fuselage" in app.readout_text()
    app.set_param("tilt_deg", 5.0)  # sliders are harmless before placement
    assert app.centroid is None


# ---- save / reload ---------------------------------------------------------------------------------------
def test_inlet_definition_contents(plotter, scene, example_env):
    _, path, _ = example_env
    app = make_app(plotter, scene, path, None)
    app.pick([3.0, 0.4, 1.4])
    app.set_param("tilt_deg", 20.0)
    d = app.inlet_definition()
    assert set(d) == {"centroid", "normal", "tilt_deg", "shape"}
    assert d["centroid"] == pytest.approx(list(app.centroid))
    assert d["normal"] == pytest.approx(list(inlet_frame(20.0)[2]))
    assert d["shape"] == app.shape.to_dict() and d["tilt_deg"] == 20.0


def test_save_writes_the_definition_and_keeps_the_rest(plotter, scene, writable):
    before = yaml.safe_load(writable.read_text(encoding="utf-8"))
    app = make_app(plotter, scene, writable, writable)
    app.pick([3.0, 0.4, 1.4])
    app.set_param("width_top", 0.9)
    app.set_param("tilt_deg", 12.0)
    app.save()
    assert app.status == f"Saved inlet to {writable.name}."
    after = yaml.safe_load(writable.read_text(encoding="utf-8"))
    inlet = after["ducts"]["inlet"]
    assert inlet["centroid"] == pytest.approx([round(float(v), 6) for v in app.centroid])
    assert inlet["shape"]["width_top"] == 0.9 and inlet["tilt_deg"] == 12.0
    for section in ("model", "engine", "design_point", "checks"):
        assert after[section] == before[section]


def test_reload_restores_the_saved_state(plotter, scene, writable):
    app = make_app(plotter, scene, writable, writable)
    app.pick([3.0, 0.4, 1.4])
    app.set_param("skew", -0.3)
    app.set_param("tilt_deg", 25.0)
    app.save()
    saved_centroid = app.centroid.copy()
    app.pick([4.5, 0.6, 1.4])  # wander off
    app.set_param("skew", 0.4)
    app.set_param("tilt_deg", -10.0)
    app.reload()
    assert app.status == f"Reloaded {writable.name}."
    assert app.shape.skew == pytest.approx(-0.3) and app.tilt_deg == pytest.approx(25.0)
    assert app.centroid == pytest.approx(saved_centroid, abs=1e-5)
    assert app.nudge == [0.0, 0.0]


def test_reload_picks_up_hand_edits_including_capture_area(plotter, scene, writable):
    app = make_app(plotter, scene, writable, writable)
    area_before = app.area
    cfg = yaml.safe_load(writable.read_text(encoding="utf-8"))
    cfg["design_point"]["mach"] = 0.5
    writable.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    app.reload()
    assert app.area != pytest.approx(area_before)


def test_reload_reports_errors_instead_of_crashing(plotter, scene, writable):
    app = make_app(plotter, scene, writable, writable)
    writable.write_text("engine: [unclosed\n", encoding="utf-8")
    app.reload()
    assert app.status.startswith("Reload failed")


def test_the_example_config_is_read_only_in_the_viewer(plotter, scene, example_env):
    _, path, _ = example_env
    app = make_app(plotter, scene, path, None)  # run_explore passes None for the example config
    app.pick([3.0, 0.4, 1.4])
    before = EXAMPLE_CONFIG.read_bytes()
    app.save()
    assert "read-only" in app.status
    assert EXAMPLE_CONFIG.read_bytes() == before


def test_save_before_placing_an_inlet_does_nothing(plotter, scene, example_env, writable):
    cfg, _, _ = example_env
    cfg = copy.deepcopy(cfg)
    cfg["ducts"]["inlet"]["centroid"] = None
    before = writable.read_bytes()
    app = ExploreApp(plotter, scene, cfg, writable, writable)
    app.save()
    assert app.status.startswith("Nothing to save")
    assert writable.read_bytes() == before


# ---- rendering ----------------------------------------------------------------------------------------------
def test_scene_renders_something(plotter, scene, example_env):
    _, path, _ = example_env
    make_app(plotter, scene, path, None)
    image = plotter.screenshot(return_img=True)
    assert image.shape[:2] == (480, 640)
    assert (image != 255).any()  # not an empty white frame
