# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Real-display test of the embedded 3D view (needs a desktop session; it opens a window for a moment).

Opt in with:   DUCTMGR_GUI_TESTS=1 python -m pytest tests/test_qt_gui_3d.py
Qt's offscreen platform cannot host an OpenGL render window, so this runs only on a real display.
"""
import os

import pytest

if not os.environ.get("DUCTMGR_GUI_TESTS"):
    pytest.skip("set DUCTMGR_GUI_TESTS=1 to run the real-display 3D tests", allow_module_level=True)

import copy  # noqa: E402

import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from ductmgr.explore_model import InletModel  # noqa: E402
from ductmgr.qt_explorer import CLICK_TOLERANCE_PX, ExploreWindow  # noqa: E402
from ductmgr.viz import SceneView, load_scene  # noqa: E402


@pytest.fixture()
def gui(example_env):
    cfg, path, _ = example_env
    cfg = copy.deepcopy(cfg)
    inlet = cfg["ducts"]["inlet"]
    inlet["origin"] = cfg["ducts"]["key_sections"] = None  # start without an inlet
    scene = load_scene(cfg, path)
    model = InletModel(scene, cfg, path, None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    win = ExploreWindow(scene, model)
    win.show()
    for _ in range(5):
        app.processEvents()
    yield app, win, model
    win.ask_unsaved = lambda: "discard"
    win.close()


def display_of(plotter, point):
    r = plotter.renderer
    r.SetWorldPoint(*point, 1.0)
    r.WorldToDisplay()
    d = r.GetDisplayPoint()
    return int(round(d[0])), int(round(d[1]))


def click(win, pos, release=None):
    """A press at ``pos`` and a release at ``release`` (default: the same place) through the real handlers."""
    iren = win.plotter.iren.interactor
    iren.SetEventPosition(*pos)
    win._on_left_press()
    iren.SetEventPosition(*(release or pos))
    win._on_left_release()


def top_of(model, x, y):
    return float(np.array(model.scene.fuselage.ray_trace([x, y, 5.0], [x, y, -5.0])[0])[:, 2].max())


def test_clicking_the_geometry_does_nothing_until_place_mode_is_on(gui):
    app, win, model = gui
    pos = display_of(win.plotter, [3.0, 0.4, 1.0])
    click(win, pos)
    assert not model.placed and not model.dirty and win.mode_label.text() == "View"


def test_place_mode_places_on_the_next_click_and_ends(gui):
    app, win, model = gui
    win.view.set_view("top")
    win.plotter.render()
    z = top_of(model, 3.0, 0.4)
    pos = display_of(win.plotter, [3.0, 0.4, z])
    win.act_place.trigger()
    assert win.mode_label.text().startswith("Placing inlet")
    click(win, pos)
    assert model.placed and model.valid  # the plane origin on the skin, with a default sketch
    assert model.state.plane.origin[2] > 1.0 and model.state.plane.origin[0] == pytest.approx(3.0, abs=0.1)
    assert win.mode_label.text() == "View" and win.windowTitle().endswith("*")
    first = model.state
    click(win, display_of(win.plotter, [4.5, 0.5, top_of(model, 4.5, 0.5)]))  # the mode ended: ignored
    assert model.state == first


def test_a_drag_in_place_mode_rotates_the_camera_and_does_not_place(gui):
    app, win, model = gui
    win.view.set_view("top")
    win.plotter.render()
    pos = display_of(win.plotter, [3.0, 0.4, top_of(model, 3.0, 0.4)])
    win.act_place.trigger()
    click(win, pos, release=(pos[0] + CLICK_TOLERANCE_PX + 20, pos[1]))
    assert not model.placed and model.mode == "placing"  # still waiting for a real click


def test_a_click_on_the_background_keeps_place_mode(gui):
    app, win, model = gui
    win.act_place.trigger()
    click(win, (2, 2))
    assert not model.placed and model.mode == "placing"
    assert "No fuselage" in win.message_label.text()
    win.act_cancel.trigger()
    assert model.mode == "view"


def test_the_3d_view_shows_the_opening_the_section_and_the_frame_and_follows_the_sketcher(gui):
    app, win, model = gui
    win.view.set_view("top")
    win.plotter.render()
    win.act_place.trigger()
    click(win, display_of(win.plotter, [3.0, 0.4, top_of(model, 3.0, 0.4)]))
    acts = win.plotter.renderer.actors
    for name in SceneView.INLET_ACTORS:
        assert name in acts  # both openings, the section curve, the u / v / n arrows, the guide line
    before = np.asarray(acts["inlet_right_edge"].GetMapper().GetInput().points).copy()
    c = win.sketch.canvas
    c.resize(640, 480)
    c.fit_view()
    c.press(c.to_screen(model.state.sections[0].shape.vertices[3]))
    c.move(c.to_screen((0.0, 0.7)), alt=True)  # drag the apex up in the sketch
    c.release()
    after = np.asarray(win.plotter.renderer.actors["inlet_right_edge"].GetMapper().GetInput().points)
    assert after.shape == before.shape and np.abs(after - before).max() > 0.05  # the 3D opening moved with it


def test_an_invalid_sketch_turns_the_3d_opening_red(gui):
    app, win, model = gui
    win.act_place.trigger()
    win.view.set_view("top")
    win.plotter.render()
    click(win, display_of(win.plotter, [3.0, 0.4, top_of(model, 3.0, 0.4)]))
    model.drag_vertex(0, (model.state.sections[0].shape.vertices[0][0], -0.8), snap_tolerance=None)
    assert not model.valid and not win.btn_save.isEnabled()
    colour = win.plotter.renderer.actors["inlet_right_edge"].GetProperty().GetColor()
    assert colour[0] > 0.8 and colour[1] < 0.5  # red


def test_view_buttons_and_toggles_act_on_the_3d_view(gui):
    app, win, model = gui
    assert win.act_envelope.isEnabled() and win.view.is_visible("envelope")
    win.act_envelope.trigger()
    assert not win.view.is_visible("envelope")
    win.act_envelope.trigger()
    assert win.view.is_visible("envelope")
    win.view_actions["top"].trigger()
    pos = np.array(win.plotter.camera_position[0]) - np.array(win.plotter.camera_position[1])
    assert np.abs(pos).argmax() == 2  # looking down Z


def test_the_embedded_view_renders_the_dark_scene(gui):
    app, win, model = gui
    image = win.plotter.screenshot(return_img=True)
    assert image.shape[2] == 3 and (image != image[0, 0]).any()
    assert tuple(image[0, 0]) == (0x15, 0x18, 0x1C)
