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

import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from ductmgr.config import load_config  # noqa: E402
from ductmgr.explore_model import InletModel  # noqa: E402
from ductmgr.qt_explorer import CLICK_TOLERANCE_PX, ExploreWindow  # noqa: E402
from ductmgr.viz import load_scene  # noqa: E402


@pytest.fixture()
def gui(example_env):
    cfg, path, _ = example_env
    cfg = dict(cfg)
    cfg["ducts"] = {**cfg["ducts"], "inlet": {**cfg["ducts"]["inlet"], "centroid": None}}
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


def test_clicking_the_geometry_does_nothing_until_place_mode_is_on(gui):
    app, win, model = gui
    pos = display_of(win.plotter, [3.0, 0.4, 1.0])
    click(win, pos)
    assert model.centroid is None and not model.dirty and win.mode_label.text() == "View"


def test_place_mode_places_on_the_next_click_and_ends(gui):
    app, win, model = gui
    win.view.set_view("top")
    win.plotter.render()
    pos = display_of(win.plotter, [3.0, 0.4, 1.4])
    win.act_place.trigger()
    assert win.mode_label.text().startswith("Placing inlet")
    click(win, pos)
    assert model.centroid is not None and model.on_surface and model.centroid[2] > 1.0
    assert win.mode_label.text() == "View" and win.windowTitle().endswith("*")
    first = model.state
    click(win, display_of(win.plotter, [4.5, 0.5, 1.4]))  # the mode ended: this click is ignored
    assert model.state == first


def test_a_drag_in_place_mode_rotates_the_camera_and_does_not_place(gui):
    app, win, model = gui
    win.view.set_view("top")
    win.plotter.render()
    pos = display_of(win.plotter, [3.0, 0.4, 1.4])
    win.act_place.trigger()
    click(win, pos, release=(pos[0] + CLICK_TOLERANCE_PX + 20, pos[1]))
    assert model.centroid is None and model.mode == "placing"  # still waiting for a real click


def test_a_click_on_the_background_keeps_place_mode(gui):
    app, win, model = gui
    win.act_place.trigger()
    click(win, (2, 2))
    assert model.centroid is None and model.mode == "placing"
    assert "No fuselage" in win.message_label.text()
    win.act_cancel.trigger()
    assert model.mode == "view"


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
