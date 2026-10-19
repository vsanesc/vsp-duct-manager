# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The Bodies list: visibility in every view, hidden bodies still in the checks, and the skin selection."""
import os

if not os.environ.get("DUCTMGR_GUI_TESTS"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv
import shapely
import yaml

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import bodies  # noqa: E402
from ductmgr import qt_explorer as qe  # noqa: E402
from ductmgr import qt_paths as qp  # noqa: E402
from ductmgr import section as sec  # noqa: E402
from ductmgr.config import EXAMPLE_CONFIG, load_config, skin_names_from_config  # noqa: E402
from ductmgr.explore_model import InletModel  # noqa: E402
from ductmgr.geometry import InletPlane  # noqa: E402
from ductmgr.viz import SceneView  # noqa: E402


class Clock:
    t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def cylinder_at(z, radius=1.0, x0=0.0, length=8.0):
    return pv.Cylinder(center=(x0 + length / 2, 0.0, z), direction=(1, 0, 0), radius=radius, height=length,
                       resolution=360).triangulate().clean()


def box(bounds):
    return pv.Box(bounds=bounds).triangulate().clean()


def scene_with(parts, engine=None):
    return SimpleNamespace(
        fuselage=parts[0][1], sizing=SimpleNamespace(capture_area_per_inlet=0.06, n_inlets=2), engine_check=None,
        warnings=[], face_center=np.array([6.0, 0.0, 0.0]), face_diameter=0.5, wings=[], parts=list(parts),
        engine_mesh=engine if engine is not None else pv.PolyData(), engine_name="Engine" if engine is not None else None,
        context_parts=list(parts[1:]), context=pv.PolyData(), hub_to_tip_ratio=0.0, envelope_diameter=None,
        face_source="config", skin_names=(parts[0][0],))


def model_for(parts, cfg_extra=None, clock=None, y=0.4, engine=None, z=0.8, zc=0.0):
    r = math.hypot(y, z)  # the origin goes onto the unit circle about the X axis (the first cylinder's skin)
    cfg = {"ducts": {"symmetry": "xz", "inlets": [{"name": "R", "origin": [4.0, y / r, zc + z / r], "mirror": True}]}}
    cfg.update(cfg_extra or {})
    return InletModel(scene_with(parts, engine), cfg, EXAMPLE_CONFIG, None, **({"clock": clock} if clock else {}))


PARTS = [("Fuselage", bodies.cylinder(1.0)), ("Fin", box((5.0, 5.4, -0.6, 0.6, -0.6, 0.6))),
         ("Missile", cylinder_at(-1.2, 0.2))]


# ---- visibility -----------------------------------------------------------------------------------------------------------------
def test_the_body_rows_list_every_geom_with_visibility_and_skin():
    m = model_for(PARTS, engine=pv.Sphere(radius=0.3, center=(6, 0, 0)))
    rows = m.body_rows()
    assert [r[0] for r in rows] == ["Fuselage", "Fin", "Missile", "Engine"]
    assert [r[3] for r in rows] == ["skin", "context", "context", "engine"]
    assert all(r[1] for r in rows) and [r[2] for r in rows] == [True, False, False, False]


def test_hiding_a_body_removes_its_outline_from_both_2d_views_and_the_3d_actor():
    m = model_for(PARTS)
    for across in (2, 1):
        assert {n for n, _, _ in m.silhouettes(across)} == {"Fuselage", "Fin", "Missile"}
    pl = pv.Plotter(off_screen=True, window_size=(300, 200))
    view = SceneView(pl, m.scene)
    view.update_bodies(m)
    assert view.is_visible("body_Fin") and view.is_visible("body_Missile")
    assert m.set_body_visible("Fin", False) and "Fin" in m.hidden
    for across in (2, 1):
        assert {n for n, _, _ in m.silhouettes(across)} == {"Fuselage", "Missile"}
    view.update_bodies(m)
    assert not view.is_visible("body_Fin") and view.is_visible("body_Missile")
    m.set_body_visible("Fin", True)
    view.update_bodies(m)
    assert view.is_visible("body_Fin")
    assert not m.set_body_visible("Nothing", False)
    pl.close()


def test_the_context_toggle_hides_every_non_skin_non_engine_body_in_all_views():
    m = model_for(PARTS, engine=pv.Sphere(radius=0.3, center=(6, 0, 0)))
    assert m.context_visible
    m.set_context_visible(False)
    assert not m.context_visible and m.hidden == {"Fin", "Missile"}  # the skin and the engine are not context
    for across in (2, 1):
        assert {n for n, _, _ in m.silhouettes(across)} == {"Fuselage", "Engine"}
    m.set_context_visible(True)
    assert m.context_visible and not m.hidden


def test_visibility_is_not_part_of_the_undo_history_or_the_dirty_flag():
    m = model_for(PARTS)
    dirty = m.dirty
    m.set_body_visible("Fin", False)
    assert m.dirty == dirty and not m.can_undo


def test_hidden_bodies_are_still_in_the_clash_checks(clock=None):
    m = model_for(PARTS)
    m.state  # a duct through the box "Fin" (x 3..3.5, z 0.2..0.9) at the inlet height
    m.set_body_visible("Fin", False)
    assert [s.name for s in m.other_solids()] == ["Fin", "Missile"]  # hidden or not: every non-skin Geom
    assert m.ensure_built()
    clashes = m.checks.intersections
    shown = model_for(PARTS)
    shown.ensure_built()
    assert clashes == shown.checks.intersections and any("Fin" in c for c in clashes)


def test_the_engine_is_never_a_clash_target():
    m = model_for(PARTS, engine=pv.Sphere(radius=0.3, center=(6, 0, 0)))
    assert "Engine" not in [s.name for s in m.other_solids()]


def test_the_silhouettes_are_cached_per_body_and_view():
    m = model_for(PARTS)
    a = m.silhouettes(2)
    assert m.silhouettes(2)[0][2] is a[0][2]  # the same rings, not recomputed
    assert m.silhouettes(1)[0][2] is not a[0][2]


def test_the_2d_views_paint_a_hidden_body_no_more(qapp_session=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    m = model_for(PARTS)
    c = qp.PathCanvas(m, "top")
    c.resize(700, 340)
    c.fit_view()

    def grab():
        img = c.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3].copy()
        rgb = np.array([int(qp.CONTEXT_LINE[i:i + 2], 16) for i in (1, 3, 5)])
        return int((np.abs(arr.astype(int) - rgb).max(axis=2) <= 12).sum())

    shown = grab()
    assert shown > 50
    m.set_context_visible(False)
    assert grab() < shown * 0.3


# ---- the skin ------------------------------------------------------------------------------------------------------------------------
def two_cylinders():
    return [("Upper", cylinder_at(1.2)), ("Lower", cylinder_at(0.0))]  # the inlet sits on top of "Upper"


def circle_union_area(d, r=1.0):
    return 2 * math.pi * r * r - (2 * r * r * math.acos(d / (2 * r)) - 0.5 * d * math.sqrt(4 * r * r - d * d))


def section_area(section):
    return shapely.Polygon(section.points).area


def test_the_default_skin_is_the_first_body_without_configuration():
    m = model_for(PARTS)
    assert m.state.skin == ("Fuselage",)


def test_one_skin_body_gives_its_own_section_and_two_give_the_union_section():
    m = model_for(two_cylinders())
    plane = InletPlane((4.0, 0.0, 0.8))
    one = m._section_for(plane, ("Lower",))
    assert section_area(one) == pytest.approx(math.pi, rel=2e-3)
    union = m._section_for(plane, ("Upper", "Lower"))
    assert section_area(union) == pytest.approx(circle_union_area(1.2), rel=2e-3)
    assert section_area(union) > section_area(one) and union.closed


def test_the_union_section_is_the_boundary_of_the_united_polygons():
    m = model_for(two_cylinders())
    plane = InletPlane((4.0, 0.0, 0.6))
    a, b = (shapely.Polygon(m._section_for(plane, (n,)).points) for n in ("Upper", "Lower"))
    union = shapely.Polygon(m._section_for(plane, ("Upper", "Lower")).points)
    assert union.symmetric_difference(a.union(b)).area < 1e-3 * union.area


def test_the_skin_solid_of_two_bodies_is_inside_either():
    m = model_for(two_cylinders())
    m.set_skin_bodies(["Upper", "Lower"])
    solid = m.fuselage_solid()
    pts = np.array([[4.0, 0.0, -0.9], [4.0, 0.0, 2.1], [4.0, 0.0, 0.6], [4.0, 0.0, 3.5]])
    inside = solid.signed(pts) < 0
    assert inside.tolist() == [True, True, True, False] and solid.closed
    assert solid.signed([[4.0, 0.0, 0.6]])[0] < solid.signed([[4.0, 0.0, 3.5]])[0]


def test_changing_the_skin_rebuilds_the_section_and_reruns_the_checks_with_undo():
    clock = Clock()
    m = model_for(two_cylinders(), clock=clock, y=0.3, zc=1.2)
    area_one = section_area(m.section)
    m.ensure_built()
    assert m.checks_current
    clock.advance(10.0)
    assert m.set_skin_bodies(["Upper", "Lower"])
    assert m.state.skin == ("Upper", "Lower") and "Skin" in m.message
    assert section_area(m.section) > area_one * 0.99 and not m.checks_current  # the section follows the new skin
    assert m.dirty and m.can_undo
    m.ensure_built()
    assert m.checks_current
    assert m.undo() and m.state.skin == ("Upper",)
    assert section_area(m.section) == pytest.approx(area_one, rel=1e-9)
    assert m.redo() and m.state.skin == ("Upper", "Lower")


def test_the_skin_needs_a_body_and_unchanged_is_not_a_change():
    m = model_for(PARTS)
    assert not m.set_skin_bodies([]) and "at least one" in m.message
    assert not m.set_skin_bodies(["Fuselage"]) and not m.can_undo
    assert not m.toggle_skin_body("Fuselage", False)  # the last skin body cannot be unchecked


def test_a_body_named_like_a_store_is_a_normal_body_the_user_may_make_the_skin():
    m = model_for(PARTS)
    assert [r[3] for r in m.body_rows()][2] == "context"  # nothing special by name
    assert m.set_skin_bodies(["Missile"]) and m.state.skin == ("Missile",)
    assert "Fuselage" in [s.name for s in m.other_solids()] and "Missile" not in [s.name for s in m.other_solids()]


def test_skin_bodies_are_never_clash_targets_and_the_rest_always_are():
    m = model_for(two_cylinders())
    m.set_skin_bodies(["Upper"])
    assert [s.name for s in m.other_solids()] == ["Lower"]
    m.set_skin_bodies(["Upper", "Lower"])
    assert m.other_solids() == []


# ---- config -----------------------------------------------------------------------------------------------------------------------------
def test_the_config_names_the_skin_by_skin_bodies_or_the_older_fuselage_name():
    assert skin_names_from_config({"model": {"fuselage_name": "F"}}) == ["F"]
    assert skin_names_from_config({"model": {"fuselage_name": "F", "skin_bodies": ["A", "B"]}}) == ["A", "B"]
    assert skin_names_from_config({}) == []
    for bad in ([], "A", [""], [1]):
        with pytest.raises(ValueError, match="skin_bodies"):
            skin_names_from_config({"model": {"skin_bodies": bad}})


def test_a_configured_skin_is_used_in_the_models_order_and_unknown_names_are_reported():
    m = model_for(two_cylinders(), {"model": {"skin_bodies": ["Lower", "Upper"]}})
    assert m.state.skin == ("Upper", "Lower")
    with pytest.raises(ValueError, match="no Geom called 'Nope'"):
        model_for(two_cylinders(), {"model": {"skin_bodies": ["Nope"]}})
    older = model_for(two_cylinders(), {"model": {"fuselage_name": "Lower"}})
    assert older.state.skin == ("Lower",)


def test_the_skin_round_trips_through_save_and_reload(tmp_path):
    target = tmp_path / "duct_config.yaml"
    shutil.copyfile(EXAMPLE_CONFIG, target)
    before = yaml.safe_load(target.read_text(encoding="utf-8"))
    cfg, _ = load_config(target)
    cfg["model"]["fuselage_name"] = "Upper"
    entry = cfg["ducts"]["inlets"][0]
    entry["origin"], entry["key_sections"] = [4.0, 0.4, 2.1165], None  # on the top of "Upper"
    scene = scene_with(two_cylinders())
    m = InletModel(scene, cfg, target, target, clock=Clock())
    assert m.state.skin == ("Upper",)
    m.set_skin_bodies(["Upper", "Lower"])
    assert m.save()
    saved = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert saved["model"]["skin_bodies"] == ["Upper", "Lower"] and saved["model"]["fuselage_name"] == before["model"]["fuselage_name"]  # not rewritten
    assert saved["model"]["vsp3_path"] == before["model"]["vsp3_path"]  # the other model keys are left alone
    cfg2, _ = load_config(target)
    again = InletModel(scene_with(two_cylinders()), cfg2, target, None)
    assert again.state.skin == ("Upper", "Lower") and not again.dirty
    m.set_skin_bodies(["Upper"])
    assert m.revert() and m.state.skin == ("Upper", "Lower") and not m.dirty


# ---- the window ----------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    return app


@pytest.fixture()
def window(qapp):
    made = []

    def factory(parts, engine=None):
        m = model_for(parts, engine=engine, clock=Clock())
        w = qe.ExploreWindow(m.scene, m, with_3d=False)
        made.append(w)
        return w, m

    yield factory
    for w in made:
        w.ask_unsaved = lambda: "discard"
        w.model._listeners.clear()
        w.close()


def cell(w, row, col):
    return w.body_table.item(row, col)


def test_the_bodies_table_lists_every_geom_with_show_and_skin_boxes(window):
    w, m = window(PARTS, pv.Sphere(radius=0.3, center=(6, 0, 0)))
    t = w.body_table
    assert t.rowCount() == 4 and [cell(w, r, 0).text() for r in range(4)] == ["Fuselage", "Fin", "Missile", "Engine"]
    checked = lambda r, c: cell(w, r, c).checkState() == QtCore.Qt.CheckState.Checked
    assert [checked(r, 1) for r in range(4)] == [True] * 4
    assert [checked(r, 2) for r in range(4)] == [True, False, False, False]
    assert not (cell(w, 3, 2).flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)  # the engine cannot be skin


def test_the_show_box_and_the_context_action_drive_the_model(window):
    w, m = window(PARTS)
    cell(w, 1, 1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert m.hidden == {"Fin"} and not m.can_undo and w.act_context.isChecked() is False
    w.act_context.setChecked(True)
    w.act_context.trigger()  # toggles to off
    assert m.hidden == {"Fin", "Missile"} and not m.context_visible
    assert cell(w, 2, 1).checkState() == QtCore.Qt.CheckState.Unchecked  # the table follows
    w.act_context.trigger()
    assert not m.hidden and cell(w, 1, 1).checkState() == QtCore.Qt.CheckState.Checked


def test_the_skin_box_changes_the_skin_with_undo_and_refuses_the_last_one(window):
    w, m = window(two_cylinders())
    cell(w, 1, 2).setCheckState(QtCore.Qt.CheckState.Checked)
    assert m.state.skin == ("Upper", "Lower") and m.dirty and w.windowTitle().endswith("*")
    w.act_undo.trigger()
    assert m.state.skin == ("Upper",) and cell(w, 1, 2).checkState() == QtCore.Qt.CheckState.Unchecked
    cell(w, 0, 2).setCheckState(QtCore.Qt.CheckState.Unchecked)  # refused: the last skin body, the box goes back
    assert m.state.skin == ("Upper",) and cell(w, 0, 2).checkState() == QtCore.Qt.CheckState.Checked
    assert "at least one" in m.message


def test_the_context_action_is_disabled_without_context_bodies(window):
    w, m = window(two_cylinders()[:1])
    assert not w.act_context.isEnabled()


def test_the_3d_view_restyles_a_body_when_it_becomes_skin_and_picks_only_visible_skin():
    m = model_for(two_cylinders())
    pl = pv.Plotter(off_screen=True, window_size=(300, 200))
    view = SceneView(pl, m.scene)
    view.update_bodies(m)
    assert [a for a in view.skin_actors()] == [view.actors["body_Upper"]]
    assert view.actors["body_Upper"].GetPickable() and not view.actors["body_Lower"].GetPickable()
    m.set_skin_bodies(["Upper", "Lower"])
    view.update_bodies(m)
    assert len(view.skin_actors()) == 2 and view.actors["body_Lower"].GetPickable()
    m.set_body_visible("Upper", False)
    view.update_bodies(m)
    assert view.skin_actors() == [view.actors["body_Lower"]]  # a hidden body cannot be picked
    pl.close()
