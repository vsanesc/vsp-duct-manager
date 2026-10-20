# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Live checks of the duct bodies: external region, wall left, other Geoms, line of sight."""
import math

import numpy as np
import pytest
import pyvista as pv

import bodies as test_bodies
from ductmgr import duct as dm
from ductmgr import ductcheck as dc
from ductmgr import outline as ol
from ductmgr.centreline import CentrelineParams
from ductmgr.geometry import InletPlane
from ductmgr.solids import RayCaster, Solid

AXIS = np.array([1.0, 0.0, 0.0])
CYL = Solid(test_bodies.cylinder(1.0), "fuselage")  # radius 1, x = 0 .. 8, on the X axis


def circle_key(r):
    shape = ol.circle_shape(r)
    return dm.KeyOutline(0.0, ol.build_outline(ol.edge_paths(shape)[0]), np.array(shape.vertices))


def straight(z=0.0, r=0.25, wall=0.01, **kw):
    """A single-inlet duct of constant radius ``r`` flying straight at height z (inlet and face coincide in size)."""
    spec = dm.single_spec(InletPlane((1.0, 0.0, z)), (circle_key(r),), np.array([6.0, 0.0, z]), AXIS, 2 * r,
                       mirror=False, wall_thickness=wall, trunk=(circle_key(r),), **kw)
    return dm.Duct(spec)


def s_duct(offset=1.0, r=0.25):
    spec = dm.single_spec(InletPlane((1.0, 0.0, offset)), (circle_key(r),), np.array([6.0, 0.0, 0.0]), AXIS, 2 * r,
                       mirror=False, trunk=(circle_key(r),))
    return dm.Duct(spec)


def run(duct, fuselage=CYL, parts=(), **settings):
    return dc.run_checks(duct, fuselage, list(parts), dc.CheckSettings(**settings))


# ---- solids ------------------------------------------------------------------------------------------------------------------
def test_signed_distance_is_negative_inside():
    d = CYL.signed([[4, 0, 0], [4, 0, 1.5], [4, 0.5, 0.5]])
    assert d[0] == pytest.approx(-1.0, abs=1e-3) and d[1] == pytest.approx(0.5, abs=1e-3) and d[2] < 0
    assert CYL.closed


def test_an_open_mesh_with_a_few_open_ends_is_capped():
    open_cyl = pv.Cylinder(center=(4, 0, 0), direction=(1, 0, 0), radius=1.0, height=8.0, capping=False).triangulate()
    s = Solid(open_cyl)
    assert s.closed and s.healed and s.signed([[4, 0, 0]])[0] < 0


def test_the_ray_caster_sees_walls():
    rc = RayCaster(test_bodies.cylinder(1.0))
    assert rc.blocked([4, 0, 0], [4, 0, 2]) and not rc.blocked([4, 0, 0], [4, 0, 0.5])


# ---- external region -----------------------------------------------------------------------------------------------------------
def test_a_duct_flying_inside_the_skin_has_no_external_region():
    d = straight(z=0.0)
    c, _ = run(d)
    assert c.external_length == 0.0 and c.external_area == 0.0 and not c.external_unplanned


def test_a_positive_poking_out_along_the_whole_duct_is_an_unplanned_bump():
    d = straight(z=0.9, r=0.1, wall=0.02)  # the top of the positive is at z = 1.02, above the skin (z = 1)
    c, bodies = run(d)
    assert c.external_unplanned
    assert c.external_extent == pytest.approx(5.0, abs=0.2)  # all the way to the face
    assert c.external_length == pytest.approx(5.0, abs=0.2)
    assert c.external_area > 0.5
    flags = c.external_cells["pos_branch_R"]
    assert flags.any() and not flags.all()  # the top faces are outside, the bottom ones are not
    assert any("unplanned bump" in v for _, v, _ in c.findings)


def test_the_external_region_near_the_opening_is_intended_when_the_duct_dives():
    spec = dm.single_spec(InletPlane((1.0, 0.0, 0.9)), (circle_key(0.1),), np.array([6.0, 0.0, 0.0]), AXIS, 0.2,
                       mirror=False, wall_thickness=0.02)
    d = dm.Duct(spec)
    c_short, _ = run(d, lip_length=0.15)
    assert c_short.external_unplanned and c_short.external_extent > 0.15
    c_long, _ = run(d, lip_length=1.5)
    assert 0.0 < c_long.external_extent < 1.5 and not c_long.external_unplanned
    assert c_long.external_length > 0.0 and c_long.external_area > 0.0
    assert not any("unplanned" in v for _, v, _ in c_long.findings)


def test_the_external_area_is_measured_on_the_positive_surface_outside_the_skin():
    d = straight(z=0.95, r=0.1, wall=0.02)  # only the top cap of the tube pokes out
    c, _ = run(d)
    # an upper strip of the tube, roughly 5 m long and a few cm wide
    assert 0.0 < c.external_area < 5.0 * 2 * math.pi * 0.12 * 0.5


# ---- wall left --------------------------------------------------------------------------------------------------------------------
def test_the_wall_left_around_a_buried_duct_is_the_distance_to_the_skin():
    d = straight(z=0.0, r=0.25)
    c, _ = run(d)
    assert c.min_wall == pytest.approx(0.75, abs=0.01) and c.wall_ok


def test_the_external_wall_is_the_thickness_and_is_reported_separately():
    c, _ = run(straight(z=0.9, r=0.1, wall=0.01), min_clearance=0.02)
    assert c.external_wall == pytest.approx(0.01, abs=2e-3)
    assert any(lbl == "Wall (external)" and "not compared" in v for lbl, v, _ in c.findings)
    # a warning, if any, is about the internal wall (bore to skin), never about the external thickness
    assert not any("min_clearance" in v and "skin" not in v for lbl, v, _ in c.findings if lbl != "Wall (external)")


def test_the_clearance_applies_only_to_the_internal_part():
    # a bore skimming the skin has a thin internal wall: that is a real warning ...
    skimming, _ = run(straight(z=0.9, r=0.1, wall=0.004), min_clearance=0.02)
    assert skimming.min_wall is not None and skimming.min_wall < 0.02 and skimming.wall_ok is False
    assert any("between the negative and the skin" in v for _, v, _ in skimming.findings)
    # ... but the thin external lip alone is not
    prism = Solid(test_bodies.chined_prism(), "prism")
    spec = dm.single_spec(InletPlane((1.0, 0.0, 0.41 - 0.1 * math.cos(math.radians(45))), 45.0), (circle_key(0.1),),
                       np.array([3.0, 0.0, -0.3]), AXIS, 0.2, mirror=False, wall_thickness=0.004)
    ok, _ = run(dm.Duct(spec), fuselage=prism, min_clearance=0.02)
    assert ok.external_area > 0.0  # the positive does leave the skin at the lip, with a wall of 4 mm < 20 mm ...
    assert ok.min_wall >= 0.02 and ok.wall_ok  # ... and that raises no clearance warning
    assert not any(lvl == "warn" and "clearance" in v for _, v, lvl in ok.findings)


def test_the_example_config_has_independent_thickness_and_clearance():
    from ductmgr.config import EXAMPLE_CONFIG, load_config

    cfg = load_config(EXAMPLE_CONFIG)[0]
    assert cfg["ducts"]["wall_thickness"] < cfg["checks"]["min_clearance"]


def test_a_wall_too_thick_for_a_concave_section_is_an_error():
    star = ol.SectionShape.polygon([(0, 0.2), (-0.02, 0.03), (-0.2, 0), (-0.02, -0.03), (0, -0.2), (0.02, -0.03),
                                    (0.2, 0), (0.02, 0.03)])
    key = dm.KeyOutline(0.0, ol.build_outline(ol.edge_paths(star)[0]), np.array(star.vertices))
    d = dm.Duct(dm.single_spec(InletPlane((1.0, 0.0, 0.0)), (key,), np.array([6.0, 0.0, 0.0]), AXIS, 0.5, mirror=False,
                            wall_thickness=0.08))
    c, _ = run(d)
    assert any(lvl == "error" and "too large" in v for _, v, lvl in c.findings)
    ok = dm.Duct(dm.single_spec(InletPlane((1.0, 0.0, 0.0)), (key,), np.array([6.0, 0.0, 0.0]), AXIS, 0.5, mirror=False,
                             wall_thickness=0.004))
    assert not any(lvl == "error" for _, _, lvl in run(ok)[0].findings)


def test_a_missing_or_open_fuselage_skips_the_skin_checks_with_a_warning():
    c, _ = dc.run_checks(straight(), None, [], dc.CheckSettings())
    assert any(lbl == "Fuselage" and lvl == "warn" for lbl, _, lvl in c.findings) and c.min_wall is None


# ---- other Geoms ---------------------------------------------------------------------------------------------------------------------
def box(bounds, name):
    return Solid(pv.Box(bounds=bounds).triangulate().clean(), name)


def test_a_geom_crossing_the_duct_is_reported_and_a_distant_one_is_not():
    d = straight(z=0.0, r=0.25)
    crossing = box((3.0, 3.5, -0.4, 0.4, -0.1, 0.1), "Fin")
    c, _ = run(d, parts=[crossing])
    assert c.intersections and "Fin" in c.intersections[0] and c.intersecting
    far = box((3.0, 3.5, 0.6, 0.9, -0.1, 0.1), "Wing")
    c2, _ = run(d, parts=[far])
    assert not c2.intersections and not c2.intersecting


def test_a_small_geom_inside_the_duct_is_found_through_its_own_vertices():
    d = straight(z=0.0, r=0.25)
    inside = box((3.0, 3.2, -0.05, 0.05, -0.05, 0.05), "Probe")
    c, _ = run(d, parts=[inside])
    assert c.intersections


# ---- line of sight ----------------------------------------------------------------------------------------------------------------------
def test_a_straight_duct_shows_the_whole_face():
    c, _ = run(straight(z=0.0, r=0.25))
    assert c.los_x == pytest.approx(1.0) and c.los_n == pytest.approx(1.0) and c.los_any == pytest.approx(1.0)


def test_an_s_duct_with_an_offset_larger_than_the_face_diameter_hides_the_face():
    d = s_duct(offset=1.0, r=0.25)  # offset 1.0 m > face diameter 0.5 m
    c, _ = run(d)
    assert c.los_x == 0.0 and c.los_n == 0.0 and c.los_any == 0.0
    assert any("hidden" in v for _, v, _ in c.findings)


def test_a_small_offset_shows_part_of_the_face():
    c, _ = run(s_duct(offset=0.2, r=0.25))
    assert 0.0 < c.los_x < 1.0


# ---- numbers ---------------------------------------------------------------------------------------------------------------------------------
def test_offset_over_length_curvature_and_cone_angles_are_reported():
    d = s_duct(offset=0.6)
    c, _ = run(d)
    assert c.offset == pytest.approx(0.6) and c.offset_over_length == pytest.approx(0.6 / d.centreline.length)
    assert c.max_curvature > 0.0
    labels = [lbl for lbl, _, _ in c.findings]
    assert "Offset / length" in labels and "Centreline curvature" in labels and "Cone half-angle branch" in labels


def test_the_cone_half_angle_limit_warns():
    d = dm.Duct(dm.single_spec(InletPlane((1.0, 0.0, 0.0)), (circle_key(0.1),), np.array([2.0, 0.0, 0.0]), AXIS, 0.6,
                            mirror=False))
    c, _ = run(d, max_half_angle_deg=2.0)
    assert c.half_angle_exceeded and any("half-angle exceeds" in v for _, v, _ in c.findings)
    ok, _ = run(d, max_half_angle_deg=60.0)
    assert not ok.half_angle_exceeded


def test_a_shrinking_area_law_is_flagged():
    from ductmgr.areacurve import AreaPoints

    spec = dm.single_spec(InletPlane((1.0, 0.0, 0.0)), (circle_key(0.2),), np.array([6.0, 0.0, 0.0]), AXIS, 0.5,
                       mirror=False, area_points=AreaPoints(None, ((0.4, 0.2), (0.7, 0.05)), None))
    c, _ = run(dm.Duct(spec))
    assert "branch" in c.area_decreasing and any("A(s) decreases" in v for _, v, _ in c.findings)


def test_settings_from_config_with_defaults():
    s = dc.CheckSettings.from_config({"checks": {"min_clearance": 0.03, "lip_length": 0.2}})
    assert s.min_clearance == 0.03 and s.lip_length == 0.2 and s.max_half_angle_deg == dc.DEFAULT_MAX_HALF_ANGLE_DEG
    assert dc.CheckSettings.from_config({}) == dc.CheckSettings()
