# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Inlet skin fit on an analytic test body: a cylinder of radius 1 m along X (x = 0..8)."""
import numpy as np
import pyvista as pv
import pytest

from ductmgr import skinfit
from ductmgr.geometry import InletShape, inlet_frame
from ductmgr.skinfit import FLUSH_WARNING, SKIN_FIT_MODES, fit_inlet, skin_profile

RADIUS = 1.0
ANCHOR = np.array([4.0, 0.0, 1.0])  # on top of the cylinder
SHAPE = InletShape(width_bottom=1.0, width_top=0.7, height=1.0, apex_height=0.3, skew=0.0)
TOLERANCE = 5e-3  # the required 0.5 % area tolerance


@pytest.fixture(scope="module")
def cylinder():
    return pv.Cylinder(center=(4.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), radius=RADIUS, height=8.0,
                       resolution=360).triangulate().clean()


def frontal_area(points):
    """Outline area projected normal to the freestream (+X): shoelace on (y, z), written out independently."""
    y, z = points[:, 1], points[:, 2]
    return 0.5 * abs(np.dot(y, np.roll(z, -1)) - np.dot(np.roll(y, -1), z))


def radial_error(points):
    """Distance of points from the cylinder surface (radius 1 about the X axis)."""
    return np.abs(np.hypot(points[:, 1], points[:, 2]) - RADIUS)


# ---- skin profile ------------------------------------------------------------------------------------
def test_profile_follows_the_circle_and_stops_before_the_side_wall(cylinder):
    p = skin_profile(cylinder, ANCHOR, tilt_deg=0.0, half_range=1.3)
    assert p.valid and p.covers(-0.8, 0.8)
    u = np.linspace(-0.8, 0.8, 33)
    assert p.v_at(u) == pytest.approx(np.sqrt(1.0 - u**2) - 1.0, abs=5e-4)
    assert 0.9 < p.u.max() < 1.0 and -1.0 < p.u.min() < -0.9  # does not jump to the underside branch
    assert p.v_at(0.0) == pytest.approx(0.0, abs=1e-6)  # the anchor is on the skin


def test_profile_of_a_plane_that_misses_the_body_is_invalid(cylinder):
    far = skin_profile(cylinder, [20.0, 0.0, 1.0], 0.0, 1.0)
    assert not far.valid and far.v_at(0.3) == 0.0 and not far.covers(-0.1, 0.1)


# ---- bottom_edge -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("tilt", [0.0, 25.0, -15.0])
@pytest.mark.parametrize("target", [0.05, 0.12, 0.3])
def test_bottom_edge_follows_the_skin_stays_planar_and_matches_the_area(cylinder, tilt, target):
    out = fit_inlet(cylinder, ANCHOR, tilt, SHAPE, target, "bottom_edge")
    u_hat, v_hat, n = inlet_frame(tilt)

    assert out.planar and (out.points - ANCHOR) @ n == pytest.approx(np.zeros(len(out.points)), abs=1e-9)
    assert out.max_gap < 1e-3  # zero gap on the cylinder, up to the facet sag
    assert abs(frontal_area(out.points) / target - 1.0) < TOLERANCE
    assert abs(out.area / target - 1.0) < TOLERANCE and abs(out.area_error) < TOLERANCE
    assert out.target_area == target and out.mode == "bottom_edge"
    assert not any("did not converge" in w or "do not reach" in w for w in out.warnings)

    # the bottom edge spans the lower vertices' lateral positions, centred on the anchor, and lies on the skin
    lateral = (out.points - ANCHOR) @ u_hat
    assert lateral[0] == pytest.approx(-out.scale * SHAPE.width_bottom / 2.0, abs=1e-12)
    n_bottom = int(np.argmax(lateral))  # last bottom point = bottom-right vertex
    assert lateral[n_bottom] == pytest.approx(out.scale * SHAPE.width_bottom / 2.0, abs=1e-12)
    assert radial_error(out.points[: n_bottom + 1]).max() < 1e-3 if tilt == 0.0 else out.max_gap < 1e-3


def test_bottom_edge_keeps_the_shape_ratios_above_the_skin(cylinder):
    out = fit_inlet(cylinder, ANCHOR, 0.0, SHAPE, 0.12, "bottom_edge")
    u_hat, v_hat, _ = inlet_frame(0.0)
    s = out.scale
    uv = out.uv
    top = uv[-3:]  # top-right, apex, top-left
    assert top[0] == pytest.approx([s * 0.7 / 2.0, s * 1.0], abs=1e-9)  # (wt/2, h) above the skin at the centre
    assert top[1] == pytest.approx([0.0, s * 1.0 * 1.3], abs=1e-9)  # apex: h * (1 + apex_height)
    assert top[2] == pytest.approx([-s * 0.7 / 2.0, s * 1.0], abs=1e-9)
    assert len(uv) > 5  # the bottom edge is a curve, not a straight segment


def test_bottom_edge_opening_height_matches_an_analytic_value(cylinder):
    out = fit_inlet(cylinder, ANCHOR, 0.0, SHAPE, 0.12, "bottom_edge")
    top = out.uv[-3:]
    expected = max(v - (np.sqrt(1.0 - u**2) - 1.0) for u, v in top)  # height above the circle at each vertex
    assert out.opening_height == pytest.approx(expected, abs=1e-3)


def test_bottom_edge_quadrilateral(cylinder):
    quad = InletShape(1.0, 0.7, 1.0, apex_height=0.0)
    out = fit_inlet(cylinder, ANCHOR, 0.0, quad, 0.12, "bottom_edge")
    assert out.uv[-2:].shape == (2, 2) and abs(out.area / 0.12 - 1.0) < TOLERANCE
    assert out.planar


def test_bottom_edge_warns_when_the_skin_does_not_cover_the_inlet(cylinder):
    wide = InletShape(width_bottom=4.0, width_top=1.0, height=0.3, apex_height=0.0)  # wider than the 2 m cylinder
    out = fit_inlet(cylinder, ANCHOR, 0.0, wide, 0.45, "bottom_edge")  # bottom edge ends up > 2 m wide
    assert out.scale * wide.width_bottom > 2.0
    assert any("skin curve does not cover" in w for w in out.warnings)


# ---- flush ------------------------------------------------------------------------------------------------------
# A flush outline needs skin that is inclined to the freestream, so it has a frontal area at all:
# the forward-upper quadrant of a sphere (radius 2 m, centre at the origin; the nose is towards -X).
SPHERE_R = 2.0
SPHERE_ANCHOR = SPHERE_R * np.array([-np.sin(np.radians(40.0)), 0.0, np.cos(np.radians(40.0))])


@pytest.fixture(scope="module")
def sphere():
    return pv.Sphere(radius=SPHERE_R, theta_resolution=240, phi_resolution=240).triangulate().clean()


@pytest.mark.parametrize("tilt", [0.0, 20.0, 35.0])
@pytest.mark.parametrize("target", [0.05, 0.15])
def test_flush_outline_lies_on_the_skin_and_matches_the_area(sphere, tilt, target):
    out = fit_inlet(sphere, SPHERE_ANCHOR, tilt, SHAPE, target, "flush")
    _, _, n = inlet_frame(tilt)
    assert np.abs(np.linalg.norm(out.points, axis=1) - SPHERE_R).max() < 1e-3  # every outline point is on the skin
    assert out.max_gap < 1e-3
    assert abs(frontal_area(out.points) / target - 1.0) < TOLERANCE  # projected normal to the freestream
    assert abs(out.area / target - 1.0) < TOLERANCE
    assert not out.planar and np.abs((out.points - SPHERE_ANCHOR) @ n).max() > 1e-3  # genuinely non-planar
    assert FLUSH_WARNING in out.warnings
    assert not any("do not reach" in w or "did not converge" in w for w in out.warnings)
    assert out.opening_height == 0.0


def test_flush_along_the_freestream_keeps_the_planar_frontal_area(sphere):
    # projecting along X changes only x, so the (y, z) outline, and its frontal area, equal the planar polygon's
    out = fit_inlet(sphere, SPHERE_ANCHOR, 0.0, SHAPE, 0.1, "flush")
    assert out.planar_area == pytest.approx(0.1, rel=1e-3)
    assert out.area == pytest.approx(out.planar_area, rel=1e-3)


def test_flush_on_skin_parallel_to_the_freestream_degrades_with_a_clear_warning(cylinder):
    out = fit_inlet(cylinder, ANCHOR, 60.0, SHAPE, 0.12, "flush")  # frontal area of an outline on the cylinder ~ 0
    assert np.isfinite(out.scale) and np.isfinite(out.points).all()
    assert any("did not converge" in w and "nearly parallel to the freestream" in w for w in out.warnings)
    assert FLUSH_WARNING in out.warnings


def test_flush_warning_text_is_exact():
    assert FLUSH_WARNING == (
        "flush inlet: low pressure recovery and boundary-layer ingestion; not recommended for a main engine inlet"
    )


def test_flush_along_the_cylinder_axis_cannot_reach_the_skin_and_says_so(cylinder):
    out = fit_inlet(cylinder, ANCHOR, 0.0, SHAPE, 0.12, "flush")  # axis parallel to the skin: degenerate
    assert any("do not reach the skin" in w for w in out.warnings)
    assert out.max_gap > 1e-2


# ---- none -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_none_is_the_planar_polygon_with_the_exact_planar_area(cylinder, tilt):
    out = fit_inlet(cylinder, ANCHOR, tilt, SHAPE, 0.12, "none")
    _, _, n = inlet_frame(tilt)
    assert out.planar and (out.points - ANCHOR) @ n == pytest.approx(np.zeros(len(out.points)), abs=1e-9)
    assert out.area == pytest.approx(0.12, rel=1e-12) and out.planar_area == pytest.approx(0.12, rel=1e-12)
    assert out.points.mean(axis=0) != pytest.approx(ANCHOR)  # vertex mean is not the area centroid...
    assert len(out.points) == 5
    assert out.max_gap > 1e-2  # ...and the bottom edge floats away from the skin
    assert out.warnings == ()


def test_none_centres_the_polygon_on_the_anchor(cylinder):
    from ductmgr.geometry import polygon_centroid

    out = fit_inlet(cylinder, ANCHOR, 0.0, SHAPE, 0.12, "none")
    u_hat, v_hat, _ = inlet_frame(0.0)
    local = np.c_[(out.points - ANCHOR) @ u_hat, (out.points - ANCHOR) @ v_hat]
    assert polygon_centroid(local) == pytest.approx([0.0, 0.0], abs=1e-12)


# ---- all modes ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode, tilt", [("none", 0.0), ("bottom_edge", 0.0), ("bottom_edge", 20.0), ("flush", 25.0)])
@pytest.mark.parametrize("shape", [SHAPE, InletShape(0.6, 1.2, 1.5, 0.0, 0.0), InletShape(1.2, 0.5, 0.8, 0.6, 0.25)])
def test_area_matches_the_target_in_every_mode(cylinder, sphere, mode, tilt, shape):
    target = 0.08
    body, anchor = (sphere, SPHERE_ANCHOR) if mode == "flush" else (cylinder, ANCHOR)
    out = fit_inlet(body, anchor, tilt, shape, target, mode)
    assert abs(out.area / target - 1.0) < TOLERANCE
    if mode != "none":  # frontal projection, computed independently of the module
        assert abs(frontal_area(out.points) / target - 1.0) < TOLERANCE


def test_mode_names_and_validation(cylinder):
    assert SKIN_FIT_MODES == ("none", "bottom_edge", "flush") and skinfit.DEFAULT_SKIN_FIT == "none"
    with pytest.raises(ValueError, match="skin_fit"):
        fit_inlet(cylinder, ANCHOR, 0.0, SHAPE, 0.1, "wrap")
    with pytest.raises(ValueError, match="positive"):
        fit_inlet(cylinder, ANCHOR, 0.0, SHAPE, 0.0, "none")
    for mode in SKIN_FIT_MODES:
        assert skinfit.validate_mode(mode) == mode


def test_the_same_inputs_give_the_same_outline(cylinder):
    a = fit_inlet(cylinder, ANCHOR, 20.0, SHAPE, 0.1, "bottom_edge")
    b = fit_inlet(cylinder, ANCHOR, 20.0, SHAPE, 0.1, "bottom_edge")
    assert np.array_equal(a.points, b.points) and a.scale == b.scale
