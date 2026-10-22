# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The export plan without OpenVSP: frames and Stack parameters, the Edit Curve fit of corresponding rings, the adaptive
choice of XSecs, the settings."""
import math

import numpy as np
import pytest

from ductmgr import cedit
from ductmgr import sectionfit as sf
from ductmgr import vspexport as vx
from ductmgr.geometry import polygon_area


# ---- frames and Stack parameters ----------------------------------------------------------------------------------------------
def random_rotation(rng):
    a, b, c = rng.uniform(-80, 80, 3)
    return vx.rot_x(math.radians(a)) @ vx.rot_y(math.radians(b)) @ vx.rot_z(math.radians(c))


def test_euler_angles_reproduce_the_rotation():
    rng = np.random.default_rng(3)
    for _ in range(50):
        m = random_rotation(rng)
        a, b, c = (math.radians(x) for x in vx.euler_xyz(m))
        assert np.allclose(vx.rot_x(a) @ vx.rot_y(b) @ vx.rot_z(c), m, atol=1e-12)


def test_euler_angles_at_gimbal_lock_still_reproduce_the_rotation():
    m = vx.rot_x(math.radians(30)) @ vx.rot_y(math.radians(90))
    a, b, c = (math.radians(x) for x in vx.euler_xyz(m))
    assert np.allclose(vx.rot_x(a) @ vx.rot_y(b) @ vx.rot_z(c), m, atol=1e-9)


class Sec:
    def __init__(self, origin, R):
        self.origin, self.R = np.asarray(origin, float), R


def test_stack_parameters_replay_to_the_same_positions_and_frames():
    rng = np.random.default_rng(5)
    secs = [Sec(rng.uniform(-3, 3, 3), random_rotation(rng)) for _ in range(12)]
    for (p, R), s in zip(vx.replay_stack(vx.stack_parameters(secs)), secs):
        assert np.allclose(p, s.origin, atol=1e-12) and np.allclose(R, s.R, atol=1e-12)


def test_a_duct_flowing_along_x_with_v_up_is_the_identity_frame():
    T, V = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    U = np.cross(T, V)
    R = vx.frame_matrix(T, U, V)
    assert np.allclose(R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_the_frame_of_a_station_is_a_proper_rotation():
    rng = np.random.default_rng(7)
    for _ in range(20):
        T = rng.normal(size=3)
        T /= np.linalg.norm(T)
        V = np.cross(T, rng.normal(size=3))
        V /= np.linalg.norm(V)
        R = vx.frame_matrix(T, np.cross(T, V), V)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)


# ---- the Edit Curve fit ----------------------------------------------------------------------------------------------------------
def ellipse_ring(a, b, n=64, phase=0.0):
    t = phase - 2 * math.pi * np.arange(n) / n  # clockwise, like OpenVSP's own curves
    return np.c_[a * np.cos(t), b * np.sin(t)]


def pentagon_ring(n=64):
    """A pentagon resampled at n equal arc-length steps (its corners fall between samples), clockwise."""
    v = np.array([(0.0, 1.0), (0.95, 0.31), (0.59, -0.81), (-0.59, -0.81), (-0.95, 0.31)])
    loop = np.vstack([v, v[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, cum[-1], n, endpoint=False)
    return np.c_[np.interp(t, cum, loop[:, 0]), np.interp(t, cum, loop[:, 1])]


def test_an_ellipse_is_fitted_with_few_segments_and_its_area_is_kept_exactly():
    rings = [ellipse_ring(0.3, 0.2), ellipse_ring(0.25, 0.25), ellipse_ring(0.2, 0.3)]
    fit = sf.fit_rings(rings)
    assert not fit.linear and fit.n_segments <= 10 and fit.max_dev < sf.DEFAULT_TOLERANCE
    for ring, c in zip(rings, fit.sections):
        assert abs(polygon_area(cedit.sample_cedit(c))) == pytest.approx(abs(polygon_area(ring)), rel=2e-3)


def test_every_ring_of_a_set_has_the_same_anchors_and_so_the_same_number_of_segments():
    fit = sf.fit_rings([pentagon_ring(), ellipse_ring(1, 1), ellipse_ring(0.8, 1.1)])
    assert {c.n_segments for c in fit.sections} == {fit.n_segments}
    assert len(fit.anchors) == fit.n_segments and fit.anchors[0] == 0


def test_corners_get_anchors_on_both_sides_and_straight_runs_stay_straight():
    ring = pentagon_ring()
    idx = sf.choose_anchors([ring], 5)
    turn = sf.turning_angles(ring)
    assert all(i in idx for i in np.flatnonzero(turn > math.radians(sf.DEFAULT_CORNER_DEG)))
    fit = sf.fit_rings([ring])
    assert not fit.linear and fit.max_dev < sf.DEFAULT_TOLERANCE
    pts = np.array(fit.sections[0].points)
    # a straight segment is a degenerate Bezier: its handles lie on the chord at 1/3 and 2/3
    straight = 0
    for k in range(fit.n_segments):
        p = pts[3 * k:3 * k + 4]
        if np.allclose(p[1], p[0] + (p[3] - p[0]) / 3, atol=1e-9):
            straight += 1
    assert straight >= 4


def test_when_the_tolerance_cannot_be_met_the_sections_become_linear_cedits():
    rng = np.random.default_rng(1)
    ring = ellipse_ring(1.0, 0.6) * (1 + 0.03 * rng.random((64, 1)))  # a noisy ring no short chain follows
    fit = sf.fit_rings([ring], anchors=4, tolerance=1e-9)
    assert fit.linear and fit.n_segments == 64
    c = fit.sections[0]
    assert c.n_segments == 64
    pts = np.array(c.points) * [c.width, c.height] + np.array(c.centre)
    anchors = pts[::3][:64]
    assert all(np.abs(anchors - p).sum(axis=1).min() < 1e-9 for p in ring)  # every anchor is a ring point
    back = cedit.sample_cedit(c)
    assert polygon_area(back) == pytest.approx(polygon_area(ring), rel=1e-9)


def test_the_fitted_curve_stays_within_the_tolerance_of_the_ring():
    ring = pentagon_ring()
    fit = sf.fit_rings([ring], tolerance=0.002)
    curve = cedit.sample_cedit(fit.sections[0])
    assert sf.relative_deviation(ring, curve) <= 0.002 + 1e-9 or fit.linear


def test_the_cedit_curve_runs_clockwise_from_the_first_ring_point():
    ring = ellipse_ring(0.3, 0.2)
    c = sf.fit_rings([ring]).sections[0]
    assert polygon_area(cedit.sample_cedit(c)) < 0.0  # clockwise
    first = np.array(c.points[0]) * [c.width, c.height] + np.array(c.centre)
    assert np.allclose(first, ring[0], atol=2e-3)


# ---- the adaptive choice of XSecs -------------------------------------------------------------------------------------------------
def tube(centres, ring):
    return np.array([ring + c for c in centres])


def test_a_straight_prism_needs_only_its_end_rings():
    ring3 = np.c_[np.zeros(16), ellipse_ring(0.2, 0.1, 16)]
    dist = np.linspace(0, 2, 30)
    pts = tube([(x, 0, 0) for x in dist], ring3)
    assert vx.select_rings(pts, dist, 1e-4, min_count=2) == [0, 29]


def test_a_bend_gets_more_rings_than_a_straight_run_and_every_dropped_ring_is_within_tolerance():
    ring3 = np.c_[np.zeros(16), ellipse_ring(0.2, 0.1, 16)]
    dist = np.linspace(0, 2, 60)
    centres = [(x, 0.4 * math.exp(-((x - 1.0) / 0.25) ** 2), 0) for x in dist]  # a localised bend
    pts = tube(centres, ring3)
    kept = vx.select_rings(pts, dist, 1e-3, min_count=3)
    assert 3 < len(kept) < 60
    err = vx._hermite_errors(pts, dist, kept)
    assert err.max() <= 1e-3
    near = [dist[i] for i in kept if 0.6 < dist[i] < 1.4]
    far = [dist[i] for i in kept if dist[i] <= 0.6 or dist[i] >= 1.4]
    assert len(near) > len(far)  # denser where the curvature is


def test_mandatory_rings_and_the_minimum_count_are_kept():
    ring3 = np.c_[np.zeros(8), ellipse_ring(0.2, 0.1, 8)]
    dist = np.linspace(0, 1, 20)
    pts = tube([(x, 0, 0) for x in dist], ring3)
    kept = vx.select_rings(pts, dist, 1e-3, mandatory=[3, 4], min_count=5)
    assert {0, 19, 3, 4} <= set(kept) and len(kept) == 5


# ---- settings and names ----------------------------------------------------------------------------------------------------------------
def test_export_settings_default_and_read_from_the_config():
    assert vx.ExportSettings.from_config({}) == vx.ExportSettings()
    s = vx.ExportSettings.from_config({"export": {"fine_stations": 30, "verify": False, "cap_thickness": 0.002}})
    assert s.fine_stations == 30 and s.verify is False and s.cap_thickness == 0.002
    assert s.station_tolerance == vx.ExportSettings().station_tolerance


@pytest.mark.parametrize("bad", [{"fine_stations": 2}, {"min_stations": 1}, {"cedit_tolerance": 0.0},
                                 {"cap_thickness": -1.0}, {"prism_min": -0.1}])
def test_export_settings_reject_nonsense(bad):
    with pytest.raises(ValueError):
        vx.ExportSettings.from_config({"export": bad})


def test_geom_names_are_cleaned_of_spaces_and_punctuation():
    assert vx.sanitize("inlet 1") == "inlet_1" and vx.sanitize("R/L (top)") == "R_L_top" and vx.sanitize("--") == "inlet"


def test_the_curves_can_start_at_a_later_anchor_on_every_ring():
    rings = [pentagon_ring(), ellipse_ring(1, 1), ellipse_ring(0.8, 1.1)]
    base = sf.fit_rings(rings)
    shifted = sf.fit_rings(rings, start_shift=2)
    assert shifted.anchors == base.anchors and shifted.n_segments == base.n_segments
    for ring, c0, c2 in zip(rings, base.sections, shifted.sections):
        first = np.array(c2.points[0]) * [c2.width, c2.height] + np.array(c2.centre)
        assert np.allclose(first, ring[base.anchors[2]], atol=2e-3)  # starts at the third anchor
        # the same curve, only started elsewhere
        assert abs(polygon_area(cedit.sample_cedit(c2)) - polygon_area(cedit.sample_cedit(c0))) < 1e-9


def test_the_start_setting_is_read_from_the_config():
    assert vx.ExportSettings.from_config({"export": {"cedit_start": 3}}).cedit_start == 3
