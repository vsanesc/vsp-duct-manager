# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""The general section outline: vertices, straight / curved edges, Bézier handles, continuity, edits."""
import math

import numpy as np
import pytest

import bodies
from ductmgr import outline as ol
from ductmgr import section as sec
from ductmgr import sketch as sk
from ductmgr.geometry import InletPlane, polygon_area
from ductmgr.outline import SectionShape

R = 1.0
PENTAGON = [(0.0, 1.0), (-0.95, 0.31), (-0.59, -0.81), (0.59, -0.81), (0.95, 0.31)]  # CCW, apex up


def P(verts=PENTAGON, snapped=None):
    return SectionShape.polygon(verts, snapped)


def outline_of(shape, section=None):
    return ol.build_outline(ol.edge_paths(shape, section)[0])


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def valid(shape):
    """A plain section is valid: positive area, no self-crossing."""
    return sk.evaluate(shape, None, None, max(ol.outline_area(shape), 1e-6)).valid


# ---- construction and validation -----------------------------------------------------------------------------
def test_a_polygon_is_straight_with_corner_vertices_and_no_handles():
    s = P()
    assert s.n == 5 and s.edges == ("straight",) * 5 and s.handles == (None,) * 5 and s.continuity == ("corner",) * 5
    assert s.snapped == (False,) * 5 and not any(s.edge_is_curved(i) for i in range(5))


@pytest.mark.parametrize("n", [3, 4, 5, 6, 9])
def test_any_vertex_count_from_three_is_allowed(n):
    verts = [(math.cos(2 * math.pi * k / n + 1.0), math.sin(2 * math.pi * k / n + 1.0)) for k in range(n)]
    s = P(verts)
    assert s.n == n and ol.outline_area(s) == pytest.approx(0.5 * n * math.sin(2 * math.pi / n), rel=1e-12)


def test_fewer_than_three_vertices_is_rejected():
    with pytest.raises(ValueError, match="at least 3"):
        SectionShape.polygon([(0, 0), (1, 0)])


def test_inconsistent_data_is_rejected():
    s = P()
    with pytest.raises(ValueError, match="edge type"):
        s.with_(edges=("wavy",) + s.edges[1:])
    with pytest.raises(ValueError, match="continuity"):
        s.with_(continuity=("round",) + s.continuity[1:])
    with pytest.raises(ValueError, match="handles"):
        s.with_(edges=("curved",) + s.edges[1:])  # curved without handles
    with pytest.raises(ValueError, match="one entry per vertex"):
        s.with_(snapped=(False,) * 4)
    with pytest.raises(ValueError, match="only a curved"):
        s.control_points(0)


# ---- the circle from four Béziers ---------------------------------------------------------------------------------
def test_a_circle_from_four_symmetric_beziers_has_the_analytic_area_within_0_1_percent():
    c = ol.circle_shape(R)
    assert c.n == 4 and c.edges == ("curved",) * 4 and c.continuity == ("smooth",) * 4
    assert ol.outline_area(c) == pytest.approx(math.pi * R**2, rel=1e-3)
    big = ol.circle_shape(0.37)
    assert ol.outline_area(big) == pytest.approx(math.pi * 0.37**2, rel=1e-3)


def test_the_circle_is_round_symmetric_and_counter_clockwise():
    c = ol.circle_shape(R)
    pts = outline_of(c)
    assert np.abs(np.hypot(pts[:, 0], pts[:, 1]) - R).max() < 3e-4  # the classic 0.027 % radial error
    assert polygon_area(pts) > 0 and c.vertices[0] == (0.0, R)  # CCW, starting at the top
    for k in (1, 2, 3):  # a rotation by 90 degrees maps the area onto itself
        rot = np.array([[0.0, -1.0], [1.0, 0.0]]) ** 1
        pts = pts @ rot.T
        assert abs(polygon_area(pts)) == pytest.approx(math.pi * R**2, rel=1e-3)
    shifted = ol.circle_shape(R, centre=(0.3, -0.2))
    assert ol.outline_centroid(shifted) == pytest.approx([0.3, -0.2], abs=1e-6)


def test_the_outline_is_sampled_finely():
    c = ol.circle_shape(R)
    paths, flags, lengths = ol.edge_paths(c)
    assert all(len(p) == ol.BEZIER_SAMPLES + 1 for p in paths) and flags == [False] * 4
    assert sum(lengths) == pytest.approx(2 * math.pi * R, rel=1e-3)  # arc length of the circle
    assert ol.build_outline(paths).shape == (4 * ol.BEZIER_SAMPLES, 2)


# ---- Bézier helpers -----------------------------------------------------------------------------------------------------
def test_splitting_a_bezier_reproduces_the_same_curve():
    ctrl = np.array([(0.0, 0.0), (0.2, 0.9), (0.8, -0.4), (1.0, 0.3)])
    for t in (0.2, 0.5, 0.77):
        left, right = ol.split_bezier(ctrl, t)
        assert left[3] == pytest.approx(ol.bezier_at(ctrl, t)) and right[0] == pytest.approx(left[3])
        assert ol.bezier_at(left, 0.5) == pytest.approx(ol.bezier_at(ctrl, t * 0.5), abs=1e-12)
        assert ol.bezier_at(right, 0.5) == pytest.approx(ol.bezier_at(ctrl, t + (1 - t) * 0.5), abs=1e-12)


def test_the_nearest_parameter_on_a_bezier():
    ctrl = np.array([(0.0, 0.0), (0.3, 1.0), (0.7, 1.0), (1.0, 0.0)])
    for t in (0.1, 0.4, 0.5, 0.83):
        p = ol.bezier_at(ctrl, t)
        assert ol.nearest_t_on_bezier(ctrl, p) == pytest.approx(t, abs=1e-6)
        away = p + 0.01 * np.array([0.0, 1.0])
        assert abs(ol.nearest_t_on_bezier(ctrl, away) - t) < 0.05  # off the curve: still the neighbourhood


# ---- edge type ----------------------------------------------------------------------------------------------------------------
def test_a_new_curved_edge_starts_as_the_straight_chord():
    s = P()
    c = ol.set_edge_type(s, 1, "curved")
    a, b = np.array(s.vertices[1]), np.array(s.vertices[2])
    h0, h1 = (np.array(h) for h in c.handles[1])
    assert h0 == pytest.approx((b - a) / 3) and h1 == pytest.approx(-(b - a) / 3)  # handles at 1/3 and 2/3
    assert ol.outline_area(c) == pytest.approx(ol.outline_area(s), rel=1e-12)  # the same shape until a handle moves
    assert c.edge_is_curved(1) and not c.edge_is_curved(0)
    back = ol.set_edge_type(c, 1, "straight")
    assert back == s


def test_a_snapped_edge_cannot_be_edited():
    s = P(snapped=[True, True, False, False, False])
    with pytest.raises(ValueError, match="snapped"):
        ol.set_edge_type(s, 0, "curved")
    curved = ol.set_edge_type(P(), 0, "curved").with_(snapped=(True, True, False, False, False))
    with pytest.raises(ValueError, match="free curved"):
        ol.move_handle(curved, 0, 0, (0.0, 0.0))
    assert not curved.edge_is_curved(0)  # between two snapped vertices it follows the skin


def test_moving_a_vertex_carries_its_handles():
    c = ol.set_edge_type(ol.set_edge_type(P(), 1, "curved"), 2, "curved")
    h_before = c.handles
    moved = ol.move_vertex(c, 2, (-0.5, -0.9))
    assert moved.vertices[2] == (-0.5, -0.9)
    assert moved.handles[1][1] == h_before[1][1] and moved.handles[2][0] == h_before[2][0]  # offsets, so they follow
    assert np.array(moved.control_points(1)[2]) == pytest.approx(np.array([-0.5, -0.9]) + np.array(h_before[1][1]))


# ---- continuity ---------------------------------------------------------------------------------------------------------------------
def two_curved(mode):
    s = ol.set_edge_type(ol.set_edge_type(P(), 1, "curved"), 2, "curved")  # both edges at vertex 2 are curved
    s = ol.move_handle(s, 1, 1, np.array(s.vertices[2]) + [0.1, 0.3])  # an uneven pair of handles (corner mode)
    s = ol.move_handle(s, 2, 0, np.array(s.vertices[2]) + [0.4, -0.1])
    return ol.set_continuity(s, 2, mode), s


def handles_at(s, v):
    n = s.n
    return np.array(s.handles[(v - 1) % n][1]), np.array(s.handles[v][0])  # (in-handle, out-handle)


def test_corner_handles_are_independent():
    s, raw = two_curved("corner")
    hin, hout = handles_at(s, 2)
    assert abs(hin[0] * hout[1] - hin[1] * hout[0]) > 1e-3 and s == ol.set_continuity(raw, 2, "corner")
    moved = ol.move_handle(s, 2, 0, np.array(s.vertices[2]) + [0.3, 0.2])
    assert handles_at(moved, 2)[0] == pytest.approx(hin)  # the other handle did not follow


def test_smooth_keeps_the_handles_collinear_and_their_lengths_independent():
    s, _ = two_curved("smooth")
    hin, hout = handles_at(s, 2)
    assert abs(hin[0] * hout[1] - hin[1] * hout[0]) < 1e-12 and np.dot(hin, hout) < 0  # collinear, opposite
    assert np.linalg.norm(hout) == pytest.approx(np.linalg.norm([0.4, -0.1]))  # the dominant (out) handle is kept
    moved = ol.move_handle(s, 1, 1, np.array(s.vertices[2]) + [0.0, 0.5])  # drag the in-handle (edge 1, at vertex 2): it dominates now
    hin2, hout2 = handles_at(moved, 2)
    assert hin2 == pytest.approx([0.0, 0.5]) and unit(hout2) == pytest.approx(-unit(hin2), abs=1e-12)
    assert np.linalg.norm(hout2) == pytest.approx(np.linalg.norm(hout))  # its own length survives (G1, not C1)


def test_symmetric_handles_are_equal_and_opposite():
    s, _ = two_curved("symmetric")
    hin, hout = handles_at(s, 2)
    assert hin == pytest.approx(-hout, abs=1e-12)
    moved = ol.move_handle(s, 2, 0, np.array(s.vertices[2]) + [0.2, 0.25])
    hin2, hout2 = handles_at(moved, 2)
    assert hin2 == pytest.approx(-hout2, abs=1e-12) and hout2 == pytest.approx([0.2, 0.25])


def test_a_smooth_vertex_next_to_a_straight_edge_aligns_the_curved_handle_with_it():
    s = ol.set_edge_type(P(), 2, "curved")  # edge 2 (vertex 2 -> 3) is curved, edge 1 (1 -> 2) stays straight
    s = ol.set_continuity(s, 2, "smooth")
    t = unit(np.array(s.vertices[2]) - np.array(s.vertices[1]))  # direction of the arriving straight edge
    assert unit(s.handles[2][0]) == pytest.approx(t, abs=1e-12)
    moved = ol.move_vertex(s, 1, (-0.9, 0.0))  # the straight edge turns: the handle follows it
    t2 = unit(np.array(moved.vertices[2]) - np.array(moved.vertices[1]))
    assert unit(moved.handles[2][0]) == pytest.approx(t2, abs=1e-12) and t2 != pytest.approx(t)
    # and at the other end: a straight edge leaving a smooth vertex
    e = ol.set_edge_type(P(), 1, "curved")
    e = ol.set_continuity(e, 2, "smooth")  # edge 2 (2 -> 3) straight leaves vertex 2
    t_out = unit(np.array(e.vertices[3]) - np.array(e.vertices[2]))
    assert unit(-np.array(e.handles[1][1])) == pytest.approx(t_out, abs=1e-12)


def test_dragging_a_handle_enforces_continuity_at_its_vertex_only():
    s, _ = two_curved("smooth")
    other = ol.set_continuity(ol.set_edge_type(s, 3, "curved"), 3, "corner")
    moved = ol.move_handle(other, 2, 1, np.array(other.vertices[3]) + [0.1, 0.4])  # edge 2's handle at vertex 3
    assert handles_at(moved, 3)[0] == pytest.approx([0.1, 0.4])
    assert np.array(handles_at(moved, 2)) == pytest.approx(np.array(handles_at(other, 2)))  # vertex 2 untouched


# ---- insert and delete a vertex -------------------------------------------------------------------------------------------------------
def closed_and_same_area(before, after):
    """The outline of ``after`` is a closed, simple polygon with the area of ``before``."""
    out = outline_of(after)
    assert valid(after) and polygon_area(out) > 0
    assert ol.outline_area(after) == pytest.approx(ol.outline_area(before), rel=1e-9)


@pytest.mark.parametrize("edge", range(5))
def test_inserting_on_a_straight_edge_keeps_the_outline(edge):
    s = P()
    mid = (np.array(s.vertices[edge]) + np.array(s.vertices[(edge + 1) % 5])) / 2 + [0.01, 0.0]
    new, at = ol.insert_vertex(s, edge, mid)
    assert new.n == 6 and at == edge + 1
    on_edge = np.array(new.vertices[at])
    a, b = np.array(s.vertices[edge]), np.array(s.vertices[(edge + 1) % 5])
    assert abs((on_edge - a)[0] * (b - a)[1] - (on_edge - a)[1] * (b - a)[0]) < 1e-12  # exactly on the edge
    assert new.continuity[at] == "corner" and new.edges[edge] == "straight" and new.edges[at] == "straight"
    closed_and_same_area(s, new)
    # the outline is closed: every edge starts where the previous one ends
    paths = ol.edge_paths(new)[0]
    for i in range(new.n):
        assert paths[i][-1] == pytest.approx(paths[(i + 1) % new.n][0])


def test_inserting_on_a_curved_edge_splits_it_exactly_and_makes_the_new_vertex_smooth():
    c = ol.circle_shape(R)
    before = outline_of(c)
    new, at = ol.insert_vertex(c, 0, (-0.7, 0.72))  # near the middle of the top-left arc
    assert new.n == 5 and at == 1 and new.edges == ("curved",) * 5 and new.continuity[at] == "smooth"
    after = outline_of(new)
    assert ol.outline_area(new) == pytest.approx(ol.outline_area(c), rel=1e-4)  # the curve is re-sampled
    # the same curve: every point of the old outline lies on the new one
    d = sec.distance_to_section(sec.make_section(after, True), before)  # distance to the new outline as a polyline
    assert d.max() < 5e-4 * R  # only the chord error of the sampled curve remains
    hin, hout = handles_at(new, at)
    assert abs(hin[0] * hout[1] - hin[1] * hout[0]) < 1e-12 and np.dot(hin, hout) < 0  # G1 at the split
    assert new.vertices[at][0] ** 2 + new.vertices[at][1] ** 2 == pytest.approx(R * R, abs=3e-4)


def test_inserting_on_the_closing_edge_appends_the_vertex():
    s = P()
    new, at = ol.insert_vertex(s, 4, (0.5, 0.7))  # edge 4 closes the polygon (vertex 4 -> vertex 0)
    assert at == 5 and new.n == 6 and new.vertices[:5] == s.vertices
    closed_and_same_area(s, new)


def test_inserting_on_a_skin_edge_gives_a_snapped_vertex(prism=None):
    body = bodies.chined_prism()
    plane = InletPlane((4.0, 0.0, 0.4))
    s = sec.compute_section(body, plane)
    shape = sk.default_sketch(s, plane, 0.06)
    assert shape.edge_is_snapped(0)
    point = sec.nearest_on_section(s, (0.0, 0.0))[0]
    new, at = ol.insert_vertex(shape, 0, point)
    assert new.snapped[at] is True and new.edge_is_snapped(0) and new.edge_is_snapped(1)  # both halves follow the skin
    paths_new = ol.edge_paths(new, s)[0]
    paths_old = ol.edge_paths(shape, s)[0]
    assert ol.outline_area(new, s) == pytest.approx(ol.outline_area(shape, s), rel=1e-9)
    assert sec.distance_to_section(s, np.array(new.vertices[at]))[0] < 1e-6 and len(paths_new) == 6 and len(paths_old) == 5


@pytest.mark.parametrize("i", range(5))
def test_deleting_a_vertex_keeps_a_closed_valid_outline(i):
    s = P()
    new = ol.delete_vertex(s, i)
    assert new.n == 4 and new.vertices == s.vertices[:i] + s.vertices[i + 1:]
    assert valid(new) and polygon_area(outline_of(new)) > 0
    paths = ol.edge_paths(new)[0]
    for k in range(new.n):
        assert paths[k][-1] == pytest.approx(paths[(k + 1) % new.n][0])  # closed
    assert ol.outline_area(new) < ol.outline_area(s)  # a convex polygon loses a corner


@pytest.mark.parametrize("i", range(4))
def test_deleting_a_vertex_of_the_circle_leaves_a_valid_curved_outline(i):
    c = ol.circle_shape(R)
    new = ol.delete_vertex(c, i)
    assert new.n == 3 and valid(new) and "curved" in new.edges
    assert 1.5 < ol.outline_area(new) < math.pi  # a valid curved triangle-like shape


def test_deleting_between_a_curved_and_a_straight_edge_makes_a_curved_one():
    s = ol.set_edge_type(P(), 1, "curved")
    new = ol.delete_vertex(s, 2)  # edge 1 (curved) and edge 2 (straight) merge
    assert new.edges.count("curved") == 1 and valid(new)
    both_straight = ol.delete_vertex(P(), 2)
    assert both_straight.edges == ("straight",) * 4


def test_a_section_keeps_at_least_three_vertices():
    s = P([(0, 0), (1, 0), (0, 1)])
    with pytest.raises(ValueError, match="at least 3"):
        ol.delete_vertex(s, 0)
    four = ol.delete_vertex(P([(0, 0), (1, 0), (1, 1), (0, 1)]), 2)
    assert four.n == 3 and valid(four)


def test_insert_then_delete_restores_a_straight_outline():
    s = P()
    new, at = ol.insert_vertex(s, 2, (0.0, -0.81))
    back = ol.delete_vertex(new, at)
    assert back == s


# ---- scaling and translating ---------------------------------------------------------------------------------------------------------
def test_scale_free_leaves_snapped_vertices_and_their_handles_alone():
    s = ol.set_edge_type(ol.set_edge_type(P(snapped=[True, False, False, False, False]), 0, "curved"), 3, "curved")
    scaled = ol.scale_free(s, (0.0, 0.0), 2.0)
    assert scaled.vertices[0] == s.vertices[0]  # snapped: fixed
    assert np.array(scaled.vertices[1]) == pytest.approx(2 * np.array(s.vertices[1]))
    assert scaled.handles[0][0] == s.handles[0][0]  # the handle at the snapped vertex
    assert np.array(scaled.handles[0][1]) == pytest.approx(2 * np.array(s.handles[0][1]))  # at the free end
    assert np.array(scaled.handles[3][0]) == pytest.approx(2 * np.array(s.handles[3][0]))


def test_scale_all_and_translate():
    c = ol.circle_shape(1.0)
    big = ol.scale_all(c, (0.0, 0.0), 3.0)
    assert ol.outline_area(big) == pytest.approx(9.0 * ol.outline_area(c), rel=1e-12)
    moved = ol.translate(c, (0.5, -0.25))
    assert ol.outline_centroid(moved) == pytest.approx(ol.outline_centroid(c) + [0.5, -0.25], abs=1e-12)


# ---- config round trip ------------------------------------------------------------------------------------------------------------------
def test_a_shape_round_trips_through_the_config_dict():
    s = ol.set_edge_type(ol.set_edge_type(P(snapped=[True, True, False, False, False]), 2, "curved"), 3, "curved")
    s = ol.set_continuity(s, 3, "symmetric")
    data = ol.shape_to_config(s)
    assert set(data) == {"vertices", "snapped", "edges", "handles", "continuity"}
    assert data["handles"][0] is None and data["handles"][2] is not None
    assert ol.shape_from_config(data) == s


def test_the_config_defaults_are_straight_unsnapped_corners():
    s = ol.shape_from_config({"vertices": [[0, 0], [1, 0], [0, 1]]})
    assert s == SectionShape.polygon([(0, 0), (1, 0), (0, 1)])


def test_a_curved_edge_without_stored_handles_starts_as_the_chord():
    s = ol.shape_from_config({"vertices": [[0, 0], [3, 0], [0, 3]], "edges": ["curved", "straight", "straight"]})
    assert s.handles[0] == ((1.0, 0.0), (-1.0, 0.0))


@pytest.mark.parametrize("patch, message", [
    ({"vertices": [[0, 0], [1, 0]]}, "at least 3"),
    ({"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}, "pairs"),
    ({"snapped": [True, False]}, "snapped"),
    ({"snapped": [1, 0, 0]}, "snapped"),
    ({"edges": ["straight", "wavy", "straight"]}, "edges"),
    ({"continuity": ["corner", "round", "corner"]}, "continuity"),
    ({"handles": [None, None]}, "handles"),
    ({"edges": ["curved", "straight", "straight"], "handles": [[[0, 0]], None, None]}, "handles"),
])
def test_malformed_sections_are_rejected_with_a_readable_message(patch, message):
    data = {"vertices": [[0, 0], [1, 0], [0, 1]]}
    data.update(patch)
    with pytest.raises(ValueError, match=message):
        ol.shape_from_config(data, "key section")
