# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Automatic verification of a saved export (Phase 4).

The saved file is reopened and:

 * every Geom that is not ours has exactly the parameter values it had in the input (hash of all parameters), and the
   file holds nothing but the model and our Geoms;
 * CompGeom on a temporary set {skin Geoms, our Geoms} runs and returns a non-empty mesh;
 * the volume CompGeom finds is compared with our own estimate of (skin ∪ positives) − negatives, and so is the volume
   the duct takes out of the skin;
 * every exported negative is sliced at each design station and the cut area is compared with A(s) (and the cut shape
   with the design section);
 * the line of sight through the exported bores is recomputed on the exported mesh and compared with the viewer's value.

The temporary set and the mesh Geom CompGeom creates are removed again (the file is never written back).
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from ductmgr import cedit
from ductmgr import duct as dm
from ductmgr import ductcheck, vspio
from ductmgr.geometry import polygon_area
from ductmgr.solids import RayCaster, Solid

SLICE_MERGE = 1e-3  # m: gaps up to this in a sliced contour are closed (OpenVSP's patches do not always share vertices)
CAP_PLANE_TOL = 2e-5  # m: a triangle whose corners all lie beyond (or this close to) an end section of a body belongs to its cap
INSIDE_TOL = 1e-6  # m: a triangle centroid this deep inside another body is an internal face
END_OFFSET = 5e-4  # m: a section is sliced this far to either side of the plane (never exactly in an XSec)
LOS_AGREE = 0.05  # the exported and the viewer line of sight agree within this fraction of the face
SKIN_MC_POINTS = 400000  # sample points for the volume of a skin made of several bodies
MIN_DELTA_VOLUME = 1e-9  # m^3


Finding = tuple[str, str, str]  # (label, value, level): level is "info", "pass", "warn" or "error"


@dataclass
class Verification:
    findings: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)  # the figures behind the findings (areas, volumes, ...)

    def add(self, label: str, value: str, level: str = "info") -> None:
        self.findings.append((label, value, level))

    @property
    def failed(self) -> bool:
        return any(level == "error" for _, _, level in self.findings)

    @property
    def warned(self) -> bool:
        return any(level == "warn" for _, _, level in self.findings)

    def lines(self) -> list[str]:
        tag = {"pass": "PASS", "warn": "WARN", "error": "FAIL", "info": "info"}
        return [f"  [{tag[level]:4s}] {label}: {value}" if label else f"         {value}"
                for label, value, level in self.findings]


# ---- the file ----------------------------------------------------------------------------------------------------------
def check_untouched(v: Verification, before: dict, names: dict, prefix: str, plan_names: list[str]) -> None:
    """Every non-prefixed Geom: same parameter hash as in the input; and the file holds nothing else."""
    after = vspio.param_hashes(prefix)
    changed = sorted(names.get(g, g) for g in before if after.get(g) != before[g])
    extra = sorted(g for g in after if g not in before)
    if changed or extra:
        what = ([f"changed: {', '.join(changed)}"] if changed else []) + \
               ([f"{len(extra)} unexpected Geom(s)"] if extra else [])
        v.add("Untouched Geoms", "FAIL " + "; ".join(what), "error")
    else:
        v.add("Untouched Geoms", f"PASS all {len(before)} other Geom(s) have identical parameters", "pass")
    present = sorted(vspio.vsp.GetGeomName(g) for g in vspio.geoms_with_prefix(prefix))
    if present != sorted(plan_names):
        v.add("File contents", f"FAIL {prefix}* Geoms in the file {present} differ from the plan {sorted(plan_names)}",
              "error")
    else:
        v.add("File contents", f"PASS the model and {len(present)} {prefix}* Geoms, nothing else", "pass")


XSEC_TOLERANCE = 1e-3  # of the section size: how far a stored Edit Curve may be from the outline it was written from


def check_xsecs(v: Verification, plan) -> None:
    """Every Edit Curve XSec in the saved file keeps the control points, the Width and Height it was written with (and has
    its curve symmetry off): OpenVSP's own Edit Curve symmetry would silently rewrite an asymmetric section."""
    import openvsp as vsp

    worst, where, count, symmetric = 0.0, "", 0, []
    for g in plan.geoms:
        surf = vsp.GetXSecSurf(vspio.find_geom(g.name).id, 0)
        for i, s in enumerate(g.sections):
            if s.cedit is None:
                continue
            xs = vsp.GetXSec(surf, i)
            count += 1
            if vsp.GetParmVal(vsp.GetXSecParm(xs, "SymType")) != float(vsp.SYM_NONE):
                symmetric.append(f"{g.name} XSec {i}")
            size = max(vsp.GetXSecWidth(xs) / s.cedit.width, vsp.GetXSecHeight(xs) / s.cedit.height, 1.0)
            dev = max(cedit.stored_deviation(xs, s.cedit), abs(vsp.GetXSecWidth(xs) / s.cedit.width - 1.0),
                      abs(vsp.GetXSecHeight(xs) / s.cedit.height - 1.0)) * size
            if dev > worst:
                worst, where = dev, f"{g.name} XSec {i}"
    v.numbers.update(xsec_deviation=worst)
    if symmetric or worst > XSEC_TOLERANCE:
        v.add("Edit Curve XSecs", f"FAIL {'symmetry on in ' + ', '.join(symmetric[:3]) if symmetric else ''}"
              f"{'; ' if symmetric else ''}stored curves deviate up to {100 * worst:.3f} % of the section size at {where}", "error")
    else:
        v.add("Edit Curve XSecs", f"PASS {count} XSecs stored as written (max deviation {100 * worst:.4f} % of the section size, "
              "curve symmetry off)", "pass")


# ---- CompGeom and volumes -------------------------------------------------------------------------------------------------
def sample_tube(rings: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points uniformly distributed in the tube whose cross-sections are the corresponding ``rings`` (R, N, 3)
    joined linearly (the volume ``tube_mesh`` encloses)."""
    r = len(rings)
    cen = rings.mean(axis=1)
    nxt = np.roll(rings, -1, axis=1)
    tri = 0.5 * np.linalg.norm(np.cross(rings - cen[:, None], nxt - cen[:, None]), axis=2)  # (R, N) fan areas
    area = tri.sum(axis=1)
    length = np.linalg.norm(np.diff(cen, axis=0), axis=1)
    seg_vol = 0.5 * (area[:-1] + area[1:]) * length
    k = rng.choice(r - 1, size=n, p=seg_vol / seg_vol.sum())
    a0, a1 = area[k], area[k + 1]
    # the section area grows linearly along the segment: invert its cumulative distribution
    u = rng.random(n)
    d = a1 - a0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(d) < 1e-12 * np.maximum(a0, 1e-300), u,
                     (-a0 + np.sqrt(a0**2 + u * (a1**2 - a0**2))) / np.where(d == 0, 1.0, d))
    t = np.clip(t, 0.0, 1.0)
    ring_t = rings[k] * (1 - t)[:, None, None] + rings[k + 1] * t[:, None, None]  # (n, N, 3)
    cen_t = ring_t.mean(axis=1)
    nxt_t = np.roll(ring_t, -1, axis=1)
    fan = 0.5 * np.linalg.norm(np.cross(ring_t - cen_t[:, None], nxt_t - cen_t[:, None]), axis=2)  # (n, N)
    cdf = np.cumsum(fan, axis=1)
    j = (rng.random(n)[:, None] * cdf[:, -1:] > cdf).sum(axis=1)
    j = np.minimum(j, rings.shape[1] - 1)
    s1, s2 = rng.random(n), rng.random(n)
    fold = s1 + s2 > 1.0
    s1, s2 = np.where(fold, 1 - s1, s1), np.where(fold, 1 - s2, s2)
    idx = np.arange(n)
    pa, pb = ring_t[idx, j], ring_t[idx, (j + 1) % rings.shape[1]]
    return cen_t + s1[:, None] * (pa - cen_t) + s2[:, None] * (pb - cen_t)


def duct_volume_change(bodies: dict, skin, samples: int, rng_seed: int = 0) -> dict:
    """Volumes of the intended model's duct terms from the design bodies, by sampling the tubes: V(P \\ skin), the part of
    the positives outside the skin, and V(N ∩ (skin ∪ P)), the part of the negatives that really cuts. Overlaps between
    bodies of one kind are counted once."""
    rng = np.random.default_rng(rng_seed)
    solids = {name: Solid(b.mesh) for name, b in bodies.items()}
    pos = [n for n, b in bodies.items() if b.kind == "positive"]
    neg = [n for n, b in bodies.items() if b.kind == "negative"]
    out_p = cut_n = 0.0
    for names, kind in ((pos, "positive"), (neg, "negative")):
        for name in names:
            body = bodies[name]
            pts = sample_tube(body.rings, samples, rng)
            mult = np.zeros(len(pts))
            for other in names:
                mult += solids[other].signed(pts) < 0.0
            weight = abs(body.mesh.volume) / samples / np.maximum(mult, 1.0)
            if kind == "positive":
                out_p += float((weight * (skin.signed(pts) > 0.0)).sum())
            else:
                inside = skin.signed(pts) < 0.0
                for other in pos:
                    inside |= solids[other].signed(pts) < 0.0
                cut_n += float((weight * inside).sum())
    return {"positive_outside": out_p, "negative_cut": cut_n}


def skin_volume(skin, solids_for_union=None) -> float | None:
    """Volume of the skin from its own mesh when it is a single closed body."""
    if isinstance(skin, Solid) and skin.closed:
        return abs(float(skin.mesh.volume))
    return None


def check_volume(v: Verification, comp_all: dict, comp_skin: dict, bodies: dict, skin, settings, skin_mesh_volume) -> None:
    if comp_all["triangles"] <= 0 or comp_all["meshes"] <= 0 or comp_all["wet_volume"] <= 0.0:
        v.add("CompGeom", "FAIL the result is empty (no triangles / meshes / volume)", "error")
        return
    v.add("CompGeom", f"PASS {comp_all['meshes']} mesh(es), {comp_all['triangles']} triangles, volume "
          f"{comp_all['wet_volume']:.4f} m^3", "pass")
    terms = duct_volume_change(bodies, skin, settings.volume_samples)
    v_skin = skin_mesh_volume if skin_mesh_volume is not None else comp_skin["wet_volume"]
    ours = v_skin + terms["positive_outside"] - terms["negative_cut"]
    diff = (comp_all["wet_volume"] - ours) / ours
    level = "pass" if abs(diff) <= settings.volume_tolerance else "warn"
    v.numbers.update(volume_compgeom=comp_all["wet_volume"], volume_ours=ours, **terms)
    v.add("Volume", f"CompGeom {comp_all['wet_volume']:.4f} m^3, ours {ours:.4f} m^3 (skin {v_skin:.4f} + external "
          f"positive {terms['positive_outside']:.4f} - negative {terms['negative_cut']:.4f}): {100 * diff:+.2f} %", level)
    if level == "warn":
        v.add("", f"WARN the volumes differ by more than {100 * settings.volume_tolerance:g} %", "warn")
    # the volume the duct takes out of the skin: far more sensitive than the total
    removed_vsp = comp_skin["wet_volume"] - comp_all["wet_volume"]
    removed_ours = terms["negative_cut"] - terms["positive_outside"]
    v.numbers.update(removed_compgeom=removed_vsp, removed_ours=removed_ours)
    if abs(removed_ours) > MIN_DELTA_VOLUME:
        d2 = (removed_vsp - removed_ours) / abs(removed_ours)
        level = "pass" if abs(d2) <= settings.volume_tolerance else "warn"
        v.add("Volume removed", f"CompGeom {removed_vsp:.4f} m^3, ours {removed_ours:.4f} m^3: {100 * d2:+.2f} %", level)


# ---- sections of the exported negatives -----------------------------------------------------------------------------------
def slice_polygons(mesh: pv.PolyData, centre, T, U, V) -> list[Polygon]:
    """The closed contours of ``mesh`` cut by the plane through ``centre`` with normal ``T``, as polygons in (U, V)
    coordinates about ``centre``. Gaps in the contour up to SLICE_MERGE are closed."""
    sl = mesh.slice(normal=T, origin=centre)
    if sl.n_cells == 0:
        return []
    pts = np.asarray(sl.points)
    rel = pts - np.asarray(centre)
    xy = np.c_[rel @ U, rel @ V]
    lines, cells, i = [], sl.lines, 0
    while i < len(cells):
        n = int(cells[i])
        ids = cells[i + 1:i + 1 + n]
        lines += [LineString([xy[a], xy[b]]) for a, b in zip(ids[:-1], ids[1:]) if a != b]
        i += n + 1
    if not lines:
        return []
    band = unary_union(lines).buffer(SLICE_MERGE, join_style="mitre")
    polys = [band] if isinstance(band, Polygon) else list(band.geoms)
    holes = [Polygon(h).buffer(SLICE_MERGE, join_style="mitre") for p in polys for h in p.interiors]
    return holes


def section_at(mesh, st, only_inside: bool = False) -> Polygon | None:
    """The contour of ``mesh`` in the plane of design station ``st`` (the one around the station centre). A plane exactly at
    a section of the Stack hits mesh edges lying in it, so it is cut a little on each side and the larger contour is
    taken (``only_inside``: only on the upstream side, for the last station of a body)."""
    best = None
    for shift in ((-END_OFFSET,) if only_inside else (-END_OFFSET, END_OFFSET)):
        polys = slice_polygons(mesh, st.centre + shift * st.T, st.T, st.U, st.V)
        if polys:
            p = min(polys, key=lambda q: q.distance(Point(0.0, 0.0)))
            if best is None or p.area > best.area:
                best = p
    return best


def check_areas(v: Verification, plan, design: dm.Duct, meshes: dict, settings) -> None:
    worst, worst_shape = (0.0, ""), (0.0, "")
    missing = 0
    for g in plan.geoms:
        if not g.negative:
            continue
        if g.part == "branch":
            bg = next(b for b in design.branches if not b.twin and b.name == g.inlet)
            stations = bg.stations
        else:
            stations = design.trunk
        last = len(stations) - 1
        for k, st in enumerate(stations):
            poly = section_at(meshes[g.name], st, g.part == "trunk" and k == last)
            where = f"{g.name} station {k} (dist {st.dist:.3f} m)"
            if poly is None or poly.area <= 0.0:
                missing += 1
                continue
            err = (poly.area - st.area) / st.area
            if abs(err) > abs(worst[0]):
                worst = (err, where)
            ring = Polygon(st.ring)
            size = float(np.ptp(st.ring, axis=0).max())
            dev = poly.exterior.hausdorff_distance(ring.exterior) / size
            if dev > worst_shape[0]:
                worst_shape = (dev, where)
    v.numbers.update(area_error=worst[0], shape_error=worst_shape[0])
    if missing:
        v.add("Area", f"FAIL {missing} design station(s) are not cut by the exported negative", "error")
        return
    level = "pass" if abs(worst[0]) <= settings.area_tolerance else "warn"
    v.add("Area", f"max error {100 * worst[0]:+.2f} % of A(s) at {worst[1]} (limit {100 * settings.area_tolerance:g} %)",
          level)
    if level == "warn":
        v.add("", f"WARN the exported negative departs from A(s) by more than {100 * settings.area_tolerance:g} %: "
              "increase the number of stations (export.station_tolerance)", "warn")
    v.add("Section shape", f"max deviation {100 * worst_shape[0]:.2f} % of the section size at {worst_shape[1]}",
          "pass" if worst_shape[0] <= settings.area_tolerance else "warn")


# ---- line of sight on the exported mesh ----------------------------------------------------------------------------------
def _bare(mesh: pv.PolyData) -> pv.PolyData:
    """The triangles of ``mesh`` without any point or cell data (those do not survive extracting cells)."""
    return pv.PolyData(np.asarray(mesh.points).copy(), mesh.faces.copy())


def _split_twin(mesh: pv.PolyData, side: float) -> tuple[pv.PolyData, pv.PolyData]:
    """(primary, twin) of a mesh that holds a body and its mirror image about y = 0; ``side`` is the sign of y of the body."""
    centroids = mesh.cell_centers().points
    primary = centroids[:, 1] * side >= 0.0
    take = lambda mask: _bare(mesh.extract_cells(np.flatnonzero(mask)).extract_surface(algorithm="dataset_surface"))  # noqa: E731
    return take(primary), take(~primary)


def exported_wall_caster(plan, meshes: dict) -> RayCaster | None:
    """The walls of the exported bores as one ray-caster: every negative's surface (a mirrored twin included) without its
    end caps and without the parts inside another negative (where branches overlap into the trunk)."""
    comps = []  # (mesh, planes) per body
    for g in plan.geoms:
        if not g.negative:
            continue
        mesh = _bare(meshes[g.name])
        real = g.real_sections
        first, last = real[0], real[-1]
        planes = [(first.origin, -first.R[:, 0]), (last.origin, last.R[:, 0])]  # normals point out of the body
        if g.mirrored:
            flip = np.array([1.0, -1.0, 1.0])
            side = 1.0 if float(np.mean([s.origin[1] for s in g.sections])) >= 0.0 else -1.0
            primary, twin = _split_twin(mesh, side)
            comps += [(primary, planes), (twin, [(o * flip, t * flip) for o, t in planes])]
        else:
            comps.append((mesh, planes))
    comps = [(m, p) for m, p in comps if m.n_cells]
    if not comps:
        return None
    solids = [Solid(_bare(m.clean(tolerance=SLICE_MERGE, absolute=True))) for m, _ in comps]
    walls = []
    for i, (m, planes) in enumerate(comps):
        tri = m.faces.reshape(-1, 4)[:, 1:]
        pts = m.points
        keep = np.ones(len(tri), dtype=bool)
        for origin, normal in planes:  # cap triangles lie beyond an end section
            beyond = ((pts[tri] - origin) @ normal) > -CAP_PLANE_TOL
            keep &= ~beyond.all(axis=1)
        centroids = pts[tri].mean(axis=1)
        for j, other in enumerate(solids):
            if j != i and other.closed:
                keep &= ~(other.signed(centroids) < -INSIDE_TOL)
        if keep.any():
            walls.append(_bare(m.extract_cells(np.flatnonzero(keep)).extract_surface(algorithm="dataset_surface")))
    return RayCaster(*walls) if walls else None


def check_line_of_sight(v: Verification, plan, design: dm.Duct, meshes: dict, viewer: dict | None) -> None:
    caster = exported_wall_caster(plan, meshes)
    if caster is None:
        v.add("Line of sight", "no exported negative to cast rays through", "warn")
        return
    lx, ln = ductcheck.los_fractions(design, [caster] * len(design.branches))
    v.numbers.update(los_x=lx, los_n=ln)
    text = f"{100 * lx:.0f} % of the face along +X, {100 * ln:.0f} % along the inlet normal (exported mesh)"
    if viewer is None:
        v.add("Line of sight", text, "info")
        return
    gap = max(abs(lx - viewer["x"]), abs(ln - viewer["n"]))
    level = "pass" if gap <= LOS_AGREE else "warn"
    v.add("Line of sight", f"{text}; viewer {100 * viewer['x']:.0f} % / {100 * viewer['n']:.0f} %", level)
    if level == "warn":
        v.add("", f"WARN the line of sight on the exported mesh differs from the viewer's by {100 * gap:.0f} points",
              "warn")


# ---- all of it --------------------------------------------------------------------------------------------------------------
def verify_export(saved_path, plan, design: dm.Duct, skin_names: list[str], skin, settings, before: dict, names: dict,
                  prefix: str, viewer_los: dict | None = None, bodies: dict | None = None, progress=None) -> Verification:
    """Reopen ``saved_path`` and run every check. ``before`` / ``names`` are the parameter hashes and names of the input's
    non-prefixed Geoms; ``skin`` is the skin as a Solid; ``viewer_los`` is {"x": .., "n": ..} from the viewer."""
    say = progress or (lambda text: None)
    v = Verification()
    say("Reopening the saved file")
    vspio.open_model(saved_path)
    check_untouched(v, before, names, prefix, [g.name for g in plan.geoms])
    check_xsecs(v, plan)
    geoms = vspio.list_geoms()
    skin_geoms = [vspio.find_geom(n, geoms) for n in skin_names]
    duct_geoms = [g for g in geoms if g.name.startswith(prefix)]
    say("CompGeom on the skin and the duct Geoms")
    try:
        comp_skin = vspio.comp_geom(skin_geoms)
        comp_all = vspio.comp_geom(skin_geoms + duct_geoms)
    except Exception as exc:  # OpenVSP raised: the file may still be fine, but the check cannot run
        v.add("CompGeom", f"FAIL CompGeom raised {type(exc).__name__}: {exc}", "error")
        return v
    say("Volumes")
    bodies = bodies if bodies is not None else design.bodies()
    check_volume(v, comp_all, comp_skin, bodies, skin, settings, skin_volume(skin))
    say("Slicing the exported negatives")
    meshes = {g.name: vspio.export_mesh([vspio.find_geom(g.name, geoms)]) for g in plan.geoms if g.negative}
    check_areas(v, plan, design, meshes, settings)
    say("Line of sight")
    check_line_of_sight(v, plan, design, meshes, viewer_los)
    return v
