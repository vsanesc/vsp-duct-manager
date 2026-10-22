# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Read-only access to an OpenVSP model (Phase 2): geoms, surface meshes, engine face.

Surfaces are obtained through the OpenVSP API: the wanted geoms are put in a temporary set, exported to a
temporary STL, and loaded with PyVista. The temporary set is restored and the STL deleted afterwards.
Nothing here saves the model.
"""
import contextlib
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvsp as vsp
import pyvista as pv

DEFAULT_FACE_TO_ENVELOPE_RANGE = (0.80, 0.98)  # engine.face_to_envelope_range overrides
TEMP_SET_NAME = "ductmgr_tmp_export"
STL_NAME = "export.stl"
BOUNDS_MISMATCH_TOLERANCE = 0.01  # relative to the model size: mesh bounds vs API bounding boxes


@dataclass(frozen=True)
class GeomInfo:
    id: str
    name: str
    type_name: str
    bbox_min: np.ndarray
    bbox_max: np.ndarray


@dataclass(frozen=True)
class EngineFace:
    """Where the engine face is, from the engine Geom. The fan diameter is NOT here: it comes from the
    config (the Geom's bounding box is the casing, i.e. the envelope)."""

    geom: GeomInfo
    center: np.ndarray  # centre of the front (min-x) end of the Geom
    envelope_diameter: float  # larger of the Geom's y and z extents (the casing)


@dataclass(frozen=True)
class FaceEnvelopeCheck:
    face_diameter: float  # from the config (fan)
    envelope_diameter: float  # from the model (casing)
    ratio: float
    ratio_range: tuple[float, float]
    status: str  # "PASS" or "WARN"
    message: str


def open_model(path) -> None:
    """Load a .vsp3 into the OpenVSP session (read-only use: the file is never written)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(str(path))
    vsp.Update()


def list_geoms() -> list[GeomInfo]:
    geoms = []
    for gid in vsp.FindGeoms():
        lo, hi = vsp.GetGeomBBoxMin(gid), vsp.GetGeomBBoxMax(gid)
        geoms.append(
            GeomInfo(
                id=gid,
                name=vsp.GetGeomName(gid),
                type_name=vsp.GetGeomTypeName(gid),
                bbox_min=np.array([lo.x(), lo.y(), lo.z()]),
                bbox_max=np.array([hi.x(), hi.y(), hi.z()]),
            )
        )
    return geoms


def find_geom(name: str, geoms: list[GeomInfo] | None = None) -> GeomInfo:
    geoms = geoms if geoms is not None else list_geoms()
    matches = [g for g in geoms if g.name == name]
    if not matches:
        raise LookupError(f"no geom named '{name}' in the model (found: {sorted({g.name for g in geoms})})")
    if len(matches) > 1:
        raise LookupError(f"{len(matches)} geoms are named '{name}'; the name must be unique")
    return matches[0]


def _free_set_index() -> int:
    ids = vsp.FindGeoms()
    for idx in range(vsp.SET_FIRST_USER, vsp.GetNumSets()):
        if not any(vsp.GetSetFlag(gid, idx) for gid in ids):
            return idx
    raise RuntimeError("no free user set available for the temporary export")


@contextlib.contextmanager
def _without_compgeom_files():
    """CompGeom (also behind the STL export) writes <model>_CompGeom.txt / .csv next to the model file: remove what a
    block creates."""
    name = vsp.GetVSPFileName()
    if not name:  # a model that was never read from / written to a file: nothing is written next to it
        yield
        return
    model_file = Path(name)
    side = [model_file.with_name(f"{model_file.stem}_CompGeom{ext}") for ext in (".txt", ".csv")]
    existed = [f.exists() for f in side]
    try:
        yield
    finally:
        for f, before in zip(side, existed):
            if not before and f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass


def read_stl(path) -> pv.PolyData:
    """Read an STL file. OpenVSP writes ASCII STL, which VTK's reader sometimes fails on (an empty mesh or an error), so
    an ASCII file is parsed here; anything else goes to PyVista."""
    raw = Path(path).read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:400]:
        xyz = np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw), dtype=float)
        n = len(xyz) // 3
        faces = np.c_[np.full(n, 3), np.arange(3 * n).reshape(n, 3)]
        return pv.PolyData(xyz, faces.ravel())
    return pv.read(str(path))


def export_mesh(geoms: list[GeomInfo]) -> pv.PolyData:
    """Triangulated surface of the given geoms (all symmetric copies), via a temporary set and STL."""
    if not geoms:
        return pv.PolyData()
    idx = _free_set_index()
    old_name = vsp.GetSetName(idx)
    ids_before = set(vsp.FindGeoms())
    shown = {gid: vsp.GetSetFlag(gid, vsp.SET_SHOWN) for gid in ids_before}
    vsp.SetSetName(idx, TEMP_SET_NAME)
    for g in geoms:
        vsp.SetSetFlag(g.id, idx, True)
    try:
        with tempfile.TemporaryDirectory(prefix="ductmgr_") as tmp:
            stl = Path(tmp) / STL_NAME
            with _without_compgeom_files():
                vsp.ExportFile(str(stl), idx, vsp.EXPORT_STL)
            if not stl.is_file():
                raise RuntimeError("OpenVSP did not write the temporary STL")
            mesh = read_stl(stl)
    finally:
        # Leave the session as we found it. ExportFile hides the geoms and leaves a transient "MeshGeom"
        # behind; remove only geoms that did not exist before the export.
        for gid in set(vsp.FindGeoms()) - ids_before:
            vsp.DeleteGeom(gid)
        for g in geoms:
            vsp.SetSetFlag(g.id, idx, False)
        vsp.SetSetName(idx, old_name)
        for gid, flag in shown.items():
            vsp.SetSetFlag(gid, vsp.SET_SHOWN, flag)
    return mesh.clean().compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False)


def check_mesh_bounds(mesh: pv.PolyData, geoms: list[GeomInfo]) -> str | None:
    """WARN text if the STL bounds disagree with the API bounding boxes (e.g. a unit mismatch)."""
    if mesh.n_points == 0 or not geoms:
        return None
    lo = np.min([g.bbox_min for g in geoms], axis=0)
    hi = np.max([g.bbox_max for g in geoms], axis=0)
    size = float(np.max(hi - lo))
    b = np.array(mesh.bounds).reshape(3, 2)
    tol = BOUNDS_MISMATCH_TOLERANCE * size
    # Symmetry copies (about XZ) only extend Y, so X and Z must agree and Y must contain the bbox.
    ok = (
        all(abs(b[i, 0] - lo[i]) <= tol and abs(b[i, 1] - hi[i]) <= tol for i in (0, 2))
        and b[1, 0] <= lo[1] + tol
        and b[1, 1] >= hi[1] - tol
    )
    if size > 0.0 and not ok:
        return (
            f"WARN exported mesh bounds {np.round(mesh.bounds, 3).tolist()} do not match the model bounding "
            f"boxes ({np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}); check export units"
        )
    return None


def fuselage_mesh(fuselage_name: str) -> pv.PolyData:
    """Surface of only the fuselage geom."""
    return export_mesh([find_geom(fuselage_name)])


def context_geoms(exclude_names: list[str]) -> list[GeomInfo]:
    """Every geom whose name is not excluded (wing, fins, stores...)."""
    return [g for g in list_geoms() if g.name not in exclude_names]


def engine_face(geom_name: str) -> EngineFace:
    """Face centre = centre of the front (min-x) end of the geom's bounding box (the engine envelope)."""
    geom = find_geom(geom_name)
    lo, hi = geom.bbox_min, geom.bbox_max
    center = np.array([lo[0], 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])])
    return EngineFace(geom=geom, center=center, envelope_diameter=float(max(hi[1] - lo[1], hi[2] - lo[2])))


def check_face_to_envelope(
    face: EngineFace, face_diameter: float, ratio_range: tuple[float, float]
) -> FaceEnvelopeCheck:
    """Compare the configured fan diameter with the model's envelope diameter (nothing is changed)."""
    ratio = face_diameter / face.envelope_diameter
    lo, hi = ratio_range
    status = "PASS" if lo <= ratio <= hi else "WARN"
    message = (
        f"engine '{face.geom.name}': face diameter {face_diameter:.4f} m (config) / envelope diameter "
        f"{face.envelope_diameter:.4f} m (model) = {ratio:.3f}  [{status}: expected {lo:g}-{hi:g}]"
    )
    return FaceEnvelopeCheck(face_diameter, face.envelope_diameter, ratio, (lo, hi), status, message)


# ---- wing planform edges (from VSP parameters, not from the mesh) ------------------------------------------
ROTATION_TOLERANCE_DEG = 1e-6  # a wing rotated more than this is reported (its sweeps are in its own frame)
WING_TYPE_NAME = "Wing"


@dataclass(frozen=True)
class WingSection:
    y0: float  # spanwise extent in aircraft coordinates (planform), y0 <= y1
    y1: float
    le_deg: float  # planform sweep of the leading edge, positive = swept back
    te_deg: float  # planform sweep of the trailing edge, positive = swept back


@dataclass(frozen=True)
class WingPlanform:
    geom_id: str
    name: str
    sections: tuple[WingSection, ...]
    rotated: bool  # the wing has a non-zero relative rotation


@dataclass(frozen=True)
class EdgeSweep:
    angle_deg: float
    section_index: int  # 0-based
    n_sections: int
    y0: float
    y1: float
    outside: bool  # the requested y is outside the wing's span: the nearest section was used


def section_edge_sweeps(sweep_deg: float, sweep_location: float, root_chord: float, tip_chord: float,
                        span: float, dihedral_deg: float) -> tuple[float, float]:
    """(leading-edge, trailing-edge) planform sweep in degrees of one VSP wing section.

    The section's Sweep is measured at the chord fraction ``Sweep_Location`` in the wing plane, so the leading edge
    differs from it by the taper: tan(LE) = tan(sweep) + s (root - tip) / span, and the trailing edge adds the chord
    change (tip - root) / span. The wing plane is tilted by the dihedral, which stretches the spanwise axis of the
    planform: tan(planform) = tan(wing plane) / cos(dihedral). (``Sec_Sweep_Location`` does not enter; checked
    against the exported surface.)
    """
    if span <= 0.0:
        raise ValueError("wing section span must be positive")
    tan_le = math.tan(math.radians(sweep_deg)) + sweep_location * (root_chord - tip_chord) / span
    tan_te = tan_le + (tip_chord - root_chord) / span
    c = math.cos(math.radians(dihedral_deg))
    return math.degrees(math.atan(tan_le / c)), math.degrees(math.atan(tan_te / c))


def read_wings(geoms: list[GeomInfo] | None = None) -> list[WingPlanform]:
    """Planform edge sweeps of every Wing-type geom, per section, straight from the VSP parameters."""
    geoms = geoms if geoms is not None else list_geoms()
    wings = []
    for g in geoms:
        if g.type_name != WING_TYPE_NAME:
            continue
        surf = vsp.GetXSecSurf(g.id, 0)
        y = float(vsp.GetParmVal(g.id, "Y_Rel_Location", "XForm"))
        sections = []
        for i in range(1, vsp.GetNumXSec(surf)):  # XSec i holds the parameters of the section ending at it
            xs = vsp.GetXSec(surf, i)

            def parm(name, xs=xs):
                return float(vsp.GetParmVal(vsp.GetXSecParm(xs, name)))

            proj = parm("ProjectedSpan")
            le, te = section_edge_sweeps(parm("Sweep"), parm("Sweep_Location"), parm("Root_Chord"),
                                         parm("Tip_Chord"), parm("Span"), parm("Dihedral"))
            sections.append(WingSection(y, y + proj, le, te))
            y += proj
        rotated = any(abs(float(vsp.GetParmVal(g.id, k, "XForm"))) > ROTATION_TOLERANCE_DEG
                      for k in ("X_Rel_Rotation", "Y_Rel_Rotation", "Z_Rel_Rotation"))
        wings.append(WingPlanform(g.id, g.name, tuple(sections), rotated))
    return wings


def edge_sweep_at(wing: WingPlanform, y: float, edge: str = "LE") -> EdgeSweep:
    """Sweep of the leading ('LE') or trailing ('TE') edge in the section that spans ``y`` (the nearest section
    when ``y`` is outside the wing)."""
    if edge not in ("LE", "TE"):
        raise ValueError("edge must be 'LE' or 'TE'")
    if not wing.sections:
        raise ValueError(f"wing '{wing.name}' has no sections")
    y = abs(float(y))  # the right-hand side; a symmetric wing has the same sweep on the left
    index = next((i for i, s in enumerate(wing.sections) if s.y0 <= y <= s.y1), None)
    outside = index is None
    if outside:
        index = 0 if y < wing.sections[0].y0 else len(wing.sections) - 1
    s = wing.sections[index]
    return EdgeSweep(s.le_deg if edge == "LE" else s.te_deg, index, len(wing.sections), s.y0, s.y1, outside)


# ---- surface queries ---------------------------------------------------------------------------------
def _outward(normal: np.ndarray, point: np.ndarray, mesh: pv.PolyData) -> np.ndarray:
    """Flip ``normal`` to point away from the mesh's local axis (the cross-section centre at this x)."""
    b = mesh.bounds
    axis_point = np.array([point[0], 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])])
    n = normal / np.linalg.norm(normal)
    return n if np.dot(n, point - axis_point) >= 0.0 else -n


def surface_normal(mesh: pv.PolyData, point) -> np.ndarray:
    """Outward unit normal of the mesh cell closest to ``point``."""
    p = np.asarray(point, dtype=float)
    cell = mesh.find_closest_cell(p)
    return _outward(np.array(mesh.cell_data["Normals"][cell]), p, mesh)


def snap_to_surface(mesh: pv.PolyData, x: float, y: float, z_ref: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Surface point above/below (x, y) nearest to ``z_ref`` (ray along Z), and its outward normal.

    Returns None when the vertical line through (x, y) misses the mesh.
    """
    b = mesh.bounds
    pad = 0.05 * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    points, _ = mesh.ray_trace([x, y, b[5] + pad], [x, y, b[4] - pad])
    if len(points) == 0:
        return None
    hit = points[int(np.argmin(np.abs(points[:, 2] - z_ref)))]
    return hit, surface_normal(mesh, hit)


# ---- writing the duct (Phase 4) --------------------------------------------------------------------------------------
SKINNING_SIDES = ("Top", "Right", "Bottom", "Left")


def geoms_with_prefix(prefix: str) -> list[str]:
    """Ids of the geoms whose name starts with ``prefix`` (the only ones this program ever touches)."""
    return [gid for gid in vsp.FindGeoms() if vsp.GetGeomName(gid).startswith(prefix)]


def delete_geoms_with_prefix(prefix: str) -> list[str]:
    """Delete every geom whose name starts with ``prefix``; returns their names. Nothing else is touched."""
    removed = []
    for gid in geoms_with_prefix(prefix):
        removed.append(vsp.GetGeomName(gid))
        vsp.DeleteGeom(gid)
    vsp.Update()
    return removed


def write_stack(plan, settings) -> str:
    """Create the Stack geom of a ``vspexport.GeomPlan`` and return its id: one Edit Curve XSec per section (positions and
    frames from the plan), a POINT XSec at each end as the flat cap, skinning tangents along the section normals, the
    negative-volume flag and the planar XZ symmetry as planned."""
    from ductmgr import cedit, vspexport

    gid = vsp.AddGeom("STACK")
    vsp.SetGeomName(gid, plan.name)
    vsp.Update()
    surf = vsp.GetXSecSurf(gid, 0)
    n = len(plan.sections)
    while vsp.GetNumXSec(surf) > n:
        vsp.CutXSec(gid, 1)
    while vsp.GetNumXSec(surf) < n:
        vsp.InsertXSec(gid, 1, vsp.XS_ELLIPSE)
    vsp.Update()
    params = vspexport.stack_parameters(plan.sections)
    # First the whole path: every XSec (still an ellipse of about its final size) gets its position, frame and skinning
    # tangents, so no intermediate Stack is folded or wildly off while the sections are turned into Edit Curves one by one.
    for i, (section, q) in enumerate(zip(plan.sections, params)):
        xs = vsp.GetXSec(surf, i)
        if section.cedit is not None:
            vsp.SetParmVal(vsp.GetXSecParm(xs, "Ellipse_Width"), section.cedit.width)
            vsp.SetParmVal(vsp.GetXSecParm(xs, "Ellipse_Height"), section.cedit.height)
        for name, value in q.items():
            vsp.SetParmVal(vsp.GetXSecParm(xs, name), value)
        for side in SKINNING_SIDES:
            for lr in ("L", "R"):
                # along the section normal, except at a cap point: there in the section plane (a flat disc), and the
                # section next to a cap gets a weak tangent towards it so the disc does not bulge
                facing = (i == 0 and lr == "R") or (i == n - 1 and lr == "L")
                vsp.SetParmVal(vsp.GetXSecParm(xs, f"{side}{lr}Angle"), 90.0 if section.cedit is None and facing else 0.0)
                if (i == 1 and lr == "L") or (i == n - 2 and lr == "R"):
                    vsp.SetParmVal(vsp.GetXSecParm(xs, f"{side}{lr}Strength"), settings.cap_strength)
    vsp.Update()
    for i, section in enumerate(plan.sections):
        if section.cedit is not None:
            cedit.apply_cedit(cedit.make_edit_curve(surf, i), section.cedit)
    vsp.Update()
    vsp.SetParmVal(gid, "Tess_W", "Shape", float(settings.tess_w))
    vsp.SetParmVal(gid, "Tess_U", "Shape", float(settings.tess_u * (n - 1) + 1))
    vsp.SetParmVal(gid, "Negative_Volume_Flag", "Negative_Volume_Props", 1.0 if plan.negative else 0.0)
    vsp.SetParmVal(gid, "Sym_Planar_Flag", "Sym", float(vsp.SYM_XZ) if plan.mirrored else 0.0)
    vsp.Update()
    return gid


def save_model(path) -> Path:
    """Write the whole model to ``path`` (never call this with the input file)."""
    path = Path(path)
    vsp.WriteVSPFile(str(path), vsp.SET_ALL)
    if not path.is_file():
        raise RuntimeError(f"OpenVSP did not write {path}")
    return path


def param_hashes(prefix: str) -> dict[str, str]:
    """A hash of every parameter value (geom, XSec and XSec-curve parameters, in order) of each geom whose name does NOT
    start with ``prefix``, keyed by geom id: the fingerprint of everything this program must leave alone."""
    import hashlib

    out = {}
    for gid in vsp.FindGeoms():
        name = vsp.GetGeomName(gid)
        if name.startswith(prefix):
            continue
        pids = list(vsp.GetGeomParmIDs(gid))
        for k in range(vsp.GetNumXSecSurfs(gid)):
            surf = vsp.GetXSecSurf(gid, k)
            for i in range(vsp.GetNumXSec(surf)):
                pids += list(vsp.GetXSecParmIDs(vsp.GetXSec(surf, i)))
        h = hashlib.sha256()
        h.update(f"{name}|{vsp.GetGeomTypeName(gid)}|{len(pids)}\n".encode())
        for pid in pids:
            h.update(f"{vsp.GetParmGroupName(pid)}|{vsp.GetParmName(pid)}|{vsp.GetParmVal(pid)!r}\n".encode())
        out[gid] = h.hexdigest()
    return out


def comp_geom(geoms: list[GeomInfo]) -> dict:
    """Run CompGeom on a temporary set of ``geoms`` and return its numbers; the set, the mesh geom CompGeom creates and
    the visibility flags are restored afterwards. Keys: volume (theoretical: every component counted whole),
    wet_volume (m^3: what the exposed surfaces enclose, negative volumes subtracted), area (wetted), triangles,
    meshes."""
    idx = _free_set_index()
    old_name = vsp.GetSetName(idx)
    ids_before = set(vsp.FindGeoms())
    shown = {gid: vsp.GetSetFlag(gid, vsp.SET_SHOWN) for gid in ids_before}
    vsp.SetSetName(idx, TEMP_SET_NAME)
    for g in geoms:
        vsp.SetSetFlag(g.id, idx, True)
    try:
        vsp.SetAnalysisInputDefaults("CompGeom")
        vsp.SetIntAnalysisInput("CompGeom", "Set", [idx])
        vsp.SetIntAnalysisInput("CompGeom", "HalfMeshFlag", [0])
        with _without_compgeom_files():
            res = vsp.ExecAnalysis("CompGeom")

        def number(getter, name):
            v = getter(res, name)
            return v[0] if len(v) else 0

        return {"volume": float(number(vsp.GetDoubleResults, "Total_Theo_Vol")),
                "wet_volume": float(number(vsp.GetDoubleResults, "Total_Wet_Vol")),
                "area": float(number(vsp.GetDoubleResults, "Total_Wet_Area")),
                "triangles": int(number(vsp.GetIntResults, "Total_Num_Tris")),
                "meshes": int(number(vsp.GetIntResults, "Total_Num_Meshes"))}
    finally:
        for gid in set(vsp.FindGeoms()) - ids_before:
            vsp.DeleteGeom(gid)
        for g in geoms:
            vsp.SetSetFlag(g.id, idx, False)
        vsp.SetSetName(idx, old_name)
        for gid, flag in shown.items():
            vsp.SetSetFlag(gid, vsp.SET_SHOWN, flag)
