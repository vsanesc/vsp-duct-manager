# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Read-only access to an OpenVSP model (Phase 2): geoms, surface meshes, engine face.

Surfaces are obtained through the OpenVSP API: the wanted geoms are put in a temporary set, exported to a
temporary STL, and loaded with PyVista. The temporary set is restored and the STL deleted afterwards.
Nothing here saves the model.
"""
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
            vsp.ExportFile(str(stl), idx, vsp.EXPORT_STL)
            if not stl.is_file():
                raise RuntimeError("OpenVSP did not write the temporary STL")
            mesh = pv.read(str(stl))
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
