# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Distance, containment and ray queries against triangle meshes (VTK implicit distance and cell locator)."""
import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter


HOLE_SIZE_FACTOR = 1.0  # holes up to this multiple of the mesh size are capped when the mesh is not watertight


class Solid:
    """A mesh with signed-distance queries (negative inside). The signs are only meaningful for a closed mesh
    (``closed``); the magnitude is always the distance to the surface."""

    def __init__(self, mesh: pv.PolyData, name: str = ""):
        self.name = name
        m = mesh.triangulate().clean()
        self.healed = False
        if m.n_cells > 0 and m.n_open_edges > 0:  # a few open ends (e.g. an uncapped fuselage tail): cap them
            try:
                capped = m.fill_holes(float(m.length) * HOLE_SIZE_FACTOR).triangulate().clean()
                if capped.n_open_edges == 0:
                    m, self.healed = capped, True
            except Exception:  # the mesh stays open; the caller is told through ``closed``
                pass
        self.closed = m.n_cells > 0 and m.n_open_edges == 0
        if self.closed:
            m = m.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True,
                                  consistent_normals=True, split_vertices=False)
        self.mesh = m
        self._f = vtkImplicitPolyDataDistance()
        self._f.SetInput(m)

    @property
    def bounds(self):
        return self.mesh.bounds

    def signed(self, points) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        cloud = pv.PolyData(np.ascontiguousarray(pts))
        flt = vtkSampleImplicitFunctionFilter()
        flt.SetInputData(cloud)
        flt.SetImplicitFunction(self._f)
        flt.ComputeGradientsOff()
        flt.Update()
        out = pv.wrap(flt.GetOutput())
        return np.array(out.point_data["Implicit scalars"], dtype=float)

    def distance(self, points) -> np.ndarray:
        return np.abs(self.signed(points))

    def inside(self, points, tol: float = 0.0) -> np.ndarray:
        return self.signed(points) < -tol


class RayCaster:
    """Does a segment cross a surface? Backed by a static cell locator over the (merged) surface."""

    TOLERANCE = 1e-9

    def __init__(self, *meshes: pv.PolyData):
        merged = pv.merge([m.triangulate() for m in meshes]) if len(meshes) > 1 else meshes[0].triangulate()
        self.mesh = merged
        self._tree = vtkStaticCellLocator()
        self._tree.SetDataSet(merged)
        self._tree.BuildLocator()

    def blocked(self, a, b) -> bool:
        pts, ids = vtkPoints(), vtkIdList()
        return bool(self._tree.IntersectWithLine(np.asarray(a, dtype=float).tolist(),
                                                 np.asarray(b, dtype=float).tolist(), self.TOLERANCE, pts, ids))
