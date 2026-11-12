"""Physical S3 V2D skin position and work map for contact candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .overlap import reference_shell_face


@dataclass(frozen=True)
class S3SkinWorkAdapter:
    body_id: int
    element_id: int
    side: str
    node_ids: tuple[int, int, int]
    director: np.ndarray
    physical_offset: float
    reference_positions: np.ndarray
    work_map: sparse.csr_matrix

    @classmethod
    def from_model(cls, model, body_id: int, element_id: int,
                   side: str) -> S3SkinWorkAdapter:
        from anyfesolver.e4_pl_s3_v2d_element import (
            NativeParityE4PLS3V2DShellElement,
        )

        element = model.mesh.elements.get(element_id)
        if type(element) is not NativeParityE4PLS3V2DShellElement:
            raise ValueError(f"element {element_id} is not the admitted S3 V2D formulation")
        coordinate = {"bottom": -1.0, "midsurface": 0.0, "top": 1.0}.get(side)
        if coordinate is None:
            raise ValueError("S3 skin side must be bottom, midsurface, or top")
        reference = reference_shell_face(model, body_id, element_id, side)
        director = np.asarray(element.physical_reference_director, dtype=float).copy()
        if (director.shape != (3,) or not np.all(np.isfinite(director))
                or abs(float(np.linalg.norm(director)) - 1.0) > 1e-10):
            raise ValueError("S3 physical director must be a finite unit vector")
        offset = float(element.material_surface_offset_from_reference(coordinate))
        if not np.isfinite(offset):
            raise ValueError("S3 physical skin offset must be finite")
        node_ids = tuple(int(node) for node in element.node_ids)
        if len(node_ids) != 3 or len(set(node_ids)) != 3:
            raise ValueError("S3 skin requires three distinct material nodes")
        dofs = np.asarray([
            model.mesh.dof_manager.get_node_dofs(node)[:6]
            for node in node_ids
        ], dtype=np.intp)
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        basis = np.eye(3)
        for local, nodal in enumerate(dofs):
            for component in range(3):
                rows.append(3*local + component)
                columns.append(int(nodal[component]))
                data.append(1.0)
            for rotation in range(3):
                variation = offset * np.cross(basis[rotation], director)
                for component, coefficient in enumerate(variation):
                    if coefficient:
                        rows.append(3*local + component)
                        columns.append(int(nodal[3 + rotation]))
                        data.append(float(coefficient))
        mapping = sparse.coo_matrix(
            (data, (rows, columns)),
            shape=(9, model.mesh.dof_manager.total_dofs),
        ).tocsr()
        director.setflags(write=False)
        reference_positions = reference.corners.copy()
        reference_positions.setflags(write=False)
        return cls(int(body_id), int(element_id), side, node_ids,
                   director, offset, reference_positions, mapping)

    def current_positions(self, displacement: np.ndarray) -> np.ndarray:
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if (u.size != self.work_map.shape[1]
                or not np.all(np.isfinite(u))):
            raise ValueError("S3 skin displacement has invalid size or values")
        return self.reference_positions + np.asarray(
            self.work_map @ u,
        ).reshape(3, 3)

    def pullback(self, physical_gradient: np.ndarray,
                 physical_hessian: np.ndarray | None = None
                 ) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        gradient = np.asarray(physical_gradient, dtype=float).reshape(-1)
        if gradient.size != 9 or not np.all(np.isfinite(gradient)):
            raise ValueError("S3 skin physical gradient needs nine finite entries")
        global_gradient = np.asarray(self.work_map.T @ gradient).reshape(-1)
        if physical_hessian is None:
            return global_gradient, None
        hessian = np.asarray(physical_hessian, dtype=float)
        if hessian.shape != (9, 9) or not np.all(np.isfinite(hessian)):
            raise ValueError("S3 skin physical Hessian needs a finite 9x9 array")
        global_hessian = (self.work_map.T @ sparse.csr_matrix(hessian)
                          @ self.work_map).tocsr()
        return global_gradient, global_hessian
