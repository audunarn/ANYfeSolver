"""Planar qualified Q4 physical skin/work map for small-rotation contact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .overlap import reference_shell_face


@dataclass(frozen=True)
class Q4SkinWorkAdapter:
    body_id: int
    element_id: int
    side: str
    node_ids: tuple[int, int, int, int]
    director: np.ndarray
    physical_offset: float
    reference_positions: np.ndarray
    work_map: sparse.csr_matrix

    @classmethod
    def from_model(cls, model, body_id: int, element_id: int,
                   side: str) -> Q4SkinWorkAdapter:
        from anyfesolver.e4_pl_element import QualifiedE4PLShellElement

        element = model.mesh.elements.get(element_id)
        if type(element) is not QualifiedE4PLShellElement:
            raise ValueError(f"element {element_id} is not the admitted qualified Q4")
        coordinate = {"bottom": -1.0, "midsurface": 0.0, "top": 1.0}.get(side)
        if coordinate is None:
            raise ValueError("Q4 skin side must be bottom, midsurface, or top")
        reference = reference_shell_face(model, body_id, element_id, side)
        if reference.corners.shape != (4, 3):
            raise ValueError("Q4 physical skin needs four planar corners")
        director = element.physical_reference_director
        if director is None:
            nodes = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            director = np.cross(nodes[1]-nodes[0], nodes[3]-nodes[0])
            director /= np.linalg.norm(director)
        director = np.asarray(director, dtype=float).copy()
        if (director.shape != (3,) or not np.all(np.isfinite(director))
                or abs(float(np.linalg.norm(director))-1.0) > 1e-10):
            raise ValueError("Q4 physical director must be a finite unit vector")
        offset = 0.5*float(element.thickness)*coordinate
        if not np.isfinite(offset):
            raise ValueError("Q4 skin offset must be finite")
        node_ids = tuple(int(node) for node in element.node_ids)
        if len(node_ids) != 4 or len(set(node_ids)) != 4:
            raise ValueError("Q4 skin needs four distinct material nodes")
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
                rows.append(3*local+component)
                columns.append(int(nodal[component]))
                data.append(1.0)
            for rotation in range(3):
                variation = offset*np.cross(basis[rotation], director)
                for component, coefficient in enumerate(variation):
                    if coefficient:
                        rows.append(3*local+component)
                        columns.append(int(nodal[3+rotation]))
                        data.append(float(coefficient))
        mapping = sparse.coo_matrix(
            (data, (rows, columns)),
            shape=(12, model.mesh.dof_manager.total_dofs),
        ).tocsr()
        director.setflags(write=False)
        reference_positions = reference.corners.copy()
        reference_positions.setflags(write=False)
        return cls(int(body_id), int(element_id), side, node_ids,
                   director, offset, reference_positions, mapping)

    def current_positions(self, displacement: np.ndarray) -> np.ndarray:
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != self.work_map.shape[1] or not np.all(np.isfinite(u)):
            raise ValueError("Q4 skin displacement has invalid size or values")
        return self.reference_positions + np.asarray(self.work_map@u).reshape(4, 3)
