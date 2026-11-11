"""Explicit Q1 pressure subdivision of planar Q4 physical shell surfaces.

The virtual pressure nodes have no structural degrees of freedom. Their
positions and variations are bilinear interpolants of the original physical
skin nodes. This is an internal development primitive, not an admitted profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .s3_patch_mortar_energy import S3PatchFace


@dataclass(frozen=True)
class Q4PressureOverlay:
    faces: tuple[S3PatchFace, ...]
    structural_node_ids: tuple[int, ...]
    pressure_node_ids: tuple[int, ...]
    reference_positions: np.ndarray
    interpolation: sparse.csr_matrix

    def __post_init__(self):
        self.reference_positions.setflags(write=False)
        for face in self.faces:
            face.reference_positions.setflags(write=False)
        for values in (self.interpolation.data,
                       self.interpolation.indices,
                       self.interpolation.indptr):
            values.setflags(write=False)


def subdivide_q4_pressure_faces(
    source_faces: tuple[S3PatchFace, ...], *,
    u_divisions: int, v_divisions: int, first_pressure_node_id: int,
    reserved_node_ids: tuple[int, ...] = (),
    maximum_pressure_nodes: int, maximum_pressure_faces: int,
) -> Q4PressureOverlay:
    """Create conforming child Q4s and their sparse physical-position map.

    The caller explicitly budgets the new pressure modes and integration faces.
    Every shared material edge must have the same number of child intervals on
    both incident faces. The returned matrix maps original skin positions to
    virtual pressure-node positions; use its Kronecker product with I3 for work.
    """
    for name, value in (
        ("u_divisions", u_divisions), ("v_divisions", v_divisions),
        ("first_pressure_node_id", first_pressure_node_id),
        ("maximum_pressure_nodes", maximum_pressure_nodes),
        ("maximum_pressure_faces", maximum_pressure_faces),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not source_faces or any(len(face.node_ids) != 4 for face in source_faces):
        raise ValueError("pressure overlay needs one or more planar Q4 faces")
    if len({frozenset(face.node_ids) for face in source_faces}) != len(source_faces):
        raise ValueError("pressure overlay has duplicate material faces")
    first = source_faces[0].reference_positions
    plane_normal = np.cross(first[1]-first[0], first[3]-first[0])
    plane_normal /= np.linalg.norm(plane_normal)
    scale = max(1.0, *(float(np.max(np.abs(face.reference_positions)))
                       for face in source_faces))
    for face in source_faces:
        corners = face.reference_positions
        normal = np.cross(corners[1]-corners[0], corners[3]-corners[0])
        normal /= np.linalg.norm(normal)
        if (float(normal @ plane_normal) < 1.0-1e-10
                or float(np.max(np.abs((corners-first[0]) @ plane_normal)))
                    > 1e-10*scale):
            raise ValueError("pressure overlay needs one consistently oriented plane")
    child_count = len(source_faces) * u_divisions * v_divisions
    if child_count > maximum_pressure_faces:
        raise MemoryError("pressure overlay exceeds explicit child-face budget")
    material_nodes = tuple(sorted({node for face in source_faces
                                   for node in face.node_ids}))
    if set(material_nodes) & set(reserved_node_ids):
        raise ValueError("pressure source and reserved nodes overlap")
    if first_pressure_node_id <= max((*material_nodes, *reserved_node_ids),
                                     default=0):
        raise ValueError("virtual pressure ids must follow material and reserved ids")

    original_positions: dict[int, np.ndarray] = {}
    for face in source_faces:
        for node, position in zip(face.node_ids, face.reference_positions):
            previous = original_positions.setdefault(node, position)
            if not np.allclose(previous, position, rtol=0.0, atol=1e-12):
                raise ValueError("shared material node has inconsistent physical position")

    edge_intervals: dict[tuple[int, int], int] = {}
    for face in source_faces:
        ids = face.node_ids
        for a, b, intervals in (
            (ids[0], ids[1], u_divisions),
            (ids[1], ids[2], v_divisions),
            (ids[2], ids[3], u_divisions),
            (ids[3], ids[0], v_divisions),
        ):
            edge = (min(a, b), max(a, b))
            old = edge_intervals.setdefault(edge, intervals)
            if old != intervals:
                raise ValueError("shared edge has incompatible pressure subdivision")
    key_to_index: dict[tuple, int] = {}
    coefficients: list[dict[int, float]] = []
    child_keys: list[tuple[tuple, tuple, tuple, tuple]] = []

    def vertex_key(face_index: int, ids: tuple[int, ...], i: int, j: int):
        if (i, j) in {(0, 0), (u_divisions, 0),
                      (u_divisions, v_divisions), (0, v_divisions)}:
            corner = {(0, 0): 0, (u_divisions, 0): 1,
                      (u_divisions, v_divisions): 2,
                      (0, v_divisions): 3}[(i, j)]
            return ("corner", ids[corner])
        if j == 0:
            a, b, intervals, step = ids[0], ids[1], u_divisions, i
        elif i == u_divisions:
            a, b, intervals, step = ids[1], ids[2], v_divisions, j
        elif j == v_divisions:
            a, b, intervals, step = ids[3], ids[2], u_divisions, i
        elif i == 0:
            a, b, intervals, step = ids[0], ids[3], v_divisions, j
        else:
            return ("face", face_index, i, j)
        edge = (min(a, b), max(a, b))
        if edge_intervals[edge] != intervals:
            raise ValueError("shared edge has incompatible pressure subdivision")
        return ("edge", *edge, step if a < b else intervals - step)

    def add_vertex(face_index: int, face: S3PatchFace, i: int, j: int):
        key = vertex_key(face_index, face.node_ids, i, j)
        u, v = i / u_divisions, j / v_divisions
        weights = ((1-u)*(1-v), u*(1-v), u*v, (1-u)*v)
        terms = {node: float(weight) for node, weight in zip(face.node_ids, weights)
                 if weight > 0.0}
        if key in key_to_index:
            previous = coefficients[key_to_index[key]]
            all_nodes = set(previous) | set(terms)
            if any(abs(previous.get(node, 0.0)-terms.get(node, 0.0)) > 1e-12
                   for node in all_nodes):
                raise ValueError("shared pressure node has inconsistent work map")
        else:
            if len(coefficients) >= maximum_pressure_nodes:
                raise MemoryError("pressure overlay exceeds explicit node budget")
            key_to_index[key] = len(coefficients)
            coefficients.append(terms)
        return key

    for face_index, face in enumerate(source_faces):
        grid = {(i, j): add_vertex(face_index, face, i, j)
                for j in range(v_divisions + 1)
                for i in range(u_divisions + 1)}
        for j in range(v_divisions):
            for i in range(u_divisions):
                child_keys.append((grid[i, j], grid[i+1, j],
                                   grid[i+1, j+1], grid[i, j+1]))

    source_index = {node: index for index, node in enumerate(material_nodes)}
    row, column, data = [], [], []
    for pressure_index, terms in enumerate(coefficients):
        for node, weight in terms.items():
            row.append(pressure_index)
            column.append(source_index[node])
            data.append(weight)
    interpolation = sparse.csr_matrix(
        (data, (row, column)),
        shape=(len(coefficients), len(material_nodes)),
    )
    reference = interpolation @ np.asarray(
        [original_positions[node] for node in material_nodes], dtype=float,
    )
    pressure_ids = tuple(first_pressure_node_id + i
                         for i in range(len(coefficients)))
    child_faces = tuple(S3PatchFace(
        tuple(pressure_ids[key_to_index[key]] for key in keys),
        np.asarray([reference[key_to_index[key]] for key in keys]),
    ) for keys in child_keys)
    return Q4PressureOverlay(
        child_faces, material_nodes, pressure_ids,
        reference, interpolation,
    )
