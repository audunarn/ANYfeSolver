"""Compiled S4 batches for orthotropic and generalized shell sections.

The scalar :class:`~anyfesolver.elements.ShellElement` remains the reference
implementation.  This module only handles four-node quadrilateral shells and
keeps the same integration rules, material-axis construction, MITC4 shear
field, drilling stabilization, and blockwise local/global transformation.

The generalized coupling matrix is deliberately allowed to be nonsymmetric.
The energy-symmetric section operator is assembled as ``[A B; B.T D]``; in
particular, the lower coupling block must never be replaced by ``B``.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

from .jit_compiler import njit, prange
from .vectorized_stiffness import (
    _build_drilling_b_matrix_jit,
    _build_shell_b_matrices_jit,
    _compute_4node_shape_functions,
    _local_frame_and_derivatives_jit,
    _mitc4_shear_b_matrix_jit,
    _mitc4_shear_samples_jit,
)


@njit(cache=True)
def _material_angle_jit(
    frame: np.ndarray,
    direction: np.ndarray,
    has_direction: bool,
    angle_offset: float,
) -> float:
    """Match ``ShellElement._material_angle`` for one local frame."""

    if not has_direction:
        return angle_offset
    component_0 = (
        direction[0] * frame[0, 0]
        + direction[1] * frame[1, 0]
        + direction[2] * frame[2, 0]
    )
    component_1 = (
        direction[0] * frame[0, 1]
        + direction[1] * frame[1, 1]
        + direction[2] * frame[2, 1]
    )
    in_plane_norm = np.sqrt(component_0 * component_0 + component_1 * component_1)
    direction_norm = np.sqrt(
        direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2]
    )
    if in_plane_norm <= 1.0e-10 * max(direction_norm, 1.0):
        raise ValueError("shell material_direction is parallel to the shell normal")
    return np.arctan2(component_1, component_0) + angle_offset


@njit(cache=True)
def _in_plane_transforms_jit(
    angle: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Engineering strain/resultant and transverse-shear transforms."""

    c = np.cos(angle)
    s = np.sin(angle)
    axes = np.array([[c, -s], [s, c]])
    strain_to_section = np.zeros((3, 3))
    resultant_to_local = np.zeros((3, 3))
    for column in range(3):
        engineering = np.zeros(3)
        engineering[column] = 1.0
        local_tensor = np.array(
            [
                [engineering[0], 0.5 * engineering[2]],
                [0.5 * engineering[2], engineering[1]],
            ]
        )
        section_tensor = axes.T @ local_tensor @ axes
        strain_to_section[0, column] = section_tensor[0, 0]
        strain_to_section[1, column] = section_tensor[1, 1]
        strain_to_section[2, column] = 2.0 * section_tensor[0, 1]

        resultant = np.zeros(3)
        resultant[column] = 1.0
        section_resultant = np.array(
            [
                [resultant[0], resultant[2]],
                [resultant[2], resultant[1]],
            ]
        )
        local_resultant = axes @ section_resultant @ axes.T
        resultant_to_local[0, column] = local_resultant[0, 0]
        resultant_to_local[1, column] = local_resultant[1, 1]
        resultant_to_local[2, column] = local_resultant[0, 1]
    return strain_to_section, resultant_to_local, axes


@njit(cache=True)
def _rotate_section_jit(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    As: np.ndarray,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate section-axis matrices with the scalar section convention."""

    strain, resultant, axes = _in_plane_transforms_jit(angle)
    abd = np.zeros((6, 6))
    abd[:3, :3] = A
    abd[:3, 3:] = B
    abd[3:, :3] = B.T
    abd[3:, 3:] = D
    strain6 = np.zeros((6, 6))
    resultant6 = np.zeros((6, 6))
    strain6[:3, :3] = strain
    strain6[3:, 3:] = strain
    resultant6[:3, :3] = resultant
    resultant6[3:, 3:] = resultant
    local = resultant6 @ abd @ strain6
    local = 0.5 * (local + local.T)
    shear = axes @ As @ axes.T
    shear = 0.5 * (shear + shear.T)
    return local[:3, :3], local[:3, 3:], local[3:, 3:], shear


@njit(cache=True)
def _accumulate_rotated_matrix_jit(
    target: np.ndarray,
    local: np.ndarray,
    frame: np.ndarray,
) -> None:
    """Accumulate ``T.T @ local @ T`` using direct 3x3 blocks."""

    for block_i in range(8):
        for block_j in range(8):
            for row in range(3):
                for column in range(3):
                    value = 0.0
                    for inner_0 in range(3):
                        for inner_1 in range(3):
                            value += (
                                frame[row, inner_0]
                                * local[3 * block_i + inner_0, 3 * block_j + inner_1]
                                * frame[column, inner_1]
                            )
                    target[3 * block_i + row, 3 * block_j + column] += value


@njit(cache=True, parallel=True)
def compute_s4_generalized_stiffness_matrices_jit(
    coords_all: np.ndarray,
    A_all: np.ndarray,
    B_all: np.ndarray,
    D_all: np.ndarray,
    As_all: np.ndarray,
    material_directions: np.ndarray,
    has_material_direction: np.ndarray,
    angle_offsets: np.ndarray,
    drilling_constants: np.ndarray,
    drilling_from_a22: np.ndarray,
    drilling_stabilization: np.ndarray,
    gauss_points: np.ndarray,
    gauss_weights: np.ndarray,
) -> np.ndarray:
    """Return batched S4 stiffness matrices for section-axis ABD/As inputs."""

    element_count = coords_all.shape[0]
    matrices = np.zeros((element_count, 24, 24))
    for element_index in prange(element_count):
        coords = coords_all[element_index]
        stiffness = np.zeros((24, 24))
        for gp_index in range(gauss_points.shape[0]):
            xi = gauss_points[gp_index, 0]
            eta = gauss_points[gp_index, 1]
            weight = gauss_weights[gp_index]
            shape, derivative_xi, derivative_eta = _compute_4node_shape_functions(
                xi, eta
            )
            frame, derivative_x, derivative_y, determinant = (
                _local_frame_and_derivatives_jit(
                    coords, derivative_xi, derivative_eta
                )
            )
            angle = _material_angle_jit(
                frame,
                material_directions[element_index],
                has_material_direction[element_index],
                angle_offsets[element_index],
            )
            A, B, D, _As = _rotate_section_jit(
                A_all[element_index],
                B_all[element_index],
                D_all[element_index],
                As_all[element_index],
                angle,
            )
            B_m, B_b, _B_s = _build_shell_b_matrices_jit(
                shape, derivative_x, derivative_y, 24
            )
            B_d = _build_drilling_b_matrix_jit(
                shape, derivative_x, derivative_y, 24
            )
            drilling = drilling_constants[element_index]
            if drilling_from_a22[element_index]:
                drilling = A[2, 2] * drilling_stabilization[element_index]
            scale = determinant * weight
            local = np.zeros((24, 24))
            for row in range(24):
                for column in range(24):
                    value = 0.0
                    for i in range(3):
                        for j in range(3):
                            value += B_m[i, row] * (
                                A[i, j] * B_m[j, column]
                                + B[i, j] * B_b[j, column]
                            )
                            # Lower generalized coupling block is B.T.
                            value += B_b[i, row] * (
                                B[j, i] * B_m[j, column]
                                + D[i, j] * B_b[j, column]
                            )
                    value += B_d[0, row] * drilling * B_d[0, column]
                    local[row, column] = value * scale
            _accumulate_rotated_matrix_jit(stiffness, local, frame)

        # MITC4 transverse shear uses the element-centre frame in the scalar
        # element and is integrated on the fixed 2x2 rule.
        shape, derivative_xi, derivative_eta = _compute_4node_shape_functions(
            0.0, 0.0
        )
        center_frame, _dx, _dy, _det = _local_frame_and_derivatives_jit(
            coords, derivative_xi, derivative_eta
        )
        center_angle = _material_angle_jit(
            center_frame,
            material_directions[element_index],
            has_material_direction[element_index],
            angle_offsets[element_index],
        )
        _A, _B, _D, shear = _rotate_section_jit(
            A_all[element_index],
            B_all[element_index],
            D_all[element_index],
            As_all[element_index],
            center_angle,
        )
        planar, samples = _mitc4_shear_samples_jit(coords, center_frame, 24)
        root = np.sqrt(3.0)
        shear_points = np.array(
            [
                [-1.0 / root, -1.0 / root],
                [1.0 / root, -1.0 / root],
                [-1.0 / root, 1.0 / root],
                [1.0 / root, 1.0 / root],
            ]
        )
        for shear_index in range(4):
            B_s, determinant = _mitc4_shear_b_matrix_jit(
                planar,
                samples,
                shear_points[shear_index, 0],
                shear_points[shear_index, 1],
                24,
            )
            local = np.zeros((24, 24))
            for row in range(24):
                for column in range(24):
                    value = 0.0
                    for i in range(2):
                        for j in range(2):
                            value += B_s[i, row] * shear[i, j] * B_s[j, column]
                    local[row, column] = value * determinant
            _accumulate_rotated_matrix_jit(stiffness, local, center_frame)
        matrices[element_index] = stiffness
    return matrices


@njit(cache=True, parallel=True)
def compute_s4_section_mass_matrices_jit(
    coords_all: np.ndarray,
    mass_per_area: np.ndarray,
    rotary_inertia_per_area: np.ndarray,
    gauss_points: np.ndarray,
    gauss_weights: np.ndarray,
) -> np.ndarray:
    """Batched consistent S4 mass with per-element section properties."""

    element_count = coords_all.shape[0]
    matrices = np.zeros((element_count, 24, 24))
    for element_index in prange(element_count):
        coords = coords_all[element_index]
        matrix = np.zeros((24, 24))
        for gp_index in range(gauss_points.shape[0]):
            shape, derivative_xi, derivative_eta = _compute_4node_shape_functions(
                gauss_points[gp_index, 0], gauss_points[gp_index, 1]
            )
            _frame, _dx, _dy, determinant = _local_frame_and_derivatives_jit(
                coords, derivative_xi, derivative_eta
            )
            scale = determinant * gauss_weights[gp_index]
            for node_i in range(4):
                for node_j in range(4):
                    factor = shape[node_i] * shape[node_j] * scale
                    translation = mass_per_area[element_index] * factor
                    rotation = rotary_inertia_per_area[element_index] * factor
                    for component in range(3):
                        matrix[6 * node_i + component, 6 * node_j + component] += (
                            translation
                        )
                        matrix[
                            6 * node_i + 3 + component,
                            6 * node_j + 3 + component,
                        ] += rotation
        matrices[element_index] = matrix
    return matrices


def prepare_s4_generalized_stiffness_batch(
    model: Any,
    elements: Sequence[Any],
) -> Tuple[np.ndarray, dict[str, int]]:
    """Prepare exact section-axis inputs and run the compiled S4 kernel."""

    from .elements import _elastic_compliance, _elastic_symmetry

    count = len(elements)
    coords = np.empty((count, 4, 3), dtype=float)
    A = np.empty((count, 3, 3), dtype=float)
    B = np.zeros((count, 3, 3), dtype=float)
    D = np.empty((count, 3, 3), dtype=float)
    As = np.empty((count, 2, 2), dtype=float)
    directions = np.zeros((count, 3), dtype=float)
    has_direction = np.zeros(count, dtype=np.bool_)
    angles = np.empty(count, dtype=float)
    drilling_constants = np.zeros(count, dtype=float)
    drilling_from_a22 = np.zeros(count, dtype=np.bool_)
    drilling_stabilization = np.empty(count, dtype=float)
    orthotropic_count = 0
    generalized_count = 0

    membrane_indices = np.array([0, 1, 5], dtype=np.intp)
    shear_indices = np.array([4, 3], dtype=np.intp)
    for index, element in enumerate(elements):
        if not bool(getattr(element, "_is_4node", False)):
            raise ValueError("advanced shell batch only supports S4 elements")
        coords[index] = element.get_node_coordinates(model.mesh)
        angles[index] = np.deg2rad(float(element.material_angle_deg))
        drilling_stabilization[index] = float(element.drilling_stabilization)
        if element.material_direction is not None:
            directions[index] = np.asarray(element.material_direction, dtype=float)
            has_direction[index] = True

        section = element.shell_section
        if section is not None:
            A[index] = section.A
            B[index] = section.B
            D[index] = section.D
            As[index] = section.As
            drilling_from_a22[index] = True
            generalized_count += 1
            continue

        material = model.get_material(element.material_name)
        if _elastic_symmetry(material) != "orthotropic":
            raise ValueError(
                "advanced shell batch expects an orthotropic material or generalized section"
            )
        compliance = _elastic_compliance(material)
        Q = np.linalg.inv(compliance[np.ix_(membrane_indices, membrane_indices)])
        G = np.linalg.inv(compliance[np.ix_(shear_indices, shear_indices)])
        thickness = float(element.thickness)
        A[index] = thickness * Q
        D[index] = thickness**3 / 12.0 * Q
        As[index] = (5.0 / 6.0) * thickness * G
        drilling_constants[index] = (
            (1.0 / float(compliance[5, 5]))
            * thickness
            * float(element.drilling_stabilization)
        )
        orthotropic_count += 1

    first = elements[0]
    matrices = compute_s4_generalized_stiffness_matrices_jit(
        np.ascontiguousarray(coords),
        np.ascontiguousarray(A),
        np.ascontiguousarray(B),
        np.ascontiguousarray(D),
        np.ascontiguousarray(As),
        np.ascontiguousarray(directions),
        np.ascontiguousarray(has_direction),
        np.ascontiguousarray(angles),
        np.ascontiguousarray(drilling_constants),
        np.ascontiguousarray(drilling_from_a22),
        np.ascontiguousarray(drilling_stabilization),
        np.ascontiguousarray(first.gauss_points, dtype=float),
        np.ascontiguousarray(first.gauss_weights, dtype=float),
    )
    return matrices, {
        "orthotropic_element_count": int(orthotropic_count),
        "generalized_element_count": int(generalized_count),
    }


def prepare_s4_section_mass_batch(model: Any, elements: Sequence[Any]) -> np.ndarray:
    """Prepare per-element section mass inputs and run the compiled kernel."""

    count = len(elements)
    coords = np.empty((count, 4, 3), dtype=float)
    mass = np.empty(count, dtype=float)
    rotary = np.empty(count, dtype=float)
    for index, element in enumerate(elements):
        if not bool(getattr(element, "_is_4node", False)):
            raise ValueError("section mass batch only supports S4 elements")
        coords[index] = element.get_node_coordinates(model.mesh)
        material = model.get_material(element.material_name)
        section = element.shell_section
        mass[index] = (
            float(section.mass_per_area)
            if section is not None and section.mass_per_area is not None
            else float(material.density) * float(element.thickness)
        )
        rotary[index] = (
            float(section.rotary_inertia_per_area)
            if section is not None
            and section.rotary_inertia_per_area is not None
            else float(material.density) * float(element.thickness) ** 3 / 12.0
        )
    first = elements[0]
    return compute_s4_section_mass_matrices_jit(
        np.ascontiguousarray(coords),
        np.ascontiguousarray(mass),
        np.ascontiguousarray(rotary),
        np.ascontiguousarray(first.gauss_points, dtype=float),
        np.ascontiguousarray(first.gauss_weights, dtype=float),
    )


__all__ = [
    "compute_s4_generalized_stiffness_matrices_jit",
    "compute_s4_section_mass_matrices_jit",
    "prepare_s4_generalized_stiffness_batch",
    "prepare_s4_section_mass_batch",
]
