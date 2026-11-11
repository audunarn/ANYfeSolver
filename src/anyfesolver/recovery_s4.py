"""Compiled elastic isotropic S4 stress recovery.

The kernel mirrors :meth:`ShellElement.compute_stresses` for the qualified
four-node isotropic case, including the element's MITC4 assumed-shear field and
optional local/global top/bottom tensors.  Unsupported formulations continue
through the scalar oracle in :mod:`anyfesolver.recovery`.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .jit_compiler import njit, prange
from .vectorized_stiffness import (
    _build_shell_b_matrices_jit,
    _compute_4node_shape_functions,
    _local_frame_and_derivatives_jit,
    _mitc4_shear_b_matrix_jit,
    _mitc4_shear_samples_jit,
)


BASE_FIELDS = (
    "membrane_xx",
    "membrane_yy",
    "membrane_xy",
    "bending_xx",
    "bending_yy",
    "bending_xy",
    "shear_xz",
    "shear_yz",
    "von_mises",
    "equivalent_stress",
    "hill_utilization",
)
SURFACE_FIELDS = tuple(
    f"{frame}_{component}_{surface}"
    for frame in ("local", "global")
    for surface in ("top", "bot")
    for component in ("xx", "yy", "zz", "xy", "yz", "xz")
)
NUMERIC_FIELDS = BASE_FIELDS + SURFACE_FIELDS
_SURFACE_OFFSET = len(BASE_FIELDS)


@njit(cache=True)
def _rotate_dofs_to_local(
    global_dofs: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    local = np.empty(24, dtype=np.float64)
    for node in range(4):
        for kind in range(2):
            start = 6 * node + 3 * kind
            for row in range(3):
                value = 0.0
                for column in range(3):
                    value += rotation[column, row] * global_dofs[start + column]
                local[start + row] = value
    return local


@njit(cache=True)
def _matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    for row in range(matrix.shape[0]):
        value = 0.0
        for column in range(matrix.shape[1]):
            value += matrix[row, column] * vector[column]
        result[row] = value
    return result


@njit(cache=True)
def _rotate_stress_tensor(
    rotation: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    tau_xy: float,
    tau_xz: float,
    tau_yz: float,
) -> np.ndarray:
    local = np.zeros((3, 3), dtype=np.float64)
    local[0, 0] = sigma_x
    local[1, 1] = sigma_y
    local[0, 1] = tau_xy
    local[1, 0] = tau_xy
    local[0, 2] = tau_xz
    local[2, 0] = tau_xz
    local[1, 2] = tau_yz
    local[2, 1] = tau_yz
    result = np.zeros((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            value = 0.0
            for inner_row in range(3):
                for inner_column in range(3):
                    value += (
                        rotation[row, inner_row]
                        * local[inner_row, inner_column]
                        * rotation[column, inner_column]
                    )
            result[row, column] = value
    return result


@njit(cache=True, parallel=True)
def _recover_isotropic_s4_jit(
    coords_all: np.ndarray,
    dof_mappings: np.ndarray,
    q_all: np.ndarray,
    g_all: np.ndarray,
    thickness_all: np.ndarray,
    selected_indices: np.ndarray,
    displacements: np.ndarray,
    gauss_points: np.ndarray,
    return_global: bool,
) -> np.ndarray:
    result = np.zeros(
        (selected_indices.size, gauss_points.shape[0], len(NUMERIC_FIELDS)),
        dtype=np.float64,
    )
    for output_row in prange(selected_indices.size):
        batch_row = selected_indices[output_row]
        coords = coords_all[batch_row]
        q_local = q_all[batch_row]
        g_local = g_all[batch_row]
        thickness = thickness_all[batch_row]
        global_dofs = np.empty(24, dtype=np.float64)
        for dof in range(24):
            global_dofs[dof] = displacements[dof_mappings[batch_row, dof]]

        _center_n, center_dxi, center_deta = _compute_4node_shape_functions(0.0, 0.0)
        center_rotation, _dx, _dy, _det = _local_frame_and_derivatives_jit(
            coords,
            center_dxi,
            center_deta,
        )
        planar, samples = _mitc4_shear_samples_jit(coords, center_rotation, 24)
        center_dofs = _rotate_dofs_to_local(global_dofs, center_rotation)

        for gp_index in range(gauss_points.shape[0]):
            xi = gauss_points[gp_index, 0]
            eta = gauss_points[gp_index, 1]
            shape, dxi, deta = _compute_4node_shape_functions(xi, eta)
            rotation, dshape_dx, dshape_dy, _det_j = _local_frame_and_derivatives_jit(
                coords,
                dxi,
                deta,
            )
            local_dofs = _rotate_dofs_to_local(global_dofs, rotation)
            b_membrane, b_bending, _b_shear = _build_shell_b_matrices_jit(
                shape,
                dshape_dx,
                dshape_dy,
                24,
            )
            b_mitc, _mitc_det = _mitc4_shear_b_matrix_jit(
                planar,
                samples,
                xi,
                eta,
                24,
            )
            membrane_strain = _matvec(b_membrane, local_dofs)
            curvature = _matvec(b_bending, local_dofs)
            shear_strain = _matvec(b_mitc, center_dofs)
            sigma_membrane = _matvec(q_local, membrane_strain)
            sigma_bending = _matvec(q_local, curvature)
            bending_scale = (0.5 * thickness**3) / max(thickness**2, 1.0e-12)
            for component in range(3):
                sigma_bending[component] *= bending_scale
            tau_shear = _matvec(g_local, shear_strain)
            tau_shear[0] *= 5.0 / 6.0
            tau_shear[1] *= 5.0 / 6.0

            for component in range(3):
                result[output_row, gp_index, component] = sigma_membrane[component]
                result[output_row, gp_index, 3 + component] = sigma_bending[component]
            result[output_row, gp_index, 6] = tau_shear[0]
            result[output_row, gp_index, 7] = tau_shear[1]

            top_x = sigma_membrane[0] + sigma_bending[0]
            top_y = sigma_membrane[1] + sigma_bending[1]
            top_xy = sigma_membrane[2] + sigma_bending[2]
            bottom_x = sigma_membrane[0] - sigma_bending[0]
            bottom_y = sigma_membrane[1] - sigma_bending[1]
            bottom_xy = sigma_membrane[2] - sigma_bending[2]
            shear_square = tau_shear[0] ** 2 + tau_shear[1] ** 2
            vm_top = np.sqrt(
                top_x**2
                + top_y**2
                - top_x * top_y
                + 3.0 * (top_xy**2 + shear_square)
            )
            vm_bottom = np.sqrt(
                bottom_x**2
                + bottom_y**2
                - bottom_x * bottom_y
                + 3.0 * (bottom_xy**2 + shear_square)
            )
            equivalent = max(vm_top, vm_bottom)
            result[output_row, gp_index, 8] = equivalent
            result[output_row, gp_index, 9] = equivalent

            if return_global:
                top_global = _rotate_stress_tensor(
                    rotation,
                    top_x,
                    top_y,
                    top_xy,
                    tau_shear[0],
                    tau_shear[1],
                )
                bottom_global = _rotate_stress_tensor(
                    rotation,
                    bottom_x,
                    bottom_y,
                    bottom_xy,
                    tau_shear[0],
                    tau_shear[1],
                )
                local_values = np.array(
                    [
                        top_x,
                        top_y,
                        0.0,
                        top_xy,
                        tau_shear[1],
                        tau_shear[0],
                        bottom_x,
                        bottom_y,
                        0.0,
                        bottom_xy,
                        tau_shear[1],
                        tau_shear[0],
                    ]
                )
                for field in range(12):
                    result[output_row, gp_index, _SURFACE_OFFSET + field] = local_values[field]
                global_offset = _SURFACE_OFFSET + 12
                result[output_row, gp_index, global_offset + 0] = top_global[0, 0]
                result[output_row, gp_index, global_offset + 1] = top_global[1, 1]
                result[output_row, gp_index, global_offset + 2] = top_global[2, 2]
                result[output_row, gp_index, global_offset + 3] = top_global[0, 1]
                result[output_row, gp_index, global_offset + 4] = top_global[1, 2]
                result[output_row, gp_index, global_offset + 5] = top_global[0, 2]
                result[output_row, gp_index, global_offset + 6] = bottom_global[0, 0]
                result[output_row, gp_index, global_offset + 7] = bottom_global[1, 1]
                result[output_row, gp_index, global_offset + 8] = bottom_global[2, 2]
                result[output_row, gp_index, global_offset + 9] = bottom_global[0, 1]
                result[output_row, gp_index, global_offset + 10] = bottom_global[1, 2]
                result[output_row, gp_index, global_offset + 11] = bottom_global[0, 2]
    return result


def recover_isotropic_s4(
    batch: object,
    selected_indices: np.ndarray,
    displacements: np.ndarray,
    *,
    return_global: bool,
) -> Mapping[int, Dict[str, np.ndarray]]:
    """Evaluate selected rows of a ``RecoveryS4Batch`` into public dictionaries."""

    indices = np.asarray(selected_indices, dtype=np.intp).reshape(-1)
    numeric = _recover_isotropic_s4_jit(
        batch.coords,
        batch.dof_mappings,
        batch.q_local,
        batch.g_local,
        batch.thickness,
        indices,
        np.asarray(displacements, dtype=float),
        batch.gauss_points,
        bool(return_global),
    )
    field_count = len(NUMERIC_FIELDS) if return_global else len(BASE_FIELDS)
    recovered: Dict[int, Dict[str, np.ndarray]] = {}
    for output_row, batch_row in enumerate(indices):
        values = {
            field: numeric[output_row, :, field_index].copy()
            for field_index, field in enumerate(NUMERIC_FIELDS[:field_count])
        }
        values["equivalent_stress_measure"] = "von_mises"
        recovered[int(batch.element_ids[int(batch_row)])] = values
    return recovered
