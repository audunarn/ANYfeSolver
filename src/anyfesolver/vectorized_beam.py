"""Compiled batch kernels for ordinary elastic quadratic beam elements."""

from __future__ import annotations

import numpy as np

from .jit_compiler import njit, prange


@njit(cache=True, parallel=True)
def quadratic_beam_nonlinear_into_buffers_jit(
    displacements: np.ndarray,
    dof_mappings: np.ndarray,
    transforms: np.ndarray,
    strain_rows: np.ndarray,
    jacobian_weights: np.ndarray,
    properties: np.ndarray,
    force_positions: np.ndarray,
    tangent_positions: np.ndarray,
    force_values: np.ndarray,
    tangent_values: np.ndarray,
    tangent: bool,
) -> None:
    """Evaluate straight elastic Beam3 von Karman responses in one batch.

    ``strain_rows`` stores, in order, axial, transverse-y, transverse-z,
    curvature-y, curvature-z, torsion, shear-xy and shear-xz rows.  The
    algebra is the exact scalar :class:`QuadraticBeamElement` potential and
    consistent tangent, including the axial-force geometric term.
    """

    count = dof_mappings.shape[0]
    for element_index in prange(count):
        transform = transforms[element_index]
        local_displacement = np.zeros(18, dtype=np.float64)
        for row in range(18):
            value = 0.0
            for column in range(18):
                value += transform[row, column] * displacements[
                    dof_mappings[element_index, column]
                ]
            local_displacement[row] = value

        local_force = np.zeros(18, dtype=np.float64)
        local_tangent = np.zeros((18, 18), dtype=np.float64)
        EA = properties[element_index, 0]
        EIy = properties[element_index, 1]
        EIz = properties[element_index, 2]
        GAy = properties[element_index, 3]
        GAz = properties[element_index, 4]
        GJ = properties[element_index, 5]

        for gp_index in range(3):
            rows = strain_rows[element_index, gp_index]
            b_axial = rows[0]
            b_v = rows[1]
            b_w = rows[2]
            b_ry = rows[3]
            b_rz = rows[4]
            b_torsion = rows[5]
            b_shear_xy = rows[6]
            b_shear_xz = rows[7]
            v_gradient = 0.0
            w_gradient = 0.0
            axial_linear = 0.0
            curvature_y = 0.0
            curvature_z = 0.0
            twist = 0.0
            shear_xy = 0.0
            shear_xz = 0.0
            for dof in range(18):
                displacement = local_displacement[dof]
                v_gradient += b_v[dof] * displacement
                w_gradient += b_w[dof] * displacement
                axial_linear += b_axial[dof] * displacement
                curvature_y += b_ry[dof] * displacement
                curvature_z += b_rz[dof] * displacement
                twist += b_torsion[dof] * displacement
                shear_xy += b_shear_xy[dof] * displacement
                shear_xz += b_shear_xz[dof] * displacement
            axial_strain = axial_linear + 0.5 * (
                v_gradient * v_gradient + w_gradient * w_gradient
            )
            axial_force = EA * axial_strain
            jacobian_weight = jacobian_weights[element_index, gp_index]
            membrane = np.empty(18, dtype=np.float64)
            for row in range(18):
                membrane[row] = (
                    b_axial[row]
                    + v_gradient * b_v[row]
                    + w_gradient * b_w[row]
                )
                local_force[row] += jacobian_weight * (
                    axial_force * membrane[row]
                    + EIy * curvature_y * b_ry[row]
                    + EIz * curvature_z * b_rz[row]
                    + GAy * shear_xy * b_shear_xy[row]
                    + GAz * shear_xz * b_shear_xz[row]
                    + GJ * twist * b_torsion[row]
                )
            if tangent:
                for row in range(18):
                    for column in range(18):
                        local_tangent[row, column] += jacobian_weight * (
                            EA * membrane[row] * membrane[column]
                            + axial_force
                            * (
                                b_v[row] * b_v[column]
                                + b_w[row] * b_w[column]
                            )
                            + EIy * b_ry[row] * b_ry[column]
                            + EIz * b_rz[row] * b_rz[column]
                            + GAy * b_shear_xy[row] * b_shear_xy[column]
                            + GAz * b_shear_xz[row] * b_shear_xz[column]
                            + GJ * b_torsion[row] * b_torsion[column]
                        )

        for global_row in range(18):
            force_value = 0.0
            for local_row in range(18):
                force_value += transform[local_row, global_row] * local_force[local_row]
            force_values[force_positions[element_index, global_row]] = force_value
        if tangent:
            for global_row in range(18):
                for global_column in range(18):
                    value = 0.0
                    for local_row in range(18):
                        left = transform[local_row, global_row]
                        if left == 0.0:
                            continue
                        for local_column in range(18):
                            value += (
                                left
                                * local_tangent[local_row, local_column]
                                * transform[local_column, global_column]
                            )
                    tangent_values[
                        tangent_positions[element_index, global_row * 18 + global_column]
                    ] = value
