"""Batch B optimizations for persistent nonlinear shell assembly.

This module completes the nonlinear-assembly redesign for ordinary Python runs
with Numba available.  Elastic isotropic, orthotropic, and pre-integrated
generalized S4 groups use a dedicated in-place kernel that writes element force
and tangent entries directly into the persistent assembly plan buffers.  The
plastic shell path and unsupported element formulations remain unchanged.

The elastic path avoids, per Newton evaluation:

* plastic-strain and hardening-state work arrays,
* through-thickness layer-strain arrays,
* broadcast constitutive tensor batches,
* per-batch force/tangent result arrays, and
* Python reconstruction of new elastic state arrays.

The kernel relies on the shell transformation's existing 3x3 block-diagonal
layout (translations and rotations for every node).  Each local 3x3 tangent
block is transformed in place, so no second dense element tangent batch is
required.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .jit_compiler import njit, prange
from .elements import (
    _elastic_compliance,
    _shell_field_rows,
    _shell_material_matrices,
)
from .plasticity import lobatto_layers
from . import nonlinear_performance as _performance
from .nonlinear_element_evaluation import (
    evaluate_nonlinear_element,
    require_legacy_direct_nonlinear_element,
)
from .nonlinear_state import (
    NonlinearStateStore,
    begin_state_evaluation,
    finish_state_evaluation,
)


@njit(cache=True, parallel=True)
def _elastic_shell_batch_into_buffers(
    displacements: np.ndarray,
    dof_mappings: np.ndarray,
    T0_batch: np.ndarray,
    B_m_batch: np.ndarray,
    B_b_batch: np.ndarray,
    B_d_batch: np.ndarray,
    Gw_batch: np.ndarray,
    detw_batch: np.ndarray,
    B_s_batch: np.ndarray,
    detw_shear_batch: np.ndarray,
    membrane_matrices: np.ndarray,
    coupling_matrices: np.ndarray,
    bending_matrices: np.ndarray,
    initial_membrane_resultants: np.ndarray,
    initial_bending_resultants: np.ndarray,
    shear_matrices: np.ndarray,
    drilling_stiffnesses: np.ndarray,
    force_positions: np.ndarray,
    tangent_positions: np.ndarray,
    force_values: np.ndarray,
    tangent_values: np.ndarray,
    u_local_work: np.ndarray,
    tangent: bool,
) -> None:
    """Evaluate elastic geometrically nonlinear shells into persistent buffers."""
    n_elem = dof_mappings.shape[0]
    n_dof = dof_mappings.shape[1]
    n_gp = detw_batch.shape[1]
    n_shear = detw_shear_batch.shape[1]

    for element_index in prange(n_elem):
        T0 = T0_batch[element_index]
        u_local = u_local_work[element_index]
        membrane_matrix = membrane_matrices[element_index]
        coupling_matrix = coupling_matrices[element_index]
        bending_matrix = bending_matrices[element_index]
        shear_matrix = shear_matrices[element_index]
        drilling_stiffness = drilling_stiffnesses[element_index]

        # Gather global element DOFs and transform to the reference local frame.
        for local_row in range(n_dof):
            value = 0.0
            for local_col in range(n_dof):
                value += (
                    T0[local_row, local_col]
                    * displacements[dof_mappings[element_index, local_col]]
                )
            u_local[local_row] = value

        # The persistent plan clears the flat buffers before this kernel.  Work
        # directly in the element's assigned slices.
        force_pos = force_positions[element_index]
        tangent_pos = tangent_positions[element_index]

        B_eff = np.empty((3, n_dof), dtype=np.float64)
        membrane_times_B = np.empty((3, n_dof), dtype=np.float64)
        bending_times_B = np.empty((3, n_dof), dtype=np.float64)

        for gp_index in range(n_gp):
            B_m = B_m_batch[element_index, gp_index]
            B_b = B_b_batch[element_index, gp_index]
            B_d = B_d_batch[element_index, gp_index]
            Gw = Gw_batch[element_index, gp_index]
            detw = detw_batch[element_index, gp_index]

            theta_0 = 0.0
            theta_1 = 0.0
            membrane_strain_0 = 0.0
            membrane_strain_1 = 0.0
            membrane_strain_2 = 0.0
            curvature_0 = 0.0
            curvature_1 = 0.0
            curvature_2 = 0.0
            drilling_rotation = 0.0

            for dof_index in range(n_dof):
                displacement = u_local[dof_index]
                theta_0 += Gw[0, dof_index] * displacement
                theta_1 += Gw[1, dof_index] * displacement
                membrane_strain_0 += B_m[0, dof_index] * displacement
                membrane_strain_1 += B_m[1, dof_index] * displacement
                membrane_strain_2 += B_m[2, dof_index] * displacement
                curvature_0 += B_b[0, dof_index] * displacement
                curvature_1 += B_b[1, dof_index] * displacement
                curvature_2 += B_b[2, dof_index] * displacement
                drilling_rotation += B_d[0, dof_index] * displacement

            membrane_strain_0 += 0.5 * theta_0 * theta_0
            membrane_strain_1 += 0.5 * theta_1 * theta_1
            membrane_strain_2 += theta_0 * theta_1

            N_0 = (
                membrane_matrix[0, 0] * membrane_strain_0
                + membrane_matrix[0, 1] * membrane_strain_1
                + membrane_matrix[0, 2] * membrane_strain_2
                + coupling_matrix[0, 0] * curvature_0
                + coupling_matrix[0, 1] * curvature_1
                + coupling_matrix[0, 2] * curvature_2
            )
            N_1 = (
                membrane_matrix[1, 0] * membrane_strain_0
                + membrane_matrix[1, 1] * membrane_strain_1
                + membrane_matrix[1, 2] * membrane_strain_2
                + coupling_matrix[1, 0] * curvature_0
                + coupling_matrix[1, 1] * curvature_1
                + coupling_matrix[1, 2] * curvature_2
            )
            N_2 = (
                membrane_matrix[2, 0] * membrane_strain_0
                + membrane_matrix[2, 1] * membrane_strain_1
                + membrane_matrix[2, 2] * membrane_strain_2
                + coupling_matrix[2, 0] * curvature_0
                + coupling_matrix[2, 1] * curvature_1
                + coupling_matrix[2, 2] * curvature_2
            )
            M_0 = (
                coupling_matrix[0, 0] * membrane_strain_0
                + coupling_matrix[1, 0] * membrane_strain_1
                + coupling_matrix[2, 0] * membrane_strain_2
                + bending_matrix[0, 0] * curvature_0
                + bending_matrix[0, 1] * curvature_1
                + bending_matrix[0, 2] * curvature_2
            )
            M_1 = (
                coupling_matrix[0, 1] * membrane_strain_0
                + coupling_matrix[1, 1] * membrane_strain_1
                + coupling_matrix[2, 1] * membrane_strain_2
                + bending_matrix[1, 0] * curvature_0
                + bending_matrix[1, 1] * curvature_1
                + bending_matrix[1, 2] * curvature_2
            )
            M_2 = (
                coupling_matrix[0, 2] * membrane_strain_0
                + coupling_matrix[1, 2] * membrane_strain_1
                + coupling_matrix[2, 2] * membrane_strain_2
                + bending_matrix[2, 0] * curvature_0
                + bending_matrix[2, 1] * curvature_1
                + bending_matrix[2, 2] * curvature_2
            )
            N_0 += initial_membrane_resultants[element_index, gp_index, 0]
            N_1 += initial_membrane_resultants[element_index, gp_index, 1]
            N_2 += initial_membrane_resultants[element_index, gp_index, 2]
            M_0 += initial_bending_resultants[element_index, gp_index, 0]
            M_1 += initial_bending_resultants[element_index, gp_index, 1]
            M_2 += initial_bending_resultants[element_index, gp_index, 2]

            for dof_index in range(n_dof):
                B_eff[0, dof_index] = B_m[0, dof_index] + theta_0 * Gw[0, dof_index]
                B_eff[1, dof_index] = B_m[1, dof_index] + theta_1 * Gw[1, dof_index]
                B_eff[2, dof_index] = (
                    B_m[2, dof_index]
                    + theta_0 * Gw[1, dof_index]
                    + theta_1 * Gw[0, dof_index]
                )

                force_values[force_pos[dof_index]] += (
                    B_eff[0, dof_index] * N_0
                    + B_eff[1, dof_index] * N_1
                    + B_eff[2, dof_index] * N_2
                    + B_b[0, dof_index] * M_0
                    + B_b[1, dof_index] * M_1
                    + B_b[2, dof_index] * M_2
                    + B_d[0, dof_index] * drilling_stiffness * drilling_rotation
                ) * detw

            if not tangent:
                continue

            for row in range(3):
                for dof_index in range(n_dof):
                    membrane_value = 0.0
                    bending_value = 0.0
                    for constitutive_index in range(3):
                        membrane_value += (
                            membrane_matrix[row, constitutive_index]
                            * B_eff[constitutive_index, dof_index]
                            + coupling_matrix[row, constitutive_index]
                            * B_b[constitutive_index, dof_index]
                        )
                        bending_value += (
                            coupling_matrix[constitutive_index, row]
                            * B_eff[constitutive_index, dof_index]
                            + bending_matrix[row, constitutive_index]
                            * B_b[constitutive_index, dof_index]
                        )
                    membrane_times_B[row, dof_index] = membrane_value
                    bending_times_B[row, dof_index] = bending_value

            for local_row in range(n_dof):
                gw0_row = Gw[0, local_row]
                gw1_row = Gw[1, local_row]
                bd_row = B_d[0, local_row]
                for local_col in range(n_dof):
                    material_value = 0.0
                    bending_value = 0.0
                    for constitutive_index in range(3):
                        material_value += (
                            B_eff[constitutive_index, local_row]
                            * membrane_times_B[constitutive_index, local_col]
                        )
                        bending_value += (
                            B_b[constitutive_index, local_row]
                            * bending_times_B[constitutive_index, local_col]
                        )
                    geometric_value = (
                        gw0_row * (N_0 * Gw[0, local_col] + N_2 * Gw[1, local_col])
                        + gw1_row * (N_2 * Gw[0, local_col] + N_1 * Gw[1, local_col])
                    )
                    drilling_value = bd_row * drilling_stiffness * B_d[0, local_col]
                    tangent_values[
                        tangent_pos[local_row * n_dof + local_col]
                    ] += (
                        material_value
                        + bending_value
                        + geometric_value
                        + drilling_value
                    ) * detw

        for shear_index in range(n_shear):
            B_s = B_s_batch[element_index, shear_index]
            detw_shear = detw_shear_batch[element_index, shear_index]
            gamma_0 = 0.0
            gamma_1 = 0.0
            for dof_index in range(n_dof):
                gamma_0 += B_s[0, dof_index] * u_local[dof_index]
                gamma_1 += B_s[1, dof_index] * u_local[dof_index]

            shear_resultant_0 = (
                shear_matrix[0, 0] * gamma_0 + shear_matrix[0, 1] * gamma_1
            )
            shear_resultant_1 = (
                shear_matrix[1, 0] * gamma_0 + shear_matrix[1, 1] * gamma_1
            )
            for dof_index in range(n_dof):
                force_values[force_pos[dof_index]] += (
                    B_s[0, dof_index] * shear_resultant_0
                    + B_s[1, dof_index] * shear_resultant_1
                ) * detw_shear

            if tangent:
                for local_row in range(n_dof):
                    for local_col in range(n_dof):
                        shear_value = (
                            B_s[0, local_row]
                            * (
                                shear_matrix[0, 0] * B_s[0, local_col]
                                + shear_matrix[0, 1] * B_s[1, local_col]
                            )
                            + B_s[1, local_row]
                            * (
                                shear_matrix[1, 0] * B_s[0, local_col]
                                + shear_matrix[1, 1] * B_s[1, local_col]
                            )
                        )
                        tangent_values[
                            tangent_pos[local_row * n_dof + local_col]
                        ] += shear_value * detw_shear

        # Transform force blocks from local to global in place.  T0 maps global
        # displacements to local displacements, hence forces use T0.T.
        for block_start in range(0, n_dof, 3):
            local_0 = force_values[force_pos[block_start]]
            local_1 = force_values[force_pos[block_start + 1]]
            local_2 = force_values[force_pos[block_start + 2]]
            force_values[force_pos[block_start]] = (
                T0[block_start, block_start] * local_0
                + T0[block_start + 1, block_start] * local_1
                + T0[block_start + 2, block_start] * local_2
            )
            force_values[force_pos[block_start + 1]] = (
                T0[block_start, block_start + 1] * local_0
                + T0[block_start + 1, block_start + 1] * local_1
                + T0[block_start + 2, block_start + 1] * local_2
            )
            force_values[force_pos[block_start + 2]] = (
                T0[block_start, block_start + 2] * local_0
                + T0[block_start + 1, block_start + 2] * local_1
                + T0[block_start + 2, block_start + 2] * local_2
            )

        if not tangent:
            continue

        # T0 is block diagonal with 3x3 translation/rotation blocks.  Each
        # K_global block depends only on the corresponding K_local block, so it
        # can be transformed and overwritten safely without a second matrix.
        local_block = np.empty((3, 3), dtype=np.float64)
        intermediate = np.empty((3, 3), dtype=np.float64)
        global_block = np.empty((3, 3), dtype=np.float64)
        for row_block in range(0, n_dof, 3):
            for col_block in range(0, n_dof, 3):
                for row in range(3):
                    for col in range(3):
                        local_block[row, col] = tangent_values[
                            tangent_pos[
                                (row_block + row) * n_dof + col_block + col
                            ]
                        ]

                for row in range(3):
                    for col in range(3):
                        value = 0.0
                        for inner in range(3):
                            value += (
                                local_block[row, inner]
                                * T0[col_block + inner, col_block + col]
                            )
                        intermediate[row, col] = value

                for row in range(3):
                    for col in range(3):
                        value = 0.0
                        for inner in range(3):
                            value += (
                                T0[row_block + inner, row_block + row]
                                * intermediate[inner, col]
                            )
                        global_block[row, col] = value

                for row in range(3):
                    for col in range(3):
                        tangent_values[
                            tangent_pos[
                                (row_block + row) * n_dof + col_block + col
                            ]
                        ] = global_block[row, col]


_ORIGINAL_BATCH_BUILD = _performance._ShellBatchPlan.build.__func__
_ORIGINAL_PLAN_ASSEMBLE = _performance.NonlinearAssemblyPlan.assemble
_ORIGINAL_PLAN_DIAGNOSTICS = _performance.NonlinearAssemblyPlan.diagnostics
_ORIGINAL_STATE_FINALIZER = None
_INSTALLED = False


def _finalize_batch_b_element_states(
    model,
    displacements: np.ndarray,
    element_states: Dict[int, Any],
    num_layers: int,
    *,
    kinematics: str = "von_karman",
) -> Dict[int, Any]:
    """Recover elastic shell layer strains once, after the nonlinear solve.

    Batch B intentionally omits displacement-derived layer strains from every
    Newton iteration because elastic materials have no constitutive history to
    commit.  Public result states still need those strains for stress recovery
    and strain summaries, so reconstruct them in vectorized batch form at the
    final displacement.
    """

    states = element_states
    if _ORIGINAL_STATE_FINALIZER is not None:
        states = _ORIGINAL_STATE_FINALIZER(
            model,
            displacements,
            states,
            num_layers,
            kinematics=kinematics,
        )
    if str(kinematics) != "von_karman" or not states:
        return states

    from .nonlinear_performance_bootstrap import get_nonlinear_assembly_plan

    plan = get_nonlinear_assembly_plan(model, int(num_layers))
    elastic_batches = tuple(
        batch
        for batch in plan.shell_batches
        if getattr(batch, "_batch_b_elastic", False)
        and not getattr(batch, "_batch_b_generalized", False)
    )
    if not elastic_batches:
        return states

    displacement_array = np.asarray(displacements, dtype=float).reshape(-1)
    finalized = dict(states)
    for batch in elastic_batches:
        element_displacements = displacement_array[batch.dof_mappings]
        local_displacements = np.einsum(
            "eij,ej->ei",
            batch.T0,
            element_displacements,
            optimize=True,
        )
        rotations = np.einsum(
            "egij,ej->egi",
            batch.Gw,
            local_displacements,
            optimize=True,
        )
        membrane_strain = np.einsum(
            "egij,ej->egi",
            batch.B_m,
            local_displacements,
            optimize=True,
        )
        membrane_strain[..., 0] += 0.5 * rotations[..., 0] ** 2
        membrane_strain[..., 1] += 0.5 * rotations[..., 1] ** 2
        membrane_strain[..., 2] += rotations[..., 0] * rotations[..., 1]
        curvature = np.einsum(
            "egij,ej->egi",
            batch.B_b,
            local_displacements,
            optimize=True,
        )
        layer_positions, _layer_weights = lobatto_layers(
            int(num_layers),
            float(batch.thickness),
        )
        layer_strain = (
            membrane_strain[:, :, None, :]
            + layer_positions[None, None, :, None] * curvature[:, :, None, :]
        ).reshape(len(batch.element_ids), batch.n_gp * int(num_layers), 3)

        for index, element_id in enumerate(batch.element_ids):
            element_key = int(element_id)
            existing = finalized.get(element_key)
            if isinstance(existing, Mapping) and any(
                key in existing
                for key in (
                    "initial_membrane_stress",
                    "initial_bending_stress",
                    "initial_membrane_prestrain",
                    "initial_curvature_prestrain",
                )
            ):
                # Newton iterations use integrated initial resultants in the
                # compiled kernel. Reconstruct the complete layer/provenance
                # state once at the converged displacement for public recovery.
                element = batch.elements[index]
                material = model.get_material(element.material_name)
                require_legacy_direct_nonlinear_element(
                    element,
                    context="Batch B finalized initial-field recovery",
                )
                _force, _tangent, recovered_state = element.compute_nonlinear_response(
                    model.mesh,
                    material,
                    displacement_array[batch.dof_mappings[index]],
                    existing,
                    int(num_layers),
                    False,
                )
                if recovered_state is not None:
                    finalized[element_key] = recovered_state
                continue
            if isinstance(existing, Mapping):
                state = dict(existing)
            else:
                state = dict(batch.elastic_states[index])
            state["layer_strain"] = layer_strain[index]
            # Any carried elastic stress array describes an earlier
            # displacement. Downstream recovery reconstructs it from the
            # current strain and elastic constitutive matrix.
            state.pop("layer_stress", None)
            finalized[element_key] = state
    return finalized


def _batch_b_shell_build(
    cls,
    model,
    key,
    items,
    num_layers,
):
    batch = _ORIGINAL_BATCH_BUILD(cls, model, key, items, num_layers)
    if batch.has_plasticity or (key and key[0] == "isotropic_q4_assumed_strain_plastic"):
        batch._batch_b_elastic = False
        return batch

    element_count = len(batch.elements)
    batch._batch_b_elastic = True
    batch._batch_b_membrane_matrix = np.empty(
        (element_count, 3, 3), dtype=float
    )
    batch._batch_b_coupling_matrix = np.zeros(
        (element_count, 3, 3), dtype=float
    )
    batch._batch_b_bending_matrix = np.empty(
        (element_count, 3, 3), dtype=float
    )
    batch._batch_b_shear_matrix = np.empty(
        (element_count, 2, 2), dtype=float
    )
    batch._batch_b_drilling_stiffness = np.empty(element_count, dtype=float)
    generalized = bool(batch.elements[0].shell_section is not None)
    batch._batch_b_generalized = generalized
    isotropic = bool(
        not generalized
        and getattr(batch.material, "elastic_symmetry", "isotropic")
        == "isotropic"
    )
    if isotropic:
        # The group key already guarantees one material, thickness and
        # drilling stabilization value.  Isotropic constitutive matrices are
        # frame-invariant, so compute them once and broadcast into the
        # per-element arrays required by the generalized kernel.  Keeping the
        # old one-computation setup path avoids making mature isotropic plan
        # construction scale with the element count.
        elastic_matrix, shear_matrix, _strain, _stress = (
            _shell_material_matrices(batch.material, 0.0)
        )
        thickness = float(batch.thickness)
        batch._batch_b_membrane_matrix[:] = thickness * elastic_matrix
        batch._batch_b_bending_matrix[:] = (
            thickness**3 / 12.0 * elastic_matrix
        )
        batch._batch_b_shear_matrix[:] = (
            (5.0 / 6.0) * thickness * shear_matrix
        )
        drilling_modulus = 1.0 / float(
            _elastic_compliance(batch.material)[5, 5]
        )
        batch._batch_b_drilling_stiffness.fill(
            drilling_modulus
            * thickness
            * float(batch.drilling_stabilization)
        )
    else:
        for index, element in enumerate(batch.elements):
            cache = element._nonlinear_geometry(model.mesh)
            frame = cache["R0"]
            material = model.get_material(element.material_name)
            if element.shell_section is not None:
                section = element._generalized_section_in_frame(frame)
                assert section is not None
                batch._batch_b_membrane_matrix[index] = section.A
                batch._batch_b_coupling_matrix[index] = section.B
                batch._batch_b_bending_matrix[index] = section.D
                batch._batch_b_shear_matrix[index] = section.As
                batch._batch_b_drilling_stiffness[index] = (
                    float(section.A[2, 2])
                    * float(element.drilling_stabilization)
                )
                continue

            elastic_matrix, shear_matrix, _strain, _stress = (
                _shell_material_matrices(
                    material,
                    element._material_angle(frame),
                )
            )
            thickness = float(element.thickness)
            batch._batch_b_membrane_matrix[index] = thickness * elastic_matrix
            batch._batch_b_bending_matrix[index] = (
                thickness**3 / 12.0 * elastic_matrix
            )
            batch._batch_b_shear_matrix[index] = (
                (5.0 / 6.0) * thickness * shear_matrix
            )
            drilling_modulus = 1.0 / float(
                _elastic_compliance(material)[5, 5]
            )
            batch._batch_b_drilling_stiffness[index] = (
                drilling_modulus
                * thickness
                * float(element.drilling_stabilization)
            )
    batch._batch_b_initial_membrane_resultants = np.zeros(
        (len(batch.elements), batch.n_gp, 3), dtype=float
    )
    batch._batch_b_initial_bending_resultants = np.zeros_like(
        batch._batch_b_initial_membrane_resultants
    )
    batch._batch_b_initial_fields_supported = isotropic

    # Elastic groups do not require mutable constitutive history.  Release the
    # per-element plastic work arrays allocated by the compatibility builder.
    batch.plastic_work = np.empty((0, 0, 3), dtype=float)
    batch.alpha_work = np.empty((0, 0), dtype=float)

    points_per_element = int(batch.n_gp * batch.num_layers)
    shared_plastic = np.zeros((points_per_element, 3), dtype=float)
    shared_alpha = np.zeros(points_per_element, dtype=float)
    shared_plastic.setflags(write=False)
    shared_alpha.setflags(write=False)
    batch.elastic_states = tuple(
        (
            {"generalized_section": True}
            if generalized
            else {
                "plastic_strain": shared_plastic,
                "alpha": shared_alpha,
            }
        )
        for _ in batch.elements
    )
    batch._batch_b_elastic_state_mapping = {
        int(element_id): state
        for element_id, state in zip(batch.element_ids, batch.elastic_states)
    }
    return batch


def _populate_initial_field_resultants(batch, committed_states: Mapping[int, Any]) -> int:
    """Integrate elastic initial fields into membrane/bending resultants."""

    if not getattr(batch, "_batch_b_initial_fields_supported", False):
        batch._batch_b_initial_membrane_resultants.fill(0.0)
        batch._batch_b_initial_bending_resultants.fill(0.0)
        return 0
    membrane_resultants = batch._batch_b_initial_membrane_resultants
    bending_resultants = batch._batch_b_initial_bending_resultants
    membrane_resultants.fill(0.0)
    bending_resultants.fill(0.0)
    layer_positions, layer_weights = lobatto_layers(
        int(batch.num_layers), float(batch.thickness)
    )
    membrane_weight = float(np.sum(layer_weights))
    second_weight = float(np.sum(layer_weights * layer_positions**2))
    bending_stress_weight = 2.0 * second_weight / float(batch.thickness)
    count = 0
    zero = np.zeros((batch.n_gp, 3), dtype=float)
    for index, element_id in enumerate(batch.element_ids):
        state = committed_states.get(int(element_id))
        if not _performance._state_has_initial_fields(state):
            continue
        membrane_stress = _shell_field_rows(
            state.get("initial_membrane_stress", zero),
            batch.n_gp,
            "initial_membrane_stress",
        )
        bending_stress = _shell_field_rows(
            state.get("initial_bending_stress", zero),
            batch.n_gp,
            "initial_bending_stress",
        )
        membrane_prestrain = _shell_field_rows(
            state.get("initial_membrane_prestrain", zero),
            batch.n_gp,
            "initial_membrane_prestrain",
        )
        curvature_prestrain = _shell_field_rows(
            state.get("initial_curvature_prestrain", zero),
            batch.n_gp,
            "initial_curvature_prestrain",
        )
        elastic_matrix = (
            batch._batch_b_membrane_matrix[index]
            / float(batch.elements[index].thickness)
        )
        membrane_resultants[index] = (
            membrane_weight * membrane_stress
            - membrane_weight * (membrane_prestrain @ elastic_matrix.T)
        )
        bending_resultants[index] = (
            bending_stress_weight * bending_stress
            - second_weight * (curvature_prestrain @ elastic_matrix.T)
        )
        count += 1
    return count


def _generalized_trial_states(batch) -> Dict[int, Dict[str, Any]]:
    """Materialize the scalar generalized-resultant state from batch arrays."""

    local_displacements = batch.u_work
    rotations = np.einsum(
        "egij,ej->egi", batch.Gw, local_displacements, optimize=True
    )
    membrane_strain = np.einsum(
        "egij,ej->egi", batch.B_m, local_displacements, optimize=True
    )
    membrane_strain[..., 0] += 0.5 * rotations[..., 0] ** 2
    membrane_strain[..., 1] += 0.5 * rotations[..., 1] ** 2
    membrane_strain[..., 2] += rotations[..., 0] * rotations[..., 1]
    curvature = np.einsum(
        "egij,ej->egi", batch.B_b, local_displacements, optimize=True
    )
    shear_strain = np.einsum(
        "esij,ej->esi", batch.B_s, local_displacements, optimize=True
    )
    membrane_resultants = (
        np.einsum(
            "eij,egj->egi",
            batch._batch_b_membrane_matrix,
            membrane_strain,
            optimize=True,
        )
        + np.einsum(
            "eij,egj->egi",
            batch._batch_b_coupling_matrix,
            curvature,
            optimize=True,
        )
    )
    bending_resultants = (
        np.einsum(
            "eji,egj->egi",
            batch._batch_b_coupling_matrix,
            membrane_strain,
            optimize=True,
        )
        + np.einsum(
            "eij,egj->egi",
            batch._batch_b_bending_matrix,
            curvature,
            optimize=True,
        )
    )
    shear_resultants = np.einsum(
        "eij,esj->esi",
        batch._batch_b_shear_matrix,
        shear_strain,
        optimize=True,
    )
    states: Dict[int, Dict[str, Any]] = {}
    for index, element_id in enumerate(batch.element_ids):
        states[int(element_id)] = {
            "generalized_section": True,
            "geometric_nonlinearity": "von_karman",
            "membrane_strain": membrane_strain[index].copy(),
            "curvature": curvature[index].copy(),
            "transverse_shear_strain": shear_strain[index].copy(),
            "membrane_resultants": membrane_resultants[index].copy(),
            "bending_resultants": bending_resultants[index].copy(),
            "transverse_shear_resultants": shear_resultants[index].copy(),
            "membrane_resultant_order": ("11", "22", "12"),
            "transverse_shear_resultant_order": ("13", "23"),
            "recovery_scope": "section_resultants_only",
        }
    return states


def _recover_initial_field_states(
    plan,
    batch,
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    trial_states: Dict[int, Any],
) -> None:
    """Populate complete result state for residual-only terminal evaluations."""

    displacement_array = np.asarray(displacements, dtype=float)
    model = plan.model
    for index, (element_id, element) in enumerate(zip(batch.element_ids, batch.elements)):
        element_key = int(element_id)
        existing = committed_states.get(element_key)
        if not _performance._state_has_initial_fields(existing):
            continue
        require_legacy_direct_nonlinear_element(
            element,
            context="Batch B residual-only initial-field recovery",
        )
        _force, _tangent, recovered = element.compute_nonlinear_response(
            model.mesh,
            model.get_material(element.material_name),
            displacement_array[batch.dof_mappings[index]],
            existing,
            plan.num_layers,
            False,
        )
        if recovered is not None:
            trial_states[element_key] = recovered


def _batch_b_plan_assemble(
    self,
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    tangent: bool = True,
    deleted_element_ids: Sequence[int] = (),
    residual_stiffness_fraction: float = 1.0,
) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[int, Any]]:
    """Assemble with direct-buffer elastic batches and legacy plastic batches."""
    if tuple(deleted_element_ids or ()):
        return _ORIGINAL_PLAN_ASSEMBLE(
            self,
            displacements,
            committed_states,
            tangent=tangent,
            deleted_element_ids=tuple(deleted_element_ids),
            residual_stiffness_fraction=float(residual_stiffness_fraction),
        )
    with self._lock:
        state_token = begin_state_evaluation(
            committed_states,
            model=self.model,
            displacements=displacements,
        )
        start_total = time.perf_counter()
        self.timings.calls += 1
        if tangent:
            self.timings.tangent_calls += 1
        else:
            self.timings.residual_only_calls += 1

        self.force_values.fill(0.0)
        if tangent:
            self.tangent_values.fill(0.0)
        trial_states: Dict[int, Any] = {}

        for batch in self.shell_batches:
            used_persistent_state = False
            if getattr(batch, "_batch_b_elastic", False):
                initialized_count = _populate_initial_field_resultants(
                    batch, committed_states
                )
                kernel_start = time.perf_counter()
                _elastic_shell_batch_into_buffers(
                    np.asarray(displacements, dtype=float),
                    batch.dof_mappings,
                    batch.T0,
                    batch.B_m,
                    batch.B_b,
                    batch.B_d,
                    batch.Gw,
                    batch.detw,
                    batch.B_s,
                    batch.detw_shear,
                    batch._batch_b_membrane_matrix,
                    batch._batch_b_coupling_matrix,
                    batch._batch_b_bending_matrix,
                    batch._batch_b_initial_membrane_resultants,
                    batch._batch_b_initial_bending_resultants,
                    batch._batch_b_shear_matrix,
                    batch._batch_b_drilling_stiffness,
                    batch.force_positions,
                    batch.tangent_positions,
                    self.force_values,
                    self.tangent_values,
                    batch.u_work,
                    bool(tangent),
                )
                if np.any(batch.linear_corrections):
                    element_displacements = np.asarray(
                        displacements, dtype=float
                    )[batch.dof_mappings]
                    correction_forces = np.einsum(
                        "eij,ej->ei",
                        batch.linear_corrections,
                        element_displacements,
                        optimize=True,
                    )
                    self.force_values[batch.force_positions] += correction_forces
                    if tangent:
                        self.tangent_values[batch.tangent_positions] += (
                            batch.linear_corrections.reshape(
                                len(batch.elements), -1
                            )
                        )
                self.timings.shell_kernel_seconds += time.perf_counter() - kernel_start
                self.timings.initial_field_accelerated_elements += initialized_count
                if getattr(batch, "_batch_b_generalized", False):
                    trial_states.update(_generalized_trial_states(batch))
                else:
                    elastic_states = batch._batch_b_elastic_state_mapping
                    # Preserve explicitly supplied/previous elastic states
                    # without allocating a new mapping in the normal case.
                    use_cached_mapping = True
                    for element_id in batch.element_ids:
                        existing = committed_states.get(int(element_id))
                        if (
                            isinstance(existing, dict)
                            and existing is not elastic_states[int(element_id)]
                        ):
                            use_cached_mapping = False
                            break
                    if use_cached_mapping:
                        trial_states.update(elastic_states)
                    else:
                        for element_id in batch.element_ids:
                            element_key = int(element_id)
                            existing = committed_states.get(element_key)
                            trial_states[element_key] = (
                                existing
                                if isinstance(existing, dict)
                                else elastic_states[element_key]
                            )
                if initialized_count and not tangent:
                    _recover_initial_field_states(
                        self,
                        batch,
                        displacements,
                        committed_states,
                        trial_states,
                    )
            else:
                persistent_trial = (
                    committed_states.shell_trial_for_layout(
                        state_token,
                        batch.state_layout,
                    )
                    if isinstance(committed_states, NonlinearStateStore)
                    and state_token is not None
                    else None
                )
                if persistent_trial is not None and batch.persistent_state_eligibility(
                    persistent_trial[0]
                )[0]:
                    used_persistent_state = True
                    F_batch, K_batch, kernel_seconds = batch.evaluate_persistent(
                        displacements,
                        persistent_trial[0],
                        persistent_trial[1],
                        tangent,
                    )
                    batch_states = {}
                else:
                    F_batch, K_batch, batch_states, kernel_seconds = batch.evaluate(
                        displacements,
                        committed_states,
                        tangent,
                    )
                self.timings.shell_kernel_seconds += kernel_seconds
                self.force_values[batch.force_positions.reshape(-1)] = np.asarray(
                    F_batch, dtype=float
                ).reshape(-1)
                if tangent and K_batch is not None:
                    self.tangent_values[batch.tangent_positions.reshape(-1)] = np.asarray(
                        K_batch, dtype=float
                    ).reshape(-1)
                trial_states.update(batch_states)

            if not used_persistent_state:
                override_start = time.perf_counter()
                override_count = _performance._apply_initial_field_shell_overrides(
                    self,
                    batch,
                    displacements,
                    committed_states,
                    tangent,
                    trial_states,
                )
                self.timings.initial_field_override_seconds += (
                    time.perf_counter() - override_start
                )
                self.timings.initial_field_override_elements += override_count

        for batch in self.quadratic_beam_batches:
            self.timings.non_shell_seconds += batch.evaluate_into_buffers(
                self,
                displacements,
                committed_states,
                tangent,
                trial_states,
            )

        non_shell_start = time.perf_counter()
        model = self.model
        mesh = model.mesh
        for record in self.non_shell_elements:
            material = model.get_material(record.element.material_name)
            u_element = np.asarray(displacements, dtype=float)[record.dof_mapping]
            f_element, k_element, trial_state = evaluate_nonlinear_element(
                record.element,
                mesh,
                material,
                u_element,
                committed_states.get(record.element_id),
                self.num_layers,
                tangent,
                committed_states=committed_states,
                state_token=state_token,
                element_id=record.element_id,
                kinematics="von_karman",
            )
            self.force_values[record.force_positions] = np.asarray(
                f_element, dtype=float
            ).reshape(-1)
            if tangent and k_element is not None:
                self.tangent_values[record.tangent_positions] = np.asarray(
                    k_element, dtype=float
                ).reshape(-1)
            if trial_state is not None:
                trial_states[record.element_id] = trial_state
        self.timings.non_shell_seconds += time.perf_counter() - non_shell_start

        scatter_start = time.perf_counter()
        scatter = _performance._scatter_sum
        if not self.shell_batches and not self.quadratic_beam_batches:
            scatter = _performance._generic_beam_scatter_kernel(self)
        force = scatter(
            self.force_values,
            self.force_dofs_flat,
            self.total_dofs,
        )
        self.timings.force_scatter_seconds += time.perf_counter() - scatter_start

        tangent_matrix: Optional[sparse.csr_matrix]
        if tangent:
            scatter_start = time.perf_counter()
            csr_data = scatter(
                self.tangent_values,
                self.tangent_scatter,
                self.nnz,
            )
            tangent_matrix = sparse.csr_matrix(
                (csr_data, self.csr_indices, self.csr_indptr),
                shape=(self.total_dofs, self.total_dofs),
                copy=False,
            )
            self.timings.tangent_scatter_seconds += time.perf_counter() - scatter_start
        else:
            tangent_matrix = None

        self.timings.total_seconds += time.perf_counter() - start_total
        state_payload = finish_state_evaluation(
            committed_states,
            state_token,
            trial_states,
        )
        return force, tangent_matrix, state_payload


def _batch_b_plan_diagnostics(self) -> Dict[str, Any]:
    diagnostics = _ORIGINAL_PLAN_DIAGNOSTICS(self)
    elastic_batches = [
        batch
        for batch in self.shell_batches
        if getattr(batch, "_batch_b_elastic", False)
    ]
    generalized_batches = [
        batch
        for batch in elastic_batches
        if getattr(batch, "_batch_b_generalized", False)
    ]
    orthotropic_batches = [
        batch
        for batch in elastic_batches
        if not getattr(batch, "_batch_b_generalized", False)
        and any(
            getattr(element, "shell_section", None) is None
            and getattr(
                self.model.get_material(element.material_name),
                "elastic_symmetry",
                "isotropic",
            )
            == "orthotropic"
            for element in batch.elements
        )
    ]
    diagnostics.update(
        {
            "batch_b_installed": True,
            "elastic_fast_path_batch_count": len(elastic_batches),
            "elastic_fast_path_element_count": int(
                sum(batch.element_ids.size for batch in elastic_batches)
            ),
            "plastic_batch_count": int(len(self.shell_batches) - len(elastic_batches)),
            "orthotropic_elastic_fast_path_batch_count": len(
                orthotropic_batches
            ),
            "orthotropic_elastic_fast_path_element_count": int(
                sum(batch.element_ids.size for batch in orthotropic_batches)
            ),
            "generalized_elastic_fast_path_batch_count": len(
                generalized_batches
            ),
            "generalized_elastic_fast_path_element_count": int(
                sum(batch.element_ids.size for batch in generalized_batches)
            ),
            "elastic_constitutive_state_bytes": int(
                sum(
                    batch.elastic_states[0]["plastic_strain"].nbytes
                    + batch.elastic_states[0]["alpha"].nbytes
                    for batch in elastic_batches
                    if batch.elastic_states
                    and "plastic_strain" in batch.elastic_states[0]
                )
            ),
        }
    )
    return diagnostics


def install_batch_b_optimizations() -> bool:
    global _INSTALLED, _ORIGINAL_STATE_FINALIZER
    if _INSTALLED:
        return True
    from . import nonlinear_static as _nonlinear_static

    _ORIGINAL_STATE_FINALIZER = (
        _nonlinear_static._finalize_nonlinear_element_states
    )
    _nonlinear_static._finalize_nonlinear_element_states = (
        _finalize_batch_b_element_states
    )
    _performance._EXTENDED_ELASTIC_S4_BATCHES = True
    _performance._ShellBatchPlan.build = classmethod(_batch_b_shell_build)
    _performance.NonlinearAssemblyPlan.assemble = _batch_b_plan_assemble
    _performance.NonlinearAssemblyPlan.diagnostics = _batch_b_plan_diagnostics
    _performance.clear_nonlinear_assembly_cache()
    _INSTALLED = True
    return True


def uninstall_batch_b_optimizations() -> None:
    global _INSTALLED, _ORIGINAL_STATE_FINALIZER
    if not _INSTALLED:
        return
    from . import nonlinear_static as _nonlinear_static

    if (
        _nonlinear_static._finalize_nonlinear_element_states
        is _finalize_batch_b_element_states
        and _ORIGINAL_STATE_FINALIZER is not None
    ):
        _nonlinear_static._finalize_nonlinear_element_states = (
            _ORIGINAL_STATE_FINALIZER
        )
    _ORIGINAL_STATE_FINALIZER = None
    _performance._EXTENDED_ELASTIC_S4_BATCHES = False
    _performance._ShellBatchPlan.build = classmethod(_ORIGINAL_BATCH_BUILD)
    _performance.NonlinearAssemblyPlan.assemble = _ORIGINAL_PLAN_ASSEMBLE
    _performance.NonlinearAssemblyPlan.diagnostics = _ORIGINAL_PLAN_DIAGNOSTICS
    _performance.clear_nonlinear_assembly_cache()
    _INSTALLED = False


def batch_b_status() -> Dict[str, Any]:
    return {
        "installed": bool(_INSTALLED),
        "description": (
            "in-place isotropic/orthotropic/generalized elastic shell "
            "assembly into persistent buffers"
        ),
        "parallel_kernel": True,
    }
