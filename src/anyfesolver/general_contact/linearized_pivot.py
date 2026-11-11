"""Disposable constrained gap responses for augmented shell pressure proposals.

These local linearizations propose algorithmic iterates. The coupled path must
re-equilibrate each proposal and apply its independent contact acceptance gates.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse
from scipy.linalg import null_space

from anyfesolver.linalg import MatrixClass, factorize

from .augmentation import ActiveSetPivot, propose_active_set_pivot
from .overlap import PlanarFace
from .q4_source_overlap import projected_q4_source_face
from .q4_s3_tiny_wedge import verify_tiny_wedge_derivatives
from .q4_s3_overlap_derivatives import q4_s3_overlap_derivatives
from .s3_overlap_derivatives import s3_overlap_derivatives
from .s3_patch_augmented import (
    _q4_s3_gap_operand_roundoff, _q4_s3_pair_value_rtol,
    _q4_s3_pair_values_agree,
)
from .supported_schur import SupportedSchurProposal, propose_supported_schur_step


def _projected_clip_signature(geometry, source, target) -> tuple[int, ...] | None:
    """Reject clipping edge changes hidden by equal polygon vertex counts."""
    if not hasattr(geometry, "source_face_indices"):
        return ()  # Analytical response controls have no face topology.
    signatures: list[int] = []
    for source_nodes in geometry.source_face_indices:
        source_positions = source[source_nodes]
        if len(source_positions) == 4:
            first, normal = projected_q4_source_face(source_positions)
        else:
            first = PlanarFace((1, 1), source_positions)
            normal = np.cross(first.corners[1]-first.corners[0],
                              first.corners[-1]-first.corners[0])
            normal /= np.linalg.norm(normal)
        tangent = first.corners[1]-first.corners[0]
        tangent /= np.linalg.norm(tangent)
        binormal = np.cross(normal, tangent)
        origin = first.corners[0]
        for target_nodes in geometry.target_face_indices:
            target_positions = target[target_nodes]
            projected = target_positions-((target_positions-origin) @ normal)[:, None]*normal
            second = PlanarFace((2, 1), projected)
            polygons = tuple(np.column_stack(((face.corners-origin) @ tangent,
                                              (face.corners-origin) @ binormal))
                             for face in (first, second))
            scale = max(float(np.max(np.linalg.norm(p-p[0], axis=1)))
                        for p in polygons)
            threshold = 1.0e-10*scale
            for vertices, edges in ((polygons[0], polygons[1]),
                                    (polygons[1], polygons[0])):
                for point in vertices:
                    for start, end in zip(edges, np.roll(edges, -1, axis=0)):
                        edge = end-start
                        along = float(np.dot(point-start, edge)/np.dot(edge, edge))
                        if along < -1.0e-10 or along > 1.0+1.0e-10:
                            signatures.append(0)  # Distant supporting-line extension.
                            continue
                        distance = (edge[0]*(point[1]-start[1])
                                    - edge[1]*(point[0]-start[0]))/np.linalg.norm(edge)
                        if abs(distance) <= threshold:
                            return None
                        signatures.append(1 if distance > 0.0 else -1)
    return tuple(signatures)


def _jet2_integral_gradients(geometry, source, target,
                             selected: np.ndarray, count: int,
                             maximum_evaluations: int,
                             check_cancel: Callable[[], None] | None = None,
                             *, manager=None, base=None,
                             diagnostics: dict[str, object] | None = None):
    """Differentiate W and S at one smooth pose in authoritative xyz order."""
    pair_count = len(geometry.pair_laws)
    if 3*pair_count > maximum_evaluations:
        raise ValueError("pressure response geometry evaluation budget exceeded")
    weighted = np.zeros((len(selected), count))
    support = np.zeros_like(weighted)
    selected_row = {int(node): row for row, node in enumerate(selected)}
    source_count = len(source)
    tiny_wedge_used = False
    for pair_index, law in enumerate(geometry.pair_laws):
        if check_cancel is not None:
            check_cancel()
        source_face = pair_index // len(geometry.target_faces)
        target_face = pair_index % len(geometry.target_faces)
        source_nodes = geometry.source_face_indices[source_face]
        target_nodes = geometry.target_face_indices[target_face]
        local_source = source[source_nodes]
        local_target = target[target_nodes]
        local = law.evaluate(local_source, local_target)
        if local.reference_overlap_area == 0.0:
            continue
        reference = geometry.source_faces[source_face].reference_positions
        if len(source_nodes) == 4:
            derivative = q4_s3_overlap_derivatives(
                reference, local_source, local_target,
                area_tolerance=geometry.q4_source_area_tolerance,
                weighted_gap_tolerance=geometry.q4_source_weighted_gap_tolerance,
                maximum_triangles=geometry.maximum_q4_source_integration_triangles,
                allow_warped_source=True,
                **dict(zip((
                    "support_gradient_tolerance",
                    "support_hessian_tolerance",
                    "weighted_gap_gradient_tolerance",
                    "weighted_gap_hessian_tolerance",
                ), geometry.q4_source_derivative_tolerances)),
            )
        else:
            derivative = s3_overlap_derivatives(
                reference, local_source, local_target,
            )
        if len(source_nodes) == 4:
            value_rtol = _q4_s3_pair_value_rtol(
                local.overlap_area, derivative.overlap_area,
                derivative.overlap_area_resolution,
            )
            roundoff_factor = (512.0*np.finfo(float).eps
                               * max(1, local.integration_triangles))
            support_atol = roundoff_factor*max(
                float(np.max(np.abs(local.reference_support))),
                float(np.max(np.abs(derivative.reference_support))),
            )
            gap_atol = roundoff_factor*max(
                float(np.max(np.abs(local.reference_weighted_gap))),
                float(np.max(np.abs(derivative.reference_weighted_gap))),
            )
        else:
            value_rtol = 1.0e-9
            support_atol = gap_atol = 0.0
        if len(source_nodes) == 4:
            support_agrees = _q4_s3_pair_values_agree(
                local.reference_support, derivative.reference_support,
                value_rtol=value_rtol, roundoff_atol=support_atol,
                local_area=local.overlap_area,
                derivative_area=derivative.overlap_area,
                area_resolution=derivative.overlap_area_resolution,
            )
            gap_agrees = _q4_s3_pair_values_agree(
                local.reference_weighted_gap,
                derivative.reference_weighted_gap,
                value_rtol=value_rtol, roundoff_atol=gap_atol,
                local_area=local.overlap_area,
                derivative_area=derivative.overlap_area,
                area_resolution=derivative.overlap_area_resolution,
                normwise=True,
            )
            if not gap_agrees:
                # Match the projected work owner's bounded cancellation
                # allowance when nearly coincident skin coordinates are
                # subtracted. Its edge-scale guard still rejects remote
                # origins and genuinely unresolved geometry.
                gap_atol = max(gap_atol, _q4_s3_gap_operand_roundoff(
                    local_source, local_target,
                    max(float(np.max(np.abs(local.reference_support))),
                        float(np.max(np.abs(derivative.reference_support)))),
                ))
                gap_agrees = _q4_s3_pair_values_agree(
                    local.reference_weighted_gap,
                    derivative.reference_weighted_gap,
                    value_rtol=value_rtol, roundoff_atol=gap_atol,
                    local_area=local.overlap_area,
                    derivative_area=derivative.overlap_area,
                    area_resolution=derivative.overlap_area_resolution,
                    normwise=True,
                )
        else:
            support_agrees = np.allclose(
                derivative.reference_support, local.reference_support,
                rtol=value_rtol, atol=support_atol,
            )
            gap_agrees = np.allclose(
                derivative.reference_weighted_gap,
                local.reference_weighted_gap,
                rtol=value_rtol, atol=gap_atol,
            )
        if not support_agrees or not gap_agrees:
            verified = False
            if (not tiny_wedge_used and manager is not None and base is not None
                    and len(source_nodes) == 4
                    and manager.law.near_d1_diagonal_tiny_wedge(
                        source, target, pair_index,
                    )):
                tie_step = manager.law.shared_edge_tie[0]
                support_nodes = base.reference_support[source_nodes]
                gap_nodes = base.normalized_gap[source_nodes]
                local_support = local.reference_support
                reference_face = geometry.source_faces[source_face].reference_positions
                length = min(np.linalg.norm(reference_face[index]
                                            - reference_face[(index+1) % 4])
                             for index in range(4))
                support_error = np.abs(
                    derivative.reference_support-local_support,
                )
                gap_error = np.abs(
                    derivative.reference_weighted_gap-local.reference_weighted_gap,
                )
                if (local.overlap_area <= 4.0*tie_step*tie_step
                        and np.all(support_nodes-local_support
                                   >= 1.0e6*local_support)
                        and np.all(support_error <= 1.0e-10*support_nodes)
                        and np.all(gap_error <= 1.0e-10*support_nodes
                                   * np.maximum(np.abs(gap_nodes),
                                                1.0e-8*length))):
                    pressure_state = manager.law.evaluate(
                        source, target, manager.normal_multipliers,
                    )
                    force_scale = float(pressure_state.pressure
                                        @ pressure_state.reference_support)
                    if force_scale > 0.0:
                        active_local = np.asarray([
                            3*node+axis for node in range(4)
                            for axis in (0, 2)
                        ])
                        used, scenarios = verify_tiny_wedge_derivatives(
                            law, local_source, local_target, derivative,
                            pressure_state.pressure[source_nodes],
                            gap_nodes, active_local,
                            difference_step_m=tie_step/20.0,
                            force_scale_n=force_scale,
                            length_scale_m=length,
                            maximum_evaluations=(maximum_evaluations
                                                 - 4*pair_count),
                        )
                        response_errors = []
                        for gradient_error, _hessian_error in scenarios:
                            delta = (gradient_error[1]
                                     - gap_nodes[:, None]*gradient_error[0])
                            response_errors.append(float(np.max(
                                np.linalg.norm(delta, axis=1)/support_nodes,
                            )))
                        if max(response_errors) <= 1.0e-7:
                            verified = True
                            tiny_wedge_used = True
                            if diagnostics is not None:
                                diagnostics["schur_tiny_wedge_gap_response"] = {
                                    "pair": pair_index,
                                    "overlap_area_m2": local.overlap_area,
                                    "maximum_gradient_error": max(response_errors),
                                    "float_evaluations": used,
                                }
            if not verified:
                raise ValueError(
                    "pressure response derivative disagrees with mortar geometry "
                    f"(pair={pair_index}, "
                    f"overlap_area_m2={local.overlap_area:.12g}, "
                    f"support_error_m2={np.max(np.abs(derivative.reference_support-local.reference_support)):.12g}, "
                    f"weighted_gap_error_m3={np.max(np.abs(derivative.reference_weighted_gap-local.reference_weighted_gap)):.12g})"
                )
        columns = np.asarray(
            [3*int(node)+axis for node in source_nodes for axis in range(3)]
            + [3*(source_count+int(node))+axis
               for node in target_nodes for axis in range(3)],
            dtype=np.intp,
        )
        for local_row, node in enumerate(source_nodes):
            row = selected_row.get(int(node))
            if row is not None:
                weighted[row, columns] += derivative.weighted_gap_gradient[local_row]
                support[row, columns] += derivative.support_gradient[local_row]
    return weighted, support


def _exact_artificial_seam_gap_gradient(
        manager, source, target, base, selected, *, check_cancel=None,
        diagnostics=None):
    """Bound a centered generalized gap derivative at the opt-in D1 diagonal.

    This is only an algorithmic pressure-response candidate. A coupled proposal
    must still be re-equilibrated under the ordinary acceptance gates.
    """
    law = manager.law
    tie = getattr(law, "shared_edge_tie", None)
    if (tie is None or getattr(law.geometry, "differentiation", None)
            != "jet2_q4_s3"
            or not law._only_artificial_shared_edge_ties(source, target)):
        return None
    step = float(tie[0])
    if not np.isfinite(step) or step <= 0.0:
        return None
    geometry = law.geometry
    coordinates = np.concatenate((source, target)).reshape(-1)
    source_count = len(source)
    count = len(coordinates)
    pair_count = len(geometry.pair_laws)
    # Include the caller's exact geometry state and this helper's active-set
    # reference state as well as both signs at two resolutions.
    evaluations = (4*count+2)*pair_count
    if evaluations > manager.settings.maximum_general_s3_energy_evaluations:
        raise ValueError("exact-seam pressure response geometry budget exceeded")
    baseline_active = law.evaluate(
        source, target, manager.normal_multipliers,
    ).pressure > 0.0
    center = base.normalized_gap[selected]
    estimates = []
    largest_one_sided = 0.0
    for h in (step, step/2.0):
        derivative = np.zeros((len(selected), count))
        for coordinate in range(count):
            if check_cancel is not None:
                check_cancel()
            sides = []
            for sign in (-1.0, 1.0):
                shifted = coordinates.copy()
                shifted[coordinate] += sign*h
                moved_source = shifted[:3*source_count].reshape(-1, 3)
                moved_target = shifted[3*source_count:].reshape(-1, 3)
                candidate = law.evaluate(
                    moved_source, moved_target, manager.normal_multipliers,
                )
                if (np.any(candidate.reference_support[selected] <= 0.0)
                        or not np.array_equal(candidate.pressure > 0.0,
                                              baseline_active)):
                    return None
                sides.append(candidate.normalized_gap[selected])
            minus, plus = sides
            left = (center-minus)/h
            right = (plus-center)/h
            largest_one_sided = max(largest_one_sided, float(np.max(
                np.abs(left-right), initial=0.0,
            )))
            derivative[:, coordinate] = 0.5*(left+right)
        estimates.append(derivative)
    stability = float(np.max(np.abs(estimates[0]-estimates[1]), initial=0.0))
    # Both are dimensionless derivatives of a length gap by a length DOF.
    # The threshold is a development response-accuracy guard; the final
    # coupled proposal is checked against unchanged dimensional gates.
    if (not all(np.all(np.isfinite(item)) for item in estimates)
            or not np.isfinite(largest_one_sided)
            or max(largest_one_sided, stability) > 1.0e-7):
        if diagnostics is not None:
            diagnostics["schur_exact_seam_derivative_failure"] = {
                "one_sided": largest_one_sided, "step_stability": stability,
            }
        return None
    if diagnostics is not None:
        diagnostics["schur_exact_seam_generalized_gradient"] = True
        diagnostics["schur_exact_seam_one_sided_error"] = largest_one_sided
        diagnostics["schur_exact_seam_step_error"] = stability
        diagnostics["schur_exact_seam_geometry_evaluations"] = evaluations
    return estimates[1]


def _constrained_gap_response(manager, displacement, tangent, transform,
                              control_rows, selected: np.ndarray,
                              active: np.ndarray, *, difference_step: float,
                              check_cancel: Callable[[], None] | None = None,
                              diagnostics: dict[str, object] | None = None,
                              ) -> np.ndarray | None:
    """Return d(g_selected)/d(lambda_active) from one constrained factorization."""
    source, target = manager._positions(displacement)
    geometry = manager.law.geometry
    basis = getattr(manager.law, "pressure_basis", None)
    fine_base = geometry.evaluate(source, target)
    base = (manager.law.evaluate(source, target, manager.normal_multipliers)
            if basis is not None else fine_base)
    clip_signature = _projected_clip_signature(geometry, source, target)
    exact_gap_derivative = (
        _exact_artificial_seam_gap_gradient(
            manager, source, target, base, selected,
            check_cancel=check_cancel, diagnostics=diagnostics,
        ) if clip_signature is None and basis is None else None
    )
    if clip_signature is None and exact_gap_derivative is None:
        return None
    if np.any(base.reference_support[selected] <= 0.0):
        return None
    if not len(active):
        return np.zeros((len(selected), 0))
    coordinates = np.concatenate((source, target)).reshape(-1)
    count = len(coordinates)
    pair_count = len(getattr(geometry, "pair_laws", (1,)))
    if exact_gap_derivative is not None:
        weighted = support = None
    elif getattr(geometry, "differentiation", "global") in {"jet2_s3", "jet2_q4_s3"}:
        fine_selected = (np.arange(len(fine_base.reference_support))
                         if basis is not None else selected)
        weighted, support = _jet2_integral_gradients(
            geometry, source, target, fine_selected, count,
            manager.settings.maximum_general_s3_energy_evaluations,
            check_cancel, manager=manager, base=fine_base,
            diagnostics=diagnostics,
        )
        if basis is not None:
            support, weighted = basis.transform_derivatives(support, weighted)
            support = support[selected]
            weighted = weighted[selected]
    else:
        if (2*count + 1)*pair_count > manager.settings.maximum_general_s3_energy_evaluations:
            raise ValueError("pressure response geometry evaluation budget exceeded")
        weighted = np.zeros((len(selected), count))
        support = np.zeros_like(weighted)
        source_count = len(source)
        for coordinate in range(count):
            if check_cancel is not None:
                check_cancel()
            shifted = coordinates.copy()
            shifted[coordinate] += difference_step
            plus = geometry.evaluate(
                shifted[:3*source_count].reshape(-1, 3),
                shifted[3*source_count:].reshape(-1, 3),
            )
            shifted[coordinate] -= 2*difference_step
            minus = geometry.evaluate(
                shifted[:3*source_count].reshape(-1, 3),
                shifted[3*source_count:].reshape(-1, 3),
            )
            plus_source = (coordinates.copy())
            plus_source[coordinate] += difference_step
            minus_source = (coordinates.copy())
            minus_source[coordinate] -= difference_step
            plus_clip = _projected_clip_signature(
                geometry, plus_source[:3*source_count].reshape(-1, 3),
                plus_source[3*source_count:].reshape(-1, 3),
            )
            minus_clip = _projected_clip_signature(
                geometry, minus_source[:3*source_count].reshape(-1, 3),
                minus_source[3*source_count:].reshape(-1, 3),
            )
            if (plus.overlap_signature != base.overlap_signature
                    or minus.overlap_signature != base.overlap_signature
                    or plus_clip != clip_signature
                    or minus_clip != clip_signature
                    or np.any(plus.reference_support[selected] <= 0.0)
                    or np.any(minus.reference_support[selected] <= 0.0)):
                return None
            weighted[:, coordinate] = (
                plus.reference_weighted_gap[selected]
                - minus.reference_weighted_gap[selected]
            )/(2*difference_step)
            support[:, coordinate] = (
                plus.reference_support[selected]
                - minus.reference_support[selected]
            )/(2*difference_step)
    active_positions = np.searchsorted(selected, active)
    if np.any(selected[active_positions] != active):
        raise ValueError("active pressure mode lacks positive reference support")
    if exact_gap_derivative is not None:
        gap_derivative = exact_gap_derivative
        gap_variation_active = (
            base.reference_support[active, None]
            * exact_gap_derivative[active_positions]
        )
    else:
        area = base.reference_support[selected, None]
        selected_gap = base.normalized_gap[selected]
        gap_derivative = (weighted-selected_gap[:, None]*support)/area
        gap_variation_active = (
            weighted[active_positions]
            - selected_gap[active_positions, None]*support[active_positions]
        )
    variations = manager.map @ transform
    # Contact residual is -sum_i p_i(dW_i-g_i dS_i); held-multiplier
    # coupling therefore contains the moving-support term.
    force_derivative = -np.asarray(variations.T @ gap_variation_active.T)
    gap_reduced = np.asarray(variations.T @ gap_derivative.T).T
    reduced_tangent = (transform.T @ tangent @ transform).tocsr()
    rows = np.asarray(control_rows, dtype=float)
    if diagnostics is not None:
        unconstrained_norms = np.linalg.norm(force_derivative, axis=0)
        unconstrained_scaled = force_derivative / np.where(
            unconstrained_norms > 0.0, unconstrained_norms, 1.0,
        )[None, :]
        diagnostics["schur_force_map_unconstrained_singular_values"] = list(
            map(float, np.linalg.svd(unconstrained_scaled, compute_uv=False)))
        if len(rows):
            single_control_spectra = []
            for row in rows:
                single_force = null_space(row[None, :]).T @ force_derivative
                single_norms = np.linalg.norm(single_force, axis=0)
                single_scaled = single_force / np.where(
                    single_norms > 0.0, single_norms, 1.0,
                )[None, :]
                single_control_spectra.append(list(map(
                    float, np.linalg.svd(single_scaled, compute_uv=False),
                )))
            diagnostics["schur_force_map_single_control_singular_values"] = (
                single_control_spectra
            )
        admissible_force = (
            null_space(rows).T @ force_derivative
            if len(rows) else force_derivative
        )
        column_norms = np.linalg.norm(admissible_force, axis=0)
        scaled_force = admissible_force / np.where(
            column_norms > 0.0, column_norms, 1.0,
        )[None, :]
        diagnostics["schur_force_map_column_norms"] = list(
            map(float, column_norms))
        diagnostics["schur_force_map_scaled_singular_values"] = list(
            map(float, np.linalg.svd(scaled_force, compute_uv=False)))
    matrix = sparse.bmat([
        [reduced_tangent, sparse.csr_matrix(-rows.T)],
        [sparse.csr_matrix(rows),
         sparse.csr_matrix((len(rows), len(rows)))],
    ], format="csr")
    right = np.vstack((-force_derivative,
                       np.zeros((len(rows), len(active)))))
    factor = factorize(matrix, MatrixClass.GENERAL,
                       signature="contact:pressure_response")
    if factor.status != "ok":
        if diagnostics is not None:
            diagnostics["schur_response_failure"] = (
                factor.failure_reason or factor.status
            )
        return None
    try:
        response = np.column_stack([
            np.asarray(factor.solve(right[:, column])).reshape(-1)[:transform.shape[1]]
            for column in range(len(active))
        ])
    except RuntimeError as exc:
        if diagnostics is not None:
            diagnostics["schur_response_failure"] = str(exc)[:240]
        return None
    schur = gap_reduced @ response
    return schur if np.all(np.isfinite(schur)) else None


def refined_patch_pivot(manager, displacement, tangent, transform, control_rows,
                        *, difference_step: float, maximum_pressure: float,
                        pressure_tolerance: float,
                        check_cancel: Callable[[], None] | None = None,
                        ) -> ActiveSetPivot | None:
    """Preserve the historical two-to-four-active-mode pivot route."""
    source, target = manager._positions(displacement)
    pressure = manager.normal_multipliers
    state = manager.law.evaluate(source, target, pressure)
    active = np.flatnonzero(state.pressure > 0.0)
    if not 2 <= len(active) <= 4:
        return None
    schur = _constrained_gap_response(
        manager, displacement, tangent, transform, control_rows,
        active, active, difference_step=difference_step,
        check_cancel=check_cancel,
    )
    if schur is None:
        return None
    return propose_active_set_pivot(
        pressure, state.normalized_gap, active, schur,
        maximum_pressure=maximum_pressure,
        gap_consistency=pressure_tolerance/manager.stiffness,
    )


def refined_patch_supported_schur(
    manager, displacement, tangent, transform, control_rows, *,
    difference_step: float, maximum_pressure: float,
    maximum_pressure_step: float, maximum_modes: int,
    check_cancel: Callable[[], None] | None = None,
    diagnostics: dict[str, object] | None = None,
    retain_backtrack: bool = False,
) -> SupportedSchurProposal | None:
    """Propose a bounded projected-root correction for every supported mode."""
    source, target = manager._positions(displacement)
    pressure = manager.normal_multipliers
    state = manager.law.evaluate(source, target, pressure)
    supported = state.reference_support > 0.0
    selected = np.flatnonzero(supported)
    if len(selected) > maximum_modes:
        raise MemoryError("supported pressure modes exceed the declared Schur cap")
    active = np.flatnonzero(state.pressure > 0.0)
    if not len(active):
        if diagnostics is not None:
            diagnostics["schur_ineligible"] = "no_active_pressure_modes"
        return None
    response_active = _constrained_gap_response(
        manager, displacement, tangent, transform, control_rows,
        selected, active, difference_step=difference_step,
        check_cancel=check_cancel, diagnostics=diagnostics,
    )
    if response_active is None:
        if diagnostics is not None:
            diagnostics["schur_ineligible"] = "projection_branch_or_nonfinite_response"
        return None
    response = np.zeros((len(selected), len(selected)))
    response[:, np.searchsorted(selected, active)] = response_active
    return propose_supported_schur_step(
        pressure, state.normalized_gap, supported, response,
        normal_stiffness=manager.stiffness,
        maximum_pressure=maximum_pressure,
        maximum_pressure_step=maximum_pressure_step,
        maximum_modes=maximum_modes,
        diagnostics=diagnostics,
        check_cancel=check_cancel,
        retain_backtrack=retain_backtrack,
    )
