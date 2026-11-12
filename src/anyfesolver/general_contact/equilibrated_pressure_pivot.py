"""Disposable two-mode equilibrated pressure proposal for a refined shell patch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse

from anyfesolver.linalg import MatrixClass, factorize
from anyfesolver.nonlinear_static import (
    _assemble_nonlinear_system, _discard_nonlinear_state_candidate,
)

from .augmentation import ActiveSetPivot


def equilibrated_pressure_pivot(
        manager, model, displacement, reduced_coordinates,
        control_multipliers, committed_states, num_layers,
        transform, offset, rows, reduced_external, target, *,
        pressure_step: float, maximum_pressure: float,
        max_iterations: int, tolerance: float,
        check_cancel: Callable[[], None] | None = None,
        ) -> ActiveSetPivot | None:
    """Differentiate the projected map along equilibrated pressure trials.

    A coordinate-wise geometric derivative can cross a projection boundary
    that the actual equilibrated pressure path does not. These three trials
    use the complete structural/contact tangent, then the caller verifies any
    proposal in its ordinary augmented equilibrium loop.
    """

    if (not np.isfinite(pressure_step) or pressure_step <= 0.0
            or not np.isfinite(maximum_pressure) or maximum_pressure <= 0.0
            or type(max_iterations) is not int or max_iterations < 1):
        raise ValueError("equilibrated pressure pivot controls are invalid")
    original = manager.normal_multipliers.copy()
    original_reset = manager.last_support_reset
    try:
        projected = manager.projected_normal_multipliers(displacement)
        active = np.flatnonzero(projected > 0.0)
        if len(active) != 2:
            return None
        source, target_positions = manager._positions(displacement)
        base_geometry = manager.law.geometry.evaluate(source, target_positions)
        if np.any(base_geometry.reference_support[active] <= 0.0):
            return None
        baseline = np.zeros_like(original)
        baseline[active] = original[active]
        if np.any(baseline[active] <= 0.0):
            return None
        retained_amplitude = float(np.max(baseline[active]))
        if retained_amplitude > maximum_pressure:
            return None
        reactions = np.asarray(rows).T
        pressure_vectors = [baseline]
        for index in active:
            perturbed = baseline.copy()
            perturbed[index] += pressure_step
            pressure_vectors.append(perturbed)
        residuals = []
        signatures = []
        for pressure in pressure_vectors:
            if check_cancel is not None:
                check_cancel()
            manager.set_normal_multipliers(pressure)
            q = np.asarray(reduced_coordinates, dtype=float).copy()
            controls = np.asarray(control_multipliers, dtype=float).copy()
            for iteration in range(max_iterations):
                if check_cancel is not None:
                    check_cancel()
                u = np.asarray(transform @ q + offset).reshape(-1)
                internal, tangent, _trial = _assemble_nonlinear_system(
                    model, u, committed_states, num_layers,
                    kinematics="von_karman",
                )
                residual = (reduced_external + reactions @ controls
                            - np.asarray(transform.T @ internal).reshape(-1))
                constraint = target - rows @ q
                scale = max(1.0, float(np.linalg.norm(
                    reduced_external + reactions @ controls,
                )))
                if (np.linalg.norm(residual) <= tolerance*scale
                        and np.max(np.abs(constraint)) <= tolerance):
                    certificate = manager.evaluate(u, tangent=False)[2]
                    if (not certificate.gap_tolerance_met
                            or certificate.active_pressure_dofs != 2):
                        return None
                    source, target_positions = manager._positions(u)
                    signature = manager.law.geometry.evaluate(
                        source, target_positions,
                    ).overlap_signature
                    updated = manager.projected_normal_multipliers(u)
                    if np.any(updated[np.setdiff1d(
                            np.arange(len(updated)), active,
                    )] > 0.0):
                        return None
                    residuals.append(updated[active]-pressure[active])
                    signatures.append(signature)
                    break
                reduced = (transform.T @ tangent @ transform).tocsr()
                matrix = sparse.bmat([
                    [reduced, sparse.csr_matrix(-reactions)],
                    [sparse.csr_matrix(rows),
                     sparse.csr_matrix((len(rows), len(rows)))],
                ], format="csr")
                delta = np.asarray(factorize(
                    matrix, MatrixClass.GENERAL,
                    signature="contact_path:equilibrated_pressure_pivot",
                ).solve(np.r_[residual, constraint])).reshape(-1)
                proposed = np.asarray(transform @ (q+delta[:len(q)]) + offset).reshape(-1)
                fraction = manager.admissible_step_fraction(u, proposed)
                if fraction <= 1.0e-12:
                    return None
                q += fraction*delta[:len(q)]
                controls += fraction*delta[len(q):]
            else:
                return None
        if not all(item == signatures[0] for item in signatures[1:]):
            return None
        values = np.asarray(residuals)
        derivative = ((values[1:]-values[0])/pressure_step).T
        if not np.all(np.isfinite(derivative)):
            return None
        condition = float(np.linalg.cond(derivative))
        if not np.isfinite(condition) or condition > 1.0e7:
            return None
        root = baseline[active] - np.linalg.solve(derivative, values[0])
        if (not np.all(np.isfinite(root)) or np.any(root <= 0.0)
                or np.max(root) > maximum_pressure
                or np.max(np.abs(root-baseline[active])) > 2.0*retained_amplitude):
            return None
        proposal = np.zeros_like(original)
        proposal[active] = root
        return ActiveSetPivot(proposal, (0, 1), condition)
    finally:
        manager.set_normal_multipliers(original)
        manager.last_support_reset = original_reset
        _discard_nonlinear_state_candidate(committed_states)
