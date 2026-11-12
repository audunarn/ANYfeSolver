"""Bounded algorithmic proposals for a supported augmented pressure space.

The caller supplies the equilibrated normalized-gap response to multipliers.
This module never accepts a mechanical increment or commits physical history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SupportedSchurProposal:
    multipliers: np.ndarray
    active_indices: tuple[int, ...]
    scaled_condition: float | None
    step_fraction: float
    predicted_residual_before: float
    predicted_residual_after: float
    backtrack_data: SupportedSchurBacktrackData | None = None


@dataclass(frozen=True)
class SupportedSchurBacktrackData:
    """The frozen local response used only to propose a smaller trial step."""

    base_multipliers: np.ndarray
    supported_indices: np.ndarray
    base_gap: np.ndarray
    equilibrated_gap_response: np.ndarray
    normal_stiffness: float
    maximum_pressure: float
    maximum_pressure_step: float
    maximum_modes: int
    kink_roundoff: float


def propose_supported_schur_backtrack(
    full_proposal: SupportedSchurProposal,
    *,
    diagnostics: dict[str, object] | None = None,
) -> SupportedSchurProposal | None:
    """Find the first predicted-safe interior fraction of an existing step.

    This only evaluates the saved linearized gap map. The static owner must
    restore the base equilibrium and verify the candidate after a new solve.
    """
    data = full_proposal.backtrack_data
    if data is None:
        if diagnostics is not None:
            diagnostics["schur_backtrack_reason"] = "response_unavailable"
        return None
    base = data.base_multipliers
    full = np.asarray(full_proposal.multipliers, dtype=float)
    modes = data.supported_indices
    H = data.equilibrated_gap_response
    if (base.ndim != 1 or full.shape != base.shape or not len(base)
            or modes.ndim != 1 or not len(modes)
            or not np.issubdtype(modes.dtype, np.integer)
            or np.any(modes < 0) or np.any(modes >= len(base))
            or np.any(np.diff(modes) <= 0)
            or len(modes) > data.maximum_modes
            or data.base_gap.shape != (len(modes),)
            or H.shape != (len(modes), len(modes))
            or not np.isfinite(data.normal_stiffness)
            or data.normal_stiffness <= 0.0
            or not np.isfinite(data.maximum_pressure)
            or data.maximum_pressure <= 0.0
            or not np.isfinite(data.maximum_pressure_step)
            or data.maximum_pressure_step <= 0.0
            or not np.isfinite(data.kink_roundoff)
            or data.kink_roundoff < 0.0
            or not np.all(np.isfinite(full))
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(data.base_gap))
            or not np.all(np.isfinite(H))
            or not np.isfinite(full_proposal.predicted_residual_before)
            or full_proposal.predicted_residual_before <= 0.0
            or np.any(base < 0.0) or np.any(full < 0.0)
            or np.any(base[np.setdiff1d(np.arange(len(base)), modes)] != 0.0)
            or np.any(full[np.setdiff1d(np.arange(len(full)), modes)] != 0.0)
            or np.max(base, initial=0.0) > data.maximum_pressure
            or np.max(full, initial=0.0) > data.maximum_pressure
            or np.max(np.abs(full-base), initial=0.0)
            > data.maximum_pressure_step):
        if diagnostics is not None:
            diagnostics["schur_backtrack_reason"] = "inconsistent_full_proposal"
        return None
    direction = full-base
    examined = 0
    # Nineteen interior grid points give a bounded, deterministic search.
    # Ascending order minimizes the nonlinear pressure move among candidates
    # with a predicted 30% merit margin (owner requires only 25% in reality).
    for tick in range(1, 20):
        examined += 1
        fraction = tick/20.0
        candidate = base + fraction*direction
        step = candidate-base
        if (np.any(candidate < 0.0)
                or np.max(candidate, initial=0.0) > data.maximum_pressure
                or np.max(np.abs(step), initial=0.0)
                > data.maximum_pressure_step):
            continue
        local_pressure = candidate[modes]
        argument = local_pressure-data.normal_stiffness*(
            data.base_gap + H @ step[modes]
        )
        if (not np.all(np.isfinite(argument))
                or np.any(np.abs(argument) <= data.kink_roundoff)):
            continue
        residual = float(np.max(np.abs(
            np.maximum(0.0, argument)-local_pressure), initial=0.0))
        if residual > 0.70*full_proposal.predicted_residual_before:
            continue
        active = tuple(int(modes[index]) for index in
                       np.flatnonzero(argument > 0.0))
        if diagnostics is not None:
            diagnostics["schur_backtrack_examined"] = examined
            diagnostics["schur_backtrack_fraction"] = fraction
            diagnostics["schur_backtrack_predicted_residual_pa"] = residual
        return SupportedSchurProposal(
            candidate, active, full_proposal.scaled_condition,
            fraction*full_proposal.step_fraction,
            full_proposal.predicted_residual_before, residual,
        )
    if diagnostics is not None:
        diagnostics["schur_backtrack_examined"] = examined
        diagnostics["schur_backtrack_reason"] = "no_predicted_safe_interior_fraction"
    return None


def propose_supported_schur_secant_backtrack(
    full_proposal: SupportedSchurProposal,
    actual_full_gap: np.ndarray,
    *,
    diagnostics: dict[str, object] | None = None,
) -> SupportedSchurProposal | None:
    """Suggest an interior trial from the observed full-step gap curvature.

    The quadratic interpolant matches the base gap, its saved local slope,
    and the rejected full-step gap. It is only a proposal: the owner must
    re-equilibrate and apply its unchanged physical and merit checks.
    """
    data = full_proposal.backtrack_data

    def decline(reason: str) -> None:
        if diagnostics is not None:
            diagnostics["schur_secant_backtrack_reason"] = reason
        return None

    if data is None:
        return decline("response_unavailable")
    base = data.base_multipliers
    full = np.asarray(full_proposal.multipliers, dtype=float)
    actual_gap = np.asarray(actual_full_gap, dtype=float)
    modes = data.supported_indices
    H = data.equilibrated_gap_response
    if (base.ndim != 1 or not len(base) or full.shape != base.shape
            or actual_gap.shape != base.shape
            or modes.ndim != 1 or not len(modes)
            or not np.issubdtype(modes.dtype, np.integer)
            or np.any(modes < 0) or np.any(modes >= len(base))
            or np.any(np.diff(modes) <= 0)
            or len(modes) > data.maximum_modes
            or data.base_gap.shape != (len(modes),)
            or H.shape != (len(modes), len(modes))
            or not np.isfinite(data.normal_stiffness)
            or data.normal_stiffness <= 0.0
            or not np.isfinite(data.maximum_pressure)
            or data.maximum_pressure <= 0.0
            or not np.isfinite(data.maximum_pressure_step)
            or data.maximum_pressure_step <= 0.0
            or not np.isfinite(data.kink_roundoff)
            or data.kink_roundoff < 0.0
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(full))
            or not np.all(np.isfinite(actual_gap))
            or not np.all(np.isfinite(data.base_gap))
            or not np.all(np.isfinite(H))
            or not np.isfinite(full_proposal.predicted_residual_before)
            or full_proposal.predicted_residual_before <= 0.0
            or np.any(base < 0.0) or np.any(full < 0.0)
            or np.any(base[np.setdiff1d(np.arange(len(base)), modes)] != 0.0)
            or np.any(full[np.setdiff1d(np.arange(len(full)), modes)] != 0.0)
            or np.max(base, initial=0.0) > data.maximum_pressure
            or np.max(full, initial=0.0) > data.maximum_pressure
            or np.max(np.abs(full-base), initial=0.0)
            > data.maximum_pressure_step):
        return decline("inconsistent_full_proposal_or_gap")

    direction = full-base
    linear_gap_change = H @ direction[modes]
    curvature = (actual_gap[modes]-data.base_gap-linear_gap_change)
    if not np.all(np.isfinite(curvature)):
        return decline("nonfinite_curvature")

    best: SupportedSchurProposal | None = None
    best_fraction = float("inf")
    minimum_scanned_residual = float("inf")
    minimum_scanned_fraction: float | None = None
    examined = 0
    eligible = 0
    for tick in range(1, 20):
        examined += 1
        fraction = tick/20.0
        candidate = base+fraction*direction
        step = candidate-base
        if (np.any(candidate < 0.0)
                or np.max(candidate, initial=0.0) > data.maximum_pressure
                or np.max(np.abs(step), initial=0.0)
                > data.maximum_pressure_step):
            continue
        local_pressure = candidate[modes]
        predicted_gap = (data.base_gap+fraction*linear_gap_change
                         +fraction*fraction*curvature)
        argument = local_pressure-data.normal_stiffness*predicted_gap
        if (not np.all(np.isfinite(argument))
                or np.any(np.abs(argument) <= data.kink_roundoff)):
            continue
        residual = float(np.max(np.abs(
            np.maximum(0.0, argument)-local_pressure), initial=0.0))
        if residual < minimum_scanned_residual:
            minimum_scanned_residual = residual
            minimum_scanned_fraction = fraction
        if residual > 0.70*full_proposal.predicted_residual_before:
            continue
        eligible += 1
        if (best is None
                or residual < best.predicted_residual_after
                or (residual == best.predicted_residual_after
                    and fraction < best_fraction)):
            active = tuple(int(modes[index]) for index in
                           np.flatnonzero(argument > 0.0))
            best = SupportedSchurProposal(
                candidate, active, full_proposal.scaled_condition,
                fraction*full_proposal.step_fraction,
                full_proposal.predicted_residual_before, residual,
            )
            best_fraction = fraction
    if diagnostics is not None:
        diagnostics["schur_secant_backtrack_examined"] = examined
        diagnostics["schur_secant_backtrack_eligible"] = eligible
        diagnostics["schur_secant_backtrack_minimum_scanned_residual_pa"] = (
            minimum_scanned_residual
            if minimum_scanned_fraction is not None else None)
        diagnostics["schur_secant_backtrack_minimum_scanned_fraction"] = (
            minimum_scanned_fraction)
        if best is not None:
            diagnostics["schur_secant_backtrack_fraction"] = best.step_fraction
            diagnostics["schur_secant_backtrack_predicted_residual_pa"] = (
                best.predicted_residual_after)
    if best is None:
        return decline("no_predicted_safe_interior_fraction")
    return best


def propose_supported_schur_step(
    multipliers: np.ndarray,
    normalized_gap: np.ndarray,
    supported: np.ndarray,
    equilibrated_gap_response: np.ndarray,
    *,
    normal_stiffness: float,
    maximum_pressure: float,
    maximum_pressure_step: float,
    maximum_modes: int,
    diagnostics: dict[str, object] | None = None,
    check_cancel: Callable[[], None] | None = None,
    retain_backtrack: bool = False,
) -> SupportedSchurProposal | None:
    """Propose one projected-root Newton step on a stable active branch.

    On an active branch, ``R=max(0,lambda-epsilon*g)-lambda`` has derivative
    ``-epsilon*H`` where ``H=dg_equilibrated/dlambda``. Inactive rows have
    derivative ``-I``. Fixed-branch backtracking is tried first; projected and
    bounded branch-aware candidates may change the predicted active set. Every
    proposal must reduce the local residual by the existing verification
    threshold. The analysis owner re-equilibrates and verifies the actual
    residual, active set and physical gap.
    """

    lam = np.asarray(multipliers, dtype=float).reshape(-1)
    gap = np.asarray(normalized_gap, dtype=float).reshape(-1)
    mask = np.asarray(supported, dtype=bool).reshape(-1)
    modes = np.flatnonzero(mask)
    count = len(modes)
    H = np.asarray(equilibrated_gap_response, dtype=float)
    if (gap.shape != lam.shape or mask.shape != lam.shape
            or H.shape != (count, count)
            or not all(np.all(np.isfinite(item)) for item in (lam, gap, H))
            or np.any(lam < 0.0) or np.any(lam[~mask] != 0.0)
            or type(maximum_modes) is not int or maximum_modes < 1):
        raise ValueError("supported Schur inputs are inconsistent")
    for name, value in (("normal_stiffness", normal_stiffness),
                        ("maximum_pressure", maximum_pressure),
                        ("maximum_pressure_step", maximum_pressure_step)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if count > maximum_modes:
        raise MemoryError("supported pressure modes exceed the declared Schur cap")
    def decline(reason: str) -> None:
        if diagnostics is not None:
            diagnostics["schur_ineligible"] = reason
        return None

    if not count:
        return decline("no_supported_pressure_modes")
    pressure = lam[modes]
    physical_gap = gap[modes]
    projected_argument = pressure - normal_stiffness*physical_gap
    roundoff = 64.0*np.finfo(float).eps*max(
        1.0, maximum_pressure, float(np.max(np.abs(projected_argument))),
    )
    if np.any(np.abs(projected_argument) <= roundoff):
        return decline("projected_active_set_kink")
    active = projected_argument > 0.0
    residual = np.maximum(0.0, projected_argument) - pressure
    before = float(np.linalg.norm(residual, ord=np.inf))
    if before == 0.0:
        return decline("projected_residual_zero")
    backtrack_data = None
    if retain_backtrack:
        def frozen(values: np.ndarray) -> np.ndarray:
            copy = np.array(values, copy=True)
            copy.setflags(write=False)
            return copy

        backtrack_data = SupportedSchurBacktrackData(
            frozen(lam), frozen(modes), frozen(physical_gap), frozen(H),
            normal_stiffness, maximum_pressure, maximum_pressure_step,
            maximum_modes, roundoff,
        )
    jacobian = -np.eye(count)
    jacobian[active] = -normal_stiffness*H[active]
    row_magnitude = np.max(np.abs(jacobian), axis=1)
    row_scale = np.where(row_magnitude > 0.0, row_magnitude, 1.0)
    column_magnitude = np.max(np.abs(jacobian/row_scale[:, None]), axis=0)
    column_scale = np.where(column_magnitude > 0.0, column_magnitude, 1.0)
    scaled = jacobian/row_scale[:, None]/column_scale[None, :]
    try:
        left_modes, singular_values, right_modes = np.linalg.svd(scaled)
        condition = (float(singular_values[0]/singular_values[-1])
                     if singular_values[-1] > 0.0 else float("inf"))
        if diagnostics is not None:
            diagnostics["schur_scaled_condition"] = (
                condition if np.isfinite(condition) else None
            )
            diagnostics["schur_scaled_singular_values"] = list(
                map(float, singular_values))
            scaled_rhs = -residual/row_scale
            modal_rhs = left_modes.T @ scaled_rhs
            diagnostics["schur_scaled_rhs_modal"] = list(map(float, modal_rhs))
            diagnostics["schur_scaled_weak_right_mode"] = list(
                map(float, right_modes[-1]))
            diagnostics["schur_scaled_weak_rhs_fraction"] = float(
                abs(modal_rhs[-1])/max(np.linalg.norm(scaled_rhs),
                                        np.finfo(float).tiny)
            )
        if not np.isfinite(singular_values[0]) or singular_values[0] <= 0.0:
            return decline("scaled_response_singular")
        retained = singular_values > singular_values[0]/1.0e8
        rank = int(np.count_nonzero(retained))
        if diagnostics is not None:
            diagnostics["schur_effective_rank"] = rank
            diagnostics["schur_truncated_response"] = rank < count
        if rank == 0:
            return decline("scaled_response_singular")
        if rank < count:
            direction = (
                right_modes[:rank].T
                @ ((left_modes[:, :rank].T @ (-residual/row_scale))
                   / singular_values[:rank])
            )/column_scale
        else:
            direction = np.linalg.solve(scaled, -residual/row_scale)/column_scale
    except np.linalg.LinAlgError:
        return decline("scaled_response_singular")
    if not np.all(np.isfinite(direction)):
        return decline("nonfinite_pressure_direction")
    if diagnostics is not None:
        diagnostics["schur_full_step_max_pa"] = float(np.max(np.abs(direction)))
        diagnostics["schur_initial_pressure_pa"] = list(map(float, pressure))
        diagnostics["schur_full_direction_pa"] = list(map(float, direction))
        diagnostics["schur_backtrack_rejections"] = {
            "step_cap": 0, "negative_pressure": 0,
            "pressure_cap": 0, "active_branch": 0, "predicted_merit": 0,
        }
    best_fixed_branch_residual = None
    fixed_proposal = None
    for halvings in range(9):
        fraction = 0.5**halvings
        step = fraction*direction
        candidate = pressure + step
        if np.max(np.abs(step)) > maximum_pressure_step:
            if diagnostics is not None:
                diagnostics["schur_backtrack_rejections"]["step_cap"] += 1
            continue
        if np.any(candidate < 0.0):
            if diagnostics is not None:
                diagnostics["schur_backtrack_rejections"]["negative_pressure"] += 1
            continue
        if np.max(candidate) > maximum_pressure:
            if diagnostics is not None:
                diagnostics["schur_backtrack_rejections"]["pressure_cap"] += 1
            continue
        predicted_gap = physical_gap + H @ step
        predicted_argument = candidate - normal_stiffness*predicted_gap
        if np.any((predicted_argument > 0.0) != active):
            if diagnostics is not None:
                diagnostics["schur_backtrack_rejections"]["active_branch"] += 1
            continue
        after = float(np.linalg.norm(
            np.maximum(0.0, predicted_argument)-candidate,
            ord=np.inf,
        ))
        if diagnostics is not None:
            if (best_fixed_branch_residual is None
                    or after < best_fixed_branch_residual):
                best_fixed_branch_residual = after
                diagnostics["schur_best_fixed_branch_fraction"] = fraction
                diagnostics["schur_best_fixed_branch_residual_pa"] = after
        # The static owner rejects a re-equilibrated proposal unless its
        # projected residual falls below 75% of the previous value.  Do not
        # spend a nonlinear solve on a linearly predicted rejection.
        if after >= 0.75*before:
            if diagnostics is not None:
                diagnostics["schur_backtrack_rejections"]["predicted_merit"] += 1
            continue
        proposed = lam.copy()
        proposed[modes] = candidate
        fixed_proposal = SupportedSchurProposal(
            proposed, tuple(int(modes[index]) for index in np.flatnonzero(active)),
            condition if np.isfinite(condition) else None,
            fraction, before, after,
            backtrack_data,
        )
        break
    if (fixed_proposal is not None
            and fixed_proposal.predicted_residual_after <= 1.0e-8*before):
        if diagnostics is not None:
            diagnostics["schur_proposal_kind"] = "fixed_active_branch"
        return fixed_proposal
    # If positivity blocks the fixed branch, project the same response step
    # to the admissible pressure cone.  Recompute the projected residual and
    # active set at the projected candidate; the owner re-equilibrates and
    # verifies this predicted set before accepting an algorithmic update.
    best_projected_residual = None
    projected_proposal = None
    for halvings in range(9):
        fraction = 0.5**halvings
        candidate = np.maximum(0.0, pressure + fraction*direction)
        step = candidate-pressure
        if (np.max(np.abs(step)) > maximum_pressure_step
                or np.max(candidate) > maximum_pressure):
            continue
        predicted_gap = physical_gap + H @ step
        predicted_argument = candidate - normal_stiffness*predicted_gap
        if np.any(np.abs(predicted_argument) <= roundoff):
            continue
        after = float(np.linalg.norm(
            np.maximum(0.0, predicted_argument)-candidate,
            ord=np.inf,
        ))
        if diagnostics is not None and (best_projected_residual is None
                                        or after < best_projected_residual):
            best_projected_residual = after
            diagnostics["schur_best_projected_fraction"] = fraction
            diagnostics["schur_best_projected_residual_pa"] = after
        if after >= 0.75*before:
            continue
        predicted_active = tuple(int(modes[index]) for index in
                                 np.flatnonzero(predicted_argument > 0.0))
        proposed = lam.copy()
        proposed[modes] = candidate
        trial = SupportedSchurProposal(
            proposed, predicted_active,
            condition if np.isfinite(condition) else None,
            fraction, before, after,
            backtrack_data,
        )
        projected_proposal = trial
        break

    def fallback_proposal():
        candidates = [
            (trial, kind) for trial, kind in (
                (fixed_proposal, "fixed_active_branch"),
                (projected_proposal, "projected_active_set"),
            ) if trial is not None
        ]
        if not candidates:
            return None
        selected, kind = candidates[0]
        tie_tolerance = 1.0e-12+1.0e-10*before
        for trial, trial_kind in candidates[1:]:
            if (trial.predicted_residual_after
                    < selected.predicted_residual_after-tie_tolerance):
                selected, kind = trial, trial_kind
        if diagnostics is not None:
            diagnostics["schur_proposal_kind"] = kind
        return selected
    # The first Newton direction can point outside the pressure cone even
    # when another active branch has a small admissible root.  Search the
    # same local, equilibrated *linearized* gap map with a branch Jacobian.
    # This remains only a disposable proposal: the owner must solve the full
    # mechanics again and verify the unchanged merit, gap and active-set gates.
    lower = np.maximum(0.0, pressure-maximum_pressure_step)
    upper = np.minimum(maximum_pressure, pressure+maximum_pressure_step)
    if np.any(lower >= upper):
        fallback = fallback_proposal()
        if fallback is not None:
            return fallback
        return decline("bounded_step_or_predicted_merit_failed")

    def local_argument(candidate: np.ndarray) -> np.ndarray:
        return candidate-normal_stiffness*(
            physical_gap + H @ (candidate-pressure)
        )

    def local_residual(candidate: np.ndarray) -> np.ndarray:
        if check_cancel is not None:
            check_cancel()
        return np.maximum(0.0, local_argument(candidate))-candidate

    def local_branch_jacobian(candidate: np.ndarray) -> np.ndarray:
        if check_cancel is not None:
            check_cancel()
        branch = local_argument(candidate) > 0.0
        derivative = -np.eye(count)
        derivative[branch] = -normal_stiffness*H[branch]
        return derivative

    maximum_local_evaluations = min(8*count+16, 256)
    proposals = []
    evaluations = 0
    for seed in (pressure, lower):
        try:
            bounded = least_squares(
                local_residual, seed, jac=local_branch_jacobian,
                bounds=(lower, upper), max_nfev=maximum_local_evaluations,
                xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12,
            )
        except (ValueError, np.linalg.LinAlgError, MemoryError):
            continue
        evaluations += int(bounded.nfev)
        candidate = bounded.x
        if not bounded.success or not np.all(np.isfinite(candidate)):
            continue
        argument = local_argument(candidate)
        after = float(np.linalg.norm(local_residual(candidate), ord=np.inf))
        if (not np.all(np.isfinite(argument)) or not np.isfinite(after)
                or np.any(candidate < lower-1.0e-12)
                or np.any(candidate > upper+1.0e-12)
                or np.any(np.abs(argument) <= roundoff)):
            continue
        proposals.append((after, float(np.linalg.norm(candidate-pressure)),
                          float(np.linalg.norm(candidate)), candidate,
                          argument))
    if diagnostics is not None:
        diagnostics["schur_bounded_semismooth_evaluations"] = evaluations
        diagnostics["schur_bounded_semismooth_candidates"] = len(proposals)
    if not proposals:
        fallback = fallback_proposal()
        if fallback is not None:
            return fallback
        return decline("bounded_semismooth_local_solve_failed")
    after, _step_norm, _pressure_norm, candidate, argument = min(
        proposals, key=lambda item: item[:3],
    )
    if diagnostics is not None:
        diagnostics["schur_bounded_semismooth_residual_pa"] = after
    fallback = fallback_proposal()
    tie_tolerance = 1.0e-12+1.0e-10*before
    if (after >= 0.75*before
            or (fallback is not None
                and after >= fallback.predicted_residual_after-tie_tolerance)):
        if fallback is not None:
            return fallback
        return decline("bounded_step_or_predicted_merit_failed")
    proposed = lam.copy()
    proposed[modes] = candidate
    if diagnostics is not None:
        diagnostics["schur_proposal_kind"] = "bounded_semismooth_active_set"
    return SupportedSchurProposal(
        proposed,
        tuple(int(modes[index]) for index in np.flatnonzero(argument > 0.0)),
        condition if np.isfinite(condition) else None,
        1.0, before, after,
        backtrack_data,
    )
