"""One static contact-path increment with coupled primal/dual Newton work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from anyfesolver.control import cancellation_safe_point
from anyfesolver.linalg import MatrixClass, factorize
from anyfesolver.nonlinear_static import _assemble_nonlinear_system
from anyfesolver.nonlinear_static import _discard_nonlinear_state_candidate

from .core import suspend_contact
from .initial_contact_seed import propose_linearized_kkt_seed
from .semismooth import complementarity, linearize, reference_nodal_support
from .semismooth_block import prepare_frozen_block_response


@dataclass(frozen=True)
class SemismoothStep:
    completed: bool
    reason: str | None
    q: np.ndarray
    reactions: np.ndarray
    nodal_forces: np.ndarray
    candidate_states: object | None
    contact: object | None
    iterations: int
    residual_norm: float
    constraint_norm: float
    complementarity_norm: float
    candidate_evaluations: int
    backtracks: int
    diagnostics: dict | None = None
    initial_seed: dict | None = None


def _coupled_contact_blocks(structural_tangent, contact, transform):
    """Map the structural/contact tangent and gap work to free coordinates."""
    gap_q = np.asarray(contact.gap_gradient @ transform)
    kqq = (transform.T @ (structural_tangent+contact.tangent) @ transform).tocsr()
    force_gradient = (contact.gap_gradient if contact.force_gradient is None
                      else contact.force_gradient)
    kqz = (sparse.csr_matrix(-np.asarray(force_gradient @ transform).T)
           if contact.force_pressure_gradient is None else
           sparse.csr_matrix(transform.T @ contact.force_pressure_gradient))
    return kqq, kqz, gap_q


def _proposal_gap_diagnostic(manager, proposed_u):
    """Keep invalid rejected geometry from masking the solver's non-success."""
    try:
        lower = (manager.point_gap_lower(proposed_u) if hasattr(manager,'point_gap_lower')
                 else manager._gap_lower(manager._positions(proposed_u)))
    except ValueError as exc:
        return {"swept_proposed_gap_lower_m": None,
                "swept_proposed_geometry_error": f"{type(exc).__name__}: {exc}"}
    if not np.isfinite(lower):
        return {"swept_proposed_gap_lower_m": None,
                "swept_proposed_geometry_error": "nonfinite or unresolved physical gap bound"}
    return {"swept_proposed_gap_lower_m": lower,
            "swept_proposed_geometry_error": None}


def _assemble_direct_matrix(kqq, kqz, gap_q, control_reactions,
                            control_rows, scales, active):
    """Assemble one fixed active branch of the direct monolithic Newton system."""
    modes = len(scales)
    branch_q = np.zeros_like(gap_q)
    branch_q[active] = scales[active, None]*gap_q[active]
    branch_z = np.ones(modes)
    branch_z[active] = 0.0
    return sparse.bmat((
        (kqq, sparse.csr_matrix(-control_reactions), kqz),
        (sparse.csr_matrix(control_rows),
         sparse.csr_matrix((control_rows.shape[0], control_rows.shape[0])),
         sparse.csr_matrix((control_rows.shape[0], modes))),
        (sparse.csr_matrix(branch_q),
         sparse.csr_matrix((modes, control_rows.shape[0])),
         sparse.diags(branch_z, format="csr")),
    ), format="csr")


def _current_merit_slope(*, delta, q_size, mu_size, kqq, kqz,
                         gap_q, control_reactions, control_rows, residual,
                         constraint, ncp, z, gap, support, c, scale,
                         control_scales):
    """Directional derivative of the current scaled maximum residual merit.

    Predicting the full-step active set does not differentiate the residual
    at the current base. At projected-map ties use its one-sided derivative,
    including zero residual components in the L1/Linf directional norms.
    This is a search-order check, never a substitute for trial acceptance.
    """
    dq = delta[:q_size]
    dmu = delta[q_size:q_size+mu_size]
    dz = delta[q_size+mu_size:]
    dr = -kqq @ dq + control_reactions @ dmu - kqz @ dz
    dc = -control_rows @ dq
    dg = gap_q @ dq
    y = z-c*gap
    supported = support > 0.0
    dncp = np.asarray(dz).copy()
    active = supported & (y > 0.0)
    ties = supported & (y == 0.0)
    dncp[active] = c[active]*dg[active]
    dncp[ties] -= np.maximum(0.0, dz[ties]-c[ties]*dg[ties])
    rnorm = float(np.linalg.norm(residual))
    rvalue = rnorm/scale
    rslope = (float(np.dot(residual, dr))/rnorm if rnorm > 0.0
              else float(np.linalg.norm(dr)))/scale
    scaled = constraint/control_scales
    dscaled = dc/control_scales
    cvalue = float(np.max(np.abs(scaled), initial=0.0))
    if len(scaled):
        cslope = float(np.max(np.where(scaled == 0.0, np.abs(dscaled),
                                     np.sign(scaled)*dscaled)[np.abs(scaled) == cvalue]))
    else:
        cslope = 0.0
    nvalue = float(np.sum(np.abs(ncp)))/scale
    nslope = float(np.sum(np.where(ncp == 0.0, np.abs(dncp),
                                  np.sign(ncp)*dncp)))/scale
    values = np.asarray((rvalue, cvalue, nvalue))
    slopes = np.asarray((rslope, cslope, nslope))
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(slopes)):
        return np.inf
    return float(np.max(slopes[values == np.max(values)]))


def _original_direction_event_fraction(event, certified, last_trial):
    """Return an event step only inside its own proposal's certificate."""
    fraction = min(1.0, 0.5*event, certified)
    return (fraction if last_trial is not None
            and 1.0e-12 < fraction < last_trial else None)


#: Private diagnostics hook. When not None it receives the real seed
#: admission decision payload — the untouched origin pose, the proposed
#: pose, the live origin/candidate normalized gaps, the candidate
#: force/constraint/NCP blocks and the unchanged production decision
#: record — immediately after the seed decision and before any Newton
#: iteration. It is None in production, never runs on the default-none
#: route and changes no acceptance predicate or transaction gate.
_INITIAL_SEED_DECISION_HOOK = None


def solve_increment(*, model, manager, settings, q, reactions, nodal_forces,
                    committed_states, transform, offset, control_rows,
                    control_reactions, target, reduced_external, num_layers,
                    max_iterations, tolerance, control_scales,
                    path_force_magnitude, cancellation_token, step,
                    stagnation_patience=None, stagnation_fraction=None,
                    stagnation_merit_rate=None, initial_contact_seed=None):
    """Solve one declared path step without committing its material state.

    Optional stagnation controls end this increment early with the declared
    reason ``contact_semismooth_stagnation`` once ``stagnation_patience``
    consecutive iterations have each accepted a swept admissible fraction
    at or below ``stagnation_fraction`` while improving the merit by less
    than ``stagnation_merit_rate`` relative. The bail changes no
    acceptance predicate; the caller decides whether a bounded physical
    continuation retries the same declared target.
    """
    q = np.asarray(q, dtype=float).copy()
    mu = np.asarray(reactions, dtype=float).copy()
    z = np.asarray(nodal_forces, dtype=float).copy()
    control_scales = np.asarray(control_scales, dtype=float).reshape(-1)
    if (control_scales.shape != mu.shape
            or not np.all(np.isfinite(control_scales))
            or np.any(control_scales <= 0.0)):
        raise ValueError("contact-path control scales must match positive controls")
    stagnation_enabled = stagnation_patience is not None
    if stagnation_enabled and (
            type(stagnation_patience) is not int or stagnation_patience < 1
            or type(stagnation_fraction) is not float
            or not 0.0 < stagnation_fraction <= 1.0
            or type(stagnation_merit_rate) is not float
            or not 0.0 < stagnation_merit_rate < 1.0):
        raise ValueError("contact-path stagnation controls need an integer "
                         "patience >= 1, a fraction in (0, 1] and a merit "
                         "rate in (0, 1)")
    scale = settings.complementarity_force_scale_n
    support_ref = reference_nodal_support(manager.law)
    c = manager.stiffness*support_ref
    modes = len(z)
    if z.shape != (len(support_ref),):
        raise ValueError("active-set checkpoint has incompatible pressure modes")
    evaluations = 0
    backtracks = 0
    stagnation_run = 0
    stagnation_trace = []

    def evaluate(at_q, at_mu, at_z, *, tangent):
        nonlocal evaluations
        cancellation_safe_point(cancellation_token,
                                f"contact_path.step:{step}.semismooth.candidate")
        u = np.asarray(transform @ at_q + offset).reshape(-1)
        friction_path = getattr(manager, "require_friction_path", None)
        if friction_path is not None:
            origin = np.asarray(transform @ q + offset).reshape(-1)
            friction_path(origin, u)
            friction_path(manager.committed_displacement, u)
        finite = getattr(manager, "finite_skin", None) is not None
        if finite:
            origin = np.asarray(transform @ q + offset).reshape(-1)
            if not manager._swept_admissible(origin, u):
                raise ValueError("actual finite shell candidate path is unsafe or uncertified")
            manager.require_finite_deformation(u)
        with suspend_contact():
            internal, stiffness, material_trial = _assemble_nonlinear_system(
                model, u, committed_states, num_layers, tangent=tangent,
                kinematics="corotational" if finite else "von_karman",
                **({"corotational_tangent": "consistent"} if finite else {}),
            )
        contact = linearize(manager, u, at_z, tangent=tangent)
        evaluations += 1
        total_internal = np.asarray(internal).reshape(-1) + contact.force
        residual = (reduced_external + control_reactions @ at_mu
                    - np.asarray(transform.T @ total_internal).reshape(-1))
        constraint = target - control_rows @ at_q
        ncp, active, ncp_q, ncp_z = complementarity(
            contact.state, at_z, c, contact.gap_gradient, tangent=tangent,
        )
        value = max(float(np.linalg.norm(residual))/scale,
                    float(np.max(np.abs(constraint)/control_scales, initial=0.0)),
                    float(np.sum(np.abs(ncp)))/scale)
        return (u, material_trial, contact, stiffness, residual, constraint,
                ncp, active, ncp_q, ncp_z, value)

    seed_record = None
    if initial_contact_seed in ("linearized_kkt_dev",
                                "linearized_kkt_merit_globalization_dev"):
        # Opt-in initial-contact seed: one linearized KKT pressure proposal
        # at the committed origin, reevaluated through the same actual
        # mechanics, full scaled merit and conservative swept admissibility
        # before it may initialize Newton. A refused seed leaves the origin
        # untouched; no acceptance predicate changes here or below. The
        # merit-globalization selection additionally allows bounded
        # backtracking of the complete algorithmic correction as a
        # disposable initial Newton iterate when the full proposal is
        # unsafe or nonimproving; every trial passes the same actual
        # gates, and an explicit refusal stands when none does.
        cancellation_safe_point(cancellation_token,
                                f"contact_path.step:{step}.semismooth.initial_seed")
        seed_origin = (q, mu, z)
        proposal = None
        seed_candidate = None
        try:
            (origin_u, _origin_trial, origin_contact, origin_structural_k,
             origin_residual, origin_constraint, _origin_ncp, _origin_active,
             _origin_ncp_q, _origin_ncp_z, origin_merit) = evaluate(
                q, mu, z, tangent=True)
        except (ValueError, MemoryError) as exc:
            _discard_nonlinear_state_candidate(committed_states)
            seed_record = {"selection": initial_contact_seed,
                           "outcome": "refused",
                           "reason": f"origin_evaluation_failed: {exc}"[:200]}
        else:
            _discard_nonlinear_state_candidate(committed_states)
            seed_kqq, seed_kqz, seed_gap_q = _coupled_contact_blocks(
                origin_structural_k, origin_contact, transform,
            )
            proposal = propose_linearized_kkt_seed(
                kqq=seed_kqq, kqz=seed_kqz, gap_q=seed_gap_q,
                control_reactions=control_reactions,
                control_rows=control_rows, residual=origin_residual,
                constraint=origin_constraint, q=q,
                normalized_gap=origin_contact.state.normalized_gap,
                support=origin_contact.state.reference_support,
                signature=f"contact_path:{step}:initial_seed",
                selection=initial_contact_seed,
            )
            if proposal.refused:
                seed_record = {"selection": initial_contact_seed,
                               "outcome": "refused",
                               "reason": proposal.reason,
                               **proposal.diagnostics}
            else:
                seed_q = np.asarray(proposal.q, dtype=float).copy()
                seed_mu = np.asarray(proposal.reactions, dtype=float).copy()
                seed_z = np.asarray(proposal.nodal_forces, dtype=float).copy()
                unsupported = (np.asarray(
                    origin_contact.state.reference_support) <= 0.0)
                seed_z[unsupported] = 0.0
                seed_u = np.asarray(transform @ seed_q + offset).reshape(-1)
                seed_fraction = manager.admissible_step_fraction(origin_u,
                                                                 seed_u)
                seed_merit = np.inf
                seed_error = None
                try:
                    seed_candidate = evaluate(seed_q, seed_mu, seed_z,
                                             tangent=False)
                    seed_merit = float(seed_candidate[-1])
                except (ValueError, MemoryError) as exc:
                    seed_error = f"{type(exc).__name__}: {exc}"[:200]
                _discard_nonlinear_state_candidate(committed_states)
                seed_base = {"selection": initial_contact_seed,
                             "origin_merit": float(origin_merit),
                             "admissible_fraction": float(seed_fraction),
                             **proposal.diagnostics}
                if seed_error is not None:
                    seed_record = {**seed_base, "outcome": "refused",
                                   "reason": "seed_candidate_evaluation_failed",
                                   "detail": seed_error}
                elif not np.isfinite(seed_merit):
                    seed_record = {**seed_base, "outcome": "refused",
                                   "reason": "nonfinite_seed_candidate"}
                elif np.any(seed_z < 0.0):
                    seed_record = {**seed_base, "outcome": "refused",
                                   "reason": "negative_seed_pressure_force"}
                elif seed_fraction != 1.0:
                    seed_record = {**seed_base, "outcome": "refused",
                                   "reason": "seed_not_swept_admissible"}
                elif not seed_merit < origin_merit*(1.0-1.0e-4):
                    seed_record = {**seed_base, "outcome": "refused",
                                   "reason": "seed_not_merit_improving",
                                   "seed_merit": seed_merit}
                else:
                    q, mu, z = seed_q, seed_mu, seed_z
                    seed_record = {**seed_base, "outcome": "admitted",
                                   "seed_merit": seed_merit,
                                   "seed_candidate_evaluations": 1}
                if (initial_contact_seed
                        == "linearized_kkt_merit_globalization_dev"
                        and proposal is not None
                        and not proposal.refused
                        and seed_record is not None
                        and seed_record.get("outcome") == "refused"):
                    # Adopted bounded merit globalization: the complete
                    # algorithmic correction (q, reactions, z) of the
                    # refused full linearized proposal is backtracked as a
                    # disposable initial Newton iterate at the unchanged
                    # requested load/control target. Only an actually
                    # constructed, unrefused finite proposal is backtracked;
                    # an origin-evaluation failure or a proposal refusal
                    # keeps its explicit refusal record untouched. Every
                    # trial is reevaluated for finite/nonnegative supported
                    # forces, the ACTUAL coupled residual/control/NCP merit
                    # and full conservative swept-gap admission from the
                    # same committed origin; only a real strict merit
                    # improvement is admitted. No gap monotonicity is
                    # assumed, no endpoint is bypassed and the complete
                    # correction is scaled as one consistent iterate. If no
                    # trial passes, the explicit refusal stands and the
                    # existing unchanged route proceeds from the origin.
                    origin_q0, origin_mu0, origin_z0 = seed_origin
                    correction_q = seed_q - origin_q0
                    correction_mu = seed_mu - origin_mu0
                    correction_z = np.asarray(seed_z - origin_z0,
                                              dtype=float)
                    correction_z[unsupported] = 0.0
                    bounded_trials = []
                    bounded_admitted = None
                    for bounded_fraction in (0.5, 0.25, 0.125, 0.0625,
                                             0.03125):
                        cancellation_safe_point(
                            cancellation_token,
                            f"contact_path.step:{step}.semismooth."
                            f"initial_seed.bounded:{bounded_fraction}")
                        trial_q = origin_q0 + bounded_fraction*correction_q
                        trial_mu = origin_mu0 + bounded_fraction*correction_mu
                        trial_z = np.asarray(
                            origin_z0 + bounded_fraction*correction_z,
                            dtype=float)
                        trial_z[unsupported] = 0.0
                        trial = {"fraction": float(bounded_fraction)}
                        if not (np.all(np.isfinite(trial_q))
                                and np.all(np.isfinite(trial_mu))
                                and np.all(np.isfinite(trial_z))):
                            trial["outcome"] = "nonfinite_trial_iterate"
                            bounded_trials.append(trial)
                            continue
                        if np.any(trial_z[~unsupported] < 0.0):
                            trial["outcome"] = "negative_trial_pressure_force"
                            bounded_trials.append(trial)
                            continue
                        trial_u = np.asarray(
                            transform @ trial_q + offset).reshape(-1)
                        trial_fraction = manager.admissible_step_fraction(
                            origin_u, trial_u)
                        trial["admissible_fraction"] = float(trial_fraction)
                        if trial_fraction != 1.0:
                            trial["outcome"] = "trial_not_swept_admissible"
                            bounded_trials.append(trial)
                            continue
                        try:
                            trial_candidate = evaluate(trial_q, trial_mu,
                                                      trial_z, tangent=False)
                            trial_merit = float(trial_candidate[-1])
                        except (ValueError, MemoryError) as exc:
                            _discard_nonlinear_state_candidate(
                                committed_states)
                            trial["outcome"] = "trial_evaluation_failed"
                            trial["detail"] = (
                                f"{type(exc).__name__}: {exc}"[:200])
                            bounded_trials.append(trial)
                            continue
                        _discard_nonlinear_state_candidate(committed_states)
                        trial["merit"] = trial_merit
                        if (np.isfinite(trial_merit)
                                and trial_merit
                                < origin_merit*(1.0-1.0e-4)):
                            trial["outcome"] = "admitted"
                            bounded_admitted = (trial_q, trial_mu, trial_z,
                                                trial_merit, trial_candidate)
                            bounded_trials.append(trial)
                            break
                        trial["outcome"] = "trial_not_merit_improving"
                        bounded_trials.append(trial)
                    if bounded_admitted is not None:
                        (trial_q, trial_mu, trial_z, trial_merit,
                         trial_candidate) = bounded_admitted
                        q, mu, z = trial_q, trial_mu, trial_z
                        seed_candidate = trial_candidate
                        seed_record = {**seed_base, "outcome": "admitted",
                                       "seed_merit": trial_merit,
                                       "seed_candidate_evaluations":
                                           1 + len(bounded_trials),
                                       "bounded_globalization": {
                                           "full_proposal_reason":
                                               seed_record["reason"],
                                           "admitted_fraction": float(
                                               bounded_trials[-1]["fraction"]),
                                           "trials": bounded_trials}}
                    else:
                        seed_record = {**seed_record,
                                       "bounded_globalization": {
                                           "outcome":
                                               "no_safe_bounded_trial",
                                           "trials": bounded_trials}}
        if _INITIAL_SEED_DECISION_HOOK is not None:
            decision = {"record": seed_record,
                        "origin_q": seed_origin[0],
                        "origin_reactions": seed_origin[1],
                        "origin_nodal_forces": seed_origin[2]}
            if proposal is not None and not proposal.refused:
                decision["proposed_q"] = seed_q
                decision["proposed_reactions"] = seed_mu
                decision["proposed_nodal_forces"] = seed_z
                decision["origin_normalized_gap"] = (
                    origin_contact.state.normalized_gap)
                decision["origin_residual_n"] = origin_residual
                decision["origin_constraint"] = origin_constraint
                if (seed_record is not None
                        and seed_record.get("outcome") == "admitted"
                        and "bounded_globalization" in seed_record):
                    decision["admitted_q"] = q
                    decision["admitted_reactions"] = mu
                    decision["admitted_nodal_forces"] = z
                if seed_candidate is not None:
                    (_candidate_u, _candidate_trial, candidate_contact,
                     _candidate_k, candidate_residual, candidate_constraint,
                     candidate_ncp, _candidate_active, _candidate_ncp_q,
                     _candidate_ncp_z, candidate_merit) = seed_candidate
                    decision["candidate_normalized_gap"] = (
                        candidate_contact.state.normalized_gap)
                    decision["candidate_residual_n"] = candidate_residual
                    decision["candidate_constraint"] = candidate_constraint
                    decision["candidate_ncp_n"] = candidate_ncp
                    decision["candidate_merit"] = candidate_merit
            _INITIAL_SEED_DECISION_HOOK(decision)

    for iteration in range(1, max_iterations+1):
        cancellation_safe_point(cancellation_token,
                                f"contact_path.step:{step}.semismooth.iteration:{iteration}")
        try:
            (u, trial, contact, structural_k, residual, constraint, ncp,
             active, ncp_q, ncp_z, merit) = evaluate(q, mu, z, tangent=True)
        except (ValueError, MemoryError) as exc:
            _discard_nonlinear_state_candidate(committed_states)
            return SemismoothStep(False, f"contact_semismooth_candidate_failed: {exc}",
                                  q, mu, z, None, None, iteration, np.inf,
                                  np.inf, np.inf, evaluations, backtracks, initial_seed=seed_record)
        residual_norm = float(np.linalg.norm(residual))
        constraint_norm = float(np.max(np.abs(constraint), initial=0.0))
        ncp_force = float(np.sum(np.abs(ncp)))
        ncp_norm = ncp_force/scale
        instantaneous_force_scale = max(
            float(path_force_magnitude),
            float(np.linalg.norm(control_reactions @ mu)),
        )
        instantaneous_ncp_met = (
            ncp_force <= 1.0e-8*instantaneous_force_scale
            if instantaneous_force_scale > 0.0 else ncp_force == 0.0
        )
        unsupported = np.asarray(contact.state.reference_support) <= 0.0
        force_limit = tolerance*max(
            1.0, float(np.linalg.norm(reduced_external
                                      + control_reactions @ mu)),
        )
        if (residual_norm <= force_limit
                and np.all(np.abs(constraint) <= tolerance*control_scales)
                and ncp_norm <= 1.0e-8
                and instantaneous_ncp_met
                and np.all(z[~unsupported] >= 0.0)
                and np.all(np.abs(z[unsupported])
                           <= 64.0*np.finfo(float).eps*scale)):
            positions = manager._positions(u)
            if (manager.point_gap_lower(u) if hasattr(manager,'point_gap_lower')
                    else manager._gap_lower(positions)) >= -manager.gap_tolerance:
                z[unsupported] = 0.0
                return SemismoothStep(True, None, q, mu, z, trial, contact,
                                      iteration, residual_norm, constraint_norm,
                                      ncp_norm, evaluations, backtracks, initial_seed=seed_record)
        kqq, kqz, gap_q = _coupled_contact_blocks(
            structural_k, contact, transform,
        )
        q_size, mu_size = len(q), len(mu)
        block_response = None
        if settings.monolithic_linear_solver == "block_response_dev":
            try:
                block_response = prepare_frozen_block_response(
                    kqq=kqq, control_reactions=control_reactions,
                    control_rows=control_rows, kqz=kqz,
                    gap_q=gap_q,
                    residual=residual, constraint=constraint,
                    signature=f"contact_path:{step}:semismooth:{iteration}:block",
                )
            except Exception as exc:
                _discard_nonlinear_state_candidate(committed_states)
                return SemismoothStep(
                    False, "contact_semismooth_block_response_failed",
                    q, mu, z, None, None, iteration, residual_norm,
                    constraint_norm, ncp_norm, evaluations, backtracks,
                    {"detail": f"{type(exc).__name__}: {exc}"[:200]},
                initial_seed=seed_record,
                )
        pivot_active = active.copy()
        seen_sets = set()
        pivot_trace = []
        current_branch_delta = None
        pivot_limit = min(settings.monolithic_maximum_pivots, modes+2)
        for pivot in range(pivot_limit):
            cancellation_safe_point(cancellation_token,
                                    f"contact_path.step:{step}.semismooth.pivot:{pivot}")
            signature = tuple(np.flatnonzero(pivot_active))
            if signature in seen_sets:
                # The current active branch is a valid generalized Newton
                # derivative even when full-step active-set prediction cycles.
                # Let the existing swept check and residual merit decide its
                # step length instead of rejecting the whole increment.
                if current_branch_delta is None:
                    raise RuntimeError("active-set pivot cycle has no initial direction")
                delta = current_branch_delta
                dq = delta[:q_size]
                dmu = delta[q_size:q_size+mu_size]
                dz = delta[q_size+mu_size:]
                pivot_trace.append({"cycle": list(signature),
                                    "fallback": "current_branch"})
                break
            seen_sets.add(signature)
            try:
                if block_response is None:
                    branch_c = np.where(pivot_active,
                                        c*contact.state.normalized_gap, z)
                    matrix = _assemble_direct_matrix(
                        kqq, kqz, gap_q, control_reactions,
                        control_rows, c, pivot_active,
                    )
                    rhs = np.r_[residual, constraint, -branch_c]
                    delta = np.asarray(factorize(
                        matrix, MatrixClass.GENERAL,
                        signature=f"contact_path:{step}:semismooth:{iteration}:pivot:{pivot}",
                    ).solve(rhs)).reshape(-1)
                else:
                    parts = block_response.direction(
                        pivot_active, z, contact.state.normalized_gap,
                    )
                    delta = np.concatenate(parts)
            except ArithmeticError as exc:
                _discard_nonlinear_state_candidate(committed_states)
                return SemismoothStep(False, "contact_semismooth_block_backward_error",
                                      q, mu, z, None, None, iteration, residual_norm,
                                      constraint_norm, ncp_norm, evaluations, backtracks,
                                      {"pivot_trace": pivot_trace,
                                       "detail": f"{type(exc).__name__}: {exc}"[:200]},
                                      initial_seed=seed_record)
            except Exception as exc:
                _discard_nonlinear_state_candidate(committed_states)
                return SemismoothStep(False, "singular_contact_semismooth_tangent",
                                      q, mu, z, None, None, iteration, residual_norm,
                                      constraint_norm, ncp_norm, evaluations, backtracks,
                                      {"pivot_trace": pivot_trace,
                                       "detail": f"{type(exc).__name__}: {exc}"[:200]},
                                      initial_seed=seed_record)
            if not np.all(np.isfinite(delta)):
                _discard_nonlinear_state_candidate(committed_states)
                return SemismoothStep(False, "nonfinite_contact_semismooth_increment",
                                      q, mu, z, None, None, iteration, residual_norm,
                                      constraint_norm, ncp_norm, evaluations, backtracks,
                                      {"pivot_trace": pivot_trace},
                                      initial_seed=seed_record)
            if current_branch_delta is None:
                current_branch_delta = delta.copy()
            dq = delta[:q_size]
            dmu = delta[q_size:q_size+mu_size]
            dz = delta[q_size+mu_size:]
            predicted_gap = (contact.state.normalized_gap
                             + contact.gap_gradient @ (transform @ dq))
            predicted_z = z+dz
            predicted_active = (
                (contact.state.reference_support > 0.0)
                & (predicted_z-c*predicted_gap > 0.0)
            )
            pivot_trace.append({"from": list(signature),
                                "to": np.flatnonzero(predicted_active).tolist()})
            if np.array_equal(predicted_active, pivot_active):
                break
            pivot_active = predicted_active
        else:
            # A bounded predictor that has not stabilized still leaves the
            # current branch's generalized Newton direction available. The
            # swept certificate and residual merit decide whether to take it.
            if current_branch_delta is None:
                raise RuntimeError("active-set pivot limit has no initial direction")
            delta = current_branch_delta
            dq = delta[:q_size]
            dmu = delta[q_size:q_size+mu_size]
            dz = delta[q_size+mu_size:]
            pivot_trace.append({"limit": pivot_limit,
                                "fallback": "current_branch"})
        predicted_branch_delta = delta
        merit_direction_slopes = None
        if (current_branch_delta is not None
                and not np.array_equal(delta, current_branch_delta)):
            slope_args = dict(q_size=q_size, mu_size=mu_size, kqq=kqq,
                              kqz=kqz, gap_q=gap_q,
                              control_reactions=control_reactions,
                              control_rows=control_rows, residual=residual,
                              constraint=constraint, ncp=ncp, z=z,
                              gap=contact.state.normalized_gap,
                              support=contact.state.reference_support,
                              c=c, scale=scale, control_scales=control_scales)
            predicted_slope = _current_merit_slope(delta=delta, **slope_args)
            current_slope = _current_merit_slope(
                delta=current_branch_delta, **slope_args)
            merit_direction_slopes = {"predicted": predicted_slope,
                                      "current": current_slope}
            if (predicted_slope >= -1.0e-4*merit
                    and current_slope < -1.0e-4*merit):
                delta = current_branch_delta
                dq = delta[:q_size]
                dmu = delta[q_size:q_size+mu_size]
                dz = delta[q_size+mu_size:]
                pivot_trace.append({"fallback": "current_branch_before_line_search",
                                    "predicted_merit_slope": predicted_slope,
                                    "current_merit_slope": current_slope})
        proposed_u = np.asarray(transform @ (q+dq) + offset).reshape(-1)
        fraction = min(1.0, manager.admissible_step_fraction(u, proposed_u))
        swept_initial_fraction = fraction
        _discard_nonlinear_state_candidate(committed_states)
        improved = False
        accepted_fraction = None
        accepted_merit = None
        attempts = []
        last_candidate = None
        last_z = None
        for attempt in range(12):
            cancellation_safe_point(cancellation_token,
                                    f"contact_path.step:{step}.semismooth.backtrack:{attempt}")
            if fraction <= 1.0e-12:
                break
            trial_q = q + fraction*dq
            trial_mu = mu + fraction*dmu
            trial_z = z + fraction*dz
            try:
                candidate = evaluate(trial_q, trial_mu, trial_z, tangent=False)
                last_candidate, last_z = candidate, trial_z
                new_merit = candidate[-1]
                candidate_error = None
                candidate_norms = {
                    "equilibrium": float(np.linalg.norm(candidate[4]))/scale,
                    "constraint": float(np.max(
                        np.abs(candidate[5])/control_scales, initial=0.0)),
                    "complementarity": float(np.sum(np.abs(candidate[6])))/scale,
                    "active_modes": int(np.count_nonzero(candidate[7])),
                }
            except (ValueError, MemoryError) as exc:
                new_merit = np.inf
                candidate_error = f"{type(exc).__name__}: {exc}"[:200]
                candidate_norms = {}
            attempts.append({
                "fraction": float(fraction), "merit": float(new_merit),
                "error": candidate_error, **candidate_norms,
            })
            _discard_nonlinear_state_candidate(committed_states)
            if np.isfinite(new_merit) and new_merit < merit*(1.0-1.0e-4*fraction):
                q, mu, z = trial_q, trial_mu, trial_z
                accepted_fraction = float(fraction)
                accepted_merit = float(new_merit)
                improved = True
                break
            backtracks += 1
            fraction *= 0.5
        original_last_trial_fraction = (attempts[-1]["fraction"]
                                        if attempts else None)
        if (not improved and current_branch_delta is not None
                and not np.array_equal(delta, current_branch_delta)):
            # A predicted active set can give a non-descent direction for the
            # current projected residual. The derivative of the *current*
            # branch is also a valid generalized Newton choice. Try it under
            # the same swept and merit gates before rejecting the increment.
            branch_dq = current_branch_delta[:q_size]
            branch_dmu = current_branch_delta[q_size:q_size+mu_size]
            branch_dz = current_branch_delta[q_size+mu_size:]
            branch_u = np.asarray(
                transform @ (q+branch_dq) + offset,
            ).reshape(-1)
            branch_fraction = min(1.0, manager.admissible_step_fraction(u, branch_u))
            pivot_trace.append({"fallback": "current_branch_after_line_search"})
            for attempt in range(12):
                cancellation_safe_point(
                    cancellation_token,
                    f"contact_path.step:{step}.semismooth.current_branch:{attempt}",
                )
                if branch_fraction <= 1.0e-12:
                    break
                trial_q = q + branch_fraction*branch_dq
                trial_mu = mu + branch_fraction*branch_dmu
                trial_z = z + branch_fraction*branch_dz
                try:
                    candidate = evaluate(trial_q, trial_mu, trial_z, tangent=False)
                    last_candidate, last_z = candidate, trial_z
                    new_merit = candidate[-1]
                    candidate_error = None
                except (ValueError, MemoryError) as exc:
                    new_merit = np.inf
                    candidate_error = f"{type(exc).__name__}: {exc}"[:200]
                attempts.append({
                    "fraction": float(branch_fraction), "merit": float(new_merit),
                    "error": candidate_error, "kind": "current_branch",
                })
                _discard_nonlinear_state_candidate(committed_states)
                if (np.isfinite(new_merit)
                        and new_merit < merit*(1.0-1.0e-4*branch_fraction)):
                    q, mu, z = trial_q, trial_mu, trial_z
                    accepted_fraction = float(branch_fraction)
                    accepted_merit = float(new_merit)
                    improved = True
                    break
                backtracks += 1
                branch_fraction *= 0.5
        if not improved:
            # The projected complementarity map is smooth only up to the
            # first active-set event. A narrow rim mode may cross before the
            # geometric half-step grid samples that smooth branch.
            y = z-c*contact.state.normalized_gap
            dy = dz-c*(contact.gap_gradient @ (transform @ dq))
            crossing = active & (dy < 0.0)
            # A swept certificate can forbid every trial, leaving no
            # backtrack sample from which an active-set event may be taken.
            if original_last_trial_fraction is not None and np.any(crossing):
                event = float(np.min(y[crossing]/(-dy[crossing])))
                # This event uses dq/proposed_u, not a fallback proposal.
                # Its own swept certificate and own last trial are binding.
                event_fraction = _original_direction_event_fraction(
                    event, swept_initial_fraction, original_last_trial_fraction)
                if event_fraction is not None:
                    event_q = q + event_fraction*dq
                    event_mu = mu + event_fraction*dmu
                    event_z = z + event_fraction*dz
                    try:
                        event_candidate = evaluate(event_q, event_mu, event_z,
                                                   tangent=False)
                        event_merit = float(event_candidate[-1])
                        event_error = None
                    except (ValueError, MemoryError) as exc:
                        event_merit = np.inf
                        event_error = f"{type(exc).__name__}: {exc}"[:200]
                    _discard_nonlinear_state_candidate(committed_states)
                    attempts.append({"fraction": event_fraction,
                                     "merit": event_merit,
                                     "error": event_error,
                                     "kind": "before_active_event"})
                    if (np.isfinite(event_merit)
                            and event_merit < merit*(1.0-1.0e-4*event_fraction)):
                        q, mu, z = event_q, event_mu, event_z
                        accepted_fraction = float(event_fraction)
                        accepted_merit = float(event_merit)
                        improved = True
        if improved and stagnation_enabled:
            # Declared stagnation bail: patience consecutive iterations have
            # each accepted a swept admissible fraction at or below the
            # threshold while improving the merit by less than the declared
            # relative rate. This changes no acceptance predicate; the
            # caller decides whether a bounded physical continuation
            # retries the same declared target.
            relative = ((merit-accepted_merit)/merit
                        if merit > 0.0 else 0.0)
            stagnation_trace.append({
                "iteration": iteration,
                "accepted_fraction": (float(accepted_fraction)
                                      if accepted_fraction is not None else None),
                "merit": float(merit),
                "accepted_merit": (float(accepted_merit)
                                   if accepted_merit is not None else None),
                "relative_improvement": float(relative),
            })
            if (accepted_fraction is not None
                    and accepted_fraction <= stagnation_fraction
                    and relative < stagnation_merit_rate):
                stagnation_run += 1
            else:
                stagnation_run = 0
            if stagnation_run >= stagnation_patience:
                return SemismoothStep(
                    False, "contact_semismooth_stagnation",
                    q, mu, z, None, None, iteration, residual_norm,
                    constraint_norm, ncp_norm, evaluations, backtracks,
                    {"iteration": iteration,
                     "consecutive_stagnating_iterations": stagnation_run,
                     "stagnation_patience": stagnation_patience,
                     "stagnation_fraction_threshold": stagnation_fraction,
                     "stagnation_merit_rate": stagnation_merit_rate,
                     "last_accepted_fraction": float(accepted_fraction),
                     "last_merit": float(merit),
                     "last_accepted_merit": float(accepted_merit),
                     "last_relative_improvement": float(relative),
                     "stagnation_trace": stagnation_trace},
                    initial_seed=seed_record)
        if not improved:
            return SemismoothStep(False, "contact_semismooth_backtracking_failed",
                                  q, mu, z, None, None, iteration, residual_norm,
                                  constraint_norm, ncp_norm, evaluations, backtracks,
                                  {"iteration": iteration, "merit": merit,
                                   "equilibrium": residual_norm/scale,
                                   "constraint": float(np.max(
                                       np.abs(constraint)/control_scales,
                                       initial=0.0)),
                                   "complementarity": ncp_norm,
                                   "active_modes": int(np.count_nonzero(active)),
                                   "z": z.tolist(),
                                   "c": c.tolist(),
                                   "support": contact.state.reference_support.tolist(),
                                   "gap": contact.state.normalized_gap.tolist(),
                                   "y": (z-c*contact.state.normalized_gap).tolist(),
                                   "ncp": ncp.tolist(),
                                   "last_trial": ({
                                       "z": last_z.tolist(),
                                       "support": last_candidate[2].state.reference_support.tolist(),
                                       "gap": last_candidate[2].state.normalized_gap.tolist(),
                                       "y": (last_z-c*last_candidate[2].state.normalized_gap).tolist(),
                                   } if last_candidate is not None else None),
                                   "direction_q_norm": float(np.linalg.norm(dq)),
                                   "direction_z_min": float(np.min(dz)),
                                   "direction_z_max": float(np.max(dz)),
                                   "pivot_trace": pivot_trace,
                                   "swept_initial_fraction": swept_initial_fraction,
                                   "swept_start_gap_lower_m": (manager.point_gap_lower(u)
                                       if hasattr(manager,'point_gap_lower') else manager._gap_lower(manager._positions(u))),
                                   **_proposal_gap_diagnostic(manager, proposed_u),
                                   "minimum_sampled_gap_m":
                                   contact.state.minimum_physical_gap,
                                   "overlap_signature":
                                   list(contact.state.overlap_signature),
                                   "attempts": attempts},
                                  initial_seed=seed_record)
    return SemismoothStep(False, "contact_semismooth_iterations_exhausted",
                          q, mu, z, None, None, max_iterations, residual_norm,
                          constraint_norm, ncp_norm, evaluations, backtracks,
                          {"iteration": max_iterations,
                           "equilibrium_residual_n": residual_norm,
                           "constraint_error_m": constraint_norm,
                           "complementarity_relative_force": ncp_norm,
                           "active_modes": int(np.count_nonzero(active)),
                           "nodal_force_n": z.tolist(),
                           "normalized_gap_m": contact.state.normalized_gap.tolist(),
                           "last_line_search": attempts},
                          initial_seed=seed_record)
