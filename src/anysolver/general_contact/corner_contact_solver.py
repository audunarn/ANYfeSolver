"""Primal-dual active-set NCP Newton core for corner contact (no penalty).

Frictionless corner law per physical sphere corner: g >= 0, lambda >= 0,
lambda * g = 0 with lambda a compressive FORCE in newtons. The algebraic
complementarity map Phi = min(lambda / F_ref, g / L_ref) only scales the NCP;
it is not a physical penalty. Contact geometric stiffness is omitted from the
iteration matrix ONLY; the residual and complementarity are fully re-evaluated
for every trial. Nothing is committed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import os
import time
import json


def _observe(event, **values):
    if os.environ.get("ANYSOLVER_CORNER_CONTACT_TRACE") == "1":
        print("[newton] " + json.dumps(dict(event=event, **values)), flush=True)

from .corner_contact_certificate import CornerContactRefusal

MAX_ITERATIONS = 20
MAX_REDUCTIONS = 8
GAP_TOL_M = 1e-12
NCP_TOL = 1e-8
RELATIVE_FORCE_TOL = 1e-8


class NewtonFailure(RuntimeError):
    def __init__(self, code, message, trace=None):
        self.code = code
        self.trace = trace or []
        super().__init__(message)


class Cancelled(RuntimeError):
    """Raised by a caller supplied cancellation hook; accepted state is untouched."""


@dataclass
class Evaluation:
    residual: np.ndarray        # reduced equilibrium residual (full imbalance pulled back)
    stiffness: np.ndarray       # dense reduced iteration matrix (no contact curvature)
    ids: list                   # candidate corner ids
    gaps: np.ndarray            # signed gaps (m)
    gradients: np.ndarray       # reduced gap gradients, one row per candidate
    guide_error: np.ndarray     # guide row residuals
    guide_rows: np.ndarray      # reduced guide rows
    extra: object = field(default=None)
    # ||reduced external + reduced guide force|| at the evaluated state; when given,
    # the equilibrium criterion is 1e-8 * max(1, force_scale) exactly as the existing
    # semismooth path does (F_ref only scales the algebraic NCP map and the merit).
    force_scale: float | None = field(default=None)


def _lam_array(ev, lam):
    return np.array([lam.get(i, 0.) for i in ev.ids], dtype=float)


def _phi(ev, lam_arr, f_ref, l_ref):
    return np.minimum(lam_arr / f_ref, ev.gaps / l_ref) if len(ev.ids) else np.zeros(0)


def _merit(ev, lam, f_ref, l_ref):
    phi = _phi(ev, _lam_array(ev, lam), f_ref, l_ref)
    return (float(np.sum(ev.residual ** 2)) / f_ref ** 2 + float(np.sum(phi ** 2))
            + float(np.sum(ev.guide_error ** 2)) / l_ref ** 2)


def _affine_guide_progress(ev, trial, alpha):
    """Feasibility-first filter for explicitly declared affine controls only.

    This is a trial-step policy, never an acceptance tolerance. A full Newton
    step solves A*dq=-guide_error, so the trial guide error must equal
    (1-alpha)*guide_error. Once feasible, no trial may leave that manifold.
    """
    if ev.guide_error.size == 0:
        return False, True
    expected = (1.-alpha)*ev.guide_error
    if (trial.guide_error.shape != ev.guide_error.shape
            or not np.all(np.isfinite(trial.guide_error))
            or float(np.max(np.abs(trial.guide_error-expected))) > GAP_TOL_M):
        raise NewtonFailure("nonaffine_guides", "declared affine guide identity failed")
    before = float(np.max(np.abs(ev.guide_error)))
    after = float(np.max(np.abs(trial.guide_error)))
    if before > GAP_TOL_M:
        return after < (1.-1e-4*alpha)*before, True
    return False, after <= GAP_TOL_M


def converged(ev, lam, f_ref, l_ref, force_tolerance):
    lam_arr = _lam_array(ev, lam)
    if ev.force_scale is not None:
        force_tolerance = RELATIVE_FORCE_TOL * max(1.0, float(ev.force_scale))
    if float(np.linalg.norm(ev.residual)) > force_tolerance:
        return False
    if ev.guide_error.size and float(np.max(np.abs(ev.guide_error))) > GAP_TOL_M:
        return False
    if np.any(lam_arr < 0.):
        return False
    for l, g in zip(lam_arr, ev.gaps):
        if l > 0. and abs(g) > GAP_TOL_M:
            return False
        if l == 0. and g < -GAP_TOL_M:
            return False
    phi = _phi(ev, lam_arr, f_ref, l_ref)
    return bool(phi.size == 0 or float(np.max(np.abs(phi))) <= NCP_TOL)


def solve_ncp(problem, q, mu, lam, *, f_ref, l_ref=1.0, force_tolerance,
              max_iterations=MAX_ITERATIONS, max_reductions=MAX_REDUCTIONS, cancel=None):
    """Semismooth Newton on (q, mu, lambda); returns the converged evaluation."""
    q, mu = np.array(q, dtype=float), np.array(mu, dtype=float)
    started = time.perf_counter()
    ev = problem.evaluate(q, mu, lam)
    lam = {i: lam.get(i, 0.) for i in ev.ids}
    trace = []
    for iteration in range(max_iterations + 1):
        if cancel is not None and cancel():
            raise Cancelled("cancelled between Newton iterations")
        _observe("iteration", iteration=iteration, elapsed_s=time.perf_counter()-started,
                 residual_N=float(np.linalg.norm(ev.residual)),
                 minimum_gap_m=float(np.min(ev.gaps)) if ev.gaps.size else None,
                 guide_error_m=float(np.max(np.abs(ev.guide_error))) if ev.guide_error.size else 0.)
        if converged(ev, lam, f_ref, l_ref, force_tolerance):
            return dict(q=q, mu=mu, lam={i: v for i, v in lam.items() if v > 0.}, ev=ev,
                        iterations=iteration, trace=trace)
        if iteration == max_iterations:
            break
        # A line-search residual can be accepted without constructing a tangent.
        # Rebuild the current iteration tangent only when another Newton step is needed.
        if ev.stiffness is None:
            ev = problem.evaluate(q, mu, lam)
        lam_arr = _lam_array(ev, lam)
        active = (lam_arr / f_ref >= ev.gaps / l_ref) if lam_arr.size else np.zeros(0, bool)
        def direction(active):
            a = np.nonzero(active)[0]
            n, k, m = q.size, mu.size, a.size
            matrix = np.zeros((n + k + m, n + k + m))
            # Symmetric algebraic scaling of the multiplier unknowns/rows only (equal
            # solution): unknown multiplier steps are s * y, constraint rows are scaled by s.
            s = max(1.0, float(np.sqrt(np.max(np.abs(ev.stiffness)))))
            matrix[:n, :n] = ev.stiffness
            rhs = np.zeros(n + k + m)
            rhs[:n] = -ev.residual
            # Inactive NCP rows prescribe delta_lambda = -lambda.  Their old
            # force is still present in ev.residual, so include its removal in
            # the equilibrium linearization K*dq - G.T*delta_lambda = -residual.
            if lam_arr.size:
                rhs[:n] -= ev.gradients[~active].T @ lam_arr[~active]
            if k:
                matrix[:n, n:n + k] = -s * ev.guide_rows.T
                matrix[n:n + k, :n] = s * ev.guide_rows
                rhs[n:n + k] = -s * ev.guide_error
            if m:
                matrix[:n, n + k:] = -s * ev.gradients[a].T
                matrix[n + k:, :n] = s * ev.gradients[a]
                rhs[n + k:] = -s * ev.gaps[a]
            linear_start = time.perf_counter()
            try:
                step = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError as error:
                raise NewtonFailure("singular_matrix", str(error), trace) from error
            _observe("linear_solve", iteration=iteration, active=int(m), size=int(n+k+m),
                     seconds=time.perf_counter()-linear_start)
            if not np.all(np.isfinite(step)):
                raise NewtonFailure("nonfinite_step", "nonfinite Newton step", trace)
            step[n:] *= s
            return step, a, n, k, m

        # Solve the linearized unilateral working set before a nonlinear trial.
        # Gap-based admission alone can demand tensile forces from nearly
        # coincident facet-local corners. Such a direction does not solve the
        # linearized complementarity problem and need not reduce the NCP merit.
        # Remove the most tensile constraint, or add the most overclosing
        # inactive constraint, one at a time. Identities/gradients stay intact.
        # This is iteration algebra only: every trial and final gate still uses
        # the original nonlinear gaps, forces and acceptance criteria.
        seen_working_sets = set()
        for working_iteration in range(2 * len(ev.ids) + 1):
            signature = tuple(np.flatnonzero(active))
            if signature in seen_working_sets:
                raise NewtonFailure("working_set_cycle",
                                    "linearized contact working set cycled", trace)
            seen_working_sets.add(signature)
            step, a, n, k, m = direction(active)
            predicted_lam = np.zeros_like(lam_arr)
            predicted_lam[a] = lam_arr[a] + step[n + k:]
            tensile = a[predicted_lam[a] < 0.]
            if tensile.size:
                j = tensile[np.argmin(predicted_lam[tensile])]
                active[j] = False
                _observe("working_set_remove", iteration=iteration,
                         corner_id=int(ev.ids[j]), multiplier_N=float(predicted_lam[j]))
                continue
            predicted = ev.gaps + ev.gradients @ step[:n]
            blockers = np.flatnonzero((~active) & (predicted < -GAP_TOL_M))
            if blockers.size:
                j = blockers[np.argmin(predicted[blockers])]
                active[j] = True
                _observe("working_set_add", iteration=iteration,
                         corner_id=int(ev.ids[j]), predicted_gap_m=float(predicted[j]))
                continue
            break
        else:
            raise NewtonFailure("working_set_limit",
                                "linearized contact working set did not settle", trace)
        delta_lam = -lam_arr.copy()
        delta_lam[a] = step[n + k:]
        merit0 = _merit(ev, lam, f_ref, l_ref)
        alpha, accepted, last = 1.0, False, None
        for _ in range(max_reductions + 1):
            if cancel is not None and cancel():
                raise Cancelled("cancelled during Newton line search")
            # Scale the entire Newton step, including inactive multiplier
            # removal, rather than dropping those forces at every alpha.
            lam_try = {i: float(lam_arr[j] + alpha * delta_lam[j])
                       for j, i in enumerate(ev.ids)}
            try:
                trial_start = time.perf_counter()
                trial = getattr(problem, "evaluate_trial", problem.evaluate)(
                    q + alpha * step[:n], mu + alpha * step[n:n + k], lam_try)
            except CornerContactRefusal as refusal:
                last = refusal
                _observe("trial_refused", iteration=iteration, alpha=alpha,
                         code=refusal.code, seconds=time.perf_counter()-trial_start)
                alpha *= 0.5
                continue
            lam_new = {i: lam_try.get(i, 0.) for i in trial.ids}
            merit_trial = _merit(trial, lam_new, f_ref, l_ref)
            _observe("trial", iteration=iteration, alpha=alpha, merit=merit_trial,
                     merit0=merit0, seconds=time.perf_counter()-trial_start)
            guide_progress, guide_feasible = False, True
            if getattr(problem, "affine_guide_filter", False):
                guide_progress, guide_feasible = _affine_guide_progress(ev,trial,alpha)
            if guide_feasible and np.isfinite(merit_trial) and (guide_progress or
                    merit_trial < (1. - 1e-4 * alpha) * merit0):
                if guide_progress:
                    _observe("guide_filter", iteration=iteration, alpha=alpha,
                             guide_error_m=float(np.max(np.abs(trial.guide_error))))
                q, mu, ev, lam, accepted = (q + alpha * step[:n], mu + alpha * step[n:n + k],
                                            trial, lam_new, True)
                break
            alpha *= 0.5
        trace.append(dict(iteration=iteration, active=int(m), alpha=alpha,
                          residual=float(np.linalg.norm(ev.residual)), merit0=merit0))
        if not accepted:
            if last is not None:
                raise last
            raise NewtonFailure("line_search_failed",
                                f"no merit reduction after {max_reductions} reductions", trace)
    raise NewtonFailure("iteration_limit", f"not converged in {max_iterations} iterations", trace)
