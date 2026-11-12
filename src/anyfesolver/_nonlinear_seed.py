"""Private uncommitted accepted-secant equilibrium seeds for coupled DC.

The plane is a temporary initial-guess equation.  This helper never accepts a
load increment or material history; the caller must still solve its original
control equation.  Every global solve is reported against the caller's budget.
"""
import numpy as np


def secant_plane_seed(q, load_factor, previous, evaluate, solve, *,
                      maximum_solves, tolerance, force_scale, domain_error):
    """Find one frozen-history family root at the next accepted-secant plane.

Coordinates use the fixed metric translation/1m, rotation/1rad, lambda/1.
``evaluate`` returns actual ``residual``, ``reference``, ``tangent`` and the
negative proportional ``load_column``; it owns all guards and copied histories.
``solve`` performs one general bordered solve (or returns None on failure).
"""
    origin = np.r_[np.asarray(q, dtype=float), float(load_factor)]
    older = np.r_[np.asarray(previous[0], dtype=float), float(previous[1])]
    info = dict(solves=0, evaluations=0, status='invalid_accepted_secant')
    if (origin.shape != older.shape or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(older)) or maximum_solves <= 0):
        return origin.copy(), info
    secant = origin-older
    span = float(np.linalg.norm(secant))
    if not np.isfinite(span) or span <= 64*np.linalg.norm(np.spacing(origin)):
        return origin.copy(), info
    direction = secant/span
    candidate = origin+secant
    info['span'] = span

    def observe(x, tangent):
        info['evaluations'] += 1
        value = evaluate(x[:-1], float(x[-1]), tangent)
        plane = float(direction@(x-origin)-span)
        residual = np.r_[value['residual']/force_scale, plane]
        return value, plane, residual

    for correction in range(int(maximum_solves)+1):
        try:
            value, plane, residual = observe(candidate, True)
        except domain_error:
            info['status'] = 'seed_domain_refused'
            return origin.copy(), info
        info.update(residual_norm=float(np.linalg.norm(value['residual'])),
                    plane_error=plane, load_factor=float(candidate[-1]))
        if (np.all(np.isfinite(residual))
                and np.linalg.norm(value['residual']) <= tolerance*value['reference']
                and abs(plane) <= tolerance):
            info['status'] = 'equilibrated_uncommitted_seed'
            return candidate.copy(), info
        if correction == maximum_solves:
            info['status'] = 'seed_solve_budget'
            return origin.copy(), info
        matrix = np.vstack((np.column_stack((value['tangent']/force_scale,
            value['load_column']/force_scale)), direction))
        info['solves'] += 1
        delta = solve(matrix, -residual)
        if delta is None or not np.all(np.isfinite(delta)):
            info['status'] = 'seed_linear_solve_failed'
            return origin.copy(), info
        merit = .5*float(residual@residual)
        accepted = None
        for search in range(8):
            alpha = 2.**(-search)
            trial = candidate+alpha*delta
            try:
                _value, _plane, rtrial = observe(trial, False)
            except domain_error:
                continue
            trial_merit = .5*float(rtrial@rtrial)
            if np.isfinite(trial_merit) and trial_merit <= (1.-1e-4*alpha)*merit:
                accepted = trial
                break
        if accepted is None:
            info['status'] = 'seed_merit_no_improvement'
            return origin.copy(), info
        candidate = accepted
    raise AssertionError('bounded secant seed loop exhausted unexpectedly')
