"""Bounded branch-endpoint event localization for affine descriptor Newmark.

Endpoint brackets alone do not certify a complete path. The owner must
reequilibrate at the proposed duration and verify its entire lifted interval.
"""
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh, null_space
from numpy.polynomial.polynomial import polyadd, polymul


class EventNeedsCutback(ValueError):
    """No reliable event bracket within the current bounded step."""
    def __init__(self,message,endpoint_evaluations=0,diagnostics=None):
        super().__init__(message)
        self.endpoint_evaluations = endpoint_evaluations
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class LocalizedEvent:
    duration: float
    support_indices: tuple[int, ...]
    endpoint_evaluations: int
    position: np.ndarray


def _root_intervals(coefficients, left, tolerance, maximum_iterations):
    """Bernstein sign variation isolation on a positive duration interval.

    Every coefficient is retained; no small modal stiffness or polynomial tail
    is discarded. Unresolved multiple roots exhaust a finite subdivision budget.
    """
    coefficients = np.trim_zeros(coefficients,'b')
    degree = len(coefficients)-1
    if degree < 1:
        return ()
    coefficients = coefficients/np.max(np.abs(coefficients))
    translated = np.array([sum(coefficients[k]*comb(k,j)*left**(k-j)
                               *(1-left)**j for k in range(j,degree+1))
                           for j in range(degree+1)])
    bernstein = np.array([sum(translated[j]*comb(k,j)/comb(degree,j)
                             for j in range(k+1)) for k in range(degree+1)])
    if not np.all(np.isfinite(bernstein)):
        raise EventNeedsCutback("nonfinite affine event Bernstein coefficients")
    queue = [(left,1.,bernstein)]
    intervals, iterations = [], 0
    while queue:
        a,b,values = queue.pop()
        iterations += 1
        if iterations > maximum_iterations*degree*4:
            raise EventNeedsCutback("affine event root-isolation budget exhausted")
        signs = np.sign(values[values != 0.])
        variations = int(np.count_nonzero(signs[1:] != signs[:-1]))
        if values[0] == 0.:
            intervals.append((a,a))
        if values[-1] == 0.:
            intervals.append((b,b))
        if variations == 0:
            continue
        if variations == 1:
            intervals.append((a,b))
            continue
        if b-a <= tolerance:
            raise EventNeedsCutback("unresolved multiple affine event roots")
        layers = [values]
        for _ in range(degree):
            layers.append(.5*(layers[-1][:-1]+layers[-1][1:]))
        lower = np.array([layer[0] for layer in layers])
        upper = np.array([layer[-1] for layer in layers[::-1]])
        mid = .5*(a+b)
        queue.extend(((mid,b,upper),(a,mid,lower)))
    return tuple(sorted(set(intervals)))


def localize_branch_event(structural, mass, gradient, reference_gap, weights,
                          force_offset, position, velocity, acceleration,
                          start_load, end_load, duration, active,
                          required_indices, *, minimum_dt, time_tolerance,
                          gap_tolerance, maximum_iterations=64, check_cancel=None,
                          null_basis=None):
    """Find the first bracketed support root of the fixed-branch endpoint family.

    Forces are affine over the requested duration. No trial state is mutated.
    A missing bracket, a root below minimum_dt or unresolved event is explicit.
    The caller verifies that no earlier represented crossing was overlooked.
    """
    K, M, B = (np.asarray(v, dtype=float) for v in (structural, mass, gradient))
    g, w, b, q, v, a, f0, f1 = (np.asarray(x, dtype=float) for x in
        (reference_gap, weights, force_offset, position, velocity, acceleration,
         start_load, end_load))
    active = np.asarray(active, dtype=bool)
    n, count = q.size, g.size
    Z = np.zeros((n,0)) if null_basis is None else np.asarray(null_basis,dtype=float)
    if (q.ndim != 1 or K.shape != (n,n) or M.shape != K.shape
            or B.shape != (count,n) or active.shape != (count,)
            or w.shape != (count,) or np.any(w <= 0.)
            or any(x.shape != q.shape for x in (b,v,a,f0,f1))
            or any(not np.all(np.isfinite(x)) for x in (K,M,B,g,w,b,q,v,a,f0,f1))
            or not 0. < minimum_dt < duration
            or not 0. < time_tolerance < duration or gap_tolerance <= 0.
            or not np.isfinite(gap_tolerance)
            or type(maximum_iterations) is not int or maximum_iterations < 1
            or not required_indices or any(type(i) is not int or not 0 <= i < count
                                           for i in required_indices)
            or Z.ndim != 2 or Z.shape[0] != n or not np.all(np.isfinite(Z))):
        raise ValueError("invalid affine event localization data")
    selected = B[active]
    branch = K+selected.T @ (w[active,None]*selected)
    offset = b+selected.T @ (w[active]*g[active])
    orientation = np.where(active, -1., 1.)
    for matrix in (M,branch):
        if np.linalg.norm(matrix-matrix.T) > 512*np.finfo(float).eps*max(1.,np.linalg.norm(matrix)):
            raise ValueError("affine event family requires symmetric work and inertia")
    # Condense only mass-null algebraic work for the endpoint family. Physical
    # inertia is retained. Generalized eigenmodes yield exact rational gaps;
    # their polynomial numerators isolate duration roots before scalar refinement.
    P = null_space(Z.T) if Z.shape[1] else np.eye(n)
    if not 0 < P.shape[1] <= 3:
        raise EventNeedsCutback("affine event physical-coordinate cap exceeded")
    if np.linalg.norm(M @ Z) > 1e-10*max(1.,np.linalg.norm(M)):
        raise ValueError("event null basis does not preserve physical inertia")
    fstart, fslope = f0-offset, (f1-f0)/duration
    effective, gap_map = P.T @ branch @ P, B @ P
    subtraction_scale = np.linalg.norm(branch,2)
    reduced_force, reduced_slope = P.T @ fstart, P.T @ fslope
    gap_constant, gap_slope = g.copy(), np.zeros(count)
    if Z.shape[1]:
        algebraic = Z.T @ branch @ Z
        try:
            np.linalg.cholesky(algebraic)
            cross = np.linalg.solve(algebraic,Z.T @ branch @ P)
            force = np.linalg.solve(algebraic,Z.T @ fstart)
            slope = np.linalg.solve(algebraic,Z.T @ fslope)
        except np.linalg.LinAlgError as exc:
            raise EventNeedsCutback("nonpositive affine event algebraic work") from exc
        effective -= P.T @ branch @ Z @ cross
        subtraction_scale += np.linalg.norm(P.T @ branch @ Z,2)*np.linalg.norm(cross,2)
        reduced_force -= P.T @ branch @ Z @ force
        reduced_slope -= P.T @ branch @ Z @ slope
        gap_map -= B @ Z @ cross
        gap_constant += B @ Z @ force
        gap_slope = B @ Z @ slope
    physical_mass = P.T @ M @ P
    try:
        eigenvalues, modes = eigh(effective, physical_mass)
    except (ValueError,np.linalg.LinAlgError) as exc:
        raise EventNeedsCutback("nonpositive affine event physical mass") from exc
    # A rigid Schur mode is obtained by subtraction of much larger blocks.
    # Bound that arithmetic using the original work and physical mass scales,
    # not the near-zero final eigenvalues. Retain every computed eigenvalue;
    # none is clipped or substituted into the actual endpoint equation.
    spectral_roundoff = (512*np.finfo(float).eps*subtraction_scale
                         /np.min(np.linalg.eigvalsh(physical_mass)))
    if (np.min(eigenvalues) < -spectral_roundoff
            or np.any(1.+.25*duration*duration*eigenvalues <= 0.)):
        raise EventNeedsCutback("affine event endpoint family is not positive")
    denominators = [np.array([1.,0.,.25*duration*duration*value]) for value in eigenvalues]
    numerators = np.stack((modes.T @ physical_mass @ (P.T @ q),
        duration*modes.T @ physical_mass @ (P.T @ v),
        .25*duration**2*(modes.T @ physical_mass @ (P.T @ a)+modes.T @ reduced_force),
        .25*duration**3*modes.T @ reduced_slope),axis=1)
    product = np.array([1.])
    for denominator in denominators:
        product = polymul(product,denominator)
    coefficients = []
    for i in range(count):
        polynomial = polymul([gap_constant[i],duration*gap_slope[i]],product)
        for j,value in enumerate((gap_map @ modes)[i]):
            others = np.array([1.])
            for k,denominator in enumerate(denominators):
                if k != j:
                    others = polymul(others,denominator)
            polynomial = polyadd(polynomial,value*polymul(numerators[j],others))
        if not np.all(np.isfinite(polynomial)):
            raise EventNeedsCutback("nonfinite affine event root polynomial")
        coefficients.append(polynomial)
    calls = 0
    last_endpoint = None

    def gaps(h):
        nonlocal calls, last_endpoint
        if check_cancel is not None:
            check_cancel("transient contact event endpoint evaluation")
        calls += 1
        # Multiplication by beta*h² avoids 1/h² overflow. This owner uses
        # average acceleration beta=1/4, gamma=1/2 exclusively.
        h2 = .25*h*h
        right = M @ (q+h*v+h2*a)+h2*(f0+(h/duration)*(f1-f0)-offset)
        try:
            endpoint = np.linalg.solve(M+h2*branch, right)
        except np.linalg.LinAlgError as exc:
            raise EventNeedsCutback("singular affine event endpoint family",calls) from exc
        value = g+B @ endpoint
        if not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(value)):
            raise EventNeedsCutback("nonfinite affine event endpoint family",calls)
        last_endpoint = endpoint
        return orientation*value

    probe = max(minimum_dt, 64*np.finfo(float).eps*duration)
    low, high = gaps(probe), gaps(duration)
    if np.any(low < -gap_tolerance):
        raise EventNeedsCutback("event is below the minimum resolved duration",calls,
            dict(duration_s=float(duration),probe_s=float(probe),active=active.tolist(),
                 start_oriented_gaps_m=(orientation*(g+B @ q)).tolist(),
                 probe_oriented_gaps_m=low.tolist(),
                 initial_gap_rates_m_per_s=(B @ v).tolist()))
    roots = {}
    for i,polynomial in enumerate(coefficients):
        scale = np.max(np.abs(polynomial))
        if scale == 0.:
            continue  # identically stationary, with no sign departure
        try:
            intervals = _root_intervals(polynomial,probe/duration,
                                        .25*time_tolerance/duration,maximum_iterations)
        except EventNeedsCutback as exc:
            exc.endpoint_evaluations = calls
            raise
        for lower,upper in intervals:
            left, right = duration*lower, duration*upper
            if lower == upper:
                # Exact endpoint/subdivision roots have zero-width isolators.
                # Classify the actual family on each side, never discard them
                # because the two coincident evaluations have identical signs.
                left = max(probe,left-time_tolerance)
                right += time_tolerance
            left_value, right_value = gaps(left)[i], gaps(right)[i]
            if not (left_value > 0. and (right_value <= 0. or
                    (upper == 1. and abs(right_value) <= gap_tolerance))):
                continue  # a grazing root or return to the starting side
            if right_value > 0.:
                roots[i] = duration
                break
            try:
                root = brentq(lambda h: float(gaps(h)[i]), left, right,
                              xtol=min(.25*time_tolerance,4*np.finfo(float).eps*duration),
                              rtol=4*np.finfo(float).eps,
                              maxiter=maximum_iterations)
            except (RuntimeError, ValueError) as exc:
                raise EventNeedsCutback("affine event root localization exhausted",calls) from exc
            if root > duration:
                if root-duration > time_tolerance or abs(high[i]) > gap_tolerance:
                    raise EventNeedsCutback("event lies beyond its boundary budgets",calls)
                root = duration
            roots[i] = float(root)
            break
        if i not in roots and low[i] > 0. and abs(high[i]) <= gap_tolerance:
            # A root at a requested checkpoint boundary can round just outside
            # [0,duration] in the condensed polynomial. Check its actual branch
            # endpoint family on both sides within the explicit time budget.
            # The extended affine load is only a numerical boundary bracket;
            # no state or work beyond the requested duration is committed.
            before = gaps(max(probe,duration-time_tolerance))[i]
            after = gaps(duration+time_tolerance)[i]
            if before > 0. and after < 0.:
                roots[i] = float(duration)
    if any(i not in roots for i in required_indices):
        raise EventNeedsCutback("first represented transition has no isolated endpoint bracket",calls)
    if not roots:
        raise EventNeedsCutback("no supported affine event root",calls)
    first = min(roots.values())
    errors = np.abs(gaps(first))
    supports = tuple(i for i,t in roots.items()
                     if abs(t-first) <= time_tolerance and errors[i] <= gap_tolerance)
    if not supports or any(errors[i] > gap_tolerance
                           for i,t in roots.items() if t == first):
        raise EventNeedsCutback("localized event exceeds its dimensional gap budget",calls)
    position_at_event = last_endpoint.copy()
    position_at_event.flags.writeable = False
    return LocalizedEvent(first, supports, calls, position_at_event)
