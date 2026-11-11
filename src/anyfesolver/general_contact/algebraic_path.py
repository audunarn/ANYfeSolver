"""Bounded affine elastic descriptor path; not a general collision kernel."""

from dataclasses import dataclass
from itertools import product

import numpy as np


def _roots(coefficient):
    """Real roots of a constant/linear/quadratic polynomial, with stable pairs."""
    c, b, a = coefficient
    scale = max(abs(c), abs(b), abs(a))
    if scale == 0.:
        return ()
    c, b, a = np.asarray(coefficient)/scale
    if a == 0.:
        return () if b == 0. else (-c/b,)
    discriminant = b*b-4*a*c
    if discriminant < 0.:
        return ()
    root = -.5*(b+np.copysign(np.sqrt(discriminant), b))
    return (-b/(2*a),) if root == 0. else (root/a, c/root)


def _extrema(coefficients, left, right):
    at_left = coefficients[0]+left*(coefficients[1]+left*coefficients[2])
    at_right = coefficients[0]+right*(coefficients[1]+right*coefficients[2])
    lower, upper = np.minimum(at_left, at_right), np.maximum(at_left, at_right)
    curved = coefficients[2] != 0.
    vertex = np.zeros_like(coefficients[0])
    with np.errstate(over="ignore", divide="ignore"):
        np.divide(-.5*coefficients[1], coefficients[2], out=vertex, where=curved)
    inside = curved & (vertex > left) & (vertex < right)
    vertex = np.where(inside, vertex, 0.)
    values = coefficients[0]+vertex*(coefficients[1]+vertex*coefficients[2])
    lower = np.where(inside, np.minimum(lower, values), lower)
    upper = np.where(inside, np.maximum(upper, values), upper)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("nonfinite affine descriptor path extrema")
    return lower, upper


@dataclass(frozen=True)
class AlgebraicPathSegment:
    start: float
    end: float
    active: tuple[bool, ...]
    coefficients: np.ndarray

    def position(self, fraction):
        s = float(fraction)
        if not self.start <= s <= self.end:
            raise ValueError("fraction outside algebraic path segment")
        return self.coefficients[0]+s*(self.coefficients[1]+s*self.coefficients[2])


@dataclass(frozen=True)
class AlgebraicContactPath:
    segments: tuple[AlgebraicPathSegment, ...]
    maximum_branch_gap_error: float

    @property
    def transition_fractions(self):
        return tuple(segment.start for segment in self.segments[1:])

    def minimum_gap(self, reference_gap, gap_gradient):
        """Complete quadratic extrema of an independently supplied affine map."""
        base = np.asarray(reference_gap, dtype=float)
        gradient = np.asarray(gap_gradient, dtype=float)
        n = self.segments[0].coefficients.shape[1]
        if (base.ndim != 1 or gradient.shape != (base.size, n) or base.size == 0
                or not np.all(np.isfinite(base)) or not np.all(np.isfinite(gradient))):
            raise ValueError("invalid algebraic path gap map")
        minimum = float("inf")
        for segment in self.segments:
            coefficients = segment.coefficients @ gradient.T
            coefficients[0] += base
            lower, _ = _extrema(coefficients, segment.start, segment.end)
            minimum = min(minimum, float(np.min(lower)))
        return minimum

    def linear_load_work(self, start_force, end_force):
        """Exact work for a force affine in the declared path parameter."""
        start = np.asarray(start_force, dtype=float)
        end = np.asarray(end_force, dtype=float)
        n = self.segments[0].coefficients.shape[1]
        if (start.shape != (n,) or end.shape != (n,)
                or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end))):
            raise ValueError("invalid algebraic path load")
        difference = end-start
        work = 0.
        for segment in self.segments:
            _, linear, quadratic = segment.coefficients
            a, b = segment.start, segment.end
            work += ((b-a)*(linear @ start)
                     +.5*(b*b-a*a)*(linear @ difference+2*quadratic @ start)
                     +(2/3)*(b*b*b-a*a*a)*(quadratic @ difference))
        if not np.isfinite(work):
            raise ValueError("nonfinite algebraic path load work")
        return float(work)


def affine_descriptor_path(structural_tangent, null_basis, physical_coefficients,
                           gap_gradient, reference_gap, weights, *, force=None,
                           force_rate=None, force_curvature=None, maximum_pressure_dofs=3):
    """Lift p(s) through Z.T(Kq-f+B.T w min(gref+Bq,0))=0.

    Requires exact affine structural work and positive definite algebraic work.
    Enumerates a small, explicit pressure-space cap. All gap-polynomial roots
    partition the interval; no sampled gaps stand in for interval minima.
    ``force_rate``/``force_curvature`` add f(s)=force+s*force_rate+s*s*force_curvature;
    omitted, the force is constant and results are unchanged.
    Floating branch/continuity checks use 512 epsilon times arithmetic scales,
    unrelated to the caller's dimensional penetration tolerance.
    """
    K = np.asarray(structural_tangent, dtype=float)
    Z = np.asarray(null_basis, dtype=float)
    p = np.asarray(physical_coefficients, dtype=float)
    B = np.asarray(gap_gradient, dtype=float)
    g = np.asarray(reference_gap, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = p.shape[1] if p.ndim == 2 else 0
    f = np.zeros(n) if force is None else np.asarray(force, dtype=float)
    rate = None if force_rate is None else np.asarray(force_rate, dtype=float)
    curvature = None if force_curvature is None else np.asarray(force_curvature, dtype=float)
    if (p.shape != (3, n) or n == 0 or K.shape != (n, n) or Z.ndim != 2
            or Z.shape[0] != n or Z.shape[1] == 0 or B.ndim != 2 or B.shape[1] != n
            or g.shape != (B.shape[0],) or w.shape != g.shape or f.shape != (n,)
            or any(v is not None and (v.shape != (n,) or not np.all(np.isfinite(v)))
                   for v in (rate, curvature))
            or type(maximum_pressure_dofs) is not int or maximum_pressure_dofs < 1
            or not 0 < g.size <= maximum_pressure_dofs or np.any(w <= 0.)
            or any(not np.all(np.isfinite(v)) for v in (K, Z, p, B, g, w, f))):
        raise ValueError("invalid or over-cap affine descriptor path data")
    eps = 512*np.finfo(float).eps
    try:
        gram = Z.T @ Z
        if not np.all(np.isfinite(gram)):
            raise ValueError("nonfinite affine descriptor path null-basis Gram matrix")
        p = p-(Z @ np.linalg.solve(gram, Z.T @ p.T)).T
        algebraic = Z.T @ K @ Z
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(algebraic)):
            raise ValueError("nonfinite affine descriptor path projection")
        if np.max(np.abs(algebraic-algebraic.T)) > eps*np.max(np.abs(algebraic)):
            raise ValueError("affine descriptor algebraic work is nonsymmetric")
        np.linalg.cholesky(algebraic)
        branches = []
        boundaries = {0., 1.}
        for flags in product((False, True), repeat=g.size):
            active = np.asarray(flags, dtype=bool)
            tangent = K+B[active].T @ (w[active, None]*B[active])
            rhs = tangent @ p.T
            rhs[:, 0] += B[active].T @ (w[active]*g[active])-f
            if rate is not None:
                rhs[:, 1] -= rate
            if curvature is not None:
                rhs[:, 2] -= curvature
            if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(rhs)):
                raise ValueError("nonfinite affine descriptor path branch work")
            block, projected_rhs = Z.T @ tangent @ Z, Z.T @ rhs
            if not np.all(np.isfinite(block)) or not np.all(np.isfinite(projected_rhs)):
                raise ValueError("nonfinite affine descriptor path projected branch work")
            coefficients = p-(Z @ np.linalg.solve(block, projected_rhs)).T
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("nonfinite affine descriptor path branch solution")
            gaps = coefficients @ B.T
            gaps[0] += g
            tolerance = eps*(np.abs(g)+np.abs(B) @ np.sum(np.abs(coefficients), axis=0))
            if not np.all(np.isfinite(gaps)) or not np.all(np.isfinite(tolerance)):
                raise ValueError("nonfinite affine descriptor path gap polynomial")
            for polynomial in gaps.T:
                boundaries.update(float(s) for s in _roots(polynomial) if 0. < s < 1.)
            branches.append((flags, coefficients, gaps, tolerance))
    except np.linalg.LinAlgError as exc:
        raise ValueError("affine descriptor path requires positive definite algebraic work") from exc
    # More active modes win exact ties deterministically; no positive support is removed.
    branches.sort(key=lambda branch: (sum(branch[0]), branch[0]), reverse=True)
    segments = []
    error = 0.
    partition = sorted(boundaries)
    for left, right in zip(partition, partition[1:]):
        for flags, coefficients, gaps, tolerance in branches:
            active = np.asarray(flags, dtype=bool)
            lower, upper = _extrema(gaps, left, right)
            violation = np.where(active, np.maximum(upper, 0.), np.maximum(-lower, 0.))
            if np.all(violation <= tolerance):
                _extrema(coefficients, left, right)  # Reject nonfinite physical coordinates too.
                error = max(error, float(np.max(violation)))
                if segments and segments[-1].active == flags:
                    old = segments.pop()
                    segments.append(AlgebraicPathSegment(old.start, right, flags, old.coefficients))
                else:
                    if segments:
                        previous = segments[-1].position(left)
                        current = coefficients[0]+left*(coefficients[1]+left*coefficients[2])
                        scale = max(float(np.max(np.sum(np.abs(coefficients), axis=0))),
                                    float(np.max(np.sum(np.abs(segments[-1].coefficients), axis=0))))
                        if np.max(np.abs(previous-current)) > eps*scale:
                            raise ValueError("discontinuous affine descriptor path")
                    stored = coefficients.copy()
                    stored.setflags(write=False)
                    segments.append(AlgebraicPathSegment(left, right, flags, stored))
                break
        else:
            raise ValueError("unresolved affine descriptor path interval")
    return AlgebraicContactPath(tuple(segments), error)
