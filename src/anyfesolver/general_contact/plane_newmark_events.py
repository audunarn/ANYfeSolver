"""Development helpers for pressure-mode branch checks and Newmark events.

Inputs are normalized pressure gaps in metres (weighted gaps divided by their
positive pressure support), not physical corner gaps. Active pressure branches
have gap <= 0. The owner must check physical gaps separately.

An arbitrary nonlinear endpoint-duration callback cannot certify its first
root. Localization returns a bracketed proposal from one committed origin;
reequilibration and a complete accepted Hermite-path check remain obligatory.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt, nextafter, inf

import numpy as np

from .event_subdivision import EventNeedsCutback


@dataclass(frozen=True)
class HermiteBranchExtrema:
    minimum_gaps: np.ndarray
    maximum_gaps: np.ndarray


@dataclass(frozen=True)
class BranchPathBounds:
    admissible: bool
    minimum_gaps: np.ndarray
    maximum_gaps: np.ndarray
    oriented_minimum_gaps: np.ndarray


@dataclass(frozen=True)
class PlaneNewmarkEvent:
    duration: float
    support_indices: tuple[int, ...]
    endpoint_evaluations: int


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite and positive')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not np.isfinite(result) or result <= 0.:
        raise ValueError(f'{name} must be finite and positive')
    return result


def _vector(value, name, count=None):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain finite gaps or rates') from exc
    if (result.ndim != 1 or not result.size
            or (count is not None and result.size != count)
            or not np.all(np.isfinite(result))):
        raise ValueError(f'{name} must be a nonempty finite vector of the expected size')
    return result.copy()


def _active(value, count):
    result = np.asarray(value)
    if result.dtype != np.dtype(bool) or result.shape != (count,):
        raise ValueError('active must contain one boolean per pressure mode')
    return result.copy()


def _owned(values):
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _multiply(left, right):
    products = [a*b for a in left for b in right]
    return min(products), max(products)


def _polynomial_interval(coefficients, interval):
    value = (coefficients[-1], coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        lower, upper = _multiply(value, interval)
        value = lower+coefficient, upper+coefficient
    return value


def _sqrt_interval(value):
    # floor(sqrt(n/d)*d*2**96) is exactly isqrt(n*d*2**192).
    # All operations are integer/rational; even a near-double derivative root
    # retains an enclosure without a floating discriminant cancellation.
    scale = value.denominator << 96
    lower = Fraction(isqrt((value.numerator*value.denominator) << 192), scale)
    if lower*lower == value:
        return lower, lower
    return lower, lower+Fraction(1, scale)


def _stationary_intervals(coefficients):
    a, b, c = 3*coefficients[3], 2*coefficients[2], coefficients[1]
    if a == 0:
        return () if b == 0 else ((-c/b, -c/b),)
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return ()
    low, high = _sqrt_interval(discriminant)
    roots = []
    for sign in (-1, 1):
        pair = [(-b+sign*root)/(2*a) for root in (low, high)]
        roots.append((min(pair), max(pair)))
    return tuple(roots)


def _outward(value, direction):
    try:
        result = nextafter(float(value), direction)
    except OverflowError as exc:
        raise ValueError('Hermite gap enclosure cannot be represented finitely') from exc
    if not np.isfinite(result):
        raise ValueError('Hermite gap enclosure cannot be represented finitely')
    return result


def hermite_branch_extrema(start_gaps, end_gaps, start_gap_rates,
                           end_gap_rates, duration):
    """Enclose every cubic's min/max, including endpoints, with outward rounding.

    Binary float input values are exact rationals. Stationary points use exact
    discriminants and rational square-root intervals; interval evaluation and
    one outward float step enclose the represented cubic, including roundoff.
    """
    duration = _positive(duration, 'duration')
    first = _vector(start_gaps, 'start_gaps')
    last, rate0, rate1 = (_vector(value, name, first.size) for value, name in (
        (end_gaps, 'end_gaps'), (start_gap_rates, 'start_gap_rates'),
        (end_gap_rates, 'end_gap_rates')))
    h = Fraction(duration)
    minimum, maximum = [], []
    for g0, g1, v0, v1 in zip(first, last, rate0, rate1):
        g0, g1, v0, v1 = map(Fraction, (g0, g1, v0, v1))
        coefficients = (g0, h*v0, 3*(g1-g0)-h*(2*v0+v1),
                        2*(g0-g1)+h*(v0+v1))
        lower, upper = min(g0, g1), max(g0, g1)
        for a, b in _stationary_intervals(coefficients):
            a, b = max(a, Fraction(0)), min(b, Fraction(1))
            if a <= b:
                low, high = _polynomial_interval(coefficients, (a, b))
                lower, upper = min(lower, low), max(upper, high)
        minimum.append(_outward(lower, -inf))
        maximum.append(_outward(upper, inf))
    return HermiteBranchExtrema(_owned(minimum), _owned(maximum))


def branch_path_admissible(start_gaps, end_gaps, start_gap_rates,
                           end_gap_rates, duration, *, active, gap_tolerance):
    """Check the complete represented cubic against each selected branch sign."""
    tolerance = _positive(gap_tolerance, 'gap_tolerance')
    bounds = hermite_branch_extrema(start_gaps, end_gaps, start_gap_rates,
                                    end_gap_rates, duration)
    active = _active(active, bounds.minimum_gaps.size)
    oriented = _owned(np.where(active, -bounds.maximum_gaps, bounds.minimum_gaps))
    return BranchPathBounds(bool(np.all(oriented >= -tolerance)),
                            bounds.minimum_gaps, bounds.maximum_gaps, oriented)


def localize_endpoint_event(endpoint_gaps, duration, active, required_indices,
                            *, minimum_dt, time_tolerance, gap_tolerance,
                            maximum_evaluations, check_cancel=None, batch_front=False):
    """Propose the earliest of explicitly required, bracketed branch crossings.

    The callback returns the actual equilibrated endpoint's normalized pressure
    gaps at positive duration h, always from ONE committed origin. It must not
    commit a candidate. A finite endpoint bracket is no certificate against
    hidden earlier crossings of a nonlinear duration family. In particular a
    required grazing/hidden excursion with no bracket requests a cutback.

    The opt-in batch front brackets min(oriented required gaps) with shared
    vector evaluations. For single-crossing endpoint families it proposes the
    earliest crossing, independent of row count. Neither mode certifies hidden
    excursions or reentry in an arbitrary nonlinear endpoint family. The caller
    must still reequilibrate and check the complete represented step path.
    """
    duration = _positive(duration, 'duration')
    minimum_dt = _positive(minimum_dt, 'minimum_dt')
    time_tolerance = _positive(time_tolerance, 'time_tolerance')
    gap_tolerance = _positive(gap_tolerance, 'gap_tolerance')
    if (minimum_dt >= duration or time_tolerance >= duration
            or type(maximum_evaluations) is not int or maximum_evaluations < 2
            or not callable(endpoint_gaps) or type(batch_front) is not bool
            or (check_cancel is not None and not callable(check_cancel))):
        raise ValueError('invalid plane Newmark localization controls')
    active = np.asarray(active)
    if active.ndim != 1 or not active.size:
        raise ValueError('active must contain pressure-mode booleans')
    active = _active(active, active.size)
    try:
        indices = tuple(required_indices)
    except TypeError as exc:
        raise ValueError('required_indices must name pressure modes') from exc
    if (not indices or any(type(index) is not int or not 0 <= index < active.size
                           for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('required_indices must be unique valid integer modes')
    indices = tuple(sorted(indices))
    orientation = np.where(active, -1., 1.)
    cache = {}
    evaluations = 0

    def fail(message):
        raise EventNeedsCutback(message, endpoint_evaluations=evaluations)

    def evaluate(h):
        nonlocal evaluations
        if check_cancel is not None:
            check_cancel('plane Newmark endpoint event')
        if h not in cache:
            if evaluations >= maximum_evaluations:
                fail('plane Newmark endpoint evaluation budget exhausted')
            evaluations += 1
            try:
                values = endpoint_gaps(h)
            except EventNeedsCutback as exc:
                raise EventNeedsCutback(str(exc),endpoint_evaluations=evaluations,
                                        diagnostics=exc.diagnostics) from exc
            except ValueError as exc:
                raise EventNeedsCutback('plane Newmark endpoint candidate failed: '+str(exc),
                                        endpoint_evaluations=evaluations) from exc
            try:
                gaps = _vector(values, 'endpoint_gaps', active.size)
            except ValueError as exc:
                raise EventNeedsCutback('invalid equilibrated plane Newmark endpoint gaps',
                                        endpoint_evaluations=evaluations) from exc
            cache[h] = gaps
        return cache[h]

    initial, terminal = evaluate(minimum_dt), evaluate(duration)
    start_sign, end_sign = orientation*initial, orientation*terminal
    if np.any(start_sign < -gap_tolerance):
        fail('a pressure-mode event is below minimum_dt or has an inconsistent initial phase')
    if any(start_sign[index] <= 0. for index in indices):
        fail('required plane event is below minimum_dt or has ambiguous initial phase')
    if any(end_sign[index] >= 0. for index in indices):
        fail('required plane event has no sign bracket; grazing or hidden path needs cutback')
    crossings = set(np.flatnonzero((start_sign > 0.) & (end_sign < 0.)).tolist())
    if not crossings.issubset(indices):
        fail('required_indices omit a bracketed pressure-mode crossing')
    proposals = []
    # A vector-valued equilibrium already supplies every row. A common bracket
    # avoids repeating the same nonlinear solve for distinct crossing fronts.
    searches = (None,) if batch_front else indices
    for index in searches:
        left, right = minimum_dt, duration
        lower = float(np.min(start_sign[list(indices)])) if batch_front else start_sign[index]
        upper = float(np.min(end_sign[list(indices)])) if batch_front else end_sign[index]
        while True:
            proposed = left if abs(lower) < abs(upper) else right
            if right-left <= time_tolerance and min(abs(lower), abs(upper)) <= gap_tolerance:
                proposals.append(proposed)
                break
            midpoint = left+.5*(right-left)
            if not left < midpoint < right:
                fail('plane Newmark event cannot resolve time and gap tolerances')
            values = orientation*evaluate(midpoint)
            value = float(np.min(values[list(indices)])) if batch_front else float(values[index])
            if value > 0.:
                left, lower = midpoint, value
            else:
                right, upper = midpoint, value
    proposed = min(proposals)
    gaps = evaluate(proposed)
    support = tuple(int(index) for index in np.flatnonzero(np.abs(gaps) <= gap_tolerance))
    if not support:
        fail('plane Newmark event proposal has no resolved pressure-mode support')
    return PlaneNewmarkEvent(proposed, support, evaluations)
