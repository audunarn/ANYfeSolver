"""Exact affine-path certificates for physical B3 endpoint projection.

This developmental sufficient condition requires globally positive F_t for
F(t)=(r(t)-p).r_t(t). It deliberately leaves nonconvex distance profiles
unresolved. Full physical-curve regularity, common time subdivision, topology,
reciprocity and analytic work (b3_distance_jet) remain caller-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
from numbers import Integral

from .beam_curve_pair import CurvePairGapCertificate
from .beam_temporal_bounds import _split
from .capped_cylinder import _restrict, _sqrt_bound
from .sphere_temporal_bounds import _axis, _squared_bounds, _squared_controls


@dataclass(frozen=True)
class EndpointInterval:
    """Fixed closest branch and strict source ownership on the entire path.

    Negative ownership means source-cap ownership; positive means side
    ownership. Open certificates use zero ownership and bounds [0,1].
    Target endpoint indices use B3 nodal order [start,mid,end]: 0 or 2.
    Parameter bounds are exact rational target coordinates in [0,1].
    """
    open: bool
    ownership: int
    target_endpoint: int | None
    parameter_lower: Fraction
    parameter_upper: Fraction


def _rational(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('B3 endpoint geometry must be finite')
    return Fraction.from_float(value)


def _positions(values):
    result = tuple(tuple(_rational(value) for value in row) for row in values)
    if len(result) != 3 or any(len(row) != 3 for row in result):
        raise ValueError('B3 endpoint geometry requires three 3D nodal positions')
    return result


def _curve_controls(positions):
    # Exact Q2 nodal-to-Bernstein conversion; never discard curvature modes.
    return tuple((a, 2*b-(a+c)/2, c) for a, b, c in zip(*positions))


def _outward_float(value, direction):
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError('B3 curve gap bound is not representable')
    result = math.nextafter(result, direction)
    if not math.isfinite(result):
        raise ArithmeticError('B3 outward curve gap bound is not representable')
    return result


def certify_curve_pair_gap(a, b, radius, *, accuracy, maximum_cells,
                           maximum_depth):
    """Bound the complete two-Q2-curve minimum gap with exact tensor controls.

    Relative controls A(s)-B(t) have degree (2,2); squared distance has degree
    (4,4). Every patch lower bound is the maximum of the exact coordinate-box
    lower bound, its squared-distance Bernstein minimum and zero. Actual
    corners and centre supply only an upper bound. Best-first refinement ends
    when all remaining patch lower distances are within the dimensional
    accuracy of the best sampled upper distance. Pruning requires a proved
    inability to improve that sampled minimum, never a sampled lower bound.
    The certificate's minimum_sampled_gap is rounded outward upwards and its
    conservative_lower_gap downwards. Their reported width is <= accuracy.
    Complete curve regularity and scientific acceptance remain caller-owned.
    """
    first, second = map(_curve_controls, map(_positions, (a, b)))
    radius, accuracy = _rational(radius), _rational(accuracy)
    if radius <= 0 or accuracy <= 0:
        raise ValueError('B3 curve gap radius and accuracy must be positive')
    if (type(maximum_cells) is not int or maximum_cells < 1
            or type(maximum_depth) is not int or maximum_depth < 0):
        raise ValueError('B3 curve gap cell/depth budgets are invalid')
    controls = tuple(tuple(tuple(x-y for y in other) for x in lane)
                     for lane, other in zip(first, second))
    squared = _squared_controls(controls)
    lower, upper = _squared_bounds(controls, squared)
    heap = [(lower, 0, 0, controls, squared)]
    checked, serial = 1, 0
    while True:
        lower = min(heap[0][0], upper) if heap else upper
        lower_root, upper_root = _sqrt_bound(lower), _sqrt_bound(upper, True)
        if upper_root-lower_root <= accuracy:
            lo = _outward_float(lower_root-radius, -math.inf)
            hi = _outward_float(upper_root-radius, math.inf)
            if _rational(hi)-_rational(lo) <= accuracy:
                return CurvePairGapCertificate(hi, lo, checked, float(accuracy))
            if not heap or lower == upper:
                raise MemoryError('B3 curve gap accuracy is below report precision')
        if checked+2 > maximum_cells:
            raise MemoryError('B3 exact curve gap cell budget exhausted')
        _, _, depth, controls, squared = heapq.heappop(heap)
        if depth >= maximum_depth:
            raise MemoryError('B3 exact curve gap depth budget exhausted')
        axis = _axis(controls)
        pairs = tuple(_split(lane, axis) for lane in controls)
        squared_pair = _split(squared, axis)
        for side in (0, 1):
            child = tuple(pair[side] for pair in pairs)
            child_squared = squared_pair[side]
            child_lower, candidate = _squared_bounds(child, child_squared)
            checked += 1
            upper = min(upper, candidate)
            # Discarded support cannot lower the actual global minimum.
            if child_lower < upper:
                serial += 1
                heapq.heappush(heap, (child_lower, serial, depth+1,
                                      child, child_squared))


def _values(tensor):
    return tuple(value for row in tensor for value in row)


def _space_derivative(tensor):
    degree = len(tensor)-1
    return tuple(tuple(degree*(b-a) for a, b in zip(first, last))
                 for first, last in zip(tensor, tensor[1:]))


def _dot_tensor(first, second):
    # Polarization reuses the existing exact tensor multiplication engine.
    plus = tuple(tuple(tuple(a+b for a, b in zip(row, other))
                       for row, other in zip(lane, target))
                 for lane, target in zip(first, second))
    minus = tuple(tuple(tuple(a-b for a, b in zip(row, other))
                        for row, other in zip(lane, target))
                  for lane, target in zip(first, second))
    a, b = _squared_controls(plus), _squared_controls(minus)
    return tuple(tuple((x-y)/4 for x, y in zip(row, other))
                 for row, other in zip(a, b))


def _interior_bracket(stationarity):
    """Enclose all temporal roots using strict exact Bernstein sign bounds.

    Each search refines target coordinate only. A midpoint whose time hull
    crosses zero cannot move the certified edge. Twelve steps are a finite
    developmental bound; inability to resolve ownership is returned to caller.
    """
    bounds = []
    for lower in (True, False):
        left, right = Fraction(0), Fraction(1)
        current = stationarity
        certified = left if lower else right
        for _ in range(12):
            first, last = _split(current, 0)
            middle = (left+right)/2
            values = first[-1]
            sign_proved = max(values) < 0 if lower else min(values) > 0
            if lower:
                if sign_proved:
                    certified, left, current = middle, middle, last
                else:
                    right, current = middle, first
            else:
                if sign_proved:
                    certified, right, current = middle, middle, first
                else:
                    left, current = middle, last
        bounds.append(certified)
    return tuple(bounds)


def _restrict_space(vector, left, right):
    # All spatial lanes have degree two. The bracket is dyadic, so _restrict's
    # scalar conversion is exact; geometry controls themselves stay rational.
    return tuple(tuple(zip(*_restrict(tuple(zip(*lane)), left, right)))
                 for lane in vector)


def endpoint_interval(source0, source1, target0, target1, endpoint, radius):
    """Return a whole affine-path certificate, or None when unresolved.

    Distance squared has exact tensor degree (4,2) in target coordinate and
    time. Its first derivative divided by two is F; its second derivative
    divided by two is F_t. Positive controls for F_t prove strict convexity
    over the full target, making strict endpoint signs a unique branch proof.
    Ownership uses (p-r).outward_actual_B3_tangent on time x root bracket.
    No path samples establish admission. Active projection/ownership ties
    fail closed; a globally open distance certificate may bypass such ties.
    """
    if endpoint not in (0, 2):
        raise ValueError('B3 endpoint index must be 0 or 2')
    source0, source1, target0, target1 = map(
        _positions, (source0, source1, target0, target1))
    radius = _rational(radius)
    if radius < 0:
        raise ValueError('combined radius must be nonnegative')
    first, last = _curve_controls(target0), _curve_controls(target1)
    # r(t,time)-p(time), spatial-major tensor controls per Cartesian lane.
    relative = tuple(tuple((a-source0[endpoint][axis],
                            b-source1[endpoint][axis])
                           for a, b in zip(first[axis], last[axis]))
                     for axis in range(3))
    near = []
    for lane in relative:
        low, high = min(_values(lane)), max(_values(lane))
        near.append(low if low > 0 else -high if high < 0 else Fraction(0))
    if sum(value*value for value in near) >= radius*radius:
        return EndpointInterval(True, 0, None, Fraction(0), Fraction(1))
    squared = _squared_controls(relative)
    if min(_values(squared)) >= radius*radius:
        return EndpointInterval(True, 0, None, Fraction(0), Fraction(1))
    stationarity = tuple(tuple(value/2 for value in row)
                         for row in _space_derivative(squared))
    curvature = _space_derivative(stationarity)
    if min(_values(curvature)) <= 0:
        return None
    if min(stationarity[0]) > 0:
        target_endpoint, left, right = 0, Fraction(0), Fraction(0)
    elif max(stationarity[-1]) < 0:
        target_endpoint, left, right = 2, Fraction(1), Fraction(1)
    elif max(stationarity[0]) < 0 and min(stationarity[-1]) > 0:
        target_endpoint = None
        left, right = _interior_bracket(stationarity)
    else:
        return None
    source_first, source_last = map(_curve_controls, (source0, source1))
    direction = -1 if endpoint == 0 else 1
    index = 0 if endpoint == 0 else 1
    outward = tuple((direction*2*(a[index+1]-a[index]),
                      direction*2*(b[index+1]-b[index]))
                     for a, b in zip(source_first, source_last))
    separation = tuple(tuple(tuple(-value for value in row) for row in lane)
                       for lane in _restrict_space(relative, left, right))
    tangent = tuple((lane, lane, lane) for lane in outward)
    ownership = _values(_dot_tensor(separation, tangent))
    if min(ownership) > 0:
        sign = 1
    elif max(ownership) < 0:
        sign = -1
    else:
        return None
    return EndpointInterval(False, sign, target_endpoint, left, right)
