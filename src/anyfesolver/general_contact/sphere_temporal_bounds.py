"""Complete B2/B3 beam and translating sphere Hermite gap enclosures.

Three exact tensor Bernstein fields describe beam centreline minus sphere
centre. A control box and the exact squared-distance Bernstein polynomial
certify lower bounds; actual patch corners and centre provide upper bounds
on the minimum. Both trajectories use
their declared endpoint velocities. This bounds the declared interpolation,
not a continuum collision trajectory outside that interpolation.
"""

from __future__ import annotations

from fractions import Fraction
import heapq
import math
from numbers import Integral
from typing import Callable

from .beam_temporal_bounds import (
    BeamHermiteGapBounds,
    BeamHermiteGapBoundsError,
    _cell_controls,
    _finite_scalar,
    _finite_vector,
    _fraction,
    _outward,
    _split,
    _split_curve,
)


_ScalarControls = tuple[tuple[Fraction, ...], ...]
_VectorControls = tuple[_ScalarControls, ...]


def _relative_controls(cell, states, step, sphere) -> _VectorControls:
    zero = Fraction(0)
    lanes = []
    for component in range(3):
        normal = tuple(Fraction(axis == component) for axis in range(3))
        absolute = _cell_controls(cell, states, step, (zero, zero, zero), normal, zero)
        lanes.append(tuple(tuple(value - sphere[component][time]
                                 for time, value in enumerate(row))
                           for row in absolute))
    return tuple(lanes)


def _squared_controls(controls: _VectorControls) -> _ScalarControls:
    """Exact Bernstein product for sum of squares, degree (2/4, 6).

    B_i^n B_j^n = C(n,i) C(n,j) / C(2n,i+j) * B_(i+j)^(2n).
    Apply that identity in both tensor coordinates, preserving correlation
    which the component box cannot retain near a turning point.
    """
    spatial, temporal = len(controls[0])-1, len(controls[0][0])-1
    result = [[Fraction(0) for _ in range(2*temporal+1)]
              for _ in range(2*spatial+1)]
    spatial_weights = tuple(tuple(Fraction(math.comb(spatial, i)*math.comb(spatial, j),
                                           math.comb(2*spatial, i+j))
                                  for j in range(spatial+1)) for i in range(spatial+1))
    temporal_weights = tuple(tuple(Fraction(math.comb(temporal, i)*math.comb(temporal, j),
                                            math.comb(2*temporal, i+j))
                                   for j in range(temporal+1)) for i in range(temporal+1))
    for lane in controls:
        for i, row in enumerate(lane):
            for j, other in enumerate(lane):
                for a, value in enumerate(row):
                    for b, other_value in enumerate(other):
                        result[i+j][a+b] += (
                            spatial_weights[i][j]*temporal_weights[a][b]*value*other_value)
    return tuple(tuple(row) for row in result)


def _squared_bounds(controls: _VectorControls, squared: _ScalarControls) -> tuple[Fraction, Fraction]:
    lower = Fraction(0)
    candidates = [Fraction(0) for _ in range(5)]
    for lane in controls:
        lo = min(value for row in lane for value in row)
        hi = max(value for row in lane for value in row)
        nearest = lo if lo > 0 else -hi if hi < 0 else Fraction(0)
        lower += nearest * nearest
        centre_row = tuple(_split_curve(tuple(column))[0][-1] for column in zip(*lane))
        centre = _split_curve(centre_row)[0][-1]
        for index, value in enumerate((lane[0][0], lane[0][-1],
                                       lane[-1][0], lane[-1][-1], centre)):
            candidates[index] += value * value
    lower = max(lower, min(value for row in squared for value in row), Fraction(0))
    return lower, min(candidates)


def _axis(controls: _VectorControls) -> int:
    space = max(abs(b - a) for lane in controls
                for arow, brow in zip(lane, lane[1:]) for a, b in zip(arow, brow))
    time = max(abs(b - a) for lane in controls for row in lane
               for a, b in zip(row, row[1:]))
    return 0 if space > time else 1


def _children(controls: _VectorControls, squared: _ScalarControls):
    # Compare the certified lower bound of each candidate partition. Position
    # variation alone can select spatial splits repeatedly at a temporal
    # turning point, even when those splits do not improve the minimum bound.
    partitions = []
    for axis in (0, 1):
        pairs = tuple(_split(lane, axis) for lane in controls)
        squared_pair = _split(squared, axis)
        children = []
        for side in (0, 1):
            child = tuple(pair[side] for pair in pairs)
            child_squared = squared_pair[side]
            lower, upper = _squared_bounds(child, child_squared)
            children.append((child, child_squared, lower, upper))
        partitions.append(tuple(children))
    gains = tuple(min(child[2] for child in partition) for partition in partitions)
    axis = (0 if gains[0] > gains[1] else 1 if gains[1] > gains[0]
            else _axis(controls))
    return partitions[axis]


def _sqrt_enclosure(value: Fraction, tolerance: Fraction) -> tuple[Fraction, Fraction]:
    """Dyadic square-root bracket verified with exact squared inequalities.

    Its spacing is at most tolerance/16 and also at least 64 significant bits
    finer than the root scale. No squared distance is converted to float.
    This preserves finite gap answers when squared distances overflow float
    or when subtracting the combined radius causes severe cancellation.
    """
    if value == 0:
        return Fraction(0), Fraction(0)
    root_scale = (value.numerator.bit_length() - value.denominator.bit_length()) // 2
    tolerance_scale = tolerance.numerator.bit_length() - tolerance.denominator.bit_length()
    exponent = min(root_scale - 64, tolerance_scale - 5)
    spacing = Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**(-exponent))
    scaled = value / (spacing * spacing)
    integer = math.isqrt(scaled.numerator // scaled.denominator)
    lo, hi = integer * spacing, (integer + 1) * spacing
    if lo * lo == value:
        hi = lo
    elif hi * hi == value:
        lo = hi
    if not (0 <= lo and lo * lo <= value <= hi * hi):
        raise BeamHermiteGapBoundsError("sphere square-root enclosure failed")
    return lo, hi


def _gap_interval(lower, upper, radius, tolerance) -> tuple[float, float]:
    lower_root = _sqrt_enclosure(lower, tolerance)[0]
    upper_root = _sqrt_enclosure(upper, tolerance)[1]
    lo = _outward(lower_root - radius, -math.inf)
    hi = _outward(upper_root - radius, math.inf)
    # Verify the final outward conversion too, using exact sums and squares.
    lo_distance, hi_distance = _fraction(lo) + radius, _fraction(hi) + radius
    if ((lo_distance > 0 and lo_distance * lo_distance > lower)
            or hi_distance < 0 or hi_distance * hi_distance < upper):
        raise BeamHermiteGapBoundsError("sphere gap float roundoff enclosure failed")
    return lo, hi


def beam_sphere_hermite_gap_bounds(
    manager, start, end, start_velocity, end_velocity, dt,
    *, sphere_start_point, sphere_end_point,
    sphere_start_velocity, sphere_end_velocity,
    tolerance_m: float, maximum_subdivisions: int,
    check_cancel: Callable[[], object] | None = None,
) -> BeamHermiteGapBounds:
    """Enclose the minimum of distance(beam, sphere centre) minus summed radii.

    Beam displacements and velocities use full DOF indexing; sphere points and
    velocities are absolute three-vectors. ``manager.radius`` is the summed
    beam/sphere radius. All cells, eta in [0,1] and the full time fraction in
    [0,1] are retained. Finite input data are interpreted as exact binary64
    values. ``maximum_subdivisions`` limits binary splits across all cells.
    An unresolved enclosure, unrepresentable report or cancellation fails
    explicitly. A cancellation callback may raise or return a truthy value.
    """
    states = tuple(_finite_vector(value, name) for value, name in (
        (start, "start"), (end, "end"),
        (start_velocity, "start_velocity"), (end_velocity, "end_velocity")))
    if not states[0].size or any(value.shape != states[0].shape for value in states[1:]):
        raise ValueError("full displacement and velocity vectors must have equal nonempty shape")
    step = _fraction(_finite_scalar(dt, "dt", positive=True))
    tolerance = _fraction(_finite_scalar(tolerance_m, "tolerance_m", positive=True))
    if (isinstance(maximum_subdivisions, bool)
            or not isinstance(maximum_subdivisions, Integral) or maximum_subdivisions < 0):
        raise ValueError("maximum_subdivisions must be a nonnegative integer")
    if check_cancel is not None and not callable(check_cancel):
        raise ValueError("check_cancel must be callable")
    sphere_states = tuple(_finite_vector(value, name) for value, name in (
        (sphere_start_point, "sphere_start_point"), (sphere_end_point, "sphere_end_point"),
        (sphere_start_velocity, "sphere_start_velocity"),
        (sphere_end_velocity, "sphere_end_velocity")))
    if any(value.shape != (3,) for value in sphere_states):
        raise ValueError("sphere points and velocities must have three components")
    radius_value = _finite_scalar(manager.radius, "combined beam/sphere radius")
    if radius_value < 0:
        raise ValueError("combined beam/sphere radius must be nonnegative")
    radius = _fraction(radius_value)
    sphere = tuple((
        _fraction(sphere_states[0][axis]),
        _fraction(sphere_states[0][axis]) + step * _fraction(sphere_states[2][axis]) / 3,
        _fraction(sphere_states[1][axis]) - step * _fraction(sphere_states[3][axis]) / 3,
        _fraction(sphere_states[1][axis]),
    ) for axis in range(3))

    def cancelled():
        if check_cancel is not None and check_cancel():
            raise BeamHermiteGapBoundsError("beam sphere Hermite gap enclosure cancelled")

    heap, upper, serial = [], None, 0
    for cell in manager.cells:
        cancelled()
        controls = _relative_controls(cell, states, step, sphere)
        squared = _squared_controls(controls)
        lower, candidate = _squared_bounds(controls, squared)
        upper = candidate if upper is None else min(upper, candidate)
        heapq.heappush(heap, (lower, serial, controls, squared))
        serial += 1
    if upper is None:
        raise ValueError("beam sphere gap enclosure requires at least one B2/B3 cell")
    subdivisions = 0
    while True:
        cancelled()
        lower = min(heap[0][0], upper) if heap else upper
        lo, hi = _gap_interval(lower, upper, radius, tolerance)
        if _fraction(hi) - _fraction(lo) <= tolerance:
            return BeamHermiteGapBounds(lo, hi, subdivisions)
        if lower == upper:
            raise BeamHermiteGapBoundsError("requested sphere gap tolerance is below float roundoff enclosure")
        if subdivisions >= maximum_subdivisions:
            raise BeamHermiteGapBoundsError(
                f"beam sphere Hermite gap enclosure unresolved after {subdivisions} subdivisions; "
                f"gap interval [{lo!r}, {hi!r}]")
        _, _, controls, squared = heapq.heappop(heap)
        for child, child_squared, child_lower, candidate in _children(controls, squared):
            upper = min(upper, candidate)
            # Only a proved inability to improve the global minimum removes
            # a patch. Every initial cell contributes its complete support.
            if child_lower < upper:
                heapq.heappush(heap, (child_lower, serial, child, child_squared))
                serial += 1
        subdivisions += 1
