"""Bounded ownership partitions for the existing Q2 union-distance potential.

Every returned interval is covered by a complete distance-separation bound,
a convex shared-endpoint proof, or a uniform bound on the distance-difference
derivative. The last bound proves that there is at most one root; endpoint
signs then enumerate that root completely. Samples alone never certify an
owner or root inventory. Unsupported convexity, tangencies and spent budgets
produce non-success.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt

import numpy as np
from scipy.optimize import brentq

from .beam_curve_pair import (
    _bezier_controls, _curve, _endpoint_projection_cuts, closest_on_b3,
)


@dataclass(frozen=True)
class CurvedOwnershipPartition:
    pieces: tuple[tuple[float, float, int], ...]
    # Root, owner immediately to its left, and owner immediately to its right.
    # station, left/right owners, root uncertainty, slope lower, convexity.
    switches: tuple[tuple[float, int, int, float, float, float], ...]
    visited: int
    # Enclosed junction bands and true integral/root-H bounds in the 30 point chart.
    bands: tuple = ()


def physical_point_map():
    """36 cell-position slots from 30 independent physical length coordinates.

    Each beam has endpoint, midpoint, shared endpoint, midpoint, endpoint.
    Only its material shared endpoint is duplicated; no spatially coincident
    remote points are merged. The map is fixed by native cell topology.
    """
    result = np.zeros((36, 30))
    for row, column in enumerate((0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9)):
        result[3*row:3*row+3, 3*column:3*column+3] = np.eye(3)
    return result


def _quadratic_range(coefficients, left, right):
    constant, linear, quadratic = coefficients
    stations = [left, right]
    if quadratic != 0.0:
        critical = -linear/(2.0*quadratic)
        if left < critical < right:
            stations.append(critical)
    values = [constant+station*(linear+station*quadratic)
              for station in stations]
    return min(values), max(values)


def _convexity_lower(source_hull, target, guard):
    """Bernstein lower bound on d²/dt² |target(t)-source|²."""
    first = -3.0*target[0]+4.0*target[1]-target[2]
    second = 2.0*target[0]-4.0*target[1]+2.0*target[2]
    bounds = []
    for point in source_hull:
        constant = 2.0*float(first@first)+4.0*float((target[0]-point)@second)
        linear = 12.0*float(first@second)
        quadratic = 12.0*float(second@second)
        bounds.extend((constant, constant+0.5*linear,
                       constant+linear+quadratic))
    return min(bounds)-guard


def _shared_endpoint_owner(source, targets, left, right, guard):
    """Under strict squared-distance convexity, an endpoint is a global min.

    Endpoint stationarity is quadratic in source station, so its full range
    includes every critical point. Strict positive margins cover floating
    uncertainty; near-zero endpoint stationarity is not this certificate.
    """
    origin = source[0]
    linear = -3.0*source[0]+4.0*source[1]-source[2]
    quadratic = 2.0*source[0]-4.0*source[1]+2.0*source[2]
    junction = targets[0][2]
    tangent_left = _curve(targets[0], 1.0)[1]
    tangent_right = _curve(targets[1], 0.0)[1]
    for owner, tangent, sign in ((1, tangent_left, -1.0),
                                 (0, tangent_right, 1.0)):
        minimum, _ = _quadratic_range((
            sign*float((junction-origin)@tangent),
            -sign*float(linear@tangent),
            -sign*float(quadratic@tangent),
        ), left, right)
        if minimum >= guard:
            return owner
    return None


def _product_range(first, second):
    values = [a*b for a in first for b in second]
    return min(values), max(values)


def _exact_curve(points, station):
    t = Fraction(float(station))
    rows = tuple(tuple(Fraction(float(value)) for value in row) for row in points)
    first = tuple(-3*a+4*b-c for a, b, c in zip(*rows))
    second = tuple(2*a-4*b+2*c for a, b, c in zip(*rows))
    position = tuple(a+t*(b+t*c) for a, b, c in zip(rows[0], first, second))
    tangent = tuple(b+2*t*c for b, c in zip(first, second))
    return position, tangent


def _exact_smooth_junction(targets):
    """Exact positive tangent collinearity makes the joined curve convex.

    A positive rescaling of each cell parameter makes first derivatives
    continuous at the junction. Strict convexity on both cells then proves
    unique union projection; all owner changes occur at the shared endpoint.
    Spatial coincidence or approximately aligned tangents is insufficient.
    """
    first_point, first = _exact_curve(targets[0], 1.0)
    second_point, second = _exact_curve(targets[1], 0.0)
    if first_point != second_point:
        return False
    cross = (first[1]*second[2]-first[2]*second[1],
             first[2]*second[0]-first[0]*second[2],
             first[0]*second[1]-first[1]*second[0])
    return all(value == 0 for value in cross) and sum(a*b for a, b in zip(first, second)) > 0


def _up(value):
    return float(np.nextafter(float(value), np.inf))


def _down(value):
    return float(np.nextafter(float(value), -np.inf))


def _dot(first, second):
    return sum(a*b for a, b in zip(first, second))


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _sqrt_up(value):
    return _up(sqrt(max(0.0, _up(value))))


def _sqrt_down(value):
    return max(0.0, _down(sqrt(max(0.0, _down(value)))))


def _near_junction_band(search, left, right, convexities, reference, stiffness,
                        budget):
    """Complete near-junction inventory and bounded bulk/root approximation.

    Exact binary64 polynomials define geometry here; no coordinate is snapped.
    A forward tangent cone and positive source crossing make both endpoint
    stationarity polynomials strictly decreasing. Before both zeros the left
    cell wins; after both zeros the right wins. Between them there is either a
    shared-endpoint plateau, or a connected two-interior overlap. In the latter,
    every distinct equal-distance crossing has D_s >= beta*|Delta n| > 0,
    hence exactly one crossing. Strict target convexity excludes extra minima.

    We retain midpoint bulk energy/work/Hessian in that enclosed band and bound
    the entire true bulk integral plus its returned approximation separately.
    All possible moving-root Hessian is also bounded. This is integration error
    accounting, never a claim that a microscopic kink is exactly C1.
    """
    source, targets = search.source, search.targets
    junction, tangent_l = _exact_curve(targets[0], 1.0)
    junction_r, tangent_r = _exact_curve(targets[1], 0.0)
    if junction != junction_r:
        return None
    start = _exact_curve(targets[0], 0.0)[0]
    end = _exact_curve(targets[1], 1.0)[0]
    axis = tuple(b-a for a, b in zip(start, end))
    axis_square = _dot(axis, axis)
    if axis_square <= 0:
        return None
    target_tangents = tuple(_exact_curve(target, t)[1]
                            for target in targets for t in (0.0, 1.0))
    axial = tuple(_dot(axis, tangent) for tangent in target_tangents)
    if min(axial) <= 0:
        return None
    # Ratios for the exact unnormalized axis avoid catastrophic cancellation
    # when the cone is tiny. Positive affine axial components make endpoint
    # cones cover every interior target tangent.
    kappa = _sqrt_up(max(_dot(_cross(axis, tangent), _cross(axis, tangent))/dot**2
                         for tangent, dot in zip(target_tangents, axial)))
    a = _sqrt_down(min(axial)**2/axis_square)
    source_tangents = tuple(_exact_curve(source, t)[1] for t in (left, right))
    source_axial = min(_dot(axis, tangent) for tangent in source_tangents)
    if source_axial <= 0 or a <= 0:
        return None
    source_axial_low = _sqrt_down(source_axial**2/axis_square)
    source_perp = _sqrt_up(max(_dot(_cross(axis, tangent), _cross(axis, tangent))
                               /axis_square for tangent in source_tangents))
    beta = _down(_down(source_axial_low-_up(kappa*source_perp))
                 /_sqrt_up(1.0+_up(kappa*kappa)))
    if beta <= 0:
        return None

    def phi(station, tangent):
        point = _exact_curve(source, station)[0]
        return _dot(tuple(j-p for j, p in zip(junction, point)), tangent)

    brackets = []
    for tangent in (tangent_l, tangent_r):
        lo, hi = left, right
        if not phi(lo, tangent) > 0 > phi(hi, tangent):
            return None  # endpoint/noncontained roots retain the regular proof
        for _ in range(64):
            mid = 0.5*(lo+hi)
            if mid == lo or mid == hi:
                break
            value = phi(mid, tangent)
            if value == 0:
                lo, hi = float(np.nextafter(mid, -np.inf)), float(np.nextafter(mid, np.inf))
                break
            if value > 0:
                lo = mid
            else:
                hi = mid
        else:
            return None
        if not (phi(lo, tangent) >= 0 >= phi(hi, tangent)):
            return None
        brackets.append((lo, hi))
    begin = min(value[0] for value in brackets)
    stop = max(value[1] for value in brackets)
    if not left < begin < stop < right or not 0.0 < begin < stop < 1.0:
        return None
    def bound(begin, stop, remote):
        width = _up(stop-begin)
        middle = 0.5*(begin+stop)
        point = _exact_curve(source, middle)[0]
        distance_j = _sqrt_down(_dot(tuple(p-j for p, j in zip(point, junction)),
                                    tuple(p-j for p, j in zip(point, junction))))
        source_speed = _sqrt_up(max(_dot(t, t) for t in source_tangents))
        reach = _up(source_speed*_up(max(middle-begin, stop-middle)))
        phi_max = (max(Fraction(0), phi(begin, tangent_l)),
                   max(Fraction(0), -phi(stop, tangent_r)))
        tau = tuple(min(1.0, _up(2*value/Fraction(float(m))))
                    for value, m in zip(phi_max, convexities))
        # Endpoint strong-convexity value surplus is quadratic in exact residual;
        # it explicitly lowers the distance floor before normals/Hessian bounds.
        junction_low = _down(distance_j-reach-search.roundoff)
        if junction_low <= search.roundoff:
            return None
        surplus = max(_up(_up(2*value*value/Fraction(float(m)))/junction_low)
                      for value, m in zip(phi_max, convexities))
        distance_low = _down(junction_low-surplus)
        distance_high = _up(distance_j+reach+2.0*search.roundoff)
        if distance_low <= search.roundoff:
            return None
        speeds = tuple(_sqrt_up(max(_dot(_exact_curve(target, t)[1],
                                        _exact_curve(target, t)[1]) for t in (0.0, 1.0)))
                       for target in targets)
        normal_jump = min(2.0, _up(_up(sum(_up(speed*t) for speed, t in zip(speeds, tau)))
                                  /distance_low))
        c = _up(3.0+_up(6.0*distance_high/a))
        measure = _sqrt_up(max(_dot(_exact_curve(reference, t)[1],
                                   _exact_curve(reference, t)[1]) for t in (begin, stop)))
        penetration = max(0.0, _up(search.radius-distance_low))
        h_distance = _up(16.0/distance_low+(6.0+4.0*max(speeds)/distance_low)**2
                         *2.0*distance_high/min(convexities))
        # These are norm bounds in physical length coordinates, before cancellation.
        bulk = np.array((.25*stiffness*measure*penetration**2,
                         2.0*stiffness*measure*penetration,
                         .5*stiffness*measure*(16.0+penetration*h_distance)))*width
        root_h = _up(.5*stiffness*measure*penetration*c*c*normal_jump/beta)
        # Conversion/evaluation guards multiply absolute bounds, not a canceled sum.
        bulk *= 1.0+4096.0*np.finfo(float).eps
        root_h *= 1.0+4096.0*np.finfo(float).eps
        return bulk, root_h if remote else 0.0

    if brackets[0][1] < brackets[1][0]:
        # Exact monotone endpoint inventory proves a genuine shared-map plateau.
        # Only its two zero brackets require approximation; its full interior
        # is integrated normally, irrespective of plateau width.
        spans = ((brackets[0][0], brackets[0][1], False),
                 (brackets[1][0], brackets[1][1], False))
        pieces = ((left, brackets[0][0], 0),
                  (brackets[0][1], brackets[1][0], 0),
                  (brackets[1][1], right, 1))
    else:
        spans = ((begin, stop, True),)
        pieces = ((left, begin, 0), (stop, right, 1))
    bands = []
    estimate = np.zeros(3)
    for first, last, remote in spans:
        result = bound(first, last, remote)
        if result is None:
            return None
        bulk, root_h = result
        estimate += 2.0*bulk+np.array((0.0, 0.0, root_h))
        bands.append((first, last, bulk, root_h))
    if not np.all(np.isfinite(estimate)) or np.any(estimate > budget):
        return None
    return pieces, tuple(bands)


def _stable_difference(source, targets, station, projections, convexities):
    """Factored exact-coordinate distance difference and projection error.

    At a convex interior minimum, the squared-distance value error is
    quadratic in its stationarity residual. Endpoint KKT signs give zero
    value error; otherwise the same strong-convexity bound applies.
    """
    point, _ = _exact_curve(source, station)
    values, errors, target_points = [], [], []
    for target, projection, convexity in zip(targets, projections, convexities):
        t = projection.minimizers[0]
        projected, tangent = _exact_curve(target, t)
        offset = tuple(a-b for a, b in zip(projected, point))
        square = sum(value*value for value in offset)
        distance = sqrt(float(square))
        phi = sum(a*b for a, b in zip(offset, tangent))
        endpoint_kkt = (t == 0.0 and phi >= 0) or (t == 1.0 and phi <= 0)
        value_error = (0.0 if endpoint_kkt else
                       2.0*float(phi)**2/(convexity*max(distance, np.finfo(float).tiny)))
        values.append(distance)
        errors.append(value_error)
        target_points.append(projected)
    numerator = sum((b-a)*(2*p-a-b)
                    for p, a, b in zip(point, *target_points))
    if values[0]+values[1] <= 0.0:
        raise ValueError("coincident curved centrelines cannot define contact distance")
    difference = float(numerator)/(values[0]+values[1])
    error = sum(errors)+8.0*np.finfo(float).eps*abs(difference)
    return difference, error


def moving_boundary_error_bound(source, targets, reference, station,
                               projections, delta_gradient, gap, slope,
                               uncertainty, slope_lower, convexity,
                               stiffness, roundoff, point_map):
    """Bound root placement and float boundary-Hessian evaluation errors.

    Strong convexity bounds projection-station sensitivity. Q2 shape norms
    are <=2 and derivative norms <=6 on [0,1]. These give normal/gradient
    variation bounds and a local interval for c=-k*J*g/(2*D_s). The Hessian
    error is cmax*(2*|V|*dV+dV²)+dc*|V|² for B=c*V*V.T. Misallocated
    integration beside an uncertain root is also bounded, not discarded.
    """
    source_speed = max(float(np.linalg.norm(_curve(source, t)[1])) for t in (0.0, 1.0))
    target_speed = max(float(np.linalg.norm(_curve(target, t)[1]))
                       for target in targets for t in (0.0, 1.0))
    second_source = float(np.linalg.norm(_curve(source, 0.0)[2]))
    reference_speed = float(np.linalg.norm(_curve(reference, station)[1]))
    reference_second = float(np.linalg.norm(_curve(reference, 0.0)[2]))
    point, _ = _exact_curve(source, station)
    station_error = 0.0
    for target, projection in zip(targets, projections):
        t = projection.minimizers[0]
        projected, tangent = _exact_curve(target, t)
        phi = sum((a-b)*v for a, b, v in zip(projected, point, tangent))
        endpoint_kkt = (t == 0.0 and phi >= 0) or (t == 1.0 and phi <= 0)
        if not endpoint_kkt:
            station_error = max(station_error, 2.0*abs(float(phi))/convexity)
    station_speed = 2.0*target_speed*source_speed/convexity
    movement = uncertainty+station_error/max(station_speed, np.finfo(float).tiny)
    _, projection_distance_error = _stable_difference(
        source, targets, station, projections, (convexity, convexity),
    )
    minimum_distance = (min(p.distance for p in projections)-source_speed*movement
                        -projection_distance_error-2.0*roundoff)
    if minimum_distance <= roundoff:
        raise MemoryError("curved boundary distance uncertainty unresolved")
    normal_speed = (source_speed+target_speed*station_speed)/minimum_distance
    gradient_speed = 2.0*(6.0+6.0*station_speed+4.0*normal_speed)
    gradient_error = (gradient_speed*movement
                      +6.0*(8.0*roundoff/minimum_distance+2048.0*np.finfo(float).eps))
    slope_error = ((2.0*normal_speed*source_speed+2.0*second_source)*movement
                   +8.0*roundoff*source_speed/minimum_distance)
    denominator_low = max(slope_lower, slope-slope_error)
    denominator_high = slope+slope_error
    if denominator_low <= 0.0:
        raise MemoryError("curved boundary slope uncertainty unresolved")
    measure_error = reference_second*movement+2.0*roundoff
    gap_error = source_speed*movement+projection_distance_error+2.0*roundoff
    map_norm = np.sqrt(2.0)  # fixed Boolean physical_point_map operator norm
    curve_magnitude = float(np.linalg.norm(delta_gradient))
    # Endpoint rows in the 36 curve coordinates are independent slots, while
    # their material node is one coordinate in the physical 30 length chart.
    # Bound after that fixed map, preserving its cancellation and N/m units.
    magnitude = float(np.linalg.norm(point_map.T @ delta_gradient))
    physical_gradient_error = map_norm*gradient_error
    coefficient = 0.5*stiffness*reference_speed*abs(gap)/slope
    maximum = (0.5*stiffness*(reference_speed+measure_error)
               *(abs(gap)+gap_error)/denominator_low)
    minimum = (0.5*stiffness*max(0.0, reference_speed-measure_error)
               *max(0.0, abs(gap)-gap_error)/denominator_high)
    coefficient_error = max(abs(maximum-coefficient), abs(coefficient-minimum))
    coefficient_error += 2048.0*np.finfo(float).eps*maximum
    boundary_error = (maximum*(2.0*magnitude*physical_gradient_error+physical_gradient_error**2)
                      +coefficient_error*magnitude**2)

    # Fixed-branch Hessian bounds use projection Schur curvature >=m/(2*d).
    distance_high = max(p.distance for p in projections)+source_speed*movement+2.0*roundoff
    distance_hessian = (16.0/minimum_distance
                        +(6.0+4.0*target_speed/minimum_distance)**2
                        *2.0*distance_high/convexity)
    measure = reference_speed+measure_error
    penetration = abs(gap)+gap_error
    energy_error = stiffness*measure*penetration*source_speed*uncertainty**2
    curve_force_error = (0.5*stiffness*measure
                         *(penetration*(curve_magnitude+gradient_error)
                           +8.0*source_speed*uncertainty)*uncertainty)
    force_error = 2.0*curve_force_error
    tangent_error = (4.0*stiffness*measure
                     *(16.0+penetration*distance_hessian)*uncertainty
                     +boundary_error)
    result = np.array((energy_error, force_error, tangent_error))
    if not np.all(np.isfinite(result)):
        raise MemoryError("curved boundary error bound is nonfinite")
    return result


def _normal_box(source_hull, source_point, target, projection, reach, convexity, roundoff,
                derivative_guard):
    """Enclose all normals using strong-convexity projection sensitivity.

    For phi=(r-p).r', phi_t>=convexity/2. Changing p by delta changes
    phi by at most max|r'|*|delta|, including constrained endpoint minima.
    Thus projection station changes by at most 2*max|r'|*reach/convexity.
    """
    speed = max(float(np.linalg.norm(_curve(target, t)[1])) for t in (0.0, 1.0))
    station = projection.minimizers[0]
    projected, tangent = _exact_curve(target, station)
    exact_source = tuple(Fraction(float(value)) for value in source_point)
    phi = sum((a-b)*v for a, b, v in zip(projected, exact_source, tangent))
    endpoint_kkt = (station == 0.0 and phi >= 0) or (station == 1.0 and phi <= 0)
    residual = 0.0 if endpoint_kkt else abs(float(phi))
    uncertainty = 2.0*(speed*(reach+roundoff)+residual+derivative_guard)/convexity
    begin, stop = max(0.0, station-uncertainty), min(1.0, station+uncertainty)
    target_hull = _bezier_controls(target, begin, stop)
    separation_low = np.min(source_hull, axis=0)-np.max(target_hull, axis=0)-roundoff
    separation_high = np.max(source_hull, axis=0)-np.min(target_hull, axis=0)+roundoff
    projection_surplus = (0.0 if endpoint_kkt else
                          2.0*residual**2/(convexity*max(projection.distance, np.finfo(float).tiny)))
    distance_low = projection.distance-reach-roundoff-projection_surplus
    distance_high = projection.distance+reach+roundoff
    if distance_low <= roundoff:
        return None
    values = np.array((separation_low/distance_low,
                       separation_low/distance_high,
                       separation_high/distance_low,
                       separation_high/distance_high))
    return np.min(values, axis=0), np.max(values, axis=0)


def curved_ownership_partition(search, left, right, *, remaining_cells,
                               reference=None, stiffness=None, band_budget=None):
    """Certify all active ownership pieces and isolated simple switches."""
    source, targets = search.source, search.targets
    scale = max(1.0, *(float(np.max(np.abs(row))) for row in (source, *targets)))
    derivative_guard = 2048.0*np.finfo(float).eps*scale*scale
    if search.roundoff > min(search.distance_tolerance, search.feature_tolerance):
        raise MemoryError("curved pair coordinates exceed ownership resolution")
    pending = [(left, right, 0)]
    smooth_junction = _exact_smooth_junction(targets)
    pieces, switches, bands = [], [], []
    visited = 0

    def projections(station):
        point = _curve(source, station)[0]
        result = tuple(closest_on_b3(point, target) for target in targets)
        if any(not projection.unique for projection in result):
            raise ValueError("curved pair has competing minima within a target cell")
        return result

    while pending:
        begin, stop, depth = pending.pop()
        visited += 1
        if visited > remaining_cells:
            raise MemoryError("curved pair ownership search cell budget exhausted")
        midpoint = 0.5*(begin+stop)
        if midpoint == begin or midpoint == stop:
            raise MemoryError("curved pair ownership parameter resolution exhausted")
        point = _curve(source, midpoint)[0]
        hull = _bezier_controls(source, begin, stop)
        reach = max(float(np.linalg.norm(control-point)) for control in hull)
        projected = projections(midpoint)
        delta = projected[0].distance-projected[1].distance
        convexities = tuple(_convexity_lower(hull, target, derivative_guard)
                            for target in targets)
        if min(convexities) > derivative_guard:
            if abs(delta) > 2.0*reach+2.0*search.roundoff:
                pieces.append((begin, stop, 0 if delta < 0.0 else 1))
                continue
            if smooth_junction:
                cuts = (begin, *sorted(set(
                    root for target in targets
                    for root in _endpoint_projection_cuts(source, target)
                    if begin < root < stop
                )), stop)
                for first, last in zip(cuts[:-1], cuts[1:]):
                    if last <= first:
                        raise MemoryError("curved shared-junction partition unresolved")
                    nearest = projections(0.5*(first+last))
                    owner = 0 if nearest[0].distance <= nearest[1].distance else 1
                    pieces.append((first, last, owner))
                continue
            owner = _shared_endpoint_owner(source, targets, begin, stop,
                                            derivative_guard)
            if owner is not None:
                pieces.append((begin, stop, owner))
                continue
            if reference is not None and band_budget is not None:
                near = _near_junction_band(
                    search, begin, stop, convexities, reference, stiffness,
                    band_budget-sum((2.0*band[2]+np.array((0.0, 0.0, band[3]))
                                     for band in bands), start=np.zeros(3)),
                )
                if near is not None:
                    near_pieces, near_bands = near
                    pieces.extend(near_pieces)
                    bands.extend(near_bands)
                    continue
            normals = tuple(_normal_box(
                hull, point, target, projection, reach, convexity, search.roundoff,
                derivative_guard,
            ) for target, projection, convexity in zip(targets, projected, convexities))
            if all(normal is not None for normal in normals):
                first_low, first_high = normals[0]
                second_low, second_high = normals[1]
                normal_low, normal_high = first_low-second_high, first_high-second_low
                tangents = np.array((_curve(source, begin)[1], _curve(source, stop)[1]))
                tangent_low = np.min(tangents, axis=0)-search.roundoff
                tangent_high = np.max(tangents, axis=0)+search.roundoff
                products = tuple(_product_range(
                    (normal_low[axis], normal_high[axis]),
                    (tangent_low[axis], tangent_high[axis]),
                ) for axis in range(3))
                slope_low = sum(value[0] for value in products)-derivative_guard
                slope_high = sum(value[1] for value in products)+derivative_guard
                if slope_low > derivative_guard or slope_high < -derivative_guard:
                    def difference(station):
                        return _stable_difference(source, targets, station,
                                                  projections(station), convexities)[0]

                    first, first_error = _stable_difference(
                        source, targets, begin, projections(begin), convexities,
                    )
                    last, last_error = _stable_difference(
                        source, targets, stop, projections(stop), convexities,
                    )
                    crosses = ((first+first_error < 0.0 and last-last_error > 0.0)
                               or (first-first_error > 0.0 and last+last_error < 0.0))
                    if crosses:
                        root = float(brentq(difference, begin, stop,
                                            xtol=1e-14, rtol=1e-14))
                        residual, value_error = _stable_difference(
                            source, targets, root, projections(root), convexities,
                        )
                        slope_lower = min(abs(slope_low), abs(slope_high))
                        uncertainty = (abs(residual)+value_error)/slope_lower+2e-14
                        if not begin+uncertainty < root < stop-uncertainty:
                            raise MemoryError("curved switch root bracket touches interval endpoint")
                        left_owner, right_owner = (0, 1) if first < 0.0 else (1, 0)
                        pieces.extend(((begin, root, left_owner),
                                       (root, stop, right_owner)))
                        switches.append((root, left_owner, right_owner,
                                         uncertainty, slope_lower, min(convexities)))
                    elif ((first-first_error > 0.0 and last-last_error > 0.0)
                          or (first+first_error < 0.0 and last+last_error < 0.0)):
                        pieces.append((begin, stop, 0 if first < 0.0 else 1))
                    else:
                        raise MemoryError("curved pair switch at interval boundary unresolved")
                    continue

        # Account for the scan's dyadic depth before this extra subdivision.
        initial_depth = max(0, int(np.ceil(-np.log2(right-left))))
        if initial_depth+depth >= search.maximum_depth:
            raise MemoryError(
                "curved pair ownership depth budget exhausted: "
                f"source interval [{begin}, {stop}], scan interval [{left}, {right}], "
                f"depth {initial_depth+depth}, distance difference {delta}, "
                f"reach {reach}, convexity lower {convexities}, "
                f"exact smooth junction {smooth_junction}"
            )
        pending.extend(((begin, midpoint, depth+1), (midpoint, stop, depth+1)))
    return CurvedOwnershipPartition(tuple(sorted(pieces)), tuple(sorted(switches)),
                                    visited, tuple(bands))
