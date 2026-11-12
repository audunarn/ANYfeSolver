"""Internal two-cell native beam pair distributed potential and work map."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import isfinite

import numpy as np
from scipy.optimize import brentq

from anyfesolver._ge_beam3_deterministic_reduction import deterministic_dot, deterministic_norm
from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_p5_seeded.core import sha

from .beam_curve_pair import (
    _GAUSS_8, _GAUSS_16, _bezier_controls, _curve,
    _endpoint_projection_cuts, _source_interval_target_gap_lower,
    b3_distance_jet, closest_on_b3,
)
from .native_curve_union import NativeTwoCellUnionSearch, NativeUnionNearest, resolved_physical_gap_lower
from .native_curved_pair_partition import (
    curved_ownership_partition, moving_boundary_error_bound, physical_point_map,
    _exact_curve, _sqrt_up,
    _sqrt_down, _up, _down, _convexity_lower,
)
from .native_retained_curve_cell import NativeRetainedCurveCell
from .native_retained_station import _readonly


def _joined_owned_pieces(pieces):
    """Join only contiguous pieces with the same proven owner/support status.

    Search leaf boundaries are numerical bookkeeping. The union of their
    complete ownership/unique-projection certificates remains a certificate;
    moving roots change owners and bounded junction bands leave real holes.
    Those boundaries therefore survive. Endpoint projection cuts are applied
    after joining, before quadrature, and its total length budget is preserved.
    """
    joined = []
    for begin, stop, owner, active in pieces:
        if (joined and joined[-1][1] == begin
                and joined[-1][2:] == (owner, active)):
            joined[-1] = (joined[-1][0], stop, owner, active)
        else:
            joined.append((begin, stop, owner, active))
    return joined


def _active_gap_correspondence_upper(source, target, left, right, radius):
    """Complete active-support bound from an admissible affine station map.

    Its endpoint target stations only have to be feasible, not exact closest.
    Source Q2 minus target Q2 composed with this affine map is Q2. Exact
    binary64-coefficient Bernstein controls enclose the full difference curve,
    so nearest distance cannot exceed their maximum norm. Directed conversion
    encloses the exact stored-coordinate geometry before comparing with radius.
    There is no floating curve evaluation in this bound: endpoint stations
    provide only a feasible correspondence, so projection evaluation error
    cannot invalidate it. Existing search/geometry guards remain unchanged.
    """
    stations = tuple(closest_on_b3(_curve(source, s)[0], target).minimizers[0]
                     for s in (left, right))
    if any(not 0.0 <= t <= 1.0 for t in stations):
        raise ValueError("native curved support correspondence station is infeasible")
    first, last = stations
    # Midpoint station arithmetic is exact rational inside _exact_curve only
    # when representable. Instead form target's composed Bernstein controls
    # directly from endpoint positions/tangents; the derivative factor uses
    # the exact difference of the binary64 endpoint stations.
    span_s = Fraction(float(right))-Fraction(float(left))
    span_t = Fraction(float(last))-Fraction(float(first))
    p0, v0 = _exact_curve(source, left)
    p1, _ = _exact_curve(source, right)
    r0, w0 = _exact_curve(target, first)
    r1, _ = _exact_curve(target, last)
    controls = (tuple(p-r for p, r in zip(p0, r0)),
                tuple(p-r+(span_s*v-span_t*w)/2
                      for p, r, v, w in zip(p0, r0, v0, w0)),
                tuple(p-r for p, r in zip(p1, r1)))
    distance_upper = _sqrt_up(max(sum(value*value for value in row) for row in controls))
    return float(np.nextafter(distance_upper-radius, np.inf))


def _private_active_correspondence(source, target, left, right, radius):
    """Complete active-support bound from an admissible affine station map.

    Its endpoint target stations only have to be feasible, not exact closest.
    Source Q2 minus target Q2 composed with this affine map is Q2. Exact
    binary64-coefficient Bernstein controls enclose the full difference curve,
    so nearest distance cannot exceed their maximum norm. Directed conversion
    encloses the exact stored-coordinate geometry before comparing with radius.
    There is no floating curve evaluation in this bound: endpoint stations
    provide only a feasible correspondence, so projection evaluation error
    cannot invalidate it. Existing search/geometry guards remain unchanged.
    """
    stations = tuple(closest_on_b3(_curve(source, s)[0], target).minimizers[0]
                     for s in (left, right))
    if any(not 0.0 <= t <= 1.0 for t in stations):
        raise ValueError("native curved support correspondence station is infeasible")
    first, last = stations
    # Midpoint station arithmetic is exact rational inside _exact_curve only
    # when representable. Instead form target's composed Bernstein controls
    # directly from endpoint positions/tangents; the derivative factor uses
    # the exact difference of the binary64 endpoint stations.
    span_s = Fraction(float(right))-Fraction(float(left))
    span_t = Fraction(float(last))-Fraction(float(first))
    p0, v0 = _exact_curve(source, left)
    p1, _ = _exact_curve(source, right)
    r0, w0 = _exact_curve(target, first)
    r1, _ = _exact_curve(target, last)
    controls = (tuple(p-r for p, r in zip(p0, r0)),
                tuple(p-r+(span_s*v-span_t*w)/2
                      for p, r, v, w in zip(p0, r0, v0, w0)),
                tuple(p-r for p, r in zip(p1, r1)))
    return max(sum(value*value for value in row) for row in controls) < Fraction(float(radius))**2


def _exact_controls(points, left, right):
    first, tangent = _exact_curve(points, left)
    last, _ = _exact_curve(points, right)
    span = Fraction(float(right))-Fraction(float(left))
    return (first, tuple(p+span*v/2 for p, v in zip(first, tangent)), last)


def _endpoint_distance_lower(source, target, station, convexity):
    """Exact stored-coordinate endpoint distance minus projection surplus.

    The rounded projector supplies a feasible station only. Exact source and
    target positions, endpoint KKT or strong convexity, and directed norm/value
    arithmetic certify the closest distance floor without a spatial roundoff
    band. Existing monotonicity/convexity/search guards are unchanged.
    """
    projection = closest_on_b3(_curve(source, station)[0], target)
    if convexity <= 0.0 or not projection.unique:
        return -float('inf')
    point, _ = _exact_curve(source, station)
    t = projection.minimizers[0]
    projected, tangent = _exact_curve(target, t)
    offset = tuple(a-b for a, b in zip(projected, point))
    distance = _sqrt_down(sum(a*a for a in offset))
    phi = sum(a*b for a, b in zip(offset, tangent))
    if (t == 0.0 and phi >= 0) or (t == 1.0 and phi <= 0):
        return distance
    if distance <= 0.0:
        return -float('inf')
    surplus = _up(_up(2*phi*phi/Fraction(float(convexity)))/distance)
    return _down(distance-surplus)


def _private_factored_gap(source, target, station, radius, target_station=None):
    """Exact stored-Q2 value enclosure at a feasible production station.

    For f = 0.5*distance_squared, exact strict convexity bounds surplus
    by phi_squared/m. The corrected squared-distance denominator is used.
    and outward denominator intervals preserve the physical zero surface.
    """
    if target_station is None:
        projection=closest_on_b3(_curve(source,station)[0],target)
        if not projection.unique:raise ValueError('native body competing projection minima')
        target_station=projection.minimizers[0]
    point,_=_exact_curve(source,station);q,v=_exact_curve(target,target_station)
    offset=tuple(x-y for x,y in zip(q,point));square=sum(x*x for x in offset)
    phi=sum(x*y for x,y in zip(offset,v));r0,v0=_exact_curve(target,0.);_,v1=_exact_curve(target,1.)
    a=tuple(y-x for x,y in zip(v0,v1));aa=sum(x*x for x in a);va=sum(x*y for x,y in zip(v0,a))
    t=max(Fraction(0),min(Fraction(1),-va/aa)) if aa else Fraction(0)
    m=(sum(x*x for x in v0)+sum((x-y)*z for x,y,z in zip(r0,point,a))
       +3*va*t+Fraction(3,2)*aa*t*t)
    if m<=0 or square<=0:raise ValueError('native body exact projection convexity unresolved')
    kkt=(target_station==0. and phi>=0) or (target_station==1. and phi<=0)
    surplus=Fraction(0) if kkt else phi*phi/m
    numerator=square-Fraction(float(radius))**2
    dlow=Fraction(_sqrt_down(square))+Fraction(float(radius))
    dhigh=Fraction(_sqrt_up(square))+Fraction(float(radius))
    corrected=square-surplus
    if corrected<=0:
        lower=-float(radius)
    else:
        corrected_lo=Fraction(_sqrt_down(corrected))+Fraction(float(radius))
        corrected_hi=Fraction(_sqrt_up(corrected))+Fraction(float(radius))
        lower=_down(min((numerator-surplus)/corrected_lo,(numerator-surplus)/corrected_hi))
    upper=_up(max(numerator/dlow,numerator/dhigh))
    nominal=float(numerator)/(.5*(float(dlow)+float(dhigh)))
    return lower,upper,nominal


def _private_interval_open(source,target,left,right,radius):
    """Complete exact strong-convex distance floor for a source Q2 hull.

    A feasible midpoint target station is fixed over the interval. For every
    source point, min squared distance >= feasible squared distance minus phi_squared/m.
    Boxes lower the feasible squared norm; exact hull extrema bound phi_squared.
    Convexity m=min_t f_tt is proved for every source-hull vertex (affine in p).
    No rounded distance subtraction or sampled clearance proves opening.
    """
    hull=_exact_controls(source,left,right)
    projection=closest_on_b3(_curve(source,.5*(left+right))[0],target)
    if not projection.unique:return False
    t=projection.minimizers[0];q,v=_exact_curve(target,t)
    r0,v0=_exact_curve(target,0.);_,v1=_exact_curve(target,1.)
    a=tuple(y-x for x,y in zip(v0,v1));aa=sum(x*x for x in a);va=sum(x*y for x,y in zip(v0,a))
    u=max(Fraction(0),min(Fraction(1),-va/aa)) if aa else Fraction(0)
    m=min(sum(x*x for x in v0)+sum((x-y)*z for x,y,z in zip(r0,p,a))
          +3*va*u+Fraction(3,2)*aa*u*u for p in hull)
    if m<=0:return False
    offsets=[tuple(x-y for x,y in zip(q,p)) for p in hull]
    norm_low=Fraction(0)
    for axis in range(3):
        lo=min(row[axis] for row in offsets);hi=max(row[axis] for row in offsets)
        norm_low+=0 if lo<=0<=hi else min(lo*lo,hi*hi)
    phis=[sum(x*y for x,y in zip(row,v)) for row in offsets]
    kkt=(t==0. and min(phis)>=0) or (t==1. and max(phis)<=0)
    surplus=0 if kkt else max(value*value for value in phis)/m
    return norm_low-surplus>=Fraction(float(radius))**2


def _exact_tiny_cell_omission_bound(source, target, reference, radius, stiffness,
                                    left, right):
    """Opt-in absolute strip bounds with exact stored-coordinate geometry.

    Positive norm/product operations are inflated before use; convexity and
    the distance floor use directed subtraction. The midpoint projector only
    supplies a feasible station: its exact stationarity surplus is subtracted.
    No canceled force or tangent is used to justify an omission.
    """
    source_controls = _exact_controls(source, left, right)
    target_controls = _exact_controls(target, 0.0, 1.0)
    square = lambda row: sum(v*v for v in row)
    maximum_separation = _sqrt_up(max(square(tuple(a-b for a, b in zip(p, r)))
                                      for p in source_controls for r in target_controls))
    _, first = _exact_curve(target, 0.0)
    _, last = _exact_curve(target, 1.0)
    delta = tuple(b-a for a, b in zip(first, last))
    delta_square = square(delta)
    minimum_station = (max(Fraction(0), min(Fraction(1),
        -sum(a*b for a, b in zip(first, delta))/delta_square)) if delta_square else Fraction(0))
    minimum_square = square(tuple(a+minimum_station*b for a, b in zip(first, delta)))
    max_speed = _sqrt_up(max(square(first), square(last)))
    convexity = _down(_down(minimum_square)
                      -_up(maximum_separation*_sqrt_up(delta_square)))
    if convexity <= 0.0:
        return None
    midpoint = 0.5*(left+right)
    point, _ = _exact_curve(source, midpoint)
    projection = closest_on_b3(_curve(source, midpoint)[0], target)
    if not projection.unique:
        return None
    t = projection.minimizers[0]
    projected, tangent = _exact_curve(target, t)
    offset = tuple(a-b for a, b in zip(projected, point))
    distance = _sqrt_down(square(offset))
    phi = sum(a*b for a, b in zip(offset, tangent))
    kkt = (t == 0.0 and phi >= 0) or (t == 1.0 and phi <= 0)
    if distance <= 0.0:
        return None
    # F_second >= 2*convexity; distance surplus <= phi_squared/(convexity*d).
    surplus = (0.0 if kkt else
               _up(_up(phi*phi/Fraction(float(convexity)))/distance))
    reach = _sqrt_up(max(square(tuple(a-b for a, b in zip(control, point)))
                         for control in source_controls))
    minimum_distance = _down(_down(distance-surplus)-reach)
    if minimum_distance <= 0.0:
        return None
    if minimum_distance >= radius:
        # Complete distance floor proves the owned nearest branch open. Its
        # potential, work and Hessian are exactly zero on the full interval.
        return np.zeros(3)
    penetration = max(0.0, _up(radius-minimum_distance))
    reference_speed = _sqrt_up(max(square(_exact_curve(reference, t)[1]) for t in (left, right)))
    jacobian_norm = _sqrt_up(Fraction(6))
    mixed_norm = _up(_sqrt_up(Fraction(48))+_up(jacobian_norm*max_speed/minimum_distance))
    hessian_norm = _up(6.0/minimum_distance
                       +_up(mixed_norm*mixed_norm*maximum_separation/convexity))
    measure = _up(stiffness*reference_speed*_up(Fraction(float(right))-Fraction(float(left))))
    bounds = np.array((.5*measure*penetration**2,
                       measure*penetration*jacobian_norm,
                       measure*(6.0+penetration*hessian_norm)))
    bounds *= 1.0+4096.0*np.finfo(float).eps
    return bounds if np.all(np.isfinite(bounds)) else None


class _GeometryGapUnionSearch(NativeTwoCellUnionSearch):
    """Use the union scan without requiring a mechanical projection owner.

    The minimum distance to the complete target union is defined even when
    several material points attain it. The representative fields below only
    satisfy the scan record; no normal, pressure or contact work is selected.
    """

    def nearest(self, source_station: float) -> NativeUnionNearest:
        if (type(source_station) not in (int, float)
                or not isfinite(source_station)
                or not 0.0 <= source_station <= 1.0):
            raise ValueError("native union source station must lie in [0,1]")
        point = _curve(self.source, float(source_station))[0]
        projections = tuple(closest_on_b3(point, target)
                            for target in self.targets)
        distances = tuple(float(projection.distance)
                          for projection in projections)
        representative = 0 if distances[0] <= distances[1] else 1
        station = float(projections[representative].minimizers[0])
        return NativeUnionNearest(
            min(distances)-self.radius, representative, station,
            _readonly(_curve(self.targets[representative], station)[0]),
            distances, False,
        )


class _CurvedOwnershipUnionSearch(_GeometryGapUnionSearch):
    """Remote union switches are handled by the certified integral partition."""

    def nearest(self, source_station: float) -> NativeUnionNearest:
        point = _curve(self.source, source_station)[0]
        if any(not closest_on_b3(point, target).unique for target in self.targets):
            raise ValueError("curved pair has competing minima within a target cell")
        return super().nearest(source_station)


def _straight_join_owner(source: np.ndarray,
                         targets: tuple[np.ndarray, np.ndarray],
                         left: float, right: float,
                         accuracy_m: float) -> int | None:
    """Certify a target owner by the shared-junction Voronoi half-space.

    For collinear, forward-ordered straight cells, a source point on one
    side of the junction is at least as close to that side's cell as to
    the opposite cell. The Bezier hull encloses the whole source interval.
    """
    first, second = targets
    start, junction, end = first[0], first[2], second[2]
    span = end-start
    length = float(np.linalg.norm(span))
    roundoff = 256.0*np.finfo(float).eps*max(
        1.0, float(np.max(np.abs(source))),
        float(np.max(np.abs(first))), float(np.max(np.abs(second))),
    )
    if roundoff > accuracy_m or length <= roundoff:
        return None
    axis = span/length
    if (float(np.dot(junction-start, axis)) <= roundoff
            or float(np.dot(end-junction, axis)) <= roundoff):
        return None
    for cell in targets:
        if (float(np.linalg.norm(cell[1]-0.5*(cell[0]+cell[2]))) > roundoff
                or any(float(np.linalg.norm(
                    point-start-np.dot(point-start, axis)*axis,
                )) > roundoff for point in cell)):
            return None
    projections = [float(np.dot(point-junction, axis))
                   for point in _bezier_controls(source, left, right)]
    if max(projections) <= roundoff and min(projections) < -roundoff:
        return 0
    if min(projections) >= -roundoff and max(projections) > roundoff:
        return 1
    return None


def _straight_join_partitions(source: np.ndarray,
                              targets: tuple[np.ndarray, np.ndarray],
                              left: float, right: float,
                              accuracy_m: float,
                              ) -> tuple[tuple[float, float, int], ...] | None:
    """Split a Q2 source interval at its exact straight-target Voronoi crossings."""
    start, end = targets[0][0], targets[1][2]
    span = end-start
    length = float(np.linalg.norm(span))
    if length == 0.0:
        return None
    axis = span/length
    junction = targets[0][2]
    ordinates = np.array([float(np.dot(point-junction, axis))
                          for point in source])
    quadratic = 2.0*(ordinates[0]-2.0*ordinates[1]+ordinates[2])
    linear = -3.0*ordinates[0]+4.0*ordinates[1]-ordinates[2]
    constant = ordinates[0]
    scale = max(1.0, float(np.max(np.abs(ordinates))))
    roundoff = 256.0*np.finfo(float).eps*scale
    if roundoff > accuracy_m:
        return None
    if abs(quadratic) > roundoff:
        candidates = np.roots((quadratic, linear, constant))
    elif abs(linear) > roundoff:
        candidates = (-constant/linear,)
    else:
        candidates = ()
    boundaries = [left]
    for root in candidates:
        if abs(complex(root).imag) > roundoff:
            continue
        value = float(complex(root).real)
        if left+roundoff < value < right-roundoff:
            boundaries.append(value)
    boundaries.extend((right,))
    boundaries.sort()
    pieces = []
    for begin, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop-begin <= roundoff:
            return None
        owner = _straight_join_owner(source, targets, begin, stop,
                                     accuracy_m)
        if owner is None:
            return None
        pieces.append((begin, stop, owner))
    return tuple(pieces)


def _junction_endpoint_owner(
    source: np.ndarray, targets: tuple[np.ndarray, np.ndarray],
    target_node_ids: tuple[tuple[int, int], tuple[int, int]],
    left: float, right: float, accuracy_m: float,
) -> int | None:
    """Certify ownership beside a material junction with curved target cells.

    If squared distance to the opposite cell increases from its shared
    endpoint throughout that target cell, its nearest point is the junction.
    The adjoining cell contains the same material point and is therefore no
    farther away. Bernstein bounds certify the cubic derivative for every
    source point in the interval's Q2 hull.
    """
    scale = max(1.0, float(np.max(np.abs(source))),
                *(float(np.max(np.abs(target))) for target in targets))
    roundoff = 256.0*np.finfo(float).eps*scale
    derivative_margin = 1024.0*np.finfo(float).eps*scale*scale
    if (roundoff > accuracy_m
            or target_node_ids[0][1] != target_node_ids[1][0]
            or float(np.linalg.norm(targets[0][2]-targets[1][0]))
            > roundoff):
        return None
    hull = _bezier_controls(source, left, right)

    def endpoint_minimum(target: np.ndarray, sign: float) -> bool:
        origin = target[0]
        first = -3.0*target[0]+4.0*target[1]-target[2]
        second = 2.0*target[0]-4.0*target[1]+2.0*target[2]
        for point in hull:
            constant = float((origin-point)@first)
            linear = 2.0*float((origin-point)@second)+float(first@first)
            quadratic = 3.0*float(first@second)
            cubic = 2.0*float(second@second)
            bernstein = (
                constant,
                constant+linear/3.0,
                constant+2.0*linear/3.0+quadratic/3.0,
                constant+linear+quadratic+cubic,
            )
            if min(sign*value for value in bernstein) <= derivative_margin:
                return False
        return True

    if endpoint_minimum(targets[1], 1.0):
        return 0
    if endpoint_minimum(targets[0], -1.0):
        return 1
    return None


def _straight_support_partitions(
    source: np.ndarray, targets: tuple[np.ndarray, np.ndarray],
    radius: float, left: float, right: float, accuracy_m: float,
) -> tuple[tuple[float, float, int], ...] | None:
    """Certify active line intervals for straight finite target geometry.

    The source is affine in station. Its squared distance to either physical
    endpoint or the target line is quadratic. Split first at the axial
    feature changes, then at every real activation root within each feature.
    A returned empty tuple proves this interval open. ``None`` means this
    exact straight profile does not apply and the caller must use its
    conservative general search route.
    """
    start, junction, end = targets[0][0], targets[0][2], targets[1][2]
    span = end-start
    length = float(np.linalg.norm(span))
    roundoff = 256.0*np.finfo(float).eps*max(
        1.0, float(np.max(np.abs(source))),
        *(float(np.max(np.abs(target))) for target in targets),
        radius,
    )
    if roundoff > accuracy_m or length <= roundoff:
        return None
    axis = span/length
    junction_axial = float(np.dot(junction-start, axis))
    if not roundoff < junction_axial < length-roundoff:
        return None
    for cell in targets:
        if (float(np.linalg.norm(cell[1]-0.5*(cell[0]+cell[2]))) > roundoff
                or any(float(np.linalg.norm(
                    point-start-np.dot(point-start, axis)*axis,
                )) > roundoff for point in cell)):
            return None
    if float(np.linalg.norm(source[1]-0.5*(source[0]+source[2]))) > roundoff:
        return None

    origin = source[0]
    direction = source[2]-source[0]
    axial_origin = float(np.dot(origin-start, axis))
    axial_direction = float(np.dot(direction, axis))
    boundaries = [left, right]
    if abs(axial_direction) > roundoff:
        for threshold in (0.0, junction_axial, length):
            root = (threshold-axial_origin)/axial_direction
            if left+roundoff < root < right-roundoff:
                boundaries.append(float(root))
    boundaries.sort()

    def add_roots(begin: float, stop: float) -> list[float]:
        middle = 0.5*(begin+stop)
        axial_middle = axial_origin+axial_direction*middle
        if axial_middle < 0.0:
            relative = origin-start
            slope = direction
        elif axial_middle > length:
            relative = origin-end
            slope = direction
        else:
            relative = origin-start-axial_origin*axis
            slope = direction-axial_direction*axis
        coefficients = (
            float(np.dot(slope, slope)),
            2.0*float(np.dot(relative, slope)),
            float(np.dot(relative, relative))-radius*radius,
        )
        if abs(coefficients[0]) > roundoff*roundoff:
            candidates = np.roots(coefficients)
        elif abs(coefficients[1]) > roundoff*roundoff:
            candidates = (-coefficients[2]/coefficients[1],)
        else:
            candidates = ()
        roots = []
        for candidate in candidates:
            if abs(complex(candidate).imag) > roundoff:
                continue
            root = float(complex(candidate).real)
            if begin+roundoff < root < stop-roundoff:
                roots.append(root)
        return roots

    activation = []
    for begin, stop in zip(boundaries[:-1], boundaries[1:]):
        activation.extend(add_roots(begin, stop))
    boundaries.extend(activation)
    boundaries.sort()
    pieces = []
    for begin, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop-begin <= roundoff:
            return None
        middle = 0.5*(begin+stop)
        point = origin+middle*direction
        axial = axial_origin+middle*axial_direction
        nearest = start+np.clip(axial, 0.0, length)*axis
        difference = float(np.dot(point-nearest, point-nearest))-radius*radius
        if abs(difference) <= roundoff*roundoff:
            return None
        if difference < 0.0:
            owner = _straight_join_owner(source, targets, begin, stop,
                                         accuracy_m)
            if owner is None:
                return None
            pieces.append((begin, stop, owner))
    return tuple(pieces)


def _piecewise_straight_support_partitions(
    source: np.ndarray, targets: tuple[np.ndarray, np.ndarray],
    target_node_ids: tuple[tuple[int, int], tuple[int, int]],
    radius: float, accuracy_m: float,
) -> tuple[tuple[float, float, int], ...] | None:
    """Exact finite-segment support for a straight source and two straight targets.

    The target segments may meet at a kink. Projection clamps first partition
    source station; each resulting squared segment distance is quadratic.
    Radius crossings and equality of the two quadratics complete the
    support/ownership partition. Ambiguous floating-point branches fall back
    to the general search/integration path.
    """
    scale = max(1.0, radius, float(np.max(np.abs(source))),
                *(float(np.max(np.abs(target))) for target in targets))
    roundoff = 256.0*np.finfo(float).eps*scale
    squared_roundoff = 32.0*roundoff*scale
    station_roundoff = 64.0*np.finfo(float).eps
    if roundoff > accuracy_m:
        return None
    if any(float(np.linalg.norm(curve[1]-0.5*(curve[0]+curve[2])))
           > roundoff for curve in (source, *targets)):
        return None
    source_start, direction = source[0], source[2]-source[0]
    segments = tuple((target[0], target[2]-target[0]) for target in targets)
    if any(float(np.dot(vector, vector)) <= roundoff**2
           for _, vector in segments):
        return None
    shared_junction = (
        target_node_ids[0][1] == target_node_ids[1][0]
        and float(np.linalg.norm(targets[0][2]-targets[1][0])) <= roundoff
    )

    def roots(coefficients: tuple[float, float, float],
              left: float, right: float) -> tuple[float, ...] | None:
        a, b, c = coefficients
        if max(abs(a), abs(b), abs(c)) <= squared_roundoff:
            return None
        if a == 0.0:
            candidates = () if b == 0.0 else (-c/b,)
        else:
            discriminant = b*b-4.0*a*c
            discriminant_roundoff = (1024.0*np.finfo(float).eps
                                     *(b*b+abs(4.0*a*c)))
            if abs(discriminant) <= discriminant_roundoff:
                return None
            candidates = np.roots((a, b, c))
        found = []
        for candidate in candidates:
            if abs(complex(candidate).imag) > station_roundoff:
                continue
            value = float(complex(candidate).real)
            if not np.isfinite(value):
                return None
            if left < value < right:
                if abs((a*value+b)*value+c) > squared_roundoff:
                    return None
                found.append(value)
        return tuple(found)

    boundaries = [0.0, 1.0]
    for start, vector in segments:
        squared_length = float(vector@vector)
        first = float((source_start-start)@vector)/squared_length
        slope = float(direction@vector)/squared_length
        if abs(slope) <= station_roundoff:
            if min(abs(first), abs(first-1.0)) <= station_roundoff:
                return None
            continue
        for level in (0.0, 1.0):
            root = (level-first)/slope
            if 0.0 < root < 1.0:
                boundaries.append(float(root))
    boundaries = sorted(set(boundaries))
    if any(right-left <= station_roundoff
           for left, right in zip(boundaries[:-1], boundaries[1:])):
        return None

    def quadratics(left: float, right: float):
        middle = 0.5*(left+right)
        pieces = []
        branches = []
        for start, vector in segments:
            squared_length = float(vector@vector)
            first = float((source_start-start)@vector)/squared_length
            slope = float(direction@vector)/squared_length
            projected = first+slope*middle
            if min(abs(projected), abs(projected-1.0)) <= station_roundoff:
                return None
            if projected < 0.0:
                clamp, variation = 0.0, 0.0
                branch = "start"
            elif projected > 1.0:
                clamp, variation = 1.0, 0.0
                branch = "end"
            else:
                clamp, variation = first, slope
                branch = "interior"
            offset = source_start-start-clamp*vector
            velocity = direction-variation*vector
            pieces.append((float(velocity@velocity),
                           2.0*float(offset@velocity),
                           float(offset@offset)))
            branches.append(branch)
        return tuple(pieces), tuple(branches)

    active = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        calculated = quadratics(left, right)
        if calculated is None:
            return None
        coefficients, branches = calculated
        first, second = coefficients
        junction_owner = shared_junction and branches == ("end", "start")
        cuts = [left, right]
        for index, polynomial in enumerate((
            (first[0], first[1], first[2]-radius*radius),
            (second[0], second[1], second[2]-radius*radius),
            tuple(a-b for a, b in zip(first, second)),
        )):
            if (index == 2 and junction_owner
                    and max(abs(value) for value in polynomial)
                    <= squared_roundoff):
                continue
            found = roots(polynomial, left, right)
            if found is None:
                return None
            cuts.extend(found)
        cuts = sorted(set(cuts))
        if any(stop-begin <= station_roundoff
               for begin, stop in zip(cuts[:-1], cuts[1:])):
            return None
        for begin, stop in zip(cuts[:-1], cuts[1:]):
            middle = 0.5*(begin+stop)
            distances = tuple((a*middle+b)*middle+c
                              for a, b, c in coefficients)
            nearest = min(distances)
            if abs(nearest-radius*radius) <= squared_roundoff:
                return None
            if nearest >= radius*radius:
                continue
            if (abs(distances[0]-distances[1]) <= squared_roundoff
                    and not junction_owner):
                return None
            active.append((begin, stop,
                           0 if junction_owner or distances[0] < distances[1]
                           else 1))
    return tuple(active)


def _monotone_gap_endpoints(
    source: np.ndarray, target: np.ndarray,
    left: float, right: float, roundoff_m: float,
) -> tuple[float, float] | None:
    """Certify one smooth closest-distance branch and a gap derivative sign.

    Strict convexity of squared distance in target station gives unique
    projection. A bound on the source-distance second derivative then proves
    that its first derivative cannot change sign over this source interval.
    Returned values are endpoint centreline distances, before subtracting
    the declared round radius.
    """
    source_controls = _bezier_controls(source, left, right)
    target_controls = _bezier_controls(target, 0.0, 1.0)
    maximum_separation = max(
        float(deterministic_norm(a-b))
        for a in source_controls for b in target_controls
    )
    target_first = _curve(target, 0.0)[1]
    target_last = _curve(target, 1.0)[1]
    target_delta = target_last-target_first
    target_delta_square = deterministic_dot(target_delta, target_delta)
    minimum_station = (float(np.clip(-deterministic_dot(target_first, target_delta)
                                     /target_delta_square, 0.0, 1.0))
                       if target_delta_square > 0.0 else 0.0)
    target_min_speed = float(deterministic_norm(
        target_first+minimum_station*target_delta,
    ))
    target_max_speed = max(deterministic_norm(target_first),
                           deterministic_norm(target_last))
    convexity = (target_min_speed**2
                 - maximum_separation*deterministic_norm(target_delta))
    if convexity <= 0.0:
        return None
    middle = 0.5*(left+right)
    source_point, source_speed, source_second = _curve(source, middle)
    projection = closest_on_b3(source_point, target)
    if not projection.unique:
        return None
    target_point = _curve(target, projection.minimizers[0])[0]
    normal_vector = source_point-target_point
    distance = deterministic_norm(normal_vector)
    reach = max(deterministic_norm(control-source_point)
                for control in source_controls)
    minimum_distance = distance-reach-roundoff_m
    if minimum_distance <= 0.0:
        return None
    source_max_speed = max(deterministic_norm(_curve(source, station)[1])
                           for station in (left, right))
    projection_hessian_bound = (
        1.0/minimum_distance
        + (target_max_speed/minimum_distance)**2
        * maximum_separation/convexity
    )
    second_bound = (source_max_speed**2*projection_hessian_bound
                    + deterministic_norm(source_second))
    derivative = deterministic_dot(normal_vector/distance, source_speed)
    derivative_roundoff = 256.0*np.finfo(float).eps*max(
        1.0, source_max_speed, target_max_speed,
    )
    if abs(derivative) <= second_bound*0.5*(right-left)+derivative_roundoff:
        return None
    return tuple(
        float(closest_on_b3(_curve(source, station)[0], target).distance)
        for station in (left, right)
    )


def _tiny_cell_omission_bound(
    source: np.ndarray, target: np.ndarray, reference: np.ndarray,
    radius: float, stiffness: float, left: float, right: float,
    roundoff_m: float, accuracy_m: float,
) -> np.ndarray | None:
    """Bound all three omitted line contributions on a smooth Q2 cell.

    Q2 position shapes have absolute value at most one, so the 18-coordinate
    distance Jacobian has norm at most sqrt(6). Target shape derivatives
    have absolute value at most four, giving a sqrt(48) bound. Strict
    convexity in target station bounds the projection Hessian. The returned
    bounds are for energy, maximum force component and maximum tangent entry.
    """
    if roundoff_m > accuracy_m:
        return None
    source_controls = _bezier_controls(source, left, right)
    target_controls = _bezier_controls(target, 0.0, 1.0)
    maximum_separation = max(
        float(np.linalg.norm(a-b))
        for a in source_controls for b in target_controls
    )
    if maximum_separation <= 0.0:
        return None
    target_first = _curve(target, 0.0)[1]
    target_last = _curve(target, 1.0)[1]
    target_delta = target_last-target_first
    delta_square = float(target_delta@target_delta)
    minimum_station = (float(np.clip(-float(target_first@target_delta)
                                     /delta_square, 0.0, 1.0))
                       if delta_square > 0.0 else 0.0)
    target_min_speed = float(np.linalg.norm(
        target_first+minimum_station*target_delta,
    ))
    target_max_speed = max(float(np.linalg.norm(target_first)),
                           float(np.linalg.norm(target_last)))
    convexity = (target_min_speed**2
                 - maximum_separation*float(np.linalg.norm(target_delta)))
    if convexity <= 0.0:
        return None
    middle = 0.5*(left+right)
    source_point = _curve(source, middle)[0]
    projection = closest_on_b3(source_point, target)
    if not projection.unique:
        return None
    reach = max(float(np.linalg.norm(control-source_point))
                for control in source_controls)
    minimum_distance = projection.distance-reach-roundoff_m
    if minimum_distance <= 0.0:
        return None
    penetration = max(0.0, radius-minimum_distance)
    reference_speed = max(float(np.linalg.norm(_curve(reference, station)[1]))
                          for station in (left, right))
    jacobian_norm = np.sqrt(6.0)
    mixed_norm = (np.sqrt(48.0)
                  + jacobian_norm*target_max_speed/minimum_distance)
    hessian_norm = (6.0/minimum_distance
                    + mixed_norm**2*maximum_separation/convexity)
    measure = stiffness*reference_speed*(right-left)
    bounds = np.array((
        0.5*measure*penetration**2,
        measure*penetration*jacobian_norm,
        measure*(6.0+penetration*hessian_norm),
    ))
    return bounds if np.all(np.isfinite(bounds)) else None


@dataclass(frozen=True)
class NativeRetainedWholePairResult:
    potential_j: float
    residual: np.ndarray
    tangent: np.ndarray
    beam_a_force_n: np.ndarray
    beam_b_force_n: np.ndarray
    beam_a_moment_nm: np.ndarray
    beam_b_moment_nm: np.ndarray
    conservative_lower_gap_m: float
    minimum_sampled_gap_m: float
    energy_error_j: float
    force_error_n: float
    tangent_error_n_per_m: float
    search_cells: int
    integration_intervals: int
    integration_evaluations: int
    integration_cells_checked: int
    bounded_omission_cells: int
    unresolved_search_intervals: int


def _pooled_interval_open(source, target, left, right, radius):
    """Exact Q2 hull-box separation, then the unchanged convex-distance proof."""
    a = _exact_controls(source,left,right); b = _exact_controls(target,0.,1.)
    square = Fraction(0)
    for axis in range(3):
        al,ah = min(p[axis] for p in a),max(p[axis] for p in a)
        bl,bh = min(p[axis] for p in b),max(p[axis] for p in b)
        separation = max(Fraction(0),bl-ah,al-bh)
        square += separation*separation
    return square >= Fraction(float(radius))**2 or _private_interval_open(source,target,left,right,radius)


def _pooled_connected_junction_partition(search, left, right, *, remaining_cells):
    """Opt-in caller preserves the cap even for the shared proof's fast path."""
    if remaining_cells < 1:
        raise MemoryError('pooled connected-junction search cell budget exhausted')
    from .native_retained_body_junction import connected_junction_partition
    partition = connected_junction_partition(search,left,right,remaining_cells=remaining_cells)
    if partition is not None and partition.visited > remaining_cells:
        raise MemoryError('pooled connected-junction search cell budget exhausted')
    return partition


from math import inf, nextafter

def _vertex_endpoint_bands(source, target, left, right, *, maximum_cells):
    """Enclose all simple endpoint-phi roots; refuse tangency and plateaus.

    Phi(s)=(q_endpoint-p(s)).q'_endpoint is quadratic. Exact derivative
    critical points partition its domain into monotone intervals, so exact
    endpoint signs enumerate every root. Rational bisection terminates only
    once an outward binary64 bracket is at most four adjacent-float steps.
    Domain-edge roots retain a one-sided uncertainty strip.
    """
    rows = lambda data: tuple(tuple(Fraction(float(x)) for x in row) for row in data)
    p, q = rows(source), rows(target)
    linear = lambda a: tuple(-3*x+4*y-z for x,y,z in zip(*a))
    quadratic = lambda a: tuple(2*x-4*y+2*z for x,y,z in zip(*a))
    pb, pc, qb, qc = linear(p), quadratic(p), linear(q), quadratic(q)
    dot = lambda a,b: sum(x*y for x,y in zip(a,b))
    low, high = Fraction(float(left)), Fraction(float(right))
    if not 0 <= low < high <= 1:
        raise ValueError('native body projection domain must lie in [0,1]')
    visited, bands = 0, []

    def outward(value, direction):
        result = float(value)
        if (direction < 0 and Fraction(result) > value) or (direction > 0 and Fraction(result) < value):
            result = nextafter(result, -inf if direction < 0 else inf)
        return result

    def strip(a,b):
        # One additional representable step covers an exactly representable
        # root too; no rounded scalar root is used as a zero-width cut.
        return (max(left,nextafter(outward(a,-1),-inf)),
                min(right,nextafter(outward(b,1),inf)))

    for endpoint in (0,1):
        point = q[0] if endpoint == 0 else q[2]
        tangent = qb if endpoint == 0 else tuple(x+2*y for x,y in zip(qb,qc))
        a = dot(tuple(x-y for x,y in zip(point,p[0])),tangent)
        b, c = -dot(pb,tangent), -dot(pc,tangent)
        value = lambda s: a+s*(b+s*c)
        if a == b == c == 0:
            return None,visited
        stations = [low,high]
        if c:
            critical = -b/(2*c)
            if low <= critical <= high:
                if value(critical) == 0:
                    return None,visited
                if low < critical < high:
                    stations.insert(1,critical)
        for station in stations:
            if value(station) == 0:
                bands.append(strip(station,station))
        for begin,stop in zip(stations[:-1],stations[1:]):
            first,last = value(begin),value(stop)
            if first*last >= 0:
                continue
            for _ in range(128):
                visited += 1
                if visited > maximum_cells:
                    raise MemoryError('native body endpoint projection inventory cell budget exhausted')
                band = strip(begin,stop)
                edge = band[0]
                for _ in range(4):
                    edge = nextafter(edge,inf)
                if edge >= band[1]:
                    bands.append(band)
                    break
                middle = (begin+stop)/2
                sample = value(middle)
                if sample == 0:
                    bands.append(strip(middle,middle))
                    break
                if first*sample > 0:
                    begin,first = middle,sample
                else:
                    stop = middle
            else:
                return None,visited
    merged = []
    for begin,stop in sorted(bands):
        if begin >= stop:
            return None,visited
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0],max(stop,merged[-1][1]))
        else:
            merged.append((begin,stop))
    return tuple(merged),visited


def _vertex_rows(curve):
    return tuple(tuple(Fraction(float(value)) for value in row) for row in curve)
def _vertex_dot(a,b):
    return sum(x*y for x,y in zip(a,b))
def _vertex_sub(a,b):
    return tuple(x-y for x,y in zip(a,b))
def _vertex_coefficients(curve):
    p=_vertex_rows(curve)
    return p[0],tuple(-3*x+4*y-z for x,y,z in zip(*p)),tuple(2*x-4*y+2*z for x,y,z in zip(*p))
def _vertex_at(polynomial,s):
    a,b,c=polynomial
    return a+b*s+c*s*s
def _vertex_extrema(polynomial,left,right):
    a,b,c=polynomial; lo,hi=Fraction(float(left)),Fraction(float(right))
    values=[_vertex_at(polynomial,lo),_vertex_at(polynomial,hi)]
    if c and lo < -b/(2*c) < hi:
        values.append(_vertex_at(polynomial,-b/(2*c)))
    return min(values),max(values)
def _vertex_source_hull(curve,left,right):
    a,b,c=_vertex_coefficients(curve); l,r=Fraction(float(left)),Fraction(float(right))
    p0=tuple(x+y*l+z*l*l for x,y,z in zip(a,b,c))
    p1=tuple(x+y*r+z*r*r for x,y,z in zip(a,b,c))
    middle=tuple(x+(r-l)*(y+2*z*l)/2 for x,y,z in zip(p0,b,c))
    return p0,middle,p1

def _vertex_fixed_projection_feature(source,target,left,right):
    """Strict exact BOTH endpoint KKT ranges; zero/root stays on inventory.

    Positive target stationarity convexity is separately required by caller.
    No whole-owner dominance claim certifies its winning projection feature.
    """
    p,b,c=_vertex_coefficients(source)
    q,qb,qc=_vertex_coefficients(target)
    last=_vertex_rows(target)[2]
    phis=[]
    for point,tangent in ((q,qb),(last,tuple(x+2*y for x,y in zip(qb,qc)))):
        phis.append((_vertex_dot(_vertex_sub(point,p),tangent),
                     -_vertex_dot(b,tangent),-_vertex_dot(c,tangent)))
    (first_min,first_max),(last_min,last_max)=(_vertex_extrema(phi,left,right) for phi in phis)
    if first_min > 0:
        return 'FIXED_START'
    if last_max < 0:
        return 'FIXED_END'
    if first_max < 0 < last_min:
        return 'FIXED_INTERIOR'
    return None


def _vertex_inventory(search,endpoint_bands,*,maximum_cells):
    """Return every root-free/uncertainty interval and exact visited debit.

    Classification uses exact full-interval _vertex_extrema, never midpoint signs.
    The complete target endpoint-root _vertex_inventory is charged before any None.
    This diagnostic does not return a mechanical partition or omit strips.
    """
    if maximum_cells < 1:
        raise MemoryError('endpoint _vertex_inventory search cell budget exhausted')
    targets=search.targets; source=search.source
    shared=_vertex_rows(targets[0])[2]==_vertex_rows(targets[1])[0]
    result=dict(shared_vertex=shared,visited=1,root_free=[],root_strips=[],convexity=[])
    if not shared:
        result['status']='UNSHARED_VERTEX'; return result
    hull=_vertex_source_hull(source,0.,1.)
    for target in targets:
        q,b,c=_vertex_coefficients(target); acc=tuple(2*x for x in c)
        aa=_vertex_dot(acc,acc); va=_vertex_dot(b,acc)
        t=max(Fraction(0),min(Fraction(1),-va/aa)) if aa else Fraction(0)
        bound=min(_vertex_dot(b,b)+_vertex_dot(_vertex_sub(q,p),acc)+3*va*t+Fraction(3,2)*aa*t*t for p in hull)
        result['convexity'].append(str(bound))
        if bound <= 0:
            result['status']='NONCONVEX'; return result
    p,b,c=_vertex_coefficients(source); j=_vertex_rows(targets[0])[2]
    left_b,left_c=_vertex_coefficients(targets[0])[1:]
    left_t=tuple(x+2*y for x,y in zip(left_b,left_c))
    right_t=_vertex_coefficients(targets[1])[1]
    phis=tuple((_vertex_dot(_vertex_sub(j,p),t),-_vertex_dot(b,t),-_vertex_dot(c,t)) for t in (left_t,right_t))
    result['phi_coefficients']=[[str(value) for value in phi] for phi in phis]
    # Complete material vertex containment gives global one-sided dominance.
    # Fast return additionally requires a fixed winning projection feature.
    # Zero or endpoint-root uncertainty keeps complete accounted inventory.
    lmin,lmax=_vertex_extrema(phis[0],0.,1.)
    rmin,rmax=_vertex_extrema(phis[1],0.,1.)
    if (rmin > 0 and any(phis[0])) or (lmax < 0 and any(phis[1])):
        winner=0 if rmin > 0 else 1
        feature=_vertex_fixed_projection_feature(source,targets[winner],0.,1.)
        if feature is not None:
            kind='LEFT_DOMINATES_RIGHT_VERTEX' if winner==0 else 'RIGHT_DOMINATES_LEFT_VERTEX'
            result['root_free']=[dict(interval=[0.,1.],label=kind,
                phi_left_range=[str(lmin),str(lmax)],phi_right_range=[str(rmin),str(rmax)],
                endpoint_transition_only=True,winning_projection_feature=feature)]
            result['status']='COMPLETE_EXACT_KKT_INVENTORY';return result
    bands=[]
    for target in targets:
        remaining=maximum_cells-result['visited']
        if remaining < 1:
            raise MemoryError('endpoint _vertex_inventory search cell budget exhausted')
        found,visited=endpoint_bands(source,target,0.,1.,maximum_cells=remaining)
        if visited > remaining:
            raise MemoryError('endpoint _vertex_inventory returned excess search cells')
        result['visited']+=visited
        if found is None:
            result['status']='NONCLASSICAL_ENDPOINT_KKT'; return result
        bands.extend(found)
    merged=[]
    for lo,hi in sorted(bands):
        if merged and lo <= merged[-1][1]:
            merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:
            merged.append((lo,hi))
    def label(lo,hi):
        (lmin,lmax),(rmin,rmax)=[_vertex_extrema(phi,lo,hi) for phi in phis]
        if lmax < 0 < rmin:
            kind='SHARED_VERTEX_CONE_PLATEAU'
        elif lmin > 0 and rmin > 0:
            kind='LEFT_DOMINATES_RIGHT_VERTEX'
        elif lmax < 0 and rmax < 0:
            kind='RIGHT_DOMINATES_LEFT_VERTEX'
        elif lmin > 0 > rmax:
            kind='GENUINE_TWO_INTERIOR'
        else:
            kind='UNCERTAIN_OR_ZERO_KKT'
        return dict(interval=[lo,hi],label=kind,phi_left_range=[str(lmin),str(lmax)],
            phi_right_range=[str(rmin),str(rmax)],endpoint_transition_only=(lmax<=0 or rmin>=0))
    position=0.
    for lo,hi in merged:
        if lo > position:
            result['root_free'].append(label(position,lo))
        result['root_strips'].append(label(lo,hi)); position=hi
    if position < 1.:
        result['root_free'].append(label(position,1.))
    result['status']='COMPLETE_EXACT_KKT_INVENTORY'
    return result



def _pooled_endpoint_feature_inventory(search,left,right,*,remaining_cells,
        reference,radius,stiffness,shared_workmap):
    """Exact endpoint KKT partition or complete fallback with visited debit.

    Endpoint transitions share the exact physical point/work map, hence have
    continuous E/R and no distinct-interior root-H term. Full bulk H includes
    its jump. Every root strip bounds BOTH branches in half-weighted physical30.
    Genuine two-interior or uncertain KKT is sent to the mature complete route.
    """
    from .native_curved_pair_partition import CurvedOwnershipPartition
    if remaining_cells < 1:
        raise MemoryError('pooled endpoint inventory search cell budget exhausted')
    if not shared_workmap or left != 0. or right != 1.:
        return None,1,('UNADMITTED_ENDPOINT_WORKMAP_OR_DOMAIN',)
    result=_vertex_inventory(search,_vertex_endpoint_bands,maximum_cells=remaining_cells)
    used=result['visited'];labels=tuple(row['label'] for row in result['root_free'])
    if result['status'] != 'COMPLETE_EXACT_KKT_INVENTORY':
        return None,used,(result['status'],)
    owners={'SHARED_VERTEX_CONE_PLATEAU':0,'LEFT_DOMINATES_RIGHT_VERTEX':0,
            'RIGHT_DOMINATES_LEFT_VERTEX':1}
    if any(label not in owners for label in labels):
        return None,used,labels
    if any(not row['endpoint_transition_only'] for row in result['root_strips']):
        return None,used,labels+('UNCERTAIN_ROOT_STRIP_COMPETITION',)
    pieces=tuple((*row['interval'],owners[row['label']]) for row in result['root_free'])
    charged=[]
    for row in result['root_strips']:
        lo,hi=row['interval'];bounds=[]
        for target in search.targets:
            bound=_exact_tiny_cell_omission_bound(search.source,target,reference,radius,stiffness,lo,hi)
            if bound is None or not np.all(np.isfinite(bound)) or np.any(bound<0):
                return None,used,labels+('UNBOUNDED_ROOT_STRIP',)
            bounds.append(bound)
        # Half weight times complete 18->30 amplification [1,2,4].
        # Sum both potential branches once, conservatively, with exact rounding.
        bulk=np.array([_up((Fraction(float(a))+Fraction(float(b)))*factor)
                       for a,b,factor in zip(bounds[0],bounds[1],(Fraction(1,2),Fraction(1),Fraction(2)))])
        charged.append((lo,hi,bulk,0.))
    return CurvedOwnershipPartition(pieces,(),0,tuple(charged)),used,labels


class NativeRetainedWholeCurvePair:
    """Symmetric distributed line contact over both cells of two native beams.

    The opt-in V13 curved route reports force/tangent error controls in its
    30 independent physical point coordinates, all measured in length. Its
    product residual/tangent still use the native 84 coordinate pullback;
    those mixed work coordinates are not an absolute N/m error chart.
    Historical V4/V6 quadrature/error conventions are unchanged.
    """

    def __init__(self, element_a: int,
                 reference_a: CenteredCurvedBeam3ReferenceGeometry,
                 node_ids_a: tuple[int, int, int], radius_a_m: float,
                 element_b: int,
                 reference_b: CenteredCurvedBeam3ReferenceGeometry,
                 node_ids_b: tuple[int, int, int], radius_b_m: float, *,
                 line_stiffness_n_per_m2: float,
                 distance_tolerance_m: float,
                 feature_tolerance_m: float,
                 search_parameter_tolerance: float,
                 maximum_search_depth: int,
                 maximum_search_cells: int,
                 energy_tolerance_j: float,
                 force_tolerance_n: float,
                 tangent_tolerance_n_per_m: float,
                 maximum_integration_intervals: int,
                 certify_open_intervals: bool = False,
                 certify_curved_ownership: bool = False,
                 pooled_quadrature_budget: bool = False):
        if type(pooled_quadrature_budget) is not bool or (pooled_quadrature_budget and not certify_curved_ownership):
            raise ValueError('pooled quadrature requires explicit resolved ownership')
        self.pooled_quadrature_budget = pooled_quadrature_budget
        if type(certify_open_intervals) is not bool:
            raise ValueError('native whole pair open certificate selection must be bool')
        self.certify_open_intervals = certify_open_intervals
        if type(certify_curved_ownership) is not bool:
            raise ValueError('native whole pair curved ownership selection must be bool')
        self.certify_curved_ownership = certify_curved_ownership
        if (type(element_a) is not int or type(element_b) is not int
                or element_a <= 0 or element_b <= 0 or element_a == element_b):
            raise ValueError("native whole pair needs distinct beam identities")
        for label, ids in (("A", node_ids_a), ("B", node_ids_b)):
            if (type(ids) is not tuple or len(ids) != 3
                    or len(set(ids)) != 3
                    or any(type(value) is not int or value <= 0 for value in ids)):
                raise ValueError(f"native whole pair needs exact {label} node identities")
        if set(node_ids_a) & set(node_ids_b):
            raise ValueError("native whole pair does not admit shared material nodes")
        controls = (radius_a_m, radius_b_m, line_stiffness_n_per_m2,
                    distance_tolerance_m, feature_tolerance_m,
                    search_parameter_tolerance, energy_tolerance_j,
                    force_tolerance_n, tangent_tolerance_n_per_m)
        if any(type(value) not in (int, float) or not isfinite(value) or value <= 0.0
               for value in controls):
            raise ValueError("native whole pair needs positive dimensional controls")
        if (type(maximum_search_depth) is not int or maximum_search_depth < 0
                or type(maximum_search_cells) is not int or maximum_search_cells < 1
                or type(maximum_integration_intervals) is not int
                or maximum_integration_intervals < 1):
            raise ValueError("native whole pair needs finite resource budgets")
        self.cells_a = tuple(NativeRetainedCurveCell(element_a, reference_a, c)
                             for c in (0, 1))
        self.cells_b = tuple(NativeRetainedCurveCell(element_b, reference_b, c)
                             for c in (0, 1))
        self.node_ids_a = node_ids_a
        self.node_ids_b = node_ids_b
        self.radius_a = float(radius_a_m)
        self.radius_b = float(radius_b_m)
        self.radius = self.radius_a+self.radius_b
        self.stiffness = float(line_stiffness_n_per_m2)
        self.distance_tolerance = float(distance_tolerance_m)
        self.feature_tolerance = float(feature_tolerance_m)
        self.search_parameter_tolerance = float(search_parameter_tolerance)
        self.maximum_search_depth = maximum_search_depth
        self.maximum_search_cells = maximum_search_cells
        self.energy_tolerance = float(energy_tolerance_j)
        self.force_tolerance = float(force_tolerance_n)
        self.tangent_tolerance = float(tangent_tolerance_n_per_m)
        self.maximum_integration_intervals = maximum_integration_intervals
        identity = np.tile(np.eye(3), (2, 1, 1))
        self.reference_curves = tuple(
            cell.evaluate(reference.coordinates, np.zeros((3, 3)), identity).positions
            for cells, reference in ((self.cells_a, reference_a),
                                     (self.cells_b, reference_b))
            for cell in cells
        )
        self.identity = self._fingerprint()

    def _fingerprint(self) -> str:
        return sha(dict(
            profile=("NATIVE_RETAINED_WHOLE_Q2_PAIR_DEV_V14_POOLED_PHYSICAL_GAP_QUADRATURE" if self.pooled_quadrature_budget else (("NATIVE_RETAINED_WHOLE_Q2_PAIR_DEV_V13_RESOLVED_PHYSICAL_GAP_GLOBAL_CHORD"
                      if self.certify_open_intervals else
                      "NATIVE_RETAINED_WHOLE_Q2_PAIR_DEV_V13_RESOLVED_PHYSICAL_GAP")
                     if self.certify_curved_ownership else
                     ("NATIVE_RETAINED_WHOLE_Q2_PAIR_DEV_V6_GLOBAL_CHORD"
                      if self.certify_open_intervals else "NATIVE_RETAINED_WHOLE_Q2_PAIR_DEV_V4"))),
            **(dict(pooled_quadrature_budget=True,
                    exact_endpoint_kkt_inventory_all_visited=True,
                    exact_global_endpoint_dominance_and_precision_fallback=True,
                    fast_global_owner_requires_fixed_winning_projection_feature=True,
                    exact_owned_interval_open=True,
                    complete_material_source_domains=True,
                    certify_curved_ownership=self.certify_curved_ownership,
                    certify_open_intervals=self.certify_open_intervals)
               if self.pooled_quadrature_budget else {}),
            cells_a=[(cell.element_id, cell.cell,
                      cell.reference.canonical_data()) for cell in self.cells_a],
            cells_b=[(cell.element_id, cell.cell,
                      cell.reference.canonical_data()) for cell in self.cells_b],
            node_ids_a=self.node_ids_a, node_ids_b=self.node_ids_b,
            radius_a_m=self.radius_a, radius_b_m=self.radius_b,
            line_stiffness_n_per_m2=self.stiffness,
            distance_tolerance_m=self.distance_tolerance,
            feature_tolerance_m=self.feature_tolerance,
            search_parameter_tolerance=self.search_parameter_tolerance,
            maximum_search_depth=self.maximum_search_depth,
            maximum_search_cells=self.maximum_search_cells,
            energy_tolerance_j=self.energy_tolerance,
            force_tolerance_n=self.force_tolerance,
            tangent_tolerance_n_per_m=self.tangent_tolerance,
            maximum_integration_intervals=self.maximum_integration_intervals,
            reference_curves=self.reference_curves,
        ))

    def _geometry_searches(self, poses):
        current = tuple(
            cell.evaluate(positions, low, rotations).positions
            for cells, positions, low, rotations in (
                (self.cells_a, poses[0], poses[1], poses[2]),
                (self.cells_b, poses[3], poses[4], poses[5]),
            )
            for cell in cells
        )
        searches = []
        for source_beam, target_beam in ((0, 1), (1, 0)):
            source_base, target_base = 2*source_beam, 2*target_beam
            ids = self.node_ids_b if target_beam else self.node_ids_a
            target_ids = ((ids[0], ids[1]), (ids[1], ids[2]))
            for source_cell in (0, 1):
                searches.append(_GeometryGapUnionSearch(
                    current[source_base+source_cell],
                    (current[target_base], current[target_base+1]),
                    target_ids,
                    combined_radius_m=self.radius,
                    distance_tolerance_m=self.distance_tolerance,
                    feature_tolerance_m=self.feature_tolerance,
                    parameter_tolerance=self.search_parameter_tolerance,
                    maximum_depth=self.maximum_search_depth,
                    maximum_cells=self.maximum_search_cells,
                    certify_open_intervals=self.certify_open_intervals,
                ))
        return tuple(searches)

    def physical_gap_certificate(self, positions_a, position_low_a, cell_rotations_a,
                                 positions_b, position_low_b, cell_rotations_b):
        """Resolved physical-gap bracket for the explicit curved profile.

        Ownership/integration scan stopping rules do not define endpoint gap
        accuracy. This query encloses the complete physical Q2 union distance
        within the existing dimensional distance tolerance or fails closed.
        """
        if self._fingerprint() != self.identity:
            raise ValueError("native whole pair definition changed")
        if not self.certify_curved_ownership:
            raise ValueError("native physical gap requires curved ownership opt-in")
        searches = self._geometry_searches((positions_a, position_low_a, cell_rotations_a,
                                          positions_b, position_low_b, cell_rotations_b))
        return resolved_physical_gap_lower(searches, self.distance_tolerance)

    def physical_gap_lower(self, *poses):
        return self.physical_gap_certificate(*poses).lower_gap_m

    def geometry_gap_lower(self, positions_a: np.ndarray,
                           position_low_a: np.ndarray,
                           cell_rotations_a: np.ndarray,
                           positions_b: np.ndarray,
                           position_low_b: np.ndarray,
                           cell_rotations_b: np.ndarray, *,
                           minimum_gap_m: float | None = None) -> float:
        """Conservatively bound the complete physical pair's minimum gap.

        This geometry certificate includes both retained cells in both
        directions, including their internal rotation lifts. It does not
        admit a force law: ambiguous nearest points and unresolved intervals
        still contribute their conservative bounds. Search controls and
        finite budgets are the same as for the full mechanical evaluation.
        With a finite requested minimum, the complete-source chord bound may
        return early only when it certifies that minimum for the whole pair.
        """
        if self._fingerprint() != self.identity:
            raise ValueError("native whole pair definition changed")
        if (minimum_gap_m is not None
                and (type(minimum_gap_m) not in (int, float)
                     or not isfinite(minimum_gap_m))):
            raise ValueError("native whole pair minimum gap must be finite")
        searches = self._geometry_searches((positions_a, position_low_a, cell_rotations_a,
                                          positions_b, position_low_b, cell_rotations_b))
        if minimum_gap_m is not None:
            # Constructors above validate every physical cell, topology and
            # control before any early return. Both directional bounds cover
            # every complete target cell and include the helper's roundoff
            # guards; sampled gaps cannot authorize this certificate.
            lower = min(
                _source_interval_target_gap_lower(
                    search.source, 0.0, 1.0, target, self.radius,
                )
                for search in searches for target in search.targets
            )
            if lower >= minimum_gap_m:
                return lower
        return min(search.scan().conservative_lower_gap_m
                   for search in searches)

    def evaluate(self, positions_a: np.ndarray, position_low_a: np.ndarray,
                 cell_rotations_a: np.ndarray,
                 positions_b: np.ndarray, position_low_b: np.ndarray,
                 cell_rotations_b: np.ndarray, *, _states=None, _references=None,
                 _target_ids=None, _source_domains=None, _geometry_result=False, _diagnostics=None) -> NativeRetainedWholePairResult:
        if self._fingerprint() != self.identity:
            raise ValueError("native whole pair definition changed")
        states = tuple(
            cell.evaluate(positions, low, rotations)
            for cells, positions, low, rotations in (
                (self.cells_a, positions_a, position_low_a, cell_rotations_a),
                (self.cells_b, positions_b, position_low_b, cell_rotations_b),
            )
            for cell in cells
        ) if _states is None else tuple(_states)
        reference_curves = self.reference_curves if _references is None else tuple(_references)
        current = tuple(state.positions for state in states)
        jacobian = np.zeros((36, 84))
        for global_cell, state in enumerate(states):
            side = 0 if global_cell < 2 else 1
            jacobian[9*global_cell:9*global_cell+9,
                     42*side:42*side+42] = state.geometric_jacobian
        map_norm = (float(np.linalg.norm(jacobian, 2))
                    if self.certify_curved_ownership else 0.0)
        point_map = physical_point_map() if self.certify_curved_ownership else None
        total = np.zeros(1+36+36*36)
        error = np.zeros(3)
        boundary_error = np.zeros(3)
        pooled = None
        if self.pooled_quadrature_budget:
            from .native_pair_error_budget import PooledQuadratureBudget, within_reserve
            pooled = PooledQuadratureBudget((self.energy_tolerance,self.force_tolerance,self.tangent_tolerance))
        search_cells = intervals = evaluations = unresolved = visited = omissions = 0
        minimum_sampled = conservative_lower = float("inf")

        for source_beam, target_beam in ((0, 1), (1, 0)):
            if _source_domains is not None and source_beam != 0:
                continue
            source_base, target_base = 2*source_beam, 2*target_beam
            target_positions = (current[target_base], current[target_base+1])
            ids = self.node_ids_b if target_beam else self.node_ids_a
            target_ids = ((ids[0], ids[1]), (ids[1], ids[2]))
            if _target_ids is not None:
                target_ids = _target_ids
            for source_cell in (0, 1):
                if _source_domains is not None and source_cell not in _source_domains:
                    continue
                source_index = source_base+source_cell
                projection_cuts: dict[int, tuple[float, ...]] = {}
                search_type = (_CurvedOwnershipUnionSearch
                               if self.certify_curved_ownership
                               else NativeTwoCellUnionSearch)
                search = search_type(
                    current[source_index], target_positions, target_ids,
                    combined_radius_m=self.radius,
                    distance_tolerance_m=self.distance_tolerance,
                    feature_tolerance_m=self.feature_tolerance,
                    parameter_tolerance=self.search_parameter_tolerance,
                    maximum_depth=self.maximum_search_depth,
                    maximum_cells=self.maximum_search_cells,
                    certify_open_intervals=self.certify_open_intervals,
                )
                if _source_domains is None and pooled is None:
                    scan = search.scan()
                else:
                    # Fixed material domain: ownership is completely inventoried
                    # by the existing curved partition below. Avoid a redundant
                    # Lipschitz detection tessellation; unresolved is never open.
                    from .native_curve_union import NativeUnionScan, NativeUnionInterval
                    left, right = (0.,1.) if _source_domains is None else _source_domains[source_cell]
                    lower = min(_source_interval_target_gap_lower(
                        current[source_index], left, right, target, self.radius)
                        for target in target_positions)
                    scan = NativeUnionScan((NativeUnionInterval(left, right,
                        'open' if lower >= 0.0 else 'unresolved', None, lower, float('inf')),),
                        search.nearest(.5*(left+right)).gap_m, lower, 1)
                search_cells += scan.visited_cells
                unresolved += scan.unresolved_intervals
                minimum_sampled = min(minimum_sampled,
                                      scan.minimum_sampled_gap_m)
                conservative_lower = min(conservative_lower,
                                         scan.conservative_lower_gap_m)

                def integrand(station: float, owner: int | None,
                              certified_active: bool = False) -> tuple[np.ndarray, bool]:
                    nonlocal evaluations
                    if _source_domains is not None or (pooled is not None and owner is not None):
                        if type(owner) is not int or owner not in (0,1):
                            raise ValueError('native body integration requires an explicit target owner')
                        chosen = owner
                    else:
                        chosen = (search.nearest(station).target_cell
                                  if owner is None or self.certify_curved_ownership
                                  else owner)
                    target_index = target_base+chosen
                    if _source_domains is not None or pooled is not None:
                        from .native_retained_body_projection import strict_projection_station
                        from .beam_curve_pair import _b3_distance_jet_at_station
                        target_station = strict_projection_station(
                            current[source_index],current[target_index],station)
                        jet = _b3_distance_jet_at_station(
                            current[source_index],current[target_index],station,
                            self.radius,target_station,source_is_a=True)
                    else:
                        jet = b3_distance_jet(
                            current[source_index], current[target_index],
                            station, self.radius, source_is_a=True,
                        )
                    evaluations += 1
                    if _source_domains is not None or pooled is not None:
                        lo,hi,gap=_private_factored_gap(current[source_index],current[target_index],station,self.radius,jet.target_parameter)
                        if lo<0.0<hi:
                            raise MemoryError('native whole pair integration parameter resolution exhausted: floating activation sign contradicts certified-active interval')
                        jet=replace(jet,gap=gap)
                    if jet.gap >= 0.0:
                        if self.certify_curved_ownership and certified_active:
                            raise MemoryError(
                                "native whole pair integration parameter resolution exhausted: "
                                "floating activation sign contradicts certified-active interval"
                            )
                        return np.zeros_like(total), False
                    measure = float(deterministic_norm(
                        _curve(reference_curves[source_index], station)[1]
                    ))
                    factor = self.stiffness*measure
                    gradient = factor*jet.gap*jet.gradient
                    hessian = factor*(np.outer(jet.gradient, jet.gradient)
                                      + jet.gap*jet.hessian)
                    index = np.r_[np.arange(9*source_index, 9*source_index+9),
                                  np.arange(9*target_index, 9*target_index+9)]
                    value = np.zeros_like(total)
                    value[0] = 0.5*factor*jet.gap*jet.gap
                    value[1+index] += gradient
                    matrix = value[37:].reshape(36, 36)
                    matrix[np.ix_(index, index)] += hessian
                    return value, True

                def quadrature(left: float, right: float, rule, owner: int | None,
                               certified_active: bool = False):
                    points, weights = rule
                    result = np.zeros_like(total)
                    active = False
                    scale = 0.5*(right-left)
                    midpoint = 0.5*(left+right)
                    for point, weight in zip(points, weights):
                        value, on = integrand(midpoint+scale*float(point), owner, certified_active)
                        result += scale*float(weight)*value
                        active = active or on
                    return result, active

                def integrate(left: float, right: float, owner: int | None,
                              budget: np.ndarray,
                              certified_active: bool = False) -> np.ndarray:
                    nonlocal visited, intervals, omissions
                    visited += 1
                    if _diagnostics is not None:
                        _diagnostics.update(error=error.copy(),boundary_error=boundary_error.copy(),
                            half_reserve=.5*np.array((self.energy_tolerance,self.force_tolerance,self.tangent_tolerance)),
                            visited=visited,omissions=omissions,interval=(left,right),owner=owner,
                            source_cell=source_index,target_cell=target_base+owner,
                            source_curve=current[source_index].copy(),target_curve=current[target_base+owner].copy(),
                            pooled_quadrature=None if pooled is None else pooled.report())
                    if visited > self.maximum_integration_intervals:
                        raise MemoryError(
                            "native whole pair integration interval budget exhausted: "
                            f"source cell {source_index}, target cell "
                            f"{target_base+owner}, interval [{left}, {right}], "
                            f"visited {visited}, allocated tolerances "
                            f"{budget.tolist()}"
                        )
                    middle = 0.5*(left+right)
                    if pooled is not None and type(owner) is int:
                        # Full absolute physical30 strip bound, charged once.
                        bound = _exact_tiny_cell_omission_bound(
                            current[source_index],current[target_base+owner],
                            reference_curves[source_index],self.radius,self.stiffness,left,right)
                        if bound is not None:
                            bound = bound*np.array((1.,2.,4.))
                            limits = np.array((self.energy_tolerance,self.force_tolerance,self.tangent_tolerance))
                            if pooled.admits(bound) and within_reserve(error,bound,limits):
                                pooled.charge(bound)
                                error[:] = np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,bound)])
                                omissions += 1
                                return np.zeros_like(total)
                    if ((_source_domains is not None or pooled is not None) and all(
                            (_private_interval_open if pooled is None else _pooled_interval_open)(
                            current[source_index],target,left,right,self.radius) for target in target_positions)):
                        return np.zeros_like(total)
                    if middle == left or middle == right:
                        bound = (_exact_tiny_cell_omission_bound(
                            current[source_index], current[target_base+owner],
                            reference_curves[source_index], self.radius,
                            self.stiffness, left, right,
                        ) if self.certify_curved_ownership else _tiny_cell_omission_bound(
                            current[source_index], current[target_base+owner],
                            reference_curves[source_index],
                            self.radius, self.stiffness, left, right,
                            search.roundoff,
                            min(self.distance_tolerance,
                                self.feature_tolerance),
                        ))
                        remaining = np.array((self.energy_tolerance,
                                              self.force_tolerance,
                                              self.tangent_tolerance))-error
                        if bound is not None and np.all(bound <= remaining):
                            if self.certify_curved_ownership:
                                bound *= np.array((1.0, 2.0, 4.0))
                                tolerances = np.array((self.energy_tolerance, self.force_tolerance,
                                                       self.tangent_tolerance))
                                if ((_source_domains is None and np.any(bound > 0.5*tolerances/self.maximum_integration_intervals))
                                        or np.any(bound > remaining)
                                        or (np.any(boundary_error+bound > 0.5*tolerances) if pooled is None else not within_reserve(boundary_error,bound,tolerances,half=True))):
                                    raise MemoryError("curved pair physical-point omission budget exhausted")
                                boundary_error[:] = boundary_error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(boundary_error,bound)])
                            error[:] = error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,bound)])
                            omissions += 1
                            return np.zeros_like(total)
                        raise MemoryError(
                            "native whole pair integration parameter resolution "
                            f"exhausted: source cell {source_index}, target cell "
                            f"{target_base+owner}, interval [{left}, {right}]"
                        )
                    if (_source_domains is not None and all(_private_interval_open(
                            current[source_index],target,left,right,self.radius) for target in target_positions)):
                        return np.zeros_like(total)
                    if (_source_domains is not None
                            and not np.array_equal(_curve(current[source_index], left)[0], _curve(current[source_index], right)[0])
                            and min(_source_interval_target_gap_lower(
                            current[source_index], left, right, target, self.radius) for target in target_positions) >= 0.0):
                        # Complete source/target geometry proves zero potential;
                        # the fixed-domain ownership certificate is retained.
                        return np.zeros_like(total)
                    point = _curve(current[source_index], middle)[0]
                    projected = closest_on_b3(point, current[target_base+owner])
                    if not projected.unique:
                        raise ValueError("native whole pair has competing target minima")
                    controls = _bezier_controls(current[source_index], left, right)
                    reach = max(float(np.linalg.norm(control-point))
                                for control in controls)
                    lower_gap = projected.distance-self.radius-reach-search.roundoff
                    if lower_gap >= 0.0:
                        return np.zeros_like(total)
                    if self.certify_curved_ownership and not certified_active:
                        certified_active = (_active_gap_correspondence_upper(
                            current[source_index], current[target_base+owner],
                            left, right, self.radius,
                        ) < 0.0)
                    if _source_domains is not None and not certified_active:
                        certified_active = _private_active_correspondence(current[source_index],current[target_base+owner],left,right,self.radius)
                    if _diagnostics is not None:
                        _diagnostics['last_activation'] = dict(interval=(left,right),source=source_index,target=target_base+owner,
                            certified_active=bool(certified_active),nominal_midpoint_gap=projected.distance-self.radius,
                            conservative_gap_lower=lower_gap)
                    support = (None if certified_active else _monotone_gap_endpoints(
                        current[source_index], current[target_base+owner],
                        left, right, search.roundoff,
                    ))
                    if support is not None:
                        gap_left, gap_right = (value-self.radius
                                               for value in support)
                        open_gaps = (gap_left, gap_right)
                        if self.certify_curved_ownership:
                            scale = max(1.0, float(np.max(np.abs(current[source_index]))),
                                        float(np.max(np.abs(current[target_base+owner]))))
                            convexity = _convexity_lower(controls, current[target_base+owner],
                                2048.0*np.finfo(float).eps*scale*scale)
                            open_gaps = tuple(_down(_endpoint_distance_lower(
                                current[source_index], current[target_base+owner],
                                station, convexity,
                            )-self.radius) for station in (left, right))
                        if _source_domains is not None:
                            endpoints=tuple(_private_factored_gap(current[source_index],current[target_base+owner],t,self.radius) for t in (left,right))
                            open_gaps=tuple(value[0] for value in endpoints)
                            gap_left,gap_right=(value[2] for value in endpoints)
                        if min(open_gaps) >= 0.0:
                            return np.zeros_like(total)
                        if gap_left*gap_right < 0.0:
                            def gap_at(station: float) -> float:
                                if _source_domains is not None:
                                    return _private_factored_gap(current[source_index],current[target_base+owner],station,self.radius)[2]
                                return (closest_on_b3(
                                    _curve(current[source_index], station)[0],
                                    current[target_base+owner],
                                ).distance-self.radius)
                            root = float(brentq(
                                gap_at, left, right,
                                xtol=1e-14, rtol=1e-14,
                            ))
                            if left < root < right:
                                fraction = (root-left)/(right-left)
                                return (
                                    integrate(left, root, owner, fraction*budget)
                                    + integrate(root, right, owner,
                                                (1.0-fraction)*budget)
                                )
                    if self.certify_curved_ownership and not certified_active:
                        # Neither agreement nor inactivity at Gauss points is
                        # a support certificate. Bound the whole uncertain
                        # strip, or continue finite subdivision. This is the
                        # existing complete omission bound in the 30 physical
                        # length chart, charged to the unchanged half reserve.
                        bound = _exact_tiny_cell_omission_bound(
                            current[source_index], current[target_base+owner],
                            reference_curves[source_index], self.radius,
                            self.stiffness, left, right,
                        )
                        tolerances = np.array((self.energy_tolerance,
                                               self.force_tolerance,
                                               self.tangent_tolerance))
                        if bound is not None and _source_domains is None:
                            bound *= np.array((1.0, 2.0, 4.0))
                            # At most N strips can be omitted, because every
                            # strip consumes one of the N visited intervals.
                            # Give each a bounded share instead of allowing an
                            # early strip to consume the whole reserve and
                            # starve a later one-ulp boundary.
                            if ((_source_domains is not None or np.all(bound <= 0.5*tolerances/self.maximum_integration_intervals))
                                    and (np.all(boundary_error+bound <= 0.5*tolerances) if pooled is None else within_reserve(boundary_error,bound,tolerances,half=True))
                                    and (np.all(error+bound <= tolerances) if pooled is None else within_reserve(error,bound,tolerances))):
                                boundary_error[:] = boundary_error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(boundary_error,bound)])
                                error[:] = error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,bound)])
                                omissions += 1
                                return np.zeros_like(total)
                        return (integrate(left, middle, owner, 0.5*budget)
                                +integrate(middle, right, owner, 0.5*budget))
                    try:
                        coarse, coarse_active = quadrature(left, right, _GAUSS_8, owner, certified_active)
                        refined, refined_active = quadrature(left, right, _GAUSS_16, owner, certified_active)
                    except MemoryError as failure:
                        if (not self.certify_curved_ownership or not str(failure).endswith(
                                'floating activation sign contradicts certified-active interval')):
                            raise
                        # Discard the whole partial rule. A contradicted sample
                        # never silently loses its tangent. Bound every omitted
                        # contribution on this entire interval, or subdivide;
                        # uniform unresolved activation cannot be admitted by
                        # sample agreement or by a zero traction alone.
                        bound = _exact_tiny_cell_omission_bound(
                            current[source_index], current[target_base+owner],
                            reference_curves[source_index], self.radius,
                            self.stiffness, left, right,
                        )
                        tolerances = np.array((self.energy_tolerance,
                                               self.force_tolerance,
                                               self.tangent_tolerance))
                        if bound is not None and _source_domains is None:
                            bound *= np.array((1.0, 2.0, 4.0))
                            if ((_source_domains is not None or np.all(bound <= 0.5*tolerances/self.maximum_integration_intervals))
                                    and (np.all(boundary_error+bound <= 0.5*tolerances) if pooled is None else within_reserve(boundary_error,bound,tolerances,half=True))
                                    and (np.all(error+bound <= tolerances) if pooled is None else within_reserve(error,bound,tolerances))):
                                boundary_error[:] = boundary_error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(boundary_error,bound)])
                                error[:] = error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,bound)])
                                omissions += 1
                                return np.zeros_like(total)
                        return (integrate(left, middle, owner, 0.5*budget, certified_active)
                                +integrate(middle, right, owner, 0.5*budget, certified_active))
                    difference = np.abs(refined-coarse)
                    if self.certify_curved_ownership:
                        signed = refined-coarse
                        measured = np.array((
                            difference[0],
                            float(np.max(np.abs(point_map.T @ signed[1:37]))),
                            float(np.max(np.abs(point_map.T @ signed[37:].reshape(36, 36)
                                                 @ point_map))),
                        ))
                    else:
                        measured = np.array((
                            difference[0], float(np.max(difference[1:37])),
                            float(np.max(difference[37:])),
                        ))
                    admitted = (np.all(measured <= budget) if pooled is None else
                        pooled.admits(measured) and within_reserve(error,measured,(self.energy_tolerance,self.force_tolerance,self.tangent_tolerance)))
                    if _diagnostics is not None:
                        _diagnostics['last_rule'] = dict(interval=(left,right),source=source_index,target=target_base+owner,
                            coarse_active=coarse_active,refined_active=refined_active,measured=measured.copy(),admitted=bool(admitted))
                    if (coarse_active or refined_active) and admitted:
                        if pooled is not None:pooled.charge(measured)
                        intervals += 1
                        error[:] = error+measured if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,measured)])
                        return refined
                    if certified_active and not (coarse_active or refined_active):
                        raise MemoryError("certified native whole-pair contact absent at quadrature")
                    return (integrate(left, middle, owner, 0.5*budget,
                                      certified_active)
                            + integrate(middle, right, owner, 0.5*budget,
                                        certified_active))

                moving_switches = []
                junction_bands = []
                ownership_cells = 0
                piecewise = (None if self.certify_curved_ownership else _straight_support_partitions(
                    current[source_index], target_positions, self.radius,
                    0.0, 1.0,
                    min(self.distance_tolerance, self.feature_tolerance),
                ))
                if piecewise is None and not self.certify_curved_ownership:
                    piecewise = _piecewise_straight_support_partitions(
                        current[source_index], target_positions, target_ids,
                        self.radius,
                        min(self.distance_tolerance, self.feature_tolerance),
                    )
                if piecewise is not None:
                    source_pieces = [(left, right, owner, True)
                                     for left, right, owner in piecewise]
                else:
                    source_pieces = []
                    for interval in scan.intervals:
                        if _source_domains is not None:
                            left, right = _source_domains[source_cell]
                            if interval.right <= left or interval.left >= right:
                                continue
                            interval = type(interval)(max(left, interval.left), min(right, interval.right),
                                interval.status, interval.target_cell, interval.lower_gap_m, interval.upper_gap_m)
                        if interval.status == "open":
                            continue
                        if self.certify_curved_ownership:
                            partition = None
                            if pooled is not None:
                                shared_workmap = (target_ids[0][1] == target_ids[1][0]
                                    and np.array_equal(states[target_base].geometric_jacobian[6:9],
                                                       states[target_base+1].geometric_jacobian[:3])
                                    and np.array_equal(point_map[9*target_base+6:9*target_base+9],
                                                       point_map[9*(target_base+1):9*(target_base+1)+3]))
                                partition,debited,labels = _pooled_endpoint_feature_inventory(
                                    search,interval.left,interval.right,
                                    remaining_cells=self.maximum_search_cells-search_cells-ownership_cells,
                                    reference=reference_curves[source_index],radius=self.radius,
                                    stiffness=self.stiffness,shared_workmap=shared_workmap)
                                ownership_cells += debited
                                if _diagnostics is not None:
                                    _diagnostics.setdefault('endpoint_kkt_inventory',[]).append(
                                        dict(source=source_index,visited=debited,labels=labels,
                                             adopted=partition is not None))
                            elif _source_domains is not None:
                                from .native_retained_body_junction import connected_junction_partition
                                partition = connected_junction_partition(search,interval.left,interval.right,
                                    remaining_cells=self.maximum_search_cells-scan.visited_cells-ownership_cells)
                            partition = partition or curved_ownership_partition(
                                search, interval.left, interval.right,
                                remaining_cells=(self.maximum_search_cells
                                                 -(search_cells if pooled is not None else scan.visited_cells)
                                                 -ownership_cells),
                                reference=reference_curves[source_index],
                                stiffness=self.stiffness,
                                band_budget=np.minimum(
                                    0.5*np.array((self.energy_tolerance,
                                                  self.force_tolerance,
                                                  self.tangent_tolerance))-boundary_error,
                                    np.array((self.energy_tolerance,
                                              self.force_tolerance,
                                              self.tangent_tolerance))-error,
                                ),
                            )
                            ownership_cells += partition.visited
                            moving_switches.extend(partition.switches)
                            junction_bands.extend(partition.bands)
                            source_pieces.extend((begin, stop, owner, False)
                                                 for begin, stop, owner in partition.pieces)
                            continue
                        straight_pieces = _straight_support_partitions(
                            current[source_index], target_positions,
                            self.radius, interval.left, interval.right,
                            min(self.distance_tolerance, self.feature_tolerance),
                        )
                        pieces = straight_pieces
                        if pieces is None and interval.target_cell is None:
                            pieces = _straight_join_partitions(
                                current[source_index], target_positions,
                                interval.left, interval.right,
                                min(self.distance_tolerance,
                                    self.feature_tolerance),
                            )
                            if pieces is None:
                                owner = _junction_endpoint_owner(
                                    current[source_index], target_positions,
                                    target_ids, interval.left, interval.right,
                                    min(self.distance_tolerance,
                                        self.feature_tolerance),
                                )
                                if owner is not None:
                                    pieces = ((interval.left, interval.right,
                                               owner),)
                            if pieces is None:
                                raise ValueError(
                                    "native whole pair target ownership is unresolved: "
                                    f"source cell {source_index}, interval "
                                    f"[{interval.left}, {interval.right}]"
                                )
                        elif pieces is None:
                            pieces = ((interval.left, interval.right,
                                       interval.target_cell),)
                        source_pieces.extend(
                            (begin, stop, owner, straight_pieces is not None)
                            for begin, stop, owner in pieces
                        )
                search_cells += ownership_cells
                if _source_domains is not None:
                    domain_left, domain_right = _source_domains[source_cell]
                    source_pieces = [(max(left, domain_left), min(right, domain_right), owner, active)
                                     for left, right, owner, active in source_pieces
                                     if min(right, domain_right) > max(left, domain_left)]
                    moving_switches = [switch for switch in moving_switches
                                       if domain_left < switch[0] < domain_right]
                if self.certify_curved_ownership:
                    source_pieces = _joined_owned_pieces(source_pieces)
                for begin, stop, certified_owner, certified_active in source_pieces:
                    if _source_domains is not None:
                        from .native_retained_body_projection import endpoint_projection_bands
                        bands, cells = endpoint_projection_bands(
                            current[source_index], current[target_base+certified_owner],
                            begin, stop, maximum_cells=self.maximum_search_cells-search_cells,
                        )
                        search_cells += cells
                        segments = []
                        cursor = begin
                        for band_left, band_right in bands:
                            if cursor < band_left:
                                segments.append((cursor, band_left))
                            bound = _exact_tiny_cell_omission_bound(
                                current[source_index], current[target_base+certified_owner],
                                reference_curves[source_index], self.radius,
                                self.stiffness, band_left, band_right,
                            )
                            if bound is None or not np.all(np.isfinite(bound)) or np.any(bound < 0):
                                raise MemoryError('native body endpoint projection strip enclosure unresolved')
                            # Both one-sided constrained Hessians are enclosed
                            # by the existing strict-convex strip bound. Their
                            # envelope gradient agrees at the endpoint, hence
                            # no moving-boundary Hessian is added. Charge the
                            # complete physical30 strip before omitting it.
                            bound = np.array([_up(Fraction(float(a))*b)
                                              for a,b in zip(bound,(1,2,4))])
                            charged = np.array([_up(Fraction(float(a))+Fraction(float(b)))
                                                for a,b in zip(error,bound)])
                            boundary_charged = np.array([_up(Fraction(float(a))+Fraction(float(b)))
                                                         for a,b in zip(boundary_error,bound)])
                            limits = np.array((self.energy_tolerance,self.force_tolerance,self.tangent_tolerance))
                            if np.any(charged > limits) or np.any(boundary_charged > .5*limits):
                                raise MemoryError('native body endpoint projection strip reserve exhausted')
                            visited += 1
                            if visited > self.maximum_integration_intervals:
                                raise MemoryError('native body endpoint projection strip integration budget exhausted')
                            error[:],boundary_error[:] = charged,boundary_charged
                            omissions += 1
                            if _diagnostics is not None:
                                _diagnostics.setdefault('projection_strips',[]).append(
                                    dict(source=source_index,target=target_base+certified_owner,
                                         interval=(band_left,band_right),bound=bound.copy()))
                            cursor = band_right
                        if cursor < stop:
                            segments.append((cursor,stop))
                        for start,end in segments:
                            share = .125*(end-start)
                            budget = np.array((self.energy_tolerance*share,
                                               self.force_tolerance*share,self.tangent_tolerance*share))
                            total += .5*integrate(start,end,certified_owner,budget,certified_active)
                        continue
                    if certified_owner not in projection_cuts:
                        projection_cuts[certified_owner] = (
                            _endpoint_projection_cuts(
                                current[source_index],
                                current[target_base+certified_owner],
                            )
                        )
                    cuts = (begin, *(value for value in
                                     projection_cuts[certified_owner]
                                     if (begin+64.0*np.finfo(float).eps
                                         < value < stop-64.0*np.finfo(float).eps)),
                            stop)
                    for start, end in zip(cuts[:-1], cuts[1:]):
                        # Reserve half of each global error control for the
                        # new boundary/root uncertainty; defaults retain their
                        # historical allocation exactly.
                        share = (0.125 if self.certify_curved_ownership else 0.25)*(end-start)
                        budget = np.array((self.energy_tolerance*share,
                                           self.force_tolerance*share,
                                           self.tangent_tolerance*share))
                        total += 0.5*integrate(
                            start, end, certified_owner, budget,
                            certified_active,
                        )

                for begin, stop, bulk_bound, root_h_bound in junction_bands:
                    visited += 1
                    if visited > self.maximum_integration_intervals:
                        raise MemoryError("curved junction band integration budget exhausted")
                    if _source_domains is not None or pooled is not None:
                        from .native_retained_body_projection import charged_junction_omission
                        limits = np.array((self.energy_tolerance,self.force_tolerance,self.tangent_tolerance))
                        bound,charged,boundary_charged = charged_junction_omission(
                            bulk_bound,root_h_bound,error,boundary_error,limits)
                        error[:],boundary_error[:] = charged,boundary_charged
                        omissions += 1
                        intervals += 1
                        if _diagnostics is not None:
                            _diagnostics.setdefault('junction_omissions',[]).append(
                                dict(source=source_index,interval=(begin,stop),bound=bound.copy()))
                        # Complete half-weighted bulk and root-H bounds cover
                        # the zero approximation in physical30. Native work
                        # error amplification occurs in the body adapter.
                        continue
                    midpoint = 0.5*(begin+stop)
                    # A bounded unresolved band uses an explicit nominal
                    # branch only for its returned midpoint approximation;
                    # full bulk+returned-density bounds below cover either
                    # branch. Proven integration pieces keep their owner.
                    nominal_owner = (search.nearest(midpoint).target_cell
                                     if _source_domains is not None else None)
                    density, _ = integrand(midpoint, nominal_owner)
                    approximation = 0.5*(stop-begin)*density
                    magnitude = np.array((
                        abs(approximation[0]),
                        float(np.max(np.abs(point_map.T @ approximation[1:37]))),
                        float(np.max(np.abs(point_map.T
                                             @ approximation[37:].reshape(36, 36)
                                             @ point_map))),
                    ))
                    # Bound the full true integral and the absolute returned
                    # approximation BEFORE cancellation; all possible root-H
                    # is covered too. The existing half reserve is unchanged.
                    bound = (bulk_bound+magnitude)*(1.0+4096.0*np.finfo(float).eps)
                    bound[2] += root_h_bound
                    tolerances = np.array((self.energy_tolerance, self.force_tolerance,
                                           self.tangent_tolerance))
                    if ((np.any(boundary_error+bound > 0.5*tolerances) if pooled is None else not within_reserve(boundary_error,bound,tolerances,half=True))
                            or (np.any(error+bound > tolerances) if pooled is None else not within_reserve(error,bound,tolerances))):
                        raise MemoryError(
                            "curved junction band error budget exhausted: "
                            f"source {source_index}, band [{begin}, {stop}], "
                            f"bound {bound.tolist()}"
                        )
                    total += approximation
                    boundary_error[:] = boundary_error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(boundary_error,bound)])
                    error[:] = error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,bound)])
                    intervals += 1

                for (station, left_owner, right_owner, uncertainty,
                     slope_lower, convexity) in moving_switches:
                    # Energy densities agree at the switch, so the residual
                    # has no boundary term. Their gradients differ for remote
                    # projections: a_q=-Delta(d_q)/D_s produces this Hessian.
                    # The directional half weight is applied exactly here.
                    point, source_tangent, _ = _curve(current[source_index], station)
                    target_indices = (target_base+left_owner, target_base+right_owner)
                    projections = tuple(closest_on_b3(point, current[index])
                                        for index in target_indices)
                    points = tuple(_curve(current[index], projection.minimizers[0])[0]
                                   for index, projection in zip(target_indices, projections))
                    jets = tuple(b3_distance_jet(
                        current[source_index], current[index], station, self.radius,
                        source_is_a=True,
                    ) for index in target_indices)
                    delta_gradient = np.zeros(36)
                    for sign, index, jet in zip((1.0, -1.0), target_indices, jets):
                        indices = np.r_[np.arange(9*source_index, 9*source_index+9),
                                        np.arange(9*index, 9*index+9)]
                        delta_gradient[indices] += sign*jet.gradient
                    if float(np.linalg.norm(points[0]-points[1])) <= search.roundoff:
                        material_shared = all(
                            projection.minimizers[0] == (1.0 if index == target_base else 0.0)
                            for index, projection in zip(target_indices, projections)
                        )
                        mapped_delta = jacobian.T @ delta_gradient
                        if (not material_shared
                                or float(np.linalg.norm(mapped_delta))
                                > 2048.0*np.finfo(float).eps*max(1.0, map_norm)):
                            raise ValueError("coincident remote curved projections have different material maps")
                        continue
                    if abs(jets[0].gap-jets[1].gap) > self.distance_tolerance:
                        raise MemoryError("curved pair switch distance root unresolved")
                    gap = 0.5*(jets[0].gap+jets[1].gap)
                    if gap >= 0.0:
                        continue
                    if not uncertainty < station < 1.0-uncertainty:
                        raise ValueError("active remote curved switch at source endpoint")
                    normals = tuple((point-projected)/projection.distance
                                    for projected, projection in zip(points, projections))
                    slope = float((normals[0]-normals[1])@source_tangent)
                    if slope <= 2048.0*np.finfo(float).eps*max(
                            1.0, float(np.linalg.norm(source_tangent))):
                        raise ValueError("curved pair switch is not transversal")
                    measure = float(np.linalg.norm(
                        _curve(reference_curves[source_index], station)[1],
                    ))
                    total[37:].reshape(36, 36)[:] += (
                        -0.5*self.stiffness*measure*gap/slope
                        *np.outer(delta_gradient, delta_gradient)
                    )
                    bound = moving_boundary_error_bound(
                        current[source_index], tuple(current[index] for index in target_indices),
                        reference_curves[source_index], station,
                        projections, delta_gradient, gap, slope, uncertainty,
                        slope_lower, convexity, self.stiffness, search.roundoff,
                        point_map,
                    )
                    tolerances = np.array((self.energy_tolerance, self.force_tolerance,
                                           self.tangent_tolerance))
                    if ((np.any(boundary_error+bound > 0.5*tolerances) if pooled is None else not within_reserve(boundary_error,bound,tolerances,half=True))
                            or (np.any(error+bound > tolerances) if pooled is None else not within_reserve(error,bound,tolerances))):
                        raise MemoryError(
                            "curved pair moving-boundary error budget exhausted: "
                            f"source {source_index}, root {station}, bound {bound.tolist()}, "
                            f"root uncertainty {uncertainty}, slope {slope}, "
                            f"slope lower {slope_lower}"
                        )
                    boundary_error[:] = boundary_error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(boundary_error,bound)])
                    error[:] = error+bound if _source_domains is None and pooled is None else np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(error,bound)])
        if self.certify_curved_ownership and np.any(error > np.array((
                self.energy_tolerance, self.force_tolerance, self.tangent_tolerance))):
            raise MemoryError("curved pair aggregate error budget exhausted")
        if pooled is not None:
            limits = (self.energy_tolerance,self.force_tolerance,self.tangent_tolerance)
            if not within_reserve(boundary_error,(0.,0.,0.),limits,half=True):
                raise MemoryError("pooled whole-pair boundary reserve exhausted")
        gradient = total[1:37]
        hessian = total[37:].reshape(36, 36)
        if _diagnostics is not None:
            if pooled is not None:_diagnostics['pooled_quadrature'] = pooled.report()
            _diagnostics.update(error=error.copy(),boundary_error=boundary_error.copy(),
                half_reserve=.5*np.array((self.energy_tolerance,self.force_tolerance,self.tangent_tolerance)),
                visited=visited,omissions=omissions,search_cells=search_cells,evaluations=evaluations)
        if _geometry_result:
            return float(total[0]), gradient, hessian, error.copy(), (search_cells, intervals, evaluations, visited, omissions, unresolved)
        residual = jacobian.T @ gradient
        tangent = jacobian.T @ hessian @ jacobian
        for global_cell, state in enumerate(states):
            _, curvature = state.pullback(
                gradient[9*global_cell:9*global_cell+9],
                np.zeros((9, 9)),
            )
            side = 0 if global_cell < 2 else 1
            tangent[42*side:42*side+42,
                    42*side:42*side+42] += curvature
        physical = -gradient.reshape(4, 3, 3)
        force_a = np.sum(physical[:2], axis=(0, 1))
        force_b = np.sum(physical[2:], axis=(0, 1))
        moment = np.array([
            np.sum(np.cross(current[cell], physical[cell]), axis=0)
            for cell in range(4)
        ])
        return NativeRetainedWholePairResult(
            float(total[0]), _readonly(residual), _readonly(tangent),
            _readonly(force_a), _readonly(force_b),
            _readonly(np.sum(moment[:2], axis=0)),
            _readonly(np.sum(moment[2:], axis=0)),
            conservative_lower, minimum_sampled,
            float(error[0]), float(error[1]), float(error[2]),
            search_cells, intervals, evaluations, visited, omissions, unresolved,
        )
