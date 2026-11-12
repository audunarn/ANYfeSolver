"""Conservative two-cell nearest-feature search without dropped contact."""

import numpy as np
import pytest

from anyfesolver.general_contact.native_curve_union import NativeTwoCellUnionSearch


def _line(first, last, y):
    return np.array(((first, y, 0.0),
                     (0.5*(first+last), y, 0.0),
                     (last, y, 0.0)))


def _search(source=None, *, target_y=0.08, radius=0.1,
            maximum_cells=10000):
    if source is None:
        source = _line(0.0, 1.0, 0.0)
    return NativeTwoCellUnionSearch(
        source, (_line(0.0, 0.5, target_y),
                 _line(0.5, 1.0, target_y)),
        ((1, 2), (2, 3)),
        combined_radius_m=radius,
        distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9,
        parameter_tolerance=1e-3,
        maximum_depth=20,
        maximum_cells=maximum_cells,
    )


def _covering(scan, station):
    return next(interval for interval in scan.intervals
                if interval.left <= station <= interval.right)


def test_shared_junction_switch_and_conservative_contact_support():
    search = _search()
    assert search.nearest(0.25).target_cell == 0
    assert search.nearest(0.75).target_cell == 1
    middle = search.nearest(0.5)
    assert middle.shared_junction_tie and middle.target_cell == 0
    scan = search.scan()
    assert scan.unresolved_intervals > 0
    assert any(item.status == "active" and item.target_cell == 0
               for item in scan.intervals)
    assert any(item.status == "active" and item.target_cell == 1
               for item in scan.intervals)
    assert scan.conservative_lower_gap_m <= -0.02
    for station in np.linspace(0.0, 1.0, 101):
        if search.nearest(float(station)).gap_m < 0.0:
            assert _covering(scan, station).status != "open"


@pytest.mark.parametrize('certify_open_intervals',(False,True))
def test_open_and_narrow_contact_pocket_are_not_dropped(certify_open_intervals):
    clear_search = _search(target_y=0.3)
    clear_search.certify_open_intervals=certify_open_intervals
    clear = clear_search.scan()
    assert all(item.status == "open" for item in clear.intervals)
    source = np.array(((0.0, 0.3, 0.0),
                       (0.5, 0.09, 0.0),
                       (1.0, 0.3, 0.0)))
    search = _search(source, target_y=0.0)
    search.certify_open_intervals=certify_open_intervals
    scan = search.scan()
    assert search.nearest(0.5).gap_m < 0.0
    assert _covering(scan, 0.5).status != "open"
    assert _covering(scan, 0.0).status == "open"
    assert _covering(scan, 1.0).status == "open"
    for station in np.linspace(0.0, 1.0, 201):
        if search.nearest(float(station)).gap_m < 0.0:
            assert _covering(scan, station).status != "open"


def test_open_kink_junction_certified_against_independent_segments():
    source=_line(0.,1.,0.)
    points=np.array(((0.,.101,0.),(1.,.1002,0.),(2.,.099,0.)))
    targets=tuple(np.vstack((a,.5*(a+b),b)) for a,b in zip(points[:-1],points[1:]))
    search=NativeTwoCellUnionSearch(source,targets,((1,2),(2,3)),
        combined_radius_m=.1,distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9,parameter_tolerance=1e-3,
        maximum_depth=20,maximum_cells=10000,certify_open_intervals=True)
    scan=search.scan()
    assert all(item.status=='open' for item in scan.intervals)
    assert scan.conservative_lower_gap_m>0.
    for x in np.linspace(0.,1.,101):
        point=np.array((x,0.,0.))
        distances=[]
        for a,b in zip(points[:-1],points[1:]):
            v=b-a;t=np.clip((point-a)@v/(v@v),0.,1.)
            distances.append(np.linalg.norm(point-a-t*v))
        gap=min(distances)-.1
        assert gap>0. and _covering(scan,x).lower_gap_m<=gap


def test_curved_open_interval_bound_against_exact_quadratic_ordinate():
    source=np.array(((0.,.1001,0.),(.5,.1002,0.),(1.,.1001,0.)))
    search=NativeTwoCellUnionSearch(source,(_line(0.,.5,0.),_line(.5,1.,0.)),
        ((1,2),(2,3)),combined_radius_m=.1,distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9,parameter_tolerance=1e-3,
        maximum_depth=20,maximum_cells=10000,certify_open_intervals=True)
    scan=search.scan()
    assert all(item.status=='open' for item in scan.intervals)
    search.certify_open_intervals=False
    assert search.scan().unresolved_intervals>0
    for interval in scan.intervals:
        # The nearest segment point is (s,0,0); this concave quadratic
        # has its interval minimum at an endpoint, independently of search.
        exact=min(.1001+.0004*s*(1-s)-.1 for s in (interval.left,interval.right))
        assert 0.<interval.lower_gap_m<=exact


def test_global_chord_bound_preserves_active_parallel_gap():
    search=_search();search.certify_open_intervals=True
    scan=search.scan()
    assert -0.0200000001<=scan.conservative_lower_gap_m<=-.02
    assert scan.unresolved_intervals>0
    for station in np.linspace(0.,1.,101):
        assert _covering(scan,station).status!='open'


def test_remote_equal_minima_and_resource_exhaustion_fail_explicitly():
    source = _line(-0.2, 0.2, 1.0)
    target_a = np.array(((-1.0, 1.0, 0.0),
                         (-0.5, 0.5, 0.0),
                         (0.0, 0.0, 0.0)))
    target_b = np.array(((0.0, 0.0, 0.0),
                         (0.5, 0.5, 0.0),
                         (1.0, 1.0, 0.0)))
    ambiguous = NativeTwoCellUnionSearch(
        source, (target_a, target_b), ((1, 2), (2, 3)),
        combined_radius_m=0.8,
        distance_tolerance_m=1e-10,
        feature_tolerance_m=1e-9,
        parameter_tolerance=1e-3,
        maximum_depth=20,
        maximum_cells=10000,
    )
    with pytest.raises(ValueError, match="competing remote target minima"):
        ambiguous.nearest(0.5)
    with pytest.raises(MemoryError, match="cell budget exhausted"):
        _search(maximum_cells=1).scan()
    with pytest.raises(ValueError, match="shared material junction identity"):
        NativeTwoCellUnionSearch(
            _line(0.0, 1.0, 0.0),
            (_line(0.0, 0.5, 0.08), _line(0.5, 1.0, 0.08)),
            ((1, 2), (6, 3)),
            combined_radius_m=0.1,
            distance_tolerance_m=1e-10,
            feature_tolerance_m=1e-9,
            parameter_tolerance=1e-3,
            maximum_depth=20,
            maximum_cells=10000,
        )

def test_source_interval_gap_lower_matches_original_polynomial_oracle():
    """The local exact-coefficient interval bound reproduces the original
    numpy Polynomial arithmetic and fixed scalar reductions bit-exactly: identical conservative
    guards, identical threshold behavior across near-zero, parallel and
    endpoint geometries, and no sampled certificate."""
    from scipy.optimize import brentq
    from fractions import Fraction
    from anyfesolver.general_contact.beam_curve_pair import (
        _positions, _source_interval_target_gap_lower,
    )

    def oracle_roots(polynomial):
        coefficients = np.asarray(polynomial.coef, dtype=float)
        while len(coefficients) > 1 and coefficients[-1] == 0.0:
            coefficients = coefficients[:-1]
        if len(coefficients) <= 1:
            return ()
        normalized = np.polynomial.Polynomial(
            coefficients / float(np.max(np.abs(coefficients))),
        )
        critical = oracle_roots(normalized.deriv())
        stations = (0.0, *critical, 1.0)
        values = tuple(float(normalized(station)) for station in stations)
        evaluation_guard = 128.0 * np.finfo(float).eps * float(
            np.sum(np.abs(normalized.coef)),
        )
        roots = [station for station, value in zip(stations, values)
                 if abs(value) <= evaluation_guard]
        for left, right, value_left, value_right in zip(
            stations[:-1], stations[1:], values[:-1], values[1:],
        ):
            if value_left * value_right < 0.0:
                roots.append(float(brentq(
                    normalized, left, right, xtol=1e-14, rtol=1e-14,
                )))
        return tuple(sorted(set(float(np.clip(root, 0.0, 1.0))
                                for root in roots)))

    def oracle_squared_length(vector):
        # Independent exact-arithmetic oracle for the production contract's
        # ascending, singly rounded products/additions. A host-dispatched
        # BLAS norm is not an exact oracle for that specified operation order.
        result = 0.
        for component in vector:
            value = Fraction(float(component))
            result = float(value*value + Fraction(result))
        return result

    def oracle_bound(source, left, right, target, radius):
        from anyfesolver.general_contact.beam_curve_pair import _curve
        start = _curve(source, left)[0]
        end = _curve(source, right)[0]
        middle = _curve(source, 0.5 * (left + right))[0]
        chord_deviation = float(np.sqrt(oracle_squared_length(middle - 0.5 * (start + end))))
        vector = end - start
        squared_length = oracle_squared_length(vector)
        if squared_length <= 0.0:
            raise MemoryError("B3 pair source interval chord is degenerate")
        polynomial = np.polynomial.Polynomial
        target_curve = tuple(polynomial((
            target[0, axis],
            -3.0*target[0, axis] + 4.0*target[1, axis] - target[2, axis],
            2.0*target[0, axis] - 4.0*target[1, axis] + 2.0*target[2, axis],
        )) for axis in range(3))
        offset = tuple(component - float(start[axis])
                       for axis, component in enumerate(target_curve))
        along = sum((component*float(vector[axis])
                     for axis, component in enumerate(offset)),
                    polynomial((0.0,)))
        projection = along / squared_length
        start_distance = sum((component*component for component in offset),
                             polynomial((0.0,)))
        end_distance = sum(((component - float(end[axis]))**2
                             for axis, component in enumerate(target_curve)),
                            polynomial((0.0,)))
        interior_distance = start_distance - along*along/squared_length
        cuts = (0.0, *sorted(set((
            *oracle_roots(projection),
            *oracle_roots(projection - 1.0),
        ))), 1.0)
        minimum_squared = np.inf
        coefficient_scale = 0.0
        for first, last in zip(cuts[:-1], cuts[1:]):
            if last - first <= 1.0e-14:
                continue
            projected = float(projection(0.5*(first + last)))
            branch = (start_distance if projected <= 0.0 else
                      end_distance if projected >= 1.0
                      else interior_distance)
            coefficient_scale = max(
                coefficient_scale, float(np.sum(np.abs(branch.coef))),
            )
            stations = (first, *oracle_roots(branch.deriv()), last)
            minimum_squared = min(
                minimum_squared,
                *(float(branch(station)) for station in stations
                  if first <= station <= last),
            )
        if not np.isfinite(minimum_squared):
            raise MemoryError("B3 pair interval distance could not be bounded")
        squared_guard = 1024.0 * np.finfo(float).eps * max(
            coefficient_scale, radius*radius, np.finfo(float).tiny,
        )
        position_guard = 1024.0 * np.finfo(float).eps * max(
            float(np.max(np.abs(source))), float(np.max(np.abs(target))),
            radius,
        )
        return (float(np.sqrt(max(0.0, minimum_squared - squared_guard)))
                - radius - chord_deviation - position_guard)

    rng = np.random.default_rng(20261011)
    cases = [
        # near-parallel straight centrelines at small normal clearance
        (np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.))),
         np.array(((0., 1e-7, 0.), (.5, 1e-7, 0.), (1., 1e-7, 0.))), .01),
        # curved near-parallel centrelines
        (np.array(((0., 0., 0.), (.5, .1, 0.), (1., 0., 0.))),
         np.array(((0., 1e-4, 0.), (.5, .1 + 1e-4, 0.), (1., 1e-4, 0.))),
         .01),
        # endpoint-close configuration
        (np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.))),
         np.array(((1. + 1e-9, 0., 0.), (1.5, .2, 0.), (2., .5, 0.))), .05),
        # near-touching straight pair, bound crosses zero near the threshold
        (np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.))),
         np.array(((0., .02 - 1e-9, 0.), (.5, .02 - 1e-9, 0.),
                   (1., .02 - 1e-9, 0.))), .02),
        # crossing centrelines
        (np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.))),
         np.array(((.5, -1., 0.), (.5, 0., 0.), (.5, 1., 0.))), .05),
        # large coordinates with tiny radius
        (np.array(((1e3, 0., 0.), (1e3 + 5, 1., 0.), (1e3 + 10, 0., 0.))),
         np.array(((1e3, .05, 0.), (1e3 + 5, 1.05, 0.),
                   (1e3 + 10, .05, 0.))), 1e-4),
    ]
    cases.extend((np.column_stack([rng.uniform(-2, 2, 3) for _ in range(3)]),
                  np.column_stack([rng.uniform(-2, 2, 3) for _ in range(3)]),
                  float(rng.uniform(.01, .2))) for _ in range(24))
    intervals = ((0., 1.), (0., .5), (.5, 1.), (.25, .75),
                 (.125, .375), (0., 1e-9), (.999999, 1.))
    compared = 0
    for source, target, radius in cases:
        try:
            source = _positions(source, "source")
            target = _positions(target, "target")
        except ValueError:
            continue
        for left, right in intervals:
            expected = oracle_bound(source, left, right, target, radius)
            actual = _source_interval_target_gap_lower(
                source, left, right, target, radius)
            assert expected == actual and repr(expected) == repr(actual)
            compared += 1
    assert compared >= 200
    # A degenerate chord fails closed through both arithmetics.
    source = _positions(np.array(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.))),
                        "source")
    target = _positions(np.array(((0., .08, 0.), (.5, .08, 0.),
                                  (1., .08, 0.))), "target")
    with pytest.raises(MemoryError):
        oracle_bound(source, .3, .3, target, .1)
    with pytest.raises(MemoryError):
        _source_interval_target_gap_lower(source, .3, .3, target, .1)

def test_coef_sub_helper_matches_numpy_polyutils_sub_exactly():
    """Focused parity for the local subtraction helper against numpy
    polyutils._sub: a not-shorter second operand is negated wholesale,
    so its trailing coefficients carry the negated sign, a mid-series
    zero becomes -0.0 and survives the final trim, cancellation trims
    to the first element, and a longer first operand keeps its
    trailing coefficients unchanged."""
    import math
    from numpy.polynomial import polyutils as pu
    from anyfesolver.general_contact.beam_curve_pair import _coef_sub

    # the repaired defect direction, as an explicit canary
    assert _coef_sub((1.0,), (2.0, 3.0)) == (-1.0, -3.0)
    cases = [
        # shorter first operand: trailing second coefficients negated
        ((1.0,), (2.0, 3.0)),
        ((1.0, 2.0), (3.0, 4.0, 5.0)),
        ((0.0,), (0.0, 4.0, 0.0, 5.0)),
        # cancellation and trailing zeros trim to the first element
        ((1.0, 1.0), (1.0, 1.0)),
        ((5.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        ((2.0, -2.0, 0.0), (2.0, -2.0)),
        # signed zero: negation keeps -0.0 mid-series after the trim
        ((1.0,), (1.0, 0.0, 5.0)),
        ((0.0,), (0.0, 0.0, 2.0)),
        ((-0.0,), (-0.0, 0.0, 3.0)),
        ((1.0,), (-0.0, 0.0, -0.0, 7.0)),
        ((0.0,), (-0.0,)),
        ((-0.0,), (0.0,)),
        # longer first operand keeps its trailing coefficients
        ((1.0, 0.0, 6.0), (2.0,)),
        ((0.0, 0.0, 4.0), (0.0,)),
        # equal lengths
        ((1.0, -2.0), (3.0, 4.0)),
    ]
    rng = np.random.default_rng(20261011)
    cases.extend((tuple(rng.uniform(-4.0, 4.0, int(rng.integers(1, 6)))),
                  tuple(rng.uniform(-4.0, 4.0, int(rng.integers(1, 7)))))
                 for _ in range(200))
    for first, second in cases:
        expected = pu._sub(np.asarray(first, dtype=float),
                           np.asarray(second, dtype=float))
        actual = _coef_sub(first, second)
        assert len(actual) == len(expected), (first, second)
        for produced, reference in zip(actual, expected):
            assert produced == reference, (first, second)
            assert math.copysign(1.0, produced) == math.copysign(
                1.0, reference), (first, second)
