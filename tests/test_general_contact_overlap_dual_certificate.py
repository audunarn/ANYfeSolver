"""Independent controls for the opt-in overlap-constrained bound candidate.

No pressure, optimizer, sampled safety criterion or altered projection law.
"""
from fractions import Fraction

import numpy as np
import pytest

from anyfesolver.general_contact import q4_overlap_local_bound as bounds


def controls(values, count):
    return tuple((np.asarray(values, float).copy(),
                  np.asarray(values, float).copy()) for _ in range(count))


def constrained_controls():
    # N = b - 1 on the box: b>=0 implies N>=-1, whereas the
    # unrestricted rectangle has minimum -2. Both other constraints are 1.
    n = controls((-2., 0., 0., -2.), 4)
    b = controls((-1., 1., 1., -1.), 5)
    one = controls((1., 1., 1., 1.), 5)
    return n, (b, one, one)


def test_cubic_to_quartic_exact_rational_alignment():
    cubic = tuple(bounds._interval(np.full(4, value)) for value in (0., 0., 0., 1.))
    elevated = bounds._elevate_cubic_time_interval(cubic)
    expected = (0., 0., 0., .25, 1.)
    for coefficient, exact in zip(elevated, expected):
        assert np.all(coefficient[0] <= exact) and np.all(coefficient[1] >= exact)
    # Independent rational evaluation of t**3 vs the quartic polynomial.
    from math import comb
    for t in (Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(1)):
        value = sum(Fraction(expected[k]) * comb(4, k) * t**k * (1-t)**(4-k)
                    for k in range(5))
        assert value == t**3


def test_one_fixed_vector_validates_all_joint_controls():
    n, bary = constrained_controls()
    aligned = bounds._elevate_cubic_time_interval(n)
    lower = bounds._overlap_dual_coefficient_lower(aligned, bary, (1., 0., 0.))
    assert -1.00000000000001 < lower <= -1.
    # Exact positive-overlap values, including boundary b=0, use Q1 weights.
    for t in (Fraction(0), Fraction(1, 3), Fraction(1)):
        for x in (Fraction(1, 2), Fraction(3, 4), Fraction(1)):
            b = 2*x-1
            assert b >= 0 and b-1 >= Fraction(lower)
    found, checked = bounds._overlap_dual_numerator_lower(n, bary)
    assert checked <= 570 and -1.00000000000001 < found <= -1.
    assert abs(found-lower) <= 32*np.finfo(float).eps
    # Put the useful constraint in each other axis: this reaches NONZERO
    # axis1/axis2 searches, rather than only checking generic helper syntax.
    for axis in (1, 2):
        permuted = list(bary)
        permuted[0], permuted[axis] = permuted[axis], permuted[0]
        value, attempts = bounds._overlap_dual_numerator_lower(n, tuple(permuted))
        assert attempts > 0 and -1.00000000000001 < value <= -1.


def test_nonzero_batched_rows_match_rational_and_scalar_validation():
    n, bary = constrained_controls()
    aligned = bounds._elevate_cubic_time_interval(n)
    flat_n = np.concatenate([x[0] for x in aligned])
    flat_b = np.concatenate([x[1] for x in bary[0]])
    weights = (.25, .5, 1., 2., 3.)
    batched = bounds._overlap_dual_axis_lower(flat_n, flat_b, weights)
    exact_rows = [min(Fraction(float(a))-Fraction(w)*Fraction(float(b))
                      for a, b in zip(flat_n, flat_b)) for w in weights]
    assert Fraction(batched) <= max(exact_rows)
    scalar = max(bounds._overlap_dual_coefficient_lower(aligned, bary, (w, 0., 0.))
                 for w in weights)
    # Both independently round outward; batching the lower/upper endpoints
    # may add ULPs. This parity check is not an admission tolerance: the exact
    # rational inequality above remains the proof obligation.
    assert abs(batched-scalar) <= 64*np.finfo(float).eps*max(1., abs(scalar))
    assert batched > -1.00000000000001
    assert bounds._overlap_dual_axis_lower(flat_n, flat_b, (-1., np.nan, np.inf)) == -np.inf
    assert bounds._overlap_dual_axis_lower(np.zeros(20), np.full(20, 1e308), (1e308,)) == -np.inf


def test_coefficient_varying_multiplier_counterexample_is_not_admitted():
    n, bary = constrained_controls()
    aligned = bounds._elevate_cubic_time_interval(n)
    # Review counterexample: n=(-2,0), b=(-1,1), midpoint n=-1,b=0.
    # A varying lambda=(2,0) falsely makes both controls zero.
    assert Fraction(-2, 2) == -1
    assert np.array_equal(np.array((-2., 0.))-np.array((2., 0.))*np.array((-1., 1.)), (0., 0.))
    assert bounds._overlap_dual_coefficient_lower(aligned, bary, np.zeros((5, 4, 3))) == -np.inf
    for weight in (0., 1., 2.):
        assert bounds._overlap_dual_coefficient_lower(aligned, bary, (weight, 0., 0.)) <= -1.


@pytest.mark.parametrize('weights', ((-1., 0., 0.), (np.nan, 0., 0.),
                                   (np.inf, 0., 0.), (1., 0.)))
def test_invalid_multiplier_has_no_bound(weights):
    n, bary = constrained_controls()
    assert bounds._overlap_dual_coefficient_lower(
        bounds._elevate_cubic_time_interval(n), bary, weights) == -np.inf


def test_uncertainty_consumes_upper_barycentric_endpoint():
    n = controls((0., 0., 0., 0.), 5)
    uncertain = tuple((np.full(4, -1.), np.full(4, 1.)) for _ in range(5))
    zero = controls((0., 0., 0., 0.), 5)
    lower = bounds._overlap_dual_coefficient_lower(n, (uncertain, zero, zero), (1., 0., 0.))
    assert lower <= -1.  # subtracting the lower endpoint would falsely give +1
    broken = list(uncertain)
    broken[3] = (np.full(4, -1.), np.array((1., 1., np.nan, 1.)))
    assert bounds._overlap_dual_coefficient_lower(n, (tuple(broken), zero, zero), (0., 0., 0.)) == -np.inf
    with np.errstate(over='ignore', invalid='ignore'):
        huge = controls((1e308, 1e308, 1e308, 1e308), 5)
        assert bounds._overlap_dual_coefficient_lower(n, (huge, zero, zero), (1e308, 0., 0.)) == -np.inf


def planar_cut():
    source = np.array(((0, 0, -.5), (1, 0, .5), (1, 1, .5), (0, 1, -.5)), float)
    target = np.array(((.5, 0, 0), (2, 0, 0), (.5, 2, 0)), float)
    # R=(-1,0,1), hit_x=x+z=2x-.5. Target requires hit_x>=.5,
    # hence x>=.5 and source-normal gap >=0 throughout represented overlap.
    return source, target


def test_analytical_cut_improves_without_changing_parent_ray():
    source, target = planar_cut()
    original, original_nodes = bounds.matching_projection_swept_lower(source, source, target, target, 0)
    details = {}
    improved, nodes = bounds.matching_projection_swept_lower(
        source, source, target, target, 0, overlap_dual=True, diagnostics=details)
    assert original < -.5 and original_nodes == nodes == 1
    assert original <= improved <= 0. and improved > -1e-12
    assert details['dual_improved_boxes'] == 1 and details['dual_candidates_checked'] <= 570


def test_default_false_and_disjoint_boundary_interpretation():
    source, target = planar_cut()
    assert bounds.matching_projection_swept_lower(source, source, target, target, 3) == bounds.matching_projection_swept_lower(
        source, source, target, target, 3, overlap_dual=False)
    disjoint = target + (10., 0., 0.)
    assert bounds.matching_projection_swept_lower(source, source, disjoint, disjoint, 0, overlap_dual=True)[0] == np.inf
    # Zero barycentric boundary is not strictly separated by the prune test.
    touching = np.array(((1.5, 0, 0), (2.5, 0, 0), (1.5, 1, 0)), float)
    assert bounds.matching_projection_swept_lower(source, source, touching, touching, 0, overlap_dual=True)[0] != np.inf
    with pytest.raises(ValueError, match='overlap dual needs a bool'):
        bounds.matching_projection_swept_lower(source, source, target, target, overlap_dual=1)


def test_error_enclosure_and_actual_penetration_still_refuse():
    source = np.array(((0, 0, -1e-4), (1, 0, -1e-4), (1, 1, -1e-4), (0, 1, -1e-4)), float)
    target = np.array(((-4, -4, 0), (8, -4, 0), (-4, 8, 0)), float)
    error = np.zeros((4, 3)); error[:, 2] = 1e-3
    for errors in (None, (error, error, 0., 0.)):
        lower, visited = bounds.matching_projection_swept_lower(source, source, target, target, 3,
            gap_budget=5e-5, adaptive_depth=7, node_budget=1024,
            endpoint_errors=errors, overlap_dual=True)
        assert lower <= -1e-4 and visited <= 1024
    assert bounds.matching_projection_swept_lower(source, source, target[[0, 2, 1]], target[[0, 2, 1]],
        overlap_dual=True)[0] == -np.inf


def test_hidden_time_penetration_still_refuses():
    source = np.array(((-.01, -.01, 0), (.01, -.01, 0), (.01, .01, 0), (-.01, .01, 0)), float)
    first = np.array(((-3, 0, 1.1), (1, -10, -.9), (1, 10, -.9)), float)
    last = np.array(((-1, 0, -.9), (3, -10, 1.1), (3, 10, 1.1)), float)
    lower, nodes = bounds.matching_projection_swept_lower(source, source, first, last, 3,
        gap_budget=5e-5, adaptive_depth=7, node_budget=1024, overlap_dual=True)
    assert lower <= -.09 and nodes <= 1024


def test_full_floor_and_partial_exhaustion_keep_valid_parent():
    source, target = planar_cut()
    floor, _ = bounds.matching_projection_swept_lower(source, source, target, target, 3)
    improved, nodes = bounds.matching_projection_swept_lower(source, source, target, target, 3,
        refinement_gap_budget=5e-5, adaptive_depth=7, node_budget=1024, overlap_dual=True)
    assert floor <= improved <= 0. and nodes <= 1024
    penetrating = source.copy(); penetrating[:, 2] -= 1.
    parent, _ = bounds.matching_projection_swept_lower(penetrating, penetrating, target, target, 0)
    refused, nodes = bounds.matching_projection_swept_lower(penetrating, penetrating, target, target, 3,
        gap_budget=5e-5, adaptive_depth=7, node_budget=1, overlap_dual=True)
    assert refused == parent and nodes == 1 and refused < -5e-5
    # This is root exhaustion only. Existing partial-child parent-retention
    # coverage is inherited by impact; no partial-child result is claimed here.
