"""Independent controls for the opt-in joint overlap-constrained dual.

No pressure, optimizer, sampled safety criterion or altered projection
law. One constant nonnegative three-component weight vector applies to
all aligned time/spatial controls of a represented box; every proposal
is revalidated with directed outward arithmetic.
"""
from fractions import Fraction

import numpy as np

from anyfesolver.general_contact import q4_overlap_local_bound as bounds


def controls(values, count):
    return tuple((np.asarray(values, float).copy(),
                  np.asarray(values, float).copy()) for _ in range(count))


def joint_only_controls():
    # 20 joint rows (5 time controls x 4 box corners). The numerator
    # minimum -2 sits at corners 0/1 of time control 0, where two
    # barycentric constraints are simultaneously provably negative; the
    # exact certificate-LP optimum -2/3 needs both components at 2/3.
    # Every single-axis weight is blocked at -2 by the respectively other
    # corner's zero constraint coefficient.
    numerator = controls((-2., -2., -1.5, 0.), 1)+controls((0., 0., 0., 0.), 3)
    u = controls((-2., 0., -1., .5), 1)+controls((0., 0., 0., 0.), 4)
    v = controls((0., -2., -1., .5), 1)+controls((0., 0., 0., 0.), 4)
    w = controls((0., 0., 0., 2.), 1)+controls((0., 0., 0., 0.), 4)
    return numerator, (u, v, w)


def coefficient_counterexample_controls():
    # N = b - 1 on the box: b>=0 implies N>=-1, whereas the
    # unrestricted rectangle has minimum -2. Both other constraints are 1.
    n = controls((-2., 0., 0., -2.), 4)
    b = controls((-1., 1., 1., -1.), 5)
    one = controls((1., 1., 1., 1.), 5)
    return n, (b, one, one)


def flat_controls(n, bary):
    aligned = bounds._elevate_cubic_time_interval(n)
    return (aligned,
            np.concatenate([x[0] for x in aligned]),
            np.stack([np.concatenate([x[1] for x in controls])
                      for controls in bary]))


def planar_cut():
    source = np.array(((0, 0, -.5), (1, 0, .5), (1, 1, .5), (0, 1, -.5)), float)
    target = np.array(((.5, 0, 0), (2, 0, 0), (.5, 2, 0)), float)
    return source, target


def test_vertex_enumeration_is_bounded_deterministic_and_feasible():
    n, bary = joint_only_controls()
    _, flat_n, flat_c = flat_controls(n, bary)
    sets = bounds._OVERLAP_DUAL_VERTEX_ACTIVE_SETS
    assert sets.shape == (8855, 4)  # choose(23, 4) gap and nonnegativity rows
    first = bounds._overlap_dual_joint_candidates(flat_n, flat_c)
    second = bounds._overlap_dual_joint_candidates(flat_n, flat_c)
    assert first == second and 0 < len(first) <= 8855
    weights = np.asarray(first, dtype=float)
    assert np.all(np.isfinite(weights)) and np.all(weights >= 0.)
    # The exact rational optimum vertex (2/3, 2/3, 0) is proposed with its
    # nonnegativity-active component pinned to exact zero.
    assert any(abs(u-Fraction(2, 3)) <= Fraction(1, 10**12)
               and abs(v-Fraction(2, 3)) <= Fraction(1, 10**12)
               and w == 0. for u, v, w in first)
    assert bounds._overlap_dual_joint_candidates(np.full(20, np.nan), flat_c) == []
    assert bounds._overlap_dual_joint_candidates(flat_n, np.zeros((3, 19))) == []


def test_joint_dual_dominates_every_single_axis_optimum():
    n, bary = joint_only_controls()
    found, checked = bounds._overlap_dual_numerator_lower(n, bary)
    assert checked <= 570 and -2.0000000000001 < found <= -2.
    joint, joint_checked = bounds._overlap_dual_joint_lower(n, bary)
    assert 0 < joint_checked <= 8855
    # Exact certificate-LP optimum is -2/3; directed rounding only widens.
    assert -2/3-1e-12 <= joint <= -2/3+1e-12
    assert joint > found+1.


def test_joint_dual_preserves_coefficient_varying_counterexample():
    n, bary = coefficient_counterexample_controls()
    found, _ = bounds._overlap_dual_numerator_lower(n, bary)
    joint, checked = bounds._overlap_dual_joint_lower(n, bary)
    assert checked > 0
    # A coefficient-varying multiplier would falsely reach 0 here; one
    # fixed triple on all 20 controls cannot beat the honest optimum -1.
    assert -1.00000000000001 < joint <= -1.
    assert -1.00000000000001 < found <= -1.
    assert joint-found <= 64*np.finfo(float).eps


def test_joint_batched_rows_match_rational_and_scalar_validation():
    n, bary = joint_only_controls()
    aligned, flat_n, flat_c = flat_controls(n, bary)
    weights = ((0., 0., 0.), (.25, .5, .25), (2/3, 2/3, 0.),
               (1., 1., 0.), (2., 0., 0.))
    batched = bounds._overlap_dual_joint_weight_lower(flat_n, flat_c, weights)
    exact = [min(Fraction(float(a))-Fraction(u)*Fraction(float(cu))
                 -Fraction(v)*Fraction(float(cv))-Fraction(w)*Fraction(float(cw))
                 for a, cu, cv, cw in zip(flat_n, flat_c[0], flat_c[1], flat_c[2]))
             for u, v, w in weights]
    assert all(Fraction(float(b)) <= e for b, e in zip(batched, exact))
    scalar = [bounds._overlap_dual_coefficient_lower(aligned, bary, weight)
              for weight in weights]
    # Both independently round outward; batching may add ULPs. This parity
    # check is not an admission tolerance: the exact rational inequality
    # above remains the proof obligation.
    assert all(abs(b-s) <= 64*np.finfo(float).eps*max(1., abs(s))
               for b, s in zip(batched, scalar))
    assert bounds._overlap_dual_joint_weight_lower(
        flat_n, flat_c, (-1., 0., 0.))[0] == -np.inf
    assert bounds._overlap_dual_joint_weight_lower(
        flat_n, flat_c, (np.nan, 0., 0.))[0] == -np.inf


def test_joint_dual_invalid_structures_refuse():
    n, bary = joint_only_controls()
    nonfinite = tuple((np.full(4, -1.), np.array((1., np.nan, 1., 1.)))
                      for _ in range(4))
    assert bounds._overlap_dual_joint_lower(nonfinite, bary) == (-np.inf, 0)
    assert bounds._overlap_dual_joint_lower(n, bary[:2]) == (-np.inf, 0)
    inverted = tuple((np.full(4, 1.), np.full(4, -1.)) for _ in range(4))
    assert bounds._overlap_dual_joint_lower(inverted, bary) == (-np.inf, 0)
    short = controls((0., 0., 0., 0.), 3)
    assert bounds._overlap_dual_joint_lower(short, bary) == (-np.inf, 0)


def test_joint_stage_is_budget_gated_and_reported_in_diagnostics():
    source, target = planar_cut()
    unbudgeted = {}
    lower, nodes = bounds.matching_projection_swept_lower(
        source, source, target, target, 0, overlap_dual=True,
        diagnostics=unbudgeted)
    assert lower > -1e-12 and nodes == 1
    assert unbudgeted['dual_improved_boxes'] == 1
    # No budget: the joint stage never runs; development calls unchanged.
    assert unbudgeted['dual_joint_candidates_checked'] == 0
    certified = {}
    lower, _ = bounds.matching_projection_swept_lower(
        source, source, target, target, 0, gap_budget=5e-5,
        overlap_dual=True, diagnostics=certified)
    assert lower > -5e-5 and certified['dual_improved_boxes'] == 1
    # The axis dual already met the budget: no joint work is spent.
    assert certified['dual_joint_candidates_checked'] == 0
    penetrating = source.copy()
    penetrating[:, 2] -= 1e-4
    refused = {}
    lower, visited = bounds.matching_projection_swept_lower(
        penetrating, penetrating, target, target, 0, gap_budget=5e-5,
        overlap_dual=True, diagnostics=refused)
    # True represented-overlap infimum is -sqrt(2)*5e-5 < -5e-5: no valid
    # certificate can admit this box, joint or not.
    assert lower < -5e-5 and visited == 1
    assert refused['dual_joint_candidates_checked'] > 0
