"""Local Q4/S3 derivative checks on a resolvable diagonal sliver."""

import numpy as np
import pytest

from anyfesolver.general_contact.q4_s3_overlap_derivatives import (
    UnsupportedQ4S3Branch,
    q4_s3_overlap_derivatives,
)
from anyfesolver.general_contact.q4_source_overlap import q4_source_overlap_integrals


REFERENCE = np.array([[0.35, 0.1, 0.055], [0.6, 0.1, 0.055],
                      [0.6, 0.6, 0.055], [0.35, 0.6, 0.055]])
LEFT_TRIANGLE = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0],
                          [-0.5, 1.0, 0.0]])
RIGHT_TRIANGLE = np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0],
                           [-0.5, 1.0, 0.0]])
INTEGRATION = dict(area_tolerance=1e-9, weighted_gap_tolerance=1e-10,
                   maximum_triangles=128)
DERIVATIVES = dict(allow_warped_source=True,
                   support_gradient_tolerance=1e-5,
                   support_hessian_tolerance=1e-5,
                   weighted_gap_gradient_tolerance=1e-5,
                   weighted_gap_hessian_tolerance=1e-5)
SHIFT_DIRECTION = np.zeros(21)
SHIFT_DIRECTION[[0, 3, 6, 9]] = 1.0
NORMAL_RATE = 0.2
MIXED_DIRECTION = SHIFT_DIRECTION.copy()
MIXED_DIRECTION[[2, 5, 8, 11]] = NORMAL_RATE


def _source(shift, normal_shift=0.0):
    current = REFERENCE.copy()
    current[:, 0] += shift
    current[:, 2] += -0.01504+normal_shift
    return current


def _jet(shift, target, normal_shift=0.0):
    return q4_s3_overlap_derivatives(
        REFERENCE, _source(shift, normal_shift), target,
        **INTEGRATION, **DERIVATIVES,
    )


def _production(shift, target, normal_shift=0.0):
    return q4_source_overlap_integrals(
        REFERENCE, _source(shift, normal_shift), target, **INTEGRATION,
    )


def _directional_gradient(result):
    return float(np.sum(result.support_gradient@SHIFT_DIRECTION))


def _directional_hessian(result):
    return float(np.sum(np.einsum(
        'i,aij,j->a', SHIFT_DIRECTION, result.support_hessian,
        SHIFT_DIRECTION,
    )))


def _weighted_gradient(result):
    return float(np.sum(result.weighted_gap_gradient@MIXED_DIRECTION))


def _weighted_hessian(result):
    return float(np.sum(np.einsum(
        'i,aij,j->a', MIXED_DIRECTION, result.weighted_gap_hessian,
        MIXED_DIRECTION,
    )))


def test_positive_sliver_matches_production_and_has_resolved_derivatives():
    delta = 3e-8
    shift = 0.05-delta
    result = _jet(shift, LEFT_TRIANGLE)
    production = _production(shift, LEFT_TRIANGLE)
    assert result.overlap_area == pytest.approx(delta**2/2, rel=2e-8, abs=0)
    np.testing.assert_allclose(result.reference_support, production.support,
                               rtol=2e-8, atol=4e-25)
    np.testing.assert_allclose(result.reference_weighted_gap,
                               production.weighted_gap, rtol=2e-8, atol=4e-25)
    assert np.max(result.derivative_quadrature_differences) < 1e-5
    assert _directional_gradient(result) == pytest.approx(-delta, rel=2e-7)
    assert _directional_hessian(result) == pytest.approx(1.0, rel=2e-6)

    step = 5e-9
    jet_plus, jet_minus = _jet(shift+step, LEFT_TRIANGLE), _jet(
        shift-step, LEFT_TRIANGLE,
    )
    assert (jet_plus.branch_signature == result.branch_signature
            == jet_minus.branch_signature)
    plus, minus = _production(shift+step, LEFT_TRIANGLE), _production(
        shift-step, LEFT_TRIANGLE,
    )
    first = (plus.support.sum()-minus.support.sum())/(2*step)
    second = (plus.support.sum()-2*production.support.sum()
              + minus.support.sum())/step**2
    assert first == pytest.approx(_directional_gradient(result), rel=2e-6)
    assert second == pytest.approx(_directional_hessian(result), rel=2e-5)

    weighted_plus_jet = _jet(shift+step, LEFT_TRIANGLE,
                             NORMAL_RATE*step)
    weighted_minus_jet = _jet(shift-step, LEFT_TRIANGLE,
                              -NORMAL_RATE*step)
    assert (weighted_plus_jet.branch_signature == result.branch_signature
            == weighted_minus_jet.branch_signature)
    weighted_plus = _production(shift+step, LEFT_TRIANGLE,
                                NORMAL_RATE*step)
    weighted_minus = _production(shift-step, LEFT_TRIANGLE,
                                 -NORMAL_RATE*step)
    first_weighted = (weighted_plus.weighted_gap.sum()
                      - weighted_minus.weighted_gap.sum())/(2*step)
    second_weighted = (weighted_plus.weighted_gap.sum()
                       - 2*production.weighted_gap.sum()
                       + weighted_minus.weighted_gap.sum())/step**2
    assert first_weighted == pytest.approx(_weighted_gradient(result), rel=3e-6)
    assert second_weighted == pytest.approx(_weighted_hessian(result), rel=3e-5)


@pytest.mark.parametrize("side", [-1, 1])
def test_pair_ownership_on_both_smooth_sides(side):
    delta = 3e-8
    shift = 0.05+side*delta
    right = _jet(shift, RIGHT_TRIANGLE)
    left = _jet(shift, LEFT_TRIANGLE) if side < 0 else None
    if side > 0:
        assert _production(shift, LEFT_TRIANGLE).overlap_area == 0.0
        with pytest.raises(UnsupportedQ4S3Branch, match="zero projected overlap"):
            _jet(shift, LEFT_TRIANGLE)
    total_support = right.reference_support.sum()
    total_gradient = _directional_gradient(right)
    total_hessian = _directional_hessian(right)
    if left is not None:
        total_support += left.reference_support.sum()
        total_gradient += _directional_gradient(left)
        total_hessian += _directional_hessian(left)
    assert total_support == pytest.approx(0.05-side*0.5*delta, abs=2e-14)
    assert total_gradient == pytest.approx(-0.5, abs=2e-9)
    assert total_hessian == pytest.approx(0.0, abs=2e-6)

    total_weighted = right.reference_weighted_gap.sum()
    total_weighted_gradient = _weighted_gradient(right)
    total_weighted_hessian = _weighted_hessian(right)
    if left is not None:
        total_weighted += left.reference_weighted_gap.sum()
        total_weighted_gradient += _weighted_gradient(left)
        total_weighted_hessian += _weighted_hessian(left)
    production_weighted = _production(shift, RIGHT_TRIANGLE).weighted_gap.sum()
    if side < 0:
        production_weighted += _production(shift, LEFT_TRIANGLE).weighted_gap.sum()
    gap = 0.055-0.01504
    assert total_weighted == pytest.approx(production_weighted, abs=2e-14)
    assert total_weighted == pytest.approx(total_support*gap, abs=2e-14)
    assert total_weighted_gradient == pytest.approx(
        -0.5*gap+NORMAL_RATE*total_support, abs=3e-9,
    )
    assert total_weighted_hessian == pytest.approx(-NORMAL_RATE, abs=2e-6)

    step = 5e-9
    plus = _jet(shift+step, RIGHT_TRIANGLE)
    minus = _jet(shift-step, RIGHT_TRIANGLE)
    assert plus.branch_signature == minus.branch_signature == right.branch_signature
    first = (_production(shift+step, RIGHT_TRIANGLE).support.sum()
             - _production(shift-step, RIGHT_TRIANGLE).support.sum())/(2*step)
    second = (_directional_gradient(plus)-_directional_gradient(minus))/(2*step)
    assert first == pytest.approx(_directional_gradient(right), abs=2e-8)
    assert second == pytest.approx(_directional_hessian(right), abs=2e-5)


def test_exact_diagonal_and_roundoff_scale_ties_fail_closed():
    for shift in (0.05, 0.05-1e-15):
        with pytest.raises(UnsupportedQ4S3Branch):
            _jet(shift, LEFT_TRIANGLE)
