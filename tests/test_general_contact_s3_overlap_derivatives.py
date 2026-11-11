"""Bounded checks of the local smooth S3 overlap derivative primitive."""

import numpy as np
import pytest

from anyfesolver.general_contact.overlap import PlanarFace, planar_projected_overlap
from anyfesolver.general_contact.triangle_mortar_energy import MovingS3PrimalMortarPenalty
from anyfesolver.general_contact.s3_overlap_derivatives import (
    UnsupportedOverlapBranch,
    s3_overlap_derivatives,
)


REFERENCE = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
SOURCE = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.1], [0.0, 2.0, -0.1]])
TARGET = np.array([[0.3, 0.2, -0.11], [1.95, 0.42, -0.05],
                   [0.65, 1.75, -0.15]])


def _evaluate(flat):
    return s3_overlap_derivatives(REFERENCE, flat[:9].reshape(3, 3),
                                  flat[9:].reshape(3, 3))


def test_partial_overlap_matches_independent_polygon_integration():
    result = s3_overlap_derivatives(REFERENCE, SOURCE, TARGET)
    overlap = planar_projected_overlap(PlanarFace((1, 1), SOURCE),
                                       PlanarFace((2, 1), TARGET))
    assert overlap is not None
    assert len(overlap.polygon) >= 3
    assert result.overlap_area == pytest.approx(overlap.area, rel=1e-12)
    normal = overlap.normal
    ratio = 2.0/(0.5*np.linalg.norm(np.cross(SOURCE[1]-SOURCE[0],
                                            SOURCE[2]-SOURCE[0])))
    support = np.zeros(3)
    weighted = np.zeros(3)
    for i in range(1, len(overlap.polygon)-1):
        triangle = overlap.polygon[[0, i, i+1]]
        area = np.linalg.norm(np.cross(triangle[1]-triangle[0],
                                       triangle[2]-triangle[0]))/2
        shapes = np.array([overlap.shape_values(PlanarFace((1, 1), SOURCE), p)
                           for p in triangle])
        gaps = np.array([np.dot(p-overlap.shape_values(
            PlanarFace((2, 1), TARGET), p)@TARGET, normal) for p in triangle])
        support += ratio*area*shapes.sum(axis=0)/3
        weighted += ratio*area*(shapes.sum(axis=0)*gaps.sum()
                                + shapes.T@gaps)/12
    np.testing.assert_allclose(result.reference_support, support, rtol=1e-12)
    np.testing.assert_allclose(result.reference_weighted_gap, weighted, rtol=1e-12)
    assert np.all(result.reference_support > 0)
    assert result.reference_overlap_area == pytest.approx(ratio*overlap.area)
    existing = MovingS3PrimalMortarPenalty(
        REFERENCE, TARGET, 1.0, difference_step=1e-5,
    ).evaluate(SOURCE, TARGET)
    # Both APIs keep pressure-node order identical to the source face order.
    np.testing.assert_allclose(result.reference_support,
                               existing.reference_support, rtol=1e-12)
    np.testing.assert_allclose(result.reference_weighted_gap,
                               existing.reference_weighted_gap, rtol=1e-12)
    assert result.reference_overlap_area == pytest.approx(existing.reference_overlap_area)


def test_gradients_and_hessians_follow_moving_clipping_branch():
    x = np.concatenate((SOURCE, TARGET)).reshape(-1)
    base = _evaluate(x)
    rng = np.random.default_rng(240926)
    for _ in range(3):
        direction = rng.normal(size=18)
        direction /= np.linalg.norm(direction)
        step = 2e-5
        plus, minus = _evaluate(x+step*direction), _evaluate(x-step*direction)
        assert plus.branch_signature == minus.branch_signature == base.branch_signature
        for value, gradient, hessian, plus_value, minus_value, plus_gradient, minus_gradient in (
            (base.reference_support, base.support_gradient, base.support_hessian,
             plus.reference_support, minus.reference_support,
             plus.support_gradient, minus.support_gradient),
            (base.reference_weighted_gap, base.weighted_gap_gradient,
             base.weighted_gap_hessian, plus.reference_weighted_gap,
             minus.reference_weighted_gap, plus.weighted_gap_gradient,
             minus.weighted_gap_gradient),
        ):
            np.testing.assert_allclose((plus_value-minus_value)/(2*step),
                                       gradient@direction, rtol=3e-6, atol=1e-8)
            np.testing.assert_allclose((plus_gradient-minus_gradient)/(2*step),
                                       np.einsum('ijk,k->ij', hessian, direction),
                                       rtol=2e-5, atol=1e-7)
            np.testing.assert_allclose(hessian, np.swapaxes(hessian, 1, 2), atol=2e-12)


def test_rigid_motion_and_scalar_force_wrench_balance():
    x = np.concatenate((SOURCE, TARGET)).reshape(-1)
    base = _evaluate(x)
    theta = 0.31
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = (x.reshape(6, 3)@rotation.T
             + np.array([1.3, -0.7, 0.4])).reshape(-1)
    other = _evaluate(moved)
    np.testing.assert_allclose(other.reference_support, base.reference_support,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(other.reference_weighted_gap,
                               base.reference_weighted_gap, rtol=1e-12, atol=1e-12)
    for gradient in (base.support_gradient, base.weighted_gap_gradient):
        for row in gradient:
            forces = row.reshape(6, 3)
            np.testing.assert_allclose(forces.sum(axis=0), 0, atol=2e-11)
            np.testing.assert_allclose(np.cross(x.reshape(6, 3), forces).sum(axis=0),
                                       0, atol=2e-11)


def test_exact_boundary_incidence_reports_unsupported():
    with pytest.raises(UnsupportedOverlapBranch, match="clipping boundary"):
        s3_overlap_derivatives(REFERENCE, SOURCE, SOURCE)
