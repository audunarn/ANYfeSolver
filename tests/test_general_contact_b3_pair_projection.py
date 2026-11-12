"""Independent local checks for curved classical B3 pair projection/work."""

import numpy as np
import pytest

from anyfesolver.general_contact.beam_curve_pair import (
    b3_distance_jet, closest_on_b3,
)


def test_b3_target_projection_covers_interior_endpoints_and_competing_minima():
    straight = np.array(((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (1.0, 0.0, 0.0)))
    point = np.array((0.25, 0.3, 0.4))
    interior = closest_on_b3(point, straight)
    assert interior.unique
    assert interior.minimizers[0] == pytest.approx(0.25, abs=1e-13)
    assert interior.distance == pytest.approx(0.5, abs=1e-13)
    endpoint = closest_on_b3(np.array((-2.0, 0.3, 0.0)), straight)
    assert endpoint.unique
    assert endpoint.minimizers == (0.0,)

    arch = np.array(((-1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0),
                     (1.0, 0.0, 0.0)))
    competition = closest_on_b3(np.zeros(3), arch)
    assert len(competition.minimizers) == 2
    assert sum(competition.minimizers) == pytest.approx(1.0, abs=1e-12)
    source = np.array(((-0.2, 0.0, 0.0), (0.0, 0.0, 0.0),
                       (0.2, 0.0, 0.0)))
    with pytest.raises(ValueError, match="competing"):
        b3_distance_jet(source, arch, 0.5, 0.1)


def test_b3_curved_interior_distance_jet_matches_independent_differences_and_rigid_work():
    source = np.array(((0.0, 0.8, 0.1),
                       (0.5, 0.7, 0.12),
                       (1.0, 0.8, 0.1)))
    target = np.array(((0.0, 0.0, 0.0),
                       (0.5, 0.12, 0.02),
                       (1.0, 0.0, 0.0)))
    s = 0.43
    radius = 0.2
    state = b3_distance_jet(source, target, s, radius)
    assert state.target_branch == "interior"
    assert 0.0 < state.target_parameter < 1.0
    direction = np.array((
        (0.21, -0.32, 0.15), (-0.14, 0.24, -0.17), (0.08, 0.05, 0.19),
        (-0.19, 0.13, -0.10), (0.31, -0.11, 0.07), (0.04, 0.18, -0.22),
    ))
    h = 1.0e-6
    plus = b3_distance_jet(source + h * direction[:3],
                           target + h * direction[3:], s, radius)
    minus = b3_distance_jet(source - h * direction[:3],
                            target - h * direction[3:], s, radius)
    numerical_gradient = (plus.gap - minus.gap) / (2.0 * h)
    assert numerical_gradient == pytest.approx(
        float(state.gradient @ direction.reshape(18)), rel=1e-8, abs=1e-10,
    )
    numerical_hessian_action = (plus.gradient - minus.gradient) / (2.0 * h)
    np.testing.assert_allclose(
        numerical_hessian_action, state.hessian @ direction.reshape(18),
        rtol=1e-5, atol=2e-7,
    )
    np.testing.assert_allclose(state.hessian, state.hessian.T,
                               rtol=0, atol=1e-12)
    gradient_rows = state.gradient.reshape(6, 3)
    assert np.linalg.norm(np.sum(gradient_rows, axis=0)) < 1e-12
    positions = np.vstack((source, target))
    assert np.linalg.norm(np.sum(np.cross(positions, gradient_rows), axis=0)) < 1e-12
    reversed_layout = b3_distance_jet(source, target, s, radius,
                                       source_is_a=False)
    np.testing.assert_allclose(reversed_layout.gradient,
                               np.r_[state.gradient[9:], state.gradient[:9]],
                               rtol=0, atol=1e-14)
    permutation = np.r_[np.arange(9, 18), np.arange(0, 9)]
    np.testing.assert_allclose(reversed_layout.hessian,
                               state.hessian[np.ix_(permutation, permutation)],
                               rtol=0, atol=1e-14)
    swapped = b3_distance_jet(target, source, state.target_parameter,
                              radius, source_is_a=False)
    # Reversing a directional closest-point query can lower the distance:
    # the original source station need not be stationary on its own curve.
    assert swapped.gap <= state.gap + 1e-12


def test_b3_projection_rejects_undefined_normal_and_invalid_geometry():
    straight = np.array(((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="normal"):
        b3_distance_jet(straight, straight, 0.5, 0.1)
    with pytest.raises(ValueError, match="distinct"):
        closest_on_b3(np.zeros(3), np.zeros((3, 3)))
