"""Independent directional checks for the unprojected mortar force map."""

import numpy as np

from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


def _case():
    source_positions = np.asarray(((0., 0., .01), (.5, 0., .01),
                                   (.5, 1., .01), (0., 1., .01),
                                   (1., 0., .01), (1., 1., .01)))
    source = (
        S3PatchFace((1, 2, 3, 4), source_positions[[0, 1, 2, 3]]),
        S3PatchFace((2, 5, 6, 3), source_positions[[1, 4, 5, 2]]),
    )
    target = S3PatchFace(
        (7, 8, 9), np.asarray(((0.13, 0.17, 0.),
                              (0.84, 0.21, 0.), (0.31, 0.88, 0.))),
    )
    geometry = MovingS3PatchPrimalMortarPenalty(
        source, (target,), 100_000., difference_step=1e-6,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=128,
        q4_source_derivative_tolerances=(1e-5, 1e-5, 1e-5, 1e-5),
        differentiation="jet2_q4_s3", maximum_candidate_pairs=2,
    )
    position = np.concatenate((
        source_positions + np.asarray((0., 0., -.010025)),
        target.reference_positions,
    ))
    return MovingS3PatchProjectedNormal(geometry), position


def test_zero_force_still_has_activation_gradient_and_fixed_force_tangent():
    law, position = _case()
    count = len(law.source_node_ids)
    initial = law.geometry.evaluate(position[:count], position[count:])
    z = np.zeros(len(initial.reference_support))
    zero = law.nodal_force_linearization(position[:count], position[count:], z)
    assert np.all(initial.reference_support > 0.0)
    assert np.linalg.norm(zero.gap_gradient) > 0.0
    np.testing.assert_allclose(zero.residual, 0.0, atol=0.0)

    z = 0.4*initial.reference_support
    direction = np.random.default_rng(7).normal(size=position.size)
    direction /= np.linalg.norm(direction)

    def response(coordinates):
        points = coordinates.reshape(-1, 3)
        return law.nodal_force_linearization(points[:count], points[count:], z)

    base = response(position.reshape(-1))
    np.testing.assert_allclose(base.residual,
                               -base.gap_gradient.T @ z,
                               rtol=1e-12, atol=1e-12)
    step = 2e-7
    plus = response(position.reshape(-1) + step*direction)
    minus = response(position.reshape(-1) - step*direction)
    force_fd = (plus.residual-minus.residual)/(2*step)
    predicted = base.tangent @ direction
    assert np.linalg.norm(force_fd-predicted)/np.linalg.norm(predicted) < 1e-5
    gap_fd = (plus.state.normalized_gap-minus.state.normalized_gap)/(2*step)
    gap_predicted = base.gap_gradient @ direction
    assert np.linalg.norm(gap_fd-gap_predicted)/np.linalg.norm(gap_predicted) < 1e-5


def test_disappeared_overlap_has_no_work_and_inactive_force_row():
    law, position = _case()
    count = len(law.source_node_ids)
    source = position[:count]
    target = position[count:].copy()
    target[:, 0] += 10.0
    z = np.ones(len(law.pressure_mode_ids))*0.1
    raw = law.nodal_force_linearization(source, target, z)
    assert np.all(raw.state.reference_support == 0.0)
    np.testing.assert_array_equal(raw.residual, np.zeros_like(raw.residual))
    np.testing.assert_array_equal(raw.gap_gradient,
                                  np.zeros_like(raw.gap_gradient))
