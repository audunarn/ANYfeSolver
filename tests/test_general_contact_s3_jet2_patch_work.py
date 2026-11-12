"""Coupled shared-node control for differentiated S3 projected contact work."""

import numpy as np
import pytest

from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_overlap_derivatives import UnsupportedOverlapBranch
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty,
    S3PatchFace,
)


def _patch(differentiation):
    source = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
    target = np.array(((0.15, 0.20, 0.0), (0.85, 0.30, 0.0),
                       (0.45, 0.90, 0.0)))
    source_faces = (
        S3PatchFace((1, 2, 3), source[[0, 1, 2]]),
        S3PatchFace((1, 3, 4), source[[0, 2, 3]]),
    )
    target_faces = (S3PatchFace((5, 6, 7), target),)
    geometry = MovingS3PatchPrimalMortarPenalty(
        source_faces, target_faces, 1000.0,
        difference_step=2.0e-6, maximum_energy_evaluations=2000,
        differentiation=differentiation,
    )
    current_source = source.copy()
    current_source[:, 2] = -0.01 + 0.005*source[:, 0] + 0.002*source[:, 1]
    return MovingS3PatchProjectedNormal(geometry), current_source, target


def test_shared_s3_pressure_work_matches_independent_difference_route():
    numerical, source, target = _patch("pair_local")
    differentiated, _, _ = _patch("jet2_s3")
    multipliers = np.zeros(4)
    reference = numerical.work(source, target, multipliers, tangent=True)
    candidate = differentiated.work(source, target, multipliers, tangent=True)
    assert candidate.state.active_pressure_dofs >= 2
    np.testing.assert_allclose(candidate.state.reference_support,
                               reference.state.reference_support, rtol=1.0e-11)
    np.testing.assert_allclose(candidate.residual, reference.residual,
                               rtol=2.0e-4, atol=2.0e-7)
    np.testing.assert_allclose(candidate.tangent, reference.tangent,
                               rtol=2.0e-3, atol=2.0e-4)
    assert candidate.energy_evaluations < reference.energy_evaluations
    direction = np.linspace(-0.7, 0.8, candidate.residual.size)
    flat = np.concatenate((source, target)).reshape(-1)
    reference_scale = max(1.0, np.linalg.norm(candidate.tangent @ direction))
    errors = []
    for step in (2.0e-5, 1.0e-5):
        plus = flat + step*direction
        minus = flat - step*direction
        response_plus = differentiated.work(
            plus[:12].reshape(4, 3), plus[12:].reshape(3, 3),
            multipliers, tangent=False,
        ).residual
        response_minus = differentiated.work(
            minus[:12].reshape(4, 3), minus[12:].reshape(3, 3),
            multipliers, tangent=False,
        ).residual
        errors.append(np.linalg.norm(
            (response_plus-response_minus)/(2*step)
            - candidate.tangent @ direction,
        )/reference_scale)
    assert max(errors) < 1.0e-5
    force = candidate.residual.reshape(7, 3)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(np.cross(flat.reshape(7, 3), force).sum(axis=0),
                               0.0, atol=1.0e-10)


def test_touching_zero_area_pair_is_not_skipped_beside_active_pair():
    source = np.array(((0.0, 0.0, -0.01), (1.0, 0.0, -0.01),
                       (0.0, 1.0, -0.01), (2.0, 0.2, -0.01),
                       (3.0, 0.2, -0.01), (2.0, 1.2, -0.01)))
    target = np.array(((0.2, 0.2, 0.0), (2.0, 0.2, 0.0),
                       (0.2, 0.8, 0.0)))
    geometry = MovingS3PatchPrimalMortarPenalty(
        (S3PatchFace((1, 2, 3), source[:3]),
         S3PatchFace((4, 5, 6), source[3:])),
        (S3PatchFace((7, 8, 9), target),), 1000.0,
        difference_step=2.0e-6, differentiation="jet2_s3",
    )
    law = MovingS3PatchProjectedNormal(geometry)
    assert geometry.evaluate(source, target).reference_overlap_area > 0.0
    with pytest.raises(UnsupportedOverlapBranch, match="zero-area touching"):
        law.work(source, target, np.zeros(6), tangent=True)


def test_touching_pair_shares_an_active_pressure_node():
    source = np.array(((0.0, 0.0, -0.01), (2.0, 0.2, -0.01),
                       (0.0, 1.0, -0.01), (3.0, 0.2, -0.01),
                       (2.0, 1.2, -0.01)))
    target = np.array(((0.2, 0.2, 0.0), (2.0, 0.2, 0.0),
                       (0.2, 0.8, 0.0)))
    geometry = MovingS3PatchPrimalMortarPenalty(
        (S3PatchFace((1, 2, 3), source[[0, 1, 2]]),
         S3PatchFace((2, 4, 5), source[[1, 3, 4]])),
        (S3PatchFace((6, 7, 8), target),), 1000.0,
        difference_step=2.0e-6, differentiation="jet2_s3",
    )
    law = MovingS3PatchProjectedNormal(geometry)
    state = law.evaluate(source, target, np.zeros(5))
    assert state.pressure[1] > 0.0
    with pytest.raises(UnsupportedOverlapBranch, match="zero-area touching"):
        law.work(source, target, np.zeros(5), tangent=True)
