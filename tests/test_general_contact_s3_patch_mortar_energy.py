"""Shared-node moving S3 patch mortar development checks."""

import numpy as np
import pytest

from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


def _two_source_one_target():
    source = np.array(((0., 0., 0.08), (1., 0., 0.08),
                       (1., 1., 0.08), (0., 1., 0.08)))
    target = np.array(((-1., -1., -0.02), (4., -1., 0.08),
                       (-1., 4., -0.02)))
    source_faces = (
        S3PatchFace((1, 2, 3), source[[0, 1, 2]]),
        S3PatchFace((1, 3, 4), source[[0, 2, 3]]),
    )
    target_faces = (S3PatchFace((5, 6, 7), target),)
    return source, target, source_faces, target_faces


def test_patch_mortar_shares_pressure_nodes_and_exact_reference_support():
    source, target, source_faces, target_faces = _two_source_one_target()
    law = MovingS3PatchPrimalMortarPenalty(
        source_faces, target_faces, 1000.0, difference_step=2e-5,
    )
    assert law.source_node_ids == (1, 2, 3, 4)
    current = source.copy()
    current[:, 2] = -0.02
    state = law.evaluate(current, target)
    assert state.overlapping_pairs == 2
    np.testing.assert_allclose(state.reference_support,
                               (1/3, 1/6, 1/3, 1/6), atol=1e-14)
    assert sum(state.reference_support) == pytest.approx(1.0)
    assert state.pressure.shape == (4,)
    assert state.pressure_gram.shape == (4, 4)
    assert np.min(np.linalg.eigvalsh(state.pressure_gram)) > 0.0
    # Target plane z=0.02*x, so source gap is -0.02-0.02*x.
    expected_gaps = np.array((-.02, -.04, -.04, -.02))
    face1 = expected_gaps[[0, 1, 2]]
    face2 = expected_gaps[[0, 2, 3]]
    normalized_face1 = (np.sum(face1)+face1)/4.0
    normalized_face2 = (np.sum(face2)+face2)/4.0
    expected_normalized = np.array((
        0.5*(normalized_face1[0]+normalized_face2[0]),
        normalized_face1[1],
        0.5*(normalized_face1[2]+normalized_face2[1]),
        normalized_face2[2],
    ))
    np.testing.assert_allclose(state.normalized_gap, expected_normalized,
                               atol=1e-12)
    assert state.minimum_physical_gap == pytest.approx(-0.04)


def test_patch_mortar_unique_coordinate_work_and_tangent():
    source, target, source_faces, target_faces = _two_source_one_target()
    law = MovingS3PatchPrimalMortarPenalty(
        source_faces, target_faces, 1000.0, difference_step=2e-5,
    )
    current = source.copy()
    current[:, 2] = (-0.02, -0.025, -0.03, -0.02)
    work = law.work(current, target, tangent=True)
    assert work.energy_evaluations <= law.maximum_energy_evaluations
    assert work.residual.shape == (21,)
    assert work.tangent.shape == (21, 21)
    assert work.state.energy > 0.0
    np.testing.assert_allclose(work.residual.reshape(7, 3).sum(axis=0),
                               0.0, atol=1e-9*np.linalg.norm(work.residual))
    direction = np.zeros(21)
    direction[0], direction[5], direction[8] = 0.07, -0.11, 0.13
    direction[14], direction[17] = -0.04, 0.06
    positions = np.concatenate((current, target)).reshape(-1)
    step = 5e-6
    def energy_at(sign):
        trial = (positions+sign*step*direction).reshape(7, 3)
        return law.evaluate(trial[:4], trial[4:]).energy
    measured = (energy_at(1.0)-energy_at(-1.0))/(2*step)
    assert measured == pytest.approx(float(work.residual@direction),
                                     rel=1e-5, abs=1e-7)


def test_patch_mortar_rejects_duplicate_side_area_and_resource_loss():
    source, target, source_faces, target_faces = _two_source_one_target()
    with pytest.raises(MemoryError, match="budget"):
        MovingS3PatchPrimalMortarPenalty(
            source_faces, target_faces, 1000.0,
            difference_step=2e-5, maximum_energy_evaluations=100,
        )
    duplicate = S3PatchFace((1, 2, 3), source[[0, 1, 2]])
    with pytest.raises(ValueError, match="overlapping faces"):
        MovingS3PatchPrimalMortarPenalty(
            (source_faces[0], duplicate), target_faces,
            1000.0, difference_step=2e-5,
        )
