"""Exhaustive moving-pair controls for the D1 projected-prism search."""

import numpy as np
import pytest

from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty, S3PatchFace,
)


def _patch(pair_broadphase, *, source_kind="q4"):
    source = {}
    target = {}
    source_faces = []
    target_faces = []
    for j in range(5):
        for i in range(5):
            source[1+j*5+i] = np.asarray((0.103+i/4, 0.107+j/4, 0.0))
    for j in range(4):
        for i in range(4):
            base = 1+j*5+i
            nodes = (base, base+1, base+6, base+5)
            source_faces.extend(S3PatchFace(
                part, np.asarray([source[node] for node in part]))
                for part in ((nodes,) if source_kind == "q4" else
                             ((nodes[0], nodes[1], nodes[2]),
                              (nodes[0], nodes[2], nodes[3]))))
    for j in range(4):
        for i in range(4):
            target[101+j*4+i] = np.asarray((i/3, j/3, 0.0))
    for j in range(3):
        for i in range(3):
            base = 101+j*4+i
            for nodes in ((base, base+1, base+4),
                          (base+1, base+5, base+4)):
                target_faces.append(S3PatchFace(
                    nodes, np.asarray([target[node] for node in nodes])))
    return MovingS3PatchPrimalMortarPenalty(
        tuple(source_faces), tuple(target_faces), 1e4,
        difference_step=2e-5, maximum_energy_evaluations=14144,
        maximum_candidate_pairs=len(source_faces)*18,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
        q4_source_derivative_tolerances=(1e-5,)*4,
        differentiation=("jet2_q4_s3" if source_kind == "q4"
                         else "jet2_s3"), pair_broadphase=pair_broadphase,
    )


@pytest.mark.parametrize("pose", range(4))
def test_projected_prism_matches_exhaustive_mortar_pairs(pose):
    broad = _patch(True)
    full = _patch(False)
    source = broad.reference_source.copy()
    target = broad.reference_target.copy()
    if pose == 1:
        target[:, :2] += (0.072, -0.041)
        source[:, 2] = -1e-4
    elif pose == 2:
        angle = 0.23
        rotation = np.asarray(((1., 0., 0.),
                               (0., np.cos(angle), -np.sin(angle)),
                               (0., np.sin(angle), np.cos(angle))))
        source = source@rotation.T
        target = target@rotation.T
        target[:, :2] += (0.071, 0.037)
    elif pose == 3:
        source[:, 2] += 1e-4*source[:, 0]*source[:, 1]
        target[:, :2] += (0.053, 0.029)
    reduced = broad.evaluate(source, target)
    exhaustive = full.evaluate(source, target)
    assert reduced.overlap_signature == exhaustive.overlap_signature
    assert reduced.overlapping_pairs == exhaustive.overlapping_pairs
    assert sum(pair is not None for pair in reduced.pair_states) < 288
    for name in ("reference_support", "reference_weighted_gap", "pressure_gram",
                 "normalized_gap", "pressure"):
        np.testing.assert_allclose(getattr(reduced, name),
                                   getattr(exhaustive, name),
                                   rtol=1e-13, atol=1e-14)
    for name in ("energy", "overlap_area", "reference_overlap_area",
                 "minimum_physical_gap", "weighted_gap_integration_error",
                 "area_integration_error", "integration_triangles"):
        assert getattr(reduced, name) == getattr(exhaustive, name)


@pytest.mark.parametrize("pose", (0, 1))
def test_s3_source_projected_prism_matches_exhaustive_pairs(pose):
    broad = _patch(True, source_kind="s3")
    full = _patch(False, source_kind="s3")
    source = broad.reference_source.copy()
    target = broad.reference_target.copy()
    if pose:
        source[:, 2] = -1e-4
        target[:, :2] += (0.051, -0.031)
    reduced = broad.evaluate(source, target)
    exhaustive = full.evaluate(source, target)
    assert reduced.overlap_signature == exhaustive.overlap_signature
    assert reduced.overlapping_pairs == exhaustive.overlapping_pairs
    assert sum(pair is not None for pair in reduced.pair_states) < 576
    for name in ("reference_support", "reference_weighted_gap", "pressure_gram",
                 "normalized_gap", "pressure"):
        np.testing.assert_allclose(getattr(reduced, name),
                                   getattr(exhaustive, name),
                                   rtol=1e-13, atol=1e-14)
