"""Conservative deterministic search controls for planar contact patches."""

import numpy as np
import pytest

from anyfesolver.general_contact.overlap import PlanarFace, planar_projected_overlap
from anyfesolver.general_contact.patch_mortar import MovingPatchMortar, PatchFace
from anyfesolver.general_contact.pointwise_patch import PointwisePatchPenalty


def _grid(body, origin_x, subdivisions=4):
    faces = []
    for j in range(subdivisions):
        for i in range(subdivisions):
            x, y = origin_x + i / subdivisions, j / subdivisions
            step = 1.0 / subdivisions
            corners = np.array(((x, y, 0.0), (x + step, y, 0.0),
                                (x + step, y + step, 0.0), (x, y + step, 0.0)))
            nodes = (body + j * (subdivisions + 1) + i,
                     body + j * (subdivisions + 1) + i + 1,
                     body + (j + 1) * (subdivisions + 1) + i + 1,
                     body + (j + 1) * (subdivisions + 1) + i)
            faces.append(PatchFace(PlanarFace((body, len(faces) + 1), corners), nodes))
    return tuple(faces)


def _exhaustive_positive_pairs(patch, source_translation, target_translation):
    matches = set()
    for i, source in enumerate(patch.source_faces):
        moved_source = PlanarFace(source.geometry.feature_id,
                                  source.geometry.corners + source_translation)
        for j, target in enumerate(patch.target_faces):
            moved_target = PlanarFace(target.geometry.feature_id,
                                      target.geometry.corners + target_translation)
            if planar_projected_overlap(moved_source, moved_target) is not None:
                matches.add((i, j))
    return matches


def test_patch_broadphase_contains_every_exact_overlap_during_translation():
    patch = MovingPatchMortar(
        _grid(1, 0.0), _grid(100, 10.0), np.array((0.0, 0.0, 1.0)),
        1.0e6,
    )
    zero = np.zeros(3)
    heights_a = np.full(len(patch.source_node_ids), -1.0e-4)
    heights_b = np.zeros(len(patch.target_node_ids))
    assert patch.candidate_pairs == 0
    for target_x in (-11.2, -10.25, -10.0, -9.65, -8.8, 0.0):
        translation = np.array((target_x, 0.0, 0.0))
        candidates = patch._candidate_indices(zero, translation)
        assert candidates == tuple(sorted(set(candidates)))
        assert set(candidates) >= _exhaustive_positive_pairs(patch, zero, translation)
        state = patch.evaluate(zero, translation, heights_a, heights_b)
        assert state.candidate_pairs == len(candidates)
        assert state.overlapping_pairs == len(
            _exhaustive_positive_pairs(patch, zero, translation)
        )
    assert patch.evaluate(zero, zero, heights_a, heights_b).candidate_pairs == 0


def test_sparse_cartesian_product_is_admitted_and_actual_limit_fails_closed():
    source = (_grid(1, 0.0, 1)[0], _grid(10, 3.0, 1)[0])
    target = (_grid(100, 10.0, 1)[0], _grid(110, 13.0, 1)[0])
    patch = MovingPatchMortar(
        source, target, np.array((0.0, 0.0, 1.0)), 1.0e6,
        maximum_candidate_pairs=1,
    )
    assert patch.candidate_pairs == 0
    zero = np.zeros(3)
    with pytest.raises(MemoryError, match="candidate limit exceeded"):
        patch.evaluate(zero, np.array((-10.0, 0.0, 0.0)),
                       np.zeros(8), np.zeros(8))


def test_patch_broadphase_is_complete_in_a_rotated_contact_plane():
    rotation = np.array(((0.0, 0.0, 1.0),
                         (1.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0)))

    def rotated(faces):
        return tuple(PatchFace(PlanarFace(face.geometry.feature_id,
                                         face.geometry.corners @ rotation.T),
                               face.node_ids) for face in faces)

    normal = rotation @ np.array((0.0, 0.0, 1.0))
    patch = MovingPatchMortar(
        rotated(_grid(1, 0.0, 2)), rotated(_grid(100, 10.0, 2)),
        normal, 1.0e6,
    )
    source = np.zeros(3)
    for local_shift in (-10.25, -10.0, -9.6):
        target = rotation @ np.array((local_shift, 0.2, 0.0))
        candidates = patch._candidate_indices(source, target)
        assert set(candidates) >= _exhaustive_positive_pairs(patch, source, target)


def test_pointwise_profile_uses_same_candidate_count():
    patch = PointwisePatchPenalty(
        (_grid(1, 0.0, 1)[0],), (_grid(100, 10.0, 1)[0],),
        np.array((0.0, 0.0, 1.0)), 1.0e6,
        integration_subdivisions=0, maximum_integration_triangles=100,
    )
    zero = np.zeros(3)
    heights = np.full(4, -1.0e-4)
    empty = patch.evaluate(zero, zero, heights, np.zeros(4))
    assert empty.candidate_pairs == empty.overlapping_pairs == 0
    moved = patch.evaluate(zero, np.array((-9.75, 0.2, 0.0)),
                           heights, np.zeros(4))
    assert moved.candidate_pairs == moved.overlapping_pairs == 1
    assert moved.energy > 0.0
