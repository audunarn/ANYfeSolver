"""Exhaustive small-scene control for the same-side contact search bound."""

import numpy as np
import pytest

from anyfesolver.general_contact.overlap import PlanarFace, planar_projected_overlap
from anyfesolver.general_contact.q4_source_overlap import projected_q4_source_face
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty,
)


def _exhaustive_overlap(positions, indices):
    """Repeat the pre-search pair check without the production broad phase."""
    scale = max(1.0, float(np.max(np.abs(positions))))
    for i, first_nodes in enumerate(indices):
        first_positions = positions[first_nodes]
        if len(first_nodes) == 4:
            first, normal = projected_q4_source_face(first_positions)
        else:
            first = PlanarFace((1, i+1), first_positions)
            normal = np.cross(first_positions[1]-first_positions[0],
                              first_positions[2]-first_positions[0])
            normal /= np.linalg.norm(normal)
        for j in range(i+1, len(indices)):
            second_positions = positions[indices[j]]
            if len(second_positions) == 4:
                projected_q4_source_face(second_positions)
            projected = second_positions - (
                (second_positions-first.corners[0]) @ normal
            )[:, None]*normal
            second = PlanarFace((1, j+1), projected)
            overlap = planar_projected_overlap(first, second)
            if overlap is not None and overlap.area > 1e-12*scale**2:
                raise ValueError("S3 patch has overlapping faces on one side")


def _scene(kind, seed):
    random = np.random.default_rng(seed)
    faces = []
    for row in range(3):
        for column in range(3):
            x, y = float(column), float(row)
            corners = np.array(((x, y, 0.), (x+1, y, 0.),
                                (x+1, y+1, 0.), (x, y+1, 0.)))
            faces.extend((corners,) if kind == "q4" else
                         (corners[[0, 1, 2]], corners[[0, 2, 3]]))
    angle = 0.1+0.25*seed
    cy, sy = np.cos(angle), np.sin(angle)
    cx, sx = np.cos(0.37), np.sin(0.37)
    rotation = np.array(((cy, -sy, 0.), (sy, cy, 0.), (0., 0., 1.))) @ np.array(
        ((1., 0., 0.), (0., cx, -sx), (0., sx, cx)))
    indices = []
    vertices = []
    for face in faces:
        shift = random.uniform(-0.04, 0.04, size=2)
        moved = face.copy()
        moved[:, :2] += shift
        indices.append(np.arange(len(vertices), len(vertices)+len(face)))
        vertices.extend(moved @ rotation.T)
    return np.asarray(vertices), tuple(indices)


@pytest.mark.parametrize("kind", ("q4", "s3"))
@pytest.mark.parametrize("seed", range(4))
def test_same_side_search_matches_exhaustive_projected_overlap(kind, seed):
    positions, indices = _scene(kind, seed)
    def outcome(check):
        try:
            check(positions, indices)
        except ValueError as exc:
            return str(exc)
        return None
    assert outcome(MovingS3PatchPrimalMortarPenalty._check_same_side_overlap) \
        == outcome(_exhaustive_overlap)


def test_same_side_search_retains_remote_overlap_and_invalid_face():
    positions, indices = _scene("q4", 0)
    positions[indices[-1]] = positions[indices[0]]
    with pytest.raises(ValueError, match="overlapping faces"):
        MovingS3PatchPrimalMortarPenalty._check_same_side_overlap(
            positions, indices,
        )
    positions, indices = _scene("q4", 0)
    positions[indices[-1][2]] = positions[indices[-1][1]]
    with pytest.raises(ValueError, match="degenerate|convex|inverted"):
        MovingS3PatchPrimalMortarPenalty._check_same_side_overlap(
            positions, indices,
        )


@pytest.mark.parametrize("offset", (-1.0e-7, 1.0e-7))
def test_same_side_search_near_touch_at_large_coordinates(offset):
    first = np.array(((100., 50., 0.), (101., 50., 0.),
                      (101., 51., 0.), (100., 51., 0.)))
    second = first + np.array((1.+offset, 0., 0.))
    rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
    positions = np.vstack((first, second)) @ rotation.T
    indices = (np.arange(4), np.arange(4, 8))
    def outcome(check):
        try:
            check(positions, indices)
        except ValueError as exc:
            return str(exc)
        return None
    expected = outcome(_exhaustive_overlap)
    assert expected == ("S3 patch has overlapping faces on one side"
                        if offset < 0.0 else None)
    assert outcome(MovingS3PatchPrimalMortarPenalty._check_same_side_overlap) \
        == expected


@pytest.mark.parametrize("kind", ("q4", "s3"))
def test_same_side_tree_retains_projected_overlap_across_normal_separation(kind):
    first = np.array(((0., 0., 0.), (1., 0., 0.),
                      (1., 1., 0.), (0., 1., 0.)))
    second = first + np.array((0., 0., 12.))
    faces = ((first, second) if kind == "q4" else
             (first[[0, 1, 2]], second[[0, 1, 2]]))
    positions = np.vstack(faces)
    count = len(faces[0])
    indices = (np.arange(count), np.arange(count, 2*count))
    with pytest.raises(ValueError, match="overlapping faces"):
        _exhaustive_overlap(positions, indices)
    with pytest.raises(ValueError, match="overlapping faces"):
        MovingS3PatchPrimalMortarPenalty._check_same_side_overlap(
            positions, indices,
        )


@pytest.mark.parametrize("seed", range(6))
def test_same_side_tree_matches_exhaustive_with_varying_face_normals(seed):
    positions, indices = _scene("s3", seed)
    random = np.random.default_rng(seed+123)
    for nodes in indices:
        positions[nodes] += random.normal(0.0, 0.08, size=(3, 3))

    def outcome(check):
        try:
            check(positions, indices)
        except ValueError as exc:
            return str(exc)
        return None

    assert outcome(MovingS3PatchPrimalMortarPenalty._check_same_side_overlap) \
        == outcome(_exhaustive_overlap)


def test_same_side_tree_finds_late_remote_projection_after_many_empty_queries():
    faces = []
    for row in range(5):
        for column in range(5):
            x, y = 3.0*column, 3.0*row
            faces.append(np.array(((x, y, 0.), (x+1., y, 0.02),
                                   (x, y+1., -0.01))))
    positions = np.vstack(faces)
    indices = tuple(np.arange(3*i, 3*i+3) for i in range(len(faces)))
    _exhaustive_overlap(positions, indices)
    MovingS3PatchPrimalMortarPenalty._check_same_side_overlap(positions, indices)

    remote = faces[0] + np.array((0., 0., 40.))
    positions = np.vstack((positions, remote))
    indices += (np.arange(len(positions)-3, len(positions)),)
    with pytest.raises(ValueError, match="overlapping faces"):
        _exhaustive_overlap(positions, indices)
    with pytest.raises(ValueError, match="overlapping faces"):
        MovingS3PatchPrimalMortarPenalty._check_same_side_overlap(
            positions, indices,
        )


def test_same_side_tree_keeps_oblique_crossing_with_vertices_outside():
    first = np.array(((0., 0., 0.), (1., 0., 0.),
                      (1., 1., 0.), (0., 1., 0.)))
    second = np.array(((-2., .5, 4.), (3., .5, 4.), (.5, 3., 4.)))
    positions = np.vstack((first, second))
    indices = (np.arange(4), np.arange(4, 7))
    with pytest.raises(ValueError, match="overlapping faces"):
        _exhaustive_overlap(positions, indices)
    with pytest.raises(ValueError, match="overlapping faces"):
        MovingS3PatchPrimalMortarPenalty._check_same_side_overlap(
            positions, indices,
        )


def test_same_side_tree_keeps_valid_warped_q4_across_rotated_normal():
    first = np.array(((0., 0., 0.), (1., 0., .08),
                      (1., 1., .04), (0., 1., -.02)))
    second = first + np.array((0., 0., 6.))
    angle = 0.43
    rotation = np.array(((1., 0., 0.),
                         (0., np.cos(angle), -np.sin(angle)),
                         (0., np.sin(angle), np.cos(angle))))
    positions = np.vstack((first, second)) @ rotation.T
    indices = (np.arange(4), np.arange(4, 8))
    with pytest.raises(ValueError, match="overlapping faces"):
        _exhaustive_overlap(positions, indices)
    with pytest.raises(ValueError, match="overlapping faces"):
        MovingS3PatchPrimalMortarPenalty._check_same_side_overlap(
            positions, indices,
        )
