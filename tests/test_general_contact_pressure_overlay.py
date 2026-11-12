"""Local development checks for explicit Q4 pressure subdivision."""

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.general_contact.pressure_overlay import subdivide_q4_pressure_faces
from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_mortar_energy import S3PatchFace
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty,
)


def _two_faces():
    points = {
        1: (0.0, 0.0, 0.1), 2: (1.0, 0.0, 0.1),
        3: (2.0, 0.0, 0.1), 4: (0.0, 1.0, 0.1),
        5: (1.0, 1.0, 0.1), 6: (2.0, 1.0, 0.1),
    }
    return tuple(S3PatchFace(ids, np.asarray([points[node] for node in ids]))
                 for ids in ((1, 2, 5, 4), (2, 3, 6, 5)))


def test_shared_pressure_nodes_and_exact_bilinear_skin_map():
    overlay = subdivide_q4_pressure_faces(
        _two_faces(), u_divisions=2, v_divisions=2,
        first_pressure_node_id=100, reserved_node_ids=(20, 21, 22),
        maximum_pressure_nodes=15, maximum_pressure_faces=8,
    )
    assert len(overlay.faces) == 8
    assert len(overlay.pressure_node_ids) == 15
    assert len({node for face in overlay.faces for node in face.node_ids}) == 15
    with pytest.raises(ValueError):
        overlay.reference_positions[0, 0] = 10.0
    with pytest.raises(ValueError):
        overlay.interpolation.data[0] = 10.0
    np.testing.assert_allclose(np.asarray(overlay.interpolation.sum(axis=1)), 1.0)
    original = np.asarray([{
        node: position for face in _two_faces()
        for node, position in zip(face.node_ids, face.reference_positions)
    }[node] for node in overlay.structural_node_ids])
    np.testing.assert_allclose(overlay.reference_positions,
                               overlay.interpolation @ original, atol=0.0)
    for face in overlay.faces:
        for node, position in zip(face.node_ids, face.reference_positions):
            np.testing.assert_allclose(
                position, overlay.reference_positions[
                    overlay.pressure_node_ids.index(node)
                ], atol=0.0,
            )


def test_virtual_work_and_wrench_pull_back_to_structural_nodes():
    overlay = subdivide_q4_pressure_faces(
        _two_faces(), u_divisions=2, v_divisions=3,
        first_pressure_node_id=100,
        maximum_pressure_nodes=20, maximum_pressure_faces=12,
    )
    transfer = sparse.kron(overlay.interpolation, sparse.eye(3),
                           format="csr")
    rng = np.random.default_rng(15)
    structural_motion = rng.normal(size=3*len(overlay.structural_node_ids))
    virtual_force = rng.normal(size=3*len(overlay.pressure_node_ids))
    structural_force = transfer.T @ virtual_force
    np.testing.assert_allclose(
        virtual_force @ (transfer @ structural_motion),
        structural_force @ structural_motion, rtol=1e-14, atol=1e-14,
    )
    np.testing.assert_allclose(
        virtual_force.reshape(-1, 3).sum(axis=0),
        structural_force.reshape(-1, 3).sum(axis=0), atol=1e-13,
    )
    structural_reference = np.asarray([{
        node: position for face in _two_faces()
        for node, position in zip(face.node_ids, face.reference_positions)
    }[node] for node in overlay.structural_node_ids])
    np.testing.assert_allclose(
        np.cross(overlay.reference_positions,
                 virtual_force.reshape(-1, 3)).sum(axis=0),
        np.cross(structural_reference,
                 structural_force.reshape(-1, 3)).sum(axis=0),
        atol=1e-13,
    )


def test_pressure_modes_exceed_available_parent_motion_and_caps_fail_closed():
    one = (_two_faces()[0],)
    overlay = subdivide_q4_pressure_faces(
        one, u_divisions=4, v_divisions=2,
        first_pressure_node_id=100,
        maximum_pressure_nodes=15, maximum_pressure_faces=8,
    )
    assert len(overlay.pressure_node_ids) == 15
    assert np.linalg.matrix_rank(overlay.interpolation.toarray()) == 4
    with pytest.raises(MemoryError, match="node budget"):
        subdivide_q4_pressure_faces(
            one, u_divisions=4, v_divisions=2,
            first_pressure_node_id=100,
            maximum_pressure_nodes=14, maximum_pressure_faces=8,
        )
    with pytest.raises(MemoryError, match="child-face budget"):
        subdivide_q4_pressure_faces(
            one, u_divisions=4, v_divisions=2,
            first_pressure_node_id=100,
            maximum_pressure_nodes=15, maximum_pressure_faces=7,
        )


def test_rotated_shared_edge_rejects_incompatible_anisotropic_subdivision():
    first, second = _two_faces()
    rotated = S3PatchFace(
        (5, 2, 3, 6),
        second.reference_positions[[3, 0, 1, 2]],
    )
    with pytest.raises(ValueError, match="shared edge"):
        subdivide_q4_pressure_faces(
            (first, rotated), u_divisions=2, v_divisions=3,
            first_pressure_node_id=100,
            maximum_pressure_nodes=30, maximum_pressure_faces=12,
        )


def test_crease_is_not_merged_into_one_pressure_plane():
    first, second = _two_faces()
    bent = second.reference_positions.copy()
    bent[[1, 2], 2] += 0.1
    with pytest.raises(ValueError, match="one consistently oriented plane"):
        subdivide_q4_pressure_faces(
            (first, S3PatchFace(second.node_ids, bent)),
            u_divisions=2, v_divisions=2, first_pressure_node_id=100,
            maximum_pressure_nodes=20, maximum_pressure_faces=8,
        )


def test_smooth_contact_tangent_pulls_back_through_virtual_nodes():
    source = S3PatchFace(
        (1, 2, 3, 4), np.asarray(((0., 0., 0.01), (1., 0., 0.01),
                                 (1., 1., 0.01), (0., 1., 0.01))),
    )
    target = S3PatchFace(
        (5, 6, 7), np.asarray(((0.13, 0.17, 0.),
                              (0.84, 0.21, 0.), (0.31, 0.88, 0.))),
    )
    overlay = subdivide_q4_pressure_faces(
        (source,), u_divisions=2, v_divisions=2,
        first_pressure_node_id=8, reserved_node_ids=target.node_ids,
        maximum_pressure_nodes=9, maximum_pressure_faces=4,
    )
    geometry = MovingS3PatchPrimalMortarPenalty(
        overlay.faces, (target,), 100_000.,
        difference_step=1e-6, maximum_energy_evaluations=100_000,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=128,
        q4_source_derivative_tolerances=(1e-5, 1e-5, 1e-5, 1e-5),
        differentiation="jet2_q4_s3", maximum_candidate_pairs=4,
    )
    law = MovingS3PatchProjectedNormal(geometry)
    transfer = sparse.block_diag((
        sparse.kron(overlay.interpolation, sparse.eye(3), format="csr"),
        sparse.eye(9),
    ), format="csr")
    structural = np.concatenate((
        source.reference_positions + np.asarray((0., 0., -0.010025)),
        target.reference_positions,
    )).reshape(-1)
    multiplier = np.zeros(len(overlay.pressure_node_ids))

    def response(coordinates, *, tangent):
        physical = (transfer @ coordinates).reshape(-1, 3)
        work = law.work(physical[:len(overlay.pressure_node_ids)],
                        physical[len(overlay.pressure_node_ids):],
                        multiplier, tangent=tangent)
        force = transfer.T @ work.residual
        stiffness = (transfer.T @ work.tangent @ transfer
                     if tangent else None)
        return np.asarray(force).reshape(-1), stiffness

    force, tangent = response(structural, tangent=True)
    assert np.all(np.isfinite(force))
    direction = np.random.default_rng(7).normal(size=len(structural))
    direction /= np.linalg.norm(direction)
    step = 2e-7
    plus, _ = response(structural + step*direction, tangent=False)
    minus, _ = response(structural - step*direction, tangent=False)
    difference = (plus-minus)/(2*step)
    predicted = tangent @ direction
    assert np.linalg.norm(difference-predicted) / np.linalg.norm(predicted) < 1e-5
