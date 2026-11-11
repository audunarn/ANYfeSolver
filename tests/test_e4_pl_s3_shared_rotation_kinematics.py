"""Focused objectivity checks for shared-Q S3 physical kinematics."""

from __future__ import annotations

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3
from anyfesolver._native_rotation_state import (
    create_native_rotation_state_store,
    rotation_exponential,
)
from anyfesolver.e4_pl_s3_state import reconstruct_director_triad
from _e4_pl_s3_native_trial import native_trial_for_increment


def _triads(normal: np.ndarray) -> np.ndarray:
    triad = reconstruct_director_triad(np.asarray(normal, dtype=np.float64))
    return np.repeat(triad[None, :, :], 4, axis=0)


def _rigid_increment(nodes: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
    rotation = rotation_exponential(rotation_vector)
    increment = np.zeros(20, dtype=np.float64)
    for node, coordinate in enumerate(np.asarray(nodes, dtype=np.float64)):
        increment[6 * node : 6 * node + 3] = (rotation - np.eye(3)) @ coordinate
        increment[6 * node + 3 : 6 * node + 6] = rotation_vector
    return increment


def _assert_all_station_strains_zero(
    nodes: np.ndarray,
    triads: np.ndarray,
    increment: np.ndarray,
    reference_nodes: np.ndarray,
    reference_frame: np.ndarray,
    native_trial: object,
) -> None:
    for r, s, _weight in s3.TRIANGLE_QUADRATURE:
        values, _jacobian, _hessian = s3._native_incremental_strain_jets(
            nodes,
            triads,
            r,
            s,
            increment,
            reference_nodes=reference_nodes,
            reference_frame=reference_frame,
            native_rotation_trial=native_trial,
        )
        np.testing.assert_allclose(values, 0.0, rtol=0.0, atol=1.2e-14)


def test_complete_physical_kernel_is_objective_under_finite_shared_rotation() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (2.3, 0.1, 0.0), (0.37, 1.41, 0.0)),
        dtype=np.float64,
    )
    triads = _triads(np.asarray((0.0, 0.0, 1.0)))
    rotation_vector = np.asarray((0.31, -0.23, 0.17), dtype=np.float64)
    native_trial, increment, _store = native_trial_for_increment(
        nodes,
        triads,
        _rigid_increment(nodes, rotation_vector),
    )
    _assert_all_station_strains_zero(
        nodes,
        triads,
        increment,
        nodes,
        np.eye(3),
        native_trial,
    )
    updated = s3._update_native_director_triads(
        triads,
        increment,
        native_rotation_trial=native_trial,
    )
    np.testing.assert_allclose(
        updated[:3, :, 2], native_trial.trial_directors, rtol=0.0, atol=2.0e-15
    )
    expected_bubble = rotation_exponential(rotation_vector) @ triads[3, :, 2]
    np.testing.assert_allclose(updated[3, :, 2], expected_bubble, atol=2.0e-15)


def test_noncommuting_rebase_uses_left_product_not_added_rotation_vector() -> None:
    reference = np.asarray(
        ((0.0, 0.0, 0.0), (1.4, 0.2, 0.0), (0.1, 1.2, 0.0)),
        dtype=np.float64,
    )
    first_vector = np.asarray((0.22, -0.11, 0.07), dtype=np.float64)
    second_vector = np.asarray((-0.09, 0.18, 0.13), dtype=np.float64)
    first = rotation_exponential(first_vector)
    second = rotation_exponential(second_vector)
    committed_nodes = (first @ reference.T).T
    committed_normal = first @ np.asarray((0.0, 0.0, 1.0))
    committed_triads = _triads(committed_normal)
    increment = _rigid_increment(committed_nodes, second_vector)
    committed_q = np.repeat(first[None, :, :], 3, axis=0)
    committed_theta = np.repeat(first_vector[None, :], 3, axis=0)
    native_trial, increment, _store = native_trial_for_increment(
        committed_nodes,
        committed_triads,
        increment,
        committed_rotation_matrices=committed_q,
        committed_rotation_coordinates=committed_theta,
    )
    frame, _local, _quality = s3.triangle_frame(
        reference,
        np.asarray((0.0, 0.0, 1.0)),
    )
    _assert_all_station_strains_zero(
        committed_nodes,
        committed_triads,
        increment,
        reference,
        frame,
        native_trial,
    )
    expected_q = second @ first
    np.testing.assert_allclose(native_trial.trial_rotation_matrices[0], expected_q)
    assert np.linalg.norm(
        expected_q - rotation_exponential(first_vector + second_vector)
    ) > 1.0e-2


def test_shared_node_q_supports_distinct_element_directors_at_a_crease() -> None:
    all_nodes = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    node_ids = (1, 2, 3, 4)
    rotation_vector = np.asarray((0.27, -0.16, 0.12), dtype=np.float64)
    rotation = rotation_exponential(rotation_vector)
    full_trial = np.zeros(24, dtype=np.float64)
    trial_coordinates = (rotation @ all_nodes.T).T
    for row, coordinate in enumerate(all_nodes):
        full_trial[6 * row : 6 * row + 3] = trial_coordinates[row] - coordinate
        full_trial[6 * row + 3 : 6 * row + 6] = rotation_vector
    store = create_native_rotation_state_store(
        node_ids,
        rotational_dofs={
            node_id: (6 * row + 3, 6 * row + 4, 6 * row + 5)
            for row, node_id in enumerate(node_ids)
        },
        coordinate_rows={node_id: row for row, node_id in enumerate(node_ids)},
        coordinate_node_ids=node_ids,
        committed_full_displacement=np.zeros(24),
        committed_full_coordinates=all_nodes,
    )
    token = store.begin_trial(full_trial, trial_coordinates)
    element_data = (
        ((1, 2, 3), np.asarray((0.0, 0.0, 1.0))),
        ((1, 2, 4), np.asarray((0.0, -1.0, 0.0))),
    )
    shared_views = []
    for index, (connectivity, normal) in enumerate(element_data):
        reference_nodes = all_nodes[np.asarray(connectivity) - 1]
        reference_directors = np.repeat(normal[None, :], 3, axis=0)
        view = store.element_view(
            f"crease-{index}",
            connectivity,
            reference_directors,
            trial_token=token,
        )
        triads = _triads(normal)
        increment = np.zeros(20, dtype=np.float64)
        by_node = increment[:18].reshape(3, 6)
        by_node[:, :3] = view.coordinate_increment
        by_node[:, 3:6] = view.rotation_coordinate_increment
        frame, _local, _quality = s3.triangle_frame(reference_nodes, normal)
        _assert_all_station_strains_zero(
            view.committed_coordinates,
            triads,
            increment,
            reference_nodes,
            frame,
            view,
        )
        shared_views.append(view)
    np.testing.assert_array_equal(
        shared_views[0].trial_rotation_matrices[:2],
        shared_views[1].trial_rotation_matrices[:2],
    )
    assert not np.allclose(
        shared_views[0].trial_directors[0], shared_views[1].trial_directors[0]
    )


def test_signed_surface_inversion_is_rejected_even_with_positive_gram_determinant() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    triads = _triads(np.asarray((0.0, 0.0, 1.0)))
    increment = np.zeros(20, dtype=np.float64)
    increment[6:9] = nodes[2] - nodes[1]
    increment[12:15] = nodes[1] - nodes[2]
    native_trial, increment, _store = native_trial_for_increment(
        nodes, triads, increment
    )
    edges = np.column_stack(
        (
            native_trial.trial_coordinates[1] - native_trial.trial_coordinates[0],
            native_trial.trial_coordinates[2] - native_trial.trial_coordinates[0],
        )
    )
    assert np.linalg.det(edges.T @ edges) > 0.0
    with pytest.raises(ValueError, match="inverted relative to transported directors"):
        s3._native_incremental_strain_jets(
            nodes,
            triads,
            1.0 / 3.0,
            1.0 / 3.0,
            increment,
            reference_nodes=nodes,
            reference_frame=np.eye(3),
            native_rotation_trial=native_trial,
        )

def test_redundant_element_delta_must_match_native_trial_exactly() -> None:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    triads = _triads(np.asarray((0.0, 0.0, 1.0)))
    native_trial, increment, _store = native_trial_for_increment(
        nodes,
        triads,
        _rigid_increment(nodes, np.asarray((0.08, -0.03, 0.11))),
    )
    bad = increment.copy()
    bad[0] = np.nextafter(bad[0], np.inf)
    with pytest.raises(ValueError, match="translational increment disagrees"):
        s3._native_incremental_strain_jets(
            nodes,
            triads,
            0.2,
            0.3,
            bad,
            reference_nodes=nodes,
            reference_frame=np.eye(3),
            native_rotation_trial=native_trial,
        )
