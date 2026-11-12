"""Independent covariance checks for the private native S3 TL update.

These tests derive rigid rotations, the published Eq. (11) director gauge,
and the D3 numbering action directly.  They intentionally target private
formulation kernels without establishing a supported public API.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

import anyfesolver.e4_pl_s3_element as s3
from anyfesolver.e4_pl_s3_state import reconstruct_director_triad
from _e4_pl_s3_native_trial import native_trial_for_increment


def _normalized(vector: np.ndarray) -> np.ndarray:
    made = np.asarray(vector, dtype=float)
    return made / np.linalg.norm(made)


def _independent_rodrigues(rotation_vector: np.ndarray) -> np.ndarray:
    """Evaluate ``exp(skew(rotation_vector))`` independently."""

    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    skew = np.asarray(
        (
            (0.0, -vector[2], vector[1]),
            (vector[2], 0.0, -vector[0]),
            (-vector[1], vector[0], 0.0),
        ),
        dtype=float,
    )
    if angle == 0.0:
        return np.eye(3)
    return (
        np.eye(3)
        + (math.sin(angle) / angle) * skew
        + ((1.0 - math.cos(angle)) / (angle * angle)) * (skew @ skew)
    )


def _independent_eq11_gauge(normal: np.ndarray) -> np.ndarray:
    """Reconstruct the global-e_y/Eq. (11) chart without production code."""

    unit = _normalized(normal)
    first = np.cross(np.asarray((0.0, 1.0, 0.0)), unit)
    if np.linalg.norm(first) <= 1.0e-12:
        first = np.cross(np.asarray((0.0, 0.0, 1.0)), unit)
    first = _normalized(first)
    second = _normalized(np.cross(unit, first))
    return np.column_stack((first, second, unit))


def _four_distinct_gauges() -> np.ndarray:
    normals = np.asarray(
        (
            (0.11, 0.24, 0.964),
            (0.32, -0.17, 0.932),
            (-0.21, 0.31, 0.927),
            (0.16, 0.43, 0.888),
        ),
        dtype=float,
    )
    triads = np.asarray([_independent_eq11_gauge(item) for item in normals])
    assert all(
        not np.allclose(triads[left], triads[right])
        for left in range(4)
        for right in range(left)
    )
    return triads


def _mixed_rigid_increment(
    nodes: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    rotation_vector = float(angle) * _normalized(axis)
    rigid_rotation = _independent_rodrigues(rotation_vector)
    increment = np.zeros(20, dtype=float)
    for node, coordinate in enumerate(nodes):
        base = 6 * node
        increment[base : base + 3] = (
            rigid_rotation - np.eye(3)
        ) @ coordinate
        increment[base + 3 : base + 6] = rotation_vector
    return increment


def test_mixed_axis_exact_rigid_q_residual_is_binary64_zero() -> None:
    """The shared full exponential gives complete finite rigid objectivity."""

    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (2.3, 0.1, 0.0), (0.37, 1.41, 0.0)),
        dtype=float,
    )
    triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    axis = np.asarray((0.31, -0.47, 0.826), dtype=float)
    for angle in (0.01, 0.31, 0.91):
        native_trial, increment, _store = native_trial_for_increment(
            nodes,
            triads,
            _mixed_rigid_increment(nodes, axis, angle),
        )
        values, _jacobian, _hessian = s3._native_incremental_strain_jets(
            nodes,
            triads,
            0.23,
            0.31,
            increment,
            reference_nodes=nodes,
            reference_frame=np.eye(3),
            native_rotation_trial=native_trial,
        )
        np.testing.assert_allclose(values, 0.0, rtol=0.0, atol=8.0e-15)


def test_eq11_commit_reconstructs_gauge_instead_of_transporting_it() -> None:
    """Eq. (11) resets tangent gauge after committing the new normal."""

    triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    triads[0] = _independent_eq11_gauge(
        np.asarray((0.17, -0.29, 0.941), dtype=float)
    )
    rotation = np.asarray((0.08, -0.06, 0.12), dtype=float)
    increment = np.zeros(20, dtype=float)
    increment[3:6] = rotation
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    native_trial, increment, _store = native_trial_for_increment(
        nodes, triads, increment
    )

    updated = s3._update_native_director_triads(
        triads,
        increment,
        native_rotation_trial=native_trial,
    )[0]

    expected_normal = (
        _independent_rodrigues(rotation) @ triads[0, :, 2]
    )
    expected = _independent_eq11_gauge(expected_normal)
    q_transported = _independent_rodrigues(rotation) @ triads[0]

    np.testing.assert_allclose(updated, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(updated.T @ updated, np.eye(3), atol=2.0e-15)
    assert np.linalg.det(updated) > 0.999999999999998
    assert np.linalg.norm(updated[:, :2] - q_transported[:, :2]) > 0.1
    np.testing.assert_allclose(updated[:, 2], q_transported[:, 2], atol=2.0e-15)


def test_exact_global_ey_normal_uses_global_ez_fallback() -> None:
    for sign in (-1.0, 1.0):
        normal = np.asarray((0.0, sign, 0.0), dtype=float)
        actual = reconstruct_director_triad(normal)
        expected = _independent_eq11_gauge(normal)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, 2], normal)
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=0.0)
        assert np.linalg.det(actual) == 1.0


def test_source_node_four_mapping_uses_hierarchical_values_and_own_gauge() -> None:
    triads = _four_distinct_gauges()
    increment = np.asarray(
        (
            0.0,
            0.0,
            0.0,
            0.07,
            -0.04,
            0.09,
            0.0,
            0.0,
            0.0,
            -0.08,
            0.06,
            0.04,
            0.0,
            0.0,
            0.0,
            0.05,
            0.09,
            -0.07,
            0.031,
            -0.022,
        ),
        dtype=float,
    )
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    native_trial, increment, _store = native_trial_for_increment(
        nodes, triads, increment
    )
    rotations = increment[:18].reshape(3, 6)[:, 3:6]
    expected_source_rotation = (
        np.mean(rotations, axis=0)
        + increment[18] * triads[3, :, 0]
        + increment[19] * triads[3, :, 1]
    )
    expected_source_normal = (
        _independent_rodrigues(expected_source_rotation) @ triads[3, :, 2]
    )
    expected_source_triad = _independent_eq11_gauge(expected_source_normal)
    updated = s3._update_native_director_triads(
        triads,
        increment,
        native_rotation_trial=native_trial,
    )
    for node in range(3):
        np.testing.assert_allclose(
            updated[node, :, 2], native_trial.trial_directors[node], atol=2.0e-15
        )
    np.testing.assert_allclose(updated[3], expected_source_triad, atol=2.0e-15)


def test_deformed_current_covariants_use_explicit_reference_basis() -> None:
    reference_nodes = np.asarray(
        ((0.0, 0.0, 0.0), (2.3, 0.1, 0.0), (0.37, 1.41, 0.0)),
        dtype=float,
    )
    current_gradient = np.asarray(
        ((1.16, 0.21, 0.0), (-0.13, 0.91, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    increment_gradient = np.asarray(
        ((0.07, -0.11, 0.0), (0.04, 0.055, 0.0), (0.0, 0.0, 0.0)),
        dtype=float,
    )
    current_nodes = reference_nodes @ current_gradient.T
    translation = reference_nodes @ increment_gradient.T
    increment = np.zeros(20, dtype=float)
    for node in range(3):
        increment[6 * node : 6 * node + 3] = translation[node]
    triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    native_trial, increment, _store = native_trial_for_increment(
        current_nodes, triads, increment
    )

    actual, _jacobian, _hessian = s3._native_incremental_strain_jets(
        current_nodes,
        triads,
        0.29,
        0.22,
        increment,
        reference_nodes=reference_nodes,
        reference_frame=np.eye(3),
        native_rotation_trial=native_trial,
    )
    twice_green_lagrange_increment = (
        current_gradient.T @ increment_gradient
        + increment_gradient.T @ current_gradient
        + increment_gradient.T @ increment_gradient
    )
    expected = np.asarray(
        (
            0.5 * twice_green_lagrange_increment[0, 0],
            0.5 * twice_green_lagrange_increment[1, 1],
            twice_green_lagrange_increment[0, 1],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)

    deformed_basis_result, _bad_jacobian, _bad_hessian = (
        s3._native_incremental_strain_jets(
            current_nodes,
            triads,
            0.29,
            0.22,
            increment,
            reference_nodes=current_nodes,
            reference_frame=np.eye(3),
            native_rotation_trial=native_trial,
        )
    )
    assert np.linalg.norm(deformed_basis_result[:3] - actual[:3]) > 0.04


def test_all_six_d3_numberings_commute_with_director_commit_update() -> None:
    triads = _four_distinct_gauges()
    increment = np.asarray(
        (
            0.01,
            -0.02,
            0.03,
            0.07,
            -0.04,
            0.09,
            -0.02,
            0.03,
            0.01,
            -0.08,
            0.06,
            0.04,
            0.04,
            -0.01,
            -0.03,
            0.05,
            0.09,
            -0.07,
            0.031,
            -0.022,
        ),
        dtype=float,
    )
    nodes = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    baseline_trial, baseline_increment, _store = native_trial_for_increment(
        nodes, triads, increment
    )
    baseline = s3._update_native_director_triads(
        triads,
        baseline_increment,
        native_rotation_trial=baseline_trial,
    )
    external_by_node = increment[:18].reshape(3, 6)

    for permutation in itertools.permutations(range(3)):
        numbered_triads = np.concatenate(
            (triads[:3][list(permutation)], triads[3:]),
            axis=0,
        )
        numbered_increment = np.concatenate(
            (
                external_by_node[list(permutation)].reshape(18),
                increment[18:],
            )
        )
        numbered_nodes = nodes[list(permutation)]
        numbered_trial, numbered_increment, _numbered_store = native_trial_for_increment(
            numbered_nodes,
            numbered_triads,
            numbered_increment,
        )
        actual = s3._update_native_director_triads(
            numbered_triads,
            numbered_increment,
            native_rotation_trial=numbered_trial,
        )
        expected = np.concatenate(
            (baseline[:3][list(permutation)], baseline[3:]),
            axis=0,
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=3.0e-15,
            err_msg=f"D3 numbering {permutation}",
        )
