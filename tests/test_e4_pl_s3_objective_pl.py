from __future__ import annotations

import math
import itertools

import numpy as np
import pytest

from anyfesolver.e4_pl_s3_element import (
    NODAL_ROTATION_UPDATE_POLICY_ID,
    NONLINEAR_PL_ENERGY_POLICY_ID,
    PL_PHASE_POLICY_ID,
    PL_TWIST_POLICY_ID,
    SURFACE_ROTATION_POLICY_ID,
    _objective_pl_energy_response,
    _pl_operators,
    _rodrigues_rotation,
    triangle_frame,
)
from anyfesolver.e4_pl_s3_state import (
    PL_MINIMUM_TWIST_DENOMINATOR,
    PL_PHASE_MARGIN,
)
from _e4_pl_s3_native_trial import native_trial_for_increment


def _reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=float,
    )
    frame, local, _quality = triangle_frame(
        nodes,
        np.asarray((0.0, 0.0, 1.0), dtype=float),
    )
    return nodes, frame, local


def _identity_rotations() -> np.ndarray:
    return np.broadcast_to(np.eye(3), (3, 3, 3)).copy()


def _swing_with_twist_denominator(denominator: float) -> np.ndarray:
    cosine = float(denominator) - 1.0
    sine = math.sqrt((1.0 - cosine) * (1.0 + cosine))
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        dtype=float,
    )


def _drill_rotation(angle: float) -> np.ndarray:
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _response(
    increment: np.ndarray,
    *,
    current_nodes: np.ndarray | None = None,
    committed_rotations: np.ndarray | None = None,
    committed_twist: np.ndarray | None = None,
    k_d: float = 2.75,
) -> dict[str, object]:
    reference, frame, _local = _reference()
    current = reference if current_nodes is None else current_nodes
    rotations = (
        _identity_rotations()
        if committed_rotations is None
        else committed_rotations
    )
    triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    native_trial, exact_increment, _store = native_trial_for_increment(
        current,
        triads,
        np.concatenate((np.asarray(increment, dtype=float), np.zeros(2))),
        committed_rotation_matrices=rotations,
    )
    return _objective_pl_energy_response(
        reference,
        frame,
        current,
        rotations,
        exact_increment[:18],
        np.zeros(3) if committed_twist is None else committed_twist,
        k_d,
        native_rotation_trial=native_trial,
    )


def test_identity_linearization_is_exactly_the_frozen_barycentric_pl_operator() -> None:
    reference, _frame, local = _reference()
    k_d = 3.25
    result = _response(np.zeros(18), k_d=k_d)
    constraint, gram, condensed = _pl_operators(local, k_d)

    np.testing.assert_array_equal(result["constraint"], np.zeros(3))
    np.testing.assert_array_equal(result["force"], np.zeros(18))
    np.testing.assert_allclose(
        result["constraint_jacobian"], constraint, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(result["gram"], gram, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result["tangent"], condensed, rtol=0.0, atol=4.0e-17)
    assert result["energy"] == 0.0
    assert reference.shape == (3, 3)


def test_finite_rigid_rotation_has_zero_pl_constraint_force_and_energy() -> None:
    reference, _frame, _local = _reference()
    rotation_vector = np.asarray((0.31, -0.23, 0.17), dtype=float)
    rotation = _rodrigues_rotation(rotation_vector)
    increment = np.zeros(18, dtype=float)
    for node in range(3):
        increment[6 * node : 6 * node + 3] = rotation @ reference[node] - reference[node]
        increment[6 * node + 3 : 6 * node + 6] = rotation_vector

    result = _response(increment)

    np.testing.assert_allclose(result["constraint"], np.zeros(3), rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(result["force"], np.zeros(18), rtol=0.0, atol=2.0e-15)
    assert abs(float(result["energy"])) <= 2.0e-30


def test_finite_uniform_drill_has_exact_unwrapped_twist_and_positive_work() -> None:
    angle = 0.37
    k_d = 2.75
    increment = np.zeros(18, dtype=float)
    increment[5::6] = angle

    result = _response(increment, k_d=k_d)

    np.testing.assert_allclose(
        result["constraint"], np.full(3, angle), rtol=0.0, atol=2.0e-15
    )
    assert float(result["energy"]) > 0.0
    assert np.linalg.norm(np.asarray(result["force"], dtype=float)) > 0.0
    np.testing.assert_array_equal(result["turn_count"], np.zeros(3, dtype=np.int64))
    constraint = np.asarray(result["constraint"], dtype=float)
    multiplier = np.asarray(result["multiplier"], dtype=float)
    gram = np.asarray(result["gram"], dtype=float)
    np.testing.assert_array_equal(multiplier, k_d * constraint)
    mixed = np.asarray(
        [
            sum(
                float(gram[row, column]) * float(multiplier[column])
                for column in range(3)
            )
            for row in range(3)
        ]
    )
    np.testing.assert_array_equal(result["mixed_constraint_force"], mixed)
    np.testing.assert_array_equal(result["constraint_conjugate"], mixed)
    assert result["energy"] == sum(
        0.5 * float(constraint[row]) * float(mixed[row]) for row in range(3)
    )


def test_twist_phase_is_unwrapped_nearest_the_committed_value() -> None:
    committed_angle = 3.0
    increment_angle = 0.3
    committed_rotation = _rodrigues_rotation(
        np.asarray((0.0, 0.0, committed_angle), dtype=float)
    )
    rotations = np.broadcast_to(committed_rotation, (3, 3, 3)).copy()
    increment = np.zeros(18, dtype=float)
    increment[5::6] = increment_angle

    result = _response(
        increment,
        committed_rotations=rotations,
        committed_twist=np.full(3, committed_angle),
    )

    np.testing.assert_allclose(
        result["constraint"],
        np.full(3, committed_angle + increment_angle),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_array_equal(result["turn_count"], np.ones(3, dtype=np.int64))


def test_energy_gradient_and_force_jacobian_match_directional_differences() -> None:
    increment = np.asarray(
        (
            0.018,
            -0.011,
            0.007,
            0.12,
            -0.08,
            0.16,
            -0.006,
            0.015,
            -0.004,
            -0.09,
            0.05,
            0.11,
            0.009,
            0.004,
            0.012,
            0.07,
            0.03,
            -0.13,
        ),
        dtype=float,
    )
    direction = np.linspace(-0.7, 0.9, 18, dtype=float)
    direction /= np.linalg.norm(direction)
    step = 2.0e-6
    centre = _response(increment)
    plus = _response(increment + step * direction)
    minus = _response(increment - step * direction)

    energy_derivative = (float(plus["energy"]) - float(minus["energy"])) / (2.0 * step)
    expected_energy_derivative = float(np.dot(centre["force"], direction))
    np.testing.assert_allclose(
        energy_derivative, expected_energy_derivative, rtol=2.0e-8, atol=2.0e-10
    )
    force_derivative = (
        np.asarray(plus["force"], dtype=float)
        - np.asarray(minus["force"], dtype=float)
    ) / (2.0 * step)
    expected_force_derivative = np.asarray(centre["tangent"], dtype=float) @ direction
    np.testing.assert_allclose(
        force_derivative, expected_force_derivative, rtol=3.0e-8, atol=3.0e-9
    )
    np.testing.assert_allclose(
        centre["tangent"], np.asarray(centre["tangent"]).T, rtol=0.0, atol=0.0
    )


def test_antipodal_swing_and_improper_rotation_fail_closed() -> None:
    antipodal = _identity_rotations()
    antipodal[0] = _rodrigues_rotation(np.asarray((math.pi, 0.0, 0.0)))
    with pytest.raises(ValueError, match="antipodal swing"):
        _response(np.zeros(18), committed_rotations=antipodal)

    improper = _identity_rotations()
    improper[1, 0, 0] = -1.0
    with pytest.raises(ValueError, match="not orthogonal|not (?:a )?proper"):
        _response(np.zeros(18), committed_rotations=improper)


@pytest.mark.parametrize(
    ("denominator", "must_fail"),
    (
        (0.5 * PL_MINIMUM_TWIST_DENOMINATOR, True),
        (PL_MINIMUM_TWIST_DENOMINATOR, False),
        (1.5 * PL_MINIMUM_TWIST_DENOMINATOR, False),
    ),
    ids=("below", "at", "above"),
)
def test_twist_denominator_guard_has_frozen_below_at_above_semantics(
    denominator: float,
    must_fail: bool,
) -> None:
    rotations = _identity_rotations()
    rotations[0] = _swing_with_twist_denominator(denominator)

    if must_fail:
        with pytest.raises(ValueError, match="antipodal swing"):
            _response(np.zeros(18), committed_rotations=rotations)
    else:
        result = _response(np.zeros(18), committed_rotations=rotations)
        assert np.all(np.isfinite(np.asarray(result["constraint"], dtype=float)))


@pytest.mark.parametrize(
    ("phase", "must_fail"),
    (
        (math.pi - 1.5 * PL_PHASE_MARGIN, False),
        (math.pi - PL_PHASE_MARGIN, True),
        (math.pi - 0.5 * PL_PHASE_MARGIN, True),
    ),
    ids=("below", "at", "above"),
)
def test_phase_guard_has_frozen_below_at_above_semantics(
    phase: float,
    must_fail: bool,
) -> None:
    rotations = _identity_rotations()
    rotations[0] = _drill_rotation(phase)

    if must_fail:
        with pytest.raises(ValueError, match="unique phase boundary"):
            _response(np.zeros(18), committed_rotations=rotations)
    else:
        result = _response(np.zeros(18), committed_rotations=rotations)
        assert float(np.asarray(result["constraint"])[0]) == phase


def test_objective_pl_result_binds_every_frozen_policy_identifier() -> None:
    result = _response(np.zeros(18))
    assert result["surface_rotation_policy_id"] == SURFACE_ROTATION_POLICY_ID
    assert result["twist_policy_id"] == PL_TWIST_POLICY_ID
    assert result["rotation_update_policy_id"] == NODAL_ROTATION_UPDATE_POLICY_ID
    assert result["phase_policy_id"] == PL_PHASE_POLICY_ID
    assert result["energy_policy_id"] == NONLINEAR_PL_ENERGY_POLICY_ID


def _numbered_frame(nodes: np.ndarray, owner_normal: np.ndarray) -> np.ndarray:
    normal = owner_normal / np.linalg.norm(owner_normal)
    first = nodes[1] - nodes[0]
    first = first - float(first @ normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, normal))


def test_objective_pl_energy_force_and_tangent_cover_all_six_d3_numberings() -> None:
    reference = np.asarray(
        (
            (0.13, -0.24, 0.31),
            (1.19, 0.08, 0.52),
            (0.22, 0.91, 0.44),
        ),
        dtype=float,
    )
    owner = np.cross(reference[1] - reference[0], reference[2] - reference[0])
    owner /= np.linalg.norm(owner)
    current = reference + np.asarray(
        ((0.014, -0.008, 0.005), (-0.006, 0.011, 0.009), (0.007, 0.004, -0.003))
    )
    committed_rotations = np.asarray(
        [
            _rodrigues_rotation(np.asarray((0.07, -0.03, 0.11))),
            _rodrigues_rotation(np.asarray((-0.05, 0.08, -0.04))),
            _rodrigues_rotation(np.asarray((0.02, 0.06, 0.09))),
        ]
    )
    increment = np.asarray(
        (
            0.006,
            -0.004,
            0.003,
            0.04,
            -0.02,
            0.03,
            -0.003,
            0.005,
            0.002,
            -0.03,
            0.01,
            0.025,
            0.004,
            0.001,
            -0.005,
            0.02,
            0.035,
            -0.015,
        )
    )
    committed_twist = np.asarray((0.08, -0.04, 0.06))
    baseline_triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    baseline_view, baseline_increment, _baseline_store = native_trial_for_increment(
        current,
        baseline_triads,
        np.concatenate((increment, np.zeros(2))),
        committed_rotation_matrices=committed_rotations,
    )
    baseline = _objective_pl_energy_response(
        reference,
        _numbered_frame(reference, owner),
        current,
        committed_rotations,
        baseline_increment[:18],
        committed_twist,
        4.2,
        native_rotation_trial=baseline_view,
    )

    for permutation in itertools.permutations(range(3)):
        indices = np.asarray(permutation, dtype=np.intp)
        block_map = np.asarray(
            [6 * old_node + component for old_node in indices for component in range(6)],
            dtype=np.intp,
        )
        transform = np.zeros((18, 18), dtype=float)
        transform[np.arange(18), block_map] = 1.0
        numbered_view, numbered_increment, _numbered_store = (
            native_trial_for_increment(
                current[indices],
                baseline_triads,
                np.concatenate((increment[block_map], np.zeros(2))),
                committed_rotation_matrices=committed_rotations[indices],
            )
        )
        numbered = _objective_pl_energy_response(
            reference[indices],
            _numbered_frame(reference[indices], owner),
            current[indices],
            committed_rotations[indices],
            numbered_increment[:18],
            committed_twist[indices],
            4.2,
            native_rotation_trial=numbered_view,
        )
        np.testing.assert_allclose(
            numbered["constraint"],
            np.asarray(baseline["constraint"])[indices],
            rtol=0.0,
            atol=3.0e-15,
        )
        assert float(numbered["energy"]) == pytest.approx(
            float(baseline["energy"]), rel=2.0e-14, abs=2.0e-16
        )
        np.testing.assert_allclose(
            numbered["force"],
            transform @ np.asarray(baseline["force"]),
            rtol=2.0e-12,
            atol=3.0e-14,
        )
        np.testing.assert_allclose(
            numbered["tangent"],
            transform @ np.asarray(baseline["tangent"]) @ transform.T,
            rtol=3.0e-12,
            atol=5.0e-13,
        )
