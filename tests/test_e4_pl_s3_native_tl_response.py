from __future__ import annotations

import copy

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3
from anyfesolver.fe_core import Material
from _e4_pl_s3_native_trial import native_trial_for_increment


def _fixture() -> tuple[np.ndarray, np.ndarray, Material]:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.3, 1.0, 0.0)),
        dtype=float,
    )
    triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    return nodes, triads, Material("steel", 210.0e9, 0.3, density=7850.0)


def _zero_layered_state(num_layers: int) -> dict[str, np.ndarray]:
    points = len(s3.TRIANGLE_QUADRATURE) * num_layers
    return {
        "kinematic_layer_strain": np.zeros((points, 3), dtype=float),
        "station_generalized_strain": np.zeros(
            (len(s3.TRIANGLE_QUADRATURE), 8), dtype=float
        ),
        "plastic_strain": np.zeros((points, 3), dtype=float),
        "alpha": np.zeros(points, dtype=float),
    }


def _literal_isotropic_section(thickness: float) -> np.ndarray:
    elastic_modulus = 210.0e9
    poisson_ratio = 0.3
    membrane = elastic_modulus / (1.0 - poisson_ratio**2) * np.asarray(
        (
            (1.0, poisson_ratio, 0.0),
            (poisson_ratio, 1.0, 0.0),
            (0.0, 0.0, (1.0 - poisson_ratio) / 2.0),
        ),
        dtype=float,
    )
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    section = np.zeros((8, 8), dtype=float)
    section[:3, :3] = thickness * membrane
    section[3:6, 3:6] = thickness**3 * membrane / 12.0
    section[6:, 6:] = (
        (5.0 / 6.0) * thickness * shear_modulus * np.eye(2)
    )
    return section


def _literal_linear_uncondensed(
    nodes: np.ndarray,
    thickness: float,
) -> np.ndarray:
    local = nodes[:, :2]
    determinant = float(
        np.linalg.det(
            np.asarray(
                (
                    (local[1, 0] - local[0, 0], local[1, 1] - local[0, 1]),
                    (local[2, 0] - local[0, 0], local[2, 1] - local[0, 1]),
                )
            )
        )
    )
    constitutive = _literal_isotropic_section(thickness)
    result = np.zeros((17, 17), dtype=float)
    for r, s, weight in s3.TRIANGLE_QUADRATURE:
        operator = s3._kinematic_matrix(local, r, s)
        result += (
            abs(determinant)
            * float(weight)
            * (operator.T @ constitutive @ operator)
        )
    return result


@pytest.mark.parametrize("num_layers", [3, 5, 7])
def test_zero_layered_native_tangent_recovers_the_frozen_linear_core(
    num_layers: int,
) -> None:
    nodes, triads, material = _fixture()
    thickness = 0.1
    native_trial, increment, _store = native_trial_for_increment(
        nodes, triads, np.zeros(20)
    )
    force, tangent, trial = s3._native_layered_uncondensed_response(
        nodes,
        triads,
        increment,
        nodes,
        np.eye(3),
        material,
        0.0,
        thickness,
        _zero_layered_state(num_layers),
        num_layers,
        native_rotation_trial=native_trial,
    )
    source_columns = np.concatenate(
        (s3.PHYSICAL_EXTERNAL_INDICES, np.asarray((18, 19)))
    )
    expected = _literal_linear_uncondensed(nodes, thickness)

    np.testing.assert_array_equal(force, np.zeros(20))
    np.testing.assert_allclose(
        tangent[np.ix_(source_columns, source_columns)],
        expected,
        rtol=3.0e-14,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(tangent[:, (5, 11, 17)], 0.0)
    np.testing.assert_array_equal(tangent[(5, 11, 17), :], 0.0)
    assert trial["kinematic_layer_strain"].shape == (7 * num_layers, 3)
    assert trial["station_generalized_resultant"].shape == (7, 8)


def test_bubble_newton_schur_and_directional_tangent_are_consistent() -> None:
    nodes, triads, material = _fixture()
    state = _zero_layered_state(3)
    original = copy.deepcopy(state)

    def builder(increment: np.ndarray):
        native_trial, exact, _store = native_trial_for_increment(
            nodes, triads, increment
        )
        return s3._native_layered_uncondensed_response(
            nodes,
            triads,
            exact,
            nodes,
            np.eye(3),
            material,
            0.0,
            0.1,
            state,
            3,
            native_rotation_trial=native_trial,
        )

    rng = np.random.default_rng(20260825)
    external = 2.0e-4 * rng.standard_normal(18)
    force, tangent, _trial, metadata = s3._solve_native_bubble_equilibrium(
        external,
        np.zeros(2),
        builder,
    )
    bubble = np.asarray(metadata["bubble_increment"])
    full_force, full_tangent, _full_trial = builder(
        np.concatenate((external, bubble))
    )
    expected = full_tangent[:18, :18] - full_tangent[:18, 18:] @ np.linalg.solve(
        full_tangent[18:, 18:], full_tangent[18:, :18]
    )
    residual_correction = np.linalg.solve(
        full_tangent[18:, 18:], full_force[18:]
    )
    expected_force = (
        full_force[:18] - full_tangent[:18, 18:] @ residual_correction
    )
    np.testing.assert_allclose(force, expected_force, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tangent, expected, rtol=2.0e-14, atol=1.0e-5)
    assert metadata["bubble_force_correction_excluded"] is False
    assert (
        metadata["bubble_force_condensation_id"]
        == s3.BUBBLE_FORCE_CONDENSATION_ID
    )
    assert np.linalg.norm(full_force[18:], ord=np.inf) <= (
        s3._BUBBLE_RELATIVE_TOLERANCE
        * float(metadata["bubble_residual_scale"])
    )

    for _index in range(3):
        direction = rng.standard_normal(18)
        direction /= np.linalg.norm(direction)
        step = 1.0e-7
        plus = s3._solve_native_bubble_equilibrium(
            external + step * direction,
            bubble,
            builder,
        )[0]
        minus = s3._solve_native_bubble_equilibrium(
            external - step * direction,
            bubble,
            builder,
        )[0]
        finite_difference = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(
            tangent @ direction,
            finite_difference,
            rtol=5.0e-8,
            atol=2.0e-2,
        )

    for key in original:
        np.testing.assert_array_equal(state[key], original[key])


def test_generalized_section_uses_the_same_native_geometric_derivatives() -> None:
    nodes, triads, _material = _fixture()
    section = _literal_isotropic_section(0.1)
    section[:3, 3:6] = np.diag((3.0e4, 2.0e4, 1.0e4))
    section[3:6, :3] = section[:3, 3:6].T
    committed = np.zeros((7, 8), dtype=float)
    rng = np.random.default_rng(17)
    increment = 1.0e-4 * rng.standard_normal(20)

    native_trial, increment, _store = native_trial_for_increment(
        nodes, triads, increment
    )
    force, tangent, trial = s3._native_generalized_uncondensed_response(
        nodes,
        triads,
        increment,
        nodes,
        np.eye(3),
        section,
        committed,
        native_rotation_trial=native_trial,
    )
    direction = rng.standard_normal(20)
    direction /= np.linalg.norm(direction)
    step = 1.0e-7
    plus_trial, plus_increment, _plus_store = native_trial_for_increment(
        nodes, triads, increment + step * direction
    )
    force_plus = s3._native_generalized_uncondensed_response(
        nodes,
        triads,
        plus_increment,
        nodes,
        np.eye(3),
        section,
        committed,
        native_rotation_trial=plus_trial,
    )[0]
    minus_trial, minus_increment, _minus_store = native_trial_for_increment(
        nodes, triads, increment - step * direction
    )
    force_minus = s3._native_generalized_uncondensed_response(
        nodes,
        triads,
        minus_increment,
        nodes,
        np.eye(3),
        section,
        committed,
        native_rotation_trial=minus_trial,
    )[0]
    np.testing.assert_allclose(
        tangent @ direction,
        (force_plus - force_minus) / (2.0 * step),
        rtol=2.0e-8,
        atol=2.0e-2,
    )
    assert np.linalg.norm(force) > 0.0
    assert trial["generalized_section"] is True
    assert trial["recovery_scope"] == "section_resultants_only"


def test_bubble_solver_failure_is_typed_and_never_returns_partial_evidence() -> None:
    def singular_builder(increment: np.ndarray):
        force = np.zeros(20)
        force[18:] = (1.0, -1.0)
        return force, np.zeros((20, 20)), {"leak": increment.copy()}

    with pytest.raises(
        s3.S3BubbleEquilibriumError,
        match="singular or ill-conditioned",
    ):
        s3._solve_native_bubble_equilibrium(
            np.zeros(18),
            np.zeros(2),
            singular_builder,
        )
