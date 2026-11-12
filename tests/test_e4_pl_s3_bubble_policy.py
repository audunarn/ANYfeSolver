from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import anyfesolver.e4_pl_s3_element as s3
from anyfesolver.e4_pl_s3_state import (
    BUBBLE_CONDITION_LIMIT,
    BUBBLE_FORCE_CONDENSATION_ID,
    BUBBLE_LINE_SEARCH_MIN_FACTOR,
    BUBBLE_LINE_SEARCH_REDUCTION,
    BUBBLE_MAX_ITERATIONS,
    BUBBLE_RELATIVE_TOLERANCE,
    BUBBLE_STEP_TOLERANCE,
    formulation_fingerprint_payload,
)
from anyfesolver.fe_core import Material
from _e4_pl_s3_native_trial import native_trial_for_increment


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/reference_cases/e4_pl_s3_formulation_contract.json"


def _fixture() -> tuple[np.ndarray, np.ndarray, Material]:
    nodes = np.asarray(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.3, 1.0, 0.0)),
        dtype=float,
    )
    triads = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    material = Material("steel", 210.0e9, 0.3, density=7850.0)
    return nodes, triads, material


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


def test_residual_converged_ill_conditioned_block_is_rejected_before_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tangent = np.eye(20, dtype=float)
    tangent[18:, 18:] = np.diag(
        (1.0, 1.0 / (10.0 * BUBBLE_CONDITION_LIMIT))
    )
    tangent[18:, :18] = 1.0
    tangent[:18, 18:] = 1.0

    def builder(_increment: np.ndarray):
        return np.zeros(20, dtype=float), tangent.copy(), {"candidate": "only"}

    solve_called = False
    original_solve = np.linalg.solve

    def forbidden_solve(*args: object, **kwargs: object):
        nonlocal solve_called
        solve_called = True
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(s3.np.linalg, "solve", forbidden_solve)
    with pytest.raises(
        s3.S3BubbleEquilibriumError,
        match="singular or ill-conditioned",
    ):
        s3._solve_native_bubble_equilibrium(
            np.zeros(18, dtype=float),
            np.zeros(2, dtype=float),
            builder,
        )

    assert solve_called is False


def test_tolerated_nonzero_bubble_residual_is_condensed_from_external_force() -> None:
    bubble_block = np.asarray(((4.0, 0.5), (0.5, 3.0)), dtype=float)
    external_bubble = np.zeros((18, 2), dtype=float)
    external_bubble[0] = (2.0, -1.0)
    external_bubble[7] = (-0.5, 3.0)
    external_bubble[17] = (1.25, 0.75)
    external_block = np.diag(np.linspace(2.0, 19.0, 18))
    tangent = np.block(
        [
            [external_block, external_bubble],
            [external_bubble.T, bubble_block],
        ]
    )
    external_force = np.linspace(-0.5, 0.5, 18)
    bubble_residual = np.asarray(
        (
            0.25 * BUBBLE_RELATIVE_TOLERANCE,
            -0.5 * BUBBLE_RELATIVE_TOLERANCE,
        ),
        dtype=float,
    )
    full_force = np.concatenate((external_force, bubble_residual))
    trial = {"candidate": "accepted"}

    def builder(_increment: np.ndarray):
        return full_force.copy(), tangent.copy(), trial

    force, condensed, returned_trial, metadata = (
        s3._solve_native_bubble_equilibrium(
            np.zeros(18, dtype=float),
            np.zeros(2, dtype=float),
            builder,
        )
    )
    expected_force = external_force - external_bubble @ np.linalg.solve(
        bubble_block, bubble_residual
    )
    expected_tangent = external_block - external_bubble @ np.linalg.solve(
        bubble_block, external_bubble.T
    )

    assert np.linalg.norm(bubble_residual, ord=np.inf) > 0.0
    assert np.linalg.norm(bubble_residual, ord=np.inf) <= (
        BUBBLE_RELATIVE_TOLERANCE * metadata["bubble_residual_scale"]
    )
    np.testing.assert_array_equal(force, expected_force)
    np.testing.assert_array_equal(condensed, expected_tangent)
    assert returned_trial is trial


@pytest.mark.parametrize(
    "num_layers",
    [
        3.0,
        np.float64(3.0),
        True,
        np.bool_(True),
        2,
        4,
        6,
        12,
    ],
)
def test_native_layered_response_rejects_nonexact_or_unsupported_layer_counts(
    num_layers: object,
) -> None:
    nodes, triads, material = _fixture()
    state_layers = int(num_layers)
    native_trial, exact_increment, _store = native_trial_for_increment(
        nodes,
        triads,
        np.zeros(20, dtype=float),
    )
    with pytest.raises(
        ValueError,
        match=r"num_layers must be one of \[3, 5, 7, 9, 11\]",
    ):
        s3._native_layered_uncondensed_response(
            nodes,
            triads,
            exact_increment,
            nodes,
            np.eye(3, dtype=float),
            material,
            0.0,
            0.1,
            _zero_layered_state(state_layers),
            num_layers,  # type: ignore[arg-type]
            native_rotation_trial=native_trial,
        )


def test_bubble_equilibrium_policy_is_hash_and_contract_bound() -> None:
    expected = {
        "condition_limit": BUBBLE_CONDITION_LIMIT,
        "force_condensation_id": BUBBLE_FORCE_CONDENSATION_ID,
        "line_search_min_factor": BUBBLE_LINE_SEARCH_MIN_FACTOR,
        "line_search_reduction": BUBBLE_LINE_SEARCH_REDUCTION,
        "max_iterations": BUBBLE_MAX_ITERATIONS,
        "relative_tolerance": BUBBLE_RELATIVE_TOLERANCE,
        "step_tolerance": BUBBLE_STEP_TOLERANCE,
    }
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert (
        formulation_fingerprint_payload()["bubble_equilibrium_policy"]
        == expected
    )
    assert (
        contract["native_nonlinear_kinematics"]["bubble_equilibrium_policy"]
        == expected
    )
