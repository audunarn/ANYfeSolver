"""Analytical plane-stress algorithmic-tangent qualification."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver.plasticity import (
    PlaneStressConvergenceError,
    plane_stress_numerical_tangent,
    plane_stress_return_map,
    plane_stress_tangent_diagnostics,
)
from anyfesolver.plasticity_qualification import (
    E_STEEL,
    NU_STEEL,
    algorithmic_tangent_path_metrics,
    algorithmic_tangent_performance_metrics,
    global_newton_tangent_benchmark_metrics,
    reference_plastic_curve,
)


def test_analytical_tangent_matches_oracle_across_qualified_paths() -> None:
    metrics = algorithmic_tangent_path_metrics()

    assert metrics["method"] == "analytical_implicit_consistent"
    assert set(metrics["cases"]) == {
        "elastic",
        "yielding",
        "linear_hardening",
        "power_law_hardening",
        "near_singular_plane_stress",
        "unloading",
    }
    assert metrics["max_relative_error"] < 1.0e-7
    assert metrics["max_symmetry_relative_error"] < 1.0e-14
    assert metrics["max_fallback_count"] == 0
    assert (
        metrics["cases"]["near_singular_plane_stress"][
            "elastic_matrix_condition"
        ]
        > 1.0e6
    )
    assert metrics["cases"]["unloading"]["alpha_increment"] == pytest.approx(0.0)
    assert (
        metrics["cases"]["unloading"]["plastic_strain_increment_norm"]
        == pytest.approx(0.0)
    )


def test_explicit_numerical_mode_is_the_central_difference_oracle() -> None:
    curve = reference_plastic_curve()
    strain = np.array(
        [[0.003, 0.001, 0.0005], [0.008, -0.001, 0.0015]],
        dtype=float,
    )
    plastic = np.zeros_like(strain)
    alpha = np.zeros(strain.shape[0], dtype=float)

    _stress, requested, _plastic, _alpha = plane_stress_return_map(
        strain,
        plastic,
        alpha,
        E_STEEL,
        NU_STEEL,
        curve,
        tangent_method="numerical",
    )
    oracle = plane_stress_numerical_tangent(
        strain,
        plastic,
        alpha,
        E_STEEL,
        NU_STEEL,
        curve,
    )

    np.testing.assert_array_equal(requested, oracle)


def test_invalid_tangent_method_fails_closed() -> None:
    curve = reference_plastic_curve()
    with pytest.raises(ValueError, match="tangent_method"):
        plane_stress_return_map(
            np.array([[0.003, 0.0, 0.0]], dtype=float),
            np.zeros((1, 3), dtype=float),
            np.zeros(1, dtype=float),
            E_STEEL,
            NU_STEEL,
            curve,
            tangent_method="secant",
        )


def test_ill_conditioning_alone_retains_finite_analytical_tangent() -> None:
    curve = reference_plastic_curve()
    strain = np.array([[4.0e-15, -4.0e-15, 1.0e-15]], dtype=float)
    plastic = np.zeros_like(strain)
    alpha = np.zeros(1, dtype=float)
    pathological_nu = -1.0 + 5.0e-13

    diagnostics = plane_stress_tangent_diagnostics(
        strain,
        plastic,
        alpha,
        E_STEEL,
        pathological_nu,
        curve,
    )
    _stress, automatic, _plastic, _alpha = plane_stress_return_map(
        strain,
        plastic,
        alpha,
        E_STEEL,
        pathological_nu,
        curve,
    )
    oracle = plane_stress_numerical_tangent(
        strain,
        plastic,
        alpha,
        E_STEEL,
        pathological_nu,
        curve,
    )

    assert diagnostics["fallback_count"] == 0
    assert diagnostics["fallback_indices"] == []
    assert np.all(np.isfinite(automatic))
    np.testing.assert_allclose(automatic, oracle, rtol=1.0e-8)


def test_numerical_oracle_uses_representable_steps_near_stability_boundary() -> None:
    curve = reference_plastic_curve()
    strain = np.array([[0.003, 0.001, 0.0005]], dtype=float)
    pathological_nu = -1.0 + 5.0e-13
    oracle = plane_stress_numerical_tangent(
        strain,
        strain.copy(),
        np.zeros(1, dtype=float),
        E_STEEL,
        pathological_nu,
        curve,
    )
    expected = np.asarray(
        [
            [
                E_STEEL / (1.0 - pathological_nu**2),
                E_STEEL * pathological_nu / (1.0 - pathological_nu**2),
                0.0,
            ],
            [
                E_STEEL * pathological_nu / (1.0 - pathological_nu**2),
                E_STEEL / (1.0 - pathological_nu**2),
                0.0,
            ],
            [
                0.0,
                0.0,
                E_STEEL / (2.0 * (1.0 + pathological_nu)),
            ],
        ],
        dtype=float,
    )
    assert np.all(np.linalg.norm(oracle[0], axis=0) > 0.0)
    np.testing.assert_allclose(oracle[0], expected, rtol=1.0e-10)


def test_invalid_analytical_tangent_mask_retains_guarded_fallback() -> None:
    from anyfesolver.plasticity import _analytical_tangent_fallback_mask

    tangent = np.eye(3, dtype=float)[None, :, :]
    tangent[0, 1, 1] = np.nan
    mask = _analytical_tangent_fallback_mask(
        tangent,
        E=E_STEEL,
        nu=NU_STEEL,
    )
    np.testing.assert_array_equal(mask, [True])


def test_safeguarded_local_solve_recovers_newton_iteration_exhaustion() -> None:
    curve = reference_plastic_curve()
    strain = np.array([[0.16, 0.025, 0.01]], dtype=float)
    stress, tangent, _plastic, alpha = plane_stress_return_map(
        strain,
        np.zeros((1, 3), dtype=float),
        np.zeros(1, dtype=float),
        E_STEEL,
        NU_STEEL,
        curve,
        max_iterations=1,
    )

    assert abs(
        float(
            stress[0, 0] ** 2
            - stress[0, 0] * stress[0, 1]
            + stress[0, 1] ** 2
            + 3.0 * stress[0, 2] ** 2
            - curve.flow_stress(alpha)[0] ** 2
        )
    ) / max(curve.flow_stress(alpha)[0] ** 2, 1.0) < 1.0e-8
    assert np.all(np.isfinite(tangent))


def test_local_convergence_guard_fails_closed_on_invalid_status() -> None:
    from anyfesolver.plasticity import _require_local_convergence

    with pytest.raises(PlaneStressConvergenceError, match="maximum scaled residual"):
        _require_local_convergence(
            np.array([False]),
            np.array([1.0]),
            max_iterations=30,
        )


def test_analytical_tangent_reduces_constitutive_update_work() -> None:
    batch = algorithmic_tangent_performance_metrics(num_points=256, repeats=2)

    assert batch["return_map_evaluations_per_update"] == {
        "analytical": 1,
        "numerical": 7,
    }
    assert batch["tangent_derivative_samples"] == {
        "analytical": 0,
        "numerical": 6,
    }
    assert batch["analytical_seconds"] > 0.0
    assert batch["numerical_seconds"] > 0.0


def test_numerical_oracle_validates_inputs_and_accepts_array_like_values() -> None:
    curve = reference_plastic_curve()
    tangent = plane_stress_numerical_tangent(
        [[0.003, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [0.0],
        E_STEEL,
        NU_STEEL,
        curve,
    )
    assert tangent.shape == (1, 3, 3)
    for step in (0.0, -1.0e-7, np.nan, np.inf):
        with pytest.raises(ValueError, match="step must be finite and positive"):
            plane_stress_numerical_tangent(
                [[0.003, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [0.0],
                E_STEEL,
                NU_STEEL,
                curve,
                step=step,
            )


def test_public_return_map_validates_shape_finite_values_and_material_inputs() -> None:
    curve = reference_plastic_curve()
    valid = np.zeros((2, 3), dtype=float)
    with pytest.raises(ValueError, match="same shape"):
        plane_stress_return_map(
            valid,
            np.zeros((1, 3)),
            np.zeros(2),
            E_STEEL,
            NU_STEEL,
            curve,
        )
    invalid = valid.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        plane_stress_return_map(
            invalid,
            valid,
            np.zeros(2),
            E_STEEL,
            NU_STEEL,
            curve,
        )
    with pytest.raises(ValueError, match="strain must have shape"):
        plane_stress_return_map(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0],
            E_STEEL,
            NU_STEEL,
            curve,
        )
    with pytest.raises(ValueError, match="E must be finite and positive"):
        plane_stress_return_map(
            valid,
            valid,
            np.zeros(2),
            0.0,
            NU_STEEL,
            curve,
        )
    with pytest.raises(ValueError, match="nu must be finite"):
        plane_stress_return_map(
            valid,
            valid,
            np.zeros(2),
            E_STEEL,
            0.5,
            curve,
        )
    with pytest.raises(ValueError, match="positive integer"):
        plane_stress_return_map(
            valid,
            valid,
            np.zeros(2),
            E_STEEL,
            NU_STEEL,
            curve,
            max_iterations=0,
        )


def test_global_plastic_shell_newton_has_solution_reaction_and_state_parity() -> None:
    metrics = global_newton_tangent_benchmark_metrics()

    assert metrics["analytical"]["status"] == "completed"
    assert metrics["numerical"]["status"] == "completed"
    assert metrics["analytical"]["load_factor"] == pytest.approx(1.0)
    assert metrics["numerical"]["load_factor"] == pytest.approx(1.0)
    assert (
        metrics["analytical"]["total_newton_iterations"]
        <= metrics["numerical"]["total_newton_iterations"]
    )
    assert metrics["analytical"]["state_summary"]["yielded_element_count"] == 8
    assert metrics["numerical"]["state_summary"]["yielded_element_count"] == 8
    assert metrics["analytical"]["reaction_norm"] > 0.0
    assert metrics["numerical"]["reaction_norm"] > 0.0
    assert max(metrics["parity"].values()) < 1.0e-8
    # Timing is evidence, not a flaky correctness gate.
    assert metrics["speedup"] > 0.0
