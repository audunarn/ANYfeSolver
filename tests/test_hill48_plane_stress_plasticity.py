"""Focused qualification of the material-axis Hill-48 return mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.materials import Hill48Yield
from anyfesolver.plasticity import (
    PlaneStressConvergenceError,
    hill48_plane_stress_equivalent_stress,
    hill48_plane_stress_numerical_tangent,
    hill48_plane_stress_return_map,
    plane_stress_elastic_matrix,
    plane_stress_return_map,
)


E1 = 150.0e9
E2 = 90.0e9
NU12 = 0.25
G12 = 40.0e9


def _orthotropic_elastic_matrix() -> np.ndarray:
    nu21 = NU12 * E2 / E1
    denominator = 1.0 - NU12 * nu21
    return np.asarray(
        [
            [E1 / denominator, NU12 * E2 / denominator, 0.0],
            [NU12 * E2 / denominator, E2 / denominator, 0.0],
            [0.0, 0.0, G12],
        ],
        dtype=float,
    )


def _orthotropic_yield() -> Hill48Yield:
    return Hill48Yield(
        X=300.0e6,
        Y=240.0e6,
        Z=270.0e6,
        S12=130.0e6,
        S13=140.0e6,
        S23=120.0e6,
    )


@dataclass(frozen=True)
class _LinearHardening:
    initial_flow: float
    modulus: float

    def flow_stress(self, alpha: np.ndarray) -> np.ndarray:
        values = np.asarray(alpha, dtype=float)
        return self.initial_flow + self.modulus * values

    def hardening_modulus(self, alpha: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(alpha, dtype=float), self.modulus)


def test_directional_strengths_and_perfect_plastic_return() -> None:
    model = _orthotropic_yield()
    elastic = _orthotropic_elastic_matrix()
    strengths = np.asarray(
        [
            [model.X, 0.0, 0.0],
            [0.0, model.Y, 0.0],
            [0.0, 0.0, model.S12],
        ],
        dtype=float,
    )

    equivalent = hill48_plane_stress_equivalent_stress(strengths, model)
    np.testing.assert_allclose(equivalent, model.X, rtol=1.0e-13)

    trial_stress = 1.35 * strengths
    strain = trial_stress @ np.linalg.inv(elastic).T
    stress, _tangent, plastic, alpha = hill48_plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(3, dtype=float),
        elastic,
        model,
    )

    np.testing.assert_allclose(
        hill48_plane_stress_equivalent_stress(stress, model),
        model.X,
        rtol=2.0e-10,
    )
    assert np.all(alpha > 0.0)
    assert np.all(np.linalg.norm(plastic, axis=1) > 0.0)


def test_hardening_scales_all_strengths_and_accumulates_alpha() -> None:
    model = _orthotropic_yield()
    elastic = _orthotropic_elastic_matrix()
    curve = _LinearHardening(initial_flow=250.0e6, modulus=2.0e9)
    zero_plastic = np.zeros((1, 3), dtype=float)
    zero_alpha = np.zeros(1, dtype=float)

    stress_1, _tangent_1, plastic_1, alpha_1 = (
        hill48_plane_stress_return_map(
            np.asarray([[0.0040, -0.0005, 0.0010]], dtype=float),
            zero_plastic,
            zero_alpha,
            elastic,
            model,
            curve,
        )
    )
    stress_2, _tangent_2, _plastic_2, alpha_2 = (
        hill48_plane_stress_return_map(
            np.asarray([[0.0070, -0.0008, 0.0018]], dtype=float),
            plastic_1,
            alpha_1,
            elastic,
            model,
            curve,
        )
    )

    expected_1 = model.X * curve.flow_stress(alpha_1) / curve.flow_stress(
        np.zeros(1)
    )
    expected_2 = model.X * curve.flow_stress(alpha_2) / curve.flow_stress(
        np.zeros(1)
    )
    np.testing.assert_allclose(
        hill48_plane_stress_equivalent_stress(stress_1, model),
        expected_1,
        rtol=2.0e-10,
    )
    np.testing.assert_allclose(
        hill48_plane_stress_equivalent_stress(stress_2, model),
        expected_2,
        rtol=2.0e-10,
    )
    assert alpha_2[0] > alpha_1[0] > 0.0
    assert expected_2[0] > expected_1[0] > model.X


def test_elastic_unloading_preserves_committed_hill_state() -> None:
    model = _orthotropic_yield()
    elastic = _orthotropic_elastic_matrix()
    curve = _LinearHardening(initial_flow=250.0e6, modulus=1.0e9)
    stress, _tangent, plastic, alpha = hill48_plane_stress_return_map(
        np.asarray([[0.0050, -0.0004, 0.0010]], dtype=float),
        np.zeros((1, 3), dtype=float),
        np.zeros(1, dtype=float),
        elastic,
        model,
        curve,
    )
    assert np.linalg.norm(stress) > 0.0
    assert alpha[0] > 0.0

    unloaded_stress, unloaded_tangent, unloaded_plastic, unloaded_alpha = (
        hill48_plane_stress_return_map(
            plastic.copy(),
            plastic,
            alpha,
            elastic,
            model,
            curve,
        )
    )

    np.testing.assert_allclose(unloaded_stress, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(unloaded_tangent[0], elastic)
    np.testing.assert_array_equal(unloaded_plastic, plastic)
    np.testing.assert_array_equal(unloaded_alpha, alpha)

    _reloaded_stress, _reloaded_tangent, _reloaded_plastic, reloaded_alpha = (
        hill48_plane_stress_return_map(
            np.asarray([[0.0090, -0.0010, 0.0020]], dtype=float),
            unloaded_plastic,
            unloaded_alpha,
            elastic,
            model,
            curve,
        )
    )
    assert reloaded_alpha[0] > unloaded_alpha[0]


def test_biaxial_large_increment_stays_on_convex_hill_surface() -> None:
    model = _orthotropic_yield()
    elastic = _orthotropic_elastic_matrix()
    strain = np.asarray([[0.05, -0.02, 0.03]], dtype=float)

    stress, tangent, plastic, alpha = hill48_plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(1, dtype=float),
        elastic,
        model,
    )

    np.testing.assert_allclose(
        hill48_plane_stress_equivalent_stress(stress, model),
        model.X,
        rtol=2.0e-9,
    )
    assert alpha[0] > 0.0
    assert np.linalg.norm(plastic[0]) > 0.0
    assert np.all(np.isfinite(tangent))
    assert np.allclose(tangent, np.swapaxes(tangent, 1, 2), atol=1.0e-5)


def test_isotropic_hill_limit_matches_existing_plane_stress_j2() -> None:
    elastic_modulus = 210.0e9
    poisson_ratio = 0.3
    curve = DNVC208MaterialCurve(
        sigma_prop=250.0e6,
        sigma_yield=280.0e6,
        sigma_yield_2=300.0e6,
        eps_p_y1=0.004,
        eps_p_y2=0.020,
        K=600.0e6,
        n=0.2,
    )
    initial_yield = float(curve.flow_stress(np.zeros(1))[0])
    shear_yield = initial_yield / np.sqrt(3.0)
    model = Hill48Yield(
        X=initial_yield,
        Y=initial_yield,
        Z=initial_yield,
        S12=shear_yield,
        S13=shear_yield,
        S23=shear_yield,
    )
    strain = np.asarray(
        [
            [0.0030, 0.0003, 0.0007],
            [0.0060, -0.0010, 0.0020],
        ],
        dtype=float,
    )
    plastic = np.zeros_like(strain)
    alpha = np.zeros(strain.shape[0], dtype=float)

    j2_result = plane_stress_return_map(
        strain,
        plastic,
        alpha,
        elastic_modulus,
        poisson_ratio,
        curve,
    )
    hill_result = hill48_plane_stress_return_map(
        strain,
        plastic,
        alpha,
        plane_stress_elastic_matrix(elastic_modulus, poisson_ratio),
        model,
        curve,
    )

    for j2_values, hill_values in zip(j2_result, hill_result):
        np.testing.assert_allclose(
            hill_values,
            j2_values,
            rtol=1.0e-9,
            atol=1.0e-8,
        )


def test_exact_algorithmic_tangent_matches_central_difference_oracle() -> None:
    model = _orthotropic_yield()
    elastic = _orthotropic_elastic_matrix()
    curve = _LinearHardening(initial_flow=250.0e6, modulus=1.5e9)
    strain = np.asarray(
        [
            [0.0040, -0.0005, 0.0010],
            [0.0065, 0.0004, -0.0017],
        ],
        dtype=float,
    )
    plastic = np.zeros_like(strain)
    alpha = np.zeros(strain.shape[0], dtype=float)

    _stress, analytical, _plastic, _alpha = (
        hill48_plane_stress_return_map(
            strain,
            plastic,
            alpha,
            elastic,
            model,
            curve,
        )
    )
    oracle = hill48_plane_stress_numerical_tangent(
        strain,
        plastic,
        alpha,
        elastic,
        model,
        curve,
        step=1.0e-8,
    )
    default_oracle = hill48_plane_stress_numerical_tangent(
        strain,
        plastic,
        alpha,
        elastic,
        model,
        curve,
    )
    _stress, requested_oracle, _plastic, _alpha = (
        hill48_plane_stress_return_map(
            strain,
            plastic,
            alpha,
            elastic,
            model,
            curve,
            tangent_method="numerical",
        )
    )

    np.testing.assert_allclose(analytical, oracle, rtol=2.0e-7)
    np.testing.assert_array_equal(requested_oracle, default_oracle)
    np.testing.assert_allclose(
        analytical,
        np.swapaxes(analytical, 1, 2),
        rtol=1.0e-13,
        atol=1.0e-5,
    )


def test_invalid_elastic_matrix_and_iteration_budget_fail_closed() -> None:
    model = _orthotropic_yield()
    valid = np.zeros((1, 3), dtype=float)
    with pytest.raises(ValueError, match="positive definite"):
        hill48_plane_stress_return_map(
            valid,
            valid,
            np.zeros(1),
            np.diag([1.0, 1.0, -1.0]),
            model,
        )
    with pytest.raises(ValueError, match="max_iterations"):
        hill48_plane_stress_return_map(
            valid,
            valid,
            np.zeros(1),
            _orthotropic_elastic_matrix(),
            model,
            max_iterations=0,
        )
    with pytest.raises(
        PlaneStressConvergenceError,
        match="did not satisfy",
    ):
        hill48_plane_stress_return_map(
            np.asarray([[0.05, -0.02, 0.03]], dtype=float),
            valid,
            np.zeros(1),
            _orthotropic_elastic_matrix(),
            model,
            _LinearHardening(initial_flow=250.0e6, modulus=3.0e9),
            max_iterations=1,
        )
