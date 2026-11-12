"""Parity, eligibility, and fallback qualification for compiled Hill-48."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

from anyfesolver.material_curves import (
    DNVC208MaterialCurve,
    LinearHardeningCurve,
    PiecewiseLinearCurve,
    PowerLawHardeningCurve,
)
from anyfesolver.materials import Hill48Yield
from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.elements import create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_state import ShellStateBatch, ShellStateLayout
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.plasticity import (
    PlaneStressConvergenceError,
    _hill48_plane_stress_metric,
    _hill48_reference_flow,
    _hill48_return_map_core,
    hill48_plane_stress_return_map,
)
from anyfesolver import vectorized_hill48
from anyfesolver.vectorized_hill48 import (
    hill48_vectorized_diagnostics,
    pack_hill48_curve,
    reset_hill48_vectorized_diagnostics,
)


def _elastic_matrix() -> np.ndarray:
    return np.asarray(
        [
            [160.0e9, 25.0e9, 4.0e9],
            [25.0e9, 95.0e9, 2.0e9],
            [4.0e9, 2.0e9, 38.0e9],
        ],
        dtype=float,
    )


def _yield_model() -> Hill48Yield:
    return Hill48Yield(
        X=300.0e6,
        Y=240.0e6,
        Z=270.0e6,
        S12=130.0e6,
        S13=140.0e6,
        S23=120.0e6,
    )


def _global_hill_shell() -> tuple[FEModel, LoadCase]:
    model = FEModel("compiled_hill48_global_qualification")
    strength = 100.0e6
    shear_strength = strength / math.sqrt(3.0)
    model.add_orthotropic_material(
        "lamina",
        density=1600.0,
        elastic_modulus_1=150.0e9,
        elastic_modulus_2=12.0e9,
        elastic_modulus_3=10.0e9,
        poisson_ratio_12=0.25,
        poisson_ratio_13=0.20,
        poisson_ratio_23=0.30,
        shear_modulus_12=5.0e9,
        shear_modulus_13=4.0e9,
        shear_modulus_23=3.8e9,
        hill_yield=Hill48Yield(
            strength,
            strength,
            strength,
            shear_strength,
            shear_strength,
            shear_strength,
        ),
        hardening_curve=DNVC208MaterialCurve(
            sigma_prop=100.0e6,
            sigma_yield=105.0e6,
            sigma_yield_2=110.0e6,
            eps_p_y1=0.005,
            eps_p_y2=0.010,
            K=400.0e6,
            n=0.20,
        ),
    )
    for node_id, coordinates in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    model.add_element(
        1,
        create_element(
            "shell",
            1,
            [1, 2, 3, 4],
            "lamina",
            thickness=0.02,
            formulation="legacy",  # Hill48: no stable qualified Q4 route
            material_direction=(1.0, 0.0, 1.0),
            material_angle_deg=90.0,
        ),
    )
    model.add_boundary_condition(
        BoundaryCondition("left_x", [1, 4], {"ux": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition("pin_y", [1], {"uy": 0.0})
    )
    model.add_boundary_condition(
        BoundaryCondition(
            "in_plane",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("hill_membrane")
    load.add_nodal_load(2, [1.05e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(3, [1.05e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _curves() -> tuple[object | None, ...]:
    return (
        None,
        LinearHardeningCurve(250.0e6, 2.0e9),
        PiecewiseLinearCurve(
            plastic_strain=(0.0, 0.002, 0.010),
            flow_stress_values=(250.0e6, 270.0e6, 310.0e6),
        ),
        PowerLawHardeningCurve.from_yield(250.0e6, 700.0e6, 0.2),
        DNVC208MaterialCurve(
            sigma_prop=250.0e6,
            sigma_yield=280.0e6,
            sigma_yield_2=300.0e6,
            eps_p_y1=0.004,
            eps_p_y2=0.020,
            K=600.0e6,
            n=0.2,
        ),
    )


def _scalar_oracle(
    strain: np.ndarray,
    plastic: np.ndarray,
    alpha: np.ndarray,
    curve: object | None,
    *,
    compute_tangent: bool = True,
):
    elastic = _elastic_matrix()
    metric, strength = _hill48_plane_stress_metric(_yield_model())
    return _hill48_return_map_core(
        strain,
        plastic,
        alpha,
        elastic,
        metric,
        strength,
        curve,
        _hill48_reference_flow(curve),
        50,
        1.0e-10,
        compute_tangent,
    )


@pytest.mark.parametrize("curve", _curves())
def test_canonical_curves_match_scalar_oracle_for_mixed_batch(curve) -> None:
    rng = np.random.default_rng(20260811)
    strain = rng.normal(size=(192, 3)) * 0.007
    strain[:16] *= 1.0e-3
    plastic = rng.normal(size=(192, 3)) * 8.0e-5
    alpha = rng.random(192) * 0.015

    actual = hill48_plane_stress_return_map(
        strain,
        plastic,
        alpha,
        _elastic_matrix(),
        _yield_model(),
        curve,
    )
    expected = _scalar_oracle(strain, plastic, alpha, curve)

    for actual_values, expected_values in zip(actual, expected[:4]):
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=2.0e-12,
            atol=1.0e-4,
        )


def test_piecewise_corners_strong_hardening_and_large_batch_are_exact() -> None:
    curve = PiecewiseLinearCurve(
        plastic_strain=(0.0, 0.001, 0.004, 0.020),
        flow_stress_values=(220.0e6, 300.0e6, 420.0e6, 500.0e6),
    )
    row_count = 4096
    phase = np.linspace(0.0, 8.0 * np.pi, row_count)
    strain = np.column_stack(
        (
            0.008 * np.sin(phase),
            -0.004 * np.cos(0.5 * phase),
            0.006 * np.sin(0.25 * phase + 0.2),
        )
    )
    plastic = np.zeros_like(strain)
    alpha = np.resize(np.asarray([0.0, 0.001, 0.004, 0.020]), row_count)

    actual = hill48_plane_stress_return_map(
        strain,
        plastic,
        alpha,
        _elastic_matrix(),
        _yield_model(),
        curve,
    )
    expected = _scalar_oracle(strain, plastic, alpha, curve)

    for actual_values, expected_values in zip(actual, expected[:4]):
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=4.0e-13,
            atol=3.0e-5,
        )


def test_persistent_shell_state_views_feed_compiled_kernel_without_materializing() -> None:
    layout = ShellStateLayout.from_dimensions((11, 12), n_gp=4, num_layers=3)
    batch = ShellStateBatch(layout)
    committed = batch.committed_arrays()
    strain = np.linspace(
        -0.006,
        0.009,
        layout.state_point_count * 3,
        dtype=float,
    ).reshape(layout.state_point_count, 3)
    curve = LinearHardeningCurve(250.0e6, 1.5e9)

    stress, tangent, plastic, alpha = hill48_plane_stress_return_map(
        strain,
        committed.plastic_strain.reshape(-1, 3),
        committed.alpha.reshape(-1),
        _elastic_matrix(),
        _yield_model(),
        curve,
    )

    assert stress.shape == strain.shape
    assert tangent.shape == (layout.state_point_count, 3, 3)
    token = batch.begin_trial()
    batch.update_trial(
        token,
        plastic_strain=plastic.reshape(layout.n_elements, -1, 3),
        alpha=alpha.reshape(layout.n_elements, -1),
        layer_strain=strain.reshape(layout.n_elements, -1, 3),
    )
    batch.commit(token)
    np.testing.assert_array_equal(
        batch.committed_arrays().alpha.reshape(-1),
        alpha,
    )
    assert batch.diagnostics()["state_materialization_count"] == 0


def test_global_newton_solution_and_iteration_count_match_scalar_path(monkeypatch) -> None:
    compiled_model, compiled_load = _global_hill_shell()
    compiled = solve_static_nonlinear(
        compiled_model,
        compiled_load,
        num_steps=4,
        max_iterations=25,
        tolerance=1.0e-8,
    )
    hill_diagnostics = compiled.info["nonlinear_performance"]["hill48"]
    assert hill_diagnostics["activated"] is True
    assert hill_diagnostics["compiled_call_count"] > 0
    assert hill_diagnostics["compiled_point_count"] > 0
    monkeypatch.setattr(vectorized_hill48, "JIT_ENABLED", False)
    scalar_model, scalar_load = _global_hill_shell()
    scalar = solve_static_nonlinear(
        scalar_model,
        scalar_load,
        num_steps=4,
        max_iterations=25,
        tolerance=1.0e-8,
    )

    assert compiled.status == scalar.status == "completed"
    assert (
        compiled.info["total_newton_iterations"]
        == scalar.info["total_newton_iterations"]
    )
    np.testing.assert_allclose(
        compiled.displacements,
        scalar.displacements,
        rtol=2.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        compiled.element_states[1]["alpha"],
        scalar.element_states[1]["alpha"],
        rtol=2.0e-12,
        atol=1.0e-14,
    )


def test_global_arc_length_path_matches_scalar_path(monkeypatch) -> None:
    control = ArcLengthControl(
        initial_load_increment=0.20,
        minimum_load_increment=0.01,
        maximum_load_increment=0.20,
        max_steps=20,
        stop_after_peak_steps=10,
        maximum_absolute_load_factor=1.10,
    )
    compiled_model, compiled_load = _global_hill_shell()
    compiled = solve_static_arc_length(
        compiled_model,
        compiled_load,
        control=control,
        max_iterations=25,
        tolerance=1.0e-8,
        arc_tolerance=1.0e-8,
    )
    hill_diagnostics = compiled.info["nonlinear_performance"]["hill48"]
    assert hill_diagnostics["activated"] is True
    assert hill_diagnostics["compiled_call_count"] > 0
    assert hill_diagnostics["compiled_point_count"] > 0
    monkeypatch.setattr(vectorized_hill48, "JIT_ENABLED", False)
    scalar_model, scalar_load = _global_hill_shell()
    scalar = solve_static_arc_length(
        scalar_model,
        scalar_load,
        control=control,
        max_iterations=25,
        tolerance=1.0e-8,
        arc_tolerance=1.0e-8,
    )

    assert compiled.status == scalar.status == "load_factor_limit_reached"
    assert len(compiled.steps) == len(scalar.steps)
    assert [step.iterations for step in compiled.steps] == [
        step.iterations for step in scalar.steps
    ]
    np.testing.assert_allclose(
        compiled.displacements,
        scalar.displacements,
        rtol=3.0e-11,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        [step.load_factor for step in compiled.steps],
        [step.load_factor for step in scalar.steps],
        rtol=3.0e-11,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        compiled.element_states[1]["alpha"],
        scalar.element_states[1]["alpha"],
        rtol=3.0e-11,
        atol=2.0e-13,
    )


@dataclass(frozen=True)
class _CustomCurve:
    initial: float = 250.0e6
    modulus: float = 1.0e9

    def flow_stress(self, alpha):
        values = np.asarray(alpha, dtype=float)
        return self.initial + self.modulus * values

    def hardening_modulus(self, alpha):
        return np.full_like(np.asarray(alpha, dtype=float), self.modulus)


class _OverriddenLinearCurve(LinearHardeningCurve):
    def flow_stress(self, alpha):
        return super().flow_stress(alpha) + 1.0


@pytest.mark.parametrize("curve", (_CustomCurve(), _OverriddenLinearCurve(250.0e6, 1.0e9)))
def test_arbitrary_and_overridden_curve_protocols_use_observable_scalar_fallback(curve) -> None:
    reset_hill48_vectorized_diagnostics()
    packed, reason = pack_hill48_curve(curve)
    assert packed is None
    assert reason == "custom_curve_protocol"
    strain = np.asarray([[0.006, -0.001, 0.002]], dtype=float)

    hill48_plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(1),
        _elastic_matrix(),
        _yield_model(),
        curve,
    )

    diagnostics = hill48_vectorized_diagnostics()
    assert diagnostics["scalar_fallback_point_count"] == 1
    assert diagnostics["fallback_reason_counts"] == {"custom_curve_protocol": 1}
    assert diagnostics["last_call"]["path"] == "scalar_reference"


def test_unavailable_jit_uses_observable_scalar_fallback(monkeypatch) -> None:
    reset_hill48_vectorized_diagnostics()
    monkeypatch.setattr(vectorized_hill48, "JIT_ENABLED", False)
    strain = np.asarray([[0.006, -0.001, 0.002]], dtype=float)

    actual = hill48_plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(1),
        _elastic_matrix(),
        _yield_model(),
        LinearHardeningCurve(250.0e6, 1.0e9),
    )
    expected = _scalar_oracle(
        strain,
        np.zeros_like(strain),
        np.zeros(1),
        LinearHardeningCurve(250.0e6, 1.0e9),
    )

    for actual_values, expected_values in zip(actual, expected[:4]):
        np.testing.assert_array_equal(actual_values, expected_values)
    assert hill48_vectorized_diagnostics()["fallback_reason_counts"] == {
        "jit_unavailable": 1
    }


def test_compiled_nonconvergence_falls_back_then_preserves_fail_closed_error() -> None:
    reset_hill48_vectorized_diagnostics()
    with pytest.raises(PlaneStressConvergenceError, match="did not satisfy"):
        hill48_plane_stress_return_map(
            np.asarray([[0.05, -0.02, 0.03]], dtype=float),
            np.zeros((1, 3)),
            np.zeros(1),
            _elastic_matrix(),
            _yield_model(),
            LinearHardeningCurve(250.0e6, 3.0e9),
            max_iterations=1,
        )

    diagnostics = hill48_vectorized_diagnostics()
    assert diagnostics["fallback_reason_counts"] == {
        "compiled_nonconverged": 1
    }
    assert diagnostics["scalar_fallback_point_count"] == 1


def test_kernel_exception_is_an_observable_whole_batch_scalar_fallback(monkeypatch) -> None:
    reset_hill48_vectorized_diagnostics()
    original = vectorized_hill48.compiled_hill48_return_map

    def fail_kernel(*args, **kwargs):
        raise RuntimeError("synthetic compiled-kernel failure")

    monkeypatch.setattr(vectorized_hill48, "compiled_hill48_return_map", fail_kernel)
    strain = np.asarray(
        [[0.006, -0.001, 0.002], [0.001, 0.0002, 0.0001]],
        dtype=float,
    )
    curve = LinearHardeningCurve(250.0e6, 1.0e9)
    actual = hill48_plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(2),
        _elastic_matrix(),
        _yield_model(),
        curve,
    )
    monkeypatch.setattr(vectorized_hill48, "compiled_hill48_return_map", original)
    expected = _scalar_oracle(strain, np.zeros_like(strain), np.zeros(2), curve)

    for actual_values, expected_values in zip(actual, expected[:4]):
        np.testing.assert_array_equal(actual_values, expected_values)
    diagnostics = hill48_vectorized_diagnostics()
    assert diagnostics["fallback_reason_counts"] == {
        "compiled_kernel_exception": 2
    }
    assert diagnostics["scalar_fallback_point_count"] == 2


def test_invalid_analytical_row_and_requested_numerical_tangent_are_counted(monkeypatch) -> None:
    reset_hill48_vectorized_diagnostics()
    original = vectorized_hill48.compiled_hill48_return_map
    injected = False

    def invalidate_one_tangent(*args, **kwargs):
        nonlocal injected
        result = original(*args, **kwargs)
        if bool(args[-1]) and not injected:
            injected = True
            result[-1][0] = False
            result[1][0] = np.nan
        return result

    monkeypatch.setattr(
        vectorized_hill48,
        "compiled_hill48_return_map",
        invalidate_one_tangent,
    )
    strain = np.asarray(
        [[0.006, -0.001, 0.002], [0.008, 0.0003, -0.001]],
        dtype=float,
    )
    curve = LinearHardeningCurve(250.0e6, 1.0e9)
    _stress, tangent, _plastic, _alpha = hill48_plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(2),
        _elastic_matrix(),
        _yield_model(),
        curve,
    )
    assert np.all(np.isfinite(tangent))
    diagnostics = hill48_vectorized_diagnostics()
    assert diagnostics["fallback_reason_counts"]["analytical_tangent_invalid"] == 1

    reset_hill48_vectorized_diagnostics()
    hill48_plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(2),
        _elastic_matrix(),
        _yield_model(),
        curve,
        tangent_method="numerical",
    )
    assert hill48_vectorized_diagnostics()["numerical_tangent_row_count"] == 2
