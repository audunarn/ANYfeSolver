from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import nonlinear_performance, nonlinear_static
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.elements import Element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import dnv_c208_steel_curve
from anyfesolver.nonlinear_static import (
    NonlinearLoadProgram,
    NonlinearLoadStage,
    NonlinearConvergenceSettings,
    _armijo_characteristic_length,
    _armijo_residual_weights,
    solve_static_nonlinear,
)
from anyfesolver.plasticity import PlaneStressConvergenceError, plane_stress_return_map


class _HardeningSpring(Element):
    """Two-node axial spring with an exact cubic force and tangent."""

    def __init__(
        self,
        element_id: int,
        node_ids,
        *,
        stiffness: float = 1.0,
        cubic: float = 10.0,
    ) -> None:
        super().__init__(element_id, node_ids, "default")
        self.stiffness = float(stiffness)
        self.cubic = float(cubic)

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray(
            [mesh.get_node(node_id).coords() for node_id in self.node_ids],
            dtype=float,
        )

    def _matrix(self, tangent_value: float) -> np.ndarray:
        matrix = np.zeros((12, 12), dtype=float)
        matrix[0, 0] = tangent_value
        matrix[0, 6] = -tangent_value
        matrix[6, 0] = -tangent_value
        matrix[6, 6] = tangent_value
        return matrix

    def compute_stiffness_matrix(self, mesh, material):
        return self._matrix(self.stiffness)

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        displacement = np.asarray(u_elem, dtype=float)
        extension = float(displacement[6] - displacement[0])
        force_value = self.stiffness * extension + self.cubic * extension**3
        force = np.zeros(12, dtype=float)
        force[0] = -force_value
        force[6] = force_value
        tangent_value = self.stiffness + 3.0 * self.cubic * extension**2
        stiffness = self._matrix(tangent_value) if tangent else None
        return force, stiffness, {"spring_extension": extension}


class _IndefiniteTwoDof(Element):
    """Linear two-DOF element with one positive and one negative eigenvalue."""

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray(
            [mesh.get_node(self.node_ids[0]).coords()],
            dtype=float,
        )

    def compute_stiffness_matrix(self, mesh, material):
        return np.diag([1.0, -2.0, 1.0, 1.0, 1.0, 1.0])

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        matrix = self.compute_stiffness_matrix(mesh, material)
        force = matrix @ np.asarray(u_elem, dtype=float)
        return force, matrix if tangent else None, {}


class _PlaneStressMaterialPoint(Element):
    """Axial material point using the production J2 plane-stress update."""

    def __init__(self, element_id: int, node_ids) -> None:
        super().__init__(element_id, node_ids, "steel")
        self.modulus = 210.0e9
        self.poisson = 0.3
        self.area = 1.0e-4
        self.length = 1.0
        self.curve = dnv_c208_steel_curve("S355", 0.01)

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray(
            [mesh.get_node(node_id).coords() for node_id in self.node_ids],
            dtype=float,
        )

    @staticmethod
    def _matrix(tangent_value: float) -> np.ndarray:
        matrix = np.zeros((12, 12), dtype=float)
        matrix[0, 0] = tangent_value
        matrix[0, 6] = -tangent_value
        matrix[6, 0] = -tangent_value
        matrix[6, 6] = tangent_value
        return matrix

    def compute_stiffness_matrix(self, mesh, material):
        return self._matrix(self.modulus * self.area / self.length)

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        displacement = np.asarray(u_elem, dtype=float)
        strain = np.asarray(
            [[(displacement[6] - displacement[0]) / self.length, 0.0, 0.0]],
            dtype=float,
        )
        parent = {} if state is None else state
        plastic_strain = np.asarray(
            parent.get("plastic_strain", np.zeros((1, 3))),
            dtype=float,
        )
        alpha = np.asarray(parent.get("alpha", np.zeros(1)), dtype=float)
        stress, algorithmic, new_plastic, new_alpha = plane_stress_return_map(
            strain,
            plastic_strain,
            alpha,
            self.modulus,
            self.poisson,
            self.curve,
            compute_tangent=tangent,
        )
        force_value = float(stress[0, 0]) * self.area
        force = np.zeros(12, dtype=float)
        force[0] = -force_value
        force[6] = force_value
        stiffness = (
            self._matrix(float(algorithmic[0, 0, 0]) * self.area / self.length)
            if tangent
            else None
        )
        return force, stiffness, {
            "plastic_strain": new_plastic,
            "alpha": new_alpha,
            "total_strain": strain,
        }


def _spring_model(*, length: float = 2.0) -> tuple[FEModel, LoadCase]:
    model = FEModel("armijo-hardening-spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, length, 0.0, 0.0)
    model.add_element(1, _HardeningSpring(1, [1, 2]))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "guide",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("pull")
    load.add_nodal_load(2, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _armijo_settings(**overrides) -> dict[str, object]:
    settings: dict[str, object] = {
        "profile": "legacy",
        "line_search": "armijo",
        "max_line_search_cuts": 16,
        "line_search_reduction": 0.5,
    }
    settings.update(overrides)
    return settings


def _plastic_reversal_model() -> tuple[FEModel, NonlinearLoadProgram]:
    model = FEModel("armijo-plane-stress-material-point")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, _PlaneStressMaterialPoint(1, [1, 2]))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "guide",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    forward = LoadCase("plastic-forward")
    forward.add_nodal_load(2, [60_000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reverse = LoadCase("plastic-reverse")
    reverse.add_nodal_load(2, [-90_000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, NonlinearLoadProgram(
        (
            NonlinearLoadStage("forward", forward),
            NonlinearLoadStage("reverse", reverse),
        )
    )


def test_armijo_backtracks_and_reports_work() -> None:
    model, load = _spring_model()
    result = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=20,
        tolerance=1.0e-12,
        convergence_settings=_armijo_settings(),
    )

    assert result.status == "completed"
    extension = float(result.displacements[model.mesh.get_node(2).dofs[0]])
    assert extension + 10.0 * extension**3 == pytest.approx(2.0, abs=1.0e-11)
    assert result.info["convergence_settings"]["line_search"] == "armijo"
    assert result.info["convergence_settings"]["characteristic_length"] == pytest.approx(1.0)
    assert result.info["armijo"]["enabled"] is True
    solver = result.info["nonlinear_performance"]["solver"]
    assert solver["rejected_full_steps"] >= 1
    assert solver["backtracks"] >= 1
    assert solver["linear_factorizations"] == solver["linear_solves"]
    assert solver["event_counts"]["dead_load_projection_reuse"] > 0


def test_promoted_trial_must_pass_acceptance_after_tangent_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    assert nonlinear_performance._ORIGINAL_ASSEMBLER is not None
    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    model, load = _spring_model()
    residual_only_displacements: list[np.ndarray] = []
    promotion_corrupted = False

    def observed(*args, **kwargs):
        nonlocal promotion_corrupted
        force, tangent, states = original(*args, **kwargs)
        displacement = np.asarray(args[1], dtype=float).copy()
        with_tangent = bool(kwargs.get("tangent", True))
        if not with_tangent:
            residual_only_displacements.append(displacement)
        elif not promotion_corrupted and any(
            np.array_equal(displacement, prior)
            for prior in residual_only_displacements
        ):
            force = np.asarray(force, dtype=float).copy()
            force[model.mesh.get_node(2).dofs[0]] += 100.0
            promotion_corrupted = True
        return force, tangent, states

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", observed)
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=25,
        tolerance=1.0e-11,
        convergence_settings={
            "profile": "legacy",
            "line_search": "always",
            "line_search_reduction": 0.5,
        },
    )

    assert result.status == "completed"
    assert promotion_corrupted is True
    events = result.info["nonlinear_performance"]["solver"]["event_counts"]
    assert events["promotion_evaluation"] >= 2
    assert events["promotion_rejected"] == 1


def test_armijo_retries_typed_local_trial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    assert nonlinear_performance._ORIGINAL_ASSEMBLER is not None
    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    model, load = _spring_model()
    tangent_calls = 0

    def observed(*args, **kwargs):
        nonlocal tangent_calls
        if bool(kwargs.get("tangent", True)):
            tangent_calls += 1
            if tangent_calls == 2:
                raise PlaneStressConvergenceError("bounded Armijo trial failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", observed)
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=25,
        tolerance=1.0e-11,
        convergence_settings=_armijo_settings(),
    )

    assert result.status == "completed"
    solver = result.info["nonlinear_performance"]["solver"]
    assert solver["recoverable_trial_failures"] == 1
    assert any(
        reason.startswith("PlaneStressConvergenceError:")
        for reason in solver["failure_reason_counts"]
    )


def test_armijo_backtracks_nonfinite_trial_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    assert nonlinear_performance._ORIGINAL_ASSEMBLER is not None
    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    model, load = _spring_model()
    tangent_calls = 0

    def observed(*args, **kwargs):
        nonlocal tangent_calls
        force, tangent, states = original(*args, **kwargs)
        if bool(kwargs.get("tangent", True)):
            tangent_calls += 1
            if tangent_calls == 2:
                force = np.asarray(force, dtype=float).copy()
                force[model.mesh.get_node(2).dofs[0]] = np.nan
        return force, tangent, states

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", observed)
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=25,
        tolerance=1.0e-11,
        convergence_settings=_armijo_settings(),
    )

    assert result.status == "completed"
    solver = result.info["nonlinear_performance"]["solver"]
    assert solver["recoverable_trial_failures"] == 1
    assert solver["failure_reason_counts"]["nonfinite_trial_residual"] == 1
    assert solver["backtracks"] >= 1


def test_armijo_rejects_non_descent_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, load = _spring_model()
    real_factorize = nonlinear_static.factorize

    class _WrongDirectionHandle:
        status = "ok"
        backend_name = "test-wrong-direction"
        failure_reason = None

        @staticmethod
        def solve(residual):
            return -np.asarray(residual, dtype=float)

    monkeypatch.setattr(
        nonlinear_static,
        "factorize",
        lambda *args, **kwargs: _WrongDirectionHandle(),
    )
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=3,
        min_step_fraction=0.75,
        convergence_settings=_armijo_settings(),
    )
    monkeypatch.setattr(nonlinear_static, "factorize", real_factorize)

    assert result.status == "diverged"
    assert result.failure_reason == "minimum_load_increment_reached"
    solver = result.info["nonlinear_performance"]["solver"]
    assert solver["event_counts"]["armijo_non_descent"] == 1
    assert solver["failure_reason_counts"]["armijo_non_descent_direction"] >= 1


def test_armijo_trial_budget_exhaustion_returns_to_cutback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    assert nonlinear_performance._ORIGINAL_ASSEMBLER is not None
    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    model, load = _spring_model()

    def observed(*args, **kwargs):
        force, tangent, states = original(*args, **kwargs)
        displacement = np.asarray(args[1], dtype=float)
        if np.any(displacement):
            force = np.zeros_like(force)
        return force, tangent, states

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", observed)
    result = nonlinear_static.solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=3,
        min_step_fraction=0.75,
        convergence_settings=_armijo_settings(max_line_search_cuts=2),
    )

    assert result.status == "diverged"
    assert result.failure_reason == "minimum_load_increment_reached"
    solver = result.info["nonlinear_performance"]["solver"]
    assert solver["event_counts"]["line_search_failure"] == 1
    assert solver["backtracks"] == 2
    assert solver["failed_work"] == {
        "increment_count": 1,
        "newton_iterations": 1,
        "rejected_trial_evaluations": 2,
        "recoverable_trial_failures": 0,
    }


def test_armijo_scaling_distinguishes_force_and_moment_rows() -> None:
    model, _load = _spring_model(length=4.0)
    node = model.mesh.get_node(2)
    length = _armijo_characteristic_length(model, None)
    weights = _armijo_residual_weights(
        model,
        np.asarray([node.dofs[0], node.dofs[3]], dtype=np.intp),
        length,
    )

    assert length == pytest.approx(2.0)
    np.testing.assert_array_equal(weights, np.asarray([1.0, 0.5]))


def test_armijo_accepts_symmetric_indefinite_tangent() -> None:
    model = FEModel("armijo-indefinite")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_element(1, _IndefiniteTwoDof(1, [1], "default"))
    model.add_boundary_condition(
        BoundaryCondition(
            "retain-two-translations",
            [1],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("indefinite-load")
    load.add_nodal_load(1, [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=5,
        tolerance=1.0e-12,
        convergence_settings=_armijo_settings(characteristic_length=1.0),
    )

    assert result.status == "completed"
    node = model.mesh.get_node(1)
    np.testing.assert_allclose(
        result.displacements[[node.dofs[0], node.dofs[1]]],
        np.asarray([1.0, -1.0]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_armijo_corotational_auto_selects_consistent_and_rejects_rotated() -> None:
    model, load = _spring_model()
    result = solve_static_nonlinear(
        model,
        load,
        max_load_factor=0.05,
        num_steps=1,
        max_iterations=12,
        tolerance=1.0e-10,
        kinematics="corotational",
        corotational_tangent="auto",
        convergence_settings=_armijo_settings(),
    )
    assert result.status == "completed"
    assert result.info["corotational_tangent_requested"] == "auto"
    assert result.info["corotational_tangent"] == "consistent"

    rejected_model, rejected_load = _spring_model()
    with pytest.raises(ValueError, match="requires.*auto.*consistent"):
        solve_static_nonlinear(
            rejected_model,
            rejected_load,
            kinematics="corotational",
            corotational_tangent="rotated",
            convergence_settings=_armijo_settings(),
        )


def test_armijo_settings_reject_invalid_length_and_displacement_control() -> None:
    with pytest.raises(ValueError, match="characteristic_length"):
        NonlinearConvergenceSettings(
            line_search="armijo",
            characteristic_length=0.0,
        )

    model, load = _spring_model()
    with pytest.raises(ValueError, match="requires force control"):
        solve_static_nonlinear(
            model,
            load,
            control="displacement",
            convergence_settings=_armijo_settings(),
        )


def test_armijo_translation_only_zero_span_does_not_require_length() -> None:
    model, load = _spring_model(length=0.0)

    result = solve_static_nonlinear(
        model,
        load,
        num_steps=1,
        max_iterations=20,
        tolerance=1.0e-12,
        convergence_settings=_armijo_settings(),
    )

    assert result.status == "completed"
    assert result.info["convergence_settings"]["characteristic_length"] == 1.0


def test_armijo_rejected_trials_preserve_plane_stress_plastic_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_factorize = nonlinear_static.factorize

    class _ScaledHandle:
        def __init__(self, handle) -> None:
            self._handle = handle
            self.status = handle.status
            self.backend_name = handle.backend_name
            self.failure_reason = handle.failure_reason

        def solve(self, residual):
            return 4.0 * np.asarray(self._handle.solve(residual), dtype=float)

    def scaled_factorize(*args, **kwargs):
        return _ScaledHandle(real_factorize(*args, **kwargs))

    monkeypatch.setattr(nonlinear_static, "factorize", scaled_factorize)
    armijo_model, armijo_program = _plastic_reversal_model()
    armijo = solve_static_nonlinear(
        armijo_model,
        load_program=armijo_program,
        num_steps=2,
        max_iterations=30,
        tolerance=1.0e-10,
        convergence_settings=_armijo_settings(),
    )
    monkeypatch.setattr(nonlinear_static, "factorize", real_factorize)
    oracle_model, oracle_program = _plastic_reversal_model()
    oracle = solve_static_nonlinear(
        oracle_model,
        load_program=oracle_program,
        num_steps=2,
        max_iterations=30,
        tolerance=1.0e-10,
        convergence_settings=_armijo_settings(),
    )
    reference_model, reference_program = _plastic_reversal_model()
    reference = solve_static_nonlinear(
        reference_model,
        load_program=reference_program,
        num_steps=80,
        max_iterations=30,
        tolerance=1.0e-10,
        convergence_settings={"profile": "legacy", "line_search": "never"},
    )

    assert armijo.status == oracle.status == reference.status == "completed"
    events = armijo.info["nonlinear_performance"]["solver"]
    assert events["rejected_full_steps"] >= 1
    assert events["promotion_evaluations"] >= 1
    np.testing.assert_allclose(
        armijo.displacements,
        oracle.displacements,
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        armijo.element_states[1]["plastic_strain"],
        oracle.element_states[1]["plastic_strain"],
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        armijo.element_states[1]["alpha"],
        oracle.element_states[1]["alpha"],
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        armijo.displacements,
        reference.displacements,
        rtol=5.0e-4,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        armijo.element_states[1]["plastic_strain"],
        reference.element_states[1]["plastic_strain"],
        rtol=2.0e-2,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        armijo.element_states[1]["alpha"],
        reference.element_states[1]["alpha"],
        rtol=2.0e-2,
        atol=1.0e-10,
    )
    assert float(np.max(armijo.element_states[1]["alpha"])) > 0.0
