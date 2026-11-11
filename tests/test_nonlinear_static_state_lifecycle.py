from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

import anyfesolver.nonlinear_static as nonlinear_static
from anyfesolver import nonlinear_performance
from anyfesolver import nonlinear_performance_batch_c
from anyfesolver.control import SolveCancelled
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import FixedSupport
from anyfesolver.nonlinear_state import (
    NonlinearStateStore,
    begin_state_evaluation,
    finish_state_evaluation,
)


def _fallback_store(value: float = 1.0) -> NonlinearStateStore:
    return NonlinearStateStore.from_shell_layouts(
        (),
        {1: {"history": np.asarray([value], dtype=float)}},
    )


def _trial_view(
    store: NonlinearStateStore,
    value: float,
) -> Any:
    token = begin_state_evaluation(store)
    return finish_state_evaluation(
        store,
        token,
        {1: {"history": np.asarray([value], dtype=float)}},
    )


def _equilibrate(store: NonlinearStateStore, **kwargs: Any) -> Any:
    return nonlinear_static._equilibrate_initial_fields(
        model=FEModel("initial_equilibration_state_lifecycle"),
        T=sparse.eye(1, format="csr"),
        u0=np.zeros(1, dtype=float),
        committed_states=store,
        num_layers=1,
        max_iterations=1,
        tolerance=1.0e-10,
        kinematics="von_karman",
        corotational_tangent="rotated",
        general_tangent=False,
        **kwargs,
    )


def test_initial_field_equilibration_commits_the_converged_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _fallback_store()

    def assemble(
        _model: Any,
        _u: np.ndarray,
        committed_states: NonlinearStateStore,
        _num_layers: int,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, sparse.csr_matrix, Any]:
        return (
            np.zeros(1, dtype=float),
            sparse.eye(1, format="csr"),
            _trial_view(committed_states, 2.0),
        )

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", assemble)
    q, committed, history, failure = _equilibrate(store)

    assert committed is store
    assert failure is None
    assert len(history) == 1
    np.testing.assert_array_equal(q, np.zeros(1))
    np.testing.assert_array_equal(store[1]["history"], [2.0])
    assert store.generation == 1
    assert store.has_active_trial is False


def test_initial_field_equilibration_discards_a_failed_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _fallback_store()

    def assemble(
        _model: Any,
        _u: np.ndarray,
        committed_states: NonlinearStateStore,
        _num_layers: int,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, sparse.csr_matrix, Any]:
        return (
            np.full(1, np.nan),
            sparse.eye(1, format="csr"),
            _trial_view(committed_states, 9.0),
        )

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", assemble)
    _q, committed, _history, failure = _equilibrate(store)

    assert committed is store
    assert failure == "nonfinite_initial_state_residual"
    np.testing.assert_array_equal(store[1]["history"], [1.0])
    assert store.generation == 0
    assert store.has_active_trial is False


@pytest.mark.parametrize("error", [RuntimeError("element failed"), SolveCancelled("stop")])
def test_initial_field_equilibration_discards_on_exception_or_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    store = _fallback_store()

    def assemble(
        _model: Any,
        _u: np.ndarray,
        committed_states: NonlinearStateStore,
        _num_layers: int,
        **_kwargs: Any,
    ) -> Any:
        begin_state_evaluation(committed_states)
        raise error

    monkeypatch.setattr(nonlinear_static, "_assemble_nonlinear_system", assemble)
    with pytest.raises(type(error), match="element failed|Solve cancelled"):
        _equilibrate(store)

    np.testing.assert_array_equal(store[1]["history"], [1.0])
    assert store.generation == 0
    assert store.has_active_trial is False


class _LifecycleShell(ShellElement):
    legacy_nonlinear_batch_eligible = False

    def __init__(self, *, native: bool, fail: bool = False) -> None:
        super().__init__(1, [1, 2, 3], "steel", thickness=0.01)
        self.formulation_native_total_lagrangian = native
        self.fail = fail
        self.response_calls = 0

    def compute_stiffness_matrix(self, mesh: Any, material: Any) -> np.ndarray:
        del mesh, material
        return np.eye(self.total_dofs, dtype=float)

    def native_reference_directors(self, mesh: Any) -> np.ndarray:
        del mesh
        return np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=float)

    def compute_nonlinear_response(
        self,
        mesh: Any,
        material: Any,
        displacements: np.ndarray,
        committed_state: Any,
        num_layers: int,
        tangent: bool,
        *,
        native_rotation_trial: Any = None,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        del mesh, material, committed_state, num_layers
        self.response_calls += 1
        if self.fail:
            raise RuntimeError("native element evaluation failed")
        size = int(np.asarray(displacements).size)
        return (
            np.zeros(size, dtype=float),
            np.eye(size, dtype=float) if tangent else None,
            {
                "native_trial": True,
                "evaluation": self.response_calls,
                "native_rotation_context_received": native_rotation_trial is not None,
            },
        )


def _fully_constrained_model(element: ShellElement) -> FEModel:
    model = FEModel("native-state-lifecycle")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 0.2, 0.9, 0.0)
    model.add_element(1, element)
    model.add_boundary_condition(FixedSupport("fixed", [1, 2, 3]))
    return model


def test_virgin_qualified_s3_state_exists_before_native_rotation_activation() -> None:
    element = QualifiedE4PLS3ShellElement(
        1,
        [1, 2, 3],
        "steel",
        thickness=0.01,
        reference_normal=(0.0, 0.0, 1.0),
    )
    model = _fully_constrained_model(element)

    states, provenance = nonlinear_static._prepare_initial_states(
        model,
        initial_element_states=None,
        initial_fields=None,
        num_layers=3,
    )

    assert provenance == []
    assert set(states) == {1}
    assert states[1]["state_schema"] == "anysolver.e4_pl_s3.committed_state.v2"
    np.testing.assert_array_equal(states[1]["committed_total_u"], np.zeros(18))
    np.testing.assert_array_equal(
        states[1]["committed_nodal_rotation_matrices"],
        np.repeat(np.eye(3)[None, :, :], 3, axis=0),
    )


def test_fully_constrained_native_tl_element_is_evaluated_and_committed_without_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = _LifecycleShell(native=True)
    commit_calls: list[Any] = []
    original_commit = nonlinear_static._commit_nonlinear_state_candidate

    def recording_commit(committed: Any, candidate: Any, **kwargs: Any) -> Any:
        commit_calls.append(candidate)
        return original_commit(committed, candidate, **kwargs)

    monkeypatch.setattr(
        nonlinear_static,
        "_commit_nonlinear_state_candidate",
        recording_commit,
    )
    result = nonlinear_static.solve_static_nonlinear(
        _fully_constrained_model(element),
        num_steps=1,
        num_layers=1,
    )

    assert result.status == "empty_reduced_system"
    assert element.response_calls == 1
    assert len(commit_calls) == 1
    assert result.element_states[1]["native_trial"] is True
    assert result.element_states[1]["evaluation"] == 1
    assert result.element_states[1]["native_rotation_context_received"] is True
    assert result.info["initial_state_equilibration"]["native_tl_evaluated"] is True


def test_fully_constrained_legacy_element_without_fields_keeps_the_old_no_evaluation_path() -> None:
    element = _LifecycleShell(native=False)
    result = nonlinear_static.solve_static_nonlinear(
        _fully_constrained_model(element),
        num_steps=1,
        num_layers=1,
    )

    assert result.status == "empty_reduced_system"
    assert element.response_calls == 0
    assert result.element_states == {}


def _assert_assembly_exception_discards(assembler: Any) -> None:
    element = _LifecycleShell(native=False, fail=True)
    model = _fully_constrained_model(element)
    model.apply_boundary_conditions()
    store = _fallback_store()

    with pytest.raises(RuntimeError, match="native element evaluation failed"):
        assembler(
            model,
            np.zeros(model.mesh.dof_manager.total_dofs, dtype=float),
            store,
            1,
            tangent=False,
        )

    np.testing.assert_array_equal(store[1]["history"], [1.0])
    assert store.generation == 0
    assert store.has_active_trial is False


def test_reference_assembly_discards_the_state_token_when_an_element_raises() -> None:
    nonlinear_static._ensure_nonlinear_acceleration()
    reference = nonlinear_performance._ORIGINAL_ASSEMBLER
    assert reference is not None
    _assert_assembly_exception_discards(reference)


def test_installed_assembly_discards_the_state_token_when_an_element_raises() -> None:
    _assert_assembly_exception_discards(nonlinear_static._assemble_nonlinear_system)


def test_optimized_assembly_discards_when_post_assembly_recording_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fully_constrained_model(_LifecycleShell(native=False))
    model.apply_boundary_conditions()
    store = _fallback_store()

    def fail_recording(**_kwargs: Any) -> None:
        raise RuntimeError("assembly observer failed")

    monkeypatch.setattr(
        nonlinear_performance,
        "record_nonlinear_assembly_execution",
        fail_recording,
    )
    with pytest.raises(RuntimeError, match="assembly observer failed"):
        nonlinear_performance._optimized_assemble_nonlinear_system(
            model,
            np.zeros(model.mesh.dof_manager.total_dofs, dtype=float),
            store,
            1,
            tangent=False,
        )

    np.testing.assert_array_equal(store[1]["history"], [1.0])
    assert store.generation == 0
    assert store.has_active_trial is False


def test_direct_reduced_assembly_discards_when_post_assembly_recording_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fully_constrained_model(_LifecycleShell(native=False))
    model.apply_boundary_conditions()
    total_dofs = model.mesh.dof_manager.total_dofs
    store = _fallback_store()
    context = nonlinear_performance_batch_c._SolveContext(
        requested_model=model,
        bound_model=model,
        transformation=sparse.eye(total_dofs, format="csr"),
        activation_allowed=True,
        activation_reason="state-lifecycle-test",
    )
    stack = nonlinear_performance_batch_c._context_stack()
    stack.append(context)

    def fail_recording(**_kwargs: Any) -> None:
        raise RuntimeError("direct assembly observer failed")

    monkeypatch.setattr(
        nonlinear_performance_batch_c,
        "record_nonlinear_assembly_execution",
        fail_recording,
    )
    try:
        with pytest.raises(RuntimeError, match="direct assembly observer failed"):
            nonlinear_performance_batch_c._batch_c_assemble_nonlinear_system(
                model,
                np.zeros(total_dofs, dtype=float),
                store,
                1,
                tangent=False,
            )
    finally:
        assert stack.pop() is context

    np.testing.assert_array_equal(store[1]["history"], [1.0])
    assert store.generation == 0
    assert store.has_active_trial is False
