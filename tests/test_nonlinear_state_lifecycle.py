"""Production lifecycle qualification for solver-owned constitutive state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.nonlinear_static import (
    DisplacementControl,
    NonlinearLoadProgram,
    NonlinearLoadStage,
    solve_static_nonlinear,
)


E = 210.0e9
NU = 0.3
WIDTH = 0.2
THICKNESS = 0.01

PLASTIC_CURVE = DNVC208MaterialCurve(
    sigma_prop=320.0e6,
    sigma_yield=357.0e6,
    sigma_yield_2=363.3e6,
    eps_p_y1=0.004,
    eps_p_y2=0.015,
    K=740.0e6,
    n=0.166,
)


@pytest.fixture(scope="module", autouse=True)
def _install_accelerated_assembly_before_ab_comparisons() -> None:
    # Keep the persistent/mapping comparisons on the same already-installed
    # Batch-C dispatch; installation is otherwise lazy on the first solve.
    from anyfesolver.nonlinear_static import _ensure_nonlinear_acceleration

    _ensure_nonlinear_acceleration()


def _plastic_membrane_patch(
    *,
    prescribed_right_x: float | None = None,
) -> FEModel:
    model = FEModel("persistent_state_membrane_patch")
    model.add_material("steel", E, NU, hardening_curve=PLASTIC_CURVE)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, WIDTH, 0.0)
    model.add_node(4, 0.0, WIDTH, 0.0)
    model.add_element(
        1,
        create_shell_element(1, [1, 2, 3, 4], "steel", thickness=THICKNESS),
    )
    model.add_boundary_condition(
        BoundaryCondition("left_x", [1, 4], {"ux": 0.0})
    )
    model.add_boundary_condition(BoundaryCondition("origin_y", [1], {"uy": 0.0}))
    model.add_boundary_condition(
        BoundaryCondition(
            "plane",
            [1, 2, 3, 4],
            {"uz": 0.0, "rx": 0.0, "ry": 0.0},
        )
    )
    if prescribed_right_x is not None:
        model.add_boundary_condition(
            BoundaryCondition(
                "prescribed_right_x",
                [2, 3],
                {"ux": float(prescribed_right_x)},
            )
        )
    return model


def _membrane_load(name: str, stress: float) -> LoadCase:
    load = LoadCase(name)
    force = float(stress) * WIDTH * THICKNESS
    for node_id in (2, 3):
        load.add_nodal_load(
            node_id,
            load_vector=[0.5 * force, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    return load


def _assert_owned_state_equal(left: Mapping[int, Any], right: Mapping[int, Any]) -> None:
    assert set(left) == set(right)
    for element_id in left:
        left_state = left[element_id]
        right_state = right[element_id]
        assert set(left_state) == set(right_state)
        for key, left_value in left_state.items():
            right_value = right_state[key]
            if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
                np.testing.assert_array_equal(left_value, right_value)
            else:
                assert left_value == right_value


def _assert_solution_equal(left: Any, right: Any) -> None:
    assert left.status == right.status
    assert left.load_factor == right.load_factor
    np.testing.assert_array_equal(left.displacements, right.displacements)
    _assert_owned_state_equal(left.element_states, right.element_states)
    assert [step.to_dict() for step in left.steps] == [
        step.to_dict() for step in right.steps
    ]


def test_force_control_store_matches_mapping_and_materializes_owned_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = _membrane_load("pull", 340.0e6)
    persistent = solve_static_nonlinear(
        _plastic_membrane_patch(),
        load,
        num_steps=3,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
        record_increment_snapshots=True,
    )

    monkeypatch.setenv("FE_SOLVER_DISABLE_PERSISTENT_STATE", "1")
    legacy = solve_static_nonlinear(
        _plastic_membrane_patch(),
        load,
        num_steps=3,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
        record_increment_snapshots=True,
    )

    _assert_solution_equal(persistent, legacy)
    storage = persistent.info["nonlinear_state_storage"]
    assert storage["activated"] is True
    assert storage["eligible_batch_count"] == 1
    assert storage["generation"] == len(persistent.steps)
    assert storage["stale_token_error_count"] == 0
    assert storage["shell_batches"][0]["state_discard_count"] > 0
    assert storage["shell_batches"][0]["materialization_reasons"] == {
        "saved_state": len(persistent.snapshots),
        "final_result": 1,
    }
    assert legacy.info["nonlinear_state_storage"] == {
        "activated": False,
        "eligible_batch_count": 0,
        "fallback_reason": "persistent_state_storage_disabled",
    }

    assert len(persistent.snapshots) == len(persistent.steps)
    first = persistent.snapshots[0].element_states[1]["alpha"]
    final = persistent.element_states[1]["alpha"]
    assert not np.shares_memory(first, final)
    if len(persistent.snapshots) > 1:
        second = persistent.snapshots[1].element_states[1]["alpha"]
        assert not np.shares_memory(first, second)
        second_before = second.copy()
        first[...] = -123.0
        np.testing.assert_array_equal(second, second_before)
        assert np.all(final >= 0.0)


def test_block_displacement_control_store_matches_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = _membrane_load("reference", 340.0e6)
    control = DisplacementControl(
        weighted_dofs={(2, "ux"): 0.5, (3, "ux"): 0.5},
        target_displacement=0.003,
    )
    persistent = solve_static_nonlinear(
        _plastic_membrane_patch(),
        load,
        control="displacement",
        displacement_control=control,
        num_steps=3,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
        record_increment_snapshots=True,
    )

    monkeypatch.setenv("FE_SOLVER_DISABLE_PERSISTENT_STATE", "true")
    legacy = solve_static_nonlinear(
        _plastic_membrane_patch(),
        load,
        control="displacement",
        displacement_control=control,
        num_steps=3,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
        record_increment_snapshots=True,
    )

    _assert_solution_equal(persistent, legacy)
    assert persistent.info["displacement_control_linearization"] == "block_elimination"
    assert persistent.info["nonlinear_state_storage"]["activated"] is True
    assert persistent.info["nonlinear_state_storage"]["generation"] == len(
        persistent.steps
    )
    assert persistent.info["nonlinear_state_storage"]["stale_token_error_count"] == 0


def test_restart_pack_matches_mapping_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    permanent = _membrane_load("permanent", 325.0e6)
    environmental = _membrane_load("environmental", 20.0e6)
    preload = solve_static_nonlinear(
        _plastic_membrane_patch(),
        permanent,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
    )
    assert preload.status == "completed"

    persistent = solve_static_nonlinear(
        _plastic_membrane_patch(),
        environmental,
        constant_load_case=permanent,
        initial_element_states=preload.element_states,
        initial_displacements=preload.displacements,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
    )
    monkeypatch.setenv("FE_SOLVER_DISABLE_PERSISTENT_STATE", "yes")
    legacy = solve_static_nonlinear(
        _plastic_membrane_patch(),
        environmental,
        constant_load_case=permanent,
        initial_element_states=preload.element_states,
        initial_displacements=preload.displacements,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
    )

    _assert_solution_equal(persistent, legacy)
    assert persistent.info["nonlinear_state_storage"]["activated"] is True
    assert np.max(persistent.element_states[1]["alpha"]) > 0.0
    preload_alpha = preload.element_states[1]["alpha"].copy()
    persistent.element_states[1]["alpha"][...] = -1.0
    np.testing.assert_array_equal(preload.element_states[1]["alpha"], preload_alpha)


def test_plastic_restart_preserves_nonzero_prescribed_state_and_reactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = 3.0e-3
    preload_model = _plastic_membrane_patch(prescribed_right_x=target)
    preload = solve_static_nonlinear(
        preload_model,
        load_case=None,
        num_steps=3,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
    )
    assert preload.status == "completed"
    assert np.max(preload.element_states[1]["alpha"]) > 0.0

    restart_load = LoadCase("restart_shear")
    restart_load.add_nodal_load(
        3,
        load_vector=[0.0, 1.0e3, 0.0, 0.0, 0.0, 0.0],
    )
    persistent_model = _plastic_membrane_patch(prescribed_right_x=target)
    persistent = solve_static_nonlinear(
        persistent_model,
        restart_load,
        max_load_factor=0.1,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
        initial_element_states=preload.element_states,
        initial_displacements=preload.displacements,
        equilibrate_initial_state=False,
        record_increment_snapshots=True,
    )

    monkeypatch.setenv("FE_SOLVER_DISABLE_PERSISTENT_STATE", "1")
    legacy_model = _plastic_membrane_patch(prescribed_right_x=target)
    legacy = solve_static_nonlinear(
        legacy_model,
        restart_load,
        max_load_factor=0.1,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
        initial_element_states=preload.element_states,
        initial_displacements=preload.displacements,
        equilibrate_initial_state=False,
        record_increment_snapshots=True,
    )

    _assert_solution_equal(persistent, legacy)
    assert persistent.info["nonlinear_state_storage"]["activated"] is True
    assert persistent.info["prescribed_displacement_path"]["mode"] == (
        "restart_fixed_affine_state"
    )
    for node_id in (2, 3):
        dof = persistent_model.mesh.nodes[node_id].dofs[0]
        assert persistent.displacements[dof] == pytest.approx(target)
        assert [snapshot.displacements[dof] for snapshot in persistent.snapshots] == (
            pytest.approx([target] * len(persistent.snapshots))
        )
    assert persistent.steps[-1].support_reactions == legacy.steps[-1].support_reactions
    np.testing.assert_array_equal(
        persistent.element_states[1]["alpha"],
        legacy.element_states[1]["alpha"],
    )


def test_arc_length_store_matches_mapping_and_retries_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = _membrane_load("arc_reference", 400.0e6)
    control = ArcLengthControl(
        initial_load_increment=0.30,
        minimum_load_increment=0.0025,
        maximum_load_increment=0.30,
        target_iterations=3,
        max_steps=12,
        max_retries_per_step=8,
        maximum_absolute_load_factor=0.90,
    )
    persistent = solve_static_arc_length(
        _plastic_membrane_patch(),
        load,
        control=control,
        num_layers=3,
        max_iterations=2,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        record_increment_snapshots=True,
    )

    monkeypatch.setenv("FE_SOLVER_DISABLE_PERSISTENT_STATE", "on")
    legacy = solve_static_arc_length(
        _plastic_membrane_patch(),
        load,
        control=control,
        num_layers=3,
        max_iterations=2,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        record_increment_snapshots=True,
    )

    _assert_solution_equal(persistent, legacy)
    assert persistent.info["nonlinear_state_storage"]["activated"] is True
    assert persistent.info["nonlinear_state_storage"]["generation"] == len(
        persistent.steps
    )
    assert persistent.info["nonlinear_state_storage"]["stale_token_error_count"] == 0
    assert persistent.info["total_retries"] > 0
    assert any(
        row.get("action") == "cutback_after_nonconvergence"
        for row in persistent.info["adaptation_history"]
    )


def test_load_program_state_path_stays_exact_with_persistent_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permanent = _membrane_load("permanent", 325.0e6)
    environmental = _membrane_load("environmental", 20.0e6)
    program = NonlinearLoadProgram(
        (
            NonlinearLoadStage("permanent", permanent),
            NonlinearLoadStage("environmental", environmental),
        )
    )
    persistent = solve_static_nonlinear(
        _plastic_membrane_patch(),
        load_program=program,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
    )
    monkeypatch.setenv("FE_SOLVER_DISABLE_PERSISTENT_STATE", "1")
    legacy = solve_static_nonlinear(
        _plastic_membrane_patch(),
        load_program=program,
        num_steps=2,
        num_layers=3,
        max_iterations=20,
        tolerance=1.0e-9,
    )

    _assert_solution_equal(persistent, legacy)
    assert persistent.info["load_program_stage_factors"] == {
        "permanent": 1.0,
        "environmental": 1.0,
    }
