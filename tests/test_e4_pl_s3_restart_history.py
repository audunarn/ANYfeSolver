from __future__ import annotations

import copy

import numpy as np
import pytest

from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement
from anyfesolver.e4_pl_s3_state import (
    canonical_json_bytes,
    strict_canonical_json_loads,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import LinearHardeningCurve
from anyfesolver.nonlinear_static import (
    NonlinearLoadProgram,
    NonlinearLoadStage,
    solve_static_nonlinear,
)


def _restart_model() -> FEModel:
    model = FEModel("qualified-s3-restart-history")
    model.add_material(
        "steel",
        210.0e9,
        0.3,
        density=7850.0,
        hardening_curve=LinearHardeningCurve(250.0e6, 2.0e9),
    )
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 0.2, 0.9, 0.0)
    model.add_element(
        1,
        QualifiedE4PLS3ShellElement(
            1,
            [1, 2, 3],
            "steel",
            thickness=0.01,
            reference_normal=(0.0, 0.0, 1.0),
        ),
    )
    model.add_boundary_condition(FixedSupport("node-1", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "node-2-axial-only",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(FixedSupport("node-3", [3]))
    return model


def _axial_load(name: str, value: float) -> LoadCase:
    load = LoadCase(name)
    load.add_nodal_load(2, [float(value), 0.0, 0.0, 0.0, 0.0, 0.0])
    return load


def _mixed_q4_s3_model() -> tuple[FEModel, LoadCase]:
    model = _restart_model()
    model.add_material("elastic-q4", 210.0e9, 0.3, density=7850.0)
    for node_id, coordinates in {
        4: (0.0, 2.0, 0.0),
        5: (1.0, 2.0, 0.0),
        6: (1.0, 3.0, 0.0),
        7: (0.0, 3.0, 0.0),
    }.items():
        model.add_node(node_id, *coordinates)
    model.add_element(
        2,
        QualifiedE4PLShellElement(
            2,
            [4, 5, 6, 7],
            "elastic-q4",
            thickness=0.01,
        ),
    )
    model.add_boundary_condition(FixedSupport("q4-node-4", [4]))
    model.add_boundary_condition(
        BoundaryCondition(
            "q4-node-5-axial-only",
            [5],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(FixedSupport("q4-node-6-7", [6, 7]))
    load = LoadCase("mixed-checkpoint-path")
    load.add_nodal_load(2, [0.25e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    load.add_nodal_load(5, [0.25e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load


def _solve_preload() -> tuple[FEModel, object]:
    model = _restart_model()
    result = solve_static_nonlinear(
        model,
        _axial_load("preload", 2.0e6),
        # The ordered two-stage reference below uses four equal path steps,
        # hence exactly two steps belong to this first unit-factor stage.
        num_steps=2,
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        convergence_settings="legacy",
    )
    assert result.status == "completed"
    assert float(np.max(np.asarray(result.element_states[1]["alpha"]))) > 0.0
    return model, result


def test_canonical_s3_restart_matches_the_ordered_two_stage_path() -> None:
    _preload_model, preload = _solve_preload()
    original_state_bytes = canonical_json_bytes(preload.element_states[1])
    restored_state = strict_canonical_json_loads(original_state_bytes)

    split_model = _restart_model()
    split = solve_static_nonlinear(
        split_model,
        _axial_load("increment", 0.35e6),
        constant_load_case=_axial_load("preload", 2.0e6),
        num_steps=2,
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        initial_element_states={1: restored_state},
        initial_displacements=np.asarray(preload.displacements, dtype=float).copy(),
        equilibrate_initial_state=False,
        record_increment_snapshots=True,
    )

    program_model = _restart_model()
    program = solve_static_nonlinear(
        program_model,
        load_program=NonlinearLoadProgram(
            (
                NonlinearLoadStage("preload", _axial_load("preload", 2.0e6)),
                NonlinearLoadStage("increment", _axial_load("increment", 0.35e6)),
            )
        ),
        num_steps=4,
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        record_increment_snapshots=True,
    )

    assert split.status == program.status == "completed"
    assert split.info["prescribed_displacement_path"]["mode"] == (
        "restart_fixed_affine_state"
    )
    np.testing.assert_array_equal(split.displacements, program.displacements)
    assert canonical_json_bytes(split.element_states[1]) == canonical_json_bytes(
        program.element_states[1]
    )
    assert canonical_json_bytes(preload.element_states[1]) == original_state_bytes
    assert split.element_states[1]["state_integrity_sha256"] != (
        preload.element_states[1]["state_integrity_sha256"]
    )
    assert float(np.max(np.asarray(split.element_states[1]["alpha"]))) >= float(
        np.max(np.asarray(preload.element_states[1]["alpha"]))
    )
    assert split.info["nonlinear_state_storage"]["native_rotation_activated"] is True


def test_s3_restart_rejects_displacement_state_mismatch_before_mutation() -> None:
    _preload_model, preload = _solve_preload()
    state = copy.deepcopy(preload.element_states[1])
    state_bytes = canonical_json_bytes(state)
    mismatched_displacement = np.asarray(preload.displacements, dtype=float).copy()
    element_u = np.asarray(state["committed_total_u"], dtype=float).reshape(3, 6)
    assert np.max(np.abs(element_u)) > 0.0
    mismatched_displacement[:] = 0.0

    with pytest.raises(ValueError, match="committed_total_u|displacement"):
        solve_static_nonlinear(
            _restart_model(),
            _axial_load("increment", 0.1e6),
            constant_load_case=_axial_load("preload", 2.0e6),
            num_steps=1,
            num_layers=3,
            initial_element_states={1: state},
            initial_displacements=mismatched_displacement,
            equilibrate_initial_state=False,
        )

    assert canonical_json_bytes(state) == state_bytes


def test_canonical_solver_checkpoint_preserves_shared_so3_and_plastic_history() -> None:
    load = _axial_load("checkpoint-path", 2.0e6)
    uninterrupted = solve_static_nonlinear(
        _restart_model(),
        load,
        max_load_factor=1.0,
        num_steps=4,
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    first = solve_static_nonlinear(
        _restart_model(),
        load,
        max_load_factor=0.5,
        num_steps=2,
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    resumed = solve_static_nonlinear(
        _restart_model(),
        load,
        max_load_factor=1.0,
        num_steps=2,
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        convergence_settings="legacy",
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )

    assert uninterrupted.status == resumed.status == "completed"
    np.testing.assert_array_equal(uninterrupted.displacements, resumed.displacements)
    assert canonical_json_bytes(uninterrupted.element_states[1]) == canonical_json_bytes(
        resumed.element_states[1]
    )
    assert uninterrupted.restart_checkpoint_bytes() == resumed.restart_checkpoint_bytes()
    assert resumed.info["nonlinear_state_storage"]["native_rotation_activated"] is True


def test_arc_checkpoint_preserves_nonzero_shared_so3_continuation() -> None:
    load = _axial_load("arc-checkpoint-path", 0.35e6)
    common = {
        "initial_load_increment": 0.1,
        "minimum_load_increment": 0.1,
        "maximum_load_increment": 0.1,
        "growth_factor": 1.0,
        "stop_after_peak_steps": 20,
    }
    uninterrupted = solve_static_arc_length(
        _restart_model(),
        load,
        control=ArcLengthControl(max_steps=4, **common),
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        emit_restart_checkpoint=True,
    )
    first = solve_static_arc_length(
        _restart_model(),
        load,
        control=ArcLengthControl(max_steps=2, **common),
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        emit_restart_checkpoint=True,
    )
    resumed = solve_static_arc_length(
        _restart_model(),
        load,
        control=ArcLengthControl(max_steps=2, **common),
        num_layers=3,
        max_iterations=30,
        tolerance=1.0e-9,
        arc_tolerance=1.0e-9,
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )

    assert uninterrupted.status == resumed.status == "maximum_steps_reached"
    assert first.load_factor != 0.0
    np.testing.assert_array_equal(uninterrupted.displacements, resumed.displacements)
    assert canonical_json_bytes(uninterrupted.element_states[1]) == canonical_json_bytes(
        resumed.element_states[1]
    )
    assert uninterrupted.restart_checkpoint_bytes() == resumed.restart_checkpoint_bytes()


def test_mixed_qualified_q4_s3_checkpoint_has_ordered_complete_state() -> None:
    full_model, full_load = _mixed_q4_s3_model()
    full = solve_static_nonlinear(
        full_model,
        full_load,
        max_load_factor=1.0,
        num_steps=4,
        num_layers=3,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    first_model, first_load = _mixed_q4_s3_model()
    first = solve_static_nonlinear(
        first_model,
        first_load,
        max_load_factor=0.5,
        num_steps=2,
        num_layers=3,
        convergence_settings="legacy",
        emit_restart_checkpoint=True,
    )
    resumed_model, resumed_load = _mixed_q4_s3_model()
    resumed = solve_static_nonlinear(
        resumed_model,
        resumed_load,
        max_load_factor=1.0,
        num_steps=2,
        num_layers=3,
        convergence_settings="legacy",
        restart_checkpoint=first.restart_checkpoint_bytes(),
    )

    assert [record["element_id"] for record in first.restart_checkpoint["element_states"]] == [1, 2]
    np.testing.assert_array_equal(full.displacements, resumed.displacements)
    assert full.restart_checkpoint_bytes() == resumed.restart_checkpoint_bytes()
