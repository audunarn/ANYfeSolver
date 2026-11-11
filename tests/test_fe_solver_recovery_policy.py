"""Selective recovery and resource-policy foundation tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import (
    LoadCase,
    RecoveryConfig,
    ResourcePolicyError,
    ResourceConfig,
    TransientConfig,
    create_fe_result,
    estimate_model_memory,
    generate_recovery_policy_report,
    generate_simple_panel_mesh,
    recover_element_stresses,
    recover_element_stresses_with_report,
    select_node_displacements,
    solve_transient_newmark,
)
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def test_recovery_config_selects_nodes_elements_and_components() -> None:
    model = generate_simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    displacement[model.mesh.nodes[1].dofs[2]] = 0.01
    recovery = RecoveryConfig(node_ids=[1, 3], element_ids=[1], components=["von_mises"])

    node_displacements = select_node_displacements(model, displacement, recovery)
    stresses = recover_element_stresses(model, displacement, recovery)

    assert sorted(node_displacements) == [1, 3]
    assert sorted(stresses) == [1]
    assert sorted(stresses[1]) == ["von_mises"]


def test_create_fe_result_keeps_default_full_recovery_and_supports_selective_recovery() -> None:
    model = generate_simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)

    full = create_fe_result(model, displacement, {"solver_type": "unit"})
    selective = create_fe_result(
        model,
        displacement,
        {"solver_type": "unit"},
        recovery_config=RecoveryConfig(node_ids=[1], element_ids=[], include_reactions=False),
        resource_config=ResourceConfig(solver_threads=1, deterministic=True),
    )

    assert len(full.node_displacements) == len(model.mesh.nodes)
    assert len(full.element_stresses) == len(model.mesh.elements)
    assert sorted(selective.node_displacements) == [1]
    assert selective.element_stresses == {}
    assert selective.reactions == {}
    assert selective.solver_info["recovery_policy"]["resources"]["solver_threads"] == 1
    assert selective.solver_info["recovery_policy"]["memory_estimate"]["total_dofs"] == model.mesh.dof_manager.total_dofs
    assert selective.solver_info["recovery_policy"]["execution"]["element_stress_recovery"]["backend"] == "serial"


def test_threaded_stress_recovery_matches_serial_and_reports_workers() -> None:
    model = generate_simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=3, num_divisions_y=2)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    recovery = RecoveryConfig(components=["von_mises"])

    serial, serial_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=1),
    )
    threaded, threaded_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=2),
    )

    assert threaded_report.backend == "thread_pool"
    assert threaded_report.requested_workers == 2
    assert threaded_report.used_workers == 2
    native_policy = threaded_report.metadata["native_thread_policy"]
    assert native_policy["requested_threads"] == 1
    assert native_policy["restored"] is True
    assert sorted(serial) == sorted(threaded)
    for element_id in serial:
        np.testing.assert_allclose(serial[element_id]["von_mises"], threaded[element_id]["von_mises"])
    assert serial_report.backend == "serial"


def test_memory_estimate_reflects_selected_history_storage() -> None:
    model = generate_simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=1)
    full = estimate_model_memory(model, transient_saved_steps=5, store_full_history=True)
    selected = estimate_model_memory(
        model,
        transient_saved_steps=5,
        store_full_history=False,
        recovery_config=RecoveryConfig(node_ids=[1], store_full_histories=False),
    )

    assert full.matrix_nnz_estimate > 0
    assert full.history_bytes_estimate > selected.history_bytes_estimate
    assert selected.history_bytes_estimate == 5 * 6 * 8 * 3


def test_memory_estimate_accounts_for_recovery_state_snapshot_copy() -> None:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=1,
        num_divisions_y=1,
    )

    one_copy = estimate_model_memory(
        model,
        nonlinear_state=True,
    )
    two_copies = estimate_model_memory(
        model,
        nonlinear_state=True,
        nonlinear_state_copies=2,
    )

    assert (
        two_copies.nonlinear_state_bytes_estimate
        == 2 * one_copy.nonlinear_state_bytes_estimate
    )
    assert "2 retained state copies" in " ".join(two_copies.notes)


def test_memory_limit_rejects_result_recovery_preflight() -> None:
    model = generate_simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)

    with pytest.raises(ResourcePolicyError) as exc_info:
        create_fe_result(
            model,
            displacement,
            {"solver_type": "unit"},
            recovery_config=RecoveryConfig(),
            resource_config=ResourceConfig(memory_limit_bytes=1),
        )

    assert exc_info.value.context == "create_fe_result"
    assert exc_info.value.memory_estimate is not None


def test_resource_and_recovery_config_validation() -> None:
    with pytest.raises(ValueError):
        RecoveryConfig(history_mode="all_the_things")
    with pytest.raises(ValueError):
        ResourceConfig(solver_threads=0)
    with pytest.raises(ValueError):
        ResourceConfig(memory_limit_bytes=-1)


def _axial_sdof_model() -> tuple[FEModel, LoadCase]:
    model = FEModel("recovery_policy_sdof")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    section = {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    load_case = LoadCase("step")
    load_case.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load_case


def test_transient_records_recovery_resource_and_memory_provenance() -> None:
    model, load_case = _axial_sdof_model()
    config = TransientConfig(
        dt=0.001,
        t_end=0.002,
        recovery=RecoveryConfig(node_ids=[2], history_mode="selected", store_full_histories=False),
        resource_config=ResourceConfig(solver_threads=1, recovery_threads=1),
    )

    result = solve_transient_newmark(model, config, base_load_case=load_case)

    assert sorted(result.node_histories) == [2]
    policy = result.diagnostics["recovery_policy"]
    assert policy["recovery"]["node_ids"] == [2]
    assert policy["resources"]["solver_threads"] == 1
    assert policy["memory_estimate"]["total_dofs"] == model.mesh.dof_manager.total_dofs
    assert result.result_case["recovery"]["node_ids"] == [2]
    assert result.history_storage_mode == "selected"
    assert result.displacements.shape == (3, 6)
    assert result.velocities.shape == (3, 6)
    assert result.accelerations.shape == (3, 6)


def test_transient_envelope_mode_avoids_full_history_arrays() -> None:
    model, load_case = _axial_sdof_model()
    config = TransientConfig(
        dt=0.001,
        t_end=0.003,
        recovery=RecoveryConfig(node_ids=[2], history_mode="envelope", store_full_histories=False),
    )

    result = solve_transient_newmark(model, config, base_load_case=load_case)

    assert result.history_storage_mode == "envelope"
    assert result.displacements.shape == (0, 0)
    assert result.displacement_envelope is not None
    assert result.velocity_envelope is not None
    assert result.acceleration_envelope is not None
    assert result.node_displacement_history(model, 2).shape == (4, 6)


def test_transient_memory_limit_rejects_before_time_history_storage() -> None:
    model, load_case = _axial_sdof_model()
    config = TransientConfig(
        dt=0.001,
        t_end=0.002,
        recovery=RecoveryConfig(node_ids=[2], history_mode="selected", store_full_histories=False),
        resource_config=ResourceConfig(memory_limit_bytes=1),
    )

    with pytest.raises(ResourcePolicyError) as exc_info:
        solve_transient_newmark(model, config, base_load_case=load_case)

    assert exc_info.value.context == "solve_transient_newmark"


def test_recovery_policy_report_and_cli_write_outputs() -> None:
    report = generate_recovery_policy_report()
    assert report["status"] == "passed"
    assert report["reductions"]["history_memory_reduction_fraction"] > 0.0
    assert report["measured_parallel_recovery"]["results_match"] is True
    assert report["measured_parallel_recovery"]["threaded"]["backend"] == "thread_pool"

    output_dir = Path(".pytest_tmp_recovery_policy")
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "recovery_policy.json"
    markdown = output_dir / "recovery_policy.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_recovery_policy.py",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert markdown.exists()
