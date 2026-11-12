"""Nonlinear capacity workflow tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.capacity_workflow import (
    CapacityWorkflowConfig,
    evaluate_mode_mesh_adequacy,
    run_nonlinear_capacity_workflow,
    write_capacity_workflow_report,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def _beam_column_model(num_elements: int = 6) -> FEModel:
    model = FEModel("workflow_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    length = 4.0
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    for i in range(num_elements):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(range(1, num_elements + 2))
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_left", [1], {"ux": 0.0, "uy": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pin_right", [num_elements + 1], {"uy": 0.0}))
    return model


def _compression_load(model: FEModel) -> LoadCase:
    load = LoadCase("unit_compression")
    load.add_nodal_load(max(model.mesh.nodes), [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return load


def _prescribed_compression_model(num_elements: int = 6) -> FEModel:
    model = _beam_column_model(num_elements)
    end = num_elements + 1
    model.boundary_conditions = [
        condition
        for condition in model.boundary_conditions
        if condition.name != "pin_right"
    ]
    model.add_boundary_condition(
        BoundaryCondition(
            "prescribed_right",
            [end],
            {"ux": -0.0001, "uy": 0.0},
        )
    )
    return model


def test_capacity_workflow_runs_static_buckling_imperfection_and_nonlinear_capacity() -> None:
    model = _beam_column_model()
    load = _compression_load(model)
    config = CapacityWorkflowConfig(
        eigenmode_imperfection_amplitude=0.004,
        nonlinear_num_steps=3,
        nonlinear_max_load_factor=1.0,
        nonlinear_max_iterations=12,
        nonlinear_num_layers=3,
    )

    result = run_nonlinear_capacity_workflow(model, load, config=config)

    assert result.status == "completed"
    assert result.static_solver_info["convergence_info"]["status"] == "converged"
    assert result.prestress_summary["beam_elements"] == 6
    assert result.buckling_result.solver_status == "ok"
    assert result.critical_load_factor is not None
    assert result.imperfection.max_offset == pytest.approx(0.004)
    assert result.nonlinear_result.status == "completed"
    assert result.capacity_factor == pytest.approx(1.0)
    assert result.mesh_adequacy.status in {"ok", "warning"}
    assert result.to_dict()["imperfection"]["max_offset"] == pytest.approx(0.004)


def test_capacity_workflow_scales_prescribed_displacement_without_load_case() -> None:
    result = run_nonlinear_capacity_workflow(
        _prescribed_compression_model(),
        config=CapacityWorkflowConfig(
            num_buckling_modes=1,
            eigenmode_imperfection_amplitude=0.001,
            nonlinear_num_steps=2,
            nonlinear_max_load_factor=0.2,
            nonlinear_num_layers=3,
        ),
    )

    assert result.status == "completed"
    assert result.static_solver_info["convergence_info"]["status"] == "converged"
    assert result.buckling_result.solver_status == "ok"
    assert result.nonlinear_result.load_factor == pytest.approx(0.2)
    assert result.diagnostics["reference_action"] == "prescribed_displacement"


def test_capacity_workflow_rejects_an_actionless_model() -> None:
    with pytest.raises(ValueError, match="external reference load case or a nonzero"):
        run_nonlinear_capacity_workflow(_beam_column_model())


def test_capacity_workflow_mesh_mode_adequacy_warns_for_coarse_mode_representation() -> None:
    model = _beam_column_model(num_elements=2)
    load = _compression_load(model)
    result = run_nonlinear_capacity_workflow(
        model,
        load,
        config=CapacityWorkflowConfig(eigenmode_imperfection_amplitude=0.002, nonlinear_num_steps=1, mesh_min_elements_per_half_wave=10),
    )

    adequacy = evaluate_mode_mesh_adequacy(model, result.buckling_result, min_elements_per_half_wave=10)
    assert adequacy.status == "warning"
    assert adequacy.warnings
    assert adequacy.elements_per_half_wave < 10.0


def test_capacity_workflow_report_writer_and_cli() -> None:
    output_dir = Path(".pytest_tmp_capacity_workflow") / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = _beam_column_model()
        load = _compression_load(model)
        result = run_nonlinear_capacity_workflow(
            model,
            load,
            config=CapacityWorkflowConfig(eigenmode_imperfection_amplitude=0.001, nonlinear_num_steps=1, nonlinear_num_layers=3),
        )
        report_path = write_capacity_workflow_report(result, output_dir / "capacity.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "completed"
        assert report["buckling_solver_status"] == "ok"
        assert report["imperfection"]["max_offset"] == pytest.approx(0.001)

        completed = subprocess.run(
            [sys.executable, "scripts/run_capacity_workflow.py", "--output", str(output_dir / "cli_capacity.json"), "--steps", "2"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        cli_report = json.loads((output_dir / "cli_capacity.json").read_text(encoding="utf-8"))
        assert cli_report["status"] == "completed"
        assert cli_report["nonlinear_status"] == "completed"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
