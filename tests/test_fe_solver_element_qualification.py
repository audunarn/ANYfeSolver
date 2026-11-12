"""Q8, beam, and mass qualification expansion checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import (
    beam_qualification_metrics,
    generate_element_qualification_report,
    q4_q8_convergence_cost_sweep,
    q8_free_mode_metric,
    q8_mass_metric,
    q8_patch_metric,
    q8r_free_mode_metric,
    q8r_hourglass_assessment,
    q8r_mass_metric,
    q8r_patch_metric,
    reference_q8_geometries,
)
from anyfesolver.elements import ShellElement
from anyfesolver.fe_core import FEModel


def test_q8_free_element_has_six_rigid_modes_for_reference_geometries() -> None:
    for name, coords in reference_q8_geometries().items():
        metric = q8_free_mode_metric(coords)
        assert metric["zero_mode_count"] == 6, name
        assert abs(metric["relative_negative_eigenvalue"]) < 1.0e-12, name
        assert metric["relative_symmetry_error"] < 1.0e-12, name


def test_q8r_free_element_has_six_rigid_modes_after_hourglass_stabilization() -> None:
    for name, coords in reference_q8_geometries().items():
        metric = q8r_free_mode_metric(coords)
        assessment = q8r_hourglass_assessment(coords)
        assert metric["zero_mode_count"] == 6, name
        assert abs(metric["relative_negative_eigenvalue"]) < 1.0e-12, name
        assert metric["relative_symmetry_error"] < 1.0e-12, name
        assert assessment["extra_zero_energy_modes"] == 0, name
        assert assessment["hourglass_control"] == "active", name
        assert assessment["qc_status"] == "passed_stabilized_free_mode_check", name


def test_q8r_hourglass_cache_is_invalidated_after_geometry_change() -> None:
    model = FEModel("q8r_cache")
    model.add_material("steel", 210.0e9, 0.3)
    for node_id, coordinate in enumerate(reference_q8_geometries()["square"], start=1):
        model.add_node(node_id, *np.asarray(coordinate, dtype=float))
    element = ShellElement(
        1,
        list(range(1, 9)),
        "steel",
        thickness=0.01,
        reduced_integration=True,
    )
    model.add_element(1, element)
    element.compute_stiffness_matrix(model.mesh, model.get_material("steel"))
    assert element._hourglass_stiffness_matrix is not None
    node = model.mesh.get_node(5)
    model.set_node_coordinates(5, node.x, node.y, node.z + 1.0e-4)
    assert element._hourglass_stiffness_matrix is None


def test_q8_square_membrane_bending_and_shear_patches_are_exact() -> None:
    patch = q8_patch_metric(reference_q8_geometries()["square"])

    assert patch["membrane_max_relative_error"] < 1.0e-10
    assert patch["bending_relative_error"] < 1.0e-10
    assert patch["bending_relative_spread"] < 1.0e-10
    assert patch["shear_relative_error"] < 1.0e-10
    assert patch["shear_relative_spread"] < 1.0e-10


def test_q8r_square_membrane_bending_and_shear_patches_are_exact_at_reduced_points() -> None:
    patch = q8r_patch_metric(reference_q8_geometries()["square"])

    assert patch["membrane_max_relative_error"] < 1.0e-10
    assert patch["bending_relative_error"] < 1.0e-10
    assert patch["bending_relative_spread"] < 1.0e-10
    assert patch["shear_relative_error"] < 1.0e-10
    assert patch["shear_relative_spread"] < 1.0e-10


def test_q8_mass_matches_density_area_thickness_for_square() -> None:
    metric = q8_mass_metric(reference_q8_geometries()["square"], thickness=0.01)

    assert metric["total_mass"] == pytest.approx(metric["expected_mass_from_corner_area"])
    assert metric["relative_mass_error"] < 1.0e-12
    assert metric["assembled_translation_masses"]["x"] == pytest.approx(metric["total_mass"])
    assert metric["assembled_translation_masses"]["y"] == pytest.approx(metric["total_mass"])
    assert metric["assembled_translation_masses"]["z"] == pytest.approx(metric["total_mass"])


def test_q8r_mass_matches_density_area_thickness_for_square() -> None:
    metric = q8r_mass_metric(reference_q8_geometries()["square"], thickness=0.01)

    assert metric["total_mass"] == pytest.approx(metric["expected_mass_from_corner_area"])
    assert metric["relative_mass_error"] < 1.0e-12
    assert metric["assembled_translation_masses"]["x"] == pytest.approx(metric["total_mass"])
    assert metric["assembled_translation_masses"]["y"] == pytest.approx(metric["total_mass"])
    assert metric["assembled_translation_masses"]["z"] == pytest.approx(metric["total_mass"])


def test_q4_q8_convergence_cost_sweep_reports_status_and_refinement() -> None:
    rows = q4_q8_convergence_cost_sweep()

    assert len(rows) >= 2
    assert all(row["q4_status"] == "converged" and row["q8_status"] == "converged" for row in rows)
    assert all(row["q8_dofs"] > row["q4_dofs"] for row in rows)
    coarse, fine = rows[0], rows[-1]
    assert abs(fine["displacement_ratio_q4_to_q8"] - 1.0) < abs(coarse["displacement_ratio_q4_to_q8"] - 1.0)


def test_beam_qualification_metrics_cover_two_topologies_orientation_and_mass() -> None:
    metrics = beam_qualification_metrics()

    assert metrics["max_relative_error"] < 3.0e-3
    cases = {(row["element"], row["axis"], row["case"]) for row in metrics["response"]}
    assert ("BeamElement", "X", "strong_axis_tip") in cases
    assert ("BeamElement", "Y", "strong_axis_tip") in cases
    assert ("QuadraticBeamElement", "X", "strong_axis_tip") in cases
    assert ("QuadraticBeamElement", "Y", "strong_axis_tip") in cases
    assert ("BeamElement", "X", "torsion") in cases
    assert metrics["beam_mass"]["total_mass"] > 0.0


def test_element_qualification_report_and_cli() -> None:
    report = generate_element_qualification_report()
    assert "q8" in report
    assert "q8r" in report
    assert "beam" in report
    assert report["q8"]["geometry_metrics"]["square"]["free_modes"]["zero_mode_count"] == 6
    assert report["q8r"]["geometry_metrics"]["square"]["free_modes"]["zero_mode_count"] == 6
    assert report["q8r"]["qc_status"] == "passed_stabilized_free_mode_check"
    assert report["q8r"]["geometry_metrics"]["square"]["hourglass_assessment"]["extra_zero_energy_modes"] == 0

    output_dir = Path(".pytest_tmp_element_qualification") / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "element_qualification.json"
        md_path = output_dir / "element_qualification.md"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_element_qualification.py",
                "--output",
                str(json_path),
                "--markdown",
                str(md_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        from_disk = json.loads(json_path.read_text(encoding="utf-8"))
        assert from_disk["q8"]["geometry_metrics"]["square"]["patch"]["membrane_max_relative_error"] < 1.0e-10
        assert from_disk["q8r"]["qc_status"] == "passed_stabilized_free_mode_check"
        assert md_path.exists()
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


def test_q8_mass_exact_for_curved_midside_geometry() -> None:
    """Displaced midside nodes curve the edges; mass must match the true isoparametric area."""
    coords = reference_q8_geometries()["distorted_midside"]
    for metric in (q8_mass_metric(coords), q8r_mass_metric(coords)):
        assert metric["relative_mass_error"] < 1.0e-12
        # The straight corner-quad reference genuinely differs for curved edges.
        assert metric["corner_area_mass_deviation"] == pytest.approx(0.07, abs=0.005)
