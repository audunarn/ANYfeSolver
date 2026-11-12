"""Plasticity return-map and nonlinear tangent qualification checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from anyfesolver import (
    element_tangent_metrics,
    generate_plasticity_qualification_report,
    material_point_path_metrics,
    reference_plastic_curve,
    yield_function_residual,
)
from anyfesolver.plasticity import plane_stress_return_map
from anyfesolver.elements import _jit_integrate_nonlinear_response
from anyfesolver.vectorized_nonlinear import _jit_batch_integrate_nonlinear_response

import numpy as np


def test_material_point_paths_have_small_yield_residuals() -> None:
    metrics = material_point_path_metrics()

    assert metrics["elastic"]["stress_relative_error"] < 1.0e-14
    assert metrics["elastic"]["tangent_relative_error"] < 1.0e-14
    assert metrics["max_abs_yield_residual"] < 1.0e-8
    assert metrics["plastic_paths"]["uniaxial"]["alpha"] > 0.0
    assert metrics["plastic_paths"]["pure_shear"]["alpha"] > 0.0
    assert metrics["plastic_paths"]["unload_from_plastic"]["alpha_change"] == 0.0


def test_yield_function_residual_matches_returned_state() -> None:
    curve = reference_plastic_curve()
    strain = np.array([[0.003, 0.001, 0.0005]], dtype=float)
    stress, _tangent, _plastic, alpha = plane_stress_return_map(
        strain,
        np.zeros_like(strain),
        np.zeros(1),
        210.0e9,
        0.3,
        curve,
    )

    assert abs(yield_function_residual(stress[0], float(alpha[0]), curve)) < 1.0e-8


def test_shell_membrane_bending_coupling_uses_both_consistent_cross_terms() -> None:
    n_dof = 4
    B_m = np.array(
        [[[1.0, -0.3, 0.2, 0.4], [0.1, 0.7, -0.5, 0.2], [0.3, -0.1, 0.6, -0.2]]],
        dtype=float,
    )
    B_b = np.array(
        [[[0.2, 0.5, -0.4, 0.1], [-0.6, 0.3, 0.2, 0.8], [0.4, -0.2, 0.1, 0.5]]],
        dtype=float,
    )
    C1 = np.array([[[3.0, 0.4, -0.2], [0.4, 2.0, 0.3], [-0.2, 0.3, 1.5]]], dtype=float)
    detw = np.array([1.7], dtype=float)
    zeros_resultant = np.zeros((1, 3), dtype=float)
    zeros_modulus = np.zeros((1, 3, 3), dtype=float)
    B_d = np.zeros((1, 1, n_dof), dtype=float)
    Gw = np.zeros((1, 2, n_dof), dtype=float)
    B_s = np.zeros((0, 2, n_dof), dtype=float)
    expected = (B_m[0].T @ C1[0] @ B_b[0] + B_b[0].T @ C1[0] @ B_m[0]) * detw[0]

    _force, tangent = _jit_integrate_nonlinear_response(
        np.zeros(n_dof),
        zeros_resultant,
        zeros_resultant,
        zeros_modulus,
        C1,
        zeros_modulus,
        B_m,
        B_b,
        B_d,
        Gw,
        detw,
        B_s,
        np.zeros(0),
        np.zeros((2, 2)),
        0.0,
        True,
        True,
        n_dof,
    )
    np.testing.assert_allclose(tangent, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(tangent, tangent.T, rtol=1.0e-13, atol=1.0e-13)

    _batch_force, batch_tangent = _jit_batch_integrate_nonlinear_response(
        np.zeros((1, n_dof)),
        zeros_resultant[None, ...],
        zeros_resultant[None, ...],
        zeros_modulus[None, ...],
        C1[None, ...],
        zeros_modulus[None, ...],
        B_m[None, ...],
        B_b[None, ...],
        B_d[None, ...],
        Gw[None, ...],
        detw[None, ...],
        B_s[None, ...],
        np.zeros((1, 0)),
        np.zeros((2, 2)),
        0.0,
        True,
        True,
        n_dof,
    )
    np.testing.assert_allclose(batch_tangent[0], expected, rtol=1.0e-13, atol=1.0e-13)


def test_element_tangent_metrics_are_algorithmic_and_finite_difference_tight() -> None:
    metrics = element_tangent_metrics()

    assert metrics["beam_elastic"]["tangent_fd_relative_error"] < 1.0e-8
    assert metrics["beam_fiber_plastic"]["tangent_fd_relative_error"] < 1.0e-8
    assert metrics["shell_elastic"]["tangent_fd_relative_error"] < 1.0e-8
    assert metrics["shell_layered_plastic"]["state_summary"]["alpha_max"] > 0.0
    assert metrics["shell_layered_plastic"]["tangent_status"] == "tight"
    assert metrics["shell_layered_plastic"]["tangent_fd_relative_error"] < 1.0e-4
    assert metrics["max_algorithmic_tangent_error"] < 1.0e-4


def test_plasticity_qualification_report_exposes_limits_and_dnv_curves() -> None:
    report = generate_plasticity_qualification_report()

    assert report["status"] == "passed"
    assert "S355_0.01" in report["dnv_curves"]
    assert report["material_point"]["max_abs_yield_residual"] < 1.0e-8
    assert report["element_tangents"]["max_algorithmic_tangent_error"] < 1.0e-4
    assert any("numerical tangent" in item for item in report["known_limitations"])


def test_plasticity_qualification_cli_writes_json_and_markdown() -> None:
    output_dir = Path(".pytest_tmp_plasticity_qualification") / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "plasticity_qualification.json"
        md_path = output_dir / "plasticity_qualification.md"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_plasticity_qualification.py",
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
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["status"] == "passed"
        assert md_path.exists()
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
