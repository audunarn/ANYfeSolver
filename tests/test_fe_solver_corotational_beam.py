"""Corotational 2-node beam geometric nonlinearity tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.beam_validity import generate_beam_validity_report
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


E = 210.0e9
NU = 0.3


def _single_beam(section_extra=None) -> tuple[FEModel, BeamElement]:
    model = FEModel("corot_beam")
    model.add_material("steel", E, NU)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    section = {
        "area": 0.01,
        "Iy": 2.0e-6,
        "Iz": 1.0e-6,
        "J": 1.0e-6,
        "orientation": (0.0, 0.0, 1.0),
    }
    section.update(section_extra or {})
    element = BeamElement(1, [1, 2], "steel", section)
    model.add_element(1, element)
    return model, element


def test_corotational_beam_large_rigid_rotation_has_zero_internal_force() -> None:
    model, element = _single_beam({"geometric_nonlinearity": "corotational"})
    theta = np.deg2rad(35.0)
    length = 2.0
    u = np.zeros(12)
    u[6] = length * np.cos(theta) - length
    u[7] = length * np.sin(theta)
    u[5] = theta
    u[11] = theta

    force, tangent, state = element.compute_nonlinear_response(model.mesh, model.get_material("steel"), u, tangent=True)

    assert np.linalg.norm(force) < 1.0e-5
    assert tangent.shape == (12, 12)
    assert state["geometric_nonlinearity"] == "corotational"
    assert state["axial_extension"] == pytest.approx(0.0, abs=1.0e-12)
    assert state["basic_deformation_norm"] < 1.0e-12


def test_corotational_beam_axial_extension_matches_ea_over_l_response() -> None:
    model, element = _single_beam({"geometric_nonlinearity": "corotational"})
    extension = 0.002
    u = np.zeros(12)
    u[6] = extension

    force, _tangent, state = element.compute_nonlinear_response(model.mesh, model.get_material("steel"), u, tangent=True)

    expected = E * 0.01 / 2.0 * extension
    assert force[0] == pytest.approx(-expected)
    assert force[6] == pytest.approx(expected)
    assert state["current_length"] == pytest.approx(2.0 + extension)


def test_default_beam_path_is_not_corotational_unless_requested() -> None:
    model, element = _single_beam()
    theta = np.deg2rad(20.0)
    length = 2.0
    u = np.zeros(12)
    u[6] = length * np.cos(theta) - length
    u[7] = length * np.sin(theta)
    u[5] = theta
    u[11] = theta

    force, _tangent, state = element.compute_nonlinear_response(model.mesh, model.get_material("steel"), u, tangent=True)

    assert state is None
    assert np.linalg.norm(force) > 1.0e3


def test_beam_validity_report_and_cli() -> None:
    report = generate_beam_validity_report()
    rigid = report["corotational_v1"]["rigid_rotation"]
    axial = report["corotational_v1"]["axial_extension"]
    assert rigid["corotational_force_norm"] < 1.0e-5
    assert rigid["force_norm_ratio_corot_to_default"] < 1.0e-10
    assert axial["relative_error"] < 1.0e-12

    output_dir = Path(".pytest_tmp_beam_validity") / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_beam_validity.py",
                "--output",
                str(output_dir / "beam_validity.json"),
                "--markdown",
                str(output_dir / "beam_validity.md"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        written = json.loads((output_dir / "beam_validity.json").read_text(encoding="utf-8"))
        assert written["corotational_v1"]["rigid_rotation"]["corotational_force_norm"] < 1.0e-5
        assert (output_dir / "beam_validity.md").exists()
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
