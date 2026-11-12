"""Linear transient dynamics and slamming pressure-patch tests."""

from __future__ import annotations

import numpy as np
import pytest

from anyfesolver import (
    LoadCase,
    PressurePatch,
    TransientConfig,
    assemble_pressure_patch_load_vector,
    generate_simple_panel_mesh,
    solve_transient_newmark,
)
from anyfesolver.boundary import BoundaryCondition, FixedSupport
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.validation import load_vector_resultant


def test_pressure_patch_centroid_selection_and_resultant() -> None:
    model = generate_simple_panel_mesh(2.0, 2.0, 0.01, num_divisions_x=2, num_divisions_y=2)
    patch = PressurePatch.rectangular(
        name="lower_left",
        pressure_time=500.0,
        center=(0.5, 0.5, 0.0),
        size=(1.0, 1.0),
    )

    vector, info = assemble_pressure_patch_load_vector(model, patch, pressure=patch.pressure_at(0.0))
    resultant = load_vector_resultant(model, vector)

    assert info["num_selected_elements"] == 1
    assert info["selected_element_ids"] == [1]
    np.testing.assert_allclose(resultant.force, [0.0, 0.0, 500.0], atol=1.0e-10)


def _axial_sdof_model() -> tuple[FEModel, int]:
    model = FEModel("axial_sdof")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    section = {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    return model, model.mesh.get_node(2).dofs[0]


def test_newmark_step_load_matches_axial_sdof_solution() -> None:
    model, ux_dof = _axial_sdof_model()
    load_case = LoadCase("step")
    load_case.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    config = TransientConfig(dt=5.0e-4, t_end=0.05, output_nodes=[2])

    result = solve_transient_newmark(model, config, base_load_case=load_case)

    omega = 10.0  # sqrt(k/m) with k=100, m=1.
    expected = 1.0 / 100.0 * (1.0 - np.cos(omega * config.t_end))
    assert result.status == "completed"
    assert result.displacements[-1, ux_dof] == pytest.approx(expected, rel=2.0e-3)
    assert result.peak_displacement_node == 2
    np.testing.assert_allclose(result.node_histories[2], result.node_displacement_history(model, 2))
    assert result.diagnostics["factorization_count"] == 1
    assert result.diagnostics["factorization_reused"] is True


def test_newmark_undamped_free_vibration_conserves_energy() -> None:
    model, ux_dof = _axial_sdof_model()
    initial_displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    initial_displacement[ux_dof] = 0.01
    config = TransientConfig(dt=1.0e-3, t_end=0.2, initial_displacement=initial_displacement)

    result = solve_transient_newmark(model, config)

    assert result.status == "completed"
    assert result.diagnostics["max_relative_energy_drift"] < 1.0e-10
    assert np.max(np.abs(result.velocities[:, ux_dof])) > 0.0


def test_short_pressure_pulse_impulse_is_integrated_from_patch_load() -> None:
    model = generate_simple_panel_mesh(1.0, 1.0, 0.01, num_divisions_x=1, num_divisions_y=1)
    model.materials["steel"].density = 7850.0
    patch = PressurePatch.rectangular_pulse(
        name="slam",
        pressure=1000.0,
        start_time=0.001,
        end_time=0.003,
        element_ids=[1],
    )
    config = TransientConfig(dt=0.001, t_end=0.004, save_every=1)

    result = solve_transient_newmark(model, config, pressure_patches=[patch])

    assert result.status == "completed"
    assert result.force_impulse[2] == pytest.approx(2.0, rel=1.0e-12)
    assert np.linalg.norm(result.load_impulse) > 0.0
    assert result.diagnostics["pressure_patches"][0]["num_selected_elements"] == 1
