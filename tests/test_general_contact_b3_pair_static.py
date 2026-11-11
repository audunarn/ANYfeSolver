"""Opt-in classical B3/B3 distributed pair contact through static solver."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_curve_pair_static import B3PairManager


def _pair():
    model = FEModel("b3_round_pair")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    positions = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
                 (0.0, 0.12, 0.0), (0.5, 0.12, 0.0), (1.0, 0.12, 0.0))
    for node_id, point in enumerate(positions, 1):
        model.add_node(node_id, *point)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4,
               "J": 1.0e-4}
    model.add_element(1, QuadraticBeamElement(1, (1, 2, 3), "elastic", section))
    model.add_element(2, QuadraticBeamElement(2, (4, 5, 6), "elastic", section))
    definition = ContactDefinition(
        surfaces=(ContactSurface("a", (1,), beam_radius=0.05),
                  ContactSurface("b", (2,), beam_radius=0.05)),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "a_b", "a", "b", normal_stiffness=None,
            gap_tolerance=0.015, line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-6,
        maximum_line_subdivisions=16,
        beam_pair_gap_accuracy=1.0e-5,
        maximum_beam_pair_gap_cells=10000,
        maximum_beam_pair_gap_depth=25,
    )
    return model, definition, settings


def test_b3_pair_global_work_reaction_tangent_and_swept_crossing():
    model, definition, settings = _pair()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = B3PairManager(model, definition, settings)
    open_force, _open_tangent, open_result = manager.evaluate(
        np.zeros(model.mesh.dof_manager.total_dofs),
    )
    np.testing.assert_array_equal(open_force, 0.0)
    assert open_result.integration_points == 0
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (4, 5, 6):
        u[model.mesh.dof_manager.get_node_dofs(node)[1]] = -0.03
    force, tangent, result = manager.evaluate(u)
    assert result.minimum_gap == pytest.approx(-0.01, abs=1e-10)
    assert result.contact_energy == pytest.approx(0.5, rel=1e-9)
    np.testing.assert_allclose(result.beam_a_force, (0.0, -100.0, 0.0),
                               atol=1e-8)
    np.testing.assert_allclose(result.beam_b_force,
                               -np.asarray(result.beam_a_force), atol=1e-8)
    assert np.linalg.norm(np.asarray(result.beam_a_moment)
                          + np.asarray(result.beam_b_moment)) < 1e-8
    direction = np.zeros_like(u)
    direction[model.mesh.dof_manager.get_node_dofs(5)[1]] = 1.0
    h = 1e-7
    numerical = (manager.evaluate(u + h*direction, tangent=False)[0]
                 - manager.evaluate(u - h*direction, tangent=False)[0]) / (2*h)
    np.testing.assert_allclose(numerical, tangent @ direction,
                               rtol=1e-5, atol=1e-4)
    exchanged = replace(
        definition, surfaces=definition.surfaces[::-1],
        interactions=(replace(definition.interactions[0],
                              surface="b", target="a"),),
    )
    exchanged_force, _stiffness, exchanged_result = B3PairManager(
        model, exchanged, settings,
    ).evaluate(u)
    np.testing.assert_allclose(exchanged_force, force, rtol=0, atol=1e-8)
    assert exchanged_result.contact_energy == pytest.approx(result.contact_energy)
    through = np.zeros_like(u)
    for node in (4, 5, 6):
        through[model.mesh.dof_manager.get_node_dofs(node)[1]] = -0.24
    assert manager._certificate(through).conservative_lower_gap > 0.0
    assert not manager.swept_increment_admissible(
        np.zeros_like(u), through, 0.0, 1.0,
    )


def test_b3_pair_validation_rejects_missing_gap_controls():
    model, definition, settings = _pair()
    report = validate_general_contact(model, definition, replace(
        settings, beam_pair_gap_accuracy=None,
    ))
    assert not report.valid
    assert any("beam_pair_gap_accuracy" in issue for issue in report.issues)


def test_b3_pair_static_displacement_control_and_restart():
    model, definition, settings = _pair()
    model.add_boundary_condition(BoundaryCondition(
        "beam_a_fixed", (1, 2, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_b_guide", (4, 5, 6),
        {name: 0.0 for name in ("ux", "uz", "rx", "ry", "rz")},
    ))
    load = LoadCase("guided_b3_pair")
    load.add_nodal_load(6, forces=(0.0, -1.0, 0.0))
    common = dict(contact=definition, contact_settings=settings,
                  control="displacement",
                  displacement_control=DisplacementControl(
                      node_id=6, dof="uy", target_displacement=-0.03,
                  ), num_steps=4, max_iterations=20,
                  kinematics="von_karman")
    full = solve_static_nonlinear(
        model, load, emit_restart_checkpoint=True, **common,
    )
    assert full.status == "completed", (
        full.failure_reason, full.info.get("general_contact"),
    )
    assert full.info["general_contact"]["active_source_intervals"] > 0
    assert full.info["general_contact"]["gap_tolerance_met"]
    assert full.info["general_contact"]["projection_evaluations"] > 0
    half = solve_static_nonlinear(
        model, load, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-8)
