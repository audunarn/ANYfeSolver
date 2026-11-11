"""Mixed classical B2/B3 distributed contact through the static owner."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_curve_pair_static import B3PairManager, MIXED_PROFILE


def _pair():
    model = FEModel("mixed_round_pair")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node_id, point in enumerate((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 0.12, 0.0), (0.5, 0.12, 0.0), (1.0, 0.12, 0.0),
    ), 1):
        model.add_node(node_id, *point)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}
    model.add_element(1, BeamElement(1, (1, 2), "elastic", section))
    model.add_element(2, QuadraticBeamElement(2, (3, 4, 5), "elastic", section))
    definition = ContactDefinition(
        surfaces=(ContactSurface("b2", (1,), beam_radius=0.05),
                  ContactSurface("b3", (2,), beam_radius=0.05)),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "b2", "b3", normal_stiffness=None,
            line_stiffness=1.0e4, gap_tolerance=0.015,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-6,
        maximum_line_subdivisions=16, beam_pair_gap_accuracy=1.0e-5,
        maximum_beam_pair_gap_cells=10000,
        maximum_beam_pair_gap_depth=25,
    )
    return model, definition, settings


def test_mixed_pair_parallel_work_mapping_role_exchange_and_tangent():
    model, definition, settings = _pair()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == MIXED_PROFILE
    manager = B3PairManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (3, 4, 5):
        u[model.mesh.dof_manager.get_node_dofs(node)[1]] = -0.03
    force, tangent, result = manager.evaluate(u)
    assert result.contact_energy == pytest.approx(0.5, rel=1e-9)
    assert result.minimum_gap == pytest.approx(-0.01, abs=1e-10)
    np.testing.assert_allclose(result.beam_a_force, (0.0, -100.0, 0.0), atol=1e-8)
    np.testing.assert_allclose(result.beam_b_force, (0.0, 100.0, 0.0), atol=1e-8)
    assert np.linalg.norm(np.asarray(result.beam_a_moment)
                          + np.asarray(result.beam_b_moment)) < 1e-8
    b2_y = [model.mesh.dof_manager.get_node_dofs(node)[1] for node in (1, 2)]
    np.testing.assert_allclose(force[b2_y], (50.0, 50.0), atol=1e-8)
    direction = np.zeros_like(u)
    direction[b2_y[0]] = 1.0
    direction[model.mesh.dof_manager.get_node_dofs(4)[1]] = -0.3
    step = 1e-7
    numerical = (manager.evaluate(u + step*direction, tangent=False)[0]
                 - manager.evaluate(u - step*direction, tangent=False)[0]) / (2*step)
    np.testing.assert_allclose(numerical, tangent @ direction,
                               rtol=1e-5, atol=1e-4)
    exchanged = replace(
        definition, surfaces=definition.surfaces[::-1],
        interactions=(replace(definition.interactions[0],
                              surface="b3", target="b2"),),
    )
    exchanged_force, _, exchanged_result = B3PairManager(
        model, exchanged, settings,
    ).evaluate(u)
    np.testing.assert_allclose(exchanged_force, force, atol=1e-8, rtol=0)
    assert exchanged_result.contact_energy == pytest.approx(result.contact_energy)


def test_mixed_pair_displacement_path_and_restart():
    model, definition, settings = _pair()
    model.add_boundary_condition(BoundaryCondition(
        "b2_fixed", (1, 2),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "b3_guide", (3, 4, 5),
        {name: 0.0 for name in ("ux", "uz", "rx", "ry", "rz")},
    ))
    load = LoadCase("mixed_pair_path")
    load.add_nodal_load(5, forces=(0.0, -1.0, 0.0))
    common = dict(contact=definition, contact_settings=settings,
                  control="displacement",
                  displacement_control=DisplacementControl(
                      node_id=5, dof="uy", target_displacement=-0.03,
                  ), num_steps=4, max_iterations=20,
                  kinematics="von_karman")
    full = solve_static_nonlinear(model, load, emit_restart_checkpoint=True, **common)
    assert full.status == "completed", full.failure_reason
    assert full.info["general_contact"]["profile"] == MIXED_PROFILE
    assert full.info["general_contact"]["active_source_intervals"] > 0
    assert full.info["general_contact"]["gap_tolerance_met"]
    half = solve_static_nonlinear(
        model, load, max_load_factor=0.5, emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-8)


def test_mixed_pair_curved_target_radius_scaled_gap_path():
    model, definition, settings = _pair()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=5.0e-5,
    ),))
    model.add_boundary_condition(BoundaryCondition(
        "b2_fixed", (1, 2),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "b3_guide", (3, 4, 5),
        {name: 0.0 for name in ("ux", "uz", "rx", "ry", "rz")},
    ))
    load = LoadCase("tight_mixed_pair_path")
    load.add_nodal_load(5, forces=(0.0, -1.0, 0.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=5, dof="uy", target_displacement=-0.02004,
        ), num_steps=4, max_iterations=20, kinematics="von_karman",
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.failure_reason
    summary = result.info["general_contact"]
    assert summary["profile"] == MIXED_PROFILE
    assert summary["active_source_intervals"] > 0
    assert summary["gap_tolerance_met"]
    assert -5.0e-5 <= summary["independent_gap_lower_bound"] < 0.0
    np.testing.assert_allclose(summary["beam_a_force"],
                               -np.asarray(summary["beam_b_force"]), atol=1e-10)
    assert result.restart_checkpoint["version"] == 2
