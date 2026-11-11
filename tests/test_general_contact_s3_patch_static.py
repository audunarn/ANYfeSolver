"""Bounded nonmatching S3 patch contact through the static owner."""

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, DisplacementControl, LoadCase,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def _case():
    model = FEModel("general_s3_patch")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    source = ((0.1, 0.1, 0.065), (1.1, 0.1, 0.065),
              (1.1, 1.1, 0.065), (0.1, 1.1, 0.065))
    target = ((0., 0., 0.), (1., 0., 0.),
              (1., 1., 0.), (0., 1., 0.))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element_id, nodes in ((1, (1, 2, 4)), (2, (2, 3, 4)),
                              (3, (5, 6, 7)), (4, (5, 7, 8))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", (1, 2), "bottom"),
                  ContactSurface("target", (3, 4), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "patch", "source", "target", 1000.0, 0.02,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
    )
    return model, definition, settings


def test_s3_patch_maps_shared_pressure_and_nonuniform_skin_work():
    model, definition, settings = _case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.law.source_node_ids == (1, 2, 3, 4)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.025
    u[model.mesh.dof_manager.get_node_dofs(2)[0]] = 0.01
    force, tangent, result = manager.evaluate(u)
    assert result.overlapping_pairs >= 2
    assert result.penalty_energy > 0.0
    assert result.active_pressure_dofs == 4
    assert result.source_pressure_node_ids == (1, 2, 3, 4)
    assert tangent.shape == (u.size, u.size)
    assert np.linalg.norm(force) > 0.0
    np.testing.assert_allclose(
        np.array(result.integration_force)+np.array(result.target_force),
        0.0, atol=1e-7,
    )
    assert manager.restart_contract()["profile"] == report.profile


def test_s3_patch_rejects_through_crossing_without_dropping_pairs():
    model, definition, settings = _case()
    manager = GeneralS3PatchManager(model, definition, settings)
    first = np.zeros(model.mesh.dof_manager.total_dofs)
    last = first.copy()
    for node in (1, 2, 3, 4):
        last[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.05
    assert not manager.swept_increment_admissible(first, last, 0.0, 1.0)
    fraction = manager.admissible_step_fraction(first, last)
    assert 0.0 < fraction < 1.0
    assert manager.swept_trial_cutbacks >= 1


def test_s3_patch_internal_clipping_switch_preserves_aggregate_work():
    model, definition, settings = _case()
    manager = GeneralS3PatchManager(model, definition, settings)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -0.025
    source, target = manager._positions(u)
    base = manager.law.evaluate(source, target)
    derivatives = []
    signatures = []
    for h in (2e-5, 1e-5, 5e-6):
        values = []
        pair_signatures = []
        for sign in (-1.0, 1.0):
            trial = source.copy()
            trial[0, 0] += sign*h
            state = manager.law.evaluate(trial, target)
            values.append(state.energy)
            pair_signatures.append(state.overlap_signature)
        derivatives.append((values[1]-values[0])/(2*h))
        signatures.append(tuple(pair_signatures))
    assert signatures[0][0] != signatures[0][1]
    assert signatures[0][0] != base.overlap_signature
    assert max(derivatives)-min(derivatives) < 1e-8


def test_s3_patch_coupled_displacement_and_restart():
    model, definition, settings = _case()
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("drive")
    load.add_nodal_load(1, forces=(0.0, 0.0, -1.0))
    common = dict(contact=definition, contact_settings=settings,
                  kinematics="von_karman", control="displacement",
                  displacement_control=DisplacementControl(
                      node_id=1, dof="uz", target_displacement=-0.02,
                  ), max_iterations=15)
    full = solve_static_nonlinear(
        model, load, num_steps=2, emit_restart_checkpoint=True, **common,
    )
    assert full.status == "completed", full.failure_reason
    assert full.info["general_contact"]["overlapping_pairs"] >= 2
    assert full.info["general_contact"]["active_pressure_dofs"] > 0
    assert full.info["general_contact"]["gap_tolerance_met"]
    half = solve_static_nonlinear(
        model, load, num_steps=1, max_load_factor=0.5,
        emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, num_steps=1,
        restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)


def test_s3_patch_multi_control_press_slide_open_recontact():
    model, definition, settings = _case()
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.02),
                   (0.05, -0.02), (0.05, 0.0), (0.05, -0.02)),
        steps_per_segment=1,
    )
    common = dict(contact=definition, contact_settings=settings,
                  contact_path_program=program, max_iterations=15)
    full = solve_static_nonlinear(
        model, emit_restart_checkpoint=True, **common,
    )
    assert full.status == "completed", full.failure_reason
    phases = full.info["force_displacement_history"]
    assert len(phases) == 4
    assert phases[0]["contact"]["active_pressure_dofs"] > 0
    assert phases[1]["contact"]["active_pressure_dofs"] > 0
    assert phases[2]["contact"]["active_pressure_dofs"] == 0
    assert phases[3]["contact"]["active_pressure_dofs"] > 0
    assert all(row["contact"]["gap_tolerance_met"] for row in phases)
    half = solve_static_nonlinear(
        model, max_load_factor=0.5, emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)
