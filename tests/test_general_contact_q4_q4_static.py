"""Explicit Q4/Q4 mortar development profile through the nonlinear static owner."""

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def _case():
    model = FEModel("q4_q4_general_contact")
    model.add_material("elastic", 1e5, .3, density=1.)
    source = ((.1, .1, .055), (1.1, .1, .055),
              (1.1, 1.1, .055), (.1, 1.1, .055))
    target = ((0., 0., -.01), (1., 0., -.01),
              (1., 1., -.01), (0., 1., -.01))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element_id, nodes in ((1, (1, 2, 3, 4)),
                              (2, (5, 6, 7, 8))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=.05,
            reference_normal=(0., 0., 1.),
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", (1,), "bottom"),
                  ContactSurface("target", (2,), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "q4_pair", "source", "target", 1000., .02,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
    )
    return model, definition, settings


def test_q4_q4_direct_static_work_and_crossing():
    model, definition, settings = _case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == report.profile
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.02
    force, tangent, result = manager.evaluate(u)
    assert result.overlapping_pairs == 1
    assert result.active_pressure_dofs == 4
    assert result.penalty_energy > 0.
    assert result.gap_tolerance_met
    assert tangent.shape == (u.size, u.size)
    np.testing.assert_allclose(
        np.array(result.integration_force)+np.array(result.target_force),
        0., atol=1e-6,
    )
    warped = u.copy()
    warped[model.mesh.dof_manager.get_node_dofs(7)[2]] += .001
    _, _, warped_result = manager.evaluate(warped, tangent=False)
    assert warped_result.penalty_energy > 0.
    assert warped_result.gap_tolerance_met
    crossing = u.copy()
    for node in (1, 2, 3, 4):
        crossing[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.05
    assert not manager.swept_increment_admissible(
        np.zeros_like(u), crossing, 0., 1.,
    )


def test_q4_q4_path_and_midpoint_restart():
    model, definition, settings = _case()
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0. for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0. for name in ("uy", "rx", "ry", "rz")},
    ))
    path = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0., 0.), (0., -.02), (.05, -.02),
                   (.05, 0.), (.05, -.02)),
        steps_per_segment=1,
    )
    common = dict(contact=definition, contact_settings=settings,
                  contact_path_program=path, max_iterations=15)
    full = solve_static_nonlinear(
        model, emit_restart_checkpoint=True, **common,
    )
    assert full.status == "completed", full.failure_reason
    phases = full.info["force_displacement_history"]
    assert [row["contact"]["active_pressure_dofs"] > 0 for row in phases] == [
        True, True, False, True,
    ]
    assert all(row["contact"]["gap_tolerance_met"] for row in phases)
    half = solve_static_nonlinear(
        model, max_load_factor=.5, emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)
