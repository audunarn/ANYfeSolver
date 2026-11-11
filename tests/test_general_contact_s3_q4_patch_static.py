"""Mixed two-S3/one-Q4 development contact through the static owner."""

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
    model = FEModel("mixed_s3_q4_patch")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    source = ((.1, .1, .065), (1.1, .1, .065),
              (1.1, 1.1, .065), (.1, 1.1, .065))
    target = ((0., 0., -.01), (1., 0., -.01),
              (1., 1., -.01), (0., 1., -.01))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element_id, nodes in ((1, (1, 2, 4)), (2, (2, 3, 4))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0., 0., 1.),
            reference_surface_offset=.01,
        ))
    model.add_element(3, create_shell_element(
        3, (5, 6, 7, 8), "elastic", thickness=.05,
        reference_normal=(0., 0., 1.),
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", (1, 2), "bottom"),
                  ContactSurface("target", (3,), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "mixed", "source", "target", 1000., .02,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
        q4_target_weighted_gap_tolerance=1e-11,
        maximum_q4_target_integration_triangles=500,
    )
    return model, definition, settings


def test_mixed_patch_shared_pressure_work_and_swept_crossing():
    model, definition, settings = _case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == report.profile
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.02
    force, tangent, result = manager.evaluate(u)
    assert result.overlapping_pairs == 2
    assert result.active_pressure_dofs == 4
    assert result.penalty_energy > 0.
    assert result.gap_tolerance_met
    assert tangent.shape == (u.size, u.size)
    np.testing.assert_allclose(np.array(result.integration_force)
                               +np.array(result.target_force), 0., atol=1e-6)
    crossing = u.copy()
    for node in (1, 2, 3, 4):
        crossing[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.05
    assert not manager.swept_increment_admissible(
        np.zeros_like(u), crossing, 0., 1.,
    )


def test_mixed_patch_path_and_restart():
    model, definition, settings = _case()
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0. for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0. for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0., 0.), (0., -.02), (.05, -.02),
                   (.05, 0.), (.05, -.02)),
        steps_per_segment=1,
    )
    common = dict(contact=definition, contact_settings=settings,
                  contact_path_program=program, max_iterations=15)
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
