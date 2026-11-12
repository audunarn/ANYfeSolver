"""Q4 pressure-side patch development through ordinary nonlinear static contact."""

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def _case(origin_shift=(0., 0., 0.)):
    model = FEModel("q4_pressure_side_patch")
    model.add_material("elastic", 1e5, .3, density=1.)
    source = ((.1, .1, .055), (1.1, .1, .055),
              (1.1, 1.1, .055), (.1, 1.1, .055))
    target = ((0., 0., 0.), (1., 0., 0.),
              (1., 1., 0.), (0., 1., 0.))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *(np.asarray(position) + origin_shift))
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=.05,
        reference_normal=(0., 0., 1.),
    ))
    for element_id, nodes in ((2, (5, 6, 8)), (3, (6, 7, 8))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0., 0., 1.),
            reference_surface_offset=.01,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("q4_source", (1,), "bottom"),
                  ContactSurface("s3_target", (2, 3), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "mixed", "q4_source", "s3_target", 1000., .02,
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


def test_reported_patch_wrench_origin_and_translation():
    shift = np.array((2.0, -3.0, 0.7))
    reports = []
    for offset in (np.zeros(3), shift):
        model, definition, settings = _case(offset)
        manager = GeneralS3PatchManager(model, definition, settings)
        u = np.zeros(model.mesh.dof_manager.total_dofs)
        for node in (1, 2, 3, 4):
            u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.02
        _, _, report = manager.evaluate(u, tangent=False)
        assert report.to_dict()["wrench_origin_m"] == [0., 0., 0.]
        from anyfesolver.nonlinear_static import _augmented_contact_increment_summary
        summary = _augmented_contact_increment_summary(report)
        assert summary["wrench_origin_m"] == report.to_dict()["wrench_origin_m"]
        assert summary["integration_moment"] == report.to_dict()["integration_moment"]
        assert summary["target_moment"] == report.to_dict()["target_moment"]
        reports.append(report)
    base, moved = reports
    for body in ("integration", "target"):
        force = np.asarray(getattr(base, body + "_force"))
        np.testing.assert_allclose(getattr(moved, body + "_force"), force,
                                   rtol=1e-6, atol=1e-8)
        expected = getattr(base, body + "_moment") + np.cross(shift, force)
        np.testing.assert_allclose(getattr(moved, body + "_moment"), expected,
                                   rtol=1e-6, atol=1e-8)


def test_q4_pressure_side_patch_maps_four_q1_nodes_and_both_body_wrenches():
    model, definition, settings = _case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == report.profile
    assert manager.law.source_node_ids == (1, 2, 3, 4)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1, 2, 3, 4):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.02
    force, tangent, result = manager.evaluate(u)
    assert result.overlapping_pairs == 2
    assert result.active_pressure_dofs == 4
    assert result.penalty_energy > 0.
    assert result.gap_tolerance_met
    assert result.area_integration_error_m2 <= settings.q4_source_area_tolerance
    assert result.weighted_gap_integration_error_m3 <= settings.q4_source_weighted_gap_tolerance
    assert tangent.shape == (u.size, u.size)
    np.testing.assert_allclose(
        np.array(result.integration_force)+np.array(result.target_force),
        0., atol=1e-6,
    )
    crossing = u.copy()
    for node in (1, 2, 3, 4):
        crossing[model.mesh.dof_manager.get_node_dofs(node)[2]] = -.05
    assert not manager.swept_increment_admissible(
        np.zeros_like(u), crossing, 0., 1.,
    )


def test_q4_pressure_side_patch_four_phase_path_and_restart():
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
    assert all(row["contact"]["wrench_origin_m"] == [0., 0., 0.] for row in phases)
    half = solve_static_nonlinear(
        model, max_load_factor=.5, emit_restart_checkpoint=True, **common,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    assert all(row["contact"]["wrench_origin_m"] == [0., 0., 0.]
               for row in resumed.info["force_displacement_history"])
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)
