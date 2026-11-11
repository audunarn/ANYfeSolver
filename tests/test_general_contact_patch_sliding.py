"""Local physical work-map checks for mixed planar shell patches."""

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.patch_sliding import _RestrictedSlidingPatchFEManager
from anyfesolver.general_contact.static_path import (
    ContactPathControl, ContactPathProgram,
)


def _q4_against_two_s3(*, lower_x_shift=0.0):
    model = FEModel("mixed_patch_local")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    lower = tuple((x + lower_x_shift, y) for x, y in (
        (0.2, 0.1), (1.2, 0.1), (1.2, 1.1), (0.2, 1.1),
    ))
    upper = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for node, (x, y) in enumerate(lower, 1):
        model.add_node(node, x, y, 0.0)
    for node, (x, y) in enumerate(upper, 5):
        model.add_node(node, x, y, 0.04)
    for element, nodes in ((1, (1, 2, 3)), (2, (1, 3, 4))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    model.add_element(3, create_shell_element(
        3, (5, 6, 7, 8), "elastic", thickness=0.05,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("upper", (3,), side="bottom"),
                  ContactSurface("lower", (1, 2), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "mixed", "upper", "lower", 1.0e6, 6.0e-6,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    return model, definition, settings


def test_q4_against_two_s3_faces_uses_physical_offsets_and_shared_pressure():
    model, definition, settings = _q4_against_two_s3()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    assert len(manager.patch.source_node_ids) == 4
    assert len(manager.patch.target_node_ids) == 4
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (5, 6, 7, 8):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -1.0e-6
    force, _tangent, summary = manager.evaluate(displacement, tangent=False)
    assert summary.overlapping_pairs == 2
    assert summary.current_overlap_area == pytest.approx(0.72)
    assert summary.integration_force[2] == pytest.approx(0.72, rel=1e-7)
    assert summary.target_force[2] == pytest.approx(-0.72, rel=1e-7)
    assert summary.gap_tolerance_met
    assert summary.minimum_positive_support_m2 > 0.0
    assert summary.maximum_positive_support_m2 >= summary.minimum_positive_support_m2
    assert summary.pressure_gradient_condition >= 1.0
    assert summary.minimum_weighted_gap_m == pytest.approx(-1.0e-6, abs=1.0e-10)
    assert summary.maximum_algorithmic_pressure_Pa == pytest.approx(1.0)
    assert summary.weighted_to_physical_gap_span_m == pytest.approx(0.0, abs=1.0e-10)
    assert sum(force[model.mesh.dof_manager.get_node_dofs(node)[2]]
               for node in (5, 6, 7, 8)) == pytest.approx(-0.72)
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press")
    for node in (5, 6, 7, 8):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.1))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor, result.info.get("general_contact"),
    )
    coupled = result.info["general_contact"]
    assert coupled["gap_tolerance_met"]
    assert coupled["integration_force"][2] == pytest.approx(0.4, rel=1e-5)


def test_pointwise_penalty_patch_is_explicit_and_solves_constant_closure():
    model, definition, _settings = _q4_against_two_s3()
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
        normal_discretization="pointwise_penalty",
        pointwise_integration_subdivisions=1,
        maximum_pointwise_triangles=1000,
    )
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert "POINTWISE_PENALTY" in report.profile
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (5, 6, 7, 8):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -1.0e-6
    force, tangent, summary = manager.evaluate(displacement)
    assert tangent is not None
    assert summary.integration_force[2] == pytest.approx(0.72, rel=1.0e-7)
    assert summary.target_force[2] == pytest.approx(-0.72, rel=1.0e-7)
    assert summary.pressure_rank is None
    assert summary.active_pointwise_area_m2 == pytest.approx(0.72)
    assert summary.pointwise_integration_triangles > 0
    assert sum(force[model.mesh.dof_manager.get_node_dofs(node)[2]]
               for node in (5, 6, 7, 8)) == pytest.approx(-0.72)

    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press")
    for node in (5, 6, 7, 8):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.1))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", result.failure_reason
    assert result.info["general_contact"]["integration_force"][2] == pytest.approx(0.4)
    half = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
        max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(
        resumed.displacements, result.displacements, rtol=1.0e-7, atol=1.0e-9,
    )

    with pytest.raises(ValueError, match="normal_discretization"):
        ContactSolverSettings(normal_discretization="unsupported")


@pytest.mark.parametrize("discretization", ("primal_mortar", "pointwise_penalty"))
def test_patch_lateral_force_uses_physical_shell_offset_couple(discretization):
    model, definition, _settings = _q4_against_two_s3()
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
        normal_discretization=discretization,
        pointwise_integration_subdivisions=1,
    )
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (5, 6, 7, 8):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        displacement[dofs[0]] = 0.02
        displacement[dofs[2]] = -1.0e-6
    _force, _tangent, summary = manager.evaluate(displacement, tangent=False)
    net_force = np.asarray(summary.integration_force) + summary.target_force
    net_moment = np.asarray(summary.integration_moment) + summary.target_moment
    np.testing.assert_allclose(net_force, np.zeros(3), atol=1.0e-10)
    np.testing.assert_allclose(net_moment[:2], np.zeros(2), atol=1.0e-9)
    np.testing.assert_allclose(net_moment, np.zeros(3), atol=1.0e-9)


@pytest.mark.parametrize("discretization", ("primal_mortar", "pointwise_penalty"))
def test_patch_local_force_map_has_consistent_admitted_tangent(discretization):
    model, definition, _settings = _q4_against_two_s3()
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
        normal_discretization=discretization,
        pointwise_integration_subdivisions=1,
    )
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    direction = np.zeros_like(displacement)
    for node, normal_direction in zip((5, 6, 7, 8), (0.1, -0.1, 0.2, -0.2)):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        displacement[dofs[0]] = 0.02
        displacement[dofs[2]] = -1.0e-6
        direction[dofs[0]] = 0.2
        direction[dofs[2]] = normal_direction
    force, tangent, _summary = manager.evaluate(displacement)
    assert tangent is not None
    step = 1.0e-8
    plus = manager.evaluate(displacement + step * direction, tangent=False)[0]
    minus = manager.evaluate(displacement - step * direction, tangent=False)[0]
    numerical = (plus - minus) / (2 * step)
    np.testing.assert_allclose(numerical, tangent @ direction,
                               rtol=1.0e-5, atol=1.0e-5)
    assert np.linalg.norm(force) > 0.0


def test_q4_against_two_s3_faces_slides_and_restarts():
    model, definition, settings = _q4_against_two_s3()
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_lateral_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    master = int(model.mesh.dof_manager.get_node_dofs(5)[0])
    for node in (6, 7, 8):
        slave = int(model.mesh.dof_manager.get_node_dofs(node)[0])
        model.add_constraint_equation(
            terms=((slave, 1.0), (master, -1.0)),
            source_id=f"upper_uniform_x_{node}",
        )
    press = LoadCase("press")
    for node in (5, 6, 7, 8):
        press.add_nodal_load(node, forces=(0.0, 0.0, -0.1))
    drive = LoadCase("drive")
    drive.add_nodal_load(5, forces=(1.0, 0.0, 0.0))
    options = dict(
        constant_load_case=press, contact=definition,
        contact_settings=settings, kinematics="von_karman",
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=5, dof="ux", target_displacement=0.1,
        ),
        num_steps=2, max_iterations=12,
    )
    result = solve_static_nonlinear(model, drive, **options)
    assert result.status == "completed", (
        result.failure_reason, result.load_factor, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["current_overlap_area"] == pytest.approx(0.81)
    assert summary["overlapping_pairs"] == 2
    assert summary["gap_tolerance_met"]
    assert summary["integration_force"][2] == pytest.approx(0.4, rel=1e-5)
    half = solve_static_nonlinear(
        model, drive, **options,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, drive, **options,
        max_load_factor=1.0,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(
        resumed.displacements, result.displacements, rtol=1e-7, atol=1e-9,
    )


def test_q4_s3_patch_opens_and_recontacts_from_static_checkpoints():
    model, definition, settings = _q4_against_two_s3()
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_normal_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    press = LoadCase("press")
    for node in (5, 6, 7, 8):
        press.add_nodal_load(node, forces=(0.0, 0.0, -0.1))

    def stage(target, checkpoint=None):
        return solve_static_nonlinear(
            model, press, contact=definition, contact_settings=settings,
            kinematics="von_karman", control="displacement",
            displacement_control=DisplacementControl(
                node_id=5, dof="uz", target_displacement=target,
            ),
            num_steps=2, max_iterations=12,
            restart_checkpoint=checkpoint, emit_restart_checkpoint=True,
        )

    closed = stage(-1.0e-6)
    assert closed.status == "completed", closed.failure_reason
    closed_summary = closed.info["general_contact"]
    assert closed_summary["integration_force"][2] > 0.0
    assert closed_summary["gap_tolerance_met"]
    opened = stage(1.0e-6, closed.to_restart_checkpoint())
    assert opened.status == "completed", (
        opened.failure_reason, opened.info.get("general_contact"),
    )
    open_summary = opened.info["general_contact"]
    assert open_summary["active_pressure_dofs"] == 0
    assert open_summary["conservative_lower_gap"] > 0.0
    reclosed = stage(-1.0e-6, opened.to_restart_checkpoint())
    assert reclosed.status == "completed", (
        reclosed.failure_reason, reclosed.info.get("general_contact"),
    )
    reclosed_summary = reclosed.info["general_contact"]
    assert reclosed_summary["integration_force"][2] == pytest.approx(
        closed_summary["integration_force"][2], rel=1e-5,
    )
    assert reclosed_summary["gap_tolerance_met"]
    assert len(reclosed.steps) == 6


def test_q4_s3_patch_combines_press_slide_open_and_recontact():
    model, definition, settings = _q4_against_two_s3()
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    master = int(model.mesh.dof_manager.get_node_dofs(5)[0])
    for node in (6, 7, 8):
        slave = int(model.mesh.dof_manager.get_node_dofs(node)[0])
        model.add_constraint_equation(
            terms=((slave, 1.0), (master, -1.0)),
            source_id=f"upper_uniform_x_{node}",
        )
    program = ContactPathProgram(
        controls=(
            ContactPathControl.node(5, "ux"),
            ContactPathControl(tuple((node, "uz", 0.25)
                                     for node in (5, 6, 7, 8))),
        ),
        waypoints=((0.0, 0.0), (0.0, -5.0e-7),
                   (0.1, -5.0e-7), (0.1, 1.0e-6),
                   (0.1, -5.0e-7)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program,
        max_iterations=15, emit_restart_checkpoint=True,
        record_increment_snapshots=True,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("general_contact"),
    )
    assert len(result.steps) == 4
    assert [snapshot.step_index for snapshot in result.snapshots] == [1, 2, 3, 4]
    np.testing.assert_array_equal(result.snapshots[-1].displacements,
                                  result.displacements)
    assert result.info["general_contact"]["gap_tolerance_met"]
    assert result.info["general_contact"]["integration_force"][2] > 0.0
    history = result.info["force_displacement_history"]
    assert history[0]["contact"]["integration_force"][2] > 0.0
    assert history[1]["contact"]["current_overlap_area"] == pytest.approx(0.81)
    assert history[2]["contact"]["active_pressure_dofs"] == 0
    assert history[2]["contact"]["conservative_lower_gap"] > 0.0
    assert history[2]["contact"]["maximum_algorithmic_pressure_Pa"] == 0.0
    assert history[3]["contact"]["integration_force"][2] > 0.0
    half = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program,
        max_load_factor=0.5, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program,
        max_iterations=15, restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-7, atol=1e-9)
    from dataclasses import replace

    altered = replace(
        program,
        waypoints=(*program.waypoints[:-1], (0.1, -6.0e-7)),
    )
    with pytest.raises(ValueError, match="analysis contract is incompatible"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=altered,
            max_iterations=15,
            restart_checkpoint=half.to_restart_checkpoint(),
        )


def test_q4_s3_patch_activates_from_initially_disjoint_overlap():
    model, definition, settings = _q4_against_two_s3(lower_x_shift=1.0)
    guided = validate_general_contact(model, definition, ContactSolverSettings())
    assert not guided.valid
    assert "no projected reference overlap" in "; ".join(guided.issues)
    assert validate_general_contact(model, definition, settings).valid
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    master = int(model.mesh.dof_manager.get_node_dofs(5)[0])
    for node in (6, 7, 8):
        slave = int(model.mesh.dof_manager.get_node_dofs(node)[0])
        model.add_constraint_equation(
            terms=((slave, 1.0), (master, -1.0)),
            source_id=f"first_approach_upper_x_{node}",
        )
    program = ContactPathProgram(
        controls=(
            ContactPathControl.node(5, "ux"),
            ContactPathControl(tuple((node, "uz", 0.25)
                                     for node in (5, 6, 7, 8))),
        ),
        waypoints=((0.0, 0.0), (0.5, -5.0e-7)),
        steps_per_segment=1,
    )
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    initial = manager.evaluate(
        np.zeros(model.mesh.dof_manager.total_dofs), tangent=False,
    )[2]
    assert initial.current_overlap_area == 0.0
    assert initial.minimum_positive_support_m2 is None
    assert initial.pressure_gradient_condition is None
    assert initial.minimum_weighted_gap_m is None
    assert initial.maximum_algorithmic_pressure_Pa == 0.0
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("general_contact"),
    )
    contact = result.info["general_contact"]
    assert contact["current_overlap_area"] > 0.0
    assert contact["active_pressure_dofs"] > 0
    assert contact["gap_tolerance_met"]


def test_contact_path_rejects_unadmitted_public_controls():
    model, definition, settings = _q4_against_two_s3()
    program = ContactPathProgram(
        controls=(ContactPathControl.node(5, "ux"),
                  ContactPathControl.node(5, "uz")),
        waypoints=((0.0, 0.0), (0.0, -1.0e-7)),
        steps_per_segment=1,
    )
    with pytest.raises(ValueError, match="admitted settings"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, num_steps=2,
        )
    with pytest.raises(ValueError, match="admitted settings"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, kinematics="corotational",
        )


def test_augmented_law_requires_path_owner_and_pressure_tolerance():
    from dataclasses import replace

    model, definition, settings = _q4_against_two_s3()
    definition = replace(
        definition,
        interactions=(replace(definition.interactions[0], normal_law="augmented"),),
    )
    assert not validate_general_contact(model, definition, settings).valid
    settings = replace(settings, pressure_tolerance=1.0e-5)
    report = validate_general_contact(model, definition, settings)
    assert report.valid
    assert "AUGMENTED" in report.profile
    with pytest.raises(ValueError, match="requires the contact path analysis owner"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            num_steps=1,
        )


def _augmented_patch_path(*, averaged_normal: bool):
    from dataclasses import replace

    model, definition, settings = _q4_against_two_s3()
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    master = int(model.mesh.dof_manager.get_node_dofs(5)[0])
    for node in (6, 7, 8):
        slave = int(model.mesh.dof_manager.get_node_dofs(node)[0])
        model.add_constraint_equation(
            terms=((slave, 1.0), (master, -1.0)),
            source_id=f"augmented_upper_x_{node}",
        )
    definition = replace(
        definition,
        interactions=(replace(definition.interactions[0], normal_law="augmented"),),
    )
    settings = replace(settings, pressure_tolerance=1.0e-5,
                       maximum_augmentations=20)
    program = ContactPathProgram(
        controls=(
            ContactPathControl.node(5, "ux"),
            (ContactPathControl(tuple((node, "uz", 0.25)
                                      for node in (5, 6, 7, 8)))
             if averaged_normal else ContactPathControl.node(5, "uz")),
        ),
        waypoints=((0.0, 0.0), (0.0, -5.0e-7)),
        steps_per_segment=1 if averaged_normal else 2,
    )
    return model, definition, settings, program


def test_q4_s3_patch_augmented_press_uses_fixed_multiplier_equilibrium():
    model, definition, settings, program = _augmented_patch_path(
        averaged_normal=False,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("force_displacement_history"),
    )
    record = result.info["force_displacement_history"][-1]
    assert max(item["augmentation_iterations"]
               for item in result.info["force_displacement_history"]) >= 2
    assert record["normal_multiplier_change"] <= settings.pressure_tolerance
    assert record["contact"]["normal_law"] == "augmented"
    assert record["contact"]["gap_tolerance_met"]
    assert record["contact"]["active_pressure_dofs"] > 0
    assert record["contact"]["integration_force"][2] > 0.0
    assert record["contact"]["multiplier_norm"] > 0.0
    assert result.restart_checkpoint["version"] == 2
    half = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_load_factor=0.5,
        max_iterations=15, emit_restart_checkpoint=True,
    )
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        restart_checkpoint=half.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1.0e-7, atol=1.0e-9)


def test_q4_s3_augmented_slide_open_recontact_restarts_from_open_state():
    from dataclasses import replace

    model, definition, settings, initial = _augmented_patch_path(
        averaged_normal=False,
    )
    program = replace(
        initial,
        waypoints=((0.0, 0.0), (0.0, -5.0e-7),
                   (0.1, -5.0e-7), (0.1, 1.0e-6),
                   (0.1, -5.0e-7)),
        steps_per_segment=1,
    )
    common = dict(contact=definition, contact_settings=settings,
                  contact_path_program=program, max_iterations=15)
    full = solve_static_nonlinear(
        model, emit_restart_checkpoint=True, **common,
    )
    assert full.status == "completed", full.failure_reason
    assert len(full.steps) == 4
    history = full.info["force_displacement_history"]
    assert [row["contact"]["active_pressure_dofs"] for row in history] == [
        1, 1, 0, 1,
    ]
    assert all(row["normal_multiplier_change"] <= settings.pressure_tolerance
               for row in history)
    for row in history:
        contact = row["contact"]
        assert len(row["reaction_generalized"]) == 2
        assert np.all(np.isfinite(row["reaction_generalized"]))
        assert (contact["integration_force"][2]
                + contact["target_force"][2]) == pytest.approx(0.0, abs=1e-9)
    slid_force = history[1]["contact"]["integration_force"][2]
    assert slid_force > 0.0
    opened = history[2]["contact"]
    assert opened["conservative_lower_gap"] > 0.0
    assert opened["integration_force"][2] == 0.0
    assert opened["multiplier_norm"] == 0.0
    assert history[3]["contact"]["integration_force"][2] == pytest.approx(
        slid_force, rel=1e-4,
    )

    split = solve_static_nonlinear(
        model, max_load_factor=0.75,
        emit_restart_checkpoint=True, **common,
    )
    assert split.status == "completed", split.failure_reason
    assert len(split.steps) == 3
    assert split.info["general_contact"]["multiplier_norm"] == 0.0
    assert split.restart_checkpoint["version"] == 2
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=split.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)
    assert resumed.info["general_contact"]["integration_force"][2] == pytest.approx(
        full.info["general_contact"]["integration_force"][2], rel=1e-5,
    )


def test_q4_s3_augmented_forced_penetration_fails_without_committing():
    model, definition, settings, program = _augmented_patch_path(
        averaged_normal=True,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    assert result.status == "diverged"
    assert result.failure_reason == "maximum_augmentations_reached"
    assert result.steps == []
    assert result.restart_checkpoint is None
    np.testing.assert_allclose(result.displacements, 0.0, atol=0.0)
    assert result.info["general_contact"]["multiplier_norm"] == 0.0


def _two_s3_integration_faces(*, lower_bounds=(0.2, 1.2, 0.1, 1.1)):
    model = FEModel("s3_integration_patch_local")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    x0, x1, y0, y1 = lower_bounds
    lower = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    upper = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for node, (x, y) in enumerate(lower, 1):
        model.add_node(node, x, y, 0.0)
    for node, (x, y) in enumerate(upper, 5):
        model.add_node(node, x, y, 0.06)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
    ))
    for element, nodes in ((2, (5, 6, 7)), (3, (5, 7, 8))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("upper", (2, 3), side="bottom"),
                  ContactSurface("lower", (1,), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "mixed", "upper", "lower", 1.0e6, 6.0e-6,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    return model, definition, settings


def test_two_s3_integration_faces_share_four_material_pressure_nodes():
    model, definition, settings = _two_s3_integration_faces()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    assert len(manager.patch.source_node_ids) == 4
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (5, 6, 7, 8):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -1.0e-6
    _force, _tangent, summary = manager.evaluate(displacement, tangent=False)
    assert summary.overlapping_pairs == 2
    assert summary.pressure_rank == 4
    assert summary.current_overlap_area == pytest.approx(0.72)
    assert summary.integration_force[2] == pytest.approx(0.72, rel=1e-7)
    assert summary.target_force[2] == pytest.approx(-0.72, rel=1e-7)
    assert summary.gap_tolerance_met


@pytest.mark.parametrize("edge", (1.0, 0.8))
def test_two_s3_integration_faces_complete_consistent_uniform_press(edge):
    model, definition, settings = _two_s3_integration_faces(
        lower_bounds=(-0.1, edge if edge < 1.0 else 1.1, -0.1, 1.1),
    )
    weights = (
        edge / 2 - edge**3 / 6,
        edge**3 / 6,
        edge**2 / 2 - edge**3 / 6,
        edge / 2 - edge**2 / 2 + edge**3 / 6,
    )
    assert sum(weights) == pytest.approx(edge)
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    trial = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (5, 6, 7, 8):
        trial[model.mesh.dof_manager.get_node_dofs(node)[2]] = (
            -0.4 / (1.0e6 * edge)
        )
    contact_force, _tangent, local = manager.evaluate(trial, tangent=False)
    assert local.gap_tolerance_met
    for node, weight in zip((5, 6, 7, 8), weights):
        dof = model.mesh.dof_manager.get_node_dofs(node)[2]
        assert -contact_force[dof] == pytest.approx(
            0.4 * weight / edge, rel=1e-6,
        )
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guide", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    # Integrate the two piecewise-linear S3 bases over 0 <= x <= edge,
    # 0 <= y <= 1. This analytical load oracle does not call contact code.
    load = LoadCase("uniform_press")
    for node, weight in zip((5, 6, 7, 8), weights):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.4 * weight / edge))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=1, max_iterations=12,
    )
    assert result.status == "completed", (
        result.failure_reason, result.load_factor, result.info.get("general_contact"),
    )
    summary = result.info["general_contact"]
    assert summary["pressure_rank"] == 4
    assert summary["overlapping_pairs"] == 2
    assert summary["current_overlap_area"] == pytest.approx(edge)
    assert summary["gap_tolerance_met"]
    assert summary["integration_force"][2] == pytest.approx(0.4, rel=1e-5)
    assert summary["target_force"][2] == pytest.approx(-0.4, rel=1e-5)
