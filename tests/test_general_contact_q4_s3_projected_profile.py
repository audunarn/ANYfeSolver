"""Opt-in matching-projection gap certificates for one Q4/two-S3 pair."""

from dataclasses import replace

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    DisplacementControl, LoadCase, NonlinearConvergenceSettings,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_s3_patch_static import _case
from test_general_contact_q4_source_augmented import _path_case


def _projected_case():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented",
                gap_tolerance=1e-4),
    ))
    settings = replace(settings, pressure_tolerance=2e-2,
                       q4_source_gap_certificate="projected_matching")
    return model, definition, settings


def test_q4_s3_projected_profile_is_exact_and_restart_bound():
    model, definition, settings = _projected_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == (
        "Q4_S3_PATCH_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V2"
    )
    projected = GeneralS3PatchManager(model, definition, settings)
    assert projected.restart_contract()["settings"]["q4_source_gap_certificate"] == (
        "projected_matching"
    )
    old = GeneralS3PatchManager(
        model, definition, replace(settings, q4_source_gap_certificate="whole_face"),
    )
    assert old.profile != projected.profile
    assert "q4_source_gap_certificate" not in old.restart_contract()["settings"]

    penalty = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="penalty"),
    ))
    assert not validate_general_contact(model, penalty, settings).valid
    reversed_pair = replace(definition, interactions=(
        replace(definition.interactions[0], surface="s3_target",
                target="q4_source"),
    ))
    assert not validate_general_contact(model, reversed_pair, settings).valid


def test_q4_s3_projected_corner_trial_and_wrench():
    model, definition, settings = _projected_case()
    projected = GeneralS3PatchManager(model, definition, settings)
    old = GeneralS3PatchManager(
        model, definition, replace(settings, q4_source_gap_certificate="whole_face"),
    )
    initial = np.zeros(projected.n_dof)
    corner = initial.copy()
    corner[model.mesh.dof_manager.get_node_dofs(1)[2]] = -0.01505
    positions = projected._positions(corner)
    assert projected._gap_lower(positions) > -1e-4
    assert old._gap_lower(positions) < -1e-2
    assert projected.swept_increment_admissible(initial, corner, 0.0, 1.0)
    assert not old.swept_increment_admissible(initial, corner, 0.0, 1.0)
    projected.set_normal_multipliers(np.full(4, 10.0))
    _force, _tangent, contact = projected.evaluate(corner, tangent=False)
    np.testing.assert_allclose(
        np.asarray(contact.integration_force)+contact.target_force,
        0.0, atol=1e-7,
    )
    assert contact.pressure_gram_rank == 4


def test_q4_s3_projected_single_press_public_path():
    model, definition, settings, _program = _path_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=2e-4),
    ))
    settings = replace(settings, q4_source_gap_certificate="projected_matching")
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    contact = result.info["general_contact"]
    assert contact["profile"] == (
        "Q4_S3_PATCH_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V2"
    )
    assert contact["gap_tolerance_met"]
    assert contact["active_pressure_dofs"] > 0
    assert contact["pressure_gram_rank"] == 4
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4


def test_q4_s3_projected_four_phase_restart():
    model, definition, settings, _program = _path_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=2e-4),
    ))
    settings = replace(settings, q4_source_gap_certificate="projected_matching")
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505),
                   (0.05, -0.01505), (0.05, 0.0),
                   (0.05, -0.01505)),
        steps_per_segment=1,
    )
    common = dict(
        contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    result = solve_static_nonlinear(model, **common)
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    phases = result.info["force_displacement_history"]
    assert [row["contact"]["active_pressure_dofs"] > 0
            for row in phases] == [True, True, False, True]
    assert all(row["contact"]["minimum_gap"] >= -2e-4 for row in phases)
    assert phases[2]["contact"]["normal_multiplier_estimate_pa"] == [0.0]*4
    assert result.info["profile"] == "Q4_S3_PATCH_GENERAL_AUGMENTED_PATH_DEV_V2"
    for row in phases:
        contact = row["contact"]
        np.testing.assert_allclose(
            np.asarray(contact["integration_force"])+contact["target_force"],
            0.0, atol=1e-6,
        )
    opened = solve_static_nonlinear(model, max_load_factor=0.75, **common)
    assert opened.status == "completed", opened.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=opened.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=1e-6, atol=1e-7,
    )


def test_q4_s3_projected_ordinary_displacement_control():
    model, definition, settings = _projected_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=2e-4),
    ))
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
    result = solve_static_nonlinear(
        model, load, num_steps=2,
        contact=definition, contact_settings=settings,
        control="displacement", kinematics="von_karman",
        max_iterations=15, emit_restart_checkpoint=True,
        displacement_control=DisplacementControl(
            node_id=1, dof="uz", target_displacement=-0.01505,
        ),
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    contact = result.info["general_contact"]
    assert contact["profile"] == (
        "Q4_S3_PATCH_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V2"
    )
    assert contact["active_pressure_dofs"] > 0
    assert contact["gap_tolerance_met"]
    np.testing.assert_allclose(
        np.asarray(contact["integration_force"])+contact["target_force"],
        0.0, atol=1e-6,
    )
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4


def test_q4_s3_projected_50_micrometre_gap_candidate():
    model, definition, settings, _program = _path_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_stiffness=10000.0,
                gap_tolerance=5e-5),
    ))
    settings = replace(settings, q4_source_gap_certificate="projected_matching",
                       pressure_tolerance=2e-2, maximum_augmentations=16)
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01504)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    contact = result.info["general_contact"]
    assert contact["active_pressure_dofs"] > 0
    assert contact["minimum_gap"] >= -5e-5
    assert contact["gap_tolerance_met"]
    np.testing.assert_allclose(
        np.asarray(contact["integration_force"])+contact["target_force"],
        0.0, atol=1e-6,
    )
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4


def test_q4_s3_projected_50_micrometre_four_phase():
    model, definition, settings, _program = _path_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_stiffness=10000.0,
                gap_tolerance=5e-5),
    ))
    settings = replace(settings, q4_source_gap_certificate="projected_matching",
                       pressure_tolerance=2e-2, maximum_augmentations=16)
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01504),
                   (0.05, -0.01504), (0.05, 0.0),
                   (0.05, -0.01504)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    phases = result.info["force_displacement_history"]
    assert [row["contact"]["active_pressure_dofs"] > 0
            for row in phases] == [True, True, False, True]
    assert all(row["contact"]["minimum_gap"] >= -5e-5 for row in phases)
    assert phases[2]["contact"]["normal_multiplier_estimate_pa"] == [0.0]*4
    for row in phases:
        contact = row["contact"]
        np.testing.assert_allclose(
            np.asarray(contact["integration_force"])+contact["target_force"],
            0.0, atol=1e-6,
        )


def test_q4_s3_projected_ordinary_force_control():
    model, definition, settings = _projected_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=5e-3),
    ))
    settings = replace(settings, maximum_augmentations=16)
    for node in (1, 2, 3, 4):
        model.mesh.get_node(node).z -= 0.015
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_clamp", (1, 4), {"uz": 0.0},
    ))
    load = LoadCase("press")
    for node in (2, 3):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.05))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=4,
        max_iterations=8, min_step_fraction=0.1,
        convergence_settings=NonlinearConvergenceSettings(
            growth_factor=1.0, max_step_factor=1.0,
        ),
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    assert [row["load_factor"] for row in
            result.info["force_displacement_history"]] == [0.25, 0.5, 0.75, 1.0]
    contact = result.info["general_contact"]
    assert contact["active_pressure_dofs"] > 0
    assert contact["gap_tolerance_met"]
    np.testing.assert_allclose(
        np.asarray(contact["integration_force"])+contact["target_force"],
        0.0, atol=1e-6,
    )
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4
