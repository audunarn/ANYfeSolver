"""Opt-in Q4/Q4 projected-gap contact development profile."""

from dataclasses import replace

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    DisplacementControl, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_q4_q4_augmented import _augmented_case, _ordinary_case
from test_general_contact_q4_s3_patch_static import _case as q4_s3_case


def _projected_case():
    model, definition, settings = _augmented_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=1e-4),
    ))
    settings = replace(settings, q4_source_gap_certificate="projected_matching")
    return model, definition, settings


def test_projected_q4_pair_has_distinct_profile_and_restart_contract():
    model, definition, settings = _projected_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == (
        "Q4_Q4_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V2"
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.restart_contract()["settings"]["q4_source_gap_certificate"] == (
        "projected_matching"
    )
    old = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="whole_face"),
    )
    assert old.profile != manager.profile
    assert "q4_source_gap_certificate" not in old.restart_contract()["settings"]

    rejected = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="penalty"),
    ))
    assert not validate_general_contact(model, rejected, settings).valid
    mixed_model, mixed_definition, mixed_settings = q4_s3_case()
    mixed_definition = replace(mixed_definition, interactions=(
        replace(mixed_definition.interactions[0], normal_law="augmented"),
    ))
    mixed_settings = replace(
        mixed_settings, pressure_tolerance=0.02,
        q4_source_gap_certificate="projected_matching",
    )
    mixed_report = validate_general_contact(
        mixed_model, mixed_definition, mixed_settings,
    )
    assert mixed_report.valid, mixed_report.issues
    assert mixed_report.profile == (
        "Q4_S3_PATCH_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V2"
    )


def test_projected_q4_pair_certifies_static_and_swept_corner_trial():
    model, definition, settings = _projected_case()
    projected = GeneralS3PatchManager(model, definition, settings)
    old = GeneralS3PatchManager(
        model, definition,
        replace(settings, q4_source_gap_certificate="whole_face"),
    )
    initial = np.zeros(projected.n_dof)
    corner = initial.copy()
    corner[model.mesh.dof_manager.get_node_dofs(1)[2]] = -0.01505
    positions = projected._positions(corner)
    assert projected._gap_lower(positions) > -6e-5
    assert old._gap_lower(positions) < -1e-2
    assert projected.swept_increment_admissible(initial, corner, 0.0, 1.0)
    assert not old.swept_increment_admissible(initial, corner, 0.0, 1.0)


def test_projected_q4_pair_single_press_public_path():
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
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505)),
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
    contact = result.info["general_contact"]
    assert contact["profile"] == (
        "Q4_Q4_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V2"
    )
    assert contact["active_pressure_dofs"] == 4
    assert contact["gap_tolerance_met"]
    assert contact["minimum_gap"] >= -2e-4
    assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4


def test_projected_q4_pair_four_phase_public_path():
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
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
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
    assert all(row["contact"]["gap_tolerance_met"] for row in phases)
    assert all(row["contact"]["minimum_gap"] >= -2e-4 for row in phases)
    assert phases[2]["contact"]["normal_multiplier_estimate_pa"] == [0.0]*4
    assert result.info["profile"] == (
        "Q4_Q4_PATCH_GENERAL_AUGMENTED_PATH_DEV_V2"
    )
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


def test_projected_q4_pair_50_micrometre_gap_candidate():
    model, definition, settings = _projected_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_stiffness=10000.0,
                gap_tolerance=5e-5),
    ))
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
        waypoints=((0.0, 0.0), (0.0, -0.01504)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=40,
    )
    assert result.status == "completed", (
        result.failure_reason, result.info.get("contact_failure"),
    )
    contact = result.info["general_contact"]
    assert contact["active_pressure_dofs"] > 0
    assert contact["gap_tolerance_met"]
    assert contact["minimum_gap"] >= -5e-5


def test_projected_q4_pair_ordinary_force_and_displacement_controls():
    for force_control in (True, False):
        model, definition, settings, load = _ordinary_case(
            force_control=force_control,
        )
        definition = replace(definition, interactions=(
            replace(definition.interactions[0], gap_tolerance=2e-4),
        ))
        settings = replace(settings, q4_source_gap_certificate="projected_matching")
        kwargs = dict(
            contact=definition, contact_settings=settings,
            control="force" if force_control else "displacement",
            kinematics="von_karman", num_steps=2,
            max_iterations=15, emit_restart_checkpoint=True,
        )
        if not force_control:
            kwargs["displacement_control"] = DisplacementControl(
                node_id=1, dof="uz", target_displacement=-0.01505,
            )
        result = solve_static_nonlinear(model, load, **kwargs)
        assert result.status == "completed", (
            force_control, result.failure_reason,
            result.info.get("first_failed_iteration_reason"),
        )
        contact = result.info["general_contact"]
        assert contact["profile"] == (
            "Q4_Q4_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V2"
        )
        assert contact["gap_tolerance_met"]
        assert contact["active_pressure_dofs"] > 0
        assert len(result.to_restart_checkpoint()["path_state"]["normal_multipliers"]) == 4
