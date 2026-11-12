"""Development checks for S3 pressure nodes against a bilinear Q4 target."""

from dataclasses import replace

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_s3_q4_patch_static import _case
from test_general_contact_q4_s3_patch_static import _case as _q4_source_case
from test_general_contact_q4_q4_static import _case as _q4_pair_case


def _augmented_case():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented"),
    ))
    settings = replace(settings, pressure_tolerance=0.001,
                       maximum_augmentations=8)
    return model, definition, settings


def _move(model, displacement, nodes, component, value):
    result = displacement.copy()
    for node in nodes:
        result[model.mesh.dof_manager.get_node_dofs(node)[component]] += value
    return result


def test_mixed_projected_zero_gap_sliding_and_pressure_reaction():
    model, definition, settings = _augmented_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    manager = GeneralS3PatchManager(model, definition, settings)
    assert manager.profile == report.profile
    manager.set_normal_multipliers(np.full(4, 10.0))
    displacement = _move(model, np.zeros(manager.n_dof), (1, 2, 3, 4), 2, -0.015)
    displacement = _move(model, displacement, (1, 2, 3, 4), 0, 0.04)
    force, tangent, result = manager.evaluate(displacement)
    assert abs(result.minimum_gap) < 1e-10
    assert result.penalty_energy == 0.0
    assert result.augmented_merit_j > 0.0
    np.testing.assert_allclose(result.numerical_pressure_pa, 10.0, atol=1e-9)
    assert result.weighted_gap_integration_error_m3 < 1e-11
    assert abs(result.integration_force[0]) < 1e-7
    assert abs(result.integration_force[1]) < 1e-7
    np.testing.assert_allclose(np.asarray(result.integration_force)+result.target_force,
                               0.0, atol=1e-7)
    assert tangent.shape == (manager.n_dof, manager.n_dof)
    assert np.linalg.norm(force) > 0.0


def test_mixed_projected_warped_target_balances_wrench_and_tangent():
    model, definition, settings = _augmented_case()
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.set_normal_multipliers(np.full(4, 10.0))
    u = _move(model, np.zeros(manager.n_dof), (1, 2, 3, 4), 2, -0.016)
    u[model.mesh.dof_manager.get_node_dofs(7)[2]] += 0.001
    force, tangent, result = manager.evaluate(u)
    assert result.active_pressure_dofs == 4
    assert result.weighted_gap_integration_error_m3 < 1e-11
    np.testing.assert_allclose(np.asarray(result.integration_force)+result.target_force,
                               0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.integration_moment)+result.target_moment,
                               0.0, atol=1e-6)
    direction = np.zeros_like(u)
    for node in (1, 2, 3, 4):
        direction[model.mesh.dof_manager.get_node_dofs(node)[2]] = 0.5
    step = 1e-4
    plus = manager.evaluate(u+step*direction, tangent=False)[0]
    minus = manager.evaluate(u-step*direction, tangent=False)[0]
    np.testing.assert_allclose(tangent@direction, (plus-minus)/(2*step),
                               rtol=3e-3, atol=5e-4)


def test_q4_pressure_source_augmented_has_distinct_admitted_profile():
    model, definition, settings = _q4_source_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented"),
    ))
    report = validate_general_contact(
        model, definition, replace(settings, pressure_tolerance=0.001),
    )
    assert report.valid, report.issues
    assert report.profile == (
        "Q4_S3_PATCH_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V1"
    )


def test_q4_q4_augmented_profile_has_distinct_admission():
    model, definition, settings = _q4_pair_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented"),
    ))
    report = validate_general_contact(
        model, definition, replace(settings, pressure_tolerance=0.001),
    )
    assert report.valid, report.issues
    assert report.profile == (
        "Q4_Q4_GENERAL_NODAL_PRIMAL_MORTAR_AUGMENTED_STATIC_DEV_V1"
    )


def _path_case():
    model, definition, settings = _augmented_case()
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
        waypoints=((0.0, 0.0), (0.0, -0.0155), (0.025, -0.0155),
                   (0.025, 0.0), (0.025, -0.0155)),
        steps_per_segment=1,
    )
    return model, definition, settings, program


def test_mixed_projected_path_updates_and_restarts():
    model, definition, settings, program = _path_case()
    common = dict(contact=definition, contact_settings=settings,
                  contact_path_program=program, max_iterations=15,
                  emit_restart_checkpoint=True)
    full = solve_static_nonlinear(model, **common)
    assert full.status == "completed", full.failure_reason
    phases = full.info["force_displacement_history"]
    assert phases[0]["augmentation_iterations"] > 1
    assert [phase["contact"]["active_pressure_dofs"] > 0
            for phase in phases] == [True, True, False, True]
    assert phases[2]["contact"]["normal_multiplier_estimate_pa"] == [0.0]*4
    assert full.info["general_contact"]["normal_multiplier_estimate_pa"][0] > 0.0
    half = solve_static_nonlinear(model, max_load_factor=0.75, **common)
    assert half.status == "completed", half.failure_reason
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=half.to_restart_checkpoint(), **common,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        full.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=1e-6, atol=1e-7,
    )
