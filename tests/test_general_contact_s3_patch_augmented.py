"""Development checks for projected normal work on a moving S3 patch."""

from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from test_general_contact_s3_patch_static import _case


def _augmented_case():
    model, definition, settings = _case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], normal_law="augmented"),
    ))
    settings = replace(settings, pressure_tolerance=1e-5)
    return model, definition, settings


def _translate(model, u, nodes, axis, distance):
    result = u.copy()
    for node in nodes:
        result[model.mesh.dof_manager.get_node_dofs(node)[axis]] += distance
    return result


def test_projected_patch_zero_gap_sliding_has_no_lateral_work():
    model, definition, settings = _augmented_case()
    assert validate_general_contact(model, definition, settings).valid
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.set_normal_multipliers(np.full(4, 10.0))
    zero = np.zeros(manager.n_dof)
    zero = _translate(model, zero, (1, 2, 3, 4), 2, -0.015)
    for slide in (0.0, 0.07):
        displacement = _translate(model, zero, (1, 2, 3, 4), 0, slide)
        force, tangent, result = manager.evaluate(displacement)
        assert result.active_pressure_dofs == 4
        assert result.penalty_energy == 0.0
        assert result.augmented_merit_j > 0.0
        assert abs(result.minimum_gap) < 1e-10
        np.testing.assert_allclose(result.numerical_pressure_pa, 10.0, atol=1e-9)
        np.testing.assert_allclose(
            np.asarray(result.integration_force)+result.target_force,
            0.0, atol=1e-8,
        )
        assert abs(result.integration_force[0]) < 1e-7
        assert abs(result.integration_force[1]) < 1e-7
        np.testing.assert_allclose(manager.projected_normal_multipliers(displacement),
                                   10.0, atol=1e-9)
        assert tangent.shape == (manager.n_dof, manager.n_dof)
        assert np.linalg.norm(force) > 0.0


def test_projected_patch_opening_and_disappearing_support_release_pressure():
    model, definition, settings = _augmented_case()
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.set_normal_multipliers(np.full(4, 10.0))
    opened = np.zeros(manager.n_dof)
    opened = _translate(model, opened, (1, 2, 3, 4), 2, 0.02)
    np.testing.assert_array_equal(manager.projected_normal_multipliers(opened), 0.0)
    manager.set_normal_multipliers(np.full(4, 10.0))
    apart = _translate(model, opened, (1, 2, 3, 4), 0, 3.0)
    np.testing.assert_array_equal(manager.projected_normal_multipliers(apart), 0.0)
    force, tangent, result = manager.evaluate(apart)
    assert result.active_pressure_dofs == 0
    assert np.linalg.norm(force) == 0.0
    assert tangent.nnz == 0


def test_projected_patch_rejects_missing_pressure_tolerance():
    model, definition, settings = _augmented_case()
    report = validate_general_contact(model, definition,
                                      replace(settings, pressure_tolerance=None))
    assert not report.valid
    assert "pressure_tolerance" in " ".join(report.issues)


def test_projected_patch_weak_open_gap_is_reported_and_cannot_pass():
    model, definition, settings = _augmented_case()
    definition = replace(definition, interactions=(
        replace(definition.interactions[0], gap_tolerance=5e-5),
    ))
    manager = GeneralS3PatchManager(model, definition, settings)
    displacement = np.zeros(manager.n_dof)
    for node, vertical in ((1, -0.0151), (2, -0.014),
                           (3, -0.014), (4, -0.014)):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = vertical
    force, _tangent, result = manager.evaluate(displacement, tangent=False)
    assert min(result.normalized_weighted_gap_m) > 0.0
    assert result.active_pressure_dofs == 0
    np.testing.assert_array_equal(manager.projected_normal_multipliers(displacement),
                                  0.0)
    np.testing.assert_array_equal(force, 0.0)
    assert result.minimum_gap < -5e-5
    assert not result.gap_tolerance_met


def test_projected_patch_normal_tangent_matches_directional_residual():
    model, definition, settings = _augmented_case()
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.set_normal_multipliers(np.full(4, 10.0))
    u = _translate(model, np.zeros(manager.n_dof), (1, 2, 3, 4), 2, -0.016)
    force, tangent, _ = manager.evaluate(u)
    direction = np.zeros_like(u)
    for node in (1, 2, 3, 4):
        direction[model.mesh.dof_manager.get_node_dofs(node)[2]] = 0.5
    perturbation = 1e-4
    plus = manager.evaluate(u+perturbation*direction, tangent=False)[0]
    minus = manager.evaluate(u-perturbation*direction, tangent=False)[0]
    directional = (plus-minus)/(2*perturbation)
    predicted = tangent @ direction
    np.testing.assert_allclose(predicted, directional, rtol=2e-3, atol=5e-4)


def _path_case():
    model, definition, settings = _augmented_case()
    settings = replace(settings, pressure_tolerance=0.001,
                       maximum_augmentations=8)
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
        waypoints=((0.0, 0.0), (0.0, -0.0155), (0.025, -0.0155)),
        steps_per_segment=1,
    )
    return model, definition, settings, program


def test_projected_patch_path_updates_multipliers_and_restarts():
    model, definition, settings, program = _path_case()
    common = dict(contact=definition, contact_settings=settings,
                  contact_path_program=program, max_iterations=15,
                  emit_restart_checkpoint=True)
    full = solve_static_nonlinear(model, **common)
    assert full.status == "completed", full.failure_reason
    assert full.info["force_displacement_history"][0]["augmentation_iterations"] > 1
    assert full.info["general_contact"]["normal_multiplier_estimate_pa"][0] > 0.0
    half = solve_static_nonlinear(model, max_load_factor=0.5, **common)
    assert half.status == "completed", half.failure_reason
    assert half.to_restart_checkpoint()["path_state"]["normal_multipliers"][0] > 0.0
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


def test_projected_patch_exhausted_augmentation_rolls_back():
    model, definition, settings, program = _path_case()
    settings = replace(settings, pressure_tolerance=1e-8,
                       maximum_augmentations=1)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
    )
    assert result.status != "completed"
    assert result.failure_reason == "maximum_augmentations_reached"
    np.testing.assert_array_equal(result.displacements, 0.0)
    np.testing.assert_array_equal(
        result.info["general_contact"]["normal_multiplier_estimate_pa"], 0.0,
    )
