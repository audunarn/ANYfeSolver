"""Bounded many-mode pressure proposal through the ordinary static path."""

from dataclasses import replace
import json

import numpy as np
import pytest

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact import static_path, supported_schur
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3


@pytest.fixture
def projected_only(monkeypatch):
    """Exercise the retained directional fallback when the local solver fails."""
    def unavailable(*_args, **_kwargs):
        raise ValueError("bounded local pressure solve unavailable")

    monkeypatch.setattr(supported_schur, "least_squares", unavailable)


def _case():
    model, definition, settings = _two_q4_two_s3()
    for node in (7, 8, 9, 10):
        model.mesh.get_node(node).x -= 0.5
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=0.005,
    ),))
    settings = replace(
        settings, pressure_tolerance=0.002,
        maximum_augmentations=7,
        augmentation_acceleration="supported_schur_dev",
        augmentation_schur_maximum_modes=8,
        augmentation_schur_maximum_pressure_pa=100.0,
        augmentation_schur_maximum_step_pa=50.0,
        augmentation_schur_difference_step_m=2.0e-6,
    )
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (7, 8, 9, 10),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505)),
        steps_per_segment=1,
    )
    return model, definition, settings, program


def test_supported_schur_settings_are_explicit_and_profile_bounded():
    model, definition, settings, _ = _case()
    assert validate_general_contact(model, definition, settings).valid
    wrong = replace(settings, augmentation_schur_maximum_modes=2)
    assert validate_general_contact(model, definition, wrong).valid
    original = replace(settings, augmentation_acceleration="none",
                       augmentation_schur_maximum_modes=None,
                       augmentation_schur_maximum_pressure_pa=None,
                       augmentation_schur_maximum_step_pa=None,
                       augmentation_schur_difference_step_m=None)
    assert validate_general_contact(model, definition, original).valid
    contract = GeneralS3PatchManager(model, definition, settings).restart_contract()
    changed = GeneralS3PatchManager(
        model, definition,
        replace(settings, augmentation_schur_maximum_step_pa=25.0),
    ).restart_contract()
    assert contract != changed
    assert contract["settings"]["augmentation_schur_maximum_modes"] == 8
    assert contract["settings"]["augmentation_schur_maximum_attempts"] == 1
    assert contract["settings"]["augmentation_schur_maximum_relinearizations"] == 0
    twice = replace(settings, augmentation_schur_maximum_attempts=2)
    assert validate_general_contact(model, definition, twice).valid
    assert GeneralS3PatchManager(model, definition, twice).restart_contract() != contract
    relinearized = replace(settings, augmentation_schur_maximum_relinearizations=1)
    assert validate_general_contact(model, definition, relinearized).valid
    assert GeneralS3PatchManager(model, definition, relinearized).restart_contract() != contract


def test_projection_branch_declines_schur_and_preserves_static_failure():
    model, definition, settings, program = _case()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "diverged"
    assert result.failure_reason == "maximum_augmentations_reached"
    failure = result.info["contact_failure"]
    trace = failure["augmentation_trace"]
    assert len(trace) == settings.maximum_augmentations
    assert trace[1]["schur_proposed"] is False
    assert trace[1]["schur_ineligible"] == "projection_branch_or_nonfinite_response"
    assert trace[1]["pressure_gram_scaled_rank"] == 4
    assert failure["schur_attempts"] == 1
    assert failure["schur_applied"] == failure["schur_rejected"] == 0
    assert trace[-1]["pressure_change"] > settings.pressure_tolerance
    assert result.steps == []
    assert result.restart_checkpoint is None
    np.testing.assert_array_equal(result.displacements, 0.0)


def test_projected_active_set_re_equilibrates_before_committing_step(projected_only):
    model, definition, settings, program = _case()
    program = replace(program, controls=(
        program.controls[0], ContactPathControl.node(6, "uz"),
    ))
    settings = replace(
        settings, general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed"
    assert result.failure_reason is None
    assert len(result.steps) == 1
    assert result.restart_checkpoint is not None
    record = result.info["force_displacement_history"][0]
    trace = record["augmentation_trace"]
    assert trace[1]["schur_proposal_kind"] == "projected_active_set"
    assert trace[1]["schur_predicted_residual_pa"] < 0.75*trace[1]["pressure_change"]
    assert trace[2]["schur_verified"] is True
    assert trace[2]["pressure_change"] < 0.75*trace[1]["pressure_change"]
    assert record["schur_applied"] > 0
    assert trace[-1]["pressure_change"] <= settings.pressure_tolerance
    assert record["contact"]["gap_tolerance_met"]


def test_bounded_projected_schur_updates_accept_node6_fixture(projected_only):
    model, definition, settings, program = _case()
    program = replace(program, controls=(
        program.controls[0], ContactPathControl.node(6, "uz"),
    ))
    settings = replace(
        settings, general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
        augmentation_schur_maximum_attempts=2,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed"
    assert len(result.steps) == 1
    record = result.info["force_displacement_history"][-1]
    assert (1 <= record["schur_attempts"] == record["schur_applied"]
            <= settings.maximum_augmentations)
    assert record["schur_rejected"] == 0
    resets = [row["schur_active_epoch_reset"]
              for row in record["augmentation_trace"]
              if "schur_active_epoch_reset" in row]
    assert len(resets) >= 2
    assert all(row["prior_active_indices"] != row["current_active_indices"]
               for row in resets)
    assert all(row["prior_epoch_attempts"]
               <= settings.augmentation_schur_maximum_attempts
               for row in resets)
    assert record["normal_multiplier_change"] <= settings.pressure_tolerance
    assert record["contact"]["gap_tolerance_met"]
    assert result.restart_checkpoint is not None
    assert result.restart_checkpoint["version"] == 2


def test_original_control_rejects_inaccurate_second_proposal_and_rolls_back(projected_only):
    model, definition, settings, program = _case()
    settings = replace(
        settings, general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
        augmentation_schur_maximum_attempts=2,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "diverged"
    assert result.failure_reason == "maximum_augmentations_reached"
    failure = result.info["contact_failure"]
    assert failure["schur_attempts"] == 3
    assert failure["schur_applied"] == 1
    assert failure["schur_rejected"] == 2
    assert sum("schur_active_epoch_reset" in row
               for row in failure["augmentation_trace"]) == 1
    rejected = next(entry for entry in failure["augmentation_trace"]
                    if entry.get("schur_rejected") == "nonlinear_verification")
    assert rejected["schur_predicted_active_indices"] == [0]
    assert rejected["schur_actual_active_indices"] == [0]
    assert rejected["schur_gap_tolerance_met"]
    assert rejected["schur_actual_to_prior_ratio"] >= 0.75
    json.dumps(failure["augmentation_trace"], allow_nan=False)
    assert result.steps == []
    assert result.restart_checkpoint is None
    np.testing.assert_array_equal(result.displacements, 0.0)


def test_relinearized_original_control_completes_and_restarts(projected_only):
    model, definition, settings, program = _case()
    settings = replace(
        settings, general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
        augmentation_schur_maximum_attempts=2,
        augmentation_schur_maximum_relinearizations=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed"
    assert len(result.steps) == 1
    record = result.info["force_displacement_history"][-1]
    assert record["schur_relinearizations"] == 1
    assert record["schur_applied"] == 2
    assert record["schur_rejected"] == 0
    assert record["normal_multiplier_change"] <= settings.pressure_tolerance
    assert record["contact"]["gap_tolerance_met"]
    trace = record["augmentation_trace"]
    assert any(entry.get("schur_relinearization_proposed") for entry in trace)
    assert not any(entry.get("schur_rejected") for entry in trace)
    json.dumps(trace, allow_nan=False)
    assert result.restart_checkpoint is not None
    assert result.restart_checkpoint["version"] == 2

    fresh_model, fresh_definition, _fresh_settings, fresh_program = _case()
    resumed = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=settings,
        contact_path_program=fresh_program, max_iterations=12,
        emit_restart_checkpoint=True,
        restart_checkpoint=result.to_restart_checkpoint(),
    )
    assert resumed.status == "completed"
    np.testing.assert_array_equal(resumed.displacements, result.displacements)
    assert resumed.to_restart_checkpoint() == result.to_restart_checkpoint()


def test_unexpected_relinearization_failure_restores_algorithmic_origin(monkeypatch, projected_only):
    model, definition, settings, program = _case()
    settings = replace(
        settings, general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        maximum_q4_source_integration_triangles=128,
        augmentation_schur_maximum_attempts=2,
        augmentation_schur_maximum_relinearizations=1,
    )
    original = static_path.refined_patch_supported_schur
    calls = 0

    def fail_third_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected relinearization failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(static_path, "refined_patch_supported_schur", fail_third_call)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert calls == 3
    assert result.status == "diverged"
    assert result.failure_reason == "contact_supported_schur_relinearization_failed"
    failure = result.info["contact_failure"]
    assert any(entry.get("schur_relinearization_failure") ==
               "injected relinearization failure"
               for entry in failure["augmentation_trace"])
    assert result.steps == []
    assert result.restart_checkpoint is None
    np.testing.assert_array_equal(result.displacements, 0.0)
