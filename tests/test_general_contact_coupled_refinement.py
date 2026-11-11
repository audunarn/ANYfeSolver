"""Development regression for a coupled partial-contact pressure refinement."""

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_coupled_refinement import build_case


def test_four_q4_partial_contact_converges_with_bounded_branch_and_restarts():
    model, definition, settings, program = build_case(4)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed"
    assert len(result.steps) == 1
    record = result.info["force_displacement_history"][-1]
    assert record["normal_multiplier_change"] <= settings.pressure_tolerance
    assert record["contact"]["gap_tolerance_met"]
    assert record["contact"]["pressure_gram_scaled_rank"] == 6
    assert record["schur_applied"] == 2
    assert any(item.get("schur_proposal_kind") == "bounded_semismooth_active_set"
               for item in record["augmentation_trace"])
    np.testing.assert_allclose(
        np.asarray(record["contact"]["integration_force"])
        + np.asarray(record["contact"]["target_force"]),
        0.0, atol=1.0e-6,
    )
    assert result.restart_checkpoint is not None
    assert result.restart_checkpoint["version"] == 2

    fresh_model, fresh_definition, _fresh_settings, fresh_program = build_case(4)
    resumed = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=settings,
        contact_path_program=fresh_program, max_iterations=12,
        emit_restart_checkpoint=True,
        restart_checkpoint=result.to_restart_checkpoint(),
    )
    assert resumed.status == "completed"
    np.testing.assert_array_equal(resumed.displacements, result.displacements)
    assert resumed.to_restart_checkpoint() == result.to_restart_checkpoint()


def test_sixteen_q4_partial_contact_chooses_verified_bounded_branch():
    model, definition, settings, program = build_case(16)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed"
    assert len(result.steps) == 1
    record = result.info["force_displacement_history"][-1]
    assert record["normal_multiplier_change"] <= settings.pressure_tolerance
    assert record["contact"]["gap_tolerance_met"]
    assert record["contact"]["pressure_gram_scaled_rank"] == 16
    assert record["schur_applied"] == 2
    assert record["schur_rejected"] == 0
    assert any(item.get("schur_proposal_kind") == "bounded_semismooth_active_set"
               for item in record["augmentation_trace"])
    np.testing.assert_allclose(
        np.asarray(record["contact"]["integration_force"])
        + np.asarray(record["contact"]["target_force"]),
        0.0, atol=1.0e-6,
    )
    assert result.restart_checkpoint is not None
    assert result.restart_checkpoint["version"] == 2
