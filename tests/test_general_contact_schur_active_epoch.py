"""A changed active pressure space receives a fresh bounded Schur epoch."""

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_coupled_refinement import build_case


def test_tight_press_reuses_schur_budget_after_active_space_changes():
    model, definition, settings, program = build_case(
        2, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
        pressure_tolerance=1.0e-5,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed"
    assert len(result.steps) == 1
    record = result.info["force_displacement_history"][0]
    assert record["augmentation_iterations"] <= settings.maximum_augmentations
    assert (1 <= record["schur_attempts"] == record["schur_applied"]
            <= settings.maximum_augmentations)
    assert record["normal_multiplier_change"] <= settings.pressure_tolerance
    reset = [row["schur_active_epoch_reset"]
             for row in record["augmentation_trace"]
             if "schur_active_epoch_reset" in row]
    assert any(len(row["prior_active_indices"]) == 6
               and len(row["current_active_indices"]) == 3
               for row in reset)
    assert all(row["prior_epoch_attempts"]
               <= settings.augmentation_schur_maximum_attempts
               for row in reset)
    contact = record["contact"]
    assert contact["active_pressure_dofs"] == 3
    assert contact["gap_tolerance_met"]
    np.testing.assert_allclose(
        record["reaction_generalized"][1]
        + contact["integration_force"][2],
        0.0, atol=1.0e-6,
    )
    assert result.restart_checkpoint is not None
