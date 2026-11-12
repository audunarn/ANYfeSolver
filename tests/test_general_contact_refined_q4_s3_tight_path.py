"""Two-Q4/two-S3 ordinary static path at the declared physical gap budget."""

import os

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear, validate_general_contact
from examples.general_contact_refined_q4_s3_tight_path import (
    GAP_BUDGET_M, THICKNESS_M, make_case, solve_case,
)


def test_refined_tight_path_declares_physical_gap_and_supported_profile():
    model, definition, settings, program = make_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert GAP_BUDGET_M == pytest.approx(1.0e-3*THICKNESS_M)
    assert definition.interactions[0].gap_tolerance == GAP_BUDGET_M
    assert len(model.mesh.elements) == 4
    assert len(program.waypoints) == 5
    assert program.total_steps == 8


@pytest.mark.skipif(
    os.environ.get("ANY_RUN_C1_REFINED_TIGHT_PATH") != "1",
    reason="opt-in long two-Q4/two-S3 path and restart development route",
)
def test_refined_tight_path_press_slide_open_recontact_and_split_restart():
    full = solve_case()
    history = full.info["force_displacement_history"]
    assert full.restart_checkpoint["version"] == 2
    assert all(row["contact"]["gap_tolerance_met"] for row in history)
    assert min(row["contact"]["minimum_gap"] for row in history) >= -GAP_BUDGET_M
    assert history[5]["contact"]["integration_force"] == [0.0, 0.0, 0.0]
    assert history[7]["contact"]["active_pressure_dofs"] == 6

    model, definition, settings, program = make_case()
    options = dict(contact=definition, contact_settings=settings,
                   contact_path_program=program, max_iterations=18,
                   emit_restart_checkpoint=True)
    prefix = solve_static_nonlinear(model, max_load_factor=0.5, **options)
    assert prefix.status == "completed", prefix.failure_reason
    assert len(prefix.steps) == 4
    assert prefix.info["general_contact"]["active_pressure_dofs"] == 6
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=prefix.to_restart_checkpoint(), **options,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0.0, atol=1.0e-10)
    for field in ("integration_force", "target_force",
                  "integration_moment", "target_moment"):
        np.testing.assert_allclose(
            resumed.info["general_contact"][field],
            full.info["general_contact"][field], rtol=0.0, atol=1.0e-8,
        )
    assert (resumed.to_restart_checkpoint()["path_state"]["normal_multipliers"]
            == full.to_restart_checkpoint()["path_state"]["normal_multipliers"])
    assert resumed.to_restart_checkpoint() == full.to_restart_checkpoint()
