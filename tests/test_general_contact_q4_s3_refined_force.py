"""Feasible ordinary force-control development case for refined Q4/S3 contact."""

import os

import numpy as np
import pytest

from anyfesolver import validate_general_contact
from anyfesolver.general_contact.s3_patch_static import (
    REFINED_Q4_S3_AUGMENTED_PROFILE,
)
from examples.general_contact_refined_q4_s3_force import (
    GAP_BUDGET_M, make_case, solve_case,
)


def test_refined_q4_s3_force_case_is_admitted_without_imposed_interference():
    model, definition, settings, load = make_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == REFINED_Q4_S3_AUGMENTED_PROFILE
    assert GAP_BUDGET_M == pytest.approx(1.0e-3*0.05)
    assert definition.interactions[0].gap_tolerance == GAP_BUDGET_M
    assert model.mesh.get_node(3).x > 1.0
    assert model.mesh.get_node(6).x > 1.0
    assert len(model.mesh.elements) == 4
    assert load.name == "press"


@pytest.mark.skipif(os.environ.get("ANYSOLVER_LONG_CONTACT") != "1",
                    reason="opt-in feasible two-Q4/two-S3 ordinary force path")
def test_refined_q4_s3_force_path_meets_gap_wrench_and_restart_gate():
    result = solve_case()
    assert result.status == "completed", result.failure_reason
    contact = result.info["general_contact"]
    assert contact["profile"] == REFINED_Q4_S3_AUGMENTED_PROFILE
    assert contact["active_pressure_dofs"] == 6
    assert contact["gap_tolerance_met"]
    assert contact["minimum_gap"] >= -GAP_BUDGET_M
    assert [row["load_factor"] for row in
            result.info["force_displacement_history"]] == [
                0.25, 0.5, 0.75, 1.0,
            ]
    adaptation = result.info["convergence_adaptation"]
    assert len(adaptation) == 4
    assert all(row["action"] != "shrink_after_slow_convergence"
               for row in adaptation)
    assert any(row["iterations"] >= 9 and
               row["adaptation_iterations"] < 9 for row in adaptation)
    for row in result.info["force_displacement_history"]:
        increment_contact = row["contact"]
        assert increment_contact["profile"] == REFINED_Q4_S3_AUGMENTED_PROFILE
        assert increment_contact["gap_tolerance_met"]
        assert increment_contact["minimum_gap"] >= -GAP_BUDGET_M
        assert increment_contact["active_pressure_dofs"] > 0
        for first, second in (("integration_force", "target_force"),
                              ("integration_moment", "target_moment")):
            np.testing.assert_allclose(
                np.asarray(increment_contact[first])+increment_contact[second],
                0.0, atol=1.0e-6,
            )
    for first, second in (("integration_force", "target_force"),
                          ("integration_moment", "target_moment")):
        np.testing.assert_allclose(
            np.asarray(contact[first])+contact[second], 0.0, atol=1.0e-6,
        )
    assert result.restart_checkpoint["version"] == 2
    assert (result.restart_checkpoint["path_state"]["force_displacement_history"]
            == result.info["force_displacement_history"])
