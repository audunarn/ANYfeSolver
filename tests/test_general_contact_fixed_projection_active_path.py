"""Source-controlled active skew path for the restricted graph profile."""

import os

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear, validate_general_contact
from anyfesolver.general_contact.s3_patch_static import (
    REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE,
)
from examples.general_contact_fixed_projection_skew import (
    GAP_BUDGET_M, make_case, solve_case,
)


def test_fixed_projection_skew_case_has_explicit_restricted_settings():
    model, definition, settings, program = make_case()
    report = validate_general_contact(model, definition, settings)
    assert report.valid, report.issues
    assert report.profile == REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE
    assert len(model.mesh.elements) == 26
    assert len(definition.surfaces[0].element_ids) == 24
    assert len(program.controls) == 2 and program.total_steps == 2
    assert definition.interactions[0].gap_tolerance == GAP_BUDGET_M
    assert settings.q4_source_gap_certificate == "fixed_projection_graph"
    assert settings.general_s3_difference_step == 2.e-6


@pytest.mark.skipif(os.environ.get("ANYSOLVER_LONG_CONTACT") != "1",
                    reason="opt-in 24-Q4 active skew equilibrium and restart")
def test_fixed_projection_skew_active_path_meets_local_gates_and_reloads():
    result = solve_case()
    assert result.status == "completed", result.failure_reason
    assert len(result.steps) == 2
    contact = result.info["general_contact"]
    assert contact["profile"] == REFINED_Q4_S3_GRAPH_AUGMENTED_PROFILE
    assert contact["active_pressure_dofs"] == 1
    assert contact["pressure_gram_rank"] == 24
    assert contact["minimum_gap"] >= -GAP_BUDGET_M
    assert contact["gap_tolerance_met"]
    assert contact["area_integration_error_m2"] <= 1.e-10
    assert contact["weighted_gap_integration_error_m3"] <= 1.e-11
    history = result.info["force_displacement_history"]
    assert history[-1]["augmentation_iterations"] <= 17
    assert history[-1]["normal_multiplier_change"] <= .2
    for first, second in (("integration_force", "target_force"),
                          ("integration_moment", "target_moment")):
        source = np.asarray(contact[first])
        target = np.asarray(contact[second])
        assert np.linalg.norm(source+target)/np.linalg.norm(source) <= 1.e-6
    assert result.restart_checkpoint["version"] == 2

    model, definition, settings, program = make_case()
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=25, tolerance=1.e-8,
        restart_checkpoint=result.restart_checkpoint,
        emit_restart_checkpoint=True,
    )
    assert resumed.status == "completed", resumed.failure_reason
    np.testing.assert_array_equal(resumed.displacements, result.displacements)
    assert resumed.restart_checkpoint == result.restart_checkpoint
