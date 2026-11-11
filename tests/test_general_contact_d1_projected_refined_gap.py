"""Coupled regression for the opt-in refined projected gap certificate."""

import numpy as np

from anyfesolver.general_contact import static_path
from anyfesolver.general_contact.q4_s3_overlap_derivatives import UnsupportedQ4S3Branch
from scripts.general_contact_d1_coupled_refinement import build_case

from anyfesolver import solve_static_nonlinear, validate_general_contact


def test_refined_projected_patch_press_slide_open_recontact_with_tight_gap():
    model, definition, settings, program = build_case(
        2, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
        four_phase=True,
    )
    capability = validate_general_contact(model, definition, settings)
    assert capability.valid, capability.issues
    assert "REFINED_PROJECTED" in capability.profile
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "completed", result.info.get("contact_failure")
    assert len(result.steps) == 4
    assert result.restart_checkpoint is not None
    history = result.info["force_displacement_history"]
    assert [item["contact"]["active_pressure_dofs"] > 0
            for item in history] == [True, True, False, True]
    assert history[1]["target"][0] == 0.05
    for item in history:
        assert item["normal_multiplier_change"] <= settings.pressure_tolerance
        assert item["contact"]["minimum_gap"] >= -definition.interactions[0].gap_tolerance
        assert item["contact"]["pressure_gram_scaled_rank"] == 6
        assert abs(item["reaction_generalized"][1]
                   + item["contact"]["integration_force"][2]) <= 1e-6

    opened = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True, max_load_factor=0.75,
    )
    assert opened.status == "completed", opened.info.get("contact_failure")
    assert len(opened.steps) == 3
    assert opened.restart_checkpoint is not None
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
        restart_checkpoint=opened.to_restart_checkpoint(),
    )
    assert resumed.status == "completed", resumed.info.get("contact_failure")
    np.testing.assert_allclose(resumed.displacements, result.displacements,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        resumed.info["general_contact"]["normal_multiplier_estimate_pa"],
        result.info["general_contact"]["normal_multiplier_estimate_pa"],
        rtol=1e-6, atol=1e-7,
    )


def test_unresolved_derivative_branch_returns_structured_failure(monkeypatch):
    model, definition, settings, program = build_case(
        2, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )

    def unresolved(*_args, **_kwargs):
        raise UnsupportedQ4S3Branch("positive overlap predicate unresolved")

    monkeypatch.setattr(static_path, "_assemble_nonlinear_system", unresolved)
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    assert result.status == "diverged"
    assert result.failure_reason == "contact_derivative_branch_unresolved"
    assert not result.steps
    assert result.restart_checkpoint is None
    assert result.info["contact_failure"]["augmentation_trace"][0][
        "derivative_branch_failure"] == "positive overlap predicate unresolved"
