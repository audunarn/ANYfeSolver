"""An executable development profile must not become application admission."""

from dataclasses import replace

import pytest

from anyfesolver import report_general_contact_capability, validate_general_contact
from examples.general_contact_refined_q4_s3_tight_path import make_case


def test_refined_shell_profile_reports_exact_scope_without_acceptance():
    model, definition, settings, _program = make_case()
    report = report_general_contact_capability(model, definition, settings)

    assert report.validation == validate_general_contact(model, definition, settings)
    assert report.implemented
    assert report.analysis_mode == "nonlinear_static_force"
    assert report.surface_profiles == (
        ("q4_source", "shell", "bottom"),
        ("s3_target", "shell", "top"),
    )
    assert len(report.formulations) == 2
    assert report.materials
    assert report.friction == "none_declared"
    assert report.lifecycle == "model_validation_only"
    assert not report.tested
    assert not report.independently_accepted
    assert not report.released
    assert not report.admitted_for_application
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(report, independently_accepted=True, released=True)


def test_invalid_element_selection_is_not_reported_as_implemented():
    model, definition, settings, _program = make_case()
    bad_source = replace(definition.surfaces[0], element_ids=(999999,))
    invalid = replace(definition, surfaces=(bad_source, *definition.surfaces[1:]))
    report = report_general_contact_capability(model, invalid, settings)

    assert not report.validation.valid
    assert not report.implemented
    assert "missing_element:999999" in report.formulations
    assert not report.admitted_for_application
