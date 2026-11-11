"""Selector, contract, validation and request-identity checks for the seed.

The default-none selector must keep every legacy serialized shape: the
canonical settings contract, the request identity and the restart contract
all omit it when disabled and bind the enabled algorithm revision when
enabled. Validation refuses every non-D1 route and every non-contact-path
driver before any Newton work.
"""
import json
from dataclasses import asdict, replace

import pytest

from anyfesolver import ContactSolverSettings, validate_general_contact
from anyfesolver.general_contact.core import (
    _INITIAL_CONTACT_SEED_REVISION,
    _contact_settings_contract,
)
from anyfesolver.general_contact.request_identity import (
    identify_general_contact_request,
)

from test_general_contact_admission_globalization_driver_small_patch import (
    _case,
    _enabled,
)


def test_default_selector_is_none_and_legacy_contract_omits_it():
    settings = ContactSolverSettings()
    assert settings.initial_contact_seed == "none"
    contract = _contact_settings_contract(settings)
    assert "initial_contact_seed" not in contract
    # The disabled contract keeps the exact prior shape: it equals the
    # contract of a settings object constructed before the field existed.
    assert contract == _contact_settings_contract(
        ContactSolverSettings(**{name: value for name, value in
                                 asdict(settings).items()}))
    enabled = replace(settings, initial_contact_seed="linearized_kkt_dev",
                      d1_interior_profile=True,
                      d1_interior_coverage_tolerance_m2=1.0e-10,
                      d1_interior_boundary_force_tolerance_n=1.0e-10,
                      q4_source_gap_certificate="overlap_local_dev",
                      general_s3_differentiation="jet2_q4_s3_dev",
                      q4_source_pressure_basis="fine_nodal_q1",
                      enforcement_solver="monolithic_active_set",
                      complementarity_force_scale_n=1.0)
    enabled_contract = _contact_settings_contract(enabled)
    assert enabled_contract["initial_contact_seed"] == "linearized_kkt_dev"


def test_unsupported_selector_is_refused():
    with pytest.raises(ValueError, match="unsupported initial_contact_seed"):
        ContactSolverSettings(initial_contact_seed="projected_freeze_dev")


def test_enabled_selector_requires_the_d1_overlap_local_route():
    base = dict(
        d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1.0e-10,
        d1_interior_boundary_force_tolerance_n=1.0e-10,
        q4_source_gap_certificate="overlap_local_dev",
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_pressure_basis="fine_nodal_q1",
        enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=1.0,
    )
    ContactSolverSettings(initial_contact_seed="linearized_kkt_dev", **base)
    for name, broken in (
            ("q4_source_gap_certificate", "whole_face"),
            ("general_s3_differentiation", "jet2_s3_dev"),
            ("q4_source_pressure_basis", "macro_q1_pair_dev"),
            ("enforcement_solver", "nested_augmented"),
            ("d1_interior_profile", False)):
        with pytest.raises(ValueError, match="initial contact seed requires"):
            ContactSolverSettings(initial_contact_seed="linearized_kkt_dev",
                                  **{**base, name: broken})


def test_fixture_route_admits_the_enabled_selector():
    model, definition, settings, program = _case()
    enabled = replace(_enabled(settings),
                       initial_contact_seed="linearized_kkt_dev")
    assert enabled.initial_contact_seed == "linearized_kkt_dev"
    report = validate_general_contact(model, definition, enabled,
                                      analysis_control="contact_path")
    assert report.valid, report.issues


def test_validation_refuses_non_contact_path_drivers():
    model, definition, settings, program = _case()
    # Fixture route with the continuation halvings cleared: the seed
    # refusal is the reported issue, not the pre-existing continuation
    # refusal that would otherwise precede it.
    seeded = replace(_enabled(settings),
                     initial_contact_seed="linearized_kkt_dev",
                     path_continuation_maximum_halvings=0)
    assert seeded.path_continuation_maximum_halvings == 0
    report = validate_general_contact(model, definition, seeded,
                                      analysis_control="force")
    assert not report.valid
    assert report.issues == (
        "initial_contact_seed requires contact_path_program",)


def test_request_identity_omits_none_and_binds_enabled_revision():
    model, definition, settings, program = _case()
    disabled = _enabled(settings)
    identity = identify_general_contact_request(
        model, definition, disabled, contact_path_program=program)
    request = json.loads(identity.canonical_request)
    assert "initial_contact_seed" not in request["settings"]
    assert "contact_initial_seed_revision" not in request
    enabled = replace(disabled, initial_contact_seed="linearized_kkt_dev")
    enabled_identity = identify_general_contact_request(
        model, definition, enabled, contact_path_program=program)
    enabled_request = json.loads(enabled_identity.canonical_request)
    assert (enabled_request["settings"]["initial_contact_seed"]
            == "linearized_kkt_dev")
    assert (enabled_request["contact_initial_seed_revision"]
            == _INITIAL_CONTACT_SEED_REVISION)
    # The enabled request changes exactly the seed binding and the
    # settings selection; no other request field moves.
    changed = sorted(key for key in set(request) | set(enabled_request)
                     if request.get(key) != enabled_request.get(key))
    assert changed == ["contact_initial_seed_revision", "settings"]
