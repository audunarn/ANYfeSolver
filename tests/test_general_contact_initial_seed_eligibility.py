"""Focused F2 eligibility discrimination and truthful example prose.

The independent review (F2) suspected a silently ignored seed selection on
routes that pass validation: the runtime selection drops the seed to None
without refusal when friction is declared or when the validated profile is
not the refined Q4/S3 overlap-local one. This module closes both halves
with real production calls, no solve and no new scientific admission:

- Friction half: every friction route conflicts with the seed-pinned
  settings, and the path entry now refuses the enabled seed with friction
  explicitly (fail closed) before any route preparation, mirroring the
  driver-level refusal pattern. The check uses a real friction settings
  object on the coupled fixture.
- Profile half: the Q4/Q4-monolithic profile requires
  ``jet2_q4_q4_dev``, which the enabled seed forbids; a definition with a
  Q4 target face is refused by real validation with the enabled seed, and
  the fixture's own validating route selects exactly the seed-eligible
  overlap-local profile. No silently skipped profile shape validates.

It also pins the example's prose truth: the coupled example is UNVERIFIED
to completion and its source-bound checks are not installed testing.
"""
import importlib.util
import os
from dataclasses import replace

import pytest

from anyfesolver import (
    ContactDefinition, ContactInteraction, ContactSurface,
    validate_general_contact,
)
from anyfesolver.general_contact import static_path
from anyfesolver.general_contact.s3_patch_static import (
    REFINED_Q4_S3_OVERLAP_LOCAL_AUGMENTED_PROFILE,
)
from anyfesolver.general_contact.s3_sliding_friction import (
    S3MovingNormalFrictionSettings,
)

from test_general_contact_admission_globalization_driver_small_patch import (
    _GAP_TOLERANCE_M,
    _case,
    _enabled,
)

_EXAMPLE_NAME = "general_contact_initial_contact_seed_dev.py"


def _seeded(settings):
    return replace(_enabled(settings),
                   initial_contact_seed="linearized_kkt_dev")


def test_enabled_seed_with_friction_refuses_explicitly_before_preparation():
    model, definition, settings, program = _case()
    seeded = _seeded(settings)
    friction = S3MovingNormalFrictionSettings(0.3, 1.0e6, (1.0e-5,))
    # Fail closed at the path entry, before any route preparation or
    # Newton work: no silent drop of the requested selection.
    with pytest.raises(ValueError,
                        match="initial_contact_seed requires frictionless "
                              "contact"):
        static_path.solve_static_contact_path(
            model, definition, seeded, program, friction=friction)


def test_enabled_seed_refuses_q4_target_definition_at_validation():
    model, definition, settings, program = _case()
    seeded = _seeded(settings)
    # A Q4 target face is the only shape the Q4/Q4-monolithic profile
    # selector could order before overlap-local; with the enabled seed
    # it cannot validate, so the silent profile skip is unreachable.
    q4_target_definition = ContactDefinition(
        surfaces=(definition.surfaces[0],
                  ContactSurface("q4_target", (1,), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", definition.surfaces[0].name, "q4_target", 10000.0,
            _GAP_TOLERANCE_M, normal_law="augmented"),),
    )
    report = validate_general_contact(model, q4_target_definition, seeded,
                                      analysis_control="contact_path")
    assert not report.valid
    # Fail closed on the refined Q4/S3 route: the D1 interior-profile
    # gate refuses the Q4-target shape before any seed-eligible profile
    # could be selected, so the silent profile skip is unreachable.
    assert any("Q4/S3" in issue for issue in report.issues), (
        report.issues)


def test_fixture_route_with_enabled_seed_selects_seed_eligible_profile():
    model, definition, settings, program = _case()
    seeded = _seeded(settings)
    report = validate_general_contact(model, definition, seeded,
                                      analysis_control="contact_path")
    assert report.valid, report.issues
    assert report.profile == REFINED_Q4_S3_OVERLAP_LOCAL_AUGMENTED_PROFILE


def _load_example():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "examples", _EXAMPLE_NAME)
    assert os.path.isfile(path), path
    spec = importlib.util.spec_from_file_location(
        "initial_contact_seed_dev_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_prose_is_truthful_about_completion():
    example = _load_example()
    prose = example.__doc__ or ""
    assert "completes the requested half-load" in prose
    assert "prefix (one step)" in prose
    assert "not an installed-package" in prose
    assert "or D1 acceptance" in prose
    assert "historical coarse-fixture failures remain valid" in prose
