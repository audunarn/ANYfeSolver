"""The nonlinear Schur fallback is explicit and budgeted."""

from dataclasses import replace

import pytest

from anyfesolver import ContactSolverSettings
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_free_target_press_probe import build_free_target_case


def _settings():
    return ContactSolverSettings(
        augmentation_acceleration="supported_schur_dev",
        augmentation_schur_maximum_modes=4,
        augmentation_schur_maximum_pressure_pa=100.0,
        augmentation_schur_maximum_step_pa=10.0,
        augmentation_schur_difference_step_m=1.0e-6,
        maximum_augmentations=7,
        augmentation_schur_maximum_attempts=2,
    )


def test_backtrack_requires_explicit_opt_in_and_spare_attempt():
    base = _settings()
    assert not base.augmentation_schur_nonlinear_backtrack_dev
    assert replace(base, augmentation_schur_nonlinear_backtrack_dev=True)
    with pytest.raises(ValueError, match="at least two attempts"):
        replace(base, augmentation_schur_nonlinear_backtrack_dev=True,
                augmentation_schur_maximum_attempts=1)
    with pytest.raises(ValueError, match="must be boolean"):
        replace(base, augmentation_schur_nonlinear_backtrack_dev=1)
    with pytest.raises(ValueError, match="requires supported_schur_dev"):
        ContactSolverSettings(augmentation_schur_nonlinear_backtrack_dev=True)


def test_restart_contract_binds_opt_in_without_changing_default_identity():
    model, definition, settings, _program = build_free_target_case(8)
    ordinary = GeneralS3PatchManager(model, definition, settings)
    enabled = GeneralS3PatchManager(
        model, definition,
        replace(settings, augmentation_schur_nonlinear_backtrack_dev=True),
    )
    ordinary_contract = ordinary.restart_contract()
    enabled_contract = enabled.restart_contract()
    assert "augmentation_schur_nonlinear_backtrack_dev" not in (
        ordinary_contract["settings"])
    assert enabled_contract["settings"].pop(
        "augmentation_schur_nonlinear_backtrack_dev") is True
    assert enabled_contract == ordinary_contract
