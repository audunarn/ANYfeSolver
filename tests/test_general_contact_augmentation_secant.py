"""Safeguards for the optional projected augmented-contact predictor."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import (
    ContactSolverSettings, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.general_contact.augmentation import (
    ProjectedAugmentationSecant, propose_active_set_pivot,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from examples.general_contact_refined_q4_s3_mixed_path import make_case


def test_active_secant_predicts_independent_affine_two_pressure_root():
    root = np.array((0.12, 0.15))
    compliance = np.array(((0.217469849, 0.154498127),
                           (0.188831044, 0.172447028)))
    predictor = ProjectedAugmentationSecant()
    current = np.zeros(2)
    proposals = []
    for _ in range(7):
        projected = np.maximum(0.0, current - compliance @ (current - root))
        proposals.append(predictor.propose(current, projected))
        current = projected
    assert all(value is None for value in proposals[:3])
    np.testing.assert_allclose(proposals[3], root, atol=1.0e-8, rtol=0.0)
    assert predictor.applied >= 1


def test_active_set_pivot_selects_unique_complementarity_subset():
    current = np.ones(4)
    schur = np.eye(4)*1.0e-4
    gap = np.array((1.0, -1.0, 1.0, -2.0))*1.0e-4
    pivot = propose_active_set_pivot(
        current, gap, np.arange(4), schur,
        maximum_pressure=4.0, gap_consistency=1.0e-10,
    )
    assert pivot is not None
    assert pivot.retained_local_indices == (1, 3)
    np.testing.assert_allclose(pivot.multipliers, (0.0, 2.0, 0.0, 3.0))
    assert propose_active_set_pivot(
        current, gap, np.arange(4), schur,
        maximum_pressure=2.5, gap_consistency=1.0e-10,
    ) is None
    assert propose_active_set_pivot(
        current, gap, np.arange(4), np.zeros((4, 4)),
        maximum_pressure=4.0, gap_consistency=1.0e-10,
    ) is None


def test_active_secant_resets_at_active_set_change_and_rejects_invalid_samples():
    predictor = ProjectedAugmentationSecant()
    assert predictor.propose(np.zeros(2), np.array((0.1, 0.0))) is None
    assert predictor.propose(np.array((0.1, 0.0)), np.array((0.2, 0.0))) is None
    assert predictor.propose(np.array((0.2, 0.0)), np.array((0.3, 0.1))) is None
    assert predictor.applied == 0
    assert predictor.propose(np.array((0.3, 0.1)), np.zeros(2)) is None
    assert predictor.applied == 0
    try:
        predictor.propose(np.zeros(2), np.array((float("nan"), 0.0)))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite multiplier projection was accepted")


def test_active_secant_falls_back_for_unidentifiable_multiplier_map():
    predictor = ProjectedAugmentationSecant()
    for _ in range(4):
        assert predictor.propose(np.array((0.1, 0.1)),
                                 np.array((0.2, 0.2))) is None
    assert predictor.attempts == 1
    assert predictor.applied == 0


def test_active_secant_setting_is_explicit_and_validated():
    assert ContactSolverSettings().augmentation_acceleration == "none"
    assert ContactSolverSettings(
        augmentation_acceleration="active_secant"
    ).augmentation_acceleration == "active_secant"
    with pytest.raises(ValueError, match="augmentation_acceleration"):
        ContactSolverSettings(augmentation_acceleration="automatic")
    with pytest.raises(ValueError, match="requires positive pressure"):
        ContactSolverSettings(augmentation_acceleration="active_secant_pivot")
    with pytest.raises(ValueError, match="bounds require active_secant_pivot"):
        ContactSolverSettings(augmentation_pivot_max_pressure_pa=20.0)
    with pytest.raises(ValueError, match="at least 15 augmentations"):
        ContactSolverSettings(
            augmentation_acceleration="active_secant_pivot",
            augmentation_pivot_max_pressure_pa=20.0,
            augmentation_pivot_difference_step_m=1.0e-7,
        )
    model, definition, settings, _program = make_case()
    pivot_settings = replace(
        settings, augmentation_acceleration="active_secant_pivot",
        augmentation_pivot_max_pressure_pa=20.0,
        augmentation_pivot_difference_step_m=1.0e-7,
    )
    pivot_report = validate_general_contact(model, definition, pivot_settings)
    assert pivot_report.valid
    assert "Q4_S3_PATCH_REFINED" in pivot_report.profile
    with pytest.raises(ValueError, match="requires active_secant_pivot"):
        ContactSolverSettings(augmentation_equilibrated_pressure_step_pa=0.01)
    with pytest.raises(ValueError, match="finite and positive"):
        replace(pivot_settings, augmentation_equilibrated_pressure_step_pa=-0.01)
    mapped_settings = replace(
        pivot_settings, augmentation_equilibrated_pressure_step_pa=0.01,
    )
    assert validate_general_contact(model, definition, mapped_settings).valid
    assert (GeneralS3PatchManager(model, definition, mapped_settings)
            .restart_contract()["settings"]["augmentation_equilibrated_pressure_step_pa"]
            == 0.01)
    assert ("augmentation_equilibrated_pressure_step_pa" not in
            GeneralS3PatchManager(model, definition, pivot_settings)
            .restart_contract()["settings"])
    penalty = replace(definition, interactions=(replace(
        definition.interactions[0], normal_law="penalty",
    ),))
    report = validate_general_contact(
        model, penalty, replace(settings, augmentation_acceleration="active_secant"),
    )
    assert not report.valid
    assert "requires one augmented" in report.issues[0]
    with pytest.raises(ValueError, match="requires the contact path analysis owner"):
        solve_static_nonlinear(
            model, contact=definition,
            contact_settings=replace(
                settings, augmentation_acceleration="active_secant",
            ),
        )


def test_roundoff_pressure_support_is_opt_in_and_force_bounded(monkeypatch):
    model, definition, settings, _program = make_case()
    with pytest.raises(ValueError, match="positive force budget"):
        replace(settings, augmentation_support_resolution="roundoff")
    admitted = replace(
        settings, augmentation_support_resolution="roundoff",
        augmentation_support_force_budget_n=1.0e-9,
    )
    report = validate_general_contact(model, definition, admitted)
    assert report.valid
    assert "ROUNDOFF" in report.profile
    penalty = replace(definition, interactions=(replace(
        definition.interactions[0], normal_law="penalty",
    ),))
    assert not validate_general_contact(model, penalty, admitted).valid
    manager = GeneralS3PatchManager(model, definition, admitted)
    source, target = manager._positions(np.zeros(manager.n_dof))
    state = manager.law.geometry.evaluate(source, target)
    support = np.full(len(manager.law.source_node_ids), 0.1)
    support[-1] = 1.0e-18
    trial = replace(state, reference_support=support,
                    normalized_gap=np.zeros_like(support))
    multipliers = np.full(len(support), 10.0)
    resolved, floor = manager.law.resolved_support(trial)
    assert floor == pytest.approx(np.finfo(float).eps*sum(support))
    assert np.all(resolved[:-1]) and not resolved[-1]
    assert manager.law._project(trial, multipliers).pressure[-1] == 10.0
    np.testing.assert_array_equal(
        manager.law.projected_multiplier(trial, multipliers),
        np.r_[np.full(len(support)-1, 10.0), 0.0],
    )
    monkeypatch.setattr(manager.law, "evaluate", lambda *_: trial)
    monkeypatch.setattr(
        manager.law, "work",
        lambda _source, _target, pressure, *, tangent: SimpleNamespace(
            residual=np.array((pressure[-1]*1.0e-8,)),
        ),
    )
    manager.set_normal_multipliers(multipliers)
    with pytest.raises(ValueError, match="local force budget"):
        manager.projected_normal_multipliers(np.zeros(manager.n_dof))
    monkeypatch.setattr(
        manager.law, "work",
        lambda _source, _target, pressure, *, tangent: SimpleNamespace(
            residual=np.array((pressure[-1]*1.0e-11,)),
        ),
    )
    projected = manager.projected_normal_multipliers(np.zeros(manager.n_dof))
    assert projected[-1] == 0.0
    assert manager.last_support_reset["local_force_change_n"] == pytest.approx(1.0e-10)
