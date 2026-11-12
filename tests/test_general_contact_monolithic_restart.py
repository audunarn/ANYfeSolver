"""State continuity and identity for the opt-in monolithic contact route."""

from dataclasses import replace
import copy

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear
from anyfesolver.control import SolveCancelled
from anyfesolver.general_contact import semismooth_path
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.nonlinear_restart import _checkpoint_without_hash, _sha256
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _case():
    model, definition, settings, program = build_case(
        4, load_profile="distributed", total_force_n=0.3,
        target_right_x_m=1.3,
    )
    settings = replace(
        settings, enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=0.3,
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
    )
    return model, definition, settings, program


def test_monolithic_rejects_missing_ordinary_load_and_invalid_force_scale():
    model, definition, settings, program = _case()
    with pytest.raises(ValueError, match="needs a LoadCase"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
        )
    with pytest.raises(ValueError, match="force scale must be bounded"):
        solve_static_nonlinear(
            model, contact=definition,
            contact_settings=replace(settings, complementarity_force_scale_n=1e9),
            contact_path_program=program,
        )


def test_monolithic_active_contact_split_restart_and_mode_identity():
    model, definition, settings, program = _case()
    full = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert full.status == "completed"
    assert len(full.steps) == 2

    model, definition, settings, program = _case()
    half = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    assert half.status == "completed"
    assert len(half.steps) == 1
    assert half.restart_checkpoint is not None
    assert "normal_forces" in half.restart_checkpoint["path_state"]
    pressure = np.asarray(
        half.info["force_displacement_history"][-1]["contact"]["numerical_pressure_pa"]
    )
    support = np.asarray(
        half.info["force_displacement_history"][-1]["contact"]["reference_support_m2"]
    )
    np.testing.assert_allclose(
        pressure*support,
        half.restart_checkpoint["path_state"]["normal_forces"],
        rtol=0.0, atol=1e-12,
    )

    damaged = copy.deepcopy(half.to_restart_checkpoint())
    damaged["path_state"]["normal_multipliers"][0] += 0.01
    damaged["checkpoint_sha256"] = _sha256(_checkpoint_without_hash(damaged))
    model, definition, settings, program = _case()
    with pytest.raises(ValueError, match="pressure and nodal forces are incompatible"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            restart_checkpoint=damaged,
        )

    model, definition, settings, program = _case()
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        restart_checkpoint=half.to_restart_checkpoint(),
        emit_restart_checkpoint=True,
    )
    assert resumed.status == "completed"
    np.testing.assert_allclose(resumed.displacements, full.displacements,
                               rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        resumed.info["force_displacement_history"][-1]["contact"]["integration_force"],
        full.info["force_displacement_history"][-1]["contact"]["integration_force"],
        rtol=0.0, atol=1e-10,
    )

    changed = replace(settings, complementarity_force_scale_n=0.2)
    with pytest.raises(ValueError, match="incompatible"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=changed,
            contact_path_program=program, max_iterations=18,
            restart_checkpoint=half.to_restart_checkpoint(),
        )
    changed_pivots = replace(settings, monolithic_maximum_pivots=32)
    with pytest.raises(ValueError, match="incompatible"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=changed_pivots,
            contact_path_program=program, max_iterations=18,
            restart_checkpoint=half.to_restart_checkpoint(),
        )
    changed_linear_solver = replace(settings,
                                    monolithic_linear_solver="block_response_dev")
    with pytest.raises(ValueError, match="incompatible"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=changed_linear_solver,
            contact_path_program=program, max_iterations=18,
            restart_checkpoint=half.to_restart_checkpoint(),
        )
    changed_interaction = replace(
        definition.interactions[0],
        normal_stiffness=1.01 * definition.interactions[0].normal_stiffness,
    )
    changed_definition = replace(
        definition, interactions=(changed_interaction,),
    )
    with pytest.raises(ValueError, match="incompatible"):
        solve_static_nonlinear(
            model, contact=changed_definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            restart_checkpoint=half.to_restart_checkpoint(),
        )


def test_monolithic_cancellation_after_accepted_step_allows_clean_rerun():
    class CancelBeforeSecondStep:
        def raise_if_cancelled(self, stage=""):
            if stage == "contact_path.step:2":
                raise SolveCancelled("development interruption", stage=stage)

    model, definition, settings, program = _case()
    with pytest.raises(SolveCancelled, match="development interruption"):
        solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            cancellation_token=CancelBeforeSecondStep(),
        )
    rerun = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
    )
    fresh_model, fresh_definition, fresh_settings, fresh_program = _case()
    fresh = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=fresh_settings,
        contact_path_program=fresh_program, max_iterations=18,
    )
    assert rerun.status == fresh.status == "completed"
    np.testing.assert_allclose(rerun.displacements, fresh.displacements,
                               rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        rerun.info["general_contact"]["integration_force"],
        fresh.info["general_contact"]["integration_force"],
        rtol=0.0, atol=1e-10,
    )


@pytest.mark.parametrize("failure_type", (RuntimeError, ValueError))
def test_monolithic_contact_trial_failure_restores_reusable_model(
        monkeypatch, failure_type):
    """Candidate exceptions and rejected steps cannot leak trial state."""
    entered_second_step = [False]

    class MarkSecondStep:
        def raise_if_cancelled(self, stage=""):
            if stage == "contact_path.step:2":
                entered_second_step[0] = True

    original = semismooth_path.linearize
    failed_after_contact = []

    def raise_after_contact(*args, **kwargs):
        work = original(*args, **kwargs)
        if entered_second_step[0] and not failed_after_contact:
            failed_after_contact.append(True)
            raise failure_type("injected post-contact candidate failure")
        return work

    monkeypatch.setattr(semismooth_path, "linearize", raise_after_contact)
    model, definition, settings, program = _case()
    def interrupted_solve():
        return solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            cancellation_token=MarkSecondStep(),
        )
    if failure_type is RuntimeError:
        with pytest.raises(RuntimeError,
                           match="injected post-contact candidate failure"):
            interrupted_solve()
    else:
        stopped = interrupted_solve()
        assert stopped.status == "stopped_at_limit"
        assert len(stopped.steps) == 1
        assert "contact_semismooth_candidate_failed" in stopped.failure_reason
    assert failed_after_contact
    monkeypatch.setattr(semismooth_path, "linearize", original)

    rerun = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    fresh_model, fresh_definition, fresh_settings, fresh_program = _case()
    fresh = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=fresh_settings,
        contact_path_program=fresh_program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert rerun.status == fresh.status == "completed"
    np.testing.assert_allclose(rerun.displacements, fresh.displacements,
                               rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        rerun.info["general_contact"]["integration_force"],
        fresh.info["general_contact"]["integration_force"],
        rtol=0.0, atol=1e-10,
    )
    assert rerun.to_restart_checkpoint() == fresh.to_restart_checkpoint()


def test_monolithic_zero_swept_fraction_rolls_back_accepted_prefix(monkeypatch):
    """An unresolved swept trial must fail cleanly without leaking step two."""
    entered_second_step = [False]

    class MarkSecondStep:
        def raise_if_cancelled(self, stage=""):
            if stage == "contact_path.step:2":
                entered_second_step[0] = True

    original = GeneralS3PatchManager.admissible_step_fraction
    blocked = []

    def refuse_second_step(self, start, end):
        if entered_second_step[0]:
            blocked.append(True)
            return 0.0
        return original(self, start, end)

    monkeypatch.setattr(GeneralS3PatchManager, "admissible_step_fraction",
                        refuse_second_step)
    model, definition, settings, program = _case()
    stopped = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        cancellation_token=MarkSecondStep(),
    )
    assert blocked
    assert stopped.status == "stopped_at_limit"
    assert len(stopped.steps) == 1
    assert stopped.failure_reason == "contact_semismooth_backtracking_failed"
    monkeypatch.setattr(GeneralS3PatchManager, "admissible_step_fraction",
                        original)

    rerun = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    fresh_model, fresh_definition, fresh_settings, fresh_program = _case()
    fresh = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=fresh_settings,
        contact_path_program=fresh_program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert rerun.status == fresh.status == "completed"
    assert rerun.to_restart_checkpoint() == fresh.to_restart_checkpoint()


def test_monolithic_accepted_swept_fraction_reaches_same_committed_state(
        monkeypatch):
    """A shortened admissible trial can converge without committing a trial."""
    original = GeneralS3PatchManager.admissible_step_fraction
    shortened = []

    def shorten_first_admissible_trial(self, start, end):
        fraction = original(self, start, end)
        if not shortened and fraction == 1.0:
            shortened.append(True)
            return 0.5
        return fraction

    monkeypatch.setattr(GeneralS3PatchManager, "admissible_step_fraction",
                        shorten_first_admissible_trial)
    model, definition, settings, program = _case()
    shortened_result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert shortened and shortened_result.status == "completed", (
        shortened_result.failure_reason)
    assert shortened_result.info["force_displacement_history"][0]["backtracks"] == 0

    monkeypatch.setattr(GeneralS3PatchManager, "admissible_step_fraction",
                        original)
    fresh_model, fresh_definition, fresh_settings, fresh_program = _case()
    fresh = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=fresh_settings,
        contact_path_program=fresh_program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    assert fresh.status == "completed", fresh.failure_reason
    np.testing.assert_allclose(shortened_result.displacements,
                               fresh.displacements, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        shortened_result.info["general_contact"]["integration_force"],
        fresh.info["general_contact"]["integration_force"],
        rtol=0.0, atol=1e-10,
    )
    shortened_checkpoint = shortened_result.to_restart_checkpoint()
    fresh_checkpoint = fresh.to_restart_checkpoint()
    assert (shortened_checkpoint["model_fingerprint"]
            == fresh_checkpoint["model_fingerprint"])
    assert (shortened_checkpoint["path_state"]["accepted_step"]
            == fresh_checkpoint["path_state"]["accepted_step"] == 2)
    np.testing.assert_allclose(
        shortened_checkpoint["path_state"]["normal_forces"],
        fresh_checkpoint["path_state"]["normal_forces"],
        rtol=0.0, atol=1e-10,
    )
