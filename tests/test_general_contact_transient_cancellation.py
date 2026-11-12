"""Cancellation boundaries preserve candidate isolation and reusable models."""

import numpy as np
import pytest

from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver import TransientConfig, solve_transient_contact
import anyfesolver.general_contact.transient as driver
from anyfesolver.general_contact.core import active_contact_manager, _PlanarQ4PlaneManager
from test_general_contact_transient_plane import _guided_plate, _constant_q4_load


def _solve(model, contact, velocity, token=None):
    return solve_transient_contact(
        model, contact, TransientConfig(.0002, .011, initial_velocity=velocity),
        base_load_case=_constant_q4_load(), emit_restart_checkpoint=True,
        cancellation_token=token,
    )


def test_transient_cancelled_at_entry_runs_no_model_preparation(monkeypatch):
    model, contact, velocity = _guided_plate()
    monkeypatch.setattr(model, "apply_boundary_conditions",
                        lambda: pytest.fail("cancelled solve prepared model"))
    token = CancellationToken()
    token.cancel("before start")
    with pytest.raises(SolveCancelled, match="transient contact entry"):
        _solve(model, contact, velocity, token)
    assert active_contact_manager() is None


def test_transient_rejects_untrusted_token_before_invoking_it():
    class ForeignToken:
        def raise_if_cancelled(self, stage):
            pytest.fail("foreign callback ran")
    model, contact, velocity = _guided_plate()
    with pytest.raises(TypeError, match="exact unmodified CancellationToken"):
        _solve(model, contact, velocity, ForeignToken())
    token = CancellationToken()
    token._event._flag = ForeignToken()
    with pytest.raises(TypeError, match="exact unmodified CancellationToken"):
        _solve(model, contact, velocity, token)


@pytest.mark.parametrize("boundary", ["Newton", "acceptance"])
def test_transient_midtrial_cancel_preserves_commit_boundary_and_model_reuse(monkeypatch, boundary):
    model, contact, velocity = _guided_plate()
    token = CancellationToken()
    original_assemble = driver._assemble_nonlinear_system
    original_evaluate = _PlanarQ4PlaneManager.evaluate
    original_commit = driver.commit_state_candidate
    commits = []
    calls = 0
    def assembling(*args, **kwargs):
        nonlocal calls
        result = original_assemble(*args, **kwargs)
        calls += 1
        if boundary == "Newton" and calls == 2:
            token.cancel("during trial")
        return result
    def evaluating(self, displacement, *args, **kwargs):
        result = original_evaluate(self, displacement, *args, **kwargs)
        if boundary == "acceptance" and kwargs.get("tangent") is False and np.any(displacement):
            token.cancel("before step commit")
        return result
    def committing(*args, **kwargs):
        commits.append(True)
        return original_commit(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(driver, "_assemble_nonlinear_system", assembling)
        patch.setattr(_PlanarQ4PlaneManager, "evaluate", evaluating)
        patch.setattr(driver, "commit_state_candidate", committing)
        with pytest.raises(SolveCancelled):
            _solve(model, contact, velocity, token)
    assert commits == [True]  # initial state only; rejected step has no commit
    assert active_contact_manager() is None
    reused = _solve(model, contact, velocity)
    fresh_model, fresh_contact, fresh_velocity = _guided_plate()
    fresh = _solve(fresh_model, fresh_contact, fresh_velocity)
    for field in ("displacements", "velocities", "accelerations", "external_nodal_work_j"):
        np.testing.assert_array_equal(getattr(reused, field), getattr(fresh, field))
    assert reused.restart_checkpoint == fresh.restart_checkpoint
