"""Focused driver checks for the declared macro physical-substep continuation.

Small owner module: real static contact-path runs on the small interior-press
fixture (two macro steps, never the large eight-macro fixture) covering the
cutback substep chronology and force ramp, the report and post-commit macro
snapshot rollback boundaries, cancellation inside the first macro attempt and
inside the substep loop, stagnation-control validation, and restart refusal of
tampered or incompatible continuation history.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear
from anyfesolver.control import SolveCancelled
from anyfesolver.general_contact import static_path
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth_path import (
    SemismoothStep,
    solve_increment,
)
from anyfesolver.nonlinear_restart import (
    _CHECKPOINT_HASH_KEY,
    _checkpoint_without_hash,
    _sha256,
)
from scripts.general_contact_d1_interior_press_dev import build_interior_press

_CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION = "declared_macro_path_physical_substeps_v1"


def _case():
    return build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", force_n=0.04,
    )


def _enabled(settings):
    return replace(settings, q4_source_gap_certificate="overlap_local_dev",
                   path_continuation_maximum_halvings=6)


def _solve(model, definition, settings, program, **kwargs):
    return solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18, **kwargs)


_CACHE = {}


def _reference_full():
    if "full" not in _CACHE:
        model, definition, settings, program = _case()
        result = _solve(model, definition, _enabled(settings), program,
                        emit_restart_checkpoint=True)
        assert result.status == "completed" and len(result.steps) == 2
        checkpoint = result.to_restart_checkpoint()
        history = checkpoint["path_state"]["history"]
        assert len(history) == 2
        # A genuinely full macro attempt carries no continuation block.
        assert all("continuation" not in record for record in history)
        _CACHE["full"] = (result, checkpoint)
    return _CACHE["full"]


def _reference_half():
    if "half" not in _CACHE:
        model, definition, settings, program = _case()
        result = _solve(model, definition, _enabled(settings), program,
                        max_load_factor=0.5, emit_restart_checkpoint=True)
        assert result.status == "completed" and len(result.steps) == 1
        _CACHE["half"] = result
    return _CACHE["half"]


def _assert_matches_half(rerun):
    reference = _reference_half()
    assert rerun.status == "completed" and len(rerun.steps) == 1
    np.testing.assert_allclose(rerun.displacements, reference.displacements,
                               rtol=0., atol=1e-10)
    np.testing.assert_allclose(
        rerun.info["general_contact"]["integration_force"],
        reference.info["general_contact"]["integration_force"],
        rtol=0., atol=1e-10)
    assert rerun.to_restart_checkpoint() == reference.to_restart_checkpoint()


class _CancelAt:
    def __init__(self, stage):
        self.stage = stage

    def raise_if_cancelled(self, stage=""):
        if stage == self.stage:
            raise SolveCancelled("injected continuation cancellation",
                                 stage=stage)


def _force_first_macro_refusal(monkeypatch, calls):
    original = static_path.solve_semismooth_increment

    def wrapper(*args, **kwargs):
        calls.append({
            "target": np.asarray(kwargs["target"], dtype=float).copy(),
            "reduced_external":
                np.asarray(kwargs["reduced_external"], dtype=float).copy(),
            "iterations": None,
        })
        if len(calls) == 1:
            # Typed conformance: the production refusal path reads
            # step_result.initial_seed unconditionally, so the
            # injected refusal is a real SemismoothStep, not a loose
            # namespace mock.
            return SemismoothStep(
                completed=False, reason="contact_semismooth_stagnation",
                q=np.asarray(kwargs["q"], dtype=float).copy(),
                reactions=np.asarray(kwargs["reactions"], dtype=float).copy(),
                nodal_forces=np.asarray(kwargs["nodal_forces"], dtype=float).copy(),
                candidate_states=None, contact=None, iterations=3,
                residual_norm=float("inf"), constraint_norm=float("inf"),
                complementarity_norm=float("inf"), candidate_evaluations=2,
                backtracks=1)
        result = original(*args, **kwargs)
        calls[-1]["iterations"] = int(result.iterations)
        return result

    monkeypatch.setattr(static_path, "solve_semismooth_increment", wrapper)


def test_cutback_chronology_force_ramp_and_gates(monkeypatch):
    reference, _checkpoint = _reference_full()
    model, definition, settings, program = _case()
    calls = []
    _force_first_macro_refusal(monkeypatch, calls)
    result = _solve(model, definition, _enabled(settings), program,
                    emit_restart_checkpoint=True)
    assert result.status == "completed" and len(result.steps) == 2
    np.testing.assert_allclose(result.displacements, reference.displacements,
                               rtol=0., atol=5e-8)
    np.testing.assert_allclose(
        result.info["general_contact"]["integration_force"],
        reference.info["general_contact"]["integration_force"],
        rtol=0., atol=5e-8)
    history = result.to_restart_checkpoint()["path_state"]["history"]
    assert len(history) == 2
    first, second = history
    continuation = first["continuation"]
    assert continuation["mode"] == "declared_physical_dyadic_substeps"
    assert (continuation["revision"]
            == _CONTACT_PATH_PHYSICAL_SUBSTEPS_REVISION)
    assert continuation["cutback_reason"] == "contact_semismooth_stagnation"
    assert continuation["maximum_halvings"] == 6
    assert "continuation" not in second
    substeps = continuation["substeps"]
    assert substeps
    tau = 0.0
    for substep in substeps:
        assert substep["tau_start"] == tau
        span = substep["tau_end"] - substep["tau_start"]
        scaled = span * float(1 << 7)
        assert scaled > 0.0 and scaled.is_integer()
        tau = substep["tau_end"]
    assert tau == 1.0
    # Every substep lies on the program's own dyadic target/force ramp.
    macro = calls[0]
    sub_calls = calls[1:1 + len(substeps)]
    assert len(sub_calls) == len(substeps)
    for substep, call in zip(substeps, sub_calls):
        np.testing.assert_allclose(
            call["reduced_external"],
            substep["tau_end"] * macro["reduced_external"],
            rtol=0., atol=1e-16)
        np.testing.assert_allclose(
            call["target"],
            (1. - substep["tau_end"]) * program.target_at(0)
            + substep["tau_end"] * program.target_at(1),
            rtol=0., atol=1e-15)
    assert (first["semismooth_iterations"]
            == 3 + sum(call["iterations"] for call in sub_calls))
    assert (first["external_nodal_work_scope"]
            == "continuation_substep_trapezoid_not_exact_path_integral")
    assert "external_nodal_work_trapezoid_j" in first
    np.testing.assert_allclose(first["target"], program.target_at(1),
                               rtol=0., atol=1e-12)
    assert first["nodal_forces_n"] == program.force_values_at(1).tolist()


def test_report_boundary_exception_restores_complete_macro_snapshot(monkeypatch):
    _reference_half()
    model, definition, settings, program = _case()
    original_increment = static_path.solve_semismooth_increment
    original_evaluate = GeneralS3PatchManager.evaluate
    armed = [False]
    raised = [False]

    def arming_increment(*args, **kwargs):
        result = original_increment(*args, **kwargs)
        armed[0] = True
        return result

    def raise_once(self, displacement, *, tangent=True):
        if armed[0] and not tangent and not raised[0]:
            raised[0] = True
            raise RuntimeError("injected report failure")
        return original_evaluate(self, displacement, tangent=tangent)

    monkeypatch.setattr(static_path, "solve_semismooth_increment",
                        arming_increment)
    monkeypatch.setattr(GeneralS3PatchManager, "evaluate", raise_once)
    with pytest.raises(RuntimeError, match="injected report failure"):
        _solve(model, definition, _enabled(settings), program)
    assert raised[0]
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_post_commit_boundary_exception_restores_complete_macro_snapshot(monkeypatch):
    _reference_half()
    model, definition, settings, program = _case()
    original = static_path._support_reaction_resultants
    raised = [False]

    def raise_once(*args, **kwargs):
        if not raised[0]:
            raised[0] = True
            raise RuntimeError("injected post-commit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(static_path, "_support_reaction_resultants",
                        raise_once)
    with pytest.raises(RuntimeError, match="injected post-commit failure"):
        _solve(model, definition, _enabled(settings), program)
    assert raised[0]
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_cancellation_inside_first_macro_attempt_restores_and_propagates():
    _reference_half()
    model, definition, settings, program = _case()
    token = _CancelAt("contact_path.step:1.semismooth.iteration:1")
    with pytest.raises(SolveCancelled,
                       match="injected continuation cancellation"):
        _solve(model, definition, _enabled(settings), program,
               cancellation_token=token)
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_cancellation_during_substeps_restores_and_propagates(monkeypatch):
    _reference_half()
    model, definition, settings, program = _case()
    calls = []
    _force_first_macro_refusal(monkeypatch, calls)
    token = _CancelAt("contact_path.step:1.continuation:1")
    with pytest.raises(SolveCancelled,
                       match="injected continuation cancellation"):
        _solve(model, definition, _enabled(settings), program,
               cancellation_token=token)
    # The refused macro attempt published no substep Newton work.
    assert len(calls) == 1
    rerun = _solve(model, definition, _enabled(settings), program,
                   max_load_factor=0.5, emit_restart_checkpoint=True)
    _assert_matches_half(rerun)


def test_stagnation_controls_are_validated_before_any_work():
    model, definition, settings, _program = _case()
    manager = GeneralS3PatchManager(model, definition, _enabled(settings))
    with pytest.raises(ValueError, match="stagnation controls"):
        solve_increment(
            model=model, manager=manager, settings=_enabled(settings),
            q=np.zeros(1), reactions=np.zeros(2), nodal_forces=np.zeros(1),
            committed_states={}, transform=None, offset=None,
            control_rows=None, control_reactions=None, target=np.zeros(2),
            reduced_external=np.zeros(1), num_layers=5, max_iterations=1,
            tolerance=1e-7, control_scales=np.ones(2), path_force_magnitude=0.,
            cancellation_token=None, step=1,
            stagnation_patience=0, stagnation_fraction=.125,
            stagnation_merit_rate=.05)


def test_restart_rejects_tampered_and_incompatible_continuation_history(monkeypatch):
    model, definition, settings, program = _case()
    enabled = _enabled(settings)
    calls = []
    _force_first_macro_refusal(monkeypatch, calls)
    half = _solve(model, definition, enabled, program, max_load_factor=0.5,
                  emit_restart_checkpoint=True)
    assert half.status == "completed" and len(half.steps) == 1
    checkpoint = half.to_restart_checkpoint()
    record = checkpoint["path_state"]["history"][0]
    assert record["continuation"]["substeps"]

    def retamper(mutate):
        bad = copy.deepcopy(checkpoint)
        mutate(bad["path_state"]["history"][0]["continuation"])
        bad[_CHECKPOINT_HASH_KEY] = _sha256(_checkpoint_without_hash(bad))
        return bad

    fresh = _case()
    with pytest.raises(ValueError, match="chronology is invalid"):
        _solve(fresh[0], fresh[1], enabled, fresh[3],
               restart_checkpoint=retamper(
                   lambda cont: cont["substeps"][0].__setitem__(
                       "tau_end", 0.3)))
    fresh = _case()
    with pytest.raises(ValueError, match="continuation block is incompatible"):
        _solve(fresh[0], fresh[1], enabled, fresh[3],
               restart_checkpoint=retamper(
                   lambda cont: cont.__setitem__("revision", "forged_v0")))
    fresh = _case()
    with pytest.raises(ValueError, match="analysis contract is incompatible"):
        _solve(fresh[0], fresh[1], replace(enabled,
                                           path_continuation_maximum_halvings=0),
               fresh[3], restart_checkpoint=checkpoint)
    # A clean macro-only restart completes the trajectory consistently.
    fresh = _case()
    resumed = _solve(fresh[0], fresh[1], enabled, fresh[3],
                     restart_checkpoint=checkpoint,
                     emit_restart_checkpoint=True)
    assert resumed.status == "completed" and len(resumed.steps) == 2
    reference, _checkpoint = _reference_full()
    np.testing.assert_allclose(resumed.displacements, reference.displacements,
                               rtol=0., atol=5e-8)
    resumed_history = (resumed.to_restart_checkpoint()
                       ["path_state"]["history"])
    assert len(resumed_history) == 2
    assert "continuation" not in resumed_history[1]
