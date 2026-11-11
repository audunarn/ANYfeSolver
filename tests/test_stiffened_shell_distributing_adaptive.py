"""Reuse the existing controller with exact accepted-path material oracles."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_increment_control import decode_control, encode_control
from test_stiffened_shell_ge_distributing_finite_public import definition as elastic_definition, actions as elastic_actions
from test_stiffened_shell_ge_distributing_material import definition as material_definition, actions as material_actions


def model_and_actions(kind):
    if kind == "elastic":
        return elastic_definition(), elastic_actions()
    return material_definition(kind == "coupled"), material_actions()


def bounded_intervals(monkeypatch):
    """Inject only the existing typed Newton-exhaustion outcome; real substeps solve."""
    original = FinitePointDriver._solve_fixed
    seen = []
    def fixed(driver, actions, targets, **kwargs):
        factor = float(targets[0]); seen.append((factor, driver._load_factor))
        if len(targets) == 1 and factor-driver._load_factor > .251:
            raise ss.FiniteIncrementFailure(ss.IncrementAttempt(factor, driver._load_factor,
                factor, "rejected", 1, 2, 1, 1, 1., 0., "finite mixed backtrack budget exhausted"))
        return original(driver, actions, targets, **kwargs)
    monkeypatch.setattr(FinitePointDriver, "_solve_fixed", fixed)
    return seen


@pytest.mark.parametrize("kind", ("elastic", "generalized", "coupled"))
def test_cutback_checkpoint_cancellation_and_restore_match_identical_fixed_history(kind, monkeypatch):
    model, actions = model_and_actions(kind)
    oracle = ss.create_analysis(definition=model)
    owner = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    control = ss.AdaptiveIncrementControl(3, 12)
    seen = bounded_intervals(monkeypatch)
    token = CancellationToken()
    notifications = []
    try:
        expected = oracle.solve(actions, (.25, .5, .75, 1.))
        fixed_bytes = oracle.checkpoint()
        def observe(result):
            notifications.append(result.load_factor)
            assert owner.checkpoint() and owner.recover().generation == result.generation
            if result.load_factor == .5:
                token.cancel("complete material/control prefix")
        with pytest.raises(SolveCancelled):
            owner.solve(actions, (1.,), increment_control=control,
                cancellation_token=token, on_accepted=observe)
        assert notifications == [.25, .5]
        prefix = owner.checkpoint()
        state, inner = decode_control(prefix, owner._identity)
        assert state.targets == (1.,) and not state.completed
        assert [a.status for a in state.attempts] == ["rejected", "rejected", "accepted", "accepted"]
        restored = resumed.restore(prefix)
        assert restored.accepted_factors == (.25, .5)
        assert resumed.checkpoint() == prefix
        with pytest.raises(ValueError, match="matching targets"):
            resumed.solve(actions, (.9,), increment_control=control)
        assert resumed.checkpoint() == prefix
        final = resumed.solve(actions, (1.,), increment_control=control)
        assert final.status == "completed" and final.accepted_factors == expected.accepted_factors
        assert final.free_equilibrium_ratio <= 1e-8 and final.constraint_max <= 1e-9
        final_control, final_inner = decode_control(resumed.checkpoint(), resumed._identity)
        assert final_control.completed and len(final_control.attempts) == 6
        assert final_inner == fixed_bytes
        for field in ("shell_displacement", "beam_positions", "beam_frames", "cell_frames",
                      "cell_resultants", "multipliers", "support_reactions",
                      "beam_station_accumulated", "shell_layer_alpha"):
            np.testing.assert_array_equal(getattr(final, field), getattr(expected, field))
        if kind != "elastic":
            assert np.max(final.beam_station_accumulated) > 0.
        if kind == "coupled":
            assert np.max(final.shell_layer_alpha) > 0.
        assert (1., 0.) in seen and (.5, 0.) in seen
        assert owner.capabilities["increment_control"] == ("fixed", "experimental_adaptive_halving")
    finally:
        oracle.close(); owner.close(); resumed.close()


@pytest.mark.parametrize("targets", ((.5, .25), (.5, .5), (-1.,)))
def test_nonmonotone_or_invalid_adaptive_targets_fail_before_publication(targets):
    owner = ss.create_analysis(definition=elastic_definition())
    try:
        with pytest.raises(ValueError):
            owner.solve(elastic_actions(), targets, increment_control=ss.AdaptiveIncrementControl())
        assert owner._generation == 0 and owner._finite_driver is None
    finally: owner.close()


@pytest.mark.parametrize("corruption", ("future_targets", "inner_payload"))
def test_canonical_but_inadmissible_checkpoint_is_rejected_atomically(corruption):
    owner = ss.create_analysis(definition=elastic_definition())
    untouched = ss.create_analysis(definition=elastic_definition())
    try:
        owner.solve(elastic_actions(), (.25,), increment_control=ss.AdaptiveIncrementControl())
        state, inner = decode_control(owner.checkpoint(), owner._identity)
        if corruption == "future_targets":
            # Replay of the accepted first target remains valid. Only the
            # distributing profile's future-target admission rejects this.
            state = replace(state, targets=(.25, .2))
            expected = "increasing force targets"
        else:
            inner = b"{}"
            expected = "dedicated binary inner state"
        bad = encode_control(owner._identity, state, inner)
        with pytest.raises(ValueError, match=expected):
            untouched.restore(bad)
        assert untouched._generation == 0 and untouched._finite_driver is None
    finally:
        owner.close(); untouched.close()


@pytest.mark.parametrize("budget", ("cutbacks", "attempts"))
def test_exhaustion_preserves_an_existing_accepted_prefix(budget, monkeypatch):
    owner = ss.create_analysis(definition=elastic_definition())
    try:
        first = owner.solve(elastic_actions(), (.25,))
        saved = owner.checkpoint()
        calls = []
        def fail(driver, actions, targets, **kwargs):
            factor = float(targets[0]); calls.append(factor)
            raise ss.FiniteIncrementFailure(ss.IncrementAttempt(
                factor, driver._load_factor, factor, "rejected", 1, 2, 1, 1,
                1., 0., "finite mixed backtrack budget exhausted"))
        monkeypatch.setattr(FinitePointDriver, "_solve_fixed", fail)
        control = ss.AdaptiveIncrementControl(0, 12) if budget == "cutbacks" else ss.AdaptiveIncrementControl(3, 1)
        with pytest.raises(ss.AdaptiveIncrementExhausted, match="cutback limit" if budget == "cutbacks" else "attempt limit") as caught:
            owner.solve(elastic_actions(), (1.,), increment_control=control)
        assert caught.value.accepted_factors == (.25,) and len(caught.value.attempts) == 1
        assert calls == [1.] and owner.checkpoint() == saved
        assert owner.recover().generation == first.generation
    finally: owner.close()


def test_observer_numerical_failure_propagates_after_complete_commit():
    owner = ss.create_analysis(definition=elastic_definition())
    notifications = []
    failure = ss.FiniteIncrementFailure(ss.IncrementAttempt(
        .25, 0., .25, "rejected", 1, 2, 1, 1, 1., 0., "finite mixed backtrack budget exhausted"))
    try:
        def observe(result):
            notifications.append(result.load_factor)
            raise failure
        with pytest.raises(ss.FiniteIncrementFailure) as caught:
            owner.solve(elastic_actions(), (.25, .5), increment_control=ss.AdaptiveIncrementControl(),
                        on_accepted=observe)
        assert caught.value is failure and notifications == [.25]
        state, _ = decode_control(owner.checkpoint(), owner._identity)
        assert [row.status for row in state.attempts] == ["accepted"]
        assert owner.recover().accepted_factors == (.25,)
    finally: owner.close()


@pytest.mark.parametrize("error", (RuntimeError, MemoryError))
def test_programming_or_memory_failure_is_never_adaptive_cutback(error, monkeypatch):
    owner = ss.create_analysis(definition=elastic_definition())
    calls = []
    def fail(*args, **kwargs):
        calls.append(1); raise error("must propagate")
    monkeypatch.setattr(FinitePointDriver, "_assemble", fail)
    try:
        with pytest.raises(error, match="must propagate"):
            owner.solve(elastic_actions(), (1.,), increment_control=ss.AdaptiveIncrementControl())
        assert calls == [1] and owner._generation == 0 and owner._finite_driver is None
    finally: owner.close()
