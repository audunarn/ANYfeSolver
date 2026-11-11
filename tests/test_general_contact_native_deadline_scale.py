"""Bounded wall-clock deadline scale, refusal and expiry/cancel/rollback safety.

The cooperative deadlines are finite-work controls, not scientific checks.
These tests pin the shared effective-deadline service: the ordinary default
scale 1 reproduces the historical thresholds, an explicitly declared bounded
scale multiplies them, invalid declarations fail closed, and a scaled context
still refuses on expiry, honours cancellation first and keeps the accepted
state intact. A fake clock keeps the local simulation far below any real
budget; the outer portable-CI shard budget stays authoritative.
"""
import pytest

import anyfesolver._ge_beam3_retained_fibre_state as fibre_state
from anyfesolver._ge_beam3_effective_deadline import (
    SCALE_VARIABLE, deadline_scale, effective_deadline_seconds)
from anyfesolver._ge_beam3_retained_fibre_state import Context
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_retained_fibre import make_model


def program():
    return ForceProgram((.25, .5, 1., .5, 0., -.5, 0.), ((5, .4, -.005, 0.),), max_iterations=24)


def test_default_scale_reproduces_base_and_reads_per_call(monkeypatch):
    monkeypatch.delenv(SCALE_VARIABLE, raising=False)
    assert deadline_scale() == 1.0
    assert effective_deadline_seconds(120.) == 120.
    assert effective_deadline_seconds(60) == 60
    monkeypatch.setenv(SCALE_VARIABLE, '4')
    assert effective_deadline_seconds(120.) == 480.
    assert effective_deadline_seconds(60) == 240
    monkeypatch.setenv(SCALE_VARIABLE, '1')
    assert effective_deadline_seconds(120.) == 120.


@pytest.mark.parametrize('raw', ['0.5', '0', '8.5', '9', '-1', 'abc', 'nan', 'inf', ''])
def test_out_of_range_or_unparseable_scale_fails_closed(monkeypatch, raw):
    monkeypatch.setenv(SCALE_VARIABLE, raw)
    with pytest.raises(ValueError):
        effective_deadline_seconds(120.)


def test_scaled_context_expiry_cancellation_and_rollback_safety(monkeypatch):
    monkeypatch.delenv(SCALE_VARIABLE, raising=False)
    token = CancellationToken()
    clock = {'now': 1000.}
    monkeypatch.setattr(fibre_state, 'monotonic', lambda: clock['now'])
    context = Context(make_model(1.), program(),
                      check=lambda: token.raise_if_cancelled('retained-fibre.material'))
    started = clock['now']; accepted = context.initial
    context.guard()
    # A declared bounded execution budget admits work beyond the unscaled
    # threshold; the failure does not move earlier.
    monkeypatch.setenv(SCALE_VARIABLE, '2')
    clock['now'] = started + 121.
    context.guard()
    # Expiry still refuses fail-closed at the scaled budget, unchanged message.
    clock['now'] = started + 241.
    with pytest.raises(RuntimeError, match='retained fibre context cooperative deadline'):
        context.guard()
    # Rollback safety: the refusal leaves the accepted state intact and the
    # ordinary default budget still refuses beyond the unscaled threshold.
    clock['now'] = started + 121.
    monkeypatch.setenv(SCALE_VARIABLE, '1')
    with pytest.raises(RuntimeError, match='retained fibre context cooperative deadline'):
        context.guard()
    clock['now'] = started
    context.guard()
    assert context.initial is accepted and context.initial.completed_targets == 0
    # Cancellation fires before any deadline arithmetic regardless of scale.
    monkeypatch.setenv(SCALE_VARIABLE, '2')
    token.cancel('bounded local simulation')
    with pytest.raises(SolveCancelled):
        context.guard()


def test_nonfinite_base_and_overflowed_effective_deadline_are_refused(monkeypatch):
    monkeypatch.delenv(SCALE_VARIABLE, raising=False)
    # A nonfinite base previously disabled the finite guard; it is refused now.
    for base in (float('inf'), -float('inf'), float('nan'), True, False,
                '120', None, 0, -60):
        with pytest.raises(ValueError):
            effective_deadline_seconds(base)
    monkeypatch.setenv(SCALE_VARIABLE, '8')
    for base in (1e308, 10 ** 400):
        with pytest.raises(ValueError):
            effective_deadline_seconds(base)
    monkeypatch.setenv(SCALE_VARIABLE, '1')
    with pytest.raises(ValueError):
        effective_deadline_seconds(10 ** 400)
    assert effective_deadline_seconds(1e308) == 1e308
    assert effective_deadline_seconds(120.) == 120.
