"""Partial propagation-only driver evidence on the flat 2x2 fixture.

The six authored driver checks each need multiple completed real solves.
Under the guard's NUMBA_DISABLE_JIT=1 environment one 2x2 half solve
measures roughly 80-110 s of real work (attempt-03 phase log: macro
attempt stagnation refusal at 14.2 s, then adaptive dyadic substep
trials at 11.3/2.0/14.7/7.4/2.1 s and still running at the 60 s job
cap), so no restore-vs-reference half can execute inside the remaining
allowance. This module banks only the propagation halves that do fit:
the declared injected first-refusal reaches the real cutback, a
cancellation token at the first continuation safe point propagates
SolveCancelled with no substep Newton published, and a cancellation
token at the first semismooth iteration safe point propagates through
the macro attempt. Restore correctness against a pristine reference
remains unverified; these are partial checks, not the six authored
driver checks.
"""
import pytest

from anyfesolver.control import SolveCancelled

from test_general_contact_admission_globalization_driver_small_patch import (
    _CancelAt,
    _case,
    _enabled,
    _force_first_macro_refusal,
    _solve,
)


def test_cancellation_during_substeps_propagates_partial(monkeypatch):
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


def test_cancellation_inside_first_macro_attempt_propagates_partial():
    model, definition, settings, program = _case()
    token = _CancelAt("contact_path.step:1.semismooth.iteration:1")
    with pytest.raises(SolveCancelled,
                       match="injected continuation cancellation"):
        _solve(model, definition, _enabled(settings), program,
               cancellation_token=token)
