"""Caller cancellation reaches real native stores; it cannot publish a trial."""
from hashlib import sha256

import numpy as np
import pytest

from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._ge_beam3_native_generalized_loading import _ACTIVE
from anyfesolver._ge_beam3_native_generalized_program import _PROGRAM
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_analysis import analysis, generalized_problem, fibre_problem, pattern


def make(family):
    return analysis((generalized_problem if family == 'generalized' else fibre_problem)(True, True)[0])


def solve(made, family, **options):
    if family == 'generalized':
        return made.solve_distributed(pattern(), **options)
    return made.solve_nodal(((3, .35, -.012, .006),), **options)


def unlocked(made):
    assert made._lock.acquire(blocking=False)
    made._lock.release()
    made._guard()
    assert _PROGRAM.get() is None and _ACTIVE.get() is None


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
@pytest.mark.parametrize('restart', (False, True))
def test_cancelled_before_initialization_or_checkpoint_decode(family, restart, monkeypatch):
    made = make(family)
    def forbidden(*a, **k):
        raise AssertionError('cancelled operation entered state machinery')
    monkeypatch.setattr(made, '_initial', forbidden)
    monkeypatch.setattr(made, '_backend', forbidden)
    token = CancellationToken()
    token.cancel('caller stopped before capture')
    options = dict(checkpoint=b'invalid', expected_sha256='0'*64) if restart else {}
    with pytest.raises(SolveCancelled, match='before capture'):
        solve(made, family, cancellation_token=token, **options)
    unlocked(made)


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
def test_cancel_during_unaccepted_trial_preserves_both_stores(family, monkeypatch):
    made = make(family)
    element = made._elements[0]
    original = element.compute_nonlinear_response
    captured = []
    token = CancellationToken()
    def cancel_trial(*a, **k):
        result = original(*a, **k)
        context, view = k['native_material_context'], k['native_rotation_trial']
        if not captured and not np.array_equal(view.trial_coordinates, view.committed_coordinates):
            captured.append((context.store, canonical(context.store.materialize()),
                             context.store.generation, context.store.native_rotation_store.generation))
            token.cancel('stop unaccepted native force trial')
        return result
    monkeypatch.setattr(element, 'compute_nonlinear_response', cancel_trial)
    with pytest.raises(SolveCancelled):
        solve(made, family, cancellation_token=token)
    assert captured
    store, before, generation, rotation_generation = captured[0]
    assert canonical(store.materialize()) == before
    assert store.generation == generation
    assert store.native_rotation_store.generation == rotation_generation
    assert not store.has_active_trial and not store.native_rotation_store.has_active_trial
    unlocked(made)


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
def test_cancel_from_final_progress_never_issues_checkpoint(family, monkeypatch):
    made = make(family)
    token, events = CancellationToken(), []
    def progress(event):
        events.append(event)
        token.cancel('stop at accepted-step observation')
    def forbidden(*a, **k):
        raise AssertionError('cancelled force solve published a checkpoint')
    monkeypatch.setattr(made, '_envelope', forbidden)
    with pytest.raises(SolveCancelled):
        solve(made, family, steps=1, cancellation_token=token, progress_callback=progress)
    assert len(events) == 1 and events[0]['type'] == 'nonlinear_static_step'
    unlocked(made)


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
def test_observation_does_not_change_science_or_restart(family, tmp_path):
    first = solve(make(family), family)
    events = []
    made = make(family)
    observed = solve(made, family, cancellation_token=CancellationToken(), progress_callback=events.append)
    assert first.status == observed.status == 'completed'
    assert first.checkpoint == observed.checkpoint
    assert events and all(event['type'] == 'nonlinear_static_step' for event in events)
    before = made.recover(observed.checkpoint, expected_sha256=observed.checkpoint_sha256)
    token = CancellationToken()
    token.cancel('do not resume')
    with pytest.raises(SolveCancelled):
        solve(made, family, checkpoint=observed.checkpoint,
              expected_sha256=observed.checkpoint_sha256, cancellation_token=token)
    assert canonical(made.recover(observed.checkpoint, expected_sha256=observed.checkpoint_sha256)) == canonical(before)
    assert sha256(observed.checkpoint).hexdigest() == observed.checkpoint_sha256
    unlocked(made)
    (tmp_path/'checkpoint.json').write_bytes(observed.checkpoint)


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
def test_invalid_observer_rejected_before_initialization(family, monkeypatch):
    made = make(family)
    def forbidden(*a, **k):
        raise AssertionError('invalid observer entered mechanics')
    monkeypatch.setattr(made, '_initial', forbidden)
    with pytest.raises(ValueError, match='callable'):
        solve(made, family, progress_callback=42)
    unlocked(made)
