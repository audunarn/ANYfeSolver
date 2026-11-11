"""Operation-owned cleanup must not discard another solve's candidate."""

import pytest

from anyfesolver.nonlinear_state import (
    NonlinearStateStore, StateTransactionError,
    _register_nonlinear_state_cleanup, _run_with_nonlinear_state_cleanup,
)


def store(): return NonlinearStateStore.from_shell_layouts((), {1: {'history': 1}})


def test_normal_return_discards_only_uncommitted_owned_trial():
    owned=store();unrelated=store();other_token=unrelated.begin_trial()
    def operation():
        assert _register_nonlinear_state_cleanup(owned) is owned
        _register_nonlinear_state_cleanup(owned)
        token=owned.begin_trial();owned.set_trial_state(token, 1, {'history': 2})
        return 42
    assert _run_with_nonlinear_state_cleanup(operation)==42
    assert not owned.has_active_trial and owned.generation==0 and owned[1]['history']==1
    assert unrelated.has_active_trial;unrelated.discard_trial(other_token)


def test_nested_scopes_keep_parent_ownership_and_propagate_error():
    parent=store();child=store()
    def inner():
        _register_nonlinear_state_cleanup(child);child.begin_trial()
        raise RuntimeError('inner failure')
    def outer():
        _register_nonlinear_state_cleanup(parent);parent.begin_trial()
        with pytest.raises(RuntimeError, match='inner failure'): _run_with_nonlinear_state_cleanup(inner)
        assert parent.has_active_trial and not child.has_active_trial
    _run_with_nonlinear_state_cleanup(outer)
    assert not parent.has_active_trial


def test_cleanup_failure_does_not_skip_other_owned_store(monkeypatch):
    first=store();second=store();original=second.discard_trial
    def fail(token): raise RuntimeError('injected discard failure')
    monkeypatch.setattr(second, 'discard_trial', fail)
    def operation():
        for value in (first, second): _register_nonlinear_state_cleanup(value);value.begin_trial()
    with pytest.raises(StateTransactionError, match='cleanup failed'): _run_with_nonlinear_state_cleanup(operation)
    assert not first.has_active_trial and second.has_active_trial
    original(second.active_trial_token())


def test_successful_commit_is_not_rolled_back_by_cleanup():
    value=store()
    def operation():
        _register_nonlinear_state_cleanup(value);token=value.begin_trial()
        value.set_trial_state(token, 1, {'history': 2});value.commit(token)
    _run_with_nonlinear_state_cleanup(operation)
    assert value.generation==1 and value[1]['history']==2 and not value.has_active_trial


def test_nonstore_registration_rejected():
    with pytest.raises(TypeError): _register_nonlinear_state_cleanup({})
