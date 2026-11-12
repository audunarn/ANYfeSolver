"""Adaptive lifecycle and matched-increment controls; no new material oracle."""
from dataclasses import replace
from hashlib import sha256
import json

import anyfesolver
import numpy as np
import pytest

from anyfesolver.stiffened_shell import (
    AdaptiveIncrementControl, AdaptiveIncrementExhausted, FiniteIncrementFailure,
    IncrementAttempt, NodalAction,
)
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_ge_finite_experimental import FinitePointDriver
from anyfesolver._stiffened_shell_increment_control import decode_control, encode_control, MAGIC
from test_stiffened_shell_ge_physical_fibre_linear import _definition


ACTIONS = (NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),)


def _bounded_increment(monkeypatch):
    """Controlled failure injection tests scheduling; real local solves follow."""
    original = FinitePointDriver._solve_fixed
    observed = []
    def fixed(driver, actions, targets, **kwargs):
        factor = targets[0]
        observed.append((float(factor),driver._load_factor,driver.owner._generation))
        if factor-driver._load_factor > .251:
            raise FiniteIncrementFailure(IncrementAttempt(float(factor),
                float(driver._load_factor),float(factor),'rejected',1,2,1,1,
                1.,0.,'finite mixed backtrack budget exhausted'))
        return original(driver,actions,targets,**kwargs)
    monkeypatch.setattr(FinitePointDriver,'_solve_fixed',fixed)
    return observed


def test_cutbacks_use_committed_origins_and_match_fixed_increment_oracle(monkeypatch):
    observed = _bounded_increment(monkeypatch)
    control = AdaptiveIncrementControl(max_cutbacks=3,max_attempts=12)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    oracle = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    try:
        result = owner.solve(ACTIONS,targets=(1.,),increment_control=control)
        assert result.status == 'completed'
        assert result.accepted_factors == (.25,.5,.75,1.)
        assert observed[:3] == [(1.,0.,0),(.5,0.,0),(.25,0.,0)]
        assert [r.status for r in result.increment_attempts] == ['rejected']*2+['accepted']*4
        assert all(r.target == 1. for r in result.increment_attempts)
        matched = oracle.solve(ACTIONS,targets=result.accepted_factors)
        state,inner = decode_control(owner.checkpoint(),owner._identity)
        assert state.completed and inner == oracle.checkpoint()
        np.testing.assert_array_equal(result.cell_resultants,matched.cell_resultants)
        for actual,expected in zip(result.beam_fibre_material,matched.beam_fibre_material):
            assert actual.fibres == expected.fibres
            np.testing.assert_array_equal(actual.strain,expected.strain)
            np.testing.assert_array_equal(actual.resultants,expected.resultants)
    finally:
        owner.close();oracle.close()


def test_cancelled_prefix_restores_pending_cursor_and_chart(monkeypatch):
    _bounded_increment(monkeypatch)
    import anyfesolver._stiffened_shell_ge_finite_experimental as finite
    original = finite.cancellation_safe_point
    token = CancellationToken()
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition(False,True))
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=_definition(False,True))
    control = AdaptiveIncrementControl(3,12)
    def cancel(token_value,stage):
        if owner._generation == 1 and stage == 'stiffened-shell.finite.increment':
            token.cancel('adaptive prefix')
        return original(token_value,stage)
    try:
        with monkeypatch.context() as context:
            context.setattr(finite,'cancellation_safe_point',cancel)
            with pytest.raises(SolveCancelled,match='adaptive prefix'):
                owner.solve(ACTIONS,targets=(1.,),increment_control=control,cancellation_token=token)
        prefix = owner.recover()
        assert prefix.accepted_factors == (.25,) and prefix.status == 'accepted_prefix'
        rebased = owner.rebase_rotations(expected_generation=prefix.generation)
        saved = owner.checkpoint()
        state,inner = decode_control(saved,owner._identity)
        assert state.depth == 2 and state.cursor == 0
        restored = resumed.restore(saved)
        assert restored.status == 'accepted_prefix'
        assert restored.increment_attempts == rebased.increment_attempts
        assert resumed.checkpoint() == saved
        start=len(MAGIC)+4
        size=int.from_bytes(saved[len(MAGIC):start],'little')
        header=json.dumps(json.loads(saved[start:start+size]),indent=2).encode()
        noncanonical=MAGIC+len(header).to_bytes(4,'little')+header+saved[start+size:-32]
        noncanonical += sha256(noncanonical).digest()
        untouched = anyfesolver.stiffened_shell.create_analysis(definition=_definition(False,True))
        try:
            with pytest.raises(ValueError,match='noncanonical'):
                untouched.restore(noncanonical)
            assert untouched._generation == 0 and untouched._finite_driver is None
        finally:untouched.close()
        for kwargs in ({}, {'increment_control':control,'targets':(.75,)},
                       {'increment_control':control,'max_iterations':13}):
            with pytest.raises(ValueError,match='matching targets and controls'):
                resumed.solve(ACTIONS,**kwargs)
        with pytest.raises(ValueError,match='load pattern changed'):
            resumed.solve((NodalAction('beam',105,(0.,0.,999.,0.,0.,0.)),),
                          targets=(1.,),increment_control=control)
        assert resumed.checkpoint() == saved
        bad = encode_control(owner._identity,replace(state,cursor=1),inner)
        untouched = anyfesolver.stiffened_shell.create_analysis(definition=_definition(False,True))
        try:
            with pytest.raises(ValueError,match='cursor disagrees'):
                untouched.restore(bad)
            assert untouched._generation == 0 and untouched._finite_driver is None
        finally:untouched.close()
        direct = owner.solve(ACTIONS,targets=(1.,),increment_control=control)
        continued = resumed.solve(ACTIONS,targets=(1.,),increment_control=control)
        assert direct.accepted_factors == continued.accepted_factors == (.25,.5,.75,1.)
        assert direct.increment_attempts == continued.increment_attempts
        assert owner.checkpoint() == resumed.checkpoint()
    finally:
        owner.close();resumed.close()


def test_exhaustion_is_not_equilibrium(monkeypatch):
    _bounded_increment(monkeypatch)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    try:
        with pytest.raises(AdaptiveIncrementExhausted,match='cutback limit') as caught:
            owner.solve(ACTIONS,targets=(1.,),increment_control=AdaptiveIncrementControl(1,12))
        assert caught.value.accepted_factors == ()
        assert len(caught.value.attempts) == 2
        assert isinstance(caught.value.__cause__,FiniteIncrementFailure)
        assert owner._generation == 0 and owner._finite_driver is None
        with pytest.raises(RuntimeError,match='no accepted'):
            owner.checkpoint()
    finally:owner.close()


def test_checkpoint_resource_failure_precedes_atomic_publish(monkeypatch):
    import anyfesolver._stiffened_shell_increment_control as control_module
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    try:
        first = owner.solve(ACTIONS,targets=(.25,))
        saved = owner.checkpoint()
        monkeypatch.setattr(control_module,'MAX_CHECKPOINT_BYTES',len(saved)+32)
        with pytest.raises(ValueError,match='adaptive checkpoint exceeds'):
            owner.solve(ACTIONS,targets=(.5,),increment_control=AdaptiveIncrementControl())
        assert owner.recover() is first and owner._generation == first.generation
        assert owner.checkpoint() == saved
    finally:owner.close()


def test_attempt_budget_keeps_durable_prefix_and_is_terminal(monkeypatch):
    _bounded_increment(monkeypatch)
    control = AdaptiveIncrementControl(3,3)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    resumed = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    try:
        with pytest.raises(AdaptiveIncrementExhausted,match='attempt limit') as caught:
            owner.solve(ACTIONS,targets=(1.,),increment_control=control)
        assert caught.value.accepted_factors == (.25,)
        assert len(caught.value.attempts) == 3
        saved = owner.checkpoint()
        prefix = resumed.restore(saved)
        assert prefix.status == 'accepted_prefix' and resumed.checkpoint() == saved
        with pytest.raises(AdaptiveIncrementExhausted,match='attempt limit'):
            resumed.solve(ACTIONS,targets=(1.,),increment_control=control)
        assert resumed.recover() is prefix and resumed.checkpoint() == saved
    finally:owner.close();resumed.close()


def test_cancelled_prepared_controller_snapshot_is_not_published(monkeypatch):
    import anyfesolver._stiffened_shell_ge_finite_experimental as finite
    original = finite.cancellation_safe_point
    token = CancellationToken()
    def cancel(token_value,stage):
        if stage == 'stiffened-shell.finite.controller-publish':
            token.cancel('prepared checkpoint')
        return original(token_value,stage)
    monkeypatch.setattr(finite,'cancellation_safe_point',cancel)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    try:
        with pytest.raises(SolveCancelled,match='prepared checkpoint'):
            owner.solve(ACTIONS,targets=(.25,),increment_control=AdaptiveIncrementControl(),
                        cancellation_token=token)
        assert owner._generation == 0 and owner._finite_driver is None
        with pytest.raises(RuntimeError,match='no accepted'):
            owner.checkpoint()
    finally:owner.close()


@pytest.mark.parametrize('error', (MemoryError('memory'),ValueError('programming'),
                                   RuntimeError('backend failure')))
def test_unclassified_errors_are_not_retried(monkeypatch,error):
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(FinitePointDriver,'_solve_fixed',fail)
    owner = anyfesolver.stiffened_shell.create_analysis(definition=_definition())
    try:
        with pytest.raises(type(error)) as caught:
            owner.solve(ACTIONS,increment_control=AdaptiveIncrementControl())
        assert caught.value is error and len(calls)==1
        assert owner._generation == 0 and owner._finite_driver is None
    finally:owner.close()


@pytest.mark.parametrize('kwargs', ({'max_cutbacks':True},{'max_cutbacks':-1},
    {'max_attempts':0},{'max_attempts':1.5}))
def test_typed_controls_reject_invalid_resource_choices(kwargs):
    with pytest.raises(ValueError,match='bounded integer'):
        AdaptiveIncrementControl(**kwargs)
