from hashlib import sha256
from time import monotonic
import numpy as np
import pytest
from anyfesolver import _ge_beam3_operator_validation_scope as scope
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_elastic_seed_continuation import seed, model, programme, solve, context
from test_ge_beam3_elastic_seed_continuation import test_owner_mutations as owner_mutation
from test_ge_beam3_elastic_seed_continuation import test_rehashed_checkpoint_mutations as checkpoint_mutation
from test_ge_beam3_elastic_seed_continuation import test_cancel_retains_original_accepted_prefix as cancel_prefix


def test_boundary_return_and_mutation_rejection_even_on_error():
    checks=[]; state=[0]; marker=object()
    def full():
        checks.append(1)
        if state[0]: raise ValueError('boundary mutation')
    assert scope._boundary_call(lambda check: marker, full, lambda: None) is marker
    assert checks == [1,1]
    def fail(check):
        state[0]=1
        raise RuntimeError('calculation failed')
    with pytest.raises(ValueError, match='boundary mutation'):
        scope._boundary_call(fail, full, lambda: None)
    assert checks == [1,1,1,1]
    called=[]
    with pytest.raises(ValueError):
        scope._boundary_call(lambda check: called.append(1), full, lambda: None)
    assert called == []


def test_scope_nested_and_exception_cleanup():
    assert scope._ACTIVE.get() is None
    with pytest.raises(RuntimeError):
        with scope.operator_validation_scope():
            with pytest.raises(ValueError, match='nested'):
                with scope.operator_validation_scope(): pass
            raise RuntimeError('sentinel')
    assert scope._ACTIVE.get() is None


def test_default_callback_path_unchanged():
    class Fake:
        def recover(self, *args, check=None, **kw):
            check(); return args, kw, check
    checks=[]
    def full(): checks.append(1)
    a=scope.operator_call(Fake(), 'recover', full, monotonic(), 1, origin=2)
    assert a == ((1,),dict(origin=2),full) and checks == [1]
    with scope.operator_validation_scope(), pytest.raises(ValueError, match='exact sealed'):
        scope.operator_call(Fake(), 'recover', full, monotonic())


def test_real_local_equal_and_full_check_reduction(seed):
    owner=context(seed); p=owner.physical; op=p.elements[0][1].operator
    args=(owner.initial.mechanical.cell_rotations[0], owner.initial.mechanical.resultants[0])
    counts=dict(original=0, boundary=0)
    def original(): counts['original']+=1; owner.guard()
    def boundary(): counts['boundary']+=1; owner.guard()
    expected=scope.operator_call(op,'recover',original,p.started,*args,origin=owner.initial.origins[0])
    with scope.operator_validation_scope():
        actual=scope.operator_call(op,'recover',boundary,p.started,*args,origin=owner.initial.origins[0])
    assert canonical(actual)==canonical(expected)
    assert counts['boundary']==2 and counts['original']>counts['boundary']


@pytest.mark.parametrize('kind',('deadline','cancellation','model'))
def test_real_failure_does_not_return_result(seed,monkeypatch,kind):
    owner=context(seed); p=owner.physical; op=p.elements[0][1].operator
    token=CancellationToken(); calls=[]
    def full(): calls.append(1); owner.guard()
    if kind=='deadline':
        from anyfesolver._ge_beam3_effective_deadline import effective_deadline_seconds
        monkeypatch.setattr(scope,'monotonic',lambda:p.started+effective_deadline_seconds(120.)+.01)
    if kind=='cancellation': token.cancel()
    if kind=='model':
        original=type(op).recover
        def mutate(this,*a,**kw):
            answer=original(this,*a,**kw)
            p.model.mesh.nodes[2].x += .01
            return answer
        monkeypatch.setattr(type(op),'recover',mutate)
    with scope.operator_validation_scope(cancellation_token=token), pytest.raises((RuntimeError,ValueError,SolveCancelled)):
        scope.operator_call(op,'recover',full,p.started,owner.initial.mechanical.cell_rotations[0],
            owner.initial.mechanical.resultants[0],origin=owner.initial.origins[0])
    assert calls == [1,1]


def test_actual_load_reverse_restart_and_state_bytes_equal(seed):
    baseline=solve(seed)
    assert baseline.status=='completed',baseline.failure
    with scope.operator_validation_scope():
        optimized=solve(seed)
        prefix=solve(seed,stop_after=2)
        resumed=solve(seed,checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
        replay=context(seed); state,records=replay.restore(optimized.checkpoint,
            expected_sha256=sha256(optimized.checkpoint).hexdigest())
        before=canonical(state); replay.recover(state)
        assert canonical(state)==before and replay.checkpoint(records)==optimized.checkpoint
    assert optimized.status==resumed.status=='completed'
    assert baseline.checkpoint==optimized.checkpoint==resumed.checkpoint
    assert canonical(baseline.state)==canonical(optimized.state)==canonical(resumed.state)


@pytest.mark.parametrize('kind',('clone','foreign','state','map','program','identity','seed','target'))
def test_all_owner_mutations_still_rejected(seed,kind):
    with scope.operator_validation_scope(): owner_mutation(seed,kind)


@pytest.mark.parametrize('kind',('parameter','work','recovery','origins','cursor_bool','schema','seed_hash','path_claim','genesis'))
def test_rehashed_checkpoint_mutations_still_rejected(seed,kind):
    with scope.operator_validation_scope(): checkpoint_mutation(seed,kind)


@pytest.mark.parametrize('when',('before_commit','committed'))
def test_actual_cancellation_keeps_expected_prefix(seed,when):
    with scope.operator_validation_scope(): cancel_prefix(seed,when)
