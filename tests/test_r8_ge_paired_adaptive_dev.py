"""Prepared controls: injected mechanics, actual controller/checkpoint callers."""
from dataclasses import dataclass,replace
from hashlib import sha256
from threading import RLock
from types import SimpleNamespace as NS
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver import _stiffened_shell_displacement_adaptive as adaptive
from anyfesolver import _stiffened_shell_affine_endpoint as endpoint
from anyfesolver import _stiffened_shell_displacement_control as fixed
from anyfesolver._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC

SEED=COMPENSATED_MAGIC+b'injected original seed'
DIGEST=sha256(SEED).hexdigest()

@dataclass(frozen=True)
class Result:
    status:str='completed'
    increment_attempts:tuple=()

class FakeOwner:
    _identity='paired-control-fixture'
    _shell_count=12
    _beam_ids=()
    _shell_index={1:0,2:1}
    _beam_index={}
    _free=np.arange(12)
    _bl5_shell_material=False
    def __init__(self):
        self._generation=9;self._accepted=None;self._commit_lock=RLock()
        self._displacement_snapshot=None;self._displacement_adaptive=None
        self._finite_driver=NS(_accepted=Result(),_state=NS(q=np.zeros(12)),
            _pattern=np.ones(12),_programme=None,_external=None)
        self._finite_driver._load=lambda actions:np.ones(12)
    def checkpoint(self):
        return self._displacement_adaptive[1] if self._displacement_adaptive else SEED
    def recover(self):return self._finite_driver._accepted

@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(endpoint,'is_native_point_material',lambda o:True)
    monkeypatch.setattr(endpoint,'is_paired_displacement',lambda o:True)
    return FakeOwner()

def context(owner):
    state=adaptive.DisplacementIncrementState(ss.AdaptiveIncrementControl(2,8),(-1.,0.,1.),0.)
    return adaptive.Context(state,6,adaptive.direction_hash(owner._finite_driver._pattern),0,20,8,
                            force_seed_sha256=DIGEST)

def reject(owner):
    return ss.DisplacementCorrectorFailure('injected numerical rejection',
        ss.DisplacementFailureDetail(owner._generation,-1.,0.,1,1,1,1,1.,0.,.1))

def test_source_paired_identity_and_legacy_header(owner,monkeypatch):
    c=context(owner);raw=adaptive.encode(owner._identity,c,SEED,native=True,paired=True)
    assert raw.startswith(adaptive.PAIRED_MAGIC)
    decoded,inner=adaptive.decode(owner,raw)
    assert decoded==c and inner==SEED
    legacy=adaptive.encode(owner._identity,replace(c,force_seed_sha256=None),b'legacy')
    assert legacy.startswith(adaptive.MAGIC) and b'force_seed_sha256' not in legacy
    with pytest.raises(ValueError,match='formulation identity'):adaptive.decode(owner,legacy)
    # Golden bytes come from the immutable original wheel's encode function,
    # rather than reconstructing expected bytes with this candidate's encoder.
    from scripts.r8_ge_paired_adaptive_dev import BASE_DOC as DOC
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC as FORCE_MAGIC
    inner=FORCE_MAGIC+b'golden-inner-framing-only'
    digests={'legacy':'e02fa7e466ff996f259b0aa4dd818b1302fb912f822f31e1f6dc2677bca5f20e',
             'native':'4eb46e0f4ef4005c02a4933606e468341cf3e486aa2e90d3bcd521371775e374',
             'event':'4407bd44923706a836dbd1bff29dc193ea1b82cc48f2295c6c2e23d01d4c2fd5'}
    if not all((DOC/'legacy-golden-v1'/(label+'-adaptive.bin')).exists()
            for label in ('legacy','native','event')):
        pytest.skip('saved legacy-golden adaptive binaries '
            '(docs/agent_plans/development/r8-ge-paired-adaptive-20261010/'
            'legacy-golden-v1) are absent; the golden-byte identity control '
            'requires the exact saved binaries')
    with monkeypatch.context() as patch:
        patch.setattr(endpoint,'is_paired_displacement',lambda o:False)
        for label,native,policy in (('legacy',False,None),('native',True,None),('event',False,ss.GENERALIZED_BRANCH_EVENT_TRIAL)):
            patch.setattr(endpoint,'is_native_point_material',lambda o,value=native:value)
            owner._bl5_shell_material=not native
            expected=(DOC/'legacy-golden-v1'/(label+'-adaptive.bin')).read_bytes()
            assert sha256(expected).hexdigest()==digests[label]
            old_context=replace(c,force_seed_sha256=None,trial_policy=policy)
            assert adaptive.encode(owner._identity,old_context,inner,native=native)==expected
            decoded,restored_inner=adaptive.decode(owner,expected)
            assert decoded==old_context and restored_inner==inner

def test_source_digest_event_family_and_corruption_refusal(owner,monkeypatch):
    c=context(owner);raw=adaptive.encode(owner._identity,c,SEED,paired=True)
    with pytest.raises(ValueError):adaptive.decode(owner,raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError):adaptive.encode(owner._identity,replace(c,force_seed_sha256='0'*64),SEED,paired=True)
    with pytest.raises(ValueError):replace(c,trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL)
    with pytest.raises(ValueError):adaptive.encode(owner._identity,c,raw,paired=True)
    monkeypatch.setattr(endpoint,'is_paired_displacement',lambda o:False)
    with pytest.raises(ValueError,match='formulation identity'):adaptive.decode(owner,raw)

def test_source_seed_survives_rejected_wrapper(owner):
    c=context(owner)
    owner._displacement_adaptive=(c,adaptive.encode(owner._identity,c,SEED,paired=True))
    adaptive.require_native_seed(owner,owner._finite_driver,force_seed_sha256=DIGEST)
    assert adaptive.inner_checkpoint(owner)==SEED
    with pytest.raises(ValueError):adaptive.require_native_seed(owner,owner._finite_driver,force_seed_sha256='0'*64)

def test_source_progress_direction_and_cursor_refusal(owner):
    c=context(owner);adaptive.validate_progress(owner,c)
    with pytest.raises(ValueError,match='load/control'):adaptive.validate_progress(owner,replace(c,direction_sha256='0'*64))
    accepted=c.state.record(True,evaluations=1,linear_solves=1,rejected_trials=0)
    with pytest.raises(ValueError,match='seed mismatch'):adaptive.validate_progress(owner,replace(c,state=accepted))

def test_source_typed_cutback_and_durable_limit(owner,monkeypatch):
    monkeypatch.setattr(fixed,'solve',lambda *a,**k:(_ for _ in ()).throw(reject(owner)))
    control=ss.DisplacementControl('shell',2,0,force_seed_sha256=DIGEST)
    with pytest.raises(ss.DisplacementIncrementLimit):
        adaptive.solve(owner,(),(-1.,),control,ss.AdaptiveIncrementControl(0,1),max_iterations=20,max_backtracks=8)
    c,inner=adaptive.decode(owner,owner.checkpoint())
    assert len(c.state.attempts)==1 and not c.state.attempts[0].accepted and inner==SEED
    assert owner._generation==9 and owner._finite_driver._state.q[6]==0.
    assert owner.recover().status=='accepted_prefix'

def test_source_cancellation_and_untyped_failures_not_cutback(owner,monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    token=CancellationToken();token.cancel('injected cancellation')
    for failure in (ValueError('admission'),reject(owner).__class__('missing detail')):
        def fail(*a,**k):raise failure
        monkeypatch.setattr(fixed,'solve',fail)
        with pytest.raises(type(failure)):
            adaptive.solve(owner,(),(-1.,),ss.DisplacementControl('shell',2,0,force_seed_sha256=DIGEST),
                ss.AdaptiveIncrementControl(),max_iterations=20,max_backtracks=8)
        assert owner._displacement_adaptive is None and owner.checkpoint()==SEED
    monkeypatch.setattr(fixed,'solve',lambda *a,**k:token.raise_if_cancelled('injected'))
    with pytest.raises(SolveCancelled):
        adaptive.solve(owner,(),(-1.,),ss.DisplacementControl('shell',2,0,force_seed_sha256=DIGEST),
            ss.AdaptiveIncrementControl(),max_iterations=20,max_backtracks=8)
    assert owner._displacement_adaptive is None

def test_source_postpublication_observer_failure_not_cutback(owner,monkeypatch):
    def fail(*a,**k):owner._generation+=1;raise reject(owner)
    monkeypatch.setattr(fixed,'solve',fail)
    with pytest.raises(ss.DisplacementCorrectorFailure):
        adaptive.solve(owner,(),(-1.,),ss.DisplacementControl('shell',2,0,force_seed_sha256=DIGEST),
            ss.AdaptiveIncrementControl(),max_iterations=20,max_backtracks=8)
    assert owner._displacement_adaptive is None

def test_source_restore_atomic_cancel_stale_and_progress(owner,monkeypatch):
    c=context(owner);raw=adaptive.encode(owner._identity,c,SEED,paired=True)
    owner._finite_driver=None
    def restore(self,inner):
        source=FakeOwner();self._finite_driver=source._finite_driver
        return source.recover()
    monkeypatch.setattr(FakeOwner,'restore',restore,raising=False)
    from anyfesolver.control import CancellationToken,SolveCancelled
    token=CancellationToken();token.cancel('before restore')
    with pytest.raises(SolveCancelled):adaptive.restore_paired(owner,raw,cancellation_token=token)
    assert owner._finite_driver is None and owner._generation==9
    bad=adaptive.encode(owner._identity,replace(c,direction_sha256='0'*64),SEED,paired=True)
    with pytest.raises(ValueError):adaptive.restore_paired(owner,bad)
    assert owner._finite_driver is None
    original=adaptive.validate_progress
    def stale(prepared,context):original(prepared,context);owner._generation+=1
    monkeypatch.setattr(adaptive,'validate_progress',stale)
    with pytest.raises(ValueError,match='stale'):adaptive.restore_paired(owner,raw)
    assert owner._finite_driver is None and owner._displacement_adaptive is None
    monkeypatch.setattr(adaptive,'validate_progress',original)
    class CancelPublish(CancellationToken):
        def raise_if_cancelled(self,stage=''):
            if stage.endswith('restore.publish'):self.cancel('after isolated reconstruction')
            super().raise_if_cancelled(stage)
    with pytest.raises(SolveCancelled):adaptive.restore_paired(owner,raw,cancellation_token=CancelPublish())
    assert owner._finite_driver is None and owner._displacement_adaptive is None
    result=adaptive.restore_paired(owner,raw)
    assert result.status=='accepted_prefix' and owner.checkpoint()==raw

def test_source_public_restore_dispatch_before_legacy(owner,monkeypatch):
    from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
    owner._finite_driver=None;owner._unsupported_modes=()
    owner._require_open=lambda:None;owner._require_stateful_scope=lambda:None
    raw=adaptive.encode(owner._identity,context(FakeOwner()),SEED,paired=True)
    actual_restore=adaptive.restore_paired
    calls=[];monkeypatch.setattr(adaptive,'restore_paired',lambda *a,**k:calls.append((a,k)) or 'dispatched')
    assert ExperimentalGEAnalysis.restore.__wrapped__(owner,raw)=='dispatched'
    assert calls[0][0]==(owner,raw)
    monkeypatch.setattr(adaptive,'restore_paired',actual_restore)
    def restore(self,inner):
        source=FakeOwner();self._finite_driver=source._finite_driver;return source.recover()
    monkeypatch.setattr(FakeOwner,'restore',restore,raising=False)
    bad_model=adaptive.encode('wrong model',context(FakeOwner()),SEED,paired=True)
    for invalid in (bad_model,raw[:-1]+bytes([raw[-1]^1])):
        with pytest.raises(ValueError):ExperimentalGEAnalysis.restore.__wrapped__(owner,invalid)
        assert owner._finite_driver is None and owner._generation==9
    with monkeypatch.context() as patch:
        patch.setattr(endpoint,'is_paired_displacement',lambda o:False)
        with pytest.raises(ValueError):ExperimentalGEAnalysis.restore.__wrapped__(owner,raw)
    assert owner._finite_driver is None and owner._displacement_adaptive is None
    from anyfesolver.control import CancellationToken
    with pytest.raises(ValueError,match='cancellation'):
        ExperimentalGEAnalysis.restore.__wrapped__(owner,b'legacy unsupported',cancellation_token=CancellationToken())
    assert owner._finite_driver is None and owner._displacement_adaptive is None
    assert ExperimentalGEAnalysis.restore.__wrapped__(owner,raw).status=='accepted_prefix'
    assert owner.checkpoint()==raw

def test_source_public_solve_adaptive_dispatch(owner,monkeypatch):
    from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import POLICY
    owner._beam_parts=((NS(operator=NS(arithmetic_policy=POLICY)),None,None),)
    for name in ('_require_open','_require_stateful_scope'):setattr(owner,name,lambda:None)
    owner._require_distributing_reference_only=lambda *a:None
    for name in ('_coordinate_support','_bl5_native_material','_finite_distributing','_curved_reference','_curved_material'):setattr(owner,name,False)
    owner._unsupported_modes=();owner._arc_snapshot=None
    calls=[];monkeypatch.setattr(adaptive,'solve',lambda *a,**k:calls.append((a,k)) or 'adaptive')
    control=ss.DisplacementControl('shell',2,0,force_seed_sha256=DIGEST);limits=ss.AdaptiveIncrementControl()
    assert ExperimentalGEAnalysis.solve.__wrapped__(owner,(),(-1.,),displacement_control=control,increment_control=limits)=='adaptive'
    assert calls[0][0][3:]==(control,limits)

def test_source_attempt_observer_persists_failure_and_restores(monkeypatch,tmp_path):
    from scripts.r8_ge_paired_adaptive_dev import AttemptObserver
    flags={k:True for k in __import__('scripts.r8_ge_coupled_cycle_dev',fromlist=['ATOMIC_KEYS']).ATOMIC_KEYS}
    helper=NS(capture=lambda o:'before',atomic_flags=lambda o,b:flags)
    monkeypatch.setattr(fixed,'solve',lambda *a,**k:(_ for _ in ()).throw(ValueError('refused')))
    original=fixed.solve
    with AttemptObserver(tmp_path,atomic=helper):
        with pytest.raises(ValueError):fixed.solve(NS(_generation=9),(),(-1.,),None)
    assert fixed.solve is original
    import json
    record=json.loads((tmp_path/'attempts.jsonl').read_text())
    assert record['flags']==flags and record['exception']['type']=='ValueError'
    flags['state_words']=False
    with AttemptObserver(tmp_path,atomic=helper):
        with pytest.raises(ValueError):fixed.solve(NS(_generation=9),(),(-1.,),None)
    record=json.loads((tmp_path/'attempts.jsonl').read_text().splitlines()[-1])
    from scripts.r8_ge_coupled_cycle_dev import check_atomic
    with pytest.raises(AssertionError):check_atomic(record['flags'])
    from scripts.r8_ge_paired_adaptive_dev import write
    write(tmp_path/'nonfinite-failure.json',{'observed_norm':float('inf')})
    assert json.loads((tmp_path/'nonfinite-failure.json').read_text())=={'observed_norm':{'nonfinite':'inf'}}

def test_source_explicit_inventory_and_adoption_refusal(tmp_path):
    from scripts.r8_ge_paired_adaptive_dev import SOURCE_NAMES,require_adoption,PLAN,DOC,BASE_DOC
    assert len(SOURCE_NAMES)==12 and len(set(SOURCE_NAMES))==12
    assert (PLAN['max_cutbacks'],PLAN['max_attempts'],PLAN['max_iterations'],PLAN['max_backtracks'])==(4,96,20,8)
    assert DOC==BASE_DOC/'execution-v4'
    with pytest.raises(ValueError):require_adoption({'status':'ADOPTED','plan':{}})

def test_installed_public_adaptive_cycle():
    import os
    from pathlib import Path
    from scripts.r8_ge_paired_adaptive_dev import connected,DOC
    if 'GE_PAIRED_ADAPTIVE_STAGE' not in os.environ or 'GE_PAIRED_ADAPTIVE_OUTPUT' not in os.environ:
        pytest.skip('installed-cycle adaptive binding absent')
    assert os.environ.get('GE_PAIRED_ADAPTIVE_STAGE')=='installed-cycle'
    assert Path(os.environ['GE_PAIRED_ADAPTIVE_OUTPUT'])==DOC/'attempt-installed-cycle1'
    connected(Path(os.environ['GE_PAIRED_ADAPTIVE_OUTPUT']))
