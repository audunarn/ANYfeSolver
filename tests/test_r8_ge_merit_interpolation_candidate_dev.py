"""Prepared post-application controls; never load candidate source as a shadow package."""
from dataclasses import replace
from hashlib import sha256
from threading import RLock
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.sparse import eye,diags
from anyfesolver import _stiffened_shell_displacement_corrector as corrector
from anyfesolver import _stiffened_shell_displacement_control as fixed
from anyfesolver import _stiffened_shell_displacement_adaptive as adaptive
from anyfesolver import _stiffened_shell_affine_endpoint as endpoint
from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_checkpoint_binary import COMPENSATED_MAGIC

POLICY='PAIRED_FIXED_WORK_QUADRATIC_V1'
SEED=COMPENSATED_MAGIC+b'injected original seed'
DIGEST=sha256(SEED).hexdigest()

def fixture(monkeypatch):
    monkeypatch.setattr(endpoint,'is_paired_displacement',lambda o:True)
    monkeypatch.setattr(endpoint,'is_native_point_material',lambda o:False)
    monkeypatch.setattr(endpoint,'uses_native_material_history',lambda o:False)
    monkeypatch.setattr(endpoint,'require_native_displacement_policy',lambda *a:None)
    owner=NS(_K=eye(6,format='csr'),_shell_count=6,_beam_ids=(),_free=np.arange(6),
             _coordinate_scale=np.ones(6),_constraint_scale=np.empty(0),
             _generation=9,_reference_length=1.,_commit_lock=RLock(),_bl5_shell_material=True,
             _shell_index={1:0},_beam_index={},_identity='paired-control-fixture',
             _displacement_snapshot=None,_displacement_adaptive=None)
    state=NS(q=np.zeros(6),multipliers=np.empty(0));pattern=np.array([1.,0.,0.,0.,0.,0.])
    driver=NS(owner=owner,_state=state,_shell_origins=object(),_beam_origins=object(),
              _accepted=object(),_external=None,_programme=None,_load_factor=0.,_pattern=pattern)
    driver._increment_origin=lambda s:s
    driver._load=lambda actions:pattern
    owner._finite_driver=driver;owner.checkpoint=lambda:SEED
    amounts=[]
    def advance(s,step,t):
        amounts.append(t);return NS(q=s.q+t*step,multipliers=s.multipliers)
    driver._advance=advance
    def assemble(s,*args,**kwargs):
        x=s.q[1];r=np.zeros(6);r[1]=1.-x+9000.*x*x
        return r,diags([1.,-1.+18000.*x,1.,1.,1.,1.],format='csr'),(0. if 0.<r[1]<1. else 1.),0.,(),()
    driver._assemble=assemble
    step=np.zeros(6);step[1]=1.
    monkeypatch.setattr(corrector,'bordered_correction',lambda *a:(step.copy(),0.))
    return owner,driver,amounts

def run(driver,policy=None,**kwargs):
    return corrector.correct(driver,base_load=np.zeros(6),load_direction=driver._pattern,
        amplitude=0.,control_dof=0,target=0.,max_iterations=20,max_backtracks=8,
        **({} if policy is None else dict(merit_globalization=policy)),**kwargs)

def test_source_quadratic_scale_and_invalid_fallback():
    fn=corrector._quadratic_merit_amount
    # Exact squared-merit quadratic sampled at a rejected alpha.
    assert fn(1.,2.,-2.,1.)==pytest.approx(.2)
    assert fn(1e150,2e150,-2.,1.)==pytest.approx(.2)
    assert fn(1e-150,2e-150,-2.,1.)==pytest.approx(.2)
    for args in ((1.,2.,0.,1.),(1.,2.,float('nan'),1.),(0.,2.,-2.,1.),(1.,float('inf'),-2.,1.)):
        assert fn(*args)==.5
    assert 0.<fn(1.,9000.,-2.,1.)<1.

def test_source_corrector_actual_acceptance_full_first_and_default(monkeypatch):
    owner,driver,amounts=fixture(monkeypatch);before=driver._state
    trial=run(driver,POLICY)
    assert amounts[0]==1. and len(amounts)==2 and 0.<amounts[1]<1/9000.
    assert trial.evaluation[0][1]<1. and trial.evaluations==3 and trial.rejected_trials==1
    assert driver._state is before and owner._generation==9
    owner,driver,amounts=fixture(monkeypatch)
    with pytest.raises(corrector.DisplacementCorrectorFailure,match='backtrack budget') as caught:run(driver)
    assert amounts==[.5**i for i in range(9)]
    assert caught.value.detail.evaluations==10

def test_source_domain_fallback_and_exception_propagation(monkeypatch):
    from anyfesolver._stiffened_shell_ge_finite_experimental import _RotationStepTooLarge
    owner,driver,amounts=fixture(monkeypatch);original=driver._advance
    def domain(s,d,t):
        if t==1.:
            amounts.append(t);raise _RotationStepTooLarge('domain')
        return original(s,d,t)
    driver._advance=domain
    run(driver,POLICY)
    assert amounts[:2]==[1.,.5] and all(a>b for a,b in zip(amounts,amounts[1:]))
    for error in (MemoryError('memory'),RuntimeError('programming')):
        owner,driver,amounts=fixture(monkeypatch)
        def broken(*a,error=error):raise error
        driver._advance=broken
        with pytest.raises(type(error)):run(driver,POLICY)
        assert owner._generation==9 and not amounts

def test_source_cancellation_and_stale_corrector(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    owner,driver,amounts=fixture(monkeypatch)
    token=CancellationToken();token.cancel('prepared cancellation')
    with pytest.raises(SolveCancelled):run(driver,POLICY,cancellation_token=token)
    assert not amounts and owner._generation==9
    owner,driver,amounts=fixture(monkeypatch);original=driver._advance
    def stale(s,d,t):owner._generation+=1;return original(s,d,t)
    driver._advance=stale
    with pytest.raises(ValueError,match='stale'):run(driver,POLICY)
    assert driver._state.q[1]==0.

def context(owner,policy=None):
    state=adaptive.DisplacementIncrementState(ss.AdaptiveIncrementControl(4,96),(-1.,),0.)
    return adaptive.Context(state,0,adaptive.direction_hash(owner._finite_driver._pattern),0,20,8,
                            force_seed_sha256=DIGEST,merit_globalization=policy)

def test_source_fixed_caller_forwards_selected_only_and_refuses(monkeypatch):
    owner,driver,amounts=fixture(monkeypatch)
    control=fixed.DisplacementControl('shell',1,0,force_seed_sha256=DIGEST,merit_globalization=POLICY)
    class Reached(Exception):pass
    calls=[]
    def entered(*a,**k):calls.append(k);raise Reached()
    monkeypatch.setattr(fixed,'correct',entered)
    with pytest.raises(ValueError,match='adaptive context'):
        fixed.solve(owner,(),(-1.,),control,max_iterations=20,max_backtracks=8)
    assert not calls
    with pytest.raises(Reached):
        fixed.solve(owner,(),(-1.,),control,max_iterations=20,max_backtracks=8,_adaptive=context(owner,POLICY))
    assert calls[-1]['merit_globalization']==POLICY
    legacy=replace(control,merit_globalization=None)
    with pytest.raises(Reached):fixed.solve(owner,(),(-1.,),legacy,max_iterations=20,max_backtracks=8)
    assert 'merit_globalization' not in calls[-1]
    from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
    from anyfesolver._stiffened_shell_f06_increment_arithmetic import POLICY as ARITHMETIC
    owner._beam_parts=((NS(operator=NS(arithmetic_policy=ARITHMETIC)),None,None),)
    owner._require_open=lambda:None;owner._require_stateful_scope=lambda:None
    owner._require_distributing_reference_only=lambda *a:None
    for name in ('_coordinate_support','_bl5_native_material','_finite_distributing','_curved_reference','_curved_material'):
        setattr(owner,name,False)
    owner._unsupported_modes=();owner._arc_snapshot=None
    routed=[]
    monkeypatch.setattr(adaptive,'solve',lambda *a,**k:routed.append((a,k)) or 'selected-route')
    limits=ss.AdaptiveIncrementControl(4,96)
    assert ExperimentalGEAnalysis.solve.__wrapped__(owner,(),(-1.,),displacement_control=control,
        increment_control=limits,max_iterations=20,max_backtracks=8)=='selected-route'
    assert routed[0][0][3:]==(control,limits)
    monkeypatch.setattr(endpoint,'is_paired_displacement',lambda o:False)
    with pytest.raises(ValueError):fixed.solve(owner,(),(-1.,),control,max_iterations=20,max_backtracks=8,_adaptive=context(owner,POLICY))

def test_source_wrapper_policy_and_legacy_bytes(monkeypatch):
    owner,driver,amounts=fixture(monkeypatch);c=context(owner)
    original=adaptive.encode(owner._identity,c,SEED,paired=True)
    assert original.startswith(b'ANYSSDCP1') and b'merit_globalization' not in original
    raw=adaptive.encode(owner._identity,replace(c,merit_globalization=POLICY),SEED,paired=True)
    restored,inner=adaptive.decode(owner,raw)
    assert raw.startswith(b'ANYSSDCP2') and restored.merit_globalization==POLICY and inner==SEED
    assert adaptive.encode(owner._identity,restored,inner,paired=True)==raw
    # A re-signed legacy header with new selected policy must still refuse.
    mixed=b'ANYSSDCP1'+raw[9:-32];mixed+=sha256(mixed).digest()
    for invalid in (mixed,raw[:-1]+bytes([raw[-1]^1])):
        with pytest.raises(ValueError):adaptive.decode(owner,invalid)
    monkeypatch.setattr(endpoint,'is_paired_displacement',lambda o:False)
    with pytest.raises(ValueError):adaptive.decode(owner,raw)
    assert owner._displacement_adaptive is None

def test_source_pending_restart_policy_refusal(monkeypatch):
    owner,driver,amounts=fixture(monkeypatch);c=context(owner,POLICY)
    owner._displacement_adaptive=(c,adaptive.encode(owner._identity,c,SEED,paired=True))
    legacy=fixed.DisplacementControl('shell',1,0,force_seed_sha256=DIGEST)
    before=owner._displacement_adaptive
    with pytest.raises(ValueError,match='controls must match'):
        adaptive.solve(owner,(),(-1.,),legacy,c.state.limits,max_iterations=20,max_backtracks=8)
    assert owner._displacement_adaptive is before and owner._generation==9 and not amounts

def test_source_public_restore_selected_dispatch(monkeypatch):
    from anyfesolver._stiffened_shell_ge_experimental import ExperimentalGEAnalysis
    owner,driver,amounts=fixture(monkeypatch)
    owner._require_open=lambda:None;owner._require_stateful_scope=lambda:None;owner._unsupported_modes=()
    raw=adaptive.encode(owner._identity,context(owner,POLICY),SEED,paired=True)
    owner._finite_driver=None;owner._accepted=None
    calls=[]
    monkeypatch.setattr(adaptive,'restore_paired',lambda *a,**k:calls.append((a,k)) or 'isolated-route')
    assert ExperimentalGEAnalysis.restore.__wrapped__(owner,raw)=='isolated-route'
    assert calls[0][0]==(owner,raw) and owner._finite_driver is None
    monkeypatch.undo()
    owner,driver,amounts=fixture(monkeypatch)
    from anyfesolver.control import CancellationToken,SolveCancelled
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Result:
        status:str='completed'
        increment_attempts:tuple=()
    class Restorable(NS):
        def restore(self,inner):
            assert inner==SEED
            self._finite_driver=NS(_accepted=Result(),_state=NS(q=np.zeros(6)),_pattern=driver._pattern)
            return self._finite_driver._accepted
    owner=Restorable(**vars(owner));owner._finite_driver=None;owner._accepted=None
    class CancelPublish(CancellationToken):
        def raise_if_cancelled(self,stage=''):
            if stage.endswith('restore.publish'):self.cancel('after isolated replay')
            super().raise_if_cancelled(stage)
    with pytest.raises(SolveCancelled):adaptive.restore_paired(owner,raw,cancellation_token=CancelPublish())
    assert owner._finite_driver is None and owner._displacement_adaptive is None and owner._generation==9
    assert adaptive.restore_paired(owner,raw).status=='accepted_prefix'
    assert owner._displacement_adaptive[0].merit_globalization==POLICY
    # Exercise the actual rebase caller: it must hand the unchanged selected
    # outer context to publication, while the fixed inner remains policy-free.
    owner._shell_ids=(1,);owner._fixed=np.empty(0,dtype=int)
    owner._displacement_snapshot=(SEED,dict(seed_generation=9,seed_sha256=DIGEST,control_dof=0),
                                  dict(q=np.array([0.,0.,0.,.1,0.,0.]),shell_frames=np.eye(3)[None]))
    before=owner._displacement_adaptive[0];captured=[]
    monkeypatch.setattr(fixed,'encode_snapshot',lambda *a,**k:SEED)
    def publication(*a,**k):captured.append(k['adaptive']);return 'rebased'
    monkeypatch.setattr(fixed,'publish',publication)
    assert fixed.rebase(owner,9)=='rebased' and captured==[before]
    assert adaptive.encode(owner._identity,captured[0],SEED,paired=True).startswith(b'ANYSSDCP2')
