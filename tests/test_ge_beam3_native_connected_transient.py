"""Prepared actual connected public-controller controls; all UNEXECUTED."""
from hashlib import sha256
from dataclasses import replace
import json
import numpy as np
import pytest
from anyfesolver.dynamics import TransientConfig
from anyfesolver.ge_beam3_native_connected_dynamics import solve_transient
from anyfesolver._ge_beam3_native_transient import NativeTransientError
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from examples.native_connected_dynamic_dev import build,velocity

DT=2.**-12


@pytest.mark.parametrize('clamped',(False,True),ids=('free','root-clamped'))
def test_public_connected_independent_physical_balances(tmp_path,record_property,monkeypatch,capsys,clamped):
    import sys
    from examples import native_connected_transient_dev as public
    destination=tmp_path/('clamped'if clamped else'free')
    with monkeypatch.context()as patch:
        patch.setattr(sys,'argv',[public.__file__,'--balance-output',str(destination)]+(['--clamped']if clamped else[]))
        public.main()
    printed=json.loads(capsys.readouterr().out)
    assert printed['engineering_accepted']is False and printed['clamped']is clamped
    summary=json.loads((destination/'ASSESSMENT.json').read_bytes())
    assert summary['exact_fresh_split']is True
    assert summary['base_steps']*summary['base_dt']==8*summary['refinement_dt']==.001
    assert set(summary['assessments'])=={'full','prefix','resumed','refined'}
    for assessment in summary['assessments'].values():
        assert assessment['parity_max']<=1e-10 and assessment['balance_max']<=1e-6
    # Compact path/hash evidence; full physical histories are durable on disk.
    record_property('independent_connected_balance',json.dumps(dict(
        path=str(destination/'ASSESSMENT.json'),sha256=sha256((destination/'ASSESSMENT.json').read_bytes()).hexdigest(),
        clamped=clamped,base_parity=summary['assessments']['full']['parity_max'],
        base_balance=summary['assessments']['full']['balance_max'],
        refined_balance=summary['assessments']['refined']['balance_max'],exact_fresh_split=True),sort_keys=True))


def config(owner,steps=4,save_every=2):
    return TransientConfig(dt=DT,t_end=steps*DT,initial_velocity=velocity(owner),save_every=save_every)


def test_public_connected_rigid_trajectory_exact_split_restart():
    owner=build();full=solve_transient(owner,config(owner,2))
    assert full.success and full.to_dict()['accepted_steps']==2
    assert full.times.tolist()==[0.,2*DT]
    prefix_owner=build();prefix=solve_transient(prefix_owner,config(prefix_owner,1))
    resumed_owner=build();resumed=solve_transient(resumed_owner,config(resumed_owner,2),checkpoint=prefix.checkpoint,expected_sha256=sha256(prefix.checkpoint).hexdigest())
    assert resumed.checkpoint==full.checkpoint and resumed.to_dict()==full.to_dict()
    assert np.linalg.norm(owner.current.positions-owner.reference_positions-np.array((.01,.02,0.))*2*DT)<1e-11


def test_nonzero_physical_displacement_and_unsupported_controls():
    owner=build();c=config(owner,0);u=np.zeros(len(owner.physical));u[:3*owner.n]=np.tile((.001,0.,0.),owner.n)
    u[3*owner.n:]=np.tile((.125,0.,0.),2*owner.e);c=replace(c,initial_displacement=u)
    result=solve_transient(owner,c);assert result.success
    assert np.linalg.norm(owner.current.positions-owner.reference_positions-np.array((.001,0.,0.)))<1e-11
    bad=build();c=config(bad);c=replace(c,rayleigh_alpha=.1)
    with pytest.raises(NativeTransientError):solve_transient(bad,c)
    assert bad.current is None


@pytest.mark.parametrize('boundary',['trial_after_issue','commit_after_publish','before_commit'])
def test_real_trial_commit_interrupt_prefix_and_cleanup(monkeypatch,boundary):
    owner=build();c=config(owner,1)
    if boundary=='trial_after_issue':
        original=owner._prepare_trial
        def signal(*args,**kwargs):original(*args,**kwargs);raise KeyboardInterrupt('after issue')
        monkeypatch.setattr(owner,'_prepare_trial',signal)
    elif boundary=='commit_after_publish':
        original=owner.commit
        def signal(*args,**kwargs):original(*args,**kwargs);raise KeyboardInterrupt('after authentic commit')
        monkeypatch.setattr(owner,'commit',signal)
    else:
        import anyfesolver._native_midpoint_controller as engine
        original=engine.cancellation_safe_point
        def signal(token,stage):
            original(token,stage)
            if stage=='native_transient.before_commit':raise KeyboardInterrupt('precommit')
        monkeypatch.setattr(engine,'cancellation_safe_point',signal)
    with pytest.raises(KeyboardInterrupt) as caught:solve_transient(owner,c)
    expected=1 if boundary=='commit_after_publish' else 0
    assert caught.value.partial.to_dict()['accepted_steps']==expected
    assert owner.current.epoch==expected and not owner._candidates
    assert owner.checkpoint()==caught.value.native_checkpoint


def test_prepared_origin_and_restored_journal_failure_roll_back(monkeypatch):
    import anyfesolver._ge_beam3_connected_trajectory as adapter
    owner=build();original=adapter._result
    monkeypatch.setattr(adapter,'_result',lambda *a,**k:(_ for _ in ()).throw(ValueError('serialization')))
    with pytest.raises(ValueError):solve_transient(owner,config(owner,0))
    assert owner.current is None and not owner._issued
    monkeypatch.setattr(adapter,'_result',original)
    old=build();prefix=solve_transient(old,config(old,2));data=json.loads(prefix.checkpoint)
    data['body']['controller']['steps'][0]['midpoint_unknown_global'][0]+=.01
    data['checkpoint_sha256']=sha256(canonical(data['body'])).hexdigest();raw=canonical(data)
    fresh=build()
    with pytest.raises(NativeTransientError):solve_transient(fresh,config(fresh),checkpoint=raw,expected_sha256=sha256(raw).hexdigest())
    assert fresh.current is None and not fresh._issued


def test_cutback_is_shared_and_incompatible_config_refuses(monkeypatch):
    owner=build();original=owner._prepare_trial;calls=0
    def once(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:raise ValueError('connected midpoint iteration limit')
        return original(*args,**kwargs)
    monkeypatch.setattr(owner,'_prepare_trial',once)
    result=solve_transient(owner,config(owner,1));assert result.success
    assert result.to_dict()['steps'][0]['cutbacks']==[{'dt':DT,'reason':'connected midpoint iteration limit'}]
    restored=build();same=solve_transient(restored,config(restored,1),checkpoint=result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert same.checkpoint==result.checkpoint and same.to_dict()==result.to_dict()
    fresh=build();changed=config(fresh,2);changed=replace(changed,save_every=1)
    with pytest.raises(NativeTransientError):solve_transient(fresh,changed,checkpoint=result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert fresh.current is None


def test_actual_inflight_cancel_and_atomic_receipt_rollback(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    import anyfesolver._native_midpoint_controller as engine
    owner=build();token=CancellationToken();original=engine.cancellation_safe_point
    def cancel(token_arg,stage):
        if stage=='native_transient.before_commit':token.cancel('actual precommit')
        return original(token_arg,stage)
    with monkeypatch.context() as patch:
        patch.setattr(engine,'cancellation_safe_point',cancel)
        with pytest.raises(SolveCancelled) as caught:solve_transient(owner,config(owner,1),cancellation_token=token)
    assert caught.value.partial.to_dict()['accepted_steps']==0
    assert owner.current.epoch==0 and not owner._candidates
    assert owner.checkpoint()==caught.value.native_checkpoint
    # Callback receives authentic issuance INSIDE atomic trial; rollback makes receipt absent.
    fresh=build();state=fresh.initialize();receipts=[];raw=fresh.checkpoint()
    def receive(candidate):receipts.append(candidate);raise KeyboardInterrupt('inside atomic receipt')
    with pytest.raises(KeyboardInterrupt):fresh._prepare_trial(state,dt=DT,cancellation_token=None,prepare=receive)
    assert len(receipts)==1 and fresh.current is state and not fresh._candidates
    fresh._discard_prepared_trial(receipts[0]);assert fresh.checkpoint()==raw


def test_legacy_single_public_profile_smoke_remains_separate():
    from examples.native_contact_free_transient_dev import run as legacy_run
    _,result=legacy_run(t_end=.001,dt=.001)
    data=result.to_dict()
    assert result.success and data['profile']=='NATIVE_CONTACT_FREE_MIDPOINT_TRAJECTORY_DEV_V1'
    assert data['initial']['state']['mechanical']['positions']


def _resign_connected_controller(data):
    data['checkpoint_sha256']=sha256(canonical(data['body'])).hexdigest()
    raw=canonical(data)
    return raw,sha256(raw).hexdigest()


@pytest.mark.parametrize('case',['next_dt','truncated','cutback_reason','fixed_unknown'])
def test_resigned_controller_cadence_and_fixed_unknown_refusal(case):
    clamped=case=='fixed_unknown'
    old=build(clamped=clamped);c=config(old,1)
    if clamped:c=replace(c,initial_velocity=np.zeros(len(old.physical)))
    prefix=solve_transient(old,c);data=json.loads(prefix.checkpoint)
    controller=data['body']['controller']
    if case=='next_dt':controller['next_dt']=DT*.5
    elif case=='truncated':controller['steps'][0]['truncated']=True
    elif case=='cutback_reason':controller['steps'][0]['cutbacks']=[dict(dt=DT,reason='cancelled')]
    else:controller['steps'][0]['midpoint_unknown_global'][old.fixed[0]]=.01
    raw,digest=_resign_connected_controller(data)
    fresh=build(clamped=clamped);request=config(fresh,2)
    if clamped:request=replace(request,initial_velocity=np.zeros(len(fresh.physical)))
    message={'next_dt':'endpoint/cadence','truncated':'truncation','cutback_reason':'only connected Newton','fixed_unknown':'exactly zero'}[case]
    with pytest.raises(NativeTransientError,match=message):
        solve_transient(fresh,request,checkpoint=raw,expected_sha256=digest)
    assert fresh.current is None and not fresh._issued and not fresh._candidates
    assert not fresh._lock.locked()


@pytest.mark.parametrize('case',['reason','first_exceeds_plan','broken_halving','accepted_halving','floor','cutback_truncated','untruncated_short','extra_key','nonfinite','wrong_type','exhaustion'])
def test_closed_adaptive_policy_metadata_refusal(case):
    # Pure controller metadata control, distinct from the actual capsule replay.
    from anyfesolver._ge_beam3_connected_trajectory import _replay_planned_dt
    settings=dict(minimum_dt=DT/16,max_cutbacks=2)
    reason='connected midpoint iteration limit'
    step=dict(dt=DT/4,truncated=False,cutbacks=[dict(dt=DT,reason=reason),dict(dt=DT/2,reason=reason)])
    if case=='reason':step['cutbacks'][0]['reason']='cancelled'
    elif case=='first_exceeds_plan':step['cutbacks'][0]['dt']=2*DT
    elif case=='broken_halving':step['cutbacks'][1]['dt']=DT/4
    elif case=='accepted_halving':step['dt']=DT/8
    elif case=='floor':settings['minimum_dt']=DT/2
    elif case=='cutback_truncated':step['truncated']=True
    elif case=='untruncated_short':step['cutbacks']=[]
    elif case=='extra_key':step['cutbacks'][0]['repair']=True
    elif case=='nonfinite':step['cutbacks'][0]['dt']=float('nan')
    elif case=='wrong_type':step['cutbacks'][0]['dt']=True
    else:settings['max_cutbacks']=1
    with pytest.raises(NativeTransientError):_replay_planned_dt(step,DT,settings)


def test_closed_adaptive_policy_valid_halving_and_horizon_history():
    from anyfesolver._ge_beam3_connected_trajectory import _replay_planned_dt
    settings=dict(minimum_dt=DT/16,max_cutbacks=2)
    reason='connected midpoint backtracking failed'
    step=dict(dt=DT/4,truncated=False,cutbacks=[dict(dt=DT,reason=reason),dict(dt=DT/2,reason=reason)])
    assert _replay_planned_dt(step,DT,settings)==DT/4
    # Horizon clipping preserves the next plan even if a later request extends it.
    assert _replay_planned_dt(dict(dt=DT/8,truncated=True,cutbacks=[]),DT/4,settings)==DT/4


def test_clipped_prefix_extended_history_fresh_restore():
    first=build();request=config(first,0);request=replace(request,t_end=DT/2)
    prefix=solve_transient(first,request)
    second=build();request=config(second,0);request=replace(request,t_end=1.5*DT)
    extended=solve_transient(second,request,checkpoint=prefix.checkpoint,
        expected_sha256=sha256(prefix.checkpoint).hexdigest())
    steps=extended.to_dict()['steps']
    assert steps[0]['truncated'] and len(steps)>1
    third=build();request=config(third,0);request=replace(request,t_end=1.5*DT)
    same=solve_transient(third,request,checkpoint=extended.checkpoint,
        expected_sha256=sha256(extended.checkpoint).hexdigest())
    assert same.checkpoint==extended.checkpoint and same.to_dict()==extended.to_dict()
    fourth=build();request=config(fourth,0);request=replace(request,t_end=2.5*DT)
    continued=solve_transient(fourth,request,checkpoint=extended.checkpoint,
        expected_sha256=sha256(extended.checkpoint).hexdigest())
    assert continued.success and continued.to_dict()['steps'][:len(steps)]==steps


@pytest.mark.parametrize('boundary',['record','stage'])
def test_restart_mechanical_replay_single_lease_and_cancellation(monkeypatch,boundary):
    from anyfesolver.control import CancellationToken,SolveCancelled
    old=build();prefix=solve_transient(old,config(old,1))
    fresh=build();token=CancellationToken();seen=[]
    method='_evaluate' if boundary=='record' else '_stage';original=getattr(fresh,method)
    def cancel(*args,**kwargs):
        token_arg=args[1] if boundary=='record' else args[3]
        assert token_arg is token and fresh._lock.locked()
        assert fresh.current is None and not fresh._issued
        with pytest.raises(RuntimeError,match='concurrent use'):
            fresh.initialize()
        seen.append(method);token.cancel('during complete connected journal replay')
        return original(*args,**kwargs)
    monkeypatch.setattr(fresh,method,cancel)
    with pytest.raises(SolveCancelled):solve_transient(fresh,config(fresh,2),checkpoint=prefix.checkpoint,
        expected_sha256=sha256(prefix.checkpoint).hexdigest(),cancellation_token=token)
    assert seen==[method]
    assert fresh.current is None and not fresh._issued and not fresh._candidates
    assert not fresh._lock.locked()
