"""Native adaptive history: actual material updates and durable rejected work."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_displacement_adaptive import (
    Context,NATIVE_MAGIC,MAGIC,encode,decode,validate_progress,direction_hash)
from anyfesolver._stiffened_shell_displacement_increment import DisplacementIncrementState
from test_stiffened_shell_native_point_displacement import owner,ACTIONS,DIRECTION,CONTROL,OPTIONS
from test_stiffened_shell_native_plastic_s3_displacement import definition,compare
from test_stiffened_shell_displacement_increment import assert_single_cutback_history


def reject(driver,**args):
    detail=ss.DisplacementFailureDetail(driver.owner._generation,args['target'],0.,1,1,1,13,1.,0.,.01)
    raise ss.DisplacementCorrectorFailure('injected rejected candidate',detail)


def test_native_adaptive_actual_s3_cutback_cancel_restart(request,monkeypatch):
    model=definition(False,False)
    analysis=owner(request,model);restored=owner(request,model);bad=owner(request,model)
    seed=analysis.solve(ACTIONS,(.05,),**OPTIONS)
    initial=float(analysis._finite_driver._state.q[CONTROL.dof(analysis)])
    targets=(initial*1.01,initial*1.02)
    limits=ss.AdaptiveIncrementControl(2,8)
    kw=dict(OPTIONS,displacement_control=CONTROL,increment_control=limits)
    import anyfesolver._stiffened_shell_displacement_control as module
    original=module.correct;first=True
    def once(driver,**args):
        nonlocal first
        if first:first=False;return reject(driver,**args)
        return original(driver,**args)
    token=CancellationToken()
    with monkeypatch.context() as patch:
        patch.setattr(module,'correct',once)
        with pytest.raises(SolveCancelled):
            analysis.solve(DIRECTION,targets,**kw,cancellation_token=token,
                           on_accepted=lambda result:token.cancel('accepted native midpoint'))
    prefix=analysis.recover();raw=analysis.checkpoint()
    assert raw.startswith(NATIVE_MAGIC) and prefix.status=='accepted_prefix'
    assert [row.accepted for row in prefix.increment_attempts]==[False,True]
    assert prefix.increment_attempts[0].rejected_trials==13
    assert np.any(np.concatenate([x.alpha for x in prefix.shell_s3_material])>
                  np.concatenate([x.alpha for x in seed.shell_s3_material]))
    repeated=restored.restore(raw);compare(prefix,repeated)
    assert restored.checkpoint()==raw
    context,inner=decode(analysis,raw)
    # Native outer identity cannot be replaced by legacy/event identity.
    with pytest.raises(ValueError,match='formulation identity'):
        bad.restore(encode(analysis._identity,context,inner))
    assert bad._generation==0 and bad._finite_driver is None
    with pytest.raises(ValueError,match='load/control mismatch'):
        bad.restore(encode(analysis._identity,replace(context,direction_sha256='0'*64),inner,native=True))
    assert bad._generation==0 and bad._finite_driver is None
    with pytest.raises(ValueError,match='branch event'):
        encode(analysis._identity,replace(context,trial_policy=ss.GENERALIZED_BRANCH_EVENT_TRIAL),inner,native=True)
    with pytest.raises(ValueError,match='must match'):
        restored.solve(DIRECTION,targets,**{**kw,'increment_control':ss.AdaptiveIncrementControl(3,8)})
    with pytest.raises(ValueError,match='must match'):
        restored.solve(DIRECTION,(targets[-1],),displacement_control=CONTROL,**OPTIONS)
    a=analysis.solve(DIRECTION,targets,**kw);b=restored.solve(DIRECTION,targets,**kw)
    compare(a,b)
    assert_single_cutback_history(a,b,prefix,initial,targets,limits)
    assert analysis.checkpoint()==restored.checkpoint()
    assert a.free_equilibrium_ratio<=1e-8 and a.constraint_max<=1e-9
    assert not a.production_qualified


def test_native_mixed_rejected_work_is_durable_and_seed_guards(request,monkeypatch):
    model=definition(True,True)
    analysis=owner(request,model);restored=owner(request,model)
    raw=(Path(__file__).parent/'data/native_mixed_j2_displacement/quarter-prefix-v1.bin').read_bytes()
    prefix=analysis.restore(raw)
    import anyfesolver._stiffened_shell_displacement_control as module
    kw=dict(OPTIONS,displacement_control=CONTROL,increment_control=ss.AdaptiveIncrementControl(0,1))
    with monkeypatch.context() as patch:
        patch.setattr(module,'correct',reject)
        with pytest.raises(ss.DisplacementIncrementLimit):
            analysis.solve(DIRECTION,(prefix.target*2.,),**kw)
    saved=analysis.checkpoint();context,inner=decode(analysis,saved)
    assert saved.startswith(NATIVE_MAGIC) and inner==raw
    assert context.state.attempts[0].accepted is False
    assert analysis.recover().history==prefix.history
    repeated=restored.restore(saved);compare(prefix,repeated)
    assert restored.checkpoint()==saved
    with pytest.raises(ss.DisplacementIncrementLimit):
        restored.solve(DIRECTION,(prefix.target*2.,),**kw)
    # Seed-only restoration must enforce exclusions before outer publication.
    seed=owner(request,definition(False,False))
    seed.solve(ACTIONS,(.05,),**OPTIONS)
    d=CONTROL.dof(seed)
    context=Context(DisplacementIncrementState(ss.AdaptiveIncrementControl(),
        (float(seed._finite_driver._state.q[d]*1.01),),float(seed._finite_driver._state.q[d])),
        d,direction_hash(seed._finite_driver._load(DIRECTION)),0,20,12)
    validate_progress(seed,context)
    seed_raw=seed.checkpoint();seed_resumed=owner(request,seed._definition)
    with monkeypatch.context() as patch:
        patch.setattr(module,'correct',reject)
        with pytest.raises(ss.DisplacementIncrementLimit):
            seed.solve(DIRECTION,context.state.targets,**kw)
    seed_saved=seed.checkpoint();seed_resumed.restore(seed_saved)
    assert seed_resumed.checkpoint()==seed_saved
    assert decode(seed,seed_saved)[1]==seed_raw
    with pytest.raises(ss.DisplacementIncrementLimit):
        seed_resumed.solve(DIRECTION,context.state.targets,**kw)
    with pytest.raises(ValueError,match='nested'):
        encode(seed._identity,context,seed_saved,native=True)
    from anyfesolver._stiffened_shell_increment_control import MAGIC as FORCE_MAGIC
    with pytest.raises(ValueError,match='nested'):
        encode(seed._identity,context,FORCE_MAGIC+seed_raw,native=True)
    seed._finite_driver._adaptive_state=object()
    with pytest.raises(ValueError,match='ordinary unrebased'):validate_progress(seed,context)
    seed._finite_driver._adaptive_state=None
    seed._finite_driver._programme=object()
    with pytest.raises(ValueError,match='ordinary unrebased'):validate_progress(seed,context)
    seed._finite_driver._programme=None
    seed._finite_driver._external=object()
    with pytest.raises(ValueError,match='ordinary unrebased'):validate_progress(seed,context)
    seed._finite_driver._external=None
    seed._finite_driver._state=replace(seed._finite_driver._state,
        shell_anchors=np.tile(np.eye(3),(len(seed._shell_ids),1,1)))
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import CHART_MAGIC
    # Identity chart framing is still a nested chart, not an ordinary force seed.
    with pytest.raises(ValueError,match='nested'):
        encode(seed._identity,context,CHART_MAGIC+seed_raw,native=True)
    with pytest.raises(ValueError,match='ordinary unrebased'):validate_progress(seed,context)
    anchors=seed._finite_driver._state.shell_anchors.copy();anchors[0]=np.diag((-1.,-1.,1.))
    seed._finite_driver._state=replace(seed._finite_driver._state,shell_anchors=anchors)
    with pytest.raises(ValueError,match='ordinary unrebased'):validate_progress(seed,context)


def test_actual_mixed_adaptive_midpoint_continues_with_material_history(request):
    model=definition(True,True)
    analysis=owner(request,model);resumed=owner(request,model)
    raw=(Path(__file__).parent/'data/native_mixed_j2_displacement/adaptive-midpoint-v1.bin').read_bytes()
    prefix=analysis.restore(raw);resumed.restore(raw)
    context,_=decode(analysis,raw)
    assert [row.accepted for row in context.state.attempts]==[False,True]
    assert all(np.max(x.alpha)>0. for x in prefix.shell_q4_material+prefix.shell_s3_material)
    kw=dict(OPTIONS,displacement_control=CONTROL,increment_control=context.state.limits)
    result=analysis.solve(DIRECTION,context.state.targets,**kw)
    compare(result,resumed.solve(DIRECTION,context.state.targets,**kw))
    assert result.status=='completed' and len(result.increment_attempts)==3
    assert result.increment_attempts[:len(context.state.attempts)]==context.state.attempts
    assert analysis.checkpoint()==resumed.checkpoint()
    assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
    assert np.any(result.shell_q4_material[0].alpha>prefix.shell_q4_material[0].alpha)
    assert any(np.any(a.alpha>b.alpha) for a,b in zip(result.shell_s3_material,prefix.shell_s3_material,strict=True))
    assert not result.production_qualified
