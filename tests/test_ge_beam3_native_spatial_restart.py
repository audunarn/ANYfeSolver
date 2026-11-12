"""Actual spatial-couple accepted-chain continuation, not production authority."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_line_program import _PROGRAM, solve_line_static
from anyfesolver._ge_beam3_native_line_restart import LoadPoint as LinePoint
from anyfesolver._ge_beam3_native_line_recovery import recover_native_fields
from anyfesolver._ge_beam3_native_spatial_restart import LoadPoint, encode_checkpoint, decode_checkpoint
from anyfesolver._ge_beam3_native_spatial_couples import solve_spatial_static, _RUN
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from test_ge_beam3_native_line_restart import make, scale as line_scale, CASES
from test_ge_beam3_schur_line_program import save


def moment_scale(pattern, factor):
    return SpatialNodalMoments(tuple((node, *map(float, factor*np.array(moment))) for node, *moment in pattern.rows))


def packet(result):
    return dict(status=result.status, displacements=result.displacements, states=result.element_states)


def snapshots(result, line, moments, constant_line=LinePattern(()), constant_moments=None):
    return tuple(dict(load_point=LoadPoint(LinePoint(float(s.load_factor), constant_line, line), constant_moments, moments),
                      displacements=s.displacements, states=s.element_states) for s in result.snapshots)


@pytest.fixture(scope='module', params=CASES)
def saved(request, tmp_path_factory):
    case=request.param; root=tmp_path_factory.mktemp(case); model=make(case)
    connected=case==CASES[2]
    line=LinePattern(tuple((i, .17 if connected else .34, -.007 if connected else -.014, .0035 if connected else .007)
                           for i in sorted(model.mesh.elements)))
    moments=SpatialNodalMoments(((3, .03, .02, -.01),)) if connected else SpatialNodalMoments(((3, .12, .10, -.08),))
    initial={i:e.init_model_bound_nonlinear_state(model.mesh, e.section, 1) for i,e in model.mesh.elements.items()}
    whole, evidence=solve_spatial_static(model, moments, line_pattern=line)
    save(root/'whole.json', packet(whole)); save(root/'events.json', evidence)
    assert whole.status=='completed', whole.info
    chain=(dict(load_point=LoadPoint(LinePoint(0., LinePattern(()), line), None, moments),
                 displacements=np.zeros(model.mesh.dof_manager.total_dofs), states=initial),)+snapshots(whole,line,moments)
    save(root/'unencoded-chain.json', chain)
    raw=encode_checkpoint(model,chain); prefix=encode_checkpoint(model,chain[:2])
    save(root/'checkpoint.json',raw);save(root/'prefix.json',prefix)
    return case,root,line,moments,whole,raw,prefix


def test_roundtrip_equilibrium_and_physical_recovery(saved):
    case,root,_,_,whole,raw,_=saved;model=make(case)
    chain=decode_checkpoint(model,raw,expected_sha256=sha256(raw).hexdigest())
    assert encode_checkpoint(model,chain)==raw
    final=chain[-1];before=canonical(final);net=np.zeros_like(final['displacements']);recovery={};shared={};work_errors=[]
    for eid,e in model.mesh.elements.items():
        state=final['states'][eid];mapping=list(e.get_dof_mapping(model.mesh));net[mapping]+=state['response'].residual
        recovery[str(eid)]=recover_native_fields(e,model.mesh,state,expected_committed_total_u=final['displacements'][mapping])
        assert len(recovery[str(eid)]['stations'])==8
        for row in recovery[str(eid)]['stations']:
            variation=np.sin(np.arange(6)+.4);frame=row['current_frame']
            local=row['resultants']+row['resultants_low'];global_r=row['global_resultants']+row['global_resultants_low']
            global_v=np.r_[frame@variation[:3],frame@variation[3:]]
            work_error=float(abs(local@variation-global_r@global_v)/max(1.,abs(local@variation)))
            assert work_error<=1e-11;work_errors.append(work_error)
        for k,node in enumerate(e.node_ids):
            rot=state['committed_nodal_rotation_matrices'][k]
            if node in shared:np.testing.assert_array_equal(shared[node],rot)
            shared[node]=rot
    moments=final['load_point'].effective_moments(model)
    for node,*moment in moments.rows:net[list(model.mesh.dof_manager.get_node_dofs(node)[3:])]-=moment
    error=float(np.linalg.norm(net[6:]));assert error<=1e-11
    assert canonical(final)==before and canonical(final['states'])==canonical(whole.element_states)
    save(root/'recovery.json',dict(recovery=recovery,free_residual_error=error,work_errors=work_errors,shared_node_counted_once=True,
                                  history_unchanged=True,production_qualified=False))


def test_actual_continuation_unload_and_failed_step(saved):
    case,root,line,moments,whole,_,prefix=saved;model=make(case)
    chain=decode_checkpoint(model,prefix,expected_sha256=sha256(prefix).hexdigest())
    half_line=line_scale(line,.5);half_moments=moment_scale(moments,.5)
    resumed,events=solve_spatial_static(model,half_moments,line_pattern=half_line,steps=1,
        initial_checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest())
    save(root/'resumed.json',packet(resumed));save(root/'resume-events.json',events)
    assert resumed.status=='completed',resumed.info
    assert canonical(packet(resumed))==canonical(packet(whole))
    assert events['moment_restart_authorized'] and events['program']['restart_sha256']==sha256(prefix).hexdigest()
    complete=chain+snapshots(resumed,half_line,half_moments,half_line,half_moments)
    completed=encode_checkpoint(model,complete);save(root/'resumed-checkpoint.json',completed)
    fresh=make(case);decoded=decode_checkpoint(fresh,completed,expected_sha256=sha256(completed).hexdigest())
    assert encode_checkpoint(fresh,decoded)==completed
    negative_line=line_scale(line,-.5);negative_moments=moment_scale(moments,-.5)
    unloaded,events=solve_spatial_static(fresh,negative_moments,line_pattern=negative_line,steps=1,
        initial_checkpoint=completed,expected_sha256=sha256(completed).hexdigest())
    save(root/'unloaded.json',packet(unloaded));save(root/'unload-events.json',events)
    assert unloaded.status=='completed',unloaded.info
    final=decoded+snapshots(unloaded,negative_line,negative_moments,line,moments)
    final_raw=encode_checkpoint(fresh,final);save(root/'unloaded-checkpoint.json',final_raw)
    assert encode_checkpoint(make(case),decode_checkpoint(make(case),final_raw,expected_sha256=sha256(final_raw).hexdigest()))==final_raw
    for eid,state in unloaded.element_states.items():
        old=whole.element_states[eid]
        assert state['epoch']==3 and canonical(state['origins'])==canonical(old['response'].history)
        for a,b in zip(old['response'].history.stations,state['response'].history.stations):
            assert all(y[2]+y[3]>=x[2]+x[3] for x,y in zip(a.rows,b.rows))
    before=canonical(decoded[-1]['states'])
    failed,events=solve_spatial_static(make(case),half_moments,line_pattern=half_line,steps=1,max_iterations=1,
        initial_checkpoint=completed,expected_sha256=sha256(completed).hexdigest())
    save(root/'failed-restart.json',dict(result=packet(failed),events=events))
    assert failed.status!='completed' and canonical(failed.element_states)==before
    assert canonical(decoded[-1]['states'])==before and _PROGRAM.get() is None and _RUN.get() is None
    save(root/'continuation.json',dict(whole_state_bitwise_equal=True,unload_roundtrip=True,
                                      failed_step_preserves_accepted=True,production_qualified=False))


@pytest.mark.parametrize('mutation',['moment','missing-moment','node','path','line','history','seed','predecessor','skip','policy','schema','norm-range',
                                     'boolean','missing-state','genesis','rotation','qualification'])
def test_resealed_mutations(saved,mutation,tmp_path):
    case,_,_,_,_,raw,_=saved;value=json.loads(raw);last=value['snapshots'][-1];state=last['states'][0]['state']
    if mutation=='moment':last['load_point']['proportional_moments']['rows'][0][1]+=.01
    elif mutation=='missing-moment':last['load_point']['proportional_moments']=None
    elif mutation=='node':last['load_point']['proportional_moments']['rows'][0][0]=999
    elif mutation=='path':last['load_point']['line']['parameter']=.75
    elif mutation=='line':last['load_point']['line']['proportional']['rows'][0][1]+=.01
    elif mutation=='history':state['origins']['stations'][0]['rows'][0][0]+=.001
    elif mutation=='seed':state['seed_resultants'][0]+=.001
    elif mutation=='predecessor':state['previous_state_sha256']='0'*64
    elif mutation=='skip':value['snapshots'].pop(1)
    elif mutation=='policy':value['load_policy']='CONSERVATIVE_MOMENT_POTENTIAL'
    elif mutation=='schema':value['schema']='GE_BEAM3_SUPPORTED_NATIVE_LINE_STATIC_RESTART_V1'
    elif mutation=='norm-range':last['load_point']['constant_moments']={'rows':[[1,1e200,0.,0.]]}
    elif mutation=='boolean':last['load_point']['proportional_moments']['rows'][0][1]=True
    elif mutation=='missing-state':last['states']=[]
    elif mutation=='genesis':value['snapshots'][0]['load_point']['constant_moments']={'rows':[[1,.1,0.,0.]]}
    elif mutation=='rotation':state['committed_nodal_rotation_matrices'][0][0][0]+=.01
    else:value['production_qualified']=True
    for record in value['snapshots']:
        for row in record['states']:
            s=row['state'];s['state_sha256']=sha({k:v for k,v in s.items() if k!='state_sha256'})
    value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'});changed=canonical(value)
    with pytest.raises(ValueError):decode_checkpoint(make(case),changed,expected_sha256=sha256(changed).hexdigest())
    save(tmp_path/'rejected.json',dict(case=case,mutation=mutation,rejected=True,mutated_sha256=sha256(changed).hexdigest()))


@pytest.mark.parametrize('mutation',['duplicate','nonfinite','noncanonical','external-hash','constant-moment','constant-line','support'])
def test_rejection_before_solver(saved,mutation,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_spatial_couples as module
    from anyfesolver.boundary import BoundaryCondition
    case,_,line,moments,_,raw,_=saved;model=make(case);kwargs={}
    if mutation=='duplicate':raw=raw.replace(b'{',b'{"schema":"duplicate",',1)
    elif mutation=='nonfinite':raw=raw.replace(b'"parameter":0.0',b'"parameter":NaN',1)
    elif mutation=='noncanonical':raw+=b' '
    elif mutation=='constant-moment':kwargs['constant_moments']=moment_scale(moments,.1)
    elif mutation=='constant-line':kwargs['constant_line']=LinePattern(())
    elif mutation=='support':model.add_boundary_condition(BoundaryCondition('changed',[3],{'ux':0.}))
    expected='0'*64 if mutation=='external-hash' else sha256(raw).hexdigest()
    def forbidden(*args,**kw):raise AssertionError('entered solver before authority validation')
    monkeypatch.setattr(module,'solve_line_static',forbidden)
    with pytest.raises(ValueError):solve_spatial_static(model,moments,line_pattern=line,initial_checkpoint=raw,expected_sha256=expected,**kwargs)
    assert _PROGRAM.get() is None and _RUN.get() is None
    save(tmp_path/'rejected.json',dict(case=case,mutation=mutation,rejected_before_solver=True))


def test_cross_schema_and_missing_hash_rejected(saved,tmp_path):
    from anyfesolver._ge_beam3_native_line_restart import encode_checkpoint as encode_line
    case,_,line,moments,_,raw,_=saved;model=make(case)
    with pytest.raises(ValueError):solve_line_static(model,line,initial_checkpoint=raw,expected_sha256=sha256(raw).hexdigest())
    state={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in model.mesh.elements.items()}
    line_raw=encode_line(model,(dict(load_point=LinePoint(0.,LinePattern(()),line),
        displacements=np.zeros(model.mesh.dof_manager.total_dofs),states=state),))
    with pytest.raises(ValueError):solve_spatial_static(model,moments,initial_checkpoint=line_raw,expected_sha256=sha256(line_raw).hexdigest())
    with pytest.raises(ValueError):solve_spatial_static(model,moments,initial_checkpoint=raw)
    with pytest.raises(ValueError):solve_spatial_static(model,moments,expected_sha256=sha256(raw).hexdigest())
    save(tmp_path/'cross-schema.json',dict(case=case,rejected_both_directions=True,hash_required=True))


def test_live_checkpoint_binding(saved,tmp_path):
    from anyfesolver._ge_beam3_native_spatial_couples import _Run,decode_active_restart
    case,_,_,moments,_,raw,_=saved;model=make(case)
    run=_Run(model,moments,moments,restart_sha256='0'*64);token=_RUN.set(run)
    try:
        with pytest.raises(ValueError,match='checkpoint mismatch'):decode_active_restart(model,raw,expected_sha256=sha256(raw).hexdigest())
    finally:_RUN.reset(token)
    save(tmp_path/'live-binding.json',dict(case=case,foreign_checkpoint_rejected=True))


def test_validation_deadline(saved,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_spatial_restart as module
    case,_,_,_,_,raw,_=saved;calls=[]
    def clock():
        calls.append(True);return 61.*len(calls)
    monkeypatch.setattr(module,'monotonic',clock)
    with pytest.raises(RuntimeError,match='deadline'):decode_checkpoint(make(case),raw,expected_sha256=sha256(raw).hexdigest())
    save(tmp_path/'deadline.json',dict(case=case,deadline_rejected=True))


def test_cheap_structural_and_effective_range_guards():
    model=make(CASES[0]);empty=LinePattern(())
    for value in (None,[],(),(None,)*66):
        with pytest.raises(ValueError):encode_checkpoint(model,value)
    huge=SpatialNodalMoments(((3,1e308,0.,0.),))
    with pytest.raises(ValueError,match='range'):LoadPoint(LinePoint(1.,empty,empty),huge,huge).effective_moments(model)
    positive=SpatialNodalMoments(((3,.2,.1,0.),))
    cancelled=LoadPoint(LinePoint(1.,empty,empty),positive,moment_scale(positive,-1.))
    assert cancelled.effective_moments(model) is None
