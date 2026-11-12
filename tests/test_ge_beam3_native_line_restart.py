"""Actual load-chain restart, unloading and physical recovery, private only."""
from copy import deepcopy
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_native_line_static_element import NativeLineFibreStaticElement
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_line_program import solve_line_static,_PROGRAM
from anyfesolver._ge_beam3_native_line_restart import LoadPoint,encode_checkpoint,decode_checkpoint,capture_solver_state
from anyfesolver._ge_beam3_native_line_recovery import recover_native_fields
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_ge_beam3_native_line_loading import problem
from test_ge_beam3_native_fibre_connected_restart import make as connected_model
from test_ge_beam3_schur_line_program import save

CASES=('straight-elastic','curved-plastic','curved-connected-plastic')


def make(case):
    if case!=CASES[2]: return problem(case==CASES[1],case==CASES[1])[0]
    source,_=connected_model('curved-2-plastic-cantilever'); model=FEModel('native-line-connected-restart')
    for i,node in source.mesh.nodes.items():model.add_node(i,*node.coords())
    for i,e in source.mesh.elements.items():
        made=NativeLineFibreStaticElement(i,e.node_ids,e.operator.reference,e.section,order=4)
        model.add_element(i,made);model.materials[made.material_name]=made.section
    for bc in source.boundary_conditions:model.add_boundary_condition(bc)
    return model


def scale(pattern,factor):
    return LinePattern(tuple((eid,*map(float,factor*np.array(force))) for eid,*force in pattern.rows))


def packet(result):return dict(status=result.status,displacements=result.displacements,states=result.element_states)


def snapshots(result,proportional,constant):
    return tuple(dict(load_point=LoadPoint(float(s.load_factor),constant,proportional),displacements=s.displacements,states=s.element_states)
        for s in result.snapshots)


@pytest.fixture(scope='module',params=CASES)
def saved(request,tmp_path_factory):
    case=request.param;root=tmp_path_factory.mktemp(case);model=make(case)
    pattern=LinePattern(tuple((i,.34,-.014,.007) for i in sorted(model.mesh.elements)))
    initial={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in model.mesh.elements.items()}
    whole,events=solve_line_static(model,pattern,line_search=True)
    save(root/'whole.json',packet(whole));save(root/'events.json',events)
    assert whole.status=='completed',whole.info
    chain=(dict(load_point=LoadPoint(0.,LinePattern(()),pattern),displacements=np.zeros(model.mesh.dof_manager.total_dofs),states=initial),)+snapshots(whole,pattern,LinePattern(()))
    save(root/'unencoded-chain.json',chain)
    raw=encode_checkpoint(model,chain);prefix=encode_checkpoint(model,chain[:2])
    save(root/'checkpoint.json',raw);save(root/'prefix.json',prefix)
    return case,root,pattern,whole,raw,prefix


def test_roundtrip_recovery_and_shared_interface(saved):
    case,root,pattern,whole,raw,_=saved;model=make(case)
    chain=decode_checkpoint(model,raw,expected_sha256=sha256(raw).hexdigest())
    assert encode_checkpoint(model,chain)==raw
    state=chain[-1];before=canonical(state);recovery={};work_errors=[];position_errors=[];net=np.zeros_like(state['displacements']);shared={}
    for eid,e in model.mesh.elements.items():
        local=state['states'][eid];mapping=list(e.get_dof_mapping(model.mesh));net[mapping]+=local['response'].residual
        recovered=recover_native_fields(e,model.mesh,local,expected_committed_total_u=state['displacements'][mapping])
        assert recovered['load_pattern_sha256']==local['load_pattern'].signature
        for row in recovered['stations']:
            variation=np.sin(np.arange(6)+.4);frame=row['current_frame'];local_r=row['resultants']+row['resultants_low']
            global_r=row['global_resultants']+row['global_resultants_low'];global_v=np.r_[frame@variation[:3],frame@variation[3:]]
            error=float(abs(local_r@variation-global_r@global_v)/max(1.,abs(local_r@variation)))
            assert error<=1e-11;work_errors.append(error)
            c=row['cell'];t=row['xi']-(c-1);xyz=e.operator.reference.coordinates
            b=xyz[0]-2*xyz[1]+xyz[2];x=local['positions']+local['position_low']
            expected=(1-t)*x[c]+t*x[c+1]+local['response'].rotations[c]@(.5*t*(t-1)*b)
            err=float(np.linalg.norm(row['current_position']+row['current_position_low']-expected))
            assert err<=1e-11;position_errors.append(err)
        assert len(recovered['stations'])==8
        recovery[str(eid)]=recovered
        for k,node in enumerate(e.node_ids):
            rot=local['committed_nodal_rotation_matrices'][k]
            if node in shared: np.testing.assert_array_equal(shared[node],rot)
            shared[node]=rot
    assert np.linalg.norm(net[6:])<=1e-11 and canonical(state)==before
    plastic_rows=sum(row[2]>0 for value in state['states'].values() for station in value['response'].history.stations for row in station.rows)
    assert plastic_rows>0 if case!=CASES[0] else plastic_rows==0
    save(root/'recovery.json',dict(recovery=recovery,work_errors=work_errors,position_errors=position_errors,net_residual=net,
        history_unchanged=True,plastic_rows=plastic_rows,production_qualified=False))


def test_actual_solver_continuation_and_unload(saved):
    case,root,pattern,whole,raw,prefix=saved;model=make(case);increment=scale(pattern,.5)
    prefix_chain=decode_checkpoint(model,prefix,expected_sha256=sha256(prefix).hexdigest())
    constant=prefix_chain[-1]['load_point'].effective(model)
    resumed,events=solve_line_static(model,increment,initial_checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest(),steps=1,line_search=True)
    save(root/'resumed.json',packet(resumed));save(root/'resume-events.json',events)
    assert resumed.status=='completed',resumed.info
    assert canonical(packet(resumed))==canonical(packet(whole))
    continuation=prefix_chain+snapshots(resumed,increment,constant)
    completed=encode_checkpoint(model,continuation);save(root/'resumed-checkpoint.json',completed)
    fresh=make(case);decoded=decode_checkpoint(fresh,completed,expected_sha256=sha256(completed).hexdigest())
    assert encode_checkpoint(fresh,decoded)==completed
    unload=scale(pattern,-.5)
    result,events=solve_line_static(fresh,unload,initial_checkpoint=completed,expected_sha256=sha256(completed).hexdigest(),steps=1,line_search=True)
    save(root/'unloaded.json',packet(result));save(root/'unload-events.json',events)
    assert result.status=='completed',result.info
    final=decoded+snapshots(result,unload,pattern);final_raw=encode_checkpoint(fresh,final);save(root/'unloaded-checkpoint.json',final_raw)
    again=make(case);roundtrip=decode_checkpoint(again,final_raw,expected_sha256=sha256(final_raw).hexdigest())
    assert encode_checkpoint(again,roundtrip)==final_raw
    for eid,state in result.element_states.items():
        prior=whole.element_states[eid]
        assert state['epoch']==3 and canonical(state['origins'])==canonical(prior['response'].history)
        for old,new in zip(prior['response'].history.stations,state['response'].history.stations):
            assert all(n[2]+n[3]>=o[2]+o[3] for o,n in zip(old.rows,new.rows))
    assert _PROGRAM.get() is None
    save(root/'continuation.json',dict(bitwise_whole_state_match=True,typed_unload_roundtrip=True,production_qualified=False))


@pytest.mark.parametrize('mutation',['path','pattern','history','seed','predecessor','missing','skip','bool','schema'])
def test_resealed_mutations(saved,mutation,tmp_path):
    case,_,_,_,raw,_=saved;value=json.loads(raw);last=value['snapshots'][-1];state=last['states'][0]['state']
    if mutation=='path':last['load_point']['parameter']=.75
    elif mutation=='pattern':state['load_pattern']['rows'][0][1]+=.01
    elif mutation=='history':state['origins']['stations'][0]['rows'][0][0]+=.001
    elif mutation=='seed':state['seed_resultants'][0]+=.001
    elif mutation=='predecessor':state['previous_state_sha256']='0'*64
    elif mutation=='missing':last['states']=[]
    elif mutation=='skip':value['snapshots'].pop(1)
    elif mutation=='bool':last['displacements'][0]=False
    else:value['schema']='nodal-only'
    for record in value['snapshots']:
        for row in record['states']:
            s=row['state'];s['state_sha256']=sha({k:v for k,v in s.items() if k!='state_sha256'})
    value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'});changed=canonical(value)
    with pytest.raises(ValueError):decode_checkpoint(make(case),changed,expected_sha256=sha256(changed).hexdigest())
    save(tmp_path/'rejected.json',dict(case=case,mutation=mutation,rejected=True,mutated_sha256=sha256(changed).hexdigest()))


@pytest.mark.parametrize('mutation',['duplicate','nonfinite','noncanonical','external-hash','wrong-constant'])
def test_restart_authority_before_solver(saved,mutation,monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    case,_,pattern,_,raw,_=saved;kwargs={}
    if mutation=='duplicate':raw=raw.replace(b'{',b'{"schema":"duplicate",',1)
    elif mutation=='nonfinite':raw=raw.replace(b'"parameter":0.0',b'"parameter":NaN',1)
    elif mutation=='noncanonical':raw+=b' '
    elif mutation=='wrong-constant':kwargs['constant']=LinePattern(())
    expected='0'*64 if mutation=='external-hash' else sha256(raw).hexdigest()
    def forbidden(*args,**kw):raise AssertionError('solver entered before authority validation')
    monkeypatch.setattr(solver,'solve_static_nonlinear',forbidden)
    with pytest.raises(ValueError):solve_line_static(make(case),scale(pattern,.5),initial_checkpoint=raw,expected_sha256=expected,**kwargs)
    assert _PROGRAM.get() is None
    save(tmp_path/'rejected.json',dict(case=case,mutation=mutation,rejected_before_solver=True))


def test_owned_capture_and_bad_observation(saved):
    case,_,_,whole,_,_=saved;model=make(case);calls=[];state=whole.element_states[1]
    made=capture_solver_state(model,1,state,exact_guard=lambda *a,**k:calls.append(k['context']))
    assert len(calls)==2 and canonical(made)==canonical(state)
    assert not np.shares_memory(made['positions'],state['positions']) and not made['response'].hessian.flags.writeable
    calls.clear()
    with pytest.raises((ValueError,TypeError)):capture_solver_state(model,1,object(),exact_guard=lambda *a,**k:calls.append(k['context']))
    assert len(calls)==1


def test_failed_restarted_increment_keeps_checkpoint_state(saved):
    case,root,pattern,_,raw,_=saved;model=make(case)
    chain=decode_checkpoint(model,raw,expected_sha256=sha256(raw).hexdigest());before=canonical(chain[-1]['states'])
    result,events=solve_line_static(model,scale(pattern,.5),initial_checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),
        steps=1,max_iterations=1,line_search=False)
    save(root/'failed-restart.json',dict(result=packet(result),events=events))
    assert result.status!='completed' and canonical(result.element_states)==before
    assert canonical(chain[-1]['states'])==before and _PROGRAM.get() is None


def test_support_identity_mutation_rejected(saved):
    from anyfesolver.boundary import BoundaryCondition
    case,_,_,_,raw,_=saved;model=make(case)
    model.add_boundary_condition(BoundaryCondition('changed-support',[3],{'ux':0.}))
    with pytest.raises(ValueError,match='model'):decode_checkpoint(model,raw,expected_sha256=sha256(raw).hexdigest())
