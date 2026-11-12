"""Actual generalized global statics and typed restart/recovery development."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern,_ACTIVE
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_native_generalized_program import solve_distributed_model,_PROGRAM,_Program,model_identity
from anyfesolver._ge_beam3_native_generalized_restart import LoadPoint,encode_checkpoint,decode_checkpoint,capture_solver_state
from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_ge_beam3_native_generalized import problem
from test_ge_beam3_curved_contrast_probe import make_curved
from test_ge_beam3_generalized_ellipsoid_section import section
from anyfesolver.fe_core import FEModel
from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from anyfesolver._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from test_ge_beam3_schur_line_program import save

CASES=('straight-elastic','curved-plastic','connected-plastic')
EMPTY=DistributedPattern(LinePattern(()),())


def make(case):
    if case!=CASES[2]:return problem(case==CASES[1],case==CASES[1])[0]
    source=make_curved(100.)[0];m=FEModel('native-generalized-connected')
    for i,node in source.mesh.nodes.items():m.add_node(i,*node.coords())
    c=section();law=EllipsoidalGeneralizedSection(c.elastic,c.metric,.025,c.hardening)
    for i,old in source.mesh.elements.items():
        made=NativeGeneralizedStaticElement(i,old.node_ids,old.core.reference,law,order=4)
        m.add_element(i,made);m.materials[made.material_name]=made.section
    for bc in source.boundary_conditions:m.add_boundary_condition(bc)
    return m


def pattern(model):
    return DistributedPattern(LinePattern(tuple((i,.09,-.03,.02) for i in sorted(model.mesh.elements))),
        tuple((i,.07,-.04,.05) for i in sorted(model.mesh.elements)))


def packet(result):return dict(status=result.status,displacements=result.displacements,states=result.element_states)


def scale(load,factor):
    return DistributedPattern(LinePattern(tuple((eid,*map(float,factor*np.array(force))) for eid,*force in load.line.rows)),
        tuple((eid,*map(float,factor*np.array(density))) for eid,*density in load.couples))


def snapshots(result,proportional,constant):
    return tuple(dict(load_point=LoadPoint(float(s.load_factor),constant,proportional),displacements=s.displacements,states=s.element_states)
        for s in result.snapshots)


@pytest.fixture(scope='module',params=CASES)
def saved(request,tmp_path_factory):
    case=request.param;root=tmp_path_factory.mktemp(case);model=make(case);load=pattern(model)
    initial={i:e.init_model_bound_nonlinear_state(model.mesh,e.section,1) for i,e in model.mesh.elements.items()}
    whole,events=solve_distributed_model(model,load)
    save(root/'whole.json',packet(whole));save(root/'events.json',events)
    assert whole.status=='completed',whole.info
    chain=(dict(load_point=LoadPoint(0.,EMPTY,load),displacements=np.zeros(model.mesh.dof_manager.total_dofs),states=initial),)+snapshots(whole,load,EMPTY)
    save(root/'unencoded-chain.json',chain)
    raw=encode_checkpoint(model,chain);prefix=encode_checkpoint(model,chain[:2])
    save(root/'checkpoint.json',raw);save(root/'prefix.json',prefix)
    return case,root,load,whole,raw,prefix


def test_roundtrip_native_recovery(saved):
    case,root,_,whole,raw,_=saved;model=make(case)
    chain=decode_checkpoint(model,raw,expected_sha256=sha256(raw).hexdigest())
    assert encode_checkpoint(model,chain)==raw
    state=chain[-1];before=canonical(state);net=np.zeros_like(state['displacements']);recovery={};work_errors=[];position_errors=[];shared={}
    for eid,e in model.mesh.elements.items():
        local=state['states'][eid];mapping=list(e.get_dof_mapping(model.mesh));net[mapping]+=local['response'].residual
        recovered=recover_native_fields(e,model.mesh,local,expected_committed_total_u=state['displacements'][mapping])
        assert recovered['load_pattern_sha256']==local['load_pattern'].signature
        assert recovered['formulation_id']==e.formulation_id
        assert recovered['fibre_stress_status']=='RESULTANT_LEVEL_ONLY_NO_FIBRE_STRESSES_SUPPLIED'
        for row in recovered['stations']:
            assert 'fibres' not in row
            variation=np.sin(np.arange(6)+.4);frame=row['current_frame'];local_r=row['resultants']+row['resultants_low']
            global_r=row['global_resultants']+row['global_resultants_low'];global_v=np.r_[frame@variation[:3],frame@variation[3:]]
            error=float(abs(local_r@variation-global_r@global_v)/max(1.,abs(local_r@variation)))
            assert error<=1e-11;work_errors.append(error)
            c=row['cell'];t=row['xi']-(c-1);xyz=e.operator.reference.coordinates;b=xyz[0]-2*xyz[1]+xyz[2]
            x=local['positions']+local['position_low'];expected=(1-t)*x[c]+t*x[c+1]+local['response'].rotations[c]@(.5*t*(t-1)*b)
            error=float(np.linalg.norm(row['current_position']+row['current_position_low']-expected));assert error<=1e-11;position_errors.append(error)
        assert len(recovered['stations'])==8;recovery[str(eid)]=recovered
        for i,node in enumerate(e.node_ids):
            rot=local['committed_nodal_rotation_matrices'][i]
            if node in shared:np.testing.assert_array_equal(shared[node],rot)
            shared[node]=rot
    assert canonical(state)==before and canonical(state['states'])==canonical(whole.element_states)
    error=float(np.linalg.norm(net[6:]));assert error<=1e-11
    positive=sum(sum(station.accumulated)>0 for s in state['states'].values() for station in s['response'].history.stations)
    assert positive>0 if case!=CASES[0] else positive==0
    save(root/'recovery.json',dict(recovery=recovery,work_errors=work_errors,position_errors=position_errors,free_residual_error=error,
        positive_plastic_rows=positive,history_unchanged=True,production_qualified=False))


def test_actual_continuation_unload_and_failed_step(saved):
    case,root,load,whole,_,prefix=saved;model=make(case);increment=scale(load,.5)
    chain=decode_checkpoint(model,prefix,expected_sha256=sha256(prefix).hexdigest());constant=chain[-1]['load_point'].effective(model)
    result,events=solve_distributed_model(model,increment,initial_checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest(),steps=1)
    save(root/'resumed.json',packet(result));save(root/'resume-events.json',events)
    assert result.status=='completed',result.info
    assert events['restart_authorized'] and events['restart_sha256']==sha256(prefix).hexdigest()
    assert canonical(packet(result))==canonical(packet(whole))
    complete=chain+snapshots(result,increment,constant);completed=encode_checkpoint(model,complete);save(root/'resumed-checkpoint.json',completed)
    fresh=make(case);decoded=decode_checkpoint(fresh,completed,expected_sha256=sha256(completed).hexdigest())
    assert encode_checkpoint(fresh,decoded)==completed
    negative=scale(load,-.5)
    unloaded,events=solve_distributed_model(fresh,negative,initial_checkpoint=completed,expected_sha256=sha256(completed).hexdigest(),steps=1)
    save(root/'unloaded.json',packet(unloaded));save(root/'unload-events.json',events)
    assert unloaded.status=='completed',unloaded.info
    final=decoded+snapshots(unloaded,negative,load);raw=encode_checkpoint(fresh,final);save(root/'unloaded-checkpoint.json',raw)
    other=make(case);again=decode_checkpoint(other,raw,expected_sha256=sha256(raw).hexdigest());assert encode_checkpoint(other,again)==raw
    for eid,state in unloaded.element_states.items():
        prior=whole.element_states[eid]
        assert state['epoch']==3 and canonical(state['origins'])==canonical(prior['response'].history)
        for old,new in zip(prior['response'].history.stations,state['response'].history.stations):
            assert sum(new.accumulated)>=sum(old.accumulated)
    before=canonical(decoded[-1]['states'])
    failed,events=solve_distributed_model(make(case),increment,initial_checkpoint=completed,expected_sha256=sha256(completed).hexdigest(),steps=1,max_iterations=1)
    save(root/'failed-restart.json',dict(result=packet(failed),events=events))
    assert failed.status!='completed' and canonical(failed.element_states)==before and canonical(decoded[-1]['states'])==before
    assert _PROGRAM.get() is None and _ACTIVE.get() is None
    save(root/'continuation.json',dict(whole_state_bitwise_equal=True,unload_roundtrip=True,failed_step_preserves_accepted=True,production_qualified=False))


@pytest.mark.parametrize('mutation',['density','missing-density','path','line','origin','seed','predecessor','skip','jacobian','applied','boolean','missing-state','schema','conservative','production'])
def test_resealed_mutations(saved,mutation,tmp_path):
    case,_,_,_,raw,_=saved;value=json.loads(raw);last=value['snapshots'][-1];state=last['states'][0]['state']
    if mutation=='density':last['load_point']['proportional']['couples'][0][1]+=.01
    elif mutation=='missing-density':last['load_point']['proportional']['couples']=[]
    elif mutation=='path':last['load_point']['parameter']=.75
    elif mutation=='line':last['load_point']['proportional']['line']['rows'][0][1]+=.01
    elif mutation=='origin':state['origins']['stations'][0]['plastic'][0][0]+=.001
    elif mutation=='seed':state['seed_resultants'][0]+=.001
    elif mutation=='predecessor':state['previous_state_sha256']='0'*64
    elif mutation=='skip':value['snapshots'].pop(1)
    elif mutation=='jacobian':state['response']['spatial_jacobian'][0][0]+=.01
    elif mutation=='applied':state['response']['applied_couple'][18]+=.01
    elif mutation=='boolean':last['displacements'][0]=False
    elif mutation=='missing-state':last['states']=[]
    elif mutation=='schema':value['schema']='GE_BEAM3_SUPPORTED_NATIVE_LINE_STATIC_RESTART_V1'
    elif mutation=='conservative':state['response']['conservative_potential']=True
    else:value['production_qualified']=True
    for record in value['snapshots']:
        for row in record['states']:
            s=row['state'];s['state_sha256']=sha({k:v for k,v in s.items() if k!='state_sha256'})
    value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'});changed=canonical(value)
    with pytest.raises(ValueError):decode_checkpoint(make(case),changed,expected_sha256=sha256(changed).hexdigest())
    save(tmp_path/'rejected.json',dict(case=case,mutation=mutation,mutated_sha256=sha256(changed).hexdigest(),rejected=True))


@pytest.mark.parametrize('mutation',['duplicate','nonfinite','noncanonical','external-hash','wrong-constant','support'])
def test_authority_before_solver(saved,mutation,monkeypatch,tmp_path):
    import anyfesolver.nonlinear_static as solver
    from anyfesolver.boundary import BoundaryCondition
    case,_,load,_,raw,_=saved;model=make(case);kwargs={}
    if mutation=='duplicate':raw=raw.replace(b'{',b'{"schema":"duplicate",',1)
    elif mutation=='nonfinite':raw=raw.replace(b'"parameter":0.0',b'"parameter":NaN',1)
    elif mutation=='noncanonical':raw+=b' '
    elif mutation=='wrong-constant':kwargs['constant']=EMPTY
    elif mutation=='support':model.add_boundary_condition(BoundaryCondition('changed',[3],{'ux':0.}))
    expected='0'*64 if mutation=='external-hash' else sha256(raw).hexdigest()
    def forbidden(*a,**k):raise AssertionError('entered solver before authority validation')
    monkeypatch.setattr(solver,'solve_static_nonlinear',forbidden)
    with pytest.raises(ValueError):solve_distributed_model(model,scale(load,.5),initial_checkpoint=raw,expected_sha256=expected,**kwargs)
    assert _PROGRAM.get() is None and _ACTIVE.get() is None
    save(tmp_path/'rejected.json',dict(case=case,mutation=mutation,rejected_before_solver=True))


def test_owned_capture_and_active_chain_binding(saved,tmp_path):
    case,_,load,whole,raw,_=saved;model=make(case);chain=decode_checkpoint(model,raw,expected_sha256=sha256(raw).hexdigest())
    initial=chain[-1];programme=_Program(model,model_identity(model),scale(load,.5),load,[],initial,sha256(raw).hexdigest());token=_PROGRAM.set(programme)
    try:
        calls=[];state=whole.element_states[1]
        made=capture_solver_state(model,1,state,exact_guard=lambda *a,**k:calls.append(k['context']))
        assert len(calls)==2 and canonical(made)==canonical(state) and not np.shares_memory(made['positions'],state['positions'])
        assert not made['response'].spatial_jacobian.flags.writeable
        changed=dict(state);changed['epoch']=999
        with pytest.raises(ValueError,match='differs from authenticated'):capture_solver_state(model,1,changed,exact_guard=lambda *a,**k:None)
        programme.initial['displacements']=initial['displacements']+1.
        with pytest.raises(ValueError,match='authority changed'):programme.require(model)
    finally:_PROGRAM.reset(token)
    save(tmp_path/'binding.json',dict(case=case,owned_capture=True,foreign_state_rejected=True,changed_initial_rejected=True))


def test_strict_hash_schema_deadline_and_count(saved,monkeypatch,tmp_path):
    import anyfesolver._ge_beam3_native_generalized_restart as module
    from anyfesolver._ge_beam3_native_line_restart import decode_checkpoint as decode_line
    case,_,load,_,raw,_=saved;model=make(case)
    with pytest.raises(ValueError):decode_line(model,raw,expected_sha256=sha256(raw).hexdigest())
    with pytest.raises(ValueError):solve_distributed_model(model,load,initial_checkpoint=raw)
    with pytest.raises(ValueError):solve_distributed_model(model,load,expected_sha256=sha256(raw).hexdigest())
    for bad in (None,[],(),(None,)*66):
        with pytest.raises(ValueError):encode_checkpoint(model,bad)
    clock=[]
    def advance():clock.append(True);return 61.*len(clock)
    monkeypatch.setattr(module,'monotonic',advance)
    with pytest.raises(RuntimeError,match='deadline'):decode_checkpoint(model,raw,expected_sha256=sha256(raw).hexdigest())
    save(tmp_path/'strict.json',dict(case=case,hash_schema_count_and_deadline_rejected=True))


def test_general_factorization_and_closed_direct_entry(saved,monkeypatch,tmp_path):
    from anyfesolver.boundary import LoadCase
    import anyfesolver.nonlinear_static as solver
    from anyfesolver.linalg import MatrixClass
    case,_,load,_,_,prefix=saved;model=make(case);kinds=[];original=solver.factorize
    def record(matrix,kind,*args,**kwargs):
        if str(kwargs.get('signature','')).startswith('nonlinear.static_newton'):kinds.append(kind.value)
        return original(matrix,kind,*args,**kwargs)
    monkeypatch.setattr(solver,'factorize',record)
    result,events=solve_distributed_model(model,scale(load,.5),initial_checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest(),steps=1)
    assert result.status=='completed' and result.info['native_generalized_general_matrix']
    assert kinds and all(k==MatrixClass.GENERAL.value for k in kinds)
    other=make(case)
    def forbidden(*a,**k):raise AssertionError('direct unsupported call entered mechanics')
    for e in other.mesh.elements.values():monkeypatch.setattr(e,'compute_stiffness_matrix',forbidden)
    with pytest.raises(ValueError,match='live native generalized'):solver.solve_static_nonlinear(other,LoadCase('unissued'))
    save(tmp_path/'general.json',dict(case=case,kinds=kinds,direct_entry_rejected_before_mechanics=True))


@pytest.mark.parametrize('mutation',['boolean-pair','nonnormal-pair','negative-accumulated','foreign-section','missing-component','fibre-schema'])
def test_generalized_history_schema_rejects_resealed_mutations(saved,mutation,tmp_path):
    case,_,_,_,raw,_=saved;value=json.loads(raw);state=value['snapshots'][-1]['states'][0]['state']
    history=state['origins']['stations'][0]
    if mutation=='boolean-pair':history['plastic'][0][0]=False
    elif mutation=='nonnormal-pair':history['plastic'][0]=[1.,1.]
    elif mutation=='negative-accumulated':history['accumulated']=[-1.,0.]
    elif mutation=='foreign-section':history['section_identity']='0'*64
    elif mutation=='missing-component':history['plastic'].pop()
    else:state['origins']['stations'][0]={'section_identity':history['section_identity'],'rows':[]}
    state['state_sha256']=sha({k:v for k,v in state.items() if k!='state_sha256'})
    value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'})
    changed=canonical(value)
    with pytest.raises(ValueError):decode_checkpoint(make(case),changed,expected_sha256=sha256(changed).hexdigest())
    save(tmp_path/'history-rejected.json',dict(case=case,mutation=mutation,rejected=True))
