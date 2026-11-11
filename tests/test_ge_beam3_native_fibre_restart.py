"""Native model restart/recovery with strict typed accepted-chain validation."""
import json
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver.boundary import LoadCase,BoundaryCondition
from anyfesolver.fe_core import FEModel
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from anyfesolver._ge_beam3_native_fibre_restart import encode_checkpoint,decode_checkpoint,capture_solver_state,supported_solver_coordinates
from anyfesolver._ge_beam3_native_fibre_recovery import recover_native_fields
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_ge_beam3_native_fibre_static_element import problem
from test_ge_beam3_schur_line_program import save,numerical_difference
from test_ge_beam3_equilibrium_multielement import inputs as connected_inputs

CASES=('straight-elastic','curved-plastic')
FORCE=(.34,-.014,.007)


def make(case): return problem(case=='curved-plastic',case=='curved-plastic')


def load(scale=1.):
    result=LoadCase('restart-load'); result.add_nodal_load(3,forces=scale*np.array(FORCE)); return result


@pytest.fixture(scope='module')
def packets(tmp_path_factory):
    results={}
    for case in CASES:
        root=tmp_path_factory.mktemp(case); m,e=make(case)
        initial=e.init_model_bound_nonlinear_state(m.mesh,e.section,1)
        whole=solve_static_nonlinear(m,load(),num_steps=2,max_iterations=12,tolerance=1e-12,num_layers=1,min_step_fraction=1.,record_increment_snapshots=True)
        save(root/'whole.json',dict(status=whole.status,displacements=whole.displacements,states=whole.element_states))
        assert whole.status=='completed',whole.info
        chain=(dict(load_factor=0.,displacements=np.zeros(18),states={1:initial}),)+tuple(
            dict(load_factor=float(s.load_factor),displacements=s.displacements,states=s.element_states) for s in whole.snapshots)
        save(root/'unencoded-chain.json',chain)
        raw=encode_checkpoint(m,((3,*FORCE),),chain); prefix=encode_checkpoint(m,((3,*FORCE),),chain[:2])
        save(root/'checkpoint.json',raw); save(root/'prefix.json',prefix)
        results[case]=(root,m,e,whole,raw,prefix)
    return results


@pytest.mark.parametrize('case',CASES)
def test_strict_roundtrip_and_native_physical_recovery(packets,case):
    root,_,_,whole,raw,_=packets[case]; m,e=make(case)
    forces,chain=decode_checkpoint(m,raw,expected_sha256=sha256(raw).hexdigest())
    assert encode_checkpoint(m,forces,chain)==raw and canonical(chain[-1]['states'])==canonical(whole.element_states)
    state=chain[-1]['states'][1]; before=canonical(state)
    recovered=recover_native_fields(e,m.mesh,state,expected_committed_total_u=whole.displacements)
    assert canonical(state)==before and len(recovered['stations'])==8
    errors=[]
    for row in recovered['stations']:
        local=row['resultants']+row['resultants_low']; global_value=row['global_resultants']+row['global_resultants_low']
        variation=np.sin(np.arange(6)+.4); frame=row['current_frame']
        global_variation=np.r_[frame@variation[:3],frame@variation[3:]]
        error=abs(local@variation-global_value@global_variation)/max(1.,abs(local@variation))
        assert error<=1e-11; errors.append(error)
        assert np.isfinite(row['current_position']).all() and len(row['fibres'])==4
    save(root/'recovery.json',dict(recovery=recovered,work_errors=errors,history_unchanged=True))


@pytest.mark.parametrize('case',CASES)
def test_actual_solver_continuation_from_typed_prefix(packets,case):
    root,_,_,whole,_,prefix=packets[case]; m,e=make(case)
    _,chain=decode_checkpoint(m,prefix,expected_sha256=sha256(prefix).hexdigest()); state=chain[-1]
    resumed=solve_static_nonlinear(m,load(.5),constant_load_case=load(.5),num_steps=1,max_iterations=12,tolerance=1e-12,
        num_layers=1,min_step_fraction=1.,initial_element_states=state['states'],initial_displacements=state['displacements'],
        equilibrate_initial_state=False,record_increment_snapshots=True)
    save(root/'resumed.json',dict(status=resumed.status,displacements=resumed.displacements,states=resumed.element_states))
    assert resumed.status=='completed',resumed.info
    difference=numerical_difference(resumed.element_states,whole.element_states)
    assert canonical(resumed.element_states)==canonical(whole.element_states)
    np.testing.assert_array_equal(resumed.displacements,whole.displacements)
    save(root/'continuation.json',dict(bitwise_state_match=True,maximum_difference=difference,production_qualified=False))


def reseal(value):
    for record in value['snapshots']:
        for row in record['states']:
            state=row['state']; state['state_sha256']=sha({k:v for k,v in state.items() if k!='state_sha256'})
    value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'})
    return canonical(value)


@pytest.mark.parametrize('mutation',['bool','history','predecessor','seed','load','rotation-chain','missing','skip','schema'])
def test_resealed_checkpoint_mutations_rejected(packets,mutation,tmp_path):
    raw=packets['curved-plastic'][4]; value=json.loads(raw); last=value['snapshots'][-1]; state=last['states'][0]['state']
    if mutation=='bool': last['displacements'][0]=False
    elif mutation=='history': state['origins']['stations'][0]['rows'][0][0]+=.001
    elif mutation=='predecessor': state['previous_state_sha256']='0'*64
    elif mutation=='seed': state['seed_resultants'][0]+=.001
    elif mutation=='load': last['load_factor']+=.125
    elif mutation=='rotation-chain': last['displacements'][15]+=.005; state['committed_total_u'][15]+=.005
    elif mutation=='missing': last['states']=[]
    elif mutation=='skip': value['snapshots'].pop(1)
    else: value['schema']='legacy'
    mutated=reseal(value); m,_=make('curved-plastic')
    with pytest.raises(ValueError): decode_checkpoint(m,mutated,expected_sha256=sha256(mutated).hexdigest())
    save(tmp_path/'rejected.json',dict(mutation=mutation,mutated_sha256=sha256(mutated).hexdigest(),rejected=True))


@pytest.mark.parametrize('mutation',['duplicate','nonfinite','noncanonical','external-hash'])
def test_strict_json_and_external_hash(packets,mutation):
    raw=packets['straight-elastic'][4]; m,_=make('straight-elastic')
    if mutation=='duplicate': raw=raw.replace(b'{',b'{"schema":"duplicate",',1)
    elif mutation=='nonfinite': raw=raw.replace(b'"load_factor":0.0',b'"load_factor":NaN',1)
    elif mutation=='noncanonical': raw+=b' '
    expected='0'*64 if mutation=='external-hash' else sha256(raw).hexdigest()
    with pytest.raises(ValueError): decode_checkpoint(m,raw,expected_sha256=expected)


def test_changed_support_binding_rejected(packets):
    raw=packets['curved-plastic'][4]; m,_=make('curved-plastic')
    m.add_boundary_condition(BoundaryCondition('tip-x',[3],{'ux':0.}))
    with pytest.raises(ValueError,match='model'): decode_checkpoint(m,raw,expected_sha256=sha256(raw).hexdigest())


def test_solver_capture_detaches_typed_state(packets):
    _,_,_,whole,_,_=packets['curved-plastic']; m,_=make('curved-plastic'); calls=[]
    original=whole.element_states[1]; before=canonical(original)
    made=capture_solver_state(m,1,original,exact_guard=lambda *a,**k:calls.append(k['context']))
    assert len(calls)==2 and canonical(made)==before and canonical(original)==before
    assert not np.shares_memory(made['positions'],original['positions'])
    assert not made['positions'].flags.writeable and not made['response'].hessian.flags.writeable


def test_solver_capture_observation_guard_on_failure():
    m,_=make('straight-elastic'); calls=[]
    with pytest.raises((TypeError,ValueError)):
        capture_solver_state(m,1,object(),exact_guard=lambda *a,**k:calls.append(k['context']))
    assert len(calls)==1


def test_exact_supported_coordinate_map_and_mutation_rejection(packets):
    from scipy import sparse
    m,_=make('straight-elastic');u=packets['straight-elastic'][3].displacements
    transform=sparse.eye(18,format='csr')[:,6:];offset=np.zeros(18)
    q,scale=supported_solver_coordinates(m,transform,offset,u)
    assert scale==0. and np.array_equal(q,u[6:]) and np.array_equal(transform@q,u)
    transform.data[0]=2.
    with pytest.raises(ValueError,match='selection'): supported_solver_coordinates(m,transform,offset,u)


def test_existing_generic_copier_still_rejects_arbitrary_dataclass():
    from dataclasses import dataclass
    from anyfesolver.current_state_tangent import _guarded_owned_input_snapshot
    @dataclass
    class Unregistered:
        value: float=0.
    m,_=make('straight-elastic')
    with pytest.raises(TypeError,match='unsupported input type'):
        _guarded_owned_input_snapshot(m,Unregistered(),path='test',_exact_guard=lambda *a,**k:None)


def test_connected_model_genesis_roundtrip(tmp_path):
    source,_,_=connected_inputs('curved-2-clamped'); m=FEModel('connected-native-restart')
    for i,node in source.mesh.nodes.items(): m.add_node(i,*node.coords())
    states={}
    for i,old in source.mesh.elements.items():
        e=NativeFibreStaticElement(i,old.node_ids,old.operator.reference,old.section,order=4)
        m.add_element(i,e); m.materials[e.material_name]=e.section
    for bc in source.boundary_conditions: m.add_boundary_condition(bc)
    for i,e in m.mesh.elements.items(): states[i]=e.init_model_bound_nonlinear_state(m.mesh,e.section,1)
    snapshots=(dict(load_factor=0.,displacements=np.zeros(m.mesh.dof_manager.total_dofs),states=states),)
    raw=encode_checkpoint(m,((3,.1,0.,0.),),snapshots)
    forces,restored=decode_checkpoint(m,raw,expected_sha256=sha256(raw).hexdigest())
    assert encode_checkpoint(m,forces,restored)==raw
    save(tmp_path/'connected.json',raw)
