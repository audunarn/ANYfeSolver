"""Loaded connected actual-driver restart; no private surrogate controller."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import LoadCase
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from anyfesolver._ge_beam3_native_fibre_restart import encode_checkpoint,decode_checkpoint
from anyfesolver._ge_beam3_native_fibre_recovery import recover_native_fields
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_ge_beam3_equilibrium_multielement import inputs
from test_ge_beam3_spatial_nodal_moments import model as section_model
from test_ge_beam3_schur_line_program import save

CASES=('curved-2-plastic-cantilever','curved-2-elastic-clamped')


def make(case):
    if case not in CASES: raise ValueError('registered connected native case')
    plastic=case==CASES[0];source,_,_=inputs('curved-2-clamped')
    m=FEModel(case);section=section_model(plastic=plastic,coupled=True).mesh.elements[1].section
    for i,node in source.mesh.nodes.items():m.add_node(i,*node.coords())
    for i,old in source.mesh.elements.items():
        e=NativeFibreStaticElement(i,old.node_ids,old.operator.reference,section,order=4)
        m.add_element(i,e);m.materials[e.material_name]=section
    for bc in source.boundary_conditions[:1 if plastic else 2]:m.add_boundary_condition(bc)
    return m,((5,.30,-.010,.006),) if plastic else ((3,.02,-.04,.015),)


def load(forces,scale):
    value=LoadCase('connected-native-nodal-force')
    for node,*force in forces:value.add_nodal_load(node,forces=scale*np.array(force))
    return value


def solve(m,forces,*,scale=1.,constant=None,steps=2,state=None):
    return solve_static_nonlinear(m,load(forces,scale),constant_load_case=None if constant is None else load(forces,constant),
        num_steps=steps,max_iterations=12,tolerance=1e-12,num_layers=1,min_step_fraction=1.,record_increment_snapshots=True,
        initial_displacements=None if state is None else state['displacements'],
        initial_element_states=None if state is None else state['states'],equilibrate_initial_state=False)


def packet(result):return dict(status=result.status,displacements=result.displacements,states=result.element_states)


@pytest.fixture(scope='module',params=CASES)
def connected(request,tmp_path_factory):
    case=request.param;root=tmp_path_factory.mktemp(case);m,forces=make(case)
    initial={i:e.init_model_bound_nonlinear_state(m.mesh,e.section,1) for i,e in m.mesh.elements.items()}
    whole=solve(m,forces);save(root/'whole.json',packet(whole));assert whole.status=='completed',whole.info
    chain=(dict(load_factor=0.,displacements=np.zeros(30),states=initial),)+tuple(
        dict(load_factor=float(s.load_factor),displacements=s.displacements,states=s.element_states) for s in whole.snapshots)
    save(root/'unencoded-chain.json',chain)
    raw=encode_checkpoint(m,forces,chain);prefix=encode_checkpoint(m,forces,chain[:2]);save(root/'checkpoint.json',raw);save(root/'prefix.json',prefix)
    return case,root,m,forces,whole,raw,prefix


def test_connected_equilibrium_reactions_and_physical_recovery(connected):
    case,root,m,forces,whole,raw,_=connected
    _,chain=decode_checkpoint(m,raw,expected_sha256=sha256(raw).hexdigest());records=[]
    fixed=sorted({i for bc in m.boundary_conditions for i,_ in bc.get_constrained_dofs(m.mesh.dof_manager)})
    fixed_nodes=[node for node in m.mesh.nodes if all(i in fixed for i in m.mesh.dof_manager.get_node_dofs(node))]
    for snapshot in chain:
        u=snapshot['displacements'];states=snapshot['states'];before=canonical(states);internal=np.zeros(30);recovery={}
        for i,e in m.mesh.elements.items():
            mapping=list(e.get_dof_mapping(m.mesh));internal[mapping]+=states[i]['response'].residual
            recovery[str(i)]=recover_native_fields(e,m.mesh,states[i],expected_committed_total_u=u[mapping])
        external=np.zeros(30)
        for node,*f in forces:external[6*(node-1):6*(node-1)+3]=snapshot['load_factor']*np.array(f)
        residual=internal-external;free=[i for i in range(30) if i not in fixed]
        error=float(np.linalg.norm(residual[free]));assert error<=1e-11
        reactions=np.zeros(30);reactions[fixed]=residual[fixed]
        force_balance=reactions.reshape(5,6)[:,:3].sum(axis=0)+external.reshape(5,6)[:,:3].sum(axis=0)
        moment_balance=np.zeros(3)
        for node in m.mesh.nodes:
            offset=6*(node-1);position=np.array(m.mesh.nodes[node].coords())+u[offset:offset+3]
            wrench=reactions[offset:offset+6]+external[offset:offset+6]
            moment_balance+=np.cross(position,wrench[:3])+wrench[3:]
        assert np.linalg.norm(force_balance)<=1e-11 and np.linalg.norm(moment_balance)<=1e-11
        np.testing.assert_array_equal(states[1]['committed_nodal_rotation_matrices'][2],states[2]['committed_nodal_rotation_matrices'][0])
        interface=states[1]['response'].residual[12:18]+states[2]['response'].residual[:6]-external[12:18]
        assert np.linalg.norm(interface)<=1e-11 and canonical(states)==before
        records.append(dict(factor=snapshot['load_factor'],free_residual=error,force_balance=force_balance,moment_balance=moment_balance,
            shared_interface_residual=interface,fixed_nodes=fixed_nodes,reactions=reactions,recovery=recovery,history_unchanged=True))
    plastic_rows=sum(row[2]>0 for s in whole.element_states.values() for st in s['response'].history.stations for row in st.rows)
    assert plastic_rows>0 if case==CASES[0] else plastic_rows==0
    save(root/'equilibrium-recovery.json',dict(case=case,records=records,plastic_rows=plastic_rows,production_qualified=False))


def test_connected_restart_continuation(connected):
    case,root,_,forces,whole,_,prefix=connected;m,_=make(case)
    decoded,chain=decode_checkpoint(m,prefix,expected_sha256=sha256(prefix).hexdigest());assert decoded==forces
    resumed=solve(m,forces,scale=.5,constant=.5,steps=1,state=chain[-1]);save(root/'resumed.json',packet(resumed))
    assert resumed.status=='completed',resumed.info
    assert canonical(packet(resumed))==canonical(packet(whole))
    completed=chain+(dict(load_factor=1.,displacements=resumed.displacements,states=resumed.element_states),)
    assert encode_checkpoint(m,forces,completed)==connected[5]


def test_connected_unload_and_checkpoint(connected):
    case,root,_,forces,_,raw,_=connected;m,_=make(case)
    _,chain=decode_checkpoint(m,raw,expected_sha256=sha256(raw).hexdigest());previous=chain[-1]
    unloaded=solve(m,forces,scale=-.5,constant=1.,steps=1,state=previous);save(root/'unloaded.json',packet(unloaded))
    assert unloaded.status=='completed',unloaded.info
    complete=chain+(dict(load_factor=.5,displacements=unloaded.displacements,states=unloaded.element_states),)
    out=encode_checkpoint(m,forces,complete);save(root/'unloaded-checkpoint.json',out)
    fresh,_=make(case);f,decoded=decode_checkpoint(fresh,out,expected_sha256=sha256(out).hexdigest())
    assert encode_checkpoint(fresh,f,decoded)==out
    for i,state in unloaded.element_states.items():
        assert state['epoch']==3 and canonical(state['origins'])==canonical(previous['states'][i]['response'].history)
        for old,new in zip(previous['states'][i]['response'].history.stations,state['response'].history.stations):
            assert all(n[2]+n[3]>=o[2]+o[3] for o,n in zip(old.rows,new.rows))


@pytest.mark.parametrize('mutation',['connectivity','element-state-swap','force-pattern'])
def test_connected_identity_mutations(connected,mutation,tmp_path):
    case,_,_,_,_,raw,_=connected;m,_=make(case);value=json.loads(raw)
    if mutation=='connectivity':m.mesh.elements[2].node_ids=(5,4,3)
    elif mutation=='element-state-swap':
        rows=value['snapshots'][-1]['states'];rows[0]['state'],rows[1]['state']=rows[1]['state'],rows[0]['state']
    else:value['nodal_forces'][0][2]+=.01
    value['checkpoint_sha256']=sha({k:v for k,v in value.items() if k!='checkpoint_sha256'});changed=canonical(value)
    with pytest.raises(ValueError):decode_checkpoint(m,changed,expected_sha256=sha256(changed).hexdigest())
    save(tmp_path/'rejected.json',dict(case=case,mutation=mutation,rejected=True,checkpoint_sha256=sha256(changed).hexdigest()))
