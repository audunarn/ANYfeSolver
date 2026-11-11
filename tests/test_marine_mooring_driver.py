"""Short common-Newton development trajectories, not W6 qualification."""
from dataclasses import replace
import copy
import json
import numpy as np
import pytest

from anyfesolver.marine_mooring import (MooringLine,MooringLineAssembler,MooringNodeState,
    FixedEndpoint,DynamicLineNodeEndpoint as Node,RigidBodyAttachmentEndpoint as Fairlead)
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_contributions import (WorldWrenchContribution,DynamicContribution,
                                           ContributionResult,HISTORY_CHART)
from anyfesolver.marine_mechanical import (MarineMechanicalModel,MarineMechanicalRunSpec,
    MarineMechanicalInputError,MarineMechanicalSolveError,solve_marine_mechanical,
    model_to_dict,model_from_dict,result_to_dict,MOORING_API_VERSION,API_VERSION)


def case(force=0.):
    bodies=tuple(RigidBody6D(b,5.,(0.,0.,0.),np.diag([2.,3.,4.])) for b in ('A','B'))
    lines=tuple(MooringLine(str(i),(Fairlead('A',(0.,i-1.,0.)),Node(i),Fairlead('B',(0.,i-1.,0.))),
                           [1.,1.],[40.,40.],[0.,0.],[2.,2.],[0.,0.]) for i in range(3))
    assembler=MooringLineAssembler(bodies,lines,3,(0.,0.,0.),1025.,0.)
    contributions=() if not force else (WorldWrenchContribution('load',bodies[1],(0.,0.,0.),(force,0.,0.)),)
    model=MarineMechanicalModel(bodies,contributions=contributions,mooring=assembler)
    states={b.body_id:b.state(position=(x,0.,-2.)) for b,x in zip(bodies,(-2.2,2.2))}
    nodes=MooringNodeState([[0.,i-1.,-2.] for i in range(3)],np.zeros((3,3)),np.full((3,3),123.))
    return model,states,nodes


@pytest.mark.parametrize('force',[0.,2.])
def test_two_bodies_three_lines_match_independent_linear_newmark(force):
    model,states,nodes=case(force)
    dt=.001
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(dt,4),initial_nodes=nodes)
    # Independent five-coordinate linear system; no product assembler used.
    mass=np.diag([8.,8.,2.,2.,2.]);K=np.zeros((5,5));f=np.array([0.,force,0.,0.,0.])
    for j in range(2,5):
        for b in (0,1):K[b,b]+=40.;K[j,j]+=40.;K[b,j]-=40.;K[j,b]-=40.
    q=np.array([-1.2,1.2,0.,0.,0.]);v=np.zeros(5);a=np.linalg.solve(mass,f-K@q)
    initial_energy=None
    for frame in result.accepted:
        actual=np.r_[frame.positions['A'][0]+1.,frame.positions['B'][0]-1.,frame.nodes.positions[:,0]]
        np.testing.assert_allclose(actual,q,atol=2e-13,rtol=1e-12)
        np.testing.assert_allclose(np.r_[frame.world_accelerations['A'][0],frame.world_accelerations['B'][0],frame.nodes.accelerations[:,0]],a,atol=2e-11)
        np.testing.assert_allclose(sum(frame.mooring_diagnostics['node_force_components'].values()),np.zeros((3,3)),atol=2e-11)
        assert frame.residual_norm<=1e-10
        energy=frame.kinetic_energy+frame.mooring_diagnostics['elastic_energy']
        if initial_energy is None:initial_energy=energy
        if not force:assert energy==pytest.approx(initial_energy,rel=3e-13)
        q_pred=q+dt*v+.25*dt**2*a;v_pred=v+.5*dt*a
        a=np.linalg.solve(mass+.25*dt**2*K,f-K@q_pred)
        q=q_pred+.25*dt**2*a;v=v_pred+.5*dt*a
    assert result.status=='completed' and result.node_count==3
    assert result.accepted[0].world_accelerations['A'][0]==pytest.approx(18.)
    assert np.all(nodes.accelerations==123.)  # caller snapshot never mutated
    encoded=result_to_dict(result)
    assert encoded['api_version']==MOORING_API_VERSION and encoded['node_count']==3
    assert len(encoded['accepted'][0]['nodes']['positions'])==3
    json.dumps(encoded,allow_nan=False)


def test_cancel_retains_only_atomic_accepted_node_and_body_frames():
    model,states,nodes=case()
    cancelled=False
    def progress(frame):
        nonlocal cancelled
        if frame.step==1:cancelled=True
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.001,4),
        initial_nodes=nodes,cancel=lambda:cancelled,progress=progress)
    assert result.status=='cancelled' and [f.step for f in result.accepted]==[0,1]
    assert all(f.nodes.positions.shape==(3,3) for f in result.accepted)
    assert result_to_dict(result)['status']=='cancelled'


def test_mooring_domain_failure_keeps_initial_accepted_node_evidence():
    b=RigidBody6D('A',10.,(0.,0.,0.),np.eye(3))
    line=MooringLine('line',(FixedEndpoint((0.,0.,-.2)),Node(0),Fairlead('A',(0.,0.,0.))),
                    [.1,.1],[1.,1.],[0.,0.],[1.,1.],[0.,0.])
    assembler=MooringLineAssembler((b,),(line,),1,(0.,0.,0.),1025.,0.)
    model=MarineMechanicalModel((b,),mooring=assembler)
    nodes=MooringNodeState([[.5,0.,-.2]],np.zeros((1,3)),np.zeros((1,3)))
    with pytest.raises(MarineMechanicalSolveError) as raised:
        solve_marine_mechanical(model,{'A':b.state(position=(1.,0.,-.1),velocity=(0.,0.,2.,0.,0.,0.))},
                                MarineMechanicalRunSpec(1.,1),initial_nodes=nodes)
    error=raised.value
    assert error.reason=='mooring_domain' and error.partial.status=='failed'
    assert len(error.partial.accepted)==1
    np.testing.assert_array_equal(error.partial.accepted[0].nodes.positions,nodes.positions)
    assert result_to_dict(error.partial)['status']=='failed'


def test_named_model_roundtrip_and_old_schema_cannot_erase_lines():
    model,states,nodes=case()
    encoded=model_to_dict(model)
    assert encoded['api_version']==MOORING_API_VERSION
    restored=model_from_dict(json.loads(json.dumps(encoded)))
    assert model_to_dict(restored)==encoded
    result=solve_marine_mechanical(restored,states,MarineMechanicalRunSpec(.001,1),initial_nodes=nodes)
    assert result.node_count==3
    bad=copy.deepcopy(encoded);bad['api_version']=API_VERSION
    with pytest.raises(MarineMechanicalInputError):model_from_dict(bad)
    bad=copy.deepcopy(encoded);del bad['mooring']
    with pytest.raises(MarineMechanicalInputError):model_from_dict(bad)
    bad=copy.deepcopy(encoded);bad['mooring']['lines'][0]['endpoints'][1]['type']='python_code'
    with pytest.raises(MarineMechanicalInputError):model_from_dict(bad)
    old=model_to_dict(replace(model,mooring=None))
    assert old['api_version']==API_VERSION and 'mooring' not in old
    assert model_to_dict(model_from_dict(old))==old
    from anyfesolver.marine_mooring import data_from_dict,MooringLineError
    with pytest.raises(MooringLineError):data_from_dict(encoded['mooring'],('invalid',))


def test_node_and_gravity_mismatches_fail_before_execution():
    model,states,nodes=case()
    with pytest.raises(MarineMechanicalInputError):
        solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.001,1))
    with pytest.raises(MarineMechanicalInputError):
        solve_marine_mechanical(replace(model,mooring=None),states,MarineMechanicalRunSpec(.001,1),initial_nodes=nodes)
    with pytest.raises(MarineMechanicalInputError,match='gravity'):
        replace(model,gravity_world=(0.,0.,-9.81))
    with pytest.raises(MarineMechanicalInputError,match='exact ordered'):
        replace(model,bodies=tuple(reversed(model.bodies)))


def test_rotating_fairleads_rechart_preserves_nodes_and_physical_residual():
    model,states,nodes=case()
    states={bid:replace(s,rotation_vector=(.1,-.07,.05),velocity=(0.,0.,0.,.2,-.1,.15)) for bid,s in states.items()}
    nodes=replace(nodes,velocities=np.full((3,3),.02))
    dt=.001
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(dt,3),initial_nodes=nodes)
    for previous,frame in zip(result.accepted,result.accepted[1:]):
        expected_q=previous.nodes.positions+dt*previous.nodes.velocities+.25*dt**2*(previous.nodes.accelerations+frame.nodes.accelerations)
        expected_v=previous.nodes.velocities+.5*dt*(previous.nodes.accelerations+frame.nodes.accelerations)
        np.testing.assert_allclose(frame.nodes.positions,expected_q,atol=1e-14)
        np.testing.assert_allclose(frame.nodes.velocities,expected_v,atol=1e-14)
        recharted={bid:model.bodies[i].state(position=frame.positions[bid],anchor=frame.rotations[bid],
                    velocity=frame.world_velocities[bid],acceleration=frame.world_accelerations[bid]) for i,bid in enumerate(model.body_ids)}
        residual=model.mooring.force(recharted,frame.nodes)
        for i,b in enumerate(model.bodies):residual[6*i:6*i+6]-=b.inertial_force(recharted[b.body_id])
        np.testing.assert_allclose(residual,0.,atol=2e-10)
    assert np.linalg.norm(result.accepted[-1].rotations['A']-result.accepted[0].rotations['A'])>1e-5


@pytest.mark.parametrize('fault',['cancel','transport'])
def test_precommit_transport_failure_or_cancel_retains_body_node_and_history(fault):
    model,states,nodes=case()
    nodes=replace(nodes,velocities=np.c_[np.full(3,.1),np.zeros((3,2))])
    flag=False;last=[]
    class Memory(DynamicContribution):
        contribution_id='memory';ownership_key='test-memory';history_basis=HISTORY_CHART
        def evaluate_trial(self,history,states,layout):
            count=history.get('memory',{'steps':0})['steps']
            z=np.zeros((layout.size,layout.size))
            return ContributionResult('memory',np.zeros(layout.size),z,z,z,{},dict(steps=count+1))
        def history_transport(self,history,info):
            nonlocal flag
            assert info.for_body('A').body_id=='A'
            if history['steps']==2:
                if fault=='cancel':flag=True
                else:raise ValueError('deliberate transport failure')
            return history
    model=replace(model,contributions=(Memory(),))
    kwargs=dict(initial_nodes=nodes,cancel=lambda:flag,progress=lambda frame:last.append(frame))
    if fault=='cancel':
        result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.001,3),**kwargs)
        assert result.status=='cancelled'
    else:
        with pytest.raises(MarineMechanicalSolveError) as raised:
            solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.001,3),**kwargs)
        result=raised.value.partial
        assert result.status=='failed'
    assert [f.step for f in result.accepted]==[0,1] and result.histories['memory']['steps']==1
    assert result.accepted[-1] is last[-1]
    np.testing.assert_array_equal(result.accepted[-1].nodes.positions,last[-1].nodes.positions)
    assert result.accepted[-1].nodes.positions[0,0]>0.


def test_invalid_full_newton_mooring_candidate_reports_actual_domain():
    b=RigidBody6D('A',10.,(0.,0.,0.),np.eye(3))
    line=MooringLine('line',(FixedEndpoint((0.,0.,-.2)),Node(0),Fairlead('A',(0.,0.,0.))),
                    [.1,.1],[1.,1.],[0.,0.],[1.,1.],[0.,0.])
    assembler=MooringLineAssembler((b,),(line,),1,(0.,0.,0.),1025.,0.)
    class StepLoad(WorldWrenchContribution):
        def evaluate_trial_with_context(self,history,states,layout,info):
            if info.phase=='initialization':return super().evaluate_trial(history,states,layout)
            return WorldWrenchContribution('load',b,(0.,0.,0.),(0.,0.,400.)).evaluate_trial(history,states,layout)
    model=MarineMechanicalModel((b,),contributions=(StepLoad('load',b,(0.,0.,0.)),),mooring=assembler)
    nodes=MooringNodeState([[.5,0.,-.2]],np.zeros((1,3)),np.zeros((1,3)))
    with pytest.raises(MarineMechanicalSolveError) as raised:
        solve_marine_mechanical(model,{'A':b.state(position=(1.,0.,-.1))},MarineMechanicalRunSpec(.2,1,line_search=False),initial_nodes=nodes)
    assert raised.value.reason=='mooring_domain'
    assert len(raised.value.partial.accepted)==1
    assert any('mooring domain' in log.detail for log in raised.value.logs)
