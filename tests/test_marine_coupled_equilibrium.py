"""Independent static development checks, no BEM or qualification campaign."""
from dataclasses import replace
from importlib.util import find_spec
import numpy as np
import pytest
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_hydro_coordinates import EquilibriumHydroFrame
from anyfesolver.marine_potential_flow import (PotentialFlowDevelopmentData,
    PotentialFlowDevelopmentContribution,equilibrium_geometry_hash)
from anyfesolver.marine_contributions import (BodyLayout,CommittedHistory,WorldWrenchContribution,
    DynamicContribution,ContributionResult)
from anyfesolver.marine_mechanical import MarineMechanicalModel
from anyfesolver.marine_link import MechanicalLink6D
from anyfesolver.marine_mooring import (MooringLine,MooringLineAssembler,FixedEndpoint as Anchor,
    DynamicLineNodeEndpoint as Node,RigidBodyAttachmentEndpoint as Fairlead)
from anyfesolver.marine_coupled_equilibrium import (CoupledStaticSpec,
    solve_coupled_marine_equilibrium,evaluate_marine_static)
from anyfesolver.marine_equilibrium import MooringEquilibriumError,initialize_mooring_catenaries


def fixture(*, lines=False, members=False, finite=False, preload=True):
    bodies=tuple(RigidBody6D(bid,4.,(0.,0.,0.),np.diag([2.,3.,4.])) for bid in ('a','b'))
    positions=[(0.,0.,-2.),(3.,0.,-2.)]
    states={b.body_id:b.state(position=p,rotation_vector=(.12,-.08,.1) if finite else (0.,0.,0.))
            for b,p in zip(bodies,positions)}
    frames=tuple(EquilibriumHydroFrame(b.body_id,p,np.eye(3),np.eye(3),(0.,0.,0.)) for b,p in zip(bodies,positions))
    n=12;C=np.diag(np.tile([120.,130.,140.,60.,70.,80.],2));C[0,6]=C[6,0]=10.
    baseline=np.zeros(n);baseline[[2,8]]=4.*9.81
    # Nonzero excitation and radiation explicitly MUST NOT enter statics.
    data=PotentialFlowDevelopmentData(frames,equilibrium_geometry_hash(frames),.1,
        np.tile(np.eye(n),(3,1,1)),np.eye(n),C,baseline,np.full(n,1e5),np.full(n,-1e5),2.,
        np.tile([2.,1.],(2,1)),'synthetic static contract fixture; not source admission')
    fluid=PotentialFlowDevelopmentContribution(data)
    K=np.diag([30.,0.,0.,0.,0.,0.])
    link=MechanicalLink6D('link','a','b',(0.,0.,0.),(0.,0.,0.),np.eye(3),np.eye(3),
                          (3.,0.,0.),np.eye(3),K,np.zeros((6,6)))
    contributions=[fluid]
    if preload:contributions.append(WorldWrenchContribution('preload',bodies[0],(0.,0.,0.),(6.,0.,0.)))
    mooring=None;nodes=None
    if lines:
        records=[]
        for j,(anchor,bid,point) in enumerate((((-4.,-2.,-4.),'a',(.1,-.1,0.)),
             ((7.,-2.,-4.),'b',(-.1,-.1,0.)),((1.5,5.,-4.),'a',(0.,.2,0.)))):
            ends=(Anchor(anchor),Node(2*j),Node(2*j+1),Fairlead(bid,point))
            records.append(MooringLine('line'+str(j),ends,np.full(3,6.8/3),np.full(3,1e4),
                np.zeros(3),np.full(3,2.),np.full(3,.001)))
        mooring=MooringLineAssembler(bodies,tuple(records),6,(0.,0.,-9.81),1000.,0.)
        nodes=initialize_mooring_catenaries(mooring,states)
    if members:
        if find_spec('anyloads') is None:
            pytest.importorskip('anyloads', reason='optional marine-morison provider is not installed')
        from anyloads.marine import AiryEnvironment,CurrentProfile,MorisonSection,HydrodynamicOwnership
        from anyfesolver.marine_morison import MorisonMemberSetContribution,MorisonMember,MemberEndpoint
        rods=tuple(MorisonMember('brace'+str(j),MemberEndpoint('a',(0.,0.,-.2-j*.1)),
            MemberEndpoint('b',(0.,0.,-.2-j*.1)),MorisonSection(.1,1.,1.),
            HydrodynamicOwnership(bem_included=False,buoyancy_owner='morison')) for j in range(2))
        env=AiryEnvironment(20.,current=CurrentProfile.uniform(20.,(0.,.05,0.)),density=1000.)
        contributions.append(MorisonMemberSetContribution('members',bodies,rods,env))
    model=MarineMechanicalModel(bodies,(link,),(0.,0.,-9.81),tuple(contributions),mooring)
    return model,states,nodes


def test_closed_form_two_body_surge_equilibrium_no_excitation_or_memory():
    model,states,nodes=fixture()
    result=solve_coupled_marine_equilibrium(model,states,spec=CoupledStaticSpec(force_scale=100.,moment_scale=100.))
    assert result.status=='converged',result.message
    expected=np.linalg.solve(np.array([[150.,-20.],[-20.,150.]]),[6.,0.])
    np.testing.assert_allclose([result.states['a'].position[0],result.states['b'].position[0]-3.],expected,atol=1e-9,rtol=0.)
    for s in result.states.values():
        np.testing.assert_array_equal(s.velocity,0.);np.testing.assert_array_equal(s.rotation_vector,0.)
    assert result.audit['temporary_support_fraction']==0.
    assert result.audit['force_residual']<=result.audit['force_limit']
    assert result.audit['moment_residual']<=result.audit['moment_limit']
    assert result.audit['potential_geometry'][0]['coefficient_rebind_required']
    assert model.contributions[0].data.equilibrium_hash==equilibrium_geometry_hash(model.contributions[0].data.frames)
    with pytest.raises(TypeError):result.audit['force_residual']=100.
    with pytest.raises(TypeError):result.states['a']=states['b']


def test_already_balanced_geometry_keeps_original_coefficient_identity():
    model,states,_=fixture(preload=False)
    result=solve_coupled_marine_equilibrium(model,states)
    assert result.status=='converged',result.message
    assert not result.audit['potential_geometry'][0]['coefficient_rebind_required']
    assert all(s.iterations==0 for s in result.stages)
    assert not any(k.startswith('static-support') for k in result.audit['components'])


def test_finite_rotation_holds_and_accepted_recharts_reach_same_physical_root():
    model,states,_=fixture(finite=True)
    result=solve_coupled_marine_equilibrium(model,states,spec=CoupledStaticSpec(force_scale=100.,moment_scale=100.))
    assert result.status=='converged',result.message
    expected=np.linalg.solve(np.array([[150.,-20.],[-20.,150.]]),[6.,0.])
    np.testing.assert_allclose([result.states['a'].position[0],result.states['b'].position[0]-3.],expected,atol=2e-8,rtol=0.)
    for s in result.states.values():
        np.testing.assert_allclose(s.rotation,np.eye(3),atol=2e-8,rtol=0.)
        np.testing.assert_array_equal(s.rotation_vector,0.)
    assert any(np.linalg.norm(s.states['a'].rotation-np.eye(3))>1e-3 for s in result.stages[:-1])


@pytest.mark.parametrize('members',[False,True])
def test_two_bodies_link_three_lines_and_current_members_converge(members):
    model,states,nodes=fixture(lines=True,members=members)
    saved=np.array(nodes.positions)
    result=solve_coupled_marine_equilibrium(model,states,nodes,CoupledStaticSpec(force_scale=100.,moment_scale=100.))
    assert result.status=='converged',result.message
    assert result.line_result.status=='converged'
    out=evaluate_marine_static(model,result.states,result.nodes)
    assert result.audit['tolerance_ratio']<=1.
    assert np.max(abs(out.force[12:]))<=result.audit['force_limit']
    assert np.linalg.norm(result.nodes.positions-saved)>1e-3
    np.testing.assert_array_equal(nodes.positions,saved)
    assert len(result.states)==2 and result.nodes.positions.shape==(6,3)
    assert len(model.contributions[0].data.kernel)==3
    if members:assert np.linalg.norm(result.audit['components']['members'])>0.


def test_full_static_tangent_matches_independent_position_differences():
    model,states,nodes=fixture(lines=True,members=True,finite=True)
    out=evaluate_marine_static(model,states,nodes);h=6e-6;cols=[]
    for j in range(30):
        plus=dict(states);minus=dict(states);np_=nodes;nm_=nodes
        if j<12:
            bid=model.body_ids[j//6];k=j%6;field='position' if k<3 else 'rotation_vector';idx=k if k<3 else k-3
            xp=np.array(getattr(states[bid],field));xm=xp.copy();xp[idx]+=h;xm[idx]-=h
            plus[bid]=replace(states[bid],**{field:xp});minus[bid]=replace(states[bid],**{field:xm})
        else:
            xp=np.array(nodes.positions);xm=xp.copy();xp.flat[j-12]+=h;xm.flat[j-12]-=h
            np_=replace(nodes,positions=xp);nm_=replace(nodes,positions=xm)
        cols.append(-(evaluate_marine_static(model,plus,np_).force-evaluate_marine_static(model,minus,nm_).force)/(2*h))
    np.testing.assert_allclose(out.stiffness.toarray(),np.array(cols).T,atol=4e-6,rtol=2e-6)


def test_static_potential_baseline_restoring_work_conjugacy_and_zero_dynamic_terms():
    model,states,_=fixture(finite=True);c=model.contributions[0];layout=BodyLayout(model.body_ids)
    trial=c.evaluate_static(states,layout)
    expected=c.validate_states(states).pullback_force(c.data.baseline-c.data.restoring@c.validate_states(states).coordinates,c.data.restoring)
    np.testing.assert_allclose(trial.external_force,expected.external_force)
    np.testing.assert_allclose(sum(trial.diagnostics['components'].values()),trial.external_force)
    assert trial.trial_history is None and not np.any(trial.mass) and not np.any(trial.damping)
    # Changing dynamic arrays cannot change static force or tangent.
    changed=replace(c,data=replace(c.data,excitation_real=np.zeros(12),excitation_imag=np.zeros(12),
                                 added_mass_infinite=np.zeros((12,12)),kernel=np.zeros((3,12,12))))
    other=changed.evaluate_static(states,layout)
    np.testing.assert_array_equal(other.external_force,trial.external_force)
    np.testing.assert_array_equal(other.stiffness,trial.stiffness)


def test_distinct_force_and_moment_tolerances_are_applied_in_world():
    model,states,_=fixture(preload=False)
    load=WorldWrenchContribution('moment',model.bodies[0],(0.,0.,0.),(0.,0.,0.),(0.,0.,.2))
    model=replace(model,contributions=model.contributions+(load,))
    result=solve_coupled_marine_equilibrium(model,states,spec=CoupledStaticSpec(force_atol=1e-3,moment_atol=1e-10,relative_tolerance=0.))
    assert result.status=='converged',result.message
    assert result.audit['moment_residual']<1e-10
    assert np.linalg.norm(result.states['a'].rotation-np.eye(3))>1e-4


def test_cancellation_and_later_failure_keep_only_completed_stage_states():
    model,states,_=fixture()
    early=solve_coupled_marine_equilibrium(model,states,cancel=lambda:True)
    assert early.status=='cancelled' and not early.stages and early.states is None
    calls=[0]
    def cancel():calls[0]+=1;return calls[0]>=5
    partial=solve_coupled_marine_equilibrium(model,states,cancel=cancel)
    assert partial.status=='cancelled' and len(partial.stages)==1
    assert partial.states is partial.stages[-1].states
    # Bounds allow first quarter-load displacement, not full equilibrium.
    c=model.contributions[0];restricted=replace(c,data=replace(c.data,motion_bounds=np.tile([.012,.1],(2,1))))
    failed=solve_coupled_marine_equilibrium(replace(model,contributions=(restricted,*model.contributions[1:])),states)
    assert failed.status=='failed' and len(failed.stages)>=1
    assert failed.stages[-1].fraction<1. and failed.audit is None
    np.testing.assert_array_equal(states['a'].position,[0.,0.,-2.])


def test_unknown_history_provider_and_active_member_waves_are_not_ignored():
    class Unknown(DynamicContribution):
        contribution_id='unknown';ownership_key='unknown'
        def evaluate_trial(self,*args):raise AssertionError('not admitted')
    model,states,_=fixture()
    with pytest.raises(MooringEquilibriumError,match='static contract'):
        solve_coupled_marine_equilibrium(replace(model,contributions=model.contributions+(Unknown(),)),states)


def test_active_member_waves_are_not_ignored():
    model,states,_=fixture(members=True)
    from anyloads.marine import AiryComponent
    member=model.contributions[-1]
    member=replace(member,environment=replace(member.environment,components=(AiryComponent(.1,1.),)))
    with pytest.raises(MooringEquilibriumError,match='waves'):
        solve_coupled_marine_equilibrium(replace(model,contributions=(*model.contributions[:-1],member)),states)


@pytest.mark.parametrize('delta',[{'force_scale':True},{'moment_scale':0.},{'force_atol':-1.},
    {'relative_tolerance':1e308,'moment_scale':1e308},{'release_fractions':(.5,.5,1.)},
    {'release_fractions':(.5,.9)},{'max_line_search':True}])
def test_bad_spec_refused(delta):
    with pytest.raises(MooringEquilibriumError):CoupledStaticSpec(**delta)


def test_body_without_restoring_cannot_be_balanced_by_retained_support():
    model,states,_=fixture(preload=False)
    failed=solve_coupled_marine_equilibrium(replace(model,contributions=(),links=()),states)
    assert failed.status=='failed' and not failed.stages
    assert 'cannot balance residual' in failed.message


def test_malformed_static_state_model_and_nonstationary_rates_are_refused():
    model,states,_=fixture()
    for bad in (None,object()):
        with pytest.raises(MooringEquilibriumError):solve_coupled_marine_equilibrium(bad,states)
    with pytest.raises(MooringEquilibriumError):solve_coupled_marine_equilibrium(model,{'a':object(),'b':states['b']})
    with pytest.raises(MooringEquilibriumError):
        solve_coupled_marine_equilibrium(model,{**states,'a':replace(states['a'],velocity=[1.,0.,0.,0.,0.,0.])})


@pytest.mark.parametrize('bad',[None,{}, {'a':object(),'b':object()}])
def test_direct_public_static_methods_preserve_typed_admission(bad):
    from anyfesolver.marine_potential_flow import MarinePotentialFlowError
    from anyfesolver.marine_morison import MarineMorisonError
    model,states,_=fixture(members=True);layout=BodyLayout(model.body_ids)
    for c,error in ((model.contributions[0],MarinePotentialFlowError),(model.contributions[-1],MarineMorisonError)):
        with pytest.raises(error):c.evaluate_static(bad,layout)
        with pytest.raises(error):c.evaluate_static(states,None)
