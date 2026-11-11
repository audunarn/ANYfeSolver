"""Bounded native DEVELOPMENT tests, not W3/W4 scientific qualification."""
from dataclasses import replace,FrozenInstanceError
import copy,math
import numpy as np
import pytest
pytest.importorskip('anyloads.marine',reason='optional native Morison ANYloads API is not installed')
from anyloads.marine import (AiryComponent,AiryEnvironment,CurrentProfile,
                             MorisonSection,HydrodynamicOwnership)
from anyfesolver.marine_rigid import RigidBody6D,RigidBodyState,rotation_matrix
from anyfesolver.marine_contributions import BodyLayout,CommittedHistory,StepInformation,ContributionSet
from anyfesolver.marine_morison import (MemberEndpoint,MorisonMember,MorisonMemberSetContribution,
    MarineMorisonError,data_to_dict,data_from_dict)
from anyfesolver.marine_mechanical import (MarineMechanicalModel,MarineMechanicalRunSpec,
    MarineMechanicalInputError,model_to_dict,model_from_dict,solve_marine_mechanical)

def body(name='a'):
    return RigidBody6D(name,9.,(0.,0.,0.),tuple(map(tuple,10.*np.eye(3))))

def simple(*,ownership=None,current=(0.,3.,0.),buoyancy=False):
    b=body()
    m=MorisonMember('rod',MemberEndpoint('a',(-1.,0.,0.)),MemberEndpoint('a',(1.,0.,0.)),
        MorisonSection(.8,1.2,.7),ownership or HydrodynamicOwnership(buoyancy_owner='morison' if buoyancy else 'none'))
    p=MorisonMemberSetContribution('members',(b,),(m,),AiryEnvironment(8.,
        current=CurrentProfile.uniform(8.,current),density=7.3))
    return p,{'a':RigidBodyState((0.,0.,-2.),velocity=(0.,1.,0.,0.,0.,0.),
                               acceleration=(0.,.2,0.,0.,0.,0.))}

def evaluate(p,states):
    return p.evaluate_trial_with_context(CommittedHistory(),states,BodyLayout(tuple(states)),
        StepInformation(3,.04,'step',states))

def test_analytic_uniform_load_mass_components_virtual_work_and_dissipation():
    p,states=simple(buoyancy=True);r=evaluate(p,states)
    rho=7.3;section=p.members[0].section;length=2.
    added=rho*.7*section.area*length
    drag=.5*rho*1.2*.8*length*4.
    expected=np.array([0.,drag-added*.2,rho*9.81*section.area*length,0.,0.,0.])
    np.testing.assert_allclose(r.external_force,expected,rtol=1e-10,atol=1e-10)
    mass=np.diag([0.,added,added,0.,added/3.,added/3.])
    np.testing.assert_allclose(r.mass,mass,rtol=1e-10,atol=1e-10)
    d=r.diagnostics['members']['rod']
    np.testing.assert_allclose(sum(d['components'].values()),r.external_force,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(sum(d['drag_attribution'].values()),d['components']['drag'],rtol=1e-10,atol=1e-10)
    assert d['wet_length']==pytest.approx(length,abs=1e-12)
    assert d['relative_drag_power']==pytest.approx(-2*drag,abs=1e-10)
    # Pure body translation: covector/physical point power agree exactly.
    assert np.dot(states['a'].velocity,r.external_force)==pytest.approx(expected[1],abs=1e-10)

def complex_fixture():
    bodies=(body('a'),body('b'))
    m=MorisonMember('cross-body',MemberEndpoint('a',(-.7,.2,-1.7)),
        MemberEndpoint('b',(.6,-.3,.5)),MorisonSection(.8,1.2,.7),HydrodynamicOwnership(),6)
    env=AiryEnvironment(8.,(AiryComponent(.1,1.7,26.,.3),AiryComponent(.06,2.1,80.,-.7)),
        CurrentProfile((0.,.8,3.,8.),((.1,.4,0.),(.3,.6,.1),(-.1,.2,0.),(.2,-.2,0.))),density=7.3)
    p=MorisonMemberSetContribution('members',bodies,(m,),env)
    states={'a':RigidBodyState((-1.,0.,.2),(.17,-.24,.1),(.21,-.13,.17,.4,-.22,.31),
        (-.12,.13,.19,-.28,.33,-.1),rotation_matrix((.08,.15,-.06))),
        'b':RigidBodyState((2.,1.,0.),(-.11,.16,-.13),(-.18,.1,-.12,-.2,.34,-.17),
        (.23,-.11,.16,.19,-.12,.27),rotation_matrix((-.04,.07,.11)))}
    return p,states


@pytest.mark.parametrize('order',(2,16))
def test_distributed_law_receives_one_array_batch_per_member(monkeypatch,order):
    """Both admitted order boundaries retain one law call, not one per node."""
    from anyloads import marine
    p,states=complex_fixture()
    p=replace(p,members=(replace(p.members[0],quadrature_order=order),))
    original=marine.morison_force;calls=[]
    def batched(*args,**kwargs):
        calls.append(np.shape(args[3]))
        return original(*args,**kwargs)
    monkeypatch.setattr(marine,'morison_force',batched)
    result=evaluate(p,states)
    intervals=len(result.diagnostics['members']['cross-body']['wet_parameter_segments'])
    assert calls==[(order*intervals,3)]
    assert intervals>=2
    d=result.diagnostics['members']['cross-body']
    np.testing.assert_allclose(sum(d['components'].values()),result.external_force,rtol=1e-10,atol=1e-10)
    assert d['relative_drag_power']<=0.

def perturb(states,bid,j,field,delta):
    out=dict(states);s=states[bid]
    if field=='q':
        if j<3:
            value=np.array(s.position);value[j]+=delta;out[bid]=replace(s,position=value)
        else:
            value=np.array(s.rotation_vector);value[j-3]+=delta;out[bid]=replace(s,rotation_vector=value)
    else:
        name='velocity' if field=='v' else 'acceleration'
        value=np.array(getattr(s,name));value[j]+=delta;out[bid]=replace(s,**{name:value})
    return out

@pytest.mark.parametrize('field,tangent',(('q','stiffness'),('v','damping'),('a','mass')))
def test_full_total_tangents_cross_body_wet_boundary_current_knots(field,tangent):
    p,states=complex_fixture();r=evaluate(p,states);fd=np.zeros((12,12));h=2e-6
    for i,bid in enumerate(states):
        for j in range(6):
            plus=evaluate(p,perturb(states,bid,j,field,h)).external_force
            minus=evaluate(p,perturb(states,bid,j,field,-h)).external_force
            fd[:,6*i+j]=-(plus-minus)/(2*h)
    np.testing.assert_allclose(getattr(r,tangent),fd,rtol=1e-8,atol=1e-8)
    assert len(r.diagnostics['members']['cross-body']['wet_parameter_segments'])>=2
    np.testing.assert_allclose(r.mass,r.mass.T,rtol=0.,atol=1e-10)

def test_analytic_vertical_regular_wave_force():
    b=body();env=AiryEnvironment(8.,(AiryComponent(.1,1.6,0.,.3),),density=7.3)
    m=MorisonMember('vertical',MemberEndpoint('a',(0.,0.,-1.)),MemberEndpoint('a',(0.,0.,1.)),
        MorisonSection(.8,0.,.7),HydrodynamicOwnership(),16)
    p=MorisonMemberSetContribution('members',(b,),(m,),env)
    r=evaluate(p,{'a':RigidBodyState((0.,0.,-2.))})
    k=env.wave_numbers[0];phase=.3-1.6*.12
    integral=(math.sinh(k*7.)-math.sinh(k*5.))/(k*math.sinh(k*8.))
    expected=7.3*1.7*m.section.area*.1*1.6**2*math.sin(phase)*integral
    np.testing.assert_allclose(r.external_force[0],expected,rtol=1e-10,atol=1e-10)

def test_rechart_preserves_physical_force_and_mass_without_history_reset():
    p,states=complex_fixture();old=evaluate(p,states);T=np.zeros((12,12));moved={}
    for i,(bid,s) in enumerate(states.items()):
        T[6*i:6*i+6,6*i:6*i+6]=s.rate_map
        moved[bid]=RigidBodyState(s.position,velocity=s.world_velocity,
                                  acceleration=s.world_acceleration,anchor=s.rotation)
    new=evaluate(p,moved)
    np.testing.assert_allclose(old.external_force,T.T@new.external_force,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(old.mass,T.T@new.mass@T,rtol=1e-10,atol=1e-10)

def test_drag_only_dry_zero_speed_and_branch_diagnostics():
    own=HydrodynamicOwnership('potential_flow',True,True,'potential_flow')
    p,states=simple(ownership=own);r=evaluate(p,states)
    assert not np.any(r.mass)
    np.testing.assert_array_equal(r.diagnostics['members']['rod']['components']['added_mass'],np.zeros(6))
    dry={'a':replace(states['a'],position=(0.,0.,1.))}
    d=evaluate(p,dry)
    for name in ('external_force','mass','stiffness','damping'):assert not np.any(getattr(d,name))
    still={'a':replace(states['a'],position=(0.,0.,0.),velocity=(0.,3.,0.,0.,0.,0.))}
    zero=evaluate(p,still)
    assert not np.any(zero.external_force) and not np.any(zero.damping)
    assert any(e['kind']=='horizontal_on_plane' for e in zero.diagnostics['members']['rod']['branch_events'])

def test_named_serde_refusal_ownership_clock_history_and_immutability():
    p,states=simple();model=MarineMechanicalModel(p.bodies,contributions=(p,))
    data=model_to_dict(model);assert model_to_dict(model_from_dict(data))==data
    for change in ('unknown','ownership','endpoint','order','boolean-point'):
        bad=copy.deepcopy(data);m=bad['contributions'][0]['data']['members'][0]
        if change=='unknown':m['code']='evil()'
        if change=='ownership':m['ownership']['bem_included']=True
        if change=='endpoint':m['end_a']['body_id']='unknown'
        if change=='order':m['quadrature_order']=True
        if change=='boolean-point':m['end_a']['point_body'][0]=True
        with pytest.raises(ValueError):model_from_dict(bad)
    other=replace(p,contribution_id='duplicate')
    with pytest.raises(ValueError,match='duplicate Morison'):MarineMechanicalModel(p.bodies,contributions=(p,other))
    with pytest.raises(ValueError,match='duplicate Morison'):ContributionSet(('a',),(p,other))
    with pytest.raises(MarineMorisonError,match='StepInformation'):p.evaluate_trial(CommittedHistory(),states,BodyLayout(('a',)))
    with pytest.raises(MarineMorisonError,match='history'):
        p.evaluate_trial_with_context(CommittedHistory({'members':{'unknown':1}}),states,BodyLayout(('a',)),StepInformation(0,.1,'initialization'))
    with pytest.raises(FrozenInstanceError):p.members=()

def test_native_driver_initial_added_mass_and_three_step_history_cancellation():
    p,states=simple(current=(0.,0.,0.))
    section=replace(p.members[0].section,drag_coefficient=0.)
    p=replace(p,members=(replace(p.members[0],section=section),))
    from anyfesolver.marine_contributions import WorldWrenchContribution
    wrench=WorldWrenchContribution('push',p.bodies[0],(0.,0.,0.),(0.,2.,0.))
    model=MarineMechanicalModel(p.bodies,contributions=(p,wrench))
    states={'a':RigidBodyState((0.,0.,-2.))}
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.01,3))
    expected=2./(9.+7.3*.7*section.area*2.)
    assert result.status=='completed' and len(result.accepted)==4
    for record in result.accepted:
        np.testing.assert_allclose(record.world_accelerations['a'][1],expected,rtol=1e-10,atol=1e-10)
    assert result.histories['members'] is None
    cancelled=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.01,3),cancel=lambda:True)
    assert cancelled.status=='cancelled' and not cancelled.accepted and len(cancelled.histories)==0

def test_bed_and_swl_moving_limits_have_closed_buoyancy_tangents():
    b=body();theta=.23;rho=7.3
    m=MorisonMember('through-column',MemberEndpoint('a',(0.,0.,-4.)),MemberEndpoint('a',(0.,0.,4.)),
        MorisonSection(.8,0.,0.),HydrodynamicOwnership(buoyancy_owner='morison'))
    p=MorisonMemberSetContribution('members',(b,),(m,),AiryEnvironment(3.,density=rho))
    r=evaluate(p,{'a':RigidBodyState((0.,0.,-1.),(0.,theta,0.))})
    scale=rho*9.81*m.section.area;sec=1./math.cos(theta);tan=math.tan(theta)
    np.testing.assert_allclose(r.external_force[2],3*scale*sec,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(r.external_force[4],1.5*scale*math.sin(theta)*sec**2,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(r.stiffness[2,4],-3*scale*sec*tan,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(r.stiffness[4,4],-1.5*scale*(sec+2*tan*tan*sec),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(r.stiffness[4,2],-3*scale*math.sin(theta)*sec**2,rtol=1e-10,atol=1e-10)

@pytest.mark.parametrize('field,tangent',(('q','stiffness'),('v','damping'),('a','mass')))
def test_same_body_full_wet_finite_rotation_total_tangents(field,tangent):
    p,states=complex_fixture();m=p.members[0]
    p=replace(p,bodies=(p.bodies[0],),members=(replace(m,end_b=MemberEndpoint('a',(.6,-.3,-.5))),))
    states={'a':replace(states['a'],position=(-1.,0.,-3.))}
    r=evaluate(p,states);fd=np.zeros((6,6));h=2e-6
    for j in range(6):
        fd[:,j]=-(evaluate(p,perturb(states,'a',j,field,h)).external_force
                  -evaluate(p,perturb(states,'a',j,field,-h)).external_force)/(2*h)
    np.testing.assert_allclose(getattr(r,tangent),fd,rtol=1e-8,atol=1e-8)

def test_unknown_subclass_and_collapsed_member_refuse():
    p,states=simple();m=p.members[0]
    with pytest.raises(MarineMorisonError,match='zero current length'):
        replace(m,end_b=m.end_a)
    class Unknown(MorisonMemberSetContribution):pass
    unknown=Unknown(p.contribution_id,p.bodies,p.members,p.environment)
    with pytest.raises(MarineMechanicalInputError,match='trusted named'):
        model_to_dict(MarineMechanicalModel(p.bodies,contributions=(unknown,)))
    with pytest.raises(MarineMorisonError,match='subclass'):data_to_dict(unknown)

def test_native_coupled_fixture_plus_omitted_member_and_drag_correction():
    """Explicit synthetic BEM fixture; does not prove mesh/source exclusion."""
    from test_marine_potential_flow import fixture
    from anyfesolver.marine_hydro_coordinates import EquilibriumHydroFrame
    from anyfesolver.marine_potential_flow import PotentialFlowDevelopmentContribution
    p,states=simple()
    frame=EquilibriumHydroFrame('a',(0.,0.,-2.),np.eye(3),np.eye(3),(0.,0.,0.))
    data=fixture((frame,));data=replace(data,dt=.01)
    fluid=PotentialFlowDevelopmentContribution(data)
    for role in (HydrodynamicOwnership(),HydrodynamicOwnership('potential_flow',True,True,'potential_flow')):
        member=replace(p,members=(replace(p.members[0],ownership=role),))
        model=MarineMechanicalModel(p.bodies,contributions=(fluid,member))
        restored=model_from_dict(model_to_dict(model))
        result=solve_marine_mechanical(restored,states,MarineMechanicalRunSpec(.01,3))
        assert result.status=='completed' and len(result.accepted)==4
        assert result.histories['members'] is None
        assert result.histories[fluid.contribution_id]['step_index']==3


def test_mid_trial_cancel_preserves_accepted_fluid_history_and_member_prefix(monkeypatch):
    """Cancel inside the second-step fluid sampling, before any trial commit."""
    from test_marine_potential_flow import fixture
    from anyfesolver.marine_hydro_coordinates import EquilibriumHydroFrame
    from anyfesolver.marine_potential_flow import PotentialFlowDevelopmentContribution
    from anyfesolver.marine_mechanical import result_to_dict
    p,states=simple()
    frame=EquilibriumHydroFrame('a',(0.,0.,-2.),np.eye(3),np.eye(3),(0.,0.,0.))
    fluid=PotentialFlowDevelopmentContribution(replace(fixture((frame,)),dt=.01))
    model=MarineMechanicalModel(p.bodies,contributions=(fluid,p))
    prefix=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.01,1))
    assert prefix.status=='completed'
    original=AiryEnvironment.evaluate;flag={'cancel':False,'time':None}
    def sampled(environment,points,time):
        value=original(environment,points,time)
        if time>=.02:
            flag.update(cancel=True,time=time)
        return value
    monkeypatch.setattr(AiryEnvironment,'evaluate',sampled)
    cancelled=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.01,3),
                                     cancel=lambda:flag['cancel'])
    assert flag['time']==.02 and cancelled.status=='cancelled'
    before,after=result_to_dict(prefix),result_to_dict(cancelled)
    assert len(after['accepted'])==2 and after['accepted']==before['accepted']
    assert after['histories']==before['histories']
    assert cancelled.histories[fluid.contribution_id]['step_index']==1
    assert cancelled.histories['members'] is None

def test_cross_body_virtual_work_against_physical_endpoint_reactions():
    """Independent wrench assembly: endpoint moments, without point Jacobians."""
    from anyloads.marine import morison_force
    p,states=complex_fixture();m=p.members[0];r=evaluate(p,states)
    endpoints=(m.end_a,m.end_b)
    levers=[states[e.body_id].rotation@np.array(e.point_body) for e in endpoints]
    points=[np.array(states[e.body_id].position)+lever for e,lever in zip(endpoints,levers)]
    velocities=[s.world_velocity[:3]+np.cross(s.world_velocity[3:],lever)
                for s,lever in zip(states.values(),levers)]
    accelerations=[s.world_acceleration[:3]+np.cross(s.world_acceleration[3:],lever)
        +np.cross(s.world_velocity[3:],np.cross(s.world_velocity[3:],lever))
        for s,lever in zip(states.values(),levers)]
    delta=points[1]-points[0];length=np.linalg.norm(delta);axis=delta/length
    planes=p.environment.wet_levels
    bounds=sorted([0.,1.]+[float((level-points[0][2])/delta[2]) for level in planes
                          if 0.<(level-points[0][2])/delta[2]<1.])
    xi,wi=np.polynomial.legendre.leggauss(m.quadrature_order)
    wrench=np.zeros((2,6));virtual_work=0.
    dq=np.array([.11,-.2,.13,.07,-.17,.05,-.21,.16,-.1,.04,.13,-.09])
    virtual=[dq[6*i:6*i+3]+np.cross(s.rate_map[3:,3:]@dq[6*i+3:6*i+6],lever)
             for i,(s,lever) in enumerate(zip(states.values(),levers))]
    for lo,hi in zip(bounds,bounds[1:]):
        z=points[0][2]+.5*(lo+hi)*delta[2]
        if not -p.environment.water_depth<=z<=0.:continue
        for a,w in zip(xi,wi):
            s=.5*((1-a)*lo+(1+a)*hi);weight=.5*w*(hi-lo)*length
            x=(1-s)*points[0]+s*points[1]
            fluid=p.environment.evaluate(x,.12)
            f=morison_force(m.section,m.ownership,p.environment.density,
                fluid.velocity,fluid.acceleration,(1-s)*velocities[0]+s*velocities[1],
                (1-s)*accelerations[0]+s*accelerations[1],axis).force
            virtual_work+=weight*f@((1-s)*virtual[0]+s*virtual[1])
            for i,shape in enumerate((1-s,s)):
                reaction=weight*shape*f
                wrench[i]+=np.r_[reaction,np.cross(levers[i],reaction)]
    chart=np.concatenate([s.rate_map.T@F for s,F in zip(states.values(),wrench)])
    np.testing.assert_allclose(r.external_force,chart,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(dq@r.external_force,virtual_work,rtol=1e-10,atol=1e-10)
