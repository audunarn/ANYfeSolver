"""Static development oracles; no transient integration or W6 qualification."""
from dataclasses import replace
from importlib.util import find_spec
import math
import numpy as np
import pytest
from scipy.integrate import quad
from anyfesolver.marine_equilibrium import (elastic_catenary, initialize_mooring_catenaries,
    solve_mooring_equilibrium, MooringStaticSpec, MooringEquilibriumError)
from anyfesolver.marine_mooring import (MooringLine, MooringLineAssembler, MooringNodeState,
    FixedEndpoint as Anchor, DynamicLineNodeEndpoint as Node,
    RigidBodyAttachmentEndpoint as Fairlead)
from anyfesolver.marine_seabed import SeabedContact
from anyfesolver.marine_rigid import RigidBody6D


def model_case(count=8, *, contact=False, current=False, body=False):
    ends = (Anchor((0.,0.,-2.)),*(Node(i) for i in range(count-1)),Anchor((8.,0.,-2.)))
    bodies=();states={}
    if body:
        b=RigidBody6D('body',10.,(0.,0.,0.),np.diag([2.,3.,4.]))
        s=b.state(position=(7.5,-.2,-1.7),rotation_vector=(.2,-.1,.3))
        point=(.5,.2,-.3)
        ends=(*ends[:-1],Fairlead('body',point));bodies=(b,);states={'body':s}
    kw={}
    if contact:kw.update(contact_diameters=(.1,)*count)
    environment=None
    if current:
        if find_spec('anyloads') is None:
            pytest.importorskip('anyloads', reason='optional marine-morison provider is not installed')
        from anyloads.marine import AiryEnvironment,CurrentProfile
        from anyloads.marine_line import LineHydrodynamics
        environment=AiryEnvironment(10.,current=CurrentProfile.uniform(10.,(0.,.1,0.)),density=1000.)
        section=LineHydrodynamics(.1,1.,.1,0.,0.,1.,1.)
        kw.update(hydrodynamics=(section,)*count)
    line=MooringLine('line',ends,np.full(count,10./count),np.full(count,1e4),
                     np.zeros(count),np.full(count,2.),np.full(count,.001),**kw)
    model=MooringLineAssembler(bodies,(line,),count-1,(0.,0.,-9.81),1000.,0.,
                              environment,SeabedContact(-3.,1e5,0.) if contact else None)
    return model,states


@pytest.mark.parametrize('weight',[9.81,-9.81])
def test_continuous_catenary_matches_independent_force_integrals(weight):
    first=np.array([1.,2.,-8.]);last=np.array([9.,5.,-7.])
    length=11.;ea=12345.;s=np.linspace(0.,length,17)
    out=elastic_catenary(first,last,length,ea,weight,s)
    direction=(last-first)*[1.,1.,0.];direction/=np.linalg.norm(direction)
    h=float(out.first_force@direction);v=float(out.first_force[2])
    for j,a in enumerate(s):
        horizontal=quad(lambda t:h/math.hypot(h,v+weight*t)+h/ea,0.,a,epsabs=1e-11)[0]
        z=quad(lambda t:(v+weight*t)/math.hypot(h,v+weight*t)+(v+weight*t)/ea,0.,a,epsabs=1e-11)[0]
        np.testing.assert_allclose(out.positions[j],first+horizontal*direction+[0.,0.,z],atol=2e-10,rtol=0.)
    np.testing.assert_allclose(out.positions[-1],last,atol=2e-9,rtol=0.)
    np.testing.assert_allclose(out.first_force+out.last_force,[0.,0.,-weight*length],atol=1e-12)
    assert np.all(out.tensions>0.) and not out.positions.flags.writeable


def test_weightless_taut_and_slack_refusal():
    out=elastic_catenary((0.,0.,0.),(4.,0.,3.),4.,1000.,0.,[0.,1.,4.])
    np.testing.assert_allclose(out.positions[1],[1.,0.,.75])
    np.testing.assert_allclose(out.first_force,[200.,0.,150.])
    with pytest.raises(MooringEquilibriumError,match='slack'):
        elastic_catenary((0.,0.,0.),(4.,0.,3.),6.,1000.,0.,[0.,6.])


@pytest.mark.parametrize('weight',[1e-20,-1e-20,1e-10,-1e-10])
def test_tiny_nonzero_weight_preserves_correct_sloping_taut_limit(weight):
    out=elastic_catenary((0.,0.,-2.),(2.,0.,-1.),1.,100.,weight,[0.,.5,1.])
    force=100.*(math.sqrt(5.)-1.)*np.array([2.,0.,1.])/math.sqrt(5.)
    np.testing.assert_allclose(out.first_force,force,atol=2e-8,rtol=0.)
    np.testing.assert_allclose(out.positions,[[0.,0.,-2.],[1.,0.,-1.5],[2.,0.,-1.]],atol=2e-10,rtol=0.)


@pytest.mark.parametrize('body',[False,True])
def test_exact_discrete_initializer_and_staged_full_load_audit(body):
    model,states=model_case(body=body)
    nodes=initialize_mooring_catenaries(model,states)
    out=model.evaluate(states,nodes);offset=6*len(model.bodies)
    assert max(abs(out.force[offset:]))<2e-8
    solved=solve_mooring_equilibrium(model,states,nodes,MooringStaticSpec(force_scale=100.))
    assert solved.status=='converged',solved.message
    assert [s.fraction for s in solved.stages]==[.25,.5,.75,1.]
    assert all(s.name=='weight' for s in solved.stages)
    assert solved.audit['residual_norm']<=solved.audit['force_limit']
    assert solved.audit['max_penetration']==0.
    with pytest.raises(TypeError):solved.audit['residual_norm']=1e20
    with pytest.raises(ValueError):solved.audit['anchor_reactions'].flat[0]=1e20
    np.testing.assert_allclose(solved.nodes.positions,nodes.positions,atol=2e-8,rtol=0.)
    np.testing.assert_array_equal(nodes.velocities,0.)
    if body:assert np.linalg.norm(solved.audit['fixed_body_generalized_forces'])>0.


def test_unequal_material_segments_and_buoyant_exact_chain():
    model,states=model_case(4)
    line=replace(model.lines[0],rest_lengths=[1.,2.,3.,4.],displaced_area=np.full(4,.003))
    model=replace(model,lines=(line,))
    # Mirror the cable far below SWL so the buoyant curve remains admitted.
    line=replace(line,endpoints=(Anchor((0.,0.,-8.)),Node(0),Node(1),Node(2),Anchor((8.,0.,-8.))))
    model=replace(model,lines=(line,))
    nodes=initialize_mooring_catenaries(model,states)
    assert np.max(abs(model.force(states,nodes)))<2e-8
    assert np.all(nodes.positions[:,2]>-8.)


def test_steady_cross_current_ramp_reuses_public_drag():
    model,states=model_case(current=True)
    result=solve_mooring_equilibrium(model,states,spec=MooringStaticSpec(force_scale=100.))
    assert result.status=='converged',result.message
    assert [s.name for s in result.stages]==['weight']*4+['current']*4
    assert np.all(result.nodes.positions[:,1]>0.)
    assert np.max(abs(model.force(states,result.nodes)))<=result.audit['force_limit']
    assert model.environment.current.velocities[0]==(0.,.1,0.)


def test_contact_activation_converges_and_reports_actual_compliance():
    model,states=model_case(contact=True)
    seed=initialize_mooring_catenaries(model,states)
    result=solve_mooring_equilibrium(model,states,seed,MooringStaticSpec(force_scale=100.))
    assert result.status=='converged',result.message
    assert result.stages[-1].name=='contact'
    assert result.audit['max_penetration']>0.  # finite compliant law, not exact nonpenetration
    assert result.audit['max_penetration']<.02
    assert np.max(abs(model.force(states,result.nodes)))<=result.audit['force_limit']
    assert np.min(result.nodes.positions[:,2])>np.min(seed.positions[:,2])
    assert result.stages[-1].least_norm_steps>0  # compatible slack mechanism, no invented compression
    np.testing.assert_array_equal(seed.positions,initialize_mooring_catenaries(model,states).positions)


def test_stiff_touchdown_globalization_closes_without_more_iterations_or_penalty():
    # A residual-only merit stalls at 4.3985 N on this abrupt activation, though
    # the Newton direction can reduce conservative potential while increasing
    # an individual nodal residual. Final physical law and tolerance stay fixed.
    count=32
    ends=(Anchor((0.,0.,-2.95)),*(Node(i) for i in range(count-1)),Anchor((9.,0.,-1.5)))
    line=MooringLine('stiff-touchdown',ends,np.full(count,10/count),np.full(count,10000.),
                    np.zeros(count),np.full(count,2.),np.full(count,.001),contact_diameters=np.full(count,.1))
    model=MooringLineAssembler((),(line,),count-1,(0.,0.,-9.81),1000.,0.,seabed_contact=SeabedContact(-3.,1e7,0.))
    seed=initialize_mooring_catenaries(model,{})
    original=np.array(seed.positions)
    controls=MooringStaticSpec(force_scale=98.1,force_atol=1e-8,force_rtol=1e-10,max_iterations=40,max_line_search=20)
    result=solve_mooring_equilibrium(model,{},seed,controls)
    assert result.status=='converged',result.message
    assert result.stages[-1].iterations<=40
    assert result.audit['residual_norm']<=controls.force_limit
    assert result.audit['max_penetration']<=1e-4
    assert np.min(model.evaluate({},result.nodes).tensions)>0.
    np.testing.assert_array_equal(seed.positions,original)
    out=model.evaluate({},result.nodes)
    loads=out.force_components['weight'].reshape(-1,3)
    old_world_potential=out.elastic_energy+out.contact_diagnostics['potential_energy']-float(np.sum(loads*result.nodes.positions))
    # Also exercise an origin chosen to cancel the old absolute-world potential.
    cancellation_shift=old_world_potential/float(np.sum(loads[:,2]))
    for shift in (-7.,cancellation_shift):
        vector=np.array([0.,0.,shift])
        shifted_ends=tuple(replace(ep,position=np.array(ep.position)+vector) if type(ep) is Anchor else ep for ep in ends)
        translated=replace(model,lines=(replace(line,endpoints=shifted_ends),),still_water_level=shift,
                           seabed_contact=replace(model.seabed_contact,elevation=-3.+shift))
        shifted=solve_mooring_equilibrium(translated,{},spec=controls)
        assert shifted.status=='converged',shifted.message
        assert shifted.audit['residual_norm']<=controls.force_limit
        np.testing.assert_allclose(shifted.nodes.positions-vector,result.nodes.positions,atol=1e-8,rtol=0.)
        np.testing.assert_allclose(translated.evaluate({},shifted.nodes).anchor_reactions,out.anchor_reactions,atol=1e-7,rtol=0.)


def test_failure_and_cancellation_return_only_accepted_stages_without_mutation():
    model,states=model_case(contact=True);seed=initialize_mooring_catenaries(model,states)
    saved=np.array(seed.positions)
    failed=solve_mooring_equilibrium(model,states,seed,MooringStaticSpec(max_iterations=1))
    assert failed.status=='failed' and not failed.stages and failed.nodes is None
    calls=[0]
    def cancel():calls[0]+=1;return calls[0]>=3
    cancelled=solve_mooring_equilibrium(model,states,seed,MooringStaticSpec(load_fractions=(1.,)),cancel=cancel)
    assert cancelled.status=='cancelled'
    assert len(cancelled.stages)==1 and cancelled.stages[0].name=='weight'
    np.testing.assert_array_equal(seed.positions,saved)
    early=solve_mooring_equilibrium(model,states,seed,cancel=lambda:True)
    assert early.status=='cancelled' and early.nodes is None


def test_unrepresentable_contact_merit_retains_residual_route(monkeypatch):
    import anyfesolver.marine_equilibrium as equilibrium
    model,states=model_case(contact=True)
    def overflow(*args):
        raise OverflowError('injected unrepresentable potential summation')
    monkeypatch.setattr(equilibrium.math,'fsum',overflow)
    controls=MooringStaticSpec(force_scale=100.)
    result=solve_mooring_equilibrium(model,states,spec=controls)
    assert result.status=='converged',result.message
    assert result.audit['residual_norm']<=controls.force_limit


def test_failed_current_stage_retains_the_completed_weight_stage():
    model,states=model_case(current=True);seed=initialize_mooring_catenaries(model,states)
    failed=solve_mooring_equilibrium(model,states,seed,MooringStaticSpec(load_fractions=(1.,),max_iterations=1))
    assert failed.status=='failed',failed.message
    assert len(failed.stages)==1 and failed.stages[0].name=='weight'
    np.testing.assert_array_equal(failed.nodes.positions,seed.positions)


def test_normalization_overflow_cannot_be_accepted_as_evidence():
    model,states=model_case();seed=initialize_mooring_catenaries(model,states)
    failed=solve_mooring_equilibrium(model,states,seed,MooringStaticSpec(force_scale=1e-308,force_atol=100.,force_rtol=0.))
    assert failed.status=='failed' and not failed.stages and failed.audit is None
    assert 'normalized residual' in failed.message


def test_invalid_trials_cannot_be_accepted_and_active_waves_are_refused():
    model,states=model_case(current=True)
    from anyloads.marine import AiryComponent
    env=replace(model.environment,components=(AiryComponent(.1,1.),))
    with pytest.raises(MooringEquilibriumError,match='waves'):
        solve_mooring_equilibrium(replace(model,environment=env),states)
    nodes=initialize_mooring_catenaries(model,states)
    failed=solve_mooring_equilibrium(model,states,replace(nodes,positions=nodes.positions+[0.,0.,20.]))
    assert failed.status=='failed' and not failed.stages and 'wetting' in failed.message


@pytest.mark.parametrize('delta',[{'force_scale':True},{'force_atol':-1.},{'force_rtol':float('nan')},
    {'load_fractions':(.5,.5,1.)},{'load_fractions':(.5,.9)},{'max_iterations':True},{'max_line_search':0},
    {'force_scale':1e308,'force_rtol':1e308}])
def test_bad_spec_refused(delta):
    with pytest.raises(MooringEquilibriumError):MooringStaticSpec(**delta)


def test_heterogeneous_and_vertical_initializers_refuse_without_silent_fallback():
    model,states=model_case(2)
    with pytest.raises(MooringEquilibriumError,match='heterogeneous'):
        initialize_mooring_catenaries(replace(model,lines=(replace(model.lines[0],axial_rigidity=[1e4,2e4]),)),states)
    with pytest.raises(MooringEquilibriumError,match='vertical'):
        elastic_catenary((0.,0.,-8.),(0.,0.,-2.),8.,1e4,10.,[0.,8.])
    body_model,states=model_case(body=True)
    with pytest.raises(MooringEquilibriumError,match='RigidBodyState'):
        solve_mooring_equilibrium(body_model,{'body':object()})


def test_rigid_translation_preserves_static_polygon_and_forces():
    model,states=model_case();seed=initialize_mooring_catenaries(model,states)
    shift=np.array([123.,-456.,-3.])
    ends=tuple(replace(ep,position=np.array(ep.position)+shift) if type(ep) is Anchor else ep for ep in model.lines[0].endpoints)
    moved=replace(model,lines=(replace(model.lines[0],endpoints=ends),))
    moved_nodes=initialize_mooring_catenaries(moved,states)
    np.testing.assert_allclose(moved_nodes.positions,seed.positions+shift,atol=1e-12)
    np.testing.assert_allclose(moved.force(states,moved_nodes),model.force(states,seed),atol=3e-10)


def test_slack_unbalanced_mechanism_fails_instead_of_fabricating_stiffness():
    model,states=model_case(2)
    nodes=MooringNodeState([[4.,0.,-2.]],[[0.,0.,0.]],[[0.,0.,0.]])
    failed=solve_mooring_equilibrium(model,states,nodes)
    assert failed.status=='failed' and not failed.stages
    assert 'cannot balance residual' in failed.message
