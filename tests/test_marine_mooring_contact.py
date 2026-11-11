"""Coupled endpoint contact development checks; not registered W6 acceptance."""
from dataclasses import replace
from importlib.util import find_spec
import copy,json
import numpy as np
import pytest
from scipy import sparse
from anyfesolver.marine_seabed import SeabedContact
from anyfesolver.marine_mooring import (MooringLineError,MooringNodeState,
    CONTACT_API_VERSION,data_to_dict,data_from_dict)
from anyfesolver.marine_rigid import RigidBody6D
from anyfesolver.marine_mechanical import (MarineMechanicalModel,MarineMechanicalRunSpec,
    MarineMechanicalSolveError,solve_marine_mechanical,model_to_dict,model_from_dict,result_to_dict)
from test_marine_mooring import coupled_case,fixed_case,perturb


def coupled_contact():
    model,states,nodes=coupled_case()
    lines=tuple(replace(line,contact_diameters=[.2,.4]) for line in model.lines)
    return replace(model,lines=lines,seabed_contact=SeabedContact(-2.9,2000.,30.)),states,nodes


@pytest.mark.parametrize('field,matrix',[('q','stiffness'),('v','damping'),('a','mass')])
def test_full_body_node_contact_pullback_tangents_and_work(field,matrix):
    model,states,nodes=coupled_contact();out=model.evaluate(states,nodes);h=7e-6;cols=[]
    for j in range(model.size):
        plus=perturb(states,nodes,field,j,h);minus=perturb(states,nodes,field,j,-h)
        cols.append(-(model.force(*plus)-model.force(*minus))/(2*h))
    np.testing.assert_allclose(getattr(out,matrix).toarray(),np.asarray(cols).T,atol=8e-7,rtol=3e-7)
    assert sparse.isspmatrix_csr(getattr(out,matrix))
    np.testing.assert_allclose(sum(out.force_components.values()),out.force,atol=2e-13)
    p,v,a,J=model._kinematics(states,nodes)
    velocity=np.r_[states['body'].velocity,nodes.velocities.ravel()]
    endpoint=model._contact_values(p,v)[0]
    contact=sum(out.force_components[k] for k in ('contact_elastic','contact_damping'))
    np.testing.assert_allclose(contact@velocity,np.sum(sum(endpoint.values())*v),atol=2e-13)
    # Contact does not change the physical inertial operator.
    bare=replace(model,lines=tuple(replace(line,contact_diameters=None) for line in model.lines),seabed_contact=None)
    np.testing.assert_array_equal(out.mass.toarray(),bare.evaluate(states,nodes).mass.toarray())


def test_unequal_diameter_shared_node_reference_lumps_action_balance_and_passivity():
    model,nodes=fixed_case();line=replace(model.lines[0],contact_diameters=[.1,.2])
    bed=SeabedContact(-2.03,1000.,30.);model=replace(model,lines=(line,),seabed_contact=bed,gravity_world=(0.,0.,0.))
    nodes=replace(nodes,velocities=[[0.,0.,-.1]]);out=model.evaluate({},nodes)
    # Both independent half-reference lumps act at the shared node; current
    # length (1 m) must not replace the segment's .8 m reference length.
    areas=np.array([.5*.8*.1,.5*.8*.2]);deltas=np.array([.05-.03,.1-.03])
    elastic=float(np.sum(areas*1000.*deltas));damping=float(np.sum(areas*30.*.1))
    assert out.force_components['contact_elastic'][2]==pytest.approx(elastic,abs=2e-13)
    assert out.force_components['contact_damping'][2]==pytest.approx(damping,abs=2e-13)
    assert out.contact_diagnostics['max_penetration']==pytest.approx(.07,abs=2e-15)
    assert out.contact_diagnostics['potential_energy']==pytest.approx(float(np.sum(areas*1000.*deltas**2)),abs=1e-13)
    assert out.contact_diagnostics['dissipation']==pytest.approx(damping*.1,abs=1e-13)
    np.testing.assert_allclose(out.endpoint_forces.sum(axis=0)+out.contact_diagnostics['seabed_reaction'],0.,atol=2e-13)
    np.testing.assert_array_equal(out.anchor_reactions,-out.endpoint_forces[[0,2]])
    original=model.force({},nodes);out.force[:]=0.;out.contact_diagnostics['gaps'][0]=999.
    np.testing.assert_array_equal(model.force({},nodes),original)
    with pytest.raises(ValueError):model.lines[0].contact_diameters.setflags(write=True)


def test_static_compliance_against_closed_form_lumped_weight():
    model,nodes=fixed_case(rest=2.,area=0.)
    bed=SeabedContact(-2.1,1e5,0.);line=replace(model.lines[0],contact_diameters=[.2,.2])
    model=replace(model,lines=(line,),seabed_contact=bed)
    mass=2.*2.;stiffness=2.*(.5*2.*.2)*1e5;weight=mass*10.
    nodes=replace(nodes,positions=[[0.,0.,bed.elevation+.1-weight/stiffness]])
    out=model.evaluate({},nodes)
    np.testing.assert_allclose(out.force,0.,atol=1e-10)
    assert out.contact_diagnostics['max_penetration']==pytest.approx(weight/stiffness,abs=1e-15)
    assert out.contact_diagnostics['potential_energy']==pytest.approx(.5*weight**2/stiffness,abs=1e-13)


@pytest.mark.parametrize('failure',['version','missing-bed','diameters','bool','unknown','dry-bed','unsupported'])
def test_contact_model_serde_and_refusals(failure):
    model,states,nodes=coupled_contact();data=data_to_dict(model)
    assert data['api_version']==CONTACT_API_VERSION
    restored=data_from_dict(copy.deepcopy(data),model.bodies)
    assert data_to_dict(restored)==data
    np.testing.assert_array_equal(restored.force(states,nodes),model.force(states,nodes))
    native=MarineMechanicalModel(model.bodies,mooring=model)
    assert model_to_dict(model_from_dict(json.loads(json.dumps(model_to_dict(native)))))==model_to_dict(native)
    bad=copy.deepcopy(data)
    if failure=='version':bad['api_version']='marine_mooring_v1'
    elif failure=='missing-bed':del bad['seabed_contact']
    elif failure=='diameters':bad['lines'][0]['contact_diameters']=[.1]
    elif failure=='bool':bad['lines'][0]['contact_diameters'][0]=True
    elif failure=='unknown':bad['seabed_contact']['friction']=.5
    elif failure=='dry-bed':bad['seabed_contact']['elevation']=0.
    else:del bad['lines'][0]['contact_diameters']
    with pytest.raises(MooringLineError):data_from_dict(bad,model.bodies)


def test_combined_hydro_contact_preserves_floor_sampling_refusal_and_identity():
    from test_marine_line_hydrodynamics import hydro_case
    model,states,nodes=hydro_case();D=model.lines[0].hydrodynamics[0].diameter
    model=replace(model,lines=(replace(model.lines[0],contact_diameters=[D,D]),),
                  seabed_contact=SeabedContact(-20.,1e5,10.))
    assert data_to_dict(data_from_dict(data_to_dict(model),model.bodies))==data_to_dict(model)
    out=model.evaluate(states,replace(nodes,positions=[[-.3,.1,-19.975]]),time=.3)
    assert out.contact_diagnostics['max_penetration']>0. and out.relative_drag_power>=0.
    with pytest.raises(MooringLineError,match='diameters'):replace(model.lines[0],contact_diameters=[D,2*D])
    with pytest.raises(MooringLineError,match='water depth'):replace(model,seabed_contact=SeabedContact(-19.,1e5,10.))
    with pytest.raises(MooringLineError,match='hydrodynamic domain'):
        model.evaluate(states,replace(nodes,positions=[[-.3,.1,-20.001]]),time=.3)


def contact_oscillator(damping):
    assembler,nodes=fixed_case(rest=2.,area=0.)
    body=RigidBody6D('unused-body',5.,(0.,0.,0.),np.diag([2.,3.,4.]))
    line=replace(assembler.lines[0],contact_diameters=[.2,.2])
    assembler=replace(assembler,bodies=(body,),lines=(line,),gravity_world=(0.,0.,0.),
                      seabed_contact=SeabedContact(-2.1,1e5,damping))
    model=MarineMechanicalModel((body,),mooring=assembler)
    nodes=replace(nodes,positions=[[0.,0.,-2.001]],accelerations=[[123.,123.,123.]])
    return model,{'unused-body':body.state(position=(0.,0.,-1.))},nodes


@pytest.mark.parametrize('damping',[0.,100.])
def test_actual_common_driver_matches_independent_normal_newmark_energy(damping):
    model,states,nodes=contact_oscillator(damping);dt=.001
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(dt,3),initial_nodes=nodes)
    mass=4.;K=40000.;C=.4*damping;q=-.001;v=0.;a=-K*q/mass;previous=None
    for frame in result.accepted:
        np.testing.assert_allclose(frame.nodes.positions[0],[0.,0.,-2.+q],atol=2e-14,rtol=0.)
        assert frame.nodes.velocities[0,2]==pytest.approx(v,abs=3e-13)
        assert frame.nodes.accelerations[0,2]==pytest.approx(a,abs=3e-11)
        diagnostics=frame.mooring_diagnostics['contact']
        assert diagnostics['max_penetration']==pytest.approx(-q,abs=2e-14)
        assert diagnostics['potential_energy']==pytest.approx(.5*K*q*q,abs=3e-13)
        assert diagnostics['dissipation']==pytest.approx(C*v*v,abs=3e-13)
        assert 'mooring:contact_elastic' in frame.force_decomposition
        np.testing.assert_allclose(sum(frame.mooring_diagnostics['node_force_components'].values()),0.,atol=1e-10)
        energy=frame.kinetic_energy+diagnostics['potential_energy']
        if previous is not None:
            Eold,vold=previous;assert energy-Eold+C*((vold+v)/2.)**2*dt==pytest.approx(0.,abs=2e-13)
        previous=(energy,v)
        qp=q+dt*v+.25*dt**2*a;vp=v+.5*dt*a
        a=(-K*qp-C*vp)/(mass+.25*dt**2*K+.5*dt*C);q=qp+.25*dt**2*a;v=vp+.5*dt*a
    assert result.status=='completed' and len(result.accepted)==4
    assert np.all(nodes.accelerations==123.)
    encoded=result_to_dict(result);assert encoded['accepted'][0]['mooring_diagnostics']['contact']['potential_energy']>0.
    json.dumps(encoded,allow_nan=False)


def test_contact_cancellation_keeps_only_atomic_accepted_arrays():
    model,states,nodes=contact_oscillator(100.);stop=False
    def progress(frame):
        nonlocal stop
        if frame.step==1:stop=True
    result=solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.001,3),initial_nodes=nodes,
                                  cancel=lambda:stop,progress=progress)
    assert result.status=='cancelled' and [f.step for f in result.accepted]==[0,1]
    assert result.accepted[-1].mooring_diagnostics['contact']['potential_energy']>0.
    with pytest.raises(TypeError):result.accepted[-1].mooring_diagnostics['contact']['potential_energy']=999.


def test_failed_combined_contact_trial_preserves_last_body_node_history_and_evidence():
    if find_spec('anyloads') is None:
        pytest.importorskip('anyloads', reason='optional marine-morison provider is not installed')
    from anyloads.marine import AiryEnvironment
    from anyloads.marine_line import LineHydrodynamics
    from anyfesolver.marine_contributions import DynamicContribution,ContributionResult,HISTORY_CHART
    model,states,nodes=contact_oscillator(0.)
    class Memory(DynamicContribution):
        contribution_id='memory';ownership_key='contact-test-memory';history_basis=HISTORY_CHART
        def evaluate_trial(self,history,states,layout):
            count=history.get('memory',{'steps':7})['steps'];z=np.zeros((layout.size,layout.size))
            return ContributionResult('memory',np.zeros(layout.size),z,z,z,{},dict(steps=count+1))
        def history_transport(self,history,info):return history
    section=LineHydrodynamics(.2,0.,0.,0.,0.,0.,0.)
    line=replace(model.mooring.lines[0],hydrodynamics=(section,section))
    assembler=replace(model.mooring,lines=(line,),environment=AiryEnvironment(2.1,density=1000.),
                      seabed_contact=SeabedContact(-2.1,1.,0.))
    model=replace(model,mooring=assembler,contributions=(Memory(),))
    nodes=replace(nodes,positions=[[0.,0.,-2.08]],velocities=[[0.,0.,-1.]])
    accepted=[]
    with pytest.raises(MarineMechanicalSolveError) as raised:
        solve_marine_mechanical(model,states,MarineMechanicalRunSpec(.01,3),initial_nodes=nodes,
                               progress=accepted.append)
    error=raised.value;partial=error.partial
    assert error.reason=='mooring_domain' and partial.status=='failed'
    assert [f.step for f in partial.accepted]==[0,1,2]
    assert partial.accepted[-1] is accepted[-1]
    assert partial.histories['memory']['steps']==9 and error.histories==partial.histories
    assert partial.accepted[-1].nodes.positions[0,2]>=-2.1
    assert partial.accepted[-1].mooring_diagnostics['contact']['max_penetration']>0.
    encoded=result_to_dict(partial)
    assert encoded['accepted'][-1]['step']==2
    assert encoded['accepted'][-1]['mooring_diagnostics']['contact']['potential_energy']>0.
    np.testing.assert_array_equal(nodes.positions,[[0.,0.,-2.08]])
    np.testing.assert_array_equal(states['unused-body'].position,[0.,0.,-1.])
