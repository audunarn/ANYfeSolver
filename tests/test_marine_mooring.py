"""Stateless development checks; no line trajectories or W6 acceptance."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from anyfesolver.marine_mooring import (
    DynamicLineNodeEndpoint as Node, FixedEndpoint as Anchor,
    RigidBodyAttachmentEndpoint as Fairlead, MooringLine, MooringLineAssembler,
    MooringLineError, MooringNodeState,
)
from anyfesolver.marine_rigid import RigidBody6D, CHART_BRANCH_MARGIN


def line(endpoints, *, rest=.8, ea=100., damping=4., mass=2., area=.001):
    count=len(endpoints)-1
    return MooringLine('line', endpoints, np.full(count,rest), np.full(count,ea),
                       np.full(count,damping), np.full(count,mass), np.full(count,area))


def fixed_case(*, rest=.8, damping=4., mass=2., area=.001):
    l=line((Anchor((-1.,0.,-2.)),Node(0),Anchor((1.,0.,-2.))),
           rest=rest,damping=damping,mass=mass,area=area)
    model=MooringLineAssembler((),(l,),1,(0.,0.,-10.),1000.,0.)
    nodes=MooringNodeState([[0.,0.,-2.]],[[0.,0.,0.]],[[0.,0.,0.]])
    return model,nodes


def coupled_case():
    body=RigidBody6D('body',10.,(0.,0.,0.),np.diag([2.,3.,4.]))
    l=line((Anchor((-2.,-1.,-5.)),Node(0),Fairlead('body',(.5,.2,-.4))),rest=1.1)
    model=MooringLineAssembler((body,),(l,),1,(0.,0.,-9.81),1025.,0.)
    state=body.state(position=(1.,.6,-3.),rotation_vector=(.3,-.2,.1),
                     velocity=(.2,-.3,.1,.4,-.2,.3), acceleration=(.1,.2,-.3,-.2,.3,.1))
    nodes=MooringNodeState([[-.3,.1,-4.]],[[.1,-.2,.4]],[[.3,-.1,.2]])
    return model,{'body':state},nodes


def perturb(states,nodes,field,j,h):
    states=dict(states)
    if j<6:
        name=('position' if j<3 else 'rotation_vector') if field=='q' else {'v':'velocity','a':'acceleration'}[field]
        k=j%3 if field=='q' else j
        x=np.array(getattr(states['body'],name));x[k]+=h
        states['body']=replace(states['body'],**{name:x})
    else:
        name={'q':'positions','v':'velocities','a':'accelerations'}[field]
        x=np.array(getattr(nodes,name));x.flat[j-6]+=h
        nodes=replace(nodes,**{name:x})
    return states,nodes


def test_closed_form_taut_straight_node_blocks_and_mass():
    model,nodes=fixed_case()
    out=model.evaluate({},nodes)
    np.testing.assert_allclose(out.tensions,[25.,25.])
    np.testing.assert_allclose(out.force,[0.,0.,-8.])
    np.testing.assert_allclose(out.stiffness.toarray(),np.diag([250.,50.,50.]))
    np.testing.assert_allclose(out.damping.toarray(),np.diag([8.,0.,0.]))
    np.testing.assert_allclose(out.mass.toarray(),np.eye(3)*1.6)
    assert out.elastic_energy==pytest.approx(5.)
    assert sparse.isspmatrix_csr(out.stiffness)
    np.testing.assert_allclose(sum(out.force_components.values()),out.force)


@pytest.mark.parametrize('field,matrix',[('q','stiffness'),('v','damping'),('a','mass')])
def test_all_body_and_node_derivatives_at_finite_rotation(field,matrix):
    model,states,nodes=coupled_case()
    out=model.evaluate(states,nodes)
    h=7e-6  # distinct from the product's body differentiation step
    columns=[]
    for j in range(model.size):
        plus=perturb(states,nodes,field,j,h);minus=perturb(states,nodes,field,j,-h)
        columns.append(-(model.force(*plus)-model.force(*minus))/(2*h))
    np.testing.assert_allclose(getattr(out,matrix).toarray(),np.array(columns).T,atol=3e-7,rtol=2e-7)


def test_elastic_potential_gradient_and_virtual_work():
    model,states,nodes=coupled_case()
    states={'body':replace(states['body'],velocity=np.zeros(6),acceleration=np.zeros(6))}
    nodes=replace(nodes,velocities=np.zeros((1,3)),accelerations=np.zeros((1,3)))
    out=model.evaluate(states,nodes)
    h=6e-6;gradient=[]
    # Independent potential uses geometry and the constitutive definition only.
    def potential(ss,nn):
        p=np.array([model.lines[0].endpoints[0].position,nn.positions[0],
                    model.bodies[0].point_kinematics(ss['body'],(.5,.2,-.4)).position])
        strain=np.maximum(np.linalg.norm(np.diff(p,axis=0),axis=1)-1.1,0.)
        return np.sum(.5*100./1.1*strain**2)
    for j in range(model.size):
        gradient.append((potential(*perturb(states,nodes,'q',j,h))-
                         potential(*perturb(states,nodes,'q',j,-h)))/(2*h))
    np.testing.assert_allclose(out.force_components['elastic_force'],-np.array(gradient),atol=2e-8,rtol=2e-8)
    dq=np.array([.1,-.2,.3,-.4,.2,.1,.3,.1,-.2])
    kin=model.bodies[0].point_kinematics(states['body'],(.5,.2,-.4))
    endpoint_work=out.endpoint_forces[1]@dq[6:]+out.endpoint_forces[2]@(kin.jacobian@dq[:6])
    assert out.force@dq==pytest.approx(endpoint_work,abs=1e-12)
    np.testing.assert_allclose(out.stiffness.toarray(),out.stiffness.toarray().T,atol=2e-7)


@pytest.mark.parametrize('speed',[.2,-.2,-30.])
def test_damping_dissipation_and_no_compression(speed):
    model,nodes=fixed_case()
    nodes=replace(nodes,velocities=[[speed,0.,0.]])
    out=model.evaluate({},nodes)
    assert np.all(out.tensions>=0.)
    assert out.damping_dissipation>=0.
    assert out.force_components['damping_force']@nodes.velocities.ravel()==pytest.approx(-out.damping_dissipation)
    np.testing.assert_allclose(sum(out.force_components.values()),out.force)
    assert np.any(out.force_components['damping_force']!=0.)


def test_slack_branch_has_no_structural_force_or_stiffness():
    model,nodes=fixed_case(rest=1.2)
    out=model.evaluate({},replace(nodes,velocities=[[100.,0.,0.]]))
    np.testing.assert_array_equal(out.tensions,0.)
    np.testing.assert_array_equal(out.force_components['elastic_force'],0.)
    np.testing.assert_array_equal(out.force_components['damping_force'],0.)
    assert out.stiffness.nnz==0 and out.damping.nnz==0


def test_reference_weight_mass_force_and_moment_balance():
    model,states,nodes=coupled_case()
    out=model.evaluate(states,nodes)
    kin=model.bodies[0].point_kinematics(states['body'],(.5,.2,-.4))
    positions=np.array([(-2.,-1.,-5.),nodes.positions[0],kin.position])
    masses=np.array([1.1,2.2,1.1])
    net=masses-1025.*.001*np.array([.55,1.1,.55])
    external=net[:,None]*[0.,0.,-9.81]-masses[:,None]*np.array([[0.,0.,0.],nodes.accelerations[0],kin.acceleration])
    np.testing.assert_allclose(np.sum(out.endpoint_forces,axis=0),np.sum(external,axis=0),atol=1e-13)
    np.testing.assert_allclose(np.sum(np.cross(positions,out.endpoint_forces),axis=0),np.sum(np.cross(positions,external),axis=0),atol=2e-13)
    np.testing.assert_allclose(out.anchor_reactions,-out.endpoint_forces[:1])
    assert out.kinetic_energy==pytest.approx(.5*np.sum(masses[:,None]*np.array([[0.,0.,0.],nodes.velocities[0],kin.velocity])**2))


def test_public_wetting_refusal_and_boundary_tangent():
    b=RigidBody6D('body',10.,(0.,0.,0.),np.eye(3))
    l=line((Anchor((-1.,0.,-1.)),Fairlead('body',(0.,0.,0.))),rest=.5)
    model=MooringLineAssembler((b,),(l,),0,(0.,0.,-10.),1000.,0.)
    nodes=MooringNodeState(np.empty((0,3)),np.empty((0,3)),np.empty((0,3)))
    result=model.evaluate({'body':b.state()},nodes)
    assert np.isfinite(result.stiffness.data).all()
    for method in (model.force,model.evaluate):
        with pytest.raises(MooringLineError,match='wetting'):
            method({'body':b.state(position=(0.,0.,1e-9))},nodes)


def test_inputs_are_immutable_and_results_do_not_mutate_model():
    model,nodes=fixed_case()
    before=model.force({},nodes)
    for a in (nodes.positions,model.lines[0].rest_lengths,model._starts,model._node_ep):
        with pytest.raises(ValueError):a.flags.writeable=True
    out=model.evaluate({},nodes);out.force[:]=123.;out.stiffness.data[:]=123.
    np.testing.assert_array_equal(model.force({},nodes),before)


@pytest.mark.parametrize('bad',[True,np.nan,1+2j])
def test_bad_numeric_inputs_refused(bad):
    with pytest.raises(MooringLineError):
        MooringNodeState([[bad,0.,-1.]],[[0.,0.,0.]],[[0.,0.,0.]])


def test_unknown_body_duplicate_or_missing_nodes_and_collapsed_segments_refused():
    l=line((Anchor((-1.,0.,-2.)),Node(0),Fairlead('missing',(0.,0.,0.))))
    with pytest.raises(MooringLineError,match='unknown body'):
        MooringLineAssembler((),(l,),1,(0.,0.,-10.),1000.,0.)
    model,nodes=fixed_case()
    with pytest.raises(MooringLineError,match='exactly once'):
        replace(model,node_count=2)
    with pytest.raises(MooringLineError,match='exactly once'):
        replace(model,lines=(model.lines[0],replace(model.lines[0],line_id='other')))
    with pytest.raises(MooringLineError,match='collapsed'):
        model.force({},replace(nodes,positions=[[-1.,0.,-2.]]))


def test_three_line_inventory_has_independent_node_blocks():
    lines=[]
    for i in range(3):
        lines.append(replace(line((Anchor((-1.,float(i),-2.)),Node(i),Anchor((1.,float(i),-2.)))),line_id=str(i)))
    model=MooringLineAssembler((),lines,3,(0.,0.,-10.),1000.,0.)
    nodes=MooringNodeState([[0.,i,-2.] for i in range(3)],np.zeros((3,3)),np.zeros((3,3)))
    out=model.evaluate({},nodes)
    np.testing.assert_allclose(out.stiffness.toarray(),np.kron(np.eye(3),np.diag([250.,50.,50.])))
    assert out.stiffness.nnz==9


@pytest.mark.parametrize('gap',[1e-7,0.])
def test_fairlead_near_slack_and_at_kink_has_inactive_body_tangent(gap):
    b=RigidBody6D('body',10.,(0.,0.,0.),np.eye(3))
    l=line((Anchor((0.,0.,-2.)),Fairlead('body',(0.,0.,0.))),rest=1.)
    model=MooringLineAssembler((b,),(l,),0,(0.,0.,0.),1000.,0.)
    nodes=MooringNodeState(np.empty((0,3)),np.empty((0,3)),np.empty((0,3)))
    state=b.state(position=(1.-gap,0.,-2.))
    out=model.evaluate({'body':state},nodes)
    np.testing.assert_array_equal(out.force,0.)
    np.testing.assert_array_equal(out.stiffness.toarray(),0.)
    np.testing.assert_array_equal(out.damping.toarray(),0.)


@pytest.mark.parametrize('raw_tension',[1e-7,-1e-7,0.])
def test_fairlead_tension_clip_branch_tangents(raw_tension):
    b=RigidBody6D('body',10.,(0.,0.,0.),np.eye(3))
    l=line((Anchor((0.,0.,-2.)),Fairlead('body',(0.,0.,0.))),rest=1.,ea=100.,damping=4.)
    model=MooringLineAssembler((b,),(l,),0,(0.,0.,0.),1000.,0.)
    nodes=MooringNodeState(np.empty((0,3)),np.empty((0,3)),np.empty((0,3)))
    # Exactly representable length and elastic tension avoid accidental switches.
    state=b.state(position=(1.25,0.,-2.),velocity=((raw_tension-25.)/4.,0.,0.,0.,0.,0.))
    out=model.evaluate({'body':state},nodes)
    active=raw_tension>0.
    assert out.stiffness[0,0]==pytest.approx(100. if active else 0.,abs=2e-8)
    assert out.damping[0,0]==pytest.approx(4. if active else 0.,abs=2e-8)
    assert out.tensions[0]==pytest.approx(max(raw_tension,0.),abs=1e-13)


def test_short_segment_body_tangent_uses_analytic_geometry():
    b=RigidBody6D('body',10.,(0.,0.,0.),np.eye(3))
    l=line((Anchor((0.,0.,-2.)),Fairlead('body',(0.,0.,0.))),rest=.5e-6,ea=100.)
    model=MooringLineAssembler((b,),(l,),0,(0.,0.,0.),1000.,0.)
    nodes=MooringNodeState(np.empty((0,3)),np.empty((0,3)),np.empty((0,3)))
    out=model.evaluate({'body':b.state(position=(1e-6,0.,-2.))},nodes)
    np.testing.assert_allclose(out.stiffness.toarray()[:3,:3],np.diag([2e8,1e8,1e8]),rtol=1e-14)


def test_two_bodies_three_lines_work_and_mass_inventory():
    bodies=tuple(RigidBody6D(name,10.,(0.,0.,0.),np.diag([2.,3.,4.])) for name in ('A','B'))
    states={b.body_id:b.state(position=(x,0.,-3.),rotation_vector=(.2,.1,-.3),
                              velocity=(.1,.2,.3,-.2,.1,.3)) for b,x in zip(bodies,(-2.,2.))}
    lines=tuple(replace(line((Fairlead('A',(0.,i-1.,-.3)),Node(i),Fairlead('B',(0.,i-1.,-.3))),rest=1.),line_id=str(i)) for i in range(3))
    model=MooringLineAssembler(bodies,lines,3,(0.,0.,-9.81),1025.,0.)
    nodes=MooringNodeState([[0.,i-1.,-4.] for i in range(3)],np.full((3,3),.2),np.zeros((3,3)))
    out=model.evaluate(states,nodes)
    velocity=np.r_[states['A'].velocity,states['B'].velocity,nodes.velocities.ravel()]
    endpoint_velocities=[]
    for i in range(3):
        endpoint_velocities.extend([bodies[0].point_kinematics(states['A'],(0.,i-1.,-.3)).velocity,
                                    nodes.velocities[i],bodies[1].point_kinematics(states['B'],(0.,i-1.,-.3)).velocity])
    endpoint_velocities=np.array(endpoint_velocities)
    assert out.force@velocity==pytest.approx(np.sum(out.endpoint_forces*endpoint_velocities),abs=1e-12)
    assert .5*velocity@(out.mass@velocity)==pytest.approx(out.kinetic_energy,abs=1e-13)
    assert out.mass.shape==(21,21) and out.tensions.shape==(6,)
    np.testing.assert_allclose(sum(out.force_components.values()),out.force,atol=1e-13)


def test_large_node_inventory_retains_sparse_band_structure():
    count=1000
    endpoints=(Anchor((0.,0.,-2.)),*(Node(i) for i in range(count)),Anchor((count+1.,0.,-2.)))
    model=MooringLineAssembler((),(line(endpoints),),count,(0.,0.,-10.),1000.,0.)
    nodes=MooringNodeState(np.c_[np.arange(1,count+1),np.zeros(count),np.full(count,-2.)],np.zeros((count,3)),np.zeros((count,3)))
    out=model.evaluate({},nodes)
    assert out.stiffness.shape==(3*count,3*count)
    assert out.stiffness.nnz<=9*count and out.mass.nnz==3*count
    np.testing.assert_allclose(out.mass.diagonal(),1.6)


def test_body_tangent_probes_shrink_near_chart_seam():
    model,states,nodes=coupled_case()
    theta=(np.pi-CHART_BRANCH_MARGIN-1e-7,0.,0.)
    states={'body':replace(states['body'],rotation_vector=theta)}
    out=model.evaluate(states,nodes)
    assert np.isfinite(out.stiffness.data).all()
    assert 0.<out.body_derivative_steps['body'][0]<3e-8
    assert out.body_derivative_steps['body'][1]==2e-6
    h=8e-9
    expected=-(model.force(*perturb(states,nodes,'q',3,h))-model.force(*perturb(states,nodes,'q',3,-h)))/(2*h)
    np.testing.assert_allclose(out.stiffness.toarray()[:,3],expected,rtol=3e-6,atol=3e-5)


def test_exhausted_chart_differentiation_domain_refuses_tangent_explicitly():
    model,states,nodes=coupled_case()
    states={'body':replace(states['body'],rotation_vector=(np.pi-CHART_BRANCH_MARGIN-1e-14,0.,0.))}
    assert np.isfinite(model.force(states,nodes)).all()
    with pytest.raises(MooringLineError,match='differentiation domain'):
        model.evaluate(states,nodes)


def test_nonfinite_energy_is_refused_for_finite_input_state():
    model,nodes=fixed_case(rest=1.2,damping=0.)
    with np.errstate(over='ignore'),pytest.raises(MooringLineError,match='nonfinite assembly'):
        model.evaluate({},replace(nodes,velocities=[[1e155,0.,0.]]))


def test_nonfinite_support_force_is_refused_even_without_free_coordinates():
    l=line((Anchor((0.,0.,-2.)),Anchor((3.,0.,-2.))),rest=1.,ea=1e308,damping=0.)
    model=MooringLineAssembler((),(l,),0,(0.,0.,0.),1000.,0.)
    nodes=MooringNodeState(np.empty((0,3)),np.empty((0,3)),np.empty((0,3)))
    with np.errstate(over='ignore',invalid='ignore'),pytest.raises(MooringLineError,match='nonfinite assembled'):
        model.force({},nodes)
