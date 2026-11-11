"""Independent curved point work and coupled P2-bar checks (development)."""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar, root

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear, validate_general_contact)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_curve_ends import BeamCurveEndManager


def fixture(points, *, kinds=(3, 3), ke=50., kl=1000., radius=.05, tol=.008):
    model = FEModel('quadratic_physical_ends')
    model.add_material('elastic', 1000., .3, density=1.)
    nodes=[]; next_node=1
    for side, (coordinates, kind) in enumerate(zip(points, kinds)):
        ids=tuple(range(next_node,next_node+kind)); next_node+=kind; nodes.append(ids)
        for i,p in zip(ids,coordinates):model.add_node(i,*p)
        cls=BeamElement if kind==2 else QuadraticBeamElement
        model.add_element(side+1,cls(side+1,ids,'elastic',
            {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    definition=ContactDefinition(surfaces=tuple(ContactSurface(name,(i+1,),
        beam_radius=radius,beam_end_caps='hemisphere') for i,name in enumerate(('a','b'))),
        rigid_bodies=(),interactions=(ContactInteraction('pair','a','b',normal_stiffness=None,
            line_stiffness=kl,endpoint_stiffness=ke,gap_tolerance=tol),))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-6,
        maximum_line_subdivisions=16,maximum_swept_subdivisions=18,
        beam_pair_gap_accuracy=1e-5,maximum_beam_pair_gap_cells=10000,
        maximum_beam_pair_gap_depth=25)
    return model,definition,settings,tuple(nodes)


def curved(kinds=(3,3)):
    a=((-1.,.09,0.),(-.53,.09,0.),(-.06,.09,0.))
    if kinds[0]==2:a=(a[0],a[2])
    b=((0.,-.5,0.),(0.,0.,0.),(0.,.5,0.))
    model,definition,settings,nodes=fixture((a,b),kinds=kinds,radius=.025,tol=.04)
    manager=BeamCurveEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in nodes[0]:u[model.mesh.dof_manager.get_node_dofs(node)[0]]=.04
    u[model.mesh.dof_manager.get_node_dofs(nodes[1][1])[0]]=.006
    if kinds[0]==3:u[model.mesh.dof_manager.get_node_dofs(nodes[0][1])[1]]=.008
    return model,definition,settings,nodes,manager,u


def independent_point_energy(a,b,radius,ke):
    """Scalar closest point via a different minimization route, no contact kernels."""
    def position(p,t):
        return ((1-t)*(1-2*t)*p[0]+4*t*(1-t)*p[1]+t*(2*t-1)*p[2])
    def derivative(p,t):return (4*t-3)*p[0]+(4-8*t)*p[1]+(4*t-1)*p[2]
    E=0.;seen=set()
    for side,(source,target) in enumerate(((a,b),(b,a))):
        for endpoint,s in ((0,0.),(2,1.)):
            p=source[endpoint]
            result=minimize_scalar(lambda t:float(np.sum((p-position(target,t))**2)),
                                   bounds=(0.,1.),method='bounded',options={'xatol':1e-14})
            t=min((0.,float(result.x),1.),key=lambda t:float(np.sum((p-position(target,t))**2)))
            separation=p-position(target,t);gap=np.linalg.norm(separation)-radius
            ownership=float(separation@((-1 if endpoint==0 else 1)*derivative(source,s)))
            if gap>=0 or ownership>=0:continue
            key=tuple(sorted(((side,endpoint),(1-side,0 if t==0 else 2)))) if t in (0.,1.) else (side,endpoint,'interior')
            if key in seen:continue
            seen.add(key);E+=.5*ke*gap*gap
    return E


@pytest.mark.parametrize('kinds',((3,3),(2,3)))
def test_curved_reduced_point_work_and_complete_tangent(kinds):
    model,definition,settings,nodes,manager,u=curved(kinds)
    assert validate_general_contact(model,definition,settings).valid
    a,b=manager._positions(u)
    E,f,H,fields,count=manager._endpoint_work((a,b))
    assert count==1 and any(row.get('feature_kind')=='end_interior' for row in fields)
    assert E==pytest.approx(independent_point_energy(a,b,manager.radius,manager.endpoint_stiffness),rel=1e-12)
    rng=np.random.default_rng(845);v=rng.normal(size=18);v/=np.linalg.norm(v)
    for h in (1e-6,3e-7):
        ap,bp=a+h*v[:9].reshape(3,3),b+h*v[9:].reshape(3,3)
        am,bm=a-h*v[:9].reshape(3,3),b-h*v[9:].reshape(3,3)
        fd=(independent_point_energy(ap,bp,manager.radius,manager.endpoint_stiffness)-
            independent_point_energy(am,bm,manager.radius,manager.endpoint_stiffness))/(2*h)
        assert fd==pytest.approx(f@v,rel=1e-7,abs=1e-9)
        fp=manager._endpoint_work((ap,bp))[1];fm=manager._endpoint_work((am,bm))[1]
        assert np.linalg.norm((fp-fm)/(2*h)-H@v)/np.linalg.norm(H@v)<1e-5
    force,K,result=manager.evaluate(u)
    direction=np.zeros_like(u)
    for side in nodes:
        for node in side:direction[model.mesh.dof_manager.get_node_dofs(node)[:3]]=rng.normal(size=3)
    direction/=np.linalg.norm(direction);h=3e-7
    fd=(manager.evaluate(u+h*direction,tangent=False)[0]-manager.evaluate(u-h*direction,tangent=False)[0])/(2*h)
    assert np.linalg.norm(fd-K@direction)/np.linalg.norm(K@direction)<1e-5
    assert np.linalg.norm(np.asarray(result.beam_a_force)+result.beam_b_force)<1e-10
    assert np.linalg.norm(np.asarray(result.beam_a_moment)+result.beam_b_moment)<1e-10
    assert manager.swept_increment_admissible(np.zeros_like(u),u,0.,1.)
    assert manager.evaluate(np.zeros_like(u))[2].contact_energy==0.
    exchanged=replace(definition,surfaces=definition.surfaces[::-1],interactions=(
        replace(definition.interactions[0],surface='b',target='a'),))
    ef,eK,er=BeamCurveEndManager(model,exchanged,settings).evaluate(u)
    np.testing.assert_allclose(ef,force,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(eK.toarray(),K.toarray(),rtol=1e-12,atol=1e-10)
    assert er.contact_energy==pytest.approx(result.contact_energy,rel=1e-12)


def bars():
    model,definition,settings,nodes=fixture((((-.2,0.,0.),(-.1,0.,0.),(0.,0.,0.)),
        ((.104,0.,0.),(.204,0.,0.),(.304,0.,0.))))
    model.add_boundary_condition(BoundaryCondition('outer',(nodes[0][0],nodes[1][-1]),{'ux':0.}))
    model.add_boundary_condition(BoundaryCondition('guides',tuple(sum(nodes,())),
        {name:0. for name in ('uy','uz','rx','ry','rz')}))
    load=LoadCase('two_bar_load');load.add_nodal_load(nodes[0][-1],forces=(.2,0.,0.))
    load.add_nodal_load(nodes[1][0],forces=(-.2,0.,0.))
    return model,definition,settings,nodes,load


def independent_p2_bars():
    x=np.array((-.2,-.1,0.,.104,.204,.304));free=np.array((1,2,3,4))
    K=np.zeros((6,6));local=np.array(((7.,-8.,1.),(-8.,16.,-8.),(1.,-8.,7.)))/(.6)
    K[:3,:3]=local;K[3:,3:]=local;external=np.array((0.,0.,.2,-.2,0.,0.))
    ab,w=np.polynomial.legendre.leggauss(8)
    def residual(unknown):
        u=np.zeros(6);u[free]=unknown;p=x+u;r=K@u-external
        gap=p[3]-p[2]-.1
        if gap<0:r[2]-=50*gap;r[3]+=50*gap
        for side in (0,1):
            src=p[:3] if side==0 else p[3:];target=p[3] if side==0 else p[2]
            # Exact quadratic activation roots and polynomial Gaussian work.
            coef=np.array((src[0]-target, -3*src[0]+4*src[1]-src[2],2*src[0]-4*src[1]+2*src[2]))
            coef[0]+=.1 if side==0 else -.1
            roots=np.polynomial.Polynomial(coef).roots()
            cuts=sorted([0.,1.]+[float(t.real) for t in roots if abs(t.imag)<1e-12 and 0<t.real<1])
            for left,right in zip(cuts,cuts[1:]):
                for q,weight in zip(ab,w):
                    t=(left+right+(right-left)*q)/2
                    N=np.array(((1-t)*(1-2*t),4*t*(1-t),t*(2*t-1)))
                    g=(target-N@src-.1) if side==0 else (N@src-target-.1)
                    if g>=0:continue
                    grad=np.zeros(6)
                    if side==0:grad[:3]=-N;grad[3]=1.
                    else:grad[3:]=N;grad[2]=-1.
                    r+=.5*1000*g*grad*.2*(right-left)*weight/2
        return r[free]
    result=root(residual,np.array((.002,.004,-.004,-.002)),tol=1e-11)
    assert result.success and np.linalg.norm(residual(result.x))<1e-11
    return free,result.x


def test_coupled_p2_bar_force_response_and_active_json_restart(tmp_path):
    model,definition,settings,nodes,load=bars()
    common=dict(contact=definition,contact_settings=settings,tolerance=1e-10,max_iterations=25,
                emit_restart_checkpoint=True,num_steps=4)
    solved=solve_static_nonlinear(model,load,**common)
    assert solved.status=='completed'
    free,reference=independent_p2_bars()
    dofs=np.array([model.mesh.dof_manager.get_node_dofs(node)[0] for side in nodes for node in side])
    actual=solved.displacements[dofs[free]]
    np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-12)
    checkpoint=json.loads(json.dumps(solved.to_restart_checkpoint()))
    fresh,_,_,_,_=bars()
    again=solve_static_nonlinear(fresh,load,**common,restart_checkpoint=checkpoint)
    assert again.status=='completed';np.testing.assert_array_equal(again.displacements,solved.displacements)
    record=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'result.json'))).parent/'p2_bar.json'
    record.write_text(json.dumps({'independent_displacement_m':reference.tolist(),
        'solved_displacement_m':actual.tolist(),'relative_error':float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))},indent=2),encoding='utf-8')


def test_exact_oblique_cap_activation_retains_spring_tangent():
    radius=.21213203435596426
    a=((-1.,-1.,0.),(-.5,-.5,0.),(0.,0.,0.))
    # Prepare an open model, then examine the review's literal active pose.
    b=((.16,.16,0.),(.66,.66,0.),(1.16,1.16,0.))
    model,definition,settings,_=fixture((a,b),radius=radius/2,tol=.001)
    manager=BeamCurveEndManager(model,definition,settings)
    exact_b=np.array(((.15,.15,0.),(.65,.65,0.),(1.15,1.15,0.)))
    E,f,H,fields,count=manager._endpoint_work((np.array(a),exact_b))
    assert np.linalg.norm(np.array((.15,.15,0.)))-radius==0.
    assert E>0 and count==1 and np.linalg.norm(f)>0
    normal=np.array((1.,1.,0.))/np.sqrt(2.)
    assert normal@H[6:9,6:9]@normal==pytest.approx(50.,rel=1e-12)
    assert any(field.get('gap',0.)<0 for field in fields)


def test_projection_certificate_uses_actual_end_tangent_and_rejects_competing_minima():
    from anyfesolver.general_contact.beam_curve_end_geometry import endpoint_interval
    source=np.array(((-1.,.09,0.),(-.1,.1,0.),(-.02,.09,0.)))
    target=np.array(((0.,-.5,0.),(0.,0.,0.),(0.,.5,0.)))
    proof=endpoint_interval(source,source,target,target,2,.05)
    assert proof is not None and not proof.open and proof.ownership==1
    # Its endpoint chord points right, actual Q2 outward derivative points left.
    assert (source[2]-source[0])[0]>0 and (source[0]-4*source[1]+3*source[2])[0]<0
    symmetric=np.array(((-.2,.2,0.),(0.,0.,0.),(.2,.2,0.)))
    tip=np.array(((-.5,.2,0.),(-.25,.2,0.),(0.,.2,0.)))
    assert endpoint_interval(tip,tip,symmetric,symmetric,2,.3) is None
    far=target+np.array((10.,0.,0.))
    assert endpoint_interval(tip,tip,far,far,2,.05).open


def test_middle_node_motion_and_through_crossing_are_not_hidden_by_open_endpoints():
    model,definition,settings,nodes,load=bars()
    manager=BeamCurveEndManager(model,definition,settings)
    first=np.zeros(model.mesh.dof_manager.total_dofs);last=first.copy()
    for node in nodes[1]:last[model.mesh.dof_manager.get_node_dofs(node)[0]]=-.65
    assert manager._certificate(last).conservative_lower_gap>0.
    assert not manager.swept_increment_admissible(first,last,0.,1.)
    fraction=manager.admissible_step_fraction(first,last)
    assert 0<fraction<1 and manager.evaluate(first+fraction*(last-first))[2].gap_tolerance_met
    # Fixed external endpoints with moving middle node can create a stationary cusp.
    folded=first.copy();folded[model.mesh.dof_manager.get_node_dofs(nodes[0][1])[0]]=1.
    assert not manager.swept_increment_admissible(first,folded,0.,1.)
    with pytest.raises(ValueError,match='regularity|cusp'):
        manager.evaluate(folded)


def test_displacement_control_split_restart_and_incompatible_definitions():
    model,definition,settings,nodes,load=bars()
    common=dict(contact=definition,contact_settings=settings,tolerance=1e-10,max_iterations=25,
                emit_restart_checkpoint=True,num_steps=4)
    solved=solve_static_nonlinear(model,load,**common);assert solved.status=='completed'
    free,reference=independent_p2_bars()
    controlled=solve_static_nonlinear(model,load,**common,control='displacement',
        displacement_control=DisplacementControl(node_id=nodes[0][-1],dof='ux',
                                                 target_displacement=float(reference[1])))
    assert controlled.status=='completed',controlled.failure_reason
    np.testing.assert_allclose(controlled.displacements,solved.displacements,rtol=1e-6,atol=1e-10)
    model,_,_,_,_=bars()
    half=solve_static_nonlinear(model,load,**{**common,'num_steps':2},max_load_factor=.5)
    assert half.status=='completed' and half.info['general_contact']['active_endpoints']==1
    checkpoint=json.loads(json.dumps(half.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_,_=bars()
    resumed=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed'
    np.testing.assert_allclose(resumed.displacements,solved.displacements,rtol=1e-6,atol=1e-10)
    changed=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=51.),))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**common,'contact':changed},restart_checkpoint=checkpoint)
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**common,'contact_settings':replace(settings,maximum_swept_subdivisions=17)},restart_checkpoint=checkpoint)
    joined,_,_,_,_=bars();joined.add_node(20,0.,.1,0.)
    joined.add_element(20,BeamElement(20,(nodes[0][-1],20),'elastic',
        {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(joined,load,**common,restart_checkpoint=checkpoint)
    manager=BeamCurveEndManager(joined,definition,settings)
    u=np.zeros(joined.mesh.dof_manager.total_dofs)
    dofs=[joined.mesh.dof_manager.get_node_dofs(node)[0] for side in nodes for node in side]
    u[np.array(dofs)[free]]=reference
    with pytest.raises(ValueError,match='nonphysical'):manager.evaluate(u)


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_trial_exception_or_cancel_restores_origin_and_checkpoint(cancel,monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled
    model,definition,settings,nodes,load=bars()
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=25,
                tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False];original=BeamCurveEndManager.evaluate
    def callback(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.5:
            if cancel:token.cancel('curve end interruption')
            else:armed[0]=True
    def failing(self,*args,**kwargs):
        candidate=original(self,*args,**kwargs)
        if armed[0] and not raised[0]:
            raised[0]=True;raise RuntimeError('curve end interruption')
        return candidate
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(BeamCurveEndManager,'evaluate',failing)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='curve end interruption'):
            solve_static_nonlinear(model,load,**common,progress_callback=callback,cancellation_token=token)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**common)
    fresh,_,_,_,_=bars();reference=solve_static_nonlinear(fresh,load,**common)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_actual_curved_work_is_covariant_and_connectivity_invariant():
    model,definition,settings,nodes,manager,u=curved()
    f,K,r=manager.evaluate(u)
    for element in model.mesh.elements.values():element.node_ids=element.node_ids[::-1]
    rf,rK,rr=BeamCurveEndManager(model,definition,settings).evaluate(u)
    np.testing.assert_allclose(rf,f,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(rK.toarray(),K.toarray(),rtol=1e-10,atol=1e-8)
    assert rr.contact_energy==pytest.approx(r.contact_energy,rel=1e-10)
    angle=.37;Q=np.array(((np.cos(angle),-np.sin(angle),0.),(np.sin(angle),np.cos(angle),0.),(0.,0.,1.)))
    shift=np.array((.7,-.3,.2));rotated=np.zeros_like(u)
    for node in model.mesh.nodes.values():
        node.x,node.y,node.z=Q@node.coords()+shift
        dofs=model.mesh.dof_manager.get_node_dofs(node.id)[:3];rotated[dofs]=Q@u[dofs]
    tf,tK,tr=BeamCurveEndManager(model,definition,settings).evaluate(rotated)
    assert tr.contact_energy==pytest.approx(r.contact_energy,rel=1e-10)
    direction=np.zeros_like(u);other=np.zeros_like(u)
    for node,value in zip(model.mesh.nodes,np.random.default_rng(715).normal(size=(len(model.mesh.nodes),3))):
        dofs=model.mesh.dof_manager.get_node_dofs(node)[:3]
        direction[dofs]=value;other[dofs]=Q@value
        np.testing.assert_allclose(tf[dofs],Q@f[dofs],rtol=1e-10,atol=1e-10)
    x,y=K@direction,tK@other
    for node in model.mesh.nodes:
        dofs=model.mesh.dof_manager.get_node_dofs(node)[:3]
        np.testing.assert_allclose(y[dofs],Q@x[dofs],rtol=1e-10,atol=1e-8)


def test_open_checkpoint_and_public_example():
    model,definition,settings,nodes,load=bars()
    opening=LoadCase('open');opening.add_nodal_load(nodes[0][-1],forces=(-.2,0.,0.))
    opening.add_nodal_load(nodes[1][0],forces=(.2,0.,0.))
    common=dict(contact=definition,contact_settings=settings,num_steps=2,max_iterations=25,
                tolerance=1e-10,emit_restart_checkpoint=True)
    half=solve_static_nonlinear(model,opening,**common,max_load_factor=.5)
    assert half.status=='completed' and half.info['general_contact']['active_endpoints']==0
    checkpoint=json.loads(json.dumps(half.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_,_=bars()
    resumed=solve_static_nonlinear(fresh,opening,**common,restart_checkpoint=checkpoint)
    fresh,_,_,_,_=bars();full=solve_static_nonlinear(fresh,opening,**common)
    assert full.status==resumed.status=='completed'
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1e-6,atol=1e-10)
    from examples.general_contact_beam_curve_ends import run
    example=run();assert example.status=='completed' and example.info['general_contact']['active_endpoints']==1


def test_exact_full_curve_gap_enclosure_and_resource_rejection():
    from anyfesolver.general_contact.beam_curve_end_geometry import certify_curve_pair_gap
    model,definition,settings,nodes,manager,u=curved()
    a,b=manager._positions(u)
    def position(p,t):return (1-t)*(1-2*t)*p[0]+4*t*(1-t)*p[1]+t*(2*t-1)*p[2]
    # For this transverse monotone-source fixture the closest source is its tip;
    # scalar bounded minimization provides a separate distance solution route.
    result=minimize_scalar(lambda t:np.sum((a[2]-position(b,t))**2),bounds=(0.,1.),
                           method='bounded',options={'xatol':1e-14})
    reference=np.sqrt(result.fun)-manager.radius
    for accuracy in (1e-5,1e-7):
        gap=certify_curve_pair_gap(a,b,manager.radius,accuracy=accuracy,
                                  maximum_cells=10000,maximum_depth=25)
        assert gap.conservative_lower_gap <= reference+1e-12
        assert gap.minimum_sampled_gap >= reference-1e-12
        assert gap.minimum_sampled_gap-gap.conservative_lower_gap <= accuracy
    with pytest.raises(MemoryError,match='cell|budget'):
        certify_curve_pair_gap(a,b,manager.radius,accuracy=1e-7,maximum_cells=1,maximum_depth=25)
    with pytest.raises(MemoryError,match='depth|budget'):
        certify_curve_pair_gap(a,b,manager.radius,accuracy=1e-7,maximum_cells=10000,maximum_depth=0)


def test_capability_and_invalid_public_contract_preserve_admission_boundaries():
    from anyfesolver.general_contact import report_general_contact_capability
    from anyfesolver.general_contact.beam_curve_ends import PROFILE
    model,definition,settings,nodes,load=bars()
    capability=report_general_contact_capability(model,definition,settings)
    assert capability.validation.profile==PROFILE and capability.implemented
    assert not capability.independently_accepted and not capability.admitted_for_application
    assert not capability.released
    for name in ('beam_pair_gap_accuracy','maximum_beam_pair_gap_cells','maximum_beam_pair_gap_depth'):
        assert not validate_general_contact(model,definition,replace(settings,**{name:None})).valid
    missing=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=None),))
    assert not validate_general_contact(model,missing,settings).valid
