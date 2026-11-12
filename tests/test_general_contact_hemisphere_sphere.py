"""Independent constrained-cap distance/work and integrated sphere contact."""
from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import minimize_scalar, brentq
from anyfesolver import (BoundaryCondition, LoadCase, DisplacementControl,
    CancellationToken, SolveCancelled, solve_static_nonlinear,
    validate_general_contact, report_general_contact_capability)
from anyfesolver.general_contact.beam_hemisphere_sphere_work import hemisphere_sphere_work
from anyfesolver.general_contact.beam_hemisphere_sphere import HemisphereSphereManager, PROFILE
from anyfesolver.general_contact.beam_radial_ends import BeamRadialEndManager
from test_general_contact_radial_beam_ends import _case, _static


def test_full_radius_bound_preserves_exact_sum_of_literal_radii():
    from anyfesolver.general_contact.beam_end_radial_geometry import curve_gap_lower
    f=Fraction.from_float
    radius=f(.01)+f(.1);point=f(.11)
    lower=curve_gap_lower(((point,point),(Fraction(0),Fraction(0)),(Fraction(0),Fraction(0))),radius,1e-6,16)
    assert Fraction.from_float(lower)<=point-radius


def angular_oracle(beam,end,sign,R,O,Ro):
    # Minimize over the entire physical hemisphere in the q/tangent plane.
    # Rotational symmetry removes azimuth, without production branch logic.
    v=sign*(beam[1]-beam[0] if len(beam)==2 else (
        -3*beam[0]+4*beam[1]-beam[2] if end==0 else beam[0]-4*beam[1]+3*beam[2]))
    d=v/np.linalg.norm(v);q=beam[end]-O
    t=q-(q@d)*d;t/=np.linalg.norm(t)
    def distance(phi):return np.linalg.norm(q+R*(np.cos(phi)*d-np.sin(phi)*t))-Ro
    r=minimize_scalar(distance,bounds=(0.,np.pi/2),method='bounded',options={'xatol':1e-15})
    return min(distance(0.),distance(np.pi/2),r.fun)


@pytest.mark.parametrize('quadratic',(False,True))
@pytest.mark.parametrize('end',(0,-1))
@pytest.mark.parametrize('beta',(-.3,.3))
def test_independent_cap_minimum_all_coordinates_derivatives_and_wrench(quadratic,end,beta):
    d=np.array([beta,np.sqrt(1-beta*beta),0.]);c=np.array([.109,0.,0.])
    end=(2 if quadratic else 1) if end==-1 else 0;sign=-1 if end==0 else 1
    beam=np.array([c,c-.1*d,c-.2*d] if end==0 and quadratic else (
        [c-.2*d,c-.1*d,c] if quadratic else ([c,c-.2*d] if end==0 else [c-.2*d,c])))
    if quadratic:beam[1,2]+=.0002
    n=len(beam);O=np.array([0.,0.,0.]);x=np.r_[beam.ravel(),O]
    s=hemisphere_sphere_work(beam,end,sign,.01,O,.1)
    assert s.branch==('sphere' if beta<0 else 'rim') and s.gap<0
    assert s.gap==pytest.approx(angular_oracle(beam,end,sign,.01,O,.1),abs=2e-16)
    direction=np.random.default_rng(833).normal(size=len(x));direction/=np.linalg.norm(direction)
    for h in (3e-7,1e-7):
        p=x+h*direction;m=x-h*direction
        bp,bm=p[:3*n].reshape(n,3),m[:3*n].reshape(n,3)
        fd=(angular_oracle(bp,end,sign,.01,p[-3:],.1)-angular_oracle(bm,end,sign,.01,m[-3:],.1))/(2*h)
        assert fd==pytest.approx(s.gradient@direction,rel=1e-7,abs=1e-9)
        sp=hemisphere_sphere_work(bp,end,sign,.01,p[-3:],.1)
        sm=hemisphere_sphere_work(bm,end,sign,.01,m[-3:],.1)
        expected=s.hessian@direction
        assert np.linalg.norm((sp.gradient-sm.gradient)/(2*h)-expected)/np.linalg.norm(expected)<1e-5
    G=s.gradient.reshape(n+1,3)
    np.testing.assert_allclose(G.sum(axis=0),0.,atol=1e-14)
    np.testing.assert_allclose(np.cross(np.r_[beam,[O]],G).sum(axis=0),0.,atol=1e-14)
    np.testing.assert_allclose(G[:-1].sum(axis=0),s.normal,atol=1e-14)
    np.testing.assert_allclose(np.cross(beam,G[:-1]).sum(axis=0),np.cross(s.skin_point,s.normal),atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(s.rigid_point-O),.1,atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(s.skin_point-O),.1+s.gap,atol=1e-14)
    if quadratic and beta>0:assert np.linalg.norm(G[1])>0
    angle=.43;Qrot=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
    shift=np.array([.7,-.3,.2]);T=np.kron(np.eye(n+1),Qrot)
    rotated=hemisphere_sphere_work(beam@Qrot.T+shift,end,sign,.01,O@Qrot.T+shift,.1)
    assert rotated.gap==pytest.approx(s.gap,abs=2e-16)
    np.testing.assert_allclose(rotated.gradient,T@s.gradient,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(rotated.hessian,T@s.hessian@T.T,rtol=1e-9,atol=1e-11)


def test_exact_activation_one_sided_seam_and_nonunique_open_parallel_rim():
    for branch,c,R,Ro in (('sphere',np.array([.140625,0.,0.]),.015625,.125),
                          ('rim',np.array([.625,.375,0.]),.125,.625)):
        beam=np.array([c-[0.,.1,0.],c])
        # Sphere seam at q.d=0; rim case is strictly q.d>0.
        s=hemisphere_sphere_work(beam,1,1,R,[0.,0.,0.],Ro)
        assert s.branch==branch and s.gap==0.
        for sign in (-1,1):
            other=beam.copy();delta=np.nextafter(c[0],np.inf if sign>0 else -np.inf)-c[0]
            other[:,0]+=delta
            gap=hemisphere_sphere_work(other,1,1,R,[0.,0.,0.],Ro).gap
            assert gap*sign>0
    h=.109;L=.2;c=np.array([h,0.,0.]);beam=np.array([c-[0.,L,0.],c])
    seam=hemisphere_sphere_work(beam,1,1,.01,[0.,0.,0.],.1)
    direction=np.zeros(9);direction[0]=-L
    assert direction@seam.hessian@direction==pytest.approx(0.,abs=1e-15)
    p=beam.copy();p[0,0]-=L*1e-8
    positive=hemisphere_sphere_work(p,1,1,.01,[0.,0.,0.],.1)
    assert direction@positive.hessian@direction==pytest.approx(.01*h/(h-.01),rel=1e-7)
    np.testing.assert_allclose(positive.gradient,seam.gradient,atol=1e-8)
    parallel=hemisphere_sphere_work(np.array([[.2,0.,0.],[.3,0.,0.]]),1,1,.01,[0.,0.,0.],.1)
    assert parallel.branch=='parallel_rim' and parallel.gap>0 and parallel.skin_point is None
    np.testing.assert_array_equal(parallel.gradient,0.)


def support_case(quadratic=False):
    model,definition,settings=_case('sphere',quadratic)
    return model,definition,replace(settings,beam_end_cap_work='hemisphere_support_dev')


@pytest.mark.parametrize('quadratic',(False,True))
def test_complete_line_cap_work_sparse_tangent_and_sphere_reaction(quadratic):
    model,definition,settings=support_case(quadratic)
    manager=HemisphereSphereManager(model,definition,settings)
    other=HemisphereSphereManager(model,replace(definition,interactions=(replace(
        definition.interactions[0],endpoint_stiffness=2000.),)),settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[:3]]=(-.004,.0003,.0002)
    if quadratic:u[model.mesh.dof_manager.get_node_dofs(2)[:3]]=(-.001,.002,0.)
    f,K,r=manager.evaluate(u);f2,K2,r2=other.evaluate(u)
    assert r.active_endpoints==1 and r.profile==PROFILE
    assert r2.endpoint_contact_energy==pytest.approx(2*r.endpoint_contact_energy,rel=1e-13)
    assert r2.line_contact_energy==r.line_contact_energy
    np.testing.assert_allclose(np.array(r.beam_force)+r.rigid_force,0.,atol=1e-12)
    np.testing.assert_allclose(np.array(r.beam_moment)+r.rigid_moment,0.,atol=1e-12)
    np.testing.assert_allclose(r.rigid_moment,np.cross(manager.point,r.rigid_force),atol=1e-12)
    direction=np.random.default_rng(178).normal(size=len(u))*.1
    for h in (3e-7,1e-7):
        fp,_,rp=manager.evaluate(u+h*direction,tangent=False)
        fm,_,rm=manager.evaluate(u-h*direction,tangent=False)
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
        ep=rp.to_dict()['algorithmic_contact_potential'];em=rm.to_dict()['algorithmic_contact_potential']
        assert (ep-em)/(2*h)==pytest.approx(f@direction,rel=1e-5,abs=1e-9)


def test_active_seam_sweep_keeps_complete_cap_work_and_legacy_rejection():
    model,definition,settings=support_case()
    c=np.array([.11001,0.,0.]);L=.04
    for node,p in ((1,c-[0.,L,0.]),(2,c)):
        model.mesh.nodes[node].x,model.mesh.nodes[node].y,model.mesh.nodes[node].z=p
    definition=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=1e-5),))
    manager=HemisphereSphereManager(model,definition,settings)
    first=np.zeros(model.mesh.dof_manager.total_dofs);last=first.copy()
    for u in (first,last):
        for node in (1,2):u[model.mesh.dof_manager.get_node_dofs(node)[0]]=-1.05e-5
    first[model.mesh.dof_manager.get_node_dofs(1)[0]]+=L*1e-3
    last[model.mesh.dof_manager.get_node_dofs(1)[0]]-=L*1e-3
    assert manager.swept_increment_admissible(first,last,0.,1.), {
        'regular':manager._regular(first,last),'lower_bounds':[
            manager._minimum_gap(first),manager._minimum_gap(last),manager._minimum_gap((first+last)/2)]}
    legacy=BeamRadialEndManager(model,definition,replace(settings,beam_end_cap_work='owned_point'))
    assert not legacy.swept_increment_admissible(first,last,0.,1.)
    assert manager.evaluate(first)[2].endpoint_fields[-1]['branch']=='sphere'
    assert manager.evaluate(last)[2].endpoint_fields[-1]['branch']=='rim'
    # Open endpoint poses cannot hide crossing through the entire sphere.
    crossed=np.zeros_like(first)
    for node in (1,2):crossed[model.mesh.dof_manager.get_node_dofs(node)[0]]=-.3
    assert not manager.swept_increment_admissible(np.zeros_like(first),crossed,0.,1.)


def test_material_junction_and_b3_midnode_are_not_caps():
    from anyfesolver.elements import QuadraticBeamElement
    model,definition,settings=support_case(True)
    model.add_node(10,.112,-.1,0.);model.add_node(11,.112,.1,0.)
    model.add_element(2,QuadraticBeamElement(2,(10,1,11),'elastic',
        {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    assert set(HemisphereSphereManager(model,definition,settings).endpoints)=={3}


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_sphere_cap_exception_or_cancellation_preserves_fresh_replay(cancel,monkeypatch):
    model,definition,settings,load=_static('sphere');settings=replace(settings,beam_end_cap_work='hemisphere_support_dev')
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False];original=HemisphereSphereManager.evaluate
    def callback(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.75:
            if cancel:token.cancel('sphere cap interruption')
            else:armed[0]=True
    def failing(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if armed[0] and not raised[0]:
            raised[0]=True;raise RuntimeError('sphere cap interruption')
        return result
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(HemisphereSphereManager,'evaluate',failing)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='sphere cap interruption'):
            solve_static_nonlinear(model,load,**common,cancellation_token=token,progress_callback=callback)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**common)
    fresh,_,_,_=_static('sphere');reference=solve_static_nonlinear(fresh,load,**common)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_coupled_bar_force_displacement_restart_and_explicit_scope():
    model,definition,settings,load=_static('sphere');settings=replace(settings,beam_end_cap_work='hemisphere_support_dev')
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**common);assert result.status=='completed',result.failure_reason
    L=.2;g0=.002;EA=1.;k=1e4;ke=1e5;P=.2
    def balance(delta):
        length=L+g0+delta
        return EA/L*(g0+delta)+ke*delta+k*L*(delta**2/(2*length)-delta**3/(6*length**2))-P
    delta=brentq(balance,0.,1e-5,xtol=1e-18);dof=model.mesh.dof_manager.get_node_dofs(1)[0]
    error=abs(result.displacements[dof]/(-g0-delta)-1.)
    assert error<1e-6
    fresh,_,_,_=_static('sphere')
    prefix=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed'
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=_static('sphere')
    resumed=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed'
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    for altered in (replace(settings,beam_end_cap_work='owned_point'),replace(settings,maximum_swept_subdivisions=17)):
        with pytest.raises(ValueError,match='restart|incompatible|contract'):
            solve_static_nonlinear(fresh,load,**{**common,'contact_settings':altered},restart_checkpoint=checkpoint)
    fresh,_,_,_=_static('sphere')
    prescribed=solve_static_nonlinear(fresh,load,**common,control='displacement',
        displacement_control=DisplacementControl(node_id=1,dof='ux',target_displacement=-.002005))
    assert prescribed.status=='completed'
    report=report_general_contact_capability(model,definition,settings)
    assert report.implemented and not report.admitted_for_application
    for body in (replace(definition.rigid_bodies[0],path_translation=(.01,0.,0.)),):
        assert not validate_general_contact(model,replace(definition,rigid_bodies=(body,)),settings).valid
    cylinder=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],kind='cylinder',normal=(0.,0.,1.)),))
    # New public cylinder profile is distinct; the old sphere owner still rejects it.
    with pytest.raises(ValueError,match='sphere'):
        HemisphereSphereManager(model,cylinder,settings)
    for bad in (replace(definition,surfaces=(replace(definition.surfaces[0],beam_end_caps='none'),)),
                replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=None),))):
        assert not validate_general_contact(model,bad,settings).valid
    second=replace(definition.rigid_bodies[0],name='other',point=(1.,0.,0.))
    multi=replace(definition,rigid_bodies=(*definition.rigid_bodies,second),interactions=(
        *definition.interactions,replace(definition.interactions[0],name='other pair',target='other')))
    from anyfesolver.general_contact.multiple_beam_rigid import PROFILE_SUPPORT_ANALYTIC, MultipleBeamRigidManager
    assert validate_general_contact(model,multi,settings).profile==PROFILE_SUPPORT_ANALYTIC
    assert all(isinstance(child,HemisphereSphereManager) for _,child in MultipleBeamRigidManager(model,multi,settings).pairs)
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).write_text(json.dumps({'displacement_m':result.displacements[dof],
        'independent_displacement_m':-g0-delta,'relative_error':error,
        'gap_tolerance_m':definition.interactions[0].gap_tolerance,'contact':result.info['general_contact']},indent=2),encoding='utf-8')


def test_runnable_sphere_cap_example():
    from examples.general_contact_hemisphere_sphere import run
    result=run();assert result.status=='completed'
    assert result.info['general_contact']['profile']==PROFILE
