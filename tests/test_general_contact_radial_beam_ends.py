"""Independent radial endpoint laws, hidden ownership roots and static solves."""
from dataclasses import replace
from fractions import Fraction
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.integrate import quad
from anyfesolver import (ContactDefinition,ContactSurface,ContactInteraction,ContactSolverSettings,
    RigidContactBody,BoundaryCondition,LoadCase,DisplacementControl,solve_static_nonlinear,
    validate_general_contact,report_general_contact_capability)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement,QuadraticBeamElement
from anyfesolver.general_contact.beam_radial_ends import BeamRadialEndManager,PROFILE
from anyfesolver.general_contact.beam_end_radial_geometry import (ownership_path_admissible,
    end_controls,point_sign,regular_curve_path)


def _case(kind='sphere',quadratic=False,ke=1000.):
    model=FEModel('radial physical beam ends')
    model.add_material('elastic',1e3,.3,density=1.)
    xs=(.112,.212,.312) if quadratic else (.112,.312)
    for node,x in enumerate(xs,1):model.add_node(node,x,0.,0.)
    nodes=tuple(range(1,len(xs)+1))
    model.add_element(1,(QuadraticBeamElement if quadratic else BeamElement)(1,nodes,'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    body=RigidContactBody('tool',kind,(0.,0.,0.),normal=None if kind=='sphere' else (0.,0.,1.),radius=.1)
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),),
        (body,),(ContactInteraction('pair','tube','tool',None,.004,
            line_stiffness=1e4,endpoint_stiffness=ke),))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7,
        maximum_line_subdivisions=14,maximum_swept_subdivisions=16)
    return model,definition,settings


def _oracle(point,kind):
    p=np.array(point,dtype=float)
    if kind=='cylinder':p[2]=0.
    return np.linalg.norm(p)-.11


@pytest.mark.parametrize('kind',('sphere','cylinder'))
@pytest.mark.parametrize('quadratic',(False,True))
def test_independent_line_endpoint_energy_virtual_work_and_complete_tangent(kind,quadratic):
    model,definition,settings=_case(kind,quadratic)
    manager=BeamRadialEndManager(model,definition,settings)
    double=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=2000.),))
    other=BeamRadialEndManager(model,double,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[:3]]=(-.004,.0003,.0002)
    if quadratic:u[model.mesh.dof_manager.get_node_dofs(2)[:3]]=(-.001,.002,0.)
    f,K,r=manager.evaluate(u);f2,K2,r2=other.evaluate(u)
    p=model.mesh.nodes[1].coords()+u[model.mesh.dof_manager.get_node_dofs(1)[:3]]
    gap=_oracle(p,kind)
    assert r.active_endpoints==1 and r.profile==PROFILE
    assert r.endpoint_contact_energy==pytest.approx(.5*1000*gap**2,rel=1e-12)
    assert r2.line_contact_energy==r.line_contact_energy
    assert r2.endpoint_contact_energy==pytest.approx(2*r.endpoint_contact_energy,rel=1e-13)
    np.testing.assert_allclose(np.asarray(r.beam_force)+r.rigid_force,0.,atol=1e-12)
    direction=np.zeros_like(u)
    direction[model.mesh.dof_manager.get_node_dofs(1)[:3]]=(.3,-.4,.2)
    if quadratic:direction[model.mesh.dof_manager.get_node_dofs(2)[:3]]=(.07,.1,-.05)
    errors=[];full_errors=[]
    for h in (1e-6,3e-7,1e-7):
        fp=manager.evaluate(u+h*direction,tangent=False)[0]
        fm=manager.evaluate(u-h*direction,tangent=False)[0]
        expected=K@direction
        full_errors.append(np.linalg.norm((fp-fm)/(2*h)-expected)/max(np.linalg.norm(expected),1e-12))
        ap=other.evaluate(u+h*direction,tangent=False)[0]-fp
        am=other.evaluate(u-h*direction,tangent=False)[0]-fm
        expected=(K2-K)@direction
        errors.append(np.linalg.norm((ap-am)/(2*h)-expected)/max(np.linalg.norm(expected),1e-12))
        plus=model.mesh.nodes[1].coords()+(u+h*direction)[model.mesh.dof_manager.get_node_dofs(1)[:3]]
        minus=model.mesh.nodes[1].coords()+(u-h*direction)[model.mesh.dof_manager.get_node_dofs(1)[:3]]
        assert (.5*1000*(_oracle(plus,kind)**2-_oracle(minus,kind)**2)/(2*h))==pytest.approx(
            (f2-f)@direction,rel=1e-5,abs=1e-9)
    assert min(errors)<=1e-5,errors
    assert min(full_errors)<=1e-5,full_errors


@pytest.mark.parametrize('height',(Fraction(2,5),Fraction(1)))
def test_exact_double_ownership_root_rejects_without_endpoint_sign_change(height):
    # y=(.75,height*(t-.5),0), v=(0,.01*(t-.5),.02).
    # q=.01*height*(t-.5)^2: positive endpoints, active double root at t=.5.
    first=(((Fraction(3,4),),(-height/2,),(Fraction(0),)),
        ((Fraction(0),),(Fraction(-1,200),),(Fraction(1,50),)))
    last=(((Fraction(3,4),),(height/2,),(Fraction(0),)),
        ((Fraction(0),),(Fraction(1,200),),(Fraction(1,50),)))
    assert Fraction(3,4)<Fraction(4,5)
    if height==1:assert Fraction(9,16)+height**2/4>Fraction(16,25)  # Both samples open.
    assert not ownership_path_admissible(first,last,.8,16)
    # The same ownership root is harmless when the complete point path is open.
    opened=(((Fraction(1),),(-height/2,),(Fraction(0),)),first[1])
    opened_last=(((Fraction(1),),(height/2,),(Fraction(0),)),last[1])
    assert ownership_path_admissible(opened,opened_last,.8,16)


def test_actual_curved_b3_end_derivative_overrides_chord_and_junction_ownership():
    model,definition,settings=_case(quadratic=True)
    manager=BeamRadialEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[0]]=-.004
    u[model.mesh.dof_manager.get_node_dofs(2)[0]]=.1587-.212
    # Keep the spatial curve regular where the axial derivative changes sign.
    u[model.mesh.dof_manager.get_node_dofs(2)[1]]=.002
    result=manager.evaluate(u)[2]
    assert result.active_endpoints==0
    assert result.to_dict()['endpoint_fields'][0]['state']=='side_owned'
    # Actual initial end derivative points towards -x while the chord points +x.
    assert -3*.108+4*.1587-.312<0 and .312-.108>0
    model.add_node(10,.112,-.1,0.);model.add_node(11,.112,.1,0.)
    model.add_element(2,QuadraticBeamElement(2,(10,1,11),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    assert set(BeamRadialEndManager(model,definition,settings).endpoints)=={3}


def _static(kind='sphere'):
    model,definition,settings=_case(kind,ke=1e5)
    definition=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=1e-5),))
    model.add_boundary_condition(BoundaryCondition('root',(2,),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guide',(1,),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    load=LoadCase('press');load.add_nodal_load(1,forces=(-.2,0.,0.))
    return model,definition,settings,load


@pytest.mark.parametrize('kind',('sphere','cylinder'))
def test_actual_force_displacement_and_fresh_active_restart_with_independent_bar(kind):
    model,definition,settings,load=_static(kind)
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,tolerance=1e-10)
    result=solve_static_nonlinear(model,load,**args)
    assert result.status=='completed',result.failure_reason
    L=.2;g0=.002;EA=1.;k=1e4;ke=1e5;P=.2
    def balance(delta):
        length=L+g0+delta
        return EA/L*(g0+delta)+ke*delta+k*L*(delta**2/(2*length)-delta**3/(6*length**2))-P
    delta=brentq(balance,0.,1e-5,xtol=1e-18)
    dof=model.mesh.dof_manager.get_node_dofs(1)[0]
    assert result.displacements[dof]==pytest.approx(-g0-delta,rel=1e-6,abs=1e-10)
    assert result.info['general_contact']['minimum_gap']>=-1e-5
    model,definition,settings,load=_static(kind)
    prefix=solve_static_nonlinear(model,load,**{**args,'num_steps':2},max_load_factor=.75,
        emit_restart_checkpoint=True)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    model,definition,settings,load=_static(kind)
    resumed=solve_static_nonlinear(model,load,**{**args,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    with pytest.raises(ValueError,match='restart|incompatible|contract'):
        solve_static_nonlinear(model,load,**{**args,'contact_settings':replace(settings,
            maximum_swept_subdivisions=settings.maximum_swept_subdivisions+1)},restart_checkpoint=checkpoint)
    model,definition,settings,load=_static(kind)
    prescribed=solve_static_nonlinear(model,load,**args,control='displacement',
        displacement_control=DisplacementControl(node_id=1,dof='ux',target_displacement=-.002005))
    assert prescribed.status=='completed',prescribed.failure_reason
    assert prescribed.info['general_contact']['active_endpoints']==1
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('bar_'+kind+'.json').write_text(json.dumps({
        'displacement_m':result.displacements[dof],'independent_displacement_m':-g0-delta,
        'contact':result.info['general_contact']},indent=2))


def test_new_profile_explicit_law_bounds_and_no_application_admission():
    model,definition,settings=_case()
    report=validate_general_contact(model,definition,settings)
    assert report.valid and report.profile==PROFILE
    for change in (replace(definition.interactions[0],endpoint_stiffness=None),
        replace(definition.interactions[0],gap_tolerance=.01)):
        assert not validate_general_contact(model,replace(definition,interactions=(change,)),settings).valid
    capability=report_general_contact_capability(model,definition,settings)
    assert capability.implemented and not capability.admitted_for_application
    assert not capability.tested and not capability.independently_accepted and not capability.released


def test_tilted_cylinder_literal_map_complete_work_and_rigid_covariance():
    model,definition,settings=_case('cylinder',True)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[:3]]=(-.004,.0003,.0002)
    u[model.mesh.dof_manager.get_node_dofs(2)[:3]]=(-.001,.002,0.)
    f,K,result=BeamRadialEndManager(model,definition,settings).evaluate(u)
    axis=np.array((1.,2.,3.));axis/=np.linalg.norm(axis);angle=.71
    cross=np.array(((0.,-axis[2],axis[1]),(axis[2],0.,-axis[0]),(-axis[1],axis[0],0.)))
    Q=np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*cross@cross
    shift=np.array((.31,-.23,.19))
    other,definition2,_=_case('cylinder',True)
    v=np.zeros_like(u);T=np.eye(len(u))
    for node in other.mesh.nodes.values():
        node.x,node.y,node.z=Q@node.coords()+shift
        dofs=other.mesh.dof_manager.get_node_dofs(node.id)[:3]
        v[dofs]=Q@u[dofs];T[np.ix_(dofs,dofs)]=Q
    definition2=replace(definition2,rigid_bodies=(replace(definition2.rigid_bodies[0],
        point=tuple(shift),normal=tuple(Q[:,2])),))
    manager=BeamRadialEndManager(other,definition2,settings)
    f2,K2,r2=manager.evaluate(v)
    np.testing.assert_allclose(f2,T@f,rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(K2.toarray(),T@K.toarray()@T.T,rtol=1e-9,atol=1e-8)
    assert r2.line_contact_energy==pytest.approx(result.line_contact_energy,rel=1e-11)
    assert r2.endpoint_contact_energy==pytest.approx(result.endpoint_contact_energy,rel=1e-11)
    np.testing.assert_allclose(r2.beam_moment,Q@np.array(result.beam_moment)+np.cross(shift,r2.beam_force),
        rtol=1e-10,atol=1e-11)
    direction=np.random.default_rng(702).normal(size=len(u))*.1
    h=3e-7
    fp,_,rp=manager.evaluate(v+h*direction,tangent=False)
    fm,_,rm=manager.evaluate(v-h*direction,tangent=False)
    assert np.linalg.norm((fp-fm)/(2*h)-K2@direction)/np.linalg.norm(K2@direction)<=1e-5
    ep=rp.endpoint_contact_energy+rp.line_contact_energy
    em=rm.endpoint_contact_energy+rm.line_contact_energy
    assert (ep-em)/(2*h)==pytest.approx(f2@direction,rel=1e-5,abs=1e-9)


@pytest.mark.parametrize('kind',('sphere','cylinder'))
def test_complete_affine_sweeps_common_motion_crossing_and_singular_curve(kind):
    model,definition,settings=_case(kind)
    motion=np.array((.05,-.03,.02))
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],
        path_translation=tuple(motion)),))
    manager=BeamRadialEndManager(model,definition,settings)
    first=np.zeros(model.mesh.dof_manager.total_dofs);last=first.copy()
    for node in (1,2):last[model.mesh.dof_manager.get_node_dofs(node)[:3]]=motion
    assert manager.swept_increment_admissible(first,last,0.,1.)
    manager.set_path_factor(1.)
    np.testing.assert_allclose(manager.evaluate(last)[0],0.,atol=1e-12)
    manager.set_path_factor(0.)
    # Both endpoint poses open, but the complete beam crosses the obstacle.
    crossed=first.copy()
    for node in (1,2):crossed[model.mesh.dof_manager.get_node_dofs(node)[0]]=-.624
    assert not manager.swept_increment_admissible(first,crossed,0.,0.)
    # Open poses with reversed material parametrization collapse at the midpoint.
    reversed_pose=first.copy()
    reversed_pose[model.mesh.dof_manager.get_node_dofs(1)[0]]=.2
    reversed_pose[model.mesh.dof_manager.get_node_dofs(2)[0]]=-.2
    assert not manager.swept_increment_admissible(first,reversed_pose,0.,0.)
    assert manager.path_factor==0.


@pytest.mark.parametrize('kind',('sphere','cylinder'))
def test_prescribed_tool_activation_open_restart_work_and_separation(kind):
    model,definition,settings,_=_static(kind)
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],
        path_translation=(.004,0.,0.)),))
    manager=BeamRadialEndManager(model,definition,settings)
    activation=float((Fraction.from_float(.112)-Fraction.from_float(manager.radius))/Fraction.from_float(.004))
    for _ in range(8):
        point,_=end_controls(manager.cells[0].reference,manager.point_at(activation),manager.projector,0,-1)
        if point_sign(point,manager.radius)>=0:break
        activation=math.nextafter(activation,0.)
    else:raise AssertionError('activation stage lacks represented open pose')
    load=LoadCase('tool only')
    args=dict(contact=definition,contact_settings=settings,num_steps=2,max_iterations=20,tolerance=1e-10)
    prefix=solve_static_nonlinear(model,load,**args,max_load_factor=activation,emit_restart_checkpoint=True)
    assert prefix.status=='completed',prefix.failure_reason
    assert prefix.info['general_contact']['active_endpoints']==0
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    model,definition,settings,_=_static(kind)
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=(.004,0.,0.)),))
    result=solve_static_nonlinear(model,load,**{**args,'contact':definition},restart_checkpoint=checkpoint)
    assert result.status=='completed',result.failure_reason
    info=result.info['general_contact'];assert info['active_endpoints']==1
    u=result.displacements[model.mesh.dof_manager.get_node_dofs(1)[0]]
    storage=.5*5*u*u+info['endpoint_contact_energy']+info['line_contact_energy']
    work=info['prescribed_actuator_work_trapezoid']
    assert work==pytest.approx(storage,rel=1e-6,abs=1e-12)
    previous_factor=previous_force=0.;oracle_work=0.
    for step in result.info['force_displacement_history']:
        factor=step['load_factor'];displacement=step['control_value']
        delta=max(.004*factor-.002-displacement,0.)
        force=1e5*delta+1e4*.2*delta**2/(2*(.2-displacement))
        oracle_work+=.5*(force+previous_force)*.004*(factor-previous_factor)
        previous_factor,previous_force=factor,force
    assert work==pytest.approx(oracle_work,rel=1e-6,abs=1e-12)
    assert info['rigid_contact_work_trapezoid']==pytest.approx(-work,rel=1e-12)
    assert any(row.get('residual_evaluated') is False for row in result.info['convergence_adaptation'])
    manager=BeamRadialEndManager(model,definition,settings)
    manager.set_path_factor(1.)
    assert manager.evaluate(result.displacements)[2].active_endpoints==1
    manager.set_path_factor(0.)
    opened=manager.evaluate(result.displacements)[2]
    assert opened.active_endpoints==0 and opened.line_contact_energy==0.
    manager.set_path_factor(1.)
    assert manager.evaluate(result.displacements)[2].endpoint_contact_energy==pytest.approx(info['endpoint_contact_energy'])
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('tool_'+kind+'.json').write_text(json.dumps({
        'storage_J':storage,'work_J':work,'independent_work_J':oracle_work,'contact':info,
        'history':result.info['force_displacement_history']},indent=2))


def test_regular_b2_rotating_path_and_collapsed_reversal():
    first=np.array(((1.,0.,0.),(2.,0.,0.)))
    turned=np.array(((1.,0.,0.),(1.,1.,0.)))
    assert regular_curve_path(first,turned,16)
    assert not regular_curve_path(first,np.array(((1.,0.,0.),(0.,0.,0.))),16)


def test_two_curved_contact_pockets_independent_integral_and_explicit_budget_failure():
    model,definition,settings=_case(quadratic=True)
    for node,x in zip(model.mesh.nodes.values(),(-.15,0.,.15)):
        node.x=x;node.y=.2
    manager=BeamRadialEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,y in zip((1,2,3),(-.11,.14,-.11)):
        u[model.mesh.dof_manager.get_node_dofs(node)[1]]=y-.2
    result=manager.evaluate(u)[2]
    # x=.3*t, y=.14-t^2. q=t^4-.19*t^2+.0196.
    # Each of the two active pockets lies between two independent analytic roots.
    t2=np.roots((1.,-.19,.0196-.11**2))
    roots=sorted([.5+sign*math.sqrt(float(v)) for v in t2 for sign in (-1.,1.)])
    def gap(s):
        t=s-.5;return math.hypot(.3*t,.14-t*t)-.11
    expected=sum(quad(lambda s:.5*1e4*min(gap(s),0.)**2*.3,l,r,
        epsabs=1e-13,epsrel=1e-10)[0] for l,r in ((roots[0],roots[1]),(roots[2],roots[3])))
    assert expected>0 and result.active_endpoints==0
    assert result.line_contact_energy==pytest.approx(expected,rel=1e-6,abs=1e-12)
    tight=BeamRadialEndManager(model,definition,replace(settings,
        line_integration_relative_tolerance=1e-15,maximum_line_subdivisions=1))
    with pytest.raises(MemoryError,match='integration budget'):
        tight.evaluate(u)
    exact=replace(definition,surfaces=(replace(definition.surfaces[0],beam_radius=1/64),),
        rigid_bodies=(replace(definition.rigid_bodies[0],radius=1/16),))
    grazing=np.zeros_like(u);exact_manager=BeamRadialEndManager(model,exact,settings)
    for node in (1,2,3):grazing[model.mesh.dof_manager.get_node_dofs(node)[1]]=5/64-.2
    touch=exact_manager.evaluate(grazing)[2]
    assert touch.line_contact_energy==0. and touch.active_endpoints==0


def test_insufficient_swept_budget_rejects_an_unresolved_open_path():
    model,definition,settings=_case()
    definition=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=1e-6),))
    normal=BeamRadialEndManager(model,definition,settings)
    bounded=BeamRadialEndManager(model,definition,replace(settings,maximum_swept_subdivisions=1))
    u=np.zeros(model.mesh.dof_manager.total_dofs);last=u.copy()
    for node in (1,2):last[model.mesh.dof_manager.get_node_dofs(node)[1]]=.02
    assert normal.swept_increment_admissible(u,last,0.,0.)
    assert not bounded.swept_increment_admissible(u,last,0.,0.)


@pytest.mark.parametrize('kind',('sphere','cylinder'))
def test_public_radial_endpoint_example(kind):
    from examples.general_contact_beam_end_capped_cylinder import run
    result=run(radial=kind)
    assert result.info['general_contact']['profile']==PROFILE
    assert result.info['general_contact']['active_endpoints']==1


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_candidate_failure_or_cancellation_keeps_radial_model_reusable(cancel,monkeypatch):
    from anyfesolver import CancellationToken,SolveCancelled
    model,definition,settings,load=_static('sphere')
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken()
    armed=[False];raised=[False];original=BeamRadialEndManager.evaluate
    def interrupt(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.5:
            if cancel:token.cancel('radial end owned interruption')
            else:armed[0]=True
    def fail_after_candidate(self,*values,**options):
        candidate=original(self,*values,**options)
        if armed[0] and not raised[0]:
            raised[0]=True
            assert self._geometry_cache is None
            raise RuntimeError('radial end owned interruption')
        return candidate
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(BeamRadialEndManager,'evaluate',fail_after_candidate)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='radial end owned interruption'):
            solve_static_nonlinear(model,load,**args,progress_callback=interrupt,cancellation_token=token)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**args)
    fresh,definition2,settings2,load2=_static('sphere')
    reference=solve_static_nonlinear(fresh,load2,**{**args,'contact':definition2,'contact_settings':settings2})
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_failed_tool_evaluation_discards_geometry_and_restores_path_factor(monkeypatch):
    model,definition,settings=_case()
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],
        path_translation=(.004,0.,0.)),))
    manager=BeamRadialEndManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    def fail(*args):raise RuntimeError('owned local integration failure')
    with monkeypatch.context() as patch:
        patch.setattr(manager,'_integrate_cell',fail)
        with pytest.raises(RuntimeError,match='owned local integration failure'):
            manager.tool_work_increment(u,u,0.,1.)
    assert manager._geometry_cache is None and manager.path_factor==0.
    np.testing.assert_array_equal(manager.point,manager.reference_point)
    assert manager.evaluate(u)[2].active_endpoints==0
