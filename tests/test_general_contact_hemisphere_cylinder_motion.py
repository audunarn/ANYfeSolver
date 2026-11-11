"""Prescribed actual-cylinder-cap work with independent axial/line balance."""
from dataclasses import replace
import json, os
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import root
from anyfesolver import (LoadCase, DisplacementControl, CancellationToken, SolveCancelled,
    solve_static_nonlinear, validate_general_contact, report_general_contact_capability)
from anyfesolver.general_contact.beam_hemisphere_cylinder import HemisphereCylinderManager
from anyfesolver.general_contact.beam_hemisphere_cylinder_motion import (
    TranslatingHemisphereCylinderManager, PROFILE)
from test_general_contact_hemisphere_cylinder_static import (
    _bar_case, _support_case, _line_reference, _potential)


def fixture(quadratic=False, motion=(.004,0.,0.)):
    model,definition,settings,_=_bar_case(quadratic)
    definition=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=motion),))
    return model,definition,settings,LoadCase('prescribed tool only')


def reference(quadratic,factor):
    x=np.array((.112,.212,.312) if quadratic else (.112,.312))
    n=len(x); L=.2
    K=np.array(((7.,-8.,1.),(-8.,16.,-8.),(1.,-8.,7.)))/(3*L) if quadratic else np.array(((1.,-1.),(-1.,1.)))/L
    point=np.array((.004*factor,0.,0.))
    def evaluate(free):
        u=np.r_[free,0.]; positions=np.column_stack((x+u,np.zeros((n,2))))
        E,f=_line_reference(positions,.11,point,np.array((0.,0.,1.)),1e4,L)
        g=min(x[0]+u[0]-point[0]-.11,0.)
        f[0,0]+=1e5*g
        return K@u+f[:,0],-f[:,0].sum(),E+.5*1e5*g*g,u
    guess=np.linspace(max(.004*factor-.002,0.),0.,n)[:-1]
    solved=root(lambda v:evaluate(v)[0][:-1],guess,tol=1e-10)
    assert np.linalg.norm(evaluate(solved.x)[0][:-1])<1e-10
    residual,force,E,u=evaluate(solved.x)
    return u,force,.5*u@K@u+E,residual[-1]


@pytest.mark.parametrize('quadratic',(False,True))
def test_tool_solve_work_open_recontact_and_fresh_restart(quadratic):
    model,definition,settings,load=fixture(quadratic)
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**common)
    assert result.status=='completed',result.failure_reason
    u,F,storage,reaction=reference(quadratic,1.)
    dofs=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in sorted(model.mesh.nodes)]
    np.testing.assert_allclose(result.displacements[dofs],u,rtol=1e-6,atol=1e-10)
    info=result.info['general_contact']
    assert info['profile']==PROFILE and info['active_endpoints']==1 and info['gap_tolerance_met']
    assert info['rigid_motion_profile']=='exact_literal_axis_translation_affine_static_v2'
    np.testing.assert_allclose(info['rigid_force'],(-F,0.,0.),rtol=1e-6,atol=1e-10)
    assert result.steps[-1].support_reactions['root'][0]==pytest.approx(reaction,rel=1e-6,abs=1e-10)
    previous=previous_force=0.;work=0.
    for step in result.info['force_displacement_history']:
        factor=step['load_factor'];_,force,_,_=reference(quadratic,factor)
        work+=.5*(previous_force+force)*.004*(factor-previous)
        previous,previous_force=factor,force
    assert info['prescribed_actuator_work_trapezoid']==pytest.approx(work,rel=1e-6,abs=1e-12)
    assert info['rigid_contact_work_trapezoid']==pytest.approx(-work,rel=1e-12)
    # Compare reported trapezoid work with independently calculated storage.
    assert work==pytest.approx(storage,rel=2e-4,abs=1e-12)
    manager=TranslatingHemisphereCylinderManager(model,definition,settings)
    manager.set_path_factor(1.)
    active=manager.evaluate(result.displacements)[2]
    assert manager.swept_increment_admissible(result.displacements,result.displacements,1.,0.)
    manager.set_path_factor(0.)
    opened=manager.evaluate(result.displacements)[2]
    assert opened.active_endpoints==0 and opened.line_contact_energy==0.
    manager.set_path_factor(1.)
    assert manager.evaluate(result.displacements)[2].endpoint_contact_energy==pytest.approx(active.endpoint_contact_energy)
    fresh,_,_,_=fixture(quadratic)
    prefix=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=fixture(quadratic)
    resumed=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    for body in (replace(definition.rigid_bodies[0],path_translation=None),
                 replace(definition.rigid_bodies[0],path_translation=(.004001,0.,0.))):
        with pytest.raises(ValueError,match='restart|contract|incompatible'):
            solve_static_nonlinear(fresh,load,**{**common,'contact':replace(definition,rigid_bodies=(body,))},
                restart_checkpoint=checkpoint)
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('moving_B'+('3' if quadratic else '2')+'.json').write_text(json.dumps({
        'actual_displacement_m':result.displacements[dofs].tolist(),'independent_displacement_m':u.tolist(),
        'independent_force_N':F,'storage_J':storage,'independent_work_J':work,
        'contact':info,'history':result.info['force_displacement_history']},indent=2),encoding='utf-8')


def test_curved_rim_complete_body_translation_virtual_work():
    model,definition,settings=_support_case(True)
    motion=np.array((.0015,-.0003,.0002)); factor=.73
    moving=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=tuple(motion)),))
    manager=TranslatingHemisphereCylinderManager(model,moving,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[0]]=-.0035
    u[model.mesh.dof_manager.get_node_dofs(2)[:2]]=(.1587-.212,.004)
    manager.set_path_factor(factor)
    f,K,result=manager.evaluate(u)
    assert result.endpoint_fields[0]['branch']=='rim'
    center=np.array(definition.rigid_bodies[0].point)+factor*motion
    fixed=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],point=tuple(center)),))
    gradient=np.zeros(3); h=1e-7
    for j in range(3):
        v=np.zeros(3);v[j]=h
        plus=replace(fixed,rigid_bodies=(replace(fixed.rigid_bodies[0],point=tuple(center+v)),))
        minus=replace(fixed,rigid_bodies=(replace(fixed.rigid_bodies[0],point=tuple(center-v)),))
        gradient[j]=(sum(_potential(model,plus,u))-sum(_potential(model,minus,u)))/(2*h)
    np.testing.assert_allclose(result.rigid_force,-gradient,rtol=1e-5,atol=1e-7)
    previous=manager.path_factor
    work=manager.tool_work_increment(u,u,.72,.74)
    assert manager.path_factor==previous
    forces=[]
    for t in (.72,.74):
        manager.set_path_factor(t);forces.append(np.array(manager.evaluate(u)[2].rigid_force))
    assert work==pytest.approx(.5*(forces[0]+forces[1])@(.02*motion),rel=1e-12)
    manager.set_path_factor(previous)


def test_common_motion_hidden_crossing_and_explicit_owner_contract():
    model,definition,settings,_=fixture(False,motion=(.05,-.03,.02))
    manager=TranslatingHemisphereCylinderManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs);end=u.copy()
    for node in model.mesh.nodes:end[model.mesh.dof_manager.get_node_dofs(node)[:3]]=definition.rigid_bodies[0].path_translation
    assert manager.swept_increment_admissible(u,end,0.,1.)
    manager.set_path_factor(1.)
    np.testing.assert_allclose(manager.evaluate(end)[0],0.,atol=1e-12)
    crossed=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],path_translation=(.624,0.,0.)),))
    crossing=TranslatingHemisphereCylinderManager(model,crossed,settings)
    assert not crossing.swept_increment_admissible(u,u,0.,1.)
    assert crossing.path_factor==0.
    assert validate_general_contact(model,definition,settings).profile==PROFILE
    assert not validate_general_contact(model,definition,settings,analysis_control='displacement').valid
    with pytest.raises(ValueError,match='force continuation'):
        solve_static_nonlinear(model,LoadCase('unadmitted displacement control'),
            contact=definition,contact_settings=settings,control='displacement',
            displacement_control=DisplacementControl(node_id=1,dof='ux',target_displacement=.002))
    assert not report_general_contact_capability(model,definition,settings).admitted_for_application
    with pytest.raises(ValueError,match='fixed infinite cylinder'):
        HemisphereCylinderManager(model,definition,settings)


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_tool_interruption_replays_fresh(cancel,monkeypatch):
    model,definition,settings,load=fixture()
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,tolerance=1e-10)
    token=CancellationToken();armed=[False];raised=[False]
    original=TranslatingHemisphereCylinderManager.evaluate
    def callback(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.75:
            if cancel:token.cancel('owned translating cylinder stop')
            else:armed[0]=True
    def fail(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if armed[0] and not raised[0]:
            raised[0]=True;raise RuntimeError('owned translating cylinder stop')
        return result
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(TranslatingHemisphereCylinderManager,'evaluate',fail)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='owned translating cylinder stop'):
            solve_static_nonlinear(model,load,**common,progress_callback=callback,cancellation_token=token)
    assert raised[0] or token.is_cancelled
    replay=solve_static_nonlinear(model,load,**common)
    fresh,_,_,_=fixture()
    baseline=solve_static_nonlinear(fresh,load,**common)
    assert replay.status==baseline.status=='completed'
    np.testing.assert_array_equal(replay.displacements,baseline.displacements)


def test_open_restart_to_actual_contact_and_public_example():
    model,definition,settings,load=fixture()
    common=dict(contact=definition,contact_settings=settings,num_steps=1,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    prefix=solve_static_nonlinear(model,load,**common,max_load_factor=.25)
    assert prefix.status=='completed' and prefix.info['general_contact']['active_endpoints']==0
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=fixture()
    result=solve_static_nonlinear(fresh,load,**{**common,'num_steps':3},restart_checkpoint=checkpoint)
    assert result.status=='completed',result.failure_reason
    u,_,_,_=reference(False,1.)
    axial=[fresh.mesh.dof_manager.get_node_dofs(n)[0] for n in sorted(fresh.mesh.nodes)]
    np.testing.assert_allclose(result.displacements[axial],u,rtol=1e-6,atol=1e-10)
    assert result.info['general_contact']['active_endpoints']==1
    from examples.general_contact_hemisphere_cylinder_motion import run
    example=run()
    assert example.status=='completed' and example.info['general_contact']['profile']==PROFILE
