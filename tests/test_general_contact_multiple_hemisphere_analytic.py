"""Complete mixed-plane/sphere support work and coupled static transactions."""
from dataclasses import replace
import json,os
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from anyfesolver import (ContactDefinition, ContactSurface, RigidContactBody,
    LoadCase, DisplacementControl, CancellationToken, SolveCancelled,
    solve_static_nonlinear, validate_general_contact, report_general_contact_capability)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.general_contact.multiple_beam_rigid import (MultipleBeamRigidManager,
    PROFILE_SUPPORT, PROFILE_SUPPORT_ANALYTIC)
from anyfesolver.general_contact.beam_hemisphere_sphere import HemisphereSphereManager
from anyfesolver.general_contact.beam_hemisphere_plane import HemispherePlaneManager
from test_general_contact_multiple_hemisphere_planes import orthogonal, positions, bars as plane_bars
from test_general_contact_hemisphere_plane import scalar_gap
from test_general_contact_hemisphere_sphere import angular_oracle
from test_general_contact_multiple_beam_ends import fixture as old_bar


def mixed():
    model,definition,settings=orthogonal()
    sphere=RigidContactBody('y_tool','sphere',(-.5,-.097,.015),radius=.1)
    definition=replace(definition,rigid_bodies=(sphere,definition.rigid_bodies[1]))
    return model,definition,settings


def scalar_energy(x,definition):
    total=0.;individual={};R=definition.surfaces[0].beam_radius
    for law in definition.interactions:
        body=next(b for b in definition.rigid_bodies if b.name==law.target);O=np.array(body.point)
        def line_gap(t):
            p=(1-t)*(1-2*t)*x[0]+4*t*(1-t)*x[1]+t*(2*t-1)*x[2]
            return (np.linalg.norm(p-O)-R-body.radius if body.kind=='sphere'
                else (p-O)@np.array(body.normal)-R)
        # This fixture has one left sphere front and an all-active plane.
        upper=brentq(line_gap,0.,1.,xtol=1e-15) if body.kind=='sphere' else 1.
        E=.5*law.line_stiffness*quad(lambda t:min(0.,line_gap(t))**2,0.,upper,
            epsabs=1e-15,epsrel=1e-12)[0]  # one-metre reference length
        for end,sign in ((0,-1),(2,1)):
            g=(angular_oracle(x,end,sign,R,O,body.radius) if body.kind=='sphere'
                else scalar_gap(x,end,sign,R,O,np.array(body.normal)))
            E+=.5*law.endpoint_stiffness*min(0.,g)**2
        individual[law.name]=E;total+=E
    return total,individual


def test_shared_b3_rim_work_cross_node_tangent_and_independent_body_wrenches():
    model,definition,settings=mixed();assert validate_general_contact(model,definition,settings).valid
    manager=MultipleBeamRigidManager(model,definition,settings)
    assert manager.profile==PROFILE_SUPPORT_ANALYTIC
    assert isinstance(manager.pairs[0][1],HemisphereSphereManager)
    assert isinstance(manager.pairs[1][1],HemispherePlaneManager)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,y,z in ((1,.0128,.0096),(2,.0126,.0097),(3,.0127,.0099)):
        d=model.mesh.dof_manager.get_node_dofs(node);u[d[1]]=y-.015;u[d[2]]=z-.015
    force,K,result=manager.evaluate(u);fields=result.to_dict();x=positions(model,u)
    assert fields['active_endpoints']==3 and fields['candidate_endpoints']==4
    assert fields['pairs'][0]['contact']['endpoint_fields'][0]['branch']=='rim'
    independent,parts=scalar_energy(x,definition)
    assert fields['algorithmic_contact_potential']==pytest.approx(independent,rel=1e-11)
    for row in fields['pairs']:
        assert row['contact']['algorithmic_contact_potential']==pytest.approx(parts[row['interaction']],rel=1e-11)
        assert row['contact']['beam_end_cap_work']=='hemisphere_support_dev'
    direction=np.random.default_rng(900).normal(size=len(u));direction/=np.linalg.norm(direction)
    for h in (1e-7,3e-8):
        ep=scalar_energy(positions(model,u+h*direction),definition)[0]
        em=scalar_energy(positions(model,u-h*direction),definition)[0]
        assert (ep-em)/(2*h)==pytest.approx(force@direction,rel=1e-7,abs=1e-8)
        fp=manager.evaluate(u+h*direction,tangent=False)[0];fm=manager.evaluate(u-h*direction,tangent=False)[0]
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
    h=1e-7
    for body in fields['rigid_bodies']:
        name=next(law.name for law in definition.interactions if law.target==body['name'])
        physical=np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                p=x.copy();m=x.copy();p[i,j]+=h;m[i,j]-=h
                physical[i,j]=-(scalar_energy(p,definition)[1][name]-scalar_energy(m,definition)[1][name])/(2*h)
        np.testing.assert_allclose(body['rigid_force'],-physical.sum(axis=0),rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(body['rigid_moment'],-np.cross(x,physical).sum(axis=0),rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(np.array(fields['beam_force'])+fields['rigid_force'],0.,atol=1e-12)
    np.testing.assert_allclose(np.array(fields['beam_moment'])+fields['rigid_moment'],0.,atol=1e-12)


def bars(two_spheres=False):
    model,definition,settings,load=old_bar('sphere')
    if two_spheres:
        definition=replace(definition,rigid_bodies=(RigidContactBody('left','sphere',(0.,0.,0.),radius=.1),definition.rigid_bodies[1]))
    return model,definition,replace(settings,beam_end_cap_work='hemisphere_support_dev'),load


@pytest.mark.parametrize('two_spheres',(False,True))
def test_actual_simultaneous_bar_controls_and_restart_match_independent_balance(two_spheres):
    model,definition,settings,load=bars(two_spheres)
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**common);assert result.status=='completed',result.failure_reason
    def balance(delta):
        a=.002+delta;D=.2+2*a
        return 10*a+1e5*delta+1e4*.2*(delta**2/(2*D)-delta**3/(2*D**2)+delta**4/(6*D**3))-.2
    delta=brentq(balance,0.,1e-5,xtol=1e-18);a=.002+delta
    dofs=[model.mesh.dof_manager.get_node_dofs(n)[0] for n in (1,3)]
    error=float(np.max(np.abs(result.displacements[dofs]-[-a,a]))/a);assert error<1e-6
    fields=result.info['general_contact'];assert fields['profile']==PROFILE_SUPPORT_ANALYTIC
    assert fields['gap_tolerance_met'] and fields['active_endpoints']==2
    assert all(row['endpoint_stiffness_N_per_m']==1e5 for row in fields['pair_enforcement'])
    fresh,_,_,_=bars(two_spheres)
    controlled=solve_static_nonlinear(fresh,load,**common,control='displacement',
        displacement_control=DisplacementControl(node_id=1,dof='ux',target_displacement=-a))
    assert controlled.status=='completed',controlled.failure_reason
    np.testing.assert_allclose(controlled.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    fresh,_,_,_=bars(two_spheres)
    prefix=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed'
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=bars(two_spheres)
    resumed=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed'
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    for altered in (replace(settings,beam_end_cap_work='owned_point'),replace(settings,maximum_swept_subdivisions=17)):
        with pytest.raises(ValueError,match='restart|contract|incompatible'):
            solve_static_nonlinear(fresh,load,**{**common,'contact_settings':altered},restart_checkpoint=checkpoint)
    changed=replace(definition,rigid_bodies=(definition.rigid_bodies[0],replace(definition.rigid_bodies[1],radius=.100001)))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**common,'contact':changed},restart_checkpoint=checkpoint)
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('bar_'+('spheres' if two_spheres else 'mixed')+'.json').write_text(json.dumps({
        'relative_error':error,'actual_m':result.displacements[dofs].tolist(),'reference_m':[-a,a],
        'gap_tolerance_m':1e-5,'contact':fields},indent=2),encoding='utf-8')


@pytest.mark.parametrize('cancel',(False,True))
def test_second_sphere_child_failure_or_cancel_restores_actual_solver(cancel,monkeypatch):
    model,definition,settings,load=bars();common=dict(contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=20,tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False];original=HemisphereSphereManager.evaluate
    def callback(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.75:
            if cancel:token.cancel('mixed child interruption')
            else:armed[0]=True
    def failing(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if armed[0] and not raised[0]:
            raised[0]=True;raise RuntimeError('mixed child interruption')
        return result
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(HemisphereSphereManager,'evaluate',failing)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='mixed child interruption'):
            solve_static_nonlinear(model,load,**common,cancellation_token=token,progress_callback=callback)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**common);fresh,_,_,_=bars()
    reference=solve_static_nonlinear(fresh,load,**common)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_common_gap_fraction_and_admission_limits_preserve_old_plane_identity():
    model,definition,settings,_=bars();manager=MultipleBeamRigidManager(model,definition,settings)
    start=np.zeros(model.mesh.dof_manager.total_dofs);end=start.copy()
    end[model.mesh.dof_manager.get_node_dofs(3)[0]]=.003
    assert not manager.swept_increment_admissible(start,end,0.,1.)
    fraction=manager.admissible_step_fraction(start,end)
    assert 0<fraction<1 and manager.swept_increment_admissible(start,fraction*end,0.,1.)
    duplicate=replace(definition,interactions=(*definition.interactions,replace(definition.interactions[0],name='alias')))
    assert not validate_general_contact(model,duplicate,settings).valid
    assert not validate_general_contact(model,definition,replace(settings,maximum_candidate_pairs=1)).valid
    from anyfesolver.general_contact.multiple_beam_rigid import PROFILE_SUPPORT_CYLINDER
    cylinder=replace(definition.rigid_bodies[1],kind='cylinder',normal=(0.,0.,1.))
    cylinder_definition=replace(definition,rigid_bodies=(definition.rigid_bodies[0],cylinder))
    assert validate_general_contact(model,cylinder_definition,settings).profile==PROFILE_SUPPORT_CYLINDER
    assert validate_general_contact(model,cylinder_definition,settings).valid
    for body in (replace(definition.rigid_bodies[1],path_translation=(.001,0.,0.)),
                 replace(cylinder,path_translation=(.001,0.,0.))):
        assert not validate_general_contact(model,replace(definition,rigid_bodies=(definition.rigid_bodies[0],body)),settings).valid
    assert not validate_general_contact(model,definition,settings,analysis_control='arc').valid
    old_model,old_definition,old_settings,_=plane_bars()
    assert MultipleBeamRigidManager(old_model,old_definition,old_settings).profile==PROFILE_SUPPORT
    assert report_general_contact_capability(model,definition,settings).admitted_for_application is False


def test_shared_sphere_groups_two_beams_and_omits_true_junction_cap():
    model,definition,settings=mixed()
    for node,p in enumerate(((-.5,.015,.016),(0.,.015,.016),(.5,.015,.016)),4):model.add_node(node,*p)
    model.add_element(2,QuadraticBeamElement(2,(4,5,6),'elastic',{'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    surface=replace(definition.surfaces[0],name='other',element_ids=(2,));law=definition.interactions[0]
    definition=ContactDefinition((definition.surfaces[0],surface),(definition.rigid_bodies[0],),(
        law,replace(law,name='other pair',surface='other')))
    manager=MultipleBeamRigidManager(model,definition,settings);u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1,2,3,4,5,6):u[model.mesh.dof_manager.get_node_dofs(node)[1]]=-.0022
    fields=manager.evaluate(u)[2].to_dict()
    assert fields['active_endpoints']==2 and len(fields['rigid_bodies'])==1
    np.testing.assert_allclose(fields['rigid_bodies'][0]['rigid_force'],fields['rigid_force'],atol=1e-12)
    np.testing.assert_allclose(fields['rigid_bodies'][0]['rigid_moment'],fields['rigid_moment'],atol=1e-12)
    # Attach an unselected member with original endpoint 1 as its material
    # midnode. It becomes a genuine junction, not a cap or search adjacency.
    model.add_node(7,-.5,-.085,.015);model.add_node(8,-.5,.115,.015)
    model.add_element(3,QuadraticBeamElement(3,(7,1,8),'elastic',
        {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    revised=MultipleBeamRigidManager(model,definition,settings)
    original=next(child for name,child in revised.pairs if name==law.name)
    other=next(child for name,child in revised.pairs if name!=law.name)
    assert set(original.endpoints)=={3} and set(other.endpoints)=={4,6}


def test_runnable_mixed_bar_example():
    from examples.general_contact_multiple_hemisphere_analytic import run
    result=run();assert result.status=='completed'
    assert result.info['general_contact']['profile']==PROFILE_SUPPORT_ANALYTIC
