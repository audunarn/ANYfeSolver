"""Actual cap work in multi-plane static analysis, independent scalar checks."""
from dataclasses import replace
from pathlib import Path
import json, os
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
    ContactInteraction, ContactSolverSettings, RigidContactBody, LoadCase,
    DisplacementControl, CancellationToken, SolveCancelled,
    solve_static_nonlinear, validate_general_contact)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.multiple_beam_rigid import MultipleBeamRigidManager, PROFILE_SUPPORT
from test_general_contact_hemisphere_plane import scalar_gap
from test_general_contact_multiple_beam_rigid import fixture as orthogonal_fixture
from test_general_contact_multiple_beam_ends import fixture as old_bar


def orthogonal():
    model,definition,settings=orthogonal_fixture()
    definition=replace(definition,surfaces=(replace(definition.surfaces[0],beam_end_caps='hemisphere'),),
        interactions=tuple(replace(law,endpoint_stiffness=1e4*(i+1)) for i,law in enumerate(definition.interactions)))
    return model,definition,replace(settings,beam_end_cap_work='hemisphere_support_dev')


def positions(model,u):
    return np.array([model.mesh.nodes[node].coords()+u[model.mesh.dof_manager.get_node_dofs(node)[:3]]
                     for node in (1,2,3)])


def scalar_energy(x,definition):
    total=0.;fields={}
    for law in definition.interactions:
        body=next(b for b in definition.rigid_bodies if b.name==law.target)
        n=np.array(body.normal);n/=np.linalg.norm(n);point=np.array(body.point)
        R=definition.surfaces[0].beam_radius
        def gap(s):
            p=(1-s)*(1-2*s)*x[0]+4*s*(1-s)*x[1]+s*(2*s-1)*x[2]
            return min(0.,np.dot(p-point,n)-R)
        # Reference centreline length of the original fixture is one metre.
        E=.5*law.line_stiffness*quad(lambda s:gap(s)**2,0.,1.,epsabs=1e-14,epsrel=1e-12)[0]
        E+=sum(.5*law.endpoint_stiffness*min(0.,scalar_gap(x,i,sign,R,point,n))**2
               for i,sign in ((0,-1),(2,1)))
        fields[law.name]=E;total+=E
    return total,fields


def test_simultaneous_orthogonal_caps_complete_work_tangent_body_wrenches():
    model,definition,settings=orthogonal()
    assert validate_general_contact(model,definition,settings).valid
    manager=MultipleBeamRigidManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,y,z in ((1,.0096,.0095),(2,.0097,.0097),(3,.0098,.0099)):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        u[dofs[1]]=y-.015;u[dofs[2]]=z-.015
    f,K,r=manager.evaluate(u);fields=r.to_dict();x=positions(model,u)
    assert r.profile==PROFILE_SUPPORT and fields['active_endpoints']==4
    assert fields['candidate_endpoints']==4 and len(fields['rigid_bodies'])==2
    # Simultaneous closed cap/rim switches remain the same physical cap law.
    end=u.copy()
    for node,y,z in ((1,.0098,.0099),(2,.0097,.0097),(3,.0096,.0095)):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        end[dofs[1]]=y-.015;end[dofs[2]]=z-.015
    assert manager.swept_increment_admissible(u,end,0.,1.)
    assert manager.admissible_step_fraction(u,end)==1.
    total,individual=scalar_energy(x,definition)
    assert fields['algorithmic_contact_potential']==pytest.approx(total,rel=1e-12)
    for row in fields['pairs']:
        assert row['contact']['algorithmic_contact_potential']==pytest.approx(individual[row['interaction']],rel=1e-12)
        assert row['contact']['beam_end_cap_work']=='hemisphere_support_dev'
    direction=np.random.default_rng(799).normal(size=len(u));direction/=np.linalg.norm(direction)
    for h in (1e-7,3e-8):
        independent=(scalar_energy(positions(model,u+h*direction),definition)[0]-
                     scalar_energy(positions(model,u-h*direction),definition)[0])/(2*h)
        assert independent==pytest.approx(f@direction,rel=1e-7,abs=1e-8)
        fp=manager.evaluate(u+h*direction,tangent=False)[0]
        fm=manager.evaluate(u-h*direction,tangent=False)[0]
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
    # Independent plane wrench via scalar energy derivatives. Includes cap
    # support couples about the physical skin point and shared B3 midnode arms.
    physical=np.zeros((3,3));h=1e-7
    for i in range(3):
        for j in range(3):
            p=x.copy();m=x.copy();p[i,j]+=h;m[i,j]-=h
            physical[i,j]=-(scalar_energy(p,definition)[0]-scalar_energy(m,definition)[0])/(2*h)
    np.testing.assert_allclose(fields['beam_force'],physical.sum(axis=0),rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(fields['beam_moment'],np.cross(x,physical).sum(axis=0),rtol=1e-6,atol=1e-8)
    for body in fields['rigid_bodies']:
        selected=[row['contact'] for row in fields['pairs'] if row['rigid_body']==body['name']]
        np.testing.assert_allclose(body['rigid_force'],sum(np.array(p['rigid_force']) for p in selected),atol=1e-12)
        np.testing.assert_allclose(body['rigid_moment'],sum(np.array(p['rigid_moment']) for p in selected),atol=1e-12)


def bars():
    model,definition,settings,load=old_bar()
    right=RigidContactBody('right','plane',(.324,0.,0.),(-1.,0.,0.))
    definition=replace(definition,rigid_bodies=(definition.rigid_bodies[0],right))
    return model,definition,replace(settings,beam_end_cap_work='hemisphere_support_dev'),load


def test_two_ended_force_displacement_restart_matches_independent_bar(tmp_path):
    model,definition,settings,load=bars()
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**args)
    assert result.status=='completed',result.failure_reason
    def balance(delta):
        a=.002+delta;D=.2+2*a
        line=1e4*.2*(delta**2/(2*D)-delta**3/(2*D**2)+delta**4/(6*D**3))
        return 10*a+1e5*delta+line-.2
    delta=brentq(balance,0.,1e-5,xtol=1e-18);a=.002+delta
    dofs=[model.mesh.dof_manager.get_node_dofs(node)[0] for node in (1,3)]
    error=float(np.max(np.abs(result.displacements[dofs]-[-a,a]))/a)
    assert error<1e-6
    info=result.info['general_contact'];assert info['active_endpoints']==2 and info['gap_tolerance_met']
    assert all(row['endpoint_stiffness_N_per_m']==1e5 for row in info['pair_enforcement'])
    model,_,_,_=bars()
    controlled=solve_static_nonlinear(model,load,**args,control='displacement',
        displacement_control=DisplacementControl(node_id=1,dof='ux',target_displacement=-a))
    assert controlled.status=='completed',controlled.failure_reason
    np.testing.assert_allclose(controlled.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    model,_,_,_=bars()
    prefix=solve_static_nonlinear(model,load,**{**args,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed'
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=bars();resumed=solve_static_nonlinear(fresh,load,**{**args,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed'
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    for altered in (replace(settings,beam_end_cap_work='owned_point'),replace(settings,maximum_swept_subdivisions=17)):
        fresh,_,_,_=bars()
        with pytest.raises(ValueError,match='restart|contract|incompatible'):
            solve_static_nonlinear(fresh,load,**{**args,'contact_settings':altered},restart_checkpoint=checkpoint)
    output=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'result.json'))).with_name('two_plane_bar.json')
    output.write_text(json.dumps({'relative_error':error,'reference_displacements_m':[-a,a],
        'actual_displacements_m':result.displacements[dofs].tolist(),'reference_delta_m':delta,
        'registered_development_target':1e-6,'gap_budget_m':1e-5},indent=2),encoding='utf-8')


def test_one_child_gap_gate_common_fraction_and_duplicate_resource_rejection():
    model,definition,settings=orthogonal();manager=MultipleBeamRigidManager(model,definition,settings)
    start=np.zeros(model.mesh.dof_manager.total_dofs);end=start.copy()
    end[model.mesh.dof_manager.get_node_dofs(2)[1]]=-.0055
    end[model.mesh.dof_manager.get_node_dofs(2)[2]]=-.009
    assert not manager.swept_increment_admissible(start,end,0.,1.)
    fraction=manager.admissible_step_fraction(start,end)
    assert 0<fraction<1 and manager.swept_increment_admissible(start,start+fraction*(end-start),0.,fraction)
    duplicate=replace(definition,interactions=definition.interactions+(replace(definition.interactions[0],name='duplicate'),))
    assert not validate_general_contact(model,duplicate,settings).valid
    assert not validate_general_contact(model,definition,replace(settings,maximum_candidate_pairs=1)).valid
    from anyfesolver.general_contact.multiple_beam_rigid import PROFILE_SUPPORT_CYLINDER
    cylinder=replace(definition.rigid_bodies[0],kind='cylinder',normal=(0.,0.,1.),radius=.01)
    cylinder_definition=replace(definition,rigid_bodies=(cylinder,definition.rigid_bodies[1]))
    assert validate_general_contact(model,cylinder_definition,settings).profile==PROFILE_SUPPORT_CYLINDER
    assert validate_general_contact(model,cylinder_definition,settings).valid
    moving=replace(definition,rigid_bodies=(replace(cylinder,path_translation=(.001,0.,0.)),definition.rigid_bodies[1]))
    assert not validate_general_contact(model,moving,settings).valid
    for altered in (replace(settings,beam_chain_tangent='consistent_dev'),
                    replace(settings,beam_shell_geometry='jet2_deformable_s3_dev'),
                    replace(settings,augmentation_acceleration='active_secant')):
        assert not validate_general_contact(model,definition,altered).valid
    assert not validate_general_contact(model,definition,settings,analysis_control='arc').valid


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_aggregate_second_child_failure_or_cancel_restores_static_origin(cancel,monkeypatch):
    from anyfesolver.general_contact.beam_hemisphere_plane import HemispherePlaneManager
    model,definition,settings,load=bars()
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False];original=HemispherePlaneManager.evaluate
    def callback(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.5:
            if cancel:token.cancel('multiple hemisphere interruption')
            else:armed[0]=True
    def failing(self,*args,**kwargs):
        candidate=original(self,*args,**kwargs)
        if self.definition.rigid_bodies[0].name=='right' and armed[0] and not raised[0]:
            raised[0]=True;raise RuntimeError('multiple hemisphere interruption')
        return candidate
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(HemispherePlaneManager,'evaluate',failing)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='multiple hemisphere interruption'):
            solve_static_nonlinear(model,load,**common,progress_callback=callback,cancellation_token=token)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**common)
    fresh,_,_,fresh_load=bars();reference=solve_static_nonlinear(fresh,fresh_load,**common)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_true_junction_caps_and_shared_body_reactions():
    model,definition,settings,load=bars()
    model.add_node(20,.112,.1,0.)
    model.add_element(20,BeamElement(20,(1,20),'elastic',{'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    manager=MultipleBeamRigidManager(model,definition,settings)
    assert all(1 not in child.cap_endpoints for _,child in manager.pairs)
    # Two selections against one body: exact feature ownership, not duplicate
    # rigid bodies or coordinate-derived topology. Unselected joint is omitted.
    model,definition,settings,load=bars()
    for node,x in enumerate((.112,.212,.312),4):model.add_node(node,x,.05,0.)
    model.add_element(2,QuadraticBeamElement(2,(4,5,6),'elastic',{'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    surface=replace(definition.surfaces[0],name='other',element_ids=(2,))
    first=definition.interactions[0]
    definition=ContactDefinition((definition.surfaces[0],surface),(definition.rigid_bodies[0],),
        (first,replace(first,name='other pair',surface='other')))
    manager=MultipleBeamRigidManager(model,definition,settings);u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1,4):u[model.mesh.dof_manager.get_node_dofs(node)[0]]=-.0022
    fields=manager.evaluate(u)[2].to_dict()
    assert fields['active_endpoints']==2 and len(fields['rigid_bodies'])==1
    np.testing.assert_allclose(fields['rigid_bodies'][0]['rigid_force'],fields['rigid_force'],atol=1e-12)
    np.testing.assert_allclose(fields['rigid_bodies'][0]['rigid_moment'],fields['rigid_moment'],atol=1e-12)


def test_runnable_multi_plane_example_and_capability_remains_unaccepted():
    from examples.general_contact_multiple_hemisphere_planes import run
    from anyfesolver import report_general_contact_capability
    result=run();assert result.status=='completed'
    assert result.info['general_contact']['profile']==PROFILE_SUPPORT
    model,definition,settings,load=bars()
    report=report_general_contact_capability(model,definition,settings)
    assert report.implemented and not report.admitted_for_application
