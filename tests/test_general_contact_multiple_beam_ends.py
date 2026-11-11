"""Explicit physical-end composition with an independent two-ended bar."""
from dataclasses import replace
import json,os
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import brentq
from anyfesolver import (ContactDefinition,ContactSurface,ContactInteraction,ContactSolverSettings,
    RigidContactBody,RigidCappedCylinder,BoundaryCondition,LoadCase,DisplacementControl,
    solve_static_nonlinear,validate_general_contact,report_general_contact_capability)
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.general_contact.multiple_beam_rigid import MultipleBeamRigidManager,PROFILE_ENDS


def fixture(kind='sphere'):
    model=FEModel('one guided B3 with two physical end contacts')
    model.add_material('elastic',1e3,.3,density=1.)
    for node,x in enumerate((.112,.212,.312),1):model.add_node(node,x,0.,0.)
    model.add_element(1,QuadraticBeamElement(1,(1,2,3),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    model.add_boundary_condition(BoundaryCondition('midpoint',(2,),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('end guides',(1,3),
        {d:0. for d in ('uy','uz','rx','ry','rz')}))
    right=(RigidCappedCylinder('right',(.424,0.,0.),(1.,0.,0.),.01,.1) if kind=='capped_cylinder'
        else RigidContactBody('right',kind,(.424,0.,0.),radius=.1,
            normal=(0.,0.,1.) if kind=='cylinder' else None))
    definition=ContactDefinition((ContactSurface('tube',(1,),beam_radius=.01,beam_end_caps='hemisphere'),),
        (RigidContactBody('left','plane',(.1,0.,0.),(1.,0.,0.)),right),
        tuple(ContactInteraction(name,'tube',target,None,1e-5,line_stiffness=1e4,endpoint_stiffness=1e5)
            for name,target in (('left pair','left'),('right pair','right'))))
    settings=ContactSolverSettings(pair_motion='general',line_integration_relative_tolerance=1e-7,
        maximum_line_subdivisions=14,maximum_swept_subdivisions=16)
    load=LoadCase('outward pressure');load.add_nodal_load(1,forces=(-.2,0.,0.))
    load.add_nodal_load(3,forces=(.2,0.,0.))
    return model,definition,settings,load


@pytest.mark.parametrize('kind',('sphere','cylinder','capped_cylinder'))
def test_complete_sum_virtual_work_tangent_and_per_body_wrench(kind):
    model,definition,settings,_=fixture(kind)
    manager=MultipleBeamRigidManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    u[model.mesh.dof_manager.get_node_dofs(1)[0]]=-.0022
    u[model.mesh.dof_manager.get_node_dofs(3)[0]]=.0022
    f,K,r=manager.evaluate(u);fields=r.to_dict()
    assert r.profile==PROFILE_ENDS and fields['active_endpoints']==2
    assert fields['candidate_endpoints']==4
    individual=[child.evaluate(u) for _,child in manager.pairs]
    np.testing.assert_allclose(f,sum(p[0] for p in individual),atol=1e-12)
    np.testing.assert_allclose(K.toarray(),sum(p[1].toarray() for p in individual),atol=1e-12)
    assert fields['algorithmic_contact_potential']==pytest.approx(
        fields['line_contact_energy']+fields['endpoint_contact_energy'],rel=1e-13)
    assert [p['rigid_body'] for p in fields['pairs']]==['left','right']
    for i,body in enumerate(fields['rigid_bodies']):
        np.testing.assert_allclose(body['rigid_force'],individual[i][2].rigid_force,atol=1e-12)
    np.testing.assert_allclose(np.array(fields['beam_force'])+fields['rigid_force'],0.,atol=1e-12)
    np.testing.assert_allclose(np.array(fields['beam_moment'])+fields['rigid_moment'],0.,atol=1e-12)
    direction=np.random.default_rng(409).normal(size=len(u))*.07
    h=3e-7
    fp,_,rp=manager.evaluate(u+h*direction,tangent=False)
    fm,_,rm=manager.evaluate(u-h*direction,tangent=False)
    assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<=1e-5
    assert (rp.to_dict()['algorithmic_contact_potential']-rm.to_dict()['algorithmic_contact_potential'])/(2*h)==pytest.approx(
        f@direction,rel=1e-5,abs=1e-9)


@pytest.mark.parametrize('kind',('sphere','cylinder','capped_cylinder'))
def test_actual_two_ended_bar_force_displacement_and_active_restart(kind):
    model,definition,settings,load=fixture(kind)
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
        tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**args)
    assert result.status=='completed',result.failure_reason
    # Actual B3 end shape N1=1-3s+2s²; symmetric nodal motion [-a,0,a].
    def balance(delta):
        a=.002+delta;D=.2+2*a
        line=1e4*.2*(delta**2/(2*D)-delta**3/(2*D**2)+delta**4/(6*D**3))
        return 2/.2*a+1e5*delta+line-.2
    delta=brentq(balance,0.,1e-5,xtol=1e-18);a=.002+delta
    dofs=[model.mesh.dof_manager.get_node_dofs(node)[0] for node in (1,3)]
    np.testing.assert_allclose(result.displacements[dofs],(-a,a),rtol=1e-6,atol=1e-10)
    info=result.info['general_contact'];assert info['active_endpoints']==2 and info['gap_tolerance_met']
    assert all(p['endpoint_stiffness_N_per_m']==1e5 for p in info['pair_enforcement'])
    model,_,_,_=fixture(kind)
    prefix=solve_static_nonlinear(model,load,**{**args,'num_steps':2},max_load_factor=.75)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=fixture(kind)
    resumed=solve_static_nonlinear(fresh,load,**{**args,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    changed=replace(definition,interactions=(replace(definition.interactions[0],endpoint_stiffness=1.01e5),
        definition.interactions[1]))
    with pytest.raises(ValueError,match='restart|contract|incompatible'):
        solve_static_nonlinear(fresh,load,**{**args,'contact':changed},restart_checkpoint=checkpoint)
    control_model,_,_,_=fixture(kind)
    control=solve_static_nonlinear(control_model,load,**args,control='displacement',
        displacement_control=DisplacementControl(node_id=1,dof='ux',target_displacement=-.002005))
    assert control.status=='completed',control.failure_reason
    assert control.info['general_contact']['active_endpoints']==2
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('bar_'+kind+'.json').write_text(json.dumps({
        'independent_end_displacements_m':[-a,a],'solved_end_displacements_m':result.displacements[dofs].tolist(),
        'contact':info},indent=2))


@pytest.mark.parametrize('bad',('duplicate','missing_endpoint_law','moving','candidate_limit'))
def test_extended_aggregate_stays_fail_closed(bad):
    model,definition,settings,_=fixture()
    if bad=='duplicate':definition=replace(definition,interactions=definition.interactions+(
        replace(definition.interactions[0],name='alias'),))
    elif bad=='missing_endpoint_law':definition=replace(definition,interactions=(
        replace(definition.interactions[0],endpoint_stiffness=None),definition.interactions[1]))
    elif bad=='moving':definition=replace(definition,rigid_bodies=(definition.rigid_bodies[0],
        replace(definition.rigid_bodies[1],path_translation=(.001,0.,0.))))
    else:settings=replace(settings,maximum_candidate_pairs=1)
    assert not validate_general_contact(model,definition,settings).valid
    with pytest.raises((ValueError,MemoryError)):MultipleBeamRigidManager(model,definition,settings)


def test_common_trial_rejects_one_failed_pair_and_keeps_public_admission_false():
    model,definition,settings,_=fixture()
    manager=MultipleBeamRigidManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs);bad=u.copy()
    bad[model.mesh.dof_manager.get_node_dofs(1)[0]]=-.004
    assert not manager.swept_increment_admissible(u,bad,0.,1.)
    assert manager.admissible_step_fraction(u,bad)<1.
    np.testing.assert_array_equal(manager.evaluate(u)[0],0.)
    capability=report_general_contact_capability(model,definition,settings)
    assert capability.implemented and not capability.admitted_for_application
    assert not capability.independently_accepted and not capability.released


def test_shared_obstacle_disconnected_beams_sum_one_body_wrench():
    model,definition,settings,_=fixture()
    for node,x in enumerate((.112,.212,.312),4):model.add_node(node,x,.05,0.)
    model.add_element(2,QuadraticBeamElement(2,(4,5,6),'elastic',
        {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    second=replace(definition.surfaces[0],name='second tube',element_ids=(2,))
    definition=replace(definition,surfaces=(definition.surfaces[0],second),
        rigid_bodies=(definition.rigid_bodies[0],),interactions=(definition.interactions[0],
            replace(definition.interactions[0],name='second pair',surface=second.name,endpoint_stiffness=2e5)))
    manager=MultipleBeamRigidManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (1,4):u[model.mesh.dof_manager.get_node_dofs(node)[0]]=-.0022
    result=manager.evaluate(u)[2].to_dict()
    assert result['active_endpoints']==2 and result['candidate_endpoints']==4
    assert len(result['rigid_bodies'])==1
    np.testing.assert_allclose(result['rigid_bodies'][0]['rigid_force'],result['rigid_force'],atol=1e-12)
    np.testing.assert_allclose(result['rigid_bodies'][0]['rigid_moment'],result['rigid_moment'],atol=1e-12)
    assert result['endpoint_contact_energy']==pytest.approx(.5*3e5*.0002**2,rel=1e-11)
    # All forces are axial. Only the second beam's y=.05 produces z moment.
    second_force=np.array(result['pairs'][1]['contact']['rigid_force'])
    assert result['rigid_moment'][2]==pytest.approx(-.05*second_force[0],rel=1e-12)


def test_second_child_exception_cannot_leave_partial_aggregate_work(monkeypatch):
    model,definition,settings,_=fixture()
    manager=MultipleBeamRigidManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    def fail(*args,**kwargs):raise RuntimeError('owned second contributor failure')
    with monkeypatch.context() as patch:
        patch.setattr(manager.pairs[1][1],'evaluate',fail)
        with pytest.raises(RuntimeError,match='owned second contributor failure'):manager.evaluate(u)
    f,K,result=manager.evaluate(u)
    np.testing.assert_array_equal(f,0.)
    assert result.to_dict()['algorithmic_contact_potential']==0.


def test_extended_composition_rejects_hidden_material_collapse_and_limits_trial():
    model,definition,settings,_=fixture()
    manager=MultipleBeamRigidManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs);reversed_pose=u.copy()
    reversed_pose[model.mesh.dof_manager.get_node_dofs(1)[0]]=.2
    reversed_pose[model.mesh.dof_manager.get_node_dofs(3)[0]]=-.2
    assert not manager.swept_increment_admissible(u,reversed_pose,0.,1.)
    previous=manager.swept_trial_cutbacks
    fraction=manager.admissible_step_fraction(u,reversed_pose)
    assert 0.<fraction<.5
    assert manager.swept_trial_cutbacks>previous
    with pytest.raises(ValueError,match='singular|unresolved'):
        manager.evaluate(.5*reversed_pose)


def test_public_multiple_physical_end_example():
    from examples.general_contact_multiple_beam_ends import run
    result=run()
    assert result.info['general_contact']['profile']==PROFILE_ENDS
    assert result.info['general_contact']['active_endpoints']==2
