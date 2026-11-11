import json
import os
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from examples.general_contact_triangulated_rigid import case, analysis
from anyfesolver import (ContactPathProgram,ContactPathControl,LoadCase,solve_static_nonlinear,validate_general_contact)
from anyfesolver.general_contact import TriangulatedRigidContactBody
from anyfesolver.general_contact.triangulated_rigid_static import TriangulatedRigidManager


def test_complete_penalty_work_and_diagonal():
    model,definition,settings=case()
    manager=TriangulatedRigidManager(model,definition,settings)
    source,target=manager._positions(np.zeros(manager.n_dof))
    source[:,2]=-.00002
    a=manager.law.work(source,target)
    xy=source[:,:2]
    area=abs(np.linalg.det(np.array([xy[1]-xy[0],xy[2]-xy[0]])))/2
    expected=.5*1e6*area*(2e-5)**2
    assert abs(a.state.energy-expected)<expected*1e-12
    physical=-a.residual.reshape(-1,3)
    np.testing.assert_allclose(physical[:3].sum(0),(0,0,1e6*area*2e-5),atol=1e-10)
    assert np.linalg.norm(physical.sum(0))/(1e6*area*2e-5)<1e-6
    assert np.linalg.norm(np.cross(np.vstack((source,target)),physical).sum(0))/(1e6*area*2e-5)<1e-6
    mass=area/12*(np.eye(3)+np.ones((3,3)))
    shape=np.linalg.inv(np.column_stack((np.ones(3),xy)))[1:,:]
    normal_h=1e6*mass.T@np.linalg.solve(area/3*np.eye(3),mass)+1e6*area*(2e-5)**2*(shape.T@shape)
    np.testing.assert_allclose(a.tangent[np.ix_((2,5,8),(2,5,8))],normal_h,rtol=1e-12,atol=1e-9)
    centroid=source.mean(0)
    np.testing.assert_allclose(np.cross(source,physical[:3]).sum(0),np.cross(centroid,physical[:3].sum(0)),atol=1e-10)
    direction=np.zeros_like(a.residual);direction[2]=.3;direction[5]=-.2;direction[8]=.4
    h=1e-7
    def at(sign):
        p=np.r_[source.reshape(-1),target.reshape(-1)]+sign*h*direction
        return manager.law.work(p[:9].reshape(3,3),p[9:].reshape(-1,3))
    plus,minus=at(1),at(-1)
    assert abs((plus.state.energy-minus.state.energy)/(2*h)-a.residual@direction)/max(1.,abs(a.residual@direction))<1e-5
    assert np.linalg.norm((plus.residual-minus.residual)/(2*h)-a.tangent@direction)/max(1.,np.linalg.norm(a.tangent@direction))<1e-5
    second=TriangulatedRigidManager(*case(1))
    b=second.law.work(source,target)
    np.testing.assert_allclose(b.residual[:9],a.residual[:9],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(b.tangent[:9,:9],a.tangent[:9,:9],rtol=1e-10,atol=1e-8)
    assert abs(b.state.energy-a.state.energy)<expected*1e-12


def test_finite_footprint_refusal_and_crossing():
    model,definition,settings=case();manager=TriangulatedRigidManager(model,definition,settings)
    u=np.zeros(manager.n_dof)
    for node in (1,2,3):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        u[dofs[0]]=-2.;u[dofs[2]]=-.001
    assert manager.evaluate(u)[2].penalty_energy==0
    assert manager._gap_lower(manager._positions(u))==float('inf')
    v=u.copy()
    for node in (1,2,3):
        v[model.mesh.dof_manager.get_node_dofs(node)[0]]+=4.
    assert not manager.swept_increment_admissible(u,v,0.,1.)
    inside=u.copy()
    for node in (1,2,3): inside[model.mesh.dof_manager.get_node_dofs(node)[0]]=0.
    with pytest.raises(ValueError,match='overclosure'):
        TriangulatedRigidManager(model,definition,settings,inside)
    with pytest.raises(ValueError,match='duplicate'):
        TriangulatedRigidContactBody('bad',definition.rigid_bodies[0].vertices,((1,(0,1,2)),(2,(2,1,0))))
    changed=replace(definition.rigid_bodies[0],vertices=((0.,0.,0.),(1.,0.,0.),(1.,1.,.1),(0.,1.,0.)))
    assert not validate_general_contact(model,replace(definition,rigid_bodies=(changed,)),settings).valid
    manager.settings=replace(settings,general_s3_difference_step=2e-6)
    with pytest.raises(ValueError,match='mutated'): manager.evaluate(np.zeros(manager.n_dof))


def test_tilted_actual_path_and_normal_refusal():
    from anyfesolver.general_contact.s3_pair_static import _swept_gap_lower
    model,definition,settings=case();manager=TriangulatedRigidManager(model,definition,settings)
    first=np.zeros(manager.n_dof);last=first.copy()
    last[model.mesh.dof_manager.get_node_dofs(2)[2]]=-4e-5
    a,b=manager._positions(first),manager._positions(last)
    old=min(_swept_gap_lower(a[0][si],b[0][si],a[1][ti],b[1][ti]) for si in manager.law.source_face_indices for ti in manager.law.target_face_indices)
    assert old < -manager.gap_tolerance
    assert manager.swept_increment_admissible(first,last,0.,1.)
    inverted=first.copy()
    dofs=model.mesh.dof_manager.get_node_dofs(2)
    inverted[dofs[0]]=-1.3
    assert not manager.swept_increment_admissible(first,inverted,0.,1.)


def test_ordinary_force_and_path_restart(monkeypatch):
    import importlib
    solver=importlib.import_module('anyfesolver.nonlinear_static')
    original=solver._support_reaction_resultants
    captured={}
    def reaction(model,imbalance,**kwargs):
        captured.update(model=model,residual=np.array(imbalance,copy=True))
        return original(model,imbalance,**kwargs)
    monkeypatch.setattr(solver,'_support_reaction_resultants',reaction)
    full=analysis()
    assert full.status=='completed',full.failure_reason
    result=full.info['general_contact']
    assert np.linalg.norm(np.array(result['integration_force'])+np.array(result['target_force']))/.1<1e-6
    assert np.linalg.norm(np.array(result['integration_moment'])+np.array(result['target_moment']))/.1<1e-6
    assert result['penalty_energy']>0 and result['gap_tolerance_met']
    force=np.zeros(3);moment=np.zeros(3)
    for node in (1,2,3):
        model_=captured['model'];ids=model_.mesh.dof_manager.get_node_dofs(node)
        xyz=np.array((model_.mesh.nodes[node].x,model_.mesh.nodes[node].y,model_.mesh.nodes[node].z))+full.displacements[ids[:3]]
        r=captured['residual'][ids].copy()
        if node!=1: r[2]=0.  # These are free equilibrium coordinates, not supports.
        force+=r[:3];moment+=r[3:6]+np.cross(xyz,r[:3])
        if node!=1:
            load=np.array((0.,0.,-.05));force+=load;moment+=np.cross(xyz,load)
    force+=np.array(result['integration_force']);moment+=np.array(result['integration_moment'])
    balance={'force_n':force.tolist(),'moment_nm':moment.tolist(),'force_error':float(np.linalg.norm(force)/.1),'moment_error':float(np.linalg.norm(moment)/.1)}
    assert balance['force_error']<1e-6 and balance['moment_error']<1e-6
    monkeypatch.setattr(solver,'_support_reaction_resultants',original)
    model,definition,settings=case()
    program=ContactPathProgram(controls=(ContactPathControl.node(2,'ux'),ContactPathControl.node(2,'uz')),
        waypoints=((0.,0.),(0.,-.00004),(0.,0.),(0.,-.00004)),steps_per_segment=1)
    # Remove the controlled ux from the guide, preserving all remaining constraints.
    from anyfesolver import BoundaryCondition
    model.boundary_conditions[0]=BoundaryCondition('guide',(1,2,3),{k:0. for k in ('uy','rx','ry','rz')})
    model.add_boundary_condition(BoundaryCondition('xguide',(1,3),{'ux':0.}))
    common=dict(contact=definition,contact_settings=settings,contact_path_program=program,max_iterations=15)
    path=solve_static_nonlinear(model,emit_restart_checkpoint=True,**common)
    assert path.status=='completed',path.failure_reason
    phases=path.info['force_displacement_history']
    assert [r['contact']['active_pressure_dofs']>0 for r in phases]==[True,False,True]
    half=solve_static_nonlinear(model,max_load_factor=1/3,emit_restart_checkpoint=True,**common)
    assert half.status=='completed',half.failure_reason
    resumed=solve_static_nonlinear(model,restart_checkpoint=half.to_restart_checkpoint(),emit_restart_checkpoint=True,**common)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(path.displacements,resumed.displacements,atol=1e-12,rtol=0)
    assert json.dumps(path.to_restart_checkpoint(),sort_keys=True)==json.dumps(resumed.to_restart_checkpoint(),sort_keys=True)
    changed=replace(definition,rigid_bodies=(replace(definition.rigid_bodies[0],name='tool',vertices=((0.,0.,0.),(1.01,0.,0.),(1.,1.,0.),(0.,1.,0.))),))
    with pytest.raises(ValueError):
        solve_static_nonlinear(model,restart_checkpoint=half.to_restart_checkpoint(),**{**common,'contact':changed})
    from anyfesolver.control import CancellationToken
    token=CancellationToken();token.cancel('fixture cancellation')
    from anyfesolver.control import SolveCancelled
    with pytest.raises(SolveCancelled):
        solve_static_nonlinear(model,cancellation_token=token,**common)
    if os.environ.get('TRIANGULATED_RIGID_RESULT'):
        Path(os.environ['TRIANGULATED_RIGID_RESULT']).write_text(json.dumps({'force':result,'support_load_contact_balance':balance,'phases':phases,'restart_max_error':float(np.max(np.abs(path.displacements-resumed.displacements)))},indent=2,default=lambda v:v.item()))


def test_captured_operator_mutation_refuses():
    for mutate in (lambda m:setattr(m.law,'stiffness',0.), lambda m:setattr(m,'gap_tolerance',1.),
                   lambda m:setattr(m.law,'source_face_indices',()),lambda m:setattr(m,'reference_normals',()),lambda m:setattr(m.law,'n_coordinates',10**8),lambda m:setattr(m.law,'source_node_ids',())):
        m=TriangulatedRigidManager(*case());mutate(m)
        with pytest.raises(ValueError,match='mutated'): m.evaluate(np.zeros(m.n_dof))


def test_active_partial_footprint_complete_work_and_rigid_wrench():
    manager=TriangulatedRigidManager(*case())
    source,target=manager._positions(np.zeros(manager.n_dof))
    source[:,0]-=.27;source[:,2]=-.00002
    base=manager.law.work(source,target)
    assert 0<base.state.reference_overlap_area<.20875
    x=np.r_[source.reshape(-1),target.reshape(-1)]
    v=np.zeros_like(x);v[0:9:3]=.2;v[2]=.1;v[4]=.005
    # v[4]=.005 is the declared bottom-skin displacement from unit node-2 rx.
    h=1e-7
    def at(sign):
        z=x+sign*h*v
        return manager.law.work(z[:9].reshape(3,3),z[9:].reshape(-1,3))
    plus,minus=at(1),at(-1)
    assert abs((plus.state.energy-minus.state.energy)/(2*h)-base.residual@v)/max(1.,abs(base.residual@v))<1e-5
    assert np.linalg.norm((plus.residual-minus.residual)/(2*h)-base.tangent@v)/max(1.,np.linalg.norm(base.tangent@v))<1e-5
    physical=-base.residual.reshape(-1,3)
    assert np.linalg.norm(physical.sum(0))/1.<1e-6
    assert np.linalg.norm(np.cross(np.vstack((source,target)),physical).sum(0))/1.<1e-6
    axis=np.array((0.,0.,1.))
    target_direction=np.cross(np.broadcast_to(axis,target.shape),target)
    target_gradient=base.residual[9:].reshape(-1,3)
    moment=-np.cross(target,target_gradient).sum(0)
    energy=[]
    for angle in (-h,h):
        rotation=np.array(((np.cos(angle),-np.sin(angle),0.),(np.sin(angle),np.cos(angle),0.),(0.,0.,1.)))
        energy.append(manager.law.evaluate(source,target@rotation.T).energy)
    assert abs((energy[1]-energy[0])/(2*h)+moment[2])/max(1.,abs(moment[2]))<1e-6
    assert abs(np.sum(target_gradient*target_direction)+moment[2])<1e-12


def test_profile_kinematics_and_elastic_refusal():
    model,definition,settings=case()
    load=LoadCase('press');load.add_nodal_load(2,forces=(0.,0.,-.05))
    with pytest.raises(ValueError,match='small-rotation'):
        solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,kinematics='corotational')
    assert not validate_general_contact(model,definition,settings,analysis_control='transient').valid
    model.materials['elastic'].yield_stress=1.
    assert not validate_general_contact(model,definition,settings).valid


def test_exact_tiny_overlap_and_cancellation_scale_refuse():
    body=TriangulatedRigidContactBody('overlap',((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(1e-14,0.,0.),(1e-14,1.,0.),(1.,1.,0.)),((1,(0,1,2)),(2,(3,5,4))))
    with pytest.raises(ValueError,match='intersecting'): body.planar_positions()
    manager=TriangulatedRigidManager(*case())
    start=np.zeros(manager.n_dof);end=start.copy()
    for node in (1,2,3):
        end[manager.model.mesh.dof_manager.get_node_dofs(node)[0]]=1e14
    assert not manager.swept_increment_admissible(start,end,0.,1.)


def test_complete_fe_skin_rotation_pullback():
    manager=TriangulatedRigidManager(*case());model=manager.model
    u=np.zeros(manager.n_dof);v=u.copy()
    for node in (1,2,3):
        ids=model.mesh.dof_manager.get_node_dofs(node)
        u[ids[0]]=-.27;u[ids[2]]=-3e-5
    ids=model.mesh.dof_manager.get_node_dofs(1)
    v[ids[3]]=1.;v[ids[2]]=.1
    source,target=manager._positions(u)
    physical=np.asarray(manager.map@v).reshape(-1,3)
    assert physical[0,1]==.005  # bottom skin offset times global rx cross director
    f,k,_=manager.evaluate(u)
    h=1e-7
    fp,_,rp=manager.evaluate(u+h*v);fm,_,rm=manager.evaluate(u-h*v)
    assert abs((rp.penalty_energy-rm.penalty_energy)/(2*h)-f@v)/max(1.,abs(f@v))<1e-5
    assert np.linalg.norm((fp-fm)/(2*h)-k@v)/max(1.,np.linalg.norm(k@v))<1e-5


def test_edge_only_below_plane_is_not_open():
    manager=TriangulatedRigidManager(*case())
    source,target=manager._positions(np.zeros(manager.n_dof))
    source[:]=((1.,.2,-.00002),(1.4,.2,-.00002),(1.,.7,-.00002))
    with pytest.raises(ValueError,match='rim'): manager._gap_lower((source,target))
    source[:,2]=0.
    with pytest.raises(ValueError,match='rim'): manager._gap_lower((source,target))
    source[:,0]+=.1
    assert manager._gap_lower((source,target))==float('inf')


def test_positive_component_does_not_hide_remote_edge_contact():
    model,definition,settings=case()
    body=TriangulatedRigidContactBody('tool',((0.,0.,0.),(.9,0.,0.),(.9,.9,0.),(0.,.9,0.),(1.,.1,0.),(1.4,.1,0.),(1.,.8,0.)),((1,(0,1,2)),(2,(0,2,3)),(3,(4,5,6))))
    manager=TriangulatedRigidManager(model,replace(definition,rigid_bodies=(body,)),settings)
    source,target=manager._positions(np.zeros(manager.n_dof))
    source[:]=((.2,.2,-.00002),(1.,.2,-.00002),(1.,.7,-.00002))
    with pytest.raises(ValueError,match='rim'): manager._gap_lower((source,target))


def test_degenerate_polygon_cannot_certify_coverage():
    from types import SimpleNamespace
    manager=TriangulatedRigidManager(*case())
    source,target=manager._positions(np.zeros(manager.n_dof))
    source[:]=((1.,.2,-.00002),(1.4,.2,-.00002),(1.,.7,-.00002))
    state=SimpleNamespace(pair_states=(SimpleNamespace(overlap_vertices=3),SimpleNamespace(overlap_vertices=3)))
    with pytest.raises(ValueError,match='representation'): manager._reject_edge_only((source,target),state)
