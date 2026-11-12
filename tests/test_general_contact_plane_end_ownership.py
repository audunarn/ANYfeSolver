"""Independent affine ownership roots and coupled physical-end plane work."""
from dataclasses import replace
from fractions import Fraction
import json

import numpy as np
import pytest
from scipy.optimize import brentq
from anyfesolver import BoundaryCondition,LoadCase,solve_static_nonlinear
from anyfesolver.general_contact.beam_plane import _BeamPlaneManager
from test_general_contact_beam_plane import _model


def _pose(model,heights):
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,height in zip(sorted(model.mesh.nodes),heights):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]]=height-model.mesh.nodes[node].z
    return u


@pytest.mark.parametrize('quadratic',(False,True))
def test_closed_ownership_crossing_rejects_whole_path_and_limits_trial(quadratic):
    model,definition,settings=_model(quadratic=quadratic,end_caps='hemisphere')
    manager=_BeamPlaneManager(model,definition,settings)
    start_heights=(.009,.0105);end_heights=(.009,.0085)
    if quadratic:
        start_heights=(.009,.00975,.0105);end_heights=(.009,.00875,.0085)
    start=_pose(model,start_heights);end=_pose(model,end_heights)
    assert manager.evaluate(start)[2].endpoint_contact_energy>0
    assert not manager.swept_increment_admissible(start,end,0.,1.)
    fraction=manager.admissible_step_fraction(start,end)
    assert 0<fraction<1
    assert manager.swept_increment_admissible(start,start+fraction*(end-start),0.,fraction)
    assert manager.restart_contract()['hemisphere_ownership_revision']=='fixed_plane_exact_affine_ownership_v1'
    plain=replace(definition,surfaces=(replace(definition.surfaces[0],beam_end_caps='none'),),
        interactions=(replace(definition.interactions[0],endpoint_stiffness=None),))
    assert 'hemisphere_ownership_revision' not in _BeamPlaneManager(model,plain,settings).restart_contract()


def test_ownership_must_open_before_not_after_its_exact_root():
    model,definition,settings=_model(end_caps='hemisphere')
    manager=_BeamPlaneManager(model,definition,settings)
    # The first end is open at the final pose, but the ownership root occurs
    # while closed. Endpoint-only open/closed tests would incorrectly admit it.
    start=_pose(model,(.0095,.0105));end=_pose(model,(.0112,.0082))
    s=model.mesh.nodes[1].z+start[model.mesh.dof_manager.get_node_dofs(1)[2]]
    t=model.mesh.nodes[2].z+start[model.mesh.dof_manager.get_node_dofs(2)[2]]
    e=model.mesh.nodes[1].z+end[model.mesh.dof_manager.get_node_dofs(1)[2]]
    f=model.mesh.nodes[2].z+end[model.mesh.dof_manager.get_node_dofs(2)[2]]
    s,t,e,f=map(Fraction.from_float,(s,t,e,f))
    root=(t-s)/((t-s)-(f-e))
    assert s+root*(e-s)-Fraction.from_float(.01)<0
    assert not manager.swept_increment_admissible(start,end,0.,1.)
    # Reversing ownership after the first end opens is continuous in energy.
    opening_end=_pose(model,(.011,.0105))
    assert manager.swept_increment_admissible(start,opening_end,0.,1.)
    entirely_open=_pose(model,(.015,.014))
    assert manager.swept_increment_admissible(np.zeros_like(start),entirely_open,0.,1.)


@pytest.mark.parametrize('quadratic',(False,True))
def test_closed_tangent_candidate_rejects_including_actual_b3_derivative(quadratic):
    model,definition,settings=_model(quadratic=quadratic,end_caps='hemisphere')
    for node in model.mesh.nodes.values():node.z=.015625
    manager=_BeamPlaneManager(model,definition,settings)
    heights=(.009765625,.0107421875,.013671875) if quadratic else (.009765625,)*2
    u=_pose(model,heights)
    if quadratic:assert heights[2]>heights[0]  # Chord is not the end derivative.
    for tangent in (False,True):
        with pytest.raises(ValueError,match='unresolved cap-to-side ownership'):
            manager.evaluate(u,tangent=tangent)
    assert not manager.swept_increment_admissible(u,u,0.,1.)


def _vertical():
    model,definition,settings=_model(vertical=True,end_caps='hemisphere')
    definition=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=1e-5),))
    model.add_boundary_condition(BoundaryCondition('root',(2,),
        {d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guide',(1,),
        {d:0. for d in ('ux','uy','rx','ry','rz')}))
    load=LoadCase('axial end press');load.add_nodal_load(1,forces=(0.,0.,-.08))
    return model,definition,settings,load


def test_actual_plane_cap_force_and_active_fresh_restart_match_independent_bar():
    model,definition,settings,load=_vertical()
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,tolerance=1e-10)
    result=solve_static_nonlinear(model,load,**args)
    assert result.status=='completed',result.failure_reason
    L=.1;g0=.005;EA=1.;k=ke=1e4;P=.08
    def balance(delta):
        length=L+g0+delta
        return EA/L*(g0+delta)+ke*delta+k*L*(delta**2/(2*length)-delta**3/(6*length**2))-P
    delta=brentq(balance,0.,1e-5,xtol=1e-18)
    dof=model.mesh.dof_manager.get_node_dofs(1)[2]
    assert result.displacements[dof]==pytest.approx(-g0-delta,rel=1e-6,abs=1e-10)
    info=result.info['general_contact']
    assert info['active_endpoints']==1 and info['minimum_gap']>=-1e-5
    assert info['endpoint_contact_energy']==pytest.approx(.5*ke*delta**2,rel=1e-6)
    model,definition,settings,load=_vertical()
    prefix=solve_static_nonlinear(model,load,**{**args,'num_steps':2},max_load_factor=.75,
        emit_restart_checkpoint=True)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    model,definition,settings,load=_vertical()
    resumed=solve_static_nonlinear(model,load,**{**args,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    model,definition,settings,load=_vertical()
    with pytest.raises(ValueError,match='restart|incompatible|contract'):
        solve_static_nonlinear(model,load,**{**args,'contact_settings':replace(settings,
            maximum_swept_subdivisions=settings.maximum_swept_subdivisions+1)},restart_checkpoint=checkpoint)


@pytest.mark.parametrize('quadratic',(False,True))
def test_tilted_plane_rigid_covariance_of_end_work_and_ownership(quadratic):
    model,definition,settings=_model(quadratic=quadratic,end_caps='hemisphere')
    manager=_BeamPlaneManager(model,definition,settings)
    start=_pose(model,(.009,.01,.0105) if quadratic else (.009,.0105))
    closed=_pose(model,(.009,.0088,.0085) if quadratic else (.009,.0085))
    opened=_pose(model,(.015,.0145,.014) if quadratic else (.015,.014))
    f,K,info=manager.evaluate(start)
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);angle=.31
    W=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    Q=np.eye(3)+np.sin(angle)*W+(1-np.cos(angle))*(W@W);shift=np.array([.37,-.21,.11])
    moved,md,_,=_model(quadratic=quadratic,end_caps='hemisphere')
    for node in moved.mesh.nodes.values():
        p=Q@node.coords()+shift;node.x,node.y,node.z=p
    body=md.rigid_bodies[0]
    md=replace(md,rigid_bodies=(replace(body,point=tuple(shift),normal=tuple(Q@np.array([0.,0.,1.]))),))
    rotated=_BeamPlaneManager(moved,md,settings)
    T=np.kron(np.eye(len(start)//3),Q)
    rf,rK,ri=rotated.evaluate(T@start)
    np.testing.assert_allclose(rf,T@f,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(rK.toarray(),T@K.toarray()@T.T,rtol=1e-8,atol=1e-7)
    assert ri.endpoint_contact_energy==pytest.approx(info.endpoint_contact_energy,rel=1e-9)
    np.testing.assert_allclose(ri.beam_moment,Q@np.array(info.beam_moment)
        +np.cross(shift,Q@np.array(info.beam_force)),rtol=1e-9,atol=1e-9)
    assert not manager.swept_increment_admissible(start,closed,0.,1.)
    assert not rotated.swept_increment_admissible(T@start,T@closed,0.,1.)
    assert manager.swept_increment_admissible(start,opened,0.,1.)
    assert rotated.swept_increment_admissible(T@start,T@opened,0.,1.)


def test_public_plane_endpoint_example():
    from examples.general_contact_beam_end_capped_cylinder import run
    result=run(plane=True)
    assert result.info['general_contact']['active_endpoints']==1
    assert 'FIXED_PLANE' in result.info['general_contact']['profile']
