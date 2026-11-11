"""Independent scalar hemisphere support, full work maps and static controls."""
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.optimize import brentq
from anyfesolver import (BoundaryCondition, LoadCase, solve_static_nonlinear,
    validate_general_contact, DisplacementControl, CancellationToken, SolveCancelled)
from anyfesolver.general_contact.beam_hemisphere_support import hemisphere_plane_work
from anyfesolver.general_contact.beam_hemisphere_plane import HemispherePlaneManager, PROFILE
from test_general_contact_beam_plane import _model


def scalar_gap(x, endpoint, sign, radius, point, normal):
    # Independent ordinary scalar formula; does not call production AD/support.
    normal = np.asarray(normal)/np.linalg.norm(normal)
    v = (x[1]-x[0] if len(x)==2 else
         -3*x[0]+4*x[1]-x[2] if endpoint==0 else x[0]-4*x[1]+3*x[2])
    beta = sign*np.dot(v,normal)/np.linalg.norm(v)
    support = 1. if beta <= 0 else np.sqrt(1.-beta**2)
    return np.dot(x[endpoint]-point,normal)-radius*support


@pytest.mark.parametrize('quadratic',(False,True))
@pytest.mark.parametrize('angle',(-.22,.22))
def test_scalar_support_virtual_work_complete_tangent_and_skin_wrench(quadratic,angle):
    end=np.array([1.,.03,.095]);start=end-np.array([np.cos(angle),.1,np.sin(angle)])
    x=np.array([start,(start+end)/2+np.array([.01,.01,-.002]),end] if quadratic else [start,end])
    state=hemisphere_plane_work(x,len(x)-1,1,.1,np.zeros(3),np.array([0.,0.,2.]))
    expected=scalar_gap(x,len(x)-1,1,.1,np.zeros(3),np.array([0.,0.,2.]))
    assert state.gap==pytest.approx(expected,abs=2e-16)
    # Scalar support and AD gap derivatives; use near-parallel inactive poses
    # separately. Endpoint spring activated in the positive-angle fixture.
    direction=np.random.default_rng(643).normal(size=x.size);direction/=np.linalg.norm(direction)
    for h in (1e-6,3e-7):
        plus=x+h*direction.reshape(x.shape);minus=x-h*direction.reshape(x.shape)
        gp=scalar_gap(plus,len(x)-1,1,.1,np.zeros(3),np.array([0.,0.,2.]))
        gm=scalar_gap(minus,len(x)-1,1,.1,np.zeros(3),np.array([0.,0.,2.]))
        assert (gp-gm)/(2*h)==pytest.approx(state.gradient@direction,rel=1e-7,abs=1e-9)
        a=hemisphere_plane_work(plus,len(x)-1,1,.1,np.zeros(3),np.array([0.,0.,2.]))
        b=hemisphere_plane_work(minus,len(x)-1,1,.1,np.zeros(3),np.array([0.,0.,2.]))
        exact=state.hessian@direction
        assert np.linalg.norm((a.gradient-b.gradient)/(2*h)-exact)<1e-5*max(1.,np.linalg.norm(exact))
    np.testing.assert_allclose(state.gradient.reshape(-1,3).sum(axis=0),state.normal,atol=1e-14)
    np.testing.assert_allclose(np.cross(x,state.gradient.reshape(-1,3)).sum(axis=0),
                               np.cross(state.skin_point,state.normal),atol=1e-14)
    if quadratic and angle>0:assert np.linalg.norm(state.gradient[3:6])>0


def _support_model(quadratic=False,vertical=False):
    model,definition,settings=_model(quadratic=quadratic,vertical=vertical,end_caps='hemisphere')
    return model,definition,replace(settings,beam_end_cap_work='hemisphere_support_dev')


def _u(model,heights):
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,h in zip(sorted(model.mesh.nodes),heights):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]]=h-model.mesh.nodes[node].z
    return u


@pytest.mark.parametrize('quadratic',(False,True))
def test_closed_seam_continuity_actual_manager_and_full_tangent(quadratic):
    model,definition,settings=_support_model(quadratic)
    assert validate_general_contact(model,definition,settings).valid
    manager=HemispherePlaneManager(model,definition,settings)
    u=_u(model,(.0095,.0097,.0099) if quadratic else (.0095,.0099))
    f,K,r=manager.evaluate(u)
    assert r.profile==PROFILE and r.active_endpoints==2
    direction=np.random.default_rng(563).normal(size=u.size);direction/=np.linalg.norm(direction)
    for h in (1e-7,3e-8):
        fp,_,rp=manager.evaluate(u+h*direction,tangent=False)
        fm,_,rm=manager.evaluate(u-h*direction,tangent=False)
        E=lambda rr:rr.line_contact_energy+rr.endpoint_contact_energy
        assert (E(rp)-E(rm))/(2*h)==pytest.approx(f@direction,rel=1e-7,abs=1e-8)
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
    closed=_u(model,(.0095,)* (3 if quadratic else 2))
    seam=manager.evaluate(closed)[2]
    assert seam.active_endpoints==2 and all(a['branch']=='sphere' for a in seam.endpoint_fields)
    end=_u(model,(.0099,.0097,.0095) if quadratic else (.0099,.0095))
    assert manager.swept_increment_admissible(u,end,0.,1.)
    assert manager.admissible_step_fraction(u,end)==1.
    assert manager.restart_contract()['beam_end_cap_work']=='hemisphere_support_dev'


def test_exact_seam_one_sided_derivatives_and_no_legacy_energy_jump():
    # Genuine endpoint fixed, actual tangent crosses plane-parallel.
    def geometry(theta):
        e=np.array([1.,0.,.095]);return np.array([e-np.array([np.cos(theta),0.,np.sin(theta)]),e])
    rows=[]
    for t in (-1e-3,-1e-5,-1e-7,0.,1e-7,1e-5,1e-3):
        x=geometry(t);s=hemisphere_plane_work(x,1,1,.1,np.zeros(3),[0.,0.,1.])
        g=.095-.1*(1 if t<=0 else np.cos(t))
        assert s.gap==pytest.approx(g,abs=1e-16)
        v=np.array([[np.sin(t),0.,-np.cos(t)],[0.,0.,0.]]).reshape(-1)
        expected=0 if t<=0 else 1000*g*.1*np.sin(t)
        assert 1000*s.gap*(s.gradient@v)==pytest.approx(expected,abs=2e-15)
        rows.append(.5*1000*s.gap**2)
    assert abs(rows[2]-rows[4])<3e-15
    # Distinct valid generalized Hessians; no smooth central difference across seam.
    x=geometry(0.);s=hemisphere_plane_work(x,1,1,.1,np.zeros(3),[0.,0.,1.])
    assert np.linalg.norm(s.hessian)==0
    plus=hemisphere_plane_work(geometry(1e-7),1,1,.1,np.zeros(3),[0.,0.,1.])
    assert np.linalg.norm(plus.hessian)>0


def test_force_control_bar_restart_and_changed_cap_law_rejection():
    def fixture():
        model,definition,settings=_support_model(vertical=True)
        definition=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=1e-5),))
        model.add_boundary_condition(BoundaryCondition('root',(2,),{d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
        model.add_boundary_condition(BoundaryCondition('guide',(1,),{d:0. for d in ('ux','uy','rx','ry','rz')}))
        load=LoadCase('press');load.add_nodal_load(1,forces=(0.,0.,-.08))
        return model,definition,settings,load
    model,definition,settings,load=fixture()
    args=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,tolerance=1e-10)
    result=solve_static_nonlinear(model,load,**args)
    assert result.status=='completed',result.failure_reason
    def balance(delta):
        L=.1;length=L+.005+delta
        return 10*(.005+delta)+1e4*delta+1e4*L*(delta**2/(2*length)-delta**3/(6*length**2))-.08
    delta=brentq(balance,0.,1e-5,xtol=1e-18)
    dof=model.mesh.dof_manager.get_node_dofs(1)[2]
    assert result.displacements[dof]==pytest.approx(-.005-delta,rel=1e-6,abs=1e-10)
    model,definition,settings,load=fixture()
    prefix=solve_static_nonlinear(model,load,**{**args,'num_steps':2},max_load_factor=.75,emit_restart_checkpoint=True)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    model,definition,settings,load=fixture()
    resumed=solve_static_nonlinear(model,load,**{**args,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    model,definition,settings,load=fixture()
    with pytest.raises(ValueError,match='restart|incompatible|contract'):
        solve_static_nonlinear(model,load,**{**args,'contact_settings':replace(settings,beam_end_cap_work='owned_point')},restart_checkpoint=checkpoint)


def test_invalid_selection_parallel_open_collapse_and_hidden_crossing():
    model,definition,settings=_support_model(quadratic=True)
    assert not validate_general_contact(model,replace(definition,rigid_bodies=(replace(
        definition.rigid_bodies[0],kind='sphere',radius=.01,normal=None),)),settings).valid
    assert not validate_general_contact(model,definition,settings,analysis_control='arc').valid
    assert not validate_general_contact(model,definition,replace(settings,
        augmentation_acceleration='active_secant')).valid
    assert not validate_general_contact(model,definition,replace(settings,
        beam_shell_geometry='jet2_deformable_s3_dev')).valid
    assert not validate_general_contact(model,definition,replace(settings,beam_chain_tangent='consistent_dev')).valid
    assert not validate_general_contact(model,replace(definition,interactions=(replace(
        definition.interactions[0],gap_tolerance=.01),)),settings).valid
    manager=HemispherePlaneManager(model,definition,settings)
    start=np.zeros(model.mesh.dof_manager.total_dofs)
    crossing=_u(model,(.015,-.04,.015))
    assert not manager.swept_increment_admissible(start,crossing,0.,1.)
    collapse=start.copy()
    for node in (1,2,3):
        dofs=model.mesh.dof_manager.get_node_dofs(node)[:3]
        collapse[dofs]=[0.,0.,.015]-model.mesh.nodes[node].coords()
    with pytest.raises(ValueError,match='regularity'):manager.evaluate(collapse)
    state=hemisphere_plane_work(np.array([[0.,0.,.1],[0.,0.,.2]]),1,1,.01,[0.,0.,0.],[0.,0.,1.])
    assert state.branch=='parallel_rim' and state.skin_point is None and state.gap>0


def _prescribed_seam():
    model,definition,settings=_support_model()
    model.mesh.nodes[1].z=.0105;model.mesh.nodes[2].z=.0125
    definition=replace(definition,interactions=(replace(definition.interactions[0],gap_tolerance=.006),))
    for node,drop in ((1,-.003),(2,-.007)):
        bc={d:0. for d in ('ux','uy','uz','rx','ry','rz')};bc['uz']=drop
        if node==2:bc.pop('rx')  # Physical torsion equation remains in the solve.
        model.add_boundary_condition(BoundaryCondition(f'pose{node}',(node,),bc))
    return model,definition,settings,LoadCase('prescribed closed seam')


def test_prescribed_closed_seam_solver_reactions_and_active_open_recontact():
    model,definition,settings,load=_prescribed_seam()
    common=dict(contact=definition,contact_settings=settings,num_steps=8,
        max_iterations=20,tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**common)
    assert result.status=='completed',result.failure_reason
    assert result.steps[-1].load_factor==1.
    info=result.info['general_contact']
    assert info['active_endpoints']==2 and info['gap_tolerance_met']
    assert info['swept_trial_cutbacks']==0
    # Exact fixed-plane path crosses the closed seam at factor 1/2.
    manager=HemispherePlaneManager(model,definition,settings)
    a=_u(model,(.0105-.003*.25,.0125-.007*.25))
    b=_u(model,(.0105-.003*.75,.0125-.007*.75))
    assert manager.swept_increment_admissible(a,b,.25,.75)
    # Independent scalar total potential on this two-node physical path.
    def energy(x):
        h=x[:,2]-.01
        line=.5*1e4*(h[0]**2+h[0]*h[1]+h[1]**2)/3
        return line+sum(.5*1e4*min(0.,scalar_gap(x,i,sign,.01,
            np.zeros(3),np.array([0.,0.,1.])))**2 for i,sign in ((0,-1),(1,1)))
    positions=np.array([model.mesh.nodes[i].coords()+result.displacements[
        model.mesh.dof_manager.get_node_dofs(i)[:3]] for i in (1,2)])
    f,K,r=manager.evaluate(result.displacements)
    independent=np.zeros(6);eps=1e-7
    for j in range(6):
        p=positions.copy().reshape(-1);m=p.copy();p[j]+=eps;m[j]-=eps
        independent[j]=(energy(p.reshape(2,3))-energy(m.reshape(2,3)))/(2*eps)
    dofs=np.array([model.mesh.dof_manager.get_node_dofs(i)[:3] for i in (1,2)]).reshape(-1)
    np.testing.assert_allclose(f[dofs],independent,rtol=1e-6,atol=1e-8)
    # Contact is assembled before support reaction recovery: subtract the same
    # structural prescribed solve without contact, then compare both supports.
    plain,_,_,plain_load=_prescribed_seam()
    without=solve_static_nonlinear(plain,plain_load,num_steps=8,tolerance=1e-10)
    assert without.status=='completed'
    for node in (1,2):
        name=f'pose{node}'
        actual=(np.array(result.steps[-1].support_reactions[name][:3])-
                np.array(without.steps[-1].support_reactions[name][:3]))
        np.testing.assert_allclose(actual,f[model.mesh.dof_manager.get_node_dofs(node)[:3]],rtol=1e-6,atol=1e-8)
    open_pose=_u(model,(.02,.02));assert manager.evaluate(open_pose)[2].active_endpoints==0
    assert manager.swept_increment_admissible(result.displacements,open_pose,1.,2.)
    assert manager.swept_increment_admissible(open_pose,result.displacements,2.,3.)
    # Split the actual closed-seam continuation before/after the switch.
    m,d,s,l=_prescribed_seam()
    half=solve_static_nonlinear(m,l,**{**common,'num_steps':4},max_load_factor=.5)
    assert half.status=='completed'
    checkpoint=json.loads(json.dumps(half.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,fresh_load=_prescribed_seam()
    resumed=solve_static_nonlinear(fresh,fresh_load,**{**common,'num_steps':4},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,atol=1e-12)


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_exception_or_cancel_rollback_and_fresh_replay(cancel,monkeypatch):
    model,definition,settings,load=_prescribed_seam()
    common=dict(contact=definition,contact_settings=settings,num_steps=8,
        max_iterations=20,tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False];original=HemispherePlaneManager.evaluate
    def callback(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.5:
            if cancel:token.cancel('hemisphere support interruption')
            else:armed[0]=True
    def failing(self,*args,**kwargs):
        candidate=original(self,*args,**kwargs)
        if armed[0] and not raised[0]:
            raised[0]=True;raise RuntimeError('hemisphere support interruption')
        return candidate
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(HemispherePlaneManager,'evaluate',failing)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='hemisphere support interruption'):
            solve_static_nonlinear(model,load,**common,progress_callback=callback,cancellation_token=token)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**common)
    fresh,_,_,fresh_load=_prescribed_seam()
    reference=solve_static_nonlinear(fresh,fresh_load,**common)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_vertical_displacement_control_matches_actual_force_response():
    model,definition,settings=_support_model(vertical=True)
    model.add_boundary_condition(BoundaryCondition('root',(2,),{d:0. for d in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guide',(1,),{d:0. for d in ('ux','uy','rx','ry','rz')}))
    load=LoadCase('tip');load.add_nodal_load(1,forces=(0.,0.,-.08))
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,tolerance=1e-10)
    force=solve_static_nonlinear(model,load,**common)
    assert force.status=='completed'
    dof=model.mesh.dof_manager.get_node_dofs(1)[2]
    controlled=solve_static_nonlinear(model,load,**common,control='displacement',
        displacement_control=DisplacementControl(node_id=1,dof='uz',target_displacement=float(force.displacements[dof])))
    assert controlled.status=='completed',controlled.failure_reason
    np.testing.assert_allclose(controlled.displacements,force.displacements,rtol=1e-6,atol=1e-10)


@pytest.mark.parametrize('quadratic',(False,True))
def test_common_rigid_transform_and_connectivity_invariance(quadratic):
    model,definition,settings=_support_model(quadratic)
    manager=HemispherePlaneManager(model,definition,settings)
    u=_u(model,(.0095,.0097,.0099) if quadratic else (.0095,.0099))
    f,K,r=manager.evaluate(u)
    model.mesh.elements[1].node_ids=model.mesh.elements[1].node_ids[::-1]
    rf,rK,rr=HemispherePlaneManager(model,definition,settings).evaluate(u)
    np.testing.assert_allclose(rf,f,atol=1e-12)
    np.testing.assert_allclose(rK.toarray(),K.toarray(),atol=1e-10)
    angle=.37;Q=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
    shift=np.array([.7,-.3,.2]);rotated=np.zeros_like(u);T=np.eye(len(u))
    for node in model.mesh.nodes.values():
        dofs=model.mesh.dof_manager.get_node_dofs(node.id)[:3]
        ref=node.coords();rotated[dofs]=Q@u[dofs]
        node.x,node.y,node.z=Q@ref+shift;T[np.ix_(dofs,dofs)]=Q
    body=replace(definition.rigid_bodies[0],point=tuple(shift),normal=tuple(Q@np.array([0.,0.,1.])))
    moved=HemispherePlaneManager(model,replace(definition,rigid_bodies=(body,)),settings)
    mf,mK,mr=moved.evaluate(rotated)
    np.testing.assert_allclose(mf,T@f,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(mK.toarray(),T@K.toarray()@T.T,rtol=1e-10,atol=1e-8)
    assert mr.endpoint_contact_energy==pytest.approx(r.endpoint_contact_energy,rel=1e-10)


def test_runnable_example_capability_and_transient_rejection():
    from examples.general_contact_hemisphere_plane import run
    from anyfesolver import report_general_contact_capability
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact, ContactTransientSolverControls
    result=run()
    assert result.status=='completed' and result.info['general_contact']['profile']==PROFILE
    model,definition,settings=_support_model()
    report=report_general_contact_capability(model,definition,settings)
    assert report.implemented and not report.admitted_for_application
    with pytest.raises(ValueError,match='without end caps'):
        solve_transient_contact(model,definition,TransientConfig(.001,.002),
            contact_settings=settings,solver_controls=ContactTransientSolverControls(allow_classical_beam_dev=True))
