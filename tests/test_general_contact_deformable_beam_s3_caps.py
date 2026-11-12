"""Independent moving-normal endpoint support and coupled physical work."""
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import root
from anyfesolver import (BoundaryCondition, LoadCase, DisplacementControl, CancellationToken,
    SolveCancelled, solve_static_nonlinear, validate_general_contact)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.general_contact.beam_hemisphere_s3_support import hemisphere_s3_work
from anyfesolver.general_contact.deformable_beam_s3_caps import DeformableBeamS3CapManager, PROFILE
from test_general_contact_deformable_beam_s3_static import fixture as line_fixture, constraints
from test_general_contact_deformable_beam_s3_static import independent_contact_gradient as line_oracle
from test_general_contact_hemisphere_plane import scalar_gap
from s3_von_karman_oracle import model_triangle_von_karman


def scalar_support(beam,skin,endpoint,sign,radius,orientation):
    n=orientation*np.cross(skin[1]-skin[0],skin[2]-skin[0]);n/=np.linalg.norm(n)
    return scalar_gap(beam,endpoint,sign,radius,skin[0],n)


@pytest.mark.parametrize('quadratic',(False,True))
@pytest.mark.parametrize('orientation',(-1,1))
def test_all_coordinate_support_work_cross_blocks_and_skin_wrenches(quadratic,orientation):
    beam=np.array([[0.,0.,.009],[.25,.01,.0094],[.5,0.,.0096]] if quadratic else
                  [[0.,0.,.009],[.5,0.,.0096]])
    skin=np.array([[-1.,-1.,0.],[2.,-1.,0.],[-1.,2.,.0003]])
    beam[:,2]*=orientation;skin[:,2]*=orientation
    count=len(beam);x=np.r_[beam.ravel(),skin.ravel()]
    direction=np.random.default_rng(146).normal(size=len(x));direction/=np.linalg.norm(direction)
    for endpoint,sign in ((0,-1),(count-1,1)):
        s=hemisphere_s3_work(beam,skin,endpoint,sign,.01,orientation)
        assert s.gap==pytest.approx(scalar_support(beam,skin,endpoint,sign,.01,orientation),abs=1e-16)
        assert s.gap<0
        for h in (1e-7,3e-8):
            a=x+h*direction;b=x-h*direction
            ap,at=a[:3*count].reshape(count,3),a[3*count:].reshape(3,3)
            bp,bt=b[:3*count].reshape(count,3),b[3*count:].reshape(3,3)
            fd=(scalar_support(ap,at,endpoint,sign,.01,orientation)-
                scalar_support(bp,bt,endpoint,sign,.01,orientation))/(2*h)
            assert fd==pytest.approx(s.gradient@direction,rel=1e-7,abs=1e-9)
            plus=hemisphere_s3_work(ap,at,endpoint,sign,.01,orientation)
            minus=hemisphere_s3_work(bp,bt,endpoint,sign,.01,orientation)
            exact=s.hessian@direction
            assert np.linalg.norm((plus.gradient-minus.gradient)/(2*h)-exact)/max(1.,np.linalg.norm(exact))<1e-5
        G=s.gradient.reshape(count+3,3)
        np.testing.assert_allclose(G.sum(axis=0),0.,atol=1e-14)
        np.testing.assert_allclose(np.cross(np.r_[beam,skin],G).sum(axis=0),0.,atol=1e-14)
        np.testing.assert_allclose(G[:count].sum(axis=0),s.normal,atol=1e-14)
        np.testing.assert_allclose(np.cross(beam,G[:count]).sum(axis=0),np.cross(s.skin_point,s.normal),atol=1e-14)
        assert np.linalg.norm(s.hessian[:3*count,3*count:])>0
        if quadratic and s.branch=='rim':assert np.linalg.norm(G[1])>0


def cap_fixture(quadratic=False):
    model,definition,settings=line_fixture()
    if quadratic:
        model.add_node(6,.25,0.,.02005)
        model.mesh.elements[1]=QuadraticBeamElement(1,(1,6,2),'beam',
            {'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4})
    definition=replace(definition,surfaces=(replace(definition.surfaces[0],beam_end_caps='hemisphere'),definition.surfaces[1]),
        interactions=(replace(definition.interactions[0],endpoint_stiffness=1e3),))
    return model,definition,replace(settings,beam_end_cap_work='hemisphere_support_dev')


def scalar_total(beam,skin,manager):
    n=manager.kernel.orientation*np.cross(skin[1]-skin[0],skin[2]-skin[0]);n/=np.linalg.norm(n)
    def line(t):
        p=(1-t)*beam[0]+t*beam[1] if len(beam)==2 else (
            (1-t)*(1-2*t)*beam[0]+4*t*(1-t)*beam[1]+t*(2*t-1)*beam[2])
        return .5*manager.line_stiffness*manager.kernel.length*min(0.,(p-skin[0])@n-manager.kernel.radius)**2
    # All-active check fixtures below; no production clipping/root oracle.
    E=quad(line,0.,1.,epsabs=1e-15,epsrel=1e-12)[0]
    for _,(_,endpoint,sign) in manager.cap_endpoints.items():
        g=scalar_support(beam,skin,endpoint,sign,manager.kernel.radius,manager.kernel.orientation)
        E+=.5*manager.endpoint_stiffness*min(0.,g)**2
    return E


@pytest.mark.parametrize('quadratic',(False,True))
def test_complete_skin_pullback_deforming_target_tangent_and_offsets(quadratic):
    model,definition,settings=cap_fixture(quadratic)
    assert validate_general_contact(model,definition,settings).valid
    manager=DeformableBeamS3CapManager(model,definition,settings)
    u=np.zeros(model.mesh.dof_manager.total_dofs)
    for node,value in ((1,-.0001),(2,-.0002),(5,.0001)):
        u[model.mesh.dof_manager.get_node_dofs(node)[2]]=value
    if quadratic:u[model.mesh.dof_manager.get_node_dofs(6)[2]]=-.00012
    f,K,r=manager.evaluate(u);fields=r.to_dict();beam,skin=manager._positions(u)
    assert fields['active_endpoints']==2 and fields['profile']==PROFILE
    assert fields['algorithmic_contact_potential']==pytest.approx(scalar_total(beam,skin,manager),rel=1e-11)
    direction=np.zeros_like(u)
    for node in model.mesh.nodes:
        direction[model.mesh.dof_manager.get_node_dofs(node)[:3]]=np.random.default_rng(node+273).normal(size=3)
    direction/=np.linalg.norm(direction)
    for h in (1e-7,3e-8):
        p=u+h*direction;m=u-h*direction
        pb,ps=manager._positions(p);mb,ms=manager._positions(m)
        derivative=(scalar_total(pb,ps,manager)-scalar_total(mb,ms,manager))/(2*h)
        assert derivative==pytest.approx(f@direction,rel=1e-7,abs=1e-9)
        fp=manager.evaluate(p,tangent=False)[0];fm=manager.evaluate(m,tangent=False)[0]
        assert np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)<1e-5
    np.testing.assert_allclose(np.array(r.beam_force)+r.shell_force,0.,atol=1e-11)
    np.testing.assert_allclose(np.array(r.beam_moment)+r.shell_moment,0.,atol=1e-11)
    # Physical skin offset moments are structural support couples through W.
    for i,node in enumerate(manager.shell.node_ids):
        dofs=model.mesh.dof_manager.get_node_dofs(node)
        np.testing.assert_allclose(f[dofs[3:6]],np.cross(manager.shell.physical_offset*manager.shell.director,
            f[dofs[:3]]),atol=1e-12)
    # This local derivative fixture penetrates beyond the explicit 10 µm
    # increment budget. It must be cut back, not accepted because work matches.
    assert not manager.swept_increment_admissible(np.zeros_like(u),u,0.,1.)
    fraction=manager.admissible_step_fraction(np.zeros_like(u),u)
    assert 0<fraction<1 and manager.swept_increment_admissible(np.zeros_like(u),fraction*u,0.,fraction)
    invalid=u.copy();invalid[model.mesh.dof_manager.get_node_dofs(5)[3]]=1e-8
    with pytest.raises(ValueError,match='rotational'):manager.evaluate(invalid)


def test_plane_limit_seam_objectivity_and_nonunique_open_parallel_rim():
    skin=np.array([[-1.,-1.,0.],[2.,-1.,0.],[-1.,2.,0.]])
    for theta in (-1e-3,-1e-7,0.,1e-7,1e-3):
        end=np.array([.5,0.,.009]);beam=np.array([end-.5*np.array([np.cos(theta),0.,np.sin(theta)]),end])
        s=hemisphere_s3_work(beam,skin,1,1,.01,1)
        expected=.009-.01*(1 if theta<=0 else np.cos(theta))
        assert s.gap==pytest.approx(expected,abs=1e-16)
        angle=.31;Q=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
        shift=np.array([.7,-.3,.2])
        # At literal seam an oblique IEEE transform may choose either C1 branch;
        # gradient covariance holds, but mixed one-sided Hessians are not compared.
        other=hemisphere_s3_work(beam@Q.T+shift,skin@Q.T+shift,1,1,.01,1)
        np.testing.assert_allclose(other.gradient.reshape(5,3),s.gradient.reshape(5,3)@Q.T,rtol=1e-10,atol=1e-12)
        assert other.gap==pytest.approx(s.gap,abs=2e-16)
    s=hemisphere_s3_work(np.array([[0.,0.,.01],[0.,0.,.02]]),skin,1,1,.01,1)
    assert s.branch=='parallel_rim' and s.skin_point is None and s.gap>0


def scalar_two_dof_cap_contact(x):
    tip,w=x;A=np.sqrt(9+w*w);N=3*(.01005+tip)-w
    gap=N/A-.01
    # In this guided loading the root cap is open; actual tip tangent points
    # toward the skin and the active tip support lies on the sphere branch.
    return line_oracle(x)+1e3*min(gap,0.)*np.array([3/A,-1/A-N*w/A**3])


def coupled_fixture():
    model,definition,settings=cap_fixture();constraints(model)
    load=LoadCase('cap plus line press');load.add_nodal_load(2,forces=(0.,0.,-.001))
    return model,definition,settings,load


def test_actual_deforming_target_coupled_force_displacement_oracle_and_restart():
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    model,definition,settings,load=coupled_fixture()
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=25,
        tolerance=1e-10,emit_restart_checkpoint=True)
    result=solve_static_nonlinear(model,load,**common)
    assert result.status=='completed',result.failure_reason
    inds=np.array([model.mesh.dof_manager.get_node_dofs(n)[2] for n in (2,5)])
    x=result.displacements[inds]
    assert x[1]<-1e-7  # A genuinely deforming target, not a prescribed rigid plane.
    manager=DeformableBeamS3CapManager(model,definition,settings)
    force,_,r=manager.evaluate(result.displacements)
    fields=r.to_dict()
    assert fields['active_endpoints']==1 and fields['gap_tolerance_met']
    assert fields['endpoint_fields'][0]['active'] is False
    assert fields['endpoint_fields'][1]['branch']=='sphere'
    np.testing.assert_allclose(force[inds],scalar_two_dof_cap_contact(x),rtol=1e-6,atol=1e-12)
    K,_=assemble_stiffness_matrix(model);K2=K[inds][:,inds].toarray()
    # Retarget to the adopted S3 VK law (precedent 706bf011c): add the
    # independent skin membrane VK force (nodes 3,4 fixed, node 5 uz only;
    # fixture constants E=1e5, nu=.3, t=.02) to the linear structural force.
    def membrane(y):
        return np.array((0.,model_triangle_von_karman(model,(3,4,5),(0.,0.,1.),(0.,0.,y[1]),1.e5,.3,.02)[1][2]))
    reference=root(lambda y:K2@y+membrane(y)+scalar_two_dof_cap_contact(y)-[-.001,0.],x,tol=1e-11)
    assert reference.success
    assert np.linalg.norm(reference.x-x)/np.linalg.norm(reference.x)<1e-3
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:
        Path(record).write_text(json.dumps({'scope':'guided fixed-director moving-S3 development',
            'coordinates_m':x.tolist(),'von_karman_reference_m':reference.x.tolist(),
            'von_karman_response_relative_error':float(np.linalg.norm(reference.x-x)/np.linalg.norm(reference.x)),
            'contact_gradient_relative_error':float(np.linalg.norm(force[inds]-scalar_two_dof_cap_contact(x))/np.linalg.norm(force[inds])),
            'gap_tolerance_m':definition.interactions[0].gap_tolerance,
            'minimum_gap_m':fields['minimum_gap'],'endpoint_fields':fields['endpoint_fields']},indent=2),encoding='utf-8')
    fresh,_,_,_=coupled_fixture()
    controlled=solve_static_nonlinear(fresh,load,**{**common,'num_steps':32},control='displacement',
        displacement_control=DisplacementControl(node_id=2,dof='uz',target_displacement=float(x[0])))
    assert controlled.status=='completed',controlled.failure_reason
    np.testing.assert_allclose(controlled.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    prefix_model,_,_,_=coupled_fixture()
    prefix=solve_static_nonlinear(prefix_model,load,**{**common,'num_steps':2},max_load_factor=.5)
    assert prefix.status=='completed',prefix.failure_reason
    checkpoint=json.loads(json.dumps(prefix.to_restart_checkpoint(),allow_nan=False))
    fresh,_,_,_=coupled_fixture()
    resumed=solve_static_nonlinear(fresh,load,**{**common,'num_steps':2},restart_checkpoint=checkpoint)
    assert resumed.status=='completed',resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,result.displacements,rtol=1e-6,atol=1e-10)
    for altered in (replace(settings,beam_end_cap_work='owned_point'),replace(settings,beam_shell_maximum_swept_intervals=127)):
        fresh,_,_,_=coupled_fixture()
        with pytest.raises(ValueError,match='contact|restart|incompatible|unsupported|caps'):
            solve_static_nonlinear(fresh,load,**{**common,'contact_settings':altered},restart_checkpoint=checkpoint)


def test_rejected_first_displacement_trial_preserves_origin_state_and_failure_status():
    model,definition,settings,load=coupled_fixture()
    reference=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1e-10)
    assert reference.status=='completed'
    target=float(reference.displacements[model.mesh.dof_manager.get_node_dofs(2)[2]])
    model,definition,settings,load=coupled_fixture()
    # The coarse first trial is not certified within the unchanged gap/sweep
    # budgets. The old driver masked this rejection with an S3 seal error.
    result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
        num_steps=4,max_iterations=25,tolerance=1e-10,emit_restart_checkpoint=True,
        control='displacement',displacement_control=DisplacementControl(node_id=2,dof='uz',target_displacement=target))
    assert result.status=='contact_gap_exceeded',result.failure_reason
    assert not result.steps
    np.testing.assert_array_equal(result.displacements,0.)
    checkpoint=json.loads(json.dumps(result.to_restart_checkpoint(),allow_nan=False))
    assert checkpoint['element_states']
    # Honest failed checkpoints remain noncontinuable under the restart contract.
    with pytest.raises(ValueError,match='continuable completed'):
        solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
            num_steps=32,max_iterations=25,tolerance=1e-10,restart_checkpoint=checkpoint,
            control='displacement',displacement_control=DisplacementControl(node_id=2,dof='uz',target_displacement=target))
    fresh,_,_,_=coupled_fixture()
    direct=solve_static_nonlinear(fresh,load,contact=definition,contact_settings=settings,
        num_steps=32,max_iterations=25,tolerance=1e-10,
        control='displacement',displacement_control=DisplacementControl(node_id=2,dof='uz',target_displacement=target))
    assert direct.status=='completed'
    np.testing.assert_allclose(direct.displacements,reference.displacements,rtol=1e-6,atol=1e-10)


@pytest.mark.parametrize('cancel',(False,True))
def test_actual_moving_skin_exception_or_cancel_restores_solver_state(cancel,monkeypatch):
    model,definition,settings,load=coupled_fixture()
    common=dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=25,
        tolerance=1e-10,emit_restart_checkpoint=True)
    token=CancellationToken();armed=[False];raised=[False]
    original=DeformableBeamS3CapManager.evaluate
    def callback(event):
        if event.event_type=='nonlinear_static_step' and event.metadata['load_factor']>=.5:
            if cancel:token.cancel('moving S3 interruption')
            else:armed[0]=True
    def failing(self,*args,**kwargs):
        candidate=original(self,*args,**kwargs)
        if armed[0] and not raised[0]:
            raised[0]=True;raise RuntimeError('moving S3 interruption')
        return candidate
    with monkeypatch.context() as patch:
        if not cancel:patch.setattr(DeformableBeamS3CapManager,'evaluate',failing)
        with pytest.raises(SolveCancelled if cancel else RuntimeError,match='moving S3 interruption'):
            solve_static_nonlinear(model,load,**common,progress_callback=callback,cancellation_token=token)
    if not cancel:assert raised[0]
    rerun=solve_static_nonlinear(model,load,**common)
    fresh,_,_,fresh_load=coupled_fixture()
    reference=solve_static_nonlinear(fresh,fresh_load,**common)
    assert rerun.status==reference.status=='completed'
    np.testing.assert_array_equal(rerun.displacements,reference.displacements)
    assert rerun.to_restart_checkpoint()==reference.to_restart_checkpoint()


def test_true_material_ends_and_explicit_admission():
    model,definition,settings=cap_fixture(True)
    model.add_node(7,0.,.1,.02005)
    model.add_element(3,BeamElement(3,(1,7),'beam',{'area':.001,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    manager=DeformableBeamS3CapManager(model,definition,settings)
    assert set(manager.cap_endpoints)=={2}  # unselected junction and B3 midnode are not ends
    for altered in (replace(settings,beam_shell_geometry='translated_planar'),
                    replace(settings,beam_shell_rim_clearance=.005)):
        assert not validate_general_contact(model,definition,altered).valid
    assert not validate_general_contact(model,definition,settings,analysis_control='arc').valid
    for law in (replace(definition.interactions[0],gap_tolerance=.02),
                replace(definition.interactions[0],normal_law='augmented')):
        assert not validate_general_contact(model,replace(definition,interactions=(law,)),settings).valid


def test_runnable_moving_skin_example_and_unaccepted_capability():
    from examples.general_contact_deformable_s3_caps import run, build
    from anyfesolver import report_general_contact_capability
    result=run();assert result.status=='completed'
    assert result.info['general_contact']['profile']==PROFILE
    model,definition,settings,_=build()
    report=report_general_contact_capability(model,definition,settings)
    assert report.implemented and not report.admitted_for_application
