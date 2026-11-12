"""Coupled physical skin/free-sphere work and guided Q4 impact development."""
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver import (BoundaryCondition,ContactDefinition,ContactSurface,ContactInteraction,
    ContactSolverSettings,RigidContactBody,FEModel,LoadCase,create_shell_element)
from anyfesolver.dynamics import TransientConfig
from anyfesolver.general_contact.free_rigid import (ContactFreeRigidBody,_FreeSphereManager,
    FREE_SHELL_SPHERE_PROFILE,FREE_SHELL_SPHERE_HISTORY_PROFILE)
from anyfesolver.general_contact.sphere import _SphereShellManager
from anyfesolver.general_contact.transient import (ContactNodalLoadHistory,
    ContactTransientSolverControls,solve_transient_contact)


BODY=ContactFreeRigidBody('sphere',.002,8e-8,(0.,0.,.2),(2.,-3.,4.))
CONTROLS=ContactTransientSolverControls(allow_free_sphere_dev=True,allow_free_shell_sphere_dev=True,
    maximum_newton_iterations=20,equilibrium_tolerance=1e-10)


def model_data():
    model=FEModel('q4_free_sphere')
    model.add_material('elastic',1e3,.3,density=7812.5)
    for node,(x,y) in enumerate(((-.004,-.004),(.004,-.004),(.004,.004),(-.004,.004)),1):
        model.add_node(node,x,y,.025)
    model.add_element(1,create_shell_element(1,[1,2,3,4],'elastic',thickness=.002))
    model.add_boundary_condition(BoundaryCondition('guide',[1,2,3,4],
        {name:0. for name in ('ux','uy','rx','ry','rz')}))
    definition=ContactDefinition(surfaces=(ContactSurface('skin',(1,),side='bottom'),),
        rigid_bodies=(RigidContactBody('sphere','sphere',(0.,0.,0.),radius=.02),),
        interactions=(ContactInteraction('pair','skin','sphere',1e6,.002),))
    settings=ContactSolverSettings(pair_motion='general',maximum_shell_tilt=.05)
    return model,definition,settings


def impact(*,end=.022,dt=.0005,checkpoint=None,emit=False,body=BODY,history=None,controls=CONTROLS):
    model,definition,settings=model_data()
    velocity=np.zeros(model.mesh.dof_manager.total_dofs)
    for node in model.mesh.nodes:velocity[model.mesh.dof_manager.get_node_dofs(node)[2]]=-.2
    load=None
    if history is not None:
        load=LoadCase('normal_shell_history')
        for node in model.mesh.nodes:load.add_nodal_load(node,forces=(0.,0.,-.001/4))
    result=solve_transient_contact(model,definition,TransientConfig(dt,end,initial_velocity=velocity),
        contact_settings=settings,solver_controls=controls,free_rigid_bodies=(body,),
        restart_checkpoint=checkpoint,emit_restart_checkpoint=emit,
        base_load_case=load,nodal_load_history=history)
    return model,result


def test_shell_sphere_complete_virtual_work_tangent_and_wrench():
    model,definition,settings=model_data()
    owner=_FreeSphereManager(_SphereShellManager(model,definition,settings),BODY,allow_shell=True)
    q=np.zeros(model.mesh.dof_manager.total_dofs+3)
    for node in model.mesh.nodes:q[model.mesh.dof_manager.get_node_dofs(node)[2]]=-.006
    q[-3:]=(.0003,-.0001,.0002)
    residual,tangent,contact=owner.evaluate(q)
    assert contact.active_pressure_dofs>0
    directions=[np.random.default_rng(29).normal(size=len(q)),np.r_[np.zeros(len(q)-3),(1.,-2.,.3)]]
    for direction in directions:
        # Rotation coordinates use a dimensioned characteristic length.
        for i in (3,9,15,21):direction[i:i+3]*=10.
        h=1e-7
        rp=owner.evaluate(q+h*direction)[0];rm=owner.evaluate(q-h*direction)[0]
        finite=(rp-rm)/(2*h);exact=tangent@direction
        assert np.linalg.norm(finite-exact)/max(np.linalg.norm(exact),1e-12)<=1e-5
        ep=owner.evaluate(q+h*direction,tangent=False)[2].penalty_energy
        em=owner.evaluate(q-h*direction,tangent=False)[2].penalty_energy
        assert (ep-em)/(2*h)==pytest.approx(float(residual@direction),rel=1e-5,abs=1e-10)
    force=sum((residual[6*i:6*i+3] for i in range(4)),np.zeros(3))+residual[-3:]
    np.testing.assert_allclose(force,0.,atol=1e-12,rtol=0.)
    centre=np.array(definition.rigid_bodies[0].point)+q[-3:]
    np.testing.assert_allclose(np.asarray(contact.shell_moment)+np.cross(centre,contact.rigid_force),0.,atol=1e-12,rtol=0.)
    np.testing.assert_allclose(contact.rigid_force,-residual[-3:],atol=1e-12,rtol=0.)
    # Translating both bodies cannot change pressure, work or force.
    common=np.zeros_like(q);translation=np.array((.001,-.002,.0003))
    for i in range(4):common[6*i:6*i+3]=translation
    common[-3:]=translation
    other=owner.evaluate(q+common,tangent=False)
    np.testing.assert_allclose(other[0],residual,atol=1e-12,rtol=0.)
    assert other[2].penalty_energy==pytest.approx(contact.penalty_energy,abs=1e-14)


def test_free_shell_sphere_opt_in_and_constraint_admission():
    with pytest.raises(ValueError,match='separate Q4'):
        ContactTransientSolverControls(allow_free_shell_sphere_dev=True)
    with pytest.raises(ValueError,match='separate Q4'):
        replace(CONTROLS,allow_classical_beam_dev=True)
    model,definition,settings=model_data()
    model.boundary_conditions.clear()
    model.add_boundary_condition(BoundaryCondition('bad',[1,2,3,4],
        {name:0. for name in ('uy','rx','ry','rz')}))
    with pytest.raises(ValueError,match='constrained in-plane'):
        solve_transient_contact(model,definition,TransientConfig(.001,.002,
            initial_velocity=np.zeros(model.mesh.dof_manager.total_dofs)),contact_settings=settings,
            solver_controls=CONTROLS,free_rigid_bodies=(BODY,))


def test_selected_side_is_checked_over_entire_path_even_when_sphere_is_distant():
    from anyfesolver.general_contact.transient import _free_shell_sphere_path_on_selected_side
    from anyfesolver.general_contact.shell_sphere_temporal_bounds import q4_sphere_hermite_gap_bounds
    model,definition,settings=model_data()
    definition=replace(definition,
        surfaces=(replace(definition.surfaces[0],side='top'),),
        rigid_bodies=(replace(definition.rigid_bodies[0],point=(10.,0.,1.),radius=.1),))
    owner=_FreeSphereManager(_SphereShellManager(model,definition,settings),BODY,allow_shell=True)
    q=np.zeros(27);v0=np.zeros(27);v1=np.zeros(27)
    v0[2:24:6]=8.;v1[2:24:6]=-8.
    # Endpoints valid; w(t)=8*t*(1-t) is behind the top side at mid-step.
    owner.evaluate(q,tangent=False)
    bounds=q4_sphere_hermite_gap_bounds(owner.child,q[:24],q[:24],v0[:24],v1[:24],1.,
        sphere_start_point=owner.reference_point,sphere_end_point=owner.reference_point,
        sphere_start_velocity=v0[-3:],sphere_end_velocity=v1[-3:],
        tolerance_m=.01,maximum_subdivisions=128)
    assert bounds.lower_gap>8.
    assert not _free_shell_sphere_path_on_selected_side(owner,q,q,v0,v1,1.,128,lambda *_:None)[0]
    middle=q.copy();middle[2:24:6]=2.
    with pytest.raises(ValueError,match='behind selected'):owner.evaluate(middle,tangent=False)
    # A positive curve with a negative parent control is certified by subdivision.
    v0[2:24:6]=3.;v1[2:24:6]=-3.
    positive,splits=_free_shell_sphere_path_on_selected_side(owner,q,q,v0,v1,1.,128,lambda *_:None)
    assert positive and splits>0
    from anyfesolver.general_contact.beam_temporal_bounds import BeamHermiteGapBoundsError
    with pytest.raises(BeamHermiteGapBoundsError,match='selected-side'):
        _free_shell_sphere_path_on_selected_side(owner,q,q,v0,v1,1.,0,lambda *_:None)


def test_constraint_admission_checks_each_reduced_direction_and_rotated_normal():
    from scipy import sparse
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.general_contact.transient import _admit_free_shell_coordinates
    model,_,_=model_data()
    model.boundary_conditions.clear()
    model.add_boundary_condition(BoundaryCondition('bad',[1,2,3,4],
        {name:0. for name in ('uy','rx','ry','rz')}))
    dofs1=model.mesh.dof_manager.get_node_dofs(1)
    dofs2=model.mesh.dof_manager.get_node_dofs(2)
    model.add_constraint_equation(terms=((dofs1[2],1.),(dofs2[2],-1e14)))
    model.apply_boundary_conditions()
    _,_,T,u0,*_=build_constraint_transformation(sparse.eye(24,format='csr'),np.zeros(24),model)
    with pytest.raises(ValueError,match='in-plane'):_admit_free_shell_coordinates(model,T,u0)
    # A rotated plane permits only its own physical normal, represented by MPCs.
    model,_,_=model_data();model.boundary_conditions.clear()
    model.add_boundary_condition(BoundaryCondition('normal',[1,2,3,4],
        {name:0. for name in ('uy','rx','ry','rz')}))
    c=np.sqrt(.5)
    for node in model.mesh.nodes.values():
        x,z=node.x,node.z;node.x=c*(x+z);node.z=c*(-x+z)
        dofs=model.mesh.dof_manager.get_node_dofs(node.id)
        model.add_constraint_equation(terms=((dofs[0],1.),(dofs[2],-1.)))
    model.apply_boundary_conditions()
    _,_,T,u0,*_=build_constraint_transformation(sparse.eye(24,format='csr'),np.zeros(24),model)
    _admit_free_shell_coordinates(model,T,u0)


def test_free_sphere_and_independent_shell_normal_modes_remain_coupled():
    model,definition,settings=model_data()
    dofs=[model.mesh.dof_manager.get_node_dofs(i)[2] for i in model.mesh.nodes]
    velocity=np.zeros(model.mesh.dof_manager.total_dofs)
    velocity[dofs]=(-.2,-.12,-.2,-.12)
    result=solve_transient_contact(model,definition,TransientConfig(.0005,.02,initial_velocity=velocity),
        contact_settings=settings,solver_controls=CONTROLS,free_rigid_bodies=(BODY,))
    weights=np.full(4,.001/4)
    initial=weights@velocity[dofs]+BODY.mass_kg*.2
    assert np.max(np.ptp(result.displacements[:,dofs],axis=1))>1e-5
    assert any(contact.active_pressure_dofs for contact in result.contact)
    for i,entry in enumerate(result.free_rigid_bodies):
        body=entry['sphere']
        assert weights@result.velocities[i,dofs]+body.linear_momentum_kg_m_s[2]==pytest.approx(initial,abs=1e-10)
        np.testing.assert_allclose(np.asarray(result.contact[i].shell_force)+body.contact_force_n,0.,atol=1e-12,rtol=0.)


def independent_two_body(times,history=None):
    """Continuum quadrature of the symmetric nodal-pressure potential.

    This uses neither shell/contact kernels nor their quadrature. The centred
    fixture has four equal pressure supports and uniform normal motion.
    """
    points,weights=np.polynomial.legendre.leggauss(24)
    x,y=np.meshgrid(.004*points,.004*points,indexing='ij')
    w=np.outer(weights,weights)/4
    area=.008**2;mass=np.array((area*.002*7812.5,BODY.mass_kg))
    def external(time):
        return 0. if history is None else -.001*np.interp(time,
            [p[0] for p in history.points],[p[1] for p in history.points])
    q=np.zeros(2);v=np.array((-.2,.2));acceleration=np.array((external(times[0])/mass[0],0.))
    positions=[q.copy()];velocities=[v.copy()];forces=[0.]
    def force(distance):
        radial=np.sqrt(distance**2+x*x+y*y)
        gap=float(np.sum(w*radial))-.02
        if gap>=0:return 0.,0.
        derivative=float(np.sum(w*distance/radial))
        second=float(np.sum(w*(x*x+y*y)/radial**3))
        return -1e6*area*gap*derivative,-1e6*area*(derivative**2+gap*second)
    for index,step in enumerate(np.diff(times),1):
        predictor=q+step*v+.25*step**2*acceleration
        vp=v+.5*step*acceleration;candidate_acceleration=np.zeros(2)
        for _ in range(20):
            candidate=predictor+.25*step**2*candidate_acceleration
            f,k=force(.024+candidate[0]-candidate[1])
            residual=mass*candidate_acceleration+np.array((-f-external(times[index]),f))
            if np.linalg.norm(residual)<1e-12:break
            jacobian=np.diag(mass)-.25*step**2*k*np.array(((1.,-1.),(-1.,1.)))
            candidate_acceleration+=np.linalg.solve(jacobian,-residual)
        else:raise AssertionError('independent shell/sphere Newmark did not converge')
        q=candidate;v=vp+.5*step*candidate_acceleration;acceleration=candidate_acceleration
        positions.append(q.copy());velocities.append(v.copy());forces.append(f)
    return np.array(positions),np.array(velocities),np.array(forces)


def test_free_shell_sphere_load_history_work_and_restart_settings(tmp_path):
    import json,os
    from pathlib import Path
    history=ContactNodalLoadHistory(((0.,0.),(.004,1.),(.025,.5)))
    model,direct=impact(history=history)
    position,velocity,force=independent_two_body(direct.times,history)
    dofs=[model.mesh.dof_manager.get_node_dofs(node)[2] for node in model.mesh.nodes]
    np.testing.assert_allclose(direct.displacements[:,dofs],np.broadcast_to(position[:,0,None],direct.displacements[:,dofs].shape),atol=2e-8,rtol=0.)
    np.testing.assert_allclose(direct.velocities[:,dofs],np.broadcast_to(velocity[:,0,None],direct.velocities[:,dofs].shape),atol=2e-6,rtol=0.)
    bodies=[entry['sphere'] for entry in direct.free_rigid_bodies]
    np.testing.assert_allclose([body.centre_of_mass_m[2] for body in bodies],position[:,1],atol=2e-8,rtol=0.)
    np.testing.assert_allclose([contact.shell_force[2] for contact in direct.contact],force,atol=1e-6,rtol=0.)
    assert direct.profile==FREE_SHELL_SPHERE_HISTORY_PROFILE
    assert .004 in direct.times
    assert any(contact.active_pressure_dofs for contact in direct.contact)
    assert direct.contact[-1].active_pressure_dofs==0
    work=0.;impulse=0.
    points,weights=np.polynomial.legendre.leggauss(6)
    fractions=(points+1)/2;weights=weights/2
    for index,step in enumerate(np.diff(direct.times),1):
        q0,q1=position[index-1,0],position[index,0]
        v0,v1=velocity[index-1,0],velocity[index,0]
        # Independent polynomial force-times-velocity integral. No production work helper.
        dq=v0*step+2*(-3*q0+3*q1-2*step*v0-step*v1)*fractions+3*(2*q0-2*q1+step*v0+step*v1)*fractions**2
        f0=-.001*np.interp(direct.times[index-1],[p[0] for p in history.points],[p[1] for p in history.points])
        f1=-.001*np.interp(direct.times[index],[p[0] for p in history.points],[p[1] for p in history.points])
        work+=float(weights@((f0+(f1-f0)*fractions)*dq))
        assert direct.external_nodal_work_j[index]==pytest.approx(work,abs=1e-12)
        impulse+=.5*step*(f0+f1)
        total=.001*float(np.mean(direct.velocities[index,dofs]))+bodies[index].linear_momentum_kg_m_s[2]
        assert total==pytest.approx(-.001*.2+BODY.mass_kg*.2+impulse,abs=1e-10)
    _,prefix=impact(end=.014,history=history,emit=True)
    assert prefix.contact[-1].active_pressure_dofs>0
    _,resumed=impact(history=history,checkpoint=prefix.restart_checkpoint)
    for field in ('times','displacements','velocities','external_nodal_work_j'):
        np.testing.assert_allclose(np.concatenate((getattr(prefix,field),getattr(resumed,field)[1:])),getattr(direct,field),atol=1e-10,rtol=0.)
    combined=prefix.free_rigid_bodies+resumed.free_rigid_bodies[1:]
    state_error={}
    for field in bodies[0].__dataclass_fields__:
        actual=np.asarray([getattr(entry['sphere'],field) for entry in combined])
        expected=np.asarray([getattr(body,field) for body in bodies])
        state_error[field]=float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0.,err_msg=field)
    for changed in (replace(CONTROLS,beam_swept_gap_tolerance_m=2e-10),
                    replace(CONTROLS,maximum_beam_swept_subdivisions=511)):
        with pytest.raises(ValueError,match='incompatible|fingerprint'):
            impact(history=history,checkpoint=prefix.restart_checkpoint,controls=changed)
    with pytest.raises(ValueError,match='incompatible|fingerprint'):
        impact(history=ContactNodalLoadHistory(((0.,0.),(.004,.9),(.025,.5))),checkpoint=prefix.restart_checkpoint)
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))
    record=dict(profile=direct.profile,steps=len(direct.times)-1,cutbacks=direct.cutbacks,
        external_work_j=work,external_impulse_n_s=impulse,
        position_error_m=float(np.max(np.abs(direct.displacements[:,dofs[0]]-position[:,0]))),
        force_error_n=float(np.max(np.abs(np.array([value.shell_force[2] for value in direct.contact])-force))),
        restart_samples=len(resumed.times),restart_state_absolute_errors=state_error)
    (folder/'free_shell_sphere_history_result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


def independent_energy_defect(model,result):
    """Independent continuum potential; spin checked but excluded from error scale."""
    dofs=[model.mesh.dof_manager.get_node_dofs(i)[2] for i in model.mesh.nodes]
    points,weights=np.polynomial.legendre.leggauss(24)
    x,y=np.meshgrid(.004*points,.004*points,indexing='ij')
    weights=np.outer(weights,weights)/4
    initial_energy=.5*.001*.2**2+.5*BODY.mass_kg*.2**2
    spin=.5*BODY.isotropic_inertia_kg_m2*np.dot(
        BODY.initial_angular_velocity_rad_s,BODY.initial_angular_velocity_rad_s)
    energies=[]
    for i,entry in enumerate(result.free_rigid_bodies):
        body=entry['sphere']
        distance=.024+float(np.mean(result.displacements[i,dofs]))-body.centre_of_mass_m[2]
        gap=float(np.sum(weights*np.sqrt(distance**2+x*x+y*y)))-.02
        potential=.5*1e6*.008**2*min(gap,0.)**2
        assert result.contact[i].penalty_energy==pytest.approx(potential,abs=1e-10)
        translational=.5*BODY.mass_kg*np.dot(body.velocity_m_s,body.velocity_m_s)
        assert body.kinetic_energy_j-translational==pytest.approx(spin,abs=1e-14)
        energies.append(.5*.001*float(np.mean(result.velocities[i,dofs]))**2
                        +body.kinetic_energy_j-spin+potential)
    return float(np.max(np.abs(np.asarray(energies)-initial_energy))/initial_energy)


def test_coupled_q4_sphere_impact_keeps_both_body_momentum_and_restart(tmp_path):
    import json,os
    from pathlib import Path
    model,result=impact()
    dofs=[model.mesh.dof_manager.get_node_dofs(i)[2] for i in model.mesh.nodes]
    shell_mass=.008**2*.002*7812.5
    momentum=shell_mass*(-.2)+BODY.mass_kg*.2
    body=[entry['sphere'] for entry in result.free_rigid_bodies]
    reference_q,reference_v,reference_force=independent_two_body(result.times)
    np.testing.assert_allclose(result.displacements[:,dofs[0]],reference_q[:,0],atol=2e-8,rtol=0.)
    np.testing.assert_allclose([state.centre_of_mass_m[2] for state in body],reference_q[:,1],atol=2e-8,rtol=0.)
    np.testing.assert_allclose([state.velocity_m_s[2] for state in body],reference_v[:,1],atol=2e-6,rtol=0.)
    np.testing.assert_allclose([value.shell_force[2] for value in result.contact],reference_force,atol=1e-6,rtol=0.)
    for i,state in enumerate(body):
        shell_v=float(np.mean(result.velocities[i,dofs]))
        assert shell_mass*shell_v+state.linear_momentum_kg_m_s[2]==pytest.approx(momentum,abs=1e-10)
        np.testing.assert_allclose(np.asarray(result.contact[i].shell_force)+state.contact_force_n,0.,atol=1e-12,rtol=0.)
        if i:
            step=result.times[i]-result.times[i-1]
            impulse=.5*step*(np.asarray(body[i-1].contact_force_n)+state.contact_force_n)
            np.testing.assert_allclose(BODY.mass_kg*(np.asarray(state.velocity_m_s)-body[i-1].velocity_m_s),impulse,atol=1e-10,rtol=0.)
    assert result.profile==FREE_SHELL_SPHERE_PROFILE
    assert any(value.active_pressure_dofs for value in result.contact)
    assert result.contact[-1].active_pressure_dofs==0
    assert min(result.swept_minimum_gaps)>=-.002
    _,first=impact(end=.014,emit=True)
    _,resumed=impact(checkpoint=first.restart_checkpoint)
    indices=[int(np.argmin(np.abs(result.times-time))) for time in resumed.times]
    np.testing.assert_allclose(resumed.displacements,result.displacements[indices],atol=1e-10,rtol=0.)
    np.testing.assert_allclose([x['sphere'].centre_of_mass_m for x in resumed.free_rigid_bodies],
        [body[i].centre_of_mass_m for i in indices],atol=1e-10,rtol=0.)
    with pytest.raises(ValueError,match='incompatible|fingerprint'):
        impact(checkpoint=first.restart_checkpoint,body=replace(BODY,mass_kg=.003))
    # Prospective development criterion: <=1% and improvement on halving dt.
    # Reuse this coarse coupled route rather than repeating it for accounting.
    coarse_defect=independent_energy_defect(model,result)
    fine_model,fine=impact(dt=.00025)
    fine_defect=independent_energy_defect(fine_model,fine)
    assert fine_defect<coarse_defect
    assert fine_defect<=.01
    folder=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)));folder.mkdir(parents=True,exist_ok=True)
    record=dict(profile=result.profile,steps=len(result.times)-1,cutbacks=result.cutbacks,
        minimum_swept_gap=float(min(result.swept_minimum_gaps)),
        final_velocity=list(body[-1].velocity_m_s),restart_samples=len(resumed.times),
        energy_defect_coarse=coarse_defect,energy_defect_fine=fine_defect,
        independent_position_maximum_error_m=float(np.max(np.abs(result.displacements[:,dofs[0]]-reference_q[:,0]))),
        independent_force_maximum_error_n=float(np.max(np.abs(np.array([value.shell_force[2] for value in result.contact])-reference_force))))
    (folder/'free_shell_sphere_result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


@pytest.mark.parametrize('cancel',(False,True))
def test_free_shell_sphere_trial_exception_keeps_definition_and_model_reusable(monkeypatch,cancel):
    from anyfesolver.general_contact.core import active_contact_manager
    from anyfesolver.control import CancellationToken,SolveCancelled
    token=CancellationToken()
    model,definition,settings=model_data()
    initial=np.zeros(model.mesh.dof_manager.total_dofs)
    dofs=[model.mesh.dof_manager.get_node_dofs(node)[2] for node in model.mesh.nodes]
    initial[dofs]=-.2
    def solve(cancel_token=None):
        return solve_transient_contact(model,definition,TransientConfig(.0005,.003,initial_velocity=initial),
            contact_settings=settings,solver_controls=CONTROLS,free_rigid_bodies=(BODY,),
            cancellation_token=cancel_token)
    original=_SphereShellManager.evaluate
    def failing(owner,displacement,**kwargs):
        result=original(owner,displacement,**kwargs)
        if np.any(displacement):
            if cancel:token.cancel()
            else:raise ArithmeticError('owned candidate failure')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(_SphereShellManager,'evaluate',failing)
        with pytest.raises(SolveCancelled if cancel else ArithmeticError):solve(token)
    assert active_contact_manager() is None
    assert definition.rigid_bodies[0].point==(0.,0.,0.)
    reused=solve();_,fresh=impact(end=.003)
    np.testing.assert_array_equal(reused.displacements,fresh.displacements)
    np.testing.assert_array_equal(reused.velocities,fresh.velocities)
    assert reused.free_rigid_bodies==fresh.free_rigid_bodies
