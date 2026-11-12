"""Coupled development checks; references use physical masses and scalar law."""
import json,os
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from examples.general_contact_beam_s3_transient import build,run
from anyfesolver.general_contact.beam_s3_transient import BeamS3TransientManager,hermite_gap_bounds
from anyfesolver.general_contact.transient import ContactTransientSolverControls


def record(data):
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):
        import inspect
        path=Path(os.environ['CONTACT_APP_RESULT_RECORD'])
        path.write_text(json.dumps(data,indent=2))
        path.with_name(inspect.stack()[1].function+'.json').write_text(json.dumps(data,indent=2))


def reference(times):
    mass=np.diag((.0005,.09));D=np.array((1.,-1.));K=50000.;g0=.00005
    q=np.zeros(2);v=np.array((-.04,.01));a=np.zeros(2);qs=[q.copy()];vs=[v.copy()]
    for h in np.diff(times):
        p=q+h*v+.25*h*h*a
        qn=p if g0+D@p>=0 else np.linalg.solve(mass+.25*h*h*K*np.outer(D,D),mass@p-.25*h*h*K*g0*D)
        an=(qn-p)/(.25*h*h);v=v+.5*h*(a+an);q=qn;a=an
        qs.append(q.copy());vs.append(v.copy())
    return np.array(qs),np.array(vs)


def body_balance(model,result):
    """Recover each body from actual inertia, constraints and structural work."""
    from anyfesolver.assembly import assemble_mass_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.general_contact.runtime import suspend_contact
    M,_=assemble_mass_matrix(model);errors=[]
    for index in (0,int(np.argmin([x.minimum_gap for x in result.contact])),len(result.times)-1):
        u=result.displacements[index];inertia=np.asarray(M@result.accelerations[index]).reshape(-1)
        with suspend_contact():internal,_,_= _assemble_nonlinear_system(model,u,{},3,tangent=False,kinematics='von_karman')
        for nodes,force,moment in ((tuple(model.mesh.elements[1].node_ids),result.contact[index].beam_force,result.contact[index].beam_moment),((3,4,5),result.contact[index].shell_force,result.contact[index].shell_moment)):
            w=np.r_[force,moment].copy()
            for node in nodes:
                dofs=model.mesh.dof_manager.get_node_dofs(node)
                position=model.mesh.nodes[node].coords()+u[dofs[:3]]
                vector=result.constraint_reactions[index,dofs]-inertia[dofs]-internal[dofs]
                w[:3]+=vector[:3];w[3:]+=vector[3:]+np.cross(position,vector[:3])
            errors.append((float(np.linalg.norm(w[:3])/.5),float(np.linalg.norm(w[3:])/.25)))
    return errors


def test_actual_two_body_impact_reference_restart_and_export():
    m,r=run();d=lambda node:m.mesh.dof_manager.get_node_dofs(node)[2]
    q,v=reference(r.times);actual=r.displacements[:,[d(1),d(3)]];vel=r.velocities[:,[d(1),d(3)]]
    displacement_error=float(np.max(abs(actual-q))/.00005)
    velocity_error=float(np.max(abs(vel-v))/.05)
    energy=.5*.0005*vel[:,0]**2+.5*.09*vel[:,1]**2+np.array([x.line_contact_energy for x in r.contact])
    energy_error=float(np.max(abs(energy-energy[0]))/energy[0])
    momentum_error=float(np.max(abs(.0005*vel[:,0]+.09*vel[:,1]-(.0005*(-.04)+.09*.01)))/(.0005*.05))
    active=sum(x.active_line_length>0 for x in r.contact)
    force_error=max(np.linalg.norm(np.array(x.beam_force)+x.shell_force)/.5 for x in r.contact)
    moment_error=max(np.linalg.norm(np.array(x.beam_moment)+x.shell_moment)/.25 for x in r.contact)
    _,prefix=run(end=float(r.times[115]));_,resumed=run(restart=prefix.restart_checkpoint)
    np.testing.assert_array_equal(resumed.displacements,r.displacements[len(prefix.times)-1:])
    assert resumed.restart_checkpoint==r.restart_checkpoint
    from anyfesolver.general_contact.transient_results_archive import read_transient_contact_results
    archive=read_transient_contact_results(r.to_archive_bytes())
    assert archive.to_dict()['contact'][0]['record_type'].endswith('DeformableBeamS3Results')
    report=dict(states=len(r.times),active=active,displacement_error=displacement_error,velocity_error=velocity_error,energy_error=energy_error,momentum_error=momentum_error,
        force_error=force_error,moment_error=moment_error,minimum_gap=min(x.minimum_gap for x in r.contact),final_gap=r.contact[-1].minimum_gap,restart_equal=True)
    report['body_balance_errors']=body_balance(m,r)
    record(report)
    assert displacement_error<=1.e-6 and velocity_error<=1.e-6
    assert energy_error<=.01 and momentum_error<=1.e-6
    assert force_error<=1.e-6 and moment_error<=1.e-6
    assert active>0 and r.contact[-1].active_line_length==0
    assert report['minimum_gap']>=-1.e-5
    assert np.max(report['body_balance_errors'])<=1.e-6


def test_b3_complete_work_and_hidden_path_refusals():
    m,d,s,_=build(quadratic=True,unequal=True);manager=BeamS3TransientManager(m,d,s)
    n=m.mesh.dof_manager.total_dofs;u=np.zeros(n);axis=lambda node:m.mesh.dof_manager.get_node_dofs(node)[2]
    u[axis(1)]=u[axis(2)]=-.00006;u[axis(6)]=-.00007;u[axis(5)]=.000005
    residual,K,summary=manager.evaluate(u)
    direction=np.zeros(n);direction[axis(6)]=.3;direction[axis(4)]=-.2
    derivative_errors=[]
    for h in (1.e-7,1.e-8,1.e-9):
        rp,_,sp=manager.evaluate(u+h*direction);rm,_,sm=manager.evaluate(u-h*direction)
        H_error=float(np.linalg.norm((rp-rm)/(2*h)-K@direction)/np.linalg.norm(K@direction))
        work_error=float(abs((sp.line_contact_energy-sm.line_contact_energy)/(2*h)-residual@direction)/abs(residual@direction))
        derivative_errors.append(dict(h=h,tangent_error=H_error,work_error=work_error))
    record(dict(derivative_errors=derivative_errors))
    assert max(max(x['tangent_error'],x['work_error']) for x in derivative_errors)<=1.e-5
    z=np.zeros(n);a=z.copy();b=z.copy();a[manager.beam_dofs[:,2]]=-.001;b[manager.beam_dofs[:,2]]=.001
    assert not hermite_gap_bounds(manager,z,z,a,b,1.,maximum_depth=12)[1]
    bad=z.copy();bad[manager.shell_dofs[0,3]]=1.e-20
    with pytest.raises(ValueError,match='rotations'):hermite_gap_bounds(manager,z,z,bad,z,1.)
    target_rate=z.copy();target_rate[manager.shell_dofs[1,0]]=-12.
    assert not hermite_gap_bounds(manager,z,z,target_rate,-target_rate,1.,maximum_depth=8)[1]
    slide=z.copy();slide[manager.beam_dofs[:,0]]=10.
    assert not hermite_gap_bounds(manager,z,slide,z,z,1.,maximum_depth=4)[1]
    manager.kernel.stiffness=0.
    with pytest.raises(ValueError,match='mutated'):manager.evaluate(z)
    manager=BeamS3TransientManager(m,d,s)
    manager.shell.work_map.data[0]+=1.
    with pytest.raises(ValueError,match='mutated'):manager.evaluate(z)


def test_actual_unequal_target_motion_and_admission():
    m,r=run(unequal=True,end=.0015)
    ids=[m.mesh.dof_manager.get_node_dofs(i)[2] for i in (3,4,5)]
    assert np.ptp(r.displacements[-1,ids])>1.e-7
    assert any(x.active_line_length>0 for x in r.contact)
    assert min(x.minimum_gap for x in r.contact)>=-1.e-5
    assert max(np.linalg.norm(np.array(x.beam_force)+x.shell_force)/.5 for x in r.contact)<=1.e-6
    assert max(np.linalg.norm(np.array(x.beam_moment)+x.shell_moment)/.25 for x in r.contact)<=1.e-6
    errors=body_balance(m,r);assert np.max(errors)<=1.e-6
    record(dict(unequal_target_displacements=r.displacements[-1,ids].tolist(),minimum_gap=min(x.minimum_gap for x in r.contact),body_balance_errors=errors))
    with pytest.raises(ValueError):replace(ContactTransientSolverControls(),allow_deformable_beam_s3_dev=1)


def test_temporal_and_enforcement_reference_sensitivity():
    errors=[]
    for dt,k in ((1.e-5,1.e5),(5.e-6,1.e5),(5.e-6,2.e5)):
        m,r=run(dt=dt,line_stiffness=k);d=lambda i:m.mesh.dof_manager.get_node_dofs(i)[2]
        mass=np.array((.0005,.09));omega=np.sqrt(k*.5*np.sum(1/mass));hit=.001;release=hit+np.pi/omega
        relative=np.where(r.times<hit,.00005-.05*r.times,np.where(r.times<=release,-.05/omega*np.sin(omega*(r.times-hit)),.05*(r.times-release)))
        gap=.00005+r.displacements[:,d(1)]-r.displacements[:,d(3)]
        error=float(np.max(abs(gap-relative))/.00005)
        vel=r.velocities[:,[d(1),d(3)]]
        energy=.5*np.sum(mass*vel**2,axis=1)+np.array([x.line_contact_energy for x in r.contact])
        e=float(np.max(abs(energy-energy[0]))/energy[0]);assert e<=.01
        errors.append(dict(dt=dt,line_stiffness=k,omega=omega,contact_half_period=np.pi/omega,gap_error=error,energy_error=e))
    record(dict(sensitivity=errors))
    assert errors[1]['gap_error']<errors[0]['gap_error']


@pytest.mark.parametrize('failure',('cancel','exception','retry'))
def test_transaction_boundaries_and_restart_refusal(monkeypatch,failure):
    import anyfesolver.general_contact.transient as driver
    import anyfesolver.general_contact.beam_s3_transient as path
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    from anyfesolver.dynamics import TransientConfig
    m,d,s,v=build();token=CancellationToken();commits=[];original=driver.commit_state_candidate;calls=0
    def commit(origin,trial):
        commits.append(1)
        return original(origin,trial)
    bound=path.hermite_gap_bounds
    def checked(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:
            if failure=='cancel':token.cancel('owned beam/S3 acceptance');kwargs['check_cancel']()
            if failure=='exception':raise ArithmeticError('owned beam/S3 path')
            if failure=='retry':return -float('inf'),False,1
        return bound(*args,**kwargs)
    monkeypatch.setattr(driver,'commit_state_candidate',commit);monkeypatch.setattr(path,'hermite_gap_bounds',checked)
    kwargs=dict(contact_settings=s,solver_controls=ContactTransientSolverControls(allow_deformable_beam_s3_dev=True),cancellation_token=token,emit_restart_checkpoint=True)
    if failure!='retry':
        with pytest.raises(SolveCancelled if failure=='cancel' else ArithmeticError):driver.solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),**kwargs)
        assert len(commits)==3
    else:
        kwargs['emit_restart_checkpoint']=False
        r=driver.solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),**kwargs)
        assert r.cutbacks>=1 and len(r.times)>2
        monkeypatch.setattr(path,'hermite_gap_bounds',bound)
        m,d,s,v=build()
        r=driver.solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),**dict(kwargs,emit_restart_checkpoint=True))
        changed=replace(s,beam_shell_rim_clearance=.03)
        with pytest.raises(ValueError,match='incompatible'):
            driver.solve_transient_contact(m,d,TransientConfig(1.e-5,2.e-4,initial_velocity=v),contact_settings=changed,solver_controls=kwargs['solver_controls'],restart_checkpoint=r.restart_checkpoint)
    assert active_contact_manager() is None


def test_exact_zero_target_rates_and_historical_contract():
    from anyfesolver.general_contact.transient import solve_transient_contact
    from anyfesolver.dynamics import TransientConfig
    m,d,s,v=build();v[m.mesh.dof_manager.get_node_dofs(3)[3]]=1.e-20
    with pytest.raises(ValueError,match='exactly zero'):
        solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),contact_settings=s,solver_controls=ContactTransientSolverControls(allow_deformable_beam_s3_dev=True))
    assert ContactTransientSolverControls().allow_deformable_beam_s3_dev is False


def test_actual_b3_public_and_explicit_rotation_refusals():
    from anyfesolver.general_contact.transient import solve_transient_contact
    from anyfesolver.dynamics import TransientConfig
    m,r=run(quadratic=True)
    assert any(x.active_line_length>0 for x in r.contact) and r.contact[-1].active_line_length==0
    assert np.max(body_balance(m,r))<=1.e-6
    record(dict(states=len(r.times),active=sum(x.active_line_length>0 for x in r.contact),minimum_gap=min(x.minimum_gap for x in r.contact),body_balance_errors=body_balance(m,r)))
    m,d,s,v=build();u=np.zeros_like(v);u[m.mesh.dof_manager.get_node_dofs(3)[3]]=1.e-20
    with pytest.raises(ValueError,match='exactly zero'):
        solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v,initial_displacement=u),contact_settings=s,solver_controls=ContactTransientSolverControls(allow_deformable_beam_s3_dev=True))
    for flag in ('allow_multiple_beam_planes_dev','s3_algebraic_path_dev','s3_event_subdivision_dev','allow_classical_beam_dev','allow_guided_shell_pair_dev'):
        with pytest.raises(ValueError):ContactTransientSolverControls(allow_deformable_beam_s3_dev=True,**{flag:True})
    from anyfesolver import BoundaryCondition
    m,d,s,v=build();m.clear_boundary_conditions()
    m.add_boundary_condition(BoundaryCondition('inplane_only',list(m.mesh.nodes),dict.fromkeys(('ux','uy','rz'),0.)))
    with pytest.raises(ValueError,match='positive definite|constrained zero'):
        solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),contact_settings=s,solver_controls=ContactTransientSolverControls(allow_deformable_beam_s3_dev=True))


def test_legacy_archive_wire_is_exact():
    import hashlib,importlib.util,sys
    from anyfesolver.general_contact import transient_results_archive as current
    from test_general_contact_transient_results_archive import sample
    baseline=os.environ.get('C6_LEGACY_CODEC')
    if not baseline:pytest.skip('protected baseline evidence is opt-in/local')
    path=Path(baseline);assert path.is_file()
    spec=importlib.util.spec_from_file_location('anyfesolver.general_contact._legacy_c6_archive',path)
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy
    try:
        spec.loader.exec_module(legacy);value=sample();before=legacy.serialize_transient_contact_results(value);after=current.serialize_transient_contact_results(value)
        record(dict(baseline_path=str(path),baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),byte_equal=before==after,wire_sha256=hashlib.sha256(after).hexdigest()))
        assert before==after
    finally:sys.modules.pop(spec.name,None)
