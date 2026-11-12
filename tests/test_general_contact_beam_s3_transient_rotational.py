"""Independent affine-rotation line work and physical descriptor impact."""
import json,os
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from examples.general_contact_beam_s3_transient_rotational import build,run
from anyfesolver.general_contact.beam_s3_rotational_transient import RotationalBeamS3Manager
from anyfesolver.general_contact.beam_s3_transient import hermite_gap_bounds
from test_general_contact_beam_s3_transient_flexural import independent_beam_storage,independent_shell_coefficients,independent_mass_chart


def record(data):
    if os.environ.get('CONTACT_APP_RESULT_RECORD'):
        p=Path(os.environ['CONTACT_APP_RESULT_RECORD'])
        p.with_name('rotational_'+data['check']+'.json').write_text(json.dumps(data,indent=2))
        p.write_text(json.dumps(data,indent=2))


def contact_oracle(model,u):
    """Direct affine skin/Q2 polynomial, no production position or kernel call."""
    dof=lambda node:model.mesh.dof_manager.get_node_dofs(node)
    beam=np.array([model.mesh.nodes[n].coords()+u[dof(n)[:3]] for n in (1,6,2)])
    skin=np.array([model.mesh.nodes[n].coords()+u[dof(n)[:3]]+.01*(np.array((0.,0.,1.))+np.cross(u[dof(n)[3:]],(0.,0.,1.))) for n in (3,4,5)])
    n=np.cross(skin[1]-skin[0],skin[2]-skin[0]);n/=np.linalg.norm(n)
    c=np.array((n@(beam[0]-skin[0])-.01,n@(-3*beam[0]+4*beam[1]-beam[2]),n@(2*beam[0]-4*beam[1]+2*beam[2])))
    cuts=sorted([0.,1.]+[float(x.real) for x in np.polynomial.polynomial.polyroots(c) if abs(x.imag)<1.e-12 and 0<x.real<1])
    square=np.polynomial.polynomial.polymul(c,c);primitive=np.r_[0.,square/np.arange(1,len(square)+1)]
    integral=sum(np.polynomial.polynomial.polyval(b,primitive)-np.polynomial.polynomial.polyval(a,primitive) for a,b in zip(cuts,cuts[1:]) if np.polynomial.polynomial.polyval((a+b)/2,c)<0)
    return .5*1.e5*.5*integral


def energy(model,result):
    dof=lambda n:model.mesh.dof_manager.get_node_dofs(n)
    selected=[dof(n)[axis] for n in (1,6,2) for axis in (2,4)]+[dof(n)[2] for n in (3,4,5)]
    mass=independent_mass_chart()
    gradients,coefficient,_=independent_shell_coefficients(model)
    rows=np.array([model.mesh.dof_manager.get_node_dofs(n) for n in (3,4,5)])
    data=[]
    for u,v,summary in zip(result.displacements,result.velocities,result.contact):
        slope=gradients.T@u[rows[:,2]]
        gamma=slope+np.array((u[rows[0,4]],-u[rows[0,3]]))
        shell=.5*coefficient*(gamma@gamma)
        data.append(.5*v[selected]@(mass@v[selected])+independent_beam_storage(model,u)[0]+shell+contact_oracle(model,u))
    return np.array(data)


def test_rotational_complete_work_and_hidden_target_path():
    m,d,s,_=build();manager=RotationalBeamS3Manager(m,d,s);u=np.zeros(m.mesh.dof_manager.total_dofs)
    u[manager.beam_dofs[:,2]]=-.00008;u[manager.shell_dofs[:,2]]=(0.,1.e-5,-2.e-5)
    u[manager.shell_dofs[:,3]]=2.e-4;u[manager.shell_dofs[:,4]]=-3.e-4
    f,K,summary=manager.evaluate(u);direction=np.zeros_like(u)
    direction[manager.shell_dofs[:,3]]=.2;direction[manager.shell_dofs[:,4]]=-.1;direction[manager.beam_dofs[1,2]]=.3
    errors=[]
    for h in (1.e-7,1.e-8,1.e-9):
        fp,_,_=manager.evaluate(u+h*direction);fm,_,_=manager.evaluate(u-h*direction)
        work=abs((contact_oracle(m,u+h*direction)-contact_oracle(m,u-h*direction))/(2*h)-f@direction)/abs(f@direction)
        tangent=np.linalg.norm((fp-fm)/(2*h)-K@direction)/np.linalg.norm(K@direction)
        errors.append(dict(h=h,work=float(work),tangent=float(tangent)))
    assert max(max(x['work'],x['tangent']) for x in errors)<=1.e-5
    z=np.zeros_like(u);rate=z.copy();rate[manager.shell_dofs[1,4]]=100.
    assert not hermite_gap_bounds(manager,z,z,rate,-rate,1.,maximum_depth=8)[1]
    assert np.linalg.norm(f[manager.shell_dofs[:,3:]])>0
    report=manager.wrench_report(u)
    expected=.01*np.cross(u[manager.shell_dofs[:,3:]],(0.,0.,1.))
    force=-manager.kernel.evaluate(*manager._positions(u)).gradient.reshape(-1,3)[3:]
    np.testing.assert_allclose(report['affine_moment_discrepancy'],np.cross(expected,force).sum(axis=0),atol=1.e-16)
    manager.shell.work_map.data[0]*=2
    with pytest.raises(ValueError,match='mutated'):manager.evaluate(u)
    record(dict(check='work',errors=errors,wrench={k:v.tolist() for k,v in report.items()}))


def test_actual_rotational_impact_opening_restart_and_temporal_energy():
    from anyfesolver.assembly import assemble_mass_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.general_contact.runtime import suspend_contact
    m,r=run();_,prefix=run(end=float(r.times[110]));_,resumed=run(restart=prefix.restart_checkpoint)
    assert resumed.restart_checkpoint==r.restart_checkpoint
    np.testing.assert_array_equal(resumed.displacements,r.displacements[len(prefix.times)-1:])
    from anyfesolver.general_contact.transient_results_archive import read_transient_contact_results
    archive=read_transient_contact_results(r.to_archive_bytes()).to_dict()
    assert archive['contact'][110]['fields']['shell_nodal_work_moment']==r.contact[110].shell_nodal_work_moment
    assert archive['contact'][110]['fields']['algebraic_acceleration_semantics']==r.contact[110].algebraic_acceleration_semantics
    assert any(x.active_line_length>0 for x in r.contact) and r.contact[-1].active_line_length==0
    ids=np.array([m.mesh.dof_manager.get_node_dofs(n) for n in (3,4,5)])
    assert np.max(abs(r.displacements[:,ids[:,3:5]]))>0 and np.max(abs(r.velocities[:,ids[:,3:5]]))>0
    M,_=assemble_mass_matrix(m);assert np.count_nonzero(M[ids[:,3:].ravel()].data)==0
    manager=RotationalBeamS3Manager(m,*build()[1:3]);algebraic=[];wrench=[];discrepancy=[];potential=[]
    for i in (0,int(np.argmin([c.minimum_gap for c in r.contact])),len(r.times)-1):
        u=r.displacements[i];f,_,summary=manager.evaluate(u)
        with suspend_contact():structural,_,_=_assemble_nonlinear_system(m,u,{},3,tangent=False,kinematics='von_karman')
        algebraic.append(float(np.linalg.norm((f+structural)[ids[:,3:5]].sum(axis=0))/.5))
        report=manager.wrench_report(u);discrepancy.append(float(np.linalg.norm(report['affine_moment_discrepancy'])/.25))
        for nodes,F,L in (((1,6,2),summary.beam_force,summary.beam_moment),((3,4,5),summary.shell_force,report['nodal_work_moment'])):
            balance=np.r_[F,L].copy();inertia=M@r.accelerations[i]
            for node in nodes:
                dofs=m.mesh.dof_manager.get_node_dofs(node);p=m.mesh.nodes[node].coords()+u[dofs[:3]]
                part=r.constraint_reactions[i,dofs]-inertia[dofs]-structural[dofs]
                balance[:3]+=part[:3];balance[3:]+=part[3:]+np.cross(p,part[:3])
            wrench.append((float(np.linalg.norm(balance[:3])/.5),float(np.linalg.norm(balance[3:])/.25)))
        potential.append(float(abs(contact_oracle(m,u)-summary.line_contact_energy)))
    dof=lambda n:m.mesh.dof_manager.get_node_dofs(n)
    selected=[dof(n)[axis] for n in (1,6,2) for axis in (2,4)]+[dof(n)[2] for n in (3,4,5)]
    np.testing.assert_allclose(M[np.ix_(selected,selected)].toarray(),independent_mass_chart(),rtol=1.e-14,atol=1.e-19)
    couples=[]
    for u in r.displacements:
        f,_,_=manager.evaluate(u,tangent=False);couples.append(-f[ids[:,3:5]])
    couples=np.array(couples)
    rotation_work=float(np.sum(.5*(couples[1:]+couples[:-1])*np.diff(r.displacements[:,ids[:,3:5]],axis=0)))
    E=energy(m,r);defect=float(np.max(abs(E-E[0]))/E[0])
    m2,r2=run(dt=5.e-6);E2=energy(m2,r2);fine=float(np.max(abs(E2-E2[0]))/E2[0])
    record(dict(check='impact',states=len(r.times),active=sum(c.active_line_length>0 for c in r.contact),minimum_gap=min(c.minimum_gap for c in r.contact),final_gap=r.contact[-1].minimum_gap,energy_defect=defect,fine_energy_defect=fine,algebraic=algebraic,work_wrench=wrench,affine_discrepancy=discrepancy,independent_contact_error=potential,restart_equal=True,max_target_rotation=float(np.max(abs(r.displacements[:,ids[:,3:5]]))),max_target_rotation_rate=float(np.max(abs(r.velocities[:,ids[:,3:5]]))),max_rotational_contact_couple=float(np.max(abs(couples))),rotational_contact_work_trapezoid_j=rotation_work))
    assert np.max(abs(couples))>0
    assert defect<=.01 and fine<=.01 and fine<defect
    assert max(algebraic)<=1.e-6 and np.max(wrench)<=1.e-6 and max(discrepancy)<=1.e-6
    assert min(c.minimum_gap for c in r.contact)>=-1.e-5


def test_rotational_secant_corruption_refused():
    import copy,hashlib,importlib.util,sys
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    from anyfesolver.dynamics import TransientConfig
    _,r=run(end=.0011)
    changed=copy.deepcopy(r.restart_checkpoint)
    # Shell rx is an actual declared zero-mass reduced coordinate.
    m,d,s,v=build();m.apply_boundary_conditions()
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.assembly import build_constraint_transformation
    K,_=assemble_stiffness_matrix(m)
    _,_,T,_,ids,_=build_constraint_transformation(K,np.zeros(len(v)),m)
    target=m.mesh.dof_manager.get_node_dofs(3)[3]
    index=list(ids).index(target)
    changed['state']['acceleration'][index]+=.1
    changed.pop('checkpoint_sha256')
    changed['checkpoint_sha256']=analysis_configuration_fingerprint(changed)
    with pytest.raises(ValueError,match='secant history'):
        run(restart=changed)


@pytest.mark.skipif(not os.environ.get('C6_LEGACY_TRANSIENT'),
                    reason='historical V17 producer artifact must be explicitly provisioned')
def test_legacy_v17_producer_checkpoint_resumes():
    import hashlib,importlib.util,sys
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anyfesolver.dynamics import TransientConfig
    legacy=Path(os.environ['C6_LEGACY_TRANSIENT'])
    name='anyfesolver.general_contact._legacy_v17_transient'
    spec=importlib.util.spec_from_file_location(name,legacy);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
        from examples.general_contact_beam_s3_transient import build as old_build
        oldm,oldd,olds,oldv=old_build()
        reference=module.solve_transient_contact(oldm,oldd,TransientConfig(1.e-5,1.e-4,initial_velocity=oldv),contact_settings=olds,
            solver_controls=module.ContactTransientSolverControls(allow_deformable_beam_s3_dev=True),emit_restart_checkpoint=True)
        oldm,oldd,olds,oldv=old_build()
        actual=solve_transient_contact(oldm,oldd,TransientConfig(1.e-5,2.e-4,initial_velocity=oldv),contact_settings=olds,
            solver_controls=ContactTransientSolverControls(allow_deformable_beam_s3_dev=True),restart_checkpoint=reference.restart_checkpoint,emit_restart_checkpoint=True)
        assert actual.restart_checkpoint['version']==17
        record(dict(check='historical_restart',legacy_path=str(legacy),legacy_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),legacy_v17_resumed=True))
    finally:sys.modules.pop(name,None)


def test_rotational_initial_rate_and_profile_refusal():
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anyfesolver.dynamics import TransientConfig
    m,d,s,v=build();v[m.mesh.dof_manager.get_node_dofs(3)[3]]+=.01
    with pytest.raises(ValueError,match='velocity|constraints'):
        solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),contact_settings=s,solver_controls=ContactTransientSolverControls(allow_rotational_beam_s3_dev=True))
    with pytest.raises(ValueError):ContactTransientSolverControls(allow_rotational_beam_s3_dev=True,allow_deformable_beam_s3_dev=True)


@pytest.mark.parametrize('failure',('cancel','exception','retry'))
def test_rotational_transactions(monkeypatch,failure):
    import anyfesolver.general_contact.transient as driver
    import anyfesolver.general_contact.beam_s3_transient as path
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    from anyfesolver.dynamics import TransientConfig
    token=CancellationToken();commits=[];calls=0
    original=driver.commit_state_candidate;bound=path.hermite_gap_bounds
    def commit(origin,trial):
        commits.append(1)
        return original(origin,trial)
    def checked(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:
            if failure=='cancel':token.cancel('rotational accepted prefix');kwargs['check_cancel']()
            if failure=='exception':raise ArithmeticError('rotational represented path')
            if failure=='retry':return -float('inf'),False,1
        return bound(*args,**kwargs)
    monkeypatch.setattr(driver,'commit_state_candidate',commit)
    monkeypatch.setattr(path,'hermite_gap_bounds',checked)
    m,d,s,v=build()
    kwargs=dict(contact_settings=s,solver_controls=driver.ContactTransientSolverControls(allow_rotational_beam_s3_dev=True),cancellation_token=token)
    if failure!='retry':
        with pytest.raises(SolveCancelled if failure=='cancel' else ArithmeticError):
            driver.solve_transient_contact(m,d,TransientConfig(1.e-5,5.e-5,initial_velocity=v),**kwargs)
        assert len(commits)==3
    else:
        result=driver.solve_transient_contact(m,d,TransientConfig(1.e-5,5.e-5,initial_velocity=v),**kwargs)
        assert result.cutbacks==1 and len(result.times)>2
    assert active_contact_manager() is None
