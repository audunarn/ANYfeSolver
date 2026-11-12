"""Actual loaded impact with independent mass/beam/MIN3/line potential."""
import json,os
from pathlib import Path
import numpy as np
from examples.general_contact_beam_s3_transient_directional_loaded_impact import run,prepare,independent_reference,KINETIC_SCALE,DT,END

def test_loaded_impact_coarse(tmp_path):
    m,r,initial=run(dt=DT,end=END)
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')))
    path.with_name('accepted_full.checkpoint.json').write_text(json.dumps(r.restart_checkpoint,sort_keys=True))
    energy=[];contact_errors=[]
    for u,v,c in zip(r.displacements,r.velocities,r.contact):
        ref,_,_=independent_reference(m,u,v);energy.append(ref['energy']);contact_errors.append(abs(ref['contact']-c.line_contact_energy))
    external=np.array(initial['load_forces']);work=(r.displacements-r.displacements[0])@external
    defect=float(np.max(abs(np.array(energy)-energy[0]-work))/KINETIC_SCALE)
    active=[i for i,c in enumerate(r.contact) if c.active_line_length>0 and c.minimum_gap<0]
    opened=[i for i,c in enumerate(r.contact) if c.active_line_length==0 and c.minimum_gap>0]
    report=dict(states=len(r.times),active=active,opened=opened,minimum_gap=min(c.minimum_gap for c in r.contact),final_gap=r.contact[-1].minimum_gap,energy_defect=defect,independent_contact_error=max(contact_errors),initial=initial,dt=DT,end=END,kinetic_scale=KINETIC_SCALE,reference=ref,independent_scope='Q2_VK_MIN3_LUMPED_S3_CONSISTENT_B3_AFFINE_SKIN_LINE')
    path.write_text(json.dumps(report,indent=2))
    assert active and opened and max(opened)>min(active)
    assert report['final_gap']>0 and r.contact[-1].active_line_length==0
    assert report['minimum_gap']>=-1.e-5
    assert defect<=.01
    assert max(contact_errors)<=1.e-14


def test_loaded_impact_refinement_equilibrium_restart(tmp_path):
    from anyfesolver.general_contact.beam_s3_directional_rates import EquilibratedDirectionalBeamS3Manager
    from anyfesolver.assembly import assemble_mass_matrix,build_constraint_transformation
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.nonlinear_state import discard_active_state_candidate
    from anyfesolver.general_contact.runtime import suspend_contact
    m,r,initial=run(dt=DT,end=END)
    fine_failure=None;traces=[]
    _,fine,_=run(dt=DT/2,end=END)
    _,d,s,u,v,load,_=prepare();manager=EquilibratedDirectionalBeamS3Manager(m,d,s,initial_displacement=u)
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')))
    path.with_name('coarse_full.checkpoint.json').write_text(json.dumps(r.restart_checkpoint,sort_keys=True))
    if fine is not None:path.with_name('fine_full.checkpoint.json').write_text(json.dumps(fine.restart_checkpoint,sort_keys=True))
    ids={n:m.mesh.dof_manager.get_node_dofs(n) for n in m.mesh.nodes}
    tr=np.array([ids[n][:3] for n in ids]).ravel();ro=np.array([ids[n][3:] for n in ids]).ravel()
    M,_=assemble_mass_matrix(m);K0,_=assemble_stiffness_matrix(m)
    _,_,T,_,independent,_=build_constraint_transformation(K0,np.zeros(len(u)),m)
    basis=build_declared_algebraic_basis(m,M,(T.T@M@T).tocsr(),T,independent,dense_size_limit=512);Z=basis.reduced_basis
    tc=np.asarray(abs(T[tr]).sum(axis=0)).ravel()>0;rc=np.asarray(abs(T[ro]).sum(axis=0)).ravel()>0
    assert not np.any(tc&rc)
    scales=np.where(tc,.5,.25);ext=np.array(initial['load_forces'])
    independent_gradient=[];equilibrium=[];null=[];wrench=[];ce=[];energies=[]
    def structural(q):
        origin={}
        try:
            with suspend_contact():f,_,_=_assemble_nonlinear_system(m,q,origin,3,tangent=False,kinematics='von_karman')
            return f
        finally:discard_active_state_candidate(origin)
    ref0,_,mass=independent_reference(m,r.displacements[0],r.velocities[0])
    expected=(T.T@mass@T);actual=(T.T@M@T).toarray()
    mass_error=float(np.max(abs(expected-actual))/np.max(abs(expected)))
    for j,(q,rate,acc,c) in enumerate(zip(r.displacements,r.velocities,r.accelerations,r.contact)):
        ref,rf,_=independent_reference(m,q,rate);energies.append(ref['energy']);ce.append(abs(ref['contact']-c.line_contact_energy))
        f=structural(q);fc,_,summary=manager.evaluate(q);inertia=M@acc
        independent_gradient.append(float(np.linalg.norm(np.asarray(T.T@(f-rf)).ravel()/scales)))
        reduced=np.asarray(T.T@(inertia+f+fc-ext)).ravel()
        equilibrium.append(float(np.linalg.norm(reduced/scales)));null.append(float(np.linalg.norm(Z.T@reduced)/.25))
        for nodes,F,H in (((1,6,2),summary.beam_force,summary.beam_moment),((3,4,5),summary.shell_force,summary.shell_nodal_work_moment)):
            balance=np.r_[F,H].copy()
            for n in nodes:
                dd=ids[n];position=m.mesh.nodes[n].coords()+q[dd[:3]]
                part=r.constraint_reactions[j,dd]+ext[dd]-inertia[dd]-f[dd]
                balance[:3]+=part[:3];balance[3:]+=part[3:]+np.cross(position,part[:3])
            wrench.append((float(np.linalg.norm(balance[:3])/.5),float(np.linalg.norm(balance[3:])/.25)))
    def energy_defect(result):
        e=np.array([independent_reference(m,q,rate)[0]['energy'] for q,rate in zip(result.displacements,result.velocities)])
        work=(result.displacements-result.displacements[0])@ext
        return float(np.max(abs(e-e[0]-work))/KINETIC_SCALE)
    coarse_defect=energy_defect(r);fine_defect=energy_defect(fine) if fine is not None else None
    active=[i for i,c in enumerate(r.contact) if c.active_line_length>0 and c.minimum_gap<0]
    opened=[i for i,c in enumerate(r.contact) if c.active_line_length==0 and c.minimum_gap>0]
    splits=[]
    for label,index in (('active',active[len(active)//2]),('open',opened[len(opened)//2])):
        accepted_time=float(r.times[index]);_,prefix,_=run(dt=DT,end=accepted_time);_,resume,_=run(dt=DT,end=END,restart=prefix.restart_checkpoint)
        path.with_name(label+'_prefix.checkpoint.json').write_text(json.dumps(prefix.restart_checkpoint,sort_keys=True))
        splits.append(dict(phase=label,index=index,time=accepted_time,exact_checkpoint=resume.restart_checkpoint==r.restart_checkpoint,exact_displacements=np.array_equal(resume.displacements,r.displacements[len(prefix.times)-1:]),prefix_gap=prefix.contact[-1].minimum_gap))
    # Complete physical both-body direction on a smooth active pose, not a touch derivative.
    q=r.displacements[active[len(active)//2]];direction=np.zeros_like(q)
    for n in (1,6,2):direction[ids[n][2]]=.2;direction[ids[n][4]]=.1
    for n in (3,4,5):direction[ids[n][2]]=-.1;direction[ids[n][3]]=.02;direction[ids[n][4]]=-.03
    f,H,c=manager.evaluate(q);work=[]
    for h in (1.e-7,1.e-8):
        fp,_,_=manager.evaluate(q+h*direction);fm,_,_=manager.evaluate(q-h*direction)
        ep=independent_reference(m,q+h*direction,np.zeros_like(q))[0]['contact'];em=independent_reference(m,q-h*direction,np.zeros_like(q))[0]['contact']
        de=(ep-em)/(2*h)-f@direction;dh=(fp-fm)/(2*h)-H@direction
        work.append(dict(h=h,scalar_work_error=float(abs(de)/.175),force_rate_error=float(np.max(abs(dh[tr]))/.5),moment_rate_error=float(np.max(abs(dh[ro]))/.25),relative_H_error=float(np.linalg.norm(dh)/np.linalg.norm(H@direction))))
    data=dict(coarse_states=len(r.times),fine_states=len(fine.times) if fine is not None else None,fine_failure=fine_failure,fine_newton_trace=traces,coarse_dt=DT,fine_dt=DT/2,end=END,kinetic_scale=KINETIC_SCALE,coarse_energy_defect=coarse_defect,fine_energy_defect=fine_defect,independent_mass_error=mass_error,independent_reduced_structural_gradient=max(independent_gradient),complete_reduced_equilibrium=max(equilibrium),null_equilibrium=max(null),body_wrench_errors=np.max(wrench,axis=0).tolist(),contact_energy_error=max(ce),coarse_min_gap=min(c.minimum_gap for c in r.contact),fine_min_gap=min(c.minimum_gap for c in fine.contact) if fine is not None else None,coarse_final_gap=r.contact[-1].minimum_gap,fine_final_gap=fine.contact[-1].minimum_gap if fine is not None else None,splits=splits,work=work,scope='DEVELOPMENT_NO_C2_DAE_OR_FINITE_ROTATION_CLAIM',shell_oracle_chart='TIED_UNIFORM_RX_RY_REDUCED_MIN3')
    path.write_text(json.dumps(data,indent=2))
    assert active and opened and r.contact[-1].minimum_gap>0
    assert data['coarse_min_gap']>=-1.e-5
    assert mass_error<=1.e-12 and max(independent_gradient)<=1.e-6
    assert max(equilibrium)<=1.e-6 and max(null)<=1.e-6 and np.max(wrench)<=1.e-6
    assert coarse_defect<=.01
    assert all(x['exact_checkpoint'] and x['exact_displacements'] for x in splits)
    assert all(max(x['scalar_work_error'],x['force_rate_error'],x['moment_rate_error'],x['relative_H_error'])<=1.e-5 for x in work)

    assert fine is not None, fine_failure
    assert fine.contact[-1].minimum_gap>0 and min(c.minimum_gap for c in fine.contact)>=-1.e-5
    assert fine_defect<=.01 and fine_defect<coarse_defect


def test_acceleration_selection_and_legacy_v20_restart(tmp_path):
    import pytest
    from anyfesolver.general_contact.transient import ContactTransientSolverControls
    for value in (1,'true',None):
        with pytest.raises(ValueError,match='boolean'):ContactTransientSolverControls(equilibrated_newmark_acceleration_unknown_dev=value)
    with pytest.raises(ValueError,match='requires'):ContactTransientSolverControls(equilibrated_newmark_acceleration_unknown_dev=True)
    from examples.general_contact_beam_s3_transient_directional_tilted import run as legacy_run
    _,old,_=legacy_run();_,default,_=run(end=3.e-5,acceleration_unknown=False)
    assert default.restart_checkpoint==old.restart_checkpoint
    _,resumed,_=run(end=4.e-5,restart=old.restart_checkpoint,acceleration_unknown=False)
    with pytest.raises(ValueError):run(end=4.e-5,restart=old.restart_checkpoint)
    _,new,_=run(end=3.e-5)
    with pytest.raises(ValueError):run(end=4.e-5,restart=new.restart_checkpoint,acceleration_unknown=False)
    protected=os.environ.get('C6_LOADED_LEGACY_CHECKPOINT')
    baseline_report=None
    if protected:
        import hashlib
        raw=Path(protected).read_bytes();capsule=json.loads(raw)
        _,unchanged,_=run(acceleration_unknown=False)
        assert unchanged.restart_checkpoint==capsule
        _,continuation,_=run(end=END+DT,restart=capsule,acceleration_unknown=False)
        # Historical default has its original endpoint rounding behavior. Compare
        # continuation to the protected driver, not a newly imposed exact split gate.
        import importlib.util,sys
        driver_path=Path(os.environ['C6_LOADED_LEGACY_DRIVER'])
        name='anyfesolver.general_contact._loaded_pre_acceleration_transient'
        spec=importlib.util.spec_from_file_location(name,driver_path);prior=importlib.util.module_from_spec(spec);sys.modules[name]=prior
        try:
            spec.loader.exec_module(prior)
            from anyfesolver.dynamics import TransientConfig
            mm,dd,ss,uu,vv,ll,_=prepare()
            previous=prior.solve_transient_contact(mm,dd,TransientConfig(DT,END+DT,initial_displacement=uu,initial_velocity=vv),base_load_case=ll,contact_settings=ss,
                solver_controls=prior.ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=True,save_constraint_reactions_dev=True),restart_checkpoint=capsule,emit_restart_checkpoint=True)
            assert continuation.restart_checkpoint==previous.restart_checkpoint
        finally:sys.modules.pop(name,None)
        baseline_report=dict(path=protected,sha256=hashlib.sha256(raw).hexdigest(),driver_path=str(driver_path),driver_sha256=hashlib.sha256(driver_path.read_bytes()).hexdigest(),exact_unchanged_default=True,exact_protected_continuation_behavior=True,historical_default_fresh_split_nonexact_preserved=True)
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'selection.json')))
    path.with_name('legacy_default_evidence.json').write_text(json.dumps(baseline_report,indent=2))
    path.write_text(json.dumps(dict(legacy_exact=True,legacy_resumed=len(resumed.times),opposite_policy_refused=True,exact_bool_refused=True),indent=2))


import pytest
@pytest.mark.parametrize("failure",("cancel","exception","retry"))
def test_acceleration_transaction_boundaries(monkeypatch,tmp_path,failure):
    import anyfesolver.general_contact.transient as driver
    import anyfesolver.general_contact.beam_s3_transient as path
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.runtime import active_contact_manager
    m,d,s,u,v,load,initial=prepare();bound=path.hermite_gap_bounds;commit=driver.commit_state_candidate;calls=0;commits=[];token=CancellationToken()
    def checked(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:
            if failure=='cancel':token.cancel('loaded descriptor acceptance');kwargs['check_cancel']()
            if failure=='exception':raise ArithmeticError('loaded descriptor path')
            return -float('inf'),False,1
        return bound(*args,**kwargs)
    def accepted(origin,trial):commits.append(1);return commit(origin,trial)
    monkeypatch.setattr(path,'hermite_gap_bounds',checked);monkeypatch.setattr(driver,'commit_state_candidate',accepted)
    controls=driver.ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=True,equilibrated_newmark_acceleration_unknown_dev=True)
    if failure!='retry':
        with pytest.raises(SolveCancelled if failure=='cancel' else ArithmeticError):driver.solve_transient_contact(m,d,TransientConfig(1.e-5,3.e-5,initial_displacement=u,initial_velocity=v),base_load_case=load,contact_settings=s,solver_controls=controls,cancellation_token=token)
        assert len(commits)==2
    else:
        result=driver.solve_transient_contact(m,d,TransientConfig(1.e-5,3.e-5,initial_displacement=u,initial_velocity=v),base_load_case=load,contact_settings=s,solver_controls=controls)
        assert result.cutbacks>=1
        import numpy as np
        np.testing.assert_array_equal(result.displacements[0],u)
    assert active_contact_manager() is None


def test_independent_loaded_physical_gap(tmp_path):
    # Frozen dimensional comparison; direct own skin/Q2 polynomial only.
    agreement_m=1.e-12;gate_m=1.e-5;rows=[]
    for dt in (DT,DT/2):
        m,r,initial=run(dt=dt,end=END)
        gaps=[independent_reference(m,u,v)[0]['independent_min_gap'] for u,v in zip(r.displacements,r.velocities)]
        reported=[c.minimum_gap for c in r.contact]
        rows.append(dict(dt=dt,states=len(gaps),independent_minimum_m=min(gaps),independent_final_gap_m=gaps[-1],reported_final_gap_m=reported[-1],maximum_agreement_error_m=float(np.max(abs(np.array(gaps)-reported))),final_active_line_length=float(r.contact[-1].active_line_length),independent_gap_history_m=gaps,reported_gap_history_m=reported))
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'gap_result.json')))
    path.write_text(json.dumps(dict(agreement_m=agreement_m,physical_gate_m=gate_m,oracle='TRUE_ENDPOINTS_AND_STATIONARY_POINT_OF_DIRECT_Q2_AFFINE_SKIN_GAP',rows=rows),indent=2))
    assert all(row['independent_minimum_m']>=-gate_m for row in rows)
    assert all(row['independent_final_gap_m']>0 and row['final_active_line_length']==0 for row in rows)
    assert all(row['maximum_agreement_error_m']<=agreement_m for row in rows)
