"""First discriminator: exact supported pose/load construction vs public guard."""
import json,os
from pathlib import Path
import pytest
from examples.general_contact_beam_s3_transient_directional_tilted import prepare


def test_current_public_guard_for_equilibrated_tilted_touch(tmp_path):
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anyfesolver.dynamics import TransientConfig
    m,d,s,u,v,load,report=prepare()
    assert report['exact_touch'] and report['contact_null_norm']>0
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')))
    path.write_text(json.dumps(report,indent=2))
    with pytest.raises(ValueError,match='unforced impact'):
        solve_transient_contact(m,d,TransientConfig(1.e-5,2.e-5,initial_displacement=u,initial_velocity=v),base_load_case=load,contact_settings=s,solver_controls=ContactTransientSolverControls(allow_directional_beam_s3_dev=True),emit_restart_checkpoint=True)
    report['current_public_guard']='unforced impact';report['disposition']='EQUILIBRIUM_INITIAL_ADMISSION_BLOCKED_NOT_DYNAMIC_PASS'
    path.write_text(json.dumps(report,indent=2))



def test_public_equilibrated_tilted_touch_entry(tmp_path):
    import numpy as np
    from examples.general_contact_beam_s3_transient_directional_tilted import run
    m,r,report=run()
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')))
    path.with_name('accepted_full.checkpoint.json').write_text(json.dumps(r.restart_checkpoint,sort_keys=True))
    _,prefix,_=run(end=float(r.times[1]));_,resumed,_=run(restart=prefix.restart_checkpoint)
    report.update(profile=r.profile,version=r.restart_checkpoint['version'],states=len(r.times),minimum_gap=min(c.minimum_gap for c in r.contact),initial_gap=r.contact[0].minimum_gap,restart_exact=resumed.restart_checkpoint==r.restart_checkpoint)
    path.write_text(json.dumps(report,indent=2))
    assert report['version']==20 and report['restart_exact']
    np.testing.assert_array_equal(resumed.displacements,r.displacements[len(prefix.times)-1:])
    assert report['minimum_gap']>=-1.e-5



def test_tilted_work_wrench_and_loaded_energy(tmp_path):
    import numpy as np
    from fractions import Fraction as Q
    from examples.general_contact_beam_s3_transient_directional_tilted import run
    from test_general_contact_beam_s3_directional_coupling import _independent_gap_coefficients,_reference_directional
    from anyfesolver.general_contact.beam_s3_directional_rates import EquilibratedDirectionalBeamS3Manager,directional_coefficients,directional_operator
    from anyfesolver.assembly import assemble_mass_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.nonlinear_state import discard_active_state_candidate
    from anyfesolver.general_contact.runtime import suspend_contact
    m,r,initial=run();_,d,s,u,v,load,_=prepare();manager=EquilibratedDirectionalBeamS3Manager(m,d,s,initial_displacement=u)
    ids={n:m.mesh.dof_manager.get_node_dofs(n) for n in m.mesh.nodes}
    refs={n:tuple(Q(float(x)) for x in m.mesh.nodes[n].coords()) for n in m.mesh.nodes}
    reference,_,_,_,_,norm=_independent_gap_coefficients(refs,ids,u,Q(1,128))
    L=float(m.mesh.nodes[2].x-m.mesh.nodes[1].x);B,factor=directional_coefficients(manager,u)
    actualD,actualH,eD,eH=directional_operator(B,factor,v);expectedD,expectedH,active=_reference_directional(reference,Q(1.e5)*Q(L),v)
    assert np.max(abs(actualD-expectedD))<=eD+1.e-10 and np.max(abs(actualH-expectedH))<=eH+1.e-9
    f0,_,_=manager.evaluate(u);force_rates=[]
    trans=np.array([ids[n][:3] for n in ids]).ravel();rot=np.array([ids[n][3:] for n in ids]).ravel()
    for h in (1.e-6,3.e-7,1.e-7):
        fh,_,_=manager.evaluate(u+h*v);difference=(fh-f0)/h-expectedD
        force_rates.append(dict(h=h,error=float(max(np.max(abs(difference[trans]))/.5,np.max(abs(difference[rot]))/.25))))
    M,_=assemble_mass_matrix(m);external=np.array(initial['load_forces']);energy=[];wrench=[];equilibrium=[];rotational_balance=[];reduced_equilibrium=[];null_equilibrium=[];contact_couples=[]
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
    K0,_=assemble_stiffness_matrix(m);_,_,T,_,independent,_=build_constraint_transformation(K0,np.zeros(len(u)),m)
    basis=build_declared_algebraic_basis(m,M,(T.T@M@T).tocsr(),T,independent,dense_size_limit=512);Z=basis.reduced_basis
    # Actual reducer has class-preserving trace ties; force and couple scales stay separate.
    rotation_columns=np.asarray(abs(T[rot]).sum(axis=0)).ravel()>0
    translation_columns=np.asarray(abs(T[trans]).sum(axis=0)).ravel()>0
    assert not np.any(rotation_columns & translation_columns)

    def structural(q):
        origin={}
        try:
            with suspend_contact():f,_,_=_assemble_nonlinear_system(m,q,origin,3,tangent=False,kinematics='von_karman')
            return f
        finally:discard_active_state_candidate(origin)
    for j,(q,rate,acc,summary) in enumerate(zip(r.displacements,r.velocities,r.accelerations,r.contact)):
        # Elastic radial work: no commits; 3Gauss integrates the admitted cubic force.
        points,weights=np.polynomial.legendre.leggauss(3)
        stored=sum(.5*w*(structural(.5*(x+1)*q)@q) for x,w in zip(points,weights))
        energy.append(float(.5*rate@(M@rate)+stored+summary.line_contact_energy))
        f=structural(q);inertia=M@acc
        fc,_,_=manager.evaluate(q);contact_couples.append(float(np.max(abs(fc[np.array([ids[n][3:5] for n in (3,4,5)]).ravel()]))))
        # Do not subtract stored reactions here: accepted_reaction includes free Newton residual.
        reduced=np.asarray(T.T@(f+fc+inertia-external)).ravel()
        equilibrium.append(float(np.max(abs(reduced[translation_columns]))/.5))
        rotational_balance.append(float(np.max(abs(reduced[rotation_columns]))/.25))
        scaled=reduced.copy();scaled[translation_columns]/=.5;scaled[rotation_columns]/=.25
        reduced_equilibrium.append(float(np.linalg.norm(scaled)))
        null_equilibrium.append(float(np.linalg.norm(Z.T@reduced)/.25))
        for nodes,F,H in (((1,6,2),summary.beam_force,summary.beam_moment),((3,4,5),summary.shell_force,summary.shell_nodal_work_moment)):
            balance=np.r_[F,H].copy()
            for n in nodes:
                dofs=ids[n];p=m.mesh.nodes[n].coords()+q[dofs[:3]];part=r.constraint_reactions[j,dofs]+external[dofs]-inertia[dofs]-f[dofs]
                balance[:3]+=part[:3];balance[3:]+=part[3:]+np.cross(p,part[:3])
            wrench.append((float(np.linalg.norm(balance[:3])/.5),float(np.linalg.norm(balance[3:])/.25)))
    work=np.array([(external@(q-r.displacements[0])) for q in r.displacements]);defect=float(np.max(abs(np.array(energy)-energy[0]-work))/initial['positive_impact_kinetic_scale_j'])
    data=dict(directional_active_intervals=active,force_rate_errors=force_rates,translational_free_balance=max(equilibrium),rotational_free_balance=max(rotational_balance),complete_reduced_equilibrium=max(reduced_equilibrium),null_equilibrium=max(null_equilibrium),equilibrium_subtracts_stored_reactions=False,work_wrench=wrench,
        loaded_energy_defect=defect,positive_impact_scale_j=initial['positive_impact_kinetic_scale_j'],structural_energy_reference='SOURCE_JUSTIFIED_RADIAL_WORK_NOT_INDEPENDENT_FORMULATION_ORACLE',
        independent_directional_work_jacobian=True,maximum_rotational_contact_couple=max(contact_couples),full_line_initial_contact_null_coupling=initial['contact_null_norm'],minimum_gap=min(c.minimum_gap for c in r.contact))
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')));path.with_name('work_energy_report.json').write_text(json.dumps(data,indent=2));path.write_text(json.dumps(data,indent=2))
    assert max(row['error'] for row in force_rates)<=1.e-5
    assert max(equilibrium)<=1.e-6 and max(rotational_balance)<=1.e-6 and max(reduced_equilibrium)<=1.e-6 and max(null_equilibrium)<=1.e-6
    assert np.max(wrench)<=1.e-6 and defect<=.01
    assert max(contact_couples)>0


def test_equilibrated_initial_refusals_and_restart_binding(tmp_path):
    import numpy as np
    from dataclasses import replace
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anyfesolver.dynamics import TransientConfig
    from examples.general_contact_beam_s3_transient_directional_tilted import run
    m,d,s,u,v,load,initial=prepare();controls=ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=True)
    for bad in (1,None,'yes'):
        with pytest.raises(ValueError,match='boolean'):ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=bad)
    with pytest.raises(ValueError):ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=True,allow_directional_beam_s3_dev=True)
    def attempt(pose=u,velocity=v,dead=load,restart=None):
        return solve_transient_contact(m,d,TransientConfig(1.e-5,3.e-5,initial_displacement=pose,initial_velocity=velocity),base_load_case=dead,contact_settings=s,solver_controls=controls,restart_checkpoint=restart,emit_restart_checkpoint=True)
    bad=u.copy();bad[m.mesh.dof_manager.get_node_dofs(6)[2]]+=2.**-30
    with pytest.raises(ValueError,match='exact full-line'):attempt(pose=bad)
    with pytest.raises(ValueError,match='constant nodal'):attempt(dead=None)
    velocity=v.copy();velocity[m.mesh.dof_manager.get_node_dofs(3)[4]]+=.01;velocity[m.mesh.dof_manager.get_node_dofs(4)[4]]+=.01;velocity[m.mesh.dof_manager.get_node_dofs(5)[4]]+=.01
    with pytest.raises(ValueError,match='algebraic velocity'):attempt(velocity=velocity)
    from anyfesolver import LoadCase
    for node,axis,message in ((1,2,'initial total force'),(3,4,'initial algebraic force')):
        values=np.array(initial['load_forces']);values[m.mesh.dof_manager.get_node_dofs(node)[axis]]+=.001
        unbalanced=LoadCase('unbalanced_'+str(axis))
        for n in m.mesh.nodes:
            ids=m.mesh.dof_manager.get_node_dofs(n)
            unbalanced.add_nodal_load(n,forces=tuple(values[ids[:3]]),moments=tuple(values[ids[3:]]))
        with pytest.raises(ValueError,match=message):attempt(dead=unbalanced)
    from anyfesolver.general_contact.transient import ContactNodalLoadHistory
    with pytest.raises(ValueError,match='nodal load history'):
        solve_transient_contact(m,d,TransientConfig(1.e-5,3.e-5,initial_displacement=u,initial_velocity=v),base_load_case=load,
            nodal_load_history=ContactNodalLoadHistory(((0.,1.),(1.e-4,1.))),contact_settings=s,solver_controls=controls)
    _,r,_=run(end=1.e-5)
    from anyfesolver import LoadCase
    changed=LoadCase('changed_equilibrium');changed.add_nodal_load(1,forces=(0,0,.001))
    with pytest.raises(ValueError,match='incompatible'):attempt(dead=changed,restart=r.restart_checkpoint)
    from examples.general_contact_beam_s3_transient_directional import run as oldrun
    _,old=oldrun(end=1.e-5)
    with pytest.raises(ValueError,match='incompatible'):attempt(restart=old.restart_checkpoint)
    Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json'))).with_name('refusal_report.json').write_text(json.dumps(dict(nonexact_pose=True,wrong_null_velocity=True,missing_load=True,changed_load_restart=True,V19_cross_profile=True,unbalanced_physical_load=True,unbalanced_algebraic_load=True,positive_load_history=True)))



@pytest.mark.parametrize('failure',('cancel','exception','retry'))
def test_loaded_initial_transaction_boundaries(monkeypatch,tmp_path,failure):
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
    controls=driver.ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=True)
    if failure!='retry':
        with pytest.raises(SolveCancelled if failure=='cancel' else ArithmeticError):driver.solve_transient_contact(m,d,TransientConfig(1.e-5,3.e-5,initial_displacement=u,initial_velocity=v),base_load_case=load,contact_settings=s,solver_controls=controls,cancellation_token=token)
        assert len(commits)==2
    else:
        result=driver.solve_transient_contact(m,d,TransientConfig(1.e-5,3.e-5,initial_displacement=u,initial_velocity=v),base_load_case=load,contact_settings=s,solver_controls=controls)
        assert result.cutbacks>=1
        import numpy as np
        np.testing.assert_array_equal(result.displacements[0],u)
    assert active_contact_manager() is None


def test_V20_secant_corruption_and_legacy_profile_identity(tmp_path):
    import copy
    import numpy as np
    from examples.general_contact_beam_s3_transient_directional_tilted import run
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    from anyfesolver.assembly import build_constraint_transformation
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    m,r,initial=run(end=1.e-5);changed=copy.deepcopy(r.restart_checkpoint)
    m.apply_boundary_conditions();K,_=assemble_stiffness_matrix(m);_,_,_,_,ids,_=build_constraint_transformation(K,np.zeros(len(initial['pose'])),m)
    index=list(ids).index(m.mesh.dof_manager.get_node_dofs(3)[3]);changed['state']['acceleration'][index]+=.1
    changed.pop('checkpoint_sha256');changed['checkpoint_sha256']=analysis_configuration_fingerprint(changed)
    with pytest.raises(ValueError,match='secant history'):run(restart=changed)


@pytest.mark.skipif(not os.environ.get('ANYSOLVER_CONTACT_PRE_V20_BASELINE'),
                    reason='archived compatibility needs explicit baseline and execution authority')
def test_V20_archived_legacy_profile_identity(tmp_path):
    import importlib.util,sys,hashlib
    from anyfesolver.dynamics import TransientConfig
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    # Ordinary checks never discover or import a machine-local archived owner.
    # An authorised reference runner supplies its exact baseline explicitly.
    baseline=Path(os.environ['ANYSOLVER_CONTACT_PRE_V20_BASELINE'])
    if not baseline.is_file():raise ValueError('explicit pre-V20 baseline is not a file')
    name='anyfesolver.general_contact._pre_V20_transient';spec=importlib.util.spec_from_file_location(name,baseline);old=importlib.util.module_from_spec(spec);sys.modules[name]=old
    rows=[]
    try:
        spec.loader.exec_module(old)
        from examples.general_contact_beam_s3_transient import build as b17
        from examples.general_contact_beam_s3_transient_rotational import build as b18
        from examples.general_contact_beam_s3_transient_directional import build as b19
        for version,builder,flag in ((17,b17,'allow_deformable_beam_s3_dev'),(18,b18,'allow_rotational_beam_s3_dev'),(19,b19,'allow_directional_beam_s3_dev')):
            m,d,s,v=builder();prior=old.solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-5,initial_velocity=v),contact_settings=s,solver_controls=old.ContactTransientSolverControls(**{flag:True}),emit_restart_checkpoint=True)
            m,d,s,v=builder();resumed=solve_transient_contact(m,d,TransientConfig(1.e-5,2.e-5,initial_velocity=v),contact_settings=s,solver_controls=ContactTransientSolverControls(**{flag:True}),emit_restart_checkpoint=True,restart_checkpoint=prior.restart_checkpoint)
            m,d,s,v=builder();full=solve_transient_contact(m,d,TransientConfig(1.e-5,2.e-5,initial_velocity=v),contact_settings=s,solver_controls=ContactTransientSolverControls(**{flag:True}),emit_restart_checkpoint=True)
            assert resumed.restart_checkpoint==full.restart_checkpoint and resumed.restart_checkpoint['version']==version
            rows.append(dict(version=version,old_checkpoint_resumed=True,exact_current_full=True))
    finally:sys.modules.pop(name,None)
    path=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')));path.with_name('legacy_compatibility.json').write_text(json.dumps(dict(baseline=str(baseline),baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),profiles=rows),indent=2))
