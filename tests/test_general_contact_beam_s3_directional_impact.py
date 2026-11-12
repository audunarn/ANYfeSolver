"""Public V19 impact from exact full-line touch; same physical fixture as V18."""
import json,os
from pathlib import Path
import numpy as np
import pytest
from examples.general_contact_beam_s3_transient_directional import build,run
from test_general_contact_beam_s3_transient_rotational import energy,contact_oracle


def test_public_directional_impact_opening_restart_and_energy(tmp_path):
    from anyfesolver.assembly import assemble_mass_matrix
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.general_contact.runtime import suspend_contact,_ACTIVE_CONTACT
    from anyfesolver.general_contact.beam_s3_directional_rates import DirectionalBeamS3Manager
    m,r=run(end=.0008);destination=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')))
    destination.with_name('accepted_full.checkpoint.json').write_text(json.dumps(r.restart_checkpoint,sort_keys=True,separators=(',',':'))+'\n')
    split_time=float(r.times[len(r.times)//2])
    _,prefix=run(end=split_time);_,resumed=run(end=.0008,restart=prefix.restart_checkpoint)
    destination.with_name('accepted_prefix.checkpoint.json').write_text(json.dumps(prefix.restart_checkpoint,sort_keys=True,separators=(',',':'))+'\n')
    equal=resumed.restart_checkpoint==r.restart_checkpoint
    destination.with_name("restart_comparison.json").write_text(json.dumps(dict(exact=equal,split_time=split_time,max_displacement_difference=float(np.max(abs(resumed.displacements-r.displacements[len(prefix.times)-1:])))),indent=2))
    np.testing.assert_array_equal(resumed.displacements,r.displacements[len(prefix.times)-1:])
    m2,fine=run(dt=5.e-6,end=.0008)
    E=energy(m,r);Ef=energy(m2,fine)
    defect=float(np.max(abs(E-E[0]))/E[0]);fine_defect=float(np.max(abs(Ef-Ef[0]))/Ef[0])
    M,_=assemble_mass_matrix(m);ids=np.array([m.mesh.dof_manager.get_node_dofs(n) for n in (3,4,5)])
    manager=DirectionalBeamS3Manager(m,*build()[1:3]);algebraic=[];balance=[];potential=[]
    for i in (0,int(np.argmin([c.minimum_gap for c in r.contact])),len(r.times)-1):
        u=r.displacements[i];f,_,summary=manager.evaluate(u)
        with suspend_contact():structural,_,_=_assemble_nonlinear_system(m,u,{},3,tangent=False,kinematics='von_karman')
        algebraic.append(float(np.linalg.norm((f+structural)[ids[:,3:5]].sum(axis=0))/.5))
        report=manager.wrench_report(u);inertia=M@r.accelerations[i]
        for nodes,F,L in (((1,6,2),summary.beam_force,summary.beam_moment),((3,4,5),summary.shell_force,report['nodal_work_moment'])):
            w=np.r_[F,L].copy()
            for node in nodes:
                dofs=m.mesh.dof_manager.get_node_dofs(node);point=m.mesh.nodes[node].coords()+u[dofs[:3]]
                part=r.constraint_reactions[i,dofs]-inertia[dofs]-structural[dofs]
                w[:3]+=part[:3];w[3:]+=part[3:]+np.cross(point,part[:3])
            balance.append((float(np.linalg.norm(w[:3])/.5),float(np.linalg.norm(w[3:])/.25)))
        potential.append(float(abs(contact_oracle(m,u)-summary.line_contact_energy)))
    coupled=[]
    for u in r.displacements:
        f,_,_=manager.evaluate(u,tangent=False);coupled.append(-f[ids[:,3:5]])
    coupled=np.array(coupled)
    data=dict(profile=r.profile,version=r.restart_checkpoint['version'],states=len(r.times),initial_exact_gap=r.contact[0].minimum_gap,
        active_states=sum(c.active_line_length>0 for c in r.contact),minimum_gap=min(c.minimum_gap for c in r.contact),final_gap=r.contact[-1].minimum_gap,
        final_active=r.contact[-1].active_line_length,energy_defect=defect,fine_energy_defect=fine_defect,endpoint_algebraic=algebraic,
        body_work_wrench=balance,independent_polynomial_contact_error=potential,restart_exact=equal,
        maximum_target_rotation=float(np.max(abs(r.displacements[:,ids[:,3:5]]))),maximum_target_rotation_rate=float(np.max(abs(r.velocities[:,ids[:,3:5]]))),
        maximum_rotational_contact_couple=float(np.max(abs(coupled))),rotational_contact_work_trapezoid_j=float(np.sum(.5*(coupled[1:]+coupled[:-1])*np.diff(r.displacements[:,ids[:,3:5]],axis=0))),
        physical_mass_unchanged=bool(np.count_nonzero(M[ids[:,3:].ravel()].data)==0),dynamic_nonzero_full_touch_null_coupling_claim=False)
    destination.write_text(json.dumps(data,indent=2)+'\n')
    destination.with_name('impact_report.json').write_text(json.dumps(data,indent=2)+'\n')
    assert data['version']==19 and data['initial_exact_gap']==0.
    assert data['active_states']>0 and data['final_active']==0 and data['final_gap']>0
    assert data['minimum_gap']>=-1.e-5 and equal
    assert defect<=.01 and fine_defect<=.01 and fine_defect<defect
    assert max(algebraic)<=1.e-6 and np.max(balance)<=1.e-6
    assert max(potential)<=1.e-12 and data['physical_mass_unchanged']
    assert data['maximum_target_rotation']>0 and data['maximum_target_rotation_rate']>0 and data['maximum_rotational_contact_couple']>0
    assert _ACTIVE_CONTACT.get() is None


@pytest.mark.parametrize('failure',('cancel','exception','retry'))
def test_directional_transaction_boundaries(monkeypatch,failure,tmp_path):
    import anyfesolver.general_contact.transient as driver
    import anyfesolver.general_contact.beam_s3_transient as path
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact.runtime import active_contact_manager
    from anyfesolver.dynamics import TransientConfig
    m,d,s,v=build();token=CancellationToken();commits=[];original=driver.commit_state_candidate;bound=path.hermite_gap_bounds;calls=0
    def commit(origin,trial):
        commits.append(1);return original(origin,trial)
    def checked(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:
            if failure=='cancel':token.cancel('owned V19 acceptance');kwargs['check_cancel']()
            if failure=='exception':raise ArithmeticError('owned V19 path exception')
            return -float('inf'),False,1
        return bound(*args,**kwargs)
    monkeypatch.setattr(driver,'commit_state_candidate',commit);monkeypatch.setattr(path,'hermite_gap_bounds',checked)
    kwargs=dict(contact_settings=s,solver_controls=driver.ContactTransientSolverControls(allow_directional_beam_s3_dev=True,save_constraint_reactions_dev=True),cancellation_token=token,emit_restart_checkpoint=True)
    if failure!='retry':
        with pytest.raises(SolveCancelled if failure=='cancel' else ArithmeticError):driver.solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),**kwargs)
        assert len(commits)==3
        report=dict(mode=failure,accepted_commits=len(commits),rejected_candidate_not_committed=True)
    else:
        kwargs['emit_restart_checkpoint']=False
        result=driver.solve_transient_contact(m,d,TransientConfig(1.e-5,1.e-4,initial_velocity=v),**kwargs)
        assert result.cutbacks>=1
        monkeypatch.setattr(path,'hermite_gap_bounds',bound)
        _,reference=run(end=2.e-5)
        np.testing.assert_array_equal(result.displacements[:3],reference.displacements)
        report=dict(mode=failure,cutbacks=result.cutbacks,accepted_prefix_exact=True)
    assert active_contact_manager() is None
    base=Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json')));base.with_name('transaction_'+failure+'.json').write_text(json.dumps(report,indent=2))


def test_directional_secant_history_corruption_refused(tmp_path):
    import copy
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.assembly import build_constraint_transformation
    _,result=run(end=3.e-5);changed=copy.deepcopy(result.restart_checkpoint)
    m,_,_,v=build();m.apply_boundary_conditions();K,_=assemble_stiffness_matrix(m)
    _,_,_,_,ids,_=build_constraint_transformation(K,np.zeros(len(v)),m)
    index=list(ids).index(m.mesh.dof_manager.get_node_dofs(3)[3])
    changed['state']['acceleration'][index]+=.1
    changed.pop('checkpoint_sha256');changed['checkpoint_sha256']=analysis_configuration_fingerprint(changed)
    with pytest.raises(ValueError,match='secant history'):run(end=4.e-5,restart=changed)
    Path(os.environ.get('CONTACT_APP_RESULT_RECORD',str(tmp_path/'actual_result.json'))).write_text(json.dumps(dict(rechecksummed_null_acceleration_refused=True,version=19),indent=2))
