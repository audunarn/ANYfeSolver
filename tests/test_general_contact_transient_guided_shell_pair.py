"""Independent finite-mass Q4/S3 recoil, discrete work and public state controls."""
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
import json,os,runpy
import numpy as np
import pytest
from anyfesolver import ContactTransientSolverControls,TransientConfig,solve_transient_contact
from anyfesolver.assembly import assemble_mass_matrix
from anyfesolver.general_contact.core import ContactSolverSettings
from anyfesolver.general_contact.pair import _PlanarShellPairManager
from anyfesolver.general_contact.runtime import active_contact_manager
from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint

example=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_guided_shell_pair_impact_dev.py'))
MASS=np.diag([20.,14.4]);GAP=.0002;KN=3.e8;D=np.array([1.,-1.])


def reference(times):
    q=np.zeros(2);v=np.array([-.1,0.]);a=np.zeros(2)
    rows=[(q.copy(),v.copy(),a.copy())]
    for left,right in zip(times,times[1:]):
        h=right-left;beta=.25*h*h;prediction=q+h*v+beta*a
        candidates=[]
        for active in (False,True):
            stiffness=KN*np.outer(D,D) if active else np.zeros((2,2))
            constant=KN*GAP*D if active else np.zeros(2)
            trial_a=np.linalg.solve(MASS+beta*stiffness,-stiffness@prediction-constant)
            trial=prediction+beta*trial_a;gap=GAP+D@trial
            if (gap>1.e-14 if active else gap<-1.e-14):continue
            force=KN*min(gap,0.)*D
            if np.linalg.norm(MASS@trial_a+force)>1.e-6:continue
            candidates.append((trial,trial_a))
        assert candidates,'independent two-body branch has no solution'
        trial,trial_a=candidates[0];v=v+.5*h*(a+trial_a);q,a=trial,trial_a
        rows.append((q.copy(),v.copy(),a.copy()))
    return np.array(rows)


def continuous(times):
    # Exact continuous unilateral elastic oscillator on the admitted uniform modes.
    m=np.diag(MASS);total=m.sum();reduced=np.prod(m)/total
    omega=np.sqrt(KN/reduced);hit=GAP/.1;release=hit+np.pi/omega
    q=[];v=[]
    cm_velocity=-.1*m[0]/total
    for t in times:
        if t<=hit:r,rate=-.1*t,-.1
        elif t<=release:
            r=-GAP-.1/omega*np.sin(omega*(t-hit));rate=-.1*np.cos(omega*(t-hit))
        else:r,rate=-GAP+.1*(t-release),.1
        q.append([cm_velocity*t+m[1]/total*r,cm_velocity*t-m[0]/total*r])
        v.append([cm_velocity+m[1]/total*rate,cm_velocity-m[0]/total*rate])
    return np.array(q),np.array(v),hit,release


@pytest.mark.parametrize('kind',('q4','s3'))
def test_uniform_pair_has_independent_both_body_mass_pressure_and_wrench(kind):
    model,definition,modes=example['build_model'](target_kind=kind)
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,MASS,rtol=2.e-13,atol=1.e-12)
    manager=_PlanarShellPairManager(model,definition,ContactSolverSettings())
    np.testing.assert_allclose(manager.support,np.full(4,.25),rtol=1.e-12,atol=1.e-14)
    assert manager.pressure_nodes==(1,2,3,4)
    for z in (np.array([0.,0.]),np.array([-.00022,0.]),np.array([-.0005,-.00028])):
        f,k,row=manager.evaluate(modes@z,tangent=True)
        gap=GAP+D@z;p=KN*max(-gap,0.)
        np.testing.assert_allclose(modes.T@f,KN*min(gap,0.)*D,atol=2.e-8)
        np.testing.assert_allclose(modes.T@k@modes,KN*np.outer(D,D)*(gap<0.),rtol=2.e-12,atol=2.e-6)
        np.testing.assert_allclose(row.integration_force,[0.,0.,p],atol=2.e-8)
        np.testing.assert_allclose(row.target_force,[0.,0.,-p],atol=2.e-8)
        np.testing.assert_allclose(np.array(row.integration_moment)+row.target_moment,0.,atol=2.e-8)
        assert row.penalty_energy==pytest.approx(.5*KN*min(gap,0.)**2,rel=1.e-11,abs=1.e-14)


@pytest.fixture(scope='module')
def impact():
    model,result,modes=example['solve']()
    q,v,a=(example['modal'](value,modes) for value in
           (result.displacements,result.velocities,result.accelerations))
    record=dict(times_s=result.times.tolist(),q=q.tolist(),v=v.tolist(),a=a.tolist(),
        contact=[r.to_dict() for r in result.contact],swept_gaps=result.swept_minimum_gaps.tolist(),
        cutbacks=result.cutbacks,cutback_diagnostics=result.cutback_diagnostics,
        events_s=result.localized_contact_event_times,endpoint_requests=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations)
    (Path(os.environ['RUNNER_TEMP'])/'shell_pair_impact.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return model,result,modes


def test_shell_pair_recoils_with_equal_opposite_work_and_physical_momentum(impact):
    _,result,modes=impact
    expected=reference(result.times)
    for values,index,tolerance in ((result.displacements,0,1.e-10),(result.velocities,1,1.e-7),(result.accelerations,2,2.e-3)):
        actual=example['modal'](values,modes)
        np.testing.assert_allclose(actual,expected[:,index],rtol=0.,atol=tolerance)
        np.testing.assert_allclose(values,actual@modes.T,rtol=0.,atol=tolerance)
    q,v=expected[:,0],expected[:,1]
    momentum=v@np.diag(MASS)
    assert np.max(np.abs(momentum-momentum[0]))<1.e-8
    assert v[-1,1]<-.1 # Finite target receives momentum; it is not a fixed obstacle.
    for z,row in zip(q,result.contact):
        pressure=KN*max(-GAP-D@z,0.)
        np.testing.assert_allclose(row.integration_force,[0.,0.,pressure],atol=2.e-4)
        np.testing.assert_allclose(np.array(row.integration_force)+row.target_force,0.,atol=1.e-8)
        np.testing.assert_allclose(np.array(row.integration_moment)+row.target_moment,0.,atol=1.e-8)
    assert np.max(np.abs(result.constraint_reactions@modes))<1.e-6
    energy=.5*np.einsum('ij,jk,ik->i',v,MASS,v)+np.array([r.penalty_energy for r in result.contact])
    assert np.max(np.abs(energy/energy[0]-1.))<.01
    exact_q,exact_v,_,_=continuous(result.times)
    assert np.max(np.abs(q-exact_q))/.0002<.02
    assert np.max(np.abs(v-exact_v))/.1<.02
    assert any(row.active_pressure_dofs for row in result.contact)
    assert result.contact[-1].active_pressure_dofs==0
    assert np.min(result.swept_minimum_gaps)>=-2.e-5


def test_shell_pair_active_v11_restart_and_identity_rejection(impact):
    _,whole,modes=impact
    index=next(i for i,row in enumerate(whole.contact) if row.active_pressure_dofs
        and abs((whole.times[i]-whole.times[i-1])-5.e-5)<1.e-12)
    _,prefix,_=example['solve'](end=float(whole.times[index]),emit=True)
    assert prefix.restart_checkpoint['version']==11
    assert prefix.restart_checkpoint['state']['plane_pressure_features']==[['shell_pair',n] for n in (1,2,3,4)]
    _,resumed,_=example['solve'](checkpoint=prefix.restart_checkpoint)
    for name,tolerance in (('displacements',1.e-10),('velocities',1.e-7),('accelerations',2.e-3)):
        actual=np.concatenate((getattr(prefix,name)[:-1],getattr(resumed,name)))
        np.testing.assert_allclose(actual,getattr(whole,name),rtol=0.,atol=tolerance)
    assert [row.active_pressure_dofs for row in (*prefix.contact[:-1],*resumed.contact)]==[row.active_pressure_dofs for row in whole.contact]
    for alter in ('row_order','pair_name','boolean_node'):
        bad=deepcopy(prefix.restart_checkpoint)
        features=bad['state']['plane_pressure_features']
        if alter=='row_order':features.reverse()
        elif alter=='pair_name':features[0][0]='other'
        else:features[0][1]=True
        bad['checkpoint_sha256']=analysis_configuration_fingerprint({k:v for k,v in bad.items() if k!='checkpoint_sha256'})
        with pytest.raises(ValueError,match='pressure-feature'):
            example['solve'](checkpoint=bad)
    with pytest.raises(ValueError,match='analysis'):
        example['solve'](checkpoint=prefix.restart_checkpoint,
            controls=replace(example['CONTROLS'],batch_plane_event_fronts_dev=False))


def test_finite_target_temporal_refinement_against_continuous_reference(impact):
    _,coarse,modes=impact
    _,fine,_=example['solve'](dt=2.5e-5)
    measurements=[]
    for name,result in (('coarse',coarse),('half',fine)):
        q,v=example['modal'](result.displacements,modes),example['modal'](result.velocities,modes)
        exact_q,exact_v,_,_=continuous(result.times)
        errors=np.array([np.max(np.abs(q-exact_q))/.0002,np.max(np.abs(v-exact_v))/.1])
        assert np.max(errors)<.02
        energy=.5*np.einsum('ij,jk,ik->i',v,MASS,v)+np.array([r.penalty_energy for r in result.contact])
        defect=float(np.max(np.abs(energy/energy[0]-1.)))
        assert defect<.01
        measurements.append(dict(name=name,scaled_q_v_errors=errors.tolist(),energy_relative_defect=defect,
            times_s=result.times.tolist(),q=q.tolist(),v=v.tolist(),swept_gaps=result.swept_minimum_gaps.tolist(),
            cutbacks=result.cutbacks,events_s=result.localized_contact_event_times,
            endpoint_requests=result.event_endpoint_evaluations,stage_evaluations=result.plane_event_stage_evaluations))
    (Path(os.environ['RUNNER_TEMP'])/'temporal_refinement.json').write_text(json.dumps(measurements,indent=2),encoding='utf-8')
    assert np.all(np.array(measurements[1]['scaled_q_v_errors'])<measurements[0]['scaled_q_v_errors'])


@pytest.mark.parametrize('change',[
    dict(allow_guided_shell_pair_dev=True),
    dict(allow_guided_shell_pair_dev=1),
    dict(allow_guided_shell_pair_dev=True,q4_plane_event_subdivision_dev=True,allow_multiple_shell_planes_dev=True),
    dict(allow_guided_shell_pair_dev=True,q4_plane_event_subdivision_dev=True,allow_s3_descriptor_dev=True),
    dict(allow_guided_shell_pair_dev=True,q4_plane_event_subdivision_dev=True,allow_q4_plastic_dev=True)])
def test_pair_profile_rejects_incompatible_controls(change):
    with pytest.raises(ValueError,match='guided shell pair'):
        ContactTransientSolverControls(**change)


def test_pair_default_and_invalid_motion_rejected_before_integration():
    model,definition,modes=example['build_model']()
    config=TransientConfig(5.e-5,.0001,initial_velocity=modes@example['INITIAL_VELOCITY'])
    with pytest.raises(ValueError,match='one shell surface'):
        solve_transient_contact(model,definition,config)
    with pytest.raises(ValueError,match='two explicit surfaces'):
        solve_transient_contact(model,definition,config,solver_controls=example['CONTROLS'],
            contact_settings=ContactSolverSettings(pair_motion='general'))
    # Free one source node's normal coordinate; it has physical inertia, but
    # uniform-face contact kinematics no longer follow from its constraints.
    model.constraint_equations.pop(0)
    with pytest.raises(ValueError,match='uniform normal translation'):
        solve_transient_contact(model,definition,config,solver_controls=example['CONTROLS'])


def test_pair_public_cancellation_retains_model_and_no_partial_checkpoint(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    model,definition,modes=example['build_model']()
    from anyfesolver.nonlinear_restart import model_configuration_fingerprint
    fingerprint=model_configuration_fingerprint(model)
    token=CancellationToken();calls=0
    original=_PlanarShellPairManager.evaluate
    def counted(manager,*args,**kwargs):
        nonlocal calls
        result=original(manager,*args,**kwargs);calls+=1
        if calls==3:token.cancel('owned pair cancellation')
        return result
    monkeypatch.setattr(_PlanarShellPairManager,'evaluate',counted)
    with pytest.raises(SolveCancelled,match='owned pair cancellation'):
        solve_transient_contact(model,definition,
            TransientConfig(5.e-5,.0001,initial_velocity=modes@example['INITIAL_VELOCITY']),
            solver_controls=example['CONTROLS'],cancellation_token=token,emit_restart_checkpoint=True)
    assert calls==3
    assert model_configuration_fingerprint(model)==fingerprint
    assert active_contact_manager() is None
    monkeypatch.setattr(_PlanarShellPairManager,'evaluate',original)
    resumed=solve_transient_contact(model,definition,
        TransientConfig(5.e-5,.0001,initial_velocity=modes@example['INITIAL_VELOCITY']),
        solver_controls=example['CONTROLS'])
    assert np.all(resumed.displacements[0]==0.)
    assert all(row.active_pressure_dofs==0 for row in resumed.contact)
