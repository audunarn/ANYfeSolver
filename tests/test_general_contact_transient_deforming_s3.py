"""Independent covered-triangle projection and nonlinear coupled step checks.

The S3 structural operator remains an accepted input; new contact geometry/work
and time coupling use a separately implemented normal-nodal projection oracle.
"""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json
import os
import runpy
import numpy as np
import pytest
from scipy.optimize import root
from scipy.integrate import solve_ivp
from anyfesolver import TransientConfig, ContactTransientSolverControls, solve_transient_contact
from anyfesolver.assembly import assemble_mass_matrix
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.normal_s3_skin_path import prepare_normal_s3_skin_path
from anyfesolver.general_contact.shell_pair_hermite_sweep import shell_pair_hermite_lower
from anyfesolver.general_contact.runtime import active_contact_manager
from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint, model_configuration_fingerprint

example = runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_nonuniform_s3_impact_dev.py'))
GAP,KN,SUPPORT = (example[key] for key in ('GAP','KN','SUPPORT'))
MASS = np.diag(np.r_[20.*SUPPORT,np.full(3,49./3)])
CONTROLS = ContactTransientSolverControls(allow_normal_s3_skin_hermite_dev=True,
    save_constraint_reactions_dev=True,equilibrium_tolerance=1.e-10)
SOURCE_XY = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
TARGET_XY = np.array([[-2.,-2.],[5.,-2.],[-2.,5.]])
TARGET_PLANE = np.column_stack((np.ones(3),TARGET_XY))
FACES = ((0,1,2),(0,2,3))
TRIANGLE_MASS = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])/24.
deforming = runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_deforming_s3_impact_dev.py'))


def build():
    model,definition,_ = example['build_model'](target_mode='xy')
    model.constraint_equations.clear() # Retain every physical normal coordinate.
    modes = np.zeros((42,7))
    for i,node in enumerate(range(1,8)):
        modes[model.mesh.dof_manager.get_node_dofs(node)[2],i] = 1.
    return model,definition,modes


def gaps(q):
    """Original reference-measure P1 integrals; no clipping/production kernel.

    Both source triangles are completely covered by the large target. Intersect
    each current source point along its current unit normal with the target
    plane. Gap is (zs-a-bx*x-by*y)/(nz-bx*nx-by*ny), affine on that face.
    """
    source = np.column_stack((SOURCE_XY,np.full(4,.0102))).astype(np.result_type(q,float))
    source[:,2] += q[:4]
    plane = np.linalg.solve(TARGET_PLANE,.01+q[4:])
    weighted = np.zeros(4,dtype=source.dtype)
    for face in FACES:
        ids = np.array(face)
        points = source[ids]
        normal = np.cross(points[1]-points[0],points[2]-points[0])
        normal /= np.sqrt(normal@normal) # Analytic complex extension, no conjugation.
        denominator = normal[2]-plane[1:]@normal[:2]
        local = (points[:,2]-plane[0]-points[:,:2]@plane[1:])/denominator
        weighted[ids] += TRIANGLE_MASS@local
    return weighted/SUPPORT


def contact_force(q):
    q = np.asarray(q,dtype=float)
    gap = gaps(q)
    derivative = np.empty((4,7))
    for column in range(7):
        trial = q.astype(complex);trial[column] += 1.e-26j
        derivative[:,column] = gaps(trial).imag/1.e-26
    return KN*derivative.T@(SUPPORT*np.minimum(gap,0.))


def reference(times,structural):
    q = np.zeros(7);v = np.r_[np.full(4,-.1),np.zeros(3)];a = np.zeros(7)
    rows = [(q.copy(),v.copy(),a.copy())]
    for h in np.diff(times):
        beta = .25*h*h;predictor = q+h*v+beta*a
        def residual(value):
            position = predictor+beta*value
            return MASS@value+structural@position+contact_force(position)
        solution = root(residual,a,method='hybr',options={'xtol':1.e-10})
        force_scale = max(1.,np.linalg.norm(MASS@solution.x))
        assert np.linalg.norm(residual(solution.x)) <= 1.e-9*force_scale,solution.message
        next_a = solution.x
        q = predictor+beta*next_a
        v += .5*h*(a+next_a);a = next_a
        rows.append((q.copy(),v.copy(),a.copy()))
    return np.array(rows)


@pytest.mark.parametrize('q',(
    [-.00023,-.000229,-.000231,-.000228,-1.e-5,-9.e-6,-1.1e-5],
    [-.000195,-.00019,-.000225,-.0002,0.,3.e-6,-2.e-6]))
def test_actual_source_deformation_has_independent_pressure_work_and_tangent(q):
    model,definition,modes = build()
    manager = GeneralS3PatchManager(model,definition,example['SETTINGS'])
    path = prepare_normal_s3_skin_path(model,manager,modes)
    q = np.array(q)
    force,tangent,result = manager.evaluate(modes@q,tangent=True)
    np.testing.assert_allclose(result.normalized_weighted_gap_m,gaps(q),rtol=0.,atol=2.e-16)
    np.testing.assert_allclose(modes.T@force,contact_force(q),rtol=2.e-11,atol=1.e-6)
    direction = np.array([.1,-.2,.3,-.4,.2,.1,-.2])
    for step in (1.e-8,3.e-8,1.e-7):
        independent = (contact_force(q+step*direction)-contact_force(q-step*direction))/(2*step)
        actual = modes.T@tangent@modes@direction
        assert np.linalg.norm(actual-independent)/np.linalg.norm(independent) <= 1.e-5
    storage = .5*KN*float(SUPPORT@np.minimum(gaps(q),0.)**2)
    assert result.regularization_storage_j == pytest.approx(storage,rel=1.e-10,abs=1.e-12)
    np.testing.assert_allclose(np.array(result.integration_force)+result.target_force,0.,atol=1.e-7)
    np.testing.assert_allclose(np.array(result.integration_moment)+result.target_moment,0.,atol=1.e-7)
    path.require_motion(modes@q)
    with pytest.raises(ValueError,match='uniform'):
        from anyfesolver.general_contact.fixed_projection_s3_events import prepare_fixed_projection_s3_events
        prepare_fixed_projection_s3_events(model,manager,modes)


def test_original_reference_polarity_is_not_reset_at_each_path_start():
    a = np.array([[0.,0.,0.],[.5,0.,-1.],[0.,1.,0.]])
    b = a-[0.,0.,.1]
    v0 = np.zeros((3,3));v0[1,0] = -2.5
    v1 = -v0
    local = shell_pair_hermite_lower(a,a,b,b,v0,v1,v0,v1,1.)
    assert local.success
    original = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    persistent = shell_pair_hermite_lower(a,a,b,b,v0,v1,v0,v1,1.,
        reference_source=original,reference_target=original-[0.,0.,.1])
    assert not persistent.success
    assert persistent.lower_gap is None


def test_open_corner_approach_is_checked_even_with_other_closed_corners():
    a = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    b = a.copy();b[0,2] = -.1
    source_rate = np.zeros((3,3))
    target_rate = source_rate.copy();target_rate[0,2] = .8
    ordinary = shell_pair_hermite_lower(a,a,b,b,source_rate,source_rate,
        target_rate,-target_rate,1.,gap_tolerance_m=.2)
    assert ordinary.success # Penetration budget alone allows this represented excursion.
    guarded = shell_pair_hermite_lower(a,a,b,b,source_rate,source_rate,
        target_rate,-target_rate,1.,gap_tolerance_m=.2,
        exclude_open_endpoint_approach_tolerance_m=1.e-9)
    assert not guarded.success
    assert guarded.status == 'hidden_approach'
    assert guarded.lower_gap is None


def test_normal_skin_owner_applies_corner_guard_to_actual_mapped_skins():
    model,definition,modes = build()
    pair = replace(definition.interactions[0],gap_tolerance=.0003)
    definition = replace(definition,interactions=(pair,))
    manager = GeneralS3PatchManager(model,definition,example['SETTINGS'])
    owner = prepare_normal_s3_skin_path(model,manager,modes)
    q = np.zeros(7);q[5:] = GAP
    rate = np.zeros(7);rate[4] = .0016
    lower,failure = owner.sweep(manager,modes@q,modes@q,modes@rate,-modes@rate,1.,
                                event_gap_tolerance_m=1.e-10)
    assert lower is None
    assert 'corner pair' in failure


def test_deforming_source_and_target_have_nonlinear_coupled_reference():
    model,definition,modes = build()
    velocity = np.r_[np.full(4,-.1),np.zeros(3)]
    result = solve_transient_contact(model,definition,
        TransientConfig(2.5e-5,.003,initial_velocity=modes@velocity),
        contact_settings=example['SETTINGS'],solver_controls=CONTROLS)
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,MASS,atol=1.e-11)
    structural = modes.T@assemble_stiffness_matrix(model)[0]@modes
    expected = reference(result.times,structural)
    q,v,a = (np.asarray(value)@modes for value in
             (result.displacements,result.velocities,result.accelerations))
    for value,index,tolerance in ((q,0,1.e-10),(v,1,1.e-6),(a,2,1.e-2)):
        np.testing.assert_allclose(value,expected[:,index],rtol=0.,atol=tolerance)
    assert np.max(np.ptp(q[:,:4],axis=1)) > 1.e-9
    assert np.max(np.ptp(q[:,4:],axis=1)) > 1.e-8
    momentum = v@np.diag(MASS)
    assert np.max(np.abs(momentum-momentum[0])) < 1.e-7
    assert min(result.swept_minimum_gaps) >= -2.e-5
    assert np.max(np.abs(result.constraint_reactions@modes)) < 1.e-5
    energy = (.5*np.einsum('ij,jk,ik->i',v,MASS,v)
              +.5*np.einsum('ij,jk,ik->i',q,structural,q)
              +np.array([row.regularization_storage_j for row in result.contact]))
    assert np.max(np.abs(energy/energy[0]-1.)) < .01
    Path(os.environ['RUNNER_TEMP'],'deforming_s3_impact.json').write_text(json.dumps(dict(
        times_s=result.times.tolist(),q=q.tolist(),v=v.tolist(),a=a.tolist(),
        mass=MASS.tolist(),structural=structural.tolist(),energy=energy.tolist(),
        contact=[row.to_dict() for row in result.contact],cutbacks=result.cutbacks,
        cutback_diagnostics=result.cutback_diagnostics,
        localized_events_s=result.localized_contact_event_times,
        endpoint_queries=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations,
        swept_gap=result.swept_minimum_gaps.tolist()),indent=2),encoding='utf-8')


def test_temporal_halving_improves_trajectory_with_localized_events():
    """One registered development discriminator, not a refinement campaign.

    The reference integrates independent projected P1 contact forces, using the
    existing accepted S3 structural operator as input. It is not independent
    spatial qualification of that operator or a contact engineering acceptance.
    """
    model,_,modes = build()
    structural = modes.T@assemble_stiffness_matrix(model)[0]@modes
    initial = np.r_[np.zeros(7),np.full(4,-.1),np.zeros(3)]
    def derivative(time,state):
        q,v = state[:7],state[7:]
        return np.r_[v,-(structural@q+contact_force(q))/np.diag(MASS)]
    continuous = solve_ivp(derivative,(0.,.003),initial,method='DOP853',
        rtol=2.e-11,atol=np.r_[np.full(7,1.e-14),np.full(7,1.e-12)],
        max_step=1.e-5,dense_output=True)
    assert continuous.success,continuous.message
    records = []
    for dt in (2.5e-5,1.25e-5):
        _,result,mapping = deforming['solve'](dt=dt)
        q,v = (np.asarray(value)@mapping for value in
               (result.displacements,result.velocities))
        expected = continuous.sol(result.times).T
        energy = (.5*np.einsum('ij,jk,ik->i',v,MASS,v)
                  +.5*np.einsum('ij,jk,ik->i',q,structural,q)
                  +np.array([row.regularization_storage_j for row in result.contact]))
        records.append(dict(dt_s=dt,states=len(result.times),cutbacks=result.cutbacks,
            displacement_error_on_gap_scale=float(np.max(np.abs(q-expected[:,:7]))/GAP),
            velocity_error_on_initial_speed_scale=float(np.max(np.abs(v-expected[:,7:]))/.1),
            reachable_energy_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
            minimum_swept_gap_m=float(min(result.swept_minimum_gaps))))
    Path(os.environ['RUNNER_TEMP'],'deforming_s3_temporal.json').write_text(json.dumps(
        dict(reference=dict(method='independent DOP853',rtol=2.e-11,max_step_s=1.e-5,
             nfev=continuous.nfev,displacement_scale_m=GAP,velocity_scale_m_s=.1),
             runs=records),indent=2),encoding='utf-8')
    for row in records:
        assert row['displacement_error_on_gap_scale'] < .02
        assert row['velocity_error_on_initial_speed_scale'] < .02
        assert row['reachable_energy_relative_defect'] < .01
        assert row['minimum_swept_gap_m'] >= -2.e-5
    # The V14 event method has a distinct localization error floor. Its energy
    # refinement is checked by the separate same-dt event-tolerance experiment;
    # the failed strict dt-only energy comparison remains retained evidence.
    for key in ('displacement_error_on_gap_scale','velocity_error_on_initial_speed_scale'):
        assert records[1][key] < records[0][key],records


@pytest.mark.parametrize('kind',('tangent','rotation','unrepresented','complex','nonfinite','empty'))
def test_normal_skin_admission_rejects_unrepresented_or_invalid_motion(kind):
    model,definition,modes = build()
    manager = GeneralS3PatchManager(model,definition,example['SETTINGS'])
    if kind == 'empty':
        with pytest.raises(ValueError,match='normal S3'):
            prepare_normal_s3_skin_path(model,manager,np.zeros_like(modes))
        return
    owner = prepare_normal_s3_skin_path(model,manager,modes)
    motion = np.zeros(42)
    if kind == 'tangent':motion[0] = 1.e-20
    elif kind == 'rotation':motion[3] = 1.e-20
    elif kind == 'complex':motion = motion.astype(complex)
    elif kind == 'nonfinite':motion[2] = np.nan
    else:
        owner = replace(owner,full_size=43);motion = np.r_[motion,1.e-20]
    with pytest.raises(ValueError,match='normal S3'):
        owner.require_motion(motion)


@pytest.mark.parametrize('change',({'allow_fixed_projection_s3_target_dev':True},
    {'q4_plane_event_subdivision_dev':True},{'allow_normal_s3_skin_hermite_dev':1},
    {'allow_s3_descriptor_dev':True}))
def test_normal_profile_requires_explicit_exclusive_controls(change):
    with pytest.raises(ValueError,match='normal S3'):
        replace(CONTROLS,**change)


def test_normal_v15_active_restart_binds_method_features_and_physical_budgets():
    _,whole,_ = deforming['solve']()
    index = next(i for i,row in enumerate(whole.contact) if row.active_pressure_dofs)
    _,prefix,_ = deforming['solve'](end=float(whole.times[index]),emit=True)
    checkpoint = prefix.restart_checkpoint
    assert checkpoint['version'] == 15
    assert checkpoint['state']['plane_pressure_features'] == [['pair',node] for node in (1,2,3,4)]
    _,resumed,_ = deforming['solve'](checkpoint=checkpoint)
    for name,tolerance in (('displacements',1.e-10),('velocities',1.e-6),('accelerations',1.e-2)):
        actual = np.concatenate((getattr(prefix,name)[:-1],getattr(resumed,name)))
        np.testing.assert_allclose(actual,getattr(whole,name),rtol=0.,atol=tolerance)
    assert [row.active_pressure_dofs for row in (*prefix.contact[:-1],*resumed.contact)] == [row.active_pressure_dofs for row in whole.contact]
    for key,value in (('maximum_gap_triangles',9999),('maximum_swept_subdivisions',13)):
        with pytest.raises(ValueError,match='analysis'):
            deforming['solve'](checkpoint=checkpoint,settings=replace(example['SETTINGS'],**{key:value}))
    with pytest.raises(ValueError,match='analysis'):
        deforming['solve'](checkpoint=checkpoint,
            controls=replace(CONTROLS,plane_event_gap_tolerance_m=2.e-10))
    bad = deepcopy(checkpoint)
    bad['state']['plane_pressure_features'].reverse()
    bad['checkpoint_sha256'] = analysis_configuration_fingerprint({key:value for key,value in bad.items() if key!='checkpoint_sha256'})
    with pytest.raises(ValueError,match='pressure-feature'):
        deforming['solve'](checkpoint=bad)
    for key,value in (('plane_event_time_tolerance_s',2.e-10),
            ('maximum_plane_event_endpoint_evaluations',63),
            ('maximum_plane_event_stage_evaluations',127),('maximum_event_localizations',11)):
        with pytest.raises(ValueError,match='analysis'):
            deforming['solve'](checkpoint=checkpoint,controls=replace(CONTROLS,**{key:value}))
    legacy=deepcopy(checkpoint)
    legacy['schema']='ANYSOLVER_NORMAL_S3_SKIN_HERMITE_TRANSIENT_CONTACT_DEV_CHECKPOINT_V13'
    legacy['version']=13
    legacy['checkpoint_sha256']=analysis_configuration_fingerprint({key:value for key,value in legacy.items()
        if key!='checkpoint_sha256'})
    with pytest.raises(ValueError,match='version'):
        deforming['solve'](checkpoint=legacy)
    legacy['schema']='ANYSOLVER_NORMAL_S3_SKIN_HERMITE_EVENT_TRANSIENT_CONTACT_DEV_CHECKPOINT_V14'
    legacy['version']=14
    legacy['checkpoint_sha256']=analysis_configuration_fingerprint({key:value for key,value in legacy.items()
        if key!='checkpoint_sha256'})
    with pytest.raises(ValueError,match='version'):
        deforming['solve'](checkpoint=legacy)


def test_normal_unresolved_matched_physical_proof_cuts_back_before_commit(monkeypatch):
    from anyfesolver.general_contact import normal_s3_skin_path as module
    original = module.NormalS3SkinPath.overlap_sweep
    calls = 0
    def failed(self,*args,**kwargs):
        nonlocal calls
        proof = original(self,*args,**kwargs);calls += 1
        if calls == 1:return None,'owned matched physical proof failure'
        return proof
    monkeypatch.setattr(module.NormalS3SkinPath,'overlap_sweep',failed)
    _,result,_ = deforming['solve'](end=.0001)
    assert result.cutbacks == 1
    assert result.times[1] == pytest.approx(1.25e-5)
    assert 'owned matched physical' in result.cutback_diagnostics[0]['reason']
    assert active_contact_manager() is None


def test_normal_v15_open_restart_continues_after_contact_release():
    _,whole,_ = deforming['solve']()
    index = int(np.argmin(np.abs(whole.times-.0025)))
    assert np.all(np.asarray(whole.contact[index].normalized_weighted_gap_m) > 0.)
    assert np.linalg.norm(whole.contact[index].integration_force) == 0.
    _,prefix,_ = deforming['solve'](end=float(whole.times[index]),emit=True)
    assert prefix.restart_checkpoint['schema'].endswith('_V15')
    _,resumed,_ = deforming['solve'](checkpoint=prefix.restart_checkpoint)
    np.testing.assert_allclose(resumed.times,whole.times[index:],rtol=0.,atol=1.e-15)
    for name,tolerance in (('displacements',1.e-11),('velocities',1.e-7),('accelerations',1.e-3)):
        np.testing.assert_allclose(getattr(resumed,name),getattr(whole,name)[index:],rtol=0.,atol=tolerance)
    assert active_contact_manager() is None


def test_normal_unresolved_pressure_proof_cuts_back_before_commit(monkeypatch):
    from anyfesolver.general_contact import normal_s3_skin_path as module
    original=module.NormalS3SkinPath.pressure_sweep
    calls=0
    def failed(self,*args,**kwargs):
        nonlocal calls
        proof=original(self,*args,**kwargs);calls+=1
        if calls==1:return replace(proof,success=False,status='owned pressure proof failure')
        return proof
    monkeypatch.setattr(module.NormalS3SkinPath,'pressure_sweep',failed)
    _,result,_=deforming['solve'](end=.0001)
    assert result.cutbacks==1
    assert result.times[1]==pytest.approx(1.25e-5)
    assert 'owned pressure proof failure' in result.cutback_diagnostics[0]['reason']
    assert result.cutback_diagnostics[0]['plane_stage_evaluations']>0
    assert active_contact_manager() is None


def test_normal_cutback_reports_rejected_endpoint_and_stage_work(monkeypatch):
    from anyfesolver.general_contact import normal_s3_skin_path as module
    original=module.NormalS3SkinPath.pressure_sweep
    injected=False
    def failed(self,manager,start,end,v0,v1,dt,**kwargs):
        nonlocal injected
        proof=original(self,manager,start,end,v0,v1,dt,**kwargs)
        gaps=manager.evaluate(end,tangent=False)[2].normalized_weighted_gap_m
        if not injected and np.mean(v1[self.source_dofs[:,2]])>0. and min(abs(x) for x in gaps)<1.e-10:
            injected=True
            return replace(proof,success=False,status='owned post-endpoint pressure proof failure')
        return proof
    monkeypatch.setattr(module.NormalS3SkinPath,'pressure_sweep',failed)
    _,result,_=deforming['solve']()
    assert injected
    records=[row for row in result.cutback_diagnostics if 'owned post-endpoint' in row['reason']]
    assert len(records)==1
    assert records[0]['plane_endpoint_requests']>0
    assert records[0]['plane_stage_evaluations']>records[0]['plane_endpoint_requests']
    assert active_contact_manager() is None


def test_normal_pressure_proof_cancellation_restores_context(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact import normal_s3_skin_path as module
    token=CancellationToken()
    original=module.NormalS3SkinPath.pressure_sweep
    def cancelled(self,*args,**kwargs):
        token.cancel('owned mortar pressure-path cancellation')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(module.NormalS3SkinPath,'pressure_sweep',cancelled)
    model,definition,modes=deforming['build_model']()
    fingerprint=model_configuration_fingerprint(model)
    with pytest.raises(SolveCancelled,match='owned mortar'):
        solve_transient_contact(model,definition,
            TransientConfig(2.5e-5,.0001,initial_velocity=modes@np.r_[np.full(4,-.1),np.zeros(3)]),
            contact_settings=example['SETTINGS'],solver_controls=CONTROLS,cancellation_token=token)
    assert active_contact_manager() is None
    assert model_configuration_fingerprint(model)==fingerprint


def test_normal_stage_budget_exhaustion_restores_model_and_owner():
    model,definition,modes=deforming['build_model']()
    fingerprint=model_configuration_fingerprint(model)
    with pytest.raises(RuntimeError,match='stage-work budget exhausted'):
        solve_transient_contact(model,definition,
            TransientConfig(2.5e-5,.0021,initial_velocity=modes@np.r_[np.full(4,-.1),np.zeros(3)]),
            contact_settings=example['SETTINGS'],solver_controls=replace(CONTROLS,
                maximum_plane_event_stage_evaluations=1))
    assert active_contact_manager() is None
    assert model_configuration_fingerprint(model)==fingerprint


def test_event_tolerance_refinement_improves_localized_energy_balance():
    records=[]
    for tolerance in (1.e-10,1.e-12):
        model,result,modes=deforming['solve'](controls=replace(CONTROLS,
            plane_event_gap_tolerance_m=tolerance,plane_event_time_tolerance_s=tolerance))
        q,v=(np.asarray(value)@modes for value in (result.displacements,result.velocities))
        structural=modes.T@assemble_stiffness_matrix(model)[0]@modes
        energy=(.5*np.einsum('ij,jk,ik->i',v,MASS,v)+.5*np.einsum('ij,jk,ik->i',q,structural,q)
            +np.array([row.regularization_storage_j for row in result.contact]))
        records.append(dict(event_gap_tolerance_m=tolerance,event_time_tolerance_s=tolerance,
            reachable_energy_relative_defect=float(np.max(np.abs(energy/energy[0]-1.))),
            states=len(result.times),cutbacks=result.cutbacks,
            endpoint_queries=result.event_endpoint_evaluations,
            stage_evaluations=result.plane_event_stage_evaluations,
            localized_events_s=result.localized_contact_event_times,
            minimum_swept_gap_m=float(min(result.swept_minimum_gaps))))
    Path(os.environ['RUNNER_TEMP'],'deforming_s3_event_refinement.json').write_text(json.dumps(records,indent=2))
    assert records[1]['reachable_energy_relative_defect']<records[0]['reachable_energy_relative_defect']
    assert all(row['reachable_energy_relative_defect']<.01 and row['minimum_swept_gap_m']>=-2.e-5
               for row in records)


def test_normal_cancellation_inside_physical_proof_restores_model_and_owner(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact import normal_s3_skin_path as module
    model,definition,modes = build()
    fingerprint = model_configuration_fingerprint(model)
    token = CancellationToken();original = module.s3_overlap_hermite_gap_bounds
    def cancelled(*args,**kwargs):
        token.cancel('owned deforming-skin cancellation')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'s3_overlap_hermite_gap_bounds',cancelled)
    with pytest.raises(SolveCancelled,match='owned deforming-skin'):
        solve_transient_contact(model,definition,
            TransientConfig(2.5e-5,.0001,initial_velocity=modes@np.r_[np.full(4,-.1),np.zeros(3)]),
            contact_settings=example['SETTINGS'],solver_controls=CONTROLS,
            cancellation_token=token,emit_restart_checkpoint=True)
    assert model_configuration_fingerprint(model) == fingerprint
    assert active_contact_manager() is None


@pytest.mark.parametrize('kind',('complex','boolean','text'))
def test_normal_initial_conditions_do_not_silently_convert_invalid_types(kind):
    model,definition,modes = build()
    velocity = modes@np.r_[np.full(4,-.1),np.zeros(3)]
    if kind == 'complex':velocity = velocity.astype(complex)
    elif kind == 'boolean':velocity = velocity.astype(bool)
    else:velocity = velocity.astype(str)
    with pytest.raises(ValueError,match='real numeric'):
        solve_transient_contact(model,definition,TransientConfig(2.5e-5,.0001,initial_velocity=velocity),
            contact_settings=example['SETTINGS'],solver_controls=CONTROLS)
