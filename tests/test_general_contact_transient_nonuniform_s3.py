"""Coupled fixed-projection S3 impact with an independent pressure/step oracle.

The oracle uses the already qualified structural stiffness as an input. It
verifies new contact/Newmark coupling, not independently the S3 spatial operator.
"""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import itertools
import json
import os
import runpy
import numpy as np
import pytest
from anyfesolver import ContactTransientSolverControls, TransientConfig, solve_transient_contact
from anyfesolver.assembly import assemble_mass_matrix
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.runtime import active_contact_manager
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint, model_configuration_fingerprint

example = runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_nonuniform_s3_impact_dev.py'))
MASS, B, SUPPORT, KN, GAP = (example[key] for key in ('MASS','B','SUPPORT','KN','GAP'))


def reference(times, structural, *, mass=MASS, gradient=B):
    """Enumerate all 16 unilateral P1 pressure branches; no production contact."""
    q = np.zeros(len(mass)); v = np.zeros_like(q); v[0] = -.1; a = np.zeros_like(q)
    rows = [(q.copy(), v.copy(), a.copy())]
    for start, end in zip(times, times[1:]):
        h = end-start; beta = .25*h*h
        prediction = q+h*v+beta*a
        choices = []
        for flags in itertools.product((False,True), repeat=4):
            weight = SUPPORT*np.array(flags)
            stiffness = structural+KN*gradient.T@np.diag(weight)@gradient
            constant = KN*gradient.T@(weight*GAP)
            trial_a = np.linalg.solve(mass+beta*stiffness, -stiffness@prediction-constant)
            trial = prediction+beta*trial_a
            gaps = GAP+gradient@trial
            if np.any(np.where(flags, gaps>1.e-13, gaps<-1.e-13)):
                continue
            choices.append((trial,trial_a))
        assert choices, 'independent pressure branch has no solution'
        trial,trial_a = choices[0]
        v += .5*h*(a+trial_a); q,a = trial,trial_a
        rows.append((q.copy(),v.copy(),a.copy()))
    return np.array(rows)


def test_full_target_normal_motion_has_two_slope_reference_and_physical_mass():
    model,result,modes = example['solve'](target_mode='xy')
    mass,gradient = example['MASS_XY'],example['B_XY']
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,mass,
                              rtol=2.e-13,atol=1.e-11)
    structural = modes.T@assemble_stiffness_matrix(model)[0]@modes
    expected = reference(result.times,structural,mass=mass,gradient=gradient)
    q,v,a = (example['modal'](values,modes) for values in
             (result.displacements,result.velocities,result.accelerations))
    for actual,index,tolerance in ((q,0,1.e-10),(v,1,1.e-6),(a,2,1.e-2)):
        np.testing.assert_allclose(actual,expected[:,index],rtol=0.,atol=tolerance)
    assert np.all(np.max(np.abs(q[:,2:]),axis=0)>1.e-8)
    momentum = v[:,:2]@np.diag(mass)[:2]
    assert np.max(np.abs(momentum-momentum[0]))<1.e-7
    for position,row in zip(q,result.contact):
        force = SUPPORT*(-KN*np.minimum(GAP+gradient@position,0.))
        # B columns carry negative coordinates: physical y/x are -B + 1/3.
        expected_moment = np.array([force@(-gradient[:,3]+1/3),
                                   -force@(-gradient[:,2]+1/3),0.])
        np.testing.assert_allclose(row.integration_moment,expected_moment,atol=1.e-3)
        np.testing.assert_allclose(np.array(row.integration_moment)+row.target_moment,0.,atol=1.e-7)
    assert min(result.swept_minimum_gaps)>=-2.e-5
    Path(os.environ['RUNNER_TEMP'],'nonuniform_s3_xy.json').write_text(json.dumps(dict(
        times_s=result.times.tolist(),q=q.tolist(),v=v.tolist(),a=a.tolist(),
        mass=mass.tolist(),structural_stiffness=structural.tolist(),
        pressure_gradient=gradient.tolist(),contact=[row.to_dict() for row in result.contact],
        events=result.localized_contact_event_times,cutbacks=result.cutbacks,
        swept_gap=result.swept_minimum_gaps.tolist()),indent=2),encoding='utf-8')


def test_declared_lumped_mass_and_retained_structural_scope():
    model,_,modes = example['build_model']()
    np.testing.assert_allclose(modes.T@assemble_mass_matrix(model)[0]@modes,
        MASS, rtol=2.e-13, atol=1.e-11)
    structural = modes.T@assemble_stiffness_matrix(model)[0]@modes
    assert structural[2,2] > 0. # Real S3 deformation, not guided rigid recoil.
    expected = np.zeros((3,3)); expected[2,2] = structural[2,2]
    np.testing.assert_allclose(structural, expected, atol=1.e-8)


@pytest.fixture(scope='module')
def impact():
    model,result,modes = example['solve']()
    record = example['modal'](result.displacements,modes)
    Path(os.environ['RUNNER_TEMP'],'nonuniform_s3_impact.json').write_text(json.dumps(dict(
        times_s=result.times.tolist(), q_m=record.tolist(),
        velocities=example['modal'](result.velocities,modes).tolist(),
        contact=[row.to_dict() for row in result.contact],
        swept_gaps=result.swept_minimum_gaps.tolist(), events=result.localized_contact_event_times,
        cutbacks=result.cutbacks, diagnostics=result.cutback_diagnostics,
        endpoint_requests=result.event_endpoint_evaluations,
        stage_evaluations=result.plane_event_stage_evaluations),indent=2),encoding='utf-8')
    return model,result,modes


def test_nonuniform_target_has_coupled_work_reference_and_momentum(impact):
    model,result,modes = impact
    structural = modes.T@assemble_stiffness_matrix(model)[0]@modes
    expected = reference(result.times,structural)
    for values,index,tolerance in ((result.displacements,0,1.e-10),
                                  (result.velocities,1,1.e-6),
                                  (result.accelerations,2,1.e-2)):
        actual = example['modal'](values,modes)
        np.testing.assert_allclose(actual,expected[:,index],rtol=0.,atol=tolerance)
        np.testing.assert_allclose(values,actual@modes.T,rtol=0.,atol=tolerance)
    q,v = expected[:,0],expected[:,1]
    assert np.max(np.abs(q[:,2])) > 1.e-8
    assert v[-1,1] < -.01
    momentum = v[:,:2]@np.diag(MASS)[:2]
    assert np.max(np.abs(momentum-momentum[0])) < 1.e-7
    for position,row in zip(q,result.contact):
        gaps = GAP+B@position
        pressure = -KN*np.minimum(gaps,0.)
        force = float(SUPPORT@pressure)
        np.testing.assert_allclose(row.integration_force,[0.,0.,force],atol=1.e-3)
        np.testing.assert_allclose(np.array(row.integration_force)+row.target_force,0.,atol=1.e-7)
        np.testing.assert_allclose(np.array(row.integration_moment)+row.target_moment,0.,atol=1.e-7)
        assert row.penalty_energy == 0.
        assert row.regularization_storage_j == pytest.approx(.5*KN*float(SUPPORT@np.minimum(gaps,0.)**2),abs=1.e-7)
    assert result.profile == 'FIXED_PROJECTION_S3_TARGET_NEWMARK_EVENT_DEV_V1'
    assert min(result.swept_minimum_gaps) >= -2.e-5
    assert np.max(np.abs(result.constraint_reactions@modes)) < 1.e-6
    assert active_contact_manager() is None


@pytest.mark.parametrize('change',({'allow_guided_shell_pair_dev':True},
    {'q4_plane_event_subdivision_dev':False}, {'allow_s3_descriptor_dev':True},
    {'allow_fixed_projection_s3_target_dev':1}))
def test_nonuniform_profile_requires_explicit_exclusive_controls(change):
    with pytest.raises(ValueError):
        replace(example['CONTROLS'],**change)


def test_physical_hermite_gap_and_reachable_storage_are_checked(impact):
    model,result,modes = impact
    q,v = (example['modal'](values,modes) for values in (result.displacements,result.velocities))
    # Independent nodal skin gaps for each of the three target vertices.
    physical = np.column_stack((np.ones(3), -np.ones(3), -np.array([-7/3,14/3,-7/3])))
    exact_lowers = []
    for i,h in enumerate(np.diff(result.times)):
        start = GAP+physical@q[i]; end = GAP+physical@q[i+1]
        rate0,rate1 = h*(physical@v[i]),h*(physical@v[i+1])
        coefficient = np.column_stack((start,rate0,3*(end-start)-2*rate0-rate1,
                                        2*(start-end)+rate0+rate1))
        values = []
        for row in coefficient:
            points = [0.,1.]
            for root in np.polynomial.polynomial.polyroots([row[1],2*row[2],3*row[3]]):
                if abs(root.imag)<1.e-12 and 0.<root.real<1.:
                    points.append(float(root.real))
            values.extend(np.polynomial.polynomial.polyval(points,row))
        exact_lowers.append(min(values))
    assert min(exact_lowers) >= -2.e-5
    assert np.all(result.swept_minimum_gaps[1:] <= np.array(exact_lowers)+1.e-12)
    structural = modes.T@assemble_stiffness_matrix(model)[0]@modes
    energy = (.5*np.einsum('ij,jk,ik->i',v,MASS,v)
              +.5*np.einsum('ij,jk,ik->i',q,structural,q)
              +np.array([row.regularization_storage_j for row in result.contact]))
    defect = float(np.max(np.abs(energy/energy[0]-1.)))
    assert defect < .01
    Path(os.environ['RUNNER_TEMP'],'nonuniform_s3_accounting.json').write_text(json.dumps(dict(
        independent_swept_min_m=min(exact_lowers), production_swept_min_m=float(min(result.swept_minimum_gaps)),
        energy_relative_defect=defect, energy_j=energy.tolist()),indent=2),encoding='utf-8')


def test_nonuniform_active_v12_restart_and_identity_rejection(impact):
    _,whole,_ = impact
    index = next(i for i,row in enumerate(whole.contact) if row.active_pressure_dofs
                 and abs(whole.times[i]-whole.times[i-1]-2.5e-5)<1.e-12)
    _,prefix,_ = example['solve'](end=float(whole.times[index]),emit=True)
    checkpoint = prefix.restart_checkpoint
    assert checkpoint['version'] == 12
    assert checkpoint['state']['plane_pressure_features'] == [['pair',node] for node in (1,2,3,4)]
    _,resumed,_ = example['solve'](checkpoint=checkpoint)
    for name,tolerance in (('displacements',1.e-10),('velocities',1.e-6),('accelerations',1.e-2)):
        actual = np.concatenate((getattr(prefix,name)[:-1],getattr(resumed,name)))
        np.testing.assert_allclose(actual,getattr(whole,name),rtol=0.,atol=tolerance)
    assert [row.active_pressure_dofs for row in (*prefix.contact[:-1],*resumed.contact)] == [row.active_pressure_dofs for row in whole.contact]
    for alter in ('row_order','pair_name','boolean_node','version'):
        bad = deepcopy(checkpoint)
        features = bad['state']['plane_pressure_features']
        if alter == 'row_order': features.reverse()
        elif alter == 'pair_name': features[0][0] = 'other'
        elif alter == 'boolean_node': features[0][1] = True
        else: bad['version'] = 11
        bad['checkpoint_sha256'] = analysis_configuration_fingerprint({key:value for key,value in bad.items() if key!='checkpoint_sha256'})
        with pytest.raises(ValueError,match='pressure-feature|version'):
            example['solve'](checkpoint=bad)
    with pytest.raises(ValueError,match='analysis'):
        example['solve'](checkpoint=checkpoint,
            controls=replace(example['CONTROLS'],batch_plane_event_fronts_dev=False))
    model,definition,modes = example['build_model']()
    with pytest.raises(ValueError,match='analysis'):
        solve_transient_contact(model,definition,
            TransientConfig(2.5e-5,.003,initial_velocity=modes@example['INITIAL_VELOCITY']),
            contact_settings=replace(example['SETTINGS'],maximum_gap_triangles=9999),
            solver_controls=example['CONTROLS'],restart_checkpoint=checkpoint)


def test_nonuniform_default_and_unrepresented_source_motion_rejected():
    model,definition,modes = example['build_model']()
    config = TransientConfig(2.5e-5,.0001,initial_velocity=modes@example['INITIAL_VELOCITY'])
    with pytest.raises(ValueError):
        solve_transient_contact(model,definition,config,contact_settings=example['SETTINGS'])
    model.constraint_equations.pop(0)
    with pytest.raises(ValueError,match='fixed-projection'):
        solve_transient_contact(model,definition,config,contact_settings=example['SETTINGS'],
                                solver_controls=example['CONTROLS'])
    assert active_contact_manager() is None


@pytest.mark.parametrize('failure',('cancel','exception'))
def test_nonuniform_rejected_candidate_cleans_owner_without_partial_checkpoint(monkeypatch,failure):
    from anyfesolver.control import CancellationToken,SolveCancelled
    model,definition,modes = example['build_model']()
    fingerprint = model_configuration_fingerprint(model)
    token = CancellationToken(); calls = 0
    original = GeneralS3PatchManager.evaluate
    def injected(manager,*args,**kwargs):
        nonlocal calls
        value = original(manager,*args,**kwargs); calls += 1
        if calls == 3:
            if failure == 'cancel': token.cancel('owned nonuniform cancellation')
            else: raise ArithmeticError('owned nonuniform candidate exception')
        return value
    monkeypatch.setattr(GeneralS3PatchManager,'evaluate',injected)
    with pytest.raises(SolveCancelled if failure=='cancel' else ArithmeticError,match='owned nonuniform'):
        solve_transient_contact(model,definition,
            TransientConfig(2.5e-5,.0001,initial_velocity=modes@example['INITIAL_VELOCITY']),
            contact_settings=example['SETTINGS'],solver_controls=example['CONTROLS'],
            cancellation_token=token,emit_restart_checkpoint=True)
    assert model_configuration_fingerprint(model) == fingerprint
    assert active_contact_manager() is None
    monkeypatch.setattr(GeneralS3PatchManager,'evaluate',original)
    result = solve_transient_contact(model,definition,
        TransientConfig(2.5e-5,.0001,initial_velocity=modes@example['INITIAL_VELOCITY']),
        contact_settings=example['SETTINGS'],solver_controls=example['CONTROLS'])
    assert len(result.times) == 5


def test_unresolved_physical_sweep_cuts_back_before_acceptance(monkeypatch):
    from anyfesolver.general_contact import shell_pair_hermite_sweep as module
    original = module.shell_pair_hermite_lower
    calls = 0
    def bounded_failure(*args,**kwargs):
        nonlocal calls
        calls += 1
        result = original(*args,**kwargs)
        if calls == 1:
            return replace(result,success=False,lower_gap=None,detail='owned unresolved sweep')
        return result
    monkeypatch.setattr(module,'shell_pair_hermite_lower',bounded_failure)
    _,result,modes = example['solve'](end=.0001)
    assert result.cutbacks == 1
    assert result.times[1] == pytest.approx(1.25e-5)
    q = example['modal'](result.displacements,modes)
    expected = np.zeros_like(q);expected[:,0] = -.1*result.times
    np.testing.assert_allclose(q,expected,atol=1.e-16)
    assert active_contact_manager() is None


def test_cancellation_inside_physical_sweep_restores_owner(monkeypatch):
    from anyfesolver.control import CancellationToken,SolveCancelled
    from anyfesolver.general_contact import shell_pair_hermite_sweep as module
    model,definition,modes = example['build_model']()
    fingerprint = model_configuration_fingerprint(model)
    token = CancellationToken()
    original = module.shell_pair_hermite_lower
    def cancelled(*args,**kwargs):
        token.cancel('owned physical-sweep cancellation')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'shell_pair_hermite_lower',cancelled)
    with pytest.raises(SolveCancelled,match='owned physical-sweep cancellation'):
        solve_transient_contact(model,definition,
            TransientConfig(2.5e-5,.0001,initial_velocity=modes@example['INITIAL_VELOCITY']),
            contact_settings=example['SETTINGS'],solver_controls=example['CONTROLS'],
            cancellation_token=token,emit_restart_checkpoint=True)
    assert model_configuration_fingerprint(model) == fingerprint
    assert active_contact_manager() is None
