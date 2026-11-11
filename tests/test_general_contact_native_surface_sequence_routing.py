"""Explicit closed-event path routing and restart identity development."""
from dataclasses import asdict
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
from anyfesolver.general_contact.native_surface_event_sequence import evaluate_closed_event_sequence
from anyfesolver.general_contact.native_surface_path import material_path_pose
from test_general_contact_native_surface_closed_sequence import _recontact


def _policy(pair,policy):
    return NativeS3MaterialPointFriction(pair.source,pair.target,
        reference_weight_m2=pair.reference_weight_m2,
        normal_stiffness_Pa_per_m=pair.normal_stiffness_Pa_per_m,
        tangential_stiffness_Pa_per_m=pair.tangential_stiffness_Pa_per_m,
        friction_coefficient=pair.friction_coefficient,activation_policy=policy)


def test_explicit_closed_sequence_routing_retains_tangent_and_rejects_legacy():
    old,start,history,t,w,s=_recontact()
    pair=_policy(old,'transverse_event_sequence')
    expected=evaluate_closed_event_sequence(pair,start,history,t,w,s)
    actual=pair.evaluate_path(start,history,t,w,s)
    np.testing.assert_array_equal(actual.contribution.residual,expected.residual)
    np.testing.assert_array_equal(actual.contribution.spatial_tangent(),expected.spatial_tangent())
    assert actual.contribution.trial==expected.trial
    assert actual.contribution.history_evaluations==34
    assert pair._definition()['sequence_profile']=='NATIVE_S3_OPEN_CLOSED_EVENT_SEQUENCE_POINT_DEV_V2'
    with pytest.raises(ValueError,match='requires subdivision'):
        _policy(old,'transverse_events').evaluate_path(start,history,t,w,s)
    with pytest.raises((ValueError,MemoryError)):
        pair.evaluate_path(start,history,t,w,s,maximum_intervals=1)
    assert history.cumulative_dissipation_density==2.


def test_sequence_no_event_persistence_retains_incoming_elastic_history():
    old,start,history,t,w,s=_recontact()
    pair=_policy(old,'transverse_event_sequence')
    w[:]=0.;t[:,0]=.00002
    expected=pair.evaluate(start,material_path_pose(start,t,w,s,1.),history)
    actual=pair.evaluate_path(start,history,t,w,s).contribution
    assert actual.trial==expected.trial
    np.testing.assert_array_equal(actual.residual,expected.residual)
    np.testing.assert_array_equal(actual.spatial_tangent(),expected.spatial_tangent())
    assert actual.trial.candidate_state.partner_key==history.partner_key


def test_sequence_v2_checkpoint_roundtrip_and_v1_rejection(monkeypatch):
    from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
    from test_general_contact_native_surface_replay_owner import _patch,normal_settings,friction_settings
    def make():
        return _patch(activation_policy='transverse_event_sequence',gap_check='straight_patch',
            normal_integration='straight_segmented',normal_integration_settings=normal_settings(),
            friction_integration='replay',friction_integration_settings=friction_settings())[1]
    original=NativeS3MaterialPointFriction._definition
    def legacy(self):
        definition=original(self)
        if 'sequence_profile' in definition:
            definition['sequence_profile']='NATIVE_S3_OPEN_EVENT_SEQUENCE_POINT_DEV_V1'
        return definition
    with monkeypatch.context() as patch:
        patch.setattr(NativeS3MaterialPointFriction,'_definition',legacy)
        legacy_checkpoint=NativeS3MaterialTransaction(make()).checkpoint()
    bridge=make();transaction=NativeS3MaterialTransaction(bridge)
    checkpoint=transaction.checkpoint()
    restored=NativeS3MaterialTransaction.restore(bridge,checkpoint,sha256(checkpoint).hexdigest())
    assert restored.checkpoint()==checkpoint
    with pytest.raises(ValueError,match='definition/settings binding'):
        NativeS3MaterialTransaction.restore(bridge,legacy_checkpoint,sha256(legacy_checkpoint).hexdigest())
    assert transaction.checkpoint()==checkpoint


def test_saved_second_load_closed_point_through_history_field(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceReplayBudget,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from test_general_contact_native_surface_replay_owner import _patch,normal_settings,friction_settings
    _,bridge=_patch(activation_policy='transverse_event_sequence',gap_check='straight_patch',
        normal_integration='straight_segmented',normal_integration_settings=normal_settings(),
        friction_integration='replay',friction_integration_settings=friction_settings())
    folder=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    saved=json.loads((folder/'replay_owner_second_load_rejected_path.json').read_text())
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    budget=NativeSurfaceReplayBudget(30)
    value=field.query(.79,np.pi+.013,budget=budget)
    result={'profile':value.last_contribution.profile,
        'normal_gap_m':value.last_contribution.normal_gap_m,
        'cumulative_dissipation_density':value.history.cumulative_dissipation_density,
        'history_partner':value.history.partner_key,
        'field_identity':field.identity}
    destination=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'saved_closed_point.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert value.last_contribution.normal_gap_m<0.
    assert value.history.partner_key==value.pair.partner_key
    assert np.all(np.isfinite(value.last_contribution.residual))
    assert np.all(np.isfinite(value.last_contribution.spatial_tangent()))


def test_open_ending_sequence_retains_initial_active_history_and_matches_single_release():
    from anyfesolver.general_contact.native_surface_activation import evaluate_release
    old,start,history,t,w,s=_recontact()
    pair=_policy(old,'transverse_event_sequence');w[0,0]=.3
    actual=pair.evaluate_path(start,history,t,w,s).contribution
    expected=evaluate_release(pair,start,history,t,w,s)
    for name in ('dissipated_increment_density','released_storage_density','stored_energy_density'):
        assert getattr(actual.trial,name)==pytest.approx(getattr(expected.trial,name),rel=1e-12,abs=1e-14)
    np.testing.assert_allclose(actual.trial.plastic_slip_increment,expected.trial.plastic_slip_increment,rtol=1e-12,atol=1e-14)
    assert actual.trial.candidate_state.cumulative_dissipation_density==pytest.approx(
        expected.trial.candidate_state.cumulative_dissipation_density,rel=1e-12)
    assert actual.profile=='NATIVE_S3_RELEASED_ENDPOINT_EVENT_SEQUENCE_POINT_DEV_V2'
    assert actual.history_evaluations==pair.event_history_substeps+2
    assert len(actual.event_fractions)==1
    assert actual.trial.candidate_state.partner_key is None


def test_initially_active_three_events_end_open_with_independent_work():
    from dataclasses import replace
    from scipy.optimize import brentq
    from scipy.spatial.transform import Rotation
    from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
    from anyfesolver.general_contact.friction_point import initial_friction_point,trial_coulomb_point
    from test_general_contact_native_surface_s3_friction import _case
    _,old,start,_=_case()
    source=NativeRetainedRoundStation(7,old.source.reference,-.5,.03,0.)
    pair=NativeS3MaterialPointFriction(source,old.target,reference_weight_m2=.01,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.,
        activation_policy='transverse_event_sequence')
    positions=start.positions.copy();positions[:,2]=-.04-.03*.85*1.2+.00001
    cells=start.cell_rotations.copy();cells[0]=Rotation.from_rotvec((1.2,0.,0.)).as_matrix()
    start=replace(start,positions=positions,cell_rotations=cells)
    gap=lambda angle:.03*(np.sin(angle)-.85*angle)+.00001
    history=initial_friction_point(pair.target.physical_pose(start.shell_displacement).material_frame)
    history=trial_coulomb_point(history,material_frame=history.material_frame,
        relative_increment=(0.,.0001,0.),normal_pressure=-1000*gap(1.2),
        friction_coefficient=.3,tangential_stiffness=1000.,partner_key=pair.partner_key).candidate_state
    t=np.tile((0.,0.,.03*.85*2.4),(3,1));w=np.array(((-2.4,0.,0.),(0.,0.,0.)));s=np.zeros(18)
    value=pair.evaluate_path(start,history,t,w,s).contribution
    roots=[brentq(gap,a,b,xtol=1e-14) for a,b in ((.5,1.2),(-.5,.5),(-1.2,-.5))]
    assert value.event_fractions==pytest.approx(tuple((1.2-a)/2.4 for a in roots),abs=2e-13)
    elastic=.0001;dissipation=0.;released=0.
    for first,last in ((1.2,roots[0]),(roots[1],roots[2])):
        previous=.03*np.cos(first)
        for i in range(1,pair.event_history_substeps):
            angle=first+(last-first)*i/pair.event_history_substeps
            y=.03*np.cos(angle);trial=elastic+y-previous;cap=-.3*1000*gap(angle)
            next_elastic=np.clip(trial,-cap/1000,cap/1000)
            dissipation+=cap*abs(trial-next_elastic);elastic=next_elastic;previous=y
        released+=500*elastic**2;elastic=0.
    assert value.trial.dissipated_increment_density==pytest.approx(dissipation,rel=1e-9)
    assert value.trial.released_storage_density==pytest.approx(released,rel=1e-9)
    assert value.history_evaluations==2*pair.event_history_substeps+4
    assert value.trial.candidate_state.partner_key is None
    np.testing.assert_array_equal(value.residual,np.zeros(60))
    np.testing.assert_array_equal(value.spatial_tangent(),np.zeros((60,60)))


def test_closed_sequence_persistence_rejects_released_history():
    from dataclasses import replace
    old,start,history,t,w,s=_recontact()
    pair=_policy(old,'transverse_event_sequence');w[:]=0.
    released=replace(history,partner_key=None,elastic_slip=(0.,0.,0.))
    with pytest.raises(ValueError,match='matching active physical history'):
        pair.evaluate_path(start,released,t,w,s)
    assert released.partner_key is None
