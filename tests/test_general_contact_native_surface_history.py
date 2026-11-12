"""History queries at old/new material coordinates without state interpolation."""
from dataclasses import asdict
from hashlib import sha256

import numpy as np
import pytest

from anyfesolver._ge_beam3_p5_seeded.core import sha
from anyfesolver.general_contact.friction_point import initial_friction_point
from anyfesolver.general_contact.native_surface_history import (
    NativeSurfaceHistoryField,NativeSurfaceMotionSegment,NativeSurfaceReplayBudget)
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
from test_general_contact_native_surface_patch import _patch
from test_general_contact_native_surface_normal import _settings


def _origin(owner):
    original=owner.initial;nodes=owner.context.nodes[0]
    return NativeS3MaterialConfiguration(original.mechanical.positions[nodes],original.mechanical.position_low[nodes],
        original.mechanical.cell_rotations[0],original.shell_displacement)


def _segment(x,z):
    return NativeSurfaceMotionSegment(np.tile((x,0.,z),(3,1)),np.zeros((2,3)),np.zeros(18))


def test_exact_path_pose_reuse_preserves_arithmetic_refusal_and_mutation_isolation():
    from anyfesolver.general_contact.native_surface_path import material_path_pose, _PATH_POSES
    from anyfesolver._ge_beam3_p5.compensated_coordinates import advance
    from anyfesolver._ge_beam3_p5.algebra import rotation
    owner, _ = _patch(activation_policy='transverse_events', gap_check='straight_patch')
    start = _origin(owner)
    translations = np.arange(9, dtype=float).reshape(3, 3)*.00013
    angular = np.array(((.03, -.02, .01), (-.04, .01, .02)))
    shell = np.arange(18, dtype=float)*.00001
    _PATH_POSES.clear()
    pose = material_path_pose(start, translations, angular, shell, .37)
    assert material_path_pose(start, translations.copy(), angular.copy(), shell.copy(), .37) is pose
    for index in np.ndindex(3, 3):
        assert (pose.positions[index], pose.position_low[index]) == advance(
            float(start.positions[index]), float(start.position_low[index]), float(.37*translations[index]))
    assert np.array_equal(pose.cell_rotations, np.array([
        rotation(.37*angular[i]) @ start.cell_rotations[i] for i in range(2)]))
    assert np.array_equal(pose.shell_displacement, start.shell_displacement+.37*shell)
    for value in (pose.positions, pose.position_low, pose.cell_rotations, pose.shell_displacement):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    translations[0, 0] += .001
    changed = material_path_pose(start, translations, angular, shell, .37)
    assert changed is not pose and changed.positions[0, 0] != pose.positions[0, 0]
    bad_low = start.position_low.copy(); bad_low[0, 0] = 1.
    bad = NativeS3MaterialConfiguration(start.positions, bad_low, start.cell_rotations, start.shell_displacement)
    with pytest.raises(ValueError, match='normalized coordinate pair'):
        material_path_pose(bad, translations, angular, shell, .37)
    with pytest.raises(ValueError, match='rotation chart'):
        material_path_pose(start, translations, np.ones((2, 3))*np.pi, shell, .37)
    for fraction in np.linspace(0, 1, 80):
        material_path_pose(start, translations, angular, shell, fraction)
    assert len(_PATH_POSES) == 64


def test_new_material_query_first_touch_matches_scalar_slider_and_keeps_path():
    owner,bridge=_patch(activation_policy='transverse_events',gap_check='straight_patch')
    initial=_origin(owner);path=(_segment(.004,-.004),)
    field=NativeSurfaceHistoryField(bridge.pair,initial,path)
    before=field.identity;budget=NativeSurfaceReplayBudget(2)
    result=field.query(.43,np.pi+.013,budget=budget)
    assert all(result.pair.source.identity!=p.source.identity for p in bridge.pair.pairs)
    point=result.pair.source.evaluate(initial.positions,initial.position_low,initial.cell_rotations).surface_position
    target=result.pair.target.physical_pose(initial.shell_displacement)
    gap=float((point-np.asarray(target.position))@np.asarray(target.material_frame)[:,2])
    fraction=gap/.004;pressure=1e4*(.004-gap)
    traction=min(1000*.004*(1-fraction),.3*pressure)
    value=result.last_contribution
    assert value.event_fraction==pytest.approx(fraction,abs=1e-13)
    assert sum(value.residual[6*i] for i in range(3))==pytest.approx(traction,rel=1e-10)
    assert sum(value.residual[6*i+2] for i in range(3))==pytest.approx(-pressure,rel=1e-10)
    assert field.identity==before and budget.point_paths==2
    assert result.pair.reference_weight_m2==1.


def test_replay_matches_stored_nodes_through_open_recontact_and_caps():
    owner,bridge=_patch(activation_policy='transverse_events',gap_check='straight_patch')
    initial=_origin(owner);pose=initial;history=bridge.pair.initial_history(initial.shell_displacement)
    segments=(_segment(.004,-.004),_segment(.005,0.),_segment(0.,.006),_segment(-.003,-.006))
    for segment in segments:
        value=bridge.pair.evaluate_path(pose,history,segment.translations,segment.angular,segment.shell_increment)
        pose=value.configuration;history=value.contribution.trial.candidate_state
    field=NativeSurfaceHistoryField(bridge.pair,initial,segments)
    budget=NativeSurfaceReplayBudget((1+len(segments))*len(bridge.pair.pairs))
    for point,expected in zip(bridge.pair.pairs,history.points):
        query=field.query(point.source.xi,point.source.angle,budget=budget)
        assert asdict(query.history)==asdict(expected)
        assert sha(asdict(query.configuration))==sha(asdict(pose))
    assert any(h.cumulative_dissipation_density>0 for h in history.points)
    before=field.identity
    with pytest.raises(MemoryError,match='total replay'):
        field.query(.43,np.pi,budget=budget)
    assert field.identity==before
    with pytest.raises(MemoryError,match='segment cap'):
        NativeSurfaceHistoryField(bridge.pair,initial,segments,maximum_segments=2)


def test_history_snapshot_from_coupled_transaction_and_restored_path():
    owner,bridge=_patch(activation_policy='transverse_events',gap_check='straight_patch',
        normal_integration='straight_segmented',normal_integration_settings=_settings())
    transaction=NativeS3MaterialTransaction(bridge)
    owner.solve_material_increment(transaction,1.)
    before=transaction.checkpoint()
    field=NativeSurfaceHistoryField.from_transaction(transaction)
    assert field.accepted_origin_sha256==transaction.origin_sha256
    assert field.maximum_intervals==owner.maximum_swept_intervals
    with pytest.raises(ValueError,match='bound history cap'):
        field.query(.79,np.pi+.013,budget=NativeSurfaceReplayBudget(2),
            maximum_intervals=owner.maximum_swept_intervals+1)
    budget=NativeSurfaceReplayBudget(10)
    for pair,expected in zip(bridge.pair.pairs,transaction.accepted.friction_history.points):
        assert asdict(field.query(pair.source.xi,pair.source.angle,budget=budget).history)==asdict(expected)
    new=field.query(.79,np.pi+.013,budget=budget)
    assert new.history.partner_key is not None
    assert transaction.checkpoint()==before
    restored=NativeS3MaterialTransaction.restore(bridge,before,sha256(before).hexdigest())
    replay=NativeSurfaceHistoryField.from_transaction(restored)
    assert replay.identity==field.identity
    assert asdict(replay.query(.79,np.pi+.013,budget=NativeSurfaceReplayBudget(2)).history)==asdict(new.history)


@pytest.mark.parametrize('reason',[
    'native/S3 material history path requires subdivision: path resolution exhausted',
    'unexpected constitutive defect'])
def test_ordinary_resolution_request_uses_certified_replay_but_other_errors_remain_fatal(monkeypatch,reason):
    from test_general_contact_native_surface_support import _field
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
    motion=NativeSurfaceMotionSegment(np.zeros((3,3)),
        np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=.025,rotation=-.8,segments=(motion,))
    expected=field.query(-.5,1.5*np.pi,budget=NativeSurfaceReplayBudget(2))
    before=field.identity
    def refused(*args,**kwargs):raise ValueError(reason)
    monkeypatch.setattr(NativeS3MaterialPointFriction,'evaluate_path',refused)
    budget=NativeSurfaceReplayBudget(2)
    if reason.startswith('native/S3'):
        actual=field.query(-.5,1.5*np.pi,budget=budget)
        assert asdict(actual.history)==asdict(expected.history)
        np.testing.assert_array_equal(actual.last_contribution.residual,expected.last_contribution.residual)
        assert actual.history.cumulative_dissipation_density>0.
        assert budget.certified_intervals>0
    else:
        with pytest.raises(ValueError,match='unexpected constitutive defect'):
            field.query(-.5,1.5*np.pi,budget=budget)
    assert field.identity==before and budget.point_paths==2


def test_prefix_state_reuse_keeps_fresh_endpoint_law_budget_and_binding(monkeypatch):
    from collections import OrderedDict
    owner,bridge=_patch(activation_policy='transverse_events',gap_check='straight_patch')
    field=NativeSurfaceHistoryField(bridge.pair,_origin(owner),(_segment(.004,-.004),_segment(.0001,0.)))
    field._query_prefix_cache=OrderedDict();field._query_prefix_capacity=2
    original=field._replay_increment;calls=[]
    def observed(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(field,'_replay_increment',observed)
    first_budget=NativeSurfaceReplayBudget(3)
    first=field.query(.43,np.pi+.013,budget=first_budget)
    second_budget=NativeSurfaceReplayBudget(3)
    second=field.query(.43,np.pi+.013,budget=second_budget)
    assert len(calls)==3  # two prefix/suffix evaluations, then fresh suffix
    assert first_budget.point_paths==second_budget.point_paths==3
    assert first_budget.certified_intervals==second_budget.certified_intervals
    assert asdict(first.history)==asdict(second.history)
    np.testing.assert_array_equal(first.last_contribution.residual,second.last_contribution.residual)
    np.testing.assert_array_equal(first.last_contribution.spatial_tangent(),second.last_contribution.spatial_tangent())
    for x in (.44,.45):field.query(x,np.pi+.013,budget=NativeSurfaceReplayBudget(3))
    assert len(field._query_prefix_cache)==2
    with pytest.raises(MemoryError,match='total replay'):
        field.query(.45,np.pi+.013,budget=NativeSurfaceReplayBudget(2))
    field.segments=(_segment(.003,-.004),field.segments[-1])
    with pytest.raises(ValueError,match='definition changed'):
        field.query(.45,np.pi+.013,budget=NativeSurfaceReplayBudget(3))


def test_complete_schedule_replay_keeps_constitutive_refusal(monkeypatch):
    from test_general_contact_native_surface_support import _field
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
    import anyfesolver.general_contact.native_surface_activation as activation
    motion=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=.025,rotation=-.8,segments=(motion,))
    def ordinary(*args,**kwargs):raise AssertionError('ordinary certificate should not be duplicated')
    monkeypatch.setattr(NativeS3MaterialPointFriction,'evaluate_path',ordinary)
    pair=field.pair_at(-.5,1.5*np.pi)
    history=initial_friction_point(pair.target.physical_pose(field.initial.shell_displacement).material_frame)
    def replay():return field._sequence_replay(pair,field.initial,history,motion.translations,motion.angular,motion.shell_increment,256)
    value=replay()
    assert value.contribution.trial.candidate_state.cumulative_dissipation_density>0.
    def constitutive(*args,**kwargs):raise ValueError('unexpected constitutive defect')
    monkeypatch.setattr(activation,'trial_coulomb_point',constitutive)
    with pytest.raises(ValueError,match='unexpected constitutive defect'):
        replay()


def test_bound_whole_patch_ownership_reuses_only_the_same_point_path():
    from anyfesolver.general_contact.native_surface_support import history_target_ownership_certified,SupportIntervalBudget
    from anyfesolver.general_contact.native_surface_path import certify_material_ownership
    owner,bridge=_patch(activation_policy='transverse_events',gap_check='straight_patch')
    motion=_segment(.004,-.004)
    field=NativeSurfaceHistoryField(bridge.pair,_origin(owner),(motion,))
    original=field.query(.43,np.pi+.013,budget=NativeSurfaceReplayBudget(2))
    assert history_target_ownership_certified(field,budget=SupportIntervalBudget(128))
    field._bind_whole_history_target_ownership()
    prepared=field.query(.43,np.pi+.013,budget=NativeSurfaceReplayBudget(2))
    assert asdict(original.history)==asdict(prepared.history)
    np.testing.assert_array_equal(original.last_contribution.residual,prepared.last_contribution.residual)
    np.testing.assert_array_equal(original.last_contribution.spatial_tangent(),prepared.last_contribution.spatial_tangent())
    proof=certify_material_ownership(prepared.pair,field.initial,motion.translations,motion.angular,motion.shell_increment)
    assert proof.certified and proof.reason=='bound whole-history target ownership'
    changed=certify_material_ownership(prepared.pair,field.initial,motion.translations*.99,motion.angular,motion.shell_increment)
    assert changed.certified and changed.reason!='bound whole-history target ownership'


def test_whole_history_proof_cannot_issue_for_a_foreign_replay_motion():
    from anyfesolver.general_contact.native_surface_support import history_target_ownership_certified,SupportIntervalBudget
    owner,bridge=_patch(activation_policy='transverse_events',gap_check='straight_patch')
    field=NativeSurfaceHistoryField(bridge.pair,_origin(owner),(_segment(.004,-.004),))
    assert history_target_ownership_certified(field,budget=SupportIntervalBudget(128))
    field._bind_whole_history_target_ownership()
    pair=field.pair_at(.43,np.pi+.013)
    history=initial_friction_point(pair.target.physical_pose(field.initial.shell_displacement).material_frame)
    foreign=_segment(.005,-.004)
    field._replay_increment(pair,field.initial,history,foreign.translations,foreign.angular,
        foreign.shell_increment,NativeSurfaceReplayBudget(1),256)
    assert pair._prepared_history_ownership is None
