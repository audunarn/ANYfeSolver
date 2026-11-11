"""Multiple normal events on one actual retained-material motion segment."""
from dataclasses import replace
from math import acos,cos,sin,fsum,pi

import numpy as np
import pytest

from anyfesolver.general_contact.friction_point import initial_friction_point
from anyfesolver.general_contact.native_surface_event_sequence import locate_material_event_sequence,evaluate_open_event_sequence
from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
from test_general_contact_native_surface_support import _field


def pulse(height=.025):
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),np.zeros(18))
    field=_field(height=height,rotation=-.8,segments=(segment,))
    pair=field.pair_at(-.5,1.5*pi)
    history=initial_friction_point(pair.target.physical_pose(field.initial.shell_displacement).material_frame)
    return field,pair,history,segment


def test_analytical_two_event_roots_and_discrete_history_work():
    field,pair,history,segment=pulse()
    history=replace(history,cumulative_plastic_slip=.07,cumulative_dissipation_density=.1)
    args=(pair,field.initial,segment.translations,segment.angular,segment.shell_increment)
    sequence=locate_material_event_sequence(*args)
    theta=acos(.025/.03);expected=((.8-theta)/1.6,(.8+theta)/1.6)
    assert [e.fraction for e in sequence.events]==pytest.approx(expected,abs=2e-12)
    assert [e.opening for e in sequence.events]==[False,True]
    value=evaluate_open_event_sequence(pair,field.initial,history,*args[2:])
    # Independent scalar dry-friction return map, same declared32-step rule.
    elastic=0.;previous=.03*sin(-theta);dissipation=[];plastic=0.
    for index in range(1,pair.event_history_substeps):
        angle=-theta+2*theta*index/pair.event_history_substeps
        y=.03*sin(angle);trial=elastic+y-previous
        pressure=1000*(.03*cos(angle)-.025);cap=.3*pressure
        next_elastic=min(trial,cap/1000)
        increment=trial-next_elastic;dissipation.append(cap*increment);plastic+=increment
        elastic=next_elastic;previous=y
    assert value.trial.dissipated_increment_density==pytest.approx(fsum(dissipation),rel=1e-10)
    assert value.trial.released_storage_density==pytest.approx(.5*1000*elastic**2,rel=1e-10)
    assert value.trial.candidate_state.cumulative_dissipation_density==pytest.approx(.1+fsum(dissipation),rel=1e-12)
    assert value.trial.candidate_state.cumulative_plastic_slip==pytest.approx(.07+plastic,rel=1e-10)
    assert value.trial.candidate_state.partner_key is None
    assert np.array_equal(value.residual,np.zeros(60)) and np.array_equal(value.spatial_tangent(),np.zeros((60,60)))
    assert history.cumulative_dissipation_density==.1
    assert value.history_evaluations==pair.event_history_substeps+3


def test_sequence_counts_actual_coulomb_evaluations(monkeypatch):
    from anyfesolver.general_contact import native_surface_event_sequence as sequence_module
    from anyfesolver.general_contact import native_surface_activation as activation_module
    original=sequence_module.trial_coulomb_point;calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(sequence_module,'trial_coulomb_point',counted)
    monkeypatch.setattr(activation_module,'trial_coulomb_point',counted)
    field,pair,history,segment=pulse()
    value=evaluate_open_event_sequence(pair,field.initial,history,segment.translations,segment.angular,segment.shell_increment)
    assert value.history_evaluations==len(calls)==35


def test_explicit_sequence_policy_routes_full_saved_query_and_binds_identity():
    import json
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceReplayBudget
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from test_general_contact_native_surface_replay_owner import _replay_patch,_patch,normal_settings,friction_settings
    _,bridge=_patch(activation_policy='transverse_event_sequence',gap_check='straight_patch',
        normal_integration='straight_segmented',normal_integration_settings=normal_settings(),
        friction_integration='replay',friction_integration_settings=friction_settings())
    _,legacy=_replay_patch()
    assert bridge.pair.definition_sha256!=legacy.pair.definition_sha256
    folder=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    saved=json.loads((folder/'replay_owner_second_load_rejected_path.json').read_text())
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    value=field.query(.7244843513242318,3.1990185900472157,budget=NativeSurfaceReplayBudget(30))
    assert value.history.partner_key is None and value.history.cumulative_dissipation_density>0
    assert value.last_contribution.profile=='NATIVE_S3_OPEN_EVENT_SEQUENCE_POINT_DEV_V1'
    assert len(value.last_contribution.event_fractions)==2
    assert np.array_equal(value.last_contribution.residual,np.zeros(60))


def test_sequence_budget_and_tangency_remain_explicit_non_success():
    field,pair,history,segment=pulse()
    with pytest.raises((MemoryError,ValueError)):
        locate_material_event_sequence(pair,field.initial,segment.translations,segment.angular,segment.shell_increment,maximum_intervals=1)
    field,pair,history,segment=pulse(.03)
    with pytest.raises((MemoryError,ValueError)):
        locate_material_event_sequence(pair,field.initial,segment.translations,segment.angular,segment.shell_increment,maximum_intervals=32)


def test_two_chart_admitted_segments_accumulate_physical_history():
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    field,pair,history,segment=pulse()
    first=evaluate_open_event_sequence(pair,field.initial,history,segment.translations,segment.angular,segment.shell_increment)
    endpoint=material_path_pose(field.initial,segment.translations,segment.angular,segment.shell_increment,1.)
    both=evaluate_open_event_sequence(pair,endpoint,first.trial.candidate_state,
        segment.translations,-segment.angular,segment.shell_increment)
    assert len(both.event_fractions)==2
    assert both.history_evaluations==pair.event_history_substeps+3
    assert both.trial.dissipated_increment_density==pytest.approx(first.trial.dissipated_increment_density,rel=1e-9)
    assert both.trial.candidate_state.cumulative_dissipation_density==pytest.approx(2*first.trial.dissipated_increment_density,rel=1e-9)
    assert both.trial.released_storage_density==pytest.approx(first.trial.released_storage_density,rel=1e-9)
    assert both.trial.candidate_state.cumulative_plastic_slip==pytest.approx(2*first.trial.candidate_state.cumulative_plastic_slip,rel=1e-9)
    assert both.trial.candidate_state.partner_key is None


def test_saved_two_event_point_replays_closed_window(tmp_path):
    import json,os
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    from test_general_contact_native_surface_replay_owner import _replay_patch
    folder=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    saved=json.loads((folder/'replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch()
    field=NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])
    pair=field.pair_at(.7244843513242318,3.1990185900472157);pose=field.initial
    state=initial_friction_point(pair.target.physical_pose(pose.shell_displacement).material_frame)
    value=None
    for index,segment in enumerate(field.segments):
        if index==1:
            value=evaluate_open_event_sequence(pair,pose,state,segment.translations,segment.angular,segment.shell_increment,
                maximum_intervals=field.maximum_intervals)
            state=value.trial.candidate_state
        else:
            result=pair.evaluate_path(pose,state,segment.translations,segment.angular,segment.shell_increment,
                maximum_intervals=field.maximum_intervals)
            state=result.contribution.trial.candidate_state
        pose=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
    assert len(value.event_fractions)==2 and state.partner_key is None
    assert state.cumulative_dissipation_density>0
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'open_sequence_saved_point.json'
    output.write_text(json.dumps(dict(roots=value.event_fractions,intervals=value.ownership_intervals,
        history_evaluations=value.history_evaluations,dissipation=state.cumulative_dissipation_density,
        released_storage=value.trial.released_storage_density,gap=value.normal_gap_m),indent=2)+'\n')


def _saved_front_field():
    import json
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from test_general_contact_native_surface_replay_owner import _replay_patch
    saved=json.loads((Path(__file__).parents[1]
        /'docs/d1_review_support/contact_c5_native_surface_point_20261002'
        /'replay_owner_second_load_rejected_path.json').read_text())
    _,bridge=_replay_patch('transverse_event_sequence')
    return NativeSurfaceHistoryField(bridge.pair,
        NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in saved['initial'].items()}),
        tuple(NativeSurfaceMotionSegment(**{k:np.asarray(v) for k,v in s.items()}) for s in saved['segments']),
        maximum_intervals=saved['maximum_intervals'])


def _saved_front_second_segment_state(field,xi,angle):
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    pair=field.pair_at(xi,angle);pose=field.initial
    state=initial_friction_point(pair.target.physical_pose(pose.shell_displacement).material_frame)
    segment=field.segments[0]
    result=pair.evaluate_path(pose,state,segment.translations,segment.angular,segment.shell_increment,
        maximum_intervals=field.maximum_intervals)
    state=result.contribution.trial.candidate_state
    pose=material_path_pose(pose,segment.translations,segment.angular,segment.shell_increment,1.)
    return pair,pose,state,field.segments[1]


def test_saved_front_point_schedule_certifies_closed_end_below_binary64_resolution():
    field=_saved_front_field()
    pair,pose,state,segment=_saved_front_second_segment_state(field,.7999999999999998,3.323806549199672)
    sequence=locate_material_event_sequence(pair,pose,segment.translations,segment.angular,
        segment.shell_increment,maximum_intervals=field.maximum_intervals)
    # The literal binary64 endpoint evaluation rounds the final gap positive...
    assert sequence.endpoint_gap>0.
    # ...while the exact coefficient schedule certifies one closing root and a
    # strictly closed final endpoint (gap ~ -3.87e-18 m, certified).
    assert sequence.numerator_endpoint_signs==(1,-1)
    assert [e.opening for e in sequence.events]==[False]
    assert all(0.<e.fraction<1. for e in sequence.events)


def test_saved_front_point_certified_closed_endpoint_completes_with_corrected_primal():
    from anyfesolver.general_contact.native_event_polynomial import certified_material_gap
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceReplayBudget
    field=_saved_front_field()
    xi,angle=.7999999999999998,3.323806549199672
    pair,pose,state,segment=_saved_front_second_segment_state(field,xi,angle)
    certified=certified_material_gap(pair,pose,segment.translations,segment.angular,
        segment.shell_increment)
    assert certified<0.
    value=evaluate_open_event_sequence(pair,pose,state,segment.translations,segment.angular,
        segment.shell_increment,maximum_intervals=field.maximum_intervals)
    # The certified-closed sub-padding endpoint completes through the closed
    # endpoint law with only its Jet2 primal corrected; the former pressure
    # refusal and the misclassification refusal must not return.
    assert value.profile=='NATIVE_S3_CLOSED_EVENT_SEQUENCE_POINT_DEV_V1'
    assert value.normal_gap_m==certified
    assert value.normal_pressure_Pa==-pair.normal_stiffness_Pa_per_m*certified
    assert value.normal_pressure_Pa>0.
    assert value.normal_storage_j==pair.reference_weight_m2*(
        .5*pair.normal_stiffness_Pa_per_m*certified*certified)
    # Preserved chart: the corrected primal scales the same rank-one normal
    # chart the declared gradient/Hessian already spanned.
    nr=value.normal_residual
    expected=np.outer(nr,nr)/(pair.reference_weight_m2
        *pair.normal_stiffness_Pa_per_m*certified*certified)
    np.testing.assert_allclose(value.normal_spatial_tangent(),expected,rtol=1e-9,atol=1e-24)
    # Complete both-body wrench and finite maps.
    assert np.any(value.residual[:42]!=0.) and np.any(value.residual[42:]!=0.)
    assert np.all(np.isfinite(value.residual)) and np.all(np.isfinite(value.spatial_tangent()))
    # Physical history coherent: the closing endpoint activates the partner.
    assert value.trial.candidate_state.partner_key==pair.partner_key
    assert value.trial.dissipated_increment_density>=0.
    # The full saved query completes through the endpoint-law caller.
    query=field.query(xi,angle,budget=NativeSurfaceReplayBudget(30))
    assert query.last_contribution.profile=='NATIVE_S3_CLOSED_EVENT_SEQUENCE_POINT_DEV_V1'
    assert query.history.partner_key==pair.partner_key
    assert query.history.cumulative_dissipation_density>=0.


def test_certified_endpoint_primal_contract():
    from anyfesolver._ge_beam3_mixed_ad import Jet2
    from anyfesolver.general_contact.native_surface_activation import _certified_endpoint_primal
    padding=512*np.finfo(float).eps*150.
    g=Jet2.variable(5.06e-18,0,4)
    corrected=_certified_endpoint_primal(g,padding,-3.87e-18)
    assert corrected.value==-3.87e-18
    assert corrected.gradient is g.gradient and corrected.hessian is g.hessian
    # A binary64 evaluation beyond the padding resolved a clearly closed
    # endpoint on its own and keeps its own primal.
    clearly_closed=Jet2.variable(-1e-3,0,4)
    assert _certified_endpoint_primal(clearly_closed,padding,-1e-3+1e-17) is clearly_closed
    # Exact touch, out-of-profile, nonfinite, and above-padding contradictions
    # refuse.
    with pytest.raises(ValueError,match='exact touch'):
        _certified_endpoint_primal(g,padding,0.)
    with pytest.raises(ValueError,match='profile'):
        _certified_endpoint_primal(g,padding,1e-18)
    with pytest.raises(ValueError,match='finite'):
        _certified_endpoint_primal(g,padding,float('nan'))
    with pytest.raises(ValueError,match='contradicts endpoint work geometry'):
        _certified_endpoint_primal(g,padding,-1e-3)


def test_saved_front_point_certified_gap_refusals_preserve_rollback(monkeypatch):
    from anyfesolver.general_contact import native_event_polynomial as polynomial
    field=_saved_front_field()
    pair,pose,state,segment=_saved_front_second_segment_state(field,
        .7999999999999998,3.323806549199672)
    for bogus,message in ((0.,'exact touch'),(1e-18,'profile'),
            (-1e-3,'contradicts endpoint work geometry'),(float('nan'),'finite')):
        monkeypatch.setattr(polynomial,'certified_material_gap',lambda *args,**kwargs: bogus)
        with pytest.raises(ValueError,match=message):
            evaluate_open_event_sequence(pair,pose,state,segment.translations,segment.angular,
                segment.shell_increment,maximum_intervals=field.maximum_intervals)
        # Refusal rolls back: no physical state was committed.
        assert state.partner_key is None
    monkeypatch.undo()
    value=evaluate_open_event_sequence(pair,pose,state,segment.translations,segment.angular,
        segment.shell_increment,maximum_intervals=field.maximum_intervals)
    assert value.normal_pressure_Pa>0.


def test_certified_material_gap_requires_resolved_certified_sign(monkeypatch):
    from anyfesolver.general_contact import native_event_polynomial as polynomial
    monkeypatch.setattr(polynomial.MaterialTimeNumerator,'sign',lambda self,t:0)
    field,pair,history,segment=pulse()
    with pytest.raises(ValueError,match='resolved precise sign'):
        polynomial.certified_material_gap(pair,field.initial,segment.translations,
            segment.angular,segment.shell_increment)


def test_saved_front_neighbor_certified_open_endpoint_rounds_negative_and_completes():
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceReplayBudget
    field=_saved_front_field()
    pair,pose,state,segment=_saved_front_second_segment_state(field,.7999999999999994,3.3238065491996722)
    sequence=locate_material_event_sequence(pair,pose,segment.translations,segment.angular,
        segment.shell_increment,maximum_intervals=field.maximum_intervals)
    # Symmetric rounding flip: binary64 rounds the final gap negative while
    # the exact schedule certifies the endpoint strictly open with no events.
    assert sequence.endpoint_gap<0.
    assert sequence.numerator_endpoint_signs==(1,1)
    assert not sequence.events
    value=evaluate_open_event_sequence(pair,pose,state,segment.translations,segment.angular,
        segment.shell_increment,maximum_intervals=field.maximum_intervals)
    assert value.profile=='NATIVE_S3_OPEN_EVENT_SEQUENCE_POINT_DEV_V1'
    assert value.event_fractions==()
    assert value.trial.candidate_state.partner_key is None
    query=field.query(.7999999999999994,3.3238065491996722,budget=NativeSurfaceReplayBudget(30))
    assert query.last_contribution.profile=='NATIVE_S3_OPEN_EVENT_SEQUENCE_POINT_DEV_V1'
    assert query.history.partner_key is None


def test_certified_endpoint_state_resolution_contract():
    from anyfesolver.general_contact.native_surface_event_sequence import _resolve_endpoint_state
    padding=512*np.finfo(float).eps*150.
    assert _resolve_endpoint_state(1e-3,0,padding)==1
    assert _resolve_endpoint_state(-1e-3,0,padding)==-1
    assert _resolve_endpoint_state(1e-3,1,padding)==1
    # Certified signs resolve what binary64 cannot inside the padding.
    assert _resolve_endpoint_state(5.06e-18,-1,padding)==-1
    assert _resolve_endpoint_state(-1.07e-17,1,padding)==1
    # Exact touch and contradictions still refuse.
    with pytest.raises(ValueError,match='exact touch'):
        _resolve_endpoint_state(0.,1,padding)
    with pytest.raises(ValueError,match='unresolved'):
        _resolve_endpoint_state(5e-18,0,padding)
    with pytest.raises(ValueError,match='contradiction'):
        _resolve_endpoint_state(1e-3,-1,padding)
