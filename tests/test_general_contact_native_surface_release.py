"""Independent unloading-energy reference and real patch lifecycle."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
from test_general_contact_native_surface_s3_friction import _case
from test_general_contact_native_surface_patch import _patch


def test_transverse_release_retains_unloading_dissipation_with_refinement():
    _,original,pose,empty=_case()
    errors=[]
    for count in (16,64,256):
        pair=NativeS3MaterialPointFriction(original.source,original.target,reference_weight_m2=.01,
            normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.,
            activation_policy='transverse_events',event_history_substeps=count)
        history=replace(empty,partner_key=pair.partner_key,elastic_slip=(.003,0.,0.))
        result=pair.evaluate_path(pose,history,np.tile((0.,0.,.02),(3,1)),np.zeros((2,3)),np.zeros(18)).contribution
        # Normal pressure 10 -> 0 at half the path, held tangential displacement.
        # Coulomb unloading dissipates initial elastic energy .5*k*e^2.
        exact=.5*1000*.003**2
        assert result.event_fraction==pytest.approx(.5,abs=2e-14)
        assert result.trial.dissipated_increment_density==pytest.approx(exact*(1-1/count),rel=1e-10)
        assert result.trial.released_storage_density==pytest.approx(exact/count**2,rel=1e-9)
        errors.append(abs(result.trial.dissipated_increment_density/exact-1))
        assert result.trial.candidate_state.partner_key is None
        np.testing.assert_array_equal(result.residual,0.)
        np.testing.assert_array_equal(result.spatial_tangent(),0.)
        assert history.elastic_slip==(.003,0.,0.)
    assert errors[2]<errors[1]<errors[0] and errors[2]<.01


def test_rotating_patch_open_recontact_and_restart():
    owner,bridge=_patch(activation_policy='transverse_events')
    transaction=NativeS3MaterialTransaction(bridge)
    owner.solve_material_increment(transaction,1.)
    assert any(h.partner_key is not None for h in transaction.accepted.friction_history.points)
    for load in (.8,.6,.4,.2,0.):owner.solve_material_increment(transaction,load)
    assert all(h.partner_key is None for h in transaction.accepted.friction_history.points)
    capsule=transaction.checkpoint()
    restored=NativeS3MaterialTransaction.restore(bridge,capsule,sha256(capsule).hexdigest())
    for state in (transaction,restored):owner.solve_material_increment(state,1.05)
    assert transaction.checkpoint()==restored.checkpoint()
    assert any(h.partner_key is not None for h in transaction.accepted.friction_history.points)


def test_release_settings_bound_to_restart():
    _,bridge=_patch(activation_policy='transverse_events')
    transaction=NativeS3MaterialTransaction(bridge); before=transaction.checkpoint()
    bridge.pair.pairs[0].event_history_substeps=64
    with pytest.raises(ValueError,match='definition changed'):
        transaction.propose(np.zeros(42),np.zeros(18),0.)
    bridge.pair.pairs[0].event_history_substeps=32
    assert transaction.checkpoint()==before


def test_exact_release_endpoint_matches_beyond_endpoint_and_retains_small_work():
    from test_general_contact_native_surface_path import _touch
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    _,original,pose,empty=_case()
    pair=NativeS3MaterialPointFriction(original.source,original.target,reference_weight_m2=.01,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.,
        activation_policy='transverse_events',event_history_substeps=16)
    touch=_touch(pair,pose)
    loaded=material_path_pose(touch,np.tile((0.,0.,-.01),(3,1)),np.zeros((2,3)),np.zeros(18),1.)
    history=replace(empty,partner_key=pair.partner_key,elastic_slip=(.003,0.,0.),
                    cumulative_dissipation_density=1e14)
    results=[pair.evaluate_path(loaded,history,np.tile((0.,0.,amount),(3,1)),
        np.zeros((2,3)),np.zeros(18)).contribution for amount in (.01,.0101)]
    for result in results:
        assert result.trial.dissipated_increment_density==pytest.approx(.0045*15/16,rel=1e-10)
        assert result.trial.candidate_state.partner_key is None
    assert results[0].event_fraction==1.


def test_saved_closed_unloading_path_has_bounded_gap_certificate():
    import json
    from pathlib import Path
    from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration
    from anyfesolver.general_contact.native_surface_path import certify_material_path
    record=json.loads((Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002/release_rejected_path.json').read_text())['record']
    _,bridge=_patch(activation_policy='transverse_events')
    pair=bridge.pair.pairs[record['point']]
    from anyfesolver._ge_beam3_p5_seeded.core import sha
    assert sha(record['definition'])==pair.definition_sha256
    start=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in record['start'].items()})
    result=certify_material_path(pair,start,np.asarray(record['translations']),np.asarray(record['angular']),
        np.asarray(record['shell_increment']),closed=True,maximum_intervals=128)
    assert result.certified
    assert result.intervals<=3

