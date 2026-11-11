"""Continuous normal integration: independent saved reference and work checks."""
from dataclasses import replace
from pathlib import Path
import json,zipfile

import numpy as np
import pytest

from anyfesolver.general_contact.native_surface_normal import StraightNormalIntegrationSettings,integrate_straight_normal
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialConfiguration,NativeS3MaterialPointFriction
from anyfesolver.general_contact.native_surface_activation import evaluate_first_touch
from test_general_contact_native_surface_patch import _patch
from test_general_contact_native_surface_s3_friction import _case,_moved
from test_general_contact_native_surface_activation import _approach


def _settings():
    return StraightNormalIntegrationSettings(1e-12,1e-10,1e-7,128,1.)


def _saved(stiffness):
    directory=Path(__file__).parents[1]/'docs/d1_review_support/contact_c5_native_surface_point_20261002'
    with zipfile.ZipFile(directory/'continuous_gap_raw.zip') as archive:
        name='contact-native-continuous-'+('gap' if stiffness==1e4 else 'enforcement')+'-20261002/results.json'
        data=json.loads(archive.read(name))
    pose=NativeS3MaterialConfiguration(**{k:np.asarray(v) for k,v in data['configuration'].items()})
    _,bridge=_patch(activation_policy='transverse_events',axial_order=8,angular_order=8,normal_stiffness=stiffness)
    references=json.loads((directory/'continuous_gap_results.json').read_text())['normal_force_discriminator']
    reference=next(r for r in references if r['stiffness']==stiffness)
    return bridge.pair,pose,reference


@pytest.mark.parametrize('stiffness',(1e4,1e5))
def test_segmented_normal_matches_independent_continuous_pressure(stiffness):
    pair,pose,reference=_saved(stiffness)
    result=integrate_straight_normal(pair.rule,pair.target,pose,stiffness,_settings())
    normal=np.asarray(pair.target.physical_pose(pose.shell_displacement).material_frame)[:,2]
    source=sum((result.residual[6*i:6*i+3] for i in range(3)),np.zeros(3))
    target=sum((result.residual[42+6*i:45+6*i] for i in range(3)),np.zeros(3))
    assert -source@normal==pytest.approx(reference['continuous_normal_force'],rel=1e-7,abs=1e-11)
    np.testing.assert_allclose(source+target,0.,rtol=0.,atol=1e-12)
    assert result.normal_storage_j>0


def test_segmented_normal_virtual_work_and_full_spatial_tangent():
    pair,pose,_=_saved(1e5)
    value=integrate_straight_normal(pair.rule,pair.target,pose,1e5,_settings())
    direction=np.random.default_rng(1567).normal(size=60)*.01
    # Both bodies and retained rotations vary; algebraic directions remain zero-work.
    for step in (1e-5,2e-6):
        plus=integrate_straight_normal(pair.rule,pair.target,_moved(pose,direction,step),1e5,_settings())
        minus=integrate_straight_normal(pair.rule,pair.target,_moved(pose,direction,-step),1e5,_settings())
        derivative=(plus.residual-minus.residual)/(2*step)
        assert np.linalg.norm(value.spatial_tangent()@direction-derivative)/np.linalg.norm(derivative)<1e-5
        work=(plus.normal_storage_j-minus.normal_storage_j)/(2*step)
        assert work==pytest.approx(value.residual@direction,rel=1e-5,abs=1e-10)
    np.testing.assert_array_equal(value.residual[24:42],0.)


def test_point_normal_extraction_preserves_persistent_and_activation_work():
    _,pair,pose,history=_case()
    direction=np.zeros(60);direction[18]=.05
    moved=_moved(pose,direction,1.)
    zero=NativeS3MaterialPointFriction(pair.source,pair.target,reference_weight_m2=.01,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=0.,tangential_stiffness_Pa_per_m=1000.)
    full=pair.evaluate(pose,moved,history);normal=zero.evaluate(pose,moved,history)
    np.testing.assert_allclose(full.normal_residual,normal.residual,rtol=0.,atol=1e-14)
    np.testing.assert_allclose(full.normal_spatial_tangent(),normal.spatial_tangent(),rtol=0.,atol=1e-13)

    pair,start,history,translation,angular,shell=_approach(.003)
    zero=NativeS3MaterialPointFriction(pair.source,pair.target,reference_weight_m2=.01,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=0.,tangential_stiffness_Pa_per_m=1000.)
    full=evaluate_first_touch(pair,start,history,translation,angular,shell)
    normal=evaluate_first_touch(zero,start,history,translation,angular,shell)
    np.testing.assert_allclose(full.normal_residual,normal.residual,rtol=0.,atol=1e-14)
    np.testing.assert_allclose(full.normal_spatial_tangent(),normal.spatial_tangent(),rtol=0.,atol=1e-13)


def test_segmented_profile_coupled_friction_and_restart():
    from hashlib import sha256
    from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
    owner,bridge=_patch(activation_policy='transverse_events',axial_order=8,angular_order=8,
        normal_stiffness=1e5,gap_check='straight_patch',normal_integration='straight_segmented',
        normal_integration_settings=_settings())
    transaction=NativeS3MaterialTransaction(bridge)
    accepted=owner.solve_material_increment(transaction,1.)
    assert any(h.partner_key is not None for h in accepted.friction_history.points)
    assert any(np.linalg.norm(h.elastic_slip)>0 for h in accepted.friction_history.points)
    same=transaction.propose(np.zeros(42),np.zeros(18),1.).assembly
    assert max(same.metrics)<owner.equilibrium_tolerance
    points=same.material.contribution.points
    normal=integrate_straight_normal(bridge.pair.rule,bridge.pair.target,same.material.configuration,1e5,_settings())
    expected=normal.residual+sum((p.residual-p.normal_residual for p in points),np.zeros(60))
    np.testing.assert_allclose(same.material.contribution.residual,expected,rtol=0.,atol=1e-13)
    assert same.material.contribution.normal_storage_j==pytest.approx(normal.normal_storage_j,rel=1e-12)
    capsule=transaction.checkpoint()
    restored=NativeS3MaterialTransaction.restore(bridge,capsule,sha256(capsule).hexdigest())
    for current in (transaction,restored):owner.solve_material_increment(current,1.05)
    assert transaction.checkpoint()==restored.checkpoint()


def test_segmented_settings_and_budget_do_not_fallback():
    from hashlib import sha256
    from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
    pair,pose,_=_saved(1e5)
    with pytest.raises(MemoryError,match='interval budget'):
        integrate_straight_normal(pair.rule,pair.target,pose,1e5,replace(_settings(),max_intervals=1))
    _,sampled=_patch(gap_check='straight_patch')
    capsule=NativeS3MaterialTransaction(sampled).checkpoint()
    _,segmented=_patch(gap_check='straight_patch',normal_integration='straight_segmented',normal_integration_settings=_settings())
    with pytest.raises(ValueError,match='definition/settings'):
        NativeS3MaterialTransaction.restore(segmented,capsule,sha256(capsule).hexdigest())


def test_segmented_frictionless_enforcement_screen(tmp_path):
    """Bounded development discriminator, not a qualification gap claim."""
    import os
    from dataclasses import asdict
    from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
    rows=[]
    output=Path(os.environ.get('RUNNER_TEMP',str(tmp_path)))/'enforcement_results.json'
    for stiffness in (1e5,1e6):
        owner,bridge=_patch(activation_policy='transverse_events',axial_order=8,angular_order=8,
            normal_stiffness=stiffness,friction_coefficient=0.,gap_check='straight_patch',
            normal_integration='straight_segmented',normal_integration_settings=_settings())
        transaction=NativeS3MaterialTransaction(bridge)
        owner.solve_material_increment(transaction,1.)
        same=transaction.propose(np.zeros(42),np.zeros(18),1.).assembly
        assert max(same.metrics)<owner.equilibrium_tolerance
        value=same.material.contribution
        pose=same.material.configuration
        normal=np.asarray(bridge.pair.target.physical_pose(pose.shell_displacement).material_frame)[:,2]
        force=-sum((value.residual[6*i:6*i+3] for i in range(3)),np.zeros(3))@normal
        rows.append(dict(stiffness=stiffness,normal_force=float(force),
            continuous_gap=float(value.normal_gap_m),metrics=list(map(float,same.metrics)),
            integration=asdict(value.normal_integration_diagnostics),
            configuration={k:v.tolist() for k,v in asdict(pose).items()}))
        output.write_text(json.dumps(rows,indent=2)+'\n')
    assert abs(rows[1]['continuous_gap'])<abs(rows[0]['continuous_gap'])
