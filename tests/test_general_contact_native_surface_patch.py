"""Fixed swept-surface contribution through the shared native/S3 owner."""
from hashlib import sha256
import numpy as np
import pytest

from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
from anyfesolver.general_contact.native_surface_patch import NativeS3MaterialSurfacePatch
from anyfesolver.general_contact.native_surface_contributor import NativeS3MaterialContributor
from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
from test_general_contact_native_retained_s3_static import _analysis


def _patch(*,activation_policy='transverse_first_touch',axial_order=2,angular_order=2,
           gap_check='sampled',normal_stiffness=1e4,radius=.049,
           normal_integration='point',normal_integration_settings=None,friction_coefficient=.3,
           friction_integration='point',friction_integration_settings=None,maximum_replay_segments=256):
    owner=_analysis(targets=(1.,1.05))
    reference=owner.context.elements[0][1].operator.reference
    rule=NativeRoundSurfaceRule(1,reference,radius,xi_range=(.2,.8),
        angle_range=(np.pi-.34,np.pi+.34),axial_order=axial_order,angular_order=angular_order)
    corners=owner.contact.shell.reference_positions
    edges=np.column_stack((corners[1]-corners[0],corners[2]-corners[0]))
    targets=[]
    for sample in rule.samples:
        point=sample.station.evaluate(owner.initial.mechanical.positions,
            owner.initial.mechanical.position_low,owner.initial.mechanical.cell_rotations[0]).surface_position
        uv=np.linalg.lstsq(edges,point-corners[0],rcond=None)[0]
        targets.append(S3SkinMaterialPoint(owner.contact.shell,tuple((1-uv.sum(),*uv))))
    pair=NativeS3MaterialSurfacePatch(rule,targets,normal_stiffness_Pa_per_m=normal_stiffness,
        friction_coefficient=friction_coefficient,tangential_stiffness_Pa_per_m=1000.,
        activation_policy=activation_policy,gap_check=gap_check,
        normal_integration=normal_integration,normal_integration_settings=normal_integration_settings,
        friction_integration=friction_integration,friction_integration_settings=friction_integration_settings,
        maximum_replay_segments=maximum_replay_segments)
    return owner,NativeS3MaterialContributor(owner,pair)


def test_distributed_patch_active_owner_atomic_history_and_restart():
    owner,bridge=_patch(); transaction=NativeS3MaterialTransaction(bridge)
    accepted=owner.solve_material_increment(transaction,1.)
    assert len(accepted.friction_history.points)==4
    assert any(h.partner_key is not None for h in accepted.friction_history.points)
    assert len({p.source.identity for p in bridge.pair.pairs})==4
    same=transaction.propose(np.zeros(42),np.zeros(18),1.)
    assert max(same.assembly.metrics)<owner.equilibrium_tolerance
    contribution=same.assembly.material.contribution
    np.testing.assert_allclose(contribution.residual,
        sum((p.residual for p in contribution.points),np.zeros(60)),atol=1e-14,rtol=0)
    assert np.linalg.norm(accepted.shell_displacement)>0
    capsule=transaction.checkpoint()
    restored=NativeS3MaterialTransaction.restore(bridge,capsule,sha256(capsule).hexdigest())
    for current in (transaction,restored):owner.solve_material_increment(current,1.05)
    assert transaction.checkpoint()==restored.checkpoint()


def test_patch_history_binding_and_last_point_definition_preserve_transaction():
    from dataclasses import replace
    owner,bridge=_patch(); transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint()
    origin=transaction.accepted
    foreign=replace(origin.friction_history,definition_sha256='0'*64)
    with pytest.raises(ValueError,match='history binding'):
        bridge.evaluate(origin,foreign,np.zeros(42),np.zeros(18),0.)
    assert transaction.checkpoint()==before
    # A mutation of the last point is caught before any history can commit.
    bridge.pair.pairs[-1].source.radius*=1.01
    with pytest.raises(ValueError,match='definition changed'):
        transaction.propose(np.zeros(42),np.zeros(18),0.)
    bridge.pair.pairs[-1].source.radius/=(1.01)
    assert transaction.checkpoint()==before
