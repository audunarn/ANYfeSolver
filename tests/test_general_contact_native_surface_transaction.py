"""Bound native material transactions; open equilibrium replay, not C5 admission."""
from dataclasses import replace
from hashlib import sha256
from math import fsum

import numpy as np
import pytest

from anyfesolver._ge_beam3_p5.algebra import log_rotation
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import _load
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
from test_general_contact_native_surface_contributor import _contributor


def _increment(owner, first, last):
    value=np.zeros(owner.context.count)
    for i in range(3):
        for j in range(3):
            value[6*i+j]=fsum((float(last.positions[i,j]),float(last.position_low[i,j]),
                              -float(first.positions[i,j]),-float(first.position_low[i,j])))
        value[6*i+3:6*i+6]=log_rotation(last.nodal_frames[i]@first.nodal_frames[i].T)
    for c in range(2):
        value[18+3*c:21+3*c]=log_rotation(last.cell_rotations[0,c]@first.cell_rotations[0,c].T)
    value[24:42]=last.resultants[0]-first.resultants[0]
    return value


def test_bound_open_equilibrium_commit_split_replay_and_stale_proposal():
    owner,bridge,_=_contributor(radius=.049,targets=(.001,.002))
    transaction=NativeS3MaterialTransaction(bridge)
    initial=transaction.checkpoint()
    # Reuse the actual existing native/S3 nonlinear solve at two small loads.
    # Both declared contact geometries remain open; this is no friction claim.
    first=owner.solve(stop_after=1)
    assert first.status=='paused',first.failure
    step=_increment(owner,transaction.accepted.mechanical,first.accepted.mechanical)
    proposal=transaction.propose(step,first.accepted.shell_displacement,.001)
    assert transaction.checkpoint()==initial
    assert max(proposal.assembly.metrics)<owner.equilibrium_tolerance
    transaction.commit(proposal)
    checkpoint=transaction.checkpoint()
    with pytest.raises(ValueError,match='stale/foreign'):
        transaction.commit(proposal)
    restored=NativeS3MaterialTransaction.restore(bridge,checkpoint,sha256(checkpoint).hexdigest())
    assert restored.origin_sha256==transaction.origin_sha256
    second=owner.solve(checkpoint=first.checkpoint,expected_sha256=first.checkpoint_sha256)
    assert second.status=='completed',second.failure
    step=_increment(owner,transaction.accepted.mechanical,second.accepted.mechanical)
    delta=second.accepted.shell_displacement-transaction.accepted.shell_displacement
    for current in (transaction,restored):
        current.commit(current.propose(step,delta,.002))
    assert transaction.checkpoint()==restored.checkpoint()
    assert transaction.accepted.sequence==2
    assert transaction.accepted.friction_history.partner_key is None
    assert np.linalg.norm(transaction.accepted.mechanical.positions-owner.initial.mechanical.positions)>0.


def test_failed_equilibrium_cancellation_and_definition_change_do_not_commit():
    owner,bridge,_=_contributor(radius=.049)
    transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint()
    proposal=transaction.propose(np.zeros(42),np.zeros(18),1.)
    with pytest.raises(ValueError,match='requires coupled equilibrium'):
        transaction.commit(proposal)
    assert transaction.checkpoint()==before
    token=CancellationToken();token.cancel('test')
    with pytest.raises(SolveCancelled):transaction.commit(proposal,cancellation_token=token)
    assert transaction.checkpoint()==before
    bridge.pair.source.radius*=1.01
    with pytest.raises(ValueError,match='definition changed'):transaction.checkpoint()


def test_checkpoint_history_tampering_and_foreign_settings_are_rejected():
    _,bridge,_=_contributor(radius=.049)
    transaction=NativeS3MaterialTransaction(bridge)
    transaction.commit(transaction.propose(np.zeros(42),np.zeros(18),0.))
    checkpoint=transaction.checkpoint()
    data=_load(checkpoint.decode('ascii'))
    data['records'][0]['state']['friction_history']['cumulative_dissipation_density']=1.
    data['checkpoint_sha256']=sha({k:v for k,v in data.items() if k!='checkpoint_sha256'})
    changed=canonical(data)
    with pytest.raises(ValueError,match='physical replay mismatch'):
        NativeS3MaterialTransaction.restore(bridge,changed,sha256(changed).hexdigest())
    with pytest.raises(ValueError,match='definition/settings'):
        NativeS3MaterialTransaction.restore(bridge,checkpoint,sha256(checkpoint).hexdigest(),maximum_records=255)
    assert transaction.checkpoint()==checkpoint


def test_initial_penetration_and_record_cap_fail_before_physical_commit():
    _,preloaded,_=_contributor()
    with pytest.raises(ValueError,match='initial native material-surface penetration'):
        NativeS3MaterialTransaction(preloaded)
    _,bridge,_=_contributor(radius=.049)
    transaction=NativeS3MaterialTransaction(bridge,maximum_records=1)
    transaction.commit(transaction.propose(np.zeros(42),np.zeros(18),0.))
    before=transaction.checkpoint()
    with pytest.raises(MemoryError,match='record cap'):
        transaction.commit(transaction.propose(np.zeros(42),np.zeros(18),0.))
    assert transaction.checkpoint()==before


def test_existing_owner_material_driver_and_restored_driver_match_open_reference():
    owner,bridge,_=_contributor(radius=.049,targets=(.001,.002))
    transaction=NativeS3MaterialTransaction(bridge)
    first=owner.solve_material_increment(transaction,.001)
    assert first.sequence==1
    checkpoint=transaction.checkpoint()
    restored=NativeS3MaterialTransaction.restore(bridge,checkpoint,sha256(checkpoint).hexdigest())
    full=owner.solve_material_increment(transaction,.002)
    split=owner.solve_material_increment(restored,.002)
    assert transaction.checkpoint()==restored.checkpoint()
    reference=owner.solve()
    assert reference.status=='completed',reference.failure
    np.testing.assert_allclose(full.mechanical.positions,reference.accepted.mechanical.positions,rtol=0.,atol=1e-9)
    np.testing.assert_allclose(full.mechanical.resultants,reference.accepted.mechanical.resultants,rtol=0.,atol=1e-9)
    assert split.friction_history.partner_key is None
    assert np.linalg.norm(full.mechanical.positions-owner.initial.mechanical.positions)>1e-6


def test_existing_owner_material_driver_cancellation_preserves_transaction():
    owner,bridge,_=_contributor(radius=.049)
    transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint()
    token=CancellationToken();token.cancel('before solve')
    with pytest.raises(SolveCancelled):owner.solve_material_increment(transaction,.001,cancellation_token=token)
    assert transaction.checkpoint()==before
    # Histories remain immutable; writable copies of public snapshot arrays do
    # not alias the issued internal state.
    snapshot=transaction.accepted
    with pytest.raises(AttributeError):snapshot.native_histories[0].cell_identity='wrong'
    snapshot.shell_displacement.setflags(write=True)
    snapshot.shell_displacement[0]=1.
    assert transaction.checkpoint()==before


def test_empty_checkpoint_binds_the_initial_origin():
    _,bridge,_=_contributor(radius=.049)
    transaction=NativeS3MaterialTransaction(bridge)
    data=_load(transaction.checkpoint().decode('ascii'))
    data['initial_origin_sha256']='0'*64
    data['checkpoint_sha256']=sha({k:v for k,v in data.items() if k!='checkpoint_sha256'})
    changed=canonical(data)
    with pytest.raises(ValueError,match='definition/settings'):
        NativeS3MaterialTransaction.restore(bridge,changed,sha256(changed).hexdigest())


def test_unresolved_rotating_activation_driver_does_not_commit_a_partial_increment():
    owner,bridge,_=_contributor(radius=.049)
    transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint()
    with pytest.raises(RuntimeError,match='(backtracking/event|Newton iteration) limit'):
        owner.solve_material_increment(transaction,1.)
    assert transaction.checkpoint()==before
    assert transaction.accepted.sequence==0


def test_cancellation_at_atomic_commit_discards_staged_native_and_friction_state():
    class AtCommit(CancellationToken):
        def raise_if_cancelled(self,stage=''):
            if stage=='native material atomic commit':self.cancel('after validation')
            super().raise_if_cancelled(stage)
    _,bridge,_=_contributor(radius=.049)
    transaction=NativeS3MaterialTransaction(bridge)
    before=transaction.checkpoint()
    proposal=transaction.propose(np.zeros(42),np.zeros(18),0.)
    with pytest.raises(SolveCancelled,match='after validation'):
        transaction.commit(proposal,cancellation_token=AtCommit())
    assert transaction.checkpoint()==before
    transaction.commit(proposal)
    assert transaction.accepted.sequence==1


def test_transverse_first_touch_owner_equilibrium_and_active_restart():
    owner,bridge,_=_contributor(radius=.049,targets=(1.,1.05),activation_policy='transverse_first_touch')
    transaction=NativeS3MaterialTransaction(bridge)
    accepted=owner.solve_material_increment(transaction,1.)
    assert accepted.friction_history.partner_key==bridge.pair.partner_key
    assert np.linalg.norm(accepted.friction_history.elastic_slip)>0.
    assert accepted.sequence==1
    checkpoint=transaction.checkpoint()
    restored=NativeS3MaterialTransaction.restore(bridge,checkpoint,sha256(checkpoint).hexdigest())
    assert restored.checkpoint()==checkpoint
    assert restored.accepted.friction_history==accepted.friction_history
    # Retained internal rotation and deformable shell respond in the same solve.
    assert np.linalg.norm(accepted.mechanical.cell_rotations-owner.initial.mechanical.cell_rotations)>1e-6
    assert np.linalg.norm(accepted.shell_displacement)>1e-10
    # Committing the return map must not change equilibrium when the accepted
    # pose becomes the next history origin at the same physical load.
    same=transaction.propose(np.zeros(42),np.zeros(18),1.)
    assert max(same.assembly.metrics)<owner.equilibrium_tolerance
    for current in (transaction,restored):owner.solve_material_increment(current,1.05)
    assert transaction.checkpoint()==restored.checkpoint()
    assert transaction.accepted.sequence==2


def test_activation_selection_is_bound_into_transaction_restart():
    _,enabled,_=_contributor(radius=.049,activation_policy='transverse_first_touch')
    checkpoint=NativeS3MaterialTransaction(enabled).checkpoint()
    _,disabled,_=_contributor(radius=.049)
    with pytest.raises(ValueError,match='definition/settings'):
        NativeS3MaterialTransaction.restore(disabled,checkpoint,sha256(checkpoint).hexdigest())
