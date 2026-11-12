"""Curved material S3 work integrated in the existing retained native owner."""
import json,os,runpy
from pathlib import Path
import numpy as np
import pytest
from anyfesolver.general_contact.native_multi_s3_static import CURVED_MATERIAL_PROFILE


def _analysis(**kwargs):
    return runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_native_curved_material_s3.py'))['analysis'](**kwargs)


def test_s3_congruence_product_fixed_fused_order_and_refusals():
    """Synthetic: the specified accumulation order, not a kernel equivalence."""
    from anyfesolver.general_contact.native_replay_arithmetic import s3_congruence_product
    eps = 2.**-53
    # With M = I the first product is exactly T.T, so entry (i,j) of the result is
    # sum_k T[k,i]*T[k,j] and only the accumulation order of the second product
    # is observable. Row 0 of T.T is (eps, 1, eps, 0, ...); columns 15..17 are ones.
    T = np.zeros((18, 18)); T[0, 0], T[1, 0], T[2, 0] = eps, 1., eps
    T[:, 15:] = 1.
    R = s3_congruence_product(T, np.eye(18))
    assert R[0, 15] == 1.                     # ascending single accumulation: ties to 1
    assert R[0, 16] == R[0, 17] == 1.+2*eps   # even (eps+eps) + odd (1) accumulators
    # Exactly representable data: every order gives the exact product.
    rng = np.random.default_rng(5)
    A = rng.integers(-8, 9, (18, 18)).astype(float); B = rng.integers(-8, 9, (18, 18)).astype(float)
    np.testing.assert_array_equal(s3_congruence_product(A, B), A.T @ B @ A)
    for bad in (np.zeros((17, 18)), np.full((18, 18), np.nan)):
        with pytest.raises(ValueError):
            s3_congruence_product(bad, np.eye(18))
        with pytest.raises(ValueError):
            s3_congruence_product(np.eye(18), bad)


def test_curved_s3_coupled_first_contact_complete_retained_and_target_work():
    owner=_analysis(targets=(1.,));run=owner.solve()
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-curved-s3-owner.json').write_text(json.dumps(dict(
        status=run.status,failure=run.failure,profile=run.profile,completed_targets=run.accepted.completed_targets,
        candidate_evaluations=owner.candidate_evaluations,pair_evaluations=owner.pair_evaluations,
        trial_failures=owner.curved_trial_failures,
        contact=json.loads(run.checkpoint)['records'][-1]['contact'] if run.accepted.completed_targets else None),indent=2))
    assert run.status=='completed',run.failure
    assert run.profile==CURVED_MATERIAL_PROFILE and run.accepted.completed_targets==1
    contact=run.accepted.contact.contact
    assert contact.minimum_gap_m>=-3e-5
    assert np.linalg.norm(run.accepted.shell_displacement)>0.
    assert contact.minimum_rim_clearance_m>1.
    np.testing.assert_allclose(contact.beam_force_n+contact.shell_force_n,0.,atol=1e-10)
    np.testing.assert_allclose(contact.beam_moment_nm+contact.shell_moment_nm,0.,atol=1e-10)


def test_curved_s3_active_checkpoint_for_bounded_three_target_continuation():
    owner=_analysis();run=owner.solve(stop_after=1)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-s3-active.checkpoint').write_bytes(run.checkpoint)
        Path(path+'-curved-s3-active.json').write_text(json.dumps(dict(status=run.status,failure=run.failure,
            completed_targets=run.accepted.completed_targets,checkpoint_sha256=run.checkpoint_sha256,
            profile=run.profile,contact=json.loads(run.checkpoint)['records'][-1]['contact'] if run.accepted.completed_targets else None,
            trial_failures=owner.curved_trial_failures,candidate_evaluations=owner.candidate_evaluations),indent=2))
    assert run.status=='paused',run.failure
    assert run.profile==CURVED_MATERIAL_PROFILE and run.accepted.completed_targets==1
    contact=run.accepted.contact.contact
    assert contact.minimum_gap_m>=-3e-5 and np.linalg.norm(run.accepted.shell_displacement)>0.
    np.testing.assert_allclose(contact.beam_force_n+contact.shell_force_n,0.,atol=1e-10)
    np.testing.assert_allclose(contact.beam_moment_nm+contact.shell_moment_nm,0.,atol=1e-10)


def test_curved_s3_driver_forwards_cancellation_and_preserves_initial_prefix():
    from anyfesolver.control import CancellationToken
    owner=_analysis(targets=(1.,));token=CancellationToken()
    initial=owner._checkpoint(())
    owner._assemble=lambda *args:(np.zeros(owner.context.count+len(owner.initial.shell_displacement)),
        np.zeros((1,1)),(1.,1.,1.),(),owner.initial.contact)
    owner._step=lambda *args:(np.zeros(owner.context.count+len(owner.initial.shell_displacement)),1.)
    real=owner._swept_cancellable;calls=[]
    def injected(*args,**kwargs):
        assert kwargs['cancellation_token'] is token
        calls.append(1);token.cancel();return real(*args,**kwargs)
    owner._swept_cancellable=injected
    run=owner.solve(cancellation_token=token)
    assert calls==[1] and run.status=='cancelled'
    assert run.accepted.completed_targets==0 and run.checkpoint==initial


def test_curved_s3_owner_uses_authoritative_path_increment_columns():
    from anyfesolver.control import CancellationToken,SolveCancelled
    owner=_analysis(targets=(1.,));start=owner.initial.mechanical;u=owner.initial.shell_displacement
    step=np.zeros(owner.context.count+len(u));step[12:15]=(.001,-.002,.001)
    step[owner.context.count:]=np.linspace(.001,-.002,len(u))
    end=owner.context.advance(start,step[:owner.context.count]);last=u+step[owner.context.count:]
    token=CancellationToken();work=owner.pairs[0].work;real=work.certify_path
    calls=[]
    def captured(pose,translations,angular,shell_increment,**kwargs):
        nodes=owner.context.nodes[0]
        np.testing.assert_array_equal(translations,np.array([step[6*i:6*i+3] for i in nodes]))
        np.testing.assert_array_equal(shell_increment,step[owner.context.count:])
        assert kwargs['cancellation_token'] is token
        calls.append(1);token.cancel();return real(pose,translations,angular,shell_increment,**kwargs)
    work.certify_path=captured
    with pytest.raises(SolveCancelled):owner._swept_admissible(start,u,step,end,last,cancellation_token=token)
    assert calls==[1]


def test_shared_swept_wrapper_preserves_historical_subclass_dispatch():
    from anyfesolver.general_contact.native_multi_s3_static import NativeRetainedMultiS3Analysis
    # Exact wrapper contract check; historical subclass need not accept a token.
    class Historical:
        _curved_mode=False
        def _swept_admissible(self,*args):
            assert args==(1,2,3);return 'historical override'
    assert NativeRetainedMultiS3Analysis._swept_cancellable(Historical(),1,2,3,
        cancellation_token=object())=='historical override'


def test_curved_s3_checkpoint_binding_field_classification_and_valid_restore():
    from hashlib import sha256
    from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
    from anyfesolver._ge_beam3_p5_seeded.codec import _load
    owner=_analysis(targets=(1.,));initial=owner._checkpoint(())
    # Matching valid checkpoint restore through the real caller: no target
    # trajectory is needed, the initial checkpoint already suffices.
    run=owner.solve(checkpoint=initial,expected_sha256=sha256(initial).hexdigest(),stop_after=0)
    assert run.status=='paused' and run.failure is None
    assert run.accepted.completed_targets==0 and run.checkpoint==initial
    value=_load(initial.decode('ascii'))
    def resealed(body,inner):
        raw=canonical({**body,'checkpoint_sha256':inner})
        return raw,sha256(raw).hexdigest()
    def tampered(**changes):
        # External checksum stays valid; the inner checksum is recomputed so
        # exactly one semantic binding field is isolated per case.
        body={k: v for k,v in value.items() if k!='checkpoint_sha256'}
        body.update(changes)
        return resealed(body,sha(body))
    cases={'schema':tampered(schema='NATIVE_B3_GE_MULTIPLE_S3_CHECKPOINT_DEV_V0'),
        'profile':tampered(profile='NATIVE_B3_GE_CURVED_MATERIAL_PATCHES_S3_STATIC_DEV_V0'),
        'identity':tampered(identity='0'*64),
        # Body stays valid; the recorded inner digest is stale by itself.
        'checkpoint_sha256':resealed({k: v for k,v in value.items() if k!='checkpoint_sha256'},'0'*64)}
    for field,(raw,external) in cases.items():
        with pytest.raises(ValueError,match='multiple native/S3 checkpoint definition/settings binding: '+field) as raised:
            owner.solve(checkpoint=raw,expected_sha256=external)
        message=str(raised.value)
        if field=='checkpoint_sha256':
            assert 'body=' in message and 'recorded=' in message
        else:
            assert 'checkpoint=' in message and 'owner=' in message
        # The refusal leaves the owner's initial physical state unchanged.
        assert owner._checkpoint(())==initial
    # A combined disagreement lists every offending binding field.
    raw,external=tampered(schema='NATIVE_B3_GE_MULTIPLE_S3_CHECKPOINT_DEV_V0',identity='0'*64)
    with pytest.raises(ValueError,match='schema checkpoint=.*; identity checkpoint=') as raised:
        owner.solve(checkpoint=raw,expected_sha256=external)
    message=str(raised.value)
    # Component digests may include the owner's profile. Only explicit
    # disagreement labels classify which binding fields were refused.
    assert 'profile checkpoint=' not in message and 'checkpoint_sha256 body=' not in message
    assert owner._checkpoint(())==initial
