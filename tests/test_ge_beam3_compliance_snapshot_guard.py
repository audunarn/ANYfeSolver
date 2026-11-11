from time import perf_counter
import numpy as np
import pytest
from anyfesolver import _ge_beam3_compliance_snapshot_guard as policy
from anyfesolver._ge_beam3_native_generalized_factor_modal import compliance_factor
from anyfesolver.control import CancellationToken,SolveCancelled

@pytest.mark.parametrize('scale',(1e-8,1.,1e8))
def test_exact_factor_and_error_equality_with_fewer_full_checks(scale):
    a=np.diag(np.linspace(1.,3.,18));a+=.01*np.ones((18,18));a*=scale;before=a.tobytes()
    counts=dict(old=0,full=0,cheap=0)
    def old():counts['old']+=1
    def full():
        counts['full']+=1
        if a.tobytes()!=before:raise ValueError('source changed')
    def cheap():counts['cheap']+=1
    expected,error=compliance_factor(a,old)
    actual,other=policy.snapshot_compliance(a,full,cheap)
    assert actual.tobytes()==expected.tobytes() and other==error and a.tobytes()==before
    assert counts==dict(old=133,full=2,cheap=133)
    assert not actual.flags.writeable
def test_full_guard_before_arithmetic_and_after_mutation():
    a=np.eye(18);capture=a.tobytes();calls=[]
    def full():
        calls.append('full')
        if a.tobytes()!=capture:raise ValueError('mutated source')
    def change():a[0,0]=2.
    with pytest.raises(ValueError,match='mutated'):policy.snapshot_compliance(a,full,change)
    assert calls==['full','full']
    called=[]
    def reject():raise ValueError('invalid before')
    with pytest.raises(ValueError,match='invalid before'):policy.snapshot_compliance(np.eye(18),reject,lambda:called.append(1))
    assert called==[]
def test_inner_check_uses_immutable_copy_not_live_array():
    a=np.eye(18);full_count=[0]
    def full():full_count[0]+=1
    def mutate_original():a[0,0]=7.
    result,error=policy.snapshot_compliance(a,full,mutate_original)
    # In the real capture full_check rejects changed model/state inputs; this
    # fixture separately demonstrates the inner arithmetic cannot see mutation.
    np.testing.assert_array_equal(result,np.eye(18));assert error==0 and full_count==[2]
def test_cancellation_and_final_full_guard():
    token=CancellationToken();token.cancel();calls=[]
    with pytest.raises(SolveCancelled):
        policy.snapshot_compliance(np.eye(18),lambda:calls.append('full'),lambda:policy.deadline_checkpoint(perf_counter(),token))
    assert calls==['full','full']
def test_original_deadline_not_extended(monkeypatch):
    monkeypatch.setattr(policy,'monotonic',lambda:130.)
    policy.deadline_checkpoint(10.)
    with pytest.raises(RuntimeError,match='context deadline'):policy.deadline_checkpoint(9.999)
def test_unknown_policy_rejects_before_native_owner(monkeypatch):
    from anyfesolver import _ge_beam3_elastic_seed_modal as modal
    def forbidden(*a,**kw):raise AssertionError('owner must not run')
    monkeypatch.setattr(modal.control,'Context',forbidden)
    for invalid in ('unregistered',True,1):
        with pytest.raises(ValueError,match='guard policy'):
            modal.prepare(None,None,None,None,None,expected_seed_sha256='0'*64,expected_sha256='0'*64,compliance_guard_policy=invalid)
def test_nonpositive_compliance_still_rejected_and_guarded():
    full=[]
    with pytest.raises(np.linalg.LinAlgError):policy.snapshot_compliance(-np.eye(18),lambda:full.append(1),lambda:None)
    assert full==[1,1]
