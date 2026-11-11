"""Disposable exact-pose contact reuse; no material or committed state reuse."""
from pathlib import Path
from hashlib import sha256
from dataclasses import replace
import numpy as np
import pytest
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_general_contact_native_curved_s3_static import _analysis

def fixture():
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    raw=(Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_active_v1.checkpoint').read_bytes()
    state,_=owner._restore(raw,sha256(raw).hexdigest())
    return owner,state

def equal(a,b):
    np.testing.assert_array_equal(a.residual,b.residual)
    np.testing.assert_array_equal(a.tangent.toarray(),b.tangent.toarray())
    assert canonical(a.contact.to_dict())==canonical(b.contact.to_dict())

def test_exact_candidate_reuse_parity_invalidation_and_bounded_storage():
    owner,state=fixture();m=state.mechanical;u=state.shell_displacement
    reference=owner._contact(m,u)
    with owner.context.operation(None):
        owner._begin_candidate_context(None)
        try:
            before=owner.pair_evaluations
            first=owner._contact(m,u)
            second=owner._contact(m,u.copy())
            equal(first,reference);equal(second,reference)
            assert owner.pair_evaluations==before+1 and owner.contact_reuse_hits==1
            assert second is first
            with pytest.raises(ValueError):second.tangent.data[0]=0.
            # Each authoritative physical field invalidates; material/resultant
            # coordinates do not enter this geometry-only contributor cache.
            changed=u.copy();changed[owner.shell_free[0]]+=1e-10
            owner._contact(m,changed)
            high=m.positions.copy();high[2,2]+=1e-10
            owner._contact(replace(m,positions=high),u)
            low=m.position_low.copy();low[2,2]=0.
            owner._contact(replace(m,position_low=low),u)
            U=m.cell_rotations.copy()
            theta=1e-10;R=np.array(((np.cos(theta),0.,np.sin(theta)),(0.,1.,0.),(-np.sin(theta),0.,np.cos(theta))))
            U[0,1]=R@U[0,1]
            owner._contact(replace(m,cell_rotations=U),u)
            assert owner.pair_evaluations==before+5
            assert owner._contact_reuse_entry[1] is not first
        finally:owner._end_candidate_context()
    assert owner._contact_reuse_entry is None and not owner._contact_reuse_enabled
    before=owner.pair_evaluations
    owner._contact(m,u);owner._contact(m,u)
    assert owner.pair_evaluations==before+2

def test_reuse_rechecks_work_identity_and_failed_evaluation_clears_entry(monkeypatch):
    owner,state=fixture();m=state.mechanical;u=state.shell_displacement;work=owner.pairs[0].work
    with owner.context.operation(None):
        owner._begin_candidate_context(None)
        try:
            owner._contact(m,u)
            old=work.target;work.target=replace(old,adapter=replace(work.shell,body_id=99))
            with pytest.raises(ValueError,match='changed'):owner._contact(m,u)
            work.target=old
            def broken(*args):raise ValueError('owned integration failure')
            monkeypatch.setattr(work,'evaluate',broken)
            changed=u.copy();changed[owner.shell_free[0]]+=1e-10
            with pytest.raises(ValueError,match='owned integration failure'):owner._contact(m,changed)
            assert owner._contact_reuse_entry is None
        finally:owner._end_candidate_context()

def test_driver_cleans_cache_on_exception_and_empty_success(monkeypatch):
    owner,state=fixture()
    original=owner._assemble
    def broken(m,u,*args):
        owner._contact(m,u)
        raise RuntimeError('owned before physical commit')
    monkeypatch.setattr(owner,'_assemble',broken)
    run=owner.solve(stop_after=1)
    assert run.status=='failed' and run.accepted.completed_targets==0
    assert 'owned before physical commit' in run.failure
    assert not owner._contact_reuse_enabled and owner._contact_reuse_entry is None
    monkeypatch.setattr(owner,'_assemble',original)
    run=owner.solve(stop_after=0)
    assert run.status=='paused' and run.accepted.completed_targets==0
    assert not owner._contact_reuse_enabled and owner._contact_reuse_entry is None

def test_contact_reuse_never_reuses_constitutive_origin(monkeypatch):
    owner,state=fixture();origins=[];assemble=owner._assemble_structural
    def capture(m,u,parameter,origin):
        origins.append(origin)
        return assemble(m,u,parameter,origin)
    monkeypatch.setattr(owner,'_assemble_structural',capture)
    with owner.context.operation(None):
        owner._begin_candidate_context(None)
        try:
            first=owner._assemble(state.mechanical,state.shell_displacement,1.,owner.initial.native_histories)
            second=owner._assemble(state.mechanical,state.shell_displacement,1.,state.native_histories)
            assert origins[0] is owner.initial.native_histories and origins[1] is state.native_histories
            assert owner.contact_reuse_hits==1
            equal(first[-1],second[-1])
        finally:owner._end_candidate_context()


@pytest.mark.parametrize('which',('aggregate','pair'))
def test_reuse_rejects_sparse_buffer_rebinding(which):
    owner,state=fixture()
    with owner.context.operation(None):
        owner._begin_candidate_context(None)
        try:
            result=owner._contact(state.mechanical,state.shell_displacement)
            matrix=result.tangent if which=='aggregate' else result.pairs[0][1].tangent
            matrix.data=matrix.data.copy()
            matrix.data[0]+=1.
            with pytest.raises(ValueError,match='work mutated'):
                owner._contact(state.mechanical,state.shell_displacement)
            assert owner._contact_reuse_entry is None
        finally:owner._end_candidate_context()
