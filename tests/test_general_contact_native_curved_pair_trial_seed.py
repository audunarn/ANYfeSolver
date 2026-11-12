"""Disposable virgin-history trial initialization, without checkpoint import."""
from hashlib import sha256
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_p5_seeded.codec import _load
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken
from anyfesolver.general_contact.native_retained_pair_static import NativeRetainedRoundPairAnalysis
from test_general_contact_native_curved_fibre_pair_static import _curved_owner


def _seeded(template, data):
    return NativeRetainedRoundPairAnalysis(
        tuple(NativeBeamDefinition(raw) for raw in template.native._definitions),
        tuple(template.native.model.boundary_conditions),template.program,template.contact,
        gap_tolerance_m=template.gap_tolerance,equilibrium_tolerance=template.equilibrium_tolerance,
        maximum_swept_intervals=template.maximum_swept_intervals,
        gap_step_control=template.gap_step_control,initial_trial=data)


def _frozen_seed(template):
    raw=(Path(__file__).parent/'data/general_contact_native_curved_fibre_pair_v14.checkpoint').read_bytes()
    record=_load(raw.decode('ascii'))['records'][-1]
    return dict(native_model=template.context.identity,mechanical=record['mechanical'])


def test_seed_admission_is_disposable_model_bound_and_identity_bound():
    template=_curved_owner(8000.)
    definition=_frozen_seed(template)
    owner=_seeded(template,definition)
    origins=canonical(owner.initial.histories)
    raw=owner._checkpoint(())
    candidate,_=owner._initial_trial_candidate(owner.initial)
    assert owner._physical_gap(candidate)>=-owner.gap_tolerance
    assert canonical(owner.initial.histories)==origins
    assert owner.initial.completed_targets==0 and owner._checkpoint(())==raw
    assert owner.identity!=template.identity
    with pytest.raises(ValueError,match='binding'):
        template._restore(raw,sha256(raw).hexdigest())
    assert owner._restore(raw,sha256(raw).hexdigest())[0].completed_targets==0
    # Caller mutations cannot alter the copied immutable guess.
    definition['mechanical']['positions'][1][1]+=.001
    owner._guard()
    assert np.allclose(candidate.positions[1,1],.0038547320537078533)
    with pytest.raises(ValueError):
        owner._trial_state.cell_rotations.setflags(write=True)
    mutated=owner._trial_state.cell_rotations.copy()
    mutated[0,0,0,0]+=.001
    owner._trial_state=replace(owner._trial_state,cell_rotations=mutated)
    with pytest.raises(ValueError,match='changed'):
        owner._guard()


def test_seed_refuses_stale_malformed_unsupported_and_uncertified_path(monkeypatch):
    template=_curved_owner(8000.)
    seed=_frozen_seed(template)
    with pytest.raises(ValueError,match='binding'):
        _seeded(template,dict(seed,native_model='stale'))
    bad=dict(seed,mechanical=dict(seed['mechanical'],positions=np.full((6,3),np.nan)))
    with pytest.raises(ValueError):_seeded(template,bad)
    bad=dict(seed,mechanical=dict(seed['mechanical'],resultants=np.zeros((2,17))))
    with pytest.raises(ValueError,match='shape'):_seeded(template,bad)
    bad_positions=np.array(seed['mechanical']['positions']);bad_positions[0,0]+=.01
    with pytest.raises(ValueError,match='support'):
        _seeded(template,dict(seed,mechanical=dict(seed['mechanical'],positions=bad_positions)))
    owner=_seeded(template,seed)
    raw=owner._checkpoint(());origins=canonical(owner.initial.histories)
    monkeypatch.setattr(owner,'_swept_admissible',lambda *args:False)
    failed=owner.solve()
    assert failed.status=='failed' and 'trial path is not certified' in failed.failure
    assert failed.checkpoint==raw and canonical(failed.accepted.histories)==origins


def test_seed_cancellation_rolls_back_without_importing_history(monkeypatch):
    template=_curved_owner(8000.);owner=_seeded(template,_frozen_seed(template))
    raw=owner._checkpoint(());origins=canonical(owner.initial.histories)
    token=CancellationToken();token.cancel()
    failed=owner.solve(cancellation_token=token)
    assert failed.status=='failed' and 'SolveCancelled' in failed.failure
    assert failed.checkpoint==raw and canonical(failed.accepted.histories)==origins
    assert owner.context._token is None and not owner.context._lock.locked()
    token=CancellationToken();actual=owner._physical_gap
    def cancel_after(pose):
        value=actual(pose);token.cancel();return value
    monkeypatch.setattr(owner,'_physical_gap',cancel_after)
    failed=owner.solve(cancellation_token=token)
    assert failed.status=='failed' and 'SolveCancelled' in failed.failure
    assert failed.checkpoint==raw and canonical(failed.accepted.histories)==origins
