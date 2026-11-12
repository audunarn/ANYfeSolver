"""Focused connected-body development controls; independent straight line law."""
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json, os, time
import numpy as np
import pytest
from examples.general_contact_native_retained_body_pair import analysis
from anyfesolver.general_contact.native_retained_body_pair import omitted_cell_dominated
from anyfesolver.general_contact.native_replay_diagnostics import replay_difference_message


def test_complete_line_potential_work_crossblocks_and_refusals():
    owner=analysis();ctx=owner.context;initial=ctx.initial.mechanical
    positions=initial.positions.copy();positions[5:,1]=.09998
    pose=replace(initial,positions=positions)
    result=owner.contact.evaluate(pose);g=-.00002;k=4000.;length=2.
    assert ctx.count==156
    assert result.native_residual_component_error>=result.force_error_n
    assert result.native_tangent_component_error>=result.tangent_error_n_per_m
    assert result.potential_j==pytest.approx(.5*k*length*g*g,rel=1e-8)
    assert result.beam_a_force_n[1]==pytest.approx(k*length*g,rel=1e-8)
    assert result.beam_b_force_n[1]==pytest.approx(-k*length*g,rel=1e-8)
    assert result.beam_a_moment_nm[2]==pytest.approx(k*length*g,rel=1e-8)
    assert np.linalg.norm(result.tangent[:30,30:60])>0
    direction=np.zeros(ctx.count);direction[6*2+1]=1.
    # Shared middle junction's hat function integrates to .5 m; scalar line
    # distance derivative is -1 on body A: directional dE = -k*g*.5.
    assert result.residual@direction==pytest.approx(-k*g*.5,rel=1e-8)
    assert result.conservative_lower_gap_m<=g+1e-13
    with pytest.raises(ValueError):
        owner.contact.bodies=((11,),(12,13,14));owner._guard()


def test_endpoint_plateau_refusal_and_internal_motion():
    source=np.array(((0.,-.2,0.),(0.,0.,0.),(0.,.2,0.)))
    target=np.array(((0.,0.,0.),(.5,0.,0.),(1.,0.,0.)))
    assert not omitted_cell_dominated(source,target,(target[0],target[-1]),0.,1.,1e-12,(True,False))
    source=np.array(((-1.,.1,0.),(-.75,.1,0.),(-.5,.1,0.)))
    assert omitted_cell_dominated(source,target,(target[0],target[-1]),0.,1.,1e-12,(True,False))
    moved=target.copy();moved[1]=(-1.,.1,0.)
    assert not omitted_cell_dominated(source,moved,(target[0],target[-1]),0.,1.,1e-12,(True,False))


def test_default_saved_curved_energy_and_identity():
    from test_general_contact_native_curved_fibre_pair_static import _curved_owner, _saved_curved_fixture
    from anyfesolver._ge_beam3_p5_seeded.codec import _load
    owner=_curved_owner()
    raw,_=_saved_curved_fixture()
    capsule=_load(raw.decode('ascii'))
    assert owner.identity==capsule['identity']
    assert capsule['completed_targets']==1 and len(capsule['records'])==1
    record=capsule['records'][-1]
    # Every saved field, including all three error bounds, must replay exactly.
    # A different local record cannot replace the authenticated saved fixture.
    mechanical=owner.context.make(record['mechanical'],decoded=True)
    _,replayed=owner._record(record['target'],mechanical,owner.initial.histories,
        record['previous_sha256'])
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    assert canonical(record)==canonical(replayed), replay_difference_message(record, replayed, 'full record')
    state,records=owner._restore(raw,sha256(raw).hexdigest())
    assert state.contact.potential_j>0
    assert owner._checkpoint(records)==raw
    # Direct-caller failure integration: the unchanged production restore
    # control refuses a sealed capsule whose recorded contact does not
    # replay, and the refusal carries the bounded first-difference context.
    tampered={**replayed,'contact':{**replayed['contact'],
        'potential_j':replayed['contact']['potential_j']+1.}}
    tampered_raw=owner._checkpoint((tampered,))
    with pytest.raises(ValueError,match='physical replay mismatch') as failure:
        owner._restore(tampered_raw,sha256(tampered_raw).hexdigest())
    assert 'path=contact.potential_j' in str(failure.value)
    assert replay_difference_message(tampered,replayed,'record 1') in str(failure.value)


def test_saved_curved_full_record_replay_is_exact():
    """CI4 requires every saved field and the unchanged production restore."""
    from test_general_contact_native_curved_fibre_pair_static import _curved_owner, _saved_curved_fixture
    from anyfesolver._ge_beam3_p5_seeded.codec import _load
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    owner = _curved_owner()
    raw, _ = _saved_curved_fixture()
    capsule = _load(raw.decode('ascii'))
    assert owner.identity == capsule['identity']
    record = capsule['records'][0]
    mechanical = owner.context.make(record['mechanical'], decoded=True)
    _, replayed = owner._record(record['target'], mechanical, owner.initial.histories,
                               record['previous_sha256'])
    assert canonical(record) == canonical(replayed), replay_difference_message(record, replayed, 'full record')
    _, records = owner._restore(raw, sha256(raw).hexdigest())
    assert owner._checkpoint(records) == raw


def test_active_connected_static_restart_and_rollback():
    if os.environ.get('CONTACT_NATIVE_BODY_SOLVE')!='1':pytest.skip('explicit bounded active solve')
    owner=analysis();initial=owner._checkpoint(())
    result=owner.solve()
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).write_text(json.dumps({'result':str(result)},indent=2))
    assert result.status=='completed', result.failure
    assert result.accepted.contact.potential_j>0
    restored,records=analysis()._restore(result.checkpoint,result.checkpoint_sha256)
    assert restored.completed_targets==1
    assert analysis()._checkpoint(records)==result.checkpoint
    assert owner._checkpoint(())==initial


def test_bound_chain_and_indices_mutations_failclosed():
    owner=analysis();pose=owner.initial.mechanical
    owner.contact.chains=(owner.contact.chains[0][:-1],owner.contact.chains[1])
    with pytest.raises(ValueError):owner._physical_gap(pose)
    owner=analysis();owner.indices=()
    with pytest.raises(ValueError):owner._physical_gap(owner.initial.mechanical)

    owner=analysis();owner.contact.cells[0].geometry.cell=1
    with pytest.raises(ValueError):owner._physical_gap(owner.initial.mechanical)
