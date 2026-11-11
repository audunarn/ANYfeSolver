"""Continuation from an actual equilibrated nonvirgin native contact prefix."""
from pathlib import Path
from hashlib import sha256
import json,os
import numpy as np
import pytest
from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_general_contact_native_curved_s3_static import _analysis
from test_general_contact_native_fibre_s3 import _independent_material_history


def _active():
    return (Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_active_v1.checkpoint').read_bytes()


def test_saved_plastic_contact_has_positive_reaction_complete_work_and_replay():
    raw=_active();owner=_analysis(physical_fibres=True,plastic_fibres=True)
    state,records=owner._restore(raw,sha256(raw).hexdigest())
    contact=state.contact.contact
    assert contact.potential_j>0. and np.linalg.norm(contact.beam_force_n)>0.
    np.testing.assert_allclose(contact.beam_force_n+contact.shell_force_n,0.,atol=1e-10)
    np.testing.assert_allclose(contact.beam_moment_nm+contact.shell_moment_nm,0.,atol=1e-10)
    report,proposed=_independent_material_history(owner,state.mechanical,owner.initial.native_histories)
    assert report['yielded']>0 and tuple(proposed)==state.native_histories[0].stations
    assert owner._checkpoint(records)==raw
    # A correctly rehashed but invented irreversible history must fail replay.
    fields=json.loads(raw)
    fields['records'][0]['histories'][0]['stations'][0]['rows'][0][2]+=.001
    fields['checkpoint_sha256']=sha({k:v for k,v in fields.items() if k!='checkpoint_sha256'})
    tampered=canonical(fields)
    with pytest.raises(ValueError,match='replay mismatch'):
        owner._restore(tampered,sha256(tampered).hexdigest())


@pytest.mark.parametrize('cancelled',(False,True))
def test_rejected_material_candidate_restores_nonvirgin_contact_prefix(monkeypatch,cancelled):
    raw=_active();owner=_analysis(physical_fibres=True,plastic_fibres=True)
    token=CancellationToken();begin=owner._begin_candidate_context
    probe=owner.context._backend.probes[0];evaluate=type(probe).evaluate
    def interrupted(self,*args,**kwargs):
        if cancelled:token.cancel()
        else:raise RuntimeError('nonvirgin fibre material candidate failure')
        return evaluate(self,*args,**kwargs)
    def after_replay(token):
        begin(token)
        # Committed prefix validation must precede the candidate fault.
        monkeypatch.setattr(type(probe),'evaluate',interrupted)
    monkeypatch.setattr(owner,'_begin_candidate_context',after_replay)
    run=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),
        stop_after=2,cancellation_token=token)
    assert run.status==('cancelled' if cancelled else 'failed')
    assert run.accepted.completed_targets==1 and run.checkpoint==raw
    assert canonical(run.accepted.native_histories)==canonical(json.loads(raw)['records'][0]['histories'])
    assert owner.context._token is None and not owner.context._lock.locked()


def test_plastic_contact_unloading_continues_from_committed_history():
    raw=_active();owner=_analysis(physical_fibres=True,plastic_fibres=True)
    origins=owner.context.histories(json.loads(raw)['records'][-1]['histories'],decoded=True)
    before=canonical(origins)
    run=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),stop_after=2)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-plastic-open.checkpoint').write_bytes(run.checkpoint)
        Path(path+'-plastic-open.json').write_text(json.dumps(dict(status=run.status,failure=run.failure,
            completed=run.accepted.completed_targets,candidates=owner.candidate_evaluations),indent=2))
    assert run.status=='paused',run.failure
    assert run.accepted.completed_targets==2
    assert run.accepted.contact.contact.minimum_gap_m>0.
    assert run.accepted.contact.contact.potential_j==0.
    report,proposed=_independent_material_history(owner,run.accepted.mechanical,origins)
    assert tuple(proposed)==run.accepted.native_histories[0].stations
    assert canonical(origins)==before
    from test_ge_beam3_fibre_section import exact_pair
    for current,previous in zip(proposed,origins[0].stations):
        for new,old in zip(current.rows,previous.rows):assert exact_pair(new[2:])>=exact_pair(old[2:])
    restored,records=_analysis(physical_fibres=True,plastic_fibres=True)._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.native_histories==run.accepted.native_histories and owner._checkpoint(records)==run.checkpoint
    if path:Path(path+'-plastic-unloading-history.json').write_text(json.dumps(report,indent=2))


def test_plastic_contact_recontact_and_complete_chain_replay():
    raw=(Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_open_v1.checkpoint').read_bytes()
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    origins=owner.context.histories(json.loads(raw)['records'][-1]['histories'],decoded=True)
    before=canonical(origins)
    run=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest())
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-plastic-recontact.checkpoint').write_bytes(run.checkpoint)
        Path(path+'-plastic-recontact.json').write_text(json.dumps(dict(status=run.status,failure=run.failure,
            completed=run.accepted.completed_targets,candidates=owner.candidate_evaluations),indent=2))
    assert run.status=='completed',run.failure
    assert run.accepted.completed_targets==3
    contact=run.accepted.contact.contact
    assert contact.potential_j>0. and np.linalg.norm(contact.beam_force_n)>0.
    assert contact.minimum_gap_m>=-3e-5
    np.testing.assert_allclose(contact.beam_force_n+contact.shell_force_n,0.,atol=1e-10)
    np.testing.assert_allclose(contact.beam_moment_nm+contact.shell_moment_nm,0.,atol=1e-10)
    report,proposed=_independent_material_history(owner,run.accepted.mechanical,origins)
    assert tuple(proposed)==run.accepted.native_histories[0].stations and canonical(origins)==before
    from test_ge_beam3_fibre_section import exact_pair
    for current,previous in zip(proposed,origins[0].stations):
        for new,old in zip(current.rows,previous.rows):assert exact_pair(new[2:])>=exact_pair(old[2:])
    restored,records=_analysis(physical_fibres=True,plastic_fibres=True)._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.native_histories==run.accepted.native_histories and owner._checkpoint(records)==run.checkpoint
    assert canonical(records[:2])==canonical(json.loads(raw)['records'])
    if path:Path(path+'-plastic-recontact-history.json').write_text(json.dumps(report,indent=2))
