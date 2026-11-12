"""Bounded accepted-state continuation of the installed curved S3 candidate."""
from hashlib import sha256
from pathlib import Path
import json,os
import runpy,sys
import numpy as np
import pytest
from test_general_contact_native_curved_s3_static import _analysis
from test_general_contact_native_curved_static import _canonical_fixture_payload

ACTIVE=Path(__file__).parent/'data/general_contact_native_curved_s3_v2_active.checkpoint'
OPEN=Path(__file__).parent/'data/general_contact_native_curved_s3_v2_open.checkpoint'


def test_curved_s3_active_restart_opens_and_preserves_prefix():
    raw=_canonical_fixture_payload(ACTIVE);owner=_analysis()
    run=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),stop_after=2)
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-s3-open.checkpoint').write_bytes(run.checkpoint)
        Path(path+'-curved-s3-open.json').write_text(json.dumps(dict(status=run.status,
            failure=run.failure,completed_targets=run.accepted.completed_targets,
            checkpoint_sha256=run.checkpoint_sha256,trial_failures=owner.curved_trial_failures),indent=2))
    assert run.status=='paused',run.failure
    rows=json.loads(run.checkpoint)['records'];prefix=json.loads(raw)['records']
    assert run.accepted.completed_targets==2 and rows[:1]==prefix
    assert rows[1]['contact']['beam_force_n']==[0.,0.,0.]
    assert rows[1]['contact']['potential_j']==0. and rows[1]['contact']['minimum_gap_m']>0.
    restored,records=_analysis()._restore(run.checkpoint,run.checkpoint_sha256)
    assert restored.completed_targets==2 and owner._checkpoint(records)==run.checkpoint


@pytest.mark.parametrize('cancelled',(False,True))
def test_curved_s3_active_prefix_survives_precommit_failure(cancelled):
    from anyfesolver.control import CancellationToken
    raw=_canonical_fixture_payload(ACTIVE);owner=_analysis();record=owner._record;token=CancellationToken()
    def injected(target,*args):
        staged=record(target,*args)
        if target==2:
            if cancelled:token.cancel()
            else:raise RuntimeError('curved S3 staged exception')
        return staged
    owner._record=injected
    run=owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),stop_after=2,cancellation_token=token)
    assert run.status==('cancelled' if cancelled else 'failed')
    assert run.accepted.completed_targets==1 and run.checkpoint==raw
    expected=json.loads(raw)['records'][0]
    for field,values in expected['mechanical'].items():
        np.testing.assert_array_equal(getattr(run.accepted.mechanical,field),values)
    np.testing.assert_array_equal(run.accepted.shell_displacement,expected['shell_displacement'])


def test_curved_s3_checkpoint_rejects_changed_normal_law():
    raw=_canonical_fixture_payload(ACTIVE);owner=_analysis(normal_stiffness_Pa_per_m=2e7)
    with pytest.raises(ValueError,match='binding'):
        owner.solve(checkpoint=raw,expected_sha256=sha256(raw).hexdigest(),stop_after=1)


def test_curved_s3_installed_example_recontacts_from_open_checkpoint(monkeypatch,tmp_path,capsys):
    example=Path(__file__).parents[1]/'examples/general_contact_native_curved_material_s3.py'
    output=tmp_path/'recontact.checkpoint'
    restart=tmp_path/'curved-s3-open-canonical.checkpoint'
    restart.write_bytes(_canonical_fixture_payload(OPEN))
    monkeypatch.setattr(sys,'argv',[str(example),'--through-target','3','--restart',str(restart),'--checkpoint-out',str(output)])
    runpy.run_path(str(example),run_name='__main__')
    report=json.loads(capsys.readouterr().out);raw=output.read_bytes()
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:
        Path(path+'-curved-s3-cycle.checkpoint').write_bytes(raw)
        Path(path+'-curved-s3-cycle.json').write_text(json.dumps(report,indent=2))
    assert report['status']=='completed' and report['completed_targets']==3
    assert report['checkpoint_sha256']==sha256(raw).hexdigest()
    rows=json.loads(raw)['records'];prefix=json.loads(_canonical_fixture_payload(OPEN))['records']
    assert rows[:2]==prefix
    assert rows[2]['contact']['beam_force_n'][2]>0 and rows[2]['contact']['minimum_gap_m']>=-3e-5
    np.testing.assert_allclose(rows[2]['contact']['beam_force_n'],rows[0]['contact']['beam_force_n'],rtol=1e-7,atol=1e-10)
    for field,values in rows[0]['mechanical'].items():
        np.testing.assert_allclose(rows[2]['mechanical'][field],values,rtol=1e-10,atol=1e-10)
    restored,records=_analysis()._restore(raw,sha256(raw).hexdigest())
    assert restored.completed_targets==3 and _analysis()._checkpoint(records)==raw
