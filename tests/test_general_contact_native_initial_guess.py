"""First-target guesses preserve fine physical origins and admission gates."""
from pathlib import Path
from hashlib import sha256
from dataclasses import replace
import json,os
import numpy as np
import pytest
from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from test_general_contact_native_curved_s3_static import _analysis


def _fixture():
    raw=(Path(__file__).parent/'data/general_contact_native_fibre_s3_plastic_active_v1.checkpoint').read_bytes()
    source=_analysis(physical_fibres=True,plastic_fibres=True)
    owner=_analysis(physical_fibres=True,plastic_fibres=True,normal_stiffness_Pa_per_m=2e7)
    guess=owner.make_initial_guess(source,raw,sha256(raw).hexdigest())
    return owner,source,raw,guess


def _changed(guess,step):
    result=replace(guess,increment=np.array(step,copy=True))
    return replace(result,fingerprint=sha(result.descriptor()))


def test_guess_trial_uses_virgin_material_and_preserves_initial_prefix(monkeypatch):
    owner,source,raw,guess=_fixture()
    prefix=owner._checkpoint(());origin=canonical(owner.initial.native_histories)
    assert guess.owner_identity==owner.identity and guess.source_owner_identity==source.identity
    assert guess.source_checkpoint_sha256==sha256(raw).hexdigest()
    assert not guess.increment.flags.writeable
    assert canonical(json.loads(raw)['records'][0]['histories'])!=origin
    assert any(row[2]+row[3]>0. for cell in json.loads(raw)['records'][0]['histories']
               for station in cell['stations'] for row in station['rows'])
    assert all(not np.any(station.rows) for cell in owner.initial.native_histories for station in cell.stations)
    expected=owner.context.advance(owner.initial.mechanical,guess.increment[:owner.context.count])
    # Compare with independently decoded SOURCE fields, not another update.
    from fractions import Fraction as F
    saved=json.loads(raw)['records'][0]['mechanical']
    for high,low,source_high,source_low in zip(expected.positions.flat,expected.position_low.flat,
            np.asarray(saved['positions']).flat,np.asarray(saved['position_low']).flat):
        error=F(float(high))+F(float(low))-F(float(source_high))-F(float(source_low))
        assert abs(float(error))<=1e-14  # 2m fixture binary64 reconstruction budget
    np.testing.assert_allclose(expected.nodal_frames,saved['nodal_frames'],rtol=0.,atol=1e-13)
    np.testing.assert_allclose(expected.cell_rotations,saved['cell_rotations'],rtol=0.,atol=1e-13)
    np.testing.assert_array_equal(expected.resultants,saved['resultants'])
    assemble=owner._assemble;seen=[]
    def interrupt(trial,u,parameter,origins):
        assert canonical(origins)==origin
        assert canonical(trial.descriptor())==canonical(expected.descriptor())
        np.testing.assert_array_equal(u,owner.initial.shell_displacement+guess.increment[owner.context.count:])
        response=assemble(trial,u,parameter,origins)
        seen.append(response[2])
        raise RuntimeError('verified fine trial interruption')
    monkeypatch.setattr(owner,'_assemble',interrupt)
    run=owner.solve(stop_after=1,initial_guess=guess)
    assert seen and run.status=='failed' and 'verified fine trial interruption' in run.failure
    assert run.checkpoint==prefix and run.accepted.completed_targets==0
    assert canonical(run.accepted.native_histories)==origin
    path=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if path:Path(path+'-initial-guess-origin.json').write_text(json.dumps(dict(
        fingerprint=guess.fingerprint,source_checkpoint_sha256=guess.source_checkpoint_sha256,
        trial_metrics=seen,completed=run.accepted.completed_targets,status=run.status),indent=2))


def test_guess_source_binding_and_fresh_only_use():
    owner,source,raw,guess=_fixture()
    with pytest.raises(ValueError,match='hash'):
        owner.make_initial_guess(source,raw,'0'*64)
    different=_analysis(physical_fibres=True,plastic_fibres=True,targets=(.5,0.,1.))
    with pytest.raises(ValueError,match='structural identity'):
        owner.make_initial_guess(different,raw,sha256(raw).hexdigest())
    with pytest.raises(ValueError,match='fresh'):
        owner.solve(checkpoint=owner._checkpoint(()),expected_sha256=sha256(owner._checkpoint(())).hexdigest(),initial_guess=guess)
    with pytest.raises(ValueError,match='fresh'):owner.solve(stop_after=0,initial_guess=guess)
    assert owner.solve(stop_after=0).checkpoint==owner._checkpoint(())


@pytest.mark.parametrize('kind',('binding','shell_support','native_support','rotation'))
def test_guess_rejection_keeps_initial_state(kind):
    owner,_,_,guess=_fixture();prefix=owner._checkpoint(())
    step=guess.increment.copy()
    if kind=='binding':bad=replace(guess,source_checkpoint_sha256='0'*64)
    elif kind=='shell_support':
        step[owner.context.count+owner.shell_fixed[0]]=.001;bad=_changed(guess,step)
    elif kind=='native_support':
        step[0]=.001;bad=_changed(guess,step)
    else:
        step[owner.context.nodal_count]=.45*np.pi;bad=_changed(guess,step)
    run=owner.solve(stop_after=1,initial_guess=bad)
    word='binding' if kind=='binding' else 'rotation' if kind=='rotation' else 'support'
    assert run.status=='failed' and word in run.failure
    assert run.checkpoint==prefix and run.accepted.completed_targets==0


@pytest.mark.parametrize('cancelled',(False,True))
def test_guess_path_rejection_and_cancellation_are_atomic(monkeypatch,cancelled):
    owner,_,_,guess=_fixture();prefix=owner._checkpoint(());token=CancellationToken();seen=[]
    def path(start,u0,step,end,u1,**kwargs):
        assert start is owner.initial.mechanical
        np.testing.assert_array_equal(step,guess.increment)
        seen.append(True)
        if cancelled:token.cancel();return True
        return False
    monkeypatch.setattr(owner,'_swept_cancellable',path)
    run=owner.solve(stop_after=1,initial_guess=guess,cancellation_token=token)
    assert seen and run.status==('cancelled' if cancelled else 'failed')
    if not cancelled:assert 'swept path' in run.failure
    assert run.checkpoint==prefix and run.accepted.completed_targets==0
