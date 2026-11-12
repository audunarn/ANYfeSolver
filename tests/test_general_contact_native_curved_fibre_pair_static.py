"""Bounded actual-curved physical-fibre pair integration; development evidence."""
import os
import sys
import time
import json
from pathlib import Path
from hashlib import sha256

import numpy as np
import pytest

from test_general_contact_native_fibre_pair_static import (
    _owner, _check_coupled_commit_and_restart,
)


_CURVED_CHECKPOINT_SHA256 = 'e8d599af078d7d5108824d7eced7f0080f0ebda3c168a773fdbdc76d620f452e'
_CURVED_RESPONSE_SHA256 = '7077104031a891baa39c910a9f32c00480d512604e844be57e5b3ac0aa753dbd'
_CURVED_ORIGINAL_CHECKPOINT_SHA256 = 'a1d72a795421b1d7351d9bcbcfb06faa55c7fe1a3a732a204bd92c27de879086'
_CURVED_ROOT_RECORD_SHA256 = 'eb99cf5f9696ba316e82c22270a2683e3127a85e6ef48f7efa16352bfe27c7f6'


def _saved_curved_fixture():
    """Pinned provisional replay of an archived pose, not a new coupled solve."""
    data = Path(__file__).parent/'data'
    raw = (data/'general_contact_native_curved_fibre_pair_v14.checkpoint').read_bytes()
    response_raw = (data/'general_contact_native_curved_fibre_pair_v14_response.json').read_bytes()
    assert sha256(response_raw).hexdigest() == _CURVED_RESPONSE_SHA256
    response = json.loads(response_raw)
    assert sha256(raw).hexdigest() == response['checkpoint_sha256'] == _CURVED_CHECKPOINT_SHA256
    provenance = response['provenance']
    assert provenance['provisional'] is True
    assert provenance['coupled_solve_rerun'] is False
    assert provenance['kind'] == 'same_accepted_pose_canonical_root_replay'
    assert provenance['original_checkpoint_sha256'] == _CURVED_ORIGINAL_CHECKPOINT_SHA256
    assert provenance['new_record_artifact_sha256'] == _CURVED_ROOT_RECORD_SHA256
    return raw, response


def test_saved_curved_fixture_provisional_provenance_and_response_are_exact():
    """All four-profile record bytes and response fields remain explicitly bound."""
    from anyfesolver._ge_beam3_p5_seeded.codec import _load
    from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
    raw, response = _saved_curved_fixture()
    capsule = _load(raw.decode('ascii'))
    assert capsule['completed_targets'] == len(capsule['records']) == 1
    body = {key: value for key, value in capsule.items() if key != 'checkpoint_sha256'}
    assert capsule['checkpoint_sha256'] == sha(body)
    record = capsule['records'][0]
    artifact = json.dumps(record, sort_keys=True, allow_nan=False).encode('utf-8')
    assert sha256(artifact).hexdigest() == _CURVED_ROOT_RECORD_SHA256
    provenance = response['provenance']
    assert sha(record) == provenance['new_record_canonical_sha256']
    profiles = provenance['four_profile_evidence']
    assert {row['attempt'] for row in profiles} == {
        'win-record-03', 'linux-record-03', 'win-record-haswell-04', 'linux-record-haswell-04'}
    assert len(profiles) == 4
    assert all(row['record_artifact_sha256'] == _CURVED_ROOT_RECORD_SHA256 for row in profiles)
    for key, contact_key in (('force_a_n', 'beam_a_force_n'),
                             ('moment_a_nm', 'beam_a_moment_nm'),
                             ('gap_lower_m', 'conservative_lower_gap_m')):
        assert canonical(response[key]) == canonical(record['contact'][contact_key])
    # These describe the original completed solve. No new trajectory is claimed.
    assert response['status'] == 'completed' and response['failure'] is None
    assert response['completed'] == 1 and response['targets'] == [.05]
    assert response['line_stiffness_n_per_m2'] == 4000.


def _curved_owner(line_stiffness=4000.,initial_trial=None):
    return _owner(line_stiffness=line_stiffness,gap_step_control=True,
        reference_rise_m=.005,certify_curved_ownership=True,initial_trial=initial_trial)


def test_actual_curved_reference_and_restart_bind_opt_in():
    owner=_curved_owner()
    from anyfesolver._ge_beam3_p5_seeded.codec import _load
    frozen,_=_saved_curved_fixture()
    assert owner.identity==_load(frozen.decode())['identity']
    for curve in owner.contact.reference_curves:
        assert np.linalg.norm(curve[1]-.5*(curve[0]+curve[2]))>1e-4
    raw=owner._checkpoint(())
    state,records=_curved_owner()._restore(raw,sha256(raw).hexdigest())
    assert state.completed_targets==0 and owner._checkpoint(records)==raw
    for incompatible in (
        _owner(line_stiffness=4000.,gap_step_control=True),
        _owner(line_stiffness=4000.,gap_step_control=True,reference_rise_m=.005),
    ):
        with pytest.raises(ValueError,match='binding'):
            incompatible._restore(raw,sha256(raw).hexdigest())


def test_opt_in_straight_pair_preserves_analytic_line_law():
    from test_general_contact_native_curved_pair_ownership import _kernel,_reference,_pose
    # Unequal material-cell breaks on the same physical interval. Constant gap
    # and declared reference line measure give an independent scalar solution.
    a=_reference(.09998,(0.,1.,2.24));b=_reference(0.,(0.,1.12,2.24))
    kernel=_kernel(a,b,certify_curved_ownership=True)
    actual=kernel.evaluate(*_pose(a),*_pose(b))
    length=2.24;gap=.09998-.1;stiffness=100.
    assert actual.potential_j==pytest.approx(.5*stiffness*length*gap**2,rel=1e-8)
    np.testing.assert_allclose(actual.beam_a_force_n,(0.,-stiffness*length*gap,0.),
        rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(actual.beam_b_force_n,-actual.beam_a_force_n,atol=1e-12)


def _coupled_curved(line_stiffness=4000.,initial_trial=None):
    owner=_curved_owner(line_stiffness,initial_trial)
    evaluate=owner._contact
    query=owner._physical_gap
    def trace_context():
        frame=sys._getframe(2)
        result=dict(call_path=[])
        for _ in range(6):
            if frame is None:
                break
            result['call_path'].append(frame.f_code.co_name)
            iteration=frame.f_locals.get('iteration')
            if isinstance(iteration,int):
                result['iteration']=iteration
            frame=frame.f_back
        return result
    gap_calls=0
    def recorded_gap(state):
        nonlocal gap_calls
        gap_calls+=1
        started=time.perf_counter()
        event=dict(call=gap_calls,**trace_context())
        output=os.environ.get('CONTACT_APP_RESULT_RECORD')
        if output:
            Path(output+'-last-gap-pose.json').write_text(json.dumps(dict(
                call=gap_calls,
                pose={key:np.asarray(value).tolist()
                      for key,value in state.descriptor().items()},
                reference_rise_m=.005,gap_tolerance_m=5e-5,
                line_stiffness_n_per_m2=owner.contact.stiffness),indent=2))
        try:
            result=query(state)
            event.update(status='completed',lower_gap_m=result)
            return result
        except Exception as failure:
            event.update(status='failed',failure=f'{type(failure).__name__}: {failure}')
            raise
        finally:
            if output:
                event['elapsed_s']=time.perf_counter()-started
                with Path(output+'-gap-trace.ndjson').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps(event)+'\n')
    owner._physical_gap=recorded_gap
    calls=0
    def recorded_contact(state):
        nonlocal calls
        calls+=1
        started=time.perf_counter()
        event=dict(call=calls,**trace_context())
        output=os.environ.get('CONTACT_APP_RESULT_RECORD')
        if output:
            # Retain the latest attempted pose even when the bounded process
            # stops during a successful-but-expensive evaluation.
            Path(output+'-last-contact-pose.json').write_text(json.dumps(dict(
                call=calls,
                pose={key:np.asarray(value).tolist()
                      for key,value in state.descriptor().items()},
                reference_rise_m=.005,gap_tolerance_m=5e-5,
                line_stiffness_n_per_m2=owner.contact.stiffness),indent=2))
        try:
            result=evaluate(state)
            event.update(status='completed', **{key:getattr(result,key)
                for key in ('potential_j','conservative_lower_gap_m',
                    'minimum_sampled_gap_m',
                    'integration_intervals','integration_evaluations',
                    'integration_cells_checked')})
            return result
        except Exception as failure:
            event.update(status='failed',failure=f'{type(failure).__name__}: {failure}')
            output=os.environ.get('CONTACT_APP_RESULT_RECORD')
            if output:
                Path(output+'-failed-contact-pose.json').write_text(json.dumps(dict(
                    failure=f'{type(failure).__name__}: {failure}',
                    pose={key:np.asarray(value).tolist()
                          for key,value in state.descriptor().items()},
                    reference_rise_m=.005,gap_tolerance_m=5e-5,
                    line_stiffness_n_per_m2=owner.contact.stiffness),indent=2))
            raise
        finally:
            output=os.environ.get('CONTACT_APP_RESULT_RECORD')
            if output:
                event['elapsed_s']=time.perf_counter()-started
                with Path(output+'-contact-trace.ndjson').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps(event)+'\n')
    owner._contact=recorded_contact
    record=_check_coupled_commit_and_restart(owner,
        restart_factory=lambda:_curved_owner(line_stiffness,initial_trial))
    assert record['gap_lower_m']>=-5e-5
    return record


@pytest.mark.skipif(os.environ.get('CONTACT_NATIVE_CURVED_PAIR_SOLVE')!='1',
    reason='explicit bounded actual-curved coupled development run')
def test_actual_curved_plastic_pair_commit_material_work_and_exact_restart():
    _coupled_curved()


@pytest.mark.skipif(os.environ.get('CONTACT_NATIVE_CURVED_PAIR_REFINE')!='1',
    reason='explicit fixed-discretization normal-enforcement sensitivity run')
def test_actual_curved_pair_normal_enforcement_sensitivity():
    _normal_enforcement_sensitivity()


def _normal_enforcement_sensitivity(*,use_saved_trial=False):
    capsule,reference=_saved_curved_fixture()
    assert reference['status']=='completed' and reference['line_stiffness_n_per_m2']==4000.
    initial_trial=None
    if use_saved_trial:
        # Extract mechanical coordinates only. This is a virgin-origin solve,
        # never restoration of a checkpoint with a different contact law.
        from anyfesolver._ge_beam3_p5_seeded.codec import _load
        mechanical=_load(capsule.decode())['records'][-1]['mechanical']
        template=_curved_owner(8000.)
        initial_trial=dict(native_model=template.context.identity,mechanical=mechanical)
    current=_coupled_curved(8000.,initial_trial)
    errors={}
    for key in ('force_a_n','moment_a_nm'):
        before=np.asarray(reference[key]);after=np.asarray(current[key])
        errors[key]=float(np.linalg.norm(after-before)/np.linalg.norm(before))
    # Independently declared reference tip positions of the two curved beams.
    before=np.asarray(reference['positions']);after=np.asarray(current['positions'])
    for index,name,tip in ((2,'beam_a_tip',(2.,0.,0.)),
                           (5,'beam_b_tip',(2.,.101,0.))):
        errors[name]=float(np.linalg.norm(after[index]-before[index])/
            np.linalg.norm(before[index]-np.asarray(tip)))
    output=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if output:
        Path(output+'-normal-sensitivity.json').write_text(json.dumps(dict(
            scope='fixed-discretization development enforcement comparison only',
            reference_checkpoint_sha256=reference['checkpoint_sha256'],
            current_checkpoint_sha256=current['checkpoint_sha256'],
            initialization='disposable frozen mechanical trial; virgin physical history' if use_saved_trial else 'reference pose',
            stiffnesses_n_per_m2=[4000.,8000.],response_target=.02,
            gaps_m=[reference['gap_lower_m'],current['gap_lower_m']],errors=errors),indent=2))
    assert max(errors.values())<=.02


@pytest.mark.skipif(os.environ.get('CONTACT_NATIVE_CURVED_PAIR_SEEDED_REFINE')!='1',
    reason='explicit changed virgin-origin initial-trial enforcement comparison')
def test_actual_curved_pair_seeded_normal_enforcement_sensitivity():
    _normal_enforcement_sensitivity(use_saved_trial=True)
