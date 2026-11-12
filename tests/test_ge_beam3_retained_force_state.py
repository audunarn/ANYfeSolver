"""Owned retained-resultant states, restart replay and bounded finite trials."""
from copy import deepcopy
import json
import numpy as np
import pytest
from anyfesolver._ge_beam3_retained_force_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_retained_state import Context
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_curved_contrast_probe import make_curved
from docs.reference_cases.ge_beam3_full_resultant_force_probe import solve_two_macro_probe


def program(targets=(.5, 1.), **kwargs):
    return ForceProgram(targets, ((5, .05, -.001, 0.),), **kwargs)


@pytest.mark.parametrize('slenderness', [100., 10000., 1000000.])
def test_native_retained_state_matches_probe_and_restarts(slenderness, tmp_path):
    model, _ = make_curved(slenderness)
    value = solve_force_program(model, program())
    assert value.status == 'completed', value.failure
    baseline, _ = solve_two_macro_probe(model, [.05, -.001, 0.])
    for expected, actual in zip(baseline, value.state.descriptor().values()):
        np.testing.assert_array_equal(expected, actual)
    clone, _ = make_curved(slenderness)
    paused = solve_force_program(clone, program(), stop_after=1)
    assert paused.status == 'paused'
    restarted = solve_force_program(clone, program(), checkpoint=paused.checkpoint)
    assert restarted.status == 'completed', restarted.failure
    assert restarted.checkpoint == value.checkpoint
    replay = solve_force_program(clone, program(), checkpoint=restarted.checkpoint)
    assert replay.checkpoint == value.checkpoint
    with (tmp_path/'accepted.json').open('xb') as stream: stream.write(value.checkpoint)
    for array in value.state.descriptor().values():
        with pytest.raises(ValueError): array.setflags(write=True)


def test_retained_finite_load_unload_reverse_and_restart(tmp_path):
    model, _ = make_curved(1000000.)
    p = program((1., 2., 4., 8., 16., 8., 0., -4., 0.))
    value = solve_force_program(model, p)
    with (tmp_path/'last-accepted.json').open('xb') as stream: stream.write(value.checkpoint)
    assert value.status == 'completed', value.failure
    np.testing.assert_allclose(value.state.positions,
        [model.mesh.nodes[i].coords() for i in range(1, 6)], rtol=0, atol=1e-11)
    assert np.linalg.norm(value.state.resultants) < 1e-11
    paused = solve_force_program(model, p, stop_after=5)
    assert paused.status == 'paused', paused.failure
    assert np.linalg.norm(paused.state.positions[-1]-np.array(model.mesh.nodes[5].coords())) > .05
    np.testing.assert_allclose(paused.state.nodal_frames@np.swapaxes(paused.state.nodal_frames, -1, -2),
        np.broadcast_to(np.eye(3), (5, 3, 3)), rtol=1e-11, atol=1e-11)
    clone, _ = make_curved(1000000.)
    resumed = solve_force_program(clone, p, checkpoint=paused.checkpoint)
    assert resumed.checkpoint == value.checkpoint
    with (tmp_path/'finite-state.json').open('xb') as stream: stream.write(paused.checkpoint)


@pytest.fixture(scope='module')
def accepted():
    model, _ = make_curved(100.)
    p = program((.5, 1., 2.))
    result = solve_force_program(model, p, stop_after=1)
    assert result.status == 'paused'
    return p, result


@pytest.mark.parametrize('field', ['positions', 'position_low', 'nodal_frames', 'cell_rotations', 'resultants',
    'load_parameter', 'cursor', 'formulation', 'recovery', 'reaction', 'record', 'boolean', 'extra'])
def test_corrupted_retained_checkpoint_rejected_even_after_rehash(accepted, field):
    p, result = accepted; data = json.loads(result.checkpoint)
    if field in ('positions', 'position_low', 'resultants'):
        data['state'][field][-1][0] += .001
    elif field in ('nodal_frames', 'cell_rotations'):
        if field == 'nodal_frames': data['state'][field][-1][0][0] += .1
        else: data['state'][field][-1][0][0][0] += .1
    elif field == 'load_parameter': data[field] = 1.
    elif field == 'cursor': data['completed_targets'] = True
    elif field == 'formulation': data['formulation_id'] = 'CANDIDATE_GE_BEAM3_P5_LOAD_CORE_V5'
    elif field == 'recovery': data['recovery_sha256'] = '0'*64
    elif field == 'reaction': data['residual'][0] += .01
    elif field == 'record': data['records'][0]['iterations'] = True
    elif field == 'boolean': data['state']['positions'][0][0] = True
    else: data['extra'] = 1
    body = {k: v for k, v in data.items() if k != 'checkpoint_sha256'}
    forged = canonical({**body, 'checkpoint_sha256': sha(body)})
    model, _ = make_curved(100.)
    with pytest.raises(ValueError): solve_force_program(model, p, checkpoint=forged)


@pytest.mark.parametrize('stage', ['retained.before_assembly', 'retained.before_factorization',
    'retained.before_trial', 'retained.before_commit'])
def test_cancellation_preserves_last_accepted_capsule(accepted, stage):
    p, original = accepted; token = CancellationToken()
    model, _ = make_curved(100.)
    def observe(row):
        if row['stage'] == stage: token.cancel()
    result = solve_force_program(model, p, checkpoint=original.checkpoint,
        cancellation_token=token, progress=observe)
    assert result.status == 'cancelled'
    assert result.completed_targets == 1 and result.checkpoint == original.checkpoint
    for name, array in result.state.descriptor().items():
        np.testing.assert_array_equal(array, original.state.descriptor()[name])


def test_failed_trial_and_foreign_schedule_do_not_commit(accepted):
    p, original = accepted
    model, _ = make_curved(100.)
    with pytest.raises(ValueError): solve_force_program(model, program((.5, 2., 3.)), checkpoint=original.checkpoint)
    empty = solve_force_program(model, program(max_iterations=0), stop_after=0)
    failed = solve_force_program(model, program(max_iterations=0), checkpoint=empty.checkpoint)
    assert failed.status == 'failed' and failed.checkpoint == empty.checkpoint
    assert failed.completed_targets == 0


def test_noncanonical_duplicate_and_nonfinite_checkpoint_rejected(accepted):
    p, original = accepted; model, _ = make_curved(100.)
    for raw in (b'{"x":1,"x":1}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n',
                b' '+original.checkpoint, original.checkpoint.rstrip(), b'[]\n'):
        with pytest.raises(ValueError): solve_force_program(model, p, checkpoint=raw)


def test_production_model_inputs_are_not_modified(accepted):
    p, _ = accepted; model, _ = make_curved(100.)
    before = canonical([e.to_dict() for e in model.mesh.elements.values()])
    context = Context(model, p); value = solve_force_program(model, p)
    assert value.status == 'completed'
    assert canonical([e.to_dict() for e in model.mesh.elements.values()]) == before
    recovered = context.recover(value.state)
    assert len(recovered) == 2 and all(len(row['stations']) == 16 for row in recovered)


def test_exception_before_commit_keeps_accepted_state(accepted):
    p, original = accepted; model, _ = make_curved(100.)
    def observe(row):
        if row['stage'] == 'retained.before_commit': raise RuntimeError('injected publication failure')
    value = solve_force_program(model, p, checkpoint=original.checkpoint, progress=observe)
    assert value.status == 'failed' and value.checkpoint == original.checkpoint
    assert value.completed_targets == 1


def test_cancel_after_commit_retains_newly_accepted_state(accepted):
    p, original = accepted; model, _ = make_curved(100.); token = CancellationToken()
    def observe(row):
        if row['stage'] == 'retained.committed': token.cancel()
    value = solve_force_program(model, p, checkpoint=original.checkpoint,
        cancellation_token=token, progress=observe)
    assert value.status == 'cancelled' and value.completed_targets == 2
    clone, _ = make_curved(100.)
    expected = solve_force_program(clone, p, stop_after=2)
    assert value.checkpoint == expected.checkpoint


def test_model_mutation_before_commit_is_detected(accepted):
    p, original = accepted; model, _ = make_curved(100.)
    def observe(row):
        if row['stage'] == 'retained.before_commit': model.mesh.nodes[5].x += .01
    value = solve_force_program(model, p, checkpoint=original.checkpoint, progress=observe)
    assert value.status == 'failed' and value.checkpoint == original.checkpoint


def test_legacy_state_cannot_hot_restart_retained_candidate():
    from anyfesolver._ge_beam3_seeded_load_program import solve_force_program as historical
    model, _ = make_curved(100.)
    old = historical(model, program(), stop_after=0)
    assert old.status == 'paused'
    with pytest.raises(ValueError, match='schema'):
        solve_force_program(model, program(), checkpoint=old.checkpoint)


@pytest.mark.parametrize('targets', [(1,), (True,)])
def test_nonbinary64_schedule_rejected_before_any_trial(targets):
    model, _ = make_curved(100.)
    with pytest.raises(ValueError): solve_force_program(model, program(targets))
