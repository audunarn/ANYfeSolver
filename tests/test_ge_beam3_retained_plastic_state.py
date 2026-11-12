"""Model-bound paired-plastic continuation and adversarial state safety."""
import json
from hashlib import sha256
import numpy as np
import pytest
from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_retained_plastic_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_retained_plastic_state import Context
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from test_ge_beam3_retained_plastic_force_probe import model_with_plasticity
from docs.reference_cases.ge_beam3_retained_plastic_force_probe import solve as research_solve


def program(**kwargs):
    return ForceProgram((.25, .5, 1., .5, 0., -.5, 0.), ((5, .4, -.005, 0.),), max_iterations=24, **kwargs)


@pytest.mark.parametrize('contrast', [1., 1000000000000.])
def test_native_plastic_chain_matches_probe_and_resumes_exactly(contrast, tmp_path):
    p = program(); model = model_with_plasticity(contrast)
    before = canonical([e.to_dict() for e in model.mesh.elements.values()])
    whole = solve_force_program(model, p)
    (tmp_path/'whole.json').write_bytes(whole.checkpoint)
    assert whole.status == 'completed', whole.failure
    assert canonical([e.to_dict() for e in model.mesh.elements.values()]) == before
    probe = research_solve(model_with_plasticity(contrast), p)
    assert probe['status'] == 'completed', probe['failure']
    assert canonical(whole.state.mechanical.descriptor()) == canonical(probe['state'])
    assert canonical(whole.state.histories) == canonical(probe['histories'])
    paused = solve_force_program(model_with_plasticity(contrast), p, stop_after=3)
    assert paused.status == 'paused', paused.failure
    assert any(np.any(x[:, 2] > 0.) for x in paused.state.histories)
    resumed = solve_force_program(model_with_plasticity(contrast), p, checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    assert resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == whole.checkpoint
    final = solve_force_program(model_with_plasticity(contrast), p, checkpoint=resumed.checkpoint)
    assert final.status == 'completed' and final.checkpoint == whole.checkpoint
    context = Context(model_with_plasticity(contrast), p)
    recovery = context.recover(final.state)
    (tmp_path/'paused.json').write_bytes(paused.checkpoint)
    (tmp_path/'recovery.json').write_bytes(canonical(recovery))
    for array in (*final.state.mechanical.descriptor().values(), *final.state.origins, *final.state.histories):
        with pytest.raises(ValueError): array.setflags(write=True)


@pytest.fixture(scope='module')
def accepted():
    p = program(); result = solve_force_program(model_with_plasticity(1.), p, stop_after=3)
    assert result.status == 'paused', result.failure
    return p, result


def rehash(value):
    previous = value['initial']['record_sha256']
    for row in value['records']:
        row['previous_sha256'] = previous
        row['record_sha256'] = sha({k:v for k,v in row.items() if k != 'record_sha256'})
        previous = row['record_sha256']
    value['checkpoint_sha256'] = sha({k:v for k,v in value.items() if k != 'checkpoint_sha256'})
    return canonical(value)


@pytest.mark.parametrize('field', ['position', 'position_low', 'frame', 'cell_frame', 'resultant',
    'origin_high', 'origin_low', 'history_high', 'history_low', 'history_bool', 'history_shape',
    'reaction', 'metrics', 'recovery', 'material', 'parameter', 'target_bool', 'iterations_bool',
    'schema', 'model', 'cursor_bool', 'genesis', 'extra'])
def test_mutated_plastic_state_rejected_after_full_chain_rehash(accepted, field):
    p, original = accepted; value = json.loads(original.checkpoint); row = value['records'][-1]
    if field == 'position': row['mechanical']['positions'][-1][0] += .001
    elif field == 'position_low': row['mechanical']['position_low'][-1][0] += .001
    elif field == 'frame': row['mechanical']['nodal_frames'][-1][0][0] += .01
    elif field == 'cell_frame': row['mechanical']['cell_rotations'][-1][0][0][0] += .01
    elif field == 'resultant': row['mechanical']['resultants'][-1][0] += .01
    elif field.startswith('origin_'): row['origins'][0][0][0 if field.endswith('high') else 1] += .001
    elif field == 'history_high': row['histories'][0][0][2] += .001
    elif field == 'history_low': row['histories'][0][0][3] += .001
    elif field == 'history_bool': row['histories'][0][0][0] = True
    elif field == 'history_shape': row['histories'][0].pop()
    elif field == 'reaction': row['residual'][0] += .001
    elif field == 'metrics': row['metrics'][0] = 0.
    elif field == 'recovery': row['recovery_sha256'] = '0'*64
    elif field == 'material': row['material_sha256'] = '0'*64
    elif field == 'parameter': row['parameter'] = .99
    elif field == 'target_bool': row['target'] = True
    elif field == 'iterations_bool': row['iterations'] = True
    elif field == 'schema': value['schema'] = 'GE_BEAM3_RETAINED_ELASTIC_ACCEPTED_STATE_V1'
    elif field == 'model': value['model_sha256'] = '0'*64
    elif field == 'cursor_bool': value['completed_targets'] = True
    elif field == 'genesis': value['initial']['histories'][0][0][2] = .1
    else: row['extra'] = 1
    raw = rehash(value)
    with pytest.raises(ValueError): solve_force_program(model_with_plasticity(1.), p, checkpoint=raw)


@pytest.mark.parametrize('stage', ['paired-plastic.before_assembly', 'paired-plastic.before_factorization',
    'paired-plastic.before_trial', 'paired-plastic.before_commit'])
def test_cancellation_retains_exact_accepted_history(accepted, stage):
    p, original = accepted; token = CancellationToken()
    def observe(event):
        if event['stage'] == stage: token.cancel()
    result = solve_force_program(model_with_plasticity(1.), p, checkpoint=original.checkpoint,
        cancellation_token=token, progress=observe)
    assert result.status == 'cancelled' and result.completed_targets == 3
    assert result.checkpoint == original.checkpoint
    assert canonical(result.state.histories) == canonical(original.state.histories)


def test_cancellation_after_commit_keeps_new_state(accepted):
    p, original = accepted; token = CancellationToken()
    def observe(event):
        if event['stage'] == 'paired-plastic.committed': token.cancel()
    result = solve_force_program(model_with_plasticity(1.), p, checkpoint=original.checkpoint,
        cancellation_token=token, progress=observe)
    expected = solve_force_program(model_with_plasticity(1.), p, stop_after=4)
    assert result.status == 'cancelled' and result.completed_targets == 4
    assert result.checkpoint == expected.checkpoint


@pytest.mark.parametrize('change', ['exception', 'model', 'material'])
def test_failed_commit_does_not_publish_trial_state(accepted, change):
    p, original = accepted; model = model_with_plasticity(1.)
    def observe(event):
        if event['stage'] == 'paired-plastic.before_commit':
            if change == 'exception': raise RuntimeError('injected pre-publication failure')
            elif change == 'model': model.mesh.nodes[5].x += .001
            else: model.materials[model.mesh.elements[1].material_name] = object()
    result = solve_force_program(model, p, checkpoint=original.checkpoint, progress=observe)
    assert result.status == 'failed' and result.completed_targets == 3
    assert result.checkpoint == original.checkpoint


def test_invalid_serialization_and_foreign_model_rejected(accepted):
    p, original = accepted
    for bad in (b'{"x":1,"x":1}\n', b'{"x":NaN}\n', b'{"x":Infinity}\n',
                b' '+original.checkpoint, original.checkpoint.rstrip(), b'[]\n'):
        with pytest.raises(ValueError): solve_force_program(model_with_plasticity(1.), p, checkpoint=bad)
    with pytest.raises(ValueError): solve_force_program(model_with_plasticity(1e4), p, checkpoint=original.checkpoint)
    with pytest.raises(ValueError): solve_force_program(model_with_plasticity(1.), p, checkpoint=original.checkpoint,
        expected_checkpoint_sha256='0'*64)
    with pytest.raises(ValueError): Context(model_with_plasticity(1e4), p).recover(original.state)


def test_newton_limit_leaves_virgin_checkpoint_unchanged():
    p = ForceProgram((1.,), ((5, .4, -.005, 0.),), max_iterations=0)
    initial = solve_force_program(model_with_plasticity(1.), p, stop_after=0)
    failed = solve_force_program(model_with_plasticity(1.), p, checkpoint=initial.checkpoint)
    assert failed.status == 'failed' and failed.completed_targets == 0
    assert failed.checkpoint == initial.checkpoint


def test_replay_does_not_advance_or_resolve_newton_steps(accepted, monkeypatch):
    from anyfesolver._ge_beam3_retained_state import Context as Layout
    p, original = accepted
    def forbidden(*args, **kwargs): raise AssertionError('restart must not advance mechanics')
    monkeypatch.setattr(Layout, 'advance', forbidden)
    replay = solve_force_program(model_with_plasticity(1.), p, checkpoint=original.checkpoint, stop_after=3)
    assert replay.status == 'paused' and replay.checkpoint == original.checkpoint


def test_elastic_capsule_is_not_a_plastic_hot_restart():
    from anyfesolver._ge_beam3_retained_force_program import solve_force_program as elastic
    p = program(); model = model_with_plasticity(1.)
    old = elastic(model, p, stop_after=0)
    assert old.status == 'paused'
    with pytest.raises(ValueError, match='schema'): solve_force_program(model, p, checkpoint=old.checkpoint)
    new = solve_force_program(model, p, stop_after=0)
    with pytest.raises(ValueError): elastic(model, p, checkpoint=new.checkpoint)


def test_foreign_schedule_or_rewind_rejected(accepted):
    p, original = accepted
    changed = ForceProgram((.25, .5, 2., .5, 0., -.5, 0.), p.nodal_forces, max_iterations=24)
    with pytest.raises(ValueError): solve_force_program(model_with_plasticity(1.), changed, checkpoint=original.checkpoint)
    with pytest.raises(ValueError): solve_force_program(model_with_plasticity(1.), p, checkpoint=original.checkpoint, stop_after=2)


def test_declared_checkpoint_hash_protects_otherwise_valid_metadata(accepted):
    p, original = accepted; value = json.loads(original.checkpoint)
    # Physics cannot prove an iteration-count claim. An external exact-byte
    # authority hash additionally protects metadata from deliberate rewriting.
    value['records'][-1]['iterations'] += 1
    modified = rehash(value)
    with pytest.raises(ValueError, match='external checkpoint hash'):
        solve_force_program(model_with_plasticity(1.), p, checkpoint=modified,
            expected_checkpoint_sha256=sha256(original.checkpoint).hexdigest())
