"""Small assembled native-fibre transactions and adversarial replay tests."""
import json
from hashlib import sha256
import numpy as np
import pytest

from anyfesolver.control import CancellationToken
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_retained_fibre_program import solve_force_program
from anyfesolver._ge_beam3_retained_fibre_state import Context, Layout
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from test_ge_beam3_retained_fibre import make_model


def program(**kwargs):
    return ForceProgram((.25, .5, 1., .5, 0., -.5, 0.), ((5, .4, -.005, 0.),), max_iterations=24, **kwargs)


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_assembled_fibre_loading_unloading_reversal_and_replay(contrast, tmp_path):
    p = program(); model = make_model(contrast); progress = []
    before = canonical([e.to_dict() for e in model.mesh.elements.values()])
    whole = solve_force_program(model, p, progress=progress.append)
    with (tmp_path/'whole.json').open('xb') as stream: stream.write(whole.checkpoint)
    with (tmp_path/'summary.json').open('xb') as stream:
        stream.write(canonical(dict(status=whole.status, failure=whole.failure, completed=whole.completed_targets, progress=progress)))
    assert whole.status == 'completed', whole.failure
    assert canonical([e.to_dict() for e in model.mesh.elements.values()]) == before
    assert any(row[2] > 0 for cell in whole.state.histories for station in cell.stations for row in station.rows)
    value = json.loads(whole.checkpoint)
    # Physical support reactions balance the actual spatial dead loads at each
    # accepted configuration, including force lever arms in moment equilibrium.
    for record in value['records']:
        positions = np.array(record['mechanical']['positions'])+np.array(record['mechanical']['position_low'])
        reaction = np.array(record['residual'][:30]).reshape(5, 6)
        force = np.zeros((5, 3)); force[-1] = record['parameter']*np.array([.4, -.005, 0.])
        assert np.linalg.norm(np.sum(reaction[:, :3]+force, axis=0)) <= 1e-11
        moment = np.sum(reaction[:, 3:]+np.cross(positions, reaction[:, :3]+force), axis=0)
        assert np.linalg.norm(moment) <= 1e-11
        assert max(record['metrics']) <= 1e-11
    paused = solve_force_program(make_model(contrast), p, stop_after=3)
    assert paused.status == 'paused', paused.failure
    resumed = solve_force_program(make_model(contrast), p, checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    assert resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == whole.checkpoint
    context = Context(make_model(contrast), p)
    final, _ = context.restore(resumed.checkpoint)
    assert canonical(final.histories) == canonical(whole.state.histories)
    recovery = context.recover(final)
    with (tmp_path/'paused.json').open('xb') as stream: stream.write(paused.checkpoint)
    with (tmp_path/'recovery.json').open('xb') as stream: stream.write(canonical(recovery))
    for array in final.mechanical.descriptor().values():
        with pytest.raises(ValueError): array.setflags(write=True)


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    root = tmp_path_factory.mktemp('accepted-fibre-state')
    p = program(); result = solve_force_program(make_model(), p, stop_after=3)
    with (root/'paused.json').open('xb') as stream: stream.write(result.checkpoint)
    assert result.status == 'paused', result.failure
    return p, result


def rehash(value):
    previous = value['initial']['record_sha256']
    for row in value['records']:
        row['previous_sha256'] = previous
        row['record_sha256'] = sha({k: v for k, v in row.items() if k != 'record_sha256'})
        previous = row['record_sha256']
    value['checkpoint_sha256'] = sha({k: v for k, v in value.items() if k != 'checkpoint_sha256'})
    return canonical(value)


@pytest.mark.parametrize('field', ['position', 'frame', 'resultant', 'origin', 'origin_low', 'history', 'history_low', 'history_bool', 'history_shape',
    'cell_identity', 'section_identity', 'reaction', 'metrics', 'recovery', 'material', 'parameter', 'iterations_bool',
    'schema', 'model', 'cursor_bool', 'genesis', 'extra'])
def test_rehashed_fibre_state_mutations_are_rejected(accepted, field):
    p, original = accepted; value = json.loads(original.checkpoint); row = value['records'][-1]
    if field == 'position': row['mechanical']['positions'][-1][0] += .001
    elif field == 'frame': row['mechanical']['nodal_frames'][-1][0][0] += .01
    elif field == 'resultant': row['mechanical']['resultants'][-1][0] += .01
    elif field == 'origin': row['origins'][0]['stations'][0]['rows'][0][0] += .001
    elif field == 'origin_low': row['origins'][0]['stations'][0]['rows'][0][1] += 1e-20
    elif field == 'history': row['histories'][0]['stations'][0]['rows'][0][2] += .001
    elif field == 'history_low': row['histories'][0]['stations'][0]['rows'][0][3] += 1e-20
    elif field == 'history_bool': row['histories'][0]['stations'][0]['rows'][0][0] = True
    elif field == 'history_shape': row['histories'][0]['stations'][0]['rows'].pop()
    elif field == 'cell_identity': row['histories'][0]['cell_identity'] = '0'*64
    elif field == 'section_identity': row['histories'][0]['stations'][0]['section_identity'] = '0'*64
    elif field == 'reaction': row['residual'][0] += .001
    elif field == 'metrics': row['metrics'][0] = 0.
    elif field == 'recovery': row['recovery_sha256'] = '0'*64
    elif field == 'material': row['material_sha256'] = '0'*64
    elif field == 'parameter': row['parameter'] = .99
    elif field == 'iterations_bool': row['iterations'] = True
    elif field == 'schema': value['schema'] = 'GE_BEAM3_RETAINED_PAIRED_PLASTIC_ACCEPTED_CHAIN_V1'
    elif field == 'model': value['model_sha256'] = '0'*64
    elif field == 'cursor_bool': value['completed_targets'] = True
    elif field == 'genesis': value['initial']['histories'][0]['stations'][0]['rows'][0][2] += .1
    else: row['extra'] = 1
    with pytest.raises(ValueError): solve_force_program(make_model(), p, checkpoint=rehash(value))


@pytest.mark.parametrize('stage', ['retained-fibre.before_assembly', 'retained-fibre.before_factorization',
    'retained-fibre.before_trial', 'retained-fibre.before_commit'])
def test_cancellation_preserves_last_accepted_fibre_state(accepted, stage):
    p, original = accepted; token = CancellationToken()
    def observe(event):
        if event['stage'] == stage: token.cancel()
    result = solve_force_program(make_model(), p, checkpoint=original.checkpoint, cancellation_token=token, progress=observe)
    assert result.status == 'cancelled' and result.completed_targets == 3
    assert result.checkpoint == original.checkpoint
    assert canonical(result.state.histories) == canonical(original.state.histories)


@pytest.mark.parametrize('change', ['exception', 'node', 'material'])
def test_failed_precommit_preserves_original_capsule(accepted, change):
    p, original = accepted; model = make_model()
    def observe(event):
        if event['stage'] == 'retained-fibre.before_commit':
            if change == 'exception': raise RuntimeError('injected fibre commit failure')
            if change == 'node': model.mesh.nodes[5].x += .1
            if change == 'material': model.materials[model.mesh.elements[1].material_name] = object()
    result = solve_force_program(model, p, checkpoint=original.checkpoint, progress=observe)
    assert result.status == 'failed' and result.completed_targets == 3
    assert result.checkpoint == original.checkpoint


def test_replay_cannot_advance_geometric_state(accepted, monkeypatch):
    p, original = accepted
    def forbidden(*args, **kwargs): raise AssertionError('replay must not advance a mechanical increment')
    monkeypatch.setattr(Layout, 'advance', forbidden)
    replay = solve_force_program(make_model(), p, checkpoint=original.checkpoint, stop_after=3)
    assert replay.status == 'paused' and replay.checkpoint == original.checkpoint


def test_strict_json_foreign_schedule_and_model_binding(accepted):
    p, original = accepted
    for raw in (original.checkpoint+b' ', original.checkpoint.replace(b'"schema":', b'"schema":"duplicate","schema":', 1), b'{"x":NaN}'):
        with pytest.raises((ValueError, UnicodeError)): solve_force_program(make_model(), p, checkpoint=raw)
    with pytest.raises(ValueError): solve_force_program(make_model(1.e4), p, checkpoint=original.checkpoint)
    with pytest.raises(ValueError): solve_force_program(make_model(), p, checkpoint=original.checkpoint, expected_checkpoint_sha256='0'*64)
    with pytest.raises(ValueError): solve_force_program(make_model(), p, checkpoint=original.checkpoint, stop_after=2)
    changed = ForceProgram((.25, .5, 2., .5, 0., -.5, 0.), p.nodal_forces, max_iterations=24)
    with pytest.raises(ValueError): solve_force_program(make_model(), changed, checkpoint=original.checkpoint)


def test_newton_limit_keeps_virgin_state():
    p = ForceProgram((1.,), ((5, .4, -.005, 0.),), max_iterations=0)
    initial = solve_force_program(make_model(), p, stop_after=0)
    failed = solve_force_program(make_model(), p, checkpoint=initial.checkpoint)
    assert failed.status == 'failed' and failed.completed_targets == 0
    assert failed.checkpoint == initial.checkpoint


def test_cancellation_inside_cell_arithmetic_preserves_origin(accepted):
    p, original = accepted
    class MaterialCancellation(CancellationToken):
        armed = False
        checks = 0
        def raise_if_cancelled(self, stage=''):
            if self.armed and stage == 'retained-fibre.material':
                self.checks += 1
                if self.checks == 10: self.cancel('inside material arithmetic')
            super().raise_if_cancelled(stage)
    token = MaterialCancellation()
    def observe(event):
        if event['stage'] == 'retained-fibre.before_assembly': token.armed = True
    result = solve_force_program(make_model(), p, checkpoint=original.checkpoint, cancellation_token=token, progress=observe)
    assert token.checks == 10 and result.status == 'cancelled'
    assert result.checkpoint == original.checkpoint
    assert canonical(result.state.histories) == canonical(original.state.histories)


def test_cancellation_after_commit_retains_new_accepted_state(accepted):
    p, original = accepted; token = CancellationToken()
    def observe(event):
        if event['stage'] == 'retained-fibre.committed': token.cancel()
    result = solve_force_program(make_model(), p, checkpoint=original.checkpoint, cancellation_token=token, progress=observe)
    expected = solve_force_program(make_model(), p, stop_after=4)
    assert expected.status == 'paused', expected.failure
    assert result.status == 'cancelled' and result.completed_targets == 4
    assert result.checkpoint == expected.checkpoint


def test_external_hash_protects_unprovable_iteration_metadata(accepted):
    p, original = accepted; value = json.loads(original.checkpoint)
    value['records'][-1]['iterations'] += 1
    with pytest.raises(ValueError, match='external fibre checkpoint hash mismatch'):
        solve_force_program(make_model(), p, checkpoint=rehash(value),
                            expected_checkpoint_sha256=sha256(original.checkpoint).hexdigest())
