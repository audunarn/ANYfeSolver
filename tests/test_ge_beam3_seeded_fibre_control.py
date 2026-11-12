"""Successor bounded control checks; predecessor failure remains historical."""
from hashlib import sha256
import json
import numpy as np
import pytest
import anyfesolver._ge_beam3_seeded_fibre_control as native
from anyfesolver._ge_beam3_retained_fibre_control import Context as OldContext
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken
from test_ge_beam3_retained_fibre_control import program, make_model, arch, transformed_model
from test_ge_beam3_retained_fibre_control import rehash


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_seeded_loading_unloading_reversal_and_split_restart(contrast, tmp_path):
    p = program(); events = []
    whole = native.solve_translation_program(make_model(contrast), p, progress=events.append)
    (tmp_path/'whole.json').write_bytes(whole.checkpoint)
    (tmp_path/'progress.json').write_bytes(canonical(dict(status=whole.status, failure=whole.failure, events=events)))
    assert whole.status == 'completed', whole.failure
    paused = native.solve_translation_program(make_model(contrast), p, stop_after=2)
    assert paused.status == 'paused', paused.failure
    resumed = native.solve_translation_program(make_model(contrast), p, checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    (tmp_path/'paused.json').write_bytes(paused.checkpoint)
    assert resumed.status == 'completed' and resumed.checkpoint == whole.checkpoint
    assert any(row[2] > 0 for cell in whole.state.histories for station in cell.stations for row in station.rows)
    assert max(max(row['metrics']) for row in json.loads(whole.checkpoint)['records']) <= 1e-11
    assert not whole.production_qualified


def test_chart_fraction_keeps_every_relative_rotation_inside_original_chart():
    context = native.Context(make_model(), program()); state = context.initial.mechanical
    step = np.zeros(context.layout.count)
    for node in (1, 2, 3, 4): step[6*node+3:6*node+6] = [30., -20., 10.]
    for index in (0, 1): step[context.layout.nodal_count+24*index:context.layout.nodal_count+24*index+6] = [-25., 12., 18., 30., -20., 15.]
    fraction = native.chart_fraction(context, state, step)
    assert 0 < fraction < 1.
    made = context.layout.advance(state, step*fraction)
    context.assemble(made, 0., context.initial.histories, 0.)
    assert native.chart_fraction(context, state, step*.01) == 1.


def test_seed_preserves_origins_and_does_not_convert_predecessor_checkpoint(tmp_path):
    p = program(); context = native.Context(make_model(1.e12), p); before = canonical(context.initial.histories)
    trial = context.initialize(context.initial.mechanical, context.initial.histories, .01)
    assert canonical(context.initial.histories) == before
    assert np.max(abs(trial.resultants)) < 1.
    assert context.initial.completed_targets == 0
    old = OldContext(make_model(1.e12), p)
    with pytest.raises(ValueError): context.restore(old.checkpoint(()))
    with pytest.raises(ValueError): old.restore(context.checkpoint(()))
    (tmp_path/'seed.json').write_bytes(canonical(trial.descriptor()))


def test_spatial_residual_derivative_is_not_the_off_equilibrium_chart_hessian():
    context = native.Context(make_model(), program()); layout = context.layout
    state = context.initialize(context.initial.mechanical, context.initial.histories, .01)
    r, h, _, _ = context.assemble(state, .1, context.initial.histories, .01); original = h.copy()
    direction = .03*np.sin(np.arange(layout.count)); direction[layout.fixed] = 0.; eps = 1e-6
    plus = context.assemble(layout.advance(state, direction*eps), .1, context.initial.histories, .01)[0]
    minus = context.assemble(layout.advance(state, -direction*eps), .1, context.initial.histories, .01)[0]
    reference = (plus-minus)/(2*eps)
    tangent = native.spatial_jacobian(layout, r, h)
    assert np.linalg.norm(reference-h@direction) > 1e-6
    assert np.linalg.norm(reference-tangent@direction)/max(1., np.linalg.norm(reference)) <= 1e-7
    np.testing.assert_array_equal(h, original)
    assert np.linalg.norm(h-h.T) <= 1e-11
    np.testing.assert_array_equal(native.spatial_jacobian(layout, np.zeros_like(r), h), h)


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    p = program(); result = native.solve_translation_program(make_model(), p, stop_after=2)
    assert result.status == 'paused', result.failure
    (tmp_path_factory.mktemp('accepted-seeded-control')/'paused.json').write_bytes(result.checkpoint)
    return p, result


@pytest.mark.parametrize('incident', ['cancel', 'mismatch', 'exception'])
def test_initialization_failure_never_commits_local_histories(accepted, monkeypatch, incident):
    p, original = accepted; token = CancellationToken(); source = native.kinematic_response
    def altered(*args, **kwargs):
        value = source(*args, **kwargs)
        if incident == 'cancel': token.cancel()
        elif incident == 'mismatch': value['resultants'][0][0] += .01
        else: raise RuntimeError('injected local initialization failure')
        return value
    monkeypatch.setattr(native, 'kinematic_response', altered)
    result = native.solve_translation_program(make_model(), p, checkpoint=original.checkpoint, cancellation_token=token)
    assert result.status == ('cancelled' if incident == 'cancel' else 'failed')
    assert result.checkpoint == original.checkpoint


@pytest.mark.parametrize('incident', ['parameter', 'origin', 'history', 'initialization'])
def test_resealed_successor_mutations_fail(accepted, incident):
    p, result = accepted; value = json.loads(result.checkpoint); row = value['records'][-1]
    if incident == 'parameter': row['parameter'] += .01
    elif incident == 'origin': row['origins'][0]['stations'][0]['rows'][0][0] += .01
    elif incident == 'history': row['histories'][0]['stations'][0]['rows'][0][2] += .01
    else: value['program']['initialization'] = 'foreign'
    with pytest.raises(ValueError): native.Context(make_model(), p).restore(rehash(value))


def test_capture_and_replay_do_not_reinitialize_accepted_steps(accepted, monkeypatch):
    p, original = accepted; context = native.Context(make_model(), p)
    def forbidden(*args, **kwargs): raise AssertionError('replay cannot initialize or advance')
    monkeypatch.setattr(context, 'initialize', forbidden)
    monkeypatch.setattr(type(context.layout), 'advance', forbidden)
    state, records = context.restore(original.checkpoint)
    assert state.completed_targets == 2 and context.checkpoint(records) == original.checkpoint
    context.program_data['initialization'] = 'changed'
    with pytest.raises(ValueError, match='capture changed'): context.guard()


def test_seeded_arch_continuation_keeps_the_descending_branch(tmp_path):
    p = native.TranslationProgram((.01, .025, .04, .055, .075, .1, .15, .2), 3, (0., -1., 0.), ((3, 0., -1., 0.),))
    result = native.solve_translation_program(arch(), p)
    (tmp_path/'arch.json').write_bytes(result.checkpoint)
    assert result.status == 'completed', result.failure
    loads = [row['parameter'] for row in json.loads(result.checkpoint)['records']]
    assert any(a < b > c for a, b, c in zip(loads, loads[1:], loads[2:]))
    assert any(a > b < c for a, b, c in zip(loads, loads[1:], loads[2:]))


def test_large_rotation_covariance_of_initialized_controller(accepted, tmp_path):
    from anyfesolver._ge_beam3_p5.algebra import rotation
    p, result = accepted; r = rotation([2.6, .8, -.3])
    direction = tuple(map(float, r@np.array(p.direction)))
    forces = tuple(map(float, r@np.array(p.nodal_forces[0][1:])))
    other_program = native.TranslationProgram(p.targets, 5, direction, ((5, *forces),))
    other = native.solve_translation_program(transformed_model(r), other_program, stop_after=2)
    (tmp_path/'rotated.json').write_bytes(other.checkpoint)
    assert other.status == 'paused', other.failure
    a = json.loads(result.checkpoint)['records']; b = json.loads(other.checkpoint)['records']
    error = max(abs(x['parameter']-y['parameter'])/max(1., abs(x['parameter'])) for x, y in zip(a, b))
    (tmp_path/'covariance.json').write_bytes(canonical(dict(relative_parameter_error=error)))
    assert error <= 1e-11
