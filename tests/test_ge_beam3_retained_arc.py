"""Retained arc development gates; scalar folds are not beam qualification."""
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_arc as arc
from anyfesolver._ge_beam3_retained_arc_geometry import frame_terms, constraint
from anyfesolver._ge_beam3_p5.algebra import rotation, skew
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_generalized_modal import make
from test_ge_beam3_schur_line_program import save


def program(steps=(.01, .02, .015), sign=1., transverse=False):
    force = (0., .1, 0.) if transverse else (1., 0., 0.)
    return arc.Program(steps, 2., arc.NodalDeadForces(((3, *force),)), initial_sign=sign)


def restore(c, raw):
    return c.restore(raw, expected_sha256=sha256(raw).hexdigest())


@pytest.mark.parametrize('angle', (0., 1e-9, .2, 1.3, 2.7))
def test_current_spatial_frame_variation(angle):
    axis = np.array((2., -3., 4.)); axis /= np.linalg.norm(axis)
    q0 = rotation(np.array((.8, -.2, .4))); q = rotation(angle*axis)@q0
    w = np.array((.4, -.7, .2)); value, gradient = frame_terms(q, q0, w)
    a = q@q0.T
    assert abs(value-.5*np.sum((a-np.eye(3))*skew(w))) <= 1e-11
    # Independent coordinatewise Frobenius derivative, no production row call.
    exact = np.array([.5*np.sum((skew(e)@a)*skew(w)) for e in np.eye(3)])
    assert np.linalg.norm(gradient-exact) <= 1e-11
    h = 2e-6
    fd = np.array([(frame_terms(rotation(h*e)@q, q0, w)[0]
                  - frame_terms(rotation(-h*e)@q, q0, w)[0])/(2*h) for e in np.eye(3)])
    assert np.linalg.norm(fd-gradient) <= 1e-7
    s = rotation(np.array((1.7, -.8, .5)))
    transformed, row = frame_terms(s@q, s@q0, s@w)
    assert abs(transformed-value) <= 1e-11 and np.linalg.norm(row-s@gradient) <= 1e-11
    if angle == 0.: assert np.linalg.norm(gradient-w) <= 1e-11


def test_large_coordinate_low_part_and_internal_exclusion():
    position = np.array([[2.**30, -2.**29, 2.**28]])
    q = np.array([rotation(np.array((.3, -.2, .1)))])
    base = SimpleNamespace(positions=position, position_low=np.zeros((1, 3)), nodal_frames=q)
    now = SimpleNamespace(positions=position, position_low=np.array([[2.**-30, 0., 0.]]), nodal_frames=q)
    d = np.zeros(10); d[0] = 1.; d[6:9] = (4., 5., 6.); d[-1] = .5
    metric = np.r_[np.ones(6), np.zeros(3), 1.]
    gap, row = constraint(now, base, d, metric, 0., 0., 0.)
    assert gap == 2.**-30 and np.array_equal(row[6:9], np.zeros(3))
    with pytest.raises(ValueError, match='internal'):
        constraint(now, base, d, np.ones(10), 0., 0., 0.)
    with pytest.raises(ValueError, match='isotropic'):
        constraint(now, base, d, np.r_[2., np.ones(5), np.zeros(3), 1.], 0., 0., 0.)
    with pytest.raises(ValueError, match='cutback'):
        frame_terms(rotation(np.array((.91*np.pi, 0., 0.))), np.eye(3), np.ones(3))


def test_full_border_crosses_scalar_fold():
    # r=x^2+lambda-1. At x=0 no unbordered inverse exists. Retain the
    # incoming tangent along decreasing x to follow the same oriented branch.
    free = np.array([0]); metric = np.ones(2); previous = np.array([-1., 0.])
    tangent = arc.direction(np.zeros((1, 1)), np.ones(1), free, previous, metric)
    np.testing.assert_array_equal(tangent, previous)
    matrix = arc.bordered(np.zeros((1, 1)), np.ones(1), tangent*metric, free)
    assert np.linalg.det(matrix) != 0.
    x, lam = -.1, 1.
    for _ in range(2):
        matrix = arc.bordered(np.array([[2*x]]), np.ones(1), tangent, free)
        delta = np.linalg.solve(matrix, -np.array([x*x+lam-1., -x-.1]))
        x, lam = float(x+delta[0]), float(lam+delta[1])
    assert x < 0. and abs(lam-.99) <= 1e-11
    next_tangent = arc.direction(np.array([[2*x]]), np.ones(1), free, tangent, metric)
    assert next_tangent[0] < 0. and next_tangent[-1] < 0.


@pytest.mark.parametrize('sign', (-1., 1.))
def test_axial_actual_controller_restart(sign, tmp_path):
    model, _, _ = make(False, 1, clamped=True, coupled=False); p = program(sign=sign)
    result = arc.solve(model, p); save(tmp_path/'actual.json', result.checkpoint)
    assert result.status == 'completed', result.failure
    partial = arc.solve(model, p, stop_after=1)
    assert partial.status == 'paused', partial.failure
    resumed = arc.solve(model, p, checkpoint=partial.checkpoint, expected_sha256=sha256(partial.checkpoint).hexdigest())
    assert resumed.status == 'completed', resumed.failure
    assert resumed.checkpoint == result.checkpoint
    c = arc.Context(model, p); state, records = restore(c, result.checkpoint)
    # u(x)=lambda*x/EA. Metric norm squared = 1 + sum((x/EA)^2)/(3*L^2).
    factor = np.sqrt(1.+5./(1000.**2*12.))
    for index, raw in enumerate(records):
        row = json.loads(raw); expected = sign*sum(p.steps[:index+1])/factor
        assert abs(row['parameter']-expected) <= 1e-11
        mech = row['mechanical']
        displacement = np.array(mech['positions'])-c.physical.reference_positions+np.array(mech['position_low'])
        exact = np.zeros((3, 3)); exact[:, 0] = expected*np.arange(3)/1000.
        assert np.linalg.norm(displacement-exact) <= 1e-11
        assert max(*row['metrics'], abs(row['arc_gap']), row['correction']) <= 1e-11
        assert row['orientation'] > 0.
        assert abs(np.sum(c.metric*np.array(row['predictor'])**2)-1.) <= 1e-11
    assert state.completed_steps == 3 and not result.production_qualified
    before = canonical(state); recovery = c.recover(state)
    assert canonical(state) == before and c.checkpoint(records) == result.checkpoint
    save(tmp_path/'analytical.json', dict(sign=sign, recovery=recovery, restart_identical=True,
        actual_arc_controller=True, production_qualified=False))


def test_curved_transverse_spatial_row_and_restart(tmp_path):
    model, _, _ = make(True, 1, clamped=True); p = program(transverse=True)
    result = arc.solve(model, p); save(tmp_path/'curved.json', result.checkpoint)
    assert result.status == 'completed', result.failure
    c = arc.Context(model, p); state, records = restore(c, result.checkpoint)
    accepted, _ = restore(c, c.checkpoint(records[:1])); predictor = c.predictor(accepted)
    current = c.physical.make(json.loads(records[1])['mechanical'], decoded=True)
    parameter = json.loads(records[1])['parameter']
    gap, row = c.row(current, parameter, accepted, predictor)
    v = np.sin(np.arange(c.physical.count)+.4)*.1; v[list(c.physical.fixed)] = 0.
    h = 2e-6; dp = .3
    plus, _ = c.row(c.physical.advance(current, h*v), float(parameter+h*dp), accepted, predictor)
    minus, _ = c.row(c.physical.advance(current, -h*v), float(parameter-h*dp), accepted, predictor)
    error = abs((plus-minus)/(2*h)-row@np.r_[v, dp])
    assert error <= 1e-7 and abs(gap) <= 1e-11
    partial = arc.solve(model, p, stop_after=1)
    resumed = arc.solve(model, p, checkpoint=partial.checkpoint, expected_sha256=sha256(partial.checkpoint).hexdigest())
    assert resumed.status == 'completed' and resumed.checkpoint == result.checkpoint
    assert np.linalg.norm(state.mechanical.nodal_frames-c.physical.reference_frames) > 1e-5
    save(tmp_path/'row.json', dict(error=error, current_spatial_derivative=True, restart_identical=True,
        production_qualified=False, beam_postbuckling_qualified=False))


@pytest.fixture(scope='module')
def capsule():
    model, _, _ = make(False, 1, clamped=True, coupled=False); p = program((.01,))
    result = arc.solve(model, p)
    assert result.status == 'completed', result.failure
    return result.checkpoint


@pytest.mark.parametrize('kind', ('predictor', 'orientation', 'step_size', 'parameter', 'history', 'work',
    'predecessor', 'correction', 'recovery', 'cursor_bool', 'duplicate', 'nonfinite', 'whitespace', 'hash', 'schema'))
def test_resealed_mutations(capsule, kind):
    model, _, _ = make(False, 1, clamped=True, coupled=False); c = arc.Context(model, program((.01,)))
    value = json.loads(capsule); row = value['records'][0]
    if kind == 'predictor': row['predictor'][-1] *= -1.
    elif kind == 'orientation': row['orientation'] *= -1.
    elif kind == 'step_size': row['step_size'] *= 2.
    elif kind == 'parameter': row['parameter'] += .1
    elif kind == 'history': row['histories'] = []
    elif kind == 'work': row['work'][-1] += .1
    elif kind == 'predecessor': row['previous_sha256'] = '0'*64
    elif kind == 'correction': row['correction'] += .1
    elif kind == 'recovery': row['recovery_sha256'] = '0'*64
    elif kind == 'cursor_bool': value['completed_steps'] = True
    elif kind == 'schema': value['schema'] = 'GE_BEAM3_RETAINED_GENERALIZED_TRANSLATION_ACCEPTED_CHAIN_V1'
    row['record_sha256'] = sha({k: v for k, v in row.items() if k != 'record_sha256'})
    value['checkpoint_sha256'] = sha({k: v for k, v in value.items() if k != 'checkpoint_sha256'})
    raw = canonical(value)
    if kind == 'duplicate': raw = raw.replace(b'{', b'{"schema":"duplicate",', 1)
    elif kind == 'nonfinite': raw = raw.replace(b'"parameter":0.0', b'"parameter":NaN')
    elif kind == 'whitespace': raw = b' '+raw
    with pytest.raises(ValueError):
        c.restore(raw, expected_sha256='0'*64 if kind == 'hash' else sha256(raw).hexdigest())


@pytest.mark.parametrize('kind', ('clone', 'foreign', 'changed', 'record', 'metric', 'program', 'correction'))
def test_state_ownership_and_acceptance(kind, monkeypatch):
    model, _, _ = make(False, 1, clamped=True, coupled=False); p = program((.01,)); c = arc.Context(model, p)
    if kind == 'clone':
        with pytest.raises(ValueError, match='not issued'): c.recover(replace(c.initial))
    elif kind == 'foreign':
        other = arc.Context(model, p)
        with pytest.raises(ValueError, match='not issued'): c.recover(other.initial)
    elif kind == 'changed':
        object.__setattr__(c.initial, 'parameter', 1.)
        with pytest.raises(ValueError, match='changed'): c.recover(c.initial)
    elif kind == 'record':
        with pytest.raises(ValueError, match='not issued'): c.checkpoint((b'{}',))
    elif kind == 'metric':
        c.metric = c.metric.copy(); c.metric[0] *= 2.
        with pytest.raises(ValueError, match='maps changed'): c.checkpoint(())
    elif kind == 'program':
        object.__setattr__(p, 'steps', (.02,))
        with pytest.raises(ValueError, match='maps changed'): c.checkpoint(())
    else:
        original = c.correction
        def bad(*args):
            a, b, _, d = original(*args); return a, b, 1., d
        monkeypatch.setattr(c, 'correction', bad)
        predictor = c.predictor(c.initial)
        mech = c.physical.advance(c.initial.mechanical, .01*predictor[:-1])
        with pytest.raises(ValueError, match='fully converged'):
            c.stage(mech, float(.01*predictor[-1]), c.initial, (), 0)


@pytest.mark.parametrize('stage', ('before_predictor', 'before_assembly', 'before_factorization',
    'before_trial', 'before_commit', 'committed'))
def test_cancellation_rollback(stage, tmp_path):
    model, _, _ = make(True, 1, clamped=True); p = program(transverse=True)
    token = CancellationToken()
    def progress(row):
        if row['stage'] == 'retained-arc.'+stage: token.cancel()
    result = arc.solve(model, p, cancellation_token=token, progress=progress)
    assert result.status == 'cancelled', result.failure
    assert result.completed_steps == (1 if stage == 'committed' else 0)
    c = arc.Context(model, p); state, records = restore(c, result.checkpoint)
    assert state.completed_steps == result.completed_steps
    save(tmp_path/'rollback.json', dict(stage=stage, cursor=state.completed_steps,
        checkpoint_sha256=sha256(result.checkpoint).hexdigest(), status=result.status, production_qualified=False))


@pytest.mark.parametrize('change', ({'steps': []}, {'steps': (True,)}, {'steps': (0.,)},
    {'steps': (.3,)}, {'steps': (float('nan'),)}, {'length_scale': 0.}, {'length_scale': 1e-300},
    {'parameter_scale': float('inf')}, {'initial_sign': 0.}, {'initial_sign': True},
    {'max_iterations': True}, {'max_backtracks': 9}))
def test_program_schema(change):
    with pytest.raises(ValueError): replace(program(), **change)
