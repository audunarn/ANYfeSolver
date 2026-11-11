"""Private controlled retained state gates; not postbuckling qualification."""
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver import _ge_beam3_retained_translation_control as control
from anyfesolver import _ge_beam3_retained_nodal_loading as force
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_generalized_modal import make
from test_ge_beam3_schur_line_program import save


def program(targets=(.0001,), **kw):
    return control.Program(targets, 3, (1., 0., 0.), force.NodalDeadForces(((3, 1., 0., 0.),)), **kw)


def restore(context, raw):
    return context.restore(raw, expected_sha256=sha256(raw).hexdigest())


def rehash(value):
    for row in value['records']:
        row['record_sha256'] = sha({k: v for k, v in row.items() if k != 'record_sha256'})
    value['checkpoint_sha256'] = sha({k: v for k, v in value.items() if k != 'checkpoint_sha256'})
    return canonical(value)


@pytest.mark.parametrize('sign', (-1., 1.))
def test_analytical_load_unload_reverse_restart(sign, tmp_path):
    m, _, _ = make(False, 1, clamped=True, coupled=False)
    p = program(tuple(sign*v for v in (.0001, .0002, 0., -.0001)))
    result = control.solve(m, p); assert result.status == 'completed', result.failure
    partial = control.solve(m, p, stop_after=2); assert partial.status == 'paused', partial.failure
    resumed = control.solve(m, p, checkpoint=partial.checkpoint, expected_sha256=sha256(partial.checkpoint).hexdigest())
    assert resumed.checkpoint == result.checkpoint
    context = control.Context(m, p); state, records = restore(context, result.checkpoint)
    reference = context.physical.reference_positions
    for raw in records:
        row = json.loads(raw); target = row['displacement_target']; expected_load = 1000.*target/2.
        displacement = np.array(row['mechanical']['positions'])-reference+np.array(row['mechanical']['position_low'])
        expected = np.zeros((3, 3)); expected[:, 0] = target*(reference[:, 0]-reference[0, 0])/2.
        assert np.linalg.norm(displacement-expected) <= 1e-11
        assert abs(row['parameter']-expected_load) <= 1e-11
        assert abs(row['residual'][0]+expected_load) <= 1e-11
        assert abs(row['work'][-1]-expected_load*target) <= 1e-11
        assert max(*row['metrics'], row['correction']) <= 1e-11
    assert state.parameter == json.loads(records[-1])['parameter']
    assert state.parameter != p.targets[-1]
    assert canonical(context.recover(state)) == canonical(context.recover(state))
    assert context.checkpoint(records) == result.checkpoint and not result.production_qualified
    save(tmp_path/'controlled.json', result.checkpoint)
    save(tmp_path/'analytical.json', dict(sign=sign, analytical_EA=1000., length=2., restart_identical=True,
        production_qualified=False, actual_controlled_programme_run=True))


@pytest.mark.parametrize('macros,curved', ((1, True), (2, True)))
def test_curved_coupled_force_port(macros, curved, tmp_path):
    m, _, _ = make(curved, macros, clamped=True)
    loads = force.NodalDeadForces(((3, .005, -.003, .002),))
    fp = force.Program((.5, 1.), DistributedPattern(LinePattern(()), ()), loads)
    original = force.solve(m, fp); assert original.status == 'completed', original.failure
    direction = tuple(float(v) for v in np.array(loads.rows[0][1:])/np.linalg.norm(loads.rows[0][1:]))
    reference = m.mesh.nodes[3].coords()
    targets = []; polishing = []
    # A force-converged state can have a displacement error below 1e-11
    # but a larger inferred load-factor error after dividing by compliance.
    # Refine the COMPARISON state at its SAME load and material origin twice;
    # never alter/reissue the accepted force capsule or relax the port gate.
    physical = force.Context(m, fp)
    for row in json.loads(original.checkpoint)['records']:
        state = physical.make(row['mechanical'], decoded=True)
        origins = physical.histories(row['origins'], decoded=True); diagnostics = []
        for index in range(3):
            residual, jacobian, metrics, _, _ = physical.assemble(state, row['parameter'], origins)
            increment, remaining = physical.step(state, residual, jacobian)
            diagnostics.append(dict(index=index, metrics=metrics, remaining=remaining))
            if index < 2: state = physical.advance(state, increment)
        assert max(*metrics, remaining) <= 1e-14
        polishing.append(dict(target=row['target'], fixed_parameter=row['parameter'], diagnostics=diagnostics,
                              unaccepted_comparison_only=True))
        positions = state.positions[2]; low = state.position_low[2]
        targets.append(float(sum((Fraction(a)*(Fraction(x)+Fraction(y)-Fraction(float(z)))
            for a, x, y, z in zip(direction, positions, low, reference)), Fraction(0))))
    p = control.Program(tuple(targets), 3, direction, loads)
    result = control.solve(m, p)
    # Preserve any actually produced capsule before assertions, including a
    # failed port. These are raw outputs, never a partial canonical aggregate.
    save(tmp_path/'controlled.json', result.checkpoint)
    save(tmp_path/'force.json', original.checkpoint)
    save(tmp_path/'comparison.json', dict(targets=targets, polishing=polishing, status=result.status,
        failure=result.failure, actual_parameter=result.state.parameter, production_qualified=False))
    assert result.status == 'completed', result.failure
    assert abs(result.state.parameter-1.) <= 1e-11
    errors = {}
    for key in original.state.mechanical.__dataclass_fields__:
        a = getattr(result.state.mechanical, key); b = getattr(original.state.mechanical, key)
        errors[key] = float(np.linalg.norm(a-b))/max(1., float(np.linalg.norm(b)))
        assert errors[key] <= 1e-11
    context = control.Context(m, p); state, records = restore(context, result.checkpoint)
    assert context.checkpoint(records) == result.checkpoint
    before = canonical(state); recovery = context.recover(state); assert canonical(state) == before
    save(tmp_path/'port.json', dict(macros=macros, curved=curved, errors=errors, recovery=recovery,
        load_parameter_error=abs(state.parameter-1.), production_qualified=False))


def test_full_border_at_unbordered_fold():
    # r=x*x+lambda-1, x prescribed. At x=0, J=0 but full border is nonsingular.
    matrix = control.bordered(np.zeros((1, 1)), np.ones(1), np.ones(1), np.array([0]))
    correction = np.linalg.solve(matrix, np.array([1., 0.]))
    np.testing.assert_array_equal(correction, (0., 1.))
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(np.zeros((1, 1)), np.ones(1))


def test_exact_load_column_and_displacement_map():
    m, _, _ = make(True, 1, clamped=True)
    c = control.Context(m, program())
    r0, j0, _, _, _ = c.assemble(c.initial.mechanical, 0., c.initial.histories, 0.)
    r1, j1, _, _, _ = c.assemble(c.initial.mechanical, 1., c.initial.histories, 0.)
    np.testing.assert_array_equal(r1-r0, c.column)
    np.testing.assert_array_equal(j1, j0)
    displaced = c.project(c.initial.mechanical, .0001)
    assert abs(c.value(displaced)-.0001) <= 1e-11
    assert np.count_nonzero(c.row) == 1


@pytest.mark.parametrize('kind', ('clone', 'foreign', 'changed', 'record', 'maps', 'program', 'model', 'correction'))
def test_ownership_and_remaining_correction(kind, monkeypatch):
    m, _, _ = make(False, 1, clamped=True, coupled=False); p = program((0.,))
    c = control.Context(m, p)
    if kind == 'clone':
        with pytest.raises(ValueError, match='not issued'): c.recover(replace(c.initial))
    elif kind == 'foreign':
        other = control.Context(m, p)
        with pytest.raises(ValueError, match='not issued'): c.recover(other.initial)
    elif kind == 'changed':
        object.__setattr__(c.initial, 'parameter', 1.)
        with pytest.raises(ValueError, match='changed'): c.recover(c.initial)
    elif kind == 'record':
        with pytest.raises(ValueError, match='not issued'): c.checkpoint((b'{}',))
    elif kind == 'maps':
        c.column = c.column.copy(); c.column[12] += 1.
        with pytest.raises(ValueError, match='maps changed'): c.checkpoint(())
    elif kind == 'program':
        object.__setattr__(p, 'targets', (.1,))
        with pytest.raises(ValueError, match='program/control maps changed'): c.checkpoint(())
    elif kind == 'model':
        m.mesh.nodes[3].x += .01
        with pytest.raises(ValueError, match='changed'): c.checkpoint(())
    else:
        original = c.step
        def defective(*args):
            step, delta, _, matrix = original(*args)
            return step, delta, 1., matrix
        monkeypatch.setattr(c, 'step', defective)
        with pytest.raises(ValueError, match='fully converged'):
            c.stage(c.initial.mechanical, 0., c.initial, (), 0)


@pytest.mark.parametrize('kind', ('parameter', 'target', 'work', 'residual', 'recovery', 'history',
                                 'cursor_bool', 'duplicate', 'nonfinite', 'whitespace', 'hash', 'force_schema'))
def test_checkpoint_mutations(kind):
    m, _, _ = make(False, 1, clamped=True, coupled=False); p = program((0.,))
    c = control.Context(m, p); state, record = c.stage(c.initial.mechanical, 0., c.initial, (), 0)
    raw = c.checkpoint((record,)); value = json.loads(raw); row = value['records'][0]
    if kind == 'parameter': row['parameter'] += .1
    elif kind == 'target': row['displacement_target'] += .1
    elif kind == 'work': row['work'][-1] += .1
    elif kind == 'residual': row['residual'][0] += .1
    elif kind == 'recovery': row['recovery_sha256'] = '0'*64
    elif kind == 'history': row['origins'] = []
    elif kind == 'cursor_bool': value['completed_targets'] = True
    elif kind == 'duplicate': raw = raw.replace(b'{', b'{"schema":"duplicate",', 1)
    elif kind == 'nonfinite': raw = raw.replace(b'"parameter":0.0', b'"parameter":NaN')
    elif kind == 'whitespace': raw += b' '
    elif kind == 'force_schema': raw = c.physical.checkpoint(())
    if kind not in ('duplicate', 'nonfinite', 'whitespace', 'hash', 'force_schema'): raw = rehash(value)
    with pytest.raises((ValueError, TypeError)):
        c.restore(raw, expected_sha256='0'*64 if kind == 'hash' else sha256(raw).hexdigest())
    if kind == 'force_schema':
        controlled = c.checkpoint(())
        with pytest.raises(ValueError):
            c.physical.restore(controlled, expected_sha256=sha256(controlled).hexdigest())


@pytest.mark.parametrize('kind', ('before_commit', 'after_commit', 'failed'))
def test_cancellation_and_rollback(kind, tmp_path):
    m, _, _ = make(False, 1, clamped=True, coupled=False)
    p = program((.0001, .0002), max_iterations=1 if kind == 'failed' else 24)
    if kind == 'failed':
        # A nonlinear transverse control cannot complete in one correction.
        p = replace(p, direction=(0., 1., 0.), nodal_forces=force.NodalDeadForces(((3, 0., 1., 0.),)), targets=(.05,))
        result = control.solve(m, p)
        assert result.status == 'failed' and result.completed_targets == 0
    else:
        token = CancellationToken()
        def progress(row):
            stage = 'retained-translation.before_commit' if kind == 'before_commit' else 'retained-translation.committed'
            if row['stage'] == stage: token.cancel()
        result = control.solve(m, p, progress=progress, cancellation_token=token)
        assert result.status == 'cancelled'
        assert result.completed_targets == (0 if kind == 'before_commit' else 1)
    c = control.Context(m, p); state, records = restore(c, result.checkpoint)
    assert state.completed_targets == result.completed_targets
    if result.completed_targets == 0: assert result.checkpoint == c.checkpoint(())
    save(tmp_path/'rollback.json', dict(kind=kind, completed_targets=result.completed_targets, status=result.status,
        checkpoint_sha256=sha256(result.checkpoint).hexdigest(), production_qualified=False))


@pytest.mark.parametrize('change', ({'targets': [0.]}, {'targets': (True,)}, {'targets': (float('nan'),)},
    {'control_node': True}, {'direction': (1, 0., 0.)}, {'direction': (2., 0., 0.)},
    {'direction': (float('inf'), 0., 0.)}, {'max_iterations': True}, {'max_backtracks': 9}))
def test_program_schema(change):
    with pytest.raises(ValueError): replace(program(), **change)


@pytest.mark.parametrize('kind', ('fixed', 'unknown', 'empty'))
def test_model_control_admission(kind):
    m, _, _ = make(False, 1, clamped=True, coupled=False); p = program()
    if kind == 'fixed': p = replace(p, control_node=1)
    elif kind == 'unknown': p = replace(p, control_node=99)
    else: p = replace(p, nodal_forces=force.NodalDeadForces(()))
    with pytest.raises(ValueError): control.Context(m, p)
