"""Small controlled fibre paths, not mesh-converged post-buckling qualification."""
from hashlib import sha256
import json
import numpy as np
import pytest

import anyfesolver._ge_beam3_retained_fibre_control as control
from anyfesolver._ge_beam3_retained_fibre_program import solve_force_program
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_retained_fibre import make_model
from test_ge_beam3_retained_fibre_state import program as force_program


def program():
    return control.TranslationProgram((.01, .02, 0., -.01), 5, (1., 0., 0.), ((5, .4, -.005, 0.),))


def test_full_border_handles_a_simple_tangent_limit_point():
    h = np.diag([0., 2.]); column = np.array([-1., 0.]); row = np.array([1., 0.])
    matrix = control.bordered(h, column, row, [0, 1])
    np.testing.assert_array_equal(np.linalg.solve(matrix, [0., 0., .1]), [.1, 0., 0.])
    assert np.linalg.matrix_rank(h) == 1 and np.linalg.matrix_rank(matrix) == 3


def test_force_path_and_translation_path_agree(tmp_path):
    p = force_program(); model = make_model(); fixed = solve_force_program(model, p, stop_after=3)
    assert fixed.status == 'paused', fixed.failure
    data = json.loads(fixed.checkpoint); initial = model.mesh.nodes[5].coords()[0]
    targets = tuple(float(r['mechanical']['positions'][-1][0]+r['mechanical']['position_low'][-1][0]-initial) for r in data['records'])
    translated = control.TranslationProgram(targets, 5, (1., 0., 0.), p.nodal_forces)
    result = control.solve_translation_program(make_model(), translated)
    (tmp_path/'force.json').write_bytes(fixed.checkpoint); (tmp_path/'controlled.json').write_bytes(result.checkpoint)
    assert result.status == 'completed', result.failure
    records = json.loads(result.checkpoint)['records']
    for record, expected in zip(records, p.targets[:3]): assert abs(record['parameter']-expected) <= 1e-9
    np.testing.assert_allclose(result.state.mechanical.positions, fixed.state.mechanical.positions, atol=1e-10, rtol=0.)
    assert not result.production_qualified


@pytest.mark.parametrize('contrast', [1., 1.e12])
def test_physical_fibre_reversal_and_restart_are_transactional(contrast, tmp_path):
    p = program(); whole = control.solve_translation_program(make_model(contrast), p)
    (tmp_path/'whole.json').write_bytes(whole.checkpoint)
    assert whole.status == 'completed', whole.failure
    paused = control.solve_translation_program(make_model(contrast), p, stop_after=2)
    assert paused.status == 'paused', paused.failure
    resumed = control.solve_translation_program(make_model(contrast), p, checkpoint=paused.checkpoint,
        expected_checkpoint_sha256=sha256(paused.checkpoint).hexdigest())
    (tmp_path/'paused.json').write_bytes(paused.checkpoint)
    assert resumed.status == 'completed' and resumed.checkpoint == whole.checkpoint
    assert any(row[2] > 0 for cell in whole.state.histories for station in cell.stations for row in station.rows)
    context = control.Context(make_model(contrast), p); replayed, _ = context.restore(whole.checkpoint)
    (tmp_path/'recovery.json').write_bytes(canonical(context.recover(replayed)))
    records = json.loads(whole.checkpoint)['records']
    for record, target in zip(records, p.targets):
        assert abs(record['control_value']-target) <= 1e-11 and max(record['metrics']) <= 1e-11


def arch():
    from anyfesolver.fe_core import FEModel
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
    from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection, Fibre, FlowCurve
    model = FEModel('small-physical-fibre-shallow-arch'); frames = []
    for node, x in enumerate(np.linspace(-1., 1., 5), 1):
        model.add_node(node, float(x), float(.1*(1-x*x)), 0.)
        tangent = np.array([1., -.2*x, 0.]); tangent /= np.linalg.norm(tangent)
        second = np.array([0., 0., 1.]); frames.append(np.column_stack((tangent, second, np.cross(tangent, second))))
    fibres = tuple(Fibre(str(i), y, z, .25, 1.e6, FlowCurve.linear(1.e6, 100.))
        for i, (y, z) in enumerate([(-.01, -.01), (-.01, .01), (.01, -.01), (.01, .01)]))
    factor = np.zeros((3, 6)); factor[0, 1] = np.sqrt(4.e5); factor[1, 2] = np.sqrt(4.e5); factor[2, 3] = np.sqrt(80.)
    section = PhysicalFibreSection(fibres, factor)
    for eid, nodes in enumerate(((1, 2, 3), (3, 4, 5)), 1):
        ref = Reference(np.array([model.mesh.nodes[n].coords() for n in nodes]), np.array([frames[n-1] for n in nodes]))
        element = NativeRetainedFibreElement(eid, nodes, ref, section, order=4)
        model.add_element(eid, element); model.materials[element.material_name] = section
    model.add_boundary_condition(BoundaryCondition('ends', [1, 5], {k: 0. for k in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    return model


def test_small_curved_arch_path_crosses_a_load_maximum(tmp_path):
    p = control.TranslationProgram((.01, .025, .04, .055, .075, .1, .15, .2), 3, (0., -1., 0.), ((3, 0., -1., 0.),))
    events = []; result = control.solve_translation_program(arch(), p, progress=events.append)
    (tmp_path/'arch.json').write_bytes(result.checkpoint)
    (tmp_path/'progress.json').write_bytes(canonical(dict(status=result.status, failure=result.failure, events=events)))
    assert result.status == 'completed', result.failure
    rows = json.loads(result.checkpoint)['records']; loads = [r['parameter'] for r in rows]
    assert loads[1] > loads[0]
    assert any(a < b > c for a, b, c in zip(loads, loads[1:], loads[2:]))
    assert any(a > b < c for a, b, c in zip(loads, loads[1:], loads[2:]))
    assert max(max(r['metrics']) for r in rows) <= 1e-11
    assert not result.production_qualified


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    p = program(); result = control.solve_translation_program(make_model(), p, stop_after=2)
    assert result.status == 'paused', result.failure
    (tmp_path_factory.mktemp('accepted-control')/'paused.json').write_bytes(result.checkpoint)
    return p, result


def rehash(value):
    previous = value['initial']['record_sha256']
    for row in value['records']:
        row['previous_sha256'] = previous; row['record_sha256'] = sha({k: v for k, v in row.items() if k != 'record_sha256'})
        previous = row['record_sha256']
    value['checkpoint_sha256'] = sha({k: v for k, v in value.items() if k != 'checkpoint_sha256'})
    return canonical(value)


@pytest.mark.parametrize('incident', ['parameter', 'parameter_bool', 'plane', 'history', 'origin', 'metric', 'control_value',
    'recovery', 'iterations_bool', 'target', 'direction', 'schema', 'extra'])
def test_resealed_control_state_mutations_fail(accepted, incident):
    p, result = accepted; data = json.loads(result.checkpoint); row = data['records'][-1]
    if incident == 'parameter': row['parameter'] += .01
    elif incident == 'parameter_bool': row['parameter'] = True
    elif incident == 'plane': row['mechanical']['positions'][-1][0] += .01
    elif incident == 'history': row['histories'][0]['stations'][0]['rows'][0][2] += .01
    elif incident == 'origin': row['origins'][0]['stations'][0]['rows'][0][0] += .01
    elif incident == 'metric': row['metrics'][0] += .01
    elif incident == 'control_value': row['control_value'] += .01
    elif incident == 'recovery': row['recovery_sha256'] = '0'*64
    elif incident == 'iterations_bool': row['iterations'] = True
    elif incident == 'target': row['displacement_target'] += .01
    elif incident == 'direction': data['program']['direction'] = [0., 1., 0.]
    elif incident == 'schema': data['schema'] = 'GE_BEAM3_RETAINED_PHYSICAL_FIBRE_ACCEPTED_CHAIN_V1'
    else: row['extra'] = 1
    with pytest.raises(ValueError): control.Context(make_model(), p).restore(rehash(data))


@pytest.mark.parametrize('stage', ['fibre-control.before_assembly', 'fibre-control.before_factorization',
    'fibre-control.before_trial', 'fibre-control.before_commit'])
def test_cancel_before_commit_preserves_full_accepted_history(accepted, stage):
    p, result = accepted; token = CancellationToken()
    def observe(event):
        if event['stage'] == stage: token.cancel()
    cancelled = control.solve_translation_program(make_model(), p, checkpoint=result.checkpoint,
        cancellation_token=token, progress=observe)
    assert cancelled.status == 'cancelled' and cancelled.checkpoint == result.checkpoint


def test_cancel_after_commit_preserves_new_state(accepted):
    p, result = accepted; token = CancellationToken()
    def observe(event):
        if event['stage'] == 'fibre-control.committed': token.cancel()
    stopped = control.solve_translation_program(make_model(), p, checkpoint=result.checkpoint,
        cancellation_token=token, progress=observe)
    expected = control.solve_translation_program(make_model(), p, checkpoint=result.checkpoint, stop_after=3)
    assert stopped.status == 'cancelled' and stopped.completed_targets == 3
    assert stopped.checkpoint == expected.checkpoint


@pytest.mark.parametrize('incident', ['observer', 'model'])
def test_failed_publication_preserves_history(accepted, incident):
    p, result = accepted; model = make_model(); x = model.mesh.nodes[5].x
    def observe(event):
        if event['stage'] == 'fibre-control.before_commit':
            if incident == 'observer': raise RuntimeError('injected observer failure')
            model.mesh.nodes[5].x += .01
    try:
        failed = control.solve_translation_program(model, p, checkpoint=result.checkpoint, progress=observe)
        assert failed.status == 'failed' and failed.checkpoint == result.checkpoint
    finally: model.mesh.nodes[5].x = x


def test_replay_does_not_advance_or_solve_newton(accepted, monkeypatch):
    p, result = accepted; context = control.Context(make_model(), p)
    def forbidden(*args, **kwargs): raise AssertionError('replay must not advance geometry')
    monkeypatch.setattr(type(context.layout), 'advance', forbidden)
    state, records = context.restore(result.checkpoint)
    assert state.completed_targets == 2 and context.checkpoint(records) == result.checkpoint


def test_strict_input_and_external_hash_guards(accepted):
    p, result = accepted; context = control.Context(make_model(), p)
    for raw in (b' '+result.checkpoint, b'{"x":1,"x":2}', b'{"x":NaN}', bytearray(result.checkpoint)):
        with pytest.raises(ValueError): context.restore(raw)
    with pytest.raises(ValueError): context.restore(result.checkpoint, expected_sha256='0'*64)
    with pytest.raises(ValueError): control.solve_translation_program(make_model(), p, checkpoint=result.checkpoint, stop_after=1)
    for node, direction in ((True, (1., 0., 0.)), (0, (1., 0., 0.)), (5, (2., 0., 0.)), (5, (True, 0., 0.)), (5, (float('nan'), 0., 0.))):
        with pytest.raises(ValueError): control.TranslationProgram((.01,), node, direction, p.nodal_forces)
    with pytest.raises(ValueError): control.Context(make_model(), control.TranslationProgram((.01,), 1, (1., 0., 0.), p.nodal_forces))
    with pytest.raises(ValueError): control.Context(make_model(), control.TranslationProgram((.01,), 5, (1., 0., 0.), ()))


def test_iteration_failure_preserves_virgin_state():
    p = control.TranslationProgram((.01,), 5, (1., 0., 0.), ((5, .4, -.005, 0.),), max_iterations=0)
    before = control.solve_translation_program(make_model(), p, stop_after=0)
    result = control.solve_translation_program(make_model(), p)
    assert result.status == 'failed' and result.checkpoint == before.checkpoint


def transformed_model(transform, shift=None):
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    from anyfesolver._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
    model = make_model(); shift = np.zeros(3) if shift is None else shift
    for node in model.mesh.nodes.values(): node.x, node.y, node.z = map(float, transform@node.coords()+shift)
    for eid, element in list(model.mesh.elements.items()):
        ref = element.operator.reference
        rotated = Reference(np.array([model.mesh.nodes[n].coords() for n in element.node_ids]),
            np.einsum('ij,njk->nik', transform, ref.nodal_triads))
        made = NativeRetainedFibreElement(eid, tuple(element.node_ids), rotated, element.section, order=4)
        model.mesh.elements[eid] = made; model.materials[made.material_name] = made.section
    return model


def test_directional_control_under_large_common_rotation(accepted, tmp_path):
    from anyfesolver._ge_beam3_p5.algebra import rotation
    p, result = accepted; transform = rotation([2.6, .8, -.3])
    direction = tuple(map(float, transform@np.array(p.direction)))
    force = tuple(map(float, transform@np.array(p.nodal_forces[0][1:])))
    rotated = control.TranslationProgram(p.targets, 5, direction, ((5, *force),))
    other = control.solve_translation_program(transformed_model(transform), rotated, stop_after=2)
    (tmp_path/'rotated.json').write_bytes(other.checkpoint)
    assert other.status == 'paused', other.failure
    a = json.loads(result.checkpoint)['records']; b = json.loads(other.checkpoint)['records']
    error = max(abs(x['parameter']-y['parameter'])/max(1., abs(x['parameter'])) for x, y in zip(a, b))
    (tmp_path/'covariance.json').write_bytes(canonical(dict(relative_parameter_error=error)))
    assert error <= 1e-11
    np.testing.assert_allclose(other.state.mechanical.positions, result.state.mechanical.positions@transform.T, atol=1e-11, rtol=0.)


def test_small_control_is_preserved_under_large_common_translation():
    p = program(); context = control.Context(transformed_model(np.eye(3), np.array([2.**30, -2.**30, 2.**30])), p)
    made = context.project(context.initial.mechanical, 1e-9)
    assert context.value(made) == 1e-9
    assert made.positions[-1, 0] == context.layout.reference_positions[-1, 0]
    assert made.position_low[-1, 0] == 1e-9


@pytest.mark.parametrize('incident', ['parameter', 'history', 'cursor'])
def test_staging_rejects_an_accepted_state_detached_from_its_record(accepted, incident):
    from dataclasses import replace
    p, result = accepted; context = control.Context(make_model(), p); state, records = context.restore(result.checkpoint)
    if incident == 'parameter': changed = replace(state, parameter=state.parameter+.01)
    elif incident == 'history': changed = replace(state, histories=context.initial.histories)
    else: changed = replace(state, completed_targets=True)
    with pytest.raises(ValueError): context.stage(state.mechanical, state.parameter, changed, records, 0)
