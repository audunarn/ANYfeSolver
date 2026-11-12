"""Small native force-program correctness checks; no qualification claims."""

import json

import numpy as np
import pytest

import anyfesolver._ge_beam3_seeded_load_program as driver
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from test_ge_beam3_native_load_state import problem as legacy_problem, store_for, equilibrium
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement


def problem(*args, **kwargs):
    model = legacy_problem(*args, **kwargs)
    # Fresh candidates only: no old accepted state is migrated.
    for i, old in tuple(model.mesh.elements.items()):
        new = NativeP5BeamElement(i, old.node_ids, old.core.reference, old.core.section,
            line_force=old.core.line_force, order=old.core.order)
        model.mesh.elements[i] = new; model.materials[new.material_name] = new.core.section
    return model


def assert_done(result, count):
    assert result.status == 'completed', result.failure
    assert result.completed_targets == count
    assert not result.production_qualified
    assert np.linalg.norm(result.physical_imbalance[6:]) <= 1e-11
    assert not result.displacements.flags.writeable
    assert not result.physical_imbalance.flags.writeable


def reseal(value):
    body = {k:v for k,v in value.items() if k != 'checkpoint_sha256'}
    return canonical({**body,'checkpoint_sha256':sha(body)})


def test_elastic_program_matches_separate_assembly_loop():
    model = problem(); program = driver.ForceProgram((.5,1.))
    result = driver.solve_force_program(model,program)
    assert_done(result,2)
    reference = legacy_problem(); store = store_for(reference); total = np.zeros(18)
    for p in program.targets: total,_ = equilibrium(reference,store,total,p)
    np.testing.assert_allclose(result.displacements,total,rtol=0.,atol=1e-11)
    assert np.linalg.norm(result.physical_imbalance[:6]) > .01


def test_plastic_split_restart_is_byte_identical():
    program = driver.ForceProgram((.5,1.,0.,-.5))
    full = driver.solve_force_program(problem(plastic=True),program)
    assert_done(full,4)
    paused = driver.solve_force_program(problem(plastic=True),program,stop_after=2)
    assert paused.status == 'paused' and paused.completed_targets == 2
    resumed = driver.solve_force_program(problem(plastic=True),program,checkpoint=paused.checkpoint)
    assert_done(resumed,4)
    assert resumed.checkpoint == full.checkpoint
    np.testing.assert_array_equal(resumed.displacements,full.displacements)


def test_nodal_force_is_not_duplicated_with_line_work():
    program = driver.ForceProgram((1.,),((55,.025,-.01,.005),))
    result = driver.solve_force_program(problem(line_force=(0.,0.,0.)),program)
    assert_done(result,1)
    np.testing.assert_allclose(result.physical_imbalance[:3],[-.025,.01,-.005],atol=1e-11,rtol=0.)


@pytest.mark.parametrize('stage',['native_force.before_commit','native_force.committed'])
def test_cancellation_returns_last_accepted_capsule(stage):
    program = driver.ForceProgram((.5,1.)); token = CancellationToken()
    expected = 0 if stage.endswith('before_commit') else 1
    def progress(event):
        if event['stage'] == stage: token.cancel('fixture cancellation')
    result = driver.solve_force_program(problem(),program,cancellation_token=token,progress=progress)
    assert result.status == 'cancelled' and result.completed_targets == expected
    retained = driver.solve_force_program(problem(),program,checkpoint=result.checkpoint,stop_after=expected)
    assert retained.checkpoint == result.checkpoint
    resumed = driver.solve_force_program(problem(),program,checkpoint=result.checkpoint)
    assert_done(resumed,2)


def test_failed_iteration_bound_preserves_virgin_state():
    program = driver.ForceProgram((1.,),max_iterations=0)
    virgin = driver.solve_force_program(problem(),program,stop_after=0)
    failed = driver.solve_force_program(problem(),program)
    assert failed.status == 'failed' and 'iteration bound' in failed.failure
    assert failed.completed_targets == 0 and failed.checkpoint == virgin.checkpoint


def test_serialization_failure_does_not_commit(monkeypatch):
    program = driver.ForceProgram((.5,)); original = driver._checkpoint
    virgin = driver.solve_force_program(problem(),program,stop_after=0)
    def reject(*args):
        if args[-2]: raise ValueError('capsule staging fixture')
        return original(*args)
    monkeypatch.setattr(driver,'_checkpoint',reject)
    result = driver.solve_force_program(problem(),program)
    assert result.status == 'failed' and 'staging fixture' in result.failure
    assert result.completed_targets == 0 and result.checkpoint == virgin.checkpoint


@pytest.fixture(scope='module')
def accepted():
    program = driver.ForceProgram((.5,))
    result = driver.solve_force_program(problem(),program)
    assert_done(result,1)
    return program,result.checkpoint


@pytest.mark.parametrize('field',['reaction','displacement','epoch','record','program','hash'])
def test_resealed_global_mutation_rejected(accepted,field):
    program,raw = accepted; value = json.loads(raw)
    if field == 'reaction': value['physical_imbalance'][0] += .01
    elif field == 'displacement': value['total_displacement'][0] = .01
    elif field == 'epoch': value['completed_targets'] = 0
    elif field == 'record': value['records'][0]['target'] = 2
    elif field == 'program': value['program']['targets'] = [.75]
    elif field == 'hash': value['model_sha256'] = '0'*64
    with pytest.raises(ValueError):
        driver.solve_force_program(problem(),program,checkpoint=reseal(value))


@pytest.mark.parametrize('kind',['duplicate','nonfinite','whitespace','missing','oversize'])
def test_noncanonical_global_checkpoint_rejected(accepted,kind):
    program,raw = accepted
    if kind == 'duplicate': raw = raw[:-1]+b',"schema":"duplicate"}'
    elif kind == 'nonfinite': raw = b'{"bad":NaN}'
    elif kind == 'whitespace': raw += b' '
    elif kind == 'missing': raw = b'{}'
    elif kind == 'oversize': raw = b' '*((2*1024*1024)+1)
    with pytest.raises(ValueError): driver.solve_force_program(problem(),program,checkpoint=raw)


@pytest.mark.parametrize('kwargs',[
    {'targets':()}, {'targets':(1,)}, {'targets':(True,)}, {'targets':(float('nan'),)},
    {'targets':(1.,),'max_iterations':33}, {'targets':(1.,),'max_backtracks':True},
    {'targets':(1.,),'nodal_forces':((55,1.,0.,0.),(55,0.,0.,0.))},
])
def test_invalid_program_rejected(kwargs):
    with pytest.raises(ValueError): driver.ForceProgram(**kwargs)


def test_observer_model_mutation_is_rejected_before_evaluation():
    model = problem()
    def mutate(event): model.boundary_conditions[0].name = 'changed'
    result = driver.solve_force_program(model,driver.ForceProgram((.5,)),progress=mutate)
    assert result.status == 'failed' and 'inputs changed' in result.failure
    assert result.completed_targets == 0


def test_precancel_rejects_without_assembly(monkeypatch):
    token = CancellationToken(); token.cancel()
    def forbidden(*args,**kwargs): raise AssertionError('assembly must not run')
    monkeypatch.setattr(driver,'_assemble_nonlinear_system',forbidden)
    with pytest.raises(SolveCancelled):
        driver.solve_force_program(problem(),driver.ForceProgram((1.,)),cancellation_token=token)


def test_shared_node_large_offset_program_and_restart():
    program = driver.ForceProgram((.5,1.))
    origin = driver.solve_force_program(problem(count=2),program)
    translated = driver.solve_force_program(problem(count=2,shift=2.**40),program)
    assert_done(origin,2); assert_done(translated,2)
    np.testing.assert_array_equal(origin.displacements,translated.displacements)
    np.testing.assert_array_equal(origin.physical_imbalance,translated.physical_imbalance)
    replay = driver.solve_force_program(problem(count=2,shift=2.**40),program,checkpoint=translated.checkpoint)
    assert replay.checkpoint == translated.checkpoint
    with pytest.raises(ValueError,match='model/program/hash'):
        driver.solve_force_program(problem(count=2),program,checkpoint=translated.checkpoint)


def test_zero_force_schedule_commits_explicit_epochs():
    program = driver.ForceProgram((1.,0.,-1.))
    result = driver.solve_force_program(problem(line_force=(0.,0.,0.)),program)
    assert_done(result,3)
    np.testing.assert_array_equal(result.displacements,np.zeros(18))
    value = json.loads(result.checkpoint)
    assert [r['parameter'] for r in value['records']] == [1.,0.,-1.]
    assert all(r['iterations'] == 0 for r in value['records'])


@pytest.mark.parametrize('kind',['unsupported','partial_rotation','nonzero','wrong_element','unknown_load_node'])
def test_unqualified_model_routes_fail_before_assembly(monkeypatch,kind):
    model = problem(); program = driver.ForceProgram((1.,))
    if kind == 'unsupported': model.boundary_conditions.clear()
    elif kind == 'partial_rotation': del model.boundary_conditions[0].dof_constraints['rz']
    elif kind == 'nonzero': model.boundary_conditions[0].dof_constraints['ux'] = .001
    elif kind == 'wrong_element':
        from test_ge_beam3_centered_native import problem as old_problem
        model,_ = old_problem()
    elif kind == 'unknown_load_node': program = driver.ForceProgram((1.,),((999,.1,0.,0.),))
    def forbidden(*args,**kwargs): raise AssertionError('assembly must not run')
    monkeypatch.setattr(driver,'_assemble_nonlinear_system',forbidden)
    with pytest.raises(ValueError): driver.solve_force_program(model,program)


def test_direct_dof_support_mutation_is_detected():
    model = problem()
    def mutate(event): model.mesh.dof_manager.constrain_dof(7)
    result = driver.solve_force_program(model,driver.ForceProgram((1.,)),progress=mutate)
    assert result.status == 'failed' and 'inputs changed' in result.failure
    assert result.completed_targets == 0


def test_cooperative_deadline_preserves_checkpoint_without_retry(monkeypatch):
    from types import SimpleNamespace
    program = driver.ForceProgram((1.,))
    virgin = driver.solve_force_program(problem(),program,stop_after=0)
    clocks = iter((0.,601.))
    monkeypatch.setattr(driver,'time',SimpleNamespace(monotonic=lambda: next(clocks)))
    def forbidden(*args,**kwargs): raise AssertionError('assembly must not run')
    monkeypatch.setattr(driver,'_assemble_nonlinear_system',forbidden)
    result = driver.solve_force_program(problem(),program)
    assert result.status == 'failed' and 'deadline exceeded' in result.failure
    assert result.checkpoint == virgin.checkpoint and result.completed_targets == 0


def test_failure_after_commit_retains_accepted_history_without_retry():
    program = driver.ForceProgram((.5,1.)); attempts = []
    paused = driver.solve_force_program(problem(),program,stop_after=1)
    def fail(event):
        if event['stage'] == 'native_force.before_assembly' and event['target'] == 2:
            attempts.append(2)
            raise RuntimeError('second target fixture failure')
    result = driver.solve_force_program(problem(),program,progress=fail)
    assert result.status == 'failed' and 'second target fixture' in result.failure
    assert result.completed_targets == 1 and result.checkpoint == paused.checkpoint
    assert attempts == [2]


def test_restart_cannot_change_program_or_rewind(accepted):
    program,raw = accepted
    with pytest.raises(ValueError,match='model/program/hash'):
        driver.solve_force_program(problem(),driver.ForceProgram((.5,1.)),checkpoint=raw)
    with pytest.raises(ValueError,match='cannot rewind'):
        driver.solve_force_program(problem(),program,checkpoint=raw,stop_after=0)


def test_all_fixed_model_needs_no_factorization(monkeypatch):
    model = problem(); model.boundary_conditions[0].node_ids = list(model.mesh.nodes)
    def forbidden(*args,**kwargs): raise AssertionError('factorization must not run')
    monkeypatch.setattr(driver,'factorize',forbidden)
    result = driver.solve_force_program(model,driver.ForceProgram((1.,)))
    assert result.status == 'completed', result.failure
    assert result.completed_targets == 1
    np.testing.assert_array_equal(result.displacements,np.zeros(18))
    assert np.linalg.norm(result.physical_imbalance) > .01
