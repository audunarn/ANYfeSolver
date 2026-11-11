"""Bounded cutback transactions, not engineering/post-buckling qualification."""

import json

import numpy as np
import pytest

from anyfesolver import _ge_beam3_adaptive_arc_program as driver
from anyfesolver._ge_beam3_arc_program import ArcProgram, solve_arc_program
from anyfesolver._ge_beam3_p5_loads.core import canonical, sha
from anyfesolver.control import CancellationToken
from test_ge_beam3_native_load_state import problem


def row(attempt, root=0, depth=0, part=0, step=.02, before=0,
        outcome='CUTBACK', reason='ITERATION_LIMIT', iterations=16):
    return dict(attempt=attempt, root=root, depth=depth, part=part, step_size=step,
        accepted_before=before, outcome=outcome, reason=reason, iterations=iterations)


def reseal(value):
    body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
    return canonical({**body, 'checkpoint_sha256': sha(body)})


def done(result, count):
    assert result.status == 'completed', result.failure
    assert result.completed_targets == count and not result.production_qualified
    assert not result.displacements.flags.writeable


def test_dyadic_schedule_conserves_order_and_covers_each_root():
    program = driver.AdaptiveArcProgram(ArcProgram((.02, .04), 2.))
    rows = [row(1), row(2, depth=1, part=0, step=.01, outcome='ACCEPTED', reason='EQUILIBRIUM', iterations=2),
        row(3, depth=1, part=1, step=.01, before=1, outcome='ACCEPTED', reason='EQUILIBRIUM', iterations=2),
        row(4, root=1, step=.04, before=2, outcome='ACCEPTED', reason='EQUILIBRIUM', iterations=2)]
    queue, accepted, status = driver._history(program, rows)
    assert queue == [] and accepted == (.01, .01, .04) and status == 'completed'
    assert sum(accepted) == sum(program.arc.steps)


@pytest.mark.parametrize('kind', ['order', 'size', 'depth', 'part', 'cursor', 'attempt', 'budget', 'iteration', 'reason', 'boolean'])
def test_mutated_attempts_are_rejected(kind):
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.)); made = row(1)
    if kind == 'order': made['root'] = 1
    elif kind == 'size': made['step_size'] = .01
    elif kind == 'depth': made['depth'] = 1
    elif kind == 'part': made['part'] = 1
    elif kind == 'cursor': made['accepted_before'] = 1
    elif kind == 'attempt': made['attempt'] = 2
    elif kind == 'budget': made['outcome'] = 'FAILED'
    elif kind == 'iteration': made['iterations'] = 0
    elif kind == 'reason': made['reason'] = 'ASSEMBLY_EXCEPTION'
    elif kind == 'boolean': made['attempt'] = True
    with pytest.raises(ValueError): driver._history(program, [made])


@pytest.mark.parametrize('kwargs', [dict(max_depth=0), dict(minimum_step=.02), dict(max_attempts=1), dict(max_accepted=1)])
def test_each_cutback_budget_fails_closed(kwargs):
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.), **kwargs)
    assert driver._history(program, [row(1, outcome='FAILED')])[2] == 'failed'
    with pytest.raises(ValueError): driver._history(program, [row(1)])


@pytest.mark.parametrize('kwargs', [dict(max_depth=-1), dict(max_depth=13), dict(max_attempts=True),
    dict(max_attempts=257), dict(max_accepted=65), dict(minimum_step=0.), dict(minimum_step=float('nan'))])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError): driver.AdaptiveArcProgram(ArcProgram((.02,), 2.), **kwargs)


def test_native_no_cutback_matches_frozen_fixed_step_driver():
    native = ArcProgram((.02, .02), 2.)
    fixed = solve_arc_program(problem(), native); done(fixed, 2)
    adaptive = driver.solve_adaptive_arc_program(problem(), driver.AdaptiveArcProgram(native)); done(adaptive, 2)
    assert json.loads(adaptive.checkpoint)['kernel_checkpoint'].encode('ascii') == fixed.checkpoint


def test_actual_substeps_after_injected_exhaustion_match_explicit_program(monkeypatch):
    # Failure injection tests transaction control, not a physical failure claim.
    original = driver._attempt; calls = []
    def exhaust_root(*args):
        step = args[-2]; calls.append(step)
        if step == .02: return dict(accepted=False, reason='ITERATION_LIMIT', iterations=16)
        return original(*args)
    monkeypatch.setattr(driver, '_attempt', exhaust_root)
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.))
    first = driver.solve_adaptive_arc_program(problem(), program, stop_after_attempts=1)
    assert first.status == 'paused' and first.completed_targets == 0
    assert calls == [.02]
    full = driver.solve_adaptive_arc_program(problem(), program, checkpoint=first.checkpoint); done(full, 2)
    assert calls == [.02, .01, .01]  # Consumed root was not attempted on restart.
    fixed = solve_arc_program(problem(), ArcProgram((.01, .01), 2.)); done(fixed, 2)
    assert json.loads(full.checkpoint)['kernel_checkpoint'].encode('ascii') == fixed.checkpoint
    restored = driver.solve_adaptive_arc_program(problem(), program, checkpoint=full.checkpoint)
    assert restored.checkpoint == full.checkpoint and calls == [.02, .01, .01]


def test_plastic_split_restart_preserves_origin_and_budget():
    program = driver.AdaptiveArcProgram(ArcProgram((.2, .2), 2.))
    full = driver.solve_adaptive_arc_program(problem(plastic=True), program); done(full, 2)
    first = driver.solve_adaptive_arc_program(problem(plastic=True), program, stop_after=1)
    assert first.status == 'paused'
    resumed = driver.solve_adaptive_arc_program(problem(plastic=True), program, checkpoint=first.checkpoint)
    done(resumed, 2); assert resumed.checkpoint == full.checkpoint
    native = json.loads(json.loads(full.checkpoint)['kernel_checkpoint'])
    inner = json.loads(native['element_states'][0]['state']['payload'])['material_state']
    assert any(h['accumulated'] > 0. for h in inner['histories'])


def test_exhausted_native_zero_iteration_request_is_not_retried(monkeypatch):
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2., max_iterations=0), max_depth=0)
    failed = driver.solve_adaptive_arc_program(problem(), program)
    assert failed.status == 'failed' and failed.completed_targets == 0
    assert json.loads(failed.checkpoint)['attempts'][0]['reason'] == 'ITERATION_LIMIT'
    def forbidden(*args): raise AssertionError('terminal attempt was repeated')
    monkeypatch.setattr(driver, '_attempt', forbidden)
    restored = driver.solve_adaptive_arc_program(problem(), program, checkpoint=failed.checkpoint)
    assert restored.status == 'failed' and restored.checkpoint == failed.checkpoint


@pytest.mark.parametrize('suffix,accepted', [('before_commit', 0), ('committed', 1)])
def test_cancellation_retains_only_accepted_native_state(suffix, accepted):
    program = driver.AdaptiveArcProgram(ArcProgram((.02, .02), 2.)); token = CancellationToken()
    def cancel(event):
        if event['stage'] == 'native_adaptive_arc.'+suffix: token.cancel('adaptive fixture')
    result = driver.solve_adaptive_arc_program(problem(), program, cancellation_token=token, progress=cancel)
    assert result.status == 'cancelled' and result.completed_targets == accepted
    restored = driver.solve_adaptive_arc_program(problem(), program, checkpoint=result.checkpoint,
        stop_after=accepted, stop_after_attempts=len(json.loads(result.checkpoint)['attempts']))
    assert restored.checkpoint == result.checkpoint


def test_observer_error_with_numeric_looking_message_never_cutbacks():
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.)); observed = []
    def reject(event):
        observed.append(event)
        if event['stage'] == 'native_adaptive_arc.before_assembly':
            raise RuntimeError('arc iteration bound exhausted')
    result = driver.solve_adaptive_arc_program(problem(), program, progress=reject)
    assert result.status == 'failed' and result.completed_targets == 0
    attempts = json.loads(result.checkpoint)['attempts']
    assert len(attempts) == 1 and attempts[0]['reason'] == 'UNRECOVERABLE'
    assert not any(e['stage'].endswith('attempt_rejected') for e in observed)


@pytest.fixture(scope='module')
def accepted_capsule():
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.))
    result = driver.solve_adaptive_arc_program(problem(), program); done(result, 1)
    return program, result.checkpoint


@pytest.mark.parametrize('kind', ['size', 'reason', 'iteration', 'kernel', 'hash', 'budget'])
def test_resealed_wrapper_mutations_fail(accepted_capsule, kind):
    program, raw = accepted_capsule; value = json.loads(raw)
    if kind == 'size': value['attempts'][0]['step_size'] *= 2.
    elif kind == 'reason': value['attempts'][0]['reason'] = 'CANCELLATION'
    elif kind == 'iteration': value['attempts'][0]['iterations'] += 1
    elif kind == 'kernel':
        native = json.loads(value['kernel_checkpoint']); native['physical_imbalance'][0] += .01
        value['kernel_checkpoint'] = reseal(native).decode('ascii')
    elif kind == 'hash': value['model_sha256'] = '0'*64
    elif kind == 'budget': value['program']['max_attempts'] += 1
    with pytest.raises(ValueError): driver.solve_adaptive_arc_program(problem(), program, checkpoint=reseal(value))


def test_mutating_observer_cannot_replace_checkpoint_authority():
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.))
    virgin = driver.solve_adaptive_arc_program(problem(), program, stop_after=0)
    def mutate(event):
        if event['stage'] == 'native_adaptive_arc.before_assembly':
            object.__setattr__(program, 'max_attempts', 127)
    result = driver.solve_adaptive_arc_program(problem(), program, progress=mutate)
    assert result.status == 'failed' and result.completed_targets == 0
    assert result.checkpoint == virgin.checkpoint


def test_staging_failure_never_commits(monkeypatch):
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.))
    virgin = driver.solve_adaptive_arc_program(problem(), program, stop_after=0)
    original = driver._envelope
    def reject(program, identity, attempts, kernel):
        if attempts: raise ValueError('adaptive capsule staging fixture')
        return original(program, identity, attempts, kernel)
    monkeypatch.setattr(driver, '_envelope', reject)
    result = driver.solve_adaptive_arc_program(problem(), program)
    assert result.status == 'failed' and result.completed_targets == 0
    assert result.checkpoint == virgin.checkpoint


def test_real_native_plastic_newton_cutback_completes(tmp_path):
    program = driver.AdaptiveArcProgram(ArcProgram((.2,), 2., max_iterations=1),
        max_depth=4, max_attempts=31, max_accepted=16)
    events = []
    result = driver.solve_adaptive_arc_program(problem(plastic=True), program, progress=events.append)
    with (tmp_path/'diagnostic-checkpoint.json').open('xb') as stream: stream.write(result.checkpoint)
    with (tmp_path/'diagnostic-progress.json').open('xb') as stream: stream.write(canonical(events))
    with (tmp_path/'diagnostic-status.json').open('xb') as stream:
        stream.write(canonical(dict(status=result.status, failure=result.failure,
            accepted_steps=result.completed_targets, production_qualified=False)))
    assert result.status == 'completed', result.failure
    value = json.loads(result.checkpoint)
    rows = value['attempts']
    assert any(row['outcome'] == 'CUTBACK' for row in rows)
    assert all(row['reason'] in ('ITERATION_LIMIT', 'EQUILIBRIUM') for row in rows)
    accepted = tuple(row['step_size'] for row in rows if row['outcome'] == 'ACCEPTED')
    assert len(rows) <= 31 and len(accepted) <= 16 and sum(accepted) == .2
    assert len(accepted) == result.completed_targets
    # Compare the discovered schedule to an explicit fixed-step solve. This
    # is a small state-path equivalence check, not a new engineering reference.
    fixed = solve_arc_program(problem(plastic=True), ArcProgram(accepted, 2., max_iterations=1))
    done(fixed, len(accepted))
    assert value['kernel_checkpoint'].encode('ascii') == fixed.checkpoint
    first = driver.solve_adaptive_arc_program(problem(plastic=True), program, stop_after_attempts=3)
    assert first.status == 'paused' and first.completed_targets == 0
    assert len(json.loads(first.checkpoint)['attempts']) == 3
    resumed = driver.solve_adaptive_arc_program(problem(plastic=True), program, checkpoint=first.checkpoint)
    done(resumed, 8)
    assert resumed.checkpoint == result.checkpoint


def test_cancelled_attempts_consume_the_global_budget():
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.), max_attempts=2)
    rows = [row(i, outcome='CANCELLED', reason='CANCELLATION', iterations=0) for i in (1, 2)]
    queue, accepted, status = driver._history(program, rows)
    assert queue == [(0, 0, 0)] and accepted == () and status == 'failed'
    with pytest.raises(ValueError): driver._history(program, [*rows, row(3)])


def test_deadline_at_initialization_starts_no_attempt(monkeypatch):
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.))
    virgin = driver.solve_adaptive_arc_program(problem(), program, stop_after=0)
    calls = []
    def clock():
        calls.append(None)
        return 0. if len(calls) == 1 else 601.
    # Replace this module's time provider, not the process-global time module.
    from types import SimpleNamespace
    monkeypatch.setattr(driver, 'time', SimpleNamespace(monotonic=clock))
    result = driver.solve_adaptive_arc_program(problem(), program)
    assert result.status == 'failed' and 'deadline' in result.failure
    assert result.completed_targets == 0 and result.checkpoint == virgin.checkpoint


@pytest.mark.parametrize('kind', ['empty_schedule', 'missing_arc'])
def test_corrupted_program_cannot_break_failure_capsule_retention(kind):
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.))
    virgin = driver.solve_adaptive_arc_program(problem(), program, stop_after=0)
    def mutate(event):
        if event['stage'] == 'native_adaptive_arc.before_assembly':
            if kind == 'empty_schedule': object.__setattr__(program.arc, 'steps', ())
            else: object.__setattr__(program, 'arc', None)
    result = driver.solve_adaptive_arc_program(problem(), program, progress=mutate)
    assert result.status == 'failed' and result.completed_targets == 0
    assert result.checkpoint == virgin.checkpoint


def test_line_search_cannot_exhaust_at_an_iteration_without_a_corrector():
    program = driver.AdaptiveArcProgram(ArcProgram((.02,), 2.))
    with pytest.raises(ValueError):
        driver._history(program, [row(1, reason='LINE_SEARCH_LIMIT', iterations=16)])
