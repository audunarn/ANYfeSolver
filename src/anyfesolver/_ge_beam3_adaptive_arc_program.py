"""Private bounded dyadic cutback of native objective arc segments.

Only explicit Newton/line-search exhaustion is subdivisible. This does not
retry failed processes, relax tolerances, or select a bifurcation branch.
"""

from dataclasses import dataclass
import math
import time

import numpy as np

from anyfesolver.control import cancellation_safe_point, SolveCancelled
from anyfesolver.linalg import factorize, MatrixClass
from anyfesolver.nonlinear_state import native_trial_full_coordinates
from anyfesolver._ge_beam3_p5.arrays import _readonly
from anyfesolver._ge_beam3_p5_loads.core import canonical, sha
from anyfesolver._ge_beam3_p5_loads.codec import _load
from anyfesolver._ge_beam3_p5_loads.state import parameter_value
from anyfesolver._ge_beam3_load_program import ForceProgramResult, TOLERANCE
from anyfesolver import _ge_beam3_arc_program as arc


PROGRAM = 'GE_BEAM3_NATIVE_DYADIC_ARC_PROGRAM_V1'
SCHEMA = 'GE_BEAM3_NATIVE_DYADIC_ARC_CHECKPOINT_V1'
NUMERICAL = ('ITERATION_LIMIT', 'LINE_SEARCH_LIMIT')


@dataclass(frozen=True)
class AdaptiveArcProgram:
    arc: arc.ArcProgram
    max_depth: int = 4
    minimum_step: float = 1e-6
    max_attempts: int = 128
    max_accepted: int = 64

    def __post_init__(self):
        if type(self.arc) is not arc.ArcProgram: raise ValueError('exact native arc program required')
        for name, lower, upper in (('max_depth', 0, 12), ('max_attempts', 1, 256), ('max_accepted', 1, 64)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError('bounded integer adaptive policy required')
        if not 0. < parameter_value(self.minimum_step) <= min(self.arc.steps):
            raise ValueError('minimum arc step must be positive and admit every initial segment')
        if len(self.arc.steps) > min(self.max_attempts, self.max_accepted):
            raise ValueError('budget cannot cover the initial arc schedule')

    def descriptor(self):
        return dict(schema=PROGRAM, arc=self.arc.descriptor(), max_depth=self.max_depth,
            minimum_step=self.minimum_step, max_attempts=self.max_attempts, max_accepted=self.max_accepted,
            split_policy='ORDERED_EQUAL_HALVES_NO_GROWTH', recoverable_reasons=NUMERICAL)


def _step(program, item):
    root, depth, part = item
    return math.ldexp(program.arc.steps[root], -depth)


def _can_split(program, queue, accepted, attempted):
    root, depth, part = queue[0]
    return (depth < program.max_depth and _step(program, (root, depth+1, 2*part)) >= program.minimum_step
        and attempted < program.max_attempts and len(accepted)+len(queue)+1 <= program.max_accepted)


def _history(program, rows):
    """Reconstruct the complete pending dyadic partition; never trust a queue."""
    if type(rows) is not list or len(rows) > program.max_attempts: raise ValueError('bounded adaptive attempts required')
    queue = [(i, 0, 0) for i in range(len(program.arc.steps))]; accepted = []; fatal = False
    keys = {'attempt', 'root', 'depth', 'part', 'step_size', 'accepted_before', 'outcome', 'reason', 'iterations'}
    for index, row in enumerate(rows):
        if fatal or not queue: raise ValueError('attempt after terminal adaptive schedule')
        if type(row) is not dict or set(row) != keys: raise ValueError('exact adaptive attempt schema required')
        for name in ('attempt', 'root', 'depth', 'part', 'accepted_before', 'iterations'):
            if type(row[name]) is not int: raise ValueError('integer attempt coordinates required')
        item = (row['root'], row['depth'], row['part'])
        if (row['attempt'] != index+1 or item != queue[0] or row['accepted_before'] != len(accepted)
                or parameter_value(row['step_size']) != _step(program, item)):
            raise ValueError('adaptive schedule/order/size mismatch')
        if not 0 <= row['iterations'] <= program.arc.max_iterations: raise ValueError('adaptive iteration record mismatch')
        outcome, reason = row['outcome'], row['reason']
        if outcome == 'ACCEPTED' and reason == 'EQUILIBRIUM':
            accepted.append(row['step_size']); queue.pop(0)
        elif outcome == 'CANCELLED' and reason == 'CANCELLATION':
            pass  # Consumed attempt; explicit caller continuation may start a new one.
        elif reason in NUMERICAL and outcome in ('CUTBACK', 'FAILED'):
            if reason == 'ITERATION_LIMIT' and row['iterations'] != program.arc.max_iterations:
                raise ValueError('iteration exhaustion count mismatch')
            if reason == 'LINE_SEARCH_LIMIT' and row['iterations'] >= program.arc.max_iterations:
                raise ValueError('line-search exhaustion requires an attempted corrector')
            can_split = _can_split(program, queue, accepted, index+1)
            if (outcome == 'CUTBACK') != can_split: raise ValueError('cutback violates declared budgets')
            if can_split:
                root, depth, part = queue.pop(0)
                queue[:0] = [(root, depth+1, 2*part), (root, depth+1, 2*part+1)]
            else: fatal = True
        elif outcome == 'FAILED' and reason == 'UNRECOVERABLE':
            fatal = True
        else: raise ValueError('unregistered adaptive outcome/reason')
        if len(accepted)+len(queue) > program.max_accepted:
            raise ValueError('accepted-step budget exceeded')
    disposition = 'failed' if fatal or (queue and len(rows) == program.max_attempts) else ('ready' if queue else 'completed')
    return queue, tuple(accepted), disposition


def _kernel_program(program, accepted):
    source = program.arc
    return arc.ArcProgram(accepted or (source.steps[0],), source.length_scale, source.parameter_scale,
        source.initial_load_sign, source.nodal_forces, source.max_iterations, source.max_backtracks)


def _envelope(program, identity, attempts, kernel):
    queue, accepted, disposition = _history(program, attempts)
    body = dict(schema=SCHEMA, program=program.descriptor(), program_sha256=sha(program.descriptor()),
        model_sha256=identity, attempts=attempts, kernel_checkpoint=kernel.decode('ascii'))
    raw = canonical({**body, 'checkpoint_sha256': sha(body)})
    _load(raw.decode('ascii'))  # Preserve strict finite/duplicate/depth/size policy.
    return raw


def _restore(raw, model, program, identity):
    if type(raw) is not bytes: raise ValueError('canonical adaptive checkpoint bytes required')
    value = _load(raw.decode('ascii'))
    keys = {'schema', 'program', 'program_sha256', 'model_sha256', 'attempts', 'kernel_checkpoint', 'checkpoint_sha256'}
    if type(value) is not dict or set(value) != keys: raise ValueError('exact adaptive checkpoint schema required')
    body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
    if (value['schema'] != SCHEMA or value['model_sha256'] != identity
            or canonical(value['program']) != canonical(program.descriptor())
            or value['program_sha256'] != sha(program.descriptor()) or value['checkpoint_sha256'] != sha(body)):
        raise ValueError('adaptive program/model/hash mismatch')
    queue, accepted, disposition = _history(program, value['attempts'])
    if type(value['kernel_checkpoint']) is not str: raise ValueError('embedded native capsule required')
    kernel = value['kernel_checkpoint'].encode('ascii'); kernel_program = _kernel_program(program, accepted)
    elements, n, free, nodal, kernel_id, guard, maps, metric, initial = arc._capture(model, kernel_program)
    restored = arc._restore(kernel, model, kernel_program, kernel_id, elements, n, free, nodal, maps, metric, initial)
    cursor, p, total, reaction, states, direction, records, origin = restored
    if cursor != len(accepted): raise ValueError('adaptive accepted-state count mismatch')
    successful = [row for row in value['attempts'] if row['outcome'] == 'ACCEPTED']
    if any(row['iterations'] != point['iterations'] for row, point in zip(successful, records)):
        raise ValueError('adaptive/native accepted iteration mismatch')
    return value['attempts'], kernel, restored


def _attempt(model, source, elements, n, free, nodal, maps, metric, store, total, p, previous, step, observe):
    """One actual native solve. Only explicit convergence exhaustion returns failure."""
    def evaluate(u, parameter, iteration):
        observe('before_assembly', iteration)
        force, matrix, trial = arc._assemble(model, store, u, parameter, nodal)
        observe('after_assembly', iteration)
        physical = arc._physical_force(model, elements, trial, parameter, nodal)
        return force, matrix, trial, physical

    def force_norm(force, physical, parameter):
        return max(np.linalg.norm(force[free]), np.linalg.norm(physical[free]))/max(1., abs(parameter)*np.linalg.norm(nodal))

    force, matrix, trial, physical = evaluate(total, p, 0)
    if force_norm(force, physical, p) > TOLERANCE: raise ValueError('adaptive origin is not an equilibrium')
    column = arc._parameter_column(model, elements, store, trial, n, nodal)
    observe('before_predictor', 0)
    tangent = arc._direction(matrix, column, free, previous, metric)
    candidate = total+step*tangent[:-1]; parameter = float(p+step*tangent[-1])
    for iteration in range(source.max_iterations+1):
        force, matrix, trial, physical = evaluate(candidate, parameter, iteration)
        gap, row = arc._arc(candidate, total, tangent, metric, parameter, p, step, maps)
        norm = max(force_norm(force, physical, parameter), abs(gap))
        observe('iteration', iteration, merit=float(norm), parameter=parameter, arc_error=float(gap))
        if norm <= TOLERANCE:
            return dict(accepted=True, iterations=iteration, total=candidate, parameter=parameter, trial=trial,
                physical=physical, direction=tangent, arc_residual=float(abs(gap)))
        if iteration == source.max_iterations:
            return dict(accepted=False, reason='ITERATION_LIMIT', iterations=iteration)
        column = arc._parameter_column(model, elements, store, trial, n, nodal)
        observe('before_corrector', iteration)
        handle = factorize(arc._bordered(matrix, column, free, row), MatrixClass.GENERAL)
        delta = np.asarray(handle.solve(-np.r_[force[free], gap]), dtype=float).reshape(-1)
        if not np.isfinite(delta).all(): raise ValueError('nonfinite adaptive corrector')
        for cut in range(source.max_backtracks+1):
            proposed = candidate.copy(); proposed[free] += (.5**cut)*delta[:-1]
            next_p = float(parameter+(.5**cut)*delta[-1])
            changed, _, _, changed_physical = evaluate(proposed, next_p, iteration)
            changed_gap, _ = arc._arc(proposed, total, tangent, metric, next_p, p, step, maps)
            if max(force_norm(changed, changed_physical, next_p), abs(changed_gap)) < norm:
                candidate, parameter = proposed, next_p; break
        else:
            return dict(accepted=False, reason='LINE_SEARCH_LIMIT', iterations=iteration)


def solve_adaptive_arc_program(model, program, *, checkpoint=None, stop_after=None,
        stop_after_attempts=None, cancellation_token=None, progress=None):
    started = time.monotonic()
    if type(program) is not AdaptiveArcProgram: raise ValueError('exact adaptive arc program required')
    if progress is not None and not callable(progress): raise ValueError('callable adaptive observer required')
    elements, n, free, nodal, base_id, base_guard, maps, metric, initial = arc._capture(model, program.arc)
    descriptor = sha(program.descriptor()); identity = sha(dict(native_model=base_id, adaptive_program=descriptor))
    end = program.max_accepted if stop_after is None else stop_after
    attempt_end = program.max_attempts if stop_after_attempts is None else stop_after_attempts
    if type(end) is not int or not 0 <= end <= program.max_accepted: raise ValueError('bounded accepted stop count required')
    if type(attempt_end) is not int or not 0 <= attempt_end <= program.max_attempts: raise ValueError('bounded attempt stop count required')
    if checkpoint is None:
        states = {i: e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1) for i, e in elements}
        total = np.zeros(n); p = 0.; cursor = 0; direction = initial.copy(); records = []; origin = None
        reaction = arc._physical_force(model, elements, states, p, nodal); attempts = []
        native = _kernel_program(program, ())
        kernel_id = arc._capture(model, native)[4]
        kernel = arc._capsule(model, native, kernel_id, elements, states, total, reaction,
            cursor, p, direction, metric, records, origin)
        capsule = _envelope(program, identity, attempts, kernel)
    else:
        attempts, kernel, restored = _restore(checkpoint, model, program, identity)
        cursor, p, total, reaction, states, direction, records, origin = restored
        capsule = checkpoint
    if cursor > end or len(attempts) > attempt_end: raise ValueError('cannot rewind consumed adaptive progress')
    store = arc._store(model, states, total); failure = None; active = False; item = None; iteration_seen = 0

    def observe(stage, iteration=0, **data):
        nonlocal iteration_seen
        iteration_seen = iteration
        label = 'native_adaptive_arc.'+stage
        cancellation_safe_point(cancellation_token, label); base_guard()
        if sha(program.descriptor()) != descriptor: raise ValueError('adaptive program changed')
        if time.monotonic()-started > 600.: raise RuntimeError('adaptive cooperative deadline exceeded')
        if progress is not None:
            progress(dict(stage=label, accepted_steps=cursor, attempt=len(attempts)+(1 if active else 0), iteration=iteration, **data))
        cancellation_safe_point(cancellation_token, label); base_guard()
        if sha(program.descriptor()) != descriptor: raise ValueError('adaptive program changed')

    def row(outcome, reason, iterations):
        root, depth, part = item
        return dict(attempt=len(attempts)+1, root=root, depth=depth, part=part,
            step_size=_step(program, item), accepted_before=cursor, outcome=outcome, reason=reason, iterations=iterations)

    try:
        observe('initialized')
        while True:
            queue, accepted, disposition = _history(program, attempts)
            if disposition != 'ready':
                status = disposition
                if status == 'failed': failure = 'adaptive schedule failed or attempt budget exhausted'
                break
            if cursor == end or len(attempts) == attempt_end:
                status = 'paused'; break
            item = queue[0]; active = True; iteration_seen = 0
            origin = dict(total=total.tolist(), parameter=p, direction=direction.tolist(),
                element_states=arc._states(model, elements, store.materialize()))
            observe('attempt_started', step_size=_step(program, item), depth=item[1])
            result = _attempt(model, program.arc, elements, n, free, nodal, maps, metric,
                store, total, p, direction, _step(program, item), observe)
            if not result['accepted']:
                if store.has_active_trial: store.discard_trial(store.active_trial_token())
                outcome = 'CUTBACK' if _can_split(program, queue, accepted, len(attempts)+1) else 'FAILED'
                next_attempts = [*attempts, row(outcome, result['reason'], result['iterations'])]
                staged = _envelope(program, identity, next_attempts, kernel)
                attempts, capsule, active = next_attempts, staged, False
                observe('attempt_rejected', reason=result['reason'], outcome=outcome)
                continue
            candidate, parameter, tangent = result['total'], result['parameter'], result['direction']
            next_records = [*records, dict(step=cursor+1, step_size=_step(program, item), parameter=parameter,
                iterations=result['iterations'], arc_residual=result['arc_residual'], direction=tangent.tolist())]
            next_attempts = [*attempts, row('ACCEPTED', 'EQUILIBRIUM', result['iterations'])]
            native = _kernel_program(program, (*accepted, _step(program, item)))
            kernel_id = arc._capture(model, native)[4]
            next_kernel = arc._capsule(model, native, kernel_id, elements, result['trial'], candidate,
                result['physical'], cursor+1, parameter, tangent, metric, next_records, origin)
            staged = _envelope(program, identity, next_attempts, next_kernel)
            next_total = candidate.copy(); next_reaction = result['physical'].copy(); next_direction = tangent.copy()
            observe('before_commit', result['iterations'])
            coordinates = native_trial_full_coordinates(store, model, candidate)
            store.commit(store.active_trial_token(), accepted_full_displacement=candidate, accepted_full_coordinates=coordinates)
            total, reaction, direction = next_total, next_reaction, next_direction
            p, cursor, records = parameter, cursor+1, next_records
            attempts, kernel, capsule, active = next_attempts, next_kernel, staged, False
            observe('committed', result['iterations'])
    except Exception as error:
        status = 'cancelled' if isinstance(error, SolveCancelled) else 'failed'
        failure = type(error).__name__+': '+str(error)
        if active:
            try:
                # A hostile observer may have modified the frozen dataclass.
                # Never replace a valid capsule with changed program authority.
                if sha(program.descriptor()) != descriptor: raise ValueError('adaptive program changed')
                next_attempts = [*attempts, row('CANCELLED' if status == 'cancelled' else 'FAILED',
                    'CANCELLATION' if status == 'cancelled' else 'UNRECOVERABLE', iteration_seen)]
                capsule = _envelope(program, identity, next_attempts, kernel)
            except Exception: pass  # Serialization/authority failure retains the previous valid capsule.
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())
    return ForceProgramResult(status, cursor, p, _readonly(total), _readonly(reaction), capsule, failure)
