"""Private transactional retained-resultant force controller; no public routing."""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from anyfesolver.control import cancellation_safe_point, SolveCancelled
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_retained_state import Context, State


@dataclass(frozen=True)
class Result:
    status: str
    completed_targets: int
    state: State
    checkpoint: bytes
    failure: str | None
    production_qualified: bool = False


def solve_force_program(model, program, *, checkpoint=None, stop_after=None,
                        cancellation_token=None, progress=None):
    started = monotonic()
    cancellation_safe_point(cancellation_token, 'retained.start')
    context = Context(model, program)
    end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded retained stop target')
    if progress is not None and not callable(progress): raise ValueError('callable retained observer required')
    if checkpoint is None:
        accepted = context.initial; cursor = 0; records = []
        capsule = context.checkpoint(accepted, cursor, records)
    else:
        accepted, cursor, records = context.restore(checkpoint); capsule = checkpoint
    if cursor > end: raise ValueError('cannot rewind retained accepted history')
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None
    def safe(stage, target, iteration=0):
        cancellation_safe_point(cancellation_token, stage)
        if monotonic()-started > 120.: raise RuntimeError('retained controller cooperative deadline')
        context.guard()
        if progress is not None: progress(dict(stage=stage, target=target, iteration=iteration))
        context.guard(); cancellation_safe_point(cancellation_token, stage)
    try:
        safe('retained.initialized', cursor)
        for index in range(cursor, end):
            target = program.targets[index]; trial = accepted
            for iteration in range(program.max_iterations+1):
                safe('retained.before_assembly', index+1, iteration)
                residual, hessian, metrics = context.assemble(trial, target)
                if max(metrics) <= 1e-11:
                    next_records = [*records, dict(target=index+1, parameter=target, iterations=iteration)]
                    staged = context.checkpoint(trial, index+1, next_records)
                    safe('retained.before_commit', index+1, iteration)
                    # One publication point; every trial value is separately owned.
                    accepted, cursor, records, capsule = trial, index+1, next_records, staged
                    safe('retained.committed', cursor, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('retained Newton limit')
                safe('retained.before_factorization', index+1, iteration)
                step = np.zeros(context.count)
                step[context.free] = np.linalg.solve(hessian[np.ix_(context.free, context.free)], -residual[context.free])
                if not np.isfinite(step).all(): raise ValueError('nonfinite retained Newton step')
                for cut in range(program.max_backtracks+1):
                    safe('retained.before_trial', index+1, iteration)
                    candidate = context.advance(trial, step*(.5**cut))
                    _, _, changed = context.assemble(candidate, target)
                    if max(changed) < max(metrics):
                        trial = candidate; break
                else: raise RuntimeError('retained line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, cursor, accepted, capsule, failure)
