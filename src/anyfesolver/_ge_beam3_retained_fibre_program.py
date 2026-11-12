"""Private retained physical-fibre force controller; no public registration."""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_retained_fibre_state import Context, State


@dataclass(frozen=True)
class Result:
    status: str
    completed_targets: int
    state: State
    checkpoint: bytes
    failure: str | None
    production_qualified: bool = False


def solve_force_program(model, program, *, checkpoint=None, expected_checkpoint_sha256=None,
                        stop_after=None, cancellation_token=None, progress=None):
    started = monotonic(); cancellation_safe_point(cancellation_token, 'retained-fibre.start')
    context = Context(model, program, check=lambda: cancellation_safe_point(cancellation_token, 'retained-fibre.material'))
    layout = context.layout; end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded fibre stop target')
    if progress is not None and not callable(progress): raise ValueError('callable fibre observer required')
    if checkpoint is None:
        if expected_checkpoint_sha256 is not None: raise ValueError('external hash requires a fibre checkpoint')
        accepted, records = context.initial, (); capsule = context.checkpoint(records)
    else:
        accepted, records = context.restore(checkpoint, expected_sha256=expected_checkpoint_sha256); capsule = checkpoint
    if accepted.completed_targets > end: raise ValueError('cannot rewind accepted fibre history')
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None
    def safe(stage, target, iteration=0):
        cancellation_safe_point(cancellation_token, stage)
        if monotonic()-started > 120.: raise RuntimeError('retained fibre controller cooperative deadline')
        context.guard()
        if progress is not None: progress(dict(stage=stage, target=target, iteration=iteration))
        context.guard(); cancellation_safe_point(cancellation_token, stage)
    try:
        safe('retained-fibre.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            target = program.targets[index]; trial = accepted.mechanical; origins = accepted.histories
            for iteration in range(program.max_iterations+1):
                safe('retained-fibre.before_assembly', index+1, iteration)
                residual, hessian, metrics, _ = context.assemble(trial, target, origins)
                if max(metrics) <= 1e-11:
                    proposed, record = context.stage(trial, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('retained-fibre.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('retained-fibre.committed', accepted.completed_targets, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('retained fibre Newton limit')
                safe('retained-fibre.before_factorization', index+1, iteration)
                step = np.zeros(layout.count)
                step[layout.free] = np.linalg.solve(hessian[np.ix_(layout.free, layout.free)], -residual[layout.free])
                if not np.isfinite(step).all(): raise ValueError('nonfinite retained fibre Newton step')
                for cut in range(program.max_backtracks+1):
                    safe('retained-fibre.before_trial', index+1, iteration)
                    candidate = layout.advance(trial, step*(.5**cut))
                    _, _, changed, _ = context.assemble(candidate, target, origins)
                    if max(changed) < max(metrics): trial = candidate; break
                else: raise RuntimeError('retained fibre line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_targets, accepted, capsule, failure)
