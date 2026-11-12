"""Private retained-plastic force transactions and restart; no public routing."""
from dataclasses import dataclass
from time import monotonic
import numpy as np
from anyfesolver.control import cancellation_safe_point, SolveCancelled
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_retained_plastic_state import Context, State


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
    started = monotonic(); cancellation_safe_point(cancellation_token, 'paired-plastic.start')
    context = Context(model, program); layout = context.layout
    end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded plastic stop target')
    if progress is not None and not callable(progress): raise ValueError('callable plastic observer required')
    if checkpoint is None:
        if expected_checkpoint_sha256 is not None: raise ValueError('external hash requires a checkpoint')
        accepted, records = context.initial, (); capsule = context.checkpoint(records)
    else:
        accepted, records = context.restore(checkpoint, expected_sha256=expected_checkpoint_sha256); capsule = checkpoint
    if accepted.completed_targets > end: raise ValueError('cannot rewind accepted plastic history')
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None
    def safe(stage, target, iteration=0):
        cancellation_safe_point(cancellation_token, stage)
        if monotonic()-started > 120: raise RuntimeError('paired-plastic controller cooperative deadline')
        context.guard()
        if progress is not None: progress(dict(stage=stage, target=target, iteration=iteration))
        context.guard(); cancellation_safe_point(cancellation_token, stage)
    try:
        safe('paired-plastic.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            target = program.targets[index]; trial = accepted.mechanical; origins = accepted.histories
            for iteration in range(program.max_iterations+1):
                safe('paired-plastic.before_assembly', index+1, iteration)
                residual, hessian, metrics, _ = context.assemble(trial, target, origins)
                if max(metrics) <= 1e-11:
                    proposed, record = context.stage(trial, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('paired-plastic.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('paired-plastic.committed', accepted.completed_targets, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('paired-plastic Newton limit')
                safe('paired-plastic.before_factorization', index+1, iteration)
                step = np.zeros(layout.count)
                step[layout.free] = np.linalg.solve(hessian[np.ix_(layout.free, layout.free)], -residual[layout.free])
                if not np.isfinite(step).all(): raise ValueError('nonfinite paired-plastic Newton step')
                for cut in range(program.max_backtracks+1):
                    safe('paired-plastic.before_trial', index+1, iteration)
                    candidate = layout.advance(trial, step*(.5**cut))
                    _, _, changed, _ = context.assemble(candidate, target, origins)
                    if max(changed) < max(metrics): trial = candidate; break
                else: raise RuntimeError('paired-plastic line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_targets, accepted, capsule, failure)
