"""Private conservative line controller with local force elimination.

The element, loads, accepted-history/replay and globalization are unchanged.
Only the linear solver changes. Its distinct program identity prevents silent
cross-backend restart or reinterpretation of historical dense checkpoints.
"""
from time import monotonic
from types import SimpleNamespace
import numpy as np

from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_fibre_line_program import Context as DenseContext
from ._ge_beam3_retained_fibre_state import State
from ._ge_beam3_retained_fibre_program import Result
from ._ge_beam3_seeded_fibre_control import chart_fraction
from ._ge_beam3_refined_fibre_schur_solver import RefinedResultantSchurSolver as ResultantSchurSolver, POLICY as LINEAR_POLICY
from ._ge_beam3_p5_seeded.core import sha

PROGRAM='GE_BEAM3_CONSERVATIVE_LINE_LOCAL_FORCE_ELIMINATION_DEVELOPMENT_V1'


class Context(DenseContext):
    def __init__(self,model,program,*,line_forces,check=None,max_coordinates=256):
        super().__init__(model,program,line_forces=line_forces,check=check,max_coordinates=max_coordinates)
        parent=self.identity
        self.program_data={**self.program_data,'schema':PROGRAM,'linear_solver':LINEAR_POLICY,
            'factor_reuse':'ONE_FROZEN_SPATIAL_JACOBIAN_PER_NEWTON_ITERATION',
            'full_increment_recovery_required':True}
        self.identity=sha(dict(parent_capture=parent,program=self.program_data))
        self._program_identity=sha(self.program_data)
        old=self.initial
        self.initial=State(old.mechanical,old.origins,old.histories,0,self.identity)
        self.genesis=self._record(self.initial.mechanical,self.initial.origins,0,0,self.identity)[1]


def solve_force_program(model, program, *, line_forces, checkpoint=None,
                        expected_checkpoint_sha256=None, stop_after=None,
                        cancellation_token=None, progress=None, max_coordinates=256):
    started = monotonic(); cancellation_safe_point(cancellation_token, 'fibre-schur-line.start')
    context = Context(model, program, line_forces=line_forces, max_coordinates=max_coordinates,
        check=lambda: cancellation_safe_point(cancellation_token, 'fibre-schur-line.material'))
    layout = context.layout; end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded line stop target')
    if progress is not None and not callable(progress): raise ValueError('callable line observer required')
    if checkpoint is None:
        if expected_checkpoint_sha256 is not None: raise ValueError('external hash requires a checkpoint')
        accepted, records = context.initial, (); capsule = context.checkpoint(records)
    else:
        accepted, records = context.restore(checkpoint, expected_sha256=expected_checkpoint_sha256); capsule = checkpoint
    if accepted.completed_targets > end: raise ValueError('cannot rewind accepted line history')
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None

    def safe(stage, target, iteration=0, **data):
        cancellation_safe_point(cancellation_token, stage); context.guard()
        if monotonic()-started > 120.: raise RuntimeError('fibre Schur line cooperative deadline')
        if progress is not None: progress(dict(stage=stage, target=target, iteration=iteration, **data))
        context.guard(); cancellation_safe_point(cancellation_token, stage)

    try:
        safe('fibre-schur-line.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            parameter = program.targets[index]; trial = accepted.mechanical; origins = accepted.histories
            for iteration in range(program.max_iterations+1):
                safe('fibre-schur-line.before_assembly', index+1, iteration)
                residual, hessian, metrics, _ = context.assemble(trial, parameter, origins)
                safe('fibre-schur-line.iteration', index+1, iteration, metrics=metrics)
                if max(metrics) <= 1e-11:
                    proposed, record = context.stage(trial, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('fibre-schur-line.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('fibre-schur-line.committed', index+1, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('fibre Schur line Newton limit')
                safe('fibre-schur-line.before_factorization', index+1, iteration)
                tangent = context.tangent(residual, hessian, parameter)
                factor = ResultantSchurSolver(layout, tangent)
                correction = factor.solve(-residual).increment[layout.free]
                safe('fibre-schur-line.factorized', index+1, iteration, linear_solver=factor.diagnostics())
                norm = float(np.linalg.norm(correction))
                if not np.isfinite(correction).all() or not np.isfinite(norm) or norm == 0.:
                    raise ValueError('finite nonzero line Newton correction required')
                step = np.zeros(layout.count); step[layout.free] = correction
                fraction = chart_fraction(SimpleNamespace(layout=layout, physical=context), trial, step)
                for cut in range(program.max_backtracks+1):
                    safe('fibre-schur-line.before_trial', index+1, iteration)
                    candidate = layout.advance(trial, step*(fraction*.5**cut))
                    changed_residual, _, changed, _ = context.assemble(candidate, parameter, origins)
                    merit = float(np.linalg.norm(factor.solve(changed_residual).increment[layout.free]))
                    if not np.isfinite(merit): raise ValueError('finite line Newton merit required')
                    if max(changed) <= 1e-11 or merit < norm:
                        trial = candidate; break
                else: raise RuntimeError('fibre Schur line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_targets, accepted, capsule, failure)
