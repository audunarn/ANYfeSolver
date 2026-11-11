"""Private force/couple control with the native spatial residual Jacobian.

The native internal potential, section, history and recovery remain unchanged.
Nonconservative external moments are never included in a scalar energy merit.
This static development route does not authorize modal/buckling under moments.
"""
from time import monotonic
from types import SimpleNamespace
import numpy as np

from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_retained_fibre_state import Context as PhysicalContext, State
from ._ge_beam3_retained_fibre_program import Result
from ._ge_beam3_seeded_fibre_control import spatial_jacobian, chart_fraction
from ._ge_beam3_p5_seeded.core import sha


PROGRAM = 'GE_BEAM3_SPATIAL_FORCE_AND_COUPLE_CONTROL_DEVELOPMENT_V1'


class Context(PhysicalContext):
    def __init__(self, model, program, *, nodal_moments=None, check=None, max_coordinates=256):
        super().__init__(model, program, nodal_moments=nodal_moments, check=check, max_coordinates=max_coordinates)
        parent = self.identity
        self.program_data = {**self.program_data, 'schema': PROGRAM,
            'newton_derivative': 'INTERNAL_SPATIAL_JACOBIAN_MINUS_ZERO_CONSTANT_SPATIAL_LOAD_DERIVATIVE',
            'line_search': 'FROZEN_SPATIAL_NEWTON_CORRECTION_DECREASE',
            'chart_globalization': 'HALF_REMAINING_PRINCIPAL_CHART_MARGIN',
            'conservative_modal_buckling_authorized': False}
        self.identity = sha(dict(parent_capture=parent, program=self.program_data))
        self._program_identity = sha(self.program_data)
        initial = self.initial
        self.initial = State(initial.mechanical, initial.origins, initial.histories, 0, self.identity)
        self.genesis = self._record(self.initial.mechanical, self.initial.origins, 0, 0, self.identity)[1]

    def guard(self):
        super().guard()
        if hasattr(self, '_program_identity') and sha(self.program_data) != self._program_identity:
            raise ValueError('spatial load controller capture changed')

    def tangent(self, residual, internal_hessian, parameter):
        self.guard()
        # The Exp-chart Hessian contains the connection term formed from the
        # INTERNAL spatial moments, not from the net residual. Add external
        # moments back before removing 1/2 skew(M_int). Translations do not
        # enter that correction. Constant spatial loads have D f_ext = 0.
        internal = residual.copy()
        internal[:self.layout.nodal_count] += parameter*self.layout.force
        return spatial_jacobian(self.layout, internal, internal_hessian)

    def require_conservative_spectrum(self):
        raise ValueError('spatial force/couple development checkpoint is not conservative spectral authority')


def solve_force_program(model, program, *, nodal_moments=None, checkpoint=None,
                        expected_checkpoint_sha256=None, stop_after=None,
                        cancellation_token=None, progress=None, max_coordinates=256):
    started = monotonic(); cancellation_safe_point(cancellation_token, 'spatial-fibre.start')
    context = Context(model, program, nodal_moments=nodal_moments, max_coordinates=max_coordinates,
        check=lambda: cancellation_safe_point(cancellation_token, 'spatial-fibre.material'))
    layout = context.layout; end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded spatial force stop target')
    if progress is not None and not callable(progress): raise ValueError('callable spatial force observer required')
    if checkpoint is None:
        if expected_checkpoint_sha256 is not None: raise ValueError('external hash requires a checkpoint')
        accepted, records = context.initial, (); capsule = context.checkpoint(records)
    else:
        accepted, records = context.restore(checkpoint, expected_sha256=expected_checkpoint_sha256); capsule = checkpoint
    if accepted.completed_targets > end: raise ValueError('cannot rewind accepted history')
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None
    def safe(stage, target, iteration=0, **data):
        cancellation_safe_point(cancellation_token, stage); context.guard()
        if monotonic()-started > 120.: raise RuntimeError('spatial fibre cooperative deadline')
        if progress is not None: progress(dict(stage=stage, target=target, iteration=iteration, **data))
        context.guard(); cancellation_safe_point(cancellation_token, stage)
    try:
        safe('spatial-fibre.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            parameter = program.targets[index]; trial = accepted.mechanical; origins = accepted.histories
            for iteration in range(program.max_iterations+1):
                safe('spatial-fibre.before_assembly', index+1, iteration)
                residual, hessian, metrics, _ = context.assemble(trial, parameter, origins)
                safe('spatial-fibre.iteration', index+1, iteration, metrics=metrics)
                if max(metrics) <= 1e-11:
                    proposed, record = context.stage(trial, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('spatial-fibre.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('spatial-fibre.committed', index+1, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('spatial fibre Newton limit')
                safe('spatial-fibre.before_factorization', index+1, iteration)
                tangent = context.tangent(residual, hessian, parameter)[np.ix_(layout.free, layout.free)]
                correction = np.linalg.solve(tangent, -residual[layout.free])
                norm = float(np.linalg.norm(correction))
                if not np.isfinite(correction).all() or not np.isfinite(norm) or norm == 0.:
                    raise ValueError('finite nonzero spatial Newton correction required')
                step = np.zeros(layout.count); step[layout.free] = correction
                fraction = chart_fraction(SimpleNamespace(layout=layout, physical=context), trial, step)
                for cut in range(program.max_backtracks+1):
                    safe('spatial-fibre.before_trial', index+1, iteration)
                    candidate = layout.advance(trial, step*(fraction*.5**cut))
                    changed_residual, _, changed, _ = context.assemble(candidate, parameter, origins)
                    merit = float(np.linalg.norm(np.linalg.solve(tangent, changed_residual[layout.free])))
                    if not np.isfinite(merit): raise ValueError('finite spatial Newton merit required')
                    if max(changed) <= 1e-11 or merit < norm:
                        trial = candidate; break
                else: raise RuntimeError('spatial fibre line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_targets, accepted, capsule, failure)
