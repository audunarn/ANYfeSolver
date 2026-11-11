"""Private native physical-fibre force control with conservative lifted line work.

This successor reuses accepted-state/replay ownership, not old load mechanics.
No moments, follower loads, public element route or spectral authority is added.
"""
from time import monotonic
from types import SimpleNamespace
import numpy as np

from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_retained_fibre_state import Context as PhysicalContext, State
from ._ge_beam3_retained_fibre_program import Result
from ._ge_beam3_seeded_fibre_control import spatial_jacobian, chart_fraction
from ._ge_beam3_fibre_line_work import ReferenceLineForces, evaluate, Work
from ._ge_beam3_p5_seeded.core import sha
from ._native_reference_modal import _owned


PROGRAM = 'GE_BEAM3_RETAINED_FIBRE_CONSERVATIVE_LINE_PROGRAM_DEVELOPMENT_V1'


class Context(PhysicalContext):
    def __init__(self, model, program, *, line_forces, check=None, max_coordinates=256):
        if type(line_forces) is not ReferenceLineForces:
            raise ValueError('exact explicit reference line-force pattern required')
        if any(row[0] not in model.mesh.elements for row in line_forces.rows):
            raise ValueError('line force references an absent element')
        self.line_forces = line_forces
        self._line_identity = sha(line_forces.descriptor())
        self._line_ready = False
        super().__init__(model, program, check=check, max_coordinates=max_coordinates)
        self._line_ready = True
        parent = self.identity
        self.program_data = {**self.program_data, 'schema': PROGRAM,
            'line_forces': line_forces.descriptor(),
            'newton_derivative': 'SPATIAL_DERIVATIVE_OF_COMPLETE_INTERNAL_MINUS_EXTERNAL_RESIDUAL',
            'line_search': 'FROZEN_SPATIAL_NEWTON_CORRECTION_DECREASE',
            'chart_globalization': 'HALF_REMAINING_PRINCIPAL_CHART_MARGIN',
            'spectral_authority': False}
        self.identity = sha(dict(parent_capture=parent, program=self.program_data))
        self._program_identity = sha(self.program_data)
        initial = self.initial
        self.initial = State(initial.mechanical, initial.origins, initial.histories, 0, self.identity)
        self.genesis = self._record(self.initial.mechanical, self.initial.origins, 0, 0, self.identity)[1]

    def guard(self):
        super().guard()
        if sha(self.line_forces.descriptor()) != self._line_identity:
            raise ValueError('frozen line-force pattern changed')
        if hasattr(self, '_program_identity') and sha(self.program_data) != self._program_identity:
            raise ValueError('frozen line-force program changed')
        if hasattr(self, '_program_identity') and self._line_ready is not True:
            raise ValueError('frozen line-force assembly disabled')

    def line_work(self, mechanical):
        self.guard()
        forces = {row[0]: row[1:] for row in self.line_forces.rows}
        layout = self.layout; value = 0.
        gradient = np.zeros(layout.count); hessian = np.zeros((layout.count, layout.count))
        for i, (eid, _) in enumerate(layout.elements):
            if eid not in forces: continue
            probe = self.probes[i]; nodes = layout.nodes[i]
            work = evaluate(probe.reference, mechanical.positions[nodes], mechanical.position_low[nodes],
                mechanical.cell_rotations[i], forces[eid], order=probe.order)
            value += work.value
            gradient[layout.slots[i]] += work.gradient
            hessian[np.ix_(layout.slots[i], layout.slots[i])] += work.hessian
            self.check()
        if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise ValueError('nonfinite assembled line work')
        self.guard(); return Work(float(value), _owned(gradient), _owned(hessian))

    def assemble(self, mechanical, parameter, origins):
        if type(parameter) is not float or not np.isfinite(parameter):
            raise ValueError('finite binary64 line-load parameter required')
        residual, hessian, metrics, responses = super().assemble(mechanical, parameter, origins)
        if self._line_ready:
            work = self.line_work(mechanical)
            residual -= parameter*work.gradient
            hessian -= parameter*work.hessian
            scaled = residual/self.layout.scales
            metrics = (float(np.linalg.norm(scaled[self.layout.equilibrium])),
                       float(np.linalg.norm(scaled[self.layout.compatibility])))
        return residual, hessian, metrics, responses

    def tangent(self, residual, hessian, parameter):
        self.guard()
        # Conservative line work has its own cell-rotation connection. Remove
        # the connection of the NET residual from the TOTAL Exp-chart Hessian.
        # Separate nodal forces have no rotational components or Hessian.
        return spatial_jacobian(self.layout, residual, hessian)

    def require_conservative_spectrum(self):
        raise ValueError('line-load spectrum requires separately bound external Hessian authority')


def solve_force_program(model, program, *, line_forces, checkpoint=None,
                        expected_checkpoint_sha256=None, stop_after=None,
                        cancellation_token=None, progress=None, max_coordinates=256):
    started = monotonic(); cancellation_safe_point(cancellation_token, 'fibre-line.start')
    context = Context(model, program, line_forces=line_forces, max_coordinates=max_coordinates,
        check=lambda: cancellation_safe_point(cancellation_token, 'fibre-line.material'))
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
        if monotonic()-started > 120.: raise RuntimeError('fibre line cooperative deadline')
        if progress is not None: progress(dict(stage=stage, target=target, iteration=iteration, **data))
        context.guard(); cancellation_safe_point(cancellation_token, stage)

    try:
        safe('fibre-line.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            parameter = program.targets[index]; trial = accepted.mechanical; origins = accepted.histories
            for iteration in range(program.max_iterations+1):
                safe('fibre-line.before_assembly', index+1, iteration)
                residual, hessian, metrics, _ = context.assemble(trial, parameter, origins)
                safe('fibre-line.iteration', index+1, iteration, metrics=metrics)
                if max(metrics) <= 1e-11:
                    proposed, record = context.stage(trial, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('fibre-line.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('fibre-line.committed', index+1, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('fibre line Newton limit')
                safe('fibre-line.before_factorization', index+1, iteration)
                tangent = context.tangent(residual, hessian, parameter)[np.ix_(layout.free, layout.free)]
                correction = np.linalg.solve(tangent, -residual[layout.free])
                norm = float(np.linalg.norm(correction))
                if not np.isfinite(correction).all() or not np.isfinite(norm) or norm == 0.:
                    raise ValueError('finite nonzero line Newton correction required')
                step = np.zeros(layout.count); step[layout.free] = correction
                fraction = chart_fraction(SimpleNamespace(layout=layout, physical=context), trial, step)
                for cut in range(program.max_backtracks+1):
                    safe('fibre-line.before_trial', index+1, iteration)
                    candidate = layout.advance(trial, step*(fraction*.5**cut))
                    changed_residual, _, changed, _ = context.assemble(candidate, parameter, origins)
                    merit = float(np.linalg.norm(np.linalg.solve(tangent, changed_residual[layout.free])))
                    if not np.isfinite(merit): raise ValueError('finite line Newton merit required')
                    if max(changed) <= 1e-11 or merit < norm:
                        trial = candidate; break
                else: raise RuntimeError('fibre line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_targets, accepted, capsule, failure)
