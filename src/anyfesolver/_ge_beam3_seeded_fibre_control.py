"""Private kinematic-seeded successor of the retained-fibre control driver.

Old control authority and its failing high-contrast case remain unchanged.
Initial resultants are constructed from the same constrained fibre potential.
Every later trial retains the independent full-system resultant updates.
"""
from time import monotonic
from math import isfinite, fsum
import numpy as np
from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_retained_fibre_control import (
    Context as PreviousContext, TranslationProgram, State, Result, bordered,
)
from ._ge_beam3_fibre_kinematic_cell import response as kinematic_response, POLICY as SEED_POLICY
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5.algebra import HALVES, skew


PROGRAM = 'GE_BEAM3_KINEMATIC_SEEDED_SPATIAL_NEWTON_FIBRE_CONTROL_V1'


class Context(PreviousContext):
    def __init__(self, model, program, *, check=None, max_coordinates=256):
        if type(program) is not TranslationProgram: raise ValueError('exact physical translation program required')
        self._requested_program = program.descriptor()
        super().__init__(model, program, check=check, max_coordinates=max_coordinates)
        self.program_data = {**self.program_data, 'schema': PROGRAM, 'initialization': SEED_POLICY,
            'constitutive_trial_policy': 'INITIALIZE_ONCE_THEN_RETAIN_FULL_MIXED_NEWTON_TRIAL_V1',
            'newton_derivative': 'SPATIAL_RESIDUAL_JACOBIAN_FROM_EXP_CHART_HESSIAN_V1',
            'chart_globalization': 'HALF_REMAINING_PRINCIPAL_CHART_MARGIN_THEN_BOUNDED_BACKTRACK'}
        self.identity = sha(dict(parent_capture=self.identity, program=self.program_data))
        self._program_identity = sha(self.program_data)
        virgin = self.physical.initial
        self.initial, self.genesis = self._record(virgin.mechanical, virgin.histories, 0, 0., 0, self.identity)

    def guard(self):
        self.physical.guard()
        if canonical(self.program.descriptor()) != canonical(self._requested_program):
            raise ValueError('seeded translation program changed')
        if hasattr(self, '_program_identity') and sha(self.program_data) != self._program_identity:
            raise ValueError('seeded controller capture changed')

    def initialize(self, mechanical, origins, target):
        self.guard(); made = self.project(mechanical, target); origins = self.physical.history(origins)
        data = {key: value.copy() for key, value in made.descriptor().items()}
        for index, probe in enumerate(self.physical.probes):
            self.physical.check(); nodes = self.layout.nodes[index]
            # Read only the unchanged native geometric kinematics. Zero force
            # and virgin history avoid solving a possibly far-off-manifold
            # constitutive trial merely to extract geometry. Its material
            # response is discarded. The actual seed below uses accepted origin.
            evaluation = probe.evaluate(made.positions[nodes], made.position_low[nodes], made.nodal_frames[nodes],
                made.cell_rotations[index], np.zeros(18), origin=probe.cell.virgin(), check=self.physical.check)
            seed = kinematic_response(probe.cell, evaluation.kinematics.tolist(), origins[index], check=self.physical.check)
            data['resultants'][index] = seed['resultants'][0]
            # The global state retains binary64 resultants. Verify that this
            # conversion still represents the imposed compatibility to the
            # existing gate, rather than claiming paired values were retained.
            back = probe.cell.response(data['resultants'][index].tolist(), origins[index], check=self.physical.check)
            error = max(abs(fsum((float(k), -h, -l))) for k, h, l in zip(evaluation.kinematics, *back['gradient']))
            if error > 1e-11: raise ValueError('seeded binary64 resultants fail native cell compatibility')
        result = self.layout.make(data); self.guard(); return result


def chart_fraction(context, mechanical, step):
    """Stay inside the unchanged chart using the SO(3) distance triangle bound.

For exp(alpha*a) Q and exp(alpha*b) U, their relative geodesic angle is no
larger than its current angle plus alpha*(|a|+|b|). No frame differentiation,
chart widening, target retry or extra backtracking evaluation is involved.
"""
    layout = context.layout; limit = .9*np.pi; fraction = 1.
    if step.shape != (layout.count,) or not np.isfinite(step).all(): raise ValueError('finite complete control step required')
    for index, probe in enumerate(context.physical.probes):
        nodes = layout.nodes[index]; first = layout.nodal_count+24*index
        for cell, pair_nodes in enumerate(HALVES):
            spin = float(np.linalg.norm(step[first+3*cell:first+3*cell+3]))
            if spin: fraction = min(fraction, .5*limit/spin)
            for local in pair_nodes:
                node = nodes[local]; turn = float(np.linalg.norm(step[6*node+3:6*node+6]))
                if turn: fraction = min(fraction, .5*limit/turn)
                frame = mechanical.cell_rotations[index, cell]@probe.reference.nodal_triads[local]
                relative = frame.T@mechanical.nodal_frames[node]
                angle = float(np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.)))
                margin = limit-angle
                if margin <= 0: raise ValueError('accepted principal chart has no margin')
                if turn+spin: fraction = min(fraction, .5*margin/(turn+spin))
    if not isfinite(fraction) or not 0 < fraction <= 1.: raise ValueError('unresolved chart-safe fraction')
    return fraction


def spatial_jacobian(layout, residual, energy_hessian):
    """D M = H_exp - 1/2 skew(M) in each spatial rotational block.

grad_delta Pi(exp(delta) Q) = J_left(delta).T M(exp(delta) Q).
At zero delta, differentiating the Jacobian transpose gives +1/2 skew(M)
in the chart Hessian. Remove it for the derivative of the spatial residual
used by this controller. The symmetric energy Hessian is not modified.
"""
    result = energy_hessian.copy()
    blocks = [6*i+3 for i in range(len(layout.node_ids))]
    blocks += [layout.nodal_count+24*i+3*j for i in range(len(layout.elements)) for j in (0, 1)]
    for start in blocks:
        result[start:start+3, start:start+3] -= .5*skew(residual[start:start+3])
    return result


def solve_translation_program(model, program, *, checkpoint=None, expected_checkpoint_sha256=None,
                              stop_after=None, cancellation_token=None, progress=None, max_coordinates=256):
    started = monotonic(); cancellation_safe_point(cancellation_token, 'seeded-fibre-control.start')
    context = Context(model, program, check=lambda: cancellation_safe_point(cancellation_token, 'seeded-fibre-control.material'),
                      max_coordinates=max_coordinates)
    layout = context.layout; end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets): raise ValueError('bounded control stop target')
    if progress is not None and not callable(progress): raise ValueError('callable control observer required')
    if checkpoint is None:
        if expected_checkpoint_sha256 is not None: raise ValueError('external hash requires a control checkpoint')
        accepted, records = context.initial, (); capsule = context.checkpoint(records)
    else:
        accepted, records = context.restore(checkpoint, expected_sha256=expected_checkpoint_sha256); capsule = checkpoint
    if accepted.completed_targets > end: raise ValueError('cannot rewind controlled fibre history')
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None
    def safe(stage, target, iteration=0, **diagnostics):
        cancellation_safe_point(cancellation_token, stage)
        if monotonic()-started > 120.: raise RuntimeError('fibre control cooperative deadline')
        context.guard()
        if progress is not None: progress(dict(stage=stage, target=target, iteration=iteration, **diagnostics))
        context.guard(); cancellation_safe_point(cancellation_token, stage)
    try:
        safe('seeded-fibre-control.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            target = program.targets[index]; origins = accepted.histories; parameter = accepted.parameter
            trial = context.initialize(accepted.mechanical, origins, target)
            for iteration in range(program.max_iterations+1):
                safe('seeded-fibre-control.before_assembly', index+1, iteration)
                residual, hessian, metrics, _ = context.assemble(trial, parameter, origins, target)
                safe('seeded-fibre-control.iteration', index+1, iteration, parameter=parameter, metrics=metrics)
                if max(metrics) <= 1e-11:
                    proposed, record = context.stage(trial, parameter, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('seeded-fibre-control.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('seeded-fibre-control.committed', accepted.completed_targets, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('fibre control Newton limit')
                safe('seeded-fibre-control.before_factorization', index+1, iteration)
                augmented = bordered(spatial_jacobian(layout, residual, hessian), context.column, context.row, layout.free)
                increment = np.linalg.solve(augmented, -np.r_[residual[layout.free], context.value(trial)-target])
                if not np.isfinite(increment).all(): raise ValueError('nonfinite fibre control Newton step')
                natural_norm = float(np.linalg.norm(increment))
                if not isfinite(natural_norm) or natural_norm == 0.: raise ValueError('unresolved nonconverged control correction')
                step = np.zeros(layout.count); step[layout.free] = increment[:-1]
                initial_fraction = chart_fraction(context, trial, step)
                for cut in range(program.max_backtracks+1):
                    safe('seeded-fibre-control.before_trial', index+1, iteration)
                    fraction = initial_fraction*(.5**cut)
                    candidate = context.project(layout.advance(trial, step*fraction), target)
                    next_parameter = float(parameter+increment[-1]*fraction)
                    changed_residual, _, changed, _ = context.assemble(candidate, next_parameter, origins, target)
                    # A frozen full-border Newton correction measures the
                    # trial defect in the same unknown space as the step.
                    # Raw force and compatibility components have different
                    # units; directly comparing their maxima can stall a
                    # valid first load step. Final physical gates are unchanged.
                    natural = np.linalg.solve(augmented, np.r_[changed_residual[layout.free], context.value(candidate)-target])
                    merit = float(np.linalg.norm(natural))
                    if not isfinite(merit): raise ValueError('nonfinite natural control merit')
                    if max(changed) <= 1e-11 or merit < natural_norm:
                        trial, parameter = candidate, next_parameter; break
                else: raise RuntimeError('fibre control line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_targets, accepted, capsule, failure)
