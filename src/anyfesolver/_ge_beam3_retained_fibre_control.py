"""Private physical-translation control of the retained physical-fibre beam.

Solve [H,-f; a.T,0] for the full mixed state and common dead-load parameter.
No K-inverse load sensitivity, history advancement in rejected trials, public
selector, automatic cutback or claim of general post-buckling qualification.
"""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import isfinite
from time import monotonic
import numpy as np

from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_retained_fibre_state import Context as FibreContext, State as FibreState
from ._ge_beam3_seeded_load_program import ForceProgram
from ._ge_beam3_retained_fibre import POLICY
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._native_reference_modal import _owned


SCHEMA = 'GE_BEAM3_PHYSICAL_FIBRE_TRANSLATION_CONTROL_CHAIN_V1'
PROGRAM = 'GE_BEAM3_PHYSICAL_FIBRE_TRANSLATION_CONTROL_V1'


@dataclass(frozen=True)
class TranslationProgram:
    targets: tuple
    control_node: int
    direction: tuple
    nodal_forces: tuple
    max_iterations: int = 24
    max_backtracks: int = 8

    def __post_init__(self):
        ForceProgram(self.targets, self.nodal_forces, self.max_iterations, self.max_backtracks)
        if any(type(x) is not float for x in self.targets): raise ValueError('binary64 displacement targets required')
        if type(self.control_node) is not int or self.control_node <= 0: raise ValueError('positive control node required')
        if (type(self.direction) is not tuple or len(self.direction) != 3
                or any(type(x) is not float or not isfinite(x) for x in self.direction)
                or abs(np.linalg.norm(self.direction)-1.) > 1e-12):
            raise ValueError('explicit physical unit control direction required')

    def descriptor(self):
        return dict(schema=PROGRAM, targets=self.targets, control_node=self.control_node, direction=self.direction,
            nodal_forces=self.nodal_forces, max_iterations=self.max_iterations, max_backtracks=self.max_backtracks,
            load_policy='SPATIAL_DEAD_NODAL_FORCE_PATTERN', tolerance=1e-11,
            control='DIRECTION_DOT_PAIRED_POSITION_MINUS_REFERENCE', line_search='PROJECTED_FROZEN_BORDER_NEWTON_CORRECTION_DECREASE')


@dataclass(frozen=True)
class State:
    mechanical: object
    origins: tuple
    histories: tuple
    completed_targets: int
    parameter: float
    model_sha256: str


class Context:
    def __init__(self, model, program, *, check=None, max_coordinates=256):
        if type(program) is not TranslationProgram: raise ValueError('exact physical translation program required')
        self.program = program; self.program_data = program.descriptor()
        # Capture a load pattern, not a force-controlled execution or history.
        self.physical = FibreContext(model, ForceProgram((0.,), program.nodal_forces,
            program.max_iterations, program.max_backtracks), check=check, max_coordinates=max_coordinates)
        self.layout = self.physical.layout
        if program.control_node not in self.layout.node_index: raise ValueError('control node absent')
        self.node = self.layout.node_index[program.control_node]
        slots = list(range(6*self.node, 6*self.node+3))
        if any(i in self.layout.fixed for i in slots): raise ValueError('control node translations must be free')
        self.row = np.zeros(self.layout.count); self.row[slots] = program.direction; self.row = _owned(self.row)
        self.column = np.zeros(self.layout.count); self.column[:self.layout.nodal_count] = -self.layout.force
        self.column = _owned(self.column)
        if not np.any(self.column[self.layout.free]): raise ValueError('nonzero free dead-load pattern required')
        self.identity = sha(dict(schema=SCHEMA, formulation_id=POLICY, physical_capture=self.physical.identity,
            program=self.program_data, control_row=self.row, parameter_column=self.column))
        virgin = self.physical.initial
        self.initial, self.genesis = self._record(virgin.mechanical, virgin.histories, 0, 0., 0, self.identity)

    def guard(self):
        self.physical.guard()
        if canonical(self.program.descriptor()) != canonical(self.program_data): raise ValueError('translation program changed')

    def value(self, mechanical):
        # Three exact dyadic products, one final rounding: large common
        # translations must not erase the controlled small displacement.
        node = self.node
        answer = float(sum((Fraction(d)*(Fraction(float(h))+Fraction(float(l))-Fraction(float(r)))
            for d, h, l, r in zip(self.program.direction, mechanical.positions[node],
                mechanical.position_low[node], self.layout.reference_positions[node])), Fraction(0)))
        if not isfinite(answer): raise ValueError('nonfinite translation control')
        return answer

    def project(self, mechanical, target):
        step = np.zeros(self.layout.count)
        step[:] = self.row*((target-self.value(mechanical))/float(self.row@self.row))
        made = self.layout.advance(mechanical, step)
        if abs(self.value(made)-target) > 1e-11*max(1., abs(target)): raise ValueError('translation plane projection failed')
        return made

    def assemble(self, mechanical, parameter, origins, target):
        if type(parameter) is not float or not isfinite(parameter): raise ValueError('finite binary64 load parameter required')
        residual, hessian, metrics, responses = self.physical.assemble(mechanical, parameter, origins)
        control = abs(self.value(mechanical)-target)/max(1., abs(target))
        return residual, hessian, (*metrics, control), responses

    def recover(self, state):
        if type(state) is not State or state.model_sha256 != self.identity: raise ValueError('control recovery state/model binding')
        return self.physical.recover(FibreState(state.mechanical, state.origins, state.histories,
            state.completed_targets, self.physical.identity))

    def _record(self, mechanical, origins, cursor, parameter, iterations, previous):
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets): raise ValueError('control target cursor')
        if type(iterations) is not int or not 0 <= iterations <= self.program.max_iterations: raise ValueError('control iteration record')
        if cursor == 0 and parameter != 0.: raise ValueError('virgin parameter is zero')
        target = 0. if cursor == 0 else self.program.targets[cursor-1]
        mechanical = self.layout.make(mechanical.descriptor()); origins = self.physical.history(origins)
        residual, _, metrics, responses = self.assemble(mechanical, parameter, origins, target)
        if max(metrics) > 1e-11: raise ValueError('only equilibrated compatible controlled states may commit')
        histories = self.physical.history(tuple(r.history for r in responses))
        state = State(mechanical, origins, histories, cursor, parameter, self.identity)
        body = dict(target=cursor, displacement_target=target, parameter=parameter, iterations=iterations,
            previous_sha256=previous, mechanical=mechanical.descriptor(), origins=origins, histories=histories,
            residual=residual, metrics=metrics, control_value=self.value(mechanical),
            material_sha256=sha([r.material.decode('ascii') for r in responses]), recovery_sha256=sha(self.recover(state)))
        self.guard(); return state, canonical({**body, 'record_sha256': sha(body)})

    def stage(self, mechanical, parameter, accepted, records, iterations):
        if (type(records) is not tuple or type(accepted) is not State or accepted.model_sha256 != self.identity
                or type(accepted.completed_targets) is not int or accepted.completed_targets != len(records)):
            raise ValueError('control accepted state/cursor differs from chain')
        anchor = _load((self.genesis if not records else records[-1]).decode('ascii'))
        if (type(accepted.parameter) is not float or accepted.parameter != anchor['parameter']
                or canonical(accepted.mechanical.descriptor()) != canonical(anchor['mechanical'])
                or canonical(accepted.origins) != canonical(anchor['origins'])
                or canonical(accepted.histories) != canonical(anchor['histories'])):
            raise ValueError('accepted control state differs from previous record')
        previous = anchor['record_sha256']
        return self._record(mechanical, accepted.histories, len(records)+1, parameter, iterations, previous)

    def checkpoint(self, records):
        if type(records) is not tuple or len(records) > len(self.program.targets): raise ValueError('bounded immutable control chain required')
        body = dict(schema=SCHEMA, formulation_id=POLICY, model_sha256=self.identity, program=self.program_data,
            node_ids=self.layout.node_ids, element_ids=[i for i, _ in self.layout.elements], completed_targets=len(records),
            initial=_load(self.genesis.decode('ascii')), records=[_load(row.decode('ascii')) for row in records])
        raw = canonical({**body, 'checkpoint_sha256': sha(body)})
        if len(raw) > MAX_BYTES: raise ValueError('control checkpoint byte bound')
        self.guard(); return raw

    def restore(self, raw, *, expected_sha256=None):
        self.guard()
        if type(raw) is not bytes: raise ValueError('canonical control checkpoint bytes required')
        if expected_sha256 is not None and (type(expected_sha256) is not str or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError('external control checkpoint hash mismatch')
        value = _load(raw.decode('ascii'))
        keys = {'schema', 'formulation_id', 'model_sha256', 'program', 'node_ids', 'element_ids',
            'completed_targets', 'initial', 'records', 'checkpoint_sha256'}
        if type(value) is not dict or set(value) != keys: raise ValueError('exact control checkpoint schema')
        body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
        if (value['schema'] != SCHEMA or value['formulation_id'] != POLICY or value['model_sha256'] != self.identity
                or value['checkpoint_sha256'] != sha(body) or canonical(value['program']) != canonical(self.program_data)
                or canonical(value['node_ids']) != canonical(self.layout.node_ids)
                or canonical(value['element_ids']) != canonical([i for i, _ in self.layout.elements])):
            raise ValueError('control checkpoint model/program/formulation binding')
        cursor = value['completed_targets']
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets): raise ValueError('control checkpoint cursor')
        if type(value['records']) is not list or len(value['records']) != cursor: raise ValueError('control record count')
        if canonical(value['initial']) != self.genesis: raise ValueError('control virgin genesis differs')
        accepted = self.initial; records = ()
        record_keys = {'target', 'displacement_target', 'parameter', 'iterations', 'previous_sha256', 'mechanical', 'origins',
            'histories', 'residual', 'metrics', 'control_value', 'material_sha256', 'recovery_sha256', 'record_sha256'}
        for index, row in enumerate(value['records'], 1):
            self.guard()
            if (type(row) is not dict or set(row) != record_keys or type(row['target']) is not int or row['target'] != index
                    or type(row['displacement_target']) is not float or row['displacement_target'] != self.program.targets[index-1]):
                raise ValueError('control checkpoint target record')
            origins = self.physical.history(row['origins'], decoded=True); self.physical.history(row['histories'], decoded=True)
            if canonical(origins) != canonical(accepted.histories): raise ValueError('control origin chain is broken')
            mechanical = self.layout.make(row['mechanical'], decoded=True)
            proposed, regenerated = self.stage(mechanical, row['parameter'], accepted, records, row['iterations'])
            if regenerated != canonical(row): raise ValueError('control state/reaction/recovery/history replay mismatch')
            accepted, records = proposed, (*records, regenerated)
        if self.checkpoint(records) != raw: raise ValueError('control canonical replay mismatch')
        self.guard(); return accepted, records


def bordered(hessian, column, row, free):
    """Full border remains available when the unbordered tangent is singular."""
    return np.block([[hessian[np.ix_(free, free)], column[free, None]],
        [row[None, free], np.zeros((1, 1))]])


@dataclass(frozen=True)
class Result:
    status: str
    completed_targets: int
    state: State
    checkpoint: bytes
    failure: str | None
    production_qualified: bool = False


def solve_translation_program(model, program, *, checkpoint=None, expected_checkpoint_sha256=None,
                              stop_after=None, cancellation_token=None, progress=None):
    started = monotonic(); cancellation_safe_point(cancellation_token, 'fibre-control.start')
    context = Context(model, program, check=lambda: cancellation_safe_point(cancellation_token, 'fibre-control.material'))
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
        safe('fibre-control.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            target = program.targets[index]; origins = accepted.histories; parameter = accepted.parameter
            trial = context.project(accepted.mechanical, target)
            for iteration in range(program.max_iterations+1):
                safe('fibre-control.before_assembly', index+1, iteration)
                residual, hessian, metrics, _ = context.assemble(trial, parameter, origins, target)
                safe('fibre-control.iteration', index+1, iteration, parameter=parameter, metrics=metrics)
                if max(metrics) <= 1e-11:
                    proposed, record = context.stage(trial, parameter, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('fibre-control.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('fibre-control.committed', accepted.completed_targets, iteration)
                    break
                if iteration == program.max_iterations: raise RuntimeError('fibre control Newton limit')
                safe('fibre-control.before_factorization', index+1, iteration)
                augmented = bordered(hessian, context.column, context.row, layout.free)
                increment = np.linalg.solve(augmented, -np.r_[residual[layout.free], context.value(trial)-target])
                if not np.isfinite(increment).all(): raise ValueError('nonfinite fibre control Newton step')
                natural_norm = float(np.linalg.norm(increment))
                if not isfinite(natural_norm) or natural_norm == 0.: raise ValueError('unresolved nonconverged control correction')
                step = np.zeros(layout.count); step[layout.free] = increment[:-1]
                for cut in range(program.max_backtracks+1):
                    safe('fibre-control.before_trial', index+1, iteration)
                    candidate = context.project(layout.advance(trial, step*(.5**cut)), target)
                    next_parameter = float(parameter+increment[-1]*(.5**cut))
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
