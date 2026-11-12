"""Private retained generalized translation control; no public routing.

The physical retained operator is unchanged. A distinct issued state owns the
accepted dead-load factor and physical displacement target. Force-program
capsules cannot be reinterpreted as controlled histories.
"""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import isfinite
import numpy as np

from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_retained_nodal_loading import Context as PhysicalContext, Program as ForceProgram, NodalDeadForces
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._native_reference_modal import _owned
from ._ge_beam3_operator_validation_scope import operator_call

SCHEMA = 'GE_BEAM3_RETAINED_GENERALIZED_TRANSLATION_ACCEPTED_CHAIN_V1'
POLICY = 'CANDIDATE_GE_BEAM3_RETAINED_GENERALIZED_TRANSLATION_V1'


@dataclass(frozen=True)
class Program:
    targets: tuple
    control_node: int
    direction: tuple
    nodal_forces: NodalDeadForces
    max_iterations: int = 24
    max_backtracks: int = 8

    def __post_init__(self):
        ForceProgram(self.targets, DistributedPattern(LinePattern(()), ()), self.nodal_forces,
                     self.max_iterations, self.max_backtracks)
        if type(self.control_node) is not int or self.control_node <= 0:
            raise ValueError('positive physical control node required')
        if (type(self.direction) is not tuple or len(self.direction) != 3
                or any(type(v) is not float or not isfinite(v) for v in self.direction)
                or abs(np.linalg.norm(self.direction)-1.) > 1e-12):
            raise ValueError('finite physical unit control direction required')


@dataclass(frozen=True)
class State:
    mechanical: object
    origins: tuple
    histories: tuple
    completed_targets: int
    parameter: float
    model_sha256: str


def bordered(jacobian, column, row, free):
    """Full retained border; no inverse of the unbordered tangent."""
    return np.block([[jacobian[np.ix_(free, free)], column[free, None]],
                     [row[None, free], np.zeros((1, 1))]])


class Context:
    def __init__(self, model, program):
        if type(program) is not Program:
            raise ValueError('exact retained translation program required')
        program.__post_init__()
        self.program = program
        self.program_bytes = canonical(program)
        # This captures only the physical virgin state and load pattern. No
        # force-controlled target is executed or imported as a control record.
        self.physical = PhysicalContext(model, ForceProgram((0.,),
            DistributedPattern(LinePattern(()), ()), program.nodal_forces,
            program.max_iterations, program.max_backtracks))
        p = self.physical
        if program.control_node not in p.index:
            raise ValueError('control node absent')
        self.node = p.index[program.control_node]
        slots = list(model.mesh.dof_manager.get_node_dofs(program.control_node)[:3])
        if slots != list(range(6*self.node, 6*self.node+3)):
            raise ValueError('physical control DOF ordering differs')
        if any(v in p.fixed for v in slots):
            raise ValueError('control node translations must all be free')
        row = np.zeros(p.count); row[slots] = program.direction
        column = np.zeros(p.count); column[:p.nodal_count] = -p.nodal_external(1.)
        self.row, self.column = _owned(row), _owned(column)
        if not np.any(column[p.free]):
            raise ValueError('nonzero free dead-force pattern required')
        self.map_bytes = canonical(dict(node=self.node, row=self.row, column=self.column))
        self.identity = sha(dict(schema=SCHEMA, formulation=POLICY, physical_capture=p.identity,
                                 program=program, maps=self.map_bytes.decode()))
        self._issued = {}; self._issued_records = set()
        self.initial, self.genesis = self._record(p.initial.mechanical, p.initial.histories,
                                                 0, 0., 0, self.identity)

    def guard(self):
        self.physical.guard()
        if (canonical(self.program) != self.program_bytes
                or canonical(dict(node=self.node, row=self.row, column=self.column)) != self.map_bytes):
            raise ValueError('retained translation program/control maps changed')

    def value(self, mechanical):
        # Exact dyadic products and one rounding preserve small controls in
        # the presence of a large common translation.
        answer = float(sum((Fraction(d)*(Fraction(float(h))+Fraction(float(l))-Fraction(float(r)))
            for d, h, l, r in zip(self.program.direction, mechanical.positions[self.node],
                mechanical.position_low[self.node], self.physical.reference_positions[self.node])), Fraction(0)))
        if not isfinite(answer):
            raise ValueError('nonfinite physical translation')
        return answer

    def project(self, mechanical, target):
        self.guard()
        step = self.row*((target-self.value(mechanical))/float(self.row@self.row))
        result = self.physical.advance(mechanical, step)
        if abs(self.value(result)-target)/max(self.physical.length, abs(target)) > 1e-11:
            raise ValueError('physical translation projection failed')
        return result

    def assemble(self, mechanical, parameter, origins, target):
        self.guard()
        if type(parameter) is not float or not isfinite(parameter):
            raise ValueError('finite binary64 control load factor required')
        if type(target) is not float or not isfinite(target):
            raise ValueError('finite binary64 physical target required')
        r, j, metrics, responses, work = self.physical.assemble(mechanical, parameter, origins)
        error = abs(self.value(mechanical)-target)/max(self.physical.length, abs(target))
        return r, j, (*metrics, error), responses, work

    def correction_norm(self, mechanical, parameter, step, delta_parameter):
        p = self.physical; nodal = step[:p.nodal_count].reshape(-1, 6)
        answer = max(float(np.linalg.norm(nodal[:, :3]))/p.length,
                     float(np.linalg.norm(nodal[:, 3:])), abs(delta_parameter)/max(1., abs(parameter)))
        for i in range(len(p.elements)):
            first = p.nodal_count+24*i
            answer = max(answer, float(np.linalg.norm(step[first:first+6])),
                float(np.linalg.norm(step[first+6:first+24]))/max(1., float(np.linalg.norm(mechanical.resultants[i]))))
        if not isfinite(answer):
            raise ValueError('nonfinite bordered correction norm')
        return answer

    def step(self, mechanical, parameter, r, j, target):
        p = self.physical
        matrix = bordered(j, self.column, self.row, p.free)
        correction = np.linalg.solve(matrix, -np.r_[r[p.free], self.value(mechanical)-target])
        if not np.isfinite(correction).all():
            raise ValueError('nonfinite bordered Newton correction')
        step = np.zeros(p.count); step[p.free] = correction[:-1]
        delta = float(correction[-1])
        return step, delta, self.correction_norm(mechanical, parameter, step, delta), matrix

    def _require_issued(self, state, *, expected_snapshot=None):
        if expected_snapshot is not None and type(expected_snapshot) is not bytes:
            raise ValueError('exact expected controlled snapshot bytes required')
        self.guard(); bound = self._issued.get(id(state))
        if (type(state) is not State or bound is None or bound[0] is not state
                or state.model_sha256 != self.identity):
            raise ValueError('controlled state was not issued by this context')
        encoded=canonical(state)
        if encoded != bound[1] or (expected_snapshot is not None and encoded != expected_snapshot):
            raise ValueError('issued controlled state changed')
        return bound[2]

    def _recover_validated(self, state):
        self.guard(); rows = []
        for i, (eid, element) in enumerate(self.physical.elements):
            stations = operator_call(element.operator, 'recover', self.guard, self.physical.started,
                state.mechanical.cell_rotations[i], state.mechanical.resultants[i], origin=state.origins[i])
            if canonical([row['history'] for row in stations]) != canonical(state.histories[i].stations):
                raise ValueError('controlled recovery history mismatch')
            rows.append(dict(element_id=eid, stations=stations))
        self.guard(); return tuple(rows)

    def recover(self, state):
        self._require_issued(state); recovered = self._recover_validated(state)
        self._require_issued(state); return recovered

    def _record(self, mechanical, origins, cursor, parameter, iterations, previous):
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets):
            raise ValueError('control record cursor')
        if type(iterations) is not int or not 0 <= iterations <= self.program.max_iterations:
            raise ValueError('control iteration record')
        if cursor == 0 and parameter != 0.:
            raise ValueError('virgin control load factor must be zero')
        target = 0. if cursor == 0 else self.program.targets[cursor-1]
        p = self.physical; mechanical = p.make(mechanical.descriptor()); origins = p.histories(origins)
        r, j, metrics, responses, work = self.assemble(mechanical, parameter, origins, target)
        _, _, correction, _ = self.step(mechanical, parameter, r, j, target)
        if max(*metrics, correction) > 1e-11:
            raise ValueError('only equilibrated compatible fully converged controlled states may commit')
        histories = p.histories(tuple(response.history for response in responses))
        state = State(mechanical, origins, histories, cursor, parameter, self.identity)
        body = dict(target=cursor, displacement_target=target, parameter=parameter, iterations=iterations,
            previous_sha256=previous, mechanical=mechanical.descriptor(), origins=origins, histories=histories,
            residual=r, metrics=metrics, correction=correction, work=work, control_value=self.value(mechanical),
            material_sha256=sha([a.material.decode() for a in responses]), recovery_sha256=sha(self._recover_validated(state)))
        self.guard(); raw = canonical({**body, 'record_sha256': sha(body)})
        self._issued[id(state)] = (state, canonical(state), raw); self._issued_records.add(raw)
        return state, raw

    def _require_chain(self, records):
        self.guard()
        if type(records) is not tuple or len(records) > len(self.program.targets):
            raise ValueError('bounded issued controlled record chain required')
        previous = _load(self.genesis.decode())
        for index, raw in enumerate(records, 1):
            if type(raw) is not bytes or raw not in self._issued_records:
                raise ValueError('controlled record was not issued by this context')
            row = _load(raw.decode())
            if (row['target'] != index or row['displacement_target'] != self.program.targets[index-1]
                    or row['previous_sha256'] != previous['record_sha256']
                    or canonical(row['origins']) != canonical(previous['histories'])):
                raise ValueError('issued controlled chain is not contiguous')
            previous = row

    def stage(self, mechanical, parameter, accepted, records, iterations):
        issued = self._require_issued(accepted); self._require_chain(records)
        if (issued != (records[-1] if records else self.genesis)
                or accepted.completed_targets != len(records)):
            raise ValueError('controlled state differs from predecessor record')
        previous = _load(issued.decode())['record_sha256']
        return self._record(mechanical, accepted.histories, len(records)+1, parameter, iterations, previous)

    def checkpoint(self, records):
        self._require_chain(records)
        body = dict(schema=SCHEMA, formulation=POLICY, model_sha256=self.identity, program=self.program,
            initial=_load(self.genesis.decode()), records=[_load(raw.decode()) for raw in records], completed_targets=len(records))
        raw = canonical({**body, 'checkpoint_sha256': sha(body)})
        if len(raw) > MAX_BYTES:
            raise ValueError('controlled checkpoint byte bound')
        self.guard(); return raw

    def restore(self, raw, *, expected_sha256):
        self.guard()
        if (type(raw) is not bytes or len(raw) > MAX_BYTES or type(expected_sha256) is not str
                or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError('external controlled checkpoint authority mismatch')
        value = _load(raw.decode())
        keys = {'schema', 'formulation', 'model_sha256', 'program', 'initial', 'records', 'completed_targets', 'checkpoint_sha256'}
        if type(value) is not dict or set(value) != keys:
            raise ValueError('exact controlled checkpoint schema')
        body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
        if (value['schema'] != SCHEMA or value['formulation'] != POLICY or value['model_sha256'] != self.identity
                or canonical(value['program']) != self.program_bytes or value['checkpoint_sha256'] != sha(body)
                or canonical(value['initial']) != self.genesis):
            raise ValueError('controlled checkpoint model/program/genesis binding')
        cursor = value['completed_targets']
        if (type(cursor) is not int or not 0 <= cursor <= len(self.program.targets)
                or type(value['records']) is not list or len(value['records']) != cursor):
            raise ValueError('controlled checkpoint target extent')
        accepted, records = self.initial, ()
        record_keys = set(_load(self.genesis.decode()))
        for index, row in enumerate(value['records'], 1):
            if (type(row) is not dict or set(row) != record_keys or type(row['target']) is not int
                    or row['target'] != index or type(row['displacement_target']) is not float
                    or row['displacement_target'] != self.program.targets[index-1]):
                raise ValueError('exact controlled target record')
            mechanical = self.physical.make(row['mechanical'], decoded=True)
            proposed, regenerated = self.stage(mechanical, row['parameter'], accepted, records, row['iterations'])
            if regenerated != canonical(row):
                raise ValueError('controlled geometry/history/work/recovery replay mismatch')
            accepted, records = proposed, (*records, regenerated)
        if self.checkpoint(records) != raw:
            raise ValueError('noncanonical controlled replay')
        self.guard(); return accepted, records


@dataclass(frozen=True)
class Result:
    status: str
    completed_targets: int
    state: State
    checkpoint: bytes
    failure: str | None
    production_qualified: bool = False


def solve(model, program, *, checkpoint=None, expected_sha256=None, stop_after=None,
          cancellation_token=None, progress=None):
    cancellation_safe_point(cancellation_token, 'retained-translation.start')
    if progress is not None and not callable(progress):
        raise ValueError('callable controlled observer required')
    context = Context(model, program); p = context.physical
    end = len(program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.targets):
        raise ValueError('bounded control stop target')
    if checkpoint is None:
        if expected_sha256 is not None:
            raise ValueError('hash requires a controlled checkpoint')
        accepted, records = context.initial, (); capsule = context.checkpoint(records)
    else:
        accepted, records = context.restore(checkpoint, expected_sha256=expected_sha256); capsule = checkpoint
    if accepted.completed_targets > end:
        raise ValueError('controlled history cannot rewind')
    status = 'completed' if end == len(program.targets) else 'paused'; failure = None

    def safe(stage, target, iteration=0):
        cancellation_safe_point(cancellation_token, stage); context.guard()
        if progress is not None:
            progress(dict(stage=stage, target=target, iteration=iteration))
        context.guard(); cancellation_safe_point(cancellation_token, stage)

    try:
        safe('retained-translation.initialized', accepted.completed_targets)
        for index in range(accepted.completed_targets, end):
            target = program.targets[index]; origins = accepted.histories; parameter = accepted.parameter
            trial = context.project(accepted.mechanical, target)
            for iteration in range(program.max_iterations+1):
                safe('retained-translation.before_assembly', index+1, iteration)
                r, j, metrics, _, _ = context.assemble(trial, parameter, origins, target)
                safe('retained-translation.before_factorization', index+1, iteration)
                step, delta, correction, matrix = context.step(trial, parameter, r, j, target)
                if max(*metrics, correction) <= 1e-11:
                    proposed, record = context.stage(trial, parameter, accepted, records, iteration)
                    next_records = (*records, record); staged = context.checkpoint(next_records)
                    safe('retained-translation.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('retained-translation.committed', index+1, iteration); break
                if iteration == program.max_iterations:
                    raise RuntimeError('retained translation Newton limit')
                angular = [step[6*i+3:6*i+6] for i in range(len(p.node_ids))]
                angular.extend(step[p.nodal_count+24*i+3*c:p.nodal_count+24*i+3*c+3]
                               for i in range(len(p.elements)) for c in (0, 1))
                fraction = min(1., .45*np.pi/max(max(float(np.linalg.norm(v)) for v in angular), np.finfo(float).tiny))
                for cut in range(program.max_backtracks+1):
                    safe('retained-translation.before_trial', index+1, iteration)
                    amount = fraction*.5**cut
                    candidate = context.project(p.advance(trial, step*amount), target)
                    next_parameter = float(parameter+delta*amount)
                    changed, _, _, _, _ = context.assemble(candidate, next_parameter, origins, target)
                    # Compare defects in the SAME scaled unknown space using
                    # the frozen full border, not a residual-only merit.
                    natural = np.linalg.solve(matrix, -np.r_[changed[p.free], context.value(candidate)-target])
                    if not np.isfinite(natural).all():
                        raise ValueError('nonfinite frozen-border merit')
                    trial_step = np.zeros(p.count); trial_step[p.free] = natural[:-1]
                    merit = context.correction_norm(trial, parameter, trial_step, float(natural[-1]))
                    if merit < correction:
                        trial, parameter = candidate, next_parameter; break
                else:
                    raise RuntimeError('retained translation line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_targets, accepted, capsule, failure)
