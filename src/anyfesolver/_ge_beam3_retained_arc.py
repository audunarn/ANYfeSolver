"""Private retained frame-chord predictor/corrector and owned accepted chain.

Full mixed borders are solved even at a singular unbordered equilibrium
tangent. This candidate supplies no public route or postbuckling qualification.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
import numpy as np
from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_retained_nodal_loading import Context as PhysicalContext, Program as ForceProgram, NodalDeadForces
from ._ge_beam3_native_generalized_loading import DistributedPattern
from ._ge_beam3_native_line_loading import LinePattern
from ._ge_beam3_retained_arc_geometry import constraint, POLICY as GEOMETRY
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._native_reference_modal import _owned

SCHEMA = 'GE_BEAM3_RETAINED_FRAME_CHORD_ACCEPTED_CHAIN_V1'
POLICY = 'CANDIDATE_GE_BEAM3_RETAINED_FRAME_CHORD_ARC_V1'


def inverse_square(value):
    if type(value) is not float or not isfinite(value) or value <= 0.:
        raise ValueError('positive finite explicit arc scale required')
    inverse = 1./value
    result = inverse*inverse
    if not isfinite(result) or result <= 0.:
        raise ValueError('arc metric scale range')
    return result


@dataclass(frozen=True)
class Program:
    steps: tuple
    length_scale: float
    nodal_forces: NodalDeadForces
    parameter_scale: float = 1.
    initial_sign: float = 1.
    max_iterations: int = 24
    max_backtracks: int = 8

    def __post_init__(self):
        if (type(self.steps) is not tuple or not 1 <= len(self.steps) <= 16
                or any(type(v) is not float or not isfinite(v) or not 0. < v <= .25 for v in self.steps)):
            raise ValueError('one to sixteen explicit bounded positive arc steps required')
        inverse_square(self.length_scale); inverse_square(self.parameter_scale)
        if type(self.initial_sign) is not float or self.initial_sign not in (-1., 1.):
            raise ValueError('explicit arc orientation sign required')
        ForceProgram((0.,), DistributedPattern(LinePattern(()), ()), self.nodal_forces,
                     self.max_iterations, self.max_backtracks)


@dataclass(frozen=True)
class State:
    mechanical: object
    origins: tuple
    histories: tuple
    completed_steps: int
    parameter: float
    predictor: np.ndarray
    model_sha256: str


def bordered(jacobian, column, row, free):
    return np.block([[jacobian[np.ix_(free, free)], column[free, None]],
                     [row[None, free], np.array([[row[-1]]])]])


def direction(jacobian, column, free, previous, metric):
    """Oriented full bordered tangent, never an inverse-force sensitivity."""
    row = metric*previous; rhs = np.zeros(len(free)+1); rhs[-1] = 1.
    solution = np.linalg.solve(bordered(jacobian, column, row, free), rhs)
    value = np.zeros(len(metric)); value[free] = solution[:-1]; value[-1] = solution[-1]
    norm = float(np.sum(metric*value*value))
    if not np.isfinite(value).all() or not isfinite(norm) or norm <= 0.:
        raise ValueError('unresolved retained arc predictor')
    value /= sqrt(norm)
    if not np.isfinite(value).all() or float(row@value) <= 0.:
        raise ValueError('unresolved retained arc orientation')
    return _owned(value)


class Context:
    def __init__(self, model, program):
        if type(program) is not Program:
            raise ValueError('exact retained arc program required')
        program.__post_init__(); self.program = program; self.program_bytes = canonical(program)
        # Capture operators, virgin state and load pattern only; no force path is run.
        self.physical = PhysicalContext(model, ForceProgram((0.,), DistributedPattern(LinePattern(()), ()),
            program.nodal_forces, program.max_iterations, program.max_backtracks))
        p = self.physical; n = len(p.node_ids)
        metric = np.zeros(p.count+1)
        metric[:p.nodal_count].reshape(n, 6)[:, :3] = inverse_square(program.length_scale)/n
        metric[:p.nodal_count].reshape(n, 6)[:, 3:] = 1./n
        metric[-1] = inverse_square(program.parameter_scale)
        if np.any(metric[:p.nodal_count] <= 0.):
            raise ValueError('physical metric underflow')
        column = np.zeros(p.count); column[:p.nodal_count] = -p.nodal_external(1.)
        if not np.any(column[p.free]):
            raise ValueError('nonzero free dead-force pattern required')
        initial = np.zeros(p.count+1); initial[-1] = program.initial_sign*program.parameter_scale
        self.metric, self.column, self.orientation = map(_owned, (metric, column, initial))
        self.map_bytes = canonical(dict(metric=self.metric, column=self.column, orientation=self.orientation))
        self.identity = sha(dict(schema=SCHEMA, policy=POLICY, geometry=GEOMETRY,
            physical_capture=p.identity, program=program, maps=self.map_bytes.decode()))
        self._issued = {}; self._issued_records = set()
        self.initial, self.genesis = self._record(p.initial.mechanical, 0., None, initial, 0, self.identity)

    def guard(self):
        self.physical.guard()
        if (canonical(self.program) != self.program_bytes or canonical(dict(metric=self.metric,
                column=self.column, orientation=self.orientation)) != self.map_bytes):
            raise ValueError('retained arc program/maps changed')

    def _require_issued(self, state):
        self.guard(); bound = self._issued.get(id(state))
        if type(state) is not State or bound is None or bound[0] is not state or state.model_sha256 != self.identity:
            raise ValueError('arc state was not issued by this context')
        if canonical(state) != bound[1]:
            raise ValueError('issued arc state changed')
        return bound[2]

    def predictor(self, accepted):
        self._require_issued(accepted)
        _, j, metrics, _, _ = self.physical.assemble(accepted.mechanical, accepted.parameter, accepted.histories)
        if max(metrics) > 1e-11:
            raise ValueError('arc predictor origin is not equilibrated')
        return direction(j, self.column, self.physical.free, accepted.predictor, self.metric)

    def row(self, mechanical, parameter, accepted, predictor):
        self.guard()
        return constraint(mechanical, accepted.mechanical, predictor, self.metric, parameter,
                          accepted.parameter, self.program.steps[accepted.completed_steps])

    def correction_norm(self, mechanical, parameter, step, delta):
        p = self.physical; nodal = step[:p.nodal_count].reshape(-1, 6)
        result = max(float(np.linalg.norm(nodal[:, :3]))/p.length,
            float(np.linalg.norm(nodal[:, 3:])), abs(delta)/max(1., abs(parameter)))
        for i in range(len(p.elements)):
            first = p.nodal_count+24*i
            result = max(result, float(np.linalg.norm(step[first:first+6])),
                float(np.linalg.norm(step[first+6:first+24]))/max(1., float(np.linalg.norm(mechanical.resultants[i]))))
        if not isfinite(result):
            raise ValueError('nonfinite full arc correction')
        return result

    def correction(self, mechanical, parameter, residual, jacobian, gap, row):
        p = self.physical; matrix = bordered(jacobian, self.column, row, p.free)
        answer = np.linalg.solve(matrix, -np.r_[residual[p.free], gap])
        if not np.isfinite(answer).all():
            raise ValueError('nonfinite arc Newton correction')
        step = np.zeros(p.count); step[p.free] = answer[:-1]; delta = float(answer[-1])
        return step, delta, self.correction_norm(mechanical, parameter, step, delta), matrix

    def _recover_validated(self, state):
        self.guard(); rows = []
        for i, (eid, element) in enumerate(self.physical.elements):
            stations = element.operator.recover(state.mechanical.cell_rotations[i], state.mechanical.resultants[i],
                origin=state.origins[i], check=self.guard)
            if canonical([r['history'] for r in stations]) != canonical(state.histories[i].stations):
                raise ValueError('arc recovery history mismatch')
            rows.append(dict(element_id=eid, stations=stations))
        self.guard(); return tuple(rows)

    def recover(self, state):
        self._require_issued(state); result = self._recover_validated(state)
        self._require_issued(state); return result

    def _record(self, mechanical, parameter, accepted, predictor, iterations, previous):
        self.guard(); p = self.physical
        if type(parameter) is not float or not isfinite(parameter):
            raise ValueError('finite binary64 arc load factor required')
        if type(iterations) is not int or not 0 <= iterations <= self.program.max_iterations:
            raise ValueError('bounded arc iteration count required')
        mechanical = p.make(mechanical.descriptor())
        cursor = 0 if accepted is None else accepted.completed_steps+1
        origins = p.initial.histories if accepted is None else accepted.histories
        r, j, metrics, responses, work = p.assemble(mechanical, parameter, origins)
        if accepted is None:
            gap, row, size = 0., self.metric*self.orientation, 0.
        else:
            gap, row = self.row(mechanical, parameter, accepted, predictor)
            size = self.program.steps[cursor-1]
        _, _, remaining, _ = self.correction(mechanical, parameter, r, j, gap, row)
        if max(*metrics, abs(gap), remaining) > 1e-11:
            raise ValueError('only fully converged compatible arc states may commit')
        histories = p.histories(tuple(response.history for response in responses))
        state = State(mechanical, origins, histories, cursor, parameter, _owned(predictor), self.identity)
        orientation = float((self.metric*(self.orientation if accepted is None else accepted.predictor))@predictor)
        body = dict(step=cursor, step_size=size, parameter=parameter, predictor=state.predictor,
            orientation=orientation, iterations=iterations, previous_sha256=previous,
            mechanical=mechanical.descriptor(), origins=origins, histories=histories,
            residual=r, metrics=metrics, arc_gap=gap, correction=remaining, work=work,
            material_sha256=sha([response.material.decode() for response in responses]),
            recovery_sha256=sha(self._recover_validated(state)))
        self.guard(); raw = canonical({**body, 'record_sha256': sha(body)})
        self._issued[id(state)] = (state, canonical(state), raw); self._issued_records.add(raw)
        return state, raw

    def _require_chain(self, records):
        self.guard()
        if type(records) is not tuple or len(records) > len(self.program.steps):
            raise ValueError('bounded issued arc chain required')
        previous = _load(self.genesis.decode())
        for index, raw in enumerate(records, 1):
            if type(raw) is not bytes or raw not in self._issued_records:
                raise ValueError('arc record was not issued by this context')
            row = _load(raw.decode())
            if (row['step'] != index or row['step_size'] != self.program.steps[index-1]
                    or row['previous_sha256'] != previous['record_sha256']
                    or canonical(row['origins']) != canonical(previous['histories'])):
                raise ValueError('issued arc chain is not contiguous')
            previous = row

    def stage(self, mechanical, parameter, accepted, records, iterations):
        issued = self._require_issued(accepted); self._require_chain(records)
        if issued != (records[-1] if records else self.genesis) or accepted.completed_steps != len(records):
            raise ValueError('arc state differs from predecessor record')
        if len(records) == len(self.program.steps):
            raise ValueError('arc program is already complete')
        predictor = self.predictor(accepted)  # Never trust a serialized/supplied predictor.
        previous = _load(issued.decode())['record_sha256']
        return self._record(mechanical, parameter, accepted, predictor, iterations, previous)

    def checkpoint(self, records):
        self._require_chain(records)
        body = dict(schema=SCHEMA, formulation=POLICY, geometry=GEOMETRY, model_sha256=self.identity,
            program=self.program, initial=_load(self.genesis.decode()),
            records=[_load(r.decode()) for r in records], completed_steps=len(records))
        raw = canonical({**body, 'checkpoint_sha256': sha(body)})
        if len(raw) > MAX_BYTES:
            raise ValueError('arc checkpoint byte bound')
        self.guard(); return raw

    def restore(self, raw, *, expected_sha256):
        self.guard()
        if (type(raw) is not bytes or len(raw) > MAX_BYTES or type(expected_sha256) is not str
                or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError('external arc checkpoint authority mismatch')
        value = _load(raw.decode())
        keys = {'schema', 'formulation', 'geometry', 'model_sha256', 'program', 'initial',
                'records', 'completed_steps', 'checkpoint_sha256'}
        if type(value) is not dict or set(value) != keys:
            raise ValueError('exact arc checkpoint schema')
        body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
        if (value['schema'] != SCHEMA or value['formulation'] != POLICY or value['geometry'] != GEOMETRY
                or value['model_sha256'] != self.identity or canonical(value['program']) != self.program_bytes
                or value['checkpoint_sha256'] != sha(body) or canonical(value['initial']) != self.genesis):
            raise ValueError('arc checkpoint model/program/genesis binding')
        cursor = value['completed_steps']
        if (type(cursor) is not int or not 0 <= cursor <= len(self.program.steps)
                or type(value['records']) is not list or len(value['records']) != cursor):
            raise ValueError('arc checkpoint step extent')
        accepted, records = self.initial, (); record_keys = set(_load(self.genesis.decode()))
        for index, row in enumerate(value['records'], 1):
            if type(row) is not dict or set(row) != record_keys or type(row['step']) is not int or row['step'] != index:
                raise ValueError('exact arc step record')
            proposed, regenerated = self.stage(self.physical.make(row['mechanical'], decoded=True),
                row['parameter'], accepted, records, row['iterations'])
            if regenerated != canonical(row):
                raise ValueError('arc predictor/geometry/history/work/recovery replay mismatch')
            accepted, records = proposed, (*records, regenerated)
        if self.checkpoint(records) != raw:
            raise ValueError('noncanonical arc replay')
        self.guard(); return accepted, records


@dataclass(frozen=True)
class Result:
    status: str
    completed_steps: int
    state: State
    checkpoint: bytes
    failure: str | None
    production_qualified: bool = False


def solve(model, program, *, checkpoint=None, expected_sha256=None, stop_after=None,
          cancellation_token=None, progress=None):
    cancellation_safe_point(cancellation_token, 'retained-arc.start')
    if progress is not None and not callable(progress):
        raise ValueError('callable arc observer required')
    context = Context(model, program); p = context.physical
    end = len(program.steps) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(program.steps):
        raise ValueError('bounded arc stop step')
    if checkpoint is None:
        if expected_sha256 is not None:
            raise ValueError('hash requires an arc checkpoint')
        accepted, records = context.initial, (); capsule = context.checkpoint(records)
    else:
        accepted, records = context.restore(checkpoint, expected_sha256=expected_sha256); capsule = checkpoint
    if accepted.completed_steps > end:
        raise ValueError('arc history cannot rewind')
    status = 'completed' if end == len(program.steps) else 'paused'; failure = None

    def safe(stage, index, iteration=0):
        cancellation_safe_point(cancellation_token, stage); context.guard()
        if progress is not None:
            progress(dict(stage=stage, step=index, iteration=iteration))
        context.guard(); cancellation_safe_point(cancellation_token, stage)

    try:
        safe('retained-arc.initialized', accepted.completed_steps)
        for index in range(accepted.completed_steps, end):
            origins = accepted.histories
            safe('retained-arc.before_predictor', index+1)
            predictor = context.predictor(accepted); size = program.steps[index]
            trial = p.advance(accepted.mechanical, size*predictor[:-1])
            parameter = float(accepted.parameter+size*predictor[-1])
            for iteration in range(program.max_iterations+1):
                safe('retained-arc.before_assembly', index+1, iteration)
                residual, jacobian, metrics, _, _ = p.assemble(trial, parameter, origins)
                gap, row = context.row(trial, parameter, accepted, predictor)
                safe('retained-arc.before_factorization', index+1, iteration)
                step, delta, remaining, matrix = context.correction(trial, parameter, residual, jacobian, gap, row)
                if max(*metrics, abs(gap), remaining) <= 1e-11:
                    proposed, raw = context.stage(trial, parameter, accepted, records, iteration)
                    next_records = (*records, raw); staged = context.checkpoint(next_records)
                    safe('retained-arc.before_commit', index+1, iteration)
                    accepted, records, capsule = proposed, next_records, staged
                    safe('retained-arc.committed', index+1, iteration); break
                if iteration == program.max_iterations:
                    raise RuntimeError('retained arc Newton limit')
                angular = [step[6*i+3:6*i+6] for i in range(len(p.node_ids))]
                angular.extend(step[p.nodal_count+24*i+3*c:p.nodal_count+24*i+3*c+3]
                    for i in range(len(p.elements)) for c in (0, 1))
                largest = max(float(np.linalg.norm(v)) for v in angular)
                fraction = min(1., .45*np.pi/max(largest, np.finfo(float).tiny))
                for cut in range(program.max_backtracks+1):
                    safe('retained-arc.before_trial', index+1, iteration)
                    amount = fraction*.5**cut
                    candidate = p.advance(trial, amount*step); next_parameter = float(parameter+amount*delta)
                    changed, _, _, _, _ = p.assemble(candidate, next_parameter, origins)
                    changed_gap, _ = context.row(candidate, next_parameter, accepted, predictor)
                    natural = np.linalg.solve(matrix, -np.r_[changed[p.free], changed_gap])
                    if not np.isfinite(natural).all():
                        raise ValueError('nonfinite frozen arc border merit')
                    trial_step = np.zeros(p.count); trial_step[p.free] = natural[:-1]
                    merit = context.correction_norm(trial, parameter, trial_step, float(natural[-1]))
                    if merit < remaining:
                        trial, parameter = candidate, next_parameter; break
                else:
                    raise RuntimeError('retained arc line search limit')
    except SolveCancelled as error:
        status, failure = 'cancelled', str(error)
    except Exception as error:
        status, failure = 'failed', type(error).__name__+': '+str(error)
    return Result(status, accepted.completed_steps, accepted, capsule, failure)
