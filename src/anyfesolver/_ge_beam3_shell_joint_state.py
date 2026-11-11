"""Private global beam/shell accepted chain, distinct from standalone owners.

Both origins stay fixed through Newton/line search. A state is issued only after
global equilibrium, compatibility, the rigid joint and the correction test pass.
One immutable record contains both histories. No standalone state is fabricated,
no shell operator changes, and no generic FEModel/default routing is enabled.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from threading import Lock
import numpy as np

from .control import cancellation_safe_point, SolveCancelled
from ._ge_beam3_shell_joint_trial import ShellBeamTrialAssembly, _assemble_material_trial
from ._ge_beam3_pose_joint import _array
from ._ge_beam3_p5_seeded.core import canonical
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._ge_beam3_operator_validation_scope import operator_call
from ._ge_beam3_p5.algebra import skew
from ._native_reference_modal import _owned

SCHEMA = 'GE_BEAM3_COROTATIONAL_SHELL_COUPLED_ACCEPTED_CHAIN_V1'
POLICY = 'GE_BEAM3_GENERALIZED_BEAM_ELASTIC_SHELL_RIGID_OFFSET_COUPLING_V1'


def digest(value):
    return sha256(value).hexdigest()


def decode(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError('bounded canonical coupled bytes required')
    try:
        return _load(raw.decode('ascii'))
    except (UnicodeError, RecursionError) as error:
        raise ValueError('invalid coupled checkpoint text') from error


def decoded_array(value, shape):
    values = np.asarray(value, dtype=object)
    if values.shape != shape or any(type(x) is not float for x in values.flat):
        raise ValueError('exact binary64 coupled array required')
    return _array(np.array(values, dtype=float), shape)


def _require_rigid_support(assembly, shell_fixed):
    """Reject free rigid components without a stiffness-condition-number test.

Mechanical slenderness must not be mistaken for missing supports. Connectivity
and physical rigid motions determine this preflight, independently of K's scale.
The rigid joint connects its two components even when its offset is nonzero.
"""
    beam = assembly.beam; shell_nodes = len(assembly.coordinates)
    positions = list(assembly.coordinates)+list(beam.reference_positions)
    parents = list(range(len(positions)))
    def root(i):
        while parents[i] != i: i = parents[i]
        return i
    def join(nodes):
        for node in nodes[1:]: parents[root(node)] = root(nodes[0])
    join(list(range(shell_nodes)))
    for _, element in beam.elements:
        join([shell_nodes+beam.index[n] for n in element.node_ids])
    join([assembly.shell_node, shell_nodes+beam.index[assembly.beam_node]])
    fixed = set(shell_fixed) | {assembly.shell_count+i for i in beam.fixed}
    for component in sorted({root(i) for i in parents}):
        members = [i for i in range(len(positions)) if root(i) == component]
        xyz = np.array([positions[i] for i in members]); center = xyz.mean(axis=0)
        length = max(float(np.linalg.norm(x-center)) for x in xyz)
        if length == 0.: raise ValueError('degenerate coupled component')
        rows = []
        for node in members:
            rigid = np.block([[np.eye(3), -skew((positions[node]-center)/length)],
                [np.zeros((3, 3)), np.eye(3)]])
            rows.extend(rigid[axis] for axis in range(6) if 6*node+axis in fixed)
        if not rows or np.linalg.matrix_rank(np.array(rows)) != 6:
            raise ValueError('supports leave a coupled component with a free rigid motion')


@dataclass(frozen=True)
class CoupledState:
    mechanical: object
    shell_u: np.ndarray
    multipliers: np.ndarray
    beam_origins: tuple
    beam_histories: tuple
    shell_origin: bytes
    shell_history: bytes
    cursor: int
    definition_sha256: str
    production_qualified: bool = field(default=False, init=False)

    def descriptor(self):
        return dict(mechanical=self.mechanical, shell_u=self.shell_u, multipliers=self.multipliers,
            beam_origins=self.beam_origins, beam_histories=self.beam_histories,
            shell_origin=decode(self.shell_origin), shell_history=decode(self.shell_history),
            cursor=self.cursor, definition_sha256=self.definition_sha256, production_qualified=False)


@dataclass(frozen=True)
class CoupledRun:
    status: str
    state: CoupledState
    checkpoint: bytes
    failure: str | None
    production_qualified: bool = field(default=False, init=False)


class CoupledShellBeamAnalysis:
    """Owned global static transactions for the actual current shell/beam seam.

Shell material is the seam's explicit elastic material; generalized beam history
may be nonlinear. Multiple elements within the owned beam subdomain are allowed.
Only zero supports and nodal spatial dead forces are admitted by this version.
    The Newton step solve defaults to numpy.linalg.solve; the private
    _linear_solve hook exists only for the F06 prototype reference backend.
Shell histories are corotated, shell coordinates additive, beam frames spatial
multiplicative. Unsupported shell material, arbitrary starts, nodal couples,
fibre owners, generic MPCs and spectral use are not silently approximated.
"""

    def __init__(self, assembly, *, targets, shell_fixed, nodal_forces,
                 max_iterations=24, max_backtracks=8, _linear_solve=None):
        if type(assembly) is not ShellBeamTrialAssembly:
            raise ValueError('exact real shell/beam assembly required')
        assembly.guard()
        if (type(targets) is not tuple or not 1 <= len(targets) <= 16
                or any(type(x) is not float or not np.isfinite(x) or abs(x) > 16. for x in targets)):
            raise ValueError('frozen bounded coupled target schedule required')
        if (type(shell_fixed) is not tuple or any(type(x) is not int for x in shell_fixed)
                or tuple(sorted(set(shell_fixed))) != shell_fixed
                or any(not 0 <= x < assembly.shell_count for x in shell_fixed)):
            raise ValueError('ordered unique shell support DOFs required')
        if (type(max_iterations) is not int or not 1 <= max_iterations <= 24
                or type(max_backtracks) is not int or not 0 <= max_backtracks <= 8):
            raise ValueError('bounded coupled iteration controls required')
        if _linear_solve is not None and not callable(_linear_solve):
            raise ValueError('callable coupled Newton linear solve required')
        forces = _array(nodal_forces, (assembly.shell_count+assembly.beam.nodal_count,))
        if np.any(forces.reshape(-1, 6)[:, 3:] != 0.):
            raise ValueError('nodal couples require their separate work contract')
        _require_rigid_support(assembly, shell_fixed)
        self.assembly = assembly
        self.targets, self.shell_fixed = targets, shell_fixed
        self.forces = forces
        self.max_iterations, self.max_backtracks = max_iterations, max_backtracks
        self._linear_solve = _linear_solve
        self.shell_length = max(float(np.linalg.norm(x-y)) for x in assembly.coordinates for y in assembly.coordinates)
        self.free = tuple([i for i in range(assembly.shell_count) if i not in shell_fixed]
            + [assembly.shell_count+int(i) for i in assembly.beam.free]
            + list(range(assembly.count-6, assembly.count)))
        self.scale = _owned(np.r_[np.ones(assembly.shell_count), assembly.beam.scale, np.ones(6)])
        self.identity = digest(canonical(self._definition()))
        self._lock = Lock()
        self._issued = {}; self._issued_records = set()
        if not assembly._lock.acquire(blocking=False):
            raise RuntimeError('assembly in concurrent use')
        try:
            self.initial, self.genesis = self._propose(None, assembly.beam.initial.mechanical,
                np.zeros(assembly.shell_count), np.zeros(6), 0, 0)
            self._issue(self.initial, self.genesis)
        finally:
            assembly._lock.release()

    def _definition(self):
        return dict(schema=SCHEMA, policy=POLICY, assembly=self.assembly.identity,
            targets=self.targets, shell_fixed=self.shell_fixed, free=self.free,
            nodal_forces=self.forces, scale=self.scale, max_iterations=self.max_iterations,
            max_backtracks=self.max_backtracks, shell_length=self.shell_length, production_qualified=False)

    def guard(self):
        self.assembly.guard()
        if digest(canonical(self._definition())) != self.identity:
            raise ValueError('coupled definition changed')

    def _require(self, state):
        self.guard()
        bound = self._issued.get(id(state))
        if (type(state) is not CoupledState or bound is None or bound[0] is not state
                or state.definition_sha256 != self.identity
                or state.production_qualified is not False
                or canonical(state.descriptor()) != bound[1]):
            raise ValueError('coupled state not issued by this global owner')
        return bound[2]

    def _issue(self, state, record):
        self._issued[id(state)] = (state, canonical(state.descriptor()), record)
        self._issued_records.add(record)

    def _arrays(self, mechanical, u, multipliers):
        mechanical = self.assembly.beam.make(mechanical.descriptor())
        u = _array(u, (self.assembly.shell_count,)); multipliers = _array(multipliers, (6,))
        if np.any(u[list(self.shell_fixed)] != 0.):
            raise ValueError('fixed shell coordinates changed')
        return mechanical, u, multipliers

    def _evaluate(self, predecessor, mechanical, u, multipliers, cursor, token=None):
        self.guard()
        mechanical, u, multipliers = self._arrays(mechanical, u, multipliers)
        if predecessor is None:
            origins = self.assembly.beam.initial.histories
            shell_origin = self.assembly.origin_bytes
            previous = self.identity
        else:
            origins = predecessor.beam_histories; shell_origin = predecessor.shell_history
            previous = digest(canonical(predecessor.descriptor()))
        parameter = 0. if cursor == 0 else self.targets[cursor-1]
        response = _assemble_material_trial(self.assembly, mechanical, u, multipliers,
            origins, decode(shell_origin), parameter, previous, token)
        residual = response.residual.copy()
        residual[:len(self.forces)] -= parameter*self.forces
        self.guard()
        return response, residual, origins, shell_origin

    def _step(self, response, residual, mechanical, u, multipliers):
        step = np.zeros(self.assembly.count)
        solve = np.linalg.solve if self._linear_solve is None else self._linear_solve
        step[list(self.free)] = solve(response.tangent[np.ix_(self.free, self.free)],
            -residual[list(self.free)])
        if not np.isfinite(step).all():
            raise ValueError('nonfinite coupled Newton step')
        size = self.assembly.shell_count; beam = self.assembly.beam
        # Coordinates, trace rotations, physical cell rotations, stress resultants
        # and joint multipliers all participate in the correction acceptance.
        shell_step = step[:size].reshape(-1, 6)
        beam_step = step[size:size+beam.nodal_count].reshape(-1, 6)
        correction = max(float(np.linalg.norm(shell_step[:, :3]))/self.shell_length,
            float(np.linalg.norm(shell_step[:, 3:])),
            float(np.linalg.norm(beam_step[:, :3]))/beam.length,
            float(np.linalg.norm(beam_step[:, 3:])),
            float(np.linalg.norm(step[-6:]))/max(1., float(np.linalg.norm(multipliers))))
        for i in range(len(beam.elements)):
            first = size+beam.nodal_count+24*i
            correction = max(correction, float(np.linalg.norm(step[first:first+6])),
                float(np.linalg.norm(step[first+6:first+24]))/max(1., float(np.linalg.norm(mechanical.resultants[i]))))
        metric = float(np.linalg.norm((residual/self.scale)[list(self.free)]))
        return step, metric, correction

    def _beam_recovery(self, state):
        beam = self.assembly.beam; rows = []
        for i, (eid, element) in enumerate(beam.elements):
            rows_i = operator_call(element.operator, 'recover', self.guard, beam.started,
                state.mechanical.cell_rotations[i], state.mechanical.resultants[i], origin=state.beam_origins[i])
            if canonical([row['history'] for row in rows_i]) != canonical(state.beam_histories[i].stations):
                raise ValueError('coupled beam recovery/history disagreement')
            rows.append(dict(element_id=eid, stations=rows_i))
        return tuple(rows)

    def _propose(self, predecessor, mechanical, u, multipliers, cursor, iterations):
        if (type(cursor) is not int or not 0 <= cursor <= len(self.targets)
                or type(iterations) is not int or not 0 <= iterations <= self.max_iterations
                or (predecessor is None) != (cursor == 0)
                or (predecessor is not None and cursor != predecessor.cursor+1)):
            raise ValueError('contiguous coupled proposal required')
        mechanical, u, multipliers = self._arrays(mechanical, u, multipliers)
        response, residual, origins, shell_origin = self._evaluate(predecessor, mechanical, u, multipliers, cursor)
        _, metric, correction = self._step(response, residual, mechanical, u, multipliers)
        if max(metric, correction, float(np.linalg.norm(response.constraints))) > 1e-11:
            raise ValueError('only globally equilibrated compatible joint states may commit')
        histories = self.assembly.beam.histories(decode(response.beam_histories), decoded=True)
        decode(response.shell_candidate)
        state = CoupledState(mechanical, u, multipliers, origins, histories, shell_origin,
            response.shell_candidate, cursor, self.identity)
        body = dict(cursor=cursor, parameter=0. if cursor == 0 else self.targets[cursor-1],
            iterations=iterations, previous_sha256=self.identity if predecessor is None else
                digest(canonical(predecessor.descriptor())), state=state.descriptor(),
            residual=residual, constraints=response.constraints, metric=metric, correction=correction,
            tangent_sha256=digest(canonical(response.tangent)),
            recovery_sha256=digest(canonical(self._beam_recovery(state))))
        record = canonical({**body, 'record_sha256': digest(canonical(body))})
        self.guard()
        return state, record

    def _capsule(self, records):
        raw = canonical(dict(schema=SCHEMA, policy=POLICY, definition_sha256=self.identity,
            definition=self._definition(), initial=decode(self.genesis),
            records=[decode(r) for r in records], production_qualified=False))
        if len(raw) > MAX_BYTES:
            raise ValueError('coupled checkpoint size limit')
        return raw

    def _restore(self, raw, expected_sha256, cancellation_token=None):
        if type(expected_sha256) is not str or type(raw) is not bytes or digest(raw) != expected_sha256:
            raise ValueError('external coupled checkpoint SHA-256 required')
        value = decode(raw)
        if (type(value) is not dict or set(value) != {'schema', 'policy', 'definition_sha256',
                'definition', 'initial', 'records', 'production_qualified'}
                or value['schema'] != SCHEMA or value['policy'] != POLICY
                or value['definition_sha256'] != self.identity
                or canonical(value['definition']) != canonical(self._definition())
                or canonical(value['initial']) != self.genesis or value['production_qualified'] is not False
                or type(value['records']) is not list or len(value['records']) > len(self.targets)):
            raise ValueError('coupled checkpoint schema/definition/genesis binding')
        accepted = self.initial; records = (); staged = []
        for cursor, row in enumerate(value['records'], 1):
            cancellation_safe_point(cancellation_token, 'coupled.replay-record')
            data = row['state']
            mechanical = self.assembly.beam.make(data['mechanical'], decoded=True)
            u = decoded_array(data['shell_u'], (self.assembly.shell_count,))
            lm = decoded_array(data['multipliers'], (6,))
            proposed, record = self._propose(accepted, mechanical, u, lm, cursor, row['iterations'])
            cancellation_safe_point(cancellation_token, 'coupled.replay-record-complete')
            if record != canonical(row):
                raise ValueError('coupled geometry/history/work replay disagreement')
            accepted = proposed; records = (*records, record); staged.append((accepted, record))
        if self._capsule(records) != raw:
            raise ValueError('noncanonical coupled replay')
        # Failed replay never issues even a valid prefix from the rejected input.
        self.guard()
        cancellation_safe_point(cancellation_token, 'coupled.replay-complete')
        for state, record in staged: self._issue(state, record)
        return accepted, records

    def recover(self, state):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('concurrent coupled analysis forbidden')
        held = False
        try:
            held = self.assembly._lock.acquire(blocking=False)
            if not held: raise RuntimeError('assembly in concurrent use')
            self._require(state)
            result = dict(beam=self._beam_recovery(state), shell_corotated_state=decode(state.shell_history),
                shell_coordinates=state.shell_u, multipliers=state.multipliers,
                definition_sha256=self.identity, coupled_cursor=state.cursor, production_qualified=False)
            self._require(state)
            return result
        finally:
            if held: self.assembly._lock.release()
            self._lock.release()

    def solve(self, *, checkpoint=None, expected_sha256=None, stop_after=None,
              cancellation_token=None, progress=None):
        cancellation_safe_point(cancellation_token, 'coupled.start')
        if progress is not None and not callable(progress):
            raise ValueError('callable coupled observer required')
        end = len(self.targets) if stop_after is None else stop_after
        if type(end) is not int or not 0 <= end <= len(self.targets):
            raise ValueError('bounded coupled stop cursor required')
        if (checkpoint is None) != (expected_sha256 is None):
            raise ValueError('coupled checkpoint and external hash must be paired')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('concurrent coupled analysis forbidden')
        held = False
        try:
            held = self.assembly._lock.acquire(blocking=False)
            if not held: raise RuntimeError('assembly in concurrent use')
            self.guard()
            accepted, records = (self.initial, ()) if checkpoint is None else self._restore(checkpoint, expected_sha256, cancellation_token)
            if accepted.cursor > end: raise ValueError('coupled accepted history cannot rewind')
            capsule = self._capsule(records)
            status = 'completed' if end == len(self.targets) else 'paused'; failure = None
            def safe(stage, cursor, iteration=0):
                cancellation_safe_point(cancellation_token, stage); self.guard()
                if progress is not None: progress(dict(stage=stage, target=cursor, iteration=iteration))
                self.guard(); cancellation_safe_point(cancellation_token, stage)
            try:
                safe('coupled.initialized', accepted.cursor)
                for index in range(accepted.cursor, end):
                    mechanical, u, lm = accepted.mechanical, accepted.shell_u, accepted.multipliers
                    for iteration in range(self.max_iterations+1):
                        safe('coupled.before_assembly', index+1, iteration)
                        response, residual, _, _ = self._evaluate(accepted, mechanical, u, lm, index+1, cancellation_token)
                        step, metric, correction = self._step(response, residual, mechanical, u, lm)
                        if max(metric, correction, float(np.linalg.norm(response.constraints))) <= 1e-11:
                            proposed, record = self._propose(accepted, mechanical, u, lm, index+1, iteration)
                            next_records = (*records, record); staged = self._capsule(next_records)
                            safe('coupled.before_commit', index+1, iteration)
                            self._issue(proposed, record)
                            accepted, records, capsule = proposed, next_records, staged
                            safe('coupled.committed', index+1, iteration)
                            break
                        if iteration == self.max_iterations: raise RuntimeError('coupled Newton limit')
                        size = self.assembly.shell_count
                        for cut in range(self.max_backtracks+1):
                            safe('coupled.before_trial', index+1, iteration)
                            delta = step*(.5**cut)
                            try:
                                candidate = self.assembly.beam.advance(mechanical, delta[size:-6])
                                cu = u+delta[:size]; cm = lm+delta[-6:]
                                cr, rr, _, _ = self._evaluate(accepted, candidate, cu, cm, index+1, cancellation_token)
                            except ValueError as error:
                                if 'requires cutback' in str(error): continue
                                raise
                            changed = float(np.linalg.norm((rr/self.scale)[list(self.free)]))
                            if changed < metric:
                                mechanical, u, lm = candidate, cu, cm
                                break
                        else: raise RuntimeError('coupled line search limit')
            except SolveCancelled as error:
                status, failure = 'cancelled', str(error)
            except Exception as error:
                status, failure = 'failed', type(error).__name__+': '+str(error)
            return CoupledRun(status, accepted, capsule, failure)
        finally:
            if held: self.assembly._lock.release()
            self._lock.release()
