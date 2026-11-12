"""Private model-bound paired plastic state, history-chain replay, and recovery.

Uses the old retained context only for standalone capture and geometric DOF
operations. Its elastic assembly and its capsules are never used for plastic
state. Restart re-evaluates recorded accepted states, not Newton load steps.
"""
from dataclasses import dataclass
from decimal import localcontext
from hashlib import sha256
from time import monotonic
import numpy as np
from anyfesolver._ge_beam3_retained_state import Context as MechanicalLayout, State as MechanicalState
from anyfesolver._ge_beam3_retained_plastic import RetainedPlasticOperator, POLICY
from anyfesolver._ge_beam3_station_resultant_cell import history_row
from anyfesolver._native_reference_modal import _owned
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import _load, MAX_BYTES

SCHEMA = 'GE_BEAM3_RETAINED_PAIRED_PLASTIC_ACCEPTED_CHAIN_V1'
PROGRAM = 'GE_BEAM3_RETAINED_PAIRED_PLASTIC_FORCE_PROGRAM_V1'


@dataclass(frozen=True)
class State:
    mechanical: MechanicalState
    origins: tuple
    histories: tuple
    completed_targets: int
    model_sha256: str


class Context:
    def __init__(self, model, program):
        self.started = monotonic(); self.layout = MechanicalLayout(model, program)
        self.program = program
        self.probes = tuple(RetainedPlasticOperator(e) for _, e in self.layout.elements)
        self.program_data = {**program.descriptor(), 'schema': PROGRAM,
            'line_search': 'RETAINED_EQUILIBRIUM_COMPATIBILITY_RESIDUAL_DECREASE_V1'}
        self.identity = sha(dict(formulation_id=POLICY, captured_model=self.layout.identity,
            operator_ids=[p.identity for p in self.probes], program=self.program_data, state_schema=SCHEMA))
        histories = tuple(_owned(np.zeros((len(p.stations), 4))) for p in self.probes)
        self.initial = State(self.layout.initial, histories, histories, 0, self.identity)
        self.genesis = self._record(self.initial.mechanical, histories, 0, 0, self.identity)[1]
        self.guard()

    def guard(self):
        if monotonic()-self.started > 120: raise RuntimeError('paired-plastic context cooperative deadline')
        self.layout.guard()
        for p in self.probes: p.guard()

    def history(self, values, *, decoded=False):
        if type(values) not in (list, tuple) or len(values) != len(self.probes):
            raise ValueError('one paired-history array per retained element required')
        result = []
        with localcontext() as ctx:
            ctx.prec = 80
            for value, probe in zip(values, self.probes):
                if decoded:
                    raw = np.array(value, dtype=object)
                    if any(type(x) is not float for x in raw.flat): raise ValueError('strict binary64 plastic history')
                array = _owned(value)
                if array.shape != (len(probe.stations), 4): raise ValueError('paired plastic history shape')
                for row in array.tolist(): history_row(row)
                result.append(array)
        return tuple(result)

    def assemble(self, mechanical, parameter, origins):
        self.guard(); layout = self.layout
        origins = self.history(origins)
        residual = np.zeros(layout.count); hessian = np.zeros((layout.count, layout.count)); responses = []
        for i, probe in enumerate(self.probes):
            nodes = layout.nodes[i]
            response = probe.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes],
                mechanical.nodal_frames[nodes], mechanical.cell_rotations[i], mechanical.resultants[i],
                origins=origins[i].tolist())
            residual[layout.slots[i]] += response.residual
            hessian[np.ix_(layout.slots[i], layout.slots[i])] += response.hessian+response.hessian_low
            responses.append(response)
        residual[:layout.nodal_count] -= parameter*layout.force
        scaled = residual/layout.scales
        metrics = (float(np.linalg.norm(scaled[layout.equilibrium])), float(np.linalg.norm(scaled[layout.compatibility])))
        self.guard()
        return residual, hessian, metrics, tuple(responses)

    def recover(self, state):
        if type(state) is not State or state.model_sha256 != self.identity: raise ValueError('plastic recovery state model binding')
        self.guard(); rows = []
        for i, ((element_id, _), probe) in enumerate(zip(self.layout.elements, self.probes)):
            stations = probe.recover(state.mechanical.cell_rotations[i], state.mechanical.resultants[i],
                origins=state.origins[i].tolist())
            if canonical([r['history'] for r in stations]) != canonical(state.histories[i]):
                raise ValueError('paired recovery does not reproduce committed history')
            rows.append(dict(element_id=element_id, stations=stations))
        self.guard(); return tuple(rows)

    def _record(self, mechanical, origins, cursor, iterations, previous):
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets): raise ValueError('plastic target cursor')
        if type(iterations) is not int or not 0 <= iterations <= self.program.max_iterations:
            raise ValueError('plastic Newton iteration record')
        parameter = 0. if cursor == 0 else self.program.targets[cursor-1]
        mechanical = self.layout.make(mechanical.descriptor())
        origins = self.history(origins)
        residual, _, metrics, responses = self.assemble(mechanical, parameter, origins)
        if max(metrics) > 1e-11: raise ValueError('only equilibrated compatible plastic states may commit')
        histories = self.history(tuple(r.history for r in responses))
        state = State(mechanical, origins, histories, cursor, self.identity)
        body = dict(target=cursor, parameter=parameter, iterations=iterations, previous_sha256=previous,
            mechanical=mechanical.descriptor(), origins=origins, histories=histories, residual=residual,
            metrics=metrics, material_sha256=sha([r.material.decode('ascii') for r in responses]),
            recovery_sha256=sha(self.recover(state)))
        self.guard()
        return state, canonical({**body, 'record_sha256': sha(body)})

    def stage(self, mechanical, accepted, records, iterations):
        if type(accepted) is not State or accepted.model_sha256 != self.identity or accepted.completed_targets != len(records):
            raise ValueError('plastic accepted-state cursor differs from chain')
        previous = _load((self.genesis if not records else records[-1]).decode('ascii'))['record_sha256']
        return self._record(mechanical, accepted.histories, len(records)+1, iterations, previous)

    def checkpoint(self, records):
        if type(records) is not tuple or len(records) > len(self.program.targets): raise ValueError('bounded immutable plastic chain')
        body = dict(schema=SCHEMA, formulation_id=POLICY, model_sha256=self.identity,
            program=self.program_data, node_ids=self.layout.node_ids,
            element_ids=[i for i, _ in self.layout.elements], completed_targets=len(records),
            initial=_load(self.genesis.decode('ascii')), records=[_load(r.decode('ascii')) for r in records])
        raw = canonical({**body, 'checkpoint_sha256': sha(body)})
        if len(raw) > MAX_BYTES: raise ValueError('paired plastic checkpoint byte limit')
        self.guard(); return raw

    def restore(self, raw, *, expected_sha256=None):
        self.guard()
        if type(raw) is not bytes: raise ValueError('canonical plastic checkpoint bytes required')
        if expected_sha256 is not None and (type(expected_sha256) is not str or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError('external checkpoint hash authority mismatch')
        value = _load(raw.decode('ascii'))
        keys = {'schema', 'formulation_id', 'model_sha256', 'program', 'node_ids', 'element_ids',
                'completed_targets', 'initial', 'records', 'checkpoint_sha256'}
        if type(value) is not dict or set(value) != keys: raise ValueError('exact paired-plastic checkpoint schema')
        body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
        if (value['schema'] != SCHEMA or value['formulation_id'] != POLICY or value['model_sha256'] != self.identity
                or value['checkpoint_sha256'] != sha(body) or canonical(value['program']) != canonical(self.program_data)
                or canonical(value['node_ids']) != canonical(self.layout.node_ids)
                or canonical(value['element_ids']) != canonical([i for i, _ in self.layout.elements])):
            raise ValueError('plastic checkpoint model/program/formulation binding')
        cursor = value['completed_targets']
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets): raise ValueError('plastic checkpoint cursor')
        if type(value['records']) is not list or len(value['records']) != cursor: raise ValueError('plastic checkpoint record count')
        if canonical(value['initial']) != self.genesis: raise ValueError('plastic checkpoint virgin genesis differs')
        accepted = self.initial; records = ()
        record_keys = {'target', 'parameter', 'iterations', 'previous_sha256', 'mechanical', 'origins',
            'histories', 'residual', 'metrics', 'material_sha256', 'recovery_sha256', 'record_sha256'}
        for index, row in enumerate(value['records'], 1):
            self.guard()
            if (type(row) is not dict or set(row) != record_keys or type(row['target']) is not int or row['target'] != index
                    or type(row['parameter']) is not float or row['parameter'] != self.program.targets[index-1]):
                raise ValueError('plastic checkpoint target record')
            origins = self.history(row['origins'], decoded=True)
            self.history(row['histories'], decoded=True)
            if canonical(origins) != canonical(accepted.histories): raise ValueError('plastic origin chain is broken')
            mechanical = self.layout.make(row['mechanical'], decoded=True)
            proposed, regenerated = self.stage(mechanical, accepted, records, row['iterations'])
            if regenerated != canonical(row): raise ValueError('plastic state/reaction/recovery/history replay mismatch')
            accepted, records = proposed, (*records, regenerated)
        if self.checkpoint(records) != raw: raise ValueError('plastic checkpoint canonical replay mismatch')
        self.guard(); return accepted, records
