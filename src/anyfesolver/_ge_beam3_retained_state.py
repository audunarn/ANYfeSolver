"""Private retained-resultant state ownership and exact model-bound replay.

No historical V5 state is translated or mutated. Only virgin elastic
retained-resultant states are represented; this is not plastic restart parity.
"""
from dataclasses import dataclass
import numpy as np
from anyfesolver._ge_beam3_seeded_load_program import _capture, ForceProgram
from anyfesolver._ge_beam3_p5_seeded.core import canonical, sha
from anyfesolver._ge_beam3_p5_seeded.codec import _load
from anyfesolver._ge_beam3_p5.arrays import _frames
from anyfesolver._ge_beam3_p5.compensated_coordinates import validate_pair, split_sum
from anyfesolver._native_reference_modal import _owned
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_retained_elastic import RetainedElasticOperator, POLICY


SCHEMA = 'GE_BEAM3_RETAINED_ELASTIC_ACCEPTED_STATE_V1'
PROGRAM = 'GE_BEAM3_RETAINED_ELASTIC_FORCE_PROGRAM_V1'


@dataclass(frozen=True)
class State:
    positions: np.ndarray
    position_low: np.ndarray
    nodal_frames: np.ndarray
    cell_rotations: np.ndarray
    resultants: np.ndarray

    def descriptor(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class Context:
    """Captured homogeneous standalone beam model, never a shell/joint adapter."""
    def __init__(self, model, program):
        self.model, self.program = model, program
        if type(program) is not ForceProgram or any(type(v) is not float for v in program.targets):
            raise ValueError('retained program requires explicit binary64 targets')
        self.elements, self.nodal_count, self.nodal_free, self.force, identity, self.guard = _capture(model, program)
        self.count = self.nodal_count+24*len(self.elements)
        if self.count > 256: raise ValueError('bounded retained-resultant model required')
        self.program_data = {**program.descriptor(), 'schema': PROGRAM,
            'line_search': 'RETAINED_EQUILIBRIUM_COMPATIBILITY_RESIDUAL_DECREASE_V1'}
        self.identity = sha(dict(formulation_id=POLICY, captured_model_program=identity,
            retained_program=self.program_data, state_schema=SCHEMA))
        self.node_ids = tuple(sorted(model.mesh.nodes,
            key=lambda i: model.mesh.dof_manager.get_node_dofs(i)[0]))
        for i, node in enumerate(self.node_ids):
            if tuple(model.mesh.dof_manager.get_node_dofs(node)) != tuple(range(6*i, 6*i+6)):
                raise ValueError('complete contiguous six-DOF node blocks required')
        self.node_index = {node: i for i, node in enumerate(self.node_ids)}
        self.probes = tuple(RetainedElasticOperator(e) for _, e in self.elements)
        self.slots = []; self.nodes = []; frames = [None]*len(self.node_ids)
        self.equilibrium = list(int(v) for v in self.nodal_free); self.compatibility = []
        self.scales = np.ones(self.count)
        length = max(e.core.length for _, e in self.elements)
        for i in range(len(frames)): self.scales[6*i+3:6*i+6] = length
        for i, (_, e) in enumerate(self.elements):
            first = self.nodal_count+24*i
            nodes = [self.node_index[node] for node in e.node_ids]
            self.nodes.append(nodes)
            self.slots.append(list(e.get_dof_mapping(model.mesh))+list(range(first, first+24)))
            self.equilibrium.extend(range(first, first+6)); self.compatibility.extend(range(first+6, first+24))
            self.scales[first:first+12] = length
            for local, node in enumerate(nodes):
                frame = e.core.reference.nodal_triads[local]
                if frames[node] is not None and not np.array_equal(frames[node], frame):
                    raise ValueError('shared-node material frame authority differs')
                frames[node] = frame
        if any(frame is None for frame in frames): raise ValueError('unconnected node not admitted')
        self.free = list(int(v) for v in self.nodal_free)+list(range(self.nodal_count, self.count))
        self.fixed = sorted(set(range(self.nodal_count))-set(self.nodal_free))
        self.reference_positions = _owned([model.mesh.nodes[i].coords() for i in self.node_ids])
        self.reference_frames = _owned(frames)
        self.initial = self.make(dict(positions=self.reference_positions,
            position_low=np.zeros_like(self.reference_positions), nodal_frames=self.reference_frames,
            cell_rotations=np.tile(np.eye(3), (len(self.elements), 2, 1, 1)),
            resultants=np.zeros((len(self.elements), 18))))
        self.guard()

    def make(self, data, *, decoded=False):
        if type(data) is not dict or set(data) != set(State.__dataclass_fields__):
            raise ValueError('exact retained state fields required')
        n = len(self.node_ids); e = len(self.elements)
        shapes = ((n, 3), (n, 3), (n, 3, 3), (e, 2, 3, 3), (e, 18))
        values = []
        for (key, shape) in zip(State.__dataclass_fields__, shapes):
            if decoded:
                raw = np.array(data[key], dtype=object)
                if raw.shape != shape or any(type(x) is not float for x in raw.flat):
                    raise ValueError('strict binary64 retained state arrays required')
            array = _owned(data[key])
            if array.shape != shape: raise ValueError('retained state array shape mismatch')
            values.append(array)
        state = State(*values)
        _frames(state.nodal_frames, n, 'retained nodal frames')
        _frames(state.cell_rotations.reshape(2*e, 3, 3), 2*e, 'retained cell rotations')
        for high, low in zip(state.positions.flat, state.position_low.flat):
            validate_pair(float(high), float(low))
        for dof in self.fixed:
            node, component = divmod(dof, 6)
            if component < 3:
                if state.positions[node, component] != self.reference_positions[node, component] or state.position_low[node, component] != 0.:
                    raise ValueError('retained state violates translation support')
            elif not np.array_equal(state.nodal_frames[node], self.reference_frames[node]):
                raise ValueError('retained state violates rotation support')
        return state

    def assemble(self, state, parameter):
        self.guard()
        residual = np.zeros(self.count); hessian = np.zeros((self.count, self.count))
        for i, probe in enumerate(self.probes):
            nodes = self.nodes[i]
            value = probe.evaluate(state.positions[nodes], state.position_low[nodes], state.nodal_frames[nodes],
                state.cell_rotations[i], state.resultants[i])
            residual[self.slots[i]] += value.residual
            hessian[np.ix_(self.slots[i], self.slots[i])] += value.hessian
        residual[:self.nodal_count] -= parameter*self.force
        scaled = residual/self.scales
        metrics = (float(np.linalg.norm(scaled[self.equilibrium])), float(np.linalg.norm(scaled[self.compatibility])))
        self.guard()
        return residual, hessian, metrics

    def advance(self, state, step):
        step = _owned(step)
        if step.shape != (self.count,) or np.any(step[self.fixed]):
            raise ValueError('supported retained increment required')
        data = {key: value.copy() for key, value in state.descriptor().items()}
        for node in range(len(self.node_ids)):
            for axis in range(3):
                data['positions'][node, axis], data['position_low'][node, axis] = split_sum((
                    float(state.positions[node, axis]), float(state.position_low[node, axis]), float(step[6*node+axis])))
            angular = step[6*node+3:6*node+6]
            if np.linalg.norm(angular) >= .9*np.pi: raise ValueError('retained nodal chart requires cutback')
            data['nodal_frames'][node] = rotation(angular)@state.nodal_frames[node]
        for i in range(len(self.elements)):
            first = self.nodal_count+24*i
            for cell in (0, 1):
                angular = step[first+3*cell:first+3*cell+3]
                if np.linalg.norm(angular) >= .9*np.pi: raise ValueError('retained cell chart requires cutback')
                data['cell_rotations'][i, cell] = rotation(angular)@state.cell_rotations[i, cell]
            data['resultants'][i] += step[first+6:first+24]
        return self.make(data)

    def recover(self, state):
        self.guard()
        result = tuple(dict(element_id=eid, stations=probe.recover(state.cell_rotations[i], state.resultants[i]))
            for i, ((eid, _), probe) in enumerate(zip(self.elements, self.probes)))
        self.guard()
        return result

    def checkpoint(self, state, cursor, records):
        parameter = 0. if cursor == 0 else self.program.targets[cursor-1]
        residual, _, metrics = self.assemble(state, parameter)
        if max(metrics) > 1e-11: raise ValueError('only equilibrated compatible retained states may commit')
        body = dict(schema=SCHEMA, formulation_id=POLICY, model_sha256=self.identity,
            program=self.program_data, node_ids=self.node_ids, element_ids=[i for i, _ in self.elements],
            completed_targets=cursor, load_parameter=parameter, records=records,
            state=state.descriptor(), residual=residual, recovery_sha256=sha(self.recover(state)))
        return canonical({**body, 'checkpoint_sha256': sha(body)})

    def restore(self, raw):
        if type(raw) is not bytes: raise ValueError('canonical retained checkpoint bytes required')
        value = _load(raw.decode('ascii'))
        keys = {'schema', 'formulation_id', 'model_sha256', 'program', 'node_ids', 'element_ids',
            'completed_targets', 'load_parameter', 'records', 'state', 'residual', 'recovery_sha256', 'checkpoint_sha256'}
        if type(value) is not dict or set(value) != keys: raise ValueError('exact retained checkpoint schema required')
        body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
        if (value['schema'] != SCHEMA or value['formulation_id'] != POLICY or value['model_sha256'] != self.identity
                or value['checkpoint_sha256'] != sha(body) or canonical(value['program']) != canonical(self.program_data)
                or canonical(value['node_ids']) != canonical(self.node_ids)
                or canonical(value['element_ids']) != canonical([i for i, _ in self.elements])):
            raise ValueError('retained checkpoint model/program/formulation mismatch')
        cursor = value['completed_targets']
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets): raise ValueError('retained checkpoint cursor')
        parameter = 0. if cursor == 0 else self.program.targets[cursor-1]
        if type(value['load_parameter']) is not float or value['load_parameter'] != parameter:
            raise ValueError('retained checkpoint load parameter')
        records = value['records']
        if type(records) is not list or len(records) != cursor: raise ValueError('retained checkpoint records')
        for i, row in enumerate(records):
            if (type(row) is not dict or set(row) != {'target', 'parameter', 'iterations'}
                    or type(row['target']) is not int or row['target'] != i+1
                    or type(row['parameter']) is not float or row['parameter'] != self.program.targets[i]
                    or type(row['iterations']) is not int or not 0 <= row['iterations'] <= self.program.max_iterations):
                raise ValueError('retained checkpoint step record')
        state = self.make(value['state'], decoded=True)
        if cursor == 0 and canonical(state.descriptor()) != canonical(self.initial.descriptor()):
            raise ValueError('retained checkpoint virgin state differs')
        # Regenerate the complete reaction/recovery-bound capsule, not only its hash.
        if self.checkpoint(state, cursor, records) != raw:
            raise ValueError('retained checkpoint reaction/recovery replay mismatch')
        return state, cursor, records
