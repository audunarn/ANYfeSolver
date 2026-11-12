"""Private FEModel-bound physical-fibre transactions, replay and recovery.

Only geometric make/advance operations are reused from the older retained
layout. No old element, section, assembly, recovery or state capsule is used.
"""
from dataclasses import dataclass
from decimal import localcontext
from hashlib import sha256
from time import monotonic
import numpy as np
from scipy import sparse

from .assembly import build_constraint_transformation
from ._ge_beam3_retained_state import Context as GeometricOperations, State as MechanicalState
from ._ge_beam3_seeded_load_program import ForceProgram
from ._ge_beam3_retained_fibre_element import NativeRetainedFibreElement
from ._ge_beam3_retained_fibre import POLICY
from ._ge_beam3_fibre_cell import CellHistory
from ._ge_beam3_fibre_section import FibreHistory
from ._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from ._native_reference_modal import _owned
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES
from ._ge_beam3_effective_deadline import effective_deadline_seconds


SCHEMA = 'GE_BEAM3_RETAINED_PHYSICAL_FIBRE_ACCEPTED_CHAIN_V1'
PROGRAM = 'GE_BEAM3_RETAINED_PHYSICAL_FIBRE_FORCE_PROGRAM_V1'
# The original default remains unchanged. The separately selected 512-coordinate
# budget bounds one binary64 square workspace to 2 MiB (not total process RSS).
MAX_RETAINED_COORDINATES = 256
ADMITTED_DENSE_COORDINATE_BUDGETS = (256, 512)


class Layout:
    make = GeometricOperations.make
    advance = GeometricOperations.advance

    def __init__(self, model, program, *, max_coordinates=MAX_RETAINED_COORDINATES, nodal_moments=None):
        if type(program) is not ForceProgram or any(type(x) is not float for x in program.targets):
            raise ValueError('explicit binary64 retained fibre force program required')
        self.model, self.program = model, program
        if type(max_coordinates) is not int or max_coordinates not in ADMITTED_DENSE_COORDINATE_BUDGETS:
            raise ValueError('explicit admitted dense coordinate budget required')
        self.max_coordinates = max_coordinates
        if nodal_moments is not None and type(nodal_moments) is not SpatialNodalMoments:
            raise ValueError('exact explicit spatial nodal moment pattern required')
        self.nodal_moments = nodal_moments
        self.elements = tuple(sorted(model.mesh.elements.items()))
        self.nodal_count = model.mesh.dof_manager.total_dofs
        self.count = self.nodal_count+24*len(self.elements)
        if not self.elements or not 1 <= self.count <= self.max_coordinates:
            raise ValueError('bounded standalone retained fibre model required')
        if any(type(e) is not NativeRetainedFibreElement for _, e in self.elements):
            raise ValueError('exact retained fibre elements only; mixed formulations/joints are not admitted')
        self.node_ids = tuple(sorted(model.mesh.nodes, key=lambda i: model.mesh.dof_manager.get_node_dofs(i)[0]))
        if self.nodal_count != 6*len(self.node_ids): raise ValueError('complete six-DOF nodes required')
        for i, node in enumerate(self.node_ids):
            if type(node) is not int or tuple(model.mesh.dof_manager.get_node_dofs(node)) != tuple(range(6*i, 6*i+6)):
                raise ValueError('exact contiguous six-DOF node blocks required')
        _, _, transform, offset, free, info = build_constraint_transformation(
            sparse.eye(self.nodal_count, format='csr'), np.zeros(self.nodal_count), model)
        if (info['slave_dofs'] or np.any(offset) or len(free) == self.nodal_count
                or transform.nnz != len(free) or np.any(transform.data != 1.)):
            raise ValueError('homogeneous supported standalone model without MPCs required')
        self.fixed = sorted(set(range(self.nodal_count))-set(free))
        for i in range(len(self.node_ids)):
            if len(set(range(6*i+3, 6*i+6)) & set(self.fixed)) not in (0, 3):
                raise ValueError('partial rotation supports require a qualified constraint contract')
        self.node_index = {node: i for i, node in enumerate(self.node_ids)}
        self.force = np.zeros(self.nodal_count)
        for node, *force in program.nodal_forces:
            if node not in self.node_index: raise ValueError('force references an absent fibre node')
            i = self.node_index[node]; self.force[6*i:6*i+3] = force
        if nodal_moments is not None:
            for node, *moment in nodal_moments.rows:
                if node not in self.node_index: raise ValueError('moment references an absent fibre node')
                i = self.node_index[node]; self.force[6*i+3:6*i+6] = moment
        self.force = _owned(self.force)
        self.probes = tuple(e.operator for _, e in self.elements)
        self.free = [int(v) for v in free]+list(range(self.nodal_count, self.count))
        self.equilibrium = [int(v) for v in free]; self.compatibility = []; self.slots = []; self.nodes = []
        self.scales = np.ones(self.count); frames = [None]*len(self.node_ids)
        length = max(float(np.linalg.norm(p.reference.coordinates[-1]-p.reference.coordinates[0])) for p in self.probes)
        if not np.isfinite(length) or length <= 0: raise ValueError('positive characteristic beam length required')
        for i in range(len(frames)): self.scales[6*i+3:6*i+6] = length
        for i, (_, element) in enumerate(self.elements):
            first = self.nodal_count+24*i; nodes = [self.node_index[node] for node in element.node_ids]
            self.nodes.append(nodes)
            self.slots.append(list(element.get_dof_mapping(model.mesh))+list(range(first, first+24)))
            self.equilibrium.extend(range(first, first+6)); self.compatibility.extend(range(first+6, first+24))
            self.scales[first:first+12] = length
            for local, node in enumerate(nodes):
                frame = element.operator.reference.nodal_triads[local]
                if frames[node] is not None and not np.array_equal(frames[node], frame):
                    raise ValueError('shared-node physical material-frame authority differs')
                frames[node] = frame
        if any(frame is None for frame in frames): raise ValueError('unconnected fibre node not admitted')
        self.reference_positions = _owned([model.mesh.nodes[i].coords() for i in self.node_ids])
        self.reference_frames = _owned(frames)
        self.identity = self.snapshot()
        self.initial = self.make(dict(positions=self.reference_positions, position_low=np.zeros_like(self.reference_positions),
            nodal_frames=self.reference_frames, cell_rotations=np.tile(np.eye(3), (len(self.elements), 2, 1, 1)),
            resultants=np.zeros((len(self.elements), 18))))
        self.guard()

    def snapshot(self):
        model = self.model
        if (tuple(sorted(model.mesh.elements.items())) != self.elements or model.mesh.element_activity is not None
                or model.constraint_equations or model.mesh.point_masses):
            raise ValueError('retained fibre ownership/activity/constraint mismatch')
        for element_id, element in self.elements:
            element.check(model.mesh)
            if element_id != element.element_id or model.materials.get(element.material_name) is not element.section:
                raise ValueError('retained fibre section/model ownership changed')
        return sha(dict(elements=[(i, e.to_dict(), list(e.get_dof_mapping(model.mesh))) for i, e in self.elements],
            nodes=[(i, node.coords()) for i, node in sorted(model.mesh.nodes.items())],
            boundaries=[vars(bc) for bc in model.boundary_conditions], dofs=model.mesh.dof_manager.total_dofs,
            constrained_dofs=sorted(model.mesh.dof_manager._constrained_dofs), program=self.program.descriptor(),
            **({'dense_coordinate_budget': self.max_coordinates} if self.max_coordinates != 256 else {}),
            **({'nodal_moments': self.nodal_moments.descriptor()} if self.nodal_moments is not None else {})))

    def guard(self):
        if self.snapshot() != self.identity: raise ValueError('retained fibre frozen model/program changed')


@dataclass(frozen=True)
class State:
    mechanical: MechanicalState
    origins: tuple
    histories: tuple
    completed_targets: int
    model_sha256: str


class Context:
    def __init__(self, model, program, *, check=None, max_coordinates=MAX_RETAINED_COORDINATES, nodal_moments=None):
        self.started = monotonic(); self.external_check = check
        moment_options = {} if nodal_moments is None else dict(nodal_moments=nodal_moments)
        self.layout = Layout(model, program, max_coordinates=max_coordinates, **moment_options); self.program = program; self.probes = self.layout.probes
        self.program_data = {**program.descriptor(), 'schema': PROGRAM,
            'line_search': 'RETAINED_EQUILIBRIUM_COMPATIBILITY_RESIDUAL_DECREASE_V1',
            **({'nodal_moments': nodal_moments.descriptor()} if nodal_moments is not None else {})}
        self.identity = sha(dict(formulation_id=POLICY, captured_model=self.layout.identity,
            operator_ids=[p.identity for p in self.probes], program=self.program_data, state_schema=SCHEMA))
        histories = tuple(p.cell.virgin() for p in self.probes)
        self.initial = State(self.layout.initial, histories, histories, 0, self.identity)
        self.genesis = self._record(self.initial.mechanical, histories, 0, 0, self.identity)[1]
        self.guard()

    def check(self):
        if self.external_check is not None: self.external_check()
        if monotonic()-self.started > effective_deadline_seconds(120.): raise RuntimeError('retained fibre context cooperative deadline')

    def guard(self):
        self.check(); self.layout.guard()
        for probe in self.probes: probe.guard()

    def history(self, values, *, decoded=False):
        if type(values) not in (list, tuple) or len(values) != len(self.probes):
            raise ValueError('one complete cell fibre history per element required')
        result = []
        with localcontext() as context:
            context.prec = 80
            for value, probe in zip(values, self.probes):
                if decoded:
                    if type(value) is not dict or set(value) != {'cell_identity', 'stations'} or type(value['stations']) is not list:
                        raise ValueError('exact decoded fibre cell history schema')
                    stations = []
                    for row in value['stations']:
                        if (type(row) is not dict or set(row) != {'section_identity', 'rows'} or type(row['rows']) is not list
                                or any(type(r) is not list for r in row['rows'])):
                            raise ValueError('exact decoded station fibre history schema')
                        stations.append(FibreHistory(row['section_identity'], tuple(map(tuple, row['rows']))))
                    value = CellHistory(value['cell_identity'], tuple(stations))
                probe.cell._origins(value); result.append(value)
        return tuple(result)

    def assemble(self, mechanical, parameter, origins):
        self.guard(); layout = self.layout; origins = self.history(origins)
        residual = np.zeros(layout.count); hessian = np.zeros((layout.count, layout.count)); responses = []
        for i, probe in enumerate(self.probes):
            nodes = layout.nodes[i]
            response = probe.evaluate(mechanical.positions[nodes], mechanical.position_low[nodes], mechanical.nodal_frames[nodes],
                mechanical.cell_rotations[i], mechanical.resultants[i], origin=origins[i], check=self.check)
            residual[layout.slots[i]] += response.residual
            hessian[np.ix_(layout.slots[i], layout.slots[i])] += response.hessian+response.hessian_low
            responses.append(response)
        residual[:layout.nodal_count] -= parameter*layout.force
        scaled = residual/layout.scales
        metrics = (float(np.linalg.norm(scaled[layout.equilibrium])), float(np.linalg.norm(scaled[layout.compatibility])))
        self.guard(); return residual, hessian, metrics, tuple(responses)

    def recover(self, state):
        if type(state) is not State or state.model_sha256 != self.identity: raise ValueError('fibre recovery state/model binding')
        self.guard(); rows = []
        origins = self.history(state.origins); histories = self.history(state.histories)
        for i, ((element_id, _), probe) in enumerate(zip(self.layout.elements, self.probes)):
            stations = probe.recover(state.mechanical.cell_rotations[i], state.mechanical.resultants[i], origin=origins[i], check=self.check)
            made = CellHistory(probe.cell.identity, tuple(row['history'] for row in stations))
            if canonical(made) != canonical(histories[i]): raise ValueError('physical fibre recovery does not reproduce accepted history')
            rows.append(dict(element_id=element_id, stations=stations))
        self.guard(); return tuple(rows)

    def _record(self, mechanical, origins, cursor, iterations, previous):
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets): raise ValueError('fibre target cursor')
        if type(iterations) is not int or not 0 <= iterations <= self.program.max_iterations: raise ValueError('fibre iteration record')
        parameter = 0. if cursor == 0 else self.program.targets[cursor-1]
        mechanical = self.layout.make(mechanical.descriptor()); origins = self.history(origins)
        residual, _, metrics, responses = self.assemble(mechanical, parameter, origins)
        if max(metrics) > 1e-11: raise ValueError('only equilibrated compatible fibre states may commit')
        histories = self.history(tuple(r.history for r in responses))
        state = State(mechanical, origins, histories, cursor, self.identity)
        body = dict(target=cursor, parameter=parameter, iterations=iterations, previous_sha256=previous,
            mechanical=mechanical.descriptor(), origins=origins, histories=histories, residual=residual, metrics=metrics,
            material_sha256=sha([r.material.decode('ascii') for r in responses]), recovery_sha256=sha(self.recover(state)))
        self.guard(); return state, canonical({**body, 'record_sha256': sha(body)})

    def stage(self, mechanical, accepted, records, iterations):
        if type(accepted) is not State or accepted.model_sha256 != self.identity or accepted.completed_targets != len(records):
            raise ValueError('fibre accepted cursor differs from chain')
        previous = _load((self.genesis if not records else records[-1]).decode('ascii'))['record_sha256']
        return self._record(mechanical, accepted.histories, len(records)+1, iterations, previous)

    def checkpoint(self, records):
        if type(records) is not tuple or len(records) > len(self.program.targets): raise ValueError('bounded immutable fibre chain required')
        body = dict(schema=SCHEMA, formulation_id=POLICY, model_sha256=self.identity, program=self.program_data,
            node_ids=self.layout.node_ids, element_ids=[i for i, _ in self.layout.elements], completed_targets=len(records),
            initial=_load(self.genesis.decode('ascii')), records=[_load(row.decode('ascii')) for row in records])
        raw = canonical({**body, 'checkpoint_sha256': sha(body)})
        if len(raw) > MAX_BYTES: raise ValueError('fibre checkpoint byte bound')
        self.guard(); return raw

    def restore(self, raw, *, expected_sha256=None):
        self.guard()
        if type(raw) is not bytes: raise ValueError('canonical fibre checkpoint bytes required')
        if expected_sha256 is not None and (type(expected_sha256) is not str or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError('external fibre checkpoint hash mismatch')
        value = _load(raw.decode('ascii'))
        keys = {'schema', 'formulation_id', 'model_sha256', 'program', 'node_ids', 'element_ids',
                'completed_targets', 'initial', 'records', 'checkpoint_sha256'}
        if type(value) is not dict or set(value) != keys: raise ValueError('exact fibre checkpoint schema')
        body = {key: member for key, member in value.items() if key != 'checkpoint_sha256'}
        if (value['schema'] != SCHEMA or value['formulation_id'] != POLICY or value['model_sha256'] != self.identity
                or value['checkpoint_sha256'] != sha(body) or canonical(value['program']) != canonical(self.program_data)
                or canonical(value['node_ids']) != canonical(self.layout.node_ids)
                or canonical(value['element_ids']) != canonical([i for i, _ in self.layout.elements])):
            raise ValueError('fibre checkpoint model/program/formulation binding')
        cursor = value['completed_targets']
        if type(cursor) is not int or not 0 <= cursor <= len(self.program.targets): raise ValueError('fibre checkpoint cursor')
        if type(value['records']) is not list or len(value['records']) != cursor: raise ValueError('fibre checkpoint record count')
        if canonical(value['initial']) != self.genesis: raise ValueError('fibre virgin genesis differs')
        accepted = self.initial; records = ()
        record_keys = {'target', 'parameter', 'iterations', 'previous_sha256', 'mechanical', 'origins', 'histories',
                       'residual', 'metrics', 'material_sha256', 'recovery_sha256', 'record_sha256'}
        for index, row in enumerate(value['records'], 1):
            self.guard()
            if (type(row) is not dict or set(row) != record_keys or type(row['target']) is not int or row['target'] != index
                    or type(row['parameter']) is not float or row['parameter'] != self.program.targets[index-1]):
                raise ValueError('fibre checkpoint target record')
            origins = self.history(row['origins'], decoded=True); self.history(row['histories'], decoded=True)
            if canonical(origins) != canonical(accepted.histories): raise ValueError('fibre origin chain is broken')
            mechanical = self.layout.make(row['mechanical'], decoded=True)
            proposed, regenerated = self.stage(mechanical, accepted, records, row['iterations'])
            if regenerated != canonical(row): raise ValueError('fibre state/reaction/recovery/history replay mismatch')
            accepted, records = proposed, (*records, regenerated)
        if self.checkpoint(records) != raw: raise ValueError('fibre canonical replay mismatch')
        self.guard(); return accepted, records
