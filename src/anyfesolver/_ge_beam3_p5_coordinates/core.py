"""Coordinate-state successor to the frozen internal P5 package.

Private development candidate: no public selector or qualification authority.
Unchanged physical operators are imported from the preserved P5 package.
"""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import numpy as np
from anyfesolver.elements import Element
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from anyfesolver._ge_beam3_p5.compensated import CompensatedStationaryBeam
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from anyfesolver._ge_beam3_p5.mixed import LocalForceAccuracy
from anyfesolver._ge_beam3_p5.section import DirectedHardeningSection, SectionHistory, _history
from .positions import POLICY, from_total, validate_total_pair

SCHEMA='GE_BEAM3_P5_COORDINATE_STATION_STATE_V2'


FORMULATION='CANDIDATE_GE_BEAM3_P5_COORDINATE_CORE_V2'


def canonical(value):
    def encode(item):
        if is_dataclass(item): return asdict(item)
        if isinstance(item, np.ndarray): return item.tolist()
        raise TypeError('unsupported private state value')
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
                       default=encode)+'\n').encode('ascii')


def sha(value): return hashlib.sha256(canonical(value)).hexdigest()


def seal(state):
    body={key: value for key, value in state.items() if key!='state_sha256'}
    return {**deepcopy(body), 'state_sha256': sha(body)}


class NativeMaterialError(ValueError):
    """Private P5 state/model/route mismatch; no accepted state changed."""


class StationaryCore(Element):
    formulation_native_total_lagrangian=True
    formulation_id=FORMULATION
    num_nodes=3
    dofs_per_node=6

    def __init__(self, element_id, node_ids, reference, section, *, order=8):
        if type(element_id) is not int or type(node_ids) is not tuple or len(node_ids)!=3:
            raise NativeMaterialError('exact element ID and three-node tuple required')
        if any(type(i) is not int for i in node_ids) or len(set(node_ids))!=3:
            raise NativeMaterialError('distinct integer nodes required')
        if type(section) is not DirectedHardeningSection:
            raise NativeMaterialError('only declared directed-hardening section is implemented')
        super().__init__(element_id, node_ids, material_name=f'private-p5-section-{element_id}')
        self.reference=CurvedBeam3ReferenceGeometry(reference.coordinates, reference.nodal_triads)
        self.section=deepcopy(section);self.order=order
        # Validate quadrature/section now, without solving or generating history.
        CompensatedStationaryBeam(self.reference, self.section, order=order, position_low=np.zeros((3, 3)))
        self.length=max(np.linalg.norm(a-b) for a in self.reference.coordinates for b in self.reference.coordinates)
        self.model_sha256=sha(self._identity())
        self._native_session=None

    def _identity(self):
        return dict(formulation=FORMULATION, schema=SCHEMA, coordinate_policy=POLICY, element=self.element_id, nodes=self.node_ids,
            coordinates=self.reference.coordinates, frames=self.reference.nodal_triads,
            elastic=self.section._elastic, direction=self.section._direction,
            yield_force=self.section._yield, hardening=self.section._hardening, order=self.order)

    def get_node_coordinates(self, mesh):
        return np.array([mesh.get_node(i).coords() for i in self.node_ids], dtype=float)

    def _check_model(self, mesh):
        if type(self) is not StationaryCore or sha(self._identity())!=self.model_sha256:
            raise NativeMaterialError('private model identity changed')
        expected_length=max(np.linalg.norm(a-b) for a in self.reference.coordinates for b in self.reference.coordinates)
        if self.length!=expected_length: raise NativeMaterialError('local accuracy length changed')
        if not np.array_equal(self.get_node_coordinates(mesh), self.reference.coordinates):
            raise NativeMaterialError('mesh reference geometry disagrees with frozen element')

    def native_reference_directors(self, mesh):
        self._check_model(mesh)
        return self.reference.nodal_triads[:, :, 2].copy()

    def _solve(self, positions, operators, origins, *, total):
        positions, low = from_total(self.reference.coordinates, total, positions)
        model=CompensatedStationaryBeam(self.reference, self.section, order=self.order,
            position_low=low, origins=origins)
        return model.solve(positions, operators@self.reference.nodal_triads,
            force_accuracy=LocalForceAccuracy(1e-12, float(self.length)))

    def _state(self, epoch, total, positions, operators, origins, response):
        positions, low = from_total(self.reference.coordinates, total, positions)
        return seal(dict(schema=SCHEMA, model_sha256=self.model_sha256, epoch=epoch,
            committed_total_u=np.array(total, copy=True), committed_positions=np.array(positions, copy=True),
            coordinate_policy=POLICY, committed_position_low=np.array(low, copy=True),
            committed_nodal_rotation_matrices=np.array(operators, copy=True),
            origins=tuple(origins), histories=tuple(s.response.history for s in response.stations), response=response))

    def init_model_bound_nonlinear_state(self, mesh, material=None, num_layers=1):
        self._arguments(mesh, material, num_layers)
        operators=np.tile(np.eye(3), (3, 1, 1));origins=tuple(SectionHistory() for _ in range(2*self.order))
        response=self._solve(self.reference.coordinates, operators, origins, total=np.zeros(18))
        return self._state(0, np.zeros(18), self.reference.coordinates, operators, origins, response)

    def _arguments(self, mesh, material, num_layers):
        self._check_model(mesh)
        if (material is not None and material is not self.section) or type(num_layers) is not int or num_layers!=1:
            raise NativeMaterialError('exact owned section or None and one private state layout required')

    def validate_model_bound_nonlinear_state(self, mesh, material, state, num_layers,
                                            *, expected_committed_total_u=None):
        self._arguments(mesh, material, num_layers)
        keys={'schema', 'model_sha256', 'epoch', 'committed_total_u', 'committed_positions',
              'coordinate_policy', 'committed_position_low',
              'committed_nodal_rotation_matrices', 'origins', 'histories', 'response', 'state_sha256'}
        if type(state) is not dict or set(state)!=keys:
            raise NativeMaterialError('exact native station state schema required')
        body={k: v for k, v in state.items() if k!='state_sha256'}
        if state['schema']!=SCHEMA or state['model_sha256']!=self.model_sha256 or state['state_sha256']!=sha(body):
            raise NativeMaterialError('native station identity/hash mismatch')
        if type(state['epoch']) is not int or not 0<=state['epoch']<2**31:
            raise NativeMaterialError('bounded exact station epoch required')
        u=_array(state['committed_total_u'], (18,), 'committed displacement')
        positions=_array(state['committed_positions'], (3, 3), 'committed coordinates')
        operators=_frames(state['committed_nodal_rotation_matrices'], 3, 'committed operators')
        if expected_committed_total_u is not None and not np.array_equal(u, expected_committed_total_u):
            raise NativeMaterialError('state/global displacement mismatch')
        validate_total_pair(self.reference.coordinates, u, positions,
                            state['committed_position_low'], state['coordinate_policy'])
        for name in ('origins', 'histories'):
            if type(state[name]) is not tuple or len(state[name])!=2*self.order:
                raise NativeMaterialError('exact station ordering/count required')
            for value in state[name]: _history(value)
        if state['epoch']==0:
            if (np.any(u) or not np.array_equal(positions, self.reference.coordinates)
                or not np.array_equal(operators, np.tile(np.eye(3), (3, 1, 1)))
                or any(value!=SectionHistory() for value in state['origins'])):
                raise NativeMaterialError('initial state must be declared stress-free origin')
        # Replay from the PREVIOUS accepted origins, never the new plastic history.
        replay=self._solve(positions, operators, state['origins'], total=u)
        if sha(replay)!=sha(state['response']):
            raise NativeMaterialError('accepted-origin local replay mismatch')
        if tuple(s.response.history for s in replay.stations)!=state['histories']:
            raise NativeMaterialError('station history/recovery mismatch')
        return deepcopy(state)



    def _unsupported(self, *args, **kwargs):
        raise NativeMaterialError('private native station adapter: workflow not authorized')

    compute_stiffness_matrix=_unsupported
    compute_mass_matrix=_unsupported
    compute_geometric_stiffness_matrix=_unsupported
    compute_internal_forces=_unsupported
    compute_stresses=_unsupported
    to_dict=_unsupported
