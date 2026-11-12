"""Private physical B2 matrix adapter; no graph/restart/public admission."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import numpy as np
from ._ge_beam3_g3c_b2_physical import evaluate as core_evaluate, physical_rigidities
from ._ge_beam3_g3c_b2_physical import OPERATOR_ID, RepresentabilityError, owned
from ._ge_beam3_g3c_stable.beam import deformation, array, rotations, Kinematics
from ._ge_beam3_g3c_stable.joint import _exp_terms

POLICY = 'GE_BEAM3_G3C_B2_PHYSICAL_MATRIX_ADAPTER_V1'
SCHEMA = 'GE_BEAM3_G3C_B2_PHYSICAL_MATRIX_DESCRIPTOR_V1'


class AdmissionError(ValueError):
    """Initial inherited-chart admission; not full geometry-domain parity."""


def scalar(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AdmissionError('finite nonboolean scalar required')
    return float(value)


@dataclass(frozen=True)
class PhysicalTrial:
    kinematics: Kinematics
    energy: float
    local_force: np.ndarray
    local_tangent: np.ndarray
    chart_force: np.ndarray
    chart_hessian: np.ndarray
    spatial_force: np.ndarray
    spatial_row_chart_tangent: np.ndarray
    spatial_input_tangent: np.ndarray
    chart_to_spatial: np.ndarray
    strains: np.ndarray
    resultants: np.ndarray
    strain_differential: np.ndarray
    chart_strain_differential: np.ndarray
    stations: np.ndarray
    weights: np.ndarray
    reference_frame: np.ndarray
    current_frame: np.ndarray
    definition_sha256: str
    policy: str = field(default=POLICY, init=False)
    operator_id: str = field(default=OPERATOR_ID, init=False)
    production_qualified: bool = field(default=False, init=False)
    state_committed: bool = field(default=False, init=False)


class PhysicalB2Adapter:
    __slots__ = ('_body', '_seal')

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError('write-once physical definition')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('physical definition cannot be deleted')

    def __init__(self, node_ids, coordinates, anchor_node, E, nu, section,
                 *, policy=POLICY, operator_id=OPERATOR_ID, schema=SCHEMA):
        if type(self) is not PhysicalB2Adapter:
            raise AdmissionError('exact physical B2 adapter required')
        if (policy != POLICY or operator_id != OPERATOR_ID or schema != SCHEMA
                or type(policy) is not str or type(operator_id) is not str or type(schema) is not str):
            raise AdmissionError('old or wrong physical adapter identity')
        if (type(node_ids) is not tuple or len(node_ids) != 2
                or any(type(n) is not int or n <= 0 for n in node_ids)
                or len(set(node_ids)) != 2 or type(anchor_node) is not int or anchor_node not in node_ids):
            raise AdmissionError('two distinct physical nodes and anchor required')
        ref = array(coordinates, (2, 3))
        with np.errstate(over='ignore', invalid='ignore'):
            L = float(np.linalg.norm(ref[1]-ref[0]))
        if not math.isfinite(L) or L < 1e-12:
            raise AdmissionError('inherited finite length >=1e-12 required; geometry parity remains open')
        keys = {'area', 'Iy', 'Iz', 'J', 'shear_factor_y', 'shear_factor_z', 'orientation'}
        if type(section) is not dict or set(section) != keys:
            raise AdmissionError('closed history-free scalar section required')
        values = {k: scalar(section[k]) for k in keys-{'orientation'}}
        if min(values.values()) <= 0:
            raise AdmissionError('positive physical section required')
        raw = section['orientation']
        if type(raw) not in (list, tuple) or len(raw) != 3:
            raise AdmissionError('three physical orientation scalars required')
        normal = np.array([scalar(v) for v in raw])
        scale = float(np.max(abs(normal)))
        if scale == 0:
            raise AdmissionError('nonzero physical orientation required')
        normal /= scale; normal /= np.linalg.norm(normal)
        axis = (ref[1]-ref[0])/L
        transverse = normal-axis*(axis@normal)
        if not np.linalg.norm(transverse) > 1e-6*np.linalg.norm(normal):
            raise AdmissionError('parallel physical orientation; no fallback')
        transverse /= np.linalg.norm(transverse)
        frame = owned(np.column_stack((axis, np.cross(transverse, axis), transverse)))
        e, poisson = scalar(E), scalar(nu)
        if e <= 0 or not -1 < poisson < .5:
            raise AdmissionError('physical isotropic material required')
        values['orientation'] = normal.tolist()
        body = dict(schema=schema, policy=policy, operator_id=operator_id,
                    node_ids=list(node_ids), coordinates=ref.tolist(), anchor_node=anchor_node,
                    E=e, nu=poisson, section=values, reference_frame=frame.tolist(), length=L,
                    update='SPATIAL_LEFT_EXP_ACCEPTED_MATRIX',
                    recovery='PHYSICAL_HQ_S_INVERSE_REFERENCE_SECTION', quadrature='GAUSS_LEGENDRE_3')
        self._body = (json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')
        self._seal = sha256(self._body).hexdigest()

    def descriptor(self):
        if (type(self) is not PhysicalB2Adapter or type(self._body) is not bytes
                or type(self._seal) is not str or sha256(self._body).hexdigest() != self._seal):
            raise AdmissionError('physical definition changed')
        value = json.loads(self._body)
        if (value['policy'] != POLICY or value['operator_id'] != OPERATOR_ID or value['schema'] != SCHEMA):
            raise AdmissionError('old physical definition identity')
        return value

    def evaluate(self, displacement, accepted_rotations):
        entry_body, entry_seal = self._body, self._seal
        description = self.descriptor()
        if self._body != entry_body or self._seal != entry_seal:
            raise AdmissionError('physical definition replaced during entry observation')
        u, qa = array(displacement, (12,)), rotations(accepted_rotations, 2)
        ref = np.array(description['coordinates'])
        try:
            kin = deformation(ref, u, qa, description['node_ids'].index(description['anchor_node']))
        except ValueError as exc:
            raise AdmissionError(str(exc)) from exc
        frame = np.array(description['reference_frame']); T = np.kron(np.eye(4), frame.T)
        section = description['section']
        rigidities = physical_rigidities(description['E'], description['nu'],
            *(section[k] for k in ('area', 'Iy', 'Iz', 'J', 'shear_factor_y', 'shear_factor_z')))
        core = core_evaluate(description['length'], rigidities, owned(T@kin.deformation))
        f, K = T.T@core.force, T.T@core.tangent@T
        D, second = kin.differential, kin.second
        g = D.T@f; H = D.T@K@D + np.einsum('i,ijk->jk', f, second)
        P = np.eye(12); terms = []
        for i, row in enumerate(u.reshape(2, 6)):
            _, J, dJ = _exp_terms(row[3:]); sl = slice(6*i+3, 6*i+6)
            P[sl, sl] = J; terms.append((sl, dJ))
        try:
            r = np.linalg.solve(P.T, g); correction = np.zeros((12, 12))
            for sl, dJ in terms:
                correction[sl, sl] = np.einsum('ijk,i->jk', dJ, r[sl])
            A = np.linalg.solve(P.T, H-correction)
            spatial = np.linalg.solve(P.T, A.T).T
        except np.linalg.LinAlgError as exc:
            raise RepresentabilityError('physical B2 chart solve failed') from exc
        differential = core.strain_differential@T
        result = PhysicalTrial(kin, core.energy, owned(f), owned(K), owned(g), owned(H),
            owned(r), owned(A), owned(spatial), owned(P), core.strains, core.resultants,
            owned(differential), owned(differential@D), core.stations, core.weights,
            owned(frame), owned(kin.frame@frame), entry_seal)
        self.descriptor()
        if self._body != entry_body or self._seal != entry_seal:
            raise AdmissionError('physical definition replaced during evaluation')
        return result
