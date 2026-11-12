"""Objective rigid offset joint between two authenticated spatial pose ports.

Port order is shell/master, beam/slave. This is the pose-level operator only:
it does not authenticate shell/beam accepted states, register an MPC, assemble
an FEModel, or qualify a beam-shell connection. Both ports use spatial LEFT
multiplicative variations. An additive shell chart needs its own exact pullback.
No penalty stiffness or fictitious joint mass is introduced.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import fsum, isfinite
import numpy as np

from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log
from ._ge_beam3_p5.algebra import skew
from ._ge_beam3_p5_seeded.core import canonical
from ._native_reference_modal import _owned
from .control import cancellation_safe_point

POLICY = 'GE_BEAM3_OBJECTIVE_MASTER_FRAME_OFFSET_POSE_JOINT_V1'
SCHEMA = 'GE_BEAM3_OFFSET_POSE_JOINT_DEFINITION_V1'
MAX_BYTES = 16384


def _array(value, shape):
    if (type(value) is not np.ndarray or value.dtype != np.dtype('float64')
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError('exact finite binary64 joint array required')
    return _owned(value)


def _frames(value):
    made = _array(value, (2, 3, 3))
    for q in made:
        if np.linalg.norm(q.T@q-np.eye(3)) > 1e-11 or abs(np.linalg.det(q)-1.) > 1e-11:
            raise ValueError('joint port frames must be proper rotations')
    return made


def _difference(positions, low):
    return np.array([fsum((positions[1, i], -positions[0, i], low[1, i], -low[0, i]))
        for i in range(3)])


@dataclass(frozen=True)
class PoseJointEvaluation:
    joint_sha256: str
    input_sha256: str
    potential: float
    constraints: np.ndarray
    constraint_jacobian: np.ndarray
    residual: np.ndarray
    chart_hessian: np.ndarray
    spatial_jacobian: np.ndarray
    production_qualified: bool = field(default=False, init=False)
    owner_binding_authorized: bool = field(default=False, init=False)


@dataclass(frozen=True)
class PoseJointPortEvaluation:
    pose: PoseJointEvaluation
    port_input_sha256: str
    charted_ports: tuple
    residual: np.ndarray
    tangent: np.ndarray
    coordinate_map: np.ndarray
    frames: np.ndarray
    production_qualified: bool = field(default=False, init=False)
    owner_binding_authorized: bool = field(default=False, init=False)


def _exp_terms(vector):
    """Analytic left Exp differential and derivative; no Log of total angles.

Unlike the native incremental chart helper, total shell coordinates may exceed
0.9*pi. A singular additive chart must be rebased by its actual state owner.
Common finite rotation remains admissible to the matrix-valued pose operator.
"""
    jets = so3_exp([Jet2.variable(float(v), i, 3) for i, v in enumerate(vector)])
    q = np.array([[v.value for v in row] for row in jets])
    dq = np.array([[v.gradient for v in row] for row in jets])
    ddq = np.array([[v.hessian for v in row] for row in jets])
    def axial(m):
        return np.array([m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1]])/2
    a = np.column_stack([axial(dq[:, :, j]@q.T) for j in range(3)])
    da = np.empty((3, 3, 3))
    for j in range(3):
        for k in range(3):
            da[:, j, k] = axial(ddq[:, :, j, k]@q.T+dq[:, :, j]@dq[:, :, k].T)
    return q, a, da


class RigidPoseJoint:
    """Reference offsets and relative orientations are physical supplied data.

With e=Qs0.T (rb0-rs0), A=Qs0.T Qb0, constraints are
ct=Qs.T (rb-rs)-e and cr=Log(A.T Qs.T Qb).
Multipliers are components in these material bases, not spatial dead moments.
"""
    __slots__ = ('reference_positions', 'reference_frames', '_offset', '_relative',
                 'identity', '_captured_identity', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('joint definition is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, reference_positions, reference_frames):
        self.reference_positions = _array(reference_positions, (2, 3))
        self.reference_frames = _frames(reference_frames)
        self._offset = _owned(self.reference_frames[0].T @ _difference(self.reference_positions, np.zeros((2, 3))))
        self._relative = _owned(self.reference_frames[0].T @ self.reference_frames[1])
        self.identity = sha256(canonical(self._descriptor())).hexdigest()
        self._captured_identity = self.identity
        self._sealed = True

    def _descriptor(self):
        return dict(schema=SCHEMA, policy=POLICY, port_order=['shell-master', 'beam-slave'],
            reference_positions=self.reference_positions, reference_frames=self.reference_frames,
            multiplier_basis='MATERIAL_MASTER_TRANSLATION_BEAM_REFERENCE_ROTATION',
            update='SPATIAL_LEFT_MULTIPLICATIVE_BOTH_PORTS', production_qualified=False)

    def guard(self):
        if self.identity != self._captured_identity or sha256(canonical(self._descriptor())).hexdigest() != self.identity:
            raise ValueError('joint definition changed')
        offset = self.reference_frames[0].T @ _difference(self.reference_positions, np.zeros((2, 3)))
        relative = self.reference_frames[0].T @ self.reference_frames[1]
        if not np.array_equal(offset, self._offset) or not np.array_equal(relative, self._relative):
            raise ValueError('joint compiled reference changed')

    def to_bytes(self):
        self.guard()
        return canonical(self._descriptor())

    @classmethod
    def from_bytes(cls, raw, *, expected_sha256):
        if (type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES or type(expected_sha256) is not str
                or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError('bounded externally hash-bound joint definition required')
        def pairs(rows):
            data = {}
            for key, value in rows:
                if key in data:
                    raise ValueError('duplicate joint definition key')
                data[key] = value
            return data
        def number(text):
            value = float(text)
            if not isfinite(value):
                raise ValueError('nonfinite joint number')
            return value
        def forbidden(text):
            raise ValueError('nonfinite joint definition')
        try:
            data = json.loads(raw.decode('ascii'), object_pairs_hook=pairs,
                parse_float=number, parse_constant=forbidden)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError('invalid bounded joint JSON') from error
        keys = {'schema', 'policy', 'port_order', 'reference_positions', 'reference_frames',
                'multiplier_basis', 'update', 'production_qualified'}
        if type(data) is not dict or set(data) != keys or canonical(data) != raw:
            raise ValueError('strict canonical joint definition required')
        arrays = []
        for name, shape in (('reference_positions', (2, 3)), ('reference_frames', (2, 3, 3))):
            values = np.asarray(data[name], dtype=object)
            if values.shape != shape or any(type(x) is not float for x in values.flat):
                raise ValueError('strict joint definition array')
            arrays.append(np.array(values, dtype=float))
        made = cls(*arrays)
        if made.to_bytes() != raw:
            raise ValueError('joint definition policy or identity mismatch')
        return made

    def evaluate(self, positions, frames, multipliers, *, position_low=None, frames_low=None, cancellation_token=None):
        cancellation_safe_point(cancellation_token, 'pose-joint.start')
        self.guard()
        x = _array(positions, (2, 3)); q = _frames(frames); force = _array(multipliers, (6,))
        low = np.zeros((2, 3)) if position_low is None else _array(position_low, (2, 3))
        if frames_low is not None:
            from ._stiffened_shell_f06_increment_arithmetic import validate_frames, pose_evaluation
            qlow = _array(frames_low, (2, 3, 3))
            validate_frames(q, qlow, 'point frames')
            value, constraints, jacobian, residual, chart = pose_evaluation(
                x, low, q, qlow, force, self._offset, self._relative)
            spatial = chart.copy()
            for start in (3, 9):
                spatial[start:start+3, start:start+3] -= .5*skew(residual[start:start+3])
            fingerprint = sha256(canonical(dict(positions=x, position_low=low,
                frames=q, frames_low=qlow, multipliers=force))).hexdigest()
            result = PoseJointEvaluation(self.identity, fingerprint, value, _owned(constraints),
                _owned(jacobian), _owned(residual), _owned(chart), _owned(spatial))
            self.guard()
            cancellation_safe_point(cancellation_token, 'pose-joint.complete')
            return result
        gap = _difference(x, low)
        fingerprint = sha256(canonical(dict(positions=x, position_low=low, frames=q, multipliers=force))).hexdigest()
        variables = [Jet2.variable(0., i, 18) for i in range(12)]
        variables += [Jet2.variable(float(v), 12+i, 18) for i, v in enumerate(force)]
        made = [matmul(so3_exp(variables[6*n+3:6*n+6]), constant_matrix(q[n], 18)) for n in range(2)]
        distance = [Jet2.constant(float(gap[i]), 18)+variables[6+i]-variables[i] for i in range(3)]
        ct = [v-float(e) for v, e in zip(matvec(transpose(made[0]), distance), self._offset)]
        relative = matmul(constant_matrix(self._relative.T, 18), matmul(transpose(made[0]), made[1]))
        cr = so3_log(relative)
        constraints = ct+cr
        potential = sum((v*variables[12+i] for i, v in enumerate(constraints)), Jet2.constant(0., 18))
        spatial = potential.hessian.copy()
        # The derivative of the spatial residual is NOT just its Exp-chart
        # Hessian away from equilibrium: H = Dr + .5*skew(r_rotation).
        for start in (3, 9):
            spatial[start:start+3, start:start+3] -= .5*skew(potential.gradient[start:start+3])
        result = PoseJointEvaluation(self.identity, fingerprint, potential.value,
            _owned([v.value for v in constraints]), _owned([v.gradient[:12] for v in constraints]),
            _owned(potential.gradient), _owned(potential.hessian), _owned(spatial))
        if not isfinite(result.potential):
            raise ValueError('nonfinite joint work')
        self.guard()
        cancellation_safe_point(cancellation_token, 'pose-joint.complete')
        return result

    def evaluate_ports(self, positions, anchors, multipliers, *, rotation_coordinates,
                       charted_ports, position_low=None, anchors_low=None, cancellation_token=None):
        """Exact mixed-chart interface, not an accepted shell-state adapter.

For a charted port Q=Exp(theta) anchor and theta varies additively. For a
spatial port anchor IS the authoritative current Q and theta must be zero;
its next variation is multiplicative. State owners must supply these anchors.
"""
        cancellation_safe_point(cancellation_token, 'pose-joint.ports-start')
        self.guard()
        if type(charted_ports) is not tuple or len(charted_ports) != 2 or any(type(v) is not bool for v in charted_ports):
            raise ValueError('two explicit joint port chart flags required')
        anchors = _frames(anchors)
        anchor_low = None if anchors_low is None else _array(anchors_low, (2, 3, 3))
        if anchor_low is not None and any(charted and np.any(anchor_low[i]) for i, charted in enumerate(charted_ports)):
            raise ValueError('compensated charted point anchor is not implemented')
        coordinates = _array(rotation_coordinates, (2, 3))
        frames = []; terms = []
        for i, charted in enumerate(charted_ports):
            if not charted:
                if np.any(coordinates[i] != 0.):
                    raise ValueError('spatial port requires zero coordinates and its current matrix anchor')
                frames.append(anchors[i]); terms.append(None)
                continue
            q, a, da = _exp_terms(coordinates[i])
            singular = np.linalg.svd(a, compute_uv=False)
            if not np.isfinite(singular).all() or singular[-1] <= 1e-11*max(1., singular[0]):
                raise ValueError('singular additive rotation chart; state owner must rebase')
            frames.append(q@anchors[i]); terms.append((a, da))
        frames = _frames(np.array(frames))
        pose = self.evaluate(positions, frames, multipliers, position_low=position_low, frames_low=anchor_low,
            cancellation_token=cancellation_token)
        coordinate_map = np.eye(18); extra = np.zeros((18, 18))
        for i, term in enumerate(terms):
            if term is None:
                continue
            a, da = term; start = 6*i+3
            coordinate_map[start:start+3, start:start+3] = a
            for k in range(3):
                extra[start:start+3, start+k] = da[:, :, k].T@pose.residual[start:start+3]
        residual = coordinate_map.T@pose.residual
        tangent = coordinate_map.T@pose.spatial_jacobian@coordinate_map+extra
        fingerprint = sha256(canonical(dict(pose_input=pose.input_sha256, anchors=anchors,
            coordinates=coordinates, charted_ports=charted_ports))).hexdigest()
        self.guard()
        cancellation_safe_point(cancellation_token, 'pose-joint.ports-complete')
        return PoseJointPortEvaluation(pose, fingerprint, charted_ports, _owned(residual),
            _owned(tangent), _owned(coordinate_map), frames)
