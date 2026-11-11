"""Exact equilibrium-coordinate kinematics and covector pullbacks (development).

ADR-011: fixed equilibrium geometry/axes, moving material reference point,
principal rotation log. No BEM source, motion envelope or fluid contribution
is admitted by this utility. Potential-flow validity is a separate obligation.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import numpy as np
from anyfesolver.marine_rigid import (
    RigidBodyState, left_jacobian, left_jacobian_rate,
    left_jacobian_second_rate, skew)

API_VERSION = 'marine_hydro_coordinates_v1'
LOG_BRANCH_MARGIN = 1e-6
DOF_NAMES = ('surge', 'sway', 'heave', 'roll', 'pitch', 'yaw')


class MarineHydroCoordinatesError(ValueError):
    """Invalid geometry, branch, ordering or algebra input; no fallback."""


def _array(value, shape, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or any(
                isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
            raise ValueError('real numbers required')
        result = np.array(raw, dtype=float, copy=True)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f'finite shape {shape} required')
        return result
    except (TypeError, ValueError, OverflowError) as error:
        raise MarineHydroCoordinatesError(f'{name}: {error}') from error


def _rotation(value, name):
    result = _array(value, (3, 3), name)
    if (np.linalg.norm(result.T @ result-np.eye(3), ord=np.inf) > 1e-12
            or abs(np.linalg.det(result)-1.) > 1e-12):
        raise MarineHydroCoordinatesError(f'{name} must be proper SO(3); no orthonormalization')
    return tuple(tuple(row) for row in result)


def _readonly(value):
    try:
        data = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise MarineHydroCoordinatesError(f'invalid coordinate/pullback output: {error}') from error
    if not np.isfinite(data).all():
        raise MarineHydroCoordinatesError('nonfinite coordinate/pullback output')
    # Immutable backing bytes, not a writable array with a reversible flag.
    return np.frombuffer(data.tobytes(), dtype=float).reshape(data.shape)


def _items(value, name):
    try:
        return tuple(value)
    except (TypeError, ValueError) as error:
        raise MarineHydroCoordinatesError(f'{name} must be a sequence') from error


def _force_size(value):
    try:
        raw = np.asarray(value)
        if raw.ndim != 1 or not raw.size or raw.size % 6:
            raise ValueError('nonempty complete six-DOF force vector required')
        return raw.size
    except (TypeError, ValueError) as error:
        raise MarineHydroCoordinatesError(f'invalid pullback force: {error}') from error


def _principal_log(R):
    cosine = float((np.trace(R)-1.)/2.)
    antisymmetric = .5*np.array((R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]))
    sine = float(np.linalg.norm(antisymmetric))
    angle = math.atan2(sine, cosine)
    if angle >= math.pi-LOG_BRANCH_MARGIN:
        raise MarineHydroCoordinatesError('equilibrium rotation log reaches its principal seam; no unwrapping')
    if angle < 1e-8:
        return antisymmetric
    if angle <= math.pi/2:
        return (angle/sine)*antisymmetric
    # Diagonal/symmetric-part axis extraction is stable near the refused seam.
    squares = np.maximum((np.diag(R)-cosine)/(1.-cosine), 0.)
    k = int(np.argmax(squares))
    axis = np.zeros(3)
    axis[k] = math.copysign(math.sqrt(squares[k]), antisymmetric[k])
    for i in range(3):
        if i != k:
            axis[i] = (R[i, k]+R[k, i])/(2.*(1.-cosine)*axis[k])
    return angle*axis/np.linalg.norm(axis)


def _inverse_derivative(L, phi, direction):
    return -L @ left_jacobian_rate(phi, direction) @ L


def _inverse_second(L, phi, a, b):
    Ja, Jb = left_jacobian_rate(phi, a), left_jacobian_rate(phi, b)
    return (L @ Ja @ L @ Jb @ L + L @ Jb @ L @ Ja @ L
            - L @ left_jacobian_second_rate(phi, a, b) @ L)


@dataclass(frozen=True)
class HydroCoordinateBlock:
    """Immutable per-body output. Derivative axes contain only rotations.

    jacobian_derivatives[j] = dD/dtheta_j;
    jacobian_second_derivatives[i,j] = d2D/dtheta_i/dtheta_j.
    Translational derivatives are zero. Rates are equilibrium components.
    """
    body_id: str
    chart_acceleration: np.ndarray
    coordinates: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    world_rate_map: np.ndarray
    jacobian: np.ndarray
    jacobian_derivatives: np.ndarray
    jacobian_second_derivatives: np.ndarray
    geometric_bias: np.ndarray
    bias_velocity_derivative: np.ndarray
    bias_rotation_derivative: np.ndarray

    def __post_init__(self):
        shapes = {'chart_acceleration': (6,), 'coordinates': (6,), 'velocity': (6,),
            'acceleration': (6,), 'world_rate_map': (6, 6), 'jacobian': (6, 6),
            'jacobian_derivatives': (3, 6, 6),
            'jacobian_second_derivatives': (3, 3, 6, 6), 'geometric_bias': (6,),
            'bias_velocity_derivative': (6, 6), 'bias_rotation_derivative': (6, 3)}
        if not isinstance(self.body_id, str) or not self.body_id.strip() or '::' in self.body_id:
            raise MarineHydroCoordinatesError('invalid coordinate block body identity')
        for field in self.__dataclass_fields__:
            if field != 'body_id':
                object.__setattr__(self, field, _readonly(_array(getattr(self, field), shapes[field], field)))


@dataclass(frozen=True)
class EquilibriumHydroFrame:
    """Explicit immutable geometry; axes columns are in WORLD coordinates.

    reference_point_body is a material vector from the mechanical origin,
    not a constant WORLD offset. Changing equilibrium invalidates source data;
    a frame alone is not source-admission/provenance evidence.
    """
    body_id: str
    equilibrium_position: tuple
    equilibrium_rotation: tuple
    coefficient_axes: tuple
    reference_point_body: tuple

    def __post_init__(self):
        if not isinstance(self.body_id, str) or not self.body_id.strip() or '::' in self.body_id:
            raise MarineHydroCoordinatesError('body_id must be a nonempty stable identity without ::')
        for field in ('equilibrium_position', 'reference_point_body'):
            object.__setattr__(self, field, tuple(_array(getattr(self, field), (3,), field)))
        for field in ('equilibrium_rotation', 'coefficient_axes'):
            object.__setattr__(self, field, _rotation(getattr(self, field), field))

    def evaluate(self, state):
        if not isinstance(state, RigidBodyState):
            raise MarineHydroCoordinatesError('state must be a RigidBodyState')
        E = np.asarray(self.coefficient_axes)
        R0 = np.asarray(self.equilibrium_rotation)
        rho = np.asarray(self.reference_point_body)
        R = state.rotation
        x = R @ rho
        phi = _principal_log(E.T @ R @ R0.T @ E)
        h = np.r_[E.T @ (np.asarray(state.position)+x-np.asarray(self.equilibrium_position)-R0@rho), phi]
        L = np.linalg.solve(left_jacobian(phi), np.eye(3))
        theta = np.asarray(state.rotation_vector)
        J = left_jacobian(theta)
        B = state.rate_map
        H = np.zeros((6, 6))
        H[:3, :3] = E.T
        H[:3, 3:] = -E.T @ skew(x)
        H[3:, 3:] = L @ E.T
        D = H @ B
        dH, dB, dD = [], [], []
        basis = np.eye(3)
        for j in range(3):
            dBj = np.zeros((6, 6))
            dBj[3:, 3:] = left_jacobian_rate(theta, basis[j])
            dHj = np.zeros((6, 6))
            dHj[:3, 3:] = -E.T @ skew(np.cross(J[:, j], x))
            dHj[3:, 3:] = _inverse_derivative(L, phi, D[3:, 3+j]) @ E.T
            dH.append(dHj)
            dB.append(dBj)
            dD.append(dHj @ B + H @ dBj)
        dD = np.asarray(dD)
        d2D = np.empty((3, 3, 6, 6))
        for i in range(3):
            for j in range(3):
                dxij = (np.cross(dB[j][3:, 3+i], x)
                        + np.cross(J[:, i], np.cross(J[:, j], x)))
                d2H = np.zeros((6, 6))
                d2H[:3, 3:] = -E.T @ skew(dxij)
                d2H[3:, 3:] = (
                    _inverse_second(L, phi, D[3:, 3+i], D[3:, 3+j])
                    + _inverse_derivative(L, phi, dD[j][3:, 3+i])) @ E.T
                d2B = np.zeros((6, 6))
                d2B[3:, 3:] = left_jacobian_second_rate(theta, basis[i], basis[j])
                d2D[i, j] = d2H @ B + dH[i] @ dB[j] + dH[j] @ dB[i] + H @ d2B
        v, a = np.asarray(state.velocity), np.asarray(state.acceleration)
        Ddot = np.einsum('j,jab->ab', v[3:], dD)
        bias = Ddot @ v
        bias_v = Ddot.copy()
        for j in range(3):
            bias_v[:, 3+j] += dD[j] @ v
        bias_q = np.column_stack([np.einsum('k,kab,b->a', v[3:], d2D[j], v)
                                  for j in range(3)])
        return HydroCoordinateBlock(self.body_id, a, h, D@v, D@a+bias, H, D,
                                    dD, d2D, bias, bias_v, bias_q)


@dataclass(frozen=True)
class ForcePullback:
    external_force: np.ndarray
    stiffness: np.ndarray
    world_wrench: np.ndarray

    def __post_init__(self):
        n = _force_size(self.external_force)
        for field in self.__dataclass_fields__:
            shape = (n, n) if field == 'stiffness' else (n,)
            object.__setattr__(self, field, _readonly(_array(getattr(self, field), shape, field)))


@dataclass(frozen=True)
class InertiaPullback:
    """Positive INTERNAL inertia and total partial derivatives K/C/Ma."""
    internal_force: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    mass: np.ndarray
    geometric_force: np.ndarray

    def __post_init__(self):
        n = _force_size(self.internal_force)
        for field in self.__dataclass_fields__:
            shape = (n,) if field in ('internal_force', 'geometric_force') else (n, n)
            object.__setattr__(self, field, _readonly(_array(getattr(self, field), shape, field)))


@dataclass(frozen=True)
class CoupledHydroEvaluation:
    """Full ordered six-DOF blocks. No dense N-cubed derivative allocation."""
    blocks: tuple

    def __post_init__(self):
        blocks = _items(self.blocks, 'coordinate blocks')
        if not blocks or any(not isinstance(block, HydroCoordinateBlock) for block in blocks):
            raise MarineHydroCoordinatesError('nonempty coordinate blocks required')
        if len({block.body_id for block in blocks}) != len(blocks):
            raise MarineHydroCoordinatesError('duplicate body identity')
        object.__setattr__(self, 'blocks', blocks)

    @property
    def body_ids(self):
        return tuple(block.body_id for block in self.blocks)

    @property
    def dof_keys(self):
        return tuple(f'{body_id}::{dof}' for body_id in self.body_ids for dof in DOF_NAMES)

    def _vector(self, field):
        return np.concatenate([getattr(block, field) for block in self.blocks])

    def _diagonal(self, field):
        result = np.zeros((6*len(self.blocks), 6*len(self.blocks)))
        for i, block in enumerate(self.blocks):
            result[6*i:6*i+6, 6*i:6*i+6] = getattr(block, field)
        return result

    @property
    def coordinates(self):
        return _readonly(self._vector('coordinates'))

    @property
    def velocity(self):
        return _readonly(self._vector('velocity'))

    @property
    def acceleration(self):
        return _readonly(self._vector('acceleration'))

    @property
    def jacobian(self):
        return _readonly(self._diagonal('jacobian'))

    def pullback_force(self, source_force, source_stiffness):
        """Fh and Ch=-dFh/dh in the h-gradient basis, including baseline.

        Physical-wrench source semantics must first be converted explicitly.
        This algebra is not permission to reinterpret hydrostatic source data.
        """
        n = 6*len(self.blocks)
        force = _array(source_force, (n,), 'source force')
        C = _array(source_stiffness, (n, n), 'source stiffness')
        D = self._diagonal('jacobian')
        K = D.T @ C @ D
        for i, block in enumerate(self.blocks):
            base = 6*i
            for j in range(3):
                K[base:base+6, base+3+j] -= block.jacobian_derivatives[j].T @ force[base:base+6]
        return ForcePullback(D.T@force, K, self._diagonal('world_rate_map').T@force)

    def pullback_added_inertia(self, source_matrix):
        """Constant Ah in h coordinates; no symmetry or source-validity claim.

        Returns Qint=D.T Ah hddot, Ma=D.T Ah D and exact total K/C derivatives.
        External inertia is -Qint; the geometric force is included exactly once.
        Full cross-body matrices are preserved; no mirroring/regularization.
        """
        n = 6*len(self.blocks)
        A = _array(source_matrix, (n, n), 'source inertia')
        D = self._diagonal('jacobian')
        force_h = A @ self.acceleration
        K = np.zeros((n, n))
        # Keep original rates: subtracting two large terms could lose qddot.
        chart_acceleration = self._vector('chart_acceleration')
        for i, block in enumerate(self.blocks):
            base = 6*i
            rows = slice(base, base+6)
            for j in range(3):
                derivative = np.zeros(n)
                derivative[rows] = (block.jacobian_derivatives[j] @ chart_acceleration[rows]
                                    + block.bias_rotation_derivative[:, j])
                K[:, base+3+j] = D.T @ A @ derivative
                K[rows, base+3+j] += block.jacobian_derivatives[j].T @ force_h[rows]
        return InertiaPullback(D.T@force_h, K,
            D.T@A@self._diagonal('bias_velocity_derivative'), D.T@A@D,
            D.T@A@self._vector('geometric_bias'))


def evaluate_coupled(frames, states):
    """Evaluate exactly the explicitly ordered immutable frames and states."""
    frames = _items(frames, 'equilibrium frames')
    if not frames or any(not isinstance(frame, EquilibriumHydroFrame) for frame in frames):
        raise MarineHydroCoordinatesError('nonempty explicit equilibrium frames required')
    ids = tuple(frame.body_id for frame in frames)
    if len(set(ids)) != len(ids) or not isinstance(states, Mapping) or set(states) != set(ids):
        raise MarineHydroCoordinatesError('states must match unique frame body identities exactly')
    return CoupledHydroEvaluation(tuple(frame.evaluate(states[frame.body_id]) for frame in frames))
