"""Reference-only continuum work bridge for the distinct GE distributing pose.

No mechanics owner, public curved admission or external score is supplied.
Native imports/evaluation occur only in the explicitly called local oracle.
"""
from dataclasses import dataclass, field, InitVar
import math

import numpy as np


class BridgeDomainError(ValueError):
    pass


_FACTORY_KEY = object()
_REPRESENTATION_ROUNDOFF = 256*np.finfo(float).eps


def _owned(value):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise BridgeDomainError("finite representable bridge result required")
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _array(value, shape):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise BridgeDomainError("complete real local bridge array required")
    made = np.asarray(raw, dtype=float)
    if not np.isfinite(made).all():
        raise BridgeDomainError("finite local bridge values required")
    return _owned(made)


def _skew(vector):
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


@dataclass(frozen=True)
class ReferenceBridge:
    width_m: float
    height_m: float
    thickness_m: float
    nx: int
    ny: int
    frame: np.ndarray
    origin: np.ndarray
    local_nodes: np.ndarray
    world_nodes: np.ndarray
    face_samples: np.ndarray
    area_mean: np.ndarray
    local_gradients: np.ndarray
    nodal_pose_map: np.ndarray
    nodal_sampler_blocks: np.ndarray
    pose_map: np.ndarray
    _factory_token: InitVar[object] = None
    _factory_seal: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _factory_token):
        if _factory_token is not _FACTORY_KEY:
            raise BridgeDomainError("use rectangular_bridge; direct bridge construction is unsupported")
        for name in ("frame", "origin", "local_nodes", "world_nodes", "face_samples",
                     "area_mean", "local_gradients", "nodal_pose_map",
                     "nodal_sampler_blocks", "pose_map"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "_factory_seal", _FACTORY_KEY)

    def pose(self, face_translations):
        u = _array(face_translations, self.face_samples.shape)
        return _owned(self.pose_map @ u.ravel())

    def nodal_increments(self, face_translations):
        u = _array(face_translations, self.face_samples.shape)
        n = len(self.world_nodes)
        return _owned([block @ np.r_[u[i], u[n+i]]
                       for i, block in enumerate(self.nodal_sampler_blocks)])

    def forces(self, world_wrench):
        wrench = _array(world_wrench, (6,))
        return _owned((self.pose_map.T @ wrench).reshape(self.face_samples.shape))

    def resultant(self, face_forces, *, about):
        force = _array(face_forces, self.face_samples.shape)
        origin = _array(about, (3,))
        return _owned(np.r_[force.sum(axis=0), np.cross(self.face_samples-origin, force).sum(axis=0)])

    def constraint(self, offset_world):
        r = _array(offset_world, (3,))
        e = np.eye(6)
        e[:3, 3:] = -_skew(r)
        return _owned(np.column_stack((-e @ self.pose_map, np.eye(6))))

    def connection_actions(self, multiplier_world, *, offset_world):
        lam = _array(multiplier_world, (6,))
        c = self.constraint(offset_world)
        return (_owned((c[:, :-6].T @ lam).reshape(self.face_samples.shape)),
                _owned(c[:, -6:].T @ lam))


def rectangular_bridge(width_m, height_m, thickness_m, *, frame, origin, nx=2, ny=1):
    """Uniform rectangular Q4 integration and measured parallel face-pair samples.

    Face ordering is all lower samples followed by all upper samples. Each pair
    is separated by the declared thickness along the independent reference director.
    """
    for value in (width_m, height_m, thickness_m):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise BridgeDomainError("positive finite rectangle lengths and thickness required")
    if any(type(value) is not int or not 1 <= value <= 8 for value in (nx, ny)):
        raise BridgeDomainError("bounded positive integer local rectangle subdivisions required")
    q, center = _array(frame, (3, 3)), _array(origin, (3,))
    if (np.max(abs(q.T @ q-np.eye(3))) > _REPRESENTATION_ROUNDOFF
            or abs(np.linalg.det(q)-1.) > _REPRESENTATION_ROUNDOFF):
        raise BridgeDomainError("arbitrary proper orthonormal reference frame required")
    local, weights, gradients = [], [], []
    for j in range(ny+1):
        for i in range(nx+1):
            wx = (.5 if i in (0, nx) else 1.)/nx
            wy = (.5 if j in (0, ny) else 1.)/ny
            local.append((-width_m/2+width_m*i/nx, -height_m/2+height_m*j/ny, 0.))
            weights.append(wx*wy)
            gradients.append(((-1. if i == 0 else 1. if i == nx else 0.)*wy/width_m,
                              (-1. if j == 0 else 1. if j == ny else 0.)*wx/height_m))
    local = np.asarray(local)
    world = local @ q.T+center
    normal, a, b = q[:, 2], q[:, 0], q[:, 1]
    samples = np.vstack((world-thickness_m/2*normal, world+thickness_m/2*normal))
    if not np.isfinite(samples).all() or len({tuple(p) for p in samples}) != len(samples):
        raise BridgeDomainError("distinct representable physical face-pair geometry required")
    n = len(local)
    # This is a binary64 representation bound, not a physical comparison tolerance.
    # Do not divide by nominal h when the measured lever arm materially differs.
    measured_directors = (samples[n:]-samples[:n])/thickness_m
    recovered_local = (world-center) @ q
    # Each material direction uses its own length: a long edge must not mask
    # a distorted narrow edge whose yaw derivative is divided by that edge.
    # Thickness resolves the normal component of the material face-pair trace.
    scales = np.array((width_m, height_m, thickness_m))
    midpoint_error_local = (samples[:n]/2+samples[n:]/2-world) @ q
    if (np.max(abs(measured_directors-normal)) > _REPRESENTATION_ROUNDOFF
            or np.max(abs(recovered_local-local)/scales) > _REPRESENTATION_ROUNDOFF
            or np.max(abs(midpoint_error_local)/scales) > _REPRESENTATION_ROUNDOFF):
        raise BridgeDomainError("ill-resolved rectangle or measured face separation; rebase local geometry")
    nodal, pose, blocks = np.zeros((6, 6*n)), np.zeros((6, 6*n)), []
    tilt = np.eye(3)-np.outer(normal, normal)
    spin = _skew(normal)/thickness_m
    for i, (weight, (gx, gy)) in enumerate(zip(weights, gradients)):
        block = np.zeros((6, 6))
        block[:3, :3] = block[:3, 3:] = .5*np.eye(3)
        block[3:, :3], block[3:, 3:] = -spin, spin
        blocks.append(block)
        nodal[:3, 6*i:6*i+3] = weight*np.eye(3)
        nodal[3:, 6*i:6*i+3] = .5*np.outer(normal, gx*b-gy*a)
        nodal[3:, 6*i+3:6*i+6] = weight*tilt
        pose[:, 3*i:3*i+3] = nodal[:, 6*i:6*i+6] @ block[:, :3]
        pose[:, 3*(n+i):3*(n+i)+3] = nodal[:, 6*i:6*i+6] @ block[:, 3:]
    if not all(np.isfinite(value).all() for value in (nodal, pose, blocks)):
        raise BridgeDomainError("representable reference derivative and transpose maps required")
    return ReferenceBridge(float(width_m), float(height_m), float(thickness_m), nx, ny,
        q, center, _owned(local), _owned(world), _owned(samples), _owned(weights),
        _owned(np.asarray(gradients).T), _owned(nodal), _owned(blocks), _owned(pose),
        _factory_token=_FACTORY_KEY)


def _native_reference(bridge, offset_world):
    """Existing flat-XY local prototype, evaluated at an actual common rigid pose."""
    from anyfesolver import _stiffened_shell_distributing_footprint as reference
    from anyfesolver._stiffened_shell_distributing_pose import FiniteDistributingPose

    if (type(bridge) is not ReferenceBridge
            or getattr(bridge, "_factory_seal", None) is not _FACTORY_KEY
            or bridge.nx*bridge.ny < 2):
        raise BridgeDomainError("existing native prototype requires two hosts and six nodes")
    offset = _array(offset_world, (3,))
    n, nx, ny = len(bridge.local_nodes), bridge.nx, bridge.ny
    nodes = tuple(reference.FlatQ4Node(i+1, tuple(p), 6*i) for i, p in enumerate(bridge.local_nodes))
    hosts = tuple(reference.CartesianQ4Host(j*nx+i+1,
        (j*(nx+1)+i+1, j*(nx+1)+i+2, (j+1)*(nx+1)+i+2, (j+1)*(nx+1)+i+1))
        for j in range(ny) for i in range(nx))
    beam = reference.PhysicalBeamNodePort(1001, tuple(bridge.frame.T @ offset), 6*n, reference.IDENTITY_AXES)
    definition = reference.DistributingFootprintDefinition("e03-local-work-bridge",
        tuple(h.element_id for h in hosts), 1001,
        (-bridge.width_m/2, bridge.width_m/2, -bridge.height_m/2, bridge.height_m/2))
    compiled = reference.compile_footprint(definition, nodes, hosts, beam)
    return FiniteDistributingPose(compiled), offset


def native_local_evaluation(bridge, offset_world, *, face_delta=None, beam_delta=None):
    """Deferred local oracle only; calls require later separately authorized checks."""
    from anyfesolver._ge_beam3_mixed_ad import rotation_exponential

    pose, offset = _native_reference(bridge, offset_world)
    face = np.zeros_like(bridge.face_samples) if face_delta is None else _array(face_delta, bridge.face_samples.shape)
    beam = np.zeros(6) if beam_delta is None else _array(beam_delta, (6,))
    increments = bridge.nodal_increments(face)
    frames = np.array([rotation_exponential(row[3:]) @ bridge.frame for row in increments])
    return pose.evaluate(np.asarray(bridge.world_nodes+increments[:, :3], dtype=float), frames,
        np.asarray(bridge.origin+offset+beam[:3], dtype=float),
        np.asarray(rotation_exponential(beam[3:]) @ bridge.frame, dtype=float), np.zeros(6))


def native_sample_derivative(bridge, offset_world):
    evaluation = native_local_evaluation(bridge, offset_world)
    derivative = evaluation.constraint_jacobian
    n = len(bridge.world_nodes)
    sampled = np.zeros((6, 6*n+6))
    for i, block in enumerate(bridge.nodal_sampler_blocks):
        sampled[:, 3*i:3*i+3] = derivative[:, 6*i:6*i+6] @ block[:, :3]
        sampled[:, 3*(n+i):3*(n+i)+3] = derivative[:, 6*i:6*i+6] @ block[:, 3:]
    sampled[:, -6:] = derivative[:, -6:]
    row_frame = np.zeros((6, 6))
    row_frame[:3, :3] = row_frame[3:, 3:] = evaluation.shell_frame.T
    return dict(native_rows=_owned(sampled), world_to_native_rows=_owned(row_frame),
                evaluated_shell_frame=_owned(evaluation.shell_frame),
                constraints=_owned(evaluation.constraints), production_qualified=False,
                public_curved_admission=False)
