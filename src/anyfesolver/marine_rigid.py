"""Development six-DOF rigid-body mechanics in SI and world axes.

Public instantaneous mechanics only: no time integrator, hydrodynamic source
admission, contact, history or rechart policy is implied. Coordinates are
q=(r,theta), R=Exp(theta) @ anchor; chart rates are not angular velocity.
The dry covector is INTERNAL; a dynamic residual subtracts it exactly once.
"""
from dataclasses import dataclass
import math
from numbers import Real
import numpy as np

API_VERSION = "marine_rigid_v1"
SO3_TOLERANCE = 1e-12
CHART_BRANCH_MARGIN = 1e-8  # numerical log-seam guard, not a physical envelope
SERIES_THRESHOLD = .02
_I3 = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
_ZERO6 = (0.,) * 6


class MarineRigidInputError(ValueError):
    """Malformed/unsupported mechanics inputs; never a qualified fallback."""


def _array(value, shape, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or any(
            isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat
        ):
            raise ValueError("real numeric values required")
        out = np.array(raw, dtype=float, copy=True)
        if out.shape != shape or not np.isfinite(out).all():
            raise ValueError(f"finite shape {shape} required")
        return out
    except (ValueError, TypeError, OverflowError) as error:
        raise MarineRigidInputError(f"{name}: {error}") from error


def _vector(value, count, name):
    return tuple(_array(value, (count,), name))


def _rotation(value, name):
    R = _array(value, (3, 3), name)
    if (np.linalg.norm(R.T @ R - np.eye(3), ord=np.inf) > SO3_TOLERANCE
            or abs(np.linalg.det(R) - 1.) > SO3_TOLERANCE):
        raise MarineRigidInputError(f"{name} must be a proper SO(3) matrix; no orthonormalization")
    return tuple(tuple(row) for row in R)


def skew(vector):
    x, y, z = _vector(vector, 3, "skew vector")
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _chart(theta):
    theta = _array(theta, (3,), "rotation vector")
    if np.linalg.norm(theta) >= math.pi - CHART_BRANCH_MARGIN:
        raise MarineRigidInputError(
            "rotation vector reaches the principal chart seam; supply a new "
            "anchor with transported rates before evaluating this chart"
        )
    return theta


def _coefficients(theta):
    t2 = float(theta @ theta)
    t = math.sqrt(t2)
    if t < SERIES_THRESHOLD:
        a = 1 - t2/6 + t2*t2/120 - t2**3/5040 + t2**4/362880
        b = .5 - t2/24 + t2*t2/720 - t2**3/40320 + t2**4/3628800
        c = 1/6 - t2/120 + t2*t2/5040 - t2**3/362880 + t2**4/39916800
        db = -1/12 + t2/180 - t2*t2/6720 + t2**3/453600
        dc = -1/60 + t2/1260 - t2*t2/60480 + t2**3/4989600
    else:
        a = math.sin(t)/t
        b = (1-math.cos(t))/t2
        c = (t-math.sin(t))/(t2*t)
        db = (t*math.sin(t)-2*(1-math.cos(t)))/(t2*t2)
        dc = (t*(1-math.cos(t))-3*(t-math.sin(t)))/(t2*t2*t)
    return a, b, c, db, dc


def rotation_matrix(theta):
    """Exp(theta) for an admitted principal additive chart."""
    theta = _chart(theta)
    a, b, _, _, _ = _coefficients(theta)
    S = skew(theta)
    return np.eye(3) + a*S + b*(S@S)


def left_jacobian(theta):
    theta = _chart(theta)
    _, b, c, _, _ = _coefficients(theta)
    S = skew(theta)
    return np.eye(3) + b*S + c*(S@S)


def left_jacobian_rate(theta, theta_rate):
    """Exact directional time derivative, with stable small-angle series."""
    theta = _chart(theta)
    rate = _array(theta_rate, (3,), "chart angular rate")
    _, b, c, db, dc = _coefficients(theta)
    dot = theta @ rate
    S, Sd = skew(theta), skew(rate)
    return db*dot*S + b*Sd + dc*dot*(S@S) + c*(Sd@S + S@Sd)


@dataclass(frozen=True)
class RigidBodyState:
    """Immutable trial kinematics with explicit fixed chart anchor."""
    position: tuple
    rotation_vector: tuple = (0., 0., 0.)
    velocity: tuple = _ZERO6
    acceleration: tuple = _ZERO6
    anchor: tuple = _I3

    def __post_init__(self):
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        object.__setattr__(self, "rotation_vector", tuple(_chart(self.rotation_vector)))
        object.__setattr__(self, "velocity", _vector(self.velocity, 6, "chart velocity"))
        object.__setattr__(self, "acceleration", _vector(self.acceleration, 6, "chart acceleration"))
        object.__setattr__(self, "anchor", _rotation(self.anchor, "chart anchor"))

    @property
    def rotation(self):
        return rotation_matrix(self.rotation_vector) @ np.asarray(self.anchor)

    @property
    def rate_map(self):
        B = np.eye(6)
        B[3:, 3:] = left_jacobian(self.rotation_vector)
        return B

    @property
    def rate_map_derivative(self):
        Bd = np.zeros((6, 6))
        Bd[3:, 3:] = left_jacobian_rate(self.rotation_vector, self.velocity[3:])
        return Bd

    @property
    def world_velocity(self):
        return self.rate_map @ np.asarray(self.velocity)

    @property
    def world_acceleration(self):
        return (self.rate_map @ np.asarray(self.acceleration)
                + self.rate_map_derivative @ np.asarray(self.velocity))


@dataclass(frozen=True)
class PointKinematics:
    """Fresh output arrays; changing them cannot change a body or trial state."""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jacobian: np.ndarray
    acceleration_bias: np.ndarray


@dataclass(frozen=True)
class RigidBody6D:
    """Dry mechanical authority; CG and full CG inertia are explicit inputs.

    CG and inertia are in body axes relative to the reference origin. Inertia
    uses tensor off-diagonals (not negated engineering products of inertia).
    Positive rotational inertia and principal triangle inequalities are
    required for this nondegenerate six-DOF mechanics API.
    """
    body_id: str
    mass: float
    cg_body: tuple
    inertia_cg_body: tuple
    reference_origin: tuple = (0., 0., 0.)
    reference_orientation: tuple = _I3

    def __post_init__(self):
        if not isinstance(self.body_id, str) or not self.body_id.strip():
            raise MarineRigidInputError("body_id must be a nonempty string")
        if isinstance(self.mass, (bool, np.bool_)) or not isinstance(self.mass, Real):
            raise MarineRigidInputError("dry mass must be a positive finite real number")
        try:
            mass = float(self.mass)
        except (TypeError, ValueError, OverflowError) as error:
            raise MarineRigidInputError("dry mass must be a positive finite real number") from error
        if not math.isfinite(mass) or mass <= 0:
            raise MarineRigidInputError("dry mass must be a positive finite real number")
        I = _array(self.inertia_cg_body, (3, 3), "CG inertia")
        scale = float(np.linalg.norm(I, ord=2))
        if not scale or np.linalg.norm(I-I.T, ord=np.inf) > 1e-12*scale:
            raise MarineRigidInputError("CG inertia must be symmetric and nonzero")
        eig = np.linalg.eigvalsh(I)
        if eig[0] <= 0.:
            raise MarineRigidInputError("CG inertia must be positive definite")
        if eig[2] > eig[0] + eig[1] + 1e-12*scale:
            raise MarineRigidInputError("CG inertia violates principal moment triangle inequalities")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "cg_body", _vector(self.cg_body, 3, "CG in body axes"))
        object.__setattr__(self, "inertia_cg_body", tuple(tuple(row) for row in I))
        object.__setattr__(self, "reference_origin", _vector(self.reference_origin, 3, "reference origin"))
        object.__setattr__(self, "reference_orientation", _rotation(self.reference_orientation, "reference orientation"))

    def state(self, *, position=None, rotation_vector=(0., 0., 0.),
              velocity=_ZERO6, acceleration=_ZERO6, anchor=None):
        return RigidBodyState(self.reference_origin if position is None else position,
                              rotation_vector, velocity, acceleration,
                              self.reference_orientation if anchor is None else anchor)

    def _require_state(self, state):
        if not isinstance(state, RigidBodyState):
            raise MarineRigidInputError("state must be a RigidBodyState")

    def mass_matrix(self, state):
        """World twist dry mass about the current reference origin."""
        self._require_state(state)
        R = state.rotation
        c = R @ np.asarray(self.cg_body)
        Ic = R @ np.asarray(self.inertia_cg_body) @ R.T
        Io = Ic + self.mass*((c@c)*np.eye(3) - np.outer(c, c))
        return np.block([[self.mass*np.eye(3), -self.mass*skew(c)],
                         [self.mass*skew(c), Io]])

    def gyroscopic_force(self, state):
        self._require_state(state)
        omega = state.world_velocity[3:]
        c = state.rotation @ np.asarray(self.cg_body)
        Io = self.mass_matrix(state)[3:, 3:]
        return np.r_[self.mass*np.cross(omega, np.cross(omega, c)),
                     np.cross(omega, Io@omega)]

    def inertial_force(self, state):
        """Positive internal chart covector, including acceleration/gyro bias."""
        self._require_state(state)
        return state.rate_map.T @ (self.mass_matrix(state) @ state.world_acceleration
                                  + self.gyroscopic_force(state))

    def acceleration_tangent(self, state):
        """d Q_internal / d qdd; no position/velocity tangent is implied."""
        self._require_state(state)
        B = state.rate_map
        return B.T @ self.mass_matrix(state) @ B

    def kinetic_energy(self, state):
        self._require_state(state)
        nu = state.world_velocity
        return float(.5*nu @ self.mass_matrix(state) @ nu)

    def point_kinematics(self, state, point_body):
        self._require_state(state)
        rho = _array(point_body, (3,), "point in body axes")
        p = state.rotation @ rho
        nu, nud = state.world_velocity, state.world_acceleration
        omega = nu[3:]
        J = np.column_stack((np.eye(3), -skew(p)@state.rate_map[3:, 3:]))
        bias = (np.cross(state.rate_map_derivative[3:, 3:] @ np.asarray(state.velocity[3:]), p)
                + np.cross(omega, np.cross(omega, p)))
        return PointKinematics(np.asarray(state.position)+p, nu[:3]+np.cross(omega, p),
                               nud[:3]+np.cross(nud[3:], p)+np.cross(omega, np.cross(omega, p)),
                               J, bias)

    def point_force(self, state, point_body, force_world):
        """External chart covector J.T f, work conjugate to point displacement."""
        force = _array(force_world, (3,), "world point force")
        return self.point_kinematics(state, point_body).jacobian.T @ force

    def gravity_force(self, state, gravity_world=(0., 0., -9.81)):
        """External dry gravity only, with the exact current CG lever arm."""
        return self.point_force(state, self.cg_body,
                                self.mass*_array(gravity_world, (3,), "gravity acceleration"))

    def dry_tangents(self, state):
        """Exact total chart tangents of the dry covector (fresh arrays).

        stiffness = dQdry/dq, damping = dQdry/dqdot, mass = dQdry/dqdd,
        including every transport, geometric and gyroscopic term. Chart
        translation columns of the dry stiffness are exactly zero. These
        are production analytic tangents; finite differences are only
        verification, never the assembled tangent.
        """
        self._require_state(state)
        theta = np.asarray(state.rotation_vector)
        qdot = np.asarray(state.velocity)
        qdd = np.asarray(state.acceleration)
        thd = qdot[3:]
        B = state.rate_map
        Jl = B[3:, 3:]
        M = self.mass_matrix(state)
        Io = M[3:, 3:]
        R = state.rotation
        c = R @ np.asarray(self.cg_body)
        Ic = R @ np.asarray(self.inertia_cg_body) @ R.T
        nud = state.world_acceleration
        g = self.gyroscopic_force(state)
        base = M @ nud + g
        mass_tangent = B.T @ M @ B
        damping = np.zeros((6, 6))
        omega = state.world_velocity[3:]
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            col = np.zeros(6)
            col[3:] = (left_jacobian_rate(theta, e) @ thd
                       + left_jacobian_rate(theta, thd) @ e)
            dw = Jl @ e
            dg = np.r_[self.mass*(np.cross(dw, np.cross(omega, c))
                                  + np.cross(omega, np.cross(dw, c))),
                       np.cross(dw, Io @ omega) + np.cross(omega, Io @ dw)]
            damping[:, 3+j] = B.T @ (M @ col + dg)
        stiffness = np.zeros((6, 6))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            DJl_e = left_jacobian_rate(theta, e)
            dB = np.zeros((6, 6))
            dB[3:, 3:] = DJl_e
            dBd = np.zeros((6, 6))
            dBd[3:, 3:] = left_jacobian_second_rate(theta, e, thd)
            phi = Jl @ e
            dc = np.cross(phi, c)
            Sphi = skew(phi)
            dIc = Sphi @ Ic - Ic @ Sphi
            dIo = dIc + self.mass*(2*float(c @ dc)*np.eye(3)
                                   - np.outer(dc, c) - np.outer(c, dc))
            dM = np.zeros((6, 6))
            dM[:3, 3:] = -self.mass*skew(dc)
            dM[3:, :3] = self.mass*skew(dc)
            dM[3:, 3:] = dIo
            dnud = dB @ qdd + dBd @ qdot
            dw = DJl_e @ thd
            dg = np.r_[self.mass*(np.cross(dw, np.cross(omega, c))
                                  + np.cross(omega, np.cross(dw, c))
                                  + np.cross(omega, np.cross(omega, dc))),
                       np.cross(dw, Io @ omega) + np.cross(omega, dIo @ omega)
                       + np.cross(omega, Io @ dw)]
            stiffness[:, 3+j] = dB.T @ base + B.T @ (dM @ nud + M @ dnud + dg)
        return DryTangents(stiffness, damping, mass_tangent)


@dataclass(frozen=True)
class DryTangents:
    """Fresh output arrays; exact total chart tangents of the dry covector."""

    stiffness: np.ndarray
    damping: np.ndarray
    mass: np.ndarray


SERIES_THRESHOLD_SECOND = .5


def _coefficients_second(theta):
    """Stable b''(t), c''(t) for exact Jl second directional derivatives.

    A distinct, larger series threshold is used because the second
    derivatives cancel one more power of t than the first derivatives;
    closed forms lose roughly two digits near t = 0. The series below 0.5
    radians truncate near 1e-12 absolute, inside the 1e-8 verification
    tolerance for FD checks of the assembled tangents.
    """
    t2 = float(theta @ theta)
    t = math.sqrt(t2)
    if t < SERIES_THRESHOLD_SECOND:
        bpp = -1/12 + t2/60 - t2*t2/1344 + t2**3/64800 - t2**4/5322240
        cpp = -1/60 + t2/420 - t2*t2/12096 + t2**3/712800 - t2**4/69189120
    else:
        st, ct = math.sin(t), math.cos(t)
        bpp = (t2*ct - 4*t*st + 6 - 6*ct)/(t2*t2)
        cpp = (t2*st + 6*t*ct + 6*t - 12*st)/(t2*t2*t)
    return bpp, cpp


def left_jacobian_second_rate(theta, rate_a, rate_b):
    """Exact symmetric bilinear second directional derivative of Jl.

    Returns d2 Jl(theta)[a, b]; together with left_jacobian_rate this is
    the stable analytic second derivative of the chart map. Finite
    differences remain verification only, never the production tangent.
    """
    theta = _chart(theta)
    a = _array(rate_a, (3,), "chart rate a")
    b = _array(rate_b, (3,), "chart rate b")
    _, _, cc, db, dc = _coefficients(theta)
    bpp, cpp = _coefficients_second(theta)
    t2 = float(theta @ theta)
    ddb = (bpp - db)/t2 if t2 > 0. else 1/90
    ddc = (cpp - dc)/t2 if t2 > 0. else 1/630
    S = skew(theta)
    Sa, Sb = skew(a), skew(b)
    ta, tb = float(theta @ a), float(theta @ b)
    ab = float(a @ b)
    d2b = ddb*ta*tb + db*ab
    d2c = ddc*ta*tb + dc*ab
    SS = S @ S
    return (d2b*S + db*(tb*Sa + ta*Sb) + d2c*SS
            + dc*(ta*(Sb @ S + S @ Sb) + tb*(Sa @ S + S @ Sa))
            + cc*(Sa @ Sb + Sb @ Sa))
