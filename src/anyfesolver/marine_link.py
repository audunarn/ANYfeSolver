"""Development finite-geometry six-DOF mechanical links (marine_link_v1).

Implements the adopted W4 link law: a conservative finite-attachment
spring/damper between two rigid bodies with body-fixed attachment frames,
a declared local rest translation (attachment-A axes) and rest relative
rotation, full local symmetric PSD 6x6 constitutive matrices and a local
preload covector. SI mixed translation/rotation units. No BEM, hydro,
contact, line, Morison or structural admission is implied by this module.
"""
from dataclasses import dataclass
import math
import numpy as np
from anyfesolver.marine_rigid import (MarineRigidInputError, RigidBodyState,
                                    left_jacobian, left_jacobian_rate, skew)

API_VERSION = "marine_link_v1"
PSD_RELATIVE_TOLERANCE = 1e-12
LOG_BRANCH_MARGIN = 1e-6
_I3 = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


class MarineLinkError(ValueError):
    """Malformed link inputs or refused branch/constitution; no fallback."""


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
        raise MarineLinkError(f"{name}: {error}") from error


def _vector(value, count, name):
    return tuple(_array(value, (count,), name))


def _rotation(value, name):
    R = _array(value, (3, 3), name)
    if (np.linalg.norm(R.T @ R - np.eye(3), ord=np.inf) > 1e-12
            or abs(np.linalg.det(R) - 1.) > 1e-12):
        raise MarineLinkError(f"{name} must be a proper SO(3) matrix; no orthonormalization")
    return tuple(tuple(row) for row in R)


def _psd_matrix(value, name, tolerance):
    K = _array(value, (6, 6), name)
    scale = float(np.linalg.norm(K, ord=2))
    if np.linalg.norm(K - K.T, ord=np.inf) > tolerance*max(scale, 1.):
        raise MarineLinkError(f"{name} must be symmetric within its declared tolerance")
    eig = np.linalg.eigvalsh(.5*(K + K.T))
    if eig[0] < -tolerance*max(abs(eig[-1]), 1.):
        raise MarineLinkError(f"{name} must be symmetric PSD within its declared tolerance")
    return tuple(tuple(row) for row in K)


def _so3_log(R):
    """Principal matrix logarithm with typed refusal near the pi seam.

    The margin keeps every admitted rotation vector strictly inside the
    marine_rigid principal chart guard; no silent unwrapping occurs.
    """
    tr = (float(R[0, 0]) + float(R[1, 1]) + float(R[2, 2]) - 1.)/2.
    angle = math.acos(min(1., max(-1., tr)))
    if angle >= math.pi - LOG_BRANCH_MARGIN:
        raise MarineLinkError(
            "relative rotation reaches the principal branch seam; supply new "
            "rest frames or refuse this configuration; never silently unwrap")
    if angle < 1e-8:
        return np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])*.5
    if angle <= math.pi/2:
        return (angle/(2.*math.sin(angle))
                * np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]))
    # Robust large-angle branch: axis squares from the diagonal, axis sign
    # from the skew part (2 sin(angle) n, positive throughout the admitted
    # branch); stable up to the refused pi seam.
    cos_angle = tr
    diag = np.array([R[0, 0], R[1, 1], R[2, 2]])
    axis2 = np.maximum((diag - cos_angle)/(1. - cos_angle), 0.)
    k = int(np.argmax(axis2))
    axis = np.zeros(3)
    axis[k] = math.sqrt(axis2[k])
    skew_k = R[(k+2) % 3, (k+1) % 3] - R[(k+1) % 3, (k+2) % 3]
    if skew_k < 0.:
        axis[k] = -axis[k]
    for i in range(3):
        if i != k:
            axis[i] = (R[i, k] + R[k, i])/(2.*(1.-cos_angle)*axis[k])
    norm = float(np.linalg.norm(axis))
    if not (norm > 0. and math.isfinite(norm)):
        raise MarineLinkError("relative rotation log is degenerate; refusing")
    return angle*axis/norm


@dataclass(frozen=True)
class LinkEvaluation:
    """Fresh output arrays for one link trial; mutating them changes nothing."""
    link_id: str
    body_a: str
    body_b: str
    strain: np.ndarray
    strain_rate: np.ndarray
    local_force: np.ndarray
    potential_energy: float
    damping_power: float
    jacobian: np.ndarray
    chart_force_a: np.ndarray
    chart_force_b: np.ndarray
    world_wrench_a: np.ndarray
    world_wrench_b: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    mass: np.ndarray


@dataclass(frozen=True)
class MechanicalLink6D:
    """Conservative finite-geometry six-DOF link between two rigid bodies.

    Attachment points rho_a/rho_b are body-fixed vectors from each body's
    reference origin; axes_a/axes_b are body-fixed SO(3) attachment frames.
    The declared rest translation d0 is expressed in attachment-A axes and
    the declared rest relative rotation D0 = A0.T @ B0; both are derived
    only through the explicitly named caller-selected reference-state
    factory from_reference_states. stiffness and damping are full local
    symmetric PSD 6x6 matrices in SI mixed translation/rotation units,
    retained exactly as declared. preload p0 is a local constitutive
    covector; positive axial preload pulls the endpoints together. The
    relative rotation uses the principal branch and refuses near pi; this
    is not an unbounded torsion law. Same-body links are refused: the pair
    assembly is two-body.
    """

    link_id: str
    body_a: str
    body_b: str
    rho_a: tuple
    rho_b: tuple
    axes_a: tuple
    axes_b: tuple
    rest_translation: tuple
    rest_rotation: tuple
    stiffness: tuple
    damping: tuple
    preload: tuple = (0.,)*6
    psd_tolerance: float = PSD_RELATIVE_TOLERANCE

    def __post_init__(self):
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise MarineLinkError("link_id must be a nonempty string")
        for name in ("body_a", "body_b"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise MarineLinkError(f"{name} must be a nonempty body id string")
        if self.body_a == self.body_b:
            raise MarineLinkError("a mechanical link connects two distinct bodies")
        tol = self.psd_tolerance
        if (isinstance(tol, bool) or not isinstance(tol, (int, float))
                or not math.isfinite(float(tol)) or float(tol) < 0.):
            raise MarineLinkError("psd_tolerance must be a finite nonnegative real")
        object.__setattr__(self, "psd_tolerance", float(tol))
        object.__setattr__(self, "rho_a", _vector(self.rho_a, 3, "rho_a"))
        object.__setattr__(self, "rho_b", _vector(self.rho_b, 3, "rho_b"))
        object.__setattr__(self, "axes_a", _rotation(self.axes_a, "axes_a"))
        object.__setattr__(self, "axes_b", _rotation(self.axes_b, "axes_b"))
        object.__setattr__(self, "rest_translation",
                           _vector(self.rest_translation, 3, "rest_translation"))
        object.__setattr__(self, "rest_rotation",
                           _rotation(self.rest_rotation, "rest_rotation"))
        object.__setattr__(self, "stiffness",
                           _psd_matrix(self.stiffness, "stiffness", self.psd_tolerance))
        object.__setattr__(self, "damping",
                           _psd_matrix(self.damping, "damping", self.psd_tolerance))
        object.__setattr__(self, "preload", _vector(self.preload, 6, "preload"))

    @classmethod
    def from_reference_states(cls, *, link_id, body_a, body_b, rho_a, rho_b,
                              axes_a, axes_b, state_a, state_b, stiffness,
                              damping, preload=(0.,)*6,
                              psd_tolerance=PSD_RELATIVE_TOLERANCE):
        """The only rest factory: derive d0 and D0 from caller-selected states.

        d0 = A0.T @ t0 and D0 = A0.T @ B0 at the supplied reference states.
        No rest state is ever guessed from a current trial.
        """
        for state, name in ((state_a, "state_a"), (state_b, "state_b")):
            if not isinstance(state, RigidBodyState):
                raise MarineLinkError(f"{name} must be a RigidBodyState")
        A0 = state_a.rotation @ np.asarray(axes_a)
        B0 = state_b.rotation @ np.asarray(axes_b)
        t0 = ((np.asarray(state_b.position) + state_b.rotation @ np.asarray(rho_b))
              - (np.asarray(state_a.position) + state_a.rotation @ np.asarray(rho_a)))
        return cls(link_id, body_a, body_b, rho_a, rho_b, axes_a, axes_b,
                   tuple(A0.T @ t0), tuple(tuple(row) for row in A0.T @ B0),
                   stiffness, damping, preload, psd_tolerance)

    def _require_states(self, state_a, state_b):
        if not isinstance(state_a, RigidBodyState) or not isinstance(state_b, RigidBodyState):
            raise MarineLinkError("link evaluation requires two RigidBodyState inputs")

    def _derivative_blocks(self, state_a, state_b, A, At, V, Jl_inv, phi,
                           La, Lb, Ba, Bb, pa, pb, t):
        """dG columns for all twelve chart directions (contract formula)."""
        theta_a = np.asarray(state_a.rotation_vector)
        theta_b = np.asarray(state_b.rotation_vector)
        pat = pa + t
        zero3 = np.zeros((3, 3))
        dG = []
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            dLa = np.block([[zero3, At @ skew(-e)], [zero3, zero3]])
            dG.append(np.hstack([dLa @ Ba, np.zeros((6, 6))]))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            wa = Ba[3:, 3:] @ e
            dAt = -At @ skew(wa)
            dpa = np.cross(wa, pa)
            dt = -dpa
            dphi = -V @ wa
            dV = -Jl_inv @ left_jacobian_rate(phi, dphi) @ V + Jl_inv @ dAt
            dLa = np.block([[-dAt, dAt @ skew(pat) + At @ skew(dpa + dt)],
                            [zero3, -dV]])
            dLb = np.block([[dAt, -dAt @ skew(pb)], [zero3, dV]])
            dBa = np.zeros((6, 6))
            dBa[3:, 3:] = left_jacobian_rate(theta_a, e)
            dG.append(np.hstack([dLa @ Ba + La @ dBa, dLb @ Bb]))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            dLa = np.block([[zero3, At @ skew(e)], [zero3, zero3]])
            dG.append(np.hstack([dLa @ Ba, np.zeros((6, 6))]))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.
            wb = Bb[3:, 3:] @ e
            dpb = np.cross(wb, pb)
            dt = dpb
            dphi = V @ wb
            dV = -Jl_inv @ left_jacobian_rate(phi, dphi) @ V
            dLa = np.block([[zero3, At @ skew(dt)], [zero3, -dV]])
            dLb = np.block([[zero3, -At @ skew(dpb)], [zero3, dV]])
            dBb = np.zeros((6, 6))
            dBb[3:, 3:] = left_jacobian_rate(theta_b, e)
            dG.append(np.hstack([dLa @ Ba, dLb @ Bb + Lb @ dBb]))
        return dG

    def evaluate(self, state_a, state_b):
        """Full trial evaluation: strain, wrenches, covectors and tangents."""
        self._require_states(state_a, state_b)
        Ra, Rb = state_a.rotation, state_b.rotation
        ra, rb = np.asarray(state_a.position), np.asarray(state_b.position)
        pa = Ra @ np.asarray(self.rho_a)
        pb = Rb @ np.asarray(self.rho_b)
        A = Ra @ np.asarray(self.axes_a)
        B = Rb @ np.asarray(self.axes_b)
        At = A.T
        t = (rb + pb) - (ra + pa)
        z_translation = At @ t - np.asarray(self.rest_translation)
        phi = _so3_log(At @ B @ np.asarray(self.rest_rotation).T)
        Jl_phi = left_jacobian(phi)
        Jl_inv = np.linalg.inv(Jl_phi)
        V = Jl_inv @ At
        z = np.r_[z_translation, phi]
        Ba, Bb = state_a.rate_map, state_b.rate_map
        pat = pa + t
        zero3 = np.zeros((3, 3))
        La = np.block([[-At, At @ skew(pat)], [zero3, -V]])
        Lb = np.block([[At, -At @ skew(pb)], [zero3, V]])
        G = np.hstack([La @ Ba, Lb @ Bb])
        qdot = np.r_[np.asarray(state_a.velocity), np.asarray(state_b.velocity)]
        zdot = G @ qdot
        K = np.asarray(self.stiffness)
        C = np.asarray(self.damping)
        p0 = np.asarray(self.preload)
        s = K @ z + p0 + C @ zdot
        potential = float(.5*z @ K @ z + p0 @ z)
        damping_power = float(zdot @ C @ zdot)
        pair_force = -G.T @ s
        # Physical world wrenches about the COMMON world origin, recovered
        # from the chart covectors by B^-T; the frame-dependent translation
        # spring torque on A is included, never omitted.
        force_a = A @ s[:3]
        xb = rb + pb
        wrench_a = np.r_[force_a, np.cross(xb, force_a) + V.T @ s[3:]]
        wrench_b = -wrench_a
        damping_link = G.T @ C @ G
        stiffness_link = np.zeros((12, 12))
        dG_columns = self._derivative_blocks(state_a, state_b, A, At, V, Jl_inv,
                                             phi, La, Lb, Ba, Bb, pa, pb, t)
        for j in range(12):
            dG = dG_columns[j]
            stiffness_link[:, j] = (dG.T @ s + G.T @ (K @ G[:, j]
                                                      + C @ (dG @ qdot)))
        return LinkEvaluation(self.link_id, self.body_a, self.body_b, z, zdot,
                              s, potential, damping_power, G,
                              pair_force[:6].copy(), pair_force[6:].copy(),
                              wrench_a, wrench_b, stiffness_link, damping_link,
                              np.zeros((12, 12)))
