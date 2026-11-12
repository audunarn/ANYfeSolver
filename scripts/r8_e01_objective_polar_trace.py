"""Finite objective recovered-polar trace PROPOSAL; no solver/public dispatch.

Local 30-coordinate derivatives use left exponential updates of physical node
directors and beam rotation. This is a distinct candidate connection law, not
the native nodal-pose law, BL5 or accepted E01 external correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import (
    Jet2, RotationDomainError, constant_matrix, dot, matmul, matvec,
    rotation_exponential, so3_exp, so3_log, sqrt, transpose,
)
from scripts.r8_e01_material_frame_work import rectangular_patch


IDENTITY = "R8_E01_OBJECTIVE_PROJECTED_POLAR_INDEPENDENT_DIRECTOR_V1_PROPOSAL"
SIZE = 30
# Prospective arithmetic-domain margin, not a mechanical acceptance tolerance.
# det(G)/trace(G)^2 bounds reciprocal spectral conditioning in 2D.
METRIC_MARGIN = 64.*np.finfo(float).eps


class TraceDomainError(ValueError):
    """The supplied local candidate state is outside its declared domain."""


def _owned(value, shape, label):
    if np.iscomplexobj(value):
        raise TraceDomainError(f"real {label} required")
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise TraceDomainError(f"finite {label} of shape {shape} required")
    a.setflags(write=False)
    return a


def _rotation(value):
    r = _owned(value, (3, 3), "physical beam rotation")
    if (np.max(np.abs(r.T@r-np.eye(3))) > 1e-12 or
            abs(float(np.linalg.det(r))-1.) > 1e-12):
        raise TraceDomainError("proper orthogonal physical beam rotation required")
    return r


@dataclass(frozen=True)
class PhysicalState:
    positions: np.ndarray
    directors: np.ndarray
    beam_position: np.ndarray
    beam_rotation: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "positions", _owned(self.positions, (4, 3), "shell positions"))
        d = _owned(self.directors, (4, 3), "physical shell directors")
        if np.max(np.abs(np.sum(d*d, axis=1)-1.)) > 1e-12:
            raise TraceDomainError("unit physical node directors required")
        object.__setattr__(self, "directors", d)
        object.__setattr__(self, "beam_position", _owned(self.beam_position, (3,), "beam position"))
        object.__setattr__(self, "beam_rotation", _rotation(self.beam_rotation))


@dataclass(frozen=True)
class PolarTrace:
    """Frozen flat-reference host, station, shell-frame offset and pose bond."""
    xy: np.ndarray
    station: np.ndarray
    reference_beam_position: np.ndarray
    reference_beam_rotation: np.ndarray
    shell_offset: np.ndarray

    def __post_init__(self):
        xy = _owned(self.xy, (4, 2), "reference rectangle")
        station = _owned(self.station, (2,), "reference station")
        try:
            rectangular_patch(xy, station)
        except ValueError as error:
            raise TraceDomainError(str(error)) from error
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "reference_beam_position", _owned(
            self.reference_beam_position, (3,), "reference beam position"))
        object.__setattr__(self, "reference_beam_rotation", _rotation(self.reference_beam_rotation))
        object.__setattr__(self, "shell_offset", _owned(self.shell_offset, (3,), "shell-frame offset"))

    def reference_state(self):
        return PhysicalState(np.column_stack((self.xy, np.zeros(4))),
                             np.tile((0., 0., 1.), (4, 1)),
                             self.reference_beam_position, self.reference_beam_rotation)

    @property
    def e0(self):
        # Reference shell frame is identity. Apply shell offset exactly once.
        return self.reference_beam_position-np.r_[self.station, 0.]-self.shell_offset

    @property
    def reference_arm(self):
        """Full initial beam-to-midsurface arm, independent of port gauge."""
        return self.reference_beam_position-np.r_[self.station, 0.]


@dataclass(frozen=True)
class TraceEvaluation:
    constraint: np.ndarray
    jacobian: np.ndarray
    constraint_hessians: np.ndarray
    potential: float
    covector: np.ndarray
    hessian: np.ndarray
    shell_frame: np.ndarray
    director: np.ndarray
    shell_port_position: np.ndarray
    metric_det_over_trace_squared: float
    coordinate_kind: str


def updated(state, increments):
    """Actual left physical update, independent of derivative assembly."""
    if type(state) is not PhysicalState:
        raise TraceDomainError("owned PhysicalState required")
    delta = _owned(increments, (SIZE,), "local increments")
    node = delta[:24].reshape(4, 6)
    directors = np.array([rotation_exponential(row[3:])@d
                          for row, d in zip(node, state.directors, strict=True)])
    return PhysicalState(state.positions+node[:, :3], directors,
                         state.beam_position+delta[24:27],
                         rotation_exponential(delta[27:])@state.beam_rotation)


def superposed(state, rotation, translation):
    """Common rigid motion of physical data; reference material labels stay fixed."""
    r = _rotation(rotation); t = _owned(translation, (3,), "rigid translation")
    return PhysicalState(state.positions@r.T+t, state.directors@r.T,
                         r@state.beam_position+t, r@state.beam_rotation)


def _values(matrix):
    return np.array([[x.value for x in row] for row in matrix])


def _jets(candidate, state, increments):
    delta = [Jet2.variable(value, i, SIZE) for i, value in enumerate(increments)]
    x = [[Jet2.constant(state.positions[i, j], SIZE)+delta[6*i+j]
          for j in range(3)] for i in range(4)]
    directors = [matvec(so3_exp(delta[6*i+3:6*i+6]),
                       [Jet2.constant(v, SIZE) for v in state.directors[i]]) for i in range(4)]
    rb = matmul(so3_exp(delta[27:]), constant_matrix(state.beam_rotation, SIZE))
    beam_x = [Jet2.constant(state.beam_position[j], SIZE)+delta[24+j] for j in range(3)]
    return _trace_from_jets(candidate, x, directors, beam_x, rb)


def _trace_from_jets(candidate, x, directors, beam_x, rb, *,
                     beam_origin_relative=None, reference_arm=None):
    """Bounded local AD seam for a supplied authoritative physical beam pose.

    A compensated adapter may supply beam_origin_relative = rb_position-x[0]
    directly, before rounding a high/low position to one component.
    """
    size = x[0][0].gradient.size
    n, dx, dy = rectangular_patch(candidate.xy, candidate.station)
    raw_d = [sum((n[i]*directors[i][j] for i in range(4)), Jet2.constant(0., size))
             for j in range(3)]
    norm2 = dot(raw_d, raw_d)
    if norm2.value <= METRIC_MARGIN:
        raise TraceDomainError("interpolated physical director cancels or is unresolved")
    d = [v/sqrt(norm2) for v in raw_d]
    # Remove a common origin before forming gradients. Material derivatives are
    # frozen in reference coordinates; no derivative of a geometry-based host.
    f = [[sum(((x[i][j]-x[0][j])*grad[i] for i in range(4)), Jet2.constant(0., size))
          for grad in (dx, dy)] for j in range(3)]
    e = [[f[j][a]-d[j]*sum((d[k]*f[k][a] for k in range(3)), Jet2.constant(0., size))
          for a in range(2)] for j in range(3)]
    g = matmul(transpose(e), e)
    determinant = g[0][0]*g[1][1]-g[0][1]*g[1][0]
    trace = g[0][0]+g[1][1]
    if (not np.isfinite((determinant.value, trace.value)).all() or trace.value <= 0. or
            determinant.value <= METRIC_MARGIN*trace.value**2):
        raise TraceDomainError("projected material metric is rank-collapsed or unresolved")
    # Cayley-Hamilton proves sqrt(G)=(G+s I)/sqrt(trace(G)+2s), s=sqrt(det G).
    s = sqrt(determinant); k = sqrt(trace+2.*s)
    a, b, c, h = g[0][0]+s, g[0][1], g[1][0], g[1][1]+s
    denominator = a*h-b*c
    inverse_root = [[k*h/denominator, -k*b/denominator],
                    [-k*c/denominator, k*a/denominator]]
    q = matmul(e, inverse_root)
    rs = [q[j]+[d[j]] for j in range(3)]
    frame = _values(rs)
    if not np.isfinite(frame).all() or np.linalg.det(frame) <= 0.:
        raise TraceDomainError("projected material orientation flipped")
    if np.max(np.abs(frame.T@frame-np.eye(3))) > 1e-9:
        raise TraceDomainError("projected polar frame is numerically unresolved")
    arm = matvec(rs, [Jet2.constant(v, size) for v in candidate.shell_offset])
    center_relative = [sum((n[i]*(x[i][j]-x[0][j]) for i in range(4)), Jet2.constant(0., size))
                       for j in range(3)]
    # Difference first: an arbitrary common origin is not part of the law.
    # Rs.T (rb-center-Rs*a) - (rb0-center0-a) simplifies exactly to
    # Rs.T (rb-center) - (rb0-center0). Do not form the two cancelling a terms:
    # a large frame-offset gauge can otherwise erase a real displacement while
    # the AD gradient still reports its nonzero derivative.
    base_relative = ([beam_x[j]-x[0][j] for j in range(3)] if beam_origin_relative is None
                     else beam_origin_relative)
    gap = [base_relative[j]-center_relative[j] for j in range(3)]
    arm0 = candidate.reference_arm if reference_arm is None else reference_arm
    ct = [v-v0 for v, v0 in zip(matvec(transpose(rs), gap), arm0, strict=True)]
    relative = matmul(constant_matrix(candidate.reference_beam_rotation.T, size),
                      matmul(transpose(rs), rb))
    try:
        cr = so3_log(relative)
    except RotationDomainError as error:
        raise TraceDomainError("relative pose outside existing 0.9*pi logarithm chart") from error
    position = [x[0][j]+center_relative[j]+arm[j] for j in range(3)]
    return ct+cr, rs, d, position, determinant.value/trace.value**2


def evaluate(candidate, state, multipliers, *, increments=None):
    """Differentiate lambda.c in one fixed base-state exponential chart.

    At zero increments, the covector acts on spatial translations and LEFT
    physical rotation increments. At nonzero increments it acts on parameters
    of this same anchored chart, not re-centered physical rotation increments.
    The full Hessian includes multiplier-dependent trace/offset/log curvature.
    """
    if type(candidate) is not PolarTrace or type(state) is not PhysicalState:
        raise TraceDomainError("owned candidate and physical state required")
    lam = _owned(multipliers, (6,), "constraint multipliers")
    delta = np.zeros(SIZE) if increments is None else _owned(increments, (SIZE,), "local increments")
    try:
        parts = _jets(candidate, state, delta)
        return _evaluation_from_jets(parts, lam, delta)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise TraceDomainError("finite representable local derivative domain required") from error


def _evaluation_from_jets(parts, lam, increments):
    """Shared owned local derivative result; no global gather or assembly."""
    constraint, rs, d, position, metric = parts
    size = constraint[0].gradient.size
    potential = sum((v*l for v, l in zip(constraint, lam, strict=True)), Jet2.constant(0., size))
    arrays = (np.array([v.value for v in constraint]),
              np.array([v.gradient for v in constraint]),
              np.array([v.hessian for v in constraint]), potential.gradient,
              potential.hessian, _values(rs), np.array([v.value for v in d]),
              np.array([v.value for v in position]))
    if not np.isfinite(potential.value) or any(not np.isfinite(a).all() for a in arrays):
        raise TraceDomainError("nonfinite local trace derivatives")
    copied = [_owned(a, a.shape, "result") for a in arrays]
    return TraceEvaluation(*copied[:3], potential.value, *copied[3:], metric,
                           "spatial_left_physical" if not np.any(increments) else "fixed_base_exponential_parameters")


def wrench_balance(state, evaluation, origin=(0., 0., 0.)):
    """Physical force/moment balance of the zero-increment potential covector."""
    if evaluation.coordinate_kind != "spatial_left_physical":
        raise TraceDomainError("recenter before interpreting covectors as physical wrenches")
    origin = _owned(origin, (3,), "moment origin")
    node = evaluation.covector[:24].reshape(4, 6)
    beam = evaluation.covector[24:]
    force = node[:, :3].sum(axis=0)+beam[:3]
    moment = (np.cross(state.positions-origin, node[:, :3]).sum(axis=0)+
              node[:, 3:].sum(axis=0)+np.cross(state.beam_position-origin, beam[:3])+beam[3:])
    return force, moment
