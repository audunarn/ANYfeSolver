"""Frictionless centreline-envelope hemisphere support against a fixed plane.

This geometry is for the explicit endpoint point-penalty profile. It is the
minimum plane height of a hemisphere directed by the *actual* B2/B3 endpoint
tangent; it carries no material surface coordinates or spin history. Whole
beam regularity, side contact, temporal bounds and penalty assembly belong to
the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, dot, sqrt
from .capped_cylinder import _sqrt_bound


class HemispherePlaneGeometryError(ValueError):
    """The exterior support map is unsupported or not representable."""


@dataclass(frozen=True)
class HemispherePlaneState:
    """Support gap and derivatives in flattened, node-major translations.

    ``normal`` is the unit plane normal. At ``parallel_rim`` the gap is
    positive and the skin minimizer is nonunique: ``skin_point`` is None and
    the supplied derivatives are those of plane height (zero pressure).
    At the sphere/rim seam the sphere Hessian is selected; the gap is C1,
    but its Hessian has distinct one-sided limits.
    """

    gap: float
    gradient: np.ndarray
    hessian: np.ndarray
    skin_point: np.ndarray | None
    normal: np.ndarray
    branch: str


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _vector(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise HemispherePlaneGeometryError(f"{label} must contain three finite values")
    return array


def _exact_dot(left, right) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def _cross(left, right):
    return [left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]]


def _replace_value(jet: Jet2, value: Fraction) -> Jet2:
    """Recover the exact binary-input base value without changing derivatives."""
    return Jet2(float(value), jet.gradient, jet.hessian)


def _stable_gap(height, normal_squared, tangent_squared, discriminant, radius):
    # All inputs here are Fractions. Positive rescaling of N and v preserves
    # this conjugate formula and avoids floating normal/tangent norm overflow.
    numerator = height * height * tangent_squared - radius * radius * discriminant
    denominator = (_sqrt_bound(normal_squared * tangent_squared)
                   * (height * _sqrt_bound(tangent_squared)
                      + radius * _sqrt_bound(discriminant)))
    if denominator <= 0:
        raise HemispherePlaneGeometryError("support gap denominator is not representable")
    gap = float(numerator / denominator)
    if not math.isfinite(gap) or (numerator != 0 and gap == 0.0):
        raise HemispherePlaneGeometryError("nonzero exact support gap is not representable")
    return gap


def hemisphere_plane_work(positions, endpoint, outward_sign, radius,
                          plane_point, plane_normal) -> HemispherePlaneState:
    """Evaluate a B2/B3 endpoint hemisphere in the positive-height domain.

    For endpoint x, outward tangent v, unit normal n, h=(x-p).n and
    beta=n.v/|v|, the sphere branch (beta <= 0) has g=h-R. The rim branch
    (beta > 0) has g=h-R*sqrt(1-beta**2). Its support point is
    x-R*(n-beta*v/|v|)/sqrt(1-beta**2). All nodal translations are seeded
    in Jet2, including the B3 tangent's midnode arm.

    Exact Fraction arithmetic selects the branch and activation sign from
    the supplied binary floats. No gap or beta tolerance changes activation.
    Nonpositive endpoint plane height and zero tangents are unsupported.
    """
    x = np.asarray(positions, dtype=float)
    if x.shape not in ((2, 3), (3, 3)) or not np.all(np.isfinite(x)):
        raise HemispherePlaneGeometryError("positions must be finite B2 (2,3) or B3 (3,3) nodes")
    if (isinstance(endpoint, (bool, np.bool_)) or not isinstance(endpoint, Integral)
            or endpoint not in (0, len(x) - 1)):
        raise HemispherePlaneGeometryError("endpoint must be zero or the last node index")
    if isinstance(outward_sign, (bool, np.bool_)) or outward_sign not in (-1, 1):
        raise HemispherePlaneGeometryError("outward_sign must be -1 or +1")
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HemispherePlaneGeometryError("radius must be finite and positive") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise HemispherePlaneGeometryError("radius must be finite and positive")
    point = _vector(plane_point, "plane_point")
    normal = _vector(plane_normal, "plane_normal")
    normal_scale = float(np.max(np.abs(normal)))
    if normal_scale == 0:
        raise HemispherePlaneGeometryError("plane_normal must be nonzero")

    coefficients = ((-1, 1) if len(x) == 2 else
                    ((-3, 4, -1) if endpoint == 0 else (1, -4, 3)))
    coefficients = tuple(int(outward_sign) * a for a in coefficients)
    xf = [[Fraction.from_float(float(a)) for a in row] for row in x]
    nf = [Fraction.from_float(float(a)) for a in normal]
    pf = [Fraction.from_float(float(a)) for a in point]
    vf = [sum((a * row[j] for a, row in zip(coefficients, xf)), Fraction(0))
          for j in range(3)]
    H = _exact_dot([a - b for a, b in zip(xf[endpoint], pf)], nf)
    q, VV, NN = _exact_dot(vf, nf), _exact_dot(vf, vf), _exact_dot(nf, nf)
    if H <= 0:
        raise HemispherePlaneGeometryError("endpoint plane height must be strictly positive")
    if VV == 0:
        raise HemispherePlaneGeometryError("actual endpoint tangent must be nonzero")

    # Work with exact positive scales before conversion to float. In
    # particular, NN is not inferred from an IEEE dot of a unit normal.
    ns = Fraction.from_float(normal_scale)
    vs = max(abs(a) for a in vf)
    nscaled, vscaled = [a / ns for a in nf], [a / vs for a in vf]
    hn, nn, vv, qs = H / ns, NN / ns**2, VV / vs**2, q / (ns * vs)
    D = nn * vv - qs * qs
    if D < 0:
        raise HemispherePlaneGeometryError("exact support discriminant is negative")
    norm_n_exact = _sqrt_bound(nn)
    norm_n = float(norm_n_exact)
    nunit = np.array([float(a) / norm_n for a in nscaled])
    try:
        height = float(hn / norm_n_exact)
    except (ArithmeticError, ValueError) as exc:
        raise HemispherePlaneGeometryError("endpoint plane height is not representable") from exc
    if not math.isfinite(height) or height <= 0:
        raise HemispherePlaneGeometryError("positive endpoint plane height is not representable")
    size = x.size
    jets = [[Jet2.variable(float(x[i, j]), 3 * i + j, size)
             for j in range(3)] for i in range(len(x))]
    # Center seeded jets at their input values to retain derivatives without
    # cancellation in large absolute coordinates. Fraction supplies each base.
    increments = [[entry - entry.value for entry in row] for row in jets]
    height_jet = Jet2.constant(height, size) + sum(
        (nunit[j] * increments[endpoint][j] for j in range(3)),
        Jet2.constant(0.0, size))
    rf = Fraction.from_float(radius)
    branch = "sphere" if q <= 0 else ("parallel_rim" if D == 0 else "rim")
    try:
        if branch == "sphere":
            gap = _stable_gap(hn, nn, Fraction(1), nn, rf)
            gap_jet = height_jet - radius
            skin = x[endpoint] - radius * nunit
        elif branch == "parallel_rim":
            gap, gap_jet, skin = height, height_jet, None
            if not math.isfinite(gap) or gap <= 0:
                raise HemispherePlaneGeometryError("positive parallel-rim height is not representable")
        else:
            tangent = [Jet2.constant(float(vscaled[j]), size) + sum(
                ((float(Fraction(a) / vs) * increments[i][j])
                 for i, a in enumerate(coefficients)), Jet2.constant(0.0, size))
                for j in range(3)]
            cross_exact = _cross(nscaled, vscaled)
            cross = [_replace_value(a, value) for a, value in
                     zip(_cross([float(a) for a in nscaled], tangent), cross_exact)]
            tangent_norm = sqrt(_replace_value(dot(tangent, tangent), vv))
            cross_norm = sqrt(_replace_value(dot(cross, cross), D))
            support = cross_norm / (norm_n * tangent_norm)
            gap_jet = height_jet - radius * support
            gap = _stable_gap(hn, nn, vv, D, rf)
            # v x (N x v), evaluated exactly, gives the rim support
            # direction without subtracting nearly parallel unit vectors.
            direction = _cross(vscaled, cross_exact)
            denominator = _sqrt_bound(vv) * _sqrt_bound(D)
            skin = x[endpoint] - radius * np.array([float(a / denominator) for a in direction])
    except (ArithmeticError, ValueError) as exc:
        if isinstance(exc, HemispherePlaneGeometryError):
            raise
        raise HemispherePlaneGeometryError("support derivatives are not representable") from exc

    if (not math.isfinite(gap_jet.value) or not np.all(np.isfinite(gap_jet.gradient))
            or not np.all(np.isfinite(gap_jet.hessian)) or not np.all(np.isfinite(nunit))
            or (skin is not None and not np.all(np.isfinite(skin)))):
        raise HemispherePlaneGeometryError("support state is not finite")
    # This checks floating evaluation consistency, never activation. Near
    # activation the exact conjugate value supplies the literal sign.
    consistency_scale = max(1.0, radius, abs(height))
    if abs(gap_jet.value - gap) > 64 * np.finfo(float).eps * consistency_scale:
        raise HemispherePlaneGeometryError("support AD and exact gap values disagree")
    return HemispherePlaneState(gap, _readonly(gap_jet.gradient),
                                _readonly(gap_jet.hessian),
                                None if skin is None else _readonly(skin),
                                _readonly(nunit), branch)
