"""Complete endpoint hemisphere support work on a moving physical S3 plane.

The output coordinates are all beam translations followed by the three skin
vertices, each in xyz order. The S3 physical-skin adapter owns their structural
pullback. This centreline-envelope geometry has no material spin history;
finite-face, regularity, swept admission and penalty assembly are caller work.
"""

from __future__ import annotations

from fractions import Fraction
import math
from numbers import Integral

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, dot, sqrt
from .beam_hemisphere_support import (
    HemispherePlaneGeometryError, HemispherePlaneState,
    _cross, _exact_dot, _readonly, _replace_value, _stable_gap,
)
from .capped_cylinder import _sqrt_bound


def _fractions(array):
    return [[Fraction.from_float(float(value)) for value in row] for row in array]


def hemisphere_s3_work(beam_positions, skin_positions, endpoint, outward_sign,
                       radius, orientation) -> HemispherePlaneState:
    """Differentiate hemisphere support including the moving S3 normal.

    N=orientation*((s1-s0) cross (s2-s0)), h=(xe-s0).N/|N|.
    The outward tangent is the actual B2/B3 endpoint derivative. For
    q=v.N<=0, gap=h-R. For q>0, gap=h-R*|N cross v|/(|N||v|).
    Exact binary-input Fractions select these branches and the literal gap
    sign; Jet2 differentiates every beam and skin coordinate together.

    Positive endpoint plane height is required. At q=0 the sphere Hessian
    is the selected one-sided Hessian. A parallel rim (q>0, N cross v=0)
    is strictly open, with no unique skin point; only plane-height derivatives
    are returned there. They are not derivatives of a smooth support branch.
    """
    beam, skin = np.asarray(beam_positions, float), np.asarray(skin_positions, float)
    if beam.shape not in ((2, 3), (3, 3)) or not np.all(np.isfinite(beam)):
        raise HemispherePlaneGeometryError("beam_positions must be finite B2/B3 coordinates")
    if skin.shape != (3, 3) or not np.all(np.isfinite(skin)):
        raise HemispherePlaneGeometryError("skin_positions must contain three finite physical vertices")
    if (isinstance(endpoint, (bool, np.bool_)) or not isinstance(endpoint, Integral)
            or endpoint not in (0, len(beam) - 1)):
        raise HemispherePlaneGeometryError("endpoint must be zero or the last beam node index")
    if isinstance(outward_sign, (bool, np.bool_)) or outward_sign not in (-1, 1):
        raise HemispherePlaneGeometryError("outward_sign must be -1 or +1")
    if isinstance(orientation, (bool, np.bool_)) or orientation not in (-1, 1):
        raise HemispherePlaneGeometryError("orientation must be -1 or +1")
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HemispherePlaneGeometryError("radius must be finite and positive") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise HemispherePlaneGeometryError("radius must be finite and positive")

    bf, sf = _fractions(beam), _fractions(skin)
    edges = [[sf[i][j] - sf[0][j] for j in range(3)] for i in (1, 2)]
    N = [int(orientation) * value for value in _cross(*edges)]
    NN = _exact_dot(N, N)
    if NN == 0:
        raise HemispherePlaneGeometryError("physical S3 plane must be nondegenerate")
    coefficients = ((-1, 1) if len(beam) == 2 else
                    ((-3, 4, -1) if endpoint == 0 else (1, -4, 3)))
    coefficients = tuple(int(outward_sign) * value for value in coefficients)
    v = [sum((a * row[j] for a, row in zip(coefficients, bf)), Fraction(0))
         for j in range(3)]
    relative = [bf[endpoint][j] - sf[0][j] for j in range(3)]
    H, q, VV = _exact_dot(relative, N), _exact_dot(v, N), _exact_dot(v, v)
    if H <= 0:
        raise HemispherePlaneGeometryError("endpoint plane height must be strictly positive")
    if VV == 0:
        raise HemispherePlaneGeometryError("actual endpoint tangent must be nonzero")
    # These input-dependent scales are fixed during differentiation. All
    # unscaled geometry is recovered by constant factors, including normal
    # changes from BOTH physical S3 edges.
    ns, vs = max(abs(value) for value in N), max(abs(value) for value in v)
    edge_scales = [max(abs(value) for value in edge) for edge in edges]
    nscaled, vscaled = [value / ns for value in N], [value / vs for value in v]
    hn, nn, vv, qs = H / ns, NN / ns**2, VV / vs**2, q / (ns * vs)
    D = nn * vv - qs * qs
    if D < 0:
        raise HemispherePlaneGeometryError("exact support discriminant is negative")
    branch = "sphere" if q <= 0 else ("parallel_rim" if D == 0 else "rim")
    size = 3 * (len(beam) + 3)

    try:
        coordinates = [Jet2.variable(float(value), i, size) for i, value in
                       enumerate(np.r_[beam.ravel(), skin.ravel()])]
        # Zero-base increments avoid cancellation of absolute coordinates.
        increments = [entry - entry.value for entry in coordinates]
        source = [increments[3*i:3*i+3] for i in range(len(beam))]
        target = [increments[3*(len(beam)+i):3*(len(beam)+i)+3] for i in range(3)]
        edge_jets = [[Jet2.constant(float(edges[i][j] / edge_scales[i]), size)
                      + float(1 / edge_scales[i]) * (target[i+1][j] - target[0][j])
                      for j in range(3)] for i in range(2)]
        normal_factor = float(int(orientation) * edge_scales[0] * edge_scales[1] / ns)
        normal_jets = [_replace_value(normal_factor * value, exact) for value, exact in
                       zip(_cross(*edge_jets), nscaled)]
        normal_norm = sqrt(_replace_value(dot(normal_jets, normal_jets), nn))
        relative_jets = [Jet2.constant(float(relative[j]), size)
                         + source[endpoint][j] - target[0][j] for j in range(3)]
        height_jet = _replace_value(dot(relative_jets, normal_jets), hn) / normal_norm
        height = float(hn / _sqrt_bound(nn))
        if not math.isfinite(height) or height <= 0:
            raise HemispherePlaneGeometryError("positive endpoint plane height is not representable")
        normal = np.array([float(value) / normal_norm.value for value in nscaled])
        rf = Fraction.from_float(radius)

        if branch == "sphere":
            gap = _stable_gap(hn, nn, Fraction(1), nn, rf)
            gap_jet = height_jet - radius
            skin_point = beam[endpoint] - radius * normal
        elif branch == "parallel_rim":
            gap, gap_jet, skin_point = height, height_jet, None
        else:
            tangent_jets = [Jet2.constant(float(vscaled[j]), size) + sum(
                (float(Fraction(a) / vs) * source[i][j]
                 for i, a in enumerate(coefficients)), Jet2.constant(0.0, size))
                for j in range(3)]
            tangent_norm = sqrt(_replace_value(dot(tangent_jets, tangent_jets), vv))
            cross_exact = _cross(nscaled, vscaled)
            cross_jets = [_replace_value(value, exact) for value, exact in
                          zip(_cross(normal_jets, tangent_jets), cross_exact)]
            cross_norm = sqrt(_replace_value(dot(cross_jets, cross_jets), D))
            support = cross_norm / (normal_norm * tangent_norm)
            gap_jet = height_jet - radius * support
            gap = _stable_gap(hn, nn, vv, D, rf)
            direction = _cross(vscaled, cross_exact)
            denominator = _sqrt_bound(vv) * _sqrt_bound(D)
            skin_point = beam[endpoint] - radius * np.array(
                [float(value / denominator) for value in direction])
    except (ArithmeticError, ValueError) as exc:
        if isinstance(exc, HemispherePlaneGeometryError):
            raise
        raise HemispherePlaneGeometryError("moving-plane support is not representable") from exc

    if (not math.isfinite(gap_jet.value) or not np.all(np.isfinite(gap_jet.gradient))
            or not np.all(np.isfinite(gap_jet.hessian)) or not np.all(np.isfinite(normal))
            or (skin_point is not None and not np.all(np.isfinite(skin_point)))):
        raise HemispherePlaneGeometryError("moving-plane support state is not finite")
    if abs(gap_jet.value - gap) > 64 * np.finfo(float).eps * max(1.0, radius, abs(height)):
        raise HemispherePlaneGeometryError("moving-plane support AD and exact gap values disagree")
    return HemispherePlaneState(gap, _readonly(gap_jet.gradient), _readonly(gap_jet.hessian),
                                None if skin_point is None else _readonly(skin_point),
                                _readonly(normal), branch)
