"""Value-only cancellation certificate for an S3 source/original Q4 target.

This is an operand resolution envelope, not an integration error estimate or a
physical gap tolerance. Inputs are binary64 Cartesian operands. The two measured
residuals must be maxima of the *returned* Q1 shapes' forward projection errors,
in metres. ``natural_coordinate_bound`` encloses both paths' returned natural
coordinates, including shape-recovery arithmetic, and includes [-1, 1]. The
exact inverse branch is the convex, positively oriented Q4 branch in [-1, 1].

For F=a+b*r+c*s+d*r*s, det(DF) is affine in (r,s). Thus its minimum on a box
is at a corner; the maximum Frobenius norm is bounded by the corner maximum.
With determinant lower bound d and Frobenius upper bound j, ||DF^-1|| <= j/d.
Moreover F(x)-F(y)=DF((x+y)/2)(x-y) exactly. This last identity bounds inverse
error here without assuming a generic nonlinear inverse Lipschitz theorem.
Original physical Q4 heights remain bilinear; their gradient bound multiplies
the measured residuals and the forward/frame arithmetic envelope.

The gamma256 budget is a conservative binary64 value-chain operation budget:
source edge/cross/norm/frame (<80 scalar operations), scalar reprojection and
frame evaluation (<80), Q1 interpolation/recovery (<60), and final Cartesian
gap (<30). It assumes round-to-nearest basic arithmetic and square roots, with
no overflow/underflow or fast-math reassociation. Newton stopping constants are
not included: its final forward residual is supplied explicitly. Derivative,
quadrature and support-agreement gates remain the callers' separate duties.
"""
from __future__ import annotations

import numpy as np

from .s3_overlap_derivatives import UnsupportedOverlapBranch


_EPS = np.finfo(float).eps  # conservative: twice binary64 unit roundoff
_GAMMA = 256 * _EPS / (1 - 256 * _EPS)


def _up(value: float) -> float:
    return float(np.nextafter(value, np.inf))


def _enclosure_fits(inverse: float, residual: float, outer: float, inner: float) -> bool:
    # Fail closed at a one-ulp boundary: do not compare two nearest-rounded
    # estimates as if they were certified upper/lower bounds.
    lhs = _up(inverse*residual)
    rhs = float(np.nextafter(outer-inner, -np.inf))
    return bool(np.isfinite(lhs) and np.isfinite(rhs) and lhs <= rhs)


def _within_cap(allowances: np.ndarray, support: np.ndarray, min_edge: float) -> bool:
    normalized = np.nextafter(allowances/support, np.inf)
    cap = float(np.nextafter(1e-9*min_edge, -np.inf))
    return bool(np.all(np.isfinite(normalized)) and np.all(normalized <= cap))


def _q1_derivatives(r: float, s: float) -> np.ndarray:
    return .25 * np.array(((-(1-s), 1-s, 1+s, -(1+s)),
                           (-(1-r), -(1+r), 1+r, 1-r)))


def s3_q4_gap_operand_allowance(
    source, target, support_scale, *, scalar_projection_residual_m,
    jet_projection_residual_m, natural_coordinate_bound,
    absolute_support_scale=None,
) -> np.ndarray:
    """Return immutable per-row weighted-gap allowances [m^3], or reject.

    ``absolute_support_scale`` bounds each executed positive-weight quadrature
    sum of |P1 hat| [m^2]. It need not be a polygon-wide pointwise supremum. If
    omitted, positive hats and exact support are assumed. All three positive
    rows are kept. Every allowance divided by its positive support is capped
    at 1e-9 of the smallest source edge, including constant-height targets.
    """
    try:
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        support = np.asarray(support_scale, dtype=float)
        absolute = support if absolute_support_scale is None else np.asarray(
            absolute_support_scale, dtype=float)
        residuals = np.asarray((scalar_projection_residual_m,
                                jet_projection_residual_m), dtype=float)
        box = float(natural_coordinate_bound)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnsupportedOverlapBranch("S3/Q4 gap resolution inputs are invalid") from error
    if (source.shape != (3, 3) or target.shape != (4, 3)
            or support.shape != (3,) or absolute.shape != (3,)
            or residuals.shape != (2,) or not np.isfinite(box) or box < 1
            or any(not np.all(np.isfinite(a)) for a in
                   (source, target, support, absolute, residuals))
            or np.any(support <= 0) or np.any(absolute < support)
            or np.any(residuals < 0)):
        raise UnsupportedOverlapBranch("S3/Q4 gap resolution inputs are invalid")
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            answer = _allowance(source, target, support, absolute, residuals, box)
    except (OverflowError, FloatingPointError, ZeroDivisionError) as error:
        raise UnsupportedOverlapBranch("S3/Q4 gap resolution arithmetic is unresolved") from error
    return np.frombuffer(answer.tobytes(), dtype=float)


def _allowance(source, target, support, absolute, residuals, box):
    reject = UnsupportedOverlapBranch
    # Absolute coordinates deliberately matter: a distant origin must be
    # recentered, even when a particular subtraction happens to be exact.
    operand = _up(float(np.max(np.sum(np.abs(np.vstack((source, target))), axis=1))))
    cartesian_error = _up(_GAMMA * operand)
    edge = source[1:] - source[0]
    lengths = np.linalg.norm(np.roll(source, -1, axis=0)-source, axis=1)
    edge_error = _up(2*np.sqrt(3)*cartesian_error)
    min_edge = float(np.nextafter(float(np.min(lengths))-edge_error, -np.inf))
    a, b = edge
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    raw_normal = np.cross(a, b)
    raw_length = np.linalg.norm(raw_normal)
    cross_abs = np.abs(a[[1, 2, 0]]*b[[2, 0, 1]]) + np.abs(a[[2, 0, 1]]*b[[1, 2, 0]])
    cross_error = _up((la+lb)*edge_error+edge_error**2
                      + _GAMMA*np.linalg.norm(cross_abs))
    normal_lower = float(np.nextafter(raw_length-cross_error-_GAMMA*raw_length, -np.inf))
    tangent_lower = float(np.nextafter(la-edge_error-_GAMMA*la, -np.inf))
    if (not np.all(np.isfinite((min_edge, normal_lower, tangent_lower)))
            or min_edge <= 0 or normal_lower <= 0 or tangent_lower <= 0):
        raise reject("S3/Q4 source frame is unresolved")
    # sigma_max^2 <= ||E||_F^2 and sigma_min*sigma_max=|a cross b|.
    source_condition = _up((np.linalg.norm(edge)+np.sqrt(2)*edge_error)**2/normal_lower)
    normal_error = _up(2*(cross_error+_GAMMA*raw_length)/normal_lower+_GAMMA)
    tangent_error = _up(2*(edge_error+_GAMMA*la)/tangent_lower+_GAMMA)
    binormal_error = _up(normal_error+tangent_error+normal_error*tangent_error+_GAMMA)
    frame_error = _up(np.hypot(tangent_error, binormal_error))
    normal = raw_normal/raw_length
    tangent = a/la
    frame = np.vstack((tangent, np.cross(normal, tangent)))
    relative = target-source[0]
    span = _up(float(np.max(np.linalg.norm(relative, axis=1))))
    xy = relative @ frame.T
    # Includes scalar's extra Cartesian orthogonal projection, unlike the
    # Jet path's direct frame dot. Both are compared to the same exact frame.
    corner_error = _up(2*np.sqrt(2)*cartesian_error +
                       (2*frame_error+2*normal_error+_GAMMA)*span)

    def bounds(radius):
        det_min = np.inf
        jac_max = height_max = 0.
        for r in (-radius, radius):
            for s in (-radius, radius):
                derivatives = _q1_derivatives(r, s)
                jac = (derivatives @ xy).T
                norm = float(np.linalg.norm(jac))
                # Each derivative column has l1 norm <= radius (radius>=1).
                eta = _up(np.sqrt(2)*radius*corner_error +
                          _GAMMA*float(np.linalg.norm(np.abs(derivatives) @ np.abs(xy))))
                products = abs(jac[0, 0]*jac[1, 1])+abs(jac[0, 1]*jac[1, 0])
                determinant = jac[0, 0]*jac[1, 1]-jac[0, 1]*jac[1, 0]
                determinant_error = _up(_GAMMA*products+norm*eta+eta**2)
                det_min = min(det_min, float(np.nextafter(determinant-determinant_error, -np.inf)))
                jac_max = max(jac_max, _up(norm+eta))
                # Original physical bilinear heights; no planar/affine replacement.
                physical_derivative = derivatives @ relative
                height = float(np.linalg.norm(physical_derivative @ normal))
                height_error = _up(normal_error*np.linalg.norm(physical_derivative)
                                   + np.sqrt(2)*radius*2*cartesian_error
                                   + _GAMMA*np.linalg.norm(np.abs(derivatives) @ np.abs(relative)))
                height_max = max(height_max, _up(height+height_error))
        if (not np.all(np.isfinite((det_min, jac_max, height_max)))
                or det_min <= 0 or jac_max <= 0):
            raise reject("S3/Q4 projected inverse determinant is unresolved")
        return _up(jac_max/det_min), _up(jac_max**2/det_min), height_max

    # The supplied envelope already includes returned-shape recovery error.
    # Add our arithmetic plus a bounded a posteriori frame/residual expansion.
    base_box = _up(box+_GAMMA*(1+box)**2)
    inverse, _, _ = bounds(base_box)
    forward_error = _up((base_box**2)*corner_error +
                        _GAMMA*(1+base_box)**2*(span+operand))
    residual_budget = _up(float(np.sum(residuals))+2*forward_error)
    radius = _up(inverse*residual_budget)
    expanded_box = _up(base_box+2*radius)
    inverse, target_condition, height = bounds(expanded_box)
    expanded_forward = _up(expanded_box**2*corner_error +
                           _GAMMA*(1+expanded_box)**2*(span+operand))
    expanded_residual = _up(float(np.sum(residuals))+2*expanded_forward)
    if not _enclosure_fits(inverse, expanded_residual, expanded_box, base_box):
        raise reject("S3/Q4 inverse enclosure cannot be certified")
    # A map-conditioned arithmetic certificate remains present when dh=0.
    # This avoids silently admitting ill-conditioned constant-height inverses.
    direct = _up(cartesian_error*(1+source_condition)*(1+target_condition)*expanded_box**2
                 + normal_error*span*expanded_box**2)
    distance = _up(direct+height*inverse*expanded_residual)
    allowances = np.nextafter(absolute*distance, np.inf)
    if (not np.isfinite(distance) or not np.all(np.isfinite(allowances))
            or not _within_cap(allowances, support, min_edge)):
        raise reject("S3/Q4 gap coordinate/inverse cancellation exceeds local resolution")
    return allowances
