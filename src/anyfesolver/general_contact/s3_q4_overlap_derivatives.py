"""Local S3-source/Q4-target projection derivatives; no solver admission.

Coordinates: source xyz (9), original target xyz (12). Projection uses the
source normal, matching q4_target_overlap. Target positions use its original
bilinear interpolation, including implicit projected coordinates. Six explicit
absolute controls bound measured support/gap value, gradient and Hessian errors.
"""
from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, sqrt
from .moving import _TRIANGLE_RULE
from .q4_s3_overlap_derivatives import (
    UnsupportedQ4S3Branch, _area_resolution, _clip, _cross, _cross2, _dot, _natural,
    _normalize, _parts, _q1, _signed_area, _sub, _sum, _zero,
)


@dataclass(frozen=True)
class S3Q4OverlapDerivatives:
    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    support_gradient: np.ndarray
    support_hessian: np.ndarray
    weighted_gap_gradient: np.ndarray
    weighted_gap_hessian: np.ndarray
    overlap_area: float
    reference_overlap_area: float
    overlap_area_resolution_m2: float
    branch_signature: tuple
    integration_triangles: int
    measured_errors: tuple
    projection_residual_m: float
    natural_coordinate_bound: float
    absolute_reference_support: np.ndarray


@dataclass(frozen=True)
class _S3Q4FirstOrderWork:
    """Work-query values/gradients only; no unique Hessian is exposed."""
    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    support_gradient: np.ndarray
    weighted_gap_gradient: np.ndarray
    overlap_area: float
    reference_overlap_area: float
    overlap_area_resolution_m2: float
    branch_signature: tuple
    integration_triangles: int
    measured_errors: tuple
    projection_residual_m: float
    natural_coordinate_bound: float
    absolute_reference_support: np.ndarray
    seam_events: tuple
    derivative_order: int = 1


def _p1(point, corners):
    a, b, p = _sub(corners[1], corners[0]), _sub(corners[2], corners[0]), _sub(point, corners[0])
    determinant = _cross2(a, b)
    u, v = _cross2(p, b)/determinant, _cross2(a, p)/determinant
    return (1-u-v, u, v)


def s3_q4_overlap_derivatives(reference_source, current_source, current_target, *,
        support_tolerance, weighted_gap_tolerance,
        support_gradient_tolerance, weighted_gap_gradient_tolerance,
        support_hessian_tolerance, weighted_gap_hessian_tolerance,
        maximum_triangles, relative_branch_tolerance=1e-10):
    """Strict full-Hessian profile; seam refusal remains the default."""
    return _calculate(reference_source, current_source, current_target,
        support_tolerance=support_tolerance, weighted_gap_tolerance=weighted_gap_tolerance,
        support_gradient_tolerance=support_gradient_tolerance,
        weighted_gap_gradient_tolerance=weighted_gap_gradient_tolerance,
        support_hessian_tolerance=support_hessian_tolerance,
        weighted_gap_hessian_tolerance=weighted_gap_hessian_tolerance,
        maximum_triangles=maximum_triangles,
        relative_branch_tolerance=relative_branch_tolerance)


def _s3_q4_first_order_work(reference_source, current_source, current_target, **controls):
    """Private s9 work profile; retain all six quadrature controls.

    Branch Hessians are used internally for the same refinement checks, but
    are never returned as a unique seam derivative or contact tangent.
    """
    events = []
    result = _calculate(reference_source, current_source, current_target,
                        _work_events=events, **controls)
    fields = ("reference_support", "reference_weighted_gap", "support_gradient",
              "weighted_gap_gradient", "overlap_area", "reference_overlap_area",
              "overlap_area_resolution_m2", "branch_signature", "integration_triangles",
              "measured_errors", "projection_residual_m", "natural_coordinate_bound",
              "absolute_reference_support")
    return _S3Q4FirstOrderWork(*(getattr(result, field) for field in fields), tuple(events))


def _calculate(reference_source, current_source, current_target, *,
        support_tolerance, weighted_gap_tolerance,
        support_gradient_tolerance, weighted_gap_gradient_tolerance,
        support_hessian_tolerance, weighted_gap_hessian_tolerance,
        maximum_triangles, relative_branch_tolerance=1e-10, _work_events=None):
    controls = np.array((support_tolerance, weighted_gap_tolerance,
        support_gradient_tolerance, weighted_gap_gradient_tolerance,
        support_hessian_tolerance, weighted_gap_hessian_tolerance), dtype=float)
    if (not np.all(np.isfinite(controls)) or np.min(controls) <= 0
            or type(maximum_triangles) is not int or maximum_triangles < 1
            or not np.isfinite(relative_branch_tolerance) or relative_branch_tolerance <= 0):
        raise ValueError("six positive dimensioned derivative controls and finite triangle budget required")
    reference, source, target = map(lambda x: np.asarray(x, dtype=float),
                                    (reference_source, current_source, current_target))
    if (reference.shape != (3, 3) or source.shape != (3, 3) or target.shape != (4, 3)
            or not all(np.all(np.isfinite(x)) for x in (reference, source, target))):
        raise ValueError("finite S3 reference/current and original four Q4 positions required")
    reference_normal = np.cross(reference[1]-reference[0], reference[2]-reference[0])
    reference_area = .5*np.linalg.norm(reference_normal)
    if reference_area <= 0:
        raise UnsupportedQ4S3Branch("degenerate reference S3")
    values = np.concatenate((source, target)).reshape(-1)
    points = [[Jet2.variable(values[3*i+k], 3*i+k, 21) for k in range(3)] for i in range(7)]
    s, t = points[:3], points[3:]
    edge = _sub(s[1], s[0]); raw_normal = _cross(edge, _sub(s[2], s[0]))
    raw_values = np.array([x.value for x in raw_normal])
    if np.dot(reference_normal, raw_values) <= 0:
        raise UnsupportedQ4S3Branch("source polarity differs from reference")
    normal, tangent = _normalize(raw_normal), _normalize(edge)
    binormal = _cross(normal, tangent)

    def projected(point):
        delta = _sub(point, s[0])
        return [_dot(delta, tangent), _dot(delta, binormal)]

    source_xy, target_xy = [projected(x) for x in s], [projected(x) for x in t]
    span = max(np.hypot(x[0].value, x[1].value) for x in (*source_xy, *target_xy))
    threshold = relative_branch_tolerance*span**2
    for r, q in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
        _, dr, ds = _q1(Jet2.constant(r, 21), Jet2.constant(q, 21))
        a = [_sum(dr[i]*target_xy[i][k] for i in range(4)) for k in range(2)]
        b = [_sum(ds[i]*target_xy[i][k] for i in range(4)) for k in range(2)]
        if _cross2(a, b).value <= threshold:
            raise UnsupportedQ4S3Branch("projected Q4 Jacobian is degenerate or reversed")
    if _work_events is None:
        polygon, vertex_errors, signature = _clip(source_xy, target_xy,
            relative_branch_tolerance*span, edge_count=4)
    else:
        polygon, vertex_errors, signature = _clip(source_xy, target_xy,
            relative_branch_tolerance*span, edge_count=4, _work_events=_work_events)
        if _work_events and polygon:
            # Preserve every fan triangle. Put the shortest boundary edge at
            # the fan origin to avoid subtracting nearly equal long vectors
            # when a represented point is extremely close to the seam.
            first = min(range(len(polygon)), key=lambda i: sum(
                (polygon[(i+1) % len(polygon)][k].value - polygon[i][k].value)**2
                for k in range(2)))
            polygon = polygon[first:] + polygon[:first]
            vertex_errors = vertex_errors[first:] + vertex_errors[:first]
    if len(polygon) < 3 or _signed_area(polygon).value <= threshold:
        raise UnsupportedQ4S3Branch("no resolved positive S3/Q4 overlap")
    area = _signed_area(polygon)
    area_resolution = _area_resolution(polygon, vertex_errors)
    if area.value <= area_resolution:
        raise UnsupportedQ4S3Branch("S3/Q4 overlap area is unresolved at propagated vertex resolution")
    ratio = reference_area/(sqrt(_dot(raw_normal, raw_normal))*.5)

    projection_residual = 0.0
    natural_bound = 1.0
    absolute_shape_bound = np.zeros(3)
    projected_values = np.array([[x.value for x in corner] for corner in target_xy])

    def rule(triangle, inherited_area=None):
        nonlocal projection_residual, natural_bound, absolute_shape_bound
        a = (inherited_area if inherited_area is not None else
             _cross2(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))*.5)
        if a.value <= 0:
            raise UnsupportedQ4S3Branch("degenerate integration triangle")
        support, weighted = [_zero() for _ in range(3)], [_zero() for _ in range(3)]
        for barycentric, weight in _TRIANGLE_RULE:
            point = [_sum(barycentric[i]*triangle[i][k] for i in range(3)) for k in range(2)]
            shape_a = _p1(point, source_xy)
            shape_b, _, _ = _natural(point, target_xy)
            values_b = np.array([x.value for x in shape_b])
            values_a = np.array([x.value for x in shape_a])
            values_point = np.array([x.value for x in point])
            projection_residual = max(projection_residual,
                float(np.linalg.norm(values_b@projected_values-values_point)))
            r = values_b[1]+values_b[2]-values_b[0]-values_b[3]
            q = values_b[2]+values_b[3]-values_b[0]-values_b[1]
            gamma4 = 4*np.finfo(float).eps/(1-4*np.finfo(float).eps)
            shape_l1_upper = np.nextafter(np.sum(np.abs(values_b))/(1-gamma4), np.inf)
            recovery_error = (gamma4+gamma4/(1-gamma4))*shape_l1_upper
            natural_bound = max(natural_bound,
                float(np.nextafter(abs(float(r))+recovery_error, np.inf)),
                float(np.nextafter(abs(float(q))+recovery_error, np.inf)))
            absolute_shape_bound = np.maximum(absolute_shape_bound, np.abs(values_a))
            target_point = [_sum(shape_b[i]*t[i][k] for i in range(4)) for k in range(3)]
            gap = _dot(_sub(s[0], target_point), normal)
            for node in range(3):
                contribution = ratio*a*weight*shape_a[node]
                support[node] = support[node]+contribution
                weighted[node] = weighted[node]+contribution*gap
        return support, weighted

    def children(triangle):
        a, b, c = triangle
        ab, bc, ca = ([(x+y)*.5 for x, y in zip(first, second)]
                      for first, second in ((a, b), (b, c), (c, a)))
        return ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))

    support, weighted = [_zero() for _ in range(3)], [_zero() for _ in range(3)]
    measured = np.zeros(6); evaluations = 0
    pending = [(tuple(polygon[index] for index in (0, i, i+1)), 1/(len(polygon)-2), None)
               for i in range(1, len(polygon)-1)]
    while pending:
        triangle, fraction, inherited_area = pending.pop()
        if evaluations+5 > maximum_triangles:
            raise RuntimeError("S3/Q4 derivative integration triangle budget exhausted")
        # Midpoint subdivision has exactly one quarter of the parent area.
        # Carry its jet on the C1 path: re-subtracting rounded midpoint
        # coordinates can erase an extremely thin, positive seam triangle.
        if _work_events is not None and inherited_area is None:
            inherited_area = _cross2(_sub(triangle[1], triangle[0]),
                                     _sub(triangle[2], triangle[0]))*.5
        coarse = rule(triangle, inherited_area)
        child_area = None if inherited_area is None else inherited_area*.25
        subdivided = children(triangle)
        fine = [[_zero() for _ in range(3)], [_zero() for _ in range(3)]]
        for child in subdivided:
            candidate = rule(child, child_area)
            for field in range(2):
                for node in range(3):
                    fine[field][node] = fine[field][node]+candidate[field][node]
        evaluations += 5
        low, high = [_parts(field) for field in coarse], [_parts(field) for field in fine]
        errors = np.array([np.max(np.abs(high[field][order]-low[field][order]))
                           for order in range(3) for field in range(2)])
        if not np.all(np.isfinite(errors)):
            raise UnsupportedQ4S3Branch("nonfinite S3/Q4 derivative integration error")
        if np.any(errors > fraction*controls):
            pending.extend((child, fraction*.25, child_area) for child in subdivided)
            continue
        measured += errors
        for node in range(3):
            support[node] = support[node]+fine[0][node]
            weighted[node] = weighted[node]+fine[1][node]
    sv, sg, sh = _parts(support); wv, wg, wh = _parts(weighted)
    if min(sv) <= 0 or not all(np.all(np.isfinite(x)) for x in (sv, sg, sh, wv, wg, wh)):
        raise UnsupportedQ4S3Branch("nonpositive supported mode or nonfinite derivative")
    return S3Q4OverlapDerivatives(sv, wv, sg, sh, wg, wh, area.value,
        (ratio*area).value, area_resolution, signature, evaluations, tuple(measured),
        projection_residual, float(np.nextafter(natural_bound, np.inf)),
        (ratio*area).value*absolute_shape_bound)
