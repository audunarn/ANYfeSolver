"""Second derivatives of a smooth, positive S3/S3 projected overlap.

This is a local geometric primitive.  The 18 coordinates are ordered as the
three source positions followed by the three target positions, each in xyz
order.  The clipping polygon is an integration device; its vertices are not
contact features or additional pressure degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_mixed_ad import Jet2, sqrt

class UnsupportedOverlapBranch(ValueError):
    """The positive overlap has no differentiable, unambiguous local branch."""


@dataclass(frozen=True)
class S3OverlapDerivatives:
    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    support_gradient: np.ndarray
    support_hessian: np.ndarray
    weighted_gap_gradient: np.ndarray
    weighted_gap_hessian: np.ndarray
    overlap_area: float
    reference_overlap_area: float
    branch_signature: tuple[int, ...]


def _dot(a: list[Jet2], b: list[Jet2]) -> Jet2:
    return sum((x*y for x, y in zip(a, b)), start=Jet2.constant(0, 18))


def _sub(a: list[Jet2], b: list[Jet2]) -> list[Jet2]:
    return [x-y for x, y in zip(a, b)]


def _cross(a: list[Jet2], b: list[Jet2]) -> list[Jet2]:
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]]


def _cross2(a: list[Jet2], b: list[Jet2]) -> Jet2:
    return a[0]*b[1]-a[1]*b[0]


def _xy_sub(a: list[Jet2], b: list[Jet2]) -> list[Jet2]:
    return [a[0]-b[0], a[1]-b[1]]


def _barycentric(triangle: list[list[Jet2]], point: list[Jet2]) -> list[Jet2]:
    a = _xy_sub(triangle[1], triangle[0])
    b = _xy_sub(triangle[2], triangle[0])
    p = _xy_sub(point, triangle[0])
    denominator = _cross2(a, b)
    u = _cross2(p, b)/denominator
    v = _cross2(a, p)/denominator
    return [1-u-v, u, v]


def _signed_area(polygon: list[list[Jet2]]) -> Jet2:
    return sum((_cross2(point, polygon[(i+1) % len(polygon)])
                for i, point in enumerate(polygon)),
               start=Jet2.constant(0, 18))/2


def _clip(source: list[list[Jet2]], target: list[list[Jet2]],
          threshold: float) -> tuple[list[list[Jet2]], tuple[int, ...]]:
    polygon = source
    signature: list[int] = []
    for edge_index in range(3):
        start = target[edge_index]
        edge = _xy_sub(target[(edge_index+1) % 3], start)
        if not polygon:
            raise UnsupportedOverlapBranch("projected S3 faces have no positive overlap")
        old = polygon[-1]
        old_distance = _cross2(edge, _xy_sub(old, start))
        output: list[list[Jet2]] = []
        signature.append(len(polygon))
        for current in polygon:
            distance = _cross2(edge, _xy_sub(current, start))
            if min(abs(old_distance.value), abs(distance.value)) <= threshold:
                raise UnsupportedOverlapBranch("projected vertex lies on a clipping boundary")
            old_inside = old_distance.value > 0
            inside = distance.value > 0
            signature.append(int(inside))
            if old_inside != inside:
                denominator = old_distance-distance
                if abs(denominator.value) <= threshold:
                    raise UnsupportedOverlapBranch("degenerate clipping intersection")
                fraction = old_distance/denominator
                output.append([a+fraction*(b-a) for a, b in zip(old, current)])
            if inside:
                output.append(current)
            old, old_distance = current, distance
        polygon = output
    signature.append(len(polygon))
    return polygon, tuple(signature)


def _components(jets: list[Jet2]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([j.value for j in jets]),
            np.stack([j.gradient for j in jets]),
            np.stack([j.hessian for j in jets]))


def _triangle(points: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must have three finite xyz positions")
    scale = max(np.linalg.norm(values[1]-values[0]),
                np.linalg.norm(values[2]-values[0]))
    if (scale <= 0 or np.linalg.norm(np.cross(values[1]-values[0],
                                            values[2]-values[0]))
            <= 1e-12*scale*scale):
        raise ValueError(f"{label} is degenerate")
    return values.copy()


def s3_overlap_derivatives(
    reference_source: np.ndarray, current_source: np.ndarray,
    current_target: np.ndarray, *, relative_branch_tolerance: float = 1e-10,
    stable_plane_gap: bool = False,
) -> S3OverlapDerivatives:
    """Integrate P1 source support and normal gap on a fixed smooth branch.

    The source and target are original, positively oriented physical S3 faces.
    Source and target normals must have positive dot product.  The returned
    support and weighted gap use positive *reference* source measure.  No
    derivative is supplied at zero overlap, vertex-edge incidence, or a
    vanishing integration triangle.
    """
    if not np.isfinite(relative_branch_tolerance) or relative_branch_tolerance <= 0:
        raise ValueError("relative_branch_tolerance must be finite and positive")
    rs = _triangle(reference_source, "reference source")
    source = _triangle(current_source, "current source")
    target = _triangle(current_target, "current target")
    reference_area = 0.5*np.linalg.norm(np.cross(rs[1]-rs[0], rs[2]-rs[0]))
    values = np.concatenate((source, target)).reshape(-1)
    points = [[Jet2.variable(values[3*i+k], 3*i+k, 18) for k in range(3)]
              for i in range(6)]
    s, t = points[:3], points[3:]
    source_edge = _sub(s[1], s[0])
    raw_normal = _cross(source_edge, _sub(s[2], s[0]))
    normal = [item/sqrt(_dot(raw_normal, raw_normal)) for item in raw_normal]
    reference_normal = np.cross(rs[1]-rs[0], rs[2]-rs[0])
    if np.dot(reference_normal,
              np.array([item.value for item in raw_normal])) <= 0:
        raise UnsupportedOverlapBranch("source face polarity differs from reference")
    tangent = [item/sqrt(_dot(source_edge, source_edge)) for item in source_edge]
    binormal = _cross(normal, tangent)
    target_normal = _cross(_sub(t[1], t[0]), _sub(t[2], t[0]))
    if _dot(normal, target_normal).value <= 0:
        raise UnsupportedOverlapBranch("source and target face polarities conflict")

    def projected(point: list[Jet2]) -> list[Jet2]:
        delta = _sub(point, s[0])
        return [_dot(delta, tangent), _dot(delta, binormal)]

    source_xy = [projected(point) for point in s]
    target_xy = [projected(point) for point in t]
    scale = max(np.hypot(point[0].value, point[1].value)
                for point in (*source_xy, *target_xy))
    threshold = relative_branch_tolerance*scale*scale
    if _signed_area(target_xy).value <= threshold:
        raise UnsupportedOverlapBranch("target projection is degenerate or reversed")
    polygon, signature = _clip(source_xy, target_xy, threshold)
    if len(polygon) < 3:
        raise UnsupportedOverlapBranch("projected S3 faces have no positive overlap")
    polygon_area = _signed_area(polygon)
    if polygon_area.value <= threshold:
        raise UnsupportedOverlapBranch("projected overlap has no positive area")
    source_area = sqrt(_dot(raw_normal, raw_normal))/2
    ratio = reference_area/source_area
    support = [Jet2.constant(0, 18) for _ in range(3)]
    weighted_gap = [Jet2.constant(0, 18) for _ in range(3)]
    source_shapes = [_barycentric(source_xy, point) for point in polygon]
    gaps: list[Jet2] = []
    if stable_plane_gap:
        from .s3_stable_plane_gap import stable_corner_gap_jets
        corner_gaps=stable_corner_gap_jets(s,t,source,target)
        gaps=[sum((shape[j]*corner_gaps[j] for j in range(3)),
                  start=Jet2.constant(0,18)) for shape in source_shapes]
    else:
        for point in polygon:
            target_shape = _barycentric(target_xy, point)
            target_point = [sum((target_shape[j]*t[j][k] for j in range(3)),
                                start=Jet2.constant(0, 18)) for k in range(3)]
            gaps.append(_dot(_sub(s[0], target_point), normal))
    for i in range(1, len(polygon)-1):
        indices = (0, i, i+1)
        area = _cross2(_xy_sub(polygon[i], polygon[0]),
                       _xy_sub(polygon[i+1], polygon[0]))/2
        if area.value <= threshold:
            raise UnsupportedOverlapBranch("degenerate overlap integration triangle")
        for node in range(3):
            shapes = [source_shapes[j][node] for j in indices]
            local_gaps = [gaps[j] for j in indices]
            shape_sum = sum(shapes, start=Jet2.constant(0, 18))
            gap_sum = sum(local_gaps, start=Jet2.constant(0, 18))
            diagonal = sum((a*b for a, b in zip(shapes, local_gaps)),
                           start=Jet2.constant(0, 18))
            support[node] = support[node]+ratio*area*shape_sum/3
            weighted_gap[node] = weighted_gap[node]+ratio*area*(shape_sum*gap_sum+diagonal)/12
    support_value, support_gradient, support_hessian = _components(support)
    gap_value, gap_gradient, gap_hessian = _components(weighted_gap)
    if (np.min(support_value) <= 0 or
            not all(np.all(np.isfinite(item)) for item in
                    (support_value, support_gradient, support_hessian,
                     gap_value, gap_gradient, gap_hessian))):
        raise UnsupportedOverlapBranch("nonpositive support or nonfinite derivative")
    return S3OverlapDerivatives(
        support_value, gap_value, support_gradient, support_hessian,
        gap_gradient, gap_hessian, polygon_area.value,
        (ratio*polygon_area).value, signature,
    )
