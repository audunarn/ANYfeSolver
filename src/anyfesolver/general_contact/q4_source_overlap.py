"""Reference-area Q1 mortar integrals on an original bilinear Q4 source.

Clipping uses projected four-corner polygons. The integrand always evaluates
the physical Q1 source, its varying normal, and the original target shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moving import _triangle_samples
from .overlap import PlanarFace, _q4_shape, planar_projected_overlap
from .triangle_pair_energy import _triangle_area


@dataclass(frozen=True)
class Q4SourceOverlapIntegrals:
    support: np.ndarray
    weighted_gap: np.ndarray
    pressure_gram: np.ndarray
    overlap_area: float
    reference_overlap_area: float
    minimum_sampled_gap: float | None
    overlap_vertices: int
    area_error: float
    weighted_gap_error: float
    integration_triangles: int


def _q4_corner_normals(corners: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.cross(corners[(i+1) % 4]-corners[i],
                 corners[(i-1) % 4]-corners[i])
        for i in range(4)
    ])


def projected_q4_source_face(corners: np.ndarray) -> tuple[PlanarFace, np.ndarray]:
    current = np.asarray(corners, dtype=float)
    if current.shape != (4, 3) or not np.all(np.isfinite(current)):
        raise ValueError("Q4 source needs four finite physical corners")
    normal = np.cross(current[1]-current[0], current[3]-current[0])
    length = float(np.linalg.norm(normal))
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("Q4 source has ambiguous projected normal")
    normal /= length
    if np.any(_q4_corner_normals(current) @ normal <= 0.0):
        raise ValueError("Q4 source has inverted or folded bilinear geometry")
    projected = current - ((current-current[0])@normal)[:, None]*normal
    return PlanarFace((1, 1), projected), normal


def _normal_cone_gap_lower(normals: np.ndarray, differences: np.ndarray,
                           reference_normal: np.ndarray, scale: float) -> float:
    """Conservative signed gap for convex combinations of Q1 normals/positions."""
    controls = np.asarray(normals, dtype=float).reshape(-1, 3)
    projected_norms = controls @ reference_normal
    norm_lower = float(np.min(projected_norms))
    if norm_lower <= 0.0:
        return -float("inf")
    norm_upper = float(np.max(np.linalg.norm(controls, axis=1)))
    numerator = float(np.min(np.asarray(differences).reshape(-1, 3)@controls.T))
    lower = numerator/(norm_lower if numerator < 0.0 else norm_upper)
    return lower-128.0*np.finfo(float).eps*scale


def q4_source_gap_lower(source: np.ndarray, target: np.ndarray) -> float:
    """Bound the physical varying-normal gap over both complete Q1/P1 faces."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    _, normal = projected_q4_source_face(source)
    differences = source[:, None, :]-target[None, :, :]
    scale = max(1.0, float(np.max(np.abs(source))),
                float(np.max(np.abs(target))))
    return _normal_cone_gap_lower(
        _q4_corner_normals(source), differences, normal, scale,
    )


def q4_source_planar_overlap_gap_lower(
        source: np.ndarray, target: np.ndarray) -> float | None:
    """Bound the affine normal gap over a planar Q4/S3 projected overlap.

    This is an instantaneous geometry ingredient, not a swept certificate.
    ``None`` means there is no positive-area surface overlap. Nonplanar,
    inverted or nearly grazing faces fail rather than using this bound.
    """
    first = PlanarFace((1, 1), source)
    second = PlanarFace((2, 1), target)
    if len(first.corners) != 4:
        raise ValueError("planar overlap gap needs a Q4 source")
    overlap = planar_projected_overlap(first, second)
    if overlap is None:
        return None
    target_normal = np.cross(second.corners[1]-second.corners[0],
                             second.corners[-1]-second.corners[0])
    target_normal /= np.linalg.norm(target_normal)
    cosine = float(target_normal @ overlap.normal)
    if abs(cosine) < 0.25:
        raise ValueError("planar overlap gap has ambiguous normal projection")
    signed_gap = ((overlap.polygon-second.corners[0]) @ target_normal)/cosine
    scale = max(1.0,
                float(np.max(np.linalg.norm(first.corners-first.corners[0], axis=1))),
                float(np.max(np.linalg.norm(second.corners-second.corners[0], axis=1))),
                float(np.linalg.norm(second.corners[0]-first.corners[0])))
    margin = 512.0*np.finfo(float).eps*scale/abs(cosine)
    return float(np.min(signed_gap))-margin


def q4_source_projected_gap_lower(source: np.ndarray,
                                  target: np.ndarray) -> float:
    """Lower bound the gap of Q4 points sharing the overlap projection.

    If source and target points project to the same point along ``projection``,
    their difference is ``distance * projection``.  A target-face direction
    ``axis`` therefore bounds distance by the extrema of ``axis @ position``
    on both faces.  Q1 and P1 positions are convex combinations of their
    corners.  Positive source projected Jacobians make the physical normal's
    cosine with ``projection`` lie in (0, 1], so a negative distance lower
    bound is also a lower bound on the physical signed gap.  The old full-pair
    cone remains a fallback when the projection is poorly conditioned.

    This bounds matched points only.  It is not a swept collision certificate
    and must not replace ``q4_source_swept_gap_lower``.
    """
    first = np.asarray(source, dtype=float)
    second = np.asarray(target, dtype=float)
    if (first.shape != (4, 3) or second.shape not in {(3, 3), (4, 3)}
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))):
        raise ValueError("projected Q4 gap needs finite Q4/S3 or Q4/Q4 corners")
    _, projection = projected_q4_source_face(first)
    old_lower = q4_source_gap_lower(first, second)
    last = 2 if len(second) == 3 else 3
    axis = np.cross(second[1]-second[0], second[last]-second[0])
    length = float(np.linalg.norm(axis))
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("projected Q4 target has ambiguous normal")
    denominator = float(axis @ projection)
    if denominator < 0.0:
        axis = -axis
        denominator = -denominator
    cosine = denominator/length
    if cosine < 0.25:
        return old_lower
    numerator = float(np.min(first @ axis)-np.max(second @ axis))
    scale = max(1.0, float(np.max(np.abs(first))),
                float(np.max(np.abs(second))))
    projected_lower = min(0.0, numerator/denominator)
    projected_lower -= 512.0*np.finfo(float).eps*scale/cosine
    return max(old_lower, projected_lower)


def q4_source_swept_gap_lower(source_start: np.ndarray,
                              source_end: np.ndarray,
                              target_start: np.ndarray,
                              target_end: np.ndarray,
                              reference_normal: np.ndarray) -> float:
    """Conservative Q1 varying-normal gap along a linear nodal trial path."""
    a0 = np.asarray(source_start, dtype=float)
    a1 = np.asarray(source_end, dtype=float)
    b0 = np.asarray(target_start, dtype=float)
    b1 = np.asarray(target_end, dtype=float)
    n0 = _q4_corner_normals(a0)
    nm = _q4_corner_normals((a0+a1)/2.0)
    n1 = _q4_corner_normals(a1)
    controls = np.concatenate((n0, 2.0*nm-0.5*(n0+n1), n1))
    differences = np.concatenate((
        (a0[:, None, :]-b0[None, :, :]).reshape(-1, 3),
        (a1[:, None, :]-b1[None, :, :]).reshape(-1, 3),
    ))
    scale = max(1.0, *(float(np.max(np.abs(item)))
                       for item in (a0, a1, b0, b1)))
    return _normal_cone_gap_lower(
        controls, differences, np.asarray(reference_normal, dtype=float), scale,
    )


def q4_source_projected_swept_gap_lower(
        source_start: np.ndarray, source_end: np.ndarray,
        target_start: np.ndarray, target_end: np.ndarray,
        reference_normal: np.ndarray) -> float:
    """Bound matching-projection Q4 gaps along a linear nodal trial path.

    Corner-pair differences are linear in path time; the source projection
    normal and each physical corner normal are quadratic.  Their quadratic
    Bernstein controls provide an interval-wide normal cone.  The projected
    bound is used only when both normal families stay within 41 degrees of a
    common axis, which makes their mutual dot product positive.  Otherwise
    the existing full-pair swept cone is returned.
    """
    a0, a1 = (np.asarray(item, dtype=float)
              for item in (source_start, source_end))
    b0, b1 = (np.asarray(item, dtype=float)
              for item in (target_start, target_end))
    if (a0.shape != (4, 3) or a1.shape != (4, 3)
            or b0.shape not in {(3, 3), (4, 3)} or b1.shape != b0.shape
            or not all(np.all(np.isfinite(item)) for item in (a0, a1, b0, b1))):
        raise ValueError("projected swept Q4 gap needs finite matching Q4/S3 or Q4/Q4 paths")
    old_lower = q4_source_swept_gap_lower(
        a0, a1, b0, b1, reference_normal,
    )
    last = 2 if len(b0) == 3 else 3
    axis = np.cross(b0[1]-b0[0], b0[last]-b0[0])
    length = float(np.linalg.norm(axis))
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("projected swept Q4 target has ambiguous initial normal")
    axis /= length
    projection_start = np.cross(a0[1]-a0[0], a0[3]-a0[0])
    if float(axis @ projection_start) < 0.0:
        axis = -axis

    middle = 0.5*(a0+a1)

    def controls(start, midpoint, end):
        return np.asarray((start,
                           2.0*midpoint-0.5*(start+end), end))

    projection_controls = controls(
        projection_start,
        np.cross(middle[1]-middle[0], middle[3]-middle[0]),
        np.cross(a1[1]-a1[0], a1[3]-a1[0]),
    )
    corner_start = _q4_corner_normals(a0)
    corner_middle = _q4_corner_normals(middle)
    corner_end = _q4_corner_normals(a1)
    corner_controls = np.stack([
        controls(corner_start[index], corner_middle[index],
                 corner_end[index]) for index in range(4)
    ])

    def cosine_lower(values):
        maximum = float(np.max(np.linalg.norm(values, axis=-1)))
        if maximum <= 0.0 or not np.isfinite(maximum):
            return -float("inf")
        return float(np.min(values @ axis))/maximum

    projection_cosine = cosine_lower(projection_controls)
    if (projection_cosine < 0.75
            or any(cosine_lower(item) < 0.75 for item in corner_controls)):
        return old_lower
    first_difference = a0[:, None, :]-b0[None, :, :]
    last_difference = a1[:, None, :]-b1[None, :, :]
    numerator = min(float(np.min(first_difference @ axis)),
                    float(np.min(last_difference @ axis)))
    scale = max(1.0, *(float(np.max(np.abs(item)))
                       for item in (a0, a1, b0, b1)))
    projected_lower = min(0.0, numerator/projection_cosine)
    projected_lower -= 512.0*np.finfo(float).eps*scale/projection_cosine
    return max(old_lower, projected_lower)


def q4_source_overlap_integrals(
        reference_source: np.ndarray, current_source: np.ndarray,
        current_target: np.ndarray, *, area_tolerance: float,
        weighted_gap_tolerance: float, maximum_triangles: int,
        ) -> Q4SourceOverlapIntegrals:
    reference = PlanarFace((1, 1), reference_source)
    if len(reference.corners) != 4:
        raise ValueError("Q4 source reference needs four planar corners")
    source = np.asarray(current_source, dtype=float)
    target = np.asarray(current_target, dtype=float)
    if (target.shape not in {(3, 3), (4, 3)}
            or not np.all(np.isfinite(target))):
        raise ValueError("Q4 source target needs finite S3 or Q4 corners")
    if (area_tolerance is None or weighted_gap_tolerance is None
            or maximum_triangles is None
            or not np.isfinite(area_tolerance) or area_tolerance <= 0.0
            or not np.isfinite(weighted_gap_tolerance)
            or weighted_gap_tolerance <= 0.0
            or type(maximum_triangles) is not int or maximum_triangles < 1):
        raise ValueError("Q4 source integration needs positive dimensional controls")
    projected_source, normal = projected_q4_source_face(source)
    reference_normal = np.cross(reference.corners[1]-reference.corners[0],
                                reference.corners[3]-reference.corners[0])
    reference_normal /= np.linalg.norm(reference_normal)
    if float(normal@reference_normal) <= 0.0:
        raise ValueError("Q4 source polarity reversed from reference")
    target_triples = ((0, 1, 2),) if len(target) == 3 else (
        (0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1),
    )
    for indices in target_triples:
        a, b, c = target[list(indices)]
        if float(np.cross(b-a, c-a)@normal) <= 0.0:
            raise ValueError("Q4 source target is inverted or faces away")
    projected_target = target - ((target-source[0])@normal)[:, None]*normal
    target_face = PlanarFace((2, 1), projected_target)
    overlap = planar_projected_overlap(projected_source, target_face)
    support = np.zeros(4)
    weighted = np.zeros(4)
    gram = np.zeros((4, 4))
    if overlap is None:
        return Q4SourceOverlapIntegrals(
            support, weighted, gram, 0.0, 0.0, None, 0, 0.0, 0.0, 0,
        )

    source_shape_at = overlap.prepared_shape_values(projected_source)
    target_shape_at = overlap.prepared_shape_values(target_face)

    def sample(point):
        shape = source_shape_at(point)
        r = float(-shape[0]+shape[1]+shape[2]-shape[3])
        s = float(-shape[0]-shape[1]+shape[2]+shape[3])
        shape, derivatives = _q4_shape(np.array((r, s)))
        source_jacobian = derivatives @ source
        reference_jacobian = derivatives @ reference.corners
        physical_normal = np.cross(source_jacobian[0], source_jacobian[1])
        projected_jacobian = float(physical_normal@normal)
        if projected_jacobian <= 0.0:
            raise ValueError("Q4 source projection has inverted Jacobian")
        reference_area = float(np.linalg.norm(np.cross(
            reference_jacobian[0], reference_jacobian[1],
        )))
        ratio = reference_area/projected_jacobian
        physical_normal /= np.linalg.norm(physical_normal)
        target_shape = target_shape_at(point)
        gap = float(physical_normal @ (shape@source-target_shape@target))
        values = np.concatenate((
            ratio*shape, ratio*shape*gap,
            (ratio*np.outer(shape, shape)).reshape(-1),
        ))
        return values, gap

    def rule(triangle):
        area = _triangle_area(triangle)
        total = np.zeros(24)
        minimum = float("inf")
        for point, weight in _triangle_samples(tuple(triangle), area):
            value, gap = sample(point)
            total += weight*value
            minimum = min(minimum, gap)
        return total, minimum

    leaves = 0
    visited = 0
    area_error = 0.0
    weighted_error = 0.0
    minimum_sampled = min(sample(point)[1] for point in overlap.polygon)

    def integrate(triangle, area_budget, gap_budget):
        nonlocal leaves, visited, area_error, weighted_error, minimum_sampled
        visited += 1
        if visited > maximum_triangles:
            raise MemoryError("Q4 source integration triangle budget exhausted")
        coarse, coarse_min = rule(triangle)
        a, b, c = triangle
        ab, bc, ca = (a+b)/2.0, (b+c)/2.0, (c+a)/2.0
        children = (
            np.asarray((a, ab, ca)),
            np.asarray((ab, b, bc)),
            np.asarray((ca, bc, c)),
            np.asarray((ab, bc, ca)),
        )
        refined = np.zeros(24)
        child_minimum = coarse_min
        for child in children:
            value, sampled_min = rule(child)
            refined += value
            child_minimum = min(child_minimum, sampled_min)
        minimum_sampled = min(minimum_sampled, child_minimum)
        local_area_error = float(np.max(np.abs(refined[:4]-coarse[:4])))
        local_area_error = max(local_area_error, float(np.max(
            np.abs(refined[8:]-coarse[8:])
        )))
        local_gap_error = float(np.max(np.abs(refined[4:8]-coarse[4:8])))
        if local_area_error <= area_budget and local_gap_error <= gap_budget:
            leaves += 4
            if leaves > maximum_triangles:
                raise MemoryError("Q4 source integration triangle budget exhausted")
            area_error += local_area_error
            weighted_error += local_gap_error
            return refined
        if leaves+4 > maximum_triangles:
            raise MemoryError("Q4 source integration triangle budget exhausted")
        total = np.zeros(24)
        for child in children:
            total += integrate(child, area_budget/4.0, gap_budget/4.0)
        return total

    total = np.zeros(24)
    # Each fan triangle gets a finite share of the global error budget. An
    # area-proportional share can fall below roundoff for a clipping sliver,
    # although its absolute quadrature error remains far below the global
    # tolerance. Child budgets still sum to their parent's budget.
    root_share = 1.0/(len(overlap.polygon)-2)
    for index in range(1, len(overlap.polygon)-1):
        triangle = overlap.polygon[[0, index, index+1]]
        total += integrate(
            triangle, area_tolerance*root_share,
            weighted_gap_tolerance*root_share,
        )
    support[:] = total[:4]
    weighted[:] = total[4:8]
    gram[:, :] = total[8:].reshape(4, 4)
    return Q4SourceOverlapIntegrals(
        support, weighted, gram, overlap.area, float(np.sum(support)),
        minimum_sampled, len(overlap.polygon), area_error,
        weighted_error, leaves,
    )
