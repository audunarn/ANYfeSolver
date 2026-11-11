"""Project and integrate a bilinear Q4 target against a planar S3 source.

Only the projected *boundary* is polygonal. The physical target position at
every integration point uses the original Q1 shape functions and four corners.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moving import _triangle_samples
from .overlap import PlanarFace, planar_projected_overlap
from .triangle_pair_energy import _triangle_area


@dataclass(frozen=True)
class Q4TargetOverlapIntegrals:
    support: np.ndarray
    weighted_gap: np.ndarray
    pressure_gram: np.ndarray
    overlap_area: float
    minimum_sampled_gap: float | None
    overlap_vertices: int
    weighted_gap_error: float
    integration_triangles: int
    projection_residual_m: float = 0.0
    natural_coordinate_bound: float = 1.0
    absolute_reference_support: np.ndarray | None = None


def q4_target_overlap_integrals(source: PlanarFace, target_positions: np.ndarray,
                                *, reference_area_ratio: float,
                                weighted_gap_tolerance: float,
                                maximum_triangles: int,
                                exact_clipping: bool = True
                                ) -> Q4TargetOverlapIntegrals:
    if len(source.corners) != 3:
        raise ValueError("Q4 target overlap needs triangular S3 source")
    target = np.asarray(target_positions, dtype=float)
    if target.shape != (4, 3) or not np.all(np.isfinite(target)):
        raise ValueError("Q4 target needs four finite physical corners")
    if (not np.isfinite(reference_area_ratio) or reference_area_ratio <= 0.0
            or not np.isfinite(weighted_gap_tolerance)
            or weighted_gap_tolerance <= 0.0
            or type(maximum_triangles) is not int or maximum_triangles < 1):
        raise ValueError("Q4 target integration controls must be finite and positive")
    normal = np.cross(source.corners[1]-source.corners[0],
                      source.corners[2]-source.corners[0])
    normal /= np.linalg.norm(normal)
    for indices in ((0, 1, 2), (0, 2, 3)):
        a, b, c = target[list(indices)]
        if float(np.cross(b-a, c-a)@normal) <= 0.0:
            raise ValueError("bilinear Q4 target is inverted or faces away from source")
    origin = source.corners[0]
    projected = target - ((target-origin)@normal)[:, None]*normal
    projected_face = PlanarFace((2, 1), projected)
    overlap = planar_projected_overlap(source, projected_face,
                                       exact_clipping=exact_clipping)
    support = np.zeros(3)
    weighted = np.zeros(3)
    gram = np.zeros((3, 3))
    if overlap is None:
        return Q4TargetOverlapIntegrals(
            support, weighted, gram, 0.0, None, 0, 0.0, 0,
        )

    shapes = np.asarray([overlap.shape_values(source, point)
                         for point in overlap.polygon])
    for index in range(1, len(overlap.polygon)-1):
        corners = overlap.polygon[[0, index, index+1]]
        values = shapes[[0, index, index+1]]
        area = _triangle_area(corners)
        sums = np.sum(values, axis=0)
        support += area/3.0*sums
        gram += area/12.0*(np.outer(sums, sums)+values.T@values)
    support *= reference_area_ratio
    gram *= reference_area_ratio

    # Diagnostics retain actual returned inverse values, not Newton thresholds.
    # They do not alter the interpolation, integration or pressure law.
    projected_xy = np.column_stack((
        (projected_face.corners-overlap.origin)@overlap.tangent,
        (projected_face.corners-overlap.origin)@overlap.binormal,
    ))
    projection_residual = 0.0
    natural_bound = 1.0
    absolute_shape_bound = np.zeros(3)

    def sample(point):
        nonlocal projection_residual, natural_bound, absolute_shape_bound
        shape_a = overlap.shape_values(source, point)
        shape_b = overlap.shape_values(projected_face, point)
        point_xy = np.array(((point-overlap.origin)@overlap.tangent,
                             (point-overlap.origin)@overlap.binormal))
        projection_residual = max(projection_residual,
                                  float(np.linalg.norm(shape_b@projected_xy-point_xy)))
        # Q1 partition identities recover the returned natural coordinates.
        r = shape_b[1]+shape_b[2]-shape_b[0]-shape_b[3]
        s = shape_b[2]+shape_b[3]-shape_b[0]-shape_b[1]
        gamma4 = 4*np.finfo(float).eps/(1-4*np.finfo(float).eps)
        # Four operations per Q1 shape and per recovery sum. Include both
        # shape-evaluation and recovery error, using actual absolute operands.
        shape_l1_upper = np.nextafter(np.sum(np.abs(shape_b))/(1-gamma4), np.inf)
        recovery_error = (gamma4+gamma4/(1-gamma4))*shape_l1_upper
        natural_bound = max(natural_bound,
            float(np.nextafter(abs(float(r))+recovery_error, np.inf)),
            float(np.nextafter(abs(float(s))+recovery_error, np.inf)))
        absolute_shape_bound = np.maximum(absolute_shape_bound, np.abs(shape_a))
        gap = float((point-shape_b@target)@normal)
        return shape_a*gap, gap

    def rule(triangle, inherited_area=None):
        area = (_triangle_area(triangle) if inherited_area is None
                else inherited_area)
        total = np.zeros(3)
        minimum = float("inf")
        for point, weight in _triangle_samples(tuple(triangle), area):
            integrand, gap = sample(point)
            total += weight*integrand
            minimum = min(minimum, gap)
        return total, minimum

    leaves = 0
    visited = 0
    measured_error = 0.0
    minimum_sampled = min(sample(point)[1] for point in overlap.polygon)

    def integrate(triangle, budget, inherited_area=None):
        nonlocal leaves, visited, measured_error, minimum_sampled
        visited += 1
        if visited > maximum_triangles:
            raise MemoryError("Q4 target contact integration triangle budget exhausted")
        # Retain the represented parent's area on the zero-tolerance path.
        # A midpoint child has precisely one quarter of that area; computing
        # another cross product can lose a real, extremely thin seam sliver.
        if exact_clipping and inherited_area is None:
            inherited_area = _triangle_area(triangle)
        coarse, coarse_min = rule(triangle, inherited_area)
        child_area = None if inherited_area is None else inherited_area/4.0
        a, b, c = triangle
        ab, bc, ca = (a+b)/2.0, (b+c)/2.0, (c+a)/2.0
        children = (
            np.asarray((a, ab, ca)),
            np.asarray((ab, b, bc)),
            np.asarray((ca, bc, c)),
            np.asarray((ab, bc, ca)),
        )
        refined = np.zeros(3)
        child_minimum = coarse_min
        for child in children:
            value, sampled_min = rule(child, child_area)
            refined += value
            child_minimum = min(child_minimum, sampled_min)
        minimum_sampled = min(minimum_sampled, child_minimum)
        error = reference_area_ratio*float(np.max(np.abs(refined-coarse)))
        if error <= budget:
            leaves += 4
            if leaves > maximum_triangles:
                raise MemoryError("Q4 target contact integration triangle budget exhausted")
            measured_error += error
            return refined
        if leaves+4 > maximum_triangles:
            raise MemoryError("Q4 target contact integration triangle budget exhausted")
        values = np.zeros(3)
        for child in children:
            values += integrate(child, budget/4.0, child_area)
        return values

    for index in range(1, len(overlap.polygon)-1):
        triangle = overlap.polygon[[0, index, index+1]]
        budget = weighted_gap_tolerance*_triangle_area(triangle)/overlap.area
        weighted += integrate(triangle, budget)
    weighted *= reference_area_ratio
    return Q4TargetOverlapIntegrals(
        support, weighted, gram, overlap.area, minimum_sampled,
        len(overlap.polygon), measured_error, leaves,
        projection_residual, float(np.nextafter(natural_bound, np.inf)),
        reference_area_ratio*overlap.area*absolute_shape_bound,
    )
