"""Quadratic physical-gap extremum for planar S3 and parallelogram Q4 faces.

The Q4 height field is bilinear in its affine material coordinates. Those
coordinates are affine in the common projected plane only when the projected
Q4 is an exact parallelogram. The resulting difference of two height fields
is quadratic, so its minimum on a convex polygon occurs at a vertex, an edge
stationary point, or an interior stationary point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .overlap import PlanarFace, ProjectedOverlap, _cross2, _signed_area, _to_frame


@dataclass(frozen=True)
class QuadraticGapExtremum:
    minimum: float
    guarded_lower: float
    candidate_count: int


def _height_polynomial(face: PlanarFace, heights: np.ndarray,
                       overlap: ProjectedOverlap) -> tuple[np.ndarray, float] | None:
    xy = _to_frame(face.corners, overlap.origin,
                   overlap.tangent, overlap.binormal)
    if len(xy) == 4 and not np.array_equal(xy[0] + xy[2], xy[1] + xy[3]):
        return None
    second_edge = 2 if len(xy) == 3 else 3
    edge_matrix = np.column_stack((xy[1] - xy[0],
                                   xy[second_edge] - xy[0]))
    condition = float(np.linalg.cond(edge_matrix))
    if not np.isfinite(condition) or condition > 1.0e8:
        return None
    inverse = np.linalg.inv(edge_matrix)
    alpha = np.r_[-float(inverse[0] @ xy[0]), inverse[0]]
    beta = np.r_[-float(inverse[1] @ xy[0]), inverse[1]]
    first = float(heights[1] - heights[0])
    last = float(heights[second_edge] - heights[0])
    product = (float(heights[0] - heights[1] + heights[2] - heights[3])
               if len(xy) == 4 else 0.0)
    coefficients = np.array((
        heights[0] + first * alpha[0] + last * beta[0]
        + product * alpha[0] * beta[0],
        first * alpha[1] + last * beta[1]
        + product * (alpha[0] * beta[1] + beta[0] * alpha[1]),
        first * alpha[2] + last * beta[2]
        + product * (alpha[0] * beta[2] + beta[0] * alpha[2]),
        product * alpha[1] * beta[1],
        product * (alpha[1] * beta[2] + alpha[2] * beta[1]),
        product * alpha[2] * beta[2],
    ), dtype=float)
    return coefficients, condition


def quadratic_gap_extremum(
        source: PlanarFace, target: PlanarFace,
        overlap: ProjectedOverlap, normal: np.ndarray,
        source_heights: np.ndarray, target_heights: np.ndarray,
        ) -> QuadraticGapExtremum | None:
    """Return an exact-polynomial minimum, or None for a nonaffine Q4 map."""

    source_field = _height_polynomial(
        source, source_heights + source.corners @ normal, overlap,
    )
    target_field = _height_polynomial(
        target, target_heights + target.corners @ normal, overlap,
    )
    if source_field is None or target_field is None:
        return None
    source_coefficients, source_condition = source_field
    target_coefficients, target_condition = target_field
    coefficients = source_coefficients - target_coefficients
    polygon = _to_frame(overlap.polygon, overlap.origin,
                        overlap.tangent, overlap.binormal)

    def value(point: np.ndarray) -> float:
        x, y = map(float, point)
        c0, cx, cy, cxx, cxy, cyy = coefficients
        return float(c0 + cx * x + cy * y + cxx * x * x
                     + cxy * x * y + cyy * y * y)

    candidates = [value(point) for point in polygon]
    c0, cx, cy, cxx, cxy, cyy = coefficients
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = end - start
        curvature = float(cxx * dx * dx + cxy * dx * dy + cyy * dy * dy)
        slope = float(cx * dx + cy * dy + 2.0 * cxx * start[0] * dx
                      + cxy * (start[0] * dy + start[1] * dx)
                      + 2.0 * cyy * start[1] * dy)
        if curvature > 0.0:
            location = -slope / (2.0 * curvature)
            if 0.0 < location < 1.0:
                candidates.append(value(start + location * (end - start)))
    hessian = np.array(((2.0 * cxx, cxy), (cxy, 2.0 * cyy)))
    if abs(float(np.linalg.det(hessian))) > np.finfo(float).eps * (
            float(np.linalg.norm(hessian)) ** 2):
        stationary = np.linalg.solve(hessian, -np.array((cx, cy)))
        orientation = np.sign(_signed_area(polygon))
        span = max(float(np.max(np.linalg.norm(polygon - polygon[0], axis=1))),
                   np.finfo(float).tiny)
        edge_tolerance = 128.0 * np.finfo(float).eps * span * span
        if all(orientation * _cross2(end - start, stationary - start)
               >= -edge_tolerance
               for start, end in zip(polygon, np.roll(polygon, -1, axis=0))):
            candidates.append(value(stationary))
    minimum = float(min(candidates))
    extent = max(1.0, float(np.max(np.abs(polygon))))
    scale = (abs(c0) + (abs(cx) + abs(cy)) * extent
             + (abs(cxx) + abs(cxy) + abs(cyy)) * extent * extent)
    condition = max(source_condition, target_condition)
    roundoff_guard = float(1024.0 * np.finfo(float).eps
                           * condition * max(scale, abs(minimum), np.finfo(float).tiny))
    return QuadraticGapExtremum(
        minimum, float(minimum - roundoff_guard), len(candidates),
    )
