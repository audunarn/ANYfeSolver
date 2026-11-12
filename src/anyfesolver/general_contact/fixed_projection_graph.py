"""Conservative swept gap check for a fixed-projection Q4/planar-target pair.

This is the geometry check for an opt-in restricted static profile. It is not
the general moving-overlap or finite-sliding certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedProjectionSweepCertificate:
    admissible: bool
    vertical_lower_m: float
    normal_cosine_lower: float
    expanded_overlap_vertices: int
    projection_drift_m: float


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0]*second[1]-first[1]*second[0])


def _clip_expanded_target(source_rectangle: np.ndarray,
                          target: np.ndarray, expansion: float
                          ) -> tuple[np.ndarray, ...]:
    polygon = [point.copy() for point in source_rectangle]
    for first, second in zip(target, np.roll(target, -1, axis=0)):
        edge = second-first
        length = float(np.linalg.norm(edge))
        if length <= 0.:
            raise ValueError("fixed-projection target has a zero edge")

        def signed(point):
            return _cross2(edge, point-first)/length+expansion

        output = []
        previous = polygon[-1] if polygon else None
        for current in polygon:
            a, b = signed(previous), signed(current)
            if (a >= 0.) != (b >= 0.):
                fraction = a/(a-b)
                output.append(previous+fraction*(current-previous))
            if b >= 0.:
                output.append(current)
            previous = current
        polygon = output
        if not polygon:
            break
    return tuple(polygon)


def _bilinear_height(point: np.ndarray, corners: np.ndarray,
                     bounds: tuple[float, float, float, float]) -> float:
    xlo, xhi, ylo, yhi = bounds
    u = float((point[0]-xlo)/(xhi-xlo))
    v = float((point[1]-ylo)/(yhi-ylo))
    return float((1.-u)*(1.-v)*corners[0]
                 + u*(1.-v)*corners[1]
                 + u*v*corners[2]
                 + (1.-u)*v*corners[3])


def _polygon_minimum_height(polygon: tuple[np.ndarray, ...],
                            corners: np.ndarray,
                            bounds: tuple[float, float, float, float]) -> float:
    """A bilinear graph has no strict interior minimum; edges are quadratic."""
    minimum = float("inf")
    for first, second in zip(polygon, polygon[1:]+polygon[:1]):
        start = _bilinear_height(first, corners, bounds)
        end = _bilinear_height(second, corners, bounds)
        middle = _bilinear_height((first+second)/2., corners, bounds)
        minimum = min(minimum, start, end)
        quadratic = 2.*(start+end-2.*middle)
        linear = end-start-quadratic
        if quadratic > 0.:
            root = -linear/(2.*quadratic)
            if 0. < root < 1.:
                minimum = min(minimum, (quadratic*root+linear)*root+start)
    return minimum


def fixed_projection_graph_swept_certificate(
        source_start: np.ndarray, source_end: np.ndarray,
        target_start: np.ndarray, target_end: np.ndarray,
        gap_tolerance_m: float) -> FixedProjectionSweepCertificate:
    """Exclude a first `-gap_tolerance_m` crossing on a linear nodal path.

    The source must be a Q4 graph over one fixed projected rectangle. The
    planar S3/Q4 target may translate only along its fixed normal. At a first
    threshold crossing, the source point's in-plane projection is within
    ``tau*max|normal_tangent|`` of the target polygon. We enlarge each target
    halfspace by that distance. Source height minus target height is linear
    in path time at each fixed Q4 coordinate; its spatial minimum on this
    convex polygon lies on an edge and is found from the edge quadratic.
    A lower vertical height above ``-tau*min(normal_cosine)`` excludes the
    crossing. Unsupported geometry raises rather than returning safety.
    """
    first, last, target_first, target_last = (
        np.asarray(item, dtype=float) for item in
        (source_start, source_end, target_start, target_end)
    )
    if (first.shape != (4, 3) or last.shape != (4, 3)
            or target_first.shape not in {(3, 3), (4, 3)}
            or target_last.shape != target_first.shape
            or not all(np.all(np.isfinite(item)) for item in
                       (first, last, target_first, target_last))):
        raise ValueError("fixed-projection sweep needs finite Q4/S3 or Q4/Q4 poses")
    tau = float(gap_tolerance_m)
    if not np.isfinite(tau) or tau <= 0.:
        raise ValueError("fixed-projection sweep needs a positive gap tolerance")

    origin = target_first[0]
    normal = np.cross(target_first[1]-origin, target_first[-1]-origin)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 0.:
        raise ValueError("fixed-projection target has ambiguous normal")
    normal /= normal_length
    tangent = first[1]-first[0]
    tangent -= float(tangent @ normal)*normal
    tangent_length = float(np.linalg.norm(tangent))
    if tangent_length <= 0.:
        raise ValueError("fixed-projection source has ambiguous tangent")
    tangent /= tangent_length
    binormal = np.cross(normal, tangent)

    def frame(points):
        relative = points-origin
        return (np.column_stack((relative@tangent, relative@binormal)),
                relative@normal)

    source_xy, source_height = frame(first)
    source_end_xy, source_end_height = frame(last)
    target_xy, target_height = frame(target_first)
    target_end_xy, target_end_height = frame(target_last)
    scale = max(1., *(float(np.max(np.linalg.norm(item-origin, axis=1)))
                      for item in (first, last, target_first, target_last)))
    numerical = 256.*np.finfo(float).eps*scale
    drift_source = float(np.max(np.linalg.norm(source_end_xy-source_xy, axis=1)))
    drift_target = float(np.max(np.linalg.norm(target_end_xy-target_xy, axis=1)))
    drift = max(drift_source, drift_target)
    if drift > numerical:
        raise ValueError("fixed-projection sweep forbids in-plane motion")
    if (float(np.ptp(target_height)) > numerical
            or float(np.ptp(target_end_height)) > numerical):
        raise ValueError("fixed-projection sweep needs a planar parallel target")

    xlo, xhi = float(source_xy[0, 0]), float(source_xy[1, 0])
    ylo, yhi = float(source_xy[0, 1]), float(source_xy[3, 1])
    if (xhi-xlo <= numerical or yhi-ylo <= numerical
            or abs(float(source_xy[3, 0])-xlo) > numerical
            or abs(float(source_xy[2, 0])-xhi) > numerical
            or abs(float(source_xy[1, 1])-ylo) > numerical
            or abs(float(source_xy[2, 1])-yhi) > numerical):
        raise ValueError("fixed-projection source needs a positive rectangle")
    bounds = (xlo, xhi, ylo, yhi)
    signed_area = .5*sum(_cross2(a, b) for a, b in
                           zip(target_xy, np.roll(target_xy, -1, axis=0)))
    if signed_area <= numerical*numerical:
        raise ValueError("fixed-projection target is inverted or degenerate")
    if any(_cross2(target_xy[(i+1) % len(target_xy)]-target_xy[i],
                   target_xy[(i+2) % len(target_xy)]-target_xy[(i+1) % len(target_xy)])
           <= numerical*numerical for i in range(len(target_xy))):
        raise ValueError("fixed-projection target must be convex")

    dx, dy = xhi-xlo, yhi-ylo
    slope_x = []
    slope_y = []
    for heights in (source_height, source_end_height):
        slope_x.extend(((heights[1]-heights[0])/dx,
                        (heights[2]-heights[3])/dx))
        slope_y.extend(((heights[3]-heights[0])/dy,
                        (heights[2]-heights[1])/dy))
    gradient_bound = float(np.hypot(max(map(abs, slope_x)),
                                    max(map(abs, slope_y))))
    gradient_bound += 4.*numerical/min(dx, dy)*(1.+gradient_bound)
    cosine_lower = float(1./np.sqrt(1.+gradient_bound**2))
    if cosine_lower < .25:
        raise ValueError("fixed-projection source normal is nearly grazing")
    expansion = (tau*gradient_bound/np.sqrt(1.+gradient_bound**2)
                 + drift_source+drift_target+4.*numerical)
    rectangle = np.asarray(((xlo, ylo), (xhi, ylo),
                            (xhi, yhi), (xlo, yhi)))
    polygon = _clip_expanded_target(rectangle, target_xy, expansion)
    if not polygon:
        return FixedProjectionSweepCertificate(True, float("inf"),
                                                cosine_lower, 0, drift)

    minimum = min(
        _polygon_minimum_height(polygon, source_height, bounds)
        - float(target_height[0]),
        _polygon_minimum_height(polygon, source_end_height, bounds)
        - float(target_end_height[0]),
    )
    minimum -= (1024.*np.finfo(float).eps*scale
                + max(float(np.ptp(target_height)),
                      float(np.ptp(target_end_height)))
                + gradient_bound*drift_source)
    return FixedProjectionSweepCertificate(
        bool(minimum > -tau*cosine_lower), float(minimum), cosine_lower,
        len(polygon), drift,
    )
