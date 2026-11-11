"""Moving planar mortar control with fixed rotations and nodal normal fields.

The public restricted-sliding development profile uses this candidate-pose
overlap, transported primal pressure field, and bounded trial-path geometry.
It does not represent arbitrary in-plane shell deformation or a complete C1
finite-sliding formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gap_extrema import quadratic_gap_extremum
from .overlap import (
    PlanarFace,
    _cross2,
    _q4_shape,
    _signed_area,
    _to_frame,
    planar_overlap_translation_derivatives,
    planar_projected_overlap,
    swept_planar_overlap_interval,
)


@dataclass(frozen=True)
class TranslatedPairResult:
    overlap_area: float
    gap: float
    energy: float
    source_residual: np.ndarray
    target_residual: np.ndarray
    relative_tangent: np.ndarray

    @property
    def tangent(self) -> np.ndarray:
        block = self.relative_tangent
        return np.block([[block, -block], [-block, block]])


class TranslatedPairMortar:
    """Exact smooth-branch penalty derivatives for two translating patches."""

    def __init__(self, source_faces: tuple[PlanarFace, ...],
                 target_faces: tuple[PlanarFace, ...], normal: np.ndarray,
                 stiffness: float, *, maximum_candidate_pairs: int = 100_000):
        self.source_faces = tuple(source_faces)
        self.target_faces = tuple(target_faces)
        if not self.source_faces or not self.target_faces:
            raise ValueError("translated pair needs two nonempty face patches")
        if (len(self.source_faces) * len(self.target_faces)
                > maximum_candidate_pairs):
            raise MemoryError("contact candidate limit exceeded; no pair was discarded")
        direction = np.asarray(normal, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("contact normal must be a finite 3-vector")
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise ValueError("contact normal must be nonzero")
        self.normal = direction / length
        self.stiffness = float(stiffness)
        if not np.isfinite(self.stiffness) or self.stiffness <= 0.0:
            raise ValueError("normal stiffness must be finite and positive")
        for name, faces in (("source", self.source_faces),
                            ("target", self.target_faces)):
            first = float(faces[0].corners[0] @ self.normal)
            scale = max(float(np.max(np.linalg.norm(face.corners - face.corners[0], axis=1)))
                        for face in faces)
            if max(float(np.max(np.abs(face.corners @ self.normal - first)))
                   for face in faces) > 1.0e-10 * scale:
                raise ValueError(f"translated {name} patch must occupy one plane")
        self.reference_gap = float(
            (self.source_faces[0].corners[0] - self.target_faces[0].corners[0])
            @ self.normal
        )

    def evaluate(self, source_translation: np.ndarray,
                 target_translation: np.ndarray) -> TranslatedPairResult:
        source = np.asarray(source_translation, dtype=float)
        target = np.asarray(target_translation, dtype=float)
        if (source.shape != (3,) or target.shape != (3,)
                or not np.all(np.isfinite(source))
                or not np.all(np.isfinite(target))):
            raise ValueError("pair translations must be finite 3-vectors")
        area = 0.0
        area_gradient = np.zeros(3, dtype=float)
        area_hessian = np.zeros((3, 3), dtype=float)
        for source_face in self.source_faces:
            moved_source = PlanarFace(source_face.feature_id,
                                      source_face.corners + source)
            for target_face in self.target_faces:
                moved_target = PlanarFace(target_face.feature_id,
                                          target_face.corners + target)
                overlap = planar_projected_overlap(moved_source, moved_target)
                if overlap is None:
                    continue
                area += overlap.area
                gradient, hessian = planar_overlap_translation_derivatives(
                    moved_source, moved_target
                )
                area_gradient += gradient
                area_hessian += hessian
        gap = self.reference_gap + float((source - target) @ self.normal)
        compression_gap = min(gap, 0.0)
        energy = 0.5 * self.stiffness * area * compression_gap ** 2
        relative_residual = self.stiffness * (
            0.5 * compression_gap ** 2 * area_gradient
            + area * compression_gap * self.normal
        )
        relative_tangent = np.zeros((3, 3), dtype=float)
        if gap <= 0.0 and area > 0.0:
            relative_tangent = self.stiffness * (
                0.5 * compression_gap ** 2 * area_hessian
                + compression_gap * (
                    np.outer(area_gradient, self.normal)
                    + np.outer(self.normal, area_gradient)
                )
                + area * np.outer(self.normal, self.normal)
            )
        return TranslatedPairResult(
            area, gap, energy, relative_residual, -relative_residual,
            relative_tangent,
        )


@dataclass(frozen=True)
class MovingMortarState:
    overlap_area: float
    support: np.ndarray
    weighted_gap: np.ndarray
    normalized_gap: np.ndarray
    energy: float
    normal_residual: np.ndarray
    normal_tangent: np.ndarray
    minimum_vertex_gap: float


@dataclass(frozen=True)
class MovingPairIntegrals:
    overlap_area: float
    support: np.ndarray
    weighted_gap: np.ndarray
    gap_gradient: np.ndarray
    minimum_vertex_gap: float


@dataclass(frozen=True)
class MovingGapCertificate:
    minimum_sampled_gap: float
    conservative_lower_gap: float
    tolerance_met: bool
    triangles_checked: int
    method: str = "subdivision"


@dataclass(frozen=True)
class MovingSweptGapBound:
    overlap_interval: tuple[float, float] | None
    conservative_lower_gap: float
    tolerance_certified: bool


def bound_swept_moving_mortar_gap(
        source_face: PlanarFace, target_face: PlanarFace,
        normal: np.ndarray,
        source_start: np.ndarray, source_end: np.ndarray,
        target_start: np.ndarray, target_end: np.ndarray,
        source_heights_start: np.ndarray, source_heights_end: np.ndarray,
        target_heights_start: np.ndarray, target_heights_end: np.ndarray,
        *, gap_tolerance: float) -> MovingSweptGapBound:
    """Certify a linear trial path against missed projected overlap.

    The normal position difference of every source/target nodal pair is affine
    in time. Its outward-rounded minimum over the complete [0,1] path bounds
    every convexly interpolated physical gap. The separating-axis entry/exit
    times are diagnostics; they cannot certify absence of overlap. Only an
    outward swept-projection separation proof certifies an empty path. A failed
    bound means unresolved, not confirmed penetration.
    """

    tolerance = float(gap_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("gap_tolerance must be finite and positive")
    direction = np.asarray(normal, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("contact normal must be a finite 3-vector")
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 0.0:
        raise ValueError("contact normal must be nonzero")
    direction = direction / magnitude
    translations = tuple(np.asarray(value, dtype=float) for value in (
        source_start, source_end, target_start, target_end,
    ))
    heights = tuple(np.asarray(value, dtype=float) for value in (
        source_heights_start, source_heights_end,
        target_heights_start, target_heights_end,
    ))
    if (any(value.shape != (3,) or not np.all(np.isfinite(value))
            for value in translations)
            or any(value.shape != (size,) or not np.all(np.isfinite(value))
                   for value, size in zip(heights, (
                       len(source_face.corners), len(source_face.corners),
                       len(target_face.corners), len(target_face.corners),
                   )))):
        raise ValueError("swept moving mortar fields have invalid size or nonfinite values")
    for face in (source_face, target_face):
        face_normal = np.cross(face.corners[1] - face.corners[0],
                               face.corners[-1] - face.corners[0])
        face_normal /= np.linalg.norm(face_normal)
        if abs(float(face_normal @ direction)) < 1.0 - 1.0e-10:
            raise ValueError("swept moving mortar face must be parallel to contact plane")
    interval = swept_planar_overlap_interval(
        source_face, target_face, *translations,
    )
    if _sweep_faces_disjoint(source_face, target_face, translations):
        return MovingSweptGapBound(None, float("inf"), True)

    origin = tuple(map(_sweep_point, source_face.corners[0]))
    axis = tuple(map(_sweep_point, direction))
    bounds = []
    for endpoint in (0, 1):
        positions = []
        for body, face in enumerate((source_face, target_face)):
            shift = _sweep_dot(tuple(map(_sweep_point,
                                        translations[2*body+endpoint])), axis)
            values = []
            for corner, height in zip(face.corners, heights[2*body+endpoint]):
                point = tuple(_sweep_sub(_sweep_point(x), y)
                              for x, y in zip(corner, origin))
                values.append(_sweep_add(_sweep_add(_sweep_dot(point, axis),
                    _sweep_point(height)), shift))
            positions.append(values)
        bounds.append(_sweep_sub(
            (min(item[0] for item in positions[0]),
             min(item[1] for item in positions[0])),
            (max(item[0] for item in positions[1]),
             max(item[1] for item in positions[1])))[0])
    lower = min(bounds)
    return MovingSweptGapBound(interval, lower,
                               bool(np.isfinite(lower) and lower >= -tolerance))


def _sweep_add(a, b):
    lo, hi = a[0] + b[0], a[1] + b[1]
    if np.isnan(lo) or np.isnan(hi):
        return -float("inf"), float("inf")
    return (float(np.nextafter(lo, -np.inf)),
            float(np.nextafter(hi, np.inf)))


def _sweep_sub(a, b):
    return _sweep_add(a, (-b[1], -b[0]))


def _sweep_mul(a, b):
    products = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    if any(np.isnan(value) for value in products):
        return -float("inf"), float("inf")
    return (float(np.nextafter(min(products), -np.inf)),
            float(np.nextafter(max(products), np.inf)))


def _sweep_dot(a, b):
    result = (0.0, 0.0)
    for first, second in zip(a, b):
        result = _sweep_add(result, _sweep_mul(first, second))
    return result


def _sweep_point(value):
    return float(value), float(value)


def _sweep_field(start, end, time):
    return _sweep_add(start, _sweep_mul(time, _sweep_sub(end, start)))


def _sweep_faces_disjoint(source, target, translations):
    """Fixed projected axes, all corners and both endpoints: empty-path proof."""
    origin = source.corners[0]
    tangent = source.corners[1] - origin
    tangent /= np.linalg.norm(tangent)
    normal = np.cross(source.corners[1] - origin, source.corners[-1] - origin)
    normal /= np.linalg.norm(normal)
    axes = (tangent, np.cross(normal, tangent))
    if not np.all(np.isfinite(axes)):
        return False
    origin_i = tuple(map(_sweep_point, origin))
    for axis in axes:
        direction = tuple(map(_sweep_point, axis))
        extents = []
        for body, face in enumerate((source, target)):
            points = []
            for endpoint in (0, 1):
                shift = _sweep_dot(tuple(map(_sweep_point,
                    translations[2*body+endpoint])), direction)
                for corner in face.corners:
                    point = tuple(_sweep_sub(_sweep_point(x), y)
                                  for x, y in zip(corner, origin_i))
                    points.append(_sweep_add(_sweep_dot(point, direction), shift))
            extents.append((min(p[0] for p in points), max(p[1] for p in points)))
        first, second = extents
        if first[1] < second[0] or second[1] < first[0]:
            return True
    return False


def certify_swept_moving_mortar_gap(
        source_face: PlanarFace, target_face: PlanarFace, normal: np.ndarray,
        source_start: np.ndarray, source_end: np.ndarray,
        target_start: np.ndarray, target_end: np.ndarray,
        source_heights_start: np.ndarray, source_heights_end: np.ndarray,
        target_heights_start: np.ndarray, target_heights_end: np.ndarray,
        *, gap_tolerance: float, maximum_boxes: int) -> MovingSweptGapBound:
    """Enclose the entire linear path on its changing projected overlap.

    Subdivision boxes cover projected position and path fraction together.
    Material-coordinate interval constraints prune only provably empty boxes;
    all remaining gaps are bounded using the original S3/Q4 interpolation.
    Neither endpoint checks nor sampled positions certify a time interval.

    For Q4, x=c0+A(u,v)+d*u*v. Thus A^-1(x-c0-translation)
    encloses (u,v)+(A^-1*d)*u*v. With u,v in [0,1], interval
    subtraction encloses the true inverse even for non-affine quadrilaterals.
    A wide enclosure or exhausted budget is explicit non-certification.
    All scalar interval operations round outwards; no gap tolerance is added.
    """
    if type(maximum_boxes) is not int or maximum_boxes < 1:
        raise ValueError("maximum_boxes must be a positive integer")
    coarse = bound_swept_moving_mortar_gap(
        source_face, target_face, normal, source_start, source_end,
        target_start, target_end, source_heights_start, source_heights_end,
        target_heights_start, target_heights_end, gap_tolerance=gap_tolerance,
    )
    if coarse.tolerance_certified:
        return coarse
    direction = np.asarray(normal, dtype=float)
    direction = direction / np.linalg.norm(direction)
    origin = source_face.corners[0]
    tangent = source_face.corners[1] - origin
    tangent /= np.linalg.norm(tangent)
    face_normal = np.cross(source_face.corners[1] - origin,
                           source_face.corners[-1] - origin)
    face_normal /= np.linalg.norm(face_normal)
    binormal = np.cross(face_normal, tangent)
    if not np.all(np.isfinite((direction, tangent, binormal))):
        return coarse
    axes = tuple(tuple(map(_sweep_point, axis))
                 for axis in (tangent, binormal, direction))
    origin_i = tuple(map(_sweep_point, origin))

    def prepare(face, start, end, h_start, h_end):
        corners = [tuple(_sweep_sub(_sweep_point(x), y)
                         for x, y in zip(point, origin_i))
                   for point in face.corners]
        xy = [tuple(_sweep_dot(point, axis) for axis in axes[:2])
              for point in corners]
        columns = tuple(tuple(_sweep_sub(xy[index][j], xy[0][j])
                              for j in range(2))
                        for index in (1, len(corners)-1))
        a, c = columns[0]
        b, d = columns[1]
        determinant = _sweep_sub(_sweep_mul(a, d), _sweep_mul(b, c))
        if (not np.all(np.isfinite(determinant))
                or determinant[0] <= 0.0 <= determinant[1]):
            return None
        reciprocal = (float(np.nextafter(1.0/determinant[1], -np.inf)),
                      float(np.nextafter(1.0/determinant[0], np.inf)))
        inverse = ((_sweep_mul(d, reciprocal),
                    _sweep_mul((-b[1], -b[0]), reciprocal)),
                   (_sweep_mul((-c[1], -c[0]), reciprocal),
                    _sweep_mul(a, reciprocal)))
        defect = ((0.0, 0.0), (0.0, 0.0))
        if len(corners) == 4:
            defect = tuple(_sweep_add(_sweep_sub(xy[2][j], xy[1][j]),
                                     _sweep_sub(xy[0][j], xy[3][j]))
                           for j in range(2))
        correction = tuple(_sweep_dot(row, defect) for row in inverse)
        translations = [tuple(_sweep_dot(tuple(map(_sweep_point, value)), axis)
                              for axis in axes) for value in (start, end)]
        heights = []
        for field in (h_start, h_end):
            heights.append([_sweep_add(_sweep_dot(point, axes[2]),
                                      _sweep_point(height))
                            for point, height in zip(corners, field)])
        coefficients = []
        for values in heights:
            coefficients.append((values[0], _sweep_sub(values[1], values[0]),
                _sweep_sub(values[-1], values[0]),
                (_sweep_add(_sweep_sub(values[2], values[1]),
                            _sweep_sub(values[0], values[3]))
                 if len(corners) == 4 else (0.0, 0.0))))
        return xy, inverse, correction, translations, coefficients

    prepared = [prepare(*values) for values in (
        (source_face, source_start, source_end,
         source_heights_start, source_heights_end),
        (target_face, target_start, target_end,
         target_heights_start, target_heights_end))]
    if any(item is None for item in prepared):
        return coarse
    # The intersection of the two swept AABBs contains every physical overlap
    # point, including contact born and lost strictly inside the path.
    # Cover all of [0,1]; the reported coarse overlap interval is diagnostic.
    # Refinement does not depend on rounded entry/exit times for coverage.
    time = (0.0, 1.0)
    extents = []
    for xy, _, _, translations, _ in prepared:
        shift = tuple(_sweep_field(translations[0][j], translations[1][j], time)
                      for j in range(2))
        extents.append(tuple((min(_sweep_add(p[j], shift[j])[0] for p in xy),
                              max(_sweep_add(p[j], shift[j])[1] for p in xy))
                             for j in range(2)))
    box = tuple((max(first[0], second[0]), min(first[1], second[1]))
                for first, second in zip(*extents)) + (time,)
    if not np.all(np.isfinite(box)):
        return coarse
    if any(lo > hi for lo, hi in box):
        return MovingSweptGapBound(coarse.overlap_interval, float("inf"), True)

    def height(data, cell):
        xy, inverse, correction, translations, coefficients = data
        time = cell[2]
        point = tuple(_sweep_sub(_sweep_sub(cell[j], xy[0][j]),
            _sweep_field(translations[0][j], translations[1][j], time))
            for j in range(2))
        affine = tuple(_sweep_dot(row, point) for row in inverse)
        uv = (0.0, 1.0)
        for _ in range(2):
            natural = tuple(_sweep_sub(value, _sweep_mul(error, uv))
                            for value, error in zip(affine, correction))
            natural = tuple((max(0.0, lo), min(1.0, hi))
                            for lo, hi in natural)
            if any(lo > hi for lo, hi in natural):
                return None
            if len(xy) == 3 and _sweep_add(*natural)[0] > 1.0:
                return None
            uv = _sweep_mul(*natural)
        values = tuple(_sweep_field(a, b, time)
                       for a, b in zip(*coefficients))
        result = _sweep_add(values[0], _sweep_field(
            translations[0][2], translations[1][2], time))
        for coefficient, coordinate in zip(values[1:], (*natural, uv)):
            result = _sweep_add(result, _sweep_mul(coefficient, coordinate))
        return result

    pending = [box]
    widths = tuple(hi-lo for lo, hi in box)
    lower = float("inf")
    checked = 0
    while pending:
        if checked >= maximum_boxes:
            return coarse
        cell = pending.pop()
        checked += 1
        heights = tuple(height(data, cell) for data in prepared)
        if any(value is None for value in heights):
            continue
        bound = _sweep_sub(*heights)[0]
        if np.isfinite(bound) and bound >= -gap_tolerance:
            lower = min(lower, bound)
            continue
        axis = int(np.argmax([(hi-lo)/width if width > 0.0 else 0.0
                              for (lo, hi), width in zip(cell, widths)]))
        lo, hi = cell[axis]
        midpoint = lo + 0.5*(hi-lo)
        if not lo < midpoint < hi:
            return coarse
        for interval in ((lo, midpoint), (midpoint, hi)):
            child = list(cell)
            child[axis] = interval
            pending.append(tuple(child))
    return MovingSweptGapBound(coarse.overlap_interval, lower, True)


_TRIANGLE_RULE = (
    ((1.0/3.0, 1.0/3.0, 1.0/3.0), 0.225),
    ((0.059715871789770, 0.470142064105115, 0.470142064105115),
     0.132394152788506),
    ((0.470142064105115, 0.059715871789770, 0.470142064105115),
     0.132394152788506),
    ((0.470142064105115, 0.470142064105115, 0.059715871789770),
     0.132394152788506),
    ((0.797426985353087, 0.101286507323456, 0.101286507323456),
     0.125939180544827),
    ((0.101286507323456, 0.797426985353087, 0.101286507323456),
     0.125939180544827),
    ((0.101286507323456, 0.101286507323456, 0.797426985353087),
     0.125939180544827),
)


def _triangle_samples(triangle: tuple[np.ndarray, np.ndarray, np.ndarray],
                      area: float):
    for bary, relative_weight in _TRIANGLE_RULE:
        point = sum(coefficient * vertex
                    for coefficient, vertex in zip(bary, triangle))
        yield point, area * relative_weight


def _moving_pair_integrals(
        moved_a: PlanarFace, moved_b: PlanarFace, direction: np.ndarray,
        height_a: np.ndarray, height_b: np.ndarray) -> MovingPairIntegrals:
    """Integrate one geometric pair before global pressure normalization."""

    overlap = planar_projected_overlap(moved_a, moved_b)
    rows = len(moved_a.corners)
    columns = rows + len(moved_b.corners)
    support = np.zeros(rows, dtype=float)
    weighted_gap = np.zeros(rows, dtype=float)
    gradient = np.zeros((rows, columns), dtype=float)
    if overlap is None:
        return MovingPairIntegrals(
            0.0, support, weighted_gap, gradient, float("inf"),
        )
    polygon = overlap.polygon
    centroid = np.mean(polygon, axis=0)
    for index, corner in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        area = 0.5 * float(np.linalg.norm(
            np.cross(corner - centroid, following - centroid)
        ))
        if area <= 0.0:
            continue
        triangle = (centroid, corner, following)
        for point, weight in _triangle_samples(triangle, area):
            shape_a = overlap.shape_values(moved_a, point)
            shape_b = overlap.shape_values(moved_b, point)
            gap = float(direction @ (
                shape_a @ moved_a.corners - shape_b @ moved_b.corners
            ) + shape_a @ height_a - shape_b @ height_b)
            support += weight * shape_a
            weighted_gap += weight * shape_a * gap
            gradient[:, :rows] += weight * np.outer(shape_a, shape_a)
            gradient[:, rows:] -= weight * np.outer(shape_a, shape_b)
    minimum_vertex_gap = float("inf")
    for point in polygon:
        shape_a = overlap.shape_values(moved_a, point)
        shape_b = overlap.shape_values(moved_b, point)
        minimum_vertex_gap = min(minimum_vertex_gap, float(direction @ (
            shape_a @ moved_a.corners - shape_b @ moved_b.corners
        ) + shape_a @ height_a - shape_b @ height_b))
    return MovingPairIntegrals(
        overlap.area, support, weighted_gap, gradient, minimum_vertex_gap,
    )


def moving_mortar_state(source_face: PlanarFace, target_face: PlanarFace,
                        normal: np.ndarray, stiffness: float,
                        source_translation: np.ndarray,
                        target_translation: np.ndarray,
                        source_normal_dofs: np.ndarray,
                        target_normal_dofs: np.ndarray) -> MovingMortarState:
    """Integrate a moving P1/Q1 primal mortar gap on two planar projections.

    The normal fields may vary nodally while the projected outlines translate
    without in-plane distortion. The returned normal residual and tangent are
    derivatives of this discrete pressure-space potential at fixed tangential
    translations. Translation derivatives are a separate pending force map.
    """

    direction = np.asarray(normal, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("contact normal must be a finite 3-vector")
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise ValueError("contact normal must be nonzero")
    direction = direction / length
    for name, face in (("source", source_face), ("target", target_face)):
        face_normal = np.cross(face.corners[1] - face.corners[0],
                               face.corners[-1] - face.corners[0])
        face_normal /= np.linalg.norm(face_normal)
        if abs(float(face_normal @ direction)) < 1.0 - 1.0e-10:
            raise ValueError(f"moving mortar {name} face must be parallel to the contact plane")
    penalty = float(stiffness)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("normal stiffness must be finite and positive")
    translation_a = np.asarray(source_translation, dtype=float)
    translation_b = np.asarray(target_translation, dtype=float)
    height_a = np.asarray(source_normal_dofs, dtype=float)
    height_b = np.asarray(target_normal_dofs, dtype=float)
    if (translation_a.shape != (3,) or translation_b.shape != (3,)
            or height_a.shape != (len(source_face.corners),)
            or height_b.shape != (len(target_face.corners),)
            or not all(np.all(np.isfinite(item)) for item in
                       (translation_a, translation_b, height_a, height_b))):
        raise ValueError("moving mortar fields have invalid size or nonfinite values")
    moved_a = PlanarFace(source_face.feature_id,
                         source_face.corners + translation_a)
    moved_b = PlanarFace(target_face.feature_id,
                         target_face.corners + translation_b)
    integrals = _moving_pair_integrals(
        moved_a, moved_b, direction, height_a, height_b,
    )
    rows = len(source_face.corners)
    columns = rows + len(target_face.corners)
    support = integrals.support
    weighted_gap = integrals.weighted_gap
    gradient = integrals.gap_gradient
    active_support = support > 0.0
    normalized_gap = np.zeros(rows, dtype=float)
    scale = max(float(np.max(np.linalg.norm(moved_a.corners - moved_a.corners[0], axis=1))),
                float(np.max(np.linalg.norm(moved_b.corners - moved_b.corners[0], axis=1))))
    roundoff = 256.0 * np.finfo(float).eps * scale
    weighted_gap[np.abs(weighted_gap) <= roundoff * support] = 0.0
    normalized_gap[active_support] = weighted_gap[active_support] / support[active_support]
    closed = active_support & (normalized_gap <= 0.0)
    active_gap = np.minimum(normalized_gap, 0.0)
    energy = 0.5 * penalty * float(support @ (active_gap * active_gap))
    residual = gradient.T @ (penalty * active_gap)
    tangent = np.zeros((columns, columns), dtype=float)
    if np.any(closed):
        tangent = (penalty * gradient[closed].T
                   @ np.diag(1.0 / support[closed]) @ gradient[closed])
    return MovingMortarState(
        integrals.overlap_area, support, weighted_gap, normalized_gap, energy,
        residual, tangent, integrals.minimum_vertex_gap,
    )


def _height_lipschitz(face: PlanarFace, heights: np.ndarray, overlap) -> float:
    xy = _to_frame(face.corners, overlap.origin,
                   overlap.tangent, overlap.binormal)
    if len(face.corners) == 3:
        _, gradient = overlap.shape_values_and_gradient(
            face, np.mean(face.corners, axis=0)
        )
        return float(np.linalg.norm(heights @ gradient))
    minimum_determinant = float("inf")
    maximum_jacobian_norm = 0.0
    maximum_height_derivative = 0.0
    determinant_sign = None
    for natural in ((-1.0, -1.0), (1.0, -1.0),
                    (1.0, 1.0), (-1.0, 1.0)):
        _shape, derivatives = _q4_shape(np.asarray(natural))
        jacobian = derivatives @ xy
        determinant = float(np.linalg.det(jacobian))
        sign = np.sign(determinant)
        if determinant_sign is None:
            determinant_sign = sign
        if sign != determinant_sign or determinant == 0.0:
            raise ValueError("projected Q4 Jacobian changes sign")
        minimum_determinant = min(minimum_determinant, abs(determinant))
        maximum_jacobian_norm = max(maximum_jacobian_norm,
                                    float(np.linalg.norm(jacobian)))
        maximum_height_derivative = max(
            maximum_height_derivative,
            float(np.linalg.norm(derivatives @ heights)),
        )
    return (maximum_jacobian_norm / minimum_determinant
            * maximum_height_derivative)


def certify_moving_mortar_gap(
        source_face: PlanarFace, target_face: PlanarFace,
        normal: np.ndarray, source_translation: np.ndarray,
        target_translation: np.ndarray,
        source_normal_dofs: np.ndarray, target_normal_dofs: np.ndarray,
        *, gap_tolerance: float, maximum_triangles: int,
        ) -> MovingGapCertificate:
    """Bound the represented normal gap over a moving convex overlap.

    A global shape-gradient bound limits unsampled gap variation in each
    triangle. Longest-edge subdivision continues until every lower bound meets
    the dimensional tolerance, a sampled violation is found, or the declared
    resource budget is exhausted. Failure to certify is never a passing gap.
    """

    tolerance = float(gap_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("gap_tolerance must be finite and positive")
    if type(maximum_triangles) is not int or maximum_triangles < 1:
        raise ValueError("maximum_triangles must be a positive integer")
    direction = np.asarray(normal, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("contact normal must be a finite 3-vector")
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 0.0:
        raise ValueError("contact normal must be nonzero")
    direction = direction / magnitude
    translations = (np.asarray(source_translation, dtype=float),
                    np.asarray(target_translation, dtype=float))
    heights = (np.asarray(source_normal_dofs, dtype=float),
               np.asarray(target_normal_dofs, dtype=float))
    for name, face, translation, values in zip(
        ("source", "target"), (source_face, target_face), translations, heights
    ):
        if (translation.shape != (3,) or values.shape != (len(face.corners),)
                or not np.all(np.isfinite(translation))
                or not np.all(np.isfinite(values))):
            raise ValueError(f"moving mortar {name} field has invalid size or nonfinite values")
        face_normal = np.cross(face.corners[1] - face.corners[0],
                               face.corners[-1] - face.corners[0])
        face_normal /= np.linalg.norm(face_normal)
        if abs(float(face_normal @ direction)) < 1.0 - 1.0e-10:
            raise ValueError(f"moving mortar {name} face must be parallel to the contact plane")
    moved_a = PlanarFace(source_face.feature_id,
                         source_face.corners + translations[0])
    moved_b = PlanarFace(target_face.feature_id,
                         target_face.corners + translations[1])
    heights_a, heights_b = heights
    overlap = planar_projected_overlap(moved_a, moved_b)
    if overlap is None:
        return MovingGapCertificate(float("inf"), float("inf"), True, 0, "empty")

    def gap(point: np.ndarray) -> float:
        shape_a = overlap.shape_values(moved_a, point)
        shape_b = overlap.shape_values(moved_b, point)
        return float(direction @ (
            shape_a @ moved_a.corners - shape_b @ moved_b.corners
        ) + shape_a @ heights_a - shape_b @ heights_b)

    polygon = overlap.polygon
    quadratic = quadratic_gap_extremum(
        moved_a, moved_b, overlap, direction, heights_a, heights_b,
    )
    if quadratic is not None:
        sampled = min(gap(point) for point in polygon)
        return MovingGapCertificate(
            sampled, quadratic.guarded_lower,
            bool(quadratic.guarded_lower >= -tolerance), 0, "quadratic_extrema",
        )
    lipschitz = (
        _height_lipschitz(moved_a, heights_a + moved_a.corners @ direction, overlap)
        + _height_lipschitz(moved_b, heights_b + moved_b.corners @ direction, overlap)
    )
    centre = np.mean(polygon, axis=0)
    pending = [
        (centre, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]
    minimum_sampled = float("inf")
    conservative_lower = float("inf")
    checked = 0
    while pending:
        if checked >= maximum_triangles:
            return MovingGapCertificate(
                minimum_sampled, float("-inf"), False, checked,
            )
        triangle = pending.pop()
        checked += 1
        values = tuple(gap(point) for point in triangle)
        minimum_sampled = min(minimum_sampled, *values)
        edges = (
            float(np.linalg.norm(triangle[1] - triangle[0])),
            float(np.linalg.norm(triangle[2] - triangle[1])),
            float(np.linalg.norm(triangle[0] - triangle[2])),
        )
        longest = max(edges)
        bound = min(values) - lipschitz * longest
        if minimum_sampled < -tolerance:
            return MovingGapCertificate(minimum_sampled, bound, False, checked)
        if bound >= -tolerance:
            conservative_lower = min(conservative_lower, bound)
            continue
        index = int(np.argmax(edges))
        first, second = ((0, 1), (1, 2), (2, 0))[index]
        remaining = 3 - first - second
        midpoint = 0.5 * (triangle[first] + triangle[second])
        pending.append((triangle[first], midpoint, triangle[remaining]))
        pending.append((midpoint, triangle[second], triangle[remaining]))
    return MovingGapCertificate(minimum_sampled, conservative_lower, True, checked)


def _boundary_quadrature(subject: PlanarFace, clip_face: PlanarFace,
                         overlap) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (point, outward-normal times line weight) on clipped edges."""

    subject_xy = _to_frame(subject.corners, overlap.origin,
                           overlap.tangent, overlap.binormal)
    clip_xy = _to_frame(clip_face.corners, overlap.origin,
                        overlap.tangent, overlap.binormal)
    if _signed_area(subject_xy) < 0.0:
        subject_xy = subject_xy[::-1].copy()
    if _signed_area(clip_xy) < 0.0:
        clip_xy = clip_xy[::-1].copy()
    span = max(float(np.max(np.linalg.norm(subject_xy - subject_xy[0], axis=1))),
               float(np.max(np.linalg.norm(clip_xy - clip_xy[0], axis=1))))
    tolerance = 1.0e-12 * span * span
    gauss = (-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0))
    weights = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    samples = []
    for index, start in enumerate(subject_xy):
        edge = subject_xy[(index + 1) % len(subject_xy)] - start
        lower, upper = 0.0, 1.0
        for clip_index, clip_start in enumerate(clip_xy):
            clip_edge = clip_xy[(clip_index + 1) % len(clip_xy)] - clip_start
            distance = _cross2(clip_edge, start - clip_start)
            slope = _cross2(clip_edge, edge)
            if abs(slope) <= tolerance:
                if distance < -tolerance:
                    lower, upper = 1.0, 0.0
                    break
                continue
            crossing = -distance / slope
            if slope > 0.0:
                lower = max(lower, crossing)
            else:
                upper = min(upper, crossing)
            if lower >= upper:
                break
        if upper <= lower:
            continue
        outward = edge[1] * overlap.tangent - edge[0] * overlap.binormal
        for abscissa, weight in zip(gauss, weights):
            fraction = 0.5 * (lower + upper + (upper - lower) * abscissa)
            xy = start + fraction * edge
            point = (overlap.origin + xy[0] * overlap.tangent
                     + xy[1] * overlap.binormal)
            samples.append((point, 0.5 * (upper - lower) * weight * outward))
    return samples


def _has_shared_edge_transition(source: PlanarFace, target: PlanarFace,
                                overlap) -> bool:
    """Detect positive-length coincident boundaries with nonunique transport."""

    first = _to_frame(source.corners, overlap.origin,
                      overlap.tangent, overlap.binormal)
    second = _to_frame(target.corners, overlap.origin,
                       overlap.tangent, overlap.binormal)
    scale = max(float(np.max(np.linalg.norm(first - first[0], axis=1))),
                float(np.max(np.linalg.norm(second - second[0], axis=1))))
    area_tolerance = 1.0e-12 * scale * scale
    length_tolerance = 1.0e-12 * scale
    for index, start in enumerate(first):
        edge = first[(index + 1) % len(first)] - start
        length = float(np.linalg.norm(edge))
        axis = edge / length
        for other_index, other_start in enumerate(second):
            other_edge = second[(other_index + 1) % len(second)] - other_start
            if (abs(_cross2(edge, other_edge)) > area_tolerance
                    or abs(_cross2(edge, other_start - start)) > area_tolerance):
                continue
            extent = sorted((float((other_start - start) @ axis),
                             float((other_start + other_edge - start) @ axis)))
            if min(length, extent[1]) - max(0.0, extent[0]) > length_tolerance:
                return True
    return False


@dataclass(frozen=True)
class MovingPairTransport:
    source_support_derivative: np.ndarray
    target_support_derivative: np.ndarray
    source_weighted_gap_derivative: np.ndarray
    target_weighted_gap_derivative: np.ndarray


def _moving_pair_transport(
        moved_a: PlanarFace, moved_b: PlanarFace, direction: np.ndarray,
        heights_a: np.ndarray, heights_b: np.ndarray) -> MovingPairTransport:
    """Transport local support and weighted gap before pressure coupling."""

    rows = len(heights_a)
    dm_a = np.zeros((rows, 3))
    dm_b = np.zeros((rows, 3))
    dW_a = np.zeros((rows, 3))
    dW_b = np.zeros((rows, 3))
    overlap = planar_projected_overlap(moved_a, moved_b)
    if overlap is None:
        return MovingPairTransport(dm_a, dm_b, dW_a, dW_b)
    if _has_shared_edge_transition(moved_a, moved_b, overlap):
        raise ValueError("shared-edge overlap transition needs a one-sided derivative")
    polygon = overlap.polygon
    centroid = np.mean(polygon, axis=0)
    for index, corner in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        area = 0.5 * float(np.linalg.norm(
            np.cross(corner - centroid, following - centroid)
        ))
        if area <= 0.0:
            continue
        triangle = (centroid, corner, following)
        for point, weight in _triangle_samples(triangle, area):
            shape_a, gradient_a = overlap.shape_values_and_gradient(moved_a, point)
            shape_b, gradient_b = overlap.shape_values_and_gradient(moved_b, point)
            gap = float(direction @ (
                shape_a @ moved_a.corners - shape_b @ moved_b.corners
            ) + shape_a @ heights_a - shape_b @ heights_b)
            grad_gap_a = -((heights_a + moved_a.corners @ direction) @ gradient_a)
            grad_gap_b = (heights_b + moved_b.corners @ direction) @ gradient_b
            dm_a -= weight * gradient_a
            dW_a += weight * (-gradient_a * gap
                              + shape_a[:, None] * grad_gap_a)
            dW_b += weight * shape_a[:, None] * grad_gap_b
    for point, weighted_outward in _boundary_quadrature(moved_a, moved_b, overlap):
        shape_a = overlap.shape_values(moved_a, point)
        shape_b = overlap.shape_values(moved_b, point)
        gap = float(direction @ (
            shape_a @ moved_a.corners - shape_b @ moved_b.corners
        ) + shape_a @ heights_a - shape_b @ heights_b)
        dm_a += np.outer(shape_a, weighted_outward)
        dW_a += gap * np.outer(shape_a, weighted_outward)
    for point, weighted_outward in _boundary_quadrature(moved_b, moved_a, overlap):
        shape_a = overlap.shape_values(moved_a, point)
        shape_b = overlap.shape_values(moved_b, point)
        gap = float(direction @ (
            shape_a @ moved_a.corners - shape_b @ moved_b.corners
        ) + shape_a @ heights_a - shape_b @ heights_b)
        dm_b += np.outer(shape_a, weighted_outward)
        dW_b += gap * np.outer(shape_a, weighted_outward)
    return MovingPairTransport(dm_a, dm_b, dW_a, dW_b)


def moving_mortar_translation_residual(
        source_face: PlanarFace, target_face: PlanarFace,
        normal: np.ndarray, stiffness: float,
        source_translation: np.ndarray, target_translation: np.ndarray,
        source_normal_dofs: np.ndarray, target_normal_dofs: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the moving pressure-space potential by transport.

    This includes shape changes at fixed spatial points and motion of both
    polygon boundaries. Boundary integration has a finite three-point line
    rule and requires refinement before engineering qualification.
    """

    state = moving_mortar_state(
        source_face, target_face, normal, stiffness,
        source_translation, target_translation,
        source_normal_dofs, target_normal_dofs,
    )
    if state.overlap_area == 0.0:
        return np.zeros(3), np.zeros(3)
    direction = np.asarray(normal, dtype=float)
    direction = direction / np.linalg.norm(direction)
    moved_a = PlanarFace(source_face.feature_id,
                         source_face.corners + np.asarray(source_translation))
    moved_b = PlanarFace(target_face.feature_id,
                         target_face.corners + np.asarray(target_translation))
    transport = _moving_pair_transport(
        moved_a, moved_b, direction,
        np.asarray(source_normal_dofs, dtype=float),
        np.asarray(target_normal_dofs, dtype=float),
    )
    rows = len(source_normal_dofs)
    closed = (state.support > 0.0) & (state.normalized_gap < 0.0)
    pressure = float(stiffness) * state.normalized_gap[closed]
    source_tangent = np.sum(
        pressure[:, None] * transport.source_weighted_gap_derivative[closed]
        - 0.5 * float(stiffness) * state.normalized_gap[closed, None] ** 2
        * transport.source_support_derivative[closed], axis=0,
    )
    target_tangent = np.sum(
        pressure[:, None] * transport.target_weighted_gap_derivative[closed]
        - 0.5 * float(stiffness) * state.normalized_gap[closed, None] ** 2
        * transport.target_support_derivative[closed], axis=0,
    )
    source_normal = float(np.sum(state.normal_residual[:rows])) * direction
    target_normal = float(np.sum(state.normal_residual[rows:])) * direction
    return source_tangent + source_normal, target_tangent + target_normal


@dataclass(frozen=True)
class MovingMortarReducedSystem:
    state: MovingMortarState
    residual: np.ndarray
    tangent: np.ndarray
    relative_tangent_asymmetry: float


def moving_mortar_reduced_system(
        source_face: PlanarFace, target_face: PlanarFace,
        normal: np.ndarray, stiffness: float,
        source_translation: np.ndarray, target_translation: np.ndarray,
        source_normal_dofs: np.ndarray, target_normal_dofs: np.ndarray,
        *, difference_step: float) -> MovingMortarReducedSystem:
    """Differentiate the restricted moving mortar residual on a smooth branch.

    The first derivative uses transport of current pressure-space integrals;
    its tangent is a centred directional differentiation on a smooth branch.
    At an active-set boundary, use the one-sided derivative that retains the
    base active set. If neither side retains it, reject the ambiguous column.
    This bounded control is not a deformable-shell tangent or public solver
    profile. The caller declares a dimensional perturbation and must refine
    it for its scale before using the result as evidence.
    """

    step = float(difference_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("difference_step must be finite and positive")
    source = np.asarray(source_translation, dtype=float).reshape(3)
    target = np.asarray(target_translation, dtype=float).reshape(3)
    heights_a = np.asarray(source_normal_dofs, dtype=float).reshape(-1)
    heights_b = np.asarray(target_normal_dofs, dtype=float).reshape(-1)
    variables = np.r_[source, target, heights_a, heights_b]
    split = 6 + len(heights_a)

    def evaluate_at(values: np.ndarray):
        translation_a = values[:3]
        translation_b = values[3:6]
        normal_a = values[6:split]
        normal_b = values[split:]
        state = moving_mortar_state(
            source_face, target_face, normal, stiffness,
            translation_a, translation_b, normal_a, normal_b,
        )
        residual_a, residual_b = moving_mortar_translation_residual(
            source_face, target_face, normal, stiffness,
            translation_a, translation_b, normal_a, normal_b,
        )
        return state, np.r_[residual_a, residual_b, state.normal_residual]

    base_state, residual = evaluate_at(variables)
    tangent = np.zeros((len(variables), len(variables)), dtype=float)
    base_active = (base_state.support > 0.0) & (base_state.normalized_gap <= 0.0)
    for column in range(len(variables)):
        delta = np.zeros_like(variables)
        delta[column] = step
        plus_state, plus_residual = evaluate_at(variables + delta)
        minus_state, minus_residual = evaluate_at(variables - delta)
        plus_active = ((plus_state.support > 0.0)
                       & (plus_state.normalized_gap <= 0.0))
        minus_active = ((minus_state.support > 0.0)
                        & (minus_state.normalized_gap <= 0.0))
        plus_matches = np.array_equal(plus_active, base_active)
        minus_matches = np.array_equal(minus_active, base_active)
        if plus_matches and minus_matches:
            tangent[:, column] = (plus_residual - minus_residual) / (2.0 * step)
        elif plus_matches:
            tangent[:, column] = (plus_residual - residual) / step
        elif minus_matches:
            tangent[:, column] = (residual - minus_residual) / step
        else:
            raise ValueError("moving mortar tangent crosses an active-set transition")
    scale = max(np.finfo(float).tiny, float(np.linalg.norm(tangent)))
    asymmetry = float(np.linalg.norm(tangent - tangent.T) / scale)
    return MovingMortarReducedSystem(base_state, residual, tangent, asymmetry)
