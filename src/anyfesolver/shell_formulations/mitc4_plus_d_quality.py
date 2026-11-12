"""Cold-path geometry and director quality checks for improved Q4 shells.

The routines in this module operate only on numeric finite-element input.  In
particular, they do not know about geometry documents, support-surface
objects, or a geometry-kernel tolerance policy.  Their tolerances are scaled
from the finite-element corner coordinates and machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np


def _finite_float64_array(values: object, *, label: str) -> np.ndarray:
    """Convert a real numeric array to finite ``float64`` without overflow escape."""

    source = np.asarray(values)
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values, not {source.dtype}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            made = np.asarray(source, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} cannot be represented as float64") from exc
    if not np.all(np.isfinite(made)):
        raise ValueError(f"{label} must remain finite when represented as float64")
    return made


@dataclass(frozen=True)
class Q4GeometryQuality:
    """Scale-aware reference-geometry metrics for one four-node shell."""

    area: float
    unit_normal: tuple[float, float, float]
    corner_angles_radians: tuple[float, float, float, float]
    minimum_edge_length: float
    maximum_edge_length: float
    warpage_angle_degrees: float
    characteristic_length: float
    minimum_surface_jacobian: float
    maximum_surface_jacobian: float
    minimum_jacobian_orientation: float


@dataclass(frozen=True)
class CornerDirectorQuality:
    """Compatibility metrics for the four directors of one Q4 element."""

    minimum_norm: float
    maximum_norm_error: float
    minimum_geometry_alignment: float
    maximum_pair_angle_degrees: float
    center_director_jacobian: float
    geometry: Q4GeometryQuality


def _q4_corners(corner_coordinates: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    corners = _finite_float64_array(corner_coordinates, label="Q4 corner coordinates")
    if corners.shape != (4, 3):
        raise ValueError(f"Q4 corner coordinates must have shape (4, 3), got {corners.shape}")
    return corners


def _unit(vector: np.ndarray, *, tolerance: float, label: str) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= tolerance:
        raise ValueError(f"{label} has zero or numerically unresolved length")
    return vector / norm, norm


def q4_area_vector(corner_coordinates: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return the translation-invariant oriented area vector of a Q4.

    The cross-diagonal form is the sum of the two common triangle area
    vectors.  It remains meaningful for a valid warped quadrilateral and does
    not form cross products of absolute global positions.
    """

    corners = _q4_corners(corner_coordinates)
    with np.errstate(over="ignore", invalid="ignore"):
        area_vector = 0.5 * np.cross(corners[2] - corners[0], corners[3] - corners[1])
    if not np.all(np.isfinite(area_vector)):
        raise ValueError("Q4 area vector overflowed finite float64 geometry")
    return area_vector


def q4_corner_angles(corner_coordinates: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return the four unsigned corner angles in radians."""

    corners = _q4_corners(corner_coordinates)
    with np.errstate(over="ignore", invalid="ignore"):
        edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    if not np.all(np.isfinite(edge_lengths)):
        raise ValueError("Q4 edge lengths overflowed finite float64 geometry")
    scale = max(float(np.max(edge_lengths)), 1.0)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(edge_lengths <= tolerance):
        raise ValueError("Q4 has a zero or numerically unresolved edge")

    angles = np.empty(4, dtype=np.float64)
    for corner in range(4):
        previous = corners[(corner - 1) % 4] - corners[corner]
        following = corners[(corner + 1) % 4] - corners[corner]
        previous_norm = float(np.linalg.norm(previous))
        following_norm = float(np.linalg.norm(following))
        if not math.isfinite(previous_norm) or not math.isfinite(following_norm):
            raise ValueError("Q4 corner edge norm is not finite")
        previous /= previous_norm
        following /= following_norm
        cosine = float(np.clip(np.dot(previous, following), -1.0, 1.0))
        angles[corner] = math.acos(cosine)
    return angles


def q4_geometry_quality(
    corner_coordinates: Sequence[Sequence[float]] | np.ndarray,
) -> Q4GeometryQuality:
    """Validate one Q4 and return FE-scale geometry metrics."""

    corners = _q4_corners(corner_coordinates)
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.roll(corners, -1, axis=0) - corners
        edge_lengths = np.linalg.norm(edges, axis=1)
        coordinate_span = np.ptp(corners, axis=0)
    if not np.all(np.isfinite(edges)) or not np.all(np.isfinite(edge_lengths)):
        raise ValueError("Q4 edge geometry overflowed finite float64 range")
    if not np.all(np.isfinite(coordinate_span)):
        raise ValueError("Q4 coordinate span overflowed finite float64 range")
    maximum_edge = float(np.max(edge_lengths))
    characteristic_length = max(maximum_edge, float(coordinate_span.max()))
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("Q4 has no finite characteristic length")
    length_tolerance = 64.0 * np.finfo(np.float64).eps * characteristic_length
    minimum_edge = float(np.min(edge_lengths))
    if minimum_edge <= length_tolerance:
        raise ValueError("Q4 has a zero or numerically unresolved edge")

    area_vector = q4_area_vector(corners)
    with np.errstate(over="ignore", invalid="ignore"):
        characteristic_area = float(
            np.float64(characteristic_length) * np.float64(characteristic_length)
        )
    if not math.isfinite(characteristic_area):
        raise ValueError("Q4 area tolerance scale overflowed finite float64 range")
    area_tolerance = 128.0 * np.finfo(np.float64).eps * characteristic_area
    unit_normal, _projected_area = _unit(area_vector, tolerance=area_tolerance, label="Q4 area vector")

    with np.errstate(over="ignore", invalid="ignore"):
        triangle_a = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        triangle_b = np.cross(corners[2] - corners[0], corners[3] - corners[0])
    if not np.all(np.isfinite(triangle_a)) or not np.all(np.isfinite(triangle_b)):
        raise ValueError("Q4 triangle area vectors overflowed finite float64 range")
    triangle_a_unit, triangle_a_twice_area = _unit(
        triangle_a, tolerance=area_tolerance, label="first Q4 triangle"
    )
    triangle_b_unit, triangle_b_twice_area = _unit(
        triangle_b, tolerance=area_tolerance, label="second Q4 triangle"
    )
    warpage_cosine = float(np.clip(np.dot(triangle_a_unit, triangle_b_unit), -1.0, 1.0))

    evaluation_points = (
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        *(
            (xi, eta)
            for xi in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
            for eta in (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
        ),
    )
    jacobians: list[float] = []
    jacobian_orientations: list[float] = []
    for xi, eta in evaluation_points:
        derivative_xi = 0.25 * np.asarray((-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)))
        derivative_eta = 0.25 * np.asarray((-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi))
        covariant_xi = derivative_xi @ corners
        covariant_eta = derivative_eta @ corners
        with np.errstate(over="ignore", invalid="ignore"):
            jacobian_vector = np.cross(covariant_xi, covariant_eta)
            jacobian = float(np.linalg.norm(jacobian_vector))
            signed_jacobian = float(np.dot(jacobian_vector, unit_normal))
        if (
            not np.all(np.isfinite(jacobian_vector))
            or not math.isfinite(jacobian)
            or not math.isfinite(signed_jacobian)
            or signed_jacobian <= area_tolerance
        ):
            raise ValueError("Q4 surface Jacobian is zero, unresolved, or reverses orientation")
        jacobians.append(jacobian)
        jacobian_orientations.append(signed_jacobian / jacobian)

    area = 0.5 * triangle_a_twice_area + 0.5 * triangle_b_twice_area
    if not math.isfinite(area):
        raise ValueError("Q4 surface area overflowed finite float64 range")

    return Q4GeometryQuality(
        area=area,
        unit_normal=tuple(float(value) for value in unit_normal),
        corner_angles_radians=tuple(float(value) for value in q4_corner_angles(corners)),
        minimum_edge_length=minimum_edge,
        maximum_edge_length=maximum_edge,
        warpage_angle_degrees=math.degrees(math.acos(warpage_cosine)),
        characteristic_length=characteristic_length,
        minimum_surface_jacobian=min(jacobians),
        maximum_surface_jacobian=max(jacobians),
        minimum_jacobian_orientation=min(jacobian_orientations),
    )


def corner_director_quality(
    corner_coordinates: Sequence[Sequence[float]] | np.ndarray,
    reference_directors: Sequence[Sequence[float]] | np.ndarray,
) -> CornerDirectorQuality:
    """Return director metrics in the finalized element-connectivity frame.

    Source face-use orientation is deliberately absent here.  The upstream
    adapter has already used it to choose the element node order.  The
    continuum reference Jacobian is positive only when the interpolated
    director follows the normal induced by that final connectivity.
    """

    geometry = q4_geometry_quality(corner_coordinates)
    corners = _q4_corners(corner_coordinates)
    directors = _finite_float64_array(reference_directors, label="reference directors")
    if directors.shape != (4, 3):
        raise ValueError(f"reference directors must have shape (4, 3), got {directors.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(directors, axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("reference director norms must remain finite")
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("reference directors must have nonzero length")
    unit_directors = directors / norms[:, None]
    connectivity_normal = np.asarray(geometry.unit_normal)
    alignments = unit_directors @ connectivity_normal

    derivative_xi = 0.25 * np.asarray((-1.0, 1.0, 1.0, -1.0))
    derivative_eta = 0.25 * np.asarray((-1.0, -1.0, 1.0, 1.0))
    covariant_xi = derivative_xi @ corners
    covariant_eta = derivative_eta @ corners
    center_director = np.mean(unit_directors, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        center_director_jacobian = float(
            np.dot(np.cross(covariant_xi, covariant_eta), center_director)
        )
    if not math.isfinite(center_director_jacobian):
        raise ValueError("center reference director Jacobian must remain finite")

    maximum_angle = 0.0
    for first in range(4):
        for second in range(first + 1, 4):
            cosine = float(np.clip(np.dot(unit_directors[first], unit_directors[second]), -1.0, 1.0))
            maximum_angle = max(maximum_angle, math.degrees(math.acos(cosine)))

    return CornerDirectorQuality(
        minimum_norm=float(np.min(norms)),
        maximum_norm_error=float(np.max(np.abs(norms - 1.0))),
        minimum_geometry_alignment=float(np.min(alignments)),
        maximum_pair_angle_degrees=maximum_angle,
        center_director_jacobian=center_director_jacobian,
        geometry=geometry,
    )


def numeric_payload_fingerprint(*arrays: Iterable[float] | np.ndarray) -> str:
    """Hash numeric shape, dtype, and values for cache invalidation.

    Floating values are canonicalized as little-endian ``float64``. Signed and
    unsigned integers are separately canonicalized as little-endian ``int64``
    and ``uint64`` so unsigned maximum cannot collide with the signed ``-1``
    sentinel. Unsupported object/complex arrays and non-finite float values
    fail closed. Provenance metadata has its own canonical JSON hash.
    """

    digest = hashlib.sha256()
    digest.update(b"ANYfeSolver.numeric-payload.v1\0")
    for values in arrays:
        source = np.asarray(values)
        if np.issubdtype(source.dtype, np.bool_):
            array = np.ascontiguousarray(source, dtype=np.uint8)
            category = b"bool"
        elif np.issubdtype(source.dtype, np.signedinteger):
            array = np.ascontiguousarray(source, dtype="<i8")
            category = b"signed-int64"
        elif np.issubdtype(source.dtype, np.unsignedinteger):
            array = np.ascontiguousarray(source, dtype="<u8")
            category = b"unsigned-int64"
        elif np.issubdtype(source.dtype, np.floating):
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    array = np.ascontiguousarray(source, dtype="<f8")
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("floating numeric payload cannot be represented as float64") from exc
            if not np.all(np.isfinite(array)):
                raise ValueError("floating numeric payload must remain finite as float64")
            category = b"float64"
        else:
            raise TypeError(
                "numeric payload fingerprint accepts only boolean, signed integer, "
                "unsigned integer, or floating arrays"
            )
        digest.update(category)
        digest.update(b"\0")
        digest.update(str(array.ndim).encode("ascii"))
        digest.update(b":")
        digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()
