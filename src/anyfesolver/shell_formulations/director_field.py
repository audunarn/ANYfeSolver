"""Numeric, sheet-aware reference-director preparation for improved Q4 shells.

This is a model-preparation module, not an element integration kernel.  It
accepts only numeric mesh arrays and compact integer region indices.  The
returned field is immutable and stores one director per element corner, so a
shared global node may legitimately carry different directors on different
sheets, parts, creases, or declared continuity regions.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from operator import index as integer_index
import re
from typing import Iterable, Sequence

import numpy as np

from .mitc4_plus_d_quality import (
    CornerDirectorQuality,
    _finite_float64_array,
    corner_director_quality,
    numeric_payload_fingerprint,
    q4_geometry_quality,
)


class DirectorProvenanceCode(str, Enum):
    """Permanent provenance labels for Q4 reference directors."""

    SOURCE_SURFACE_EXACT = "source_surface_exact"
    SOURCE_SURFACE_SAMPLED = "source_surface_sampled"
    SHEET_AWARE_MESH_RECONSTRUCTION = "sheet_aware_mesh_reconstruction"
    LOCAL_ELEMENT_RECONSTRUCTION = "local_element_reconstruction"
    LEGACY_CENTER_FRAME = "legacy_center_frame"
    USER_SUPPLIED = "user_supplied"


DIRECTOR_PROVENANCE_NUMERIC_CODE: dict[DirectorProvenanceCode, int] = {
    code: index for index, code in enumerate(DirectorProvenanceCode)
}
DIRECTOR_PROVENANCE_FROM_NUMERIC_CODE: dict[int, DirectorProvenanceCode] = {
    value: key for key, value in DIRECTOR_PROVENANCE_NUMERIC_CODE.items()
}
_INT64_MAX = int(np.iinfo(np.int64).max)
_CANONICAL_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREPARED_DIRECTOR_FACTORY_TOKEN = object()


def _strict_real_scalar(value: object, *, label: str) -> float:
    """Return one finite real scalar without accepting coercive containers."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{label} must be a real numeric scalar")
    try:
        made = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} cannot be represented as float64") from exc
    if not math.isfinite(made):
        raise ValueError(f"{label} must be finite")
    return made


def _strict_integer_array(
    values: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
    dtype: np.dtype | type,
) -> np.ndarray:
    """Validate integer meaning and range before any narrowing conversion."""

    source = np.asarray(values)
    if np.issubdtype(source.dtype, np.bool_):
        raise ValueError(f"{label} must contain true integers, not booleans")

    if np.issubdtype(source.dtype, np.integer):
        if np.issubdtype(source.dtype, np.signedinteger):
            outside = np.any(source < minimum) or np.any(source > maximum)
        else:
            outside = (minimum > 0 and np.any(source < minimum)) or np.any(
                source > maximum
            )
        if outside:
            raise ValueError(f"{label} values must lie in [{minimum}, {maximum}]")
        return np.asarray(source, dtype=dtype)

    if source.dtype.kind == "O":
        checked: list[int] = []
        for value in source.flat:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{label} must contain true integers, not booleans")
            try:
                made = int(integer_index(value))
            except TypeError as exc:
                raise ValueError(f"{label} must contain true integers") from exc
            if made < minimum or made > maximum:
                raise ValueError(f"{label} values must lie in [{minimum}, {maximum}]")
            checked.append(made)
        return np.asarray(checked, dtype=dtype).reshape(source.shape)

    raise ValueError(f"{label} must contain true integers")


@dataclass(frozen=True)
class DirectorValidationLimits:
    """Finite-element-scale acceptance limits for a prepared director field."""

    normalization_atol: float = 1.0e-10
    minimum_geometry_alignment: float = 1.0e-8
    maximum_corner_spread_degrees: float = 85.0
    crease_angle_degrees: float = 35.0

    def __post_init__(self) -> None:
        normalization_atol = _strict_real_scalar(
            self.normalization_atol,
            label="normalization_atol",
        )
        minimum_geometry_alignment = _strict_real_scalar(
            self.minimum_geometry_alignment,
            label="minimum_geometry_alignment",
        )
        maximum_corner_spread_degrees = _strict_real_scalar(
            self.maximum_corner_spread_degrees,
            label="maximum_corner_spread_degrees",
        )
        crease_angle_degrees = _strict_real_scalar(
            self.crease_angle_degrees,
            label="crease_angle_degrees",
        )
        if normalization_atol < 0.0:
            raise ValueError("normalization_atol must be finite and nonnegative")
        if not -1.0 < minimum_geometry_alignment <= 1.0:
            raise ValueError("minimum_geometry_alignment must lie in (-1, 1]")
        if not 0.0 <= maximum_corner_spread_degrees <= 180.0:
            raise ValueError("maximum_corner_spread_degrees must lie in [0, 180]")
        if not 0.0 <= crease_angle_degrees <= 180.0:
            raise ValueError("crease_angle_degrees must lie in [0, 180]")
        object.__setattr__(self, "normalization_atol", normalization_atol)
        object.__setattr__(self, "minimum_geometry_alignment", minimum_geometry_alignment)
        object.__setattr__(
            self,
            "maximum_corner_spread_degrees",
            maximum_corner_spread_degrees,
        )
        object.__setattr__(self, "crease_angle_degrees", crease_angle_degrees)


@dataclass(frozen=True)
class DirectorFieldQuality:
    """Whole-field diagnostics produced during one-time preparation."""

    minimum_geometry_alignment: float
    maximum_corner_spread_degrees: float
    minimum_smoothing_component_size: int
    maximum_smoothing_component_size: int
    smoothing_component_count: int
    smooth_interior_edge_count: int
    angular_crease_edge_count: int
    declared_crease_edge_count: int
    region_boundary_edge_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int


@dataclass(frozen=True)
class PreparedDirectorField:
    """Immutable per-element-corner directors and compact provenance codes.

    Instances are returned by the validated preparation functions. Direct
    construction is unsupported so callers cannot pair director arrays with
    stale quality or fingerprint records.
    """

    directors: np.ndarray
    provenance_codes: np.ndarray
    element_quality: tuple[CornerDirectorQuality, ...]
    quality: DirectorFieldQuality
    diagnostics: tuple[str, ...]
    numeric_fingerprint: str
    _factory_token: InitVar[object] = None

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PREPARED_DIRECTOR_FACTORY_TOKEN:
            raise TypeError(
                "PreparedDirectorField direct construction is unsupported; "
                "use a validated director preparation function"
            )
        directors = np.array(
            _finite_float64_array(self.directors, label="prepared directors"),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if directors.ndim != 3 or directors.shape[1:] != (4, 3):
            raise ValueError("prepared directors must have shape (n_element, 4, 3)")
        if directors.shape[0] == 0:
            raise ValueError("prepared director field requires at least one element")
        with np.errstate(over="ignore", invalid="ignore"):
            director_norms = np.linalg.norm(directors, axis=2)
        if not np.all(np.isfinite(director_norms)) or np.any(
            director_norms <= np.finfo(np.float64).tiny
        ):
            raise ValueError("prepared directors must have finite nonzero length")
        validated_provenance_codes = _strict_integer_array(
            self.provenance_codes,
            label="director provenance codes",
            minimum=0,
            maximum=int(np.iinfo(np.uint8).max),
            dtype=np.int64,
        )
        if validated_provenance_codes.shape != directors.shape[:2]:
            raise ValueError("director provenance codes must have shape (n_element, 4)")
        unknown_codes = set(
            int(value) for value in np.unique(validated_provenance_codes)
        ) - set(
            DIRECTOR_PROVENANCE_FROM_NUMERIC_CODE
        )
        if unknown_codes:
            raise ValueError(f"unknown numeric director provenance codes: {sorted(unknown_codes)}")
        provenance_codes = np.array(
            validated_provenance_codes,
            dtype=np.uint8,
            order="C",
            copy=True,
        )
        if not isinstance(self.element_quality, tuple) or any(
            not isinstance(item, CornerDirectorQuality) for item in self.element_quality
        ):
            raise TypeError("element_quality must contain CornerDirectorQuality records")
        if len(self.element_quality) != directors.shape[0]:
            raise ValueError("element_quality must contain one record per element")
        if not isinstance(self.quality, DirectorFieldQuality):
            raise TypeError("quality must be a DirectorFieldQuality record")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(value, str) or not value for value in self.diagnostics
        ):
            raise TypeError("diagnostics must contain nonempty strings")
        if not isinstance(self.numeric_fingerprint, str) or _CANONICAL_SHA256.fullmatch(
            self.numeric_fingerprint
        ) is None:
            raise ValueError(
                "numeric director fingerprint must be canonical lowercase SHA-256 hex"
            )
        fingerprint = self.numeric_fingerprint
        directors.setflags(write=False)
        provenance_codes.setflags(write=False)
        object.__setattr__(self, "directors", directors)
        object.__setattr__(self, "provenance_codes", provenance_codes)
        object.__setattr__(self, "element_quality", tuple(self.element_quality))
        object.__setattr__(self, "diagnostics", self.diagnostics)
        object.__setattr__(self, "numeric_fingerprint", fingerprint)

    @property
    def provenance_labels(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return cold-path labels corresponding to the compact numeric codes."""

        return tuple(
            tuple(DIRECTOR_PROVENANCE_FROM_NUMERIC_CODE[int(code)].value for code in row)
            for row in self.provenance_codes
        )


@dataclass(frozen=True)
class SourceCornerSamples:
    """Optional immutable UV/tangent samples retained for cold qualification.

    These arrays never replace FE interpolation or enter an element hot loop.
    Tangent vectors, when supplied, must be present as a linearly independent
    pair at every corner.
    """

    source_uv: np.ndarray | None = None
    source_tangent_1: np.ndarray | None = None
    source_tangent_2: np.ndarray | None = None
    numeric_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        uv = _optional_corner_array(self.source_uv, trailing_size=2, label="source_uv")
        tangent_1 = _optional_corner_array(
            self.source_tangent_1,
            trailing_size=3,
            label="source_tangent_1",
        )
        tangent_2 = _optional_corner_array(
            self.source_tangent_2,
            trailing_size=3,
            label="source_tangent_2",
        )
        if (tangent_1 is None) != (tangent_2 is None):
            raise ValueError("source tangent 1 and source tangent 2 must be supplied together")
        present = tuple(array for array in (uv, tangent_1, tangent_2) if array is not None)
        if not present:
            raise ValueError("at least one source corner sample array is required")
        element_counts = {array.shape[0] for array in present}
        if len(element_counts) != 1:
            raise ValueError("source UV and tangent samples must have the same element count")
        if tangent_1 is not None and tangent_2 is not None:
            unit_1 = _scaled_unit_vectors(tangent_1, label="source corner tangent 1")
            unit_2 = _scaled_unit_vectors(tangent_2, label="source corner tangent 2")
            with np.errstate(over="ignore", invalid="ignore"):
                relative_cross = np.linalg.norm(np.cross(unit_1, unit_2), axis=2)
            if not np.all(np.isfinite(relative_cross)):
                raise ValueError("normalized source corner tangent cross products must be finite")
            if np.any(relative_cross <= 128.0 * np.finfo(np.float64).eps):
                raise ValueError("source corner tangent pairs must be linearly independent")
        object.__setattr__(self, "source_uv", uv)
        object.__setattr__(self, "source_tangent_1", tangent_1)
        object.__setattr__(self, "source_tangent_2", tangent_2)
        object.__setattr__(self, "numeric_fingerprint", numeric_payload_fingerprint(*present))

    @property
    def element_count(self) -> int:
        for array in (self.source_uv, self.source_tangent_1, self.source_tangent_2):
            if array is not None:
                return int(array.shape[0])
        raise RuntimeError("source corner samples were not initialized")


def _optional_corner_array(
    values: Sequence[Sequence[Sequence[float]]] | np.ndarray | None,
    *,
    trailing_size: int,
    label: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = _finite_float64_array(values, label=label)
    if array.shape == (4, trailing_size):
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[1:] != (4, trailing_size) or array.shape[0] == 0:
        raise ValueError(
            f"{label} must have shape (n_element, 4, {trailing_size}), got {array.shape}"
        )
    made = np.array(array, dtype=np.float64, order="C", copy=True)
    made.setflags(write=False)
    return made


def _scaled_unit_vectors(vectors: np.ndarray, *, label: str) -> np.ndarray:
    """Normalize finite vectors without overflowing raw norm calculations."""

    scale = np.max(np.abs(vectors), axis=2)
    if not np.all(np.isfinite(scale)):
        raise ValueError(f"{label} scale must be finite")
    if np.any(scale <= 0.0):
        raise ValueError(f"{label} must have nonzero length")
    scaled = vectors / scale[:, :, None]
    norm = np.linalg.norm(scaled, axis=2)
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError(f"{label} normalized length must be finite and nonzero")
    unit = scaled / norm[:, :, None]
    if not np.all(np.isfinite(unit)):
        raise ValueError(f"{label} normalized vectors must be finite")
    return unit


def _batch_corner_coordinates(values: Sequence[Sequence[Sequence[float]]] | np.ndarray) -> np.ndarray:
    corners = _finite_float64_array(values, label="corner coordinates")
    if corners.shape == (4, 3):
        corners = corners[None, :, :]
    if corners.ndim != 3 or corners.shape[1:] != (4, 3):
        raise ValueError(f"corner coordinates must have shape (n_element, 4, 3), got {corners.shape}")
    if corners.shape[0] == 0:
        raise ValueError("at least one Q4 element is required")
    return np.ascontiguousarray(corners)


def _source_face_use_orientation_signs(
    values: Sequence[int] | np.ndarray | None,
    count: int,
) -> np.ndarray:
    """Validate source-relative signs retained as cold provenance only."""

    if values is None:
        signs = np.ones(count, dtype=np.int8)
    else:
        signs = np.asarray(values)
        if signs.shape != (count,):
            raise ValueError(f"face-use orientation signs must have shape ({count},)")
        signs = _strict_integer_array(
            signs,
            label="face-use orientation signs",
            minimum=-1,
            maximum=1,
            dtype=np.int8,
        )
    if not np.all(np.abs(signs) == 1):
        raise ValueError("face-use orientation signs must contain only +1 or -1")
    return signs


def validate_element_corner_directors(
    corner_coordinates: Sequence[Sequence[float]] | np.ndarray,
    reference_directors: Sequence[Sequence[float]] | np.ndarray,
    *,
    limits: DirectorValidationLimits = DirectorValidationLimits(),
) -> CornerDirectorQuality:
    """Fail closed unless directors give a positive reference Jacobian.

    The comparison frame is the normal induced by finalized element
    connectivity.  Source face-use orientation has already selected that
    connectivity upstream and must not be multiplied into directors again.
    """

    quality = corner_director_quality(corner_coordinates, reference_directors)
    if quality.maximum_norm_error > limits.normalization_atol:
        raise ValueError(
            "reference directors must be normalized before solver handoff "
            f"(maximum norm error {quality.maximum_norm_error:.3e})"
        )
    jacobian_tolerance = (
        128.0
        * np.finfo(np.float64).eps
        * quality.geometry.characteristic_length**2
    )
    if quality.center_director_jacobian <= jacobian_tolerance:
        raise ValueError(
            "reference directors do not produce a positive center reference Jacobian "
            "with finalized Q4 connectivity "
            f"({quality.center_director_jacobian:.6g})"
        )
    if quality.minimum_geometry_alignment <= limits.minimum_geometry_alignment:
        raise ValueError(
            "reference director orientation is incompatible with the oriented Q4 geometry "
            f"(minimum alignment {quality.minimum_geometry_alignment:.6g})"
        )
    if quality.maximum_pair_angle_degrees > limits.maximum_corner_spread_degrees:
        raise ValueError(
            "reference director angular variation exceeds the element acceptance limit "
            f"({quality.maximum_pair_angle_degrees:.6g} degrees)"
        )
    return quality


def prepare_supplied_corner_directors(
    corner_coordinates: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    reference_directors: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    source_face_use_orientation_signs: Sequence[int] | np.ndarray | None = None,
    provenance: DirectorProvenanceCode | str = DirectorProvenanceCode.USER_SUPPLIED,
    limits: DirectorValidationLimits = DirectorValidationLimits(),
) -> PreparedDirectorField:
    """Validate numeric directors in the finalized connectivity frame.

    ``source_face_use_orientation_signs`` records whether each source face use
    is forward or reversed relative to its underlying source face.  It is
    validated and fingerprinted as cold provenance, but never flips a
    director: node ordering has already applied that source convention.
    """

    corners = _batch_corner_coordinates(corner_coordinates)
    directors = _finite_float64_array(reference_directors, label="reference directors")
    if directors.shape == (4, 3) and corners.shape[0] == 1:
        directors = directors[None, :, :]
    if directors.shape != corners.shape:
        raise ValueError(f"reference directors must have shape {corners.shape}, got {directors.shape}")
    source_signs = _source_face_use_orientation_signs(
        source_face_use_orientation_signs,
        corners.shape[0],
    )
    try:
        provenance_code = DirectorProvenanceCode(provenance)
    except ValueError as exc:
        raise ValueError(f"unsupported director provenance code: {provenance!r}") from exc
    if provenance_code in {
        DirectorProvenanceCode.SHEET_AWARE_MESH_RECONSTRUCTION,
        DirectorProvenanceCode.LOCAL_ELEMENT_RECONSTRUCTION,
    }:
        raise ValueError("reconstruction provenance may only be emitted by reconstruct_corner_directors")

    element_quality = tuple(
        validate_element_corner_directors(
            corners[element],
            directors[element],
            limits=limits,
        )
        for element in range(corners.shape[0])
    )
    code = DIRECTOR_PROVENANCE_NUMERIC_CODE[provenance_code]
    codes = np.full((corners.shape[0], 4), code, dtype=np.uint8)
    minimum_alignment = min(item.minimum_geometry_alignment for item in element_quality)
    maximum_spread = max(item.maximum_pair_angle_degrees for item in element_quality)
    quality = DirectorFieldQuality(
        minimum_geometry_alignment=minimum_alignment,
        maximum_corner_spread_degrees=maximum_spread,
        minimum_smoothing_component_size=1,
        maximum_smoothing_component_size=1,
        smoothing_component_count=4 * corners.shape[0],
        smooth_interior_edge_count=0,
        angular_crease_edge_count=0,
        declared_crease_edge_count=0,
        region_boundary_edge_count=0,
        boundary_edge_count=0,
        nonmanifold_edge_count=0,
    )
    return PreparedDirectorField(
        directors=directors,
        provenance_codes=codes,
        element_quality=element_quality,
        quality=quality,
        diagnostics=(
            "validated_numeric_input",
            "source_face_use_orientation_applied_upstream",
            f"provenance={provenance_code.value}",
        ),
        numeric_fingerprint=numeric_payload_fingerprint(corners, directors, source_signs),
        _factory_token=_PREPARED_DIRECTOR_FACTORY_TOKEN,
    )


def _compact_indices(
    values: Sequence[int] | np.ndarray | None,
    count: int,
    *,
    label: str,
) -> np.ndarray:
    if values is None:
        return np.full(count, -1, dtype=np.int64)
    array = np.asarray(values)
    if array.shape != (count,):
        raise ValueError(f"{label} must have shape ({count},)")
    try:
        return _strict_integer_array(
            array,
            label=label,
            minimum=-1,
            maximum=_INT64_MAX,
            dtype=np.int64,
        )
    except ValueError as exc:
        raise ValueError(
            f"{label} must contain -1 (unavailable) or nonnegative compact integer "
            f"indices no greater than {_INT64_MAX}"
        ) from exc


def _edge_set(edges: Iterable[Sequence[int]], node_count: int, *, label: str) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for raw_edge in edges:
        try:
            raw_values = tuple(raw_edge)
        except TypeError as exc:
            raise ValueError(f"{label} entries must be pairs of integer node indices") from exc
        if len(raw_values) != 2:
            raise ValueError(f"{label} entries must be pairs of distinct node indices")
        made: list[int] = []
        for value in raw_values:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{label} must contain true integer node indices, not booleans")
            try:
                made.append(int(integer_index(value)))
            except TypeError as exc:
                raise ValueError(f"{label} must contain true integer node indices") from exc
        edge = (made[0], made[1])
        if edge[0] == edge[1]:
            raise ValueError(f"{label} entries must be pairs of distinct node indices")
        if min(edge) < 0 or max(edge) >= node_count:
            raise ValueError(f"{label} contains a node index outside the coordinate array")
        result.add((min(edge), max(edge)))
    return result


def _coerce_mesh(
    node_coordinates: Sequence[Sequence[float]] | np.ndarray,
    connectivity: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _finite_float64_array(node_coordinates, label="node_coordinates")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] == 0:
        raise ValueError("node_coordinates must have shape (n_node, 3)")
    elements = np.asarray(connectivity)
    if elements.ndim != 2 or elements.shape[1] != 4 or elements.shape[0] == 0:
        raise ValueError("connectivity must have shape (n_element, 4)")
    try:
        elements = _strict_integer_array(
            elements,
            label="connectivity",
            minimum=0,
            maximum=_INT64_MAX,
            dtype=np.int64,
        )
    except ValueError as exc:
        raise ValueError("connectivity must contain nonnegative integer node indices") from exc
    if np.max(elements) >= coordinates.shape[0]:
        raise ValueError("connectivity contains a node index outside node_coordinates")
    if any(len(set(int(value) for value in row)) != 4 for row in elements):
        raise ValueError("each Q4 connectivity row must contain four distinct nodes")
    return np.ascontiguousarray(coordinates), np.ascontiguousarray(elements)


def reconstruct_corner_directors(
    node_coordinates: Sequence[Sequence[float]] | np.ndarray,
    connectivity: Sequence[Sequence[int]] | np.ndarray,
    *,
    part_indices: Sequence[int] | np.ndarray | None = None,
    sheet_indices: Sequence[int] | np.ndarray | None = None,
    continuity_indices: Sequence[int] | np.ndarray | None = None,
    source_face_use_orientation_signs: Sequence[int] | np.ndarray | None = None,
    crease_edges: Iterable[Sequence[int]] = (),
    declared_intersection_edges: Iterable[Sequence[int]] = (),
    limits: DirectorValidationLimits = DirectorValidationLimits(),
) -> PreparedDirectorField:
    """Reconstruct deterministic crease- and source-region-aware directors.

    Smoothing connectivity is created only across a manifold shared edge when
    the part, sheet, and declared continuity indices all agree, the edge is not
    explicitly blocked, and its oriented facet-normal jump does not exceed the
    crease limit.  Smoothing is then performed separately for every
    ``(element corner, global node)`` fan.  Consequently, one global node may
    produce several directors without duplicating its global rotation DOFs.

    Source face-use signs are provenance relative to underlying source faces.
    They are validated and fingerprinted, but do not flip connectivity facet
    normals.  The upstream adapter must already have ordered each Q4 so its
    connectivity normal is the intended positive thickness direction.
    """

    coordinates, elements = _coerce_mesh(node_coordinates, connectivity)
    element_count = elements.shape[0]
    source_signs = _source_face_use_orientation_signs(
        source_face_use_orientation_signs,
        element_count,
    )
    parts = _compact_indices(part_indices, element_count, label="part_indices")
    sheets = _compact_indices(sheet_indices, element_count, label="sheet_indices")
    continuity = _compact_indices(continuity_indices, element_count, label="continuity_indices")
    declared_creases = _edge_set(crease_edges, coordinates.shape[0], label="crease_edges")
    intersections = _edge_set(
        declared_intersection_edges,
        coordinates.shape[0],
        label="declared_intersection_edges",
    )
    blocked_edges = declared_creases | intersections

    corners = coordinates[elements]
    geometry = tuple(q4_geometry_quality(corners[element]) for element in range(element_count))
    connectivity_normals = np.asarray(
        [np.asarray(geometry[element].unit_normal) for element in range(element_count)],
        dtype=np.float64,
    )
    corner_weights = np.asarray(
        [
            geometry[element].area * np.asarray(geometry[element].corner_angles_radians)
            for element in range(element_count)
        ],
        dtype=np.float64,
    )

    edge_to_elements: dict[tuple[int, int], list[int]] = {}
    local_corner_by_node: list[dict[int, int]] = []
    for element, row in enumerate(elements):
        local_corner_by_node.append({int(node): local for local, node in enumerate(row)})
        for local in range(4):
            first = int(row[local])
            second = int(row[(local + 1) % 4])
            edge = (min(first, second), max(first, second))
            edge_to_elements.setdefault(edge, []).append(element)

    smooth_neighbors: dict[tuple[int, int], set[int]] = {
        (int(node), element): set()
        for element, row in enumerate(elements)
        for node in row
    }
    smooth_interior_edges = 0
    angular_creases = 0
    region_boundaries = 0
    boundary_edges = 0
    nonmanifold_edges = 0
    crease_cosine = math.cos(math.radians(limits.crease_angle_degrees))
    for edge, attached in edge_to_elements.items():
        if len(attached) == 1:
            boundary_edges += 1
            continue
        if len(attached) != 2:
            nonmanifold_edges += 1
            continue
        first, second = attached
        if edge in blocked_edges:
            continue
        first_region = (int(parts[first]), int(sheets[first]), int(continuity[first]))
        second_region = (int(parts[second]), int(sheets[second]), int(continuity[second]))
        if first_region != second_region:
            region_boundaries += 1
            continue
        alignment = float(np.dot(connectivity_normals[first], connectivity_normals[second]))
        if alignment < crease_cosine:
            angular_creases += 1
            continue
        smooth_interior_edges += 1
        for node in edge:
            smooth_neighbors[(node, first)].add(second)
            smooth_neighbors[(node, second)].add(first)

    directors = np.empty((element_count, 4, 3), dtype=np.float64)
    component_sizes = np.empty((element_count, 4), dtype=np.int64)
    smoothing_component_count = 0
    incident_elements: dict[int, set[int]] = {}
    for element, row in enumerate(elements):
        for node in row:
            incident_elements.setdefault(int(node), set()).add(element)
    for node, incident in incident_elements.items():
        remaining = set(incident)
        while remaining:
            smoothing_component_count += 1
            seed = min(remaining)
            component: set[int] = set()
            pending = [seed]
            while pending:
                candidate = pending.pop()
                if candidate in component:
                    continue
                component.add(candidate)
                pending.extend(smooth_neighbors[(node, candidate)] - component)
            remaining.difference_update(component)

            weighted_normals: list[np.ndarray] = []
            for candidate in sorted(component):
                candidate_local = local_corner_by_node[candidate][node]
                weighted_normals.append(
                    corner_weights[candidate, candidate_local] * connectivity_normals[candidate]
                )
            averaged = np.asarray(
                [math.fsum(float(vector[axis]) for vector in weighted_normals) for axis in range(3)],
                dtype=np.float64,
            )
            averaged_norm = float(np.linalg.norm(averaged))
            if not math.isfinite(averaged_norm) or averaged_norm <= np.finfo(np.float64).tiny:
                raise ValueError("smooth director fan has a zero or unresolved weighted normal")
            director = averaged / averaged_norm
            for candidate in component:
                candidate_local = local_corner_by_node[candidate][node]
                directors[candidate, candidate_local] = director
                component_sizes[candidate, candidate_local] = len(component)

    element_quality = tuple(
        validate_element_corner_directors(
            corners[element],
            directors[element],
            limits=limits,
        )
        for element in range(element_count)
    )
    has_source_regions = any(value is not None for value in (part_indices, sheet_indices, continuity_indices))
    provenance = (
        DirectorProvenanceCode.SHEET_AWARE_MESH_RECONSTRUCTION
        if has_source_regions
        else DirectorProvenanceCode.LOCAL_ELEMENT_RECONSTRUCTION
    )
    provenance_codes = np.full(
        (element_count, 4),
        DIRECTOR_PROVENANCE_NUMERIC_CODE[provenance],
        dtype=np.uint8,
    )
    field_quality = DirectorFieldQuality(
        minimum_geometry_alignment=min(item.minimum_geometry_alignment for item in element_quality),
        maximum_corner_spread_degrees=max(item.maximum_pair_angle_degrees for item in element_quality),
        minimum_smoothing_component_size=int(np.min(component_sizes)),
        maximum_smoothing_component_size=int(np.max(component_sizes)),
        smoothing_component_count=smoothing_component_count,
        smooth_interior_edge_count=smooth_interior_edges,
        angular_crease_edge_count=angular_creases,
        declared_crease_edge_count=len(blocked_edges),
        region_boundary_edge_count=region_boundaries,
        boundary_edge_count=boundary_edges,
        nonmanifold_edge_count=nonmanifold_edges,
    )
    diagnostics = (
        "numeric_mesh_only",
        "element_corner_storage",
        "source_face_use_orientation_applied_upstream",
        f"provenance={provenance.value}",
        f"smooth_edges={smooth_interior_edges}",
        f"blocked_edges={len(blocked_edges)}",
        f"nonmanifold_edges={nonmanifold_edges}",
    )
    return PreparedDirectorField(
        directors=directors,
        provenance_codes=provenance_codes,
        element_quality=element_quality,
        quality=field_quality,
        diagnostics=diagnostics,
        numeric_fingerprint=numeric_payload_fingerprint(
            coordinates,
            elements,
            source_signs,
            parts,
            sheets,
            continuity,
            np.asarray(sorted(blocked_edges), dtype=np.int64).reshape((-1, 2)),
            directors,
        ),
        _factory_token=_PREPARED_DIRECTOR_FACTORY_TOKEN,
    )
