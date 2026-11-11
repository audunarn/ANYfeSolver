"""Private curved-reference geometry for the GE-B3 P4 research candidate.

This module owns geometry and reference-frame construction only.  It is not a
selectable formulation and is deliberately absent from :mod:`anyfesolver`
exports.  The mechanics that consume it must be separately preregistered and
qualified.

The centreline is the three-node quadratic Lagrange curve on ``[-1, 1]``.
Reference frames are supplied explicitly at all three nodes.  Between nodes we
use the independently derived
``PIECEWISE_SHORTEST_TRANSPORT_LINEAR_ROLL_V1`` policy.  On each half-cell it
shortest-transports the left nodal frame from the left tangent to the exact
quadratic tangent, then linearly introduces the unique admitted residual roll
needed to reach the right nodal frame.  This policy is not claimed to be a
published frame interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final

import numpy as np


GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID: Final = (
    "GE_BEAM3_CURVED_Q2_REFERENCE_GEOMETRY_SCHEMA_V1"
)
GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID: Final = (
    "PIECEWISE_SHORTEST_TRANSPORT_LINEAR_ROLL_V1_INDEPENDENT_DERIVATION"
)
GE_BEAM3_CURVED_REFERENCE_REVERSAL_ID: Final = (
    "XI_NEGATION_NODE_1_3_AND_FRAME_DIAG_NEG_POS_NEG_V1"
)
GE_BEAM3_CURVED_REFERENCE_REGULARITY_ID: Final = (
    "ANALYTIC_AFFINE_TANGENT_INTERVAL_MINIMUM_V1"
)

DEFAULT_REGULARITY_RELATIVE_TOLERANCE: Final = max(
    64.0 * np.finfo(np.float64).eps,
    1.0e-14,
)
DEFAULT_ROTATION_TOLERANCE: Final = 1.0e-11
DEFAULT_FRAME_TOLERANCE: Final = 1.0e-12
DEFAULT_HALF_FRAME_ROTATION_LIMIT: Final = 0.9 * np.pi

_NODE_PARAMETERS: Final = np.array((-1.0, 0.0, 1.0), dtype=np.float64)
_REVERSAL_FRAME_MAP: Final = np.diag((-1.0, 1.0, -1.0))


class GeBeam3CurvedReferenceError(ValueError):
    """Base error for inadmissible curved GE-B3 reference data."""


class GeBeam3CurvedGeometryError(GeBeam3CurvedReferenceError):
    """The quadratic reference centreline is not regular."""


class GeBeam3CurvedFrameError(GeBeam3CurvedReferenceError):
    """The authoritative nodal or interpolated reference frame is invalid."""


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        made = float(value)
    except (TypeError, ValueError) as exc:
        raise GeBeam3CurvedReferenceError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(made):
        raise GeBeam3CurvedReferenceError(f"{name} must be a finite scalar")
    return made


def _positive_tolerance(value: Any, *, name: str) -> float:
    made = _finite_scalar(value, name=name)
    if made <= 0.0:
        raise GeBeam3CurvedReferenceError(f"{name} must be strictly positive")
    return made


def _finite_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    try:
        made = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeBeam3CurvedReferenceError(
            f"{name} must have shape {shape} and contain finite binary64 values"
        ) from exc
    if made.shape != shape or not np.all(np.isfinite(made)):
        raise GeBeam3CurvedReferenceError(
            f"{name} must have shape {shape} and contain finite binary64 values"
        )
    return np.array(made, dtype=np.float64, copy=True)


def _readonly(value: np.ndarray) -> np.ndarray:
    made = np.array(value, dtype=np.float64, copy=True)
    made.setflags(write=False)
    return made


def _parameter(value: Any) -> float:
    xi = _finite_scalar(value, name="xi")
    if xi < -1.0 or xi > 1.0:
        raise GeBeam3CurvedReferenceError("xi must lie in the closed interval [-1, 1]")
    return 0.0 if xi == 0.0 else xi


def p2_shape_functions(xi: Any) -> np.ndarray:
    """Return ``[N1, N2, N3]`` for nodes at ``[-1, 0, 1]``."""

    value = _parameter(xi)
    return np.array(
        (
            0.5 * value * (value - 1.0),
            1.0 - value * value,
            0.5 * value * (value + 1.0),
        ),
        dtype=np.float64,
    )


def p2_shape_derivatives(xi: Any) -> np.ndarray:
    """Return derivatives of the quadratic shape functions with respect to xi."""

    value = _parameter(xi)
    return np.array((value - 0.5, -2.0 * value, value + 0.5), dtype=np.float64)


def reversal_frame_map() -> np.ndarray:
    """Return the proper local-frame map used when connectivity is reversed.

    The tangent and third director reverse while the authoritative physical
    material-orientation direction (the second director) is preserved.
    """

    return np.array(_REVERSAL_FRAME_MAP, copy=True)


def reverse_reference_coordinates(coordinates: Any) -> np.ndarray:
    """Reverse ``[end, midpoint, end]`` reference coordinates."""

    return _finite_array(coordinates, (3, 3), name="coordinates")[::-1].copy()


def reverse_reference_triads(nodal_triads: Any) -> np.ndarray:
    """Reverse nodal triads while preserving physical material orientation."""

    triads = _finite_array(nodal_triads, (3, 3, 3), name="nodal_triads")
    return np.einsum("nij,jk->nik", triads[::-1], _REVERSAL_FRAME_MAP)


def _binary64(value: float) -> str:
    made = float(value)
    if not np.isfinite(made):
        raise GeBeam3CurvedReferenceError("canonical data cannot contain nonfinite values")
    if made == 0.0:
        made = 0.0
    return made.hex()


def _binary64_array(value: np.ndarray) -> list[Any]:
    if value.ndim == 1:
        return [_binary64(entry) for entry in value]
    return [_binary64_array(row) for row in value]


def canonical_json_bytes(value: Any) -> bytes:
    """Encode this module's already-normalized plain data deterministically."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True)
class CurvedBeam3RegularityCertificate:
    """Analytic minimum certificate for ``||r0,xi||`` over ``[-1, 1]``."""

    affine_constant: tuple[float, float, float]
    affine_linear: tuple[float, float, float]
    minimizer_xi: float
    minimum_jacobian_squared: float
    characteristic_length: float
    minimum_admissible_jacobian: float

    @property
    def minimum_jacobian(self) -> float:
        return float(np.sqrt(self.minimum_jacobian_squared))

    def canonical_data(self) -> dict[str, Any]:
        return {
            "affine_constant_binary64": [_binary64(value) for value in self.affine_constant],
            "affine_linear_binary64": [_binary64(value) for value in self.affine_linear],
            "characteristic_length_binary64": _binary64(self.characteristic_length),
            "minimum_admissible_jacobian_binary64": _binary64(
                self.minimum_admissible_jacobian
            ),
            "minimum_jacobian_squared_binary64": _binary64(
                self.minimum_jacobian_squared
            ),
            "minimizer_xi_binary64": _binary64(self.minimizer_xi),
            "regularity_id": GE_BEAM3_CURVED_REFERENCE_REGULARITY_ID,
        }


@dataclass(frozen=True)
class CurvedBeam3ReferenceStation:
    """One immutable point, tangent, Jacobian and material frame evaluation."""

    xi: float
    position: np.ndarray
    derivative: np.ndarray
    jacobian: float
    frame: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _readonly(self.position))
        object.__setattr__(self, "derivative", _readonly(self.derivative))
        object.__setattr__(self, "frame", _readonly(self.frame))


def _characteristic_length(coordinates: np.ndarray) -> float:
    lengths = (
        np.linalg.norm(coordinates[1] - coordinates[0]),
        np.linalg.norm(coordinates[2] - coordinates[1]),
        np.linalg.norm(coordinates[2] - coordinates[0]),
    )
    return float(max(lengths))


def _regularity_certificate(
    coordinates: np.ndarray,
    relative_tolerance: float,
) -> CurvedBeam3RegularityCertificate:
    # r0,xi = a + xi*b exactly for the frozen P2 basis.
    affine_constant = 0.5 * (coordinates[2] - coordinates[0])
    affine_linear = coordinates[0] - 2.0 * coordinates[1] + coordinates[2]
    linear_squared = float(affine_linear @ affine_linear)
    if linear_squared == 0.0:
        minimizer = 0.0
    else:
        unconstrained = -float(affine_constant @ affine_linear) / linear_squared
        minimizer = float(np.clip(unconstrained, -1.0, 1.0))
        if minimizer == 0.0:
            minimizer = 0.0
    minimum_vector = affine_constant + minimizer * affine_linear
    minimum_squared = float(minimum_vector @ minimum_vector)
    scale = _characteristic_length(coordinates)
    if not np.isfinite(scale) or scale <= 0.0:
        raise GeBeam3CurvedGeometryError(
            "curved GE-B3 reference coordinates have zero characteristic length"
        )
    threshold = float(relative_tolerance * scale)
    # Strict inequality is deliberate: a boundary value is inadmissible.
    if not np.isfinite(minimum_squared) or minimum_squared <= threshold * threshold:
        raise GeBeam3CurvedGeometryError(
            "curved GE-B3 quadratic reference tangent is zero or below the "
            "scale-aware interval regularity threshold"
        )
    return CurvedBeam3RegularityCertificate(
        affine_constant=tuple(float(value) for value in affine_constant),
        affine_linear=tuple(float(value) for value in affine_linear),
        minimizer_xi=minimizer,
        minimum_jacobian_squared=minimum_squared,
        characteristic_length=scale,
        minimum_admissible_jacobian=threshold,
    )


def _validate_nodal_triads(
    triads: np.ndarray,
    coordinates: np.ndarray,
    *,
    rotation_tolerance: float,
) -> None:
    identity = np.eye(3)
    for index, (xi, triad) in enumerate(zip(_NODE_PARAMETERS, triads)):
        orthogonality_error = float(np.linalg.norm(triad.T @ triad - identity, ord=np.inf))
        determinant = float(np.linalg.det(triad))
        if (
            orthogonality_error > rotation_tolerance
            or abs(determinant - 1.0) > rotation_tolerance
        ):
            raise GeBeam3CurvedFrameError(
                f"nodal_triads[{index}] must be a proper SO(3) rotation"
            )
        tangent = p2_shape_derivatives(xi) @ coordinates
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm == 0.0:  # The interval certificate should catch this first.
            raise GeBeam3CurvedGeometryError(
                f"curved GE-B3 reference tangent vanishes at node {index + 1}"
            )
        unit_tangent = tangent / tangent_norm
        if float(np.linalg.norm(triad[:, 0] - unit_tangent)) > rotation_tolerance:
            raise GeBeam3CurvedFrameError(
                f"nodal_triads[{index}] first axis must match the oriented reference tangent"
            )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    cross = _skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _axis_rotation_derivative(
    axis: np.ndarray,
    axis_derivative: np.ndarray,
    angle: float,
    angle_derivative: float,
) -> np.ndarray:
    """Differentiate Rodrigues' formula for a moving unit axis."""

    cross = _skew(axis)
    cross_derivative = _skew(axis_derivative)
    sine = float(np.sin(angle))
    cosine = float(np.cos(angle))
    square = cross @ cross
    square_derivative = cross_derivative @ cross + cross @ cross_derivative
    return (
        cosine * angle_derivative * cross
        + sine * cross_derivative
        + sine * angle_derivative * square
        + (1.0 - cosine) * square_derivative
    )


def _shortest_tangent_transport(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the unique admitted shortest SO(3) rotation from left to right."""

    cosine = float(np.clip(left @ right, -1.0, 1.0))
    denominator = 1.0 + cosine
    if denominator <= 1.0 + float(np.cos(DEFAULT_HALF_FRAME_ROTATION_LIMIT)):
        raise GeBeam3CurvedFrameError(
            "reference half-cell tangent turn reaches the frozen 0.9*pi branch limit"
        )
    cross = _skew(np.cross(left, right))
    return np.eye(3) + cross + (cross @ cross) / denominator


def _shortest_tangent_transport_derivative(
    left: np.ndarray,
    right: np.ndarray,
    right_derivative: np.ndarray,
) -> np.ndarray:
    """Differentiate the admitted shortest transport with fixed ``left``."""

    cosine = float(np.clip(left @ right, -1.0, 1.0))
    denominator = 1.0 + cosine
    if denominator <= 1.0 + float(np.cos(DEFAULT_HALF_FRAME_ROTATION_LIMIT)):
        raise GeBeam3CurvedFrameError(
            "reference half-cell tangent turn reaches the frozen 0.9*pi branch limit"
        )
    cross = _skew(np.cross(left, right))
    cross_derivative = _skew(np.cross(left, right_derivative))
    denominator_derivative = float(left @ right_derivative)
    square = cross @ cross
    square_derivative = cross_derivative @ cross + cross @ cross_derivative
    return (
        cross_derivative
        + square_derivative / denominator
        - square * denominator_derivative / (denominator * denominator)
    )


def _half_frame_data(
    coordinates: np.ndarray,
    triads: np.ndarray,
    cell: int,
    *,
    rotation_tolerance: float,
) -> tuple[float, float]:
    left = cell
    right = cell + 1
    left_tangent = triads[left, :, 0]
    right_tangent = triads[right, :, 0]
    tangent_turn = float(
        np.arccos(np.clip(float(left_tangent @ right_tangent), -1.0, 1.0))
    )
    if tangent_turn >= DEFAULT_HALF_FRAME_ROTATION_LIMIT:
        raise GeBeam3CurvedFrameError(
            f"reference half-cell {cell + 1} tangent turn reaches the frozen 0.9*pi branch limit"
        )
    transport = _shortest_tangent_transport(left_tangent, right_tangent)
    base_right = transport @ triads[left]
    cosine = float(base_right[:, 1] @ triads[right, :, 1])
    sine = float(base_right[:, 2] @ triads[right, :, 1])
    residual_roll = float(np.arctan2(sine, cosine))
    if abs(residual_roll) >= DEFAULT_HALF_FRAME_ROTATION_LIMIT:
        raise GeBeam3CurvedFrameError(
            f"reference half-cell {cell + 1} residual roll reaches the frozen 0.9*pi branch limit"
        )
    reconstructed = _axis_rotation(right_tangent, residual_roll) @ base_right
    if float(np.linalg.norm(reconstructed - triads[right], ord=np.inf)) > 8.0 * rotation_tolerance:
        raise GeBeam3CurvedFrameError(
            f"reference half-cell {cell + 1} nodal frames do not admit the frozen shortest-transport roll map"
        )
    return tangent_turn, residual_roll


class CurvedBeam3ReferenceGeometry:
    """Validated quadratic centreline and authoritative reference-frame field.

    This is a private P4 building block.  It carries no mechanics, section,
    solver, selector, serialization, or qualification authority.
    """

    __slots__ = (
        "_coordinates",
        "_half_frame_data",
        "_nodal_triads",
        "_regularity",
        "_regularity_relative_tolerance",
        "_rotation_tolerance",
        "_frame_tolerance",
    )

    def __init__(
        self,
        coordinates: Any,
        nodal_triads: Any,
        *,
        regularity_relative_tolerance: float = DEFAULT_REGULARITY_RELATIVE_TOLERANCE,
        rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE,
        frame_tolerance: float = DEFAULT_FRAME_TOLERANCE,
    ) -> None:
        regularity_tolerance = _positive_tolerance(
            regularity_relative_tolerance,
            name="regularity_relative_tolerance",
        )
        proper_tolerance = _positive_tolerance(
            rotation_tolerance,
            name="rotation_tolerance",
        )
        interpolation_tolerance = _positive_tolerance(
            frame_tolerance,
            name="frame_tolerance",
        )
        if regularity_tolerance < DEFAULT_REGULARITY_RELATIVE_TOLERANCE:
            raise GeBeam3CurvedReferenceError(
                "regularity_relative_tolerance cannot weaken the frozen admission threshold"
            )
        if proper_tolerance > DEFAULT_ROTATION_TOLERANCE:
            raise GeBeam3CurvedReferenceError(
                "rotation_tolerance cannot weaken the frozen SO(3) admission threshold"
            )
        if interpolation_tolerance > DEFAULT_FRAME_TOLERANCE:
            raise GeBeam3CurvedReferenceError(
                "frame_tolerance cannot weaken the frozen frame agreement threshold"
            )
        made_coordinates = _finite_array(coordinates, (3, 3), name="coordinates")
        made_triads = _finite_array(nodal_triads, (3, 3, 3), name="nodal_triads")
        regularity = _regularity_certificate(made_coordinates, regularity_tolerance)
        _validate_nodal_triads(
            made_triads,
            made_coordinates,
            rotation_tolerance=proper_tolerance,
        )
        half_frame_data = tuple(
            _half_frame_data(
                made_coordinates,
                made_triads,
                cell,
                rotation_tolerance=proper_tolerance,
            )
            for cell in (0, 1)
        )

        self._coordinates = _readonly(made_coordinates)
        self._nodal_triads = _readonly(made_triads)
        self._regularity = regularity
        self._half_frame_data = tuple(
            (float(turn), float(roll)) for turn, roll in half_frame_data
        )
        self._regularity_relative_tolerance = regularity_tolerance
        self._rotation_tolerance = proper_tolerance
        self._frame_tolerance = interpolation_tolerance

    @property
    def coordinates(self) -> np.ndarray:
        return np.array(self._coordinates, copy=True)

    @property
    def nodal_triads(self) -> np.ndarray:
        return np.array(self._nodal_triads, copy=True)

    @property
    def regularity(self) -> CurvedBeam3RegularityCertificate:
        return self._regularity

    @property
    def frame_interpolation_id(self) -> str:
        return GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID

    def position(self, xi: Any) -> np.ndarray:
        return p2_shape_functions(xi) @ self._coordinates

    def derivative(self, xi: Any) -> np.ndarray:
        return p2_shape_derivatives(xi) @ self._coordinates

    def jacobian(self, xi: Any) -> float:
        return float(np.linalg.norm(self.derivative(xi)))

    def tangent(self, xi: Any) -> np.ndarray:
        derivative = self.derivative(xi)
        return derivative / np.linalg.norm(derivative)

    def tangent_derivative(self, xi: Any) -> np.ndarray:
        """Return the analytic derivative ``dt0/dxi``."""

        value = _parameter(xi)
        derivative = self.derivative(value)
        jacobian = float(np.linalg.norm(derivative))
        tangent = derivative / jacobian
        affine_linear = np.asarray(self._regularity.affine_linear, dtype=np.float64)
        return (
            affine_linear - tangent * float(tangent @ affine_linear)
        ) / jacobian

    def _half_cell(self, xi: float) -> tuple[int, int, float]:
        if xi < 0.0:
            return 0, 1, xi + 1.0
        return 1, 2, xi

    def _half_cell_with_trace(self, xi: float, trace: str) -> tuple[int, int, float]:
        normalized_trace = str(trace).strip().upper()
        if normalized_trace not in {"VALUE", "LEFT", "RIGHT"}:
            raise GeBeam3CurvedReferenceError(
                "trace must be VALUE, LEFT, or RIGHT"
            )
        if xi == 0.0:
            if normalized_trace == "VALUE":
                raise GeBeam3CurvedReferenceError(
                    "xi=0 has two analytic frame-derivative traces; choose LEFT or RIGHT"
                )
            return (0, 1, 1.0) if normalized_trace == "LEFT" else (1, 2, 0.0)
        if normalized_trace != "VALUE":
            raise GeBeam3CurvedReferenceError(
                "LEFT or RIGHT trace is valid only at the midpoint xi=0"
            )
        return self._half_cell(xi)

    def frame(self, xi: Any) -> np.ndarray:
        value = _parameter(xi)
        for index, node_xi in enumerate(_NODE_PARAMETERS):
            if value == node_xi:
                return np.array(self._nodal_triads[index], copy=True)
        left, _right, fraction = self._half_cell(value)
        first = self.tangent(value)
        left_tangent = self._nodal_triads[left, :, 0]
        base = _shortest_tangent_transport(left_tangent, first) @ self._nodal_triads[left]
        residual_roll = self._half_frame_data[left][1]
        frame = _axis_rotation(first, fraction * residual_roll) @ base
        # Nodal frames are admitted to the registered SO(3)/tangent tolerance.
        # The interpolant must not apply the tighter independent-checker frame
        # agreement tolerance as a second, later admission test.
        if float(np.linalg.norm(frame[:, 0] - first)) > self._rotation_tolerance:
            raise GeBeam3CurvedFrameError(
                "reference frame interpolation lost tangent alignment"
            )
        return frame

    def frame_derivative(self, xi: Any, *, trace: str = "VALUE") -> np.ndarray:
        """Return the analytic material-frame derivative ``dR0/dxi``.

        The reference frame is continuous but intentionally piecewise smooth.
        Its two derivatives at the macro midpoint are therefore selected
        explicitly with ``trace="LEFT"`` or ``trace="RIGHT"``.
        """

        value = _parameter(xi)
        left, _right, fraction = self._half_cell_with_trace(value, trace)
        tangent = self.tangent(value)
        tangent_derivative = self.tangent_derivative(value)
        left_tangent = self._nodal_triads[left, :, 0]
        transport = _shortest_tangent_transport(left_tangent, tangent)
        transport_derivative = _shortest_tangent_transport_derivative(
            left_tangent,
            tangent,
            tangent_derivative,
        )
        left_frame = self._nodal_triads[left]
        base = transport @ left_frame
        base_derivative = transport_derivative @ left_frame
        residual_roll = self._half_frame_data[left][1]
        angle = fraction * residual_roll
        rotation = _axis_rotation(tangent, angle)
        rotation_derivative = _axis_rotation_derivative(
            tangent,
            tangent_derivative,
            angle,
            residual_roll,
        )
        made = rotation_derivative @ base + rotation @ base_derivative
        frame = self.frame(value)
        spin = frame.T @ made
        if float(np.linalg.norm(spin + spin.T, ord=np.inf)) > 64.0 * self._frame_tolerance:
            raise GeBeam3CurvedFrameError(
                "analytic reference-frame derivative is not tangent to SO(3)"
            )
        return made

    def intrinsic_curvature(self, xi: Any, *, trace: str = "VALUE") -> np.ndarray:
        """Return ``axl(R0.T @ dR0/ds0)`` from analytic derivatives."""

        value = _parameter(xi)
        frame = self.frame(value)
        frame_derivative = self.frame_derivative(value, trace=trace)
        spin = frame.T @ frame_derivative / self.jacobian(value)
        skew_spin = 0.5 * (spin - spin.T)
        return np.array(
            (skew_spin[2, 1], skew_spin[0, 2], skew_spin[1, 0]),
            dtype=np.float64,
        )

    def station(self, xi: Any) -> CurvedBeam3ReferenceStation:
        value = _parameter(xi)
        derivative = self.derivative(value)
        return CurvedBeam3ReferenceStation(
            xi=value,
            position=self.position(value),
            derivative=derivative,
            jacobian=float(np.linalg.norm(derivative)),
            frame=self.frame(value),
        )

    def reversed(self) -> "CurvedBeam3ReferenceGeometry":
        return type(self)(
            reverse_reference_coordinates(self._coordinates),
            reverse_reference_triads(self._nodal_triads),
            regularity_relative_tolerance=self._regularity_relative_tolerance,
            rotation_tolerance=self._rotation_tolerance,
            frame_tolerance=self._frame_tolerance,
        )

    def rigidly_transformed(
        self,
        rotation: Any,
        translation: Any,
    ) -> "CurvedBeam3ReferenceGeometry":
        proper = _finite_array(rotation, (3, 3), name="rotation")
        identity_error = float(np.linalg.norm(proper.T @ proper - np.eye(3), ord=np.inf))
        determinant = float(np.linalg.det(proper))
        if (
            identity_error > self._rotation_tolerance
            or abs(determinant - 1.0) > self._rotation_tolerance
        ):
            raise GeBeam3CurvedFrameError("rotation must be a proper SO(3) matrix")
        offset = _finite_array(translation, (3,), name="translation")
        coordinates = np.einsum("ij,nj->ni", proper, self._coordinates) + offset
        triads = np.einsum("ij,njk->nik", proper, self._nodal_triads)
        return type(self)(
            coordinates,
            triads,
            regularity_relative_tolerance=self._regularity_relative_tolerance,
            rotation_tolerance=self._rotation_tolerance,
            frame_tolerance=self._frame_tolerance,
        )

    def canonical_data(self) -> dict[str, Any]:
        return {
            "coordinates_binary64": _binary64_array(self._coordinates),
            "frame_interpolation": {
                "authority": "INDEPENDENT_DERIVATION",
                "id": GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID,
            },
            "frame_branch_data_by_half_cell": [
                {
                    "residual_roll_binary64": _binary64(roll),
                    "tangent_turn_binary64": _binary64(turn),
                }
                for turn, roll in self._half_frame_data
            ],
            "node_parameter_coordinates_binary64": _binary64_array(_NODE_PARAMETERS),
            "nodal_triads_binary64": _binary64_array(self._nodal_triads),
            "regularity": self._regularity.canonical_data(),
            "reversal_id": GE_BEAM3_CURVED_REFERENCE_REVERSAL_ID,
            "schema": GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID,
            "tolerances_binary64": {
                "frame": _binary64(self._frame_tolerance),
                "regularity_relative": _binary64(
                    self._regularity_relative_tolerance
                ),
                "rotation": _binary64(self._rotation_tolerance),
            },
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_data())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest().upper()


__all__ = [
    "CurvedBeam3ReferenceGeometry",
    "CurvedBeam3ReferenceStation",
    "CurvedBeam3RegularityCertificate",
    "DEFAULT_FRAME_TOLERANCE",
    "DEFAULT_HALF_FRAME_ROTATION_LIMIT",
    "DEFAULT_REGULARITY_RELATIVE_TOLERANCE",
    "DEFAULT_ROTATION_TOLERANCE",
    "GE_BEAM3_CURVED_REFERENCE_INTERPOLATION_ID",
    "GE_BEAM3_CURVED_REFERENCE_REGULARITY_ID",
    "GE_BEAM3_CURVED_REFERENCE_REVERSAL_ID",
    "GE_BEAM3_CURVED_REFERENCE_SCHEMA_ID",
    "GeBeam3CurvedFrameError",
    "GeBeam3CurvedGeometryError",
    "GeBeam3CurvedReferenceError",
    "canonical_json_bytes",
    "p2_shape_derivatives",
    "p2_shape_functions",
    "reverse_reference_coordinates",
    "reverse_reference_triads",
    "reversal_frame_map",
]
