"""Committed-state authority for the private mixed GE-Beam3 candidate.

This module owns the P2 restart boundary.  Arrays are hashed and serialized as
typed, little-endian, contiguous byte strings so JSON number formatting can
never change a mechanics identity.  The state remains private until the later
selector/package gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .beam_sections import generalized_beam_mass_matrix, generalized_beam_stiffness


GE_BEAM3_MIXED_CANDIDATE_ID = "CANDIDATE_GE_BEAM3_DC_MIXED_K1_MACRO_V2"
GE_BEAM3_MIXED_FORMULATION_ID = GE_BEAM3_MIXED_CANDIDATE_ID
GE_BEAM3_MIXED_CONDENSATION_ID = "TWO_CELL_18_LOCAL_SCHUR_V1"
GE_BEAM3_MIXED_QUADRATURE_ID = "K1_FORCE_GAUSS1_MOMENT_GAUSS2_V1"
GE_BEAM3_MIXED_REFERENCE_ID = "STRAIGHT_TWO_EQUAL_CELLS_V1"
GE_BEAM3_MIXED_ROTATION_ID = "ELEMENT_P0_VERTEX_HYBRID_SPATIAL_INCREMENT_V1"

STATE_SCHEMA = "anysolver.ge_beam3_mixed.committed_state.v1"
STATE_VERSION = 1
STATE_LAYOUT_ID = "GE_BEAM3_MIXED_Q18_SHARED_SO3_OPERATOR_STATION4_V1"
STATE_INTEGRITY_ID = "SHA256_CANONICAL_COMPLETE_STATE_EXCLUDING_DIGEST_V1"
SECTION_SCHEMA = "anysolver.ge-beam3-mixed.linear-section.v1"
COMMIT_STATUS = "COMMITTED_CONVERGED"
LOCAL_STATE_ROLE = "RECOVERED_CONDENSED_DIAGNOSTIC_NOT_RESTART_PREDICTOR"
SECTION_STATE_MODE = "STATELESS_LINEAR_SPD_6X6"
SECTION_STATION_IDS = ("CELL0_GP0", "CELL0_GP1", "CELL1_GP0", "CELL1_GP1")

_GEOMETRY_TOLERANCE = 1.0e-12
_SEMANTIC_TOLERANCE = 1.0e-11
_RELATIVE_ROTATION_LIMIT = 0.9 * math.pi
_SHA256 = re.compile(r"[0-9A-F]{64}\Z")
_HEX = re.compile(r"(?:[0-9A-F]{2})*\Z")

_ARRAY_FIELDS: dict[str, tuple[str, tuple[int, ...]]] = {
    "committed_local_moments": ("<f8", (2, 2, 3)),
    "committed_local_rotation_matrices": ("<f8", (2, 3, 3)),
    "committed_nodal_rotation_matrices": ("<f8", (3, 3, 3)),
    "committed_total_u": ("<f8", (18,)),
    "node_ids": ("<i8", (3,)),
    "reference_geometry": ("<f8", (3, 3)),
    "reference_triad": ("<f8", (3, 3)),
    "station_generalized_resultant": ("<f8", (4, 6)),
    "station_generalized_strain": ("<f8", (4, 6)),
}

_STATE_KEYS = frozenset(
    {
        "candidate_id",
        "commit_status",
        "committed_local_moments",
        "committed_local_rotation_matrices",
        "committed_nodal_rotation_matrices",
        "committed_total_u",
        "condensation_id",
        "element_id",
        "element_identity_sha256",
        "formulation_id",
        "local_state_role",
        "node_ids",
        "quadrature_id",
        "reference_geometry",
        "reference_id",
        "reference_triad",
        "rotation_id",
        "section_descriptor_sha256",
        "section_state_mode",
        "section_states",
        "section_station_ids",
        "state_integrity_id",
        "state_integrity_sha256",
        "state_layout_id",
        "state_schema",
        "state_version",
        "station_generalized_resultant",
        "station_generalized_strain",
    }
)

_STATE_ENUMS = {
    "candidate_id": GE_BEAM3_MIXED_CANDIDATE_ID,
    "commit_status": COMMIT_STATUS,
    "condensation_id": GE_BEAM3_MIXED_CONDENSATION_ID,
    "formulation_id": GE_BEAM3_MIXED_FORMULATION_ID,
    "local_state_role": LOCAL_STATE_ROLE,
    "quadrature_id": GE_BEAM3_MIXED_QUADRATURE_ID,
    "reference_id": GE_BEAM3_MIXED_REFERENCE_ID,
    "rotation_id": GE_BEAM3_MIXED_ROTATION_ID,
    "section_state_mode": SECTION_STATE_MODE,
    "state_integrity_id": STATE_INTEGRITY_ID,
    "state_layout_id": STATE_LAYOUT_ID,
    "state_schema": STATE_SCHEMA,
}


class GeBeam3MixedCommittedStateError(ValueError):
    """A mixed GE-Beam3 state is malformed, stale, foreign, or inconsistent."""


# Short compatibility spelling for element-side code.
GeBeam3MixedStateContractError = GeBeam3MixedCommittedStateError


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, np.ndarray):
        return encode_typed_array(value, path=path)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise GeBeam3MixedCommittedStateError(f"nonfinite canonical value at {path}")
        return value
    if isinstance(value, Mapping):
        made: dict[str, Any] = {}
        for key, member in value.items():
            if type(key) is not str:
                raise GeBeam3MixedCommittedStateError(f"non-string canonical key at {path}")
            if key in made:
                raise GeBeam3MixedCommittedStateError(f"duplicate canonical key {key!r} at {path}")
            made[key] = _canonical_value(member, path=f"{path}.{key}")
        return made
    if _is_sequence(value):
        return [_canonical_value(member, path=f"{path}[{index}]") for index, member in enumerate(value)]
    raise GeBeam3MixedCommittedStateError(
        f"unsupported canonical value {type(value).__name__} at {path}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return sorted compact UTF-8 JSON with one LF and typed arrays."""

    return (
        json.dumps(
            _canonical_value(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def strict_canonical_json_loads(raw: bytes) -> Any:
    """Decode only the single frozen canonical JSON representation."""

    if type(raw) is not bytes:
        raise GeBeam3MixedCommittedStateError("canonical state input must be bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise GeBeam3MixedCommittedStateError("canonical state must not contain a BOM")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        made: dict[str, Any] = {}
        for key, value in pairs:
            if key in made:
                raise GeBeam3MixedCommittedStateError(f"duplicate JSON key {key!r}")
            made[key] = value
        return made

    def reject_constant(token: str) -> None:
        raise GeBeam3MixedCommittedStateError(f"nonfinite JSON token {token}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeBeam3MixedCommittedStateError("canonical state JSON is invalid") from exc
    if canonical_json_bytes(value) != raw:
        raise GeBeam3MixedCommittedStateError("state bytes are not canonical")
    return value


def _canonical_dtype(dtype: Any) -> np.dtype[Any]:
    made = np.dtype(dtype)
    if made.kind == "f" and made.itemsize == 8:
        return np.dtype("<f8")
    if made.kind == "i" and made.itemsize == 8:
        return np.dtype("<i8")
    raise GeBeam3MixedCommittedStateError(
        "typed state array dtype must be float64 or int64"
    )


def _owned_array(
    value: Any,
    dtype: str,
    shape: tuple[int, ...],
    label: str,
    *,
    strict_runtime_dtype: bool = False,
) -> np.ndarray:
    if strict_runtime_dtype and not isinstance(value, np.ndarray):
        raise GeBeam3MixedCommittedStateError(f"{label} must be a typed numpy array")
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise GeBeam3MixedCommittedStateError(f"{label} is not an array") from exc
    expected = np.dtype(dtype)
    if strict_runtime_dtype and _canonical_dtype(source.dtype) != expected:
        raise GeBeam3MixedCommittedStateError(f"{label} has the wrong dtype")
    if source.shape != shape:
        raise GeBeam3MixedCommittedStateError(f"{label} must have shape {shape}")
    if expected.kind == "i":
        if source.dtype.kind not in "iu" or source.dtype.kind == "b":
            raise GeBeam3MixedCommittedStateError(f"{label} must contain signed int64 values")
        if source.dtype.kind == "u" and source.size and int(source.max()) > np.iinfo(np.int64).max:
            raise GeBeam3MixedCommittedStateError(f"{label} exceeds signed int64")
    else:
        if source.dtype.kind not in "fiu":
            raise GeBeam3MixedCommittedStateError(f"{label} must contain binary64 values")
    try:
        made = np.array(source, dtype=expected, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeBeam3MixedCommittedStateError(f"{label} cannot be represented exactly") from exc
    if expected.kind == "f" and not np.all(np.isfinite(made)):
        raise GeBeam3MixedCommittedStateError(f"{label} contains nonfinite values")
    if expected.kind == "i" and source.dtype.kind in "iu" and not np.array_equal(made, source):
        raise GeBeam3MixedCommittedStateError(f"{label} cannot be represented as signed int64")
    made.setflags(write=False)
    return made


def encode_typed_array(value: Any, *, path: str = "$") -> dict[str, Any]:
    """Encode one canonical C-contiguous little-endian f8/i8 array."""

    if not isinstance(value, np.ndarray):
        raise GeBeam3MixedCommittedStateError(f"typed array required at {path}")
    dtype = _canonical_dtype(value.dtype)
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise GeBeam3MixedCommittedStateError(f"nonfinite typed array at {path}")
    return {
        "data_hex": array.tobytes(order="C").hex().upper(),
        "dtype": dtype.str,
        "shape": [int(value) for value in array.shape],
    }


def decode_typed_array(
    value: Any,
    *,
    dtype: str,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """Decode one exact typed-array JSON object into an owned read-only array."""

    if not isinstance(value, Mapping) or set(value) != {"data_hex", "dtype", "shape"}:
        raise GeBeam3MixedCommittedStateError(f"{label} typed array object keys mismatch")
    if value["dtype"] != dtype:
        raise GeBeam3MixedCommittedStateError(f"{label} typed array dtype mismatch")
    raw_shape = value["shape"]
    if not isinstance(raw_shape, list) or any(type(item) is not int or item < 0 for item in raw_shape):
        raise GeBeam3MixedCommittedStateError(f"{label} typed array shape is malformed")
    if tuple(raw_shape) != shape:
        raise GeBeam3MixedCommittedStateError(f"{label} typed array shape mismatch")
    data_hex = value["data_hex"]
    if type(data_hex) is not str or _HEX.fullmatch(data_hex) is None:
        raise GeBeam3MixedCommittedStateError(f"{label} typed array data_hex is not uppercase hex")
    expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    if len(data_hex) != 2 * expected_bytes:
        raise GeBeam3MixedCommittedStateError(f"{label} typed array byte count mismatch")
    try:
        data = bytes.fromhex(data_hex)
    except ValueError as exc:  # defensive; the expression above is authoritative
        raise GeBeam3MixedCommittedStateError(f"{label} typed array data_hex is invalid") from exc
    made = np.frombuffer(data, dtype=np.dtype(dtype)).copy().reshape(shape)
    if np.dtype(dtype).kind == "f" and not np.all(np.isfinite(made)):
        raise GeBeam3MixedCommittedStateError(f"{label} contains nonfinite values")
    made.setflags(write=False)
    return made


def _state_array(value: Any, name: str) -> np.ndarray:
    dtype, shape = _ARRAY_FIELDS[name]
    if isinstance(value, Mapping):
        return decode_typed_array(value, dtype=dtype, shape=shape, label=name)
    return _owned_array(value, dtype, shape, name, strict_runtime_dtype=True)


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise GeBeam3MixedCommittedStateError(f"{label} must be uppercase SHA-256")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise GeBeam3MixedCommittedStateError(f"{label} must be an integer, not a boolean")
    return value


def _proper_rotation(matrix: np.ndarray, label: str) -> None:
    defect = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord=np.inf))
    determinant = float(np.linalg.det(matrix))
    if defect > 1.0e-12 or determinant <= 0.0 or abs(determinant - 1.0) > 1.0e-12:
        raise GeBeam3MixedCommittedStateError(f"{label} is not a proper rotation")


def _validate_reference(
    geometry: np.ndarray,
    triad: np.ndarray,
    orientation: Any,
    axis_direction: Any,
) -> tuple[np.ndarray, np.ndarray]:
    chord = geometry[2] - geometry[0]
    length = float(np.linalg.norm(chord))
    if not math.isfinite(length) or length <= 0.0:
        raise GeBeam3MixedCommittedStateError("reference geometry has coincident ends")
    if float(np.linalg.norm(geometry[1] - 0.5 * (geometry[0] + geometry[2]))) > _GEOMETRY_TOLERANCE * length:
        raise GeBeam3MixedCommittedStateError("reference middle node is not the exact midpoint within tolerance")
    first = geometry[1] - geometry[0]
    second = geometry[2] - geometry[1]
    if float(np.linalg.norm(np.cross(first, chord))) > _GEOMETRY_TOLERANCE * length * length:
        raise GeBeam3MixedCommittedStateError("reference geometry is not collinear")
    if float(np.linalg.norm(np.cross(second, chord))) > _GEOMETRY_TOLERANCE * length * length:
        raise GeBeam3MixedCommittedStateError("reference geometry is not collinear")
    _proper_rotation(triad, "reference_triad")
    e1 = chord / length
    if float(np.linalg.norm(triad[:, 0] - e1, ord=np.inf)) > _GEOMETRY_TOLERANCE:
        raise GeBeam3MixedCommittedStateError("reference triad first column is not chord-aligned")
    orient = _owned_array(orientation, "<f8", (3,), "reference_orientation")
    projected = orient - float(orient @ e1) * e1
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= _GEOMETRY_TOLERANCE * float(np.linalg.norm(orient)):
        raise GeBeam3MixedCommittedStateError("reference orientation is parallel to the chord")
    if float(np.linalg.norm(triad[:, 1] - projected / projected_norm, ord=np.inf)) > _GEOMETRY_TOLERANCE:
        raise GeBeam3MixedCommittedStateError("reference triad does not reproduce the physical orientation")
    axis = _owned_array(axis_direction, "<f8", (3,), "reference_axis_direction")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0 or abs(abs(float((axis / axis_norm) @ e1)) - 1.0) > _GEOMETRY_TOLERANCE:
        raise GeBeam3MixedCommittedStateError("reference axis authority is not parallel to the chord")
    return orient, axis


def _positive_definite_matrix(value: Any, label: str) -> np.ndarray:
    made = _owned_array(value, "<f8", (6, 6), label)
    diagonal = np.diag(made)
    if np.any(diagonal <= 0.0):
        raise GeBeam3MixedCommittedStateError(f"{label} must have a positive diagonal")
    scale = 1.0 / np.sqrt(diagonal)
    normalized = scale[:, None] * made * scale[None, :]
    if not np.allclose(normalized, normalized.T, rtol=1.0e-10, atol=1.0e-12):
        raise GeBeam3MixedCommittedStateError(f"{label} must be symmetric")
    if float(np.linalg.eigvalsh(0.5 * (normalized + normalized.T))[0]) <= 1.0e-12:
        raise GeBeam3MixedCommittedStateError(f"{label} must be positive definite")
    result = np.array(0.5 * (made + made.T), dtype="<f8", order="C", copy=True)
    result.setflags(write=False)
    return result


def section_descriptor_payload(
    *, section_name: str, section_stiffness: Any, section_mass: Any
) -> dict[str, Any]:
    if type(section_name) is not str:
        raise GeBeam3MixedCommittedStateError("section_name must be a string")
    return {
        "mass_matrix_per_reference_length": _positive_definite_matrix(section_mass, "section mass"),
        "name": section_name,
        "schema": SECTION_SCHEMA,
        "stiffness_matrix": _positive_definite_matrix(section_stiffness, "section stiffness"),
    }


def section_descriptor_sha256(
    *, section_name: str, section_stiffness: Any, section_mass: Any
) -> str:
    return canonical_sha256(
        section_descriptor_payload(
            section_name=section_name,
            section_stiffness=section_stiffness,
            section_mass=section_mass,
        )
    )


def element_identity_payload(
    *,
    node_ids: Any,
    reference_geometry: Any,
    reference_triad: Any,
    reference_orientation: Any,
    reference_axis_direction: Any,
    section_descriptor_sha256_value: str,
) -> dict[str, Any]:
    nodes = _owned_array(node_ids, "<i8", (3,), "node_ids")
    if len(set(int(value) for value in nodes)) != 3:
        raise GeBeam3MixedCommittedStateError("node_ids must contain three distinct IDs")
    geometry = _owned_array(reference_geometry, "<f8", (3, 3), "reference_geometry")
    triad = _owned_array(reference_triad, "<f8", (3, 3), "reference_triad")
    orientation, axis = _validate_reference(geometry, triad, reference_orientation, reference_axis_direction)
    return {
        "candidate_id": GE_BEAM3_MIXED_CANDIDATE_ID,
        "condensation_id": GE_BEAM3_MIXED_CONDENSATION_ID,
        "node_ids": nodes,
        "quadrature_id": GE_BEAM3_MIXED_QUADRATURE_ID,
        "reference_axis_direction": axis,
        "reference_geometry": geometry,
        "reference_id": GE_BEAM3_MIXED_REFERENCE_ID,
        "reference_orientation": orientation,
        "reference_triad": triad,
        "rotation_id": GE_BEAM3_MIXED_ROTATION_ID,
        "section_descriptor_sha256": _sha(
            section_descriptor_sha256_value, "section_descriptor_sha256"
        ),
        "state_layout_id": STATE_LAYOUT_ID,
    }


def element_identity_sha256(**kwargs: Any) -> str:
    return canonical_sha256(element_identity_payload(**kwargs))


def model_bound_identity(element: Any, mesh: Any) -> dict[str, Any]:
    """Recompute the complete P2 identity from an element and current model."""

    geometry = _owned_array(element.get_node_coordinates(mesh), "<f8", (3, 3), "reference_geometry")
    triad = _owned_array(element.reference_triad(mesh), "<f8", (3, 3), "reference_triad")
    # Identity follows the same physical-axis polarity convention as mechanics.
    # A reversed coupled element stores the effective transformed section, not
    # the untransformed section object.  Prefer the element-owned methods so the
    # identity cannot drift from its actual energy/mass operators.
    stiffness_provider = getattr(element, "_section_stiffness", None)
    mass_provider = getattr(element, "_section_mass", None)
    stiffness = (
        stiffness_provider(mesh)
        if callable(stiffness_provider)
        else generalized_beam_stiffness(element.generalized_section)
    )
    mass = (
        mass_provider(mesh)
        if callable(mass_provider)
        else generalized_beam_mass_matrix(element.generalized_section)
    )
    if mass is None:
        raise GeBeam3MixedCommittedStateError(
            "mixed GE-Beam3 P2 state requires an authoritative generalized section mass"
        )
    axis = getattr(element, "reference_axis_direction", None)
    if axis is None:
        # The element's state authority uses the directed reference chord for a
        # reversal-insensitive section whose constructor did not require an
        # explicit physical-axis authority.  Persist that concrete vector: the
        # frozen typed preimage does not admit null outside section_states.
        axis = triad[:, 0]
    descriptor = section_descriptor_payload(
        section_name=str(element.generalized_section.name),
        section_stiffness=stiffness,
        section_mass=mass,
    )
    descriptor_hash = canonical_sha256(descriptor)
    identity = element_identity_payload(
        node_ids=np.asarray(element.node_ids, dtype="<i8"),
        reference_geometry=geometry,
        reference_triad=triad,
        reference_orientation=element.reference_orientation,
        reference_axis_direction=axis,
        section_descriptor_sha256_value=descriptor_hash,
    )
    return {
        "element_identity_payload": identity,
        "element_identity_sha256": canonical_sha256(identity),
        "reference_geometry": geometry,
        "reference_triad": triad,
        "section_descriptor": descriptor,
        "section_descriptor_sha256": descriptor_hash,
        "section_mass": descriptor["mass_matrix_per_reference_length"],
        "section_stiffness": descriptor["stiffness_matrix"],
    }


def _normalized_close(left: np.ndarray, right: np.ndarray, tolerance: float = _SEMANTIC_TOLERANCE) -> bool:
    # ``numpy.linalg.norm(..., ord=inf)`` is defined only through matrices;
    # registered state arrays also include rank-three cell/endpoint records.
    # The componentwise infinity norm is the intended normalized invariant for
    # every registered shape.
    scale = max(
        1.0,
        float(np.max(np.abs(left), initial=0.0)),
        float(np.max(np.abs(right), initial=0.0)),
    )
    difference = float(np.max(np.abs(left - right), initial=0.0))
    return bool(difference <= tolerance * scale)


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order="C") == right.tobytes(order="C"))


def _validate_rotations(state: Mapping[str, Any]) -> None:
    triad = state["reference_triad"]
    nodal = state["committed_nodal_rotation_matrices"]
    local = state["committed_local_rotation_matrices"]
    for index, matrix in enumerate(nodal):
        _proper_rotation(matrix, f"committed_nodal_rotation_matrices[{index}]")
    for index, matrix in enumerate(local):
        _proper_rotation(matrix, f"committed_local_rotation_matrices[{index}]")
    absolute = np.einsum("nij,jk->nik", nodal, triad)
    for left, right in ((0, 1), (1, 2)):
        relative = absolute[left].T @ absolute[right]
        cosine = max(-1.0, min(1.0, 0.5 * (float(np.trace(relative)) - 1.0)))
        if math.acos(cosine) >= _RELATIVE_ROTATION_LIMIT:
            raise GeBeam3MixedCommittedStateError(
                f"adjacent absolute material frames {left + 1}-{right + 1} exceed the rotation domain"
            )


def _normalize_state_structure(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise GeBeam3MixedCommittedStateError("committed mixed GE-Beam3 state must be a mapping")
    if set(state) != _STATE_KEYS:
        missing = sorted(_STATE_KEYS - set(state))
        extra = sorted(set(state) - _STATE_KEYS)
        raise GeBeam3MixedCommittedStateError(
            f"state keys mismatch; missing={missing}, extra={extra}"
        )
    made: dict[str, Any] = {}
    for key, expected in _STATE_ENUMS.items():
        if type(state[key]) is not str or state[key] != expected:
            raise GeBeam3MixedCommittedStateError(f"state enum {key} mismatch")
        made[key] = state[key]
    made["state_version"] = _integer(state["state_version"], "state_version")
    if made["state_version"] != STATE_VERSION:
        raise GeBeam3MixedCommittedStateError("state version mismatch")
    made["element_id"] = _integer(state["element_id"], "element_id")
    made["element_identity_sha256"] = _sha(state["element_identity_sha256"], "element_identity_sha256")
    made["section_descriptor_sha256"] = _sha(state["section_descriptor_sha256"], "section_descriptor_sha256")
    made["state_integrity_sha256"] = _sha(state["state_integrity_sha256"], "state_integrity_sha256")
    if type(state["section_station_ids"]) is not list or tuple(state["section_station_ids"]) != SECTION_STATION_IDS:
        raise GeBeam3MixedCommittedStateError("section station IDs mismatch")
    made["section_station_ids"] = list(SECTION_STATION_IDS)
    if type(state["section_states"]) is not list or state["section_states"] != [None, None, None, None]:
        raise GeBeam3MixedCommittedStateError("stateless section_states must be exactly four nulls")
    made["section_states"] = [None, None, None, None]
    for name in _ARRAY_FIELDS:
        made[name] = _state_array(state[name], name)
    return made


def _state_integrity_sha256(state: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in state.items() if key != "state_integrity_sha256"}
    )


def seal_committed_ge_beam3_mixed_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Own, normalize, and integrity-seal a complete state mapping."""

    if not isinstance(state, Mapping):
        raise GeBeam3MixedCommittedStateError("committed mixed GE-Beam3 state must be a mapping")
    expected_without_digest = _STATE_KEYS - {"state_integrity_sha256"}
    supplied = set(state)
    if supplied == expected_without_digest:
        provisional = dict(state)
        provisional["state_integrity_sha256"] = "0" * 64
    elif supplied == _STATE_KEYS:
        provisional = dict(state)
    else:
        missing = sorted(expected_without_digest - supplied)
        extra = sorted(supplied - _STATE_KEYS)
        raise GeBeam3MixedCommittedStateError(
            f"state keys mismatch before sealing; missing={missing}, extra={extra}"
        )
    normalized = _normalize_state_structure(provisional)
    normalized["state_integrity_sha256"] = _state_integrity_sha256(normalized)
    return normalized


def initialize_ge_beam3_mixed_state(
    *,
    element_id: int,
    node_ids: Any,
    reference_geometry: Any,
    reference_triad: Any,
    reference_orientation: Any,
    reference_axis_direction: Any,
    section_name: str,
    section_stiffness: Any,
    section_mass: Any,
    committed_total_u: Any | None = None,
    committed_nodal_rotation_matrices: Any | None = None,
    committed_local_rotation_matrices: Any | None = None,
    committed_local_moments: Any | None = None,
    station_generalized_strain: Any | None = None,
    station_generalized_resultant: Any | None = None,
) -> dict[str, Any]:
    """Build an owned sealed committed state from model and recovered fields."""

    _integer(element_id, "element_id")
    nodes = _owned_array(node_ids, "<i8", (3,), "node_ids")
    geometry = _owned_array(reference_geometry, "<f8", (3, 3), "reference_geometry")
    triad = _owned_array(reference_triad, "<f8", (3, 3), "reference_triad")
    descriptor = section_descriptor_payload(
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
    )
    descriptor_hash = canonical_sha256(descriptor)
    identity_hash = element_identity_sha256(
        node_ids=nodes,
        reference_geometry=geometry,
        reference_triad=triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_descriptor_sha256_value=descriptor_hash,
    )
    total_u = np.zeros(18, dtype="<f8") if committed_total_u is None else committed_total_u
    nodal = np.repeat(np.eye(3, dtype="<f8")[None, :, :], 3, axis=0) if committed_nodal_rotation_matrices is None else committed_nodal_rotation_matrices
    local = np.repeat(triad[None, :, :], 2, axis=0) if committed_local_rotation_matrices is None else committed_local_rotation_matrices
    moments = np.zeros((2, 2, 3), dtype="<f8") if committed_local_moments is None else committed_local_moments
    strains = np.zeros((4, 6), dtype="<f8") if station_generalized_strain is None else station_generalized_strain
    resultants = np.asarray(strains, dtype="<f8") @ np.asarray(descriptor["stiffness_matrix"], dtype="<f8").T if station_generalized_resultant is None else station_generalized_resultant
    state = {
        **_STATE_ENUMS,
        "committed_local_moments": _owned_array(moments, "<f8", (2, 2, 3), "committed_local_moments"),
        "committed_local_rotation_matrices": _owned_array(local, "<f8", (2, 3, 3), "committed_local_rotation_matrices"),
        "committed_nodal_rotation_matrices": _owned_array(nodal, "<f8", (3, 3, 3), "committed_nodal_rotation_matrices"),
        "committed_total_u": _owned_array(total_u, "<f8", (18,), "committed_total_u"),
        "element_id": element_id,
        "element_identity_sha256": identity_hash,
        "node_ids": nodes,
        "reference_geometry": geometry,
        "reference_triad": triad,
        "section_descriptor_sha256": descriptor_hash,
        "section_states": [None, None, None, None],
        "section_station_ids": list(SECTION_STATION_IDS),
        "state_version": STATE_VERSION,
        "station_generalized_resultant": _owned_array(resultants, "<f8", (4, 6), "station_generalized_resultant"),
        "station_generalized_strain": _owned_array(strains, "<f8", (4, 6), "station_generalized_strain"),
    }
    sealed = seal_committed_ge_beam3_mixed_state(state)
    return validate_committed_ge_beam3_mixed_state(
        sealed,
        element_id=element_id,
        node_ids=nodes,
        reference_geometry=geometry,
        reference_triad=triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_name=section_name,
        section_stiffness=descriptor["stiffness_matrix"],
        section_mass=descriptor["mass_matrix_per_reference_length"],
        expected_committed_total_u=total_u,
    )


def validate_committed_ge_beam3_mixed_state(
    state: Mapping[str, Any],
    *,
    element_id: int,
    node_ids: Any,
    reference_geometry: Any,
    reference_triad: Any,
    reference_orientation: Any,
    reference_axis_direction: Any,
    section_name: str,
    section_stiffness: Any,
    section_mass: Any,
    expected_committed_total_u: Any | None = None,
    expected_local_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate exact identity/integrity and local constitutive invariants."""

    normalized = _normalize_state_structure(state)
    if normalized["state_integrity_sha256"] != _state_integrity_sha256(normalized):
        raise GeBeam3MixedCommittedStateError("committed state integrity mismatch")
    expected_element_id = _integer(element_id, "element_id")
    nodes = _owned_array(node_ids, "<i8", (3,), "expected node_ids")
    geometry = _owned_array(reference_geometry, "<f8", (3, 3), "expected reference_geometry")
    triad = _owned_array(reference_triad, "<f8", (3, 3), "expected reference_triad")
    descriptor = section_descriptor_payload(
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
    )
    descriptor_hash = canonical_sha256(descriptor)
    identity_hash = element_identity_sha256(
        node_ids=nodes,
        reference_geometry=geometry,
        reference_triad=triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_descriptor_sha256_value=descriptor_hash,
    )
    if normalized["element_id"] != expected_element_id:
        raise GeBeam3MixedCommittedStateError("element ID binding mismatch")
    if not _bitwise_equal(normalized["node_ids"], nodes):
        raise GeBeam3MixedCommittedStateError("connectivity binding mismatch")
    if not _bitwise_equal(normalized["reference_geometry"], geometry):
        raise GeBeam3MixedCommittedStateError("reference geometry bit pattern mismatch")
    if not _bitwise_equal(normalized["reference_triad"], triad):
        raise GeBeam3MixedCommittedStateError("reference triad bit pattern mismatch")
    if normalized["section_descriptor_sha256"] != descriptor_hash:
        raise GeBeam3MixedCommittedStateError("section descriptor binding mismatch")
    if normalized["element_identity_sha256"] != identity_hash:
        raise GeBeam3MixedCommittedStateError("element identity binding mismatch")
    if expected_committed_total_u is not None:
        expected_u = _owned_array(expected_committed_total_u, "<f8", (18,), "expected committed_total_u")
        if not _bitwise_equal(normalized["committed_total_u"], expected_u):
            raise GeBeam3MixedCommittedStateError("committed displacement binding mismatch")
    _validate_rotations(normalized)
    stiffness = descriptor["stiffness_matrix"]
    expected_resultants = normalized["station_generalized_strain"] @ stiffness.T
    if not _normalized_close(normalized["station_generalized_resultant"], expected_resultants):
        raise GeBeam3MixedCommittedStateError("station resultants are not section stiffness times strain")
    expected_moments = normalized["station_generalized_resultant"][:, 3:6].reshape(2, 2, 3)
    if not _normalized_close(normalized["committed_local_moments"], expected_moments):
        raise GeBeam3MixedCommittedStateError("local moments do not match sided station resultants")
    if expected_local_state is not None:
        if not isinstance(expected_local_state, Mapping):
            raise GeBeam3MixedCommittedStateError("expected_local_state must be a mapping")
        comparisons = {
            "committed_local_rotation_matrices": (2, 3, 3),
            "committed_local_moments": (2, 2, 3),
            "station_generalized_strain": (4, 6),
            "station_generalized_resultant": (4, 6),
        }
        if set(expected_local_state) != set(comparisons):
            raise GeBeam3MixedCommittedStateError("expected local-state keys mismatch")
        for name, shape in comparisons.items():
            expected = _owned_array(expected_local_state[name], "<f8", shape, f"expected {name}")
            if not _normalized_close(normalized[name], expected):
                raise GeBeam3MixedCommittedStateError(f"recomputed local state mismatch for {name}")
    return normalized


def serialize_ge_beam3_mixed_state(state: Mapping[str, Any]) -> bytes:
    normalized = _normalize_state_structure(state)
    if normalized["state_integrity_sha256"] != _state_integrity_sha256(normalized):
        raise GeBeam3MixedCommittedStateError("cannot serialize state with invalid integrity")
    return canonical_json_bytes(normalized)


def deserialize_ge_beam3_mixed_state(raw: bytes) -> dict[str, Any]:
    decoded = strict_canonical_json_loads(raw)
    if not isinstance(decoded, Mapping):
        raise GeBeam3MixedCommittedStateError("restart payload must decode to a mapping")
    normalized = _normalize_state_structure(decoded)
    if normalized["state_integrity_sha256"] != _state_integrity_sha256(normalized):
        raise GeBeam3MixedCommittedStateError("restart state integrity mismatch")
    return normalized


# Concise aliases used by element-side code and tests.
build_committed_state = initialize_ge_beam3_mixed_state
seal_committed_state = seal_committed_ge_beam3_mixed_state
validate_committed_state = validate_committed_ge_beam3_mixed_state
serialize_committed_state = serialize_ge_beam3_mixed_state
deserialize_committed_state = deserialize_ge_beam3_mixed_state


__all__ = [
    "COMMIT_STATUS",
    "GE_BEAM3_MIXED_CANDIDATE_ID",
    "GE_BEAM3_MIXED_FORMULATION_ID",
    "GeBeam3MixedCommittedStateError",
    "GeBeam3MixedStateContractError",
    "LOCAL_STATE_ROLE",
    "SECTION_SCHEMA",
    "SECTION_STATE_MODE",
    "SECTION_STATION_IDS",
    "STATE_INTEGRITY_ID",
    "STATE_LAYOUT_ID",
    "STATE_SCHEMA",
    "STATE_VERSION",
    "build_committed_state",
    "canonical_json_bytes",
    "canonical_sha256",
    "decode_typed_array",
    "deserialize_committed_state",
    "deserialize_ge_beam3_mixed_state",
    "element_identity_payload",
    "element_identity_sha256",
    "encode_typed_array",
    "initialize_ge_beam3_mixed_state",
    "model_bound_identity",
    "seal_committed_ge_beam3_mixed_state",
    "seal_committed_state",
    "section_descriptor_payload",
    "section_descriptor_sha256",
    "serialize_committed_state",
    "serialize_ge_beam3_mixed_state",
    "strict_canonical_json_loads",
    "validate_committed_ge_beam3_mixed_state",
    "validate_committed_state",
]
