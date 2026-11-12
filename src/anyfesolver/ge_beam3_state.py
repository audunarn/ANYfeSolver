"""Qualified committed-state boundary for the opt-in mixed GE-Beam3.

The P2 candidate state remains immutable research evidence.  This module
defines the production-facing v2 identity and serialization boundary while
delegating the already-qualified array, rotation, and constitutive invariants
to that frozen implementation.  Candidate v1 records are deliberately not
migrated: callers must supply a complete qualified v2 record.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import ge_beam3_mixed_state as _p2


GE_BEAM3_QUALIFIED_FORMULATION_ID = "GE_BEAM3_DC_MIXED_K1_MACRO_V2"
QUALIFICATION_ORIGIN_CANDIDATE_ID = _p2.GE_BEAM3_MIXED_CANDIDATE_ID

GE_BEAM3_CONDENSATION_ID = _p2.GE_BEAM3_MIXED_CONDENSATION_ID
GE_BEAM3_QUADRATURE_ID = _p2.GE_BEAM3_MIXED_QUADRATURE_ID
GE_BEAM3_REFERENCE_ID = _p2.GE_BEAM3_MIXED_REFERENCE_ID
GE_BEAM3_ROTATION_ID = _p2.GE_BEAM3_MIXED_ROTATION_ID

STATE_SCHEMA = "anysolver.ge-beam3-mixed.committed_state.v2"
STATE_VERSION = 2
STATE_LAYOUT_ID = "GE_BEAM3_MIXED_Q18_SHARED_SO3_OPERATOR_STATION4_V2"
STATE_INTEGRITY_ID = "SHA256_CANONICAL_COMPLETE_STATE_EXCLUDING_DIGEST_V2"
SECTION_SCHEMA = "anysolver.ge-beam3-mixed.linear-section.v2"
COMMIT_STATUS = _p2.COMMIT_STATUS
LOCAL_STATE_ROLE = _p2.LOCAL_STATE_ROLE
SECTION_STATE_MODE = _p2.SECTION_STATE_MODE
SECTION_STATION_IDS = _p2.SECTION_STATION_IDS

_SHA256 = re.compile(r"[0-9A-F]{64}\Z")

_STATE_KEYS = frozenset(
    {
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
        "qualification_origin_candidate_id",
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
    "commit_status": COMMIT_STATUS,
    "condensation_id": GE_BEAM3_CONDENSATION_ID,
    "formulation_id": GE_BEAM3_QUALIFIED_FORMULATION_ID,
    "local_state_role": LOCAL_STATE_ROLE,
    "qualification_origin_candidate_id": QUALIFICATION_ORIGIN_CANDIDATE_ID,
    "quadrature_id": GE_BEAM3_QUADRATURE_ID,
    "reference_id": GE_BEAM3_REFERENCE_ID,
    "rotation_id": GE_BEAM3_ROTATION_ID,
    "section_state_mode": SECTION_STATE_MODE,
    "state_integrity_id": STATE_INTEGRITY_ID,
    "state_layout_id": STATE_LAYOUT_ID,
    "state_schema": STATE_SCHEMA,
}

_CHECKPOINT_ARRAY_FIELDS: dict[str, tuple[str, tuple[int, ...]]] = {
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


# Keep one exception family across the immutable candidate mechanics and the
# qualified wrapper.  This lets a caller handle either rejection uniformly,
# while the v2 exact-key gate below still distinguishes candidate records.
GeBeam3CommittedStateError = _p2.GeBeam3MixedCommittedStateError
GeBeam3StateContractError = GeBeam3CommittedStateError

canonical_json_bytes = _p2.canonical_json_bytes
canonical_sha256 = _p2.canonical_sha256
encode_typed_array = _p2.encode_typed_array
decode_typed_array = _p2.decode_typed_array
strict_canonical_json_loads = _p2.strict_canonical_json_loads


def normalize_ge_beam3_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Rehydrate only the list arrays emitted by the generic checkpoint codec.

    Standalone GE-B3 state bytes retain their stricter typed-array encoding.
    The global nonlinear checkpoint has its own canonical hash and converts
    every NumPy array to JSON lists before asking the element to revalidate it.
    Reconstructing the frozen shapes and dtypes here lets that shared producer
    and consumer use the qualified state without weakening its identity,
    integrity, SO(3), or exact-key validation.
    """

    if not isinstance(state, Mapping):
        raise GeBeam3CommittedStateError(
            "committed qualified GE-Beam3 checkpoint state must be a mapping"
        )
    made = dict(state)
    for name, (dtype, shape) in _CHECKPOINT_ARRAY_FIELDS.items():
        value = made.get(name)
        if not isinstance(value, list):
            continue
        if dtype == "<i8":
            flat: list[Any] = []

            def collect(member: Any) -> None:
                if isinstance(member, list):
                    for child in member:
                        collect(child)
                else:
                    flat.append(member)

            collect(value)
            if any(type(member) is not int for member in flat):
                raise GeBeam3CommittedStateError(
                    f"checkpoint field {name} must contain exact integers"
                )
        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeBeam3CommittedStateError(
                f"checkpoint field {name} cannot be represented as {dtype}"
            ) from exc
        if array.shape != shape:
            raise GeBeam3CommittedStateError(
                f"checkpoint field {name} must have shape {shape}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise GeBeam3CommittedStateError(
                f"checkpoint field {name} contains nonfinite values"
            )
        owned = np.array(array, dtype=dtype, order="C", copy=True)
        owned.setflags(write=False)
        made[name] = owned
    return made


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise GeBeam3CommittedStateError(f"{label} must be uppercase SHA-256")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise GeBeam3CommittedStateError(
            f"{label} must be an integer, not a boolean"
        )
    return value


def _reject_candidate_state(state: Mapping[str, Any]) -> None:
    """Reject a P2 record explicitly, before any mechanics-side validation."""

    if (
        state.get("state_schema") == _p2.STATE_SCHEMA
        or state.get("state_version") == _p2.STATE_VERSION
        or state.get("state_layout_id") == _p2.STATE_LAYOUT_ID
        or state.get("formulation_id") == _p2.GE_BEAM3_MIXED_FORMULATION_ID
        or "candidate_id" in state
    ):
        raise GeBeam3CommittedStateError(
            "P2 candidate committed state is non-migratable; a complete "
            "qualified v2 state is required"
        )


def section_descriptor_payload(
    *, section_name: str, section_stiffness: Any, section_mass: Any
) -> dict[str, Any]:
    """Return the qualified v2 linear-section descriptor."""

    candidate = _p2.section_descriptor_payload(
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
    )
    return {
        "mass_matrix_per_reference_length": candidate[
            "mass_matrix_per_reference_length"
        ],
        "name": candidate["name"],
        "schema": SECTION_SCHEMA,
        "stiffness_matrix": candidate["stiffness_matrix"],
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
    """Build the exact qualified v2 element-identity preimage."""

    candidate = _p2.element_identity_payload(
        node_ids=node_ids,
        reference_geometry=reference_geometry,
        reference_triad=reference_triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_descriptor_sha256_value=section_descriptor_sha256_value,
    )
    candidate.pop("candidate_id")
    candidate["formulation_id"] = GE_BEAM3_QUALIFIED_FORMULATION_ID
    candidate["state_layout_id"] = STATE_LAYOUT_ID
    return candidate


def element_identity_sha256(**kwargs: Any) -> str:
    return canonical_sha256(element_identity_payload(**kwargs))


def model_bound_identity(element: Any, mesh: Any) -> dict[str, Any]:
    """Recompute the complete qualified identity from an element and mesh."""

    candidate = _p2.model_bound_identity(element, mesh)
    descriptor = section_descriptor_payload(
        section_name=str(element.generalized_section.name),
        section_stiffness=candidate["section_stiffness"],
        section_mass=candidate["section_mass"],
    )
    descriptor_hash = canonical_sha256(descriptor)
    candidate_identity = candidate["element_identity_payload"]
    identity = element_identity_payload(
        node_ids=candidate_identity["node_ids"],
        reference_geometry=candidate_identity["reference_geometry"],
        reference_triad=candidate_identity["reference_triad"],
        reference_orientation=candidate_identity["reference_orientation"],
        reference_axis_direction=candidate_identity["reference_axis_direction"],
        section_descriptor_sha256_value=descriptor_hash,
    )
    return {
        "element_identity_payload": identity,
        "element_identity_sha256": canonical_sha256(identity),
        "reference_geometry": candidate["reference_geometry"],
        "reference_triad": candidate["reference_triad"],
        "section_descriptor": descriptor,
        "section_descriptor_sha256": descriptor_hash,
        "section_mass": descriptor["mass_matrix_per_reference_length"],
        "section_stiffness": descriptor["stiffness_matrix"],
    }


def _candidate_syntax_record(state: Mapping[str, Any]) -> dict[str, Any]:
    """Map v2 fields to a disposable P2-shaped record for array validation."""

    return {
        "candidate_id": _p2.GE_BEAM3_MIXED_CANDIDATE_ID,
        "commit_status": _p2.COMMIT_STATUS,
        "committed_local_moments": state["committed_local_moments"],
        "committed_local_rotation_matrices": state[
            "committed_local_rotation_matrices"
        ],
        "committed_nodal_rotation_matrices": state[
            "committed_nodal_rotation_matrices"
        ],
        "committed_total_u": state["committed_total_u"],
        "condensation_id": _p2.GE_BEAM3_MIXED_CONDENSATION_ID,
        "element_id": state["element_id"],
        "element_identity_sha256": state["element_identity_sha256"],
        "formulation_id": _p2.GE_BEAM3_MIXED_FORMULATION_ID,
        "local_state_role": _p2.LOCAL_STATE_ROLE,
        "node_ids": state["node_ids"],
        "quadrature_id": _p2.GE_BEAM3_MIXED_QUADRATURE_ID,
        "reference_geometry": state["reference_geometry"],
        "reference_id": _p2.GE_BEAM3_MIXED_REFERENCE_ID,
        "reference_triad": state["reference_triad"],
        "rotation_id": _p2.GE_BEAM3_MIXED_ROTATION_ID,
        "section_descriptor_sha256": state["section_descriptor_sha256"],
        "section_state_mode": _p2.SECTION_STATE_MODE,
        "section_states": state["section_states"],
        "section_station_ids": state["section_station_ids"],
        "state_integrity_id": _p2.STATE_INTEGRITY_ID,
        "state_integrity_sha256": "0" * 64,
        "state_layout_id": _p2.STATE_LAYOUT_ID,
        "state_schema": _p2.STATE_SCHEMA,
        "state_version": _p2.STATE_VERSION,
        "station_generalized_resultant": state["station_generalized_resultant"],
        "station_generalized_strain": state["station_generalized_strain"],
    }


def _normalize_state_structure(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise GeBeam3CommittedStateError(
            "committed qualified GE-Beam3 state must be a mapping"
        )
    _reject_candidate_state(state)
    if set(state) != _STATE_KEYS:
        missing = sorted(_STATE_KEYS - set(state))
        extra = sorted(set(state) - _STATE_KEYS)
        raise GeBeam3CommittedStateError(
            f"state keys mismatch; missing={missing}, extra={extra}"
        )
    for key, expected in _STATE_ENUMS.items():
        if type(state[key]) is not str or state[key] != expected:
            raise GeBeam3CommittedStateError(f"state enum {key} mismatch")
    version = _integer(state["state_version"], "state_version")
    if version != STATE_VERSION:
        raise GeBeam3CommittedStateError("state version mismatch")
    integrity = _sha(state["state_integrity_sha256"], "state_integrity_sha256")

    # The immutable P2 normalizer owns typed-array shapes/dtypes, station IDs,
    # stateless-section layout, element IDs, and digest field syntax.  It is
    # called only after the qualified key/enumeration gate succeeds.
    candidate = _p2.seal_committed_ge_beam3_mixed_state(
        _candidate_syntax_record(state)
    )
    normalized = {
        **_STATE_ENUMS,
        "committed_local_moments": candidate["committed_local_moments"],
        "committed_local_rotation_matrices": candidate[
            "committed_local_rotation_matrices"
        ],
        "committed_nodal_rotation_matrices": candidate[
            "committed_nodal_rotation_matrices"
        ],
        "committed_total_u": candidate["committed_total_u"],
        "element_id": candidate["element_id"],
        "element_identity_sha256": _sha(
            state["element_identity_sha256"], "element_identity_sha256"
        ),
        "node_ids": candidate["node_ids"],
        "reference_geometry": candidate["reference_geometry"],
        "reference_triad": candidate["reference_triad"],
        "section_descriptor_sha256": _sha(
            state["section_descriptor_sha256"], "section_descriptor_sha256"
        ),
        "section_states": candidate["section_states"],
        "section_station_ids": candidate["section_station_ids"],
        "state_integrity_sha256": integrity,
        "state_version": STATE_VERSION,
        "station_generalized_resultant": candidate[
            "station_generalized_resultant"
        ],
        "station_generalized_strain": candidate["station_generalized_strain"],
    }
    return normalized


def _state_integrity_sha256(state: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in state.items() if key != "state_integrity_sha256"}
    )


def seal_committed_ge_beam3_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Own, normalize, and integrity-seal one complete qualified v2 state."""

    if not isinstance(state, Mapping):
        raise GeBeam3CommittedStateError(
            "committed qualified GE-Beam3 state must be a mapping"
        )
    _reject_candidate_state(state)
    without_digest = _STATE_KEYS - {"state_integrity_sha256"}
    supplied = set(state)
    if supplied == without_digest:
        provisional = dict(state)
        provisional["state_integrity_sha256"] = "0" * 64
    elif supplied == _STATE_KEYS:
        provisional = dict(state)
    else:
        missing = sorted(without_digest - supplied)
        extra = sorted(supplied - _STATE_KEYS)
        raise GeBeam3CommittedStateError(
            f"state keys mismatch before sealing; missing={missing}, extra={extra}"
        )
    normalized = _normalize_state_structure(provisional)
    normalized["state_integrity_sha256"] = _state_integrity_sha256(normalized)
    return normalized


def _qualified_from_candidate(
    candidate: Mapping[str, Any],
    *,
    element_identity_sha256_value: str,
    section_descriptor_sha256_value: str,
) -> dict[str, Any]:
    return seal_committed_ge_beam3_state(
        {
            **_STATE_ENUMS,
            "committed_local_moments": candidate["committed_local_moments"],
            "committed_local_rotation_matrices": candidate[
                "committed_local_rotation_matrices"
            ],
            "committed_nodal_rotation_matrices": candidate[
                "committed_nodal_rotation_matrices"
            ],
            "committed_total_u": candidate["committed_total_u"],
            "element_id": candidate["element_id"],
            "element_identity_sha256": element_identity_sha256_value,
            "node_ids": candidate["node_ids"],
            "reference_geometry": candidate["reference_geometry"],
            "reference_triad": candidate["reference_triad"],
            "section_descriptor_sha256": section_descriptor_sha256_value,
            "section_states": candidate["section_states"],
            "section_station_ids": candidate["section_station_ids"],
            "state_version": STATE_VERSION,
            "station_generalized_resultant": candidate[
                "station_generalized_resultant"
            ],
            "station_generalized_strain": candidate["station_generalized_strain"],
        }
    )


def initialize_ge_beam3_state(
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
    """Initialize one qualified state through the accepted P2 invariants."""

    candidate = _p2.initialize_ge_beam3_mixed_state(
        element_id=element_id,
        node_ids=node_ids,
        reference_geometry=reference_geometry,
        reference_triad=reference_triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
        committed_total_u=committed_total_u,
        committed_nodal_rotation_matrices=committed_nodal_rotation_matrices,
        committed_local_rotation_matrices=committed_local_rotation_matrices,
        committed_local_moments=committed_local_moments,
        station_generalized_strain=station_generalized_strain,
        station_generalized_resultant=station_generalized_resultant,
    )
    descriptor_hash = section_descriptor_sha256(
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
    )
    identity_hash = element_identity_sha256(
        node_ids=node_ids,
        reference_geometry=reference_geometry,
        reference_triad=reference_triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_descriptor_sha256_value=descriptor_hash,
    )
    qualified = _qualified_from_candidate(
        candidate,
        element_identity_sha256_value=identity_hash,
        section_descriptor_sha256_value=descriptor_hash,
    )
    return validate_committed_ge_beam3_state(
        qualified,
        element_id=element_id,
        node_ids=node_ids,
        reference_geometry=reference_geometry,
        reference_triad=reference_triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
        expected_committed_total_u=(
            candidate["committed_total_u"]
            if committed_total_u is not None
            else None
        ),
    )


def validate_committed_ge_beam3_state(
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
    """Validate v2 identity/integrity and all accepted P2 state invariants."""

    normalized = _normalize_state_structure(state)
    if normalized["state_integrity_sha256"] != _state_integrity_sha256(normalized):
        raise GeBeam3CommittedStateError("committed state integrity mismatch")

    descriptor = section_descriptor_payload(
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
    )
    descriptor_hash = canonical_sha256(descriptor)
    identity_hash = element_identity_sha256(
        node_ids=node_ids,
        reference_geometry=reference_geometry,
        reference_triad=reference_triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_descriptor_sha256_value=descriptor_hash,
    )
    if normalized["section_descriptor_sha256"] != descriptor_hash:
        raise GeBeam3CommittedStateError("section descriptor binding mismatch")
    if normalized["element_identity_sha256"] != identity_hash:
        raise GeBeam3CommittedStateError("element identity binding mismatch")

    # Re-seal a disposable v1-shaped record with its own candidate descriptor
    # and identity hashes, then invoke the frozen P2 semantic validator.  This
    # is validation reuse only; no v1 input is accepted or converted for use.
    candidate_descriptor_hash = _p2.section_descriptor_sha256(
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
    )
    candidate_identity_hash = _p2.element_identity_sha256(
        node_ids=node_ids,
        reference_geometry=reference_geometry,
        reference_triad=reference_triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_descriptor_sha256_value=candidate_descriptor_hash,
    )
    candidate_source = _candidate_syntax_record(normalized)
    candidate_source["section_descriptor_sha256"] = candidate_descriptor_hash
    candidate_source["element_identity_sha256"] = candidate_identity_hash
    candidate = _p2.seal_committed_ge_beam3_mixed_state(candidate_source)
    _p2.validate_committed_ge_beam3_mixed_state(
        candidate,
        element_id=element_id,
        node_ids=node_ids,
        reference_geometry=reference_geometry,
        reference_triad=reference_triad,
        reference_orientation=reference_orientation,
        reference_axis_direction=reference_axis_direction,
        section_name=section_name,
        section_stiffness=section_stiffness,
        section_mass=section_mass,
        expected_committed_total_u=expected_committed_total_u,
        expected_local_state=expected_local_state,
    )
    return normalized


def serialize_ge_beam3_state(state: Mapping[str, Any]) -> bytes:
    normalized = _normalize_state_structure(state)
    if normalized["state_integrity_sha256"] != _state_integrity_sha256(normalized):
        raise GeBeam3CommittedStateError(
            "cannot serialize state with invalid integrity"
        )
    return canonical_json_bytes(normalized)


def deserialize_ge_beam3_state(raw: bytes) -> dict[str, Any]:
    decoded = strict_canonical_json_loads(raw)
    if not isinstance(decoded, Mapping):
        raise GeBeam3CommittedStateError("restart payload must decode to a mapping")
    normalized = _normalize_state_structure(decoded)
    if normalized["state_integrity_sha256"] != _state_integrity_sha256(normalized):
        raise GeBeam3CommittedStateError("restart state integrity mismatch")
    return normalized


build_committed_state = initialize_ge_beam3_state
seal_committed_state = seal_committed_ge_beam3_state
validate_committed_state = validate_committed_ge_beam3_state
validate_ge_beam3_state = validate_committed_ge_beam3_state
serialize_committed_state = serialize_ge_beam3_state
deserialize_committed_state = deserialize_ge_beam3_state


__all__ = [
    "COMMIT_STATUS",
    "GE_BEAM3_CONDENSATION_ID",
    "GE_BEAM3_QUALIFIED_FORMULATION_ID",
    "GE_BEAM3_QUADRATURE_ID",
    "GE_BEAM3_REFERENCE_ID",
    "GE_BEAM3_ROTATION_ID",
    "GeBeam3CommittedStateError",
    "GeBeam3StateContractError",
    "LOCAL_STATE_ROLE",
    "QUALIFICATION_ORIGIN_CANDIDATE_ID",
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
    "deserialize_ge_beam3_state",
    "element_identity_payload",
    "element_identity_sha256",
    "encode_typed_array",
    "initialize_ge_beam3_state",
    "model_bound_identity",
    "seal_committed_ge_beam3_state",
    "seal_committed_state",
    "section_descriptor_payload",
    "section_descriptor_sha256",
    "serialize_committed_state",
    "serialize_ge_beam3_state",
    "strict_canonical_json_loads",
    "validate_committed_ge_beam3_state",
    "validate_committed_state",
    "validate_ge_beam3_state",
]
