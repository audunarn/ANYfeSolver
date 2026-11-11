"""Deterministic committed-state lifecycle for the S3 V2D candidate.

The state is formulation-owned and model-bound.  It deliberately shares no
schema or hot-restart path with the rejected V1 formulation or the linear V2C
candidate.  Solver transactions may copy these dictionaries, but only a
state that passes :func:`validate_v2d_state` is accepted as committed input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


FORMULATION_ID = "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
STATE_SCHEMA = "anysolver.e4-pl-s3-v2d-native-committed-state-v3"
STATE_LAYOUT_ID = "S3_V2D_HAMMER3_LAYERED_OR_GENERALIZED_ACTIVITY_STATE_V3"
STATE_INTEGRITY_ID = "S3_V2D_CANONICAL_ACTIVITY_STATE_INTEGRITY_V3"
ACTIVITY_DISPOSITION_KEY = "qualified_s3_activity_disposition"
MATERIAL_MODES = frozenset({"LAYERED_PLANE_STRESS", "GENERALIZED_SECTION"})
SOLVER_KINEMATICS = frozenset(
    {"UNBOUND", "ELEMENT_LOCAL", "LINEAR", "VON_KARMAN", "COROTATIONAL"}
)

_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "formulation_id",
        "state_layout_id",
        "material_mode",
        "element_id",
        "node_ids",
        "element_identity_sha256",
        "num_layers",
        "committed_total_u",
        "committed_constitutive_u",
        "solver_kinematics",
        ACTIVITY_DISPOSITION_KEY,
        "plastic_strain",
        "alpha",
        "layer_strain",
        "layer_stress",
        "station_generalized_strain",
        "station_generalized_resultant",
        "initial_generalized_prestrain",
        "initial_generalized_resultant",
        "initial_field_provenance",
        "state_integrity_sha256",
    }
)


class V2DStateError(ValueError):
    """A V2D state is malformed, stale, foreign, or internally inconsistent."""


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _canonical_value(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise V2DStateError(f"nonfinite canonical value at {path}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        made: dict[str, Any] = {}
        for key, member in value.items():
            if not isinstance(key, str):
                raise V2DStateError(f"non-string canonical key at {path}")
            if key in made:
                raise V2DStateError(f"duplicate canonical key {key!r} at {path}")
            made[key] = _canonical_value(member, path=f"{path}.{key}")
        return made
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(member, path=f"{path}[{index}]")
            for index, member in enumerate(value)
        ]
    raise V2DStateError(
        f"unsupported canonical value {type(value).__name__} at {path}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def strict_canonical_json_loads(raw: bytes) -> Any:
    if not isinstance(raw, bytes):
        raise V2DStateError("V2D canonical state input must be bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise V2DStateError("V2D canonical state must not contain a BOM")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        made: dict[str, Any] = {}
        for key, value in pairs:
            if key in made:
                raise V2DStateError(f"duplicate JSON key {key!r}")
            made[key] = value
        return made

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                V2DStateError(f"nonfinite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2DStateError("V2D canonical state JSON is invalid") from exc
    if canonical_json_bytes(value) != raw:
        raise V2DStateError("V2D state bytes are not canonical")
    return value


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        made = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V2DStateError(f"V2D {label} must be numeric") from exc
    if made.size == 0 and math.prod(shape) == 0:
        made = made.reshape(shape)
    if made.shape != shape or not np.all(np.isfinite(made)):
        raise V2DStateError(f"V2D {label} must have finite shape {shape}")
    return np.array(made, dtype=np.float64, order="C", copy=True)


def _validated_num_layers(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise V2DStateError("V2D num_layers must be an integer")
    layers = int(value)
    if layers <= 0:
        raise V2DStateError("V2D num_layers must be positive")
    return layers


def _integrity(state: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_integrity_sha256"}
    return canonical_sha256({"integrity_id": STATE_INTEGRITY_ID, "state": payload})


def seal_v2d_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise V2DStateError("V2D committed state must be a mapping")
    made = copy.deepcopy(dict(state))
    made.pop("state_integrity_sha256", None)
    made["state_integrity_sha256"] = _integrity(made)
    return made


def initialize_v2d_state(
    *,
    element_id: int,
    node_ids: Sequence[int],
    element_identity_sha256: str,
    num_layers: int,
    material_mode: str,
    initial_generalized_prestrain: Any,
    initial_generalized_resultant: Any,
    initial_field_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    mode = str(material_mode)
    if mode not in MATERIAL_MODES:
        raise V2DStateError(f"unsupported V2D material mode {mode!r}")
    layers = _validated_num_layers(num_layers)
    points = 3 * layers if mode == "LAYERED_PLANE_STRESS" else 0
    state = {
        "schema": STATE_SCHEMA,
        "formulation_id": FORMULATION_ID,
        "state_layout_id": STATE_LAYOUT_ID,
        "material_mode": mode,
        "element_id": int(element_id),
        "node_ids": [int(value) for value in node_ids],
        "element_identity_sha256": str(element_identity_sha256),
        "num_layers": layers,
        "committed_total_u": np.zeros(18, dtype=np.float64),
        "committed_constitutive_u": np.zeros(18, dtype=np.float64),
        "solver_kinematics": "UNBOUND",
        ACTIVITY_DISPOSITION_KEY: None,
        "plastic_strain": np.zeros((points, 3), dtype=np.float64),
        "alpha": np.zeros(points, dtype=np.float64),
        "layer_strain": np.zeros((points, 3), dtype=np.float64),
        "layer_stress": np.zeros((points, 3), dtype=np.float64),
        "station_generalized_strain": np.zeros((3, 8), dtype=np.float64),
        "station_generalized_resultant": np.zeros((3, 8), dtype=np.float64),
        "initial_generalized_prestrain": _finite_array(
            initial_generalized_prestrain, (3, 8), "initial generalized prestrain"
        ),
        "initial_generalized_resultant": _finite_array(
            initial_generalized_resultant, (3, 8), "initial generalized resultant"
        ),
        "initial_field_provenance": dict(initial_field_provenance or {}),
    }
    return seal_v2d_state(state)


def validate_v2d_state(
    state: Mapping[str, Any],
    *,
    element_id: int,
    node_ids: Sequence[int],
    element_identity_sha256: str,
    num_layers: int,
    material_mode: str,
    expected_committed_total_u: Any | None = None,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise V2DStateError("V2D committed state must be a mapping")
    made = copy.deepcopy(dict(state))
    optional_keys = {"experimental_von_karman_law"}
    if set(made) - optional_keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - set(made))
        extra = sorted(set(made) - _REQUIRED_KEYS)
        raise V2DStateError(f"V2D state keys mismatch; missing={missing}, extra={extra}")
    if (
        made["schema"] != STATE_SCHEMA
        or made["formulation_id"] != FORMULATION_ID
        or made["state_layout_id"] != STATE_LAYOUT_ID
    ):
        raise V2DStateError("V2D state formulation/schema fingerprint mismatch")
    mode = str(material_mode)
    if mode not in MATERIAL_MODES:
        raise V2DStateError(f"unsupported V2D material mode {mode!r}")
    layers = _validated_num_layers(num_layers)
    if (
        made["material_mode"] != mode
        or made["element_id"] != int(element_id)
        or made["node_ids"] != [int(value) for value in node_ids]
        or made["element_identity_sha256"] != str(element_identity_sha256)
        or made["num_layers"] != layers
    ):
        raise V2DStateError("V2D state model binding mismatch")
    points = 3 * layers if mode == "LAYERED_PLANE_STRESS" else 0
    arrays = {
        "committed_total_u": _finite_array(
            made["committed_total_u"], (18,), "committed_total_u"
        ),
        "committed_constitutive_u": _finite_array(
            made["committed_constitutive_u"],
            (18,),
            "committed_constitutive_u",
        ),
        "plastic_strain": _finite_array(
            made["plastic_strain"], (points, 3), "plastic_strain"
        ),
        "alpha": _finite_array(made["alpha"], (points,), "alpha"),
        "layer_strain": _finite_array(
            made["layer_strain"], (points, 3), "layer_strain"
        ),
        "layer_stress": _finite_array(
            made["layer_stress"], (points, 3), "layer_stress"
        ),
        "station_generalized_strain": _finite_array(
            made["station_generalized_strain"],
            (3, 8),
            "station_generalized_strain",
        ),
        "station_generalized_resultant": _finite_array(
            made["station_generalized_resultant"],
            (3, 8),
            "station_generalized_resultant",
        ),
        "initial_generalized_prestrain": _finite_array(
            made["initial_generalized_prestrain"],
            (3, 8),
            "initial_generalized_prestrain",
        ),
        "initial_generalized_resultant": _finite_array(
            made["initial_generalized_resultant"],
            (3, 8),
            "initial_generalized_resultant",
        ),
    }
    if np.any(arrays["alpha"] < 0.0):
        raise V2DStateError("V2D alpha must be nonnegative")
    if made["solver_kinematics"] not in SOLVER_KINEMATICS:
        raise V2DStateError("V2D solver_kinematics is incompatible")
    if made["solver_kinematics"] == "VON_KARMAN":
        if made.get("experimental_von_karman_law") != "S3_V2D_VK_R1":
            raise V2DStateError("V2D historical or unknown von Karman law cannot be restarted")
    elif "experimental_von_karman_law" in made:
        raise V2DStateError("V2D experimental von Karman history cannot change kinematics")
    disposition = made[ACTIVITY_DISPOSITION_KEY]
    if disposition is not None and not isinstance(disposition, Mapping):
        raise V2DStateError("V2D activity disposition is malformed")
    if expected_committed_total_u is not None:
        expected = _finite_array(
            expected_committed_total_u, (18,), "expected committed_total_u"
        )
        if not np.array_equal(arrays["committed_total_u"], expected):
            actual = arrays["committed_total_u"]
            differing = np.flatnonzero(actual != expected)
            index = int(differing[0]) if differing.size else -1
            stored = float(actual[index]) if index >= 0 else None
            bound = float(expected[index]) if index >= 0 else None
            raise V2DStateError(
                "V2D committed displacement binding mismatch"
                f"; element_id={int(made['element_id'])}"
                f"; node_ids={[int(value) for value in made['node_ids']]}"
                f"; differing_entries={int(differing.size)}/18"
                f"; first_index={index}"
                f"; stored={stored.hex() if stored is not None else None}"
                f"; expected={bound.hex() if bound is not None else None}"
            )
    supplied_integrity = made["state_integrity_sha256"]
    if not isinstance(supplied_integrity, str) or supplied_integrity != _integrity(made):
        raise V2DStateError("V2D committed state integrity mismatch")
    if not isinstance(made["initial_field_provenance"], Mapping):
        raise V2DStateError("V2D initial-field provenance must be a mapping")
    made.update(arrays)
    made["initial_field_provenance"] = copy.deepcopy(
        dict(made["initial_field_provenance"])
    )
    made[ACTIVITY_DISPOSITION_KEY] = (
        None if disposition is None else copy.deepcopy(dict(disposition))
    )
    return made


def serialize_v2d_state(state: Mapping[str, Any]) -> bytes:
    """Serialize one already sealed V2D state to canonical bytes."""

    return canonical_json_bytes(state)


def deserialize_v2d_state(raw: bytes) -> Mapping[str, Any]:
    """Decode only; the element must still perform the model-bound validation."""

    value = strict_canonical_json_loads(raw)
    if not isinstance(value, Mapping):
        raise V2DStateError("V2D restart payload must decode to a mapping")
    if value.get("formulation_id") != FORMULATION_ID:
        raise V2DStateError("V2D restart cannot cross formulation identities")
    return value


__all__ = [
    "ACTIVITY_DISPOSITION_KEY",
    "FORMULATION_ID",
    "MATERIAL_MODES",
    "STATE_INTEGRITY_ID",
    "STATE_LAYOUT_ID",
    "STATE_SCHEMA",
    "SOLVER_KINEMATICS",
    "V2DStateError",
    "canonical_json_bytes",
    "canonical_sha256",
    "deserialize_v2d_state",
    "initialize_v2d_state",
    "seal_v2d_state",
    "serialize_v2d_state",
    "strict_canonical_json_loads",
    "validate_v2d_state",
]
