"""G6 integration policies around the frozen GE-B3 mechanics core.

This module owns initial-field work, activity epochs, and canonical gate
records.  It deliberately does not implement beam kinematics or element
operators.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


POLICY = "GE_BEAM3_G6_REMAINING_DOMAIN_PARITY_V1"
INITIAL_FIELD_SCHEMA = "GE_BEAM3_GENERALIZED_INITIAL_FIELD_V1"
ACTIVITY_EPOCH_POLICY = "GE_BEAM3_ACCEPTED_BOUNDARY_ACTIVITY_EPOCH_V1"
CONSUMER_POLICY_SCHEMA = "GE_BEAM3_EXPLICIT_CONSUMER_POLICY_V1"
TERMINAL = "PROVISIONAL_GO_GE_BEAM3_FULL_LEGACY_DOMAIN_PARITY_OPT_IN"
PRODUCTION_RESTRICTION = "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"

ROUTES = (
    "PARITY_REGISTER",
    "LEGACY_B3_BASELINES",
    "INITIAL_FIELD_STATE",
    "REFERENCE_DYNAMICS",
    "CONTACT_ACTIVITY",
    "OBJECTIVE_JOINT_CONTINUATION",
    "SCALE_PERFORMANCE",
    "ECOSYSTEM_BOUNDARY",
)

PARITY_DISPOSITIONS = {
    **{f"P{index:02d}": "ACCEPTED_GE_EVIDENCE" for index in range(1, 33)},
    **{f"U{index:02d}": "LEGACY_B3_UNSUPPORTED_NO_OBLIGATION" for index in range(1, 11)},
}
PARITY_DISPOSITIONS["P25"] = "LEGACY_B3_UNSUPPORTED_NO_OBLIGATION"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def canonical(value: Any) -> bytes:
    return (json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("ascii")


def _owned(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    source = np.asarray(value, dtype=object)
    if source.shape != shape or any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, float, np.integer, np.floating))
        or not isfinite(float(item))
        for item in source.flat
    ):
        raise ValueError(f"{label} must contain finite real values with shape {shape}")
    made = np.asarray(source, dtype="<f8").copy()
    made.setflags(write=False)
    return made


@dataclass(frozen=True)
class GeneralizedInitialField:
    """Station-owned residual resultant and generalized eigenstrain.

    The energy at a station is ``psi(e-e0)+s0.e``.  Consequently the returned
    resultant is exactly ``d energy / d e`` and no constitutive history is
    invented or committed by this value object.
    """

    prestress: np.ndarray
    prestrain: np.ndarray
    source: str
    identity: str = ""

    def __post_init__(self) -> None:
        prestress = np.asarray(self.prestress, dtype=object)
        prestrain = np.asarray(self.prestrain, dtype=object)
        if prestress.ndim != 2 or prestress.shape[1:] != (6,):
            raise ValueError("initial prestress must have shape (stations, 6)")
        if prestrain.shape != prestress.shape or not 1 <= prestress.shape[0] <= 64:
            raise ValueError("matching bounded station initial fields required")
        made_stress = _owned(prestress, prestress.shape, "initial prestress")
        made_strain = _owned(prestrain, prestrain.shape, "initial prestrain")
        if type(self.source) is not str or not self.source.isascii() or not self.source.strip():
            raise ValueError("bounded ASCII initial-field source required")
        body = {
            "schema": INITIAL_FIELD_SCHEMA,
            "prestress": made_stress,
            "prestrain": made_strain,
            "source": self.source,
        }
        identity = sha256(canonical(body)).hexdigest()
        if self.identity not in ("", identity):
            raise ValueError("initial-field identity mismatch")
        object.__setattr__(self, "prestress", made_stress)
        object.__setattr__(self, "prestrain", made_strain)
        object.__setattr__(self, "identity", identity)

    @property
    def station_count(self) -> int:
        return int(self.prestress.shape[0])

    def to_bytes(self) -> bytes:
        return canonical({
            "identity": self.identity,
            "prestrain": self.prestrain,
            "prestress": self.prestress,
            "schema": INITIAL_FIELD_SCHEMA,
            "source": self.source,
        })

    @classmethod
    def from_bytes(cls, raw: bytes, *, expected_sha256: str) -> "GeneralizedInitialField":
        if type(raw) is not bytes or not 0 < len(raw) <= 65536:
            raise ValueError("bounded initial-field bytes required")
        if type(expected_sha256) is not str or sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("external initial-field hash mismatch")

        def pairs(rows):
            result = {}
            for key, value in rows:
                if key in result:
                    raise ValueError("duplicate initial-field key")
                result[key] = value
            return result

        try:
            value = json.loads(raw.decode("ascii"), object_pairs_hook=pairs,
                               parse_constant=lambda token: (_ for _ in ()).throw(
                                   ValueError(f"nonfinite initial-field token: {token}")))
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("invalid initial-field JSON") from error
        if type(value) is not dict or set(value) != {
            "identity", "prestrain", "prestress", "schema", "source"
        } or value["schema"] != INITIAL_FIELD_SCHEMA or canonical(value) != raw:
            raise ValueError("strict canonical initial-field record required")
        return cls(value["prestress"], value["prestrain"], value["source"], value["identity"])


@dataclass(frozen=True)
class InitialFieldResponse:
    field_identity: str
    source: str
    station: int
    effective_strain: np.ndarray
    resultants: np.ndarray
    tangent: np.ndarray
    potential: float
    constitutive_response: Any


def evaluate_initial_field(law: Any, strain: Any, origin: Any,
                           field: GeneralizedInitialField, station: int) -> InitialFieldResponse:
    """Compose an admitted pure section response with frozen initial work."""
    if type(field) is not GeneralizedInitialField:
        raise ValueError("exact G6 generalized initial field required")
    if type(station) is not int or not 0 <= station < field.station_count:
        raise ValueError("initial-field station index outside definition")
    total = _owned(strain, (6,), "generalized strain")
    effective = _owned(total - field.prestrain[station], (6,), "effective strain")
    from ._ge_beam3_g1_elastic import ElasticSection

    response = (
        law.response(effective, origin)
        if type(law) is ElasticSection
        else law.response(effective, origin=origin)
    )
    if isinstance(response, Mapping):
        resultants = response["resultants"]
        tangent = response["tangent"]
        base_potential = response["potential"]
    else:
        resultants = response.resultants
        tangent = response.tangent
        base_potential = response.incremental_potential
        if isinstance(base_potential, tuple):
            base_potential = sum(base_potential)
    made_resultants = _owned(
        np.asarray(resultants, dtype=float) + field.prestress[station],
        (6,), "initial-field resultants",
    )
    made_tangent = _owned(tangent, (6, 6), "initial-field tangent")
    potential = float(base_potential) + float(field.prestress[station] @ total)
    if not isfinite(potential):
        raise ValueError("nonfinite initial-field potential")
    return InitialFieldResponse(field.identity, field.source, station, effective,
                                made_resultants, made_tangent, potential, response)


@dataclass(frozen=True)
class ActivityEpochLease:
    identity: str
    sequence: int
    element_ids: tuple[int, ...]
    activity_sha256: str

    @classmethod
    def capture(cls, activity: Any) -> "ActivityEpochLease":
        if activity is None or not callable(getattr(activity, "to_restart", None)):
            raise ValueError("serializable element activity required")
        descriptor = activity.to_restart(include_history=True)
        element_ids = tuple(int(value) for value in descriptor["element_ids"])
        body = canonical(descriptor)
        identity = sha256(canonical({
            "policy": ACTIVITY_EPOCH_POLICY,
            "sequence": int(activity.sequence),
            "element_ids": element_ids,
            "activity_sha256": sha256(body).hexdigest(),
        })).hexdigest()
        return cls(identity, int(activity.sequence), element_ids, sha256(body).hexdigest())

    def validate(self, activity: Any) -> None:
        current = type(self).capture(activity)
        if current != self:
            raise RuntimeError("element activity changed during open GE-B3 trial")


@dataclass(frozen=True)
class ConsumerPolicy:
    """Persisted opt-in policy used by ecosystem adapters.

    Missing policy is deliberately legacy.  This value never changes a beam
    alias and cannot select GE-B3 unless both the explicit selector and the
    exact qualified formulation identity are present.
    """

    enabled: bool = False
    selector: str = "legacy"
    formulation_id: str = ""

    def __post_init__(self) -> None:
        from .ge_beam3_element import (
            GE_BEAM3_QUALIFIED_FORMULATION_ID, GE_BEAM3_SELECTOR,
        )
        expected = (True, GE_BEAM3_SELECTOR, GE_BEAM3_QUALIFIED_FORMULATION_ID)
        legacy = (False, "legacy", "")
        if (self.enabled, self.selector, self.formulation_id) not in (legacy, expected):
            raise ValueError("consumer policy must be exact legacy or explicit qualified GE-B3")

    @classmethod
    def ge_beam3(cls) -> "ConsumerPolicy":
        from .ge_beam3_element import (
            GE_BEAM3_QUALIFIED_FORMULATION_ID, GE_BEAM3_SELECTOR,
        )
        return cls(True, GE_BEAM3_SELECTOR, GE_BEAM3_QUALIFIED_FORMULATION_ID)

    def to_bytes(self) -> bytes:
        return canonical({
            "enabled": self.enabled,
            "formulation_id": self.formulation_id,
            "schema": CONSUMER_POLICY_SCHEMA,
            "selector": self.selector,
        })

    @classmethod
    def from_bytes(cls, raw: bytes, *, expected_sha256: str) -> "ConsumerPolicy":
        if type(raw) is not bytes or not 0 < len(raw) <= 4096:
            raise ValueError("bounded consumer-policy bytes required")
        if type(expected_sha256) is not str or sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("consumer-policy hash mismatch")
        def pairs(rows):
            result = {}
            for key, value in rows:
                if key in result:
                    raise ValueError("duplicate consumer-policy key")
                result[key] = value
            return result
        try:
            value = json.loads(raw.decode("ascii"), object_pairs_hook=pairs,
                               parse_constant=lambda _token: (_ for _ in ()).throw(
                                   ValueError("nonfinite consumer policy")))
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("invalid consumer-policy JSON") from error
        if (type(value) is not dict or set(value) != {
                "enabled", "formulation_id", "schema", "selector"
            } or value["schema"] != CONSUMER_POLICY_SCHEMA or canonical(value) != raw
                or type(value["enabled"]) is not bool):
            raise ValueError("strict canonical consumer policy required")
        return cls(value["enabled"], value["selector"], value["formulation_id"])


def evidence(route: str, **checks: bool) -> dict[str, Any]:
    if type(route) is not str or route not in ROUTES:
        raise ValueError("registered G6 route required")
    if not checks or any(type(key) is not str or type(value) is not bool
                         for key, value in checks.items()) or not all(checks.values()):
        raise ValueError("nonempty passing Boolean G6 checks required")
    body = {"policy": POLICY, "route": route,
            "checks": dict(sorted(checks.items())), "production_default_qualified": False}
    return {**body, "record_sha256": sha256(canonical(body)).hexdigest()}


def adjudicate(records: Sequence[Mapping[str, Any]], *, parity_rows: Mapping[str, str],
               installed_artifact: Mapping[str, Any]) -> dict[str, Any]:
    if type(records) is not list or any(type(row) is not dict for row in records):
        raise ValueError("G6 records must be a list of exact mappings")
    by_route = {row.get("route"): row for row in records}
    if len(by_route) != len(records) or tuple(sorted(by_route)) != tuple(sorted(ROUTES)):
        raise ValueError("complete unique G6 route inventory required")
    for route, row in by_route.items():
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        if set(row) != {"policy", "route", "checks", "production_default_qualified",
                       "record_sha256"} or row["policy"] != POLICY or row["route"] != route:
            raise ValueError("invalid G6 route record")
        if row["production_default_qualified"] is not False or not all(row["checks"].values()):
            raise ValueError("failed or over-scoped G6 route")
        if row["record_sha256"] != sha256(canonical(body)).hexdigest():
            raise ValueError("G6 record hash mismatch")
    expected = tuple([f"P{index:02d}" for index in range(1, 33)]
                     + [f"U{index:02d}" for index in range(1, 11)])
    if type(parity_rows) is not dict or tuple(sorted(parity_rows)) != tuple(sorted(expected)):
        raise ValueError("complete 42-row parity disposition required")
    admitted = {"ACCEPTED_GE_EVIDENCE", "LEGACY_B3_UNSUPPORTED_NO_OBLIGATION"}
    if any(value not in admitted for value in parity_rows.values()):
        raise ValueError("unresolved parity-row disposition")
    if type(installed_artifact) is not dict or set(installed_artifact) != {
        "wheel_sha256", "probe_sha256", "replicas_byte_identical",
        "anyfem_probe_sha256", "anystructure_probe_sha256"
    } or installed_artifact["replicas_byte_identical"] is not True:
        raise ValueError("complete installed ecosystem artifact witness required")
    for key, value in installed_artifact.items():
        if key.endswith("sha256") and (type(value) is not str or len(value) != 64):
            raise ValueError("installed artifact hash malformed")
    return {
        "policy": POLICY,
        "terminal": TERMINAL,
        "parity_rows": dict(sorted(parity_rows.items())),
        "full_applicable_legacy_domain_parity": True,
        "explicit_opt_in_qualified": True,
        "production_default_qualified": False,
        "production_restriction": PRODUCTION_RESTRICTION,
        "installed_artifact": dict(installed_artifact),
    }


__all__ = [
    "ACTIVITY_EPOCH_POLICY", "ActivityEpochLease", "CONSUMER_POLICY_SCHEMA",
    "ConsumerPolicy", "GeneralizedInitialField", "PARITY_DISPOSITIONS",
    "INITIAL_FIELD_SCHEMA", "InitialFieldResponse", "POLICY", "ROUTES", "TERMINAL",
    "adjudicate", "canonical", "evaluate_initial_field", "evidence",
]
