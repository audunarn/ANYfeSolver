"""Neutral, immutable geometry provenance for improved shell model setup.

The public objects in this module describe an already-resolved numeric FE
handoff.  They deliberately do not import ANYgeometry, inspect a geometry
document, resolve replacement lineage, run a geometry audit, or retain live
geometry objects.  Repeated model UUIDs and document checksums live in one
header; shell elements carry only integer indices into model-owned tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from operator import index as integer_index
import re
from typing import Any, Mapping, Sequence
from uuid import UUID

import numpy as np


SUPPORTED_ANYGEOMETRY_API = ">=0.2,<0.3"
SUPPORTED_GEOMETRY_SCHEMAS = (3, 4)
FORWARD_GEOMETRY_SCHEMA = 4
_INT64_MAX = int(np.iinfo(np.int64).max)
SUPPORTED_SOURCE_ENTITY_KINDS = (
    "vertex",
    "edge",
    "face",
    "part",
    "sheet",
    "face_use",
    "coedge",
    "member",
    "member_edge_use",
    "attachment",
    "junction",
)
_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?(?:[-+].*)?$")


class GeometryAuditStatus(str, Enum):
    """Compact upstream strict-audit outcome carried by the FE handoff."""

    CLEAN_CERTIFIABLE = "clean_certifiable"
    CLEAN_NOT_CERTIFIED = "clean_not_certified"
    ISSUES_PRESENT = "issues_present"
    UNCLASSIFIED_CANDIDATE = "unclassified_candidate"
    AUDIT_NOT_RUN = "audit_not_run"
    PROVENANCE_UNAVAILABLE = "provenance_unavailable"


class SourceEntityState(str, Enum):
    """Upstream state of a handle retained in the cold model table."""

    ACTIVE = "active"
    REPLACED = "replaced"
    DELETED = "deleted"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"


class LineageResolutionStatus(str, Enum):
    """Whether an upstream replacement has a usable FE mapping."""

    RESOLVED_UNAMBIGUOUS = "resolved_unambiguous"
    DELETED_WITHOUT_DESCENDANT = "deleted_without_descendant"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"
    AMBIGUOUS_REPLACEMENT = "ambiguous_replacement"


@dataclass(frozen=True)
class GeometryProvenanceHeader:
    """One shared source-geometry and mesh provenance header."""

    source_geometry_model_id: str
    source_geometry_revision: int
    source_geometry_schema: int
    source_geometry_package_version: str
    source_geometry_document_checksum: str | None
    source_geometry_audit_status: GeometryAuditStatus | str
    source_geometry_audit_certifiable: bool
    source_geometry_tolerance_summary: tuple[tuple[str, float], ...] = ()
    source_units: str = "m"
    source_local_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_coordinate_transform_fingerprint: str = "identity"
    source_mesh_revision: int = 0
    source_mesh_generator_version: str = "unknown"
    adapter_marks_mesh_stale: bool = False

    def __post_init__(self) -> None:
        model_id = _canonical_model_id(self.source_geometry_model_id)
        object.__setattr__(self, "source_geometry_model_id", model_id)
        object.__setattr__(
            self,
            "source_geometry_revision",
            _nonnegative_integer(self.source_geometry_revision, "source_geometry_revision"),
        )
        object.__setattr__(
            self,
            "source_mesh_revision",
            _nonnegative_integer(self.source_mesh_revision, "source_mesh_revision"),
        )
        package_version = _nonempty_string(
            self.source_geometry_package_version,
            "source_geometry_package_version",
        )
        validate_anygeometry_version(package_version)
        object.__setattr__(self, "source_geometry_package_version", package_version)
        schema = _nonnegative_integer(self.source_geometry_schema, "source_geometry_schema")
        if schema not in SUPPORTED_GEOMETRY_SCHEMAS:
            raise ValueError(
                "unsupported source geometry schema "
                f"{self.source_geometry_schema}; expected one of {SUPPORTED_GEOMETRY_SCHEMAS}"
            )
        object.__setattr__(self, "source_geometry_schema", schema)
        try:
            audit_status = GeometryAuditStatus(self.source_geometry_audit_status)
        except ValueError as exc:
            raise ValueError(f"unsupported source geometry audit status: {self.source_geometry_audit_status!r}") from exc
        object.__setattr__(self, "source_geometry_audit_status", audit_status)
        if not isinstance(self.source_geometry_audit_certifiable, (bool, np.bool_)):
            raise TypeError("source_geometry_audit_certifiable must be a boolean")
        certifiable = bool(self.source_geometry_audit_certifiable)
        if certifiable != (audit_status is GeometryAuditStatus.CLEAN_CERTIFIABLE):
            raise ValueError(
                "source_geometry_audit_certifiable must be true exactly for clean_certifiable audit status"
            )
        object.__setattr__(self, "source_geometry_audit_certifiable", certifiable)
        if audit_status is GeometryAuditStatus.PROVENANCE_UNAVAILABLE:
            raise ValueError("a populated geometry provenance header cannot claim provenance_unavailable")

        checksum = self.source_geometry_document_checksum
        if checksum is not None:
            checksum = _nonempty_string(
                checksum,
                "source_geometry_document_checksum",
            )
        object.__setattr__(self, "source_geometry_document_checksum", checksum)
        units = _nonempty_string(self.source_units, "source_units")
        transform = _nonempty_string(
            self.source_coordinate_transform_fingerprint,
            "source_coordinate_transform_fingerprint",
        )
        generator = _nonempty_string(
            self.source_mesh_generator_version,
            "source_mesh_generator_version",
        )
        object.__setattr__(self, "source_units", units)
        object.__setattr__(self, "source_coordinate_transform_fingerprint", transform)
        object.__setattr__(self, "source_mesh_generator_version", generator)

        origin = tuple(
            _finite_float(value, "source_local_origin coordinate")
            for value in self.source_local_origin
        )
        if len(origin) != 3:
            raise ValueError("source_local_origin must contain three finite coordinates")
        object.__setattr__(self, "source_local_origin", origin)
        tolerance_summary = tuple(
            sorted(
                (
                    _nonempty_string(name, "source geometry tolerance name"),
                    _finite_float(value, f"source geometry tolerance {name!r}"),
                )
                for name, value in self.source_geometry_tolerance_summary
            )
        )
        if any(not name or value < 0.0 for name, value in tolerance_summary):
            raise ValueError("source geometry tolerance summary requires named finite nonnegative values")
        if len({name for name, _ in tolerance_summary}) != len(tolerance_summary):
            raise ValueError("source geometry tolerance summary names must be unique")
        object.__setattr__(self, "source_geometry_tolerance_summary", tolerance_summary)
        if not isinstance(self.adapter_marks_mesh_stale, (bool, np.bool_)):
            raise TypeError("adapter_marks_mesh_stale must be a boolean")
        object.__setattr__(self, "adapter_marks_mesh_stale", bool(self.adapter_marks_mesh_stale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_geometry_model_id": self.source_geometry_model_id,
            "source_geometry_revision": self.source_geometry_revision,
            "source_geometry_schema": self.source_geometry_schema,
            "source_geometry_package_version": self.source_geometry_package_version,
            "source_geometry_document_checksum": self.source_geometry_document_checksum,
            "source_geometry_audit_status": self.source_geometry_audit_status.value,
            "source_geometry_audit_certifiable": self.source_geometry_audit_certifiable,
            "source_geometry_tolerance_summary": [list(item) for item in self.source_geometry_tolerance_summary],
            "source_units": self.source_units,
            "source_local_origin": list(self.source_local_origin),
            "source_coordinate_transform_fingerprint": self.source_coordinate_transform_fingerprint,
            "source_mesh_revision": self.source_mesh_revision,
            "source_mesh_generator_version": self.source_mesh_generator_version,
            "adapter_marks_mesh_stale": self.adapter_marks_mesh_stale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryProvenanceHeader":
        """Load a neutral provenance header, never a geometry document."""

        return cls(
            source_geometry_model_id=payload["source_geometry_model_id"],
            source_geometry_revision=payload["source_geometry_revision"],
            source_geometry_schema=payload["source_geometry_schema"],
            source_geometry_package_version=payload["source_geometry_package_version"],
            source_geometry_document_checksum=payload.get("source_geometry_document_checksum"),
            source_geometry_audit_status=payload["source_geometry_audit_status"],
            source_geometry_audit_certifiable=payload["source_geometry_audit_certifiable"],
            source_geometry_tolerance_summary=tuple(
                (name, value)
                for name, value in payload.get("source_geometry_tolerance_summary", ())
            ),
            source_units=payload.get("source_units", "m"),
            source_local_origin=tuple(payload.get("source_local_origin", (0.0, 0.0, 0.0))),
            source_coordinate_transform_fingerprint=payload.get(
                "source_coordinate_transform_fingerprint", "identity"
            ),
            source_mesh_revision=payload.get("source_mesh_revision", 0),
            source_mesh_generator_version=payload.get("source_mesh_generator_version", "unknown"),
            adapter_marks_mesh_stale=payload.get("adapter_marks_mesh_stale", False),
        )


@dataclass(frozen=True)
class SourceEntityHandle:
    """Cold adapter input retaining the model-bound source identity."""

    model_id: str
    kind: str
    entity_id: int
    state: SourceEntityState | str = SourceEntityState.ACTIVE

    def __post_init__(self) -> None:
        model_id = _canonical_model_id(self.model_id)
        kind = _normal_kind(self.kind)
        entity_id = _positive_integer(self.entity_id, "source entity_id")
        try:
            state = SourceEntityState(self.state)
        except ValueError as exc:
            raise ValueError(f"unsupported source entity state: {self.state!r}") from exc
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class CompactSourceHandle:
    """Model-bound handle after stripping the repeated model UUID and kind."""

    kind_index: int
    entity_id: int
    state: SourceEntityState | str = SourceEntityState.ACTIVE

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind_index", _nonnegative_integer(self.kind_index, "kind_index"))
        object.__setattr__(self, "entity_id", _positive_integer(self.entity_id, "entity_id"))
        try:
            object.__setattr__(self, "state", SourceEntityState(self.state))
        except ValueError as exc:
            raise ValueError(f"unsupported source entity state: {self.state!r}") from exc


@dataclass(frozen=True)
class ReplacementLineageRecord:
    """Cold evidence for a replacement already resolved by the owner adapter."""

    original_handle_index: int
    descendant_handle_indices: tuple[int, ...]
    selected_active_handle_index: int
    status: LineageResolutionStatus | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_handle_index",
            _nonnegative_integer(self.original_handle_index, "original_handle_index"),
        )
        object.__setattr__(
            self,
            "descendant_handle_indices",
            tuple(
                _nonnegative_integer(index, "descendant_handle_index")
                for index in self.descendant_handle_indices
            ),
        )
        if len(set(self.descendant_handle_indices)) != len(self.descendant_handle_indices):
            raise ValueError("replacement lineage descendant indices must be unique")
        object.__setattr__(
            self,
            "selected_active_handle_index",
            _optional_table_index(self.selected_active_handle_index, "selected_active_handle_index"),
        )
        try:
            object.__setattr__(self, "status", LineageResolutionStatus(self.status))
        except ValueError as exc:
            raise ValueError(f"unsupported lineage resolution status: {self.status!r}") from exc


@dataclass(frozen=True)
class SupportSurfaceRecord:
    """Model-owned support-surface classification and stable fingerprint."""

    surface_type: str
    fingerprint: str

    def __post_init__(self) -> None:
        surface_type = _nonempty_string(
            self.surface_type,
            "support surface type",
        ).lower()
        fingerprint = _nonempty_string(
            self.fingerprint,
            "support surface fingerprint",
        )
        object.__setattr__(self, "surface_type", surface_type)
        object.__setattr__(self, "fingerprint", fingerprint)


@dataclass(frozen=True)
class ShellSourceAssociation:
    """Integer-only source association stored for one shell element."""

    part_handle_index: int = -1
    sheet_handle_index: int = -1
    face_use_handle_index: int = -1
    face_handle_index: int = -1
    support_surface_index: int = -1
    source_face_use_orientation: int = 1
    material_region_index: int = -1
    thickness_region_index: int = -1
    mesh_region_index: int = -1
    lineage_index: int = -1

    def __post_init__(self) -> None:
        for name in _SHELL_ASSOCIATION_INDEX_FIELDS:
            value = _optional_table_index(getattr(self, name), name)
            object.__setattr__(self, name, value)
        orientation = _strict_integer(self.source_face_use_orientation, "source_face_use_orientation")
        if orientation not in (-1, 1):
            raise ValueError("source_face_use_orientation must be +1 or -1")
        object.__setattr__(self, "source_face_use_orientation", orientation)


@dataclass(frozen=True)
class MemberSourceAssociation:
    """Integer-only source association stored for one physical beam member."""

    member_handle_index: int
    member_edge_use_handle_index: int = -1
    edge_handle_index: int = -1
    part_handle_index: int = -1
    normalized_parameter_start: float = 0.0
    normalized_parameter_end: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "member_handle_index",
            "member_edge_use_handle_index",
            "edge_handle_index",
            "part_handle_index",
        ):
            value = (
                _nonnegative_integer(getattr(self, name), name)
                if name == "member_handle_index"
                else _optional_table_index(getattr(self, name), name)
            )
            object.__setattr__(self, name, value)
        start = _finite_float(self.normalized_parameter_start, "normalized_parameter_start")
        end = _finite_float(self.normalized_parameter_end, "normalized_parameter_end")
        if not 0.0 <= start < end <= 1.0:
            raise ValueError("normalized member parameter range must satisfy 0 <= start < end <= 1")
        object.__setattr__(self, "normalized_parameter_start", start)
        object.__setattr__(self, "normalized_parameter_end", end)


@dataclass(frozen=True)
class CouplingIntentRecord:
    """Model-owned explicit attachment/junction intent; never proximity inference."""

    attachment_handle_index: int = -1
    junction_handle_index: int = -1
    attachment_kind: str | None = None
    junction_kind: str | None = None
    member_parameter: float | None = None
    face_or_edge_parameter: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        attachment = _optional_table_index(self.attachment_handle_index, "attachment_handle_index")
        junction = _optional_table_index(self.junction_handle_index, "junction_handle_index")
        if attachment < -1 or junction < -1 or (attachment < 0 and junction < 0):
            raise ValueError("coupling intent requires an attachment or junction handle index")
        object.__setattr__(self, "attachment_handle_index", attachment)
        object.__setattr__(self, "junction_handle_index", junction)
        for index_name, kind_name in (
            ("attachment_handle_index", "attachment_kind"),
            ("junction_handle_index", "junction_kind"),
        ):
            index = getattr(self, index_name)
            kind = getattr(self, kind_name)
            if index >= 0 and kind is None:
                raise ValueError(f"{kind_name} is required when {index_name} is present")
            object.__setattr__(
                self,
                kind_name,
                None if kind is None else _nonempty_string(kind, kind_name).lower(),
            )
        if self.member_parameter is not None:
            member_parameter = _finite_float(self.member_parameter, "member_parameter")
            if not 0.0 <= member_parameter <= 1.0:
                raise ValueError("member_parameter must lie in [0, 1]")
            object.__setattr__(self, "member_parameter", member_parameter)
        parameters = tuple(
            _finite_float(value, "face_or_edge_parameter")
            for value in self.face_or_edge_parameter
        )
        object.__setattr__(self, "face_or_edge_parameter", parameters)


_SHELL_ASSOCIATION_INDEX_FIELDS = (
    "part_handle_index",
    "sheet_handle_index",
    "face_use_handle_index",
    "face_handle_index",
    "support_surface_index",
    "material_region_index",
    "thickness_region_index",
    "mesh_region_index",
    "lineage_index",
)


@dataclass(frozen=True)
class GeometryProvenanceTables:
    """Immutable model-level tables plus an O(1) canonical fingerprint."""

    header: GeometryProvenanceHeader
    entity_kind_table: tuple[str, ...]
    handles: tuple[CompactSourceHandle, ...]
    lineages: tuple[ReplacementLineageRecord, ...] = ()
    support_surfaces: tuple[SupportSurfaceRecord, ...] = ()
    coupling_intents: tuple[CouplingIntentRecord, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        kinds = tuple(_normal_kind(value) for value in self.entity_kind_table)
        if len(set(kinds)) != len(kinds):
            raise ValueError("entity kind table entries must be unique")
        object.__setattr__(self, "entity_kind_table", kinds)
        object.__setattr__(self, "handles", tuple(self.handles))
        object.__setattr__(self, "lineages", tuple(self.lineages))
        object.__setattr__(self, "support_surfaces", tuple(self.support_surfaces))
        object.__setattr__(self, "coupling_intents", tuple(self.coupling_intents))
        for handle in self.handles:
            if handle.kind_index >= len(kinds):
                raise ValueError("compact source handle references a missing entity kind")
        handle_keys = tuple((handle.kind_index, handle.entity_id) for handle in self.handles)
        if len(set(handle_keys)) != len(handle_keys):
            raise ValueError("compact source handle table contains duplicate source identities")
        self._validate_lineage_table()
        self._validate_coupling_table()
        canonical = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    @classmethod
    def build(
        cls,
        header: GeometryProvenanceHeader,
        source_handles: Sequence[SourceEntityHandle],
        *,
        lineages: Sequence[ReplacementLineageRecord] = (),
        support_surfaces: Sequence[SupportSurfaceRecord] = (),
        coupling_intents: Sequence[CouplingIntentRecord] = (),
    ) -> "GeometryProvenanceTables":
        """Validate model-bound handles once and strip repeated UUID strings."""

        handles = tuple(source_handles)
        wrong_model = [
            index for index, handle in enumerate(handles) if handle.model_id != header.source_geometry_model_id
        ]
        if wrong_model:
            raise ValueError(
                "source handles belong to the wrong geometry model at table indices "
                + ", ".join(str(index) for index in wrong_model)
            )
        kinds = tuple(sorted({handle.kind for handle in handles}))
        kind_index = {kind: index for index, kind in enumerate(kinds)}
        compact = tuple(
            CompactSourceHandle(
                kind_index=kind_index[handle.kind],
                entity_id=handle.entity_id,
                state=handle.state,
            )
            for handle in handles
        )
        return cls(
            header=header,
            entity_kind_table=kinds,
            handles=compact,
            lineages=tuple(lineages),
            support_surfaces=tuple(support_surfaces),
            coupling_intents=tuple(coupling_intents),
        )

    def _validate_lineage_table(self) -> None:
        handle_count = len(self.handles)
        for lineage_index, lineage in enumerate(self.lineages):
            all_indices = (
                lineage.original_handle_index,
                *lineage.descendant_handle_indices,
            )
            if any(index < 0 or index >= handle_count for index in all_indices):
                raise ValueError(f"lineage {lineage_index} references a missing source handle")
            original = self.handles[lineage.original_handle_index]
            if lineage.status is LineageResolutionStatus.RESOLVED_UNAMBIGUOUS:
                if original.state is not SourceEntityState.REPLACED:
                    raise ValueError("resolved lineage original handle must have replaced state")
                if not lineage.descendant_handle_indices:
                    raise ValueError("resolved lineage requires at least one active descendant")
                if lineage.selected_active_handle_index not in lineage.descendant_handle_indices:
                    raise ValueError("resolved lineage selection must be one of its descendants")
                if self.handles[lineage.selected_active_handle_index].state is not SourceEntityState.ACTIVE:
                    raise ValueError("resolved lineage selection must reference an active handle")
                if any(
                    self.handles[index].state is not SourceEntityState.ACTIVE
                    for index in lineage.descendant_handle_indices
                ):
                    raise ValueError("resolved lineage descendants must all be active handles")
                original_kind = original.kind_index
                if any(
                    self.handles[index].kind_index != original_kind
                    for index in lineage.descendant_handle_indices
                ):
                    raise ValueError("replacement lineage descendants must retain the source entity kind")
            elif lineage.selected_active_handle_index != -1:
                raise ValueError("unresolved lineage must not select an active descendant")

    def _validate_coupling_table(self) -> None:
        for intent_index, intent in enumerate(self.coupling_intents):
            for handle_index, expected_kind in (
                (intent.attachment_handle_index, "attachment"),
                (intent.junction_handle_index, "junction"),
            ):
                if handle_index < 0:
                    continue
                if handle_index >= len(self.handles):
                    raise ValueError(f"coupling intent {intent_index} references a missing source handle")
                handle = self.handles[handle_index]
                if self.entity_kind_table[handle.kind_index] != expected_kind:
                    raise ValueError(
                        f"coupling intent {intent_index} does not reference a source {expected_kind}"
                    )
                if handle.state is not SourceEntityState.ACTIVE:
                    raise ValueError(f"coupling intent {intent_index} references an inactive source handle")

    def handle_kind(self, handle_index: int) -> str:
        handle = self.handles[_checked_index(handle_index, len(self.handles), "source handle")]
        return self.entity_kind_table[handle.kind_index]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract": "ANYfeSolver.s4_geometry_handoff.v1",
            "anygeometry_api": SUPPORTED_ANYGEOMETRY_API,
            "header": self.header.to_dict(),
            "entity_kind_table": list(self.entity_kind_table),
            "handle_table": [
                {
                    "kind_index": handle.kind_index,
                    "entity_id": handle.entity_id,
                    "state": handle.state.value,
                }
                for handle in self.handles
            ],
            "lineage_table": [
                {
                    "original_handle_index": item.original_handle_index,
                    "descendant_handle_indices": list(item.descendant_handle_indices),
                    "selected_active_handle_index": item.selected_active_handle_index,
                    "status": item.status.value,
                }
                for item in self.lineages
            ],
            "support_surface_table": [
                {"surface_type": item.surface_type, "fingerprint": item.fingerprint}
                for item in self.support_surfaces
            ],
            "coupling_intent_table": [
                {
                    "attachment_handle_index": item.attachment_handle_index,
                    "junction_handle_index": item.junction_handle_index,
                    "attachment_kind": item.attachment_kind,
                    "junction_kind": item.junction_kind,
                    "member_parameter": item.member_parameter,
                    "face_or_edge_parameter": list(item.face_or_edge_parameter),
                }
                for item in self.coupling_intents
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the neutral table contract; no geometry document is embedded."""

        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryProvenanceTables":
        """Reconstruct neutral tables and verify their deterministic fingerprint."""

        if payload.get("contract") != "ANYfeSolver.s4_geometry_handoff.v1":
            raise ValueError("unsupported neutral S4 geometry handoff contract")
        if payload.get("anygeometry_api") != SUPPORTED_ANYGEOMETRY_API:
            raise ValueError("neutral handoff declares an incompatible ANYgeometry API boundary")
        if "fingerprint" not in payload:
            raise ValueError("neutral S4 geometry handoff v1 requires a provenance fingerprint")
        expected = payload["fingerprint"]
        if not isinstance(expected, str) or not expected:
            raise ValueError("neutral S4 geometry handoff fingerprint must be a nonempty string")
        table = cls(
            header=GeometryProvenanceHeader.from_dict(payload["header"]),
            entity_kind_table=tuple(payload.get("entity_kind_table", ())),
            handles=tuple(
                CompactSourceHandle(
                    kind_index=item["kind_index"],
                    entity_id=item["entity_id"],
                    state=item["state"],
                )
                for item in payload.get("handle_table", ())
            ),
            lineages=tuple(
                ReplacementLineageRecord(
                    original_handle_index=item["original_handle_index"],
                    descendant_handle_indices=tuple(item.get("descendant_handle_indices", ())),
                    selected_active_handle_index=item.get("selected_active_handle_index", -1),
                    status=item["status"],
                )
                for item in payload.get("lineage_table", ())
            ),
            support_surfaces=tuple(
                SupportSurfaceRecord(item["surface_type"], item["fingerprint"])
                for item in payload.get("support_surface_table", ())
            ),
            coupling_intents=tuple(
                CouplingIntentRecord(
                    attachment_handle_index=item.get("attachment_handle_index", -1),
                    junction_handle_index=item.get("junction_handle_index", -1),
                    attachment_kind=item.get("attachment_kind"),
                    junction_kind=item.get("junction_kind"),
                    member_parameter=item.get("member_parameter"),
                    face_or_edge_parameter=tuple(item.get("face_or_edge_parameter", ())),
                )
                for item in payload.get("coupling_intent_table", ())
            ),
        )
        if expected != table.fingerprint:
            raise ValueError("neutral S4 geometry handoff fingerprint does not match its payload")
        return table


def validate_anygeometry_version(version: str) -> tuple[int, int, int]:
    """Validate the public ``>=0.2,<0.3`` API boundary without importing it."""

    normalized = _nonempty_string(version, "source geometry package version")
    match = _VERSION_PATTERN.match(normalized)
    if match is None:
        raise ValueError(f"invalid source geometry package version: {version!r}")
    parsed = (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch") or 0),
    )
    if parsed[:2] != (0, 2):
        raise ValueError(
            f"source geometry package {version!r} is outside supported API {SUPPORTED_ANYGEOMETRY_API}"
        )
    return parsed


def validate_face_use_orientation(
    source_face_use_orientation: int,
    element_source_orientation: int,
) -> None:
    """Reject a source-relative node-order mismatch before preparation.

    Both signs are measured relative to the same underlying source face.  Once
    they agree, finalized element connectivity defines the positive reference
    normal and directors must follow that connectivity normal directly.  This
    check never supplies a multiplier for element director construction.
    """

    source = _strict_integer(source_face_use_orientation, "source_face_use_orientation")
    element = _strict_integer(element_source_orientation, "element_source_orientation")
    if source not in (-1, 1) or element not in (-1, 1):
        raise ValueError("source and element orientation signs must be +1 or -1")
    if source != element:
        raise ValueError(
            "element node ordering disagrees with authoritative source face-use orientation; "
            "correct it upstream or reject the FE model"
        )


def validate_provenance_snapshot(
    tables: GeometryProvenanceTables,
    *,
    expected_model_id: str | None = None,
    expected_geometry_revision: int | None = None,
    expected_mesh_revision: int | None = None,
) -> None:
    """Perform the O(1) stale/wrong-model gate used during model/session setup."""

    header = tables.header
    if expected_model_id is not None and _canonical_model_id(expected_model_id) != header.source_geometry_model_id:
        raise ValueError("geometry provenance belongs to the wrong source model")
    if (
        expected_geometry_revision is not None
        and _nonnegative_integer(expected_geometry_revision, "expected_geometry_revision")
        != header.source_geometry_revision
    ):
        raise ValueError("source geometry revision no longer matches the immutable FE handoff")
    if (
        expected_mesh_revision is not None
        and _nonnegative_integer(expected_mesh_revision, "expected_mesh_revision")
        != header.source_mesh_revision
    ):
        raise ValueError("source mesh revision no longer matches the immutable FE handoff")
    if header.adapter_marks_mesh_stale:
        raise ValueError("upstream adapter marked the immutable FE mesh stale")


def validate_shell_source_association(
    tables: GeometryProvenanceTables,
    association: ShellSourceAssociation,
    *,
    require_complete_source: bool = True,
) -> None:
    """Validate one integer association without resolving source geometry."""

    handle_fields = (
        ("part_handle_index", "part"),
        ("sheet_handle_index", "sheet"),
        ("face_use_handle_index", "face_use"),
        ("face_handle_index", "face"),
    )
    if require_complete_source and any(getattr(association, name) < 0 for name, _ in handle_fields):
        raise ValueError("geometry-backed shell association is missing part/sheet/face-use/face identity")
    for name, expected_kind in handle_fields:
        index = getattr(association, name)
        if index < 0:
            continue
        handle = tables.handles[_checked_index(index, len(tables.handles), name)]
        kind = tables.entity_kind_table[handle.kind_index]
        if kind != expected_kind:
            raise ValueError(f"{name} references {kind!r}, expected {expected_kind!r}")
        if handle.state is not SourceEntityState.ACTIVE:
            raise ValueError(f"{name} must reference an active, already-resolved source handle")
    if association.support_surface_index >= len(tables.support_surfaces):
        raise ValueError("support_surface_index references a missing support-surface record")
    if association.lineage_index >= 0:
        lineage = tables.lineages[
            _checked_index(association.lineage_index, len(tables.lineages), "lineage")
        ]
        if lineage.status is not LineageResolutionStatus.RESOLVED_UNAMBIGUOUS:
            raise ValueError("shell source association retains unresolved or ambiguous replacement lineage")
        if association.face_handle_index != lineage.selected_active_handle_index:
            raise ValueError("shell face association must use the lineage's selected active descendant")


def validate_member_source_association(
    tables: GeometryProvenanceTables,
    association: MemberSourceAssociation,
) -> None:
    """Validate physical member identity independently of FE beam numbering."""

    expected = (
        (association.member_handle_index, "member"),
        (association.member_edge_use_handle_index, "member_edge_use"),
        (association.edge_handle_index, "edge"),
        (association.part_handle_index, "part"),
    )
    for index, expected_kind in expected:
        if index < 0:
            continue
        handle = tables.handles[_checked_index(index, len(tables.handles), expected_kind)]
        if tables.entity_kind_table[handle.kind_index] != expected_kind:
            raise ValueError(f"member association index does not reference {expected_kind!r}")
        if handle.state is not SourceEntityState.ACTIVE:
            raise ValueError("member association must reference active, already-resolved handles")


def pack_shell_source_associations(
    associations: Sequence[ShellSourceAssociation],
) -> np.ndarray:
    """Pack per-element provenance into a read-only contiguous integer array."""

    if any(not isinstance(association, ShellSourceAssociation) for association in associations):
        raise TypeError("shell source association packing requires validated records")
    columns = (*_SHELL_ASSOCIATION_INDEX_FIELDS[:5], "source_face_use_orientation", *_SHELL_ASSOCIATION_INDEX_FIELDS[5:])
    packed = np.asarray(
        [[int(getattr(association, name)) for name in columns] for association in associations],
        dtype=np.int64,
        order="C",
    )
    if packed.size == 0:
        packed = np.empty((0, len(columns)), dtype=np.int64)
    packed.setflags(write=False)
    return packed


def analysis_provenance_fingerprint(
    tables: GeometryProvenanceTables | None,
    *,
    topology_fingerprint: str,
    numeric_reference_fingerprint: str,
    material_section_mapping_fingerprint: str,
) -> str:
    """Compose the cold model/session cache key expected by coordinator wiring."""

    if tables is not None:
        validate_provenance_snapshot(tables)
    components = {
        "geometry_provenance": "unavailable" if tables is None else tables.fingerprint,
        "topology": _nonempty_string(topology_fingerprint, "topology_fingerprint"),
        "numeric_reference": _nonempty_string(
            numeric_reference_fingerprint,
            "numeric_reference_fingerprint",
        ),
        "material_section_mapping": _nonempty_string(
            material_section_mapping_fingerprint,
            "material_section_mapping_fingerprint",
        ),
    }
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normal_kind(value: str) -> str:
    kind = _nonempty_string(value, "source entity kind").lower().replace("-", "_")
    if kind not in SUPPORTED_SOURCE_ENTITY_KINDS:
        raise ValueError(f"unsupported source entity kind: {value!r}")
    return kind


def _checked_index(index: int, length: int, label: str) -> int:
    value = _strict_integer(index, label)
    if value < 0 or value >= length:
        raise ValueError(f"{label} index {value} is outside table size {length}")
    return value


def _canonical_model_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("source geometry model ID must be a string")
    try:
        made = UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("source geometry model ID must be a valid UUID") from exc
    if made.int == 0:
        raise ValueError("source geometry model ID cannot be the nil UUID")
    return str(made)


def _strict_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer, not a boolean")
    try:
        return int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer") from exc


def _nonnegative_integer(value: object, label: str) -> int:
    made = _strict_integer(value, label)
    if made < 0 or made > _INT64_MAX:
        raise ValueError(f"{label} must be nonnegative and no greater than {_INT64_MAX}")
    return made


def _positive_integer(value: object, label: str) -> int:
    made = _strict_integer(value, label)
    if made <= 0 or made > _INT64_MAX:
        raise ValueError(f"{label} must be positive and no greater than {_INT64_MAX}")
    return made


def _optional_table_index(value: object, label: str) -> int:
    made = _strict_integer(value, label)
    if made < -1 or made > _INT64_MAX:
        raise ValueError(
            f"{label} must be -1 (unavailable) or a table index in [0, {_INT64_MAX}]"
        )
    return made


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{label} must be a real numeric value, not {type(value).__name__}")
    try:
        made = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} cannot be represented as float64") from exc
    if not math.isfinite(made):
        raise ValueError(f"{label} must be finite")
    return made


def _nonempty_string(value: object, label: str) -> str:
    """Require an actual stable string instead of coercing arbitrary objects."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    made = value.strip()
    if not made:
        raise ValueError(f"{label} must be nonempty")
    return made
