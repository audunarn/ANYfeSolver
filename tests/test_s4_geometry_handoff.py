"""Neutral S4 geometry provenance and source-intent contract checks."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from anyfesolver.shell_formulations.geometry_provenance import (
    FORWARD_GEOMETRY_SCHEMA,
    SUPPORTED_ANYGEOMETRY_API,
    CompactSourceHandle,
    CouplingIntentRecord,
    GeometryAuditStatus,
    GeometryProvenanceHeader,
    GeometryProvenanceTables,
    LineageResolutionStatus,
    MemberSourceAssociation,
    ReplacementLineageRecord,
    ShellSourceAssociation,
    SourceEntityHandle,
    SourceEntityState,
    SupportSurfaceRecord,
    analysis_provenance_fingerprint,
    pack_shell_source_associations,
    validate_anygeometry_version,
    validate_face_use_orientation,
    validate_member_source_association,
    validate_provenance_snapshot,
    validate_shell_source_association,
)


MODEL_ID = "9ef01460-18af-4634-896d-802ec73e40fb"
OTHER_MODEL_ID = "427d94fc-5392-40c9-bc42-84d1ac3e8249"


def _header(**changes: object) -> GeometryProvenanceHeader:
    values: dict[str, object] = {
        "source_geometry_model_id": MODEL_ID,
        "source_geometry_revision": 18,
        "source_geometry_schema": 4,
        "source_geometry_package_version": "0.2.1",
        "source_geometry_document_checksum": "sha256:geometry-document",
        "source_geometry_audit_status": GeometryAuditStatus.CLEAN_CERTIFIABLE,
        "source_geometry_audit_certifiable": True,
        "source_geometry_tolerance_summary": (("angular", 1.0e-8), ("length", 1.0e-7)),
        "source_units": "m",
        "source_local_origin": (100.0, -25.0, 3.0),
        "source_coordinate_transform_fingerprint": "sha256:transform",
        "source_mesh_revision": 7,
        "source_mesh_generator_version": "ANYmesh-0.1",
    }
    values.update(changes)
    return GeometryProvenanceHeader(**values)


def _resolved_tables() -> tuple[GeometryProvenanceTables, ShellSourceAssociation]:
    handles = (
        SourceEntityHandle(MODEL_ID, "part", 10),
        SourceEntityHandle(MODEL_ID, "sheet", 20),
        SourceEntityHandle(MODEL_ID, "face_use", 30),
        SourceEntityHandle(MODEL_ID, "face", 40, SourceEntityState.REPLACED),
        SourceEntityHandle(MODEL_ID, "face", 41),
        SourceEntityHandle(MODEL_ID, "attachment", 50),
        SourceEntityHandle(MODEL_ID, "junction", 60),
        SourceEntityHandle(MODEL_ID, "member", 70),
        SourceEntityHandle(MODEL_ID, "member_edge_use", 80),
        SourceEntityHandle(MODEL_ID, "edge", 90),
    )
    lineage = ReplacementLineageRecord(
        original_handle_index=3,
        descendant_handle_indices=(4,),
        selected_active_handle_index=4,
        status=LineageResolutionStatus.RESOLVED_UNAMBIGUOUS,
    )
    table = GeometryProvenanceTables.build(
        _header(),
        handles,
        lineages=(lineage,),
        support_surfaces=(SupportSurfaceRecord("plane", "sha256:plane"),),
        coupling_intents=(
            CouplingIntentRecord(
                attachment_handle_index=5,
                junction_handle_index=6,
                attachment_kind="member_on_face",
                junction_kind="endpoint",
                member_parameter=0.25,
                face_or_edge_parameter=(0.1, 0.9),
            ),
        ),
    )
    association = ShellSourceAssociation(
        part_handle_index=0,
        sheet_handle_index=1,
        face_use_handle_index=2,
        face_handle_index=4,
        support_surface_index=0,
        source_face_use_orientation=1,
        material_region_index=11,
        thickness_region_index=12,
        mesh_region_index=13,
        lineage_index=0,
    )
    return table, association


def test_live_api_boundary_accepts_schema_4_and_legacy_schema_3_metadata() -> None:
    assert SUPPORTED_ANYGEOMETRY_API == ">=0.2,<0.3"
    assert FORWARD_GEOMETRY_SCHEMA == 4
    assert validate_anygeometry_version("0.2") == (0, 2, 0)
    assert validate_anygeometry_version("0.2.1") == (0, 2, 1)
    assert _header(source_geometry_schema=3).source_geometry_schema == 3
    with pytest.raises(ValueError, match="outside supported API"):
        validate_anygeometry_version("0.3.0")
    with pytest.raises(ValueError, match="unsupported source geometry schema"):
        _header(source_geometry_schema=5)


def test_model_uuid_is_validated_then_stored_once_in_compact_serialization() -> None:
    tables, association = _resolved_tables()
    validate_shell_source_association(tables, association)
    serialized = json.dumps(tables.to_dict(), sort_keys=True)
    assert serialized.count(MODEL_ID) == 1
    assert all(isinstance(handle, CompactSourceHandle) for handle in tables.handles)
    assert all(not hasattr(handle, "model_id") for handle in tables.handles)
    restored = GeometryProvenanceTables.from_dict(json.loads(serialized))
    assert restored == tables
    assert restored.fingerprint == tables.fingerprint

    missing_fingerprint = tables.to_dict()
    missing_fingerprint.pop("fingerprint")
    with pytest.raises(ValueError, match="requires a provenance fingerprint"):
        GeometryProvenanceTables.from_dict(missing_fingerprint)

    wrong_fingerprint = tables.to_dict()
    wrong_fingerprint["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        GeometryProvenanceTables.from_dict(wrong_fingerprint)


def test_wrong_model_is_rejected_before_source_handles_become_compact() -> None:
    handles = (
        SourceEntityHandle(MODEL_ID, "part", 1),
        SourceEntityHandle(OTHER_MODEL_ID, "sheet", 2),
    )
    with pytest.raises(ValueError, match="wrong geometry model"):
        GeometryProvenanceTables.build(_header(), handles)


def test_model_identity_and_entity_ids_follow_public_handle_invariants() -> None:
    with pytest.raises(ValueError, match="valid UUID"):
        _header(source_geometry_model_id="not-a-uuid")
    with pytest.raises(ValueError, match="positive"):
        SourceEntityHandle(MODEL_ID, "face", 0)


def test_model_and_association_integers_reject_values_outside_compact_int64_range() -> None:
    unsigned_maximum = np.uint64(np.iinfo(np.uint64).max)
    for too_large in (unsigned_maximum, 1 << 100):
        with pytest.raises(ValueError, match="no greater"):
            _header(source_geometry_revision=too_large)
        with pytest.raises(ValueError, match="table index"):
            ShellSourceAssociation(material_region_index=too_large)
    with pytest.raises(ValueError, match="no greater"):
        SourceEntityHandle(MODEL_ID, "face", unsigned_maximum)
    with pytest.raises(ValueError, match="table index"):
        MemberSourceAssociation(member_handle_index=0, edge_handle_index=unsigned_maximum)
    with pytest.raises(ValueError, match="table index"):
        CouplingIntentRecord(
            attachment_handle_index=unsigned_maximum,
            attachment_kind="member_on_face",
        )


def test_provenance_float_fields_reject_boolean_string_and_nonfinite_coercions() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        _header(source_local_origin=(True, 0.0, 0.0))
    with pytest.raises(TypeError, match="real numeric"):
        _header(source_geometry_tolerance_summary=(("length", "1e-7"),))
    with pytest.raises(TypeError, match="real numeric"):
        MemberSourceAssociation(member_handle_index=0, normalized_parameter_start=False)
    with pytest.raises(ValueError, match="must be finite"):
        CouplingIntentRecord(
            attachment_handle_index=0,
            attachment_kind="member_on_face",
            face_or_edge_parameter=(np.inf,),
        )


def test_fingerprint_and_identity_fields_reject_nonstring_objects() -> None:
    for field_name in (
        "source_geometry_package_version",
        "source_geometry_document_checksum",
        "source_units",
        "source_coordinate_transform_fingerprint",
        "source_mesh_generator_version",
    ):
        with pytest.raises(TypeError, match="must be a string"):
            _header(**{field_name: object()})
    with pytest.raises(TypeError, match="must be a string"):
        _header(source_geometry_tolerance_summary=((object(), 1.0e-7),))
    with pytest.raises(TypeError, match="must be a string"):
        SupportSurfaceRecord(object(), "surface-fingerprint")
    with pytest.raises(TypeError, match="must be a string"):
        SupportSurfaceRecord("plane", object())
    with pytest.raises(TypeError, match="must be a string"):
        SourceEntityHandle(object(), "face", 1)


def test_analysis_fingerprint_components_reject_nondeterministic_objects() -> None:
    for field_name in (
        "topology_fingerprint",
        "numeric_reference_fingerprint",
        "material_section_mapping_fingerprint",
    ):
        values: dict[str, object] = {
            "topology_fingerprint": "topology-a",
            "numeric_reference_fingerprint": "directors-a",
            "material_section_mapping_fingerprint": "section-a",
        }
        values[field_name] = object()
        with pytest.raises(TypeError, match="must be a string"):
            analysis_provenance_fingerprint(None, **values)


def test_stale_revision_and_adapter_stale_flag_fail_closed_in_constant_time_gate() -> None:
    tables, _ = _resolved_tables()
    validate_provenance_snapshot(
        tables,
        expected_model_id=MODEL_ID,
        expected_geometry_revision=18,
        expected_mesh_revision=7,
    )
    with pytest.raises(ValueError, match="geometry revision"):
        validate_provenance_snapshot(tables, expected_geometry_revision=19)
    with pytest.raises(ValueError, match="mesh revision"):
        validate_provenance_snapshot(tables, expected_mesh_revision=8)
    with pytest.raises(ValueError, match="wrong source model"):
        validate_provenance_snapshot(tables, expected_model_id=OTHER_MODEL_ID)

    stale = GeometryProvenanceTables.build(
        _header(adapter_marks_mesh_stale=True),
        (SourceEntityHandle(MODEL_ID, "part", 1),),
    )
    with pytest.raises(ValueError, match="marked.*stale"):
        validate_provenance_snapshot(stale)


@pytest.mark.parametrize(
    ("status", "certifiable", "accepted"),
    [
        (GeometryAuditStatus.CLEAN_CERTIFIABLE, True, True),
        (GeometryAuditStatus.CLEAN_NOT_CERTIFIED, False, True),
        (GeometryAuditStatus.ISSUES_PRESENT, False, True),
        (GeometryAuditStatus.AUDIT_NOT_RUN, False, True),
        (GeometryAuditStatus.CLEAN_CERTIFIABLE, False, False),
        (GeometryAuditStatus.ISSUES_PRESENT, True, False),
    ],
)
def test_upstream_audit_status_is_preserved_without_rerunning_audit(
    status: GeometryAuditStatus,
    certifiable: bool,
    accepted: bool,
) -> None:
    if accepted:
        header = _header(
            source_geometry_audit_status=status,
            source_geometry_audit_certifiable=certifiable,
        )
        assert header.source_geometry_audit_status is status
    else:
        with pytest.raises(ValueError, match="audit_certifiable"):
            _header(
                source_geometry_audit_status=status,
                source_geometry_audit_certifiable=certifiable,
            )


def test_unresolved_ambiguous_and_deleted_associations_are_rejected() -> None:
    handles = (
        SourceEntityHandle(MODEL_ID, "part", 1),
        SourceEntityHandle(MODEL_ID, "sheet", 2),
        SourceEntityHandle(MODEL_ID, "face_use", 3),
        SourceEntityHandle(MODEL_ID, "face", 4, SourceEntityState.REPLACED),
        SourceEntityHandle(MODEL_ID, "face", 5),
        SourceEntityHandle(MODEL_ID, "face", 6, SourceEntityState.DELETED),
    )
    ambiguous = ReplacementLineageRecord(
        original_handle_index=3,
        descendant_handle_indices=(4,),
        selected_active_handle_index=-1,
        status=LineageResolutionStatus.AMBIGUOUS_REPLACEMENT,
    )
    tables = GeometryProvenanceTables.build(_header(), handles, lineages=(ambiguous,))
    association = ShellSourceAssociation(0, 1, 2, 4, lineage_index=0)
    with pytest.raises(ValueError, match="unresolved or ambiguous"):
        validate_shell_source_association(tables, association)
    with pytest.raises(ValueError, match="active, already-resolved"):
        validate_shell_source_association(tables, ShellSourceAssociation(0, 1, 2, 5))


def test_face_use_orientation_mismatch_is_not_corrected_inside_the_element_path() -> None:
    validate_face_use_orientation(-1, -1)
    with pytest.raises(ValueError, match="disagrees"):
        validate_face_use_orientation(-1, 1)


def test_member_and_coupling_provenance_uses_explicit_active_source_identity() -> None:
    tables, _ = _resolved_tables()
    association = MemberSourceAssociation(
        member_handle_index=7,
        member_edge_use_handle_index=8,
        edge_handle_index=9,
        part_handle_index=0,
        normalized_parameter_start=0.2,
        normalized_parameter_end=0.8,
    )
    validate_member_source_association(tables, association)
    intent = tables.coupling_intents[0]
    assert tables.handle_kind(intent.attachment_handle_index) == "attachment"
    assert tables.handle_kind(intent.junction_handle_index) == "junction"

    with pytest.raises(ValueError, match="source attachment"):
        GeometryProvenanceTables.build(
            _header(),
            (SourceEntityHandle(MODEL_ID, "part", 1),),
            coupling_intents=(
                CouplingIntentRecord(
                    attachment_handle_index=0,
                    attachment_kind="member_on_face",
                ),
            ),
        )


def test_element_associations_pack_as_read_only_integer_indices() -> None:
    tables, association = _resolved_tables()
    validate_shell_source_association(tables, association)
    packed = pack_shell_source_associations((association, association))
    assert packed.shape == (2, 10)
    assert packed.dtype == np.int64
    assert packed.flags.c_contiguous
    assert not packed.flags.writeable
    with pytest.raises(ValueError):
        packed[0, 0] = 99
    with pytest.raises(TypeError, match="validated records"):
        pack_shell_source_associations((object(),))


def test_session_fingerprint_changes_for_geometry_numeric_or_mapping_revision() -> None:
    tables, _ = _resolved_tables()
    baseline = analysis_provenance_fingerprint(
        tables,
        topology_fingerprint="topology-a",
        numeric_reference_fingerprint="directors-a",
        material_section_mapping_fingerprint="section-a",
    )
    assert baseline == analysis_provenance_fingerprint(
        tables,
        topology_fingerprint="topology-a",
        numeric_reference_fingerprint="directors-a",
        material_section_mapping_fingerprint="section-a",
    )
    changed = analysis_provenance_fingerprint(
        tables,
        topology_fingerprint="topology-a",
        numeric_reference_fingerprint="directors-b",
        material_section_mapping_fingerprint="section-a",
    )
    assert changed != baseline


def test_production_handoff_modules_have_no_geometry_package_imports() -> None:
    root = Path("src/anyfesolver/shell_formulations")
    for name in ("geometry_provenance.py", "director_field.py", "mitc4_plus_d_quality.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0].lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0].lower())
        assert "anygeometry" not in imported_roots, name


def test_geometry_handoff_check_script_reports_synthetic_scope_without_claiming_activation() -> None:
    output_root = Path(".pytest_tmp_s4_geometry_handoff_cli")
    output_dir = output_root / str(os.getpid())
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "geometry_handoff.json"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/check_s4_geometry_handoff.py",
                "--full",
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "passed"
        assert report["scope"] == "synthetic_neutral_handoff"
        assert report["upstream_activation"] == "not_exercised_by_synthetic_contract_check"
        assert report["summary"] == {"failed": 0, "passed": 10}
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if output_root.exists() and not any(output_root.iterdir()):
            output_root.rmdir()
