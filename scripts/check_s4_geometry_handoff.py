"""Check the S4 neutral provenance and per-corner director handoff contract."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import anyfesolver.shell_formulations.director_field as director_field_module  # noqa: E402
from anyfesolver.shell_formulations.director_field import (  # noqa: E402
    DirectorValidationLimits,
    PreparedDirectorField,
    SourceCornerSamples,
    prepare_supplied_corner_directors,
    reconstruct_corner_directors,
)
from anyfesolver.shell_formulations.geometry_provenance import (  # noqa: E402
    GeometryAuditStatus,
    GeometryProvenanceHeader,
    GeometryProvenanceTables,
    LineageResolutionStatus,
    ReplacementLineageRecord,
    ShellSourceAssociation,
    SourceEntityHandle,
    SourceEntityState,
    SupportSurfaceRecord,
    pack_shell_source_associations,
    validate_provenance_snapshot,
    validate_shell_source_association,
)
from anyfesolver.shell_formulations.mitc4_plus_d_quality import (  # noqa: E402
    numeric_payload_fingerprint,
)


MODEL_ID = "9ef01460-18af-4634-896d-802ec73e40fb"
OTHER_MODEL_ID = "427d94fc-5392-40c9-bc42-84d1ac3e8249"


def _record_check(
    checks: list[dict[str, Any]],
    name: str,
    operation: Callable[[], MappingResult],
) -> MappingResult:
    try:
        details = operation()
    except Exception as exc:  # pragma: no cover - exercised by CLI failure behavior
        checks.append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return {}
    checks.append({"name": name, "status": "passed", **dict(details)})
    return details


MappingResult = dict[str, Any]


def _neutral_fixture() -> tuple[GeometryProvenanceTables, ShellSourceAssociation]:
    header = GeometryProvenanceHeader(
        source_geometry_model_id=MODEL_ID,
        source_geometry_revision=18,
        source_geometry_schema=4,
        source_geometry_package_version="0.2.1",
        source_geometry_document_checksum="sha256:synthetic-neutral-fixture",
        source_geometry_audit_status=GeometryAuditStatus.CLEAN_CERTIFIABLE,
        source_geometry_audit_certifiable=True,
        source_geometry_tolerance_summary=(("length", 1.0e-7),),
        source_mesh_revision=7,
        source_mesh_generator_version="synthetic-neutral-fixture",
    )
    handles = (
        SourceEntityHandle(MODEL_ID, "part", 10),
        SourceEntityHandle(MODEL_ID, "sheet", 20),
        SourceEntityHandle(MODEL_ID, "face_use", 30),
        SourceEntityHandle(MODEL_ID, "face", 40, SourceEntityState.REPLACED),
        SourceEntityHandle(MODEL_ID, "face", 41),
    )
    lineage = ReplacementLineageRecord(
        original_handle_index=3,
        descendant_handle_indices=(4,),
        selected_active_handle_index=4,
        status=LineageResolutionStatus.RESOLVED_UNAMBIGUOUS,
    )
    tables = GeometryProvenanceTables.build(
        header,
        handles,
        lineages=(lineage,),
        support_surfaces=(SupportSurfaceRecord("plane", "sha256:plane"),),
    )
    association = ShellSourceAssociation(
        part_handle_index=0,
        sheet_handle_index=1,
        face_use_handle_index=2,
        face_handle_index=4,
        support_surface_index=0,
        material_region_index=1,
        thickness_region_index=2,
        mesh_region_index=3,
        lineage_index=0,
    )
    return tables, association


def _provenance_check() -> MappingResult:
    tables, association = _neutral_fixture()
    validate_provenance_snapshot(
        tables,
        expected_model_id=MODEL_ID,
        expected_geometry_revision=18,
        expected_mesh_revision=7,
    )
    validate_shell_source_association(tables, association)
    packed = pack_shell_source_associations((association,))
    encoded = json.dumps(tables.to_dict(), sort_keys=True)
    restored = GeometryProvenanceTables.from_dict(json.loads(encoded))
    if encoded.count(MODEL_ID) != 1:
        raise AssertionError("model UUID is repeated in compact serialization")
    if restored.fingerprint != tables.fingerprint:
        raise AssertionError("provenance fingerprint changed during neutral serialization round trip")
    if packed.shape != (1, 10) or packed.dtype != np.int64 or packed.flags.writeable:
        raise AssertionError("shell association did not pack as read-only int64[1,10]")
    return {
        "schema": tables.header.source_geometry_schema,
        "package_version": tables.header.source_geometry_package_version,
        "handle_count": len(tables.handles),
        "bytes_per_shell_association": int(packed.nbytes),
        "fingerprint": tables.fingerprint,
    }


def _wrong_model_check() -> MappingResult:
    tables, _ = _neutral_fixture()
    try:
        validate_provenance_snapshot(tables, expected_model_id=OTHER_MODEL_ID)
    except ValueError as exc:
        return {"rejection": str(exc)}
    raise AssertionError("wrong source model was accepted")


def _stale_revision_check() -> MappingResult:
    tables, _ = _neutral_fixture()
    try:
        validate_provenance_snapshot(tables, expected_geometry_revision=19)
    except ValueError as exc:
        return {"rejection": str(exc)}
    raise AssertionError("stale source geometry revision was accepted")


def _fold_fixture() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        )
    )
    return coordinates, np.asarray(((0, 1, 2, 3), (1, 0, 4, 5)), dtype=np.int64)


def _sharp_fold_check() -> MappingResult:
    coordinates, connectivity = _fold_fixture()
    field = reconstruct_corner_directors(coordinates, connectivity)
    first = field.directors[0, 0]
    second = field.directors[1, 1]
    if float(np.dot(first, second)) > 1.0e-12:
        raise AssertionError("sharp fold directors were averaged across the shared node")
    return {
        "shared_node_director_dot": float(np.dot(first, second)),
        "angular_crease_edges": field.quality.angular_crease_edge_count,
        "director_fingerprint": field.numeric_fingerprint,
    }


def _sheet_boundary_check() -> MappingResult:
    coordinates, connectivity = _fold_fixture()
    field = reconstruct_corner_directors(
        coordinates,
        connectivity,
        part_indices=(1, 1),
        sheet_indices=(10, 11),
        limits=DirectorValidationLimits(crease_angle_degrees=180.0),
    )
    first = field.directors[0, 0]
    second = field.directors[1, 1]
    if float(np.dot(first, second)) > 1.0e-12:
        raise AssertionError("separate sheet directors were averaged")
    return {
        "shared_node_director_dot": float(np.dot(first, second)),
        "region_boundary_edges": field.quality.region_boundary_edge_count,
    }


def _invariance_check() -> MappingResult:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        )
    )
    connectivity = np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=np.int64)
    baseline = reconstruct_corner_directors(coordinates, connectivity)
    angle = 0.63
    rotation = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transformed = reconstruct_corner_directors(
        coordinates @ rotation.T + np.asarray((2.0e4, -3.0e4, 4.0e4)),
        connectivity[::-1],
    )
    expected = (baseline.directors @ rotation.T)[::-1]
    error = float(np.max(np.abs(transformed.directors - expected)))
    if error > 2.0e-11:
        raise AssertionError(f"director rigid-motion/element-order error {error:.3e}")
    return {"maximum_absolute_error": error}


def _reference_center_jacobian(corners: np.ndarray, directors: np.ndarray) -> float:
    derivative_xi = 0.25 * np.asarray((-1.0, 1.0, 1.0, -1.0))
    derivative_eta = 0.25 * np.asarray((-1.0, -1.0, 1.0, 1.0))
    covariant_xi = derivative_xi @ corners
    covariant_eta = derivative_eta @ corners
    return float(
        np.dot(
            np.cross(covariant_xi, covariant_eta),
            np.mean(directors, axis=0),
        )
    )


def _face_use_reference_jacobian_check() -> MappingResult:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    jacobians: dict[str, float] = {}
    for label, connectivity, source_sign in (
        ("forward", (0, 1, 2, 3), 1),
        ("reversed", (3, 2, 1, 0), -1),
    ):
        corners = coordinates[np.asarray(connectivity)]
        reconstructed = reconstruct_corner_directors(
            coordinates,
            (connectivity,),
            source_face_use_orientation_signs=(source_sign,),
        )
        supplied = prepare_supplied_corner_directors(
            corners,
            reconstructed.directors[0],
            source_face_use_orientation_signs=(source_sign,),
        )
        for path, prepared in (("reconstructed", reconstructed), ("supplied", supplied)):
            jacobian = _reference_center_jacobian(corners, prepared.directors[0])
            if jacobian <= 0.0:
                raise AssertionError(
                    f"{label} face-use {path} directors give nonpositive reference Jacobian"
                )
            jacobians[f"{label}_{path}"] = jacobian
    return jacobians


def _strip_director_counts(element_count: int) -> tuple[int, int]:
    bottom = [(float(index), 0.0, 0.0) for index in range(element_count + 1)]
    top = [(float(index), 1.0, 0.0) for index in range(element_count + 1)]
    coordinates = np.asarray((*bottom, *top))
    top_offset = element_count + 1
    connectivity = np.asarray(
        [
            (index, index + 1, top_offset + index + 1, top_offset + index)
            for index in range(element_count)
        ],
        dtype=np.int64,
    )
    field = reconstruct_corner_directors(coordinates, connectivity)
    return field.quality.smoothing_component_count, field.quality.smooth_interior_edge_count


def _linear_preprocessing_structure_check() -> MappingResult:
    small_elements, large_elements = 8, 64
    small_components, small_edges = _strip_director_counts(small_elements)
    large_components, large_edges = _strip_director_counts(large_elements)
    if small_components != 2 * (small_elements + 1):
        raise AssertionError("unexpected smooth-fan component count for small strip")
    if large_components != 2 * (large_elements + 1):
        raise AssertionError("smooth-fan components do not scale linearly with strip nodes")
    if small_edges != small_elements - 1 or large_edges != large_elements - 1:
        raise AssertionError("edge adjacency count does not scale linearly with Q4 count")
    return {
        "small_elements": small_elements,
        "small_smoothing_components": small_components,
        "large_elements": large_elements,
        "large_smoothing_components": large_components,
        "large_smooth_edges": large_edges,
    }


def _no_geometry_import_check() -> MappingResult:
    imported: set[str] = set()
    for name in ("geometry_provenance.py", "director_field.py", "mitc4_plus_d_quality.py"):
        path = SRC / "anyfesolver" / "shell_formulations" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0].lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0].lower())
    if "anygeometry" in imported:
        raise AssertionError("production handoff modules import ANYgeometry")
    return {"geometry_package_imports": 0}


def _rejection(operation: Callable[[], object], *, label: str) -> str:
    try:
        operation()
    except (TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError(f"hostile numeric input was accepted: {label}")


def _hostile_numeric_input_check() -> MappingResult:
    """Exercise the audit-reported narrowing and non-finite failure modes."""

    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    connectivity = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    prepared = reconstruct_corner_directors(coordinates, connectivity)
    unsigned_maximum = np.uint64(np.iinfo(np.uint64).max)

    invalid_directors = prepared.directors.copy()
    invalid_directors[0, 0, 0] = np.nan
    narrowed_provenance = np.full(prepared.provenance_codes.shape, 256, dtype=np.uint16)
    huge_tangent = np.tile((np.finfo(np.float64).max,) * 3, (4, 1))

    signed_fingerprint = numeric_payload_fingerprint(np.asarray((-1,), dtype=np.int64))
    unsigned_fingerprint = numeric_payload_fingerprint(
        np.asarray((unsigned_maximum,), dtype=np.uint64)
    )
    if signed_fingerprint == unsigned_fingerprint:
        raise AssertionError("signed -1 and uint64 maximum share a numeric fingerprint")

    def internal_prepared(**changes: object) -> PreparedDirectorField:
        values: dict[str, object] = {
            "directors": prepared.directors,
            "provenance_codes": prepared.provenance_codes,
            "element_quality": prepared.element_quality,
            "quality": prepared.quality,
            "diagnostics": prepared.diagnostics,
            "numeric_fingerprint": prepared.numeric_fingerprint,
        }
        values.update(changes)
        return PreparedDirectorField(
            **values,
            _factory_token=director_field_module._PREPARED_DIRECTOR_FACTORY_TOKEN,
        )

    return {
        "compact_uint64_max": _rejection(
            lambda: reconstruct_corner_directors(
                coordinates,
                connectivity,
                sheet_indices=np.asarray((unsigned_maximum,), dtype=np.uint64),
            ),
            label="compact uint64 maximum",
        ),
        "provenance_256": _rejection(
            lambda: internal_prepared(provenance_codes=narrowed_provenance),
            label="narrowed provenance 256",
        ),
        "nonfinite_director": _rejection(
            lambda: internal_prepared(directors=invalid_directors),
            label="NaN prepared director",
        ),
        "direct_construction": _rejection(
            lambda: PreparedDirectorField(
                prepared.directors,
                prepared.provenance_codes,
                prepared.element_quality,
                prepared.quality,
                prepared.diagnostics,
                prepared.numeric_fingerprint,
            ),
            label="direct prepared-field construction",
        ),
        "boolean_limit": _rejection(
            lambda: DirectorValidationLimits(normalization_atol=True),
            label="boolean director validation limit",
        ),
        "nonscalar_limit": _rejection(
            lambda: DirectorValidationLimits(normalization_atol=np.asarray(1.0)),
            label="nonscalar director validation limit",
        ),
        "noncanonical_fingerprint": _rejection(
            lambda: internal_prepared(numeric_fingerprint=object()),
            label="noncanonical prepared-field fingerprint",
        ),
        "untyped_quality": _rejection(
            lambda: internal_prepared(quality=object()),
            label="untyped prepared-field quality",
        ),
        "overflow_prone_parallel_tangents": _rejection(
            lambda: SourceCornerSamples(
                source_tangent_1=huge_tangent,
                source_tangent_2=huge_tangent,
            ),
            label="overflow-prone parallel source tangents",
        ),
        "association_uint64_max": _rejection(
            lambda: ShellSourceAssociation(material_region_index=unsigned_maximum),
            label="shell association uint64 maximum",
        ),
        "signed_unsigned_fingerprints_distinct": True,
    }


def run_geometry_handoff_checks(*, full: bool = False) -> dict[str, Any]:
    """Run lightweight synthetic contract checks and return a JSON-ready report."""

    checks: list[dict[str, Any]] = []
    _record_check(checks, "compact_provenance_round_trip", _provenance_check)
    _record_check(checks, "wrong_model_rejection", _wrong_model_check)
    _record_check(checks, "stale_revision_rejection", _stale_revision_check)
    _record_check(checks, "sharp_fold_corner_directors", _sharp_fold_check)
    _record_check(checks, "sheet_boundary_corner_directors", _sheet_boundary_check)
    _record_check(checks, "zero_production_geometry_imports", _no_geometry_import_check)
    _record_check(checks, "hostile_numeric_input_rejection", _hostile_numeric_input_check)
    if full:
        _record_check(checks, "rigid_motion_and_element_order_invariance", _invariance_check)
        _record_check(checks, "face_use_reference_jacobian", _face_use_reference_jacobian_check)
        _record_check(checks, "linear_preprocessing_structure", _linear_preprocessing_structure_check)
    failed = [item for item in checks if item["status"] != "passed"]
    return {
        "status": "failed" if failed else "passed",
        "scope": "synthetic_neutral_handoff",
        "full": bool(full),
        "checks": checks,
        "summary": {"passed": len(checks) - len(failed), "failed": len(failed)},
        "upstream_activation": "not_exercised_by_synthetic_contract_check",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also run rigid-motion and numbering invariance checks.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = run_geometry_handoff_checks(full=args.full)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
