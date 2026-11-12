from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "docs" / "reference_cases"
PROGRAM = REFERENCE_CASES / "e4_pl_s3_v2_flat_funnel_checker.py"
if str(REFERENCE_CASES) not in sys.path:
    sys.path.insert(0, str(REFERENCE_CASES))

import e4_pl_s3_v2_flat_funnel as funnel
import e4_pl_s3_v2_flat_funnel_checker as checker
import e4_pl_s3_v2_flat_funnel_producer as producer


def _plan(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest, raw = funnel.strict_json_load(producer.MANIFEST_PATH)
    records = funnel.validate_manifest(manifest, raw)
    plan = funnel.build_phase_plan(records, "4A")
    path = tmp_path / "plan.json"
    path.write_bytes(funnel.canonical_bytes(plan))
    return path, plan


def _classifying_record(
    member: dict[str, object],
    *,
    ratio: float,
    diagnostic_v1: bool = False,
) -> dict[str, object]:
    record = member["record"]
    level = int(record["level"])
    fraction = int(record["s3_area_fraction_percent"])
    baseline_error = 0.04 * (20.0 / level) ** 2
    response_error = baseline_error if fraction == 0 else ratio * baseline_error
    reference_document, reference_center = checker.reference_vector_document(level)
    energy_relative = 0.08 * (20.0 / level) ** 1.5
    reference_total = 4.0
    solution_total = 4.0
    error_total = energy_relative**2 * reference_total
    cross = 0.5 * (solution_total + reference_total - error_total)
    q4_pl = 0.08
    s3_pl = 0.04 if fraction else 0.0
    hourglass = 0.02
    physical = 2.0 - q4_pl - s3_pl - hourglass
    q4 = int(record["q4_element_count"])
    s3 = int(record["s3_element_count"])
    made: dict[str, object] = {
        "classification": (
            "NONCLASSIFYING_V1_COMPARATOR_ONLY"
            if diagnostic_v1
            else producer.CLASSIFICATION
        ),
        "connectivity_sha256": record["connectivity_sha256"],
        "diagonal": record["diagonal"],
        "element_counts": {"Q4": q4, "S3": s3},
        "energy_norm": {
            "absolute": error_total**0.5,
            "relative": energy_relative,
        },
        "formulation_counts": {
            "qualified_q4": q4,
            "v1_s3": s3 if diagnostic_v1 else 0,
            "v2a_s3": 0 if diagnostic_v1 else s3,
        },
        "level": level,
        "manifest_index": member["manifest_index"],
        "mask": record["mask"],
        "node_count": record["node_count"],
        "participation": {
            "q4_hourglass": hourglass / 2.0,
            "q4_pl": q4_pl / 2.0,
            "s3_pl": s3_pl / 2.0,
        },
        "quadratic_forms": {
            "error_total": error_total,
            "reference_total": reference_total,
            "solution_reference_cross": cross,
            "solution_total": solution_total,
        },
        "record_id": member["record_id"],
        "reference": {
            "center_transverse_displacement": reference_center,
            "dof_order": reference_document["dof_order"],
            "nodal_input_encoding": producer.REFERENCE_VECTOR_ENCODING,
            "reference_nodal_input_sha256": checker.sha256(
                checker.canonical_bytes(reference_document)
            ),
            "series_max_odd_index": producer.SERIES_MAX_ODD_INDEX,
        },
        "response": {
            "center_transverse_displacement": reference_center * (1.0 + response_error),
            "relative_error": response_error,
        },
        "s3_area_fraction_percent": fraction,
        "solution_energies": {
            "physical": physical,
            "q4_hourglass": hourglass,
            "q4_pl": q4_pl,
            "s3_pl": s3_pl,
            "total": 2.0,
        },
        "solver": {
            "free_dofs": 6 * int(record["node_count"]) - 12 * level - 4,
            "residual_relative": 1.0e-12,
            "status": "CONVERGED_DIRECT_SPARSE",
            "total_dofs": 6 * int(record["node_count"]),
        },
        "support_counts": {
            "edge_nodes": 4 * level,
            "theta_x_x_edge_constraints": 2 * (level + 1),
            "theta_y_y_edge_constraints": 2 * (level + 1),
            "translation_constraints": 12 * level,
        },
    }
    if diagnostic_v1:
        made["formulation_id"] = producer.V1_FORMULATION_ID
    return made


def _proof(
    tmp_path: Path,
    *,
    ratio: float = 1.10,
    shard_index: int = 0,
) -> tuple[Path, Path, dict[str, object]]:
    plan_path, plan = _plan(tmp_path)
    shard = plan["shards"][shard_index]
    classifying = [
        _classifying_record(member, ratio=ratio) for member in shard["records"]
    ]
    payload = {
        "assignment_id": shard["assignment_id"],
        "classifying_records": classifying,
        "diagonal": shard["diagonal"],
        "phase": "4A",
        "protocol": {
            "classification": producer.CLASSIFICATION,
            "energy_norm_id": checker.ENERGY_ID,
            "load_id": producer.LOAD_IDENTITY,
            "reference_id": producer.REFERENCE_IDENTITY,
            "support_id": producer.SUPPORT_IDENTITY,
        },
        "schema": checker.PAYLOAD_SCHEMA,
        "scope": "full",
        "v1_comparator_diagnostics": [],
        "v1_comparator_disposition": checker.HISTORICAL_V1_DISPOSITION,
    }
    ids = [member["record_id"] for member in shard["records"]]
    document = {
        "assignment_sha256": shard["assignment_sha256"],
        "plan_sha256": funnel.sha256(plan_path.read_bytes()),
        "record_count": 27,
        "record_ids": ids,
        "record_ids_sha256": funnel.sha256(funnel.canonical_bytes(ids)),
        "schema": checker.PROOF_SCHEMA,
        "scientific_payload": payload,
        "scientific_payload_sha256": funnel.sha256(funnel.canonical_bytes(payload)),
        "selector": funnel.SELECTOR,
        "terminal": "ACCEPTED_FOR_AGGREGATION",
    }
    proof_path = tmp_path / f"proof-{shard_index}.json"
    proof_path.write_bytes(funnel.canonical_bytes(document))
    return proof_path, plan_path, document


def _rewrite(path: Path, value: dict[str, object]) -> None:
    payload = value["scientific_payload"]
    value["scientific_payload_sha256"] = funnel.sha256(funnel.canonical_bytes(payload))
    path.write_bytes(funnel.canonical_bytes(value))


def test_checker_is_independent_of_producer_and_anyfesolver_mechanics() -> None:
    tree = ast.parse(PROGRAM.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(name.startswith("anyfesolver") for name in imported)
    assert not any("producer" in name for name in imported)


def test_independent_reference_encoding_matches_producer() -> None:
    for level in checker.LEVELS:
        document, center = checker.reference_vector_document(level)
        _vector, digest, producer_center = producer.mindlin_nodal_reference(level)
        assert checker.sha256(checker.canonical_bytes(document)) == digest
        assert center == producer_center


def test_independent_reference_uses_v3_shell_embedding_and_hard_trace() -> None:
    level = 2
    document, _center = checker.reference_vector_document(level)
    rows = [
        document["values"][index : index + 6]
        for index in range(0, len(document["values"]), 6)
    ]
    assert checker.REFERENCE_ID.endswith("SHELL_EMBEDDED_V3")
    assert checker.SUPPORT_ID.endswith("SHELL_ROTATIONS_V3")
    for j in range(level + 1):
        for i in range(level + 1):
            row = rows[j * (level + 1) + i]
            if i in (0, level) or j in (0, level):
                assert row[:3] == [0.0, 0.0, 0.0]
            # beta_y=-theta_x and beta_x=theta_y.  The frozen hard trace
            # therefore constrains rx on x edges and ry on y edges.
            if i in (0, level):
                assert row[3] == 0.0
            if j in (0, level):
                assert row[4] == 0.0


def test_formal_pass_is_deterministic_and_authorizes_expansion(tmp_path: Path) -> None:
    proof, plan, _document = _proof(tmp_path, ratio=1.10)
    first = checker.verify_shard(proof, plan)
    second = checker.verify_shard(proof, plan)
    assert checker.canonical_bytes(first) == checker.canonical_bytes(second)
    assert first["terminal"] == checker.PASS
    assert first["formal_failures"] == []
    assert first["advisory_review_required"] is False
    assert first["successor_expansion_authorized"] is True
    assert len(first["sequence_results"]) == 8
    assert first["v1_diagnostic_record_count"] == 0

    payload = checker.strict_json_load(proof)[0]["scientific_payload"]
    assert payload["v1_comparator_diagnostics"] == []
    assert (
        payload["v1_comparator_disposition"]
        == checker.HISTORICAL_V1_DISPOSITION
    )
    assert payload["protocol"] == {
        "classification": producer.CLASSIFICATION,
        "energy_norm_id": checker.ENERGY_ID,
        "load_id": checker.LOAD_ID,
        "reference_id": checker.REFERENCE_ID,
        "support_id": checker.SUPPORT_ID,
    }


def test_advisory_margin_pauses_without_changing_formal_terminal(tmp_path: Path) -> None:
    proof, plan, _document = _proof(tmp_path, ratio=1.20)
    result = checker.verify_shard(proof, plan)
    assert result["terminal"] == checker.PASS
    assert result["formal_failures"] == []
    assert result["advisory_review_required"] is True
    assert result["successor_expansion_authorized"] is False


def test_formal_ratio_contradiction_is_no_go(tmp_path: Path) -> None:
    proof, plan, _document = _proof(tmp_path, ratio=1.60)
    result = checker.verify_shard(proof, plan)
    assert result["terminal"] == checker.NO_GO
    assert any("FINEST_RESPONSE_ERROR_RATIO" in item for item in result["formal_failures"])
    assert result["successor_expansion_authorized"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("quadratic", "quadratic form"),
        ("reference_hash", "nodal-vector hash"),
        ("fallback", "fallback"),
        ("support", "support counts"),
    ],
)
def test_scientific_mutations_are_rejected(tmp_path: Path, mutation: str, match: str) -> None:
    proof, plan, document = _proof(tmp_path, ratio=1.10)
    made = copy.deepcopy(document)
    record = made["scientific_payload"]["classifying_records"][1]
    if mutation == "quadratic":
        record["quadratic_forms"]["error_total"] *= 2.0
    elif mutation == "reference_hash":
        record["reference"]["reference_nodal_input_sha256"] = "0" * 64
    elif mutation == "fallback":
        record["formulation_counts"]["v2a_s3"] = 0
    else:
        record["support_counts"]["translation_constraints"] += 1
    _rewrite(proof, made)
    with pytest.raises(checker.CheckerError, match=match):
        checker.verify_shard(proof, plan)


def test_runtime_v1_diagnostic_is_rejected(tmp_path: Path) -> None:
    proof, plan, document = _proof(tmp_path, ratio=1.10)
    made = copy.deepcopy(document)
    shard = checker.strict_json_load(plan)[0]["shards"][0]
    mixed_member = next(
        member
        for member in shard["records"]
        if member["record"]["s3_element_count"] > 0
    )
    made["scientific_payload"]["v1_comparator_diagnostics"].append(
        _classifying_record(mixed_member, ratio=1.8, diagnostic_v1=True)
    )
    _rewrite(proof, made)
    with pytest.raises(checker.CheckerError, match="runtime V1 comparator diagnostics"):
        checker.verify_shard(proof, plan)


@pytest.mark.parametrize("schema_location", ["proof", "payload"])
def test_correction5_proof_schemas_are_rejected(
    tmp_path: Path, schema_location: str
) -> None:
    proof, plan, document = _proof(tmp_path, ratio=1.10)
    made = copy.deepcopy(document)
    if schema_location == "proof":
        made["schema"] = "anysolver.e4-pl-s3-v2-flat-funnel-shard-scientific-v1"
    else:
        made["scientific_payload"]["schema"] = (
            "anysolver.e4-pl-s3-v2-phase4a-production-payload-v1"
        )
    _rewrite(proof, made)
    with pytest.raises(checker.CheckerError, match="identity differs"):
        checker.verify_shard(proof, plan)


def test_historical_v1_disposition_is_exact(tmp_path: Path) -> None:
    proof, plan, document = _proof(tmp_path, ratio=1.10)
    made = copy.deepcopy(document)
    made["scientific_payload"]["v1_comparator_disposition"] = (
        "NONCLASSIFYING_V1_COMPARATOR_NEVER_FALLBACK"
    )
    _rewrite(proof, made)
    with pytest.raises(checker.CheckerError, match="payload identity differs"):
        checker.verify_shard(proof, plan)


def test_duplicate_and_nonfinite_json_are_rejected() -> None:
    with pytest.raises(checker.CheckerError, match="duplicate"):
        checker.strict_json_bytes(b'{"a":1,"a":2}\n', "duplicate")
    with pytest.raises(checker.CheckerError, match="non-finite"):
        checker.strict_json_bytes(b'{"a":NaN}\n', "nonfinite")


def test_cli_writes_exclusive_byte_identical_replicas(tmp_path: Path) -> None:
    proof, plan, _document = _proof(tmp_path, ratio=1.10)
    outputs = []
    for replica in (1, 2):
        output = tmp_path / f"checker-{replica}.json"
        assert checker.main(
            [
                "--verify-proof",
                "--proof",
                str(proof),
                "--plan",
                str(plan),
                "--output",
                str(output),
            ]
        ) == 0
        outputs.append(output.read_bytes())
        assert not list(tmp_path.glob(f".{output.name}.pending-*"))
    assert outputs[0] == outputs[1]
    with pytest.raises(checker.CheckerError, match="overwrite"):
        checker.write_exclusive(tmp_path / "checker-1.json", {"blocked": False})
    assert (tmp_path / "checker-1.json").read_bytes() == outputs[0]
    assert not list(tmp_path.glob(".checker-1.json.pending-*"))
