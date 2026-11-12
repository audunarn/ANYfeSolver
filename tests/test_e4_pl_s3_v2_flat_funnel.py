from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "docs" / "reference_cases" / "e4_pl_s3_v2_flat_funnel.py"
CONTRACT = ROOT / "docs" / "reference_cases" / "e4_pl_s3_v2_flat_funnel_contract.json"
MANIFEST = ROOT / "docs" / "reference_cases" / "e4_pl_s3_mixed_mesh_connectivity_manifest.json"
BOUNDED = ROOT / "docs" / "reference_cases" / "e4_pl_s3_v2_bounded_process.py"
SOURCE = ROOT / "docs" / "reference_cases" / "e4_pl_s3_v2_source_equation_contract.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def funnel():
    return _load("_s3_v2_flat_funnel_test", PROGRAM)


@pytest.fixture(scope="module")
def records(funnel):
    value, raw = funnel.strict_json_load(MANIFEST)
    return funnel.validate_manifest(value, raw)


def _receipt(
    funnel,
    records,
    root: Path,
    phase: str,
    *,
    predecessors=(),
    terminal: str | None = None,
):
    if terminal is None:
        terminal = funnel.SCAFFOLD_RECEIPT_TERMINAL
    coordinates = {
        "4A": ("4A", "full"),
        "4B": ("4B", "full"),
        "4C_SENTINEL": ("4C", "sentinel"),
        "4C": ("4C", "completion"),
    }
    plan_phase, scope = coordinates[phase]
    phase_root = (root / phase).resolve()
    phase_root.mkdir(parents=True, exist_ok=True)
    plan = funnel.build_phase_plan(
        records,
        plan_phase,
        scope=scope,
        receipts=predecessors,
    )
    plan_path = phase_root / "plan.json"
    plan_raw = funnel.canonical_bytes(plan)
    plan_path.write_bytes(plan_raw)
    plan_sha = funnel.sha256(plan_raw)

    candidate = _candidate_binding(funnel, root)
    producer = (root / "producer.py").resolve()
    if not producer.exists():
        producer.write_text("# frozen producer\n", encoding="utf-8")
    wave_root = phase_root
    wave_manifest = funnel.build_bounded_wave_manifest(
        plan,
        plan_path=plan_path.resolve(),
        producer_program=producer,
        python_executable=Path(sys.executable).resolve(),
        cwd=ROOT,
        output_root=wave_root,
        input_paths={
            "candidate_artifact": candidate,
            "connectivity_manifest": MANIFEST,
            "flat_funnel_contract": CONTRACT,
            "source_equation_contract": SOURCE,
        },
    )
    wave_manifest_path = phase_root / "wave-manifest.json"
    wave_manifest_raw = funnel.canonical_bytes(wave_manifest)
    wave_manifest_path.write_bytes(wave_manifest_raw)
    wave_manifest_sha = funnel.sha256(wave_manifest_raw)

    worker_results = []
    current_evidence = []
    current_record_ids = []
    for worker, shard in zip(wave_manifest["workers"], plan["shards"]):
        scientific_path = Path(worker["scientific_path"])
        scientific_path.parent.mkdir(parents=True, exist_ok=True)
        Path(worker["stdout_path"]).write_bytes(b"")
        Path(worker["stderr_path"]).write_bytes(b"")
        record_ids = [member["record_id"] for member in shard["records"]]
        payload = {
            "assignment_id": shard["assignment_id"],
            "records": [
                {"exact_residual": "0", "record_id": made_id}
                for made_id in record_ids
            ],
        }
        science = {
            "assignment_sha256": shard["assignment_sha256"],
            "plan_sha256": plan_sha,
            "record_count": len(record_ids),
            "record_ids": record_ids,
            "record_ids_sha256": funnel.sha256(funnel.canonical_bytes(record_ids)),
            "schema": funnel.SHARD_SCIENTIFIC_SCHEMA,
            "scientific_payload": payload,
            "scientific_payload_sha256": funnel.sha256(
                funnel.canonical_bytes(payload)
            ),
            "selector": funnel.SELECTOR,
            "terminal": "ACCEPTED_FOR_AGGREGATION",
        }
        science_raw = funnel.canonical_bytes(science)
        scientific_path.write_bytes(science_raw)
        evidence = funnel.ValidatedShardEvidence(
            assignment_id=shard["assignment_id"],
            assignment_sha256=shard["assignment_sha256"],
            record_count=len(record_ids),
            record_ids_sha256=science["record_ids_sha256"],
            scientific_payload_sha256=science["scientific_payload_sha256"],
            scientific_sha256=funnel.sha256(science_raw),
        )
        current_evidence.append(evidence)
        current_record_ids.extend(record_ids)
        worker_results.append(
            {
                "assignment_id": worker["assignment_id"],
                "assignment_sha256": worker["assignment_sha256"],
                "cpu_100ns": 1,
                "input_hashes": worker["input_hashes"],
                "last_progress_sequence": 5,
                "peak_tree_memory_bytes": 1,
                "plan_sha256": plan_sha,
                "program_sha256": worker["program_sha256"],
                "returncode": 0,
                "scientific_byte_count": len(science_raw),
                "scientific_schema": funnel.SHARD_SCIENTIFIC_SCHEMA,
                "scientific_sha256": funnel.sha256(science_raw),
                "status": "COMPLETED",
                "stderr_sha256": funnel.sha256(b""),
                "stdout_sha256": funnel.sha256(b""),
                "termination_proven": True,
            }
        )
    wave_result = {
        "lane": "flat-proof",
        "manifest_sha256": wave_manifest_sha,
        "schema": funnel.BOUNDED_RESULT_SCHEMA,
        "terminal": "COMPLETED",
        "wave_id": wave_manifest["wave_id"],
        "workers": worker_results,
    }
    wave_result_path = phase_root / "wave-result.json"
    wave_result_raw = funnel.canonical_bytes(wave_result)
    wave_result_path.write_bytes(wave_result_raw)
    wave_result_sha = funnel.sha256(wave_result_raw)

    combined_evidence = list(current_evidence)
    combined_ids = list(current_record_ids)
    if phase == "4C":
        sentinel = predecessors[-1]
        combined_evidence = [*sentinel.shards, *combined_evidence]
        combined_ids = [*sentinel.record_ids, *combined_ids]
    combined_evidence.sort(key=lambda item: item.assignment_id)
    combined_ids.sort()
    payload_bindings = [
        {
            "assignment_id": item.assignment_id,
            "scientific_payload_sha256": item.scientific_payload_sha256,
        }
        for item in combined_evidence
    ]
    aggregate = {
        "complete_record_count": funnel.RECEIPT_COUNTS[phase],
        "current_wave_record_count": plan["record_count"],
        "manifest_sha256": funnel.MANIFEST_SHA256,
        "phase": phase,
        "plan_sha256": plan_sha,
        "prerequisite_aggregate_sha256s": [
            item.aggregate_sha256 for item in predecessors
        ],
        "record_ids": combined_ids,
        "record_ids_sha256": funnel.sha256(funnel.canonical_bytes(combined_ids)),
        "schema": funnel.SCIENTIFIC_AGGREGATE_SCHEMA,
        "scientific_payload_sha256": funnel.sha256(
            funnel.canonical_bytes(payload_bindings)
        ),
        "selector": funnel.SELECTOR,
        "shards": [item.aggregate_binding() for item in combined_evidence],
        "terminal": terminal,
        "wave_manifest_sha256": wave_manifest_sha,
        "wave_result_sha256": wave_result_sha,
    }
    aggregate_path = phase_root / "aggregate.json"
    aggregate_raw = funnel.canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_raw)
    receipt = {
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": funnel.sha256(aggregate_raw),
        "complete_record_count": funnel.RECEIPT_COUNTS[phase],
        "manifest_sha256": funnel.MANIFEST_SHA256,
        "phase": phase,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "prerequisite_receipt_paths": [
            str(item.receipt_path) for item in predecessors
        ],
        "schema": funnel.RECEIPT_SCHEMA,
        "terminal": terminal,
        "wave_manifest_path": str(wave_manifest_path),
        "wave_manifest_sha256": wave_manifest_sha,
        "wave_result_path": str(wave_result_path),
        "wave_result_sha256": wave_result_sha,
    }
    receipt_path = phase_root / "receipt.json"
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    return funnel.load_validated_receipt(receipt_path, records)


def _candidate_binding(funnel, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    artifact = (root / "candidate.whl").resolve()
    if not artifact.exists():
        artifact.write_bytes(b"frozen candidate wheel bytes\n")
    binding = {
        "artifact_path": str(artifact),
        "artifact_sha256": funnel.sha256(artifact.read_bytes()),
        "candidate_id": funnel.CANDIDATE_ID,
        "commit": "0" * 40,
        "formulation_id": funnel.FORMULATION_ID,
        "schema": funnel.CANDIDATE_BINDING_SCHEMA,
        "selector": funnel.SELECTOR,
        "tree": "1" * 40,
    }
    path = (root / "candidate-binding.json").resolve()
    path.write_bytes(funnel.canonical_bytes(binding))
    return path


def test_harness_is_standard_library_planning_only() -> None:
    source = PROGRAM.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "typing",
    }
    assert "anyfesolver" not in imports
    assert "numpy" not in imports
    assert "scipy" not in imports
    assert "sympy" not in imports
    assert "build_case_model" not in source
    assert "solve_linear" not in source


def test_contract_freezes_qv9_science_and_v2_only_selector(funnel) -> None:
    value, raw = funnel.strict_json_load(CONTRACT)
    contract = funnel.validate_contract(value)
    assert raw == funnel.canonical_bytes(value)
    assert len(raw) == funnel.REGISTERED_FLAT_FUNNEL_CONTRACT_BYTES
    assert funnel.sha256(raw) == funnel.REGISTERED_FLAT_FUNNEL_CONTRACT_SHA256
    assert contract["authority_disposition"] == {
        "artifact_commit_tree_verified": False,
        "formal_execution_authorized": False,
        "independent_scientific_payload_checker_registered": False,
        "receipt_classification": "NONCLASSIFYING_SCAFFOLD_ONLY",
        "registered_producer_identity": False,
    }
    assert contract["candidate"]["selector"] == "e4-pl-s3-v2"
    assert contract["candidate"]["v1_fallback_forbidden"] is True
    assert contract["formal_protocol"]["support_identity"] == (
        "HARD_NAVIER_TRANSLATIONS_PLUS_TANGENTIAL_ROTATIONS_V2"
    )
    assert contract["formal_protocol"]["reference_identity"] == (
        "INDEPENDENT_NAVIER_REISSNER_MINDLIN_UNIFORM_PRESSURE_V2"
    )
    assert contract["formal_protocol"]["thresholds"] == funnel.FORMAL_THRESHOLDS
    assert contract["advisory_review_triggers"]["thresholds"] == {
        "finest_error_ratio_at_25_percent": "1.35",
        "finest_error_ratio_through_10_percent": "1.15",
    }
    assert contract["advisory_review_triggers"]["classification"] == (
        "NONCLASSIFYING_INDEPENDENT_REVIEW_TRIGGER"
    )
    assert contract["phase_funnel"]["4C"]["execution_order"] == (
        "SENTINEL_18_THEN_REMAINING_45"
    )


def test_registered_contract_rejects_every_previously_unchecked_authority_mutation(
    funnel,
) -> None:
    original, _ = funnel.strict_json_load(CONTRACT)
    mutations = []

    def changed() -> dict:
        made = copy.deepcopy(original)
        mutations.append(made)
        return made

    del changed()["candidate"]["flat_candidate_id"]
    changed()["candidate"]["final_formulation_id"] = "MUTATED"
    changed()["candidate"]["v1_fallback_forbidden"] = False
    changed()["frozen_inputs"]["anyfesolver_base"]["commit"] = "0" * 40
    changed()["frozen_inputs"]["historical_qv9"]["v1_nogo_result_sha256"] = "0" * 64
    changed()["phase_funnel"]["4A"]["masks"] = ["dispersed"]
    changed()["phase_funnel"]["4B"]["levels"] = [20, 40]
    changed()["phase_funnel"]["4C"]["records_per_diagonal_shard"] = 20
    changed()["phase_funnel"]["4C"]["prerequisites"] = ["4A"]
    changed()["advisory_review_triggers"]["disposition"] = "MUTATED"
    changed()["production_boundary"]["stage_authority"] = "FORMAL"
    changed()["authority_disposition"]["formal_execution_authorized"] = True
    for mutation in mutations:
        with pytest.raises(funnel.FlatFunnelError, match="registered exact bytes"):
            funnel.validate_contract(mutation)


def test_registered_source_contract_rejects_truncation_and_authority_mutation(
    funnel,
) -> None:
    original, raw = funnel.strict_json_load(SOURCE)
    validated = funnel.validate_source_contract(original, raw)
    assert validated is original
    assert len(raw) == funnel.REGISTERED_SOURCE_CONTRACT_BYTES
    assert funnel.sha256(raw) == funnel.REGISTERED_SOURCE_CONTRACT_SHA256

    mutations = []

    def changed() -> dict:
        made = copy.deepcopy(original)
        mutations.append(made)
        return made

    del changed()["sources"]
    del changed()["equation_requirements"]
    changed()["equation_requirements"][0]["excluded_scope"] = []
    changed()["gate"]["current_terminal"] = "MUTATED"
    map_source = next(
        item for item in changed()["sources"] if item["id"] == "s3_v2_dkmt_equation_map"
    )
    map_source["sha256"] = "0" * 64
    ledger_source = next(
        item for item in changed()["sources"] if item["id"] == "s3_v2_dkmt_source_ledger"
    )
    ledger_source["sha256"] = "0" * 64
    for mutation in mutations:
        mutated_raw = funnel.canonical_bytes(mutation)
        with pytest.raises(funnel.FlatFunnelError, match="registered exact bytes"):
            funnel.validate_source_contract(mutation, mutated_raw)


def test_immutable_manifest_is_exactly_252_unique_records(funnel, records) -> None:
    assert len(records) == 252
    assert {record["level"] for record in records} == {20, 40, 80, 160}
    assert all(
        sum(record["level"] == level for record in records) == 63
        for level in (20, 40, 80, 160)
    )
    ids = [funnel.record_id(record) for record in records]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize(
    ("phase", "receipts", "total", "per_shard"),
    [
        ("4A", (), 81, 27),
        ("4B", ("4A",), 108, 36),
    ],
)
def test_phase_selection_and_diagonal_sharding_are_exact(
    funnel, records, tmp_path, phase, receipts, total, per_shard
) -> None:
    made_receipts = []
    for item in receipts:
        made_receipts.append(
            _receipt(
                funnel,
                records,
                tmp_path,
                item,
                predecessors=tuple(made_receipts),
            )
        )
    first = funnel.build_phase_plan(records, phase, receipts=made_receipts)
    second = funnel.build_phase_plan(records, phase, receipts=made_receipts)
    assert funnel.canonical_bytes(first) == funnel.canonical_bytes(second)
    assert first["record_count"] == total
    assert first["selector"] == "e4-pl-s3-v2"
    assert [shard["diagonal"] for shard in first["shards"]] == list(funnel.DIAGONALS)
    assert all(len(shard["records"]) == per_shard for shard in first["shards"])
    assert all(shard["selector"] == "e4-pl-s3-v2" for shard in first["shards"])
    assert sum(len(shard["records"]) for shard in first["shards"]) == total
    assignment_hashes = [shard["assignment_sha256"] for shard in first["shards"]]
    assert len(set(assignment_hashes)) == 3


def test_phase_c_is_gated_sentinel_then_completion(
    funnel, records, tmp_path: Path
) -> None:
    phase_a = _receipt(funnel, records, tmp_path, "4A")
    phase_b = _receipt(
        funnel, records, tmp_path, "4B", predecessors=(phase_a,)
    )
    predecessors = [phase_a, phase_b]
    sentinel = funnel.build_phase_plan(
        records,
        "4C",
        scope="sentinel",
        receipts=predecessors,
    )
    assert sentinel["record_count"] == 18
    assert all(len(shard["records"]) == 6 for shard in sentinel["shards"])
    assert all(
        member["record"]["mask"] == "none"
        or member["record"]["s3_area_fraction_percent"] == 25
        for shard in sentinel["shards"]
        for member in shard["records"]
    )
    completion = funnel.build_phase_plan(
        records,
        "4C",
        scope="completion",
        receipts=[
            *predecessors,
            _receipt(
                funnel,
                records,
                tmp_path,
                "4C_SENTINEL",
                predecessors=tuple(predecessors),
            ),
        ],
    )
    assert completion["record_count"] == 45
    assert all(len(shard["records"]) == 15 for shard in completion["shards"])
    sentinel_ids = {
        member["record_id"]
        for shard in sentinel["shards"]
        for member in shard["records"]
    }
    completion_ids = {
        member["record_id"]
        for shard in completion["shards"]
        for member in shard["records"]
    }
    assert sentinel_ids.isdisjoint(completion_ids)
    assert len(sentinel_ids | completion_ids) == 63


def test_phase_a_is_72_mixed_plus_nine_all_q4(funnel, records) -> None:
    selected = funnel.select_phase_records(records, "4A")
    mixed = [record for _index, record in selected if record["mask"] != "none"]
    baseline = [record for _index, record in selected if record["mask"] == "none"]
    assert len(mixed) == 72
    assert len(baseline) == 9
    assert {record["mask"] for record in mixed} == {"dispersed", "chain"}
    assert {record["level"] for _index, record in selected} == {20, 40, 80}


def test_phase_b_is_incremental_108_records(funnel, records) -> None:
    selected = funnel.select_phase_records(records, "4B")
    assert len(selected) == 108
    assert {record["mask"] for _index, record in selected} == {
        "compact_cluster",
        "boundary_band",
        "hole_band",
    }
    assert all(record["s3_area_fraction_percent"] != 0 for _index, record in selected)


def test_phase_c_is_all_63_n160_records(funnel, records) -> None:
    sentinel = funnel.select_phase_records(records, "4C", scope="sentinel")
    completion = funnel.select_phase_records(records, "4C", scope="completion")
    selected = (*sentinel, *completion)
    assert len(selected) == 63
    assert all(record["level"] == 160 for _index, record in selected)
    assert len([record for _index, record in selected if record["mask"] == "none"]) == 3


def test_phase_expansion_fails_closed_without_accepted_prerequisites(
    funnel, records, tmp_path: Path
) -> None:
    with pytest.raises(funnel.FlatFunnelError, match="missing=.*4A"):
        funnel.build_phase_plan(records, "4B")
    with pytest.raises(funnel.FlatFunnelError, match="not validated"):
        _receipt(funnel, records, tmp_path, "4A", terminal="BLOCKED")
    with pytest.raises(funnel.FlatFunnelError, match="not created"):
        funnel.build_phase_plan(records, "4B", receipts=[{"phase": "4A"}])
    phase_a = _receipt(funnel, records, tmp_path / "accepted", "4A")
    with pytest.raises(funnel.FlatFunnelError, match="missing=.*4B"):
        funnel.build_phase_plan(
            records,
            "4C",
            scope="sentinel",
            receipts=[phase_a],
        )
    phase_b = _receipt(
        funnel,
        records,
        tmp_path / "accepted",
        "4B",
        predecessors=(phase_a,),
    )
    with pytest.raises(funnel.FlatFunnelError, match="4C_SENTINEL"):
        funnel.build_phase_plan(
            records,
            "4C",
            scope="completion",
            receipts=[phase_a, phase_b],
        )


def test_receipt_revalidates_stored_plan_and_scientific_aggregate(
    funnel, records, tmp_path: Path
) -> None:
    plan_root = tmp_path / "plan-mutation"
    _receipt(funnel, records, plan_root, "4A")
    receipt_path = (plan_root / "4A" / "receipt.json").resolve()
    plan_path = plan_root / "4A" / "plan.json"
    aggregate_path = plan_root / "4A" / "aggregate.json"
    receipt, _ = funnel.strict_json_load(receipt_path)
    plan, _ = funnel.strict_json_load(plan_path)
    aggregate, _ = funnel.strict_json_load(aggregate_path)
    plan["selector"] = "legacy-s3"
    plan_raw = funnel.canonical_bytes(plan)
    plan_path.write_bytes(plan_raw)
    plan_sha = funnel.sha256(plan_raw)
    aggregate["plan_sha256"] = plan_sha
    aggregate_raw = funnel.canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_raw)
    receipt["plan_sha256"] = plan_sha
    receipt["aggregate_sha256"] = funnel.sha256(aggregate_raw)
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="plan identity differs"):
        funnel.load_validated_receipt(receipt_path, records)

    aggregate_root = tmp_path / "aggregate-mutation"
    _receipt(funnel, records, aggregate_root, "4A")
    receipt_path = (aggregate_root / "4A" / "receipt.json").resolve()
    aggregate_path = aggregate_root / "4A" / "aggregate.json"
    receipt, _ = funnel.strict_json_load(receipt_path)
    aggregate, _ = funnel.strict_json_load(aggregate_path)
    aggregate["complete_record_count"] = 80
    aggregate_raw = funnel.canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_raw)
    receipt["aggregate_sha256"] = funnel.sha256(aggregate_raw)
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="scientific aggregate differs"):
        funnel.load_validated_receipt(receipt_path, records)


def test_manifest_hash_or_coordinate_mutation_is_rejected(funnel) -> None:
    value, raw = funnel.strict_json_load(MANIFEST)
    mutated = json.loads(raw)
    mutated["records"][0]["level"] = 40
    with pytest.raises(funnel.FlatFunnelError, match="hash mismatch"):
        funnel.validate_manifest(mutated, funnel.canonical_bytes(mutated))
    with pytest.raises(funnel.FlatFunnelError, match="hash mismatch"):
        funnel.validate_manifest(value, raw + b" ")


def test_duplicate_and_nonfinite_json_are_rejected(funnel) -> None:
    with pytest.raises(funnel.FlatFunnelError, match="duplicate"):
        funnel.strict_json_bytes(b'{"x":1,"x":2}', label="duplicate")
    with pytest.raises(funnel.FlatFunnelError, match="non-finite"):
        funnel.strict_json_bytes(b'{"x":NaN}', label="nonfinite")


def test_progress_records_match_bounded_runner_schema(funnel, tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    first = funnel.progress_record(
        "S3_V2_FLAT_4A_SLASH",
        sequence=0,
        phase="INITIALIZED",
        completed=0,
        total=27,
    )
    second = funnel.progress_record(
        "S3_V2_FLAT_4A_SLASH",
        sequence=1,
        phase="RECORD_COMPLETED",
        completed=1,
        total=27,
    )
    funnel.append_progress(path, first)
    funnel.append_progress(path, second)
    bounded = _load("_s3_v2_bounded_for_funnel_test", BOUNDED)
    assert bounded._progress_sequence(path, "S3_V2_FLAT_4A_SLASH") == 1


def test_bounded_wave_manifest_is_three_v2_only_workers(
    funnel, records, tmp_path: Path
) -> None:
    plan = funnel.build_phase_plan(records, "4A")
    plan_path = (tmp_path / "phase-plan.json").resolve()
    plan_path.write_bytes(funnel.canonical_bytes(plan))
    producer_program = (tmp_path / "future-producer.py").resolve()
    producer_program.write_text("# frozen producer placeholder\n", encoding="utf-8")
    candidate = _candidate_binding(funnel, tmp_path)
    output_root = (tmp_path / "wave").resolve()
    made = funnel.build_bounded_wave_manifest(
        plan,
        plan_path=plan_path,
        producer_program=producer_program,
        python_executable=Path(__import__("sys").executable).resolve(),
        cwd=ROOT,
        output_root=output_root,
        input_paths={
            "candidate_artifact": candidate,
            "connectivity_manifest": MANIFEST,
            "flat_funnel_contract": CONTRACT,
            "source_equation_contract": SOURCE,
        },
    )
    assert made["lane"] == "flat-proof"
    assert len(made["workers"]) == 3
    assert all(worker["wall_seconds"] == 900 for worker in made["workers"])
    assert all(worker["plan_sha256"] == funnel.sha256(plan_path.read_bytes()) for worker in made["workers"])
    assert all(worker["program_sha256"] == funnel.sha256(producer_program.read_bytes()) for worker in made["workers"])
    assert all(worker["scientific_schema"] == funnel.SHARD_SCIENTIFIC_SCHEMA for worker in made["workers"])
    assert all(len(worker["input_hashes"]) == 4 for worker in made["workers"])
    assert all(worker["expected_selector"] == funnel.SELECTOR for worker in made["workers"])
    assert all(worker["expected_record_count"] == 27 for worker in made["workers"])
    assert all("e4-pl-s3-v2" in worker["command"] for worker in made["workers"])
    assert all("e4-pl-s3" not in worker["command"] for worker in made["workers"])
    bounded = _load("_s3_v2_bounded_manifest_test", BOUNDED)
    wave_id, lane, root, workers = bounded.validate_manifest(made)
    assert wave_id == "S3_V2_FLAT_FUNNEL_4A_FULL"
    assert lane == "flat-proof"
    assert root == output_root
    assert len(workers) == 3


def test_bounded_wave_rejects_open_ended_or_misnamed_inputs(
    funnel, records, tmp_path: Path
) -> None:
    plan = funnel.build_phase_plan(records, "4A")
    plan_path = (tmp_path / "plan.json").resolve()
    plan_path.write_bytes(funnel.canonical_bytes(plan))
    producer = (tmp_path / "producer.py").resolve()
    producer.write_text("# producer\n", encoding="utf-8")
    candidate = _candidate_binding(funnel, tmp_path)
    common = {
        "plan_path": plan_path,
        "producer_program": producer,
        "python_executable": Path(sys.executable).resolve(),
        "cwd": ROOT,
        "output_root": (tmp_path / "wave").resolve(),
    }
    with pytest.raises(funnel.FlatFunnelError, match="frozen-input names differ"):
        funnel.build_bounded_wave_manifest(
            plan,
            **common,
            input_paths={
                "candidate_artifact": candidate,
                "connectivity_manifest": MANIFEST,
            },
        )
    with pytest.raises(funnel.FlatFunnelError, match="candidate_binding"):
        funnel.build_bounded_wave_manifest(
            plan,
            **common,
            input_paths={
                "candidate_artifact": SOURCE,
                "connectivity_manifest": MANIFEST,
                "flat_funnel_contract": CONTRACT,
                "source_equation_contract": candidate,
            },
        )
    candidate.write_bytes(b"")
    with pytest.raises(funnel.FlatFunnelError, match="invalid JSON"):
        funnel.build_bounded_wave_manifest(
            plan,
            **common,
            input_paths={
                "candidate_artifact": candidate,
                "connectivity_manifest": MANIFEST,
                "flat_funnel_contract": CONTRACT,
                "source_equation_contract": SOURCE,
            },
        )


def test_receipt_rejects_noncompleted_or_rebound_wave_result(
    funnel, records, tmp_path: Path
) -> None:
    _receipt(funnel, records, tmp_path, "4A")
    root = tmp_path / "4A"
    receipt_path = (root / "receipt.json").resolve()
    result_path = root / "wave-result.json"
    aggregate_path = root / "aggregate.json"
    receipt, _ = funnel.strict_json_load(receipt_path)
    result, _ = funnel.strict_json_load(result_path)
    aggregate, _ = funnel.strict_json_load(aggregate_path)
    result["terminal"] = "BLOCKED_E4_PL_S3_V2_PROCESS_OR_EVIDENCE"
    result_raw = funnel.canonical_bytes(result)
    result_path.write_bytes(result_raw)
    result_sha = funnel.sha256(result_raw)
    aggregate["wave_result_sha256"] = result_sha
    aggregate_raw = funnel.canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_raw)
    receipt["wave_result_sha256"] = result_sha
    receipt["aggregate_sha256"] = funnel.sha256(aggregate_raw)
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="not a completed registered wave"):
        funnel.load_validated_receipt(receipt_path, records)


def test_receipt_rejects_wave_manifest_rebinding_even_when_outer_hashes_follow(
    funnel, records, tmp_path: Path
) -> None:
    _receipt(funnel, records, tmp_path, "4A")
    root = tmp_path / "4A"
    receipt_path = (root / "receipt.json").resolve()
    manifest_path = root / "wave-manifest.json"
    result_path = root / "wave-result.json"
    aggregate_path = root / "aggregate.json"
    receipt, _ = funnel.strict_json_load(receipt_path)
    manifest, _ = funnel.strict_json_load(manifest_path)
    result, _ = funnel.strict_json_load(result_path)
    aggregate, _ = funnel.strict_json_load(aggregate_path)
    manifest["workers"][0]["assignment_sha256"] = "A" * 64
    manifest_raw = funnel.canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_raw)
    manifest_sha = funnel.sha256(manifest_raw)
    result["manifest_sha256"] = manifest_sha
    result["workers"][0]["assignment_sha256"] = "A" * 64
    result_raw = funnel.canonical_bytes(result)
    result_path.write_bytes(result_raw)
    result_sha = funnel.sha256(result_raw)
    aggregate["wave_manifest_sha256"] = manifest_sha
    aggregate["wave_result_sha256"] = result_sha
    aggregate_raw = funnel.canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_raw)
    receipt["wave_manifest_sha256"] = manifest_sha
    receipt["wave_result_sha256"] = result_sha
    receipt["aggregate_sha256"] = funnel.sha256(aggregate_raw)
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="differs from its plan shard"):
        funnel.load_validated_receipt(receipt_path, records)


def test_receipt_rejects_shard_coverage_even_when_claimed_hashes_are_recomputed(
    funnel, records, tmp_path: Path
) -> None:
    _receipt(funnel, records, tmp_path, "4A")
    root = tmp_path / "4A"
    receipt_path = (root / "receipt.json").resolve()
    result_path = root / "wave-result.json"
    aggregate_path = root / "aggregate.json"
    receipt, _ = funnel.strict_json_load(receipt_path)
    result, _ = funnel.strict_json_load(result_path)
    aggregate, _ = funnel.strict_json_load(aggregate_path)
    manifest, _ = funnel.strict_json_load(root / "wave-manifest.json")
    science_path = Path(manifest["workers"][0]["scientific_path"])
    science, _ = funnel.strict_json_load(science_path)
    science["record_ids"][0], science["record_ids"][1] = (
        science["record_ids"][1],
        science["record_ids"][0],
    )
    science["record_ids_sha256"] = funnel.sha256(
        funnel.canonical_bytes(science["record_ids"])
    )
    science_raw = funnel.canonical_bytes(science)
    science_path.write_bytes(science_raw)
    science_sha = funnel.sha256(science_raw)
    result["workers"][0]["scientific_sha256"] = science_sha
    result["workers"][0]["scientific_byte_count"] = len(science_raw)
    result_raw = funnel.canonical_bytes(result)
    result_path.write_bytes(result_raw)
    result_sha = funnel.sha256(result_raw)
    aggregate["wave_result_sha256"] = result_sha
    aggregate_raw = funnel.canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_raw)
    receipt["wave_result_sha256"] = result_sha
    receipt["aggregate_sha256"] = funnel.sha256(aggregate_raw)
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="identity or coverage differs"):
        funnel.load_validated_receipt(receipt_path, records)


def test_receipt_rejects_shard_payload_and_invented_aggregate_hashes(
    funnel, records, tmp_path: Path
) -> None:
    payload_root = tmp_path / "payload"
    _receipt(funnel, records, payload_root, "4A")
    receipt_path = (payload_root / "4A" / "receipt.json").resolve()
    manifest, _ = funnel.strict_json_load(payload_root / "4A" / "wave-manifest.json")
    science_path = Path(manifest["workers"][0]["scientific_path"])
    science, _ = funnel.strict_json_load(science_path)
    science["scientific_payload"]["invented"] = True
    science_path.write_bytes(funnel.canonical_bytes(science))
    with pytest.raises(funnel.FlatFunnelError, match="payload hash mismatch"):
        funnel.load_validated_receipt(receipt_path, records)

    aggregate_root = tmp_path / "aggregate-claim"
    _receipt(funnel, records, aggregate_root, "4A")
    receipt_path = (aggregate_root / "4A" / "receipt.json").resolve()
    aggregate_path = aggregate_root / "4A" / "aggregate.json"
    receipt, _ = funnel.strict_json_load(receipt_path)
    aggregate, _ = funnel.strict_json_load(aggregate_path)
    aggregate["scientific_payload_sha256"] = "F" * 64
    aggregate_raw = funnel.canonical_bytes(aggregate)
    aggregate_path.write_bytes(aggregate_raw)
    receipt["aggregate_sha256"] = funnel.sha256(aggregate_raw)
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="scientific aggregate differs"):
        funnel.load_validated_receipt(receipt_path, records)


def test_receipt_recursion_rejects_cycles_and_plan_token_rebinding(
    funnel, records, tmp_path: Path
) -> None:
    phase_a = _receipt(funnel, records, tmp_path / "cycle", "4A")
    _receipt(
        funnel,
        records,
        tmp_path / "cycle",
        "4B",
        predecessors=(phase_a,),
    )
    receipt_path = (tmp_path / "cycle" / "4B" / "receipt.json").resolve()
    receipt, _ = funnel.strict_json_load(receipt_path)
    receipt["prerequisite_receipt_paths"] = [str(receipt_path)]
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="cyclic"):
        funnel.load_validated_receipt(receipt_path, records)

    phase_a = _receipt(funnel, records, tmp_path / "token", "4A")
    _receipt(
        funnel,
        records,
        tmp_path / "token",
        "4B",
        predecessors=(phase_a,),
    )
    receipt_path = (tmp_path / "token" / "4B" / "receipt.json").resolve()
    plan_path = tmp_path / "token" / "4B" / "plan.json"
    receipt, _ = funnel.strict_json_load(receipt_path)
    plan, _ = funnel.strict_json_load(plan_path)
    plan["prerequisites"][0]["aggregate_sha256"] = "E" * 64
    plan_raw = funnel.canonical_bytes(plan)
    plan_path.write_bytes(plan_raw)
    receipt["plan_sha256"] = funnel.sha256(plan_raw)
    receipt_path.write_bytes(funnel.canonical_bytes(receipt))
    with pytest.raises(funnel.FlatFunnelError, match="token bindings differ"):
        funnel.load_validated_receipt(receipt_path, records)


def test_cli_emits_plan_exclusively_without_scientific_execution(
    funnel, tmp_path: Path
) -> None:
    output = tmp_path / "plan.json"
    assert (
        funnel.main(
            [
                "--manifest",
                str(MANIFEST),
                "--contract",
                str(CONTRACT),
                "--phase",
                "4A",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload, _raw = funnel.strict_json_load(output)
    assert payload["record_count"] == 81
    with pytest.raises(FileExistsError):
        funnel.main(
            [
                "--manifest",
                str(MANIFEST),
                "--contract",
                str(CONTRACT),
                "--phase",
                "4A",
                "--output",
                str(output),
            ]
        )
