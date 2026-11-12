from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from e4_pl_s3_v3a_screen_checker import verify
from e4_pl_s3_v3a_screen_producer import adjudicate, produce_proof, run_bounded


def test_contract_binds_programs_predecessors_and_process_limits() -> None:
    contract = json.loads((REFERENCE / "e4_pl_s3_v3a_implementation_screen_contract.json").read_text(encoding="ascii"))
    for entry in contract["frozen_inputs"]:
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == entry["sha256"]
    assert contract["execution"] == {
        "checker_replica_count": 2,
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "exclusive_outputs": True,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_cycle_count": 2,
    }


def test_source_native_producer_does_not_import_failed_mechanics() -> None:
    path = REFERENCE / "e4_pl_s3_v3a_screen_producer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "anyfesolver.e4_pl_s3_element" not in imports
    assert "anyfesolver.e4_pl_s3_v2_element" not in imports
    assert "e4_pl_s3_v2_flat_funnel_producer" not in imports


def test_independent_exact_reconstruction_finds_registered_local_rank_contradiction() -> None:
    proof = produce_proof(include_development=True)
    report = verify(proof)
    assert report["source_identity_passed"] is True
    assert float.fromhex(report["source_identity_worst_relative_inf_hex"]) <= 3.0e-13
    assert report["rigid_modes_exact"] is True
    assert report["exact_coupling_rank"] == 5
    assert report["exact_physical_rank"] == 8
    assert report["exact_total_rank"] == 11
    assert report["local_operator_passed"] is False
    assert proof["diagnostics"]["later_screen_stages"] == "NOT_EXECUTED_LOCAL_GATE_FAILED"
    assert proof["matrices"] == proof["boundary_maps"] == {}
    assert proof["development_records"] == []


def test_payload_mutation_and_duplicate_or_nonfinite_json_fail_closed(tmp_path: Path) -> None:
    proof = produce_proof(include_development=False)
    mutated = copy.deepcopy(proof)
    mutated["local_payloads"]["G"]["values_hex"][0] = float(1.0).hex()
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(mutated)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="ascii")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="ascii")
    from e4_pl_s3_v3a_screen_checker import load_document

    with pytest.raises(ValueError, match="duplicate"):
        load_document(duplicate)
    with pytest.raises(ValueError):
        load_document(nonfinite)


def test_terminal_precedence() -> None:
    base = {
        "authority_complete": True,
        "source_identity_passed": True,
        "local_operator_passed": True,
        "mixed_interface_passed": True,
        "replacement_required": False,
    }
    assert adjudicate(checker_identical=False, report=base) == "BLOCKED_E4_PL_S3_V3_SOURCE_OR_EVIDENCE"
    assert adjudicate(checker_identical=True, report=base | {"source_identity_passed": False}) == "NO_GO_E4_PL_S3_V3_SOURCE_IDENTITY"
    assert adjudicate(checker_identical=True, report=base | {"local_operator_passed": False}) == "NO_GO_E4_PL_S3_V3A_LOCAL_OPERATOR"
    assert adjudicate(checker_identical=True, report=base | {"mixed_interface_passed": False}) == "NO_GO_E4_PL_S3_V3A_MIXED_INTERFACE"
    assert adjudicate(checker_identical=True, report=base | {"replacement_required": True}) == "UNCLASSIFIED_E4_PL_S3_V3_FORMULATION_REPLACEMENT_REQUIRED"
    assert adjudicate(checker_identical=True, report=base) == "PROVISIONAL_GO_E4_PL_S3_V3A_STAGE4A_RERUN"


def test_two_bounded_cycles_are_byte_identical_and_stop_at_local_gate(tmp_path: Path) -> None:
    aggregate = run_bounded(tmp_path / "screen", timeout_seconds=60, wave_timeout_seconds=180)
    assert aggregate["terminal"] == "NO_GO_E4_PL_S3_V3A_LOCAL_OPERATOR"
    assert aggregate["stage4a_rerun_authorized"] is False
    assert aggregate["activation_authorized"] is False
    assert len(aggregate["cycles"]) == 2
    assert aggregate["cycles"][0]["proof_sha256"] == aggregate["cycles"][1]["proof_sha256"]
    assert aggregate["cycles"][0]["checker_sha256"] == aggregate["cycles"][1]["checker_sha256"]
    assert aggregate["cycles"][0]["checker_replicas_byte_identical"] is True
    assert aggregate["cycles"][1]["checker_replicas_byte_identical"] is True


def test_production_boundary_remains_static() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v3a*"))


def test_canonical_no_go_closeout_binds_evidence_and_review() -> None:
    result_path = REFERENCE / "e4_pl_s3_v3a_implementation_screen_result.json"
    review_path = REFERENCE / "e4_pl_s3_v3a_implementation_screen_review.json"
    status = json.loads((REFERENCE / "e4_pl_s3_v3a_implementation_screen_status.json").read_text(encoding="ascii"))
    result = json.loads(result_path.read_text(encoding="ascii"))
    review = json.loads(review_path.read_text(encoding="ascii"))
    assert result["terminal"] == status["terminal"] == "NO_GO_E4_PL_S3_V3A_LOCAL_OPERATOR"
    assert result["diagnosis"]["exact_coupling_rank"] == 5
    assert result["diagnosis"]["exact_total_rank"] == 11
    assert result["later_stages"]["macrocell_trace_executed"] is False
    assert result["later_stages"]["development_n20_n40_executed"] is False
    assert review["findings"] == {"P0": [], "P1": []}
    assert status["result"] == {"bytes": result_path.stat().st_size, "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest().upper()}
    assert status["review"] == {"bytes": review_path.stat().st_size, "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest().upper()}
    assert status["activation_authorized"] is status["stage4a_rerun_authorized"] is False
