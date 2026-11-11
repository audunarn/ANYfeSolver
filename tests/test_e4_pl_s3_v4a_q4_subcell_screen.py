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

from e4_pl_s3_v4a_screen_checker import load_document, verify
from e4_pl_s3_v4a_screen_producer import adjudicate, produce_proof, run_bounded


def test_contract_binds_preregistration_q4_programs_and_process_bounds() -> None:
    contract = json.loads((REFERENCE / "e4_pl_s3_v4a_screen_contract.json").read_text(encoding="ascii"))
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


def test_screen_uses_only_qualified_q4_and_checker_does_not_import_producer() -> None:
    producer = ast.parse((REFERENCE / "e4_pl_s3_v4a_screen_producer.py").read_text(encoding="utf-8"))
    checker = ast.parse((REFERENCE / "e4_pl_s3_v4a_screen_checker.py").read_text(encoding="utf-8"))

    def imported(tree: ast.AST) -> set[str]:
        result: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result.add(node.module)
        return result

    producer_imports = imported(producer)
    checker_imports = imported(checker)
    assert "anyfesolver.e4_pl_s3_element" not in producer_imports
    assert "anyfesolver.e4_pl_s3_v2_element" not in producer_imports
    assert "e4_pl_s3_v4a_screen_producer" not in checker_imports


def test_independent_reconstruction_finds_registered_physical_rank_contradiction() -> None:
    proof = produce_proof()
    report = verify(proof)
    assert report["authority_complete"] is True
    assert report["construction_identity_passed"] is True
    assert report["independent_centre_block_rank"] == 6
    assert report["independent_physical_rank"] == 11
    assert report["independent_total_rank"] == 12
    assert float.fromhex(report["independent_rigid_residual_relative_inf_hex"]) <= 3.0e-12
    assert float.fromhex(report["independent_local_identity_worst_relative_inf_hex"]) <= 3.0e-13
    assert report["local_operator_passed"] is False
    assert report["later_stages_absent"] is True
    assert proof["macrocell"] == {}
    assert proof["development_records"] == []
    assert proof["later_stages"] == "NOT_EXECUTED_LOCAL_GATE_FAILED"


def test_payload_mutation_duplicate_and_nonfinite_json_fail_closed(tmp_path: Path) -> None:
    proof = produce_proof()
    mutated = copy.deepcopy(proof)
    mutated["local"]["payloads"]["physical"]["values_hex"][0] = float(1.0).hex()
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(mutated)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="ascii")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate"):
        load_document(duplicate)
    with pytest.raises(ValueError):
        load_document(nonfinite)


def test_terminal_precedence() -> None:
    base = {
        "authority_complete": True,
        "construction_identity_passed": True,
        "local_operator_passed": True,
        "mixed_interface_passed": True,
    }
    assert adjudicate(identical=False, report=base) == "BLOCKED_E4_PL_S3_V4A_PROCESS_OR_EVIDENCE"
    assert adjudicate(identical=True, report=base | {"construction_identity_passed": False}) == "NO_GO_E4_PL_S3_V4A_CONSTRUCTION_IDENTITY"
    assert adjudicate(identical=True, report=base | {"local_operator_passed": False}) == "NO_GO_E4_PL_S3_V4A_LOCAL_OPERATOR"
    assert adjudicate(identical=True, report=base | {"mixed_interface_passed": False}) == "NO_GO_E4_PL_S3_V4A_MIXED_INTERFACE"
    assert adjudicate(identical=True, report=base) == "PROVISIONAL_GO_E4_PL_S3_V4A_STAGE4A_RERUN"


def test_two_bounded_cycles_are_byte_identical_and_stop_at_local_gate(tmp_path: Path) -> None:
    aggregate = run_bounded(tmp_path / "screen", timeout_seconds=60, wave_timeout_seconds=180)
    assert aggregate["terminal"] == "NO_GO_E4_PL_S3_V4A_LOCAL_OPERATOR"
    assert aggregate["stage4a_rerun_authorized"] is False
    assert aggregate["activation_authorized"] is False
    assert len(aggregate["cycles"]) == 2
    assert aggregate["cycles"][0]["proof_sha256"] == aggregate["cycles"][1]["proof_sha256"]
    assert aggregate["cycles"][0]["checker_sha256"] == aggregate["cycles"][1]["checker_sha256"]
    assert all(cycle["checker_replicas_byte_identical"] is True for cycle in aggregate["cycles"])


def test_production_boundary_remains_static() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v4a*"))


def test_closeout_when_present_binds_no_go_review_and_status() -> None:
    status_path = REFERENCE / "e4_pl_s3_v4a_screen_status.json"
    if not status_path.exists():
        return
    result_path = REFERENCE / "e4_pl_s3_v4a_screen_result.json"
    review_path = REFERENCE / "e4_pl_s3_v4a_screen_review.json"
    status = json.loads(status_path.read_text(encoding="ascii"))
    result = json.loads(result_path.read_text(encoding="ascii"))
    review = json.loads(review_path.read_text(encoding="ascii"))
    assert result["terminal"] == status["terminal"] == "NO_GO_E4_PL_S3_V4A_LOCAL_OPERATOR"
    assert result["diagnosis"]["independent_physical_rank"] == 11
    assert result["diagnosis"]["independent_total_rank"] == 12
    assert result["later_stages"]["macrocell_trace_executed"] is False
    assert result["later_stages"]["development_n20_n40_executed"] is False
    assert review["findings"] == {"P0": [], "P1": []}
    assert status["result"] == {
        "bytes": result_path.stat().st_size,
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest().upper(),
    }
    assert status["review"] == {
        "bytes": review_path.stat().st_size,
        "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest().upper(),
    }
    assert status["activation_authorized"] is status["stage4a_rerun_authorized"] is False
