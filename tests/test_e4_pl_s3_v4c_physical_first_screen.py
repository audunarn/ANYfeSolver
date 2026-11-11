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

from e4_pl_s3_v4c_screen_checker import load_document, verify
from e4_pl_s3_v4c_screen_producer import adjudicate, produce_proof, run_bounded


def test_contract_binds_physical_q4_native_pl_programs_and_bounds() -> None:
    contract = load_document(REFERENCE / "e4_pl_s3_v4c_screen_contract.json")
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]
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


def test_checker_does_not_import_v4c_producer_or_failed_s3_mechanics() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v4c_screen_checker.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "e4_pl_s3_v4c_screen_producer" not in imports
    assert "anyfesolver.e4_pl_s3_element" not in imports
    assert "anyfesolver.e4_pl_s3_v2_element" not in imports


def test_local_gate_passes_and_complete_macrocell_gate_fails_independently() -> None:
    proof = produce_proof()
    report = verify(proof)
    assert report["authority_complete"] is True
    assert report["construction_identity_passed"] is True
    assert report["independent_internal_physical_block_rank"] == 5
    assert report["independent_physical_rank"] == 9
    assert report["independent_pl_rank"] == 3
    assert report["independent_total_rank"] == 12
    assert report["local_operator_passed"] is True
    assert report["independent_macrocell_record_count"] == 9
    assert float.fromhex(report["independent_macrocell_identity_worst_relative_inf_hex"]) <= 3.0e-13
    assert float.fromhex(report["independent_macrocell_trace_worst_relative_inf_hex"]) > 3.0e-12
    assert report["mixed_interface_passed"] is False
    assert report["later_stages_absent"] is True
    assert len(proof["macrocell"]["records"]) == 9
    assert proof["development_records"] == []
    assert proof["later_stages"] == "NOT_EXECUTED_MACROCELL_GATE_FAILED"


def test_payload_mutation_duplicate_and_nonfinite_json_fail_closed(tmp_path: Path) -> None:
    proof = produce_proof()
    mutated = copy.deepcopy(proof)
    mutated["macrocell"]["matrices"]["1x1:slash"]["s3"]["values_hex"][0] = float(1.0).hex()
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
    base = {"authority_complete": True, "construction_identity_passed": True, "local_operator_passed": True, "mixed_interface_passed": True}
    assert adjudicate(identical=False, report=base) == "BLOCKED_E4_PL_S3_V4C_PROCESS_OR_EVIDENCE"
    assert adjudicate(identical=True, report=base | {"construction_identity_passed": False}) == "NO_GO_E4_PL_S3_V4C_CONSTRUCTION_IDENTITY"
    assert adjudicate(identical=True, report=base | {"local_operator_passed": False}) == "NO_GO_E4_PL_S3_V4C_LOCAL_OPERATOR"
    assert adjudicate(identical=True, report=base | {"mixed_interface_passed": False}) == "NO_GO_E4_PL_S3_V4C_MIXED_INTERFACE"
    assert adjudicate(identical=True, report=base) == "PROVISIONAL_GO_E4_PL_S3_V4C_STAGE4A_RERUN"


def test_two_bounded_cycles_are_identical_and_stop_before_development(tmp_path: Path) -> None:
    aggregate = run_bounded(tmp_path / "screen", timeout_seconds=60, wave_timeout_seconds=180)
    assert aggregate["terminal"] == "NO_GO_E4_PL_S3_V4C_MIXED_INTERFACE"
    assert aggregate["activation_authorized"] is aggregate["stage4a_rerun_authorized"] is False
    assert len(aggregate["cycles"]) == 2
    assert aggregate["cycles"][0]["proof_sha256"] == aggregate["cycles"][1]["proof_sha256"]
    assert aggregate["cycles"][0]["checker_sha256"] == aggregate["cycles"][1]["checker_sha256"]
    assert all(cycle["checker_replicas_byte_identical"] is True for cycle in aggregate["cycles"])


def test_production_boundary_remains_static() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v4c*"))


def test_closeout_when_present_binds_review_and_status() -> None:
    status_path = REFERENCE / "e4_pl_s3_v4c_screen_status.json"
    if not status_path.exists():
        return
    result_path = REFERENCE / "e4_pl_s3_v4c_screen_result.json"
    review_path = REFERENCE / "e4_pl_s3_v4c_screen_review.json"
    status = json.loads(status_path.read_text(encoding="ascii"))
    result = json.loads(result_path.read_text(encoding="ascii"))
    review = json.loads(review_path.read_text(encoding="ascii"))
    assert result["terminal"] == status["terminal"] == "NO_GO_E4_PL_S3_V4C_MIXED_INTERFACE"
    assert result["diagnosis"]["independent_physical_rank"] == 9
    assert result["diagnosis"]["macrocell_record_count"] == 9
    assert result["later_stages"]["development_n20_n40_executed"] is False
    assert review["findings"] == {"P0": [], "P1": []}
    assert status["result"] == {"bytes": result_path.stat().st_size, "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest().upper()}
    assert status["review"] == {"bytes": review_path.stat().st_size, "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest().upper()}
