from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from e4_pl_s3_v4e_shear_diagnosis_checker import load_document, verify
from e4_pl_s3_v4e_shear_diagnosis_producer import adjudicate, produce_proof, run_bounded


def test_contract_binds_closed_v4d_programs_and_bounds() -> None:
    contract = load_document(REFERENCE / "e4_pl_s3_v4e_shear_diagnosis_contract.json")
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]
    assert contract["execution"]["child_wall_seconds"] == 600
    assert contract["execution"]["complete_wave_wall_seconds"] == 1800
    assert contract["execution"]["no_automatic_retry"] is True


def test_checker_does_not_import_v4e_producer_or_failed_s3_mechanics() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v4e_shear_diagnosis_checker.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "e4_pl_s3_v4e_shear_diagnosis_producer" not in imports
    assert "anyfesolver.e4_pl_s3_element" not in imports
    assert "anyfesolver.e4_pl_s3_v2_element" not in imports


def test_independent_diagnosis_is_identity_complete_and_shear_limited() -> None:
    proof = produce_proof()
    report = verify(proof)
    assert report["authority_complete"] is True
    assert report["diagnostic_identity_passed"] is True
    assert report["independent_record_count"] == 8
    assert report["zero_pl_work"] is True
    assert report["thin_limit_stable"] is True
    assert report["shear_replacement_required"] is True
    assert report["later_stages_absent"] is True
    assert proof["classification"]["shear_replacement_required"] is True
    assert len(proof["records"]) == 8


def test_record_mutation_duplicate_and_nonfinite_json_fail_closed(tmp_path: Path) -> None:
    proof = produce_proof()
    mutated = copy.deepcopy(proof)
    mutated["records"][0]["energy_ratio_hex"] = float(2.0).hex()
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
    base = {"authority_complete": True, "diagnostic_identity_passed": True, "shear_replacement_required": True}
    assert adjudicate(identical=False, report=base) == "BLOCKED_E4_PL_S3_V4E_PROCESS_OR_EVIDENCE"
    assert adjudicate(identical=True, report=base | {"diagnostic_identity_passed": False}) == "NO_GO_E4_PL_S3_V4E_DIAGNOSTIC_IDENTITY"
    assert adjudicate(identical=True, report=base | {"shear_replacement_required": False}) == "UNCLASSIFIED_E4_PL_S3_V4E_SHEAR_SOURCE_UNRESOLVED"
    assert adjudicate(identical=True, report=base) == "UNCLASSIFIED_E4_PL_S3_V4E_SHEAR_FORMULATION_REPLACEMENT_REQUIRED"


def test_two_bounded_cycles_are_identical(tmp_path: Path) -> None:
    aggregate = run_bounded(tmp_path / "diagnosis", timeout_seconds=60, wave_timeout_seconds=180)
    assert aggregate["terminal"] == "UNCLASSIFIED_E4_PL_S3_V4E_SHEAR_FORMULATION_REPLACEMENT_REQUIRED"
    assert aggregate["activation_authorized"] is aggregate["stage4a_rerun_authorized"] is False
    assert len(aggregate["cycles"]) == 2
    assert aggregate["cycles"][0]["proof_sha256"] == aggregate["cycles"][1]["proof_sha256"]
    assert aggregate["cycles"][0]["checker_sha256"] == aggregate["cycles"][1]["checker_sha256"]
    assert all(cycle["checker_replicas_byte_identical"] is True for cycle in aggregate["cycles"])


def test_production_boundary_remains_static() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v4e*"))
