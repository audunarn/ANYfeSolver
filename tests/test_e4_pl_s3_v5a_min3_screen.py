"""Authority, mechanics, evidence, and boundary tests for the V5A MIN3 screen."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from e4_pl_s3_v5a_screen_checker import reconstruct, verify
from e4_pl_s3_v5a_screen_producer import (
    adjudicate,
    local_proof,
    macrocell_proof,
    min3_components,
    produce_proof,
    relative_inf,
    run_bounded,
)


def test_source_maps_are_independent_and_agree() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v5a_screen_checker.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "e4_pl_s3_v5a_screen_producer" not in imports
    assert "anyfesolver.e4_pl_s3_element" not in imports
    assert "anyfesolver.e4_pl_s3_v2_element" not in imports
    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    produced = min3_components(vertices)
    checked = reconstruct(vertices)
    assert max(relative_inf(produced[name], value) for name, value in checked.items()) <= 3.0e-13


def test_local_macrocell_and_preregistered_development_gate_pass() -> None:
    local = local_proof()
    assert local["diagnostics"]["gate_passed"] is True
    assert local["diagnostics"]["physical_rank"] == 9
    assert local["diagnostics"]["pl_rank"] == 3
    assert local["diagnostics"]["total_rank"] == 12
    macro = macrocell_proof()
    assert macro["diagnostics"]["record_count"] == 21
    assert macro["diagnostics"]["gate_passed"] is True
    # Q4 D-to-N equality is deliberately diagnostic, not a patch gate.
    assert float.fromhex(macro["diagnostics"]["dtn_q4_diagnostic_worst_relative_inf_hex"]) > 0.0
    proof = produce_proof()
    report = verify(proof)
    assert report["authority_complete"] is True
    assert report["construction_identity_passed"] is True
    assert report["local_operator_passed"] is True
    assert report["mixed_interface_passed"] is True
    assert report["development_passed"] is True
    assert proof["development"]["diagnostics"]["record_count"] == 4
    assert proof["later_stages"] == "RELAXATION_AUTHORITY_REQUIRED"


def test_payload_mutation_duplicate_and_nonfinite_fail_closed(tmp_path: Path) -> None:
    proof = produce_proof()
    mutated = copy.deepcopy(proof)
    mutated["local"]["payloads"]["shear"]["values_hex"][0] = float(1.0).hex()
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(mutated)
    duplicate = tmp_path / "duplicate.json"
    nonfinite = tmp_path / "nonfinite.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="ascii")
    nonfinite.write_text('{"a":NaN}\n', encoding="ascii")
    from e4_pl_s3_v5a_screen_checker import load_document

    with pytest.raises(ValueError, match="duplicate"):
        load_document(duplicate)
    with pytest.raises(ValueError):
        load_document(nonfinite)


def test_terminal_precedence_and_two_bounded_cycles(tmp_path: Path) -> None:
    base = {
        "authority_complete": True,
        "construction_identity_passed": True,
        "development_passed": True,
        "local_operator_passed": True,
        "mixed_interface_passed": True,
    }
    assert adjudicate(identical=False, report=base) == "BLOCKED_E4_PL_S3_V5A_PROCESS_OR_EVIDENCE"
    assert adjudicate(identical=True, report=base | {"local_operator_passed": False}) == "NO_GO_E4_PL_S3_V5A_SOURCE_OR_LOCAL_OPERATOR"
    assert adjudicate(identical=True, report=base | {"mixed_interface_passed": False}) == "NO_GO_E4_PL_S3_V5A_MIXED_INTERFACE"
    assert adjudicate(identical=True, report=base) == "UNCLASSIFIED_E4_PL_S3_V5A_RELAXATION_AUTHORITY_REQUIRED"
    aggregate = run_bounded(tmp_path / "bounded", timeout_seconds=120, wave_timeout_seconds=600)
    assert aggregate["terminal"] == "UNCLASSIFIED_E4_PL_S3_V5A_RELAXATION_AUTHORITY_REQUIRED"
    assert aggregate["stage4a_rerun_authorized"] is aggregate["activation_authorized"] is False
    assert aggregate["relaxed_successor_screen_authorized"] is True
    assert len(aggregate["cycles"]) == 2
    assert aggregate["cycles"][0]["proof_sha256"] == aggregate["cycles"][1]["proof_sha256"]
    assert aggregate["cycles"][0]["checker_sha256"] == aggregate["cycles"][1]["checker_sha256"]
    assert all(cycle["checker_replicas_byte_identical"] is True for cycle in aggregate["cycles"])


def test_production_boundary_and_optional_closeout() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v5a*"))
    status_path = REFERENCE / "e4_pl_s3_v5a_screen_status.json"
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text(encoding="ascii"))
    result = json.loads((REFERENCE / "e4_pl_s3_v5a_screen_result.json").read_text(encoding="ascii"))
    review = json.loads((REFERENCE / "e4_pl_s3_v5a_screen_review.json").read_text(encoding="ascii"))
    assert status["terminal"] == result["terminal"] == "UNCLASSIFIED_E4_PL_S3_V5A_RELAXATION_AUTHORITY_REQUIRED"
    assert review["findings"] == {"P0": [], "P1": []}
    assert result["activation_authorized"] is result["stage4a_rerun_authorized"] is False
