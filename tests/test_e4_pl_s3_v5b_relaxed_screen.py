"""Authority, mechanics, process, and boundary tests for the V5B funnel."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

import e4_pl_s3_v5b_relaxed_screen_checker as checker
import e4_pl_s3_v5b_relaxed_screen_producer as producer


CONTRACT = REFERENCE / "e4_pl_s3_v5b_relaxed_screen_contract.json"


def test_checker_is_independent_of_v5b_producer_and_production_s3() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v5b_relaxed_screen_checker.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "e4_pl_s3_v5b_relaxed_screen_producer" not in imports
    assert "anyfesolver.e4_pl_s3_element" not in imports
    assert "anyfesolver.e4_pl_s3_v2_element" not in imports


def test_exact_source_relaxation_and_independent_reconstruction() -> None:
    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.9, 0.0)))
    produced = producer.min3_components(vertices)
    checked = checker.reconstruct(vertices)
    assert 0.0 < produced["phi_squared"] < 1.0
    alpha = produced["unrelaxed_shear_rotational_diagonal_sum"] / produced["bending_rotational_diagonal_sum"]
    assert produced["phi_squared"] == pytest.approx(1.0 / (1.0 + 0.5 * alpha), rel=0.0, abs=1.0e-18)
    arrays = [name for name, value in checked.items() if isinstance(value, np.ndarray)]
    assert max(producer.relative_inf(produced[name], checked[name]) for name in arrays) <= 3.0e-13


def test_local_macrocell_and_development_gates_pass() -> None:
    local = producer.local_proof()
    assert local["diagnostics"]["gate_passed"] is True
    assert local["diagnostics"]["physical_rank"] == 9
    assert local["diagnostics"]["pl_rank"] == 3
    assert local["diagnostics"]["total_rank"] == 12
    macro = producer.macrocell_proof()
    assert macro["diagnostics"]["record_count"] == 21
    assert macro["diagnostics"]["gate_passed"] is True
    rows = [producer._record(*spec) for spec in producer.DEVELOPMENT]
    assert producer._development_gate(rows)["gate_passed"] is True


def test_campaign_and_thin_specs_are_complete_and_nonoverlapping() -> None:
    campaign = set(producer.DISPERSED_CAMPAIGN + producer.CHAIN_HOLDOUT + producer.ALL_S3_CONTROL)
    assert len(producer.DISPERSED_CAMPAIGN) == 36
    assert len(campaign) == 42
    assert len(producer.THIN_THICKNESSES) * 2 == 12
    assert set(producer.DEVELOPMENT) <= campaign


def test_n80_chain_residual_is_hardened_without_relaxing_gate() -> None:
    row = producer._record(80, 10, "chain", "slash")
    assert float.fromhex(row["solve_residual_relative_inf_hex"]) <= 1.0e-8


def test_terminal_precedence_and_no_direct_stage4a_authority() -> None:
    base = {
        "authority_complete": True,
        "campaign_passed": True,
        "construction_identity_passed": True,
        "local_operator_passed": True,
        "mixed_interface_passed": True,
        "thin_regime_passed": True,
    }
    assert producer.adjudicate(identical=False, report=base) == "BLOCKED_E4_PL_S3_V5B_PROCESS_OR_EVIDENCE"
    assert producer.adjudicate(identical=True, report=base | {"local_operator_passed": False}) == "NO_GO_E4_PL_S3_V5B_RELAXATION_SOURCE_OR_LOCAL_OPERATOR"
    assert producer.adjudicate(identical=True, report=base | {"mixed_interface_passed": False}) == "NO_GO_E4_PL_S3_V5B_MIXED_INTERFACE"
    assert producer.adjudicate(identical=True, report=base | {"thin_regime_passed": False}) == "NO_GO_E4_PL_S3_V5B_THIN_REGIME"
    assert producer.adjudicate(identical=True, report=base) == "PROVISIONAL_GO_E4_PL_S3_V5B_STAGE4A_RERUN"


def test_payload_mutation_duplicate_and_nonfinite_fail_closed(tmp_path: Path) -> None:
    local = producer.local_proof()
    mutated = copy.deepcopy(local["payloads"]["shear"])
    mutated["values_hex"][0] = float(1.0).hex()
    with pytest.raises(ValueError, match="hash mismatch"):
        checker.decode_array(mutated)
    duplicate = tmp_path / "duplicate.json"
    nonfinite = tmp_path / "nonfinite.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="ascii")
    nonfinite.write_text('{"a":NaN}\n', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate"):
        checker.load_document(duplicate)
    with pytest.raises(ValueError):
        checker.load_document(nonfinite)


def test_contract_freezes_coverage_bounds_thresholds_and_inputs() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    assert contract["coverage"] == {
        "all_s3_control_record_count": 3,
        "campaign_record_count": 42,
        "chain_holdout_record_count": 3,
        "development_record_count": 4,
        "dispersed_record_count": 36,
        "macrocell_record_count": 21,
        "q4_baseline_record_count": 3,
        "thin_regime_record_count": 12,
    }
    assert contract["execution"] == {
        "checker_replica_count": 2,
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_cycle_count": 2,
    }
    assert contract["stage4a_rerun_authorized"] is False
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]


def test_production_boundary_remains_unchanged() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v5b*"))
