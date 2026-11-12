from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/reference_cases/e4_pl_s3_v6v_activation_gap_audit.py"
CHECKER = ROOT / "docs/reference_cases/e4_pl_s3_v6v_activation_gap_checker.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _common(checker):
    return {
        "activation_authorized": False,
        "candidate_formulation_id": "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
        "focused_test_count": 42,
        "gate_status": {name: checker.PASS for name in sorted(checker.GATES)},
        "next_gate": "V6W_FINAL_QUALIFICATION_EVIDENCE_COMPOSITION",
        "package": {
            "import_from_target": True,
            "record_sha256": "A" * 64,
            "round_trip_exact": True,
            "wheel_bytes": 1,
            "wheel_filename": "anysolver-0.3.1-py3-none-any.whl",
            "wheel_sha256": "B" * 64,
        },
        "predecessor_terminals": {},
        "production_boundary": {
            "anymesh_untouched": True,
            "default_q4_formulation": "e4-pl",
            "default_s3_formulation": "legacy-s3",
            "q4_mechanics_unchanged": True,
        },
        "production_restriction": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
        "schema": "anysolver.e4-pl-s3-v6v-activation-gap-common-v1",
        "terminal": checker.GO,
        "test_record_sha256": "C" * 64,
    }


def test_checker_is_deterministic_and_detects_gate_mutation() -> None:
    checker = _load("_v6v_checker_test", CHECKER)
    value = _common(checker)
    raw = checker.canonical_bytes(value)
    assert checker.verify(raw, value)["accepted"] is True
    changed = json.loads(raw)
    changed["gate_status"]["batching"] = "FAIL_BOUND_REGISTERED_SCOPE"
    with pytest.raises(checker.CheckerError):
        checker.verify(checker.canonical_bytes(changed), changed)


def test_checker_accepts_a_consistent_nogo_without_converting_it_to_blocked() -> None:
    checker = _load("_v6v_checker_nogo_test", CHECKER)
    value = _common(checker)
    value["gate_status"]["package_isolation"] = checker.FAIL
    value["next_gate"] = None
    value["terminal"] = checker.NO_GO
    raw = checker.canonical_bytes(value)
    assert checker.verify(raw, value)["terminal"] == checker.NO_GO


def test_checker_imports_no_production_or_runner() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert not any("v6v_activation_gap_audit" in name for name in imports)


def test_scope_bounds_and_test_inventory_are_frozen() -> None:
    runner = _load("_v6v_runner_test", RUNNER)
    assert runner.CHILD_TIMEOUT_SECONDS == 600
    assert runner.MEMORY_LIMIT_GIB == 24
    assert runner.FOCUSED_TEST_COUNT == 42
    assert len(runner.FOCUSED_TESTS) == 7
    assert len({runner.BLOCKED, runner.NO_GO, runner.GO}) == 3


def test_contract_is_canonical_and_binds_inputs() -> None:
    runner = _load("_v6v_contract_test", RUNNER)
    raw, contract = runner.validate_contract()
    assert raw == runner.canonical_bytes(contract)
    paths = [row["path"] for row in contract["frozen_inputs"]]
    assert len(paths) == len(set(paths))
    assert contract["authority_commit"]["exact_path_count"] == 6


def test_strict_loader_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    runner = _load("_v6v_loader_test", RUNNER)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"x":1,"x":2}\n')
    with pytest.raises(runner.V6VError, match="duplicate key"):
        runner.load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"x":Infinity}\n')
    with pytest.raises(runner.V6VError, match="nonfinite token"):
        runner.load(nonfinite)


def test_defaults_and_anymesh_boundary_remain_unchanged() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    runner = RUNNER.read_text(encoding="utf-8")
    assert '"anymesh_untouched": True' in runner
