from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/reference_cases/e4_pl_s3_v6u_performance.py"
CHECKER = ROOT / "docs/reference_cases/e4_pl_s3_v6u_performance_checker.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checker_accepts_only_complete_measurement_proof() -> None:
    checker = _load("_v6u_checker_test", CHECKER)
    proof = {
        "activation_authorized": False,
        "candidate_formulation_id": "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1",
        "gate_status": checker.PASS,
        "payload": {"fraction_percent": 10, "measurement_complete": True},
        "predecessor_terminal": "PROVISIONAL_GO_E4_PL_S3_V6T_PERFORMANCE_SUCCESSOR",
        "production_restriction": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
        "schema": checker.SCHEMA,
        "worker_id": "PERFORMANCE_10",
    }
    proof["scientific_payload_sha256"] = hashlib.sha256(checker.canonical_bytes(proof)).hexdigest().upper()
    assert checker.verify(proof)["accepted"] is True
    changed = dict(proof)
    changed["gate_status"] = "FAIL_MEASURED_CONTRADICTION"
    with pytest.raises(checker.CheckerError):
        checker.verify(changed)


def test_checker_has_no_production_import() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)


def test_scope_and_bounds_are_frozen() -> None:
    runner = _load("_v6u_runner_test", RUNNER)
    assert runner.WORKER_IDS == ("PERFORMANCE_0", "PERFORMANCE_10", "PERFORMANCE_25")
    assert runner.CHILD_TIMEOUT_SECONDS == 600
    assert runner.WAVE_TIMEOUT_SECONDS == 1800
    assert runner.MEMORY_LIMIT_GIB == 24
    assert runner.PRODUCER_CONCURRENCY == 3
    assert runner.CYCLES == 2
    assert len({runner.BLOCKED, runner.NO_GO, runner.GO}) == 3


def test_contract_is_canonical_and_binds_every_frozen_input() -> None:
    runner = _load("_v6u_contract_test", RUNNER)
    raw, contract = runner.validate_contract()
    assert raw == runner.canonical_bytes(contract)
    assert contract["authority_commit"]["exact_path_count"] == 6
    assert contract["terminal_precedence"] == [runner.BLOCKED, runner.NO_GO, runner.GO]
    assert len({row["path"] for row in contract["frozen_inputs"]}) == len(contract["frozen_inputs"])


def test_strict_loader_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    runner = _load("_v6u_loader_test", RUNNER)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"a":1,"a":2}\n')
    with pytest.raises(runner.V6UError, match="duplicate key"):
        runner.load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"a":NaN}\n')
    with pytest.raises(runner.V6UError, match="nonfinite token"):
        runner.load(nonfinite)


def test_defaults_remain_unchanged() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
