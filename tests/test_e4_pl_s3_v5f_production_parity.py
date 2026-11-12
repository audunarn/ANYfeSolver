from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v5f_production_parity_contract.json"


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
def producer():
    return _load("_v5f_producer", REFERENCE / "e4_pl_s3_v5f_production_parity_producer.py")


@pytest.fixture(scope="module")
def checker():
    return _load("_v5f_checker", REFERENCE / "e4_pl_s3_v5f_production_parity_checker.py")


@pytest.fixture(scope="module")
def coordinator():
    return _load("_v5f_coordinator", REFERENCE / "e4_pl_s3_v5f_production_parity_coordinator.py")


def test_checker_is_independent_of_producer_and_production_mechanics() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v5f_production_parity_checker.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "e4_pl_s3_v5f_production_parity_producer" not in imports
    assert "anyfesolver" not in imports


def test_contract_is_canonical_and_binds_exact_authority_extent() -> None:
    raw = CONTRACT.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    assert value["authority_commit"]["expected_parent"] == "dcd9db75e992e3aa20382873c070b5676925af09"
    assert value["authority_commit"]["expected_subject"] == "feat: authorize S3 V5F production parity candidate"
    assert value["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }


def test_contract_hash_dag() -> None:
    value = json.loads(CONTRACT.read_text(encoding="ascii"))
    for binding in value["frozen_inputs"]:
        raw = (ROOT / binding["path"]).read_bytes()
        assert len(raw) == binding["bytes"]
        assert hashlib.sha256(raw).hexdigest().upper() == binding["sha256"]


def test_complete_catalog_and_independent_parity(producer, checker) -> None:
    proof = producer.produce_proof()
    assert len(proof["catalog"]) == 81
    assert len(proof["catalog_cases"]) == 162
    assert len(proof["cases"]) == 169
    checked = checker.verify_proof(proof)
    assert checked["passed"] is True
    assert checked["rank_failure_case_ids"] == []


def test_matrix_and_hash_mutations_are_detected(producer, checker) -> None:
    proof = producer.produce_proof()
    changed = copy.deepcopy(proof)
    changed["cases"][0]["components"]["total"]["hex"][0] = (float.fromhex(changed["cases"][0]["components"]["total"]["hex"][0]) + 1.0).hex()
    payload = dict(changed)
    payload.pop("scientific_payload_sha256")
    changed["scientific_payload_sha256"] = checker.sha256_bytes(checker.canonical_bytes(payload))
    assert checker.verify_proof(changed)["passed"] is False
    changed = copy.deepcopy(proof)
    changed["scientific_payload_sha256"] = "0" * 64
    with pytest.raises(checker.ProductionParityCheckError, match="payload hash"):
        checker.verify_proof(changed)


def _cycle(*, passed=True, digest="1" * 64):
    return {
        "checker_replicas_byte_identical": True,
        "passed": passed,
        "scientific_payload_sha256": digest,
    }


def test_terminal_precedence_and_bounds(coordinator) -> None:
    first = _cycle()
    assert coordinator.adjudicate([first, copy.deepcopy(first)]) == coordinator.PASS
    assert coordinator.adjudicate([_cycle(passed=False), _cycle(passed=False)]) == coordinator.NO_GO
    assert coordinator.adjudicate([first], process_complete=False) == coordinator.BLOCKED
    assert coordinator.adjudicate([first, _cycle(digest="2" * 64)]) == coordinator.BLOCKED
    assert coordinator.CHILD_TIMEOUT_SECONDS == 600
    assert coordinator.WAVE_TIMEOUT_SECONDS == 1800
    assert coordinator.MEMORY_LIMIT_GIB == 24
    assert coordinator.CHECKER_REPLICAS == 2
    assert coordinator.CYCLES == 2


def test_defaults_and_q4_mechanics_remain_unchanged() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    q4 = next(item for item in contract["frozen_inputs"] if item["role"] == "UNCHANGED_QUALIFIED_Q4_MECHANICS")
    raw = (ROOT / q4["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest().upper() == q4["sha256"]
