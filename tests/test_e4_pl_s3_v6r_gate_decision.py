from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v6r_gate_decision_contract.json"
CHECKER = REFERENCE / "e4_pl_s3_v6r_gate_decision_checker.py"
COORDINATOR = REFERENCE / "e4_pl_s3_v6r_gate_decision_coordinator.py"


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_contract_is_canonical_hash_bound_and_evidence_only() -> None:
    contract = json.loads(CONTRACT.read_bytes())
    assert CONTRACT.read_bytes() == canonical(contract)
    assert contract["evidence_policy"] == {
        "exact_metadata_identity_required": True,
        "independent_and_producer_decisions_must_agree": True,
        "mechanics_execution_forbidden": True,
        "raw_metric_identity_disposition": "NONCLASSIFYING_DERIVED_BINARY_REPRODUCTION_DIAGNOSTIC",
        "raw_metric_identity_threshold_changed": False,
        "scientific_thresholds_changed": False,
    }
    assert contract["execution"]["child_wall_seconds"] == 600
    assert contract["execution"]["cycle_wall_seconds"] == 1800
    assert contract["execution"]["memory_limit_gib_per_process_tree"] == 24
    assert contract["execution"]["automatic_retry"] is False
    for name, path in (("checker", CHECKER), ("coordinator", COORDINATOR)):
        assert contract["programs"][name] == {"bytes": path.stat().st_size, "sha256": digest(path)}
    for item in contract["frozen_inputs"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"]
    for binding in contract["proofs"].values():
        path = Path(binding["path"])
        assert path.stat().st_size == binding["bytes"]
        assert digest(path) == binding["sha256"]


def test_authority_extent_and_production_boundary() -> None:
    contract = json.loads(CONTRACT.read_bytes())
    authority = contract["authority_commit"]
    assert authority["expected_parent"] == "94c4b7b82c7b11427512d183c1da12152aba0015"
    assert authority["exact_path_count"] == len(authority["expected_paths"]) == 6
    assert contract["production_boundary"]["default_q4_formulation"] == "e4-pl"
    assert contract["production_boundary"]["default_s3_formulation"] == "legacy-s3"
    assert contract["production_boundary"]["q4_mechanics_unchanged"] is True


def test_programs_are_syntax_valid_and_checker_has_no_production_import() -> None:
    for path in (CHECKER, COORDINATOR):
        ast.parse(path.read_text(encoding="utf-8"))
    text = CHECKER.read_text(encoding="utf-8")
    assert "from anyfesolver" not in text
    assert "import anyfesolver" not in text
    assert "raw_metric_identity_disposition" in text
    assert "scientific_gate_passed" in text


def test_terminal_precedence() -> None:
    coordinator = load(COORDINATOR, "_test_v6r_coordinator")
    clean = {"cycle": 1, "formal_failure_count": 0, "record_count": 18}
    fail = {"cycle": 1, "formal_failure_count": 1, "record_count": 18}
    assert coordinator.adjudicate([], complete=False) == coordinator.BLOCKED
    assert coordinator.adjudicate([clean, clean | {"cycle": 2, "record_count": 17}]) == coordinator.BLOCKED
    assert coordinator.adjudicate([fail, fail | {"cycle": 2}]) == coordinator.NO_GO
    assert coordinator.adjudicate([clean, clean | {"cycle": 2}]) == coordinator.PASS


def test_authority_validates_without_execution_authorization() -> None:
    coordinator = load(COORDINATOR, "_test_v6r_authority")
    contract, raw = coordinator.validate_authority(execution=False)
    assert contract["activation_authorized"] is False
    assert coordinator.sha256(raw) == digest(CONTRACT)
