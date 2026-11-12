from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v5i_r1_diagnosis_contract.json"
CHECKER = REFERENCE / "e4_pl_s3_v5i_r1_diagnosis_checker.py"
COORDINATOR = REFERENCE / "e4_pl_s3_v5i_r1_diagnosis_coordinator.py"


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


def test_contract_is_canonical_hash_bound_and_nonactivating() -> None:
    raw = CONTRACT.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    for row in value["frozen_inputs"]:
        content = (ROOT / row["path"]).read_bytes()
        assert len(content) == row["bytes"]
        assert hashlib.sha256(content).hexdigest().upper() == row["sha256"]
    assert value["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }


def test_independent_checker_imports_no_production_or_producer() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert "e4_pl_s3_v5i_r1_diagnosis_producer" not in imports


def test_pair_subspace_formula_and_terminal_precedence() -> None:
    checker = _load("_v5i_r1_checker_test", CHECKER)
    coordinator = _load("_v5i_r1_coordinator_test", COORDINATOR)
    cross = [[0.9999, 0.001], [-0.001, 0.9998]]
    assert checker._minimum_singular_squared(cross) > 0.999
    passed = {
        "checker_replicas_byte_identical": True,
        "scientific_payload_sha256": "A" * 64,
        "terminal": coordinator.PASS,
    }
    assert coordinator.adjudicate([passed, dict(passed)]) == coordinator.PASS
    genuine = dict(passed, terminal=coordinator.GENUINE)
    assert coordinator.adjudicate([genuine, dict(genuine)]) == coordinator.GENUINE
    incomplete = dict(passed, terminal=coordinator.INCOMPLETE)
    assert coordinator.adjudicate([incomplete, dict(incomplete)]) == coordinator.INCOMPLETE
    assert coordinator.adjudicate([passed], process_complete=False) == coordinator.BLOCKED


def test_bounds_and_defaults_are_preserved() -> None:
    coordinator = _load("_v5i_r1_coordinator_bounds", COORDINATOR)
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert coordinator.CHILD_TIMEOUT_SECONDS == 600
    assert coordinator.WAVE_TIMEOUT_SECONDS == 1800
    assert coordinator.MEMORY_LIMIT_GIB == 24
    assert coordinator.CYCLES == 2
    assert coordinator.CHECKER_REPLICAS == 2
