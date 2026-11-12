from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v5h_local_parity_contract.json"
PRODUCER = REFERENCE / "e4_pl_s3_v5h_local_parity_producer.py"
CHECKER = REFERENCE / "e4_pl_s3_v5h_local_parity_checker.py"
COORDINATOR = REFERENCE / "e4_pl_s3_v5h_local_parity_coordinator.py"


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
    return _load("_v5h_producer", PRODUCER)


@pytest.fixture(scope="module")
def checker():
    return _load("_v5h_checker", CHECKER)


@pytest.fixture(scope="module")
def coordinator():
    return _load("_v5h_coordinator", COORDINATOR)


@pytest.fixture(scope="module")
def proof(producer):
    return producer.produce_proof()


def test_contract_is_canonical_complete_and_exact_extent() -> None:
    raw = CONTRACT.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    for binding in value["frozen_inputs"]:
        content = (ROOT / binding["path"]).read_bytes()
        assert len(content) == binding["bytes"]
        assert hashlib.sha256(content).hexdigest().upper() == binding["sha256"]
    result_path = REFERENCE / "e4_pl_s3_v5h_local_parity_result.json"
    if result_path.is_file():
        authority_commit = json.loads(result_path.read_text(encoding="ascii"))[
            "authority"
        ]["commit"]
        changed = subprocess.check_output(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                authority_commit,
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
        assert changed == value["authority"]["exact_paths"]
    else:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", value["authority"]["expected_parent"]],
            cwd=ROOT,
            text=True,
        ).splitlines()
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        assert sorted(set(changed) | set(untracked)) == value["authority"]["exact_paths"]
    assert value["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
        "stage4b_execution_authorized": False,
    }


def test_independent_checker_has_no_production_or_producer_import() -> None:
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imported)
    assert "e4_pl_s3_v5h_local_parity_producer" not in imported


def test_complete_local_parity_proof_passes(proof, checker) -> None:
    checked = checker.verify_proof(proof)
    assert proof["case_count"] == 14
    assert checked["passed"] is True
    assert checked["failure_case_ids"] == []
    assert max(
        float.fromhex(checked[name])
        for name in (
            "buckling_worst_hex",
            "component_worst_hex",
            "geometric_worst_hex",
            "mass_worst_hex",
            "modal_worst_hex",
            "pressure_worst_hex",
            "work_worst_hex",
        )
    ) <= checker.RELATIVE_LIMIT


def _rehash(value: dict, producer) -> None:
    value.pop("scientific_payload_sha256", None)
    value["scientific_payload_sha256"] = producer.sha256_bytes(
        producer.canonical_bytes(value)
    )


@pytest.mark.parametrize(
    ("surface", "mutate"),
    (
        ("mass", lambda case: case["mass"]["hex"].__setitem__(0, float(1.0).hex())),
        (
            "geometric",
            lambda case: case["geometric_stiffness"]["hex"].__setitem__(0, float(1.0).hex()),
        ),
        (
            "modal",
            lambda case: case["modal_eigenvalues"]["hex"].__setitem__(0, float(1.0).hex()),
        ),
        (
            "serialization",
            lambda case: case["serialized"].__setitem__("mass_policy_id", "BAD"),
        ),
    ),
)
def test_scientific_mutations_are_detected(
    proof,
    producer,
    checker,
    surface,
    mutate,
) -> None:
    changed = copy.deepcopy(proof)
    mutate(changed["cases"][0])
    _rehash(changed, producer)
    assert checker.verify_proof(changed)["passed"] is False, surface


def test_payload_hash_and_terminal_precedence(proof, checker, coordinator) -> None:
    changed = copy.deepcopy(proof)
    changed["cases"][0]["v2b_static_byte_identical"] = False
    with pytest.raises(checker.LocalParityCheckError, match="payload hash"):
        checker.verify_proof(changed)
    passed = {
        "checker_replicas_byte_identical": True,
        "passed": True,
        "scientific_payload_sha256": "A" * 64,
    }
    assert coordinator.adjudicate([passed, dict(passed)]) == coordinator.PASS
    no_go = dict(passed, passed=False)
    assert coordinator.adjudicate([passed, no_go]) == coordinator.NO_GO
    disagree = dict(passed, scientific_payload_sha256="B" * 64)
    assert coordinator.adjudicate([passed, disagree]) == coordinator.BLOCKED
    assert coordinator.adjudicate([passed], process_complete=False) == coordinator.BLOCKED


def test_bounds_and_defaults_are_preserved(coordinator) -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert coordinator.CHILD_TIMEOUT_SECONDS == 600
    assert coordinator.WAVE_TIMEOUT_SECONDS == 1800
    assert coordinator.MEMORY_LIMIT_GIB == 24
    assert coordinator.CHECKER_REPLICAS == 2
