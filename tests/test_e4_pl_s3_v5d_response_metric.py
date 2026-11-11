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
REFERENCE = ROOT / "docs" / "reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v5d_response_metric_contract.json"


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
    return _load("_v5d_response_producer", REFERENCE / "e4_pl_s3_v5d_response_metric_producer.py")


@pytest.fixture(scope="module")
def checker():
    return _load("_v5d_response_checker", REFERENCE / "e4_pl_s3_v5d_response_metric_checker.py")


@pytest.fixture(scope="module")
def coordinator():
    return _load("_v5d_response_coordinator", REFERENCE / "e4_pl_s3_v5d_response_metric_coordinator.py")


def test_checker_does_not_import_diagnostic_producer():
    tree = ast.parse((REFERENCE / "e4_pl_s3_v5d_response_metric_checker.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "e4_pl_s3_v5d_response_metric_producer" not in imports


def test_contract_is_canonical_and_binds_exact_authority_extent():
    raw = CONTRACT.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    assert value["authority_commit"] == {
        "expected_parent": "0cca12330d8a675810c177d314daca6f77988a5d",
        "expected_path_set": [
            "docs/agent_plans/S3_E4_PL_V5D_RESPONSE_METRIC_DIAGNOSIS_PLAN.md",
            "docs/reference_cases/e4_pl_s3_v5d_response_metric_checker.py",
            "docs/reference_cases/e4_pl_s3_v5d_response_metric_contract.json",
            "docs/reference_cases/e4_pl_s3_v5d_response_metric_coordinator.py",
            "docs/reference_cases/e4_pl_s3_v5d_response_metric_producer.py",
            "tests/test_e4_pl_s3_v5d_response_metric.py",
        ],
        "expected_subject": "docs: authorize S3 V5D response-metric diagnosis",
    }
    assert value["diagnosis_execution_authorized"] is True
    assert value["activation_authorized"] is value["stage4a_reclassified"] is False


def test_contract_frozen_hash_dag():
    value = json.loads(CONTRACT.read_text(encoding="ascii"))
    for binding in value["frozen_inputs"]:
        raw = (ROOT / binding["path"]).read_bytes()
        assert len(raw) == binding["bytes"]
        assert hashlib.sha256(raw).hexdigest().upper() == binding["sha256"]


def test_exact_45_record_and_12_sequence_catalog(producer):
    catalog = [spec for diagonal in producer.DIAGONALS for spec in producer._specs(diagonal)]
    sequences = {(diagonal, mask, fraction) for diagonal in producer.DIAGONALS for mask in producer.MASKS for fraction in producer.FRACTIONS}
    assert len(catalog) == len(set(catalog)) == 45
    assert all(len(producer._specs(diagonal)) == 15 for diagonal in producer.DIAGONALS)
    assert len(sequences) == 12


def test_independent_spatial_record_identity(producer, checker):
    made = producer._state_record(20, 25, "chain", "slash")
    independent = checker._state_record(20, 25, "chain", "slash")
    assert checker._identity(made, independent) == 0.0
    assert float.fromhex(made["solve_residual_relative_inf_hex"]) <= 1.0e-8
    assert float.fromhex(made["w_relative_l2_error_hex"]) > 0.0
    assert float.fromhex(made["energy_relative_error_hex"]) > 0.0


def test_spatial_record_mutations_are_detected(producer, checker):
    made = producer._state_record(20, 25, "chain", "slash")
    changed = dict(made)
    changed["w_relative_l2_error_hex"] = (float.fromhex(made["w_relative_l2_error_hex"]) + 1.0e-5).hex()
    assert checker._identity(changed, made) > 3.0e-12
    changed = dict(made)
    changed["connectivity_sha256"] = "0" * 64
    with pytest.raises(checker.DiagnosisCheckerError):
        checker._identity(changed, made)


def _cycle(coordinator, *, center=None, spatial=None, energy=None, controls=None, digest="1" * 64):
    return {
        "center_failure_sequences": list(coordinator.EXPECTED_CENTRE_FAILURES if center is None else center),
        "energy_failure_sequences": list(energy or []),
        "expected_v5c_center_failures_reproduced": center is None,
        "record_count": 45,
        "sequence_count": 12,
        "sequence_results_sha256": digest,
        "shards": [],
        "spatial_failure_sequences": list(spatial or []),
        "ten_percent_control_failures": list(controls or []),
    }


def test_terminal_precedence_and_deterministic_identity(coordinator):
    first = _cycle(coordinator)
    second = copy.deepcopy(first)
    assert coordinator.adjudicate([first, second]) == coordinator.PROTOCOL
    failed = _cycle(coordinator, spatial=["slash:chain:25"])
    assert coordinator.adjudicate([failed, copy.deepcopy(failed)]) == coordinator.NO_GO
    inconclusive = _cycle(coordinator, center=[], controls=["slash:chain:10"])
    assert coordinator.adjudicate([inconclusive, copy.deepcopy(inconclusive)]) == coordinator.INCONCLUSIVE
    mismatched = copy.deepcopy(second)
    mismatched["sequence_results_sha256"] = "2" * 64
    assert coordinator.adjudicate([first, mismatched]) == coordinator.BLOCKED
    assert coordinator.adjudicate([first], process_complete=False) == coordinator.BLOCKED


def test_production_boundary_is_unchanged():
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    assert contract["production_boundary"] == {
        "activation_authorized": False,
        "anymesh_untouched": True,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
