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
CONTRACT = REFERENCE / "e4_pl_s3_v5e_stage4a_spatial_contract.json"


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
    return _load("_v5e_spatial_producer", REFERENCE / "e4_pl_s3_v5e_stage4a_spatial_producer.py")


@pytest.fixture(scope="module")
def checker():
    return _load("_v5e_spatial_checker", REFERENCE / "e4_pl_s3_v5e_stage4a_spatial_checker.py")


@pytest.fixture(scope="module")
def coordinator():
    return _load("_v5e_spatial_coordinator", REFERENCE / "e4_pl_s3_v5e_stage4a_spatial_coordinator.py")


def test_checker_does_not_import_v5e_or_v5d_producer():
    tree = ast.parse((REFERENCE / "e4_pl_s3_v5e_stage4a_spatial_checker.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "e4_pl_s3_v5e_stage4a_spatial_producer" not in imports
    assert "e4_pl_s3_v5d_response_metric_producer" not in imports


def test_contract_is_canonical_and_binds_exact_authority_extent():
    raw = CONTRACT.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    assert value["authority_commit"] == {
        "expected_parent": "e16a1df01006e96307c7acb1d4f68ffe295cdca2",
        "expected_path_set": [
            "docs/agent_plans/S3_E4_PL_V5E_STAGE4A_SPATIAL_RESPONSE_PLAN.md",
            "docs/reference_cases/e4_pl_s3_v5e_stage4a_spatial_checker.py",
            "docs/reference_cases/e4_pl_s3_v5e_stage4a_spatial_contract.json",
            "docs/reference_cases/e4_pl_s3_v5e_stage4a_spatial_coordinator.py",
            "docs/reference_cases/e4_pl_s3_v5e_stage4a_spatial_producer.py",
            "tests/test_e4_pl_s3_v5e_stage4a_spatial.py",
        ],
        "expected_subject": "docs: reauthorize S3 V5E spatial Stage 4A execution",
    }
    assert value["stage4a_execution_authorized"] is True
    assert value["activation_authorized"] is value["v5c_reclassified"] is False


def test_contract_frozen_hash_dag():
    value = json.loads(CONTRACT.read_text(encoding="ascii"))
    for binding in value["frozen_inputs"]:
        raw = (ROOT / binding["path"]).read_bytes()
        assert len(raw) == binding["bytes"]
        assert hashlib.sha256(raw).hexdigest().upper() == binding["sha256"]


def test_exact_81_record_and_24_sequence_catalog(producer):
    catalog = [spec for diagonal in producer.DIAGONALS for spec in producer._specs(diagonal)]
    assert len(catalog) == len(set(catalog)) == 81
    assert all(len(producer._specs(diagonal)) == 27 for diagonal in producer.DIAGONALS)
    assert len(producer.DIAGONALS) * len(producer.MASKS) * len(producer.FRACTIONS) == 24


def test_independent_one_percent_spatial_record_and_catalog_restoration(producer, checker):
    producer_catalog = producer.v5d.FRACTIONS
    checker_catalog = checker.v5d_check.FRACTIONS
    made = producer._record(20, 1, "dispersed", "slash")
    independent = checker._record(20, 1, "dispersed", "slash")
    assert checker.v5d_check._identity(made, independent) == 0.0
    assert producer.v5d.FRACTIONS == producer_catalog == (10, 25)
    assert checker.v5d_check.FRACTIONS == checker_catalog == (10, 25)


def test_record_mutations_are_detected(producer, checker):
    made = producer._record(20, 1, "dispersed", "slash")
    changed = dict(made)
    changed["w_relative_l2_error_hex"] = (float.fromhex(made["w_relative_l2_error_hex"]) + 1.0e-5).hex()
    assert checker.v5d_check._identity(changed, made) > 3.0e-12
    changed = dict(made)
    changed["reference_sha256"] = "0" * 64
    with pytest.raises(checker.v5d_check.DiagnosisCheckerError):
        checker.v5d_check._identity(changed, made)


def test_historical_center_metric_is_nonclassifying(checker):
    passing = {
        "diagonal": "slash",
        "energy_slope_lower_95_hex": (1.0).hex(),
        "fraction_percent": 25,
        "mask": "chain",
        "spatial_finest_ratio_hex": (1.0).hex(),
        "spatial_slope_deficit_hex": (0.0).hex(),
        "spatial_slope_hex": (2.0).hex(),
        "spatial_successive_passed": True,
    }
    assert checker._formal_failures(passing) == []
    failing = dict(passing)
    failing["spatial_slope_hex"] = (1.7).hex()
    assert checker._formal_failures(failing) == ["slash:chain:25:SPATIAL_RESPONSE_SLOPE"]


def _cycle(failures=None, digest="1" * 64):
    failures = list(failures or [])
    return {
        "center_diagnostic_failure_sequences": ["slash:chain:25"],
        "formal_failure_count": len(failures),
        "formal_failures": failures,
        "record_count": 81,
        "response_metric": "NODAL_UZ_RELATIVE_L2",
        "sequence_count": 24,
        "sequence_results_sha256": digest,
        "shards": [],
    }


def test_terminal_precedence_and_cycle_identity(coordinator):
    first = _cycle()
    assert coordinator.adjudicate([first, copy.deepcopy(first)]) == coordinator.PASS
    failed = _cycle(["slash:chain:25:SPATIAL_RESPONSE_SLOPE"])
    assert coordinator.adjudicate([failed, copy.deepcopy(failed)]) == coordinator.NO_GO
    mismatched = copy.deepcopy(first)
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
