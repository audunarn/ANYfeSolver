from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/reference_cases/e4_pl_s3_v5m_parity_plan.json"
CONTRACT = ROOT / "docs/reference_cases/e4_pl_s3_v5m_parity_contract.json"
REVIEWER = ROOT / "docs/reference_cases/e4_pl_s3_v5m_protocol_review.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v5m_protocol_is_complete_bounded_and_additive() -> None:
    plan = json.loads(PLAN.read_text(encoding="ascii"))
    assert plan["bounds"] == {
        "checker_concurrency": 4,
        "checker_replicas_per_worker": 2,
        "child_timeout_seconds": 600,
        "cycles": 2,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "worker_concurrency": 3,
        "worker_ids": ["BATCH_4096", "SERIALIZATION_RESTART", "PACKAGE_WHEEL"],
        "wave_timeout_seconds": 1800,
    }
    assert plan["acceptance"]["batch"]["element_count"] == 4096
    assert plan["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }


def test_v5m_contract_has_exact_protocol_and_implementation_extents() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    assert contract["authority"]["exact_paths"] == [
        "docs/reference_cases/e4_pl_s3_v5m_parity_contract.json",
        "docs/reference_cases/e4_pl_s3_v5m_parity_plan.json",
        "docs/reference_cases/e4_pl_s3_v5m_protocol_review.py",
        "tests/test_e4_pl_s3_v5m_protocol.py",
    ]
    assert contract["implementation_extent"] == [
        "docs/reference_cases/e4_pl_s3_v5m_checker.py",
        "docs/reference_cases/e4_pl_s3_v5m_parity.py",
        "tests/test_e4_pl_s3_v5m_parity.py",
    ]


def test_v5m_protocol_reviewer_is_independent_and_empty() -> None:
    tree = ast.parse(REVIEWER.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    reviewer = _load("_v5m_protocol_reviewer_test", REVIEWER)
    assert reviewer.review()["findings"] == []
