from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/reference_cases/e4_pl_s3_v5n_activation_audit.py"
PLAN = ROOT / "docs/reference_cases/e4_pl_s3_v5n_activation_audit_plan.json"


def _load():
    spec = importlib.util.spec_from_file_location("_v5n_activation_audit_test", AUDIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v5n_synthesis_identifies_native_parity_gaps_without_activation() -> None:
    audit = _load()
    result = audit.synthesize()
    assert result["terminal"] == audit.UNCLASSIFIED
    assert result["activation_authorized"] is False
    assert result["next_gate"] == "V6_NATIVE_PARITY_SOURCE_SELECTION"
    assert result["required_gate_count"] == 20
    assert set(result["missing_gate_ids"]) >= {
        "FULL_252_MIXED_TOPOLOGY_N20_N40_N80_N160",
        "SAME_FORMULATION_RESTART",
        "NONLINEAR_GEOMETRY",
        "MATERIAL_NONLINEARITY",
        "GENERALIZED_SECTIONS",
        "MIGRATION",
        "CROSS_WHEEL_ECOSYSTEM",
    }
    assert result["gate_status"]["MODAL"] == audit.PASS
    assert result["gate_status"]["BUCKLING"] == audit.PASS
    assert result["gate_status"]["BATCH"] == audit.PASS


def test_v5n_auditor_imports_only_standard_library_and_not_mechanics() -> None:
    tree = ast.parse(AUDIT.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert set(imports) <= {"__future__", "argparse", "ast", "hashlib", "json", "pathlib", "typing"}


def test_v5n_plan_freezes_complete_gate_inventory_and_precedence() -> None:
    plan = json.loads(PLAN.read_text(encoding="ascii"))
    assert len(plan["required_activation_gates"]) == 20
    assert len(set(plan["required_activation_gates"])) == 20
    assert plan["terminal_precedence"] == [
        "BLOCKED_E4_PL_S3_V5N_EVIDENCE_OR_REVIEW",
        "NO_GO_E4_PL_S3_V5N_ACTIVATION_QUALIFICATION",
        "UNCLASSIFIED_E4_PL_S3_V5N_NATIVE_PARITY_SOURCE_REQUIRED",
        "PROVISIONAL_GO_E4_PL_S3_V5N_ACTIVATION_EXECUTION_PREPARATION",
    ]


def test_v5n_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    audit = _load()
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="ascii")
    try:
        audit.load(path)
    except audit.V5NActivationAuditError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate key was accepted")
