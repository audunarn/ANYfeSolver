from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "docs/reference_cases/e4_pl_s3_v6_native_parity_source_authority.py"
PLAN = ROOT / "docs/reference_cases/e4_pl_s3_v6_native_parity_source_plan.json"


def _load():
    spec = importlib.util.spec_from_file_location("_v6_native_parity_source_test", AUTHORITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v6_source_selection_authorizes_v2d_implementation_only() -> None:
    authority = _load()
    result = authority.validate()
    assert result["terminal"] == authority.PASS
    assert result["activation_authorized"] is False
    assert result["complete_activation_execution_authorized"] is False
    assert result["v2d_native_parity_implementation_authorized"] is True
    assert result["next_gate"] == "V6A_V2D_NATIVE_PARITY_IMPLEMENTATION"
    assert result["decision_count"] == 9
    assert result["external_file_count"] == 8
    assert result["local_file_count"] == 10


def test_v6_selection_rejects_missing_decision() -> None:
    authority = _load()
    selection = authority.load(authority.SELECTION)[1]
    selection["decisions"].pop("MATERIAL_STATE_LIFECYCLE")
    try:
        authority.validate(selection)
    except authority.V6SourceAuthorityError as error:
        assert "decision inventory" in str(error)
    else:
        raise AssertionError("incomplete source decision inventory was accepted")


def test_v6_selection_rejects_operator_mutation() -> None:
    authority = _load()
    selection = authority.load(authority.SELECTION)[1]
    selection["local_sources"][0]["sha256"] = "0" * 64
    try:
        authority.validate(selection)
    except authority.V6SourceAuthorityError as error:
        assert "local source mismatch" in str(error)
    else:
        raise AssertionError("mutated accepted V2C source was accepted")


def test_v6_plan_freezes_successor_and_production_boundary() -> None:
    plan = json.loads(PLAN.read_text(encoding="ascii"))
    assert plan["candidate"]["formulation_id"] == "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"
    assert plan["implementation_boundary"]["accepted_v2c_small_strain_operator_unchanged"] is True
    assert plan["implementation_boundary"]["generic_legacy_tri3_mechanics_forbidden"] is True
    assert plan["implementation_boundary"]["q4_mechanics_reuse_forbidden"] is True
    assert plan["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }


def test_v6_authority_imports_only_standard_library_and_not_mechanics() -> None:
    tree = ast.parse(AUTHORITY.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imports)
    assert set(imports) <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "subprocess",
        "typing",
    }


def test_v6_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    authority = _load()
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="ascii")
    try:
        authority.load(path)
    except authority.V6SourceAuthorityError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate JSON key was accepted")
