from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "docs/reference_cases/e4_pl_s3_v6w_final_qualification.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PROGRAM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_program_imports_only_the_standard_library() -> None:
    tree = ast.parse(PROGRAM.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    assert set(imports) <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "pathlib",
        "subprocess",
        "typing",
    }


def test_contract_is_canonical_and_all_bound_inputs_match() -> None:
    program = _load("_v6w_contract_test")
    raw, contract = program.validate_contract()
    assert raw == program.canonical_bytes(contract)
    paths = [row["path"] for row in contract["frozen_inputs"]]
    assert len(paths) == len(set(paths))
    assert contract["authority_commit"]["exact_path_count"] == 5


def test_current_evidence_adjudicates_to_explicit_opt_in_qualification() -> None:
    program = _load("_v6w_adjudication_test")
    _raw, contract = program.validate_contract()
    result = program.adjudicate(contract)
    assert result["terminal"] == program.GO
    assert result["qualified_selector"] == "e4-pl-s3-v2d"
    assert result["default_activation_authorized"] is False
    assert set(result["checks"].values()) == {True}


def test_package_gate_mutation_produces_nogo(tmp_path: Path) -> None:
    program = _load("_v6w_mutation_test")
    _raw, contract = program.validate_contract()
    source = ROOT / contract["evidence_paths"]["v6v_package"]
    value = json.loads(source.read_bytes())
    value["checks"]["package_isolation"] = False
    mutated = tmp_path / "mutated.json"
    mutated.write_bytes(program.canonical_bytes(value))
    changed = json.loads(json.dumps(contract))
    changed["evidence_paths"]["v6v_package"] = str(mutated)
    result = program.adjudicate(changed)
    assert result["terminal"] == program.NO_GO
    assert result["qualified_selector"] is None


def test_strict_loader_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    program = _load("_v6w_loader_test")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"x":1,"x":2}\n')
    with pytest.raises(program.V6WError, match="duplicate key"):
        program.load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"x":NaN}\n')
    with pytest.raises(program.V6WError, match="nonfinite token"):
        program.load(nonfinite)


def test_defaults_remain_unactivated() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
