from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
SELECTION = REFERENCE / "e4_pl_s3_v5g_stage4b_extension_source_selection.json"
PROGRAM = REFERENCE / "e4_pl_s3_v5g_stage4b_extension_authority.py"


def _load():
    spec = importlib.util.spec_from_file_location("_v5g_authority", PROGRAM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_selection_is_canonical_and_source_complete() -> None:
    raw = SELECTION.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    assert [row["role"] for row in value["external_source"]["files"]] == [
        "TRIA3_MASS_POLICY",
        "TRIA3_GEOMETRIC_STIFFNESS_POLICY",
        "MIN3_STIFFNESS_AND_RECOVERY",
        "RELAXATION_IDENTITY",
        "INDEPENDENT_RUNTIME_DISCLOSURE",
    ]
    assert value["scope"]["v2c_implementation_authorized"] is True
    assert value["scope"]["stage4b_execution_authorized"] is False


def test_external_graph_and_v5f_predecessor_validate() -> None:
    authority = _load()
    result = authority.validate()
    assert result["terminal"] == authority.PASS
    assert result["external_file_count"] == 5
    assert result["v2c_implementation_authorized"] is True
    assert result["stage4b_execution_authorized"] is False


def test_source_hash_scope_and_terminal_mutations_fail() -> None:
    authority = _load()
    value = json.loads(SELECTION.read_text(encoding="ascii"))
    changed = copy.deepcopy(value)
    changed["external_source"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(authority.ExtensionAuthorityError, match="binding mismatch"):
        authority.validate(changed)
    changed = copy.deepcopy(value)
    changed["predecessor"]["result_sha256"] = "0" * 64
    with pytest.raises(authority.ExtensionAuthorityError, match="predecessor mismatch"):
        authority.validate(changed)
    changed = copy.deepcopy(value)
    changed["scope"]["stage4b_execution_authorized"] = True
    with pytest.raises(authority.ExtensionAuthorityError, match="scope or terminal"):
        authority.validate(changed)


def test_q4_and_defaults_remain_unchanged() -> None:
    value = json.loads(SELECTION.read_text(encoding="ascii"))
    assert value["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
