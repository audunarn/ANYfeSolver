from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from e4_pl_s3_v4e_preregistration import load_document, validate


def test_contract_binds_closed_v4d_and_frozen_diagnostic() -> None:
    contract = load_document(REFERENCE / "e4_pl_s3_v4e_preregistration_contract.json")
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]
    assert contract["diagnostic"] == {
        "diagonals": ["slash", "backslash"],
        "pl_work": "EXACT_ZERO",
        "thickness_ratios": ["1", "1/10", "1/100", "1/1000"],
        "trace": "LINEAR_ROTATION_W0_THETA_X_X_THETA_Y_Y",
    }


def test_validator_is_standard_library_only() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v4e_preregistration.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert not imports.intersection({"numpy", "sympy", "scipy", "anyfesolver", "anymesh", "anymesher"})


def test_preregistration_authorizes_only_bounded_diagnosis() -> None:
    result = validate()
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V4E_BOUNDED_SHEAR_DIAGNOSIS"
    assert result["next_gate_authorized"] is True
    assert result["stage4a_rerun_authorized"] is result["activation_authorized"] is False


def test_production_boundary_remains_static() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v4e*"))
