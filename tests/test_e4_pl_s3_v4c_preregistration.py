from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from e4_pl_s3_v4c_preregistration import external_embedding, load_document, physical_restriction, validate


def test_contract_binds_v4b_q4_and_native_pl_authority() -> None:
    contract = load_document(REFERENCE / "e4_pl_s3_v4c_preregistration_contract.json")
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]
    assert contract["physical_construction"]["q4_drilling_rows_and_columns"] == "EXCLUDED"
    assert contract["numerical_completion"]["source"] == "ACCEPTED_BARYCENTRIC_S3_PL_ONLY"
    assert contract["execution"]["child_wall_seconds"] == 600
    assert contract["execution"]["complete_wave_wall_seconds"] == 1800


def test_exact_physical_restriction_and_embedding() -> None:
    restriction = physical_restriction()
    embedding = external_embedding()
    assert len(restriction) == 42 and all(len(row) == 20 for row in restriction)
    assert len(embedding) == 18 and all(len(row) == 15 for row in embedding)
    assert all(not any(restriction[6 * point + 5]) for point in range(7))
    result = validate()
    assert result["construction"] == {
        "external_embedding_columns": 15,
        "external_embedding_rank": 15,
        "external_embedding_rows": 18,
        "internal_physical_coordinate_count": 5,
        "physical_restriction_columns": 20,
        "physical_restriction_rank": 20,
        "physical_restriction_rows": 42,
        "zero_q4_drill_row_count": 7,
        "zero_q4_drill_rows": True,
    }


def test_validator_is_standard_library_only() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v4c_preregistration.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert not imports.intersection({"numpy", "sympy", "scipy", "anyfesolver", "anymesh", "anymesher"})


def test_preregistration_authorizes_only_bounded_screen() -> None:
    result = validate()
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V4C_BOUNDED_PHYSICAL_FIRST_SCREEN"
    assert result["next_gate_authorized"] is True
    assert result["stage4a_rerun_authorized"] is result["activation_authorized"] is False


def test_production_boundary_remains_static() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v4c*"))
