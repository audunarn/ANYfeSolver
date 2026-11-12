from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from e4_pl_s3_v4b_preregistration import FORMULATION_ID, load_document, restriction, validate


def test_contract_binds_closed_v4a_q4_and_frozen_drill_release() -> None:
    contract = load_document(REFERENCE / "e4_pl_s3_v4b_preregistration_contract.json")
    for item in contract["frozen_inputs"]:
        path = ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == item["sha256"]
    assert contract["candidate"]["formulation_id"] == FORMULATION_ID
    assert contract["construction"]["midpoint_drill_internal_coordinates"] == ["m01_drill", "m12_drill", "m20_drill"]
    assert contract["construction"]["total_internal_coordinate_count"] == 9
    assert contract["execution"] == {
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_cycle_count": 2,
    }


def test_exact_restriction_has_frozen_shape_rank_and_coordinate_roles() -> None:
    made = restriction()
    assert len(made) == 42 and all(len(row) == 27 for row in made)
    assert all(any(row) for row in made)
    result = validate()
    assert result["construction"] == {
        "barycentre_internal_coordinate_count": 6,
        "covered_area": "1/2",
        "external_coordinate_count": 18,
        "midpoint_drill_internal_coordinate_count": 3,
        "physical_midpoint_affine_row_count": 15,
        "positive_subcell_count": 3,
        "restriction_columns": 27,
        "restriction_rank": 27,
        "restriction_rows": 42,
        "rows_nonempty": True,
        "total_internal_coordinate_count": 9,
    }
    for edge in range(3):
        drill_row = 6 * (edge + 3) + 5
        assert made[drill_row][18 + edge] == 1
        assert sum(item != 0 for item in made[drill_row]) == 1


def test_validator_is_standard_library_only_and_does_not_import_mechanics() -> None:
    tree = ast.parse((REFERENCE / "e4_pl_s3_v4b_preregistration.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert not imports.intersection({"numpy", "sympy", "scipy", "anyfesolver", "anymesh", "anymesher"})


def test_canonical_parser_rejects_duplicate_nonfinite_and_hash_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="ascii")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate"):
        load_document(duplicate)
    with pytest.raises(ValueError):
        load_document(nonfinite)
    contract_path = REFERENCE / "e4_pl_s3_v4b_preregistration_contract.json"
    mutated = copy.deepcopy(load_document(contract_path))
    mutated["frozen_inputs"][0]["sha256"] = "0" * 64
    replacement = tmp_path / "contract.json"
    replacement.write_bytes((json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
    import e4_pl_s3_v4b_preregistration as module

    monkeypatch.setattr(module, "CONTRACT", replacement)
    with pytest.raises(ValueError, match="frozen input mismatch"):
        module.validate()


def test_preregistration_authorizes_only_bounded_screen() -> None:
    result = validate()
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V4B_BOUNDED_DRILL_RELEASE_SCREEN"
    assert result["next_gate_authorized"] is True
    assert result["stage4a_rerun_authorized"] is result["activation_authorized"] is False


def test_production_boundary_remains_static() -> None:
    source = (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert not any(path.is_file() for path in (ROOT / "src").rglob("*v4b*"))
