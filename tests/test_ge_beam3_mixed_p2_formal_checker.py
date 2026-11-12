"""Independent recomputation and mutation tests for the P2 checker."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
PRODUCER_PATH = REFERENCE / "ge_beam3_mixed_p2_formal_producer.py"
CHECKER_PATH = REFERENCE / "ge_beam3_mixed_p2_formal_checker.py"


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    made = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(made)
    return made


producer = _module(PRODUCER_PATH, "ge_beam3_mixed_p2_formal_producer_checker_test")
checker = _module(CHECKER_PATH, "ge_beam3_mixed_p2_formal_checker_test")


@pytest.fixture(scope="module")
def proof() -> dict[str, Any]:
    return producer.build_proof()


def _seal(value: dict[str, Any]) -> bytes:
    made = copy.deepcopy(value)
    made.pop("content_sha256", None)
    made["content_sha256"] = checker._sha(checker.canonical_bytes(made))
    return checker.canonical_bytes(made)


def _mutate_array(value: dict[str, Any], index: int = 0) -> None:
    raw = bytearray.fromhex(value["data_hex"])
    position = 8 * index
    number = __import__("struct").unpack("<d", raw[position : position + 8])[0]
    raw[position : position + 8] = __import__("struct").pack("<d", number + max(1.0, abs(number)))
    value["data_hex"] = bytes(raw).hex().upper()


def test_checker_accepts_complete_proof_and_is_deterministic(proof: dict[str, Any]) -> None:
    raw = producer.canonical_bytes(proof)
    first = checker.verify_payload(raw)
    second = checker.verify_payload(raw)
    assert checker.canonical_bytes(first) == checker.canonical_bytes(second)
    assert first["terminal"] == checker.PASS
    assert first["finding_groups"] == []
    assert first["findings"] == []
    assert first["group_pass"] == {
        "SOLVER_CHART_OR_STATE": True,
        "LOAD_MASS_OR_RECOVERY": True,
        "MODAL_OR_BUCKLING": True,
    }
    assert first["predicates"] == {
        "evidence_valid": True,
        "independent_recomputation_complete": True,
    }


@pytest.mark.parametrize(
    ("record_group", "case_id", "field", "expected_group"),
    [
        ("solver_chart_state", "NONZERO_SOLVER_CHART", "analytic_directional_force", "SOLVER_CHART_OR_STATE"),
        ("load_mass_recovery", "UNIFORM_LINE_SPATIAL_DEAD", "element_vector", "LOAD_MASS_OR_RECOVERY"),
        ("load_mass_recovery", "REFERENCE_MASS_DIAGONAL_UNIT", "mass_matrix", "LOAD_MASS_OR_RECOVERY"),
    ],
)
def test_numeric_mutations_are_scientific_findings(
    proof: dict[str, Any], record_group: str, case_id: str, field: str, expected_group: str
) -> None:
    changed = copy.deepcopy(proof)
    record = next(item for item in changed["records"][record_group] if item["case_id"] == case_id)
    _mutate_array(record[field])
    result = checker.verify_payload(_seal(changed))
    assert result["terminal"] == checker.FINDING
    assert expected_group in result["finding_groups"]
    assert result["predicates"]["evidence_valid"] is True


def test_modal_and_all_group_mutations_follow_terminal_order(proof: dict[str, Any]) -> None:
    changed = copy.deepcopy(proof)
    solver = next(item for item in changed["records"]["solver_chart_state"] if item["case_id"] == "NONZERO_SOLVER_CHART")
    _mutate_array(solver["analytic_directional_force"])
    load = next(item for item in changed["records"]["load_mass_recovery"] if item["case_id"] == "UNIFORM_LINE_SPATIAL_DEAD")
    _mutate_array(load["element_vector"])
    modal = next(item for item in changed["records"]["modal_buckling"] if item["case_id"] == "CANTILEVER_REFERENCE_LINEAR")
    modal["samples"][-1]["family_frequencies_hz"]["AXIAL"] = float(1.0e30).hex()
    result = checker.verify_payload(_seal(changed))
    assert result["terminal"] == checker.FINDING
    assert result["finding_groups"] == list(checker.GROUP_ORDER)
    assert [checker.GROUP_ORDER.index(item["group"]) for item in result["findings"]] == sorted(
        checker.GROUP_ORDER.index(item["group"]) for item in result["findings"]
    )


@pytest.mark.parametrize("mutation", ["HASH", "BINDING", "OVERLAY", "COUNT", "TYPED_SHAPE"])
def test_identity_and_encoding_mutations_are_malformed(proof: dict[str, Any], mutation: str) -> None:
    changed = copy.deepcopy(proof)
    if mutation == "HASH":
        changed["content_sha256"] = "0" * 64
        raw = producer.canonical_bytes(changed)
    elif mutation == "BINDING":
        changed["bindings"]["ge_beam3_mixed_p2_cases.json"]["sha256"] = "0" * 64
        raw = _seal(changed)
    elif mutation == "OVERLAY":
        changed["effective_recovery_case_ids"].append("RECOVERY_UNIFORM_SHEAR_Y_NATIVE_STATE")
        changed["effective_recovery_case_ids"].sort()
        raw = _seal(changed)
    elif mutation == "COUNT":
        changed["counts"]["total"] = 22
        raw = _seal(changed)
    else:
        item = changed["records"]["solver_chart_state"][0]["base_force"]
        item["shape"] = [17]
        raw = _seal(changed)
    with pytest.raises(checker.MalformedProof):
        checker.verify_payload(raw)


def test_strict_decoder_rejects_duplicates_and_nonfinite() -> None:
    for raw in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n'):
        with pytest.raises(checker.MalformedProof):
            checker._decode(raw)


def test_checker_import_boundary_excludes_production_and_legacy_mechanics() -> None:
    tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "numpy",
        "pathlib",
        "typing",
    }
    assert "anyfesolver" not in imports


def test_checker_cli_writes_malformed_record_and_uses_exclusive_output(
    tmp_path: Path,
) -> None:
    proof_path = tmp_path / "proof.json"
    proof_path.write_bytes(b'{"schema":"wrong"}\n')
    output = tmp_path / "check.json"
    original = list(__import__("sys").argv)
    try:
        __import__("sys").argv = [
            str(CHECKER_PATH), "--proof", str(proof_path), "--output", str(output)
        ]
        assert checker.main() == 2
        record = json.loads(output.read_text())
        assert record["terminal"] == checker.MALFORMED
        assert record["predicates"]["evidence_valid"] is False
        with pytest.raises(FileExistsError):
            checker.main()
    finally:
        __import__("sys").argv = original
