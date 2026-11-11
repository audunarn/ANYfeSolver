from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver.e4_pl_s3_v2d_element import (
    BATCH_POLICY_ID,
    BLOCKED_OPERATIONS,
    CAPABILITY_MATRIX,
    FORMULATION_ID,
    FORMULATION_SCHEMA,
    NATIVE_STATE_LAYOUT_ID,
    NATIVE_STATE_SCHEMA_ID,
    SERIALIZATION_POLICY_ID,
    SUPPORTED_OPERATIONS,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v6d_v2d_activity_contact_batch_contract.json"


def _canonical(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict:
        made: dict[str, object] = {}
        for key, value in pairs:
            if key in made:
                raise ValueError(f"duplicate key {key}")
            made[key] = value
        return made

    value = json.loads(
        raw,
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"nonfinite token {token}")
        ),
    )
    expected = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    assert raw == expected
    return raw, value


def test_v6d_contract_binds_v6c_and_current_candidate_identity() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract["authority"]["expected_parent"], "HEAD"],
        cwd=ROOT,
        check=True,
    ).returncode == 0
    for name, expected in (
        ("contract", contract["authority"]["v6c_contract_sha256"]),
        ("result", contract["authority"]["v6c_result_sha256"]),
        ("review", contract["authority"]["v6c_review_sha256"]),
        ("status", contract["authority"]["v6c_status_sha256"]),
    ):
        raw = (
            REFERENCE / f"e4_pl_s3_v6c_v2d_offset_load_restart_{name}.json"
        ).read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == expected
    assert contract["candidate"] == {
        "batch_policy_id": BATCH_POLICY_ID,
        "formulation_id": FORMULATION_ID,
        "formulation_schema": FORMULATION_SCHEMA,
        "implementation_id": "E4_PL_S3_V2D_ACTIVITY_CONTACT_BATCH_GATE_V1",
        "selector": "e4-pl-s3-v2d",
        "serialization_policy_id": SERIALIZATION_POLICY_ID,
        "state_layout_id": NATIVE_STATE_LAYOUT_ID,
        "state_schema": NATIVE_STATE_SCHEMA_ID,
    }


def test_v6d_capability_boundary_and_defaults_are_exact() -> None:
    required = {"activity_state", "contact_state", "qualified_batch_path"}
    assert required <= SUPPORTED_OPERATIONS
    assert all(CAPABILITY_MATRIX[name] == "SUPPORTED" for name in required)
    assert set(BLOCKED_OPERATIONS) == {
        "default_activation",
        "python_pickle_restart",
        "v2_or_earlier_state_hot_migration",
    }
    assert all(
        CAPABILITY_MATRIX[name] == "BLOCKED_PENDING_SUCCESSOR_GATE"
        for name in BLOCKED_OPERATIONS
    )
    assert anyfesolver.DEFAULT_Q4_FORMULATION == "e4-pl"
    assert anyfesolver.DEFAULT_S3_FORMULATION == "legacy-s3"


def test_v6d_routes_are_orchestration_only_and_fail_closed() -> None:
    element_source = (ROOT / "src/anyfesolver/e4_pl_s3_v2d_element.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(element_source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert not any(name.endswith("e4_pl_s3_element") for name in imports)
    assert "QualifiedE4PLS3ShellElement" not in element_source
    for route in (
        "seal_noncurrent_deleted_state",
        "validate_noncurrent_deleted_state",
        "compute_noncurrent_deleted_residual_operator",
        "native_reference_directors",
    ):
        assert route in element_source
    batch_source = (ROOT / "src/anyfesolver/s3_v2d_fast_assembly.py").read_text(
        encoding="utf-8"
    )
    assert "e4_pl_s3_element" not in batch_source
    assert "compute_stiffness_matrix" in batch_source
    assert "speedup_claimed" in batch_source


def test_v6d_contract_has_exact_bounded_extent_and_terminal_order() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert len(contract["exact_implementation_extent"]) == 11
    assert len(set(contract["exact_implementation_extent"])) == 11
    assert contract["runtime_policy"] == {
        "automatic_retry": False,
        "focused_cycle_limit_seconds": 600,
        "long_running_scientific_cycle_authorized": False,
        "required_focused_cycles": 2,
    }
    assert contract["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }
    assert contract["terminal_precedence"][-1] == (
        "PROVISIONAL_GO_E4_PL_S3_V6D_FINAL_PARITY_REVIEW"
    )
