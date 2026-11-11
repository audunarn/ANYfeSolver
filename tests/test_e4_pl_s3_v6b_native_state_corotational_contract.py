from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

import anyfesolver
from anyfesolver.e4_pl_s3_v2d_element import (
    BLOCKED_OPERATIONS,
    CAPABILITY_MATRIX,
    COROTATIONAL_POLICY_ID,
    FORMULATION_ID,
    IMPLEMENTATION_ID,
    MATERIAL_LIFECYCLE_POLICY_ID,
    NATIVE_STATE_LAYOUT_ID,
    NATIVE_STATE_SCHEMA_ID,
    NativeParityE4PLS3V2DShellElement,
)
from anyfesolver.e4_pl_s3_v2d_state import V2DStateError, strict_canonical_json_loads
from anyfesolver.elements import create_shell_element


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v6b_v2d_native_state_corotational_contract.json"
INCIDENT = REFERENCE / "e4_pl_s3_v6b_v6a_checkout_hash_incident.json"


def _canonical(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = json.loads(
        raw,
        object_pairs_hook=lambda pairs: _reject_duplicate_pairs(pairs),
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


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict:
    made: dict[str, object] = {}
    for key, value in pairs:
        if key in made:
            raise ValueError(f"duplicate key {key}")
        made[key] = value
    return made


def _commit_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def test_v6b_contract_is_canonical_and_binds_the_accepted_v6a_chain() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            contract["authority"]["expected_parent"],
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    ).returncode == 0
    for name, expected in (
        ("contract", contract["authority"]["v6a_contract_sha256"]),
        ("result", contract["authority"]["v6a_result_sha256"]),
        ("review", contract["authority"]["v6a_review_sha256"]),
        ("status", contract["authority"]["v6a_status_sha256"]),
    ):
        raw = (
            REFERENCE / f"e4_pl_s3_v6a_v2d_linear_native_parity_{name}.json"
        ).read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == expected


def test_v6b_incident_binds_authority_blobs_without_reclassifying_v6a() -> None:
    _raw, incident = _canonical(INCIDENT)
    assert incident["classification"] == "REFERENCE_CHECKOUT_LINE_ENDING_HASH_BINDING_DEFECT"
    assert incident["conclusion"] == {
        "authority_commit_unchanged": True,
        "authority_commit_tree_remains_primary": True,
        "implementation_behavior_unchanged": True,
        "scientific_arrays_unchanged": True,
        "v6a_terminal_preserved": True,
    }
    commit = incident["v6a_authority"]["commit"]
    for record in incident["affected_records"]:
        payload = _commit_blob(commit, record["path"])
        assert len(payload) == record["authority_blob_size"]
        assert hashlib.sha256(payload).hexdigest().upper() == record["authority_blob_sha256"]
        oid = subprocess.run(
            ["git", "rev-parse", f"{commit}:{record['path']}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        assert oid == record["authority_blob_oid"]


def test_v6b_identity_capabilities_and_defaults_are_fail_closed() -> None:
    _raw, contract = _canonical(CONTRACT)
    candidate = contract["candidate"]
    source = _commit_blob(
        "afe691759a611278e341be7d58ffaae4f71bde89",
        "src/anyfesolver/e4_pl_s3_v2d_element.py",
    ).decode("utf-8")
    assignments: dict[str, object] = {}
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert assignments["FORMULATION_ID"] == candidate["formulation_id"]
    assert assignments["IMPLEMENTATION_ID"] == candidate["implementation_id"]
    assert assignments["COROTATIONAL_POLICY_ID"] == candidate["corotational_policy_id"]
    assert assignments["MATERIAL_LIFECYCLE_POLICY_ID"] == candidate[
        "material_lifecycle_policy_id"
    ]
    state_source = _commit_blob(
        "afe691759a611278e341be7d58ffaae4f71bde89",
        "src/anyfesolver/e4_pl_s3_v2d_state.py",
    ).decode("utf-8")
    state_assignments: dict[str, object] = {}
    for node in ast.parse(state_source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                state_assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert state_assignments["STATE_SCHEMA"] == candidate["state_schema"]
    assert state_assignments["STATE_LAYOUT_ID"] == candidate["state_layout_id"]
    assert anyfesolver.DEFAULT_Q4_FORMULATION == "e4-pl"
    assert anyfesolver.DEFAULT_S3_FORMULATION == "legacy-s3"
    element = create_shell_element(
        1,
        [1, 2, 3],
        "steel",
        formulation="e4-pl-s3-v2d",
        thickness=0.01,
        reference_normal=(0.0, 0.0, 1.0),
    )
    assert type(element) is NativeParityE4PLS3V2DShellElement
    assert set(contract["prohibited_scope"]).issuperset(
        {
            "qualified_s3_v1_physical_mechanics",
            "q4_physical_mechanics",
            "solver_integrated_hot_restart",
            "default_activation",
        }
    )
    assert contract["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }


def test_v6b_state_json_rejects_duplicate_nonfinite_noncanonical_and_bom() -> None:
    for raw in (
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{"b":1, "a":2}\n',
        b'\xef\xbb\xbf{}\n',
    ):
        with pytest.raises(V2DStateError):
            strict_canonical_json_loads(raw)


def test_v6b_implementation_does_not_import_qualified_v1_or_q4_mechanics() -> None:
    modules = {}
    for relative in (
        "src/anyfesolver/e4_pl_s3_v2d_element.py",
        "src/anyfesolver/e4_pl_s3_v2d_state.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        modules[relative] = imported
    all_imports = set().union(*modules.values())
    assert not any(name.endswith("e4_pl_s3_element") for name in all_imports)
    source = _commit_blob(
        "afe691759a611278e341be7d58ffaae4f71bde89",
        "src/anyfesolver/e4_pl_s3_v2d_element.py",
    ).decode("utf-8")
    assert "QualifiedE4PLS3ShellElement" not in source
    # The exact Q4 class is admitted only as a mixed-model registry identity;
    # no Q4 physical operator or method is dispatched.
    assert source.count("QualifiedE4PLShellElement") == 2
    v2d_tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "QualifiedE4PLShellElement"
        for node in ast.walk(v2d_tree)
    )
    assert "compute_shape_functions" not in source


def test_v6b_contract_has_exact_bounded_successor_extent() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert len(contract["exact_implementation_extent"]) == 9
    assert len(set(contract["exact_implementation_extent"])) == 9
    assert contract["runtime_policy"] == {
        "automatic_retry": False,
        "focused_cycle_limit_seconds": 600,
        "long_running_scientific_cycle_authorized": False,
        "required_focused_cycles": 2,
    }
    assert contract["terminal_precedence"][-1] == (
        "PROVISIONAL_GO_E4_PL_S3_V6B_OFFSET_LOAD_RESTART_INTEGRATION"
    )
    assert np.isfinite(contract["test_contract"]["plastic_tangent_relative_limit"])
