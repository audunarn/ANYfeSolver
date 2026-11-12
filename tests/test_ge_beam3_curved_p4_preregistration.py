"""Static authority checks for the private GE Beam3 curved P4 reference gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "agent_plans" / "GE_BEAM3_CURVED_P4_REFERENCE_CORE_PLAN.md"
REFERENCE = ROOT / "docs" / "reference_cases"
BASELINE = REFERENCE / "ge_beam3_curved_p4_baseline.json"
CASES = REFERENCE / "ge_beam3_curved_p4_cases.json"
CONTRACT = REFERENCE / "ge_beam3_curved_p4_contract.json"
MAP_A = REFERENCE / "ge_beam3_curved_p4_equation_map_a.json"
MAP_B = REFERENCE / "ge_beam3_curved_p4_equation_map_b.json"
REVIEW = REFERENCE / "ge_beam3_curved_p4_plan_review.json"
SOURCE_LEDGER = REFERENCE / "ge_beam3_curved_p4_source_ledger.json"
REFERENCE_SCHEMA = REFERENCE / "ge_beam3_curved_p4_reference_schema.json"

BASE_COMMIT = "8ac156cbb7632f2442f904e3ab73d6a7eb867670"
BASE_TREE = "fac6e105a64f837bfca0f73933ec3298c80bde2a"
P3_CLOSEOUT = "7aa359c18d1cf5db3dfb84d364afd9870e2da994"
P3_TREE = "8e3bde1484c5291b2bb74e4f03071d3cf23e4600"
SUBJECT = "docs: preregister GE Beam3 curved P4 reference core"
CANDIDATE_ID = "CANDIDATE_GE_BEAM3_DC_MIXED_CURVED_Q2_MACRO_V1"
RESERVED_FORMULATION_ID = "GE_BEAM3_DC_MIXED_CURVED_Q2_MACRO_V1"
PATHS = {
    "docs/agent_plans/GE_BEAM3_CURVED_P4_REFERENCE_CORE_PLAN.md",
    "docs/reference_cases/ge_beam3_curved_p4_baseline.json",
    "docs/reference_cases/ge_beam3_curved_p4_cases.json",
    "docs/reference_cases/ge_beam3_curved_p4_contract.json",
    "docs/reference_cases/ge_beam3_curved_p4_equation_map_a.json",
    "docs/reference_cases/ge_beam3_curved_p4_equation_map_b.json",
    "docs/reference_cases/ge_beam3_curved_p4_plan_review.json",
    "docs/reference_cases/ge_beam3_curved_p4_reference_schema.json",
    "docs/reference_cases/ge_beam3_curved_p4_source_ledger.json",
    "tests/test_ge_beam3_curved_p4_preregistration.py",
}
PROTECTED_STRAIGHT_BLOBS = {
    "src/anyfesolver/ge_beam3_element.py": "70542da7fea26dcb2da5b5f9da5fc0b0b8d484ab",
    "src/anyfesolver/ge_beam3_mixed_element.py": "f49062b55b4bf749a1654100cf3a4ed6ffb61d37",
    "src/anyfesolver/ge_beam3_mixed_state.py": "74caa1814448245f907bd3efdb6c8a1cc1d2269d",
    "src/anyfesolver/ge_beam3_state.py": "ba8f65761197c62cb2b7ce74bf488e75f2c24818",
}
TERMINALS = [
    "BLOCKED_GE_BEAM3_P4_BASELINE_OR_AUTHORITY",
    "BLOCKED_GE_BEAM3_P4_REFERENCE_PROCESS_OR_EVIDENCE",
    "NO_GO_GE_BEAM3_P4_REFERENCE_REGULARITY",
    "NO_GO_GE_BEAM3_P4_REFERENCE_FRAME_IDENTITY",
    "NO_GO_GE_BEAM3_P4_REFERENCE_REVERSAL_OR_OBJECTIVITY",
    "PROVISIONAL_GO_GE_BEAM3_P4_PRIVATE_CURVED_REFERENCE_CORE",
]

INPUT_IDENTITIES = {
    "docs/agent_plans/GE_BEAM3_CURVED_P4_REFERENCE_CORE_PLAN.md": (
        9417,
        "6E7E5B71FD23D2AE8D44E615A4EA1D581095609CA0949CCDA4BFBA0A33403986",
    ),
    "docs/reference_cases/ge_beam3_curved_p4_baseline.json": (
        3410,
        "02A078D49AB8CB2DE3B8156561FBCD662FD25448AA2F94518D7487A35E8E929F",
    ),
    "docs/reference_cases/ge_beam3_curved_p4_cases.json": (
        7129,
        "7D3DDC0B8E760D5E5AD4491BADAFD826588F0910469E497FBDE19844835A32EE",
    ),
    "docs/reference_cases/ge_beam3_curved_p4_contract.json": (
        7480,
        "123B4423EB7FD809961297D6100EFFDEE7C61739D1C520F4A5E37A2A75B5142F",
    ),
    "docs/reference_cases/ge_beam3_curved_p4_equation_map_a.json": (
        6517,
        "F360F30FB3CDD9D5E4C9BD762338F5F833106B0ABEE84D889DFA4808E7FE047C",
    ),
    "docs/reference_cases/ge_beam3_curved_p4_equation_map_b.json": (
        7462,
        "70CFA9E52D91A60256C11B6C354A16AE084991254B95B9030BF0A52E84B4E977",
    ),
    "docs/reference_cases/ge_beam3_curved_p4_reference_schema.json": (
        5306,
        "364287E0D8044C06A110F68F6428DB516CAA4A5F6E8052D8F96B24C7B16B8B9E",
    ),
    "docs/reference_cases/ge_beam3_curved_p4_source_ledger.json": (
        8483,
        "A11BBEAAEAD98EE6B4C12FC21741437DAE450206F8BA2586633A7351452C5F21",
    ),
}


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    made: dict[str, object] = {}
    for key, value in pairs:
        if key in made:
            raise ValueError(f"duplicate key: {key}")
        made[key] = value
    return made


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite value: {value}")


def _repository_canonical_bytes(path: Path) -> bytes:
    """Return Git-canonical text bytes despite a reversible CRLF checkout."""

    raw = path.read_bytes()
    if b"\r" not in raw:
        return raw
    normalized = raw.replace(b"\r\n", b"\n")
    assert b"\r" not in normalized, f"noncanonical carriage return in {path}"
    return normalized


def _canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = _repository_canonical_bytes(path)
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique,
        parse_constant=_reject_constant,
    )
    assert isinstance(value, dict)
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert raw == expected
    return raw, value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _sanitized_git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )


def _is_explicit_github_shallow_boundary() -> bool:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return False
    shallow_repository = _sanitized_git("rev-parse", "--is-shallow-repository")
    shallow_name = _sanitized_git("rev-parse", "--git-path", "shallow")
    head = _sanitized_git("rev-parse", "HEAD")
    if (
        shallow_repository.returncode
        or shallow_repository.stdout.strip() != b"true"
        or shallow_name.returncode
        or head.returncode
    ):
        return False
    shallow = Path(os.fsdecode(shallow_name.stdout.strip()))
    if not shallow.is_absolute():
        shallow = (ROOT / shallow).resolve()
    return shallow.is_file() and head.stdout.decode("ascii").strip() in shallow.read_text(
        encoding="ascii"
    ).splitlines()


def _git(*arguments: str) -> str:
    result = _sanitized_git(*arguments)
    if result.returncode and _is_explicit_github_shallow_boundary():
        pytest.skip("required history is beyond the explicit GitHub shallow boundary")
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout.decode("utf-8").strip()


def _git_blob(commit: str, path: str) -> bytes:
    result = _sanitized_git("show", "--no-ext-diff", "--no-textconv", f"{commit}:{path}")
    if result.returncode and _is_explicit_github_shallow_boundary():
        pytest.skip("required historical blob is beyond the explicit GitHub shallow boundary")
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout


def test_preregistered_inputs_are_exact_canonical_and_independently_reviewed() -> None:
    test_path = "tests/test_ge_beam3_curved_p4_preregistration.py"
    expected_paths = PATHS - {
        "docs/reference_cases/ge_beam3_curved_p4_plan_review.json",
        test_path,
    }
    assert set(INPUT_IDENTITIES) == expected_paths
    observed: dict[str, bytes] = {}
    for path, identity in INPUT_IDENTITIES.items():
        raw = _repository_canonical_bytes(ROOT / path)
        if path.endswith(".json"):
            assert _canonical(ROOT / path)[0] == raw
        assert (len(raw), _sha(raw)) == identity
        observed[path] = raw
    observed[test_path] = _repository_canonical_bytes(ROOT / test_path)

    review_raw, review = _canonical(REVIEW)
    assert review_raw
    assert set(review) == {
        "findings",
        "reviewed_inputs",
        "reviewer_independence",
        "schema",
        "verdict",
    }
    assert review["findings"] == []
    assert review["verdict"] == (
        "ACCEPT_GE_BEAM3_CURVED_P4_REFERENCE_PREREGISTRATION_NO_P0_P1"
    )
    reviewed = {row["path"]: row for row in review["reviewed_inputs"]}
    assert set(reviewed) == set(observed)
    for path, raw in observed.items():
        assert reviewed[path] == {
            "bytes": len(raw),
            "path": path,
            "sha256": _sha(raw),
        }


def test_duplicate_nonfinite_and_noncanonical_json_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        json.loads('{"x":1,"x":2}', object_pairs_hook=_unique)
    with pytest.raises(ValueError, match="nonfinite"):
        json.loads('{"x":NaN}', parse_constant=_reject_constant)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b'{"b":2, "a":1}\n')
    with pytest.raises(AssertionError):
        _canonical(noncanonical)


def test_base_p3_closeout_and_preregistration_topology_are_exact() -> None:
    _raw, baseline = _canonical(BASELINE)
    assert baseline["base"] == {
        "commit": BASE_COMMIT,
        "parent": P3_CLOSEOUT,
        "subject": "test: make GE Beam3 closeout checkout neutral",
        "tree": BASE_TREE,
    }
    assert baseline["p3_authority"]["closeout_commit"] == P3_CLOSEOUT
    assert baseline["p3_authority"]["closeout_tree"] == P3_TREE
    assert _git("rev-parse", f"{BASE_COMMIT}^{{tree}}") == BASE_TREE
    assert _git("rev-parse", f"{BASE_COMMIT}^") == P3_CLOSEOUT

    rows = _git("log", "--format=%H%x09%s", "--all").splitlines()
    matches = [row.split("\t", 1)[0] for row in rows if row.endswith("\t" + SUBJECT)]
    if not matches:
        assert _git("rev-parse", "HEAD") == BASE_COMMIT
        return
    assert len(matches) == 1
    preregistration = matches[0]
    assert _git("rev-parse", f"{preregistration}^") == BASE_COMMIT
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", preregistration).splitlines()
    )
    assert changed == PATHS


def test_accepted_straight_core_is_bound_by_git_blob_and_content_hash() -> None:
    _raw, baseline = _canonical(BASELINE)
    rows = {row["path"]: row for row in baseline["protected_straight_blobs"]}
    assert set(PROTECTED_STRAIGHT_BLOBS) < set(rows)
    for path, git_blob in PROTECTED_STRAIGHT_BLOBS.items():
        row = rows[path]
        assert row["git_blob"] == git_blob
        assert _git("rev-parse", f"{P3_CLOSEOUT}:{path}") == git_blob
        assert _git("rev-parse", f"{BASE_COMMIT}:{path}") == git_blob
        blob = _git_blob(P3_CLOSEOUT, path)
        assert (len(blob), _sha(blob)) == (row["bytes"], row["sha256"])
    for path in ("src/anyfesolver/__init__.py", "src/anyfesolver/elements.py", "pyproject.toml"):
        row = rows[path]
        assert _git("rev-parse", f"{P3_CLOSEOUT}:{path}") == row["git_blob"]
        assert _git("rev-parse", f"{BASE_COMMIT}:{path}") == row["git_blob"]
        blob = _git_blob(P3_CLOSEOUT, path)
        assert (len(blob), _sha(blob)) == (row["bytes"], row["sha256"])


def test_source_and_independent_equation_maps_are_complete() -> None:
    _ledger_raw, ledger = _canonical(SOURCE_LEDGER)
    _map_a_raw, map_a = _canonical(MAP_A)
    _map_b_raw, map_b = _canonical(MAP_B)
    assert ledger["candidate_id"] == CANDIDATE_ID
    assert map_a["candidate_id"] == CANDIDATE_ID
    assert map_b["candidate_id"] == CANDIDATE_ID
    assert map_a["schema"] != map_b["schema"]
    assert map_a["authorship"]["map_b_used_as_derivation_input"] is False
    assert map_b["authorship"]["map_a_formula_or_json_structure_used_as_derivation_input"] is False
    authority_classes = ledger["authority_classes"]
    assert set(authority_classes) >= {"B", "D", "P"}
    assert ledger["policy"]["background_cannot_fill_derivation_gaps"] is True
    assert ledger["policy"]["empirical_stabilization_forbidden"] is True
    assert ledger["policy"]["source_hash_drift_blocks_freeze"] is True


def test_reference_schema_freezes_geometry_orientation_and_restart_boundary() -> None:
    _raw, schema = _canonical(REFERENCE_SCHEMA)
    assert schema["candidate_id"] == CANDIDATE_ID
    assert schema["reserved_formulation_id"] == RESERVED_FORMULATION_ID
    assert schema["reference_curve"]["id"] == "P2_LAGRANGE_XI_MINUS1_ZERO_PLUS1"
    assert schema["reference_curve"]["minimum_method"] == (
        "EXACT_QUADRATIC_NORM_MINIMUM_AT_ENDPOINTS_AND_CLAMPED_STATIONARY_POINT"
    )
    assert schema["orientation_authority"]["admissible"] == (
        "EXPLICIT_PHYSICAL_NODAL_MATERIAL_TRIADS"
    )
    assert schema["orientation_authority"]["global_axis_inference"] == "FORBIDDEN"
    assert schema["closed_reference_policy"]["closed_single_element"] == (
        "FORBIDDEN_BY_DISTINCT_GEOMETRY_AND_REGULARITY"
    )
    assert schema["restart_boundary"]["hot_restart"] == (
        "FORBIDDEN_BEFORE_A_SEPARATE_STATE_CONTRACT"
    )
    assert schema["reference_frame_field"]["frame_interpolation_id"] == (
        "PIECEWISE_SHORTEST_TRANSPORT_LINEAR_ROLL_V1_INDEPENDENT_DERIVATION"
    )
    assert schema["registered_defaults"] == {
        "frame_tolerance": "1E-12",
        "half_frame_rotation_limit": "0.9*PI",
        "regularity_relative_tolerance": "MAX_64_TIMES_BINARY64_EPSILON_AND_1E_MINUS14",
        "rotation_tolerance": "1E-11",
    }
    exact_keys = schema["reference_record"]["exact_keys"]
    assert len(exact_keys) == len(set(exact_keys)) == 9
    assert schema["reference_record"]["branch_record_count"] == 2


def test_cases_are_ordered_complete_bounded_and_fail_closed() -> None:
    _raw, cases = _canonical(CASES)
    assert cases["candidate_id"] == CANDIDATE_ID
    assert cases["reserved_formulation_id"] == RESERVED_FORMULATION_ID
    assert cases["selector"] == "NONE_PRIVATE_REFERENCE_GATE"
    assert len(cases["case_order"]) == len(set(cases["case_order"])) == 26
    assert [row["case_id"] for row in cases["cases"]] == cases["case_order"]
    assert cases["reference_gate_expected_counts"] == {
        "half_intervals": 2,
        "intrinsic_strain_components": 6,
        "reference_nodes": 3,
    }
    assert cases["runtime_bounds"] == {
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "inactivity_seconds": 300,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_canonical_cycle_count": 2,
    }
    assert cases["terminal_precedence"] == TERMINALS
    assert cases["evidence_policy"]["failed_partial_canonical_output"] is False


def test_contract_is_private_preregistration_authority_only() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert contract["authority_state"] == "PREREGISTERED_REFERENCE_GATE_ONLY"
    assert contract["candidate_id"] == CANDIDATE_ID
    assert contract["reserved_formulation_id"] == RESERVED_FORMULATION_ID
    assert contract["scientific_execution_authorized"] is False
    assert contract["implementation_authorized_by_preregistration_commit"] is False
    assert contract["terminal_precedence"] == TERMINALS
    assert contract["runtime_bounds"] == _canonical(CASES)[1]["runtime_bounds"]
    assert contract["stage_extents"]["PREREGISTRATION"] == sorted(PATHS)
    assert contract["production_boundary"] == {
        "accepted_straight_ge_beam3_mechanics_unchanged": True,
        "default_activation_authorized": False,
        "distribution_publication_authorized": False,
        "ecosystem_exposure_authorized": False,
        "existing_aliases_unchanged": True,
        "existing_defaults_unchanged": True,
        "new_public_export_authorized": False,
        "new_selector_authorized": False,
        "package_dependency_or_workflow_change_authorized": False,
        "qualified_q4_unchanged": True,
        "qualified_s3_v2d_unchanged": True,
        "tag_or_version_change_authorized": False,
    }
    protected = {row["path"]: row["git_blob"] for row in contract["protected_straight_blobs"]}
    assert protected == PROTECTED_STRAIGHT_BLOBS


def test_contract_hash_dag_binds_every_noncyclic_authority_input() -> None:
    _raw, contract = _canonical(CONTRACT)
    authority = {row["path"]: row for row in contract["preregistration_authority"]}
    expected = set(INPUT_IDENTITIES) - {
        "docs/reference_cases/ge_beam3_curved_p4_contract.json"
    }
    assert set(authority) == expected
    for path in expected:
        size, sha256 = INPUT_IDENTITIES[path]
        assert authority[path] == {"bytes": size, "path": path, "sha256": sha256}


def test_production_routes_and_accepted_straight_blobs_are_not_in_preregistration_extent() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert contract["allowed_current_stage_paths"] == sorted(PATHS)
    assert not any(path.startswith("src/") for path in contract["allowed_current_stage_paths"])
    assert contract["deferred"]["curved_mechanics_implementation"] is True
    assert contract["deferred"]["curved_potential_residual_tangent_condensation"] is True
    assert contract["deferred"]["reference_linear_rank_and_rigid_mode_claim"] is True
    assert contract["deferred"]["public_selector_and_serialization"] is True
    assert contract["deferred"]["beam_shell_objective_joint"] is True
    assert contract["deferred"]["curved_section_and_history_state"] is True
    assert contract["deferred"]["finite_rotation_transient_and_gyroscopic_terms"] is True
