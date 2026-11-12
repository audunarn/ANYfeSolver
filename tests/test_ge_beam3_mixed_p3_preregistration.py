"""Static authority checks for GE Beam3 mixed P3 opt-in integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "agent_plans" / "GE_BEAM3_MIXED_P3_OPT_IN_INTEGRATION_PLAN.md"
REFERENCE = ROOT / "docs" / "reference_cases"
BASELINE = REFERENCE / "ge_beam3_mixed_p3_baseline.json"
CASES = REFERENCE / "ge_beam3_mixed_p3_cases.json"
CONTRACT = REFERENCE / "ge_beam3_mixed_p3_contract.json"
REVIEW = REFERENCE / "ge_beam3_mixed_p3_plan_review.json"
STATE_SCHEMA = REFERENCE / "ge_beam3_mixed_p3_state_schema.json"

BASE_COMMIT = "e31c9e292a2fc9f6b57472bb8c5b90919a535492"
BASE_TREE = "531efe56882a4a5cb01c5c58c51844880259cd31"
SUBJECT = "docs: preregister GE Beam3 mixed P3 opt-in integration"
QUALIFIED_ID = "GE_BEAM3_DC_MIXED_K1_MACRO_V2"
SELECTOR = "ge-beam3"
PATHS = {
    "docs/agent_plans/GE_BEAM3_MIXED_P3_OPT_IN_INTEGRATION_PLAN.md",
    "docs/reference_cases/ge_beam3_mixed_p3_baseline.json",
    "docs/reference_cases/ge_beam3_mixed_p3_cases.json",
    "docs/reference_cases/ge_beam3_mixed_p3_contract.json",
    "docs/reference_cases/ge_beam3_mixed_p3_plan_review.json",
    "docs/reference_cases/ge_beam3_mixed_p3_state_schema.json",
    "tests/test_ge_beam3_mixed_p3_preregistration.py",
}
INPUT_IDENTITIES = {
    "docs/agent_plans/GE_BEAM3_MIXED_P3_OPT_IN_INTEGRATION_PLAN.md": (
        5413,
        "6A9C9E83B3EFE16D7D8DACCDE6EEAF8C611B05E57220490FBDE29F7A9756FAA1",
    ),
    "docs/reference_cases/ge_beam3_mixed_p3_baseline.json": (
        2890,
        "612822F27818AC5325FCBF2455E0A1E3C9E34520A1BE935D27CE494771887CE1",
    ),
    "docs/reference_cases/ge_beam3_mixed_p3_cases.json": (
        4625,
        "51655A793D5B33D059CA2985A8A2138132F94884E57F2983AAC7D67BE5906C5D",
    ),
    "docs/reference_cases/ge_beam3_mixed_p3_contract.json": (
        6769,
        "421BB55844F55027E53E8057A6E3AD9C6974D8559AA187984389A5CCACB7E54D",
    ),
    "docs/reference_cases/ge_beam3_mixed_p3_state_schema.json": (
        3352,
        "B36C8EDE081ECE464DC7A7E6921D4C81C98168DEB3B90271B72A8E81396D87A9",
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


def _canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
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


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(("git", "show", f"{commit}:{path}"), cwd=ROOT)


def test_preregistered_inputs_are_exact_canonical_and_independently_reviewed() -> None:
    observed: dict[str, bytes] = {}
    for path, identity in INPUT_IDENTITIES.items():
        raw = (ROOT / path).read_bytes()
        if path.endswith(".json"):
            assert _canonical(ROOT / path)[0] == raw
        assert (len(raw), _sha(raw)) == identity
        observed[path] = raw

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
        "ACCEPT_GE_BEAM3_MIXED_P3_OPT_IN_PREREGISTRATION_NO_P0_P1"
    )
    reviewed = {row["path"]: row for row in review["reviewed_inputs"]}
    test_path = "tests/test_ge_beam3_mixed_p3_preregistration.py"
    observed[test_path] = (ROOT / test_path).read_bytes()
    assert set(reviewed) == set(observed)
    for path, raw in observed.items():
        assert reviewed[path] == {
            "bytes": len(raw),
            "path": path,
            "sha256": _sha(raw),
        }


def test_duplicate_and_nonfinite_json_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        json.loads('{"x":1,"x":2}', object_pairs_hook=_unique)
    with pytest.raises(ValueError, match="nonfinite"):
        json.loads('{"x":NaN}', parse_constant=_reject_constant)


def test_p2_closeout_and_base_production_snapshot_are_hash_bound() -> None:
    _raw, baseline = _canonical(BASELINE)
    assert baseline["base"] == {
        "commit": BASE_COMMIT,
        "parent": "31b6ca4385e2f2f00dd8e9d874c2e89d0bac05dc",
        "subject": "docs: close GE Beam3 mixed P2 parity gate",
        "tree": BASE_TREE,
    }
    assert _git("rev-parse", f"{BASE_COMMIT}^{{tree}}") == BASE_TREE
    assert baseline["candidate_id"] == "CANDIDATE_GE_BEAM3_DC_MIXED_K1_MACRO_V2"
    assert baseline["qualified_formulation_id"] == QUALIFIED_ID
    assert baseline["selector"] == SELECTOR

    rows = (
        list(baseline["p2_closeout"][key] for key in ("evidence", "review", "status"))
        + list(baseline["production_snapshot"])
    )
    for row in rows:
        blob = _git_blob(BASE_COMMIT, row["path"])
        assert (len(blob), _sha(blob)) == (row["bytes"], row["sha256"])
        assert _git("rev-parse", f"{BASE_COMMIT}:{row['path']}") == row["git_blob"]

    assert baseline["p2_closeout"]["status"]["terminal"] == (
        "PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY"
    )
    assert baseline["p2_closeout"]["review"]["verdict"] == (
        "ACCEPT_PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY_NO_P0_P1"
    )


def test_stage_topology_is_exact_when_preregistration_commit_exists() -> None:
    rows = _git("log", "--format=%H%x09%s", "--all").splitlines()
    matches = [row.split("\t", 1)[0] for row in rows if row.endswith("\t" + SUBJECT)]
    if not matches:
        assert _git("rev-parse", "HEAD") == BASE_COMMIT
        return
    assert len(matches) == 1
    commit = matches[0]
    assert _git("rev-parse", f"{commit}^") == BASE_COMMIT
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
    )
    assert changed == PATHS


def test_contract_freezes_the_narrow_opt_in_boundary() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert contract["authority_state"] == "PREREGISTERED_NOT_IMPLEMENTED"
    assert contract["base"] == {
        "commit": BASE_COMMIT,
        "subject": "docs: close GE Beam3 mixed P2 parity gate",
        "tree": BASE_TREE,
    }
    integration = contract["integration_contract"]
    assert integration["class"] == "GeometricallyExactBeam3D3NElement"
    assert integration["exact_selector"] == SELECTOR
    assert integration["node_count"] == 3
    assert set(integration["forbidden_new_aliases"]) == {
        "b3",
        "beam3",
        "ge-beam-3",
        "ge_beam3",
        "gebeam3",
    }
    assert contract["qualified_formulation_id"] == QUALIFIED_ID
    assert contract["implementation_extent"]["mechanics_or_scientific_change_authorized"] is False
    assert contract["implementation_extent"]["preregistration_paths"] == sorted(PATHS)
    assert contract["production_boundary"] == {
        "default_activation_authorized": False,
        "distribution_publication_authorized": False,
        "ecosystem_exposure_authorized": False,
        "existing_aliases_unchanged": True,
        "existing_defaults_unchanged": True,
        "qualification_wheel_build_authorized": True,
        "qualified_q4_unchanged": True,
        "qualified_s3_v2d_unchanged": True,
        "tag_or_version_change_authorized": False,
    }
    protected = set(contract["implementation_extent"]["protected_unchanged"])
    assert "B2_MECHANICS_AND_ROUTING" in protected
    assert "LEGACY_B3_MECHANICS_ROUTING_AND_BATCHES" in protected
    assert "PACKAGE_METADATA_VERSION_DEPENDENCIES_AND_WORKFLOWS" in protected


def test_state_v2_is_strict_and_candidate_state_is_not_migratable() -> None:
    _raw, state = _canonical(STATE_SCHEMA)
    assert state["qualified_formulation_id"] == QUALIFIED_ID
    record = state["state_record"]
    assert record["formulation_id"] == QUALIFIED_ID
    assert record["state_schema"] == "anysolver.ge-beam3-mixed.committed_state.v2"
    assert record["state_version"] == 2
    assert len(record["exact_keys"]) == len(set(record["exact_keys"])) == 28
    rejection = state["candidate_state_rejection"]
    assert rejection["forbidden_state_version"] == 1
    assert rejection["migration"] == "FORBIDDEN"
    assert state["restart_policy"]["p2_candidate_state"] == (
        "REJECT_WITHOUT_MIGRATION"
    )
    assert "candidate_id" not in state["element_identity_preimage"]["exact_keys"]
    assert "formulation_id" in state["element_identity_preimage"]["exact_keys"]


def test_cases_package_performance_and_terminal_precedence_are_complete() -> None:
    _raw, cases = _canonical(CASES)
    _contract_raw, contract = _canonical(CONTRACT)
    assert len(cases["case_order"]) == len(set(cases["case_order"])) == 17
    assert [case["case_id"] for case in cases["cases"]] == cases["case_order"]
    assert cases["negative_selectors"] == [
        "ge_beam3",
        "beam3",
        "b3",
        "gebeam3",
        "ge-beam-3",
        "quadratic_beam",
        "beam",
    ]
    assert cases["installed_wheel"]["canonical_run_count"] == 2
    assert cases["installed_wheel"]["repository_paths"] == (
        "ABSENT_FROM_CWD_SYS_PATH_PYTHONPATH_AND_LOADED_MODULE_ORIGINS"
    )
    assert cases["performance"]["maximum_existing_path_median_ratio"] == "1.05"
    assert cases["performance"]["measured_pairs_minimum"] == 11
    assert cases["performance"]["ge_beam3_speed_claim"] == "NONE"
    expected_terminals = [
        "BLOCKED_GE_BEAM3_P3_BASELINE_OR_AUTHORITY",
        "BLOCKED_GE_BEAM3_P3_PROCESS_OR_EVIDENCE",
        "NO_GO_GE_BEAM3_P3_SELECTOR_OR_SERIALIZATION",
        "NO_GO_GE_BEAM3_P3_LINEAR_INTEGRATION",
        "NO_GO_GE_BEAM3_P3_PACKAGE_ISOLATION",
        "UNCLASSIFIED_GE_BEAM3_P3_PERFORMANCE",
        "PROVISIONAL_GO_GE_BEAM3_STRAIGHT_STANDALONE_OPT_IN",
    ]
    assert cases["terminal_precedence"] == expected_terminals
    assert contract["terminal_precedence"] == expected_terminals
    assert contract["runtime_bounds"] == {
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "inactivity_seconds": 300,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
    }


def test_base_has_no_public_ge_beam3_selector_or_export() -> None:
    elements = _git_blob(BASE_COMMIT, "src/anyfesolver/elements.py").decode("utf-8")
    package = _git_blob(BASE_COMMIT, "src/anyfesolver/__init__.py").decode("utf-8")
    assert '"ge-beam3"' not in elements
    assert "GeometricallyExactBeam3D3NElement" not in package
    assert "GE_BEAM3_QUALIFIED_FORMULATION_ID" not in package
