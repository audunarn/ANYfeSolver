"""Static authority checks for the GE Beam3 mixed P2 parity successor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "agent_plans" / "GE_BEAM3_MIXED_P2_SOLVER_PARITY_PLAN.md"
REFERENCE = ROOT / "docs" / "reference_cases"
BASELINE = REFERENCE / "ge_beam3_mixed_p2_baseline.json"
CASES = REFERENCE / "ge_beam3_mixed_p2_cases.json"
CONTRACT = REFERENCE / "ge_beam3_mixed_p2_contract.json"
INDEPENDENT_REFERENCE = REFERENCE / "ge_beam3_mixed_p2_independent_reference.py"
REVIEW = REFERENCE / "ge_beam3_mixed_p2_plan_review.json"
STATE_SCHEMA = REFERENCE / "ge_beam3_mixed_p2_state_schema.json"

BASE_COMMIT = "9d2bde784356bc7a5a8d314cd91be60c607acac2"
BASE_TREE = "ec80d9349a7f173a99fc808140f4ecaf2e85cfcd"
SUBJECT = "docs: preregister GE Beam3 mixed solver parity"
ACCEPTED_PREREGISTRATION_COMMIT = "448ae38cad85365f0058fca8174fb5a12aa57bbc"
PATHS = {
    "docs/agent_plans/GE_BEAM3_MIXED_P2_SOLVER_PARITY_PLAN.md",
    "docs/reference_cases/ge_beam3_mixed_p2_baseline.json",
    "docs/reference_cases/ge_beam3_mixed_p2_cases.json",
    "docs/reference_cases/ge_beam3_mixed_p2_contract.json",
    "docs/reference_cases/ge_beam3_mixed_p2_independent_reference.py",
    "docs/reference_cases/ge_beam3_mixed_p2_plan_review.json",
    "docs/reference_cases/ge_beam3_mixed_p2_state_schema.json",
    "tests/test_ge_beam3_mixed_p2_preregistration.py",
}


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


def _accepted_route_text(path: str) -> str:
    object_name = f"{ACCEPTED_PREREGISTRATION_COMMIT}:{path}"
    shown = _sanitized_git("show", "--no-ext-diff", "--no-textconv", object_name)
    if shown.returncode:
        assert _is_explicit_github_shallow_boundary(), (
            f"accepted route is missing outside an explicit GitHub shallow boundary: "
            f"{object_name}"
        )
        pytest.skip("accepted historical route is beyond the explicit GitHub shallow boundary")
    return shown.stdout.decode("utf-8")


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


def test_preregistered_inputs_are_exact_and_canonical() -> None:
    plan_raw = PLAN.read_bytes()
    baseline_raw, baseline = _canonical(BASELINE)
    cases_raw, cases = _canonical(CASES)
    contract_raw, contract = _canonical(CONTRACT)
    independent_raw = INDEPENDENT_REFERENCE.read_bytes()
    review_raw, review = _canonical(REVIEW)
    state_raw, state = _canonical(STATE_SCHEMA)

    assert (len(plan_raw), _sha(plan_raw)) == (
        6708,
        "0FD8628E08E511CF9381EF226705CFB485A76068EE1B21B069DF1A76BD60A3CD",
    )
    assert (len(baseline_raw), _sha(baseline_raw)) == (
        1236,
        "03326563D06303BD67325D120B78898DB3B1D63CAE0D2BCBC2B3D1635C742771",
    )
    assert (len(contract_raw), _sha(contract_raw)) == (
        8361,
        "CA5D6232CFEFE6662AAE2C5A86C3F8271620F132FF945686A45FFADBB9437478",
    )
    assert (len(cases_raw), _sha(cases_raw)) == (
        18340,
        "0A42F614F8F3C9E91208116EDEE1B30EFBA7B76FBB5F2CC72264A1F953260739",
    )
    assert (len(independent_raw), _sha(independent_raw)) == (
        24964,
        "3C400BE7A2A514517E093D5803FBC551F1749104C0E5666062EB7E186716479B",
    )
    assert (len(state_raw), _sha(state_raw)) == (
        5744,
        "3CC84F32EC88014348CACE3E80A6FDA53736A44D837B6E00960BC52F2EE86231",
    )
    assert baseline["base"]["commit"] == BASE_COMMIT
    assert baseline["base"]["tree"] == BASE_TREE
    assert contract["base"]["commit"] == BASE_COMMIT
    assert contract["base"]["tree"] == BASE_TREE
    assert cases["schema"] == "anysolver.ge-beam3-mixed-p2-cases-v1"
    assert state["state_schema"] == "anysolver.ge_beam3_mixed.committed_state.v1"
    assert set(review) == {
        "findings",
        "reviewed_inputs",
        "reviewer_independence",
        "schema",
        "verdict",
    }
    assert review["findings"] == []
    assert review["verdict"] == (
        "ACCEPT_GE_BEAM3_MIXED_P2_PREREGISTRATION_NO_P0_P1"
    )
    reviewed = {row["path"]: row for row in review["reviewed_inputs"]}
    expected_reviewed = {
        "docs/agent_plans/GE_BEAM3_MIXED_P2_SOLVER_PARITY_PLAN.md": plan_raw,
        "docs/reference_cases/ge_beam3_mixed_p2_baseline.json": baseline_raw,
        "docs/reference_cases/ge_beam3_mixed_p2_cases.json": cases_raw,
        "docs/reference_cases/ge_beam3_mixed_p2_contract.json": contract_raw,
        "docs/reference_cases/ge_beam3_mixed_p2_independent_reference.py": independent_raw,
        "docs/reference_cases/ge_beam3_mixed_p2_state_schema.json": state_raw,
        "tests/test_ge_beam3_mixed_p2_preregistration.py": Path(__file__).read_bytes(),
    }
    assert set(reviewed) == set(expected_reviewed)
    for path, raw in expected_reviewed.items():
        assert reviewed[path] == {"bytes": len(raw), "path": path, "sha256": _sha(raw)}


def test_stage_topology_is_exact_when_the_preregistration_commit_exists() -> None:
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


def test_solver_chart_state_and_terminal_contract_are_fail_closed() -> None:
    _raw, contract = _canonical(CONTRACT)

    assert contract["solver_chart"] == {
        "absolute_material_triad": "Q_OPERATOR_TRIAL_TIMES_REFERENCE_TRIAD",
        "accepted_research_path_unchanged": "EXP_X_LEFT_MULTIPLIES_Q_TRIAL",
        "required_solver_path": (
            "EXP_DELTA_PLUS_X_LEFT_MULTIPLIES_Q_OPERATOR_COMMITTED_TIMES_"
            "REFERENCE_TRIAD"
        ),
        "second_derivative": "ANALYTIC_JET2_FROM_SAME_DISCRETE_POTENTIAL",
        "spatial_operator_update": (
            "Q_OPERATOR_TRIAL_EQUALS_EXP_DELTA_TIMES_Q_OPERATOR_COMMITTED"
        ),
    }
    assert contract["state_contract"]["committed_rotation_storage"] == (
        "NODE_SHARED_SPATIAL_OPERATORS_NOT_ABSOLUTE_MATERIAL_TRIADS"
    )
    assert contract["section_state"]["history_bearing"] == (
        "REJECT_UNTIL_SEPARATE_PARTIAL_COMPLEMENTARY_CONTRACT"
    )
    assert all(contract["deferred"].values())
    assert contract["scientific_execution_authorized"] is False
    assert contract["terminal_precedence"] == [
        "BLOCKED_GE_BEAM3_P2_BASELINE_OR_AUTHORITY",
        "BLOCKED_GE_BEAM3_P2_PROCESS_OR_EVIDENCE",
        "NO_GO_GE_BEAM3_P2_SOLVER_CHART_OR_STATE",
        "NO_GO_GE_BEAM3_P2_LOAD_MASS_OR_RECOVERY",
        "NO_GO_GE_BEAM3_P2_MODAL_OR_BUCKLING",
        "PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY",
    ]


def test_execution_bounds_and_production_boundary_are_exact() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert contract["execution_bounds"] == {
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "inactivity_seconds": 300,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_canonical_cycle_count": 2,
    }
    assert contract["production_boundary"] == {
        "default_activation_authorized": False,
        "existing_aliases_unchanged": True,
        "existing_defaults_unchanged": True,
        "public_selector_before_p3": False,
        "qualified_q4_unchanged": True,
        "qualified_s3_v2d_unchanged": True,
    }
    for route in (
        "src/anyfesolver/__init__.py",
        "src/anyfesolver/elements.py",
    ):
        text = _accepted_route_text(route)
        assert "GeometricallyExactBeam3D3NElement" not in text
        assert '"ge-beam3"' not in text


def test_case_state_and_independent_reference_authority_are_complete() -> None:
    cases_raw, cases = _canonical(CASES)
    state_raw, state = _canonical(STATE_SCHEMA)
    _raw, contract = _canonical(CONTRACT)
    authority = contract["scientific_authority"]
    assert authority["case_manifest"] == {
        "bytes": len(cases_raw),
        "path": "docs/reference_cases/ge_beam3_mixed_p2_cases.json",
        "sha256": _sha(cases_raw),
    }
    assert authority["state_schema"] == {
        "bytes": len(state_raw),
        "path": "docs/reference_cases/ge_beam3_mixed_p2_state_schema.json",
        "sha256": _sha(state_raw),
    }
    assert state["hook_contract"]["native_state_consistency_required"] is False
    assert len(state["state_exact_keys"]) == len(set(state["state_exact_keys"])) == 28
    assert cases["unsupported_routes"] == [
        "CONSERVATIVE_FOLLOWER",
        "CURRENT_STATE_BUCKLING",
        "CURRENT_STATE_MODAL",
        "FINITE_ROTATION_TRANSIENT",
        "GYROSCOPIC_TERMS",
        "LINEAR_TRANSIENT_UNQUALIFIED",
        "NONCONSERVATIVE_FOLLOWER",
    ]
    result = subprocess.check_output(
        ("python", str(INDEPENDENT_REFERENCE), "--check-only"),
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result)
    assert payload == {
        "content_sha256": "D3DBB959B0F3C979CF57BB887E4BBDA2BA0B02D70026B16D5B15E38D8B8D2AE8",
        "status": "PASS",
    }


def test_p2_cannot_expose_selector_package_or_transient_routes() -> None:
    _raw, contract = _canonical(CONTRACT)
    assert contract["loads"]["descriptor_scope"] == (
        "PRIVATE_DIRECT_ELEMENT_METHOD_NOT_LOADCASE_OR_PUBLIC_SERIALIZATION"
    )
    assert contract["mass_modal_buckling"]["buckling_input"] == (
        "EXACTLY_ONE_OF_COMPRESSION_POSITIVE_AXIAL_COMPRESSION_OR_TENSION_POSITIVE_AXIAL_FORCE"
    )
    assert "LINEAR_TRANSIENT" in contract["mass_modal_buckling"]["unsupported_fail_closed"]
    assert contract["production_boundary"]["public_selector_before_p3"] is False
    assert contract["stage_extents"]["PREREGISTRATION"] == sorted(PATHS)
