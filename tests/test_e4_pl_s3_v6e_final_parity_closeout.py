from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
PREFIX = "e4_pl_s3_v6e_v2d_final_parity"


def _canonical(name: str) -> tuple[bytes, dict]:
    raw = (REFERENCE / f"{PREFIX}_{name}.json").read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return raw, value


def test_v6e_closeout_binds_frozen_authority_and_two_cycles() -> None:
    _raw, result = _canonical("result")
    assert result["authority"] == {
        "commit": "823c8d663f45d859649cd1bfbf7f041ac67aa4df",
        "contract_sha256": "E9DCF2876DF35DEA42549C4D75A94481496772AA3927FD4965974408AE0CB2DE",
        "exact_implementation_paths": 4,
        "parent": "5b04487a156adf9be04e90d94a40e3d1d7ac0bfc",
        "subject": "feat: close S3 V2D final production parity",
        "tree": "519a074bf303162449bc21f41e30060f8ee991e2",
    }
    assert result["cycles"] == {"count": 2, "tests_passed_per_cycle": 66}
    assert result["frozen_commit_review"] == {"tests_passed": 66}
    assert result["terminal"] == (
        "PROVISIONAL_GO_E4_PL_S3_V6E_INDEPENDENT_FINAL_PARITY_REVIEW"
    )


def test_v6e_review_is_empty_and_does_not_claim_scientific_review() -> None:
    _raw, review = _canonical("review")
    assert review["findings"] == []
    assert review["verdict"] == "ACCEPT_E4_PL_S3_V6E_IMPLEMENTATION_NO_P0_P1"
    assert review["review_scope"] == {
        "did_not_execute_long_scientific_cycle": True,
        "reviewed_frozen_commit": True,
        "stage4a_scientific_rerun_reviewed": False,
    }


def test_v6e_status_hashes_and_restrictions_are_exact() -> None:
    result_raw, _result = _canonical("result")
    review_raw, _review = _canonical("review")
    _raw, status = _canonical("status")
    assert status["result"] == {
        "bytes": len(result_raw),
        "sha256": hashlib.sha256(result_raw).hexdigest().upper(),
    }
    assert status["review"] == {
        "bytes": len(review_raw),
        "sha256": hashlib.sha256(review_raw).hexdigest().upper(),
    }
    assert status["activation_authorized"] is False
    assert status["stage4a_scientific_rerun_authorized"] is False
    assert status["next_gate"] == "V6F_V2D_INDEPENDENT_FINAL_PARITY_REVIEW"
    assert status["production_restriction"] == (
        "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    )


def test_v6e_authority_commit_is_reachable_and_exact() -> None:
    commit = "823c8d663f45d859649cd1bfbf7f041ac67aa4df"
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert subject == "feat: close S3 V2D final production parity"
    assert paths == [
        "docs/reference_cases/e4_pl_s3_v6e_v2d_final_parity_contract.json",
        "src/anyfesolver/e4_pl_s3_v2d_element.py",
        "tests/test_e4_pl_s3_v6d_activity_contact_batch_contract.py",
        "tests/test_e4_pl_s3_v6e_final_parity.py",
    ]
