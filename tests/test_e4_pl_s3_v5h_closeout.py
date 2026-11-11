from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
CONTRACT = REFERENCE / "e4_pl_s3_v5h_local_parity_contract.json"
RESULT = REFERENCE / "e4_pl_s3_v5h_local_parity_result.json"
REVIEW = REFERENCE / "e4_pl_s3_v5h_local_parity_review.json"
STATUS = REFERENCE / "e4_pl_s3_v5h_local_parity_status.json"
PASS = "PROVISIONAL_GO_E4_PL_S3_V5H_STAGE4B_PROTOCOL_PREPARATION"


def _canonical(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    return value


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest().upper(),
    }


def test_result_review_and_status_are_canonical_and_bound() -> None:
    result = _canonical(RESULT)
    review = _canonical(REVIEW)
    status = _canonical(STATUS)
    assert status["result"] == _binding(RESULT)
    assert status["review"] == _binding(REVIEW)
    assert status["external_aggregate"] == result["external_evidence"]["aggregate"]
    assert review["reviewed_inputs"]["aggregate_sha256"] == (
        result["external_evidence"]["aggregate"]["sha256"]
    )


def test_accepted_terminal_has_strict_nonactivation_boundary() -> None:
    result = _canonical(RESULT)
    review = _canonical(REVIEW)
    status = _canonical(STATUS)
    assert result["terminal"] == status["terminal"] == PASS
    assert set(review) == {
        "conclusions",
        "findings",
        "reviewed_inputs",
        "schema",
        "verdict",
    }
    assert review["findings"] == {"P0": [], "P1": []}
    assert review["reviewed_inputs"]["result_sha256"] == _binding(RESULT)["sha256"]
    for value in (result, status):
        assert value["activation_authorized"] is False
        assert value["stage4b_execution_authorized"] is False
        assert value["stage4b_protocol_preparation_authorized"] is True
        assert value["production_restriction"] == (
            "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
        )


def test_authority_commit_and_production_boundary_are_exact() -> None:
    contract = _canonical(CONTRACT)
    result = _canonical(RESULT)
    authority = result["authority"]
    assert authority["parent"] == contract["authority"]["expected_parent"]
    assert authority["subject"] == contract["authority"]["expected_subject"]
    actual_tree = subprocess.check_output(
        ["git", "rev-parse", f"{authority['commit']}^{{tree}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert actual_tree == authority["tree"]
    paths = subprocess.check_output(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            authority["commit"],
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert paths == contract["authority"]["exact_paths"]
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", authority["parent"], authority["commit"]],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert "src/anyfesolver/e4_pl_element.py" not in changed
