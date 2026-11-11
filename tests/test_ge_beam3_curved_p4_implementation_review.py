"""Static acceptance checks for the corrected private P4 reference core."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs" / "reference_cases" / "ge_beam3_curved_p4_implementation_review.json"
PREREGISTRATION = "5cf0fc884685b454ea645c2052c7cba60c66cbbb"
INITIAL = "214d6de76795bb7d656dc377166bd05cc35c6a3f"
CORRECTION = "d59ed224cabae23fac4ef68a74bc53ef5602b3db"
SUBJECT = "docs: accept corrected GE Beam3 curved P4 reference core"
PATHS = {
    "docs/reference_cases/ge_beam3_curved_p4_implementation_review.json",
    "tests/test_ge_beam3_curved_p4_implementation_review.py",
}
PROTECTED = {
    "src/anyfesolver/ge_beam3_element.py": "70542da7fea26dcb2da5b5f9da5fc0b0b8d484ab",
    "src/anyfesolver/ge_beam3_mixed_element.py": "f49062b55b4bf749a1654100cf3a4ed6ffb61d37",
    "src/anyfesolver/ge_beam3_mixed_state.py": "74caa1814448245f907bd3efdb6c8a1cc1d2269d",
    "src/anyfesolver/ge_beam3_state.py": "ba8f65761197c62cb2b7ce74bf488e75f2c24818",
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
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    assert b"\r" not in raw
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique,
        parse_constant=_reject_constant,
    )
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
    assert isinstance(value, dict)
    return raw, value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ("git", "show", "--no-ext-diff", "--no-textconv", f"{commit}:{path}"),
        cwd=ROOT,
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def test_review_accepts_only_the_corrected_two_path_reference_core() -> None:
    _raw, review = _canonical(REVIEW)
    assert set(review) == {
        "findings",
        "reviewed_inputs",
        "reviewer_independence",
        "schema",
        "verdict",
    }
    assert review["findings"] == []
    assert review["verdict"] == (
        "ACCEPT_CORRECTED_GE_BEAM3_CURVED_P4_PRIVATE_REFERENCE_CORE_NO_P0_P1_P2"
    )
    rows = review["reviewed_inputs"]
    commits = {row["kind"]: row for row in rows if "commit" in row}
    assert commits["INITIAL_IMPLEMENTATION_COMMIT"] == {
        "commit": INITIAL,
        "kind": "INITIAL_IMPLEMENTATION_COMMIT",
        "parent": PREREGISTRATION,
        "subject": "feat: add private GE Beam3 curved P4 reference core",
        "tree": "4cb6fc861edf721fcb246fb521a64b24aa58eeab",
    }
    assert commits["CORRECTIVE_IMPLEMENTATION_COMMIT"] == {
        "commit": CORRECTION,
        "kind": "CORRECTIVE_IMPLEMENTATION_COMMIT",
        "parent": INITIAL,
        "subject": "fix: align GE Beam3 curved frame admission",
        "tree": "2235c5f868384f0122ec5eda766cdf01c0923a1f",
    }
    assert set(_git("diff-tree", "--no-commit-id", "--name-only", "-r", INITIAL).splitlines()) == {
        "src/anyfesolver/ge_beam3_curved_reference.py",
        "tests/test_ge_beam3_curved_p4_reference.py",
    }
    assert set(_git("diff-tree", "--no-commit-id", "--name-only", "-r", CORRECTION).splitlines()) == {
        "src/anyfesolver/ge_beam3_curved_reference.py",
        "tests/test_ge_beam3_curved_p4_reference.py",
    }


def test_reviewed_blob_hashes_and_straight_nonintrusion_are_exact() -> None:
    _raw, review = _canonical(REVIEW)
    for row in review["reviewed_inputs"]:
        if "path" not in row:
            continue
        commit = INITIAL if row["kind"].startswith("INITIAL_") else CORRECTION
        blob = _git_blob(commit, row["path"])
        assert _git("rev-parse", f"{commit}:{row['path']}") == row["git_blob"]
        assert (len(blob), _sha256(blob)) == (row["bytes"], row["sha256"])
    independence = review["reviewer_independence"]
    assert independence["accepted_straight_blob_audit"] == PROTECTED
    assert independence["implementation_authorship"] is False
    assert independence["corrective_patch_authorship"] is False
    assert independence["formal_scientific_execution_performed"] is False
    assert independence["uncommitted_rehearsal_harness_excluded_from_implementation_verdict"] is True
    for path, blob in PROTECTED.items():
        assert _git("rev-parse", f"{PREREGISTRATION}:{path}") == blob
        assert _git("rev-parse", f"{CORRECTION}:{path}") == blob


def test_review_commit_topology_is_exact_when_present() -> None:
    rows = _git("log", "--format=%H%x09%s", "--all").splitlines()
    matches = [row.split("\t", 1)[0] for row in rows if row.endswith("\t" + SUBJECT)]
    if not matches:
        assert _git("rev-parse", "HEAD") == CORRECTION
        return
    assert len(matches) == 1
    commit = matches[0]
    assert _git("rev-parse", f"{commit}^") == CORRECTION
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
    )
    assert changed == PATHS
