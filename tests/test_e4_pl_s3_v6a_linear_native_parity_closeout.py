from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"


def _canonical(name: str) -> tuple[bytes, dict]:
    path = REFERENCE / name
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    assert raw == expected
    return raw, value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def test_v6a_closeout_binds_result_review_and_empty_findings() -> None:
    result_raw, result = _canonical(
        "e4_pl_s3_v6a_v2d_linear_native_parity_result.json"
    )
    review_raw, review = _canonical(
        "e4_pl_s3_v6a_v2d_linear_native_parity_review.json"
    )
    _status_raw, status = _canonical(
        "e4_pl_s3_v6a_v2d_linear_native_parity_status.json"
    )
    assert status["result"] == {"bytes": len(result_raw), "sha256": _sha(result_raw)}
    assert status["review"] == {"bytes": len(review_raw), "sha256": _sha(review_raw)}
    assert review["findings"] == {"P0": [], "P1": []}
    assert result["cycles"]["count"] == 2
    assert result["cycles"]["tests_passed_per_cycle"] == 66


def test_v6a_closeout_authorizes_only_the_native_state_successor_gate() -> None:
    _raw, result = _canonical(
        "e4_pl_s3_v6a_v2d_linear_native_parity_result.json"
    )
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V6A_STATE_IMPLEMENTATION"
    assert result["native_state_implementation_authorized"] is True
    assert result["activation_authorized"] is False
    assert result["next_gate"] == "V6B_V2D_NATIVE_STATE_AND_COROTATIONAL_PARITY"
    assert result["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }


def test_v6a_authority_commit_has_exact_six_path_extent() -> None:
    _raw, result = _canonical(
        "e4_pl_s3_v6a_v2d_linear_native_parity_result.json"
    )
    paths = subprocess.run(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            result["authority"]["commit"],
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    assert sorted(paths) == sorted(
        [
            "docs/reference_cases/e4_pl_s3_v6a_v2d_linear_native_parity_contract.json",
            "src/anyfesolver/__init__.py",
            "src/anyfesolver/e4_pl_s3_v2d_element.py",
            "src/anyfesolver/elements.py",
            "tests/test_e4_pl_s3_v2d_linear_native_parity.py",
            "tests/test_e4_pl_s3_v6a_linear_native_parity_contract.py",
        ]
    )
