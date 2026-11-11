from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import anyfesolver


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"


def _canonical(name: str) -> tuple[bytes, dict]:
    raw = (REFERENCE / name).read_bytes()

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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def test_v6d_closeout_binds_the_immutable_authority_and_exact_extent() -> None:
    _result_raw, result = _canonical(
        "e4_pl_s3_v6d_v2d_activity_contact_batch_result.json"
    )
    authority = result["authority"]
    metadata = subprocess.run(
        ["git", "show", "-s", "--format=%H%n%T%n%P%n%s", authority["commit"]],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    assert metadata == [
        authority["commit"],
        authority["tree"],
        authority["parent"],
        authority["subject"],
    ]
    paths = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", authority["commit"]],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    contract_raw, contract = _canonical(
        "e4_pl_s3_v6d_v2d_activity_contact_batch_contract.json"
    )
    assert sorted(paths) == sorted(contract["exact_implementation_extent"])
    assert len(paths) == authority["exact_implementation_paths"] == 11
    assert authority["contract_sha256"] == _sha(contract_raw)


def test_v6d_result_has_two_clean_cycles_and_only_successor_authority() -> None:
    _raw, result = _canonical(
        "e4_pl_s3_v6d_v2d_activity_contact_batch_result.json"
    )
    assert result["cycles"] == {"count": 2, "tests_passed_per_cycle": 42}
    assert all(result["checks"].values())
    assert result["terminal"] == (
        "PROVISIONAL_GO_E4_PL_S3_V6D_FINAL_PARITY_REVIEW"
    )
    assert result["next_gate"] == "V6E_V2D_FINAL_PARITY_REVIEW"
    assert result["activation_authorized"] is False
    assert result["stage4a_scientific_rerun_authorized"] is False
    assert result["production_boundary"] == {
        "activation_authorized": False,
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }
    assert anyfesolver.DEFAULT_Q4_FORMULATION == "e4-pl"
    assert anyfesolver.DEFAULT_S3_FORMULATION == "legacy-s3"


def test_v6d_review_is_five_key_empty_p0_p1_and_status_binds_bytes() -> None:
    result_raw, result = _canonical(
        "e4_pl_s3_v6d_v2d_activity_contact_batch_result.json"
    )
    review_raw, review = _canonical(
        "e4_pl_s3_v6d_v2d_activity_contact_batch_review.json"
    )
    _status_raw, status = _canonical(
        "e4_pl_s3_v6d_v2d_activity_contact_batch_status.json"
    )
    assert set(review) == {
        "conclusions",
        "findings",
        "reviewed_inputs",
        "schema",
        "verdict",
    }
    assert review["findings"] == {"P0": [], "P1": []}
    assert all(review["conclusions"].values())
    assert status["result"] == {"bytes": len(result_raw), "sha256": _sha(result_raw)}
    assert status["review"] == {"bytes": len(review_raw), "sha256": _sha(review_raw)}
    assert status["authority_commit"] == result["authority"]["commit"]
    assert status["terminal"] == result["terminal"]
    assert status["activation_authorized"] is False
