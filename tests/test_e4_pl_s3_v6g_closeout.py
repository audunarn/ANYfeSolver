from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anyfesolver.elements import DEFAULT_Q4_FORMULATION, DEFAULT_S3_FORMULATION


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
PREFIX = "e4_pl_s3_v6g_recovery_current_eigen"


def _canonical(name: str) -> tuple[bytes, dict[str, object]]:
    raw = (REFERENCE / f"{PREFIX}_{name}.json").read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return raw, value


def test_v6g_closeout_is_canonical_and_hash_bound() -> None:
    result_raw, result = _canonical("result")
    review_raw, review = _canonical("review")
    _status_raw, status = _canonical("status")
    assert status["result"] == {
        "bytes": len(result_raw),
        "sha256": hashlib.sha256(result_raw).hexdigest().upper(),
    }
    assert status["review"] == {
        "bytes": len(review_raw),
        "sha256": hashlib.sha256(review_raw).hexdigest().upper(),
    }
    assert result["authority"]["commit"] == (
        "c6e596c64321225e36aaff02b98ddb8fa81b6620"
    )
    assert result["authority"]["exact_implementation_paths"] == 10
    assert review["findings"] == []
    assert review["verdict"] == "ACCEPT_E4_PL_S3_V6G_IMPLEMENTATION_NO_P0_P1"


def test_v6g_passes_only_to_final_review_with_defaults_frozen() -> None:
    _raw, result = _canonical("result")
    _status_raw, status = _canonical("status")
    assert all(result["checks"].values())
    assert result["cycles"] == {
        "canonical_readiness_outputs_byte_identical": True,
        "count": 2,
        "focused_tests_passed_per_cycle": 23,
    }
    assert result["frozen_commit_review"] == {"tests_passed": 97}
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V6G_FINAL_REVIEW"
    assert status["terminal"] == result["terminal"]
    assert result["activation_authorized"] is False
    assert result["stage4a_scientific_rerun_authorized"] is False
    assert status["activation_authorized"] is False
    assert status["stage4a_scientific_rerun_authorized"] is False
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"


def test_v6g_preserves_v6f_historical_terminal() -> None:
    historical = json.loads(
        (REFERENCE / "e4_pl_s3_v6f_final_parity_audit_status.json").read_bytes()
    )
    assert historical["terminal"] == (
        "UNCLASSIFIED_E4_PL_S3_V6F_REMAINING_PRODUCTION_PARITY"
    )
    assert historical["activation_authorized"] is False
    assert historical["stage4a_scientific_rerun_authorized"] is False
