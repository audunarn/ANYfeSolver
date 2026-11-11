from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
RESULT = REFERENCE / "e4_pl_s3_v6v_activation_gap_result.json"
REVIEW = REFERENCE / "e4_pl_s3_v6v_activation_gap_result_review.json"
STATUS = REFERENCE / "e4_pl_s3_v6v_activation_gap_status.json"
GO = "PROVISIONAL_GO_E4_PL_S3_V6V_FINAL_QUALIFICATION_PREPARATION"


def _load(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert isinstance(value, dict)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    return raw, value


def test_v6v_result_review_and_status_are_hash_bound() -> None:
    result_raw, result = _load(RESULT)
    review_raw, review = _load(REVIEW)
    _status_raw, status = _load(STATUS)
    assert len(result_raw) == status["result"]["bytes"]
    assert hashlib.sha256(result_raw).hexdigest().upper() == status["result"]["sha256"]
    assert len(review_raw) == status["review"]["bytes"]
    assert hashlib.sha256(review_raw).hexdigest().upper() == status["review"]["sha256"]
    assert review["findings"] == []
    assert review["reviewed_inputs"]["result_sha256"] == status["result"]["sha256"]


def test_v6v_closes_every_registered_audit_gate_without_activation() -> None:
    _raw, result = _load(RESULT)
    assert result["terminal"] == GO
    assert result["activation_authorized"] is False
    assert result["checks"]["focused_test_count"] == 42
    assert all(value is True for key, value in result["checks"].items() if key != "focused_test_count")
    assert result["package"]["anyfesolver"]["import_from_isolated_target"] is True
    assert result["package"]["anyfesolver"]["round_trip_exact"] is True
    assert result["next_gate"] == "V6W_FINAL_QUALIFICATION_EVIDENCE_COMPOSITION"


def test_v6v_preserves_incidents_q4_and_s3_defaults() -> None:
    _raw, result = _load(RESULT)
    assert result["predecessor_incidents"] == {
        "v6p_reclassified": False,
        "v6q_reclassified": False,
        "v6r_is_accepted_spatial_successor": True,
        "v6v_failed_attempt_preserved": True,
    }
    assert result["production_boundary"]["default_q4_formulation"] == "e4-pl"
    assert result["production_boundary"]["default_s3_formulation"] == "legacy-s3"
    assert result["production_boundary"]["s3_activation_authorized"] is False
