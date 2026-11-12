from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
RESULT = REFERENCE / "e4_pl_s3_v5k_repair_result.json"
REVIEW = REFERENCE / "e4_pl_s3_v5k_repair_review.json"
STATUS = REFERENCE / "e4_pl_s3_v5k_repair_status.json"


def _canonical(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    return raw, value


def test_v5k_closeout_is_canonical_and_hash_bound() -> None:
    result_raw, result = _canonical(RESULT)
    review_raw, review = _canonical(REVIEW)
    _status_raw, status = _canonical(STATUS)
    assert hashlib.sha256(result_raw).hexdigest().upper() == status["result_sha256"]
    assert hashlib.sha256(review_raw).hexdigest().upper() == status["review_sha256"]
    assert review["findings"] == {"P0": [], "P1": []}
    assert status["terminal"] == result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V5K_STAGE4B_RERUN"


def test_v5k_external_evidence_and_coverage_are_exact() -> None:
    _raw, result = _canonical(RESULT)
    assert result["external_evidence"] == {
        "aggregate": {
            "bytes": 319,
            "sha256": "193FF51BE5699285FD78241ADBB84216649526BAA90562B05E3E9299B9524D9C",
        },
        "common": {
            "bytes": 823,
            "sha256": "1869AA7F7DFCE1DA022D13F2F1893566C260EC5759EE02CEF8DF8752B7521429",
        },
        "external_id": "s3-v5k-repair-20260902-e792a26",
        "stderr_logs_empty": True,
        "stdout_logs_empty": True,
    }
    assert result["cycles"] == {
        "checker_replica_pairs": 14,
        "commons_byte_identical": True,
        "proofs_byte_identical_by_worker": True,
        "worker_proofs": 14,
    }


def test_v5k_does_not_activate_s3_or_change_q4() -> None:
    _raw, result = _canonical(RESULT)
    assert result["activation_authorized"] is False
    assert result["production_boundary"] == {
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
