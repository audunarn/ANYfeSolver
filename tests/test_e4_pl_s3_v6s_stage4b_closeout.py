from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"


def _canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    return raw, value


def test_v6s_result_preserves_the_measured_no_go() -> None:
    _raw, result = _canonical(REFERENCE / "e4_pl_s3_v6s_stage4b_result.json")
    assert result["terminal"] == "NO_GO_E4_PL_S3_V6S_MIXED_PERFORMANCE"
    assert result["gate_status"] == {
        "buckling": "PASS_MEASURED_REGISTERED_SCOPE",
        "mixed_performance": "FAIL_MEASURED_CONTRADICTION",
        "modal": "PASS_MEASURED_REGISTERED_SCOPE",
    }
    assert result["cycles"]["common_byte_identical"] is True
    assert result["replicas_byte_identical"] is True
    assert result["activation_authorized"] is False


def test_v6s_status_binds_result_and_review() -> None:
    result_raw, _result = _canonical(REFERENCE / "e4_pl_s3_v6s_stage4b_result.json")
    review_raw, review = _canonical(REFERENCE / "e4_pl_s3_v6s_stage4b_result_review.json")
    _status_raw, status = _canonical(REFERENCE / "e4_pl_s3_v6s_stage4b_status.json")
    result_sha = hashlib.sha256(result_raw).hexdigest().upper()
    review_sha = hashlib.sha256(review_raw).hexdigest().upper()
    assert review["reviewed_inputs"]["result_sha256"] == result_sha
    assert status["result"] == {"bytes": len(result_raw), "sha256": result_sha}
    assert status["review"] == {"bytes": len(review_raw), "sha256": review_sha}
    assert status["stage4b_closed"] is False


def test_defaults_and_q4_boundary_remain_unchanged() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
