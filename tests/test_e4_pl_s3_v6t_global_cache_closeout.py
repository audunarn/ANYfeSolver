from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"


def _load(name: str):
    path = REFERENCE / name
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    return raw, value


def test_v6t_authorizes_only_performance_successor() -> None:
    _raw, result = _load("e4_pl_s3_v6t_global_cache_result.json")
    assert result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V6T_PERFORMANCE_SUCCESSOR"
    assert result["next_gate"] == "V6U_PERFORMANCE_ONLY_STAGE4B_SUCCESSOR"
    assert result["implementation"]["mechanics_changed"] is False
    assert result["predecessor"] == {
        "reclassified": False,
        "terminal": "NO_GO_E4_PL_S3_V6S_MIXED_PERFORMANCE",
    }
    assert result["activation_authorized"] is False


def test_v6t_status_binds_review_and_result() -> None:
    result_raw, _result = _load("e4_pl_s3_v6t_global_cache_result.json")
    review_raw, review = _load("e4_pl_s3_v6t_global_cache_result_review.json")
    _status_raw, status = _load("e4_pl_s3_v6t_global_cache_status.json")
    result_sha = hashlib.sha256(result_raw).hexdigest().upper()
    review_sha = hashlib.sha256(review_raw).hexdigest().upper()
    assert review["reviewed_inputs"]["result_sha256"] == result_sha
    assert status["result"] == {"bytes": len(result_raw), "sha256": result_sha}
    assert status["review"] == {"bytes": len(review_raw), "sha256": review_sha}


def test_defaults_remain_frozen() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
