from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
RESULT = REFERENCE / "e4_pl_s3_v6u_performance_result.json"
REVIEW = REFERENCE / "e4_pl_s3_v6u_performance_result_review.json"
STATUS = REFERENCE / "e4_pl_s3_v6u_performance_status.json"
GO = "PROVISIONAL_GO_E4_PL_S3_V6U_STAGE4B_CLOSED_ONLY"


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate key: {key}")
        value[key] = item
    return value


def _load(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    value = json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert isinstance(value, dict)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")
    return raw, value


def test_result_and_review_are_canonical_and_hash_bound() -> None:
    result_raw, result = _load(RESULT)
    review_raw, review = _load(REVIEW)
    _status_raw, status = _load(STATUS)
    assert len(result_raw) == status["result"]["bytes"]
    assert hashlib.sha256(result_raw).hexdigest().upper() == status["result"]["sha256"]
    assert len(review_raw) == status["review"]["bytes"]
    assert hashlib.sha256(review_raw).hexdigest().upper() == status["review"]["sha256"]
    assert review["findings"] == []
    assert review["reviewed_inputs"]["result_sha256"] == status["result"]["sha256"]


def test_success_closes_only_stage4b_and_preserves_predecessors() -> None:
    _raw, result = _load(RESULT)
    assert result["terminal"] == GO
    assert result["activation_authorized"] is False
    assert result["predecessors"]["v6s_reclassified"] is False
    assert result["cycles"] == {
        "checker_replica_pairs": 6,
        "checker_replicas_byte_identical": True,
        "common_byte_identical": True,
        "count": 2,
        "proofs_byte_identical_by_worker": True,
        "workers_per_cycle": 3,
    }
    assert set(result["gate_status"].values()) == {"PASS_MEASURED_REGISTERED_SCOPE"}
    assert result["next_gate"] == "V6V_PACKAGING_RESTART_BATCHING_AND_ACTIVATION_GAP_AUDIT"


def test_q4_and_s3_defaults_remain_unchanged() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
