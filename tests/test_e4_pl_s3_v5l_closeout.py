from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
RESULT = REFERENCE / "e4_pl_s3_v5l_stage4b_result.json"
REVIEW = REFERENCE / "e4_pl_s3_v5l_stage4b_review.json"
STATUS = REFERENCE / "e4_pl_s3_v5l_stage4b_status.json"


def _canonical(path: Path):
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    return raw, value


def test_v5l_closeout_hash_chain_and_terminal() -> None:
    result_raw, result = _canonical(RESULT)
    review_raw, review = _canonical(REVIEW)
    _status_raw, status = _canonical(STATUS)
    assert hashlib.sha256(result_raw).hexdigest().upper() == status["result_sha256"]
    assert hashlib.sha256(review_raw).hexdigest().upper() == status["review_sha256"]
    assert review["findings"] == {"P0": [], "P1": []}
    assert status["terminal"] == result["terminal"] == "PROVISIONAL_GO_E4_PL_S3_V5L_STAGE4B_CLOSED_ONLY"


def test_v5l_exact_external_evidence_and_coverage() -> None:
    _raw, result = _canonical(RESULT)
    assert result["external_evidence"] == {
        "aggregate":{"bytes":325,"sha256":"D37CD6B62E50907625343EC0DCD2F3D5748CD234B7A41677D384917D82789C5D"},
        "common":{"bytes":754,"sha256":"F669C072CBA92AE087445472D4E62057BFA88D2407CDA19DD7B7986E0DA6B0E7"},
        "external_id":"s3-v5l-stage4b-20260902-d6869a3",
        "stderr_logs_empty":True,
        "stdout_logs_empty":True,
    }
    assert result["cycles"] == {"checker_replica_pairs":14,"commons_byte_identical":True,"proofs_byte_identical_by_worker":True,"worker_proofs":14}
    assert result["gate_status"] == {"buckling":"PASS_MEASURED_REGISTERED_SCOPE","mixed_performance":"PASS_MEASURED_REGISTERED_SCOPE","modal":"PASS_MEASURED_REGISTERED_SCOPE"}


def test_v5l_preserves_nonactivation_boundary() -> None:
    _raw, result = _canonical(RESULT)
    assert result["activation_authorized"] is False
    assert result["production_boundary"] == {"default_q4_formulation":"e4-pl","default_s3_formulation":"legacy-s3","q4_mechanics_unchanged":True}
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
