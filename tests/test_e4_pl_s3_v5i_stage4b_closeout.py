from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
RESULT = REFERENCE / "e4_pl_s3_v5i_stage4b_result.json"
REVIEW = REFERENCE / "e4_pl_s3_v5i_stage4b_review.json"
STATUS = REFERENCE / "e4_pl_s3_v5i_stage4b_status.json"


def _canonical(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    return value


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest().upper(),
    }


def test_v5i_closeout_is_canonical_and_hash_chained() -> None:
    result = _canonical(RESULT)
    review = _canonical(REVIEW)
    status = _canonical(STATUS)
    assert status["result"] == _binding(RESULT)
    assert status["review"] == _binding(REVIEW)
    assert review["reviewed_inputs"]["result_sha256"] == _binding(RESULT)["sha256"]
    assert result["external_evidence"]["aggregate"] == status["external_aggregate"]
    assert result["cycles"]["canonical_common_byte_identical"] is True
    assert result["cycles"]["count"] == 2


def test_v5i_records_genuine_eigen_precedence_and_no_activation() -> None:
    result = _canonical(RESULT)
    review = _canonical(REVIEW)
    status = _canonical(STATUS)
    assert result["terminal"] == "NO_GO_E4_PL_S3_V5I_MIXED_EIGEN"
    assert result["gate_status"] == {
        "buckling": "FAIL_MEASURED_CONTRADICTION",
        "mixed_performance": "FAIL_MEASURED_CONTRADICTION",
        "modal": "PASS_MEASURED_REGISTERED_SCOPE",
    }
    assert float.fromhex(
        result["metrics"]["buckling"]["25"]["minimum_clustered_mac_hex"]
    ) < 0.95
    assert review["verdict"] == "ACCEPT_S3_V5I_DETERMINISTIC_NO_GO_NO_P0_P1"
    assert review["findings"] == {"P0": [], "P1": []}
    assert status["repair_diagnosis_authorized"] is True
    assert all(
        record["activation_authorized"] is False
        for record in (result, status)
    )


def test_v5i_preserves_q4_and_legacy_s3_defaults() -> None:
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
    assert _canonical(RESULT)["production_boundary"] == {
        "default_q4_formulation": "e4-pl",
        "default_s3_formulation": "legacy-s3",
        "q4_mechanics_unchanged": True,
    }
