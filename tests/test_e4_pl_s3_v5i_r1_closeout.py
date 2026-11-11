from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
RESULT = REFERENCE / "e4_pl_s3_v5i_r1_diagnosis_result.json"
REVIEW = REFERENCE / "e4_pl_s3_v5i_r1_diagnosis_review.json"
STATUS = REFERENCE / "e4_pl_s3_v5i_r1_diagnosis_status.json"


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    return value


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest().upper()}


def test_v5i_r1_closeout_is_canonical_and_hash_chained() -> None:
    result, review, status = _load(RESULT), _load(REVIEW), _load(STATUS)
    assert status["result"] == _binding(RESULT)
    assert status["review"] == _binding(REVIEW)
    assert review["reviewed_inputs"]["result_sha256"] == _binding(RESULT)["sha256"]
    assert result["external_evidence"]["aggregate"] == status["external_aggregate"]
    assert result["cycles"][0]["proof_sha256"] == result["cycles"][1]["proof_sha256"]


def test_v5i_r1_diagnosis_is_narrow_and_nonactivating() -> None:
    result, review, status = _load(RESULT), _load(REVIEW), _load(STATUS)
    assert result["terminal"] == "DIAGNOSED_E4_PL_S3_V5I_R1_PAIR_SUBSPACE_AND_ASSEMBLY_ROUTE_GAP"
    assert float.fromhex(result["buckling"]["individual_mode_five_mac_hex"]) < 0.95
    assert float.fromhex(result["buckling"]["pair_subspace_mac_hex"]) >= 0.95
    assert result["assembly"]["v2c_scalar_route_gap"] is True
    assert result["predecessor_terminal_preserved"] == "NO_GO_E4_PL_S3_V5I_MIXED_EIGEN"
    assert review["findings"] == {"P0": [], "P1": []}
    assert review["conclusions"]["mechanics_change_authorized"] is False
    assert status["activation_authorized"] is False
    source = (ROOT / "src/anyfesolver/elements.py").read_text(encoding="utf-8")
    assert 'DEFAULT_Q4_FORMULATION = "e4-pl"' in source
    assert 'DEFAULT_S3_FORMULATION = "legacy-s3"' in source
