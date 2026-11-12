from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anyfesolver.elements import DEFAULT_Q4_FORMULATION, DEFAULT_S3_FORMULATION


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/reference_cases"
PREFIX = "e4_pl_s3_v6h_stage4a_preparation"


def _canonical(name: str) -> tuple[bytes, dict[str, object]]:
    raw = (REFERENCE / f"{PREFIX}_{name}.json").read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return raw, value


def test_v6h_closeout_is_canonical_and_hash_bound() -> None:
    authority_raw, authority = _canonical("authority")
    review_raw, review = _canonical("review")
    _status_raw, status = _canonical("status")
    assert status["authority"] == {
        "bytes": len(authority_raw),
        "sha256": hashlib.sha256(authority_raw).hexdigest().upper(),
    }
    assert status["review"] == {
        "bytes": len(review_raw),
        "sha256": hashlib.sha256(review_raw).hexdigest().upper(),
    }
    assert authority["authority_commit"]["commit"] == (
        "704a2e514f172ab9ec860bc34267b9a798876fba"
    )
    assert authority["authority_commit"]["paths"] == [
        "docs/reference_cases/e4_pl_s3_v6h_stage4a_adapter.py",
        "docs/reference_cases/e4_pl_s3_v6h_stage4a_authority.py",
        "docs/reference_cases/e4_pl_s3_v6h_stage4a_authority_contract.json",
        "tests/test_e4_pl_s3_v6h_stage4a_authority.py",
    ]
    assert review["findings"] == []
    assert review["verdict"] == "ACCEPT_E4_PL_S3_V6H_PREPARATION_NO_P0_P1"


def test_v6h_authorizes_preparation_only() -> None:
    _authority_raw, authority = _canonical("authority")
    _status_raw, status = _canonical("status")
    assert authority["stage4a_preparation_authorized"] is True
    assert status["stage4a_preparation_authorized"] is True
    assert authority["stage4a_execution_authorized"] is False
    assert status["stage4a_execution_authorized"] is False
    assert authority["activation_authorized"] is False
    assert status["activation_authorized"] is False
    assert authority["terminal"] == (
        "PROVISIONAL_GO_E4_PL_S3_V6H_STAGE4A_PREPARATION"
    )
    assert status["terminal"] == authority["terminal"]
    assert authority["scientific_schema_compatibility"]["thresholds_changed"] is False
    assert authority["scientific_schema_compatibility"][
        "topology_or_case_coverage_changed"
    ] is False
    assert DEFAULT_Q4_FORMULATION == "e4-pl"
    assert DEFAULT_S3_FORMULATION == "legacy-s3"


def test_v6h_adapter_does_not_replace_the_public_factory() -> None:
    source = (
        REFERENCE / "e4_pl_s3_v6h_stage4a_adapter.py"
    ).read_text(encoding="utf-8")
    assert "element_factory.create_shell_element =" not in source
    assert 'RUNTIME_SELECTOR = "e4-pl-s3-v2d"' in source
    assert 'SCIENTIFIC_SELECTOR_SLOT = "e4-pl-s3-v2"' in source
