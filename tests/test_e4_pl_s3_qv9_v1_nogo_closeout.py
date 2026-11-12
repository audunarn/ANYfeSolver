from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference_cases"
PLAN = ROOT / "docs" / "agent_plans" / "S3_E4_PL_V2_FORMULATION_PLAN.md"
ELEMENTS = ROOT / "src" / "anyfesolver" / "elements.py"


def _load_canonical(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_reject_duplicates)
    assert raw == (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return data, raw


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


def test_qv9_result_and_reference_audit_are_exact() -> None:
    result, result_raw = _load_canonical(REFERENCE / "e4_pl_s3_qv9_v1_nogo_result.json")
    audit, audit_raw = _load_canonical(REFERENCE / "e4_pl_s3_qv9_all_q4_reference_audit.json")

    assert len(result_raw) == 2314
    assert hashlib.sha256(result_raw).hexdigest().upper() == (
        "8A497BF1ABA4E048E23ACEAD6974D263D19A02F602AAC53DD07D12257EF34804"
    )
    assert result["terminal"] == "NO_GO_E4_PL_S3_V1_QUALIFICATION"
    assert result["v1_contradiction_count"] == 2
    assert result["default_activation_authorized"] is False
    assert result["v2_plan_preparation_authorized"] is True

    assert len(audit_raw) == 1655
    assert hashlib.sha256(audit_raw).hexdigest().upper() == (
        "7D0AF3B3C5ABF84DE820E62ECD62589B65B10A04371538D219A6E17C040FA7EA"
    )
    assert audit["terminal"] == "UNCLASSIFIED_E4_PL_S3_QV9_ALL_Q4_REFERENCE_CONTROL_CLOSED"
    assert audit["all_q4_reference_control"]["q4_reference_valid"] is True


def test_closeout_status_binds_current_defaults_and_v2_scope() -> None:
    status, _ = _load_canonical(REFERENCE / "e4_pl_s3_qv9_v1_nogo_status.json")
    assert status["result"]["sha256"] == "8A497BF1ABA4E048E23ACEAD6974D263D19A02F602AAC53DD07D12257EF34804"
    assert status["all_q4_audit"]["sha256"] == "7D0AF3B3C5ABF84DE820E62ECD62589B65B10A04371538D219A6E17C040FA7EA"
    assert status["default_activation_authorized"] is False
    assert status["v2_plan_preparation_authorized"] is True
    source = ELEMENTS.read_text(encoding="utf-8")
    q4_default = re.search(r'^DEFAULT_Q4_FORMULATION = "([^"]+)"$', source, re.MULTILINE)
    s3_default = re.search(r'^DEFAULT_S3_FORMULATION = "([^"]+)"$', source, re.MULTILINE)
    assert q4_default is not None
    assert s3_default is not None
    assert q4_default.group(1) == status["default_q4_formulation"] == "e4-pl"
    assert s3_default.group(1) == status["default_s3_formulation"] == "legacy-s3"

    plan = PLAN.read_text(encoding="utf-8")
    assert "E4_PL_QUALIFIED_S3_COMPANION_V2" in plan
    assert "Implement V2 as opt-in first" in plan
    assert "Success authorizes a separate reviewed default-activation commit" in plan
