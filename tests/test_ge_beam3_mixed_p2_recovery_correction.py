"""Static authority checks for the P2 manufactured-recovery correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "agent_plans" / "GE_BEAM3_MIXED_P2_RECOVERY_CORRECTION.md"
REFERENCE = ROOT / "docs" / "reference_cases"
CORRECTION = REFERENCE / "ge_beam3_mixed_p2_recovery_correction.json"
CONTRACT = REFERENCE / "ge_beam3_mixed_p2_recovery_correction_contract.json"
CHECKER = REFERENCE / "ge_beam3_mixed_p2_recovery_correction_reference.py"
REVIEW = REFERENCE / "ge_beam3_mixed_p2_recovery_correction_review.json"
BASE_COMMIT = "448ae38cad85365f0058fca8174fb5a12aa57bbc"
SUBJECT = "docs: correct GE Beam3 mixed P2 recovery authority"
PATHS = {
    "docs/agent_plans/GE_BEAM3_MIXED_P2_RECOVERY_CORRECTION.md",
    "docs/reference_cases/ge_beam3_mixed_p2_recovery_correction.json",
    "docs/reference_cases/ge_beam3_mixed_p2_recovery_correction_contract.json",
    "docs/reference_cases/ge_beam3_mixed_p2_recovery_correction_reference.py",
    "docs/reference_cases/ge_beam3_mixed_p2_recovery_correction_review.json",
    "tests/test_ge_beam3_mixed_p2_recovery_correction.py",
}


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    made: dict[str, object] = {}
    for key, value in pairs:
        if key in made:
            raise ValueError(f"duplicate key: {key}")
        made[key] = value
    return made


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite value: {value}")


def _canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique,
        parse_constant=_reject_constant,
    )
    assert isinstance(value, dict)
    expected = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    assert raw == expected
    return raw, value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True, encoding="utf-8").strip()


def test_correction_inputs_and_bindings_are_exact() -> None:
    plan_raw = PLAN.read_bytes()
    correction_raw, correction = _canonical(CORRECTION)
    contract_raw, contract = _canonical(CONTRACT)
    checker_raw = CHECKER.read_bytes()
    expected = {
        "docs/agent_plans/GE_BEAM3_MIXED_P2_RECOVERY_CORRECTION.md": plan_raw,
        "docs/reference_cases/ge_beam3_mixed_p2_recovery_correction.json": correction_raw,
        "docs/reference_cases/ge_beam3_mixed_p2_recovery_correction_reference.py": checker_raw,
    }
    assert contract["correction_inputs"] == {
        path: {"bytes": len(raw), "sha256": _sha(raw)} for path, raw in expected.items()
    }
    assert contract["base_preregistration"]["commit"] == BASE_COMMIT
    assert contract["mechanics_change_authorized"] is False
    assert contract["tolerance_change_authorized"] is False
    assert contract["scientific_execution_authorized"] is False
    assert correction["replacement"]["case_id"] == (
        "RECOVERY_MANUFACTURED_UNIFORM_SHEAR_Y_NATIVE_STATE"
    )
    assert len(contract_raw) > 0


def test_independent_rational_stationarity_checker_passes() -> None:
    output = subprocess.check_output(
        ("python", str(CHECKER), "--check-only"), cwd=ROOT, text=True, encoding="utf-8"
    )
    result = json.loads(output)
    assert result["status"] == "PASS"
    assert len(result["content_sha256"]) == 64
    source = CHECKER.read_text(encoding="utf-8")
    for forbidden in ("anyfesolver", "numpy", "scipy", "sympy", "ge_beam3_mixed_element"):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_review_and_stage_topology_are_fail_closed() -> None:
    _raw, review = _canonical(REVIEW)
    assert set(review) == {
        "findings",
        "reviewed_inputs",
        "reviewer_independence",
        "schema",
        "verdict",
    }
    assert review["findings"] == []
    assert review["verdict"] == "ACCEPT_GE_BEAM3_MIXED_P2_RECOVERY_CORRECTION_NO_P0_P1"
    reviewed = {row["path"]: row for row in review["reviewed_inputs"]}
    reviewable = PATHS - {"docs/reference_cases/ge_beam3_mixed_p2_recovery_correction_review.json"}
    assert set(reviewed) == reviewable
    for path in reviewable:
        raw = (ROOT / path).read_bytes()
        assert reviewed[path] == {"bytes": len(raw), "path": path, "sha256": _sha(raw)}

    rows = _git("log", "--format=%H%x09%s", "--all").splitlines()
    matches = [row.split("\t", 1)[0] for row in rows if row.endswith("\t" + SUBJECT)]
    if matches:
        assert len(matches) == 1
        assert _git("rev-parse", f"{matches[0]}^") == BASE_COMMIT
        changed = set(
            _git("diff-tree", "--no-commit-id", "--name-only", "-r", matches[0]).splitlines()
        )
        assert changed == PATHS

