from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs/reference_cases/s4_stage_m_candidate_a_discretization_cases.json"
ORACLE = ROOT / "docs/reference_cases/s4_stage_m_candidate_a_discretization_oracle.py"
CONTRACT = ROOT / "docs/reference_cases/s4_stage_m_candidate_a_discretization_contract.json"
OUTPUT = ROOT / "docs/reference_cases/s4_stage_m_candidate_a_discretization_output.json"
DERIVATION = ROOT / "docs/S4_STAGE_M_CANDIDATE_A_DISCRETIZATION_DERIVATION.md"

CASES_SHA256 = "BB29F6AE7AE53E961C992BBC2EA764D50B73F935C9B5F9C21C1E21462DCC3E9C"
ORACLE_SHA256 = "3240C3C60754B44C06F790F74B553ACF2DF88070E3618082287C0D9DE175992A"
CONTRACT_SHA256 = "8861943D1339373FB36448EA376E75D4CBAB64DE1A8450D6B547A235AA62844C"
OUTPUT_SHA256 = "6904490675315E7F2E17B1AA848837B56FC3E7633B4643B8B6C91989ED8E2059"
DERIVATION_SHA256 = "A8E012E69E3FCFCDAF94E73C97C413B4715CE51A6A1DA8FDA3A50C0467580BF8"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _strict(path: Path, expected: str, *, canonical_required: bool = True) -> tuple[dict, bytes]:
    data = path.read_bytes()
    assert _sha(data) == expected
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.endswith(b"\n")
    value = json.loads(data.decode("utf-8"))
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    if canonical_required:
        assert canonical == data
    return value, data


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    local = ROOT / ".s4_stage_m_mpmath_clean"
    if local.is_dir():
        prior = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(local) if not prior else str(local) + os.pathsep + prior
    return env


def _run(*arguments: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(ORACLE), *arguments],
        cwd=ROOT,
        env=_environment() if environment is None else environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )


def test_candidate_a_static_contract_and_fail_closed_output() -> None:
    cases, _ = _strict(CASES, CASES_SHA256, canonical_required=False)
    contract, contract_bytes = _strict(CONTRACT, CONTRACT_SHA256)
    output, output_bytes = _strict(OUTPUT, OUTPUT_SHA256)
    derivation_bytes = DERIVATION.read_bytes()
    assert _sha(derivation_bytes) == DERIVATION_SHA256
    assert _sha(ORACLE.read_bytes()) == ORACLE_SHA256

    assert cases["fixed_lambda_checkpoint"]["lambda"] == "1"
    assert cases["fixed_lambda_checkpoint"]["thickness_is_separate"] is True
    assert [row["id"] for row in cases["candidate_pairs"]] == [
        "candidate_a.d4.span_r_s",
        "candidate_a.d4.span_1_rs",
    ]
    assert contract["counts"] == {"base_rows": 174, "pair_rows": 348, "pairs": 2}
    assert [row["row_count"] for row in contract["coverage_binding"]["pair_ledgers"]] == [174, 174]
    assert output["contract_sha256"] == CONTRACT_SHA256
    assert output["fixed_lambda_checkpoint"]["terminal"] == "UNCLASSIFIED_CANDIDATE_A_FIXED_LAMBDA_SPECIALIZATION"
    assert output["rotation_checkpoint"]["terminal"] == "UNCLASSIFIED_CANDIDATE_A_ROTATION_MAPPING"
    assert output["pair_terminal"] == "NOT_RUN_PREREQUISITE_UNCLOSED"
    assert output["candidate_terminal"] == "BLOCKED_CANDIDATE_A_DISCRETIZATION_UNREGISTERED"
    assert output["overall_stage_m_terminal"] == "BLOCKED_CANDIDATE_A_DISCRETIZATION_UNREGISTERED"
    assert output["candidate_b_preserved"]["terminal"] == "NO_GO_CANDIDATE_B"
    assert output["exclusions"] == {
        "candidate_b_rerun": False,
        "cleanup": False,
        "penalty_or_ctc": False,
        "production_activation": False,
        "production_source_edit": False,
        "publication": False,
        "selector": False,
    }
    assert [row["decimal_digits"] for row in output["precision_records"]] == [80, 160, 320]
    assert all(row["positive_polar_pass"] and row["objectivity_pass"] and row["guard_pass"] for row in output["precision_records"])
    assert len(contract_bytes) > 0 and len(output_bytes) > 0


def test_candidate_a_oracle_is_independent_and_deterministic() -> None:
    tree = ast.parse(ORACLE.read_text(encoding="utf-8"), filename=str(ORACLE))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "anyfesolver" or name.startswith("anyfesolver.") for name in imported)

    emitted = _run("--emit-contract")
    assert emitted.returncode == 0, emitted.stderr.decode()
    assert emitted.stderr == b""
    assert emitted.stdout == CONTRACT.read_bytes()

    first = _run("--run", "--contract", str(CONTRACT), "--contract-sha256", CONTRACT_SHA256)
    second = _run("--run", "--contract", str(CONTRACT), "--contract-sha256", CONTRACT_SHA256)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout == OUTPUT.read_bytes()


def test_candidate_a_wrong_contract_hash_blocks_before_science_import() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = _run(
        "--run",
        "--contract",
        str(CONTRACT),
        "--contract-sha256",
        "0" * 64,
        environment=env,
    )
    assert result.returncode == 2
    record = json.loads(result.stdout)
    assert record["status"] == "blocked"
    assert record["terminal"] == "BLOCKED_INPUT_IDENTITY"
