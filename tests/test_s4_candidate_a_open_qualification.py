from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/agent_plans/S4_CANDIDATE_A_OPEN_QUALIFICATION_PLAN.md"
BASELINE = ROOT / "docs/reference_cases/s4_candidate_a_open_baseline.json"
SOURCE_REGISTRY = ROOT / "docs/reference_cases/s4_candidate_a_open_source_registry.json"
ENVIRONMENT = ROOT / "docs/reference_cases/s4_candidate_a_open_environment.json"
INVENTORY = ROOT / "docs/reference_cases/s4_candidate_a_open_test_inventory.json"
A1_CERTIFICATE = ROOT / "docs/reference_cases/s4_candidate_a_open_a1_certificate.json"
A2_CERTIFICATE = ROOT / "docs/reference_cases/s4_candidate_a_open_a2_certificate.json"
ORACLE = ROOT / "docs/reference_cases/s4_candidate_a_open_oracle.py"
CONTRACT = ROOT / "docs/reference_cases/s4_candidate_a_open_contract.json"
OUTPUT = ROOT / "docs/reference_cases/s4_candidate_a_open_output.json"
REPORT = ROOT / "docs/S4_CANDIDATE_A_OPEN_QUALIFICATION_REPORT.md"

HASHES = {
    PLAN: "630EEEFD846CCFC4DE5B61C5530F8E76F5ACD33A6014A78135F4A36D8FE90999",
    BASELINE: "C3BB5E4AB79C9B6278B6E39F642AE3F99DA001ABF5DE0D1E01274FBC0187199A",
    SOURCE_REGISTRY: "8EF2E09B76046A4070A7A2BCDAC52EC16A25D50C7557F789335AC6173E5A6986",
    ENVIRONMENT: "1348DF6CE0DBC19BE84A0A28243820EAFDD7EA361AB78A5F586EBC98391D28F5",
    INVENTORY: "4F016F85EFABFC459823BC3B290F5E2AB2143677AE8765246017D19CC2A4FC11",
    A1_CERTIFICATE: "2198458DCDC7EFB4684B5CC59ADAF6E9A0EECF381951CBDD21B286D6DB11097C",
    A2_CERTIFICATE: "68691E4F1F23E23ED7DF00C4210436BDD1A730ADB58ED3E755E15CC01ECC5F3B",
    ORACLE: "7C3CE3821CE51FBA4689E437B4AB4AFD4CF1A3AA6638D1A800018E6B3FFCA682",
    CONTRACT: "7A9334964FB9A248EA1D44653C04E1F71731B3E605B053434CE7F252CCAB0D92",
    OUTPUT: "C42911E11BB1F1FA091F29FD0E3F5A3617310EF5F06C686E57C013171242B63C",
    REPORT: "C25BD9602D5189FAD4603FE6ADEE4AC3D1FF172D28614AA37BAF9F2E6A8F7246",
}

A1_REPORT_CANONICAL_LF_SHA256 = "BB6F10B29C404D339D48C4C301F23750CAC981F8F65C458BE5F2D576066F55D4"
A2_REPORT_CANONICAL_LF_SHA256 = "912583FAC4C4F5D0E13B1B7BB9E09699A605D4821CF0650254C78DA3F9F4CC76"
CONTRACT_SHA256 = HASHES[CONTRACT]
NODE_LIST_SHA256 = "7D71339F0621328AF54BC4BDFC04E3C7082EDA333B58E100C4C9550F0E9C85D9"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _canonical_lf(raw: bytes) -> bytes:
    assert not raw.startswith(b"\xef\xbb\xbf")
    without_crlf = raw.replace(b"\r\n", b"")
    assert b"\r" not in without_crlf
    if b"\r\n" in raw:
        assert b"\n" not in without_crlf
    return raw.replace(b"\r\n", b"\n")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def _json(path: Path, *, canonical: bool) -> dict[str, Any]:
    raw = path.read_bytes()
    assert _sha(raw) == HASHES[path]
    assert b"\r" not in raw and raw.endswith(b"\n")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    if canonical:
        expected = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        assert raw == expected
    return value


def _fractions(values: list[str]) -> list[Fraction]:
    return [Fraction(value) for value in values]


def _rank(matrix: list[list[Fraction]]) -> int:
    work = [row[:] for row in matrix]
    row = 0
    for column in range(len(work[0])):
        pivot = next((i for i in range(row, len(work)) if work[i][column]), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        scale = work[row][column]
        work[row] = [value / scale for value in work[row]]
        for index in range(len(work)):
            if index == row or not work[index][column]:
                continue
            scale = work[index][column]
            work[index] = [
                work[index][entry] - scale * work[row][entry]
                for entry in range(len(work[index]))
            ]
        row += 1
        if row == len(work):
            break
    return row


def _run_oracle(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(ORACLE), *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )


def test_open_packet_is_content_addressed_and_fail_closed() -> None:
    for path, expected in HASHES.items():
        assert _sha(path.read_bytes()) == expected
    assert _sha(_canonical_lf((ROOT / "docs/S4_CANDIDATE_A_OPEN_A1_RANK_CERTIFICATE.md").read_bytes())) == A1_REPORT_CANONICAL_LF_SHA256
    assert _sha(_canonical_lf((ROOT / "docs/S4_CANDIDATE_A_OPEN_A2_INF_SUP_CERTIFICATE.md").read_bytes())) == A2_REPORT_CANONICAL_LF_SHA256

    baseline = _json(BASELINE, canonical=False)
    source = _json(SOURCE_REGISTRY, canonical=False)
    environment = _json(ENVIRONMENT, canonical=True)
    inventory = _json(INVENTORY, canonical=True)
    contract = _json(CONTRACT, canonical=True)
    output = _json(OUTPUT, canonical=True)

    assert baseline["git"]["baseline_commit"] == "148ccb45ba79266d48dae1a84c4c500bdc1b4d85"
    assert baseline["git"]["baseline_tree"] == "0a0809b2111c07098058fd43891729c6f9266b06"
    assert source["offline_reproduction"] is True
    assert environment["oracle_runtime"]["dependencies"] == "python_standard_library_only"
    assert inventory["collection"]["count"] == len(inventory["collection"]["node_ids"]) == 64
    node_bytes = ("\n".join(inventory["collection"]["node_ids"]) + "\n").encode()
    assert _sha(node_bytes) == NODE_LIST_SHA256

    assert contract["coverage"]["base_rows"] == 174
    assert [row["row_count"] for row in contract["coverage"]["pair_ledgers"]] == [174, 174]
    assert contract["quadrature"] == {
        "finite_constraint_future_primary": "surface_3x3",
        "finite_constraint_future_sensitivity": "surface_4x4",
        "flat_polynomial_reproduction": ["symbolic", "tensor_2x2_gauss"],
    }
    assert contract["allowed_extent"]["production_paths"] == []
    assert output["status"] == "complete"
    assert output["pair_terminal"] == "NO_GO_CANDIDATE_A_DISCRETE_PAIR"
    assert output["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert [row["terminal"] for row in output["candidate_results"]] == [
        "PROVEN_FAIL_CANDIDATE_A1_FLAT_RANK",
        "PROVEN_FAIL_CANDIDATE_A2_INF_SUP",
    ]
    assert output["preserved"]["candidate_b"] == {
        "rerun": False,
        "terminal": "NO_GO_CANDIDATE_B",
    }
    assert output["exclusions"] == contract["exclusions"]
    assert all(value is False for value in output["exclusions"].values())


def test_open_oracle_recomputes_both_exact_certificates_independently() -> None:
    a1 = _json(A1_CERTIFICATE, canonical=True)
    rows = [
        _fractions(a1["exact_constraint_rows_raw"][mode])
        for mode in a1["candidate"]["basis_order"]
    ]
    witness_ids = a1["premises"]["accepted_B_kernel_witness_ids"]
    witnesses = [_fractions(a1["witnesses"][name]) for name in witness_ids]
    witness_matrix = [[column[row] for column in witnesses] for row in range(24)]
    assert _rank(rows) == 2
    assert _rank(witness_matrix) == 8
    assert all(
        [sum(row[i] * vector[i] for i in range(24)) for row in rows]
        == [Fraction(0), Fraction(0)]
        for vector in witnesses
    )
    assert 24 - 8 == 16
    assert (24 - 2) - 8 == 14
    assert a1["result"] == a1["result"] | {
        "candidate_terminal": "PROVEN_FAIL_CANDIDATE_A1_FLAT_RANK",
        "pair_gate_result": "PROVEN_FAIL",
    }

    a2 = _json(A2_CERTIFICATE, canonical=True)
    counterexample = a2["counterexample"]
    rs = _fractions(counterexample["local_rows"]["normalized_rs_full"])
    centre_rows = []
    for element in counterexample["topology"]["elements"]:
        index = element["nodes"].index("n11")
        centre_rows.append(rs[6 * index : 6 * (index + 1)])
    assert [sum(row[column] for row in centre_rows) for column in range(6)] == [Fraction(0)] * 6
    assert Fraction(3, 2) ** 2 * Fraction(2, 3) ** 2 * 4 == 4
    assert a2["terminal"] == {
        "beta": "0/1",
        "scope": "full_unquotiented_discontinuous_element_local_multiplier_space",
        "terminal": "PROVEN_FAIL_CANDIDATE_A2_INF_SUP",
    }

    tree = ast.parse(ORACLE.read_text(encoding="utf-8"), filename=str(ORACLE))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = ("anyfesolver", "numpy", "scipy", "mpmath")
    assert not any(name == item or name.startswith(item + ".") for name in imported for item in forbidden)


def test_open_oracle_contract_and_output_repeat_byte_exactly() -> None:
    emitted = _run_oracle("--emit-contract")
    assert emitted.returncode == 0
    assert emitted.stderr == b""
    assert emitted.stdout == CONTRACT.read_bytes()

    arguments = (
        "--run",
        "--contract",
        str(CONTRACT),
        "--contract-sha256",
        CONTRACT_SHA256,
    )
    first = _run_oracle(*arguments)
    second = _run_oracle(*arguments)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout == OUTPUT.read_bytes()


def test_open_oracle_rejects_wrong_contract_identity_without_overwrite() -> None:
    before = OUTPUT.read_bytes()
    result = _run_oracle(
        "--run",
        "--contract",
        str(CONTRACT),
        "--contract-sha256",
        "0" * 64,
    )
    assert result.returncode == 2
    assert result.stderr == b""
    blocked = json.loads(result.stdout)
    assert blocked["status"] == "blocked"
    assert blocked["terminal"] == "BLOCKED_CANDIDATE_A_NONDETERMINISTIC_EXECUTION"
    assert OUTPUT.read_bytes() == before


def test_original_eight_file_node_inventory_is_exact() -> None:
    inventory = _json(INVENTORY, canonical=True)
    env = os.environ.copy()
    roots = [
        ROOT / "tmp/s4_candidate_a_mpmath_valid",
        ROOT / ".s4_candidate_a_pinned/fileio/src",
        ROOT / ".s4_candidate_a_pinned/material/src",
        ROOT / ".s4_candidate_a_pinned/mesh/src",
        ROOT / ".s4_candidate_a_pinned/geometry/src",
    ]
    existing = [str(path) for path in roots if path.is_dir()]
    if existing:
        prior = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(existing + ([prior] if prior else []))
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_STACKSIZE"] = "1G"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q"]
    command.extend(record["path"] for record in inventory["files"])
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    observed = [
        line
        for line in result.stdout.decode("utf-8").splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    assert observed == inventory["collection"]["node_ids"]
    assert _sha(("\n".join(observed) + "\n").encode()) == NODE_LIST_SHA256
