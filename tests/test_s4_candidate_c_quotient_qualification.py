from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/agent_plans/S4_CANDIDATE_C_LINEAR_QUOTIENT_QUALIFICATION_PLAN.md"
DERIVATION = ROOT / "docs/S4_CANDIDATE_C_QUOTIENT_DERIVATION.md"
REPORT = ROOT / "docs/S4_CANDIDATE_C_QUOTIENT_QUALIFICATION_REPORT.md"
CASES = ROOT / "docs/reference_cases/s4_candidate_c_quotient_cases.json"
INVENTORY = ROOT / "docs/reference_cases/s4_candidate_c_quotient_test_inventory.json"
ORACLE = ROOT / "docs/reference_cases/s4_candidate_c_quotient_oracle.py"
CONTRACT = ROOT / "docs/reference_cases/s4_candidate_c_quotient_contract.json"
OUTPUT = ROOT / "docs/reference_cases/s4_candidate_c_quotient_output.json"

HASHES = {
    PLAN: "82762B0FAD7CC200B76C6262D3B2DFEDAA0E2261FD5E7DF1195F8BAEA85D8901",
    DERIVATION: "447AB70BE8D34A08FFAAE98C6FA5583A15DE7C384789AA8C31838A5B7FFB6ACE",
    REPORT: "51F7E5BA22C13EE0EA0642C83EA8DEBBB5888CE8517AB677572D1FB79131179E",
    CASES: "B41360811714ED7A52B40F6ED282EA9F89C91A6FD0FE818F7A0AA51CA9A84936",
    INVENTORY: "2A79F6E5F1683BC6D2F8FD049DFDBA9FF0457EDE486F1F9B0ACA8CEFCC5AA592",
    ORACLE: "4476B913393DB536CB374D8E458DC29250FF535A4C5AB340F6014C93F6323F16",
    CONTRACT: "6FC6EACBA183E93C5A0E4CFF2B4EB0E294C22BA9423CC5EA1F55E9094946D70E",
    OUTPUT: "A44ED2DD5F11A0BBF9A0CB8D01B869A1D7E12632B3E85E773A804FC2CCC140B6",
}

CONTRACT_SHA256 = HASHES[CONTRACT]
NODE_LIST_SHA256 = "AA024AA5E14FAE296B854F6E2DDE289DE5978C6B795BE8D2CE58A12B7A170CC4"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    assert _sha(raw) == HASHES[path]
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw and raw.endswith(b"\n")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert raw == canonical
    return value


def _run_oracle(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(ORACLE), *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )


def test_candidate_c_packet_is_content_addressed_and_fail_closed() -> None:
    for path, expected in HASHES.items():
        assert _sha(path.read_bytes()) == expected
    contract = _json(CONTRACT)
    output = _json(OUTPUT)
    cases = _json(CASES)
    inventory = _json(INVENTORY)

    assert contract["authority"] == {
        "base_commit": "2cb8c53cd1097380c872ba2802ec0eacc5198304",
        "base_tree": "f95d74e3ed1bb760f622e188f75f62a8b7ae43f6",
    }
    assert contract["allowed_extent"]["production_paths"] == []
    assert output["candidate_terminal"] == "NO_GO_CANDIDATE_C_QUOTIENT_INF_SUP"
    assert output["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert output["preserved"] == {
        "candidate_a_pair_terminal": "NO_GO_CANDIDATE_A_DISCRETE_PAIR",
        "candidate_b_terminal": "NO_GO_CANDIDATE_B",
        "legacy_shell_default": True,
        "rank_four_terminal": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
    }
    assert output["exclusions"] == cases["exclusions"]
    assert all(value is False for value in output["exclusions"].values())
    assert inventory["collection"]["count"] == 75
    nodes = inventory["collection"]["node_ids"]
    assert len(nodes) == len(set(nodes)) == 75
    assert _sha(("\n".join(nodes) + "\n").encode()) == NODE_LIST_SHA256


def test_candidate_c_oracle_is_stdlib_only_and_byte_repeatable() -> None:
    tree = ast.parse(ORACLE.read_text(encoding="utf-8"), filename=str(ORACLE))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = ("anyfesolver", "numpy", "scipy", "mpmath", "sympy")
    assert not any(
        name == item or name.startswith(item + ".")
        for name in imported
        for item in forbidden
    )

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


def test_candidate_c_oracle_rejects_wrong_contract_without_overwrite() -> None:
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
    assert blocked["terminal"] == "BLOCKED_CANDIDATE_C_NONDETERMINISTIC_EXECUTION"
    assert OUTPUT.read_bytes() == before


def test_candidate_c_preimplementation_75_node_inventory_is_exact() -> None:
    inventory = _json(INVENTORY)
    env = os.environ.copy()
    roots = [
        ROOT / "tmp/s4_candidate_a_mpmath_valid",
        ROOT / ".s4_candidate_a_pinned/fileio/src",
        ROOT / ".s4_candidate_a_pinned/material/src",
        ROOT / ".s4_candidate_a_pinned/mesh/src",
        ROOT / ".s4_candidate_a_pinned/geometry/src",
    ]
    existing = [str(path) for path in roots if path.is_dir()]
    prior = env.get("PYTHONPATH")
    if existing:
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
