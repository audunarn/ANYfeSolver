from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "docs/reference_cases/s4_candidate_e0_oracle.py"
CONTRACT = ROOT / "docs/reference_cases/s4_candidate_e0_contract.json"
OUTPUT = ROOT / "docs/reference_cases/s4_candidate_e0_output.json"
REPORT = ROOT / "docs/S4_CANDIDATE_E0_QUALIFICATION_REPORT.md"
BASE = "a9b45ca95303bc4b30b893fbb0d7177f9c98db03"

CONTRACT_SHA256 = "D3ACF44D4690E7ED8E257B1A1A5DB124CE91ADAFE98ACDD186305C56A4740B03"
OUTPUT_SHA256 = "513465CB4993C398C9B10334244F07A26AC9A1980D49A24F79F5FC3CC7EB04AD"
REPORT_SHA256 = "EC6F8A468384D65A75F783CC1947F6F5AC9E49D0FD3CDE741FEB02A45037C6FE"

ARTIFACTS = {
    "docs/agent_plans/S4_CANDIDATE_E0_DNV_LINEAR_BUCKLING_QUALIFICATION_PLAN.md": (5438, "082BABD49F20436BEFBC2C14C123F6904DAAA6597EA99A1F7D85FA13F80B1162"),
    "docs/S4_CANDIDATE_E0_FORMULATION_DERIVATION.md": (4103, "D9A443D2B7F92E72057AC7317083B34AD586899F5BE75F148E94C2C5798DD211"),
    "docs/S4_CANDIDATE_E0_QUALIFICATION_REPORT.md": (4178, REPORT_SHA256),
    "docs/reference_cases/s4_candidate_e0_baseline.json": (2624, "71831C68F5DEF66A3BD698ACE6CCAF06E1D3A08FA19C4C05C8F19E221F177397"),
    "docs/reference_cases/s4_candidate_e0_environment.json": (984, "31D7C666BD68BCE48CD786554674F9C90D47ECABB831C51B20B8C8AE8349F84D"),
    "docs/reference_cases/s4_candidate_e0_source_registry.json": (3705, "E31B419F141B4FC0C80010BE5BDE31A4F85ACE3A84E6B5F01E0D80B34CE617CC"),
    "docs/reference_cases/s4_candidate_e0_formulation_identity.json": (1121, "2A0A1083C655262CDD5EBC19084C3CAC4E2EE9B988618C00EF607C1E821C90B3"),
    "docs/reference_cases/s4_candidate_e0_dnv_material_fixtures.json": (2135, "A16024C81522FB783841CC790C11772A10C8D0D936F9E678BE1CA981FD3DD016"),
    "docs/reference_cases/s4_candidate_e0_gate_cases.json": (1909, "C2FB7EFF6AEAF084C00762B480288F1064D22A75527967D825742E9E91A18264"),
    "docs/reference_cases/s4_candidate_e0_test_inventory.json": (2169, "DC63B9868057080AFCF6ED229C07F967F8F95A36352D755F4329FDCA5FE79824"),
    "docs/reference_cases/s4_candidate_e0_oracle.py": (19683, "48FBD49E7011197A370492E61C76167BDC038054D47F3ADB2A7DA6DCEAF82A4A"),
    "docs/reference_cases/s4_candidate_e0_contract.json": (4528, CONTRACT_SHA256),
    "docs/reference_cases/s4_candidate_e0_output.json": (2159, OUTPUT_SHA256),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def _decode(raw: bytes) -> object:
    assert not raw.startswith(b"\xef\xbb\xbf")
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )


def _json(path: Path, sha256: str | None = None) -> dict[str, object]:
    raw = path.read_bytes()
    if sha256 is not None:
        assert _sha(raw) == sha256
    assert b"\r" not in raw and raw.endswith(b"\n")
    value = _decode(raw)
    assert isinstance(value, dict)
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
    assert raw == canonical
    return value


def _lf(path: Path, size: int, sha256: str) -> bytes:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    lf = raw.replace(b"\r\n", b"\n")
    assert b"\r" not in lf
    assert len(lf) == size
    assert _sha(lf) == sha256
    return lf


def _run(*args: str) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(ORACLE), *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def test_e0_packet_hashes_canonical_json_and_accepted_85_node_inventory() -> None:
    json_paths = {path for path in ARTIFACTS if path.endswith(".json")}
    for relative, (size, sha256) in ARTIFACTS.items():
        raw = _lf(ROOT / relative, size, sha256)
        if relative in json_paths:
            _json(ROOT / relative, sha256)
        assert raw.endswith(b"\n")

    inventory = _json(ROOT / "docs/reference_cases/s4_candidate_e0_test_inventory.json")
    base_record = inventory["composition"]["base_75"]
    base_raw = (ROOT / base_record["path"]).read_bytes()
    assert len(base_raw) == base_record["bytes"] == 10598
    base_inventory = _json(ROOT / base_record["path"], base_record["sha256"])
    assert len(base_inventory["files"]) == 11
    assert len(inventory["composition"]["appended_files"]) == 3
    for record in [*base_inventory["files"], *inventory["composition"]["appended_files"]]:
        _lf(ROOT / record["path"], record["canonical_lf_bytes"], record["canonical_lf_sha256"])
    nodes = list(base_inventory["collection"]["node_ids"])
    nodes.extend(inventory["composition"]["appended_node_ids"])
    joined = ("\n".join(nodes) + "\n").encode()
    assert len(nodes) == len(set(nodes)) == inventory["collection"]["count"] == 85
    assert len(joined) == inventory["collection"]["node_ids_canonical_lf_bytes"] == 8910
    assert _sha(joined) == inventory["collection"]["node_ids_canonical_lf_sha256"] == "966AA017D996DC3F83F0A2C98D269022803B44DBB411A934CC01BECA958E2873"


def test_e0_oracle_is_stdlib_only_and_emits_exact_contract() -> None:
    tree = ast.parse(ORACLE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "argparse", "hashlib", "json", "pathlib", "sys"}
    assert "anyfesolver" not in imports
    result = _run("--emit-contract")
    assert result.returncode == 0 and result.stderr == b""
    assert result.stdout == CONTRACT.read_bytes()
    assert _sha(result.stdout) == CONTRACT_SHA256


def test_e0_two_fresh_source_gate_runs_are_byte_identical() -> None:
    args = ("--run", "--contract", str(CONTRACT), "--contract-sha256", CONTRACT_SHA256)
    first = _run(*args)
    second = _run(*args)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout == OUTPUT.read_bytes()
    assert _sha(first.stdout) == OUTPUT_SHA256


def test_e0_wrong_or_malformed_contract_blocks_without_overwrite(tmp_path: Path) -> None:
    before = OUTPUT.read_bytes()
    result = _run("--run", "--contract", str(CONTRACT), "--contract-sha256", "0" * 64)
    assert result.returncode == 2 and result.stderr == b""
    blocked = _decode(result.stdout)
    assert blocked["status"] == "blocked"
    assert blocked["terminal"] == "BLOCKED_CANDIDATE_E_NONDETERMINISTIC_EXECUTION"
    assert OUTPUT.read_bytes() == before

    missing = _run("--run", "--contract", str(tmp_path / "missing.json"), "--contract-sha256", "0" * 64)
    assert missing.returncode == 2 and missing.stderr == b""
    assert _decode(missing.stdout)["terminal"] == "BLOCKED_CANDIDATE_E_NONDETERMINISTIC_EXECUTION"

    malformed = tmp_path / "contract.json"
    malformed.write_bytes(b'{"duplicate":1,"duplicate":2}\n')
    namespace = runpy.run_path(str(ORACLE))
    namespace["CONTRACT_PATH"] = malformed
    try:
        namespace["_load_contract"](malformed, _sha(malformed.read_bytes()))
    except namespace["ContractViolation"]:
        pass
    else:
        raise AssertionError("duplicate-key contract did not fail closed")


def test_e0_equation_level_gap_matrix_forces_source_terminal() -> None:
    registry = _json(ROOT / "docs/reference_cases/s4_candidate_e0_source_registry.json")
    claims = {record["id"]: record["status"] for record in registry["claims"]}
    assert sum(value == "PRINTED_PRIMARY_SOURCE" for value in claims.values()) == 4
    assert claims["all_four_nodes_have_independent_six_dof"] == "CONFLICTING_PRIMARY_SOURCE"
    assert claims["exact_gww1992_allman_spin_skew_force_interpolation"] == "MISSING_FULL_TEXT"
    assert sum(value == "UNSUBSTANTIATED_COMPOSITION" for value in claims.values()) == 3
    output = _json(OUTPUT, OUTPUT_SHA256)
    assert output["candidate_terminal"] == "BLOCKED_CANDIDATE_E_SOURCE_OR_IDENTITY"
    assert output["terminal_reason"] == "MISSING_EXACT_ALL_NODE_DRILL_FORMULATION_AND_SOURCE_CONFLICT"
    assert output["mechanics_results_present"] is False
    assert all(
        value in {"BLOCKED_CANDIDATE_E_SOURCE_OR_IDENTITY", "NOT_RUN_DUE_TO_SOURCE_GATE", "NOT_IN_E0_SCOPE"}
        for value in output["stages"].values()
    )


def test_e0_rp_c208_fixture_boundary_has_no_new_material_field() -> None:
    fixtures = _json(ROOT / "docs/reference_cases/s4_candidate_e0_dnv_material_fixtures.json")
    dataset = fixtures["rp_c208_dataset"]
    assert sorted(dataset["grades"]) == ["S235", "S275", "S355", "S420", "S460"]
    assert sum(len(rows) for rows in dataset["grades"].values()) == dataset["row_count"] == 17
    assert dataset["classification"] == "recommended_practice_not_ru_ship_rule"
    assert fixtures["input_contract"]["new_public_fields"] == []
    assert set(fixtures["input_contract"]["forbidden_inferred_fields"]) == {
        "cosserat_modulus", "curvature_weight", "drill_modulus", "internal_length", "rotational_micro_inertia"
    }
    assert fixtures["qualification"] == {
        "candidate_element_material_compatibility": "NOT_RUN_DUE_TO_SOURCE_GATE",
        "dnv_approval": False,
        "ordinary_material_input_shape_compatible": True,
        "rp_c208_fixture_reproduction": True,
    }

    sibling = ROOT.parent / "ANYmaterial"
    if sibling.is_dir():
        for relative, record in fixtures["anymaterial"]["files"].items():
            _lf(sibling / relative, record["canonical_lf_bytes"], record["canonical_lf_sha256"])


def test_e0_preserves_a_b_c_rank_four_and_production_identities() -> None:
    output = _json(OUTPUT, OUTPUT_SHA256)
    assert output["immutable_results"] == {
        "candidate_a": "NO_GO_CANDIDATE_A_DISCRETE_PAIR",
        "candidate_b": "NO_GO_CANDIDATE_B",
        "candidate_c": "NO_GO_CANDIDATE_C_QUOTIENT_INF_SUP",
        "rank_four": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
    }
    assert output["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert output["production"] == {
        "legacy_shell_default": True,
        "public_api_changed": False,
        "selector_available": False,
        "serialization_changed": False,
    }
    baseline = _json(ROOT / "docs/reference_cases/s4_candidate_e0_baseline.json")
    for relative, record in baseline["production_sources"].items():
        _lf(ROOT / relative, record["canonical_lf_bytes"], record["canonical_lf_sha256"])


def test_e0_extent_is_qualification_only_and_has_no_production_import() -> None:
    contract = _json(CONTRACT, CONTRACT_SHA256)
    allowed = set(contract["allowed_extent"]["new_paths"]) | {".gitattributes"}
    assert contract["allowed_extent"]["production_paths"] == []
    assert all(not path.startswith(("src/", ".github/")) for path in allowed)

    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    assert diff.returncode == 0, diff.stderr.decode(errors="replace")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    assert untracked.returncode == 0, untracked.stderr.decode(errors="replace")
    observed = set(diff.stdout.decode().splitlines()) | set(untracked.stdout.decode().splitlines())
    preserved = (
        ".s4_candidate_a_pinned/", ".s4_stage_m_execution/", ".s4_stage_m_mpmath/",
        ".s4_stage_m_mpmath_clean/", ".s4_stage_m_patch_tools/", "tmp/",
    )
    candidate_paths = {path for path in observed if not path.startswith(preserved)}
    assert candidate_paths <= allowed
    assert not any(path.startswith(("src/", "pyproject.toml", ".github/")) for path in candidate_paths)

    source = ORACLE.read_text(encoding="utf-8")
    assert "import anyfesolver" not in source and "from anyfesolver" not in source
