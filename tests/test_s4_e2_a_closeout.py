from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "281ed90e148c125edbec27e7336a8f9f0df08edc"

EXPECTED = {
    ".gitattributes": (1644, "CFA148F1B78C01C2C6E89DEBD6600458380C64097DBB7FCF54BA0C6B105A600A"),
    "docs/agent_plans/S4_E2_A_SOURCE_KINEMATICS_PLAN.md": (
        5679,
        "D8F39F3C75D19AF3C26A69845216AF9A7C948EE1F6CCB3E3BBFCF0A21C8131F4",
    ),
    "docs/S4_E2_A_FORMULATION_DERIVATION.md": (
        10061,
        "E7CE3A36E895238E1E31734E81CEB99A1A804E8D6B08101DBC2EA5452DE3B16F",
    ),
    "docs/S4_E2_A_EXTENSION_CLOSURE.md": (
        2351,
        "F6CC6AD38AEA8FCC6C402301F58CA47AFF03B738CFC2E44E1F23A6F8CD19BACA",
    ),
    "docs/S4_E2_A_QUALIFICATION_REPORT.md": (
        5579,
        "28FA4039DA2E47D9B91CD6C21620685E7DD710C6EE79AF7745022242429CE074",
    ),
    "docs/S4_E2_A_INDEPENDENT_REVIEW.md": (
        8610,
        "45EDC88299B8FC1C5618924F4B74A01B32EB7A211646E08BDDE36A0F606CB023",
    ),
    "docs/reference_cases/s4_e2_a_baseline.json": (
        1891,
        "EF62A7F2F40089A47237A17C03A1FC3C7D3BA5A9AA696D4C326CA4DB2A994A92",
    ),
    "docs/reference_cases/s4_e2_a_environment.json": (
        912,
        "2A0E7D3B568F5ACC912A7897E3B4787F7AC9BBA57739BFD346A1D5DB68B82C99",
    ),
    "docs/reference_cases/s4_e2_a_test_inventory.json": (
        1582,
        "8DD67A8940FB65601CFA43455558724E507014F327C047AF1E56A21D21A2CBA9",
    ),
    "docs/reference_cases/s4_e2_a_source_registry.json": (
        5509,
        "15AFFE358D5551EB08359267B6B0FD3FAAF6F15198C22475B37DF8AD4C014D2E",
    ),
    "docs/reference_cases/s4_e2_a_identity.json": (
        4111,
        "1D68C16149F0368E883CCD1611107068DF662C794A6D58541665A5F99472421D",
    ),
    "docs/reference_cases/s4_e2_a_cases.json": (
        2352,
        "61ED18EDB32B0DAF288E3EB66FEA522D5D4588542F11D8881B5B7762FCAC3729",
    ),
    "docs/reference_cases/s4_e2_a_oracle.py": (
        42587,
        "A1796D466DF6DDCDB420987F8FAFC3787B563C16F0B8AEC58C716C0EF194D151",
    ),
    "docs/reference_cases/s4_e2_a_contract.json": (
        3433,
        "E3AA3BC6AD8FAD7EB64564851FC558B0D1B2ACB533B292EEBA580EBA47B02D3E",
    ),
    "docs/reference_cases/s4_e2_a_output.json": (
        5821,
        "37C803C565602E1AF983AA8374C3DA090EFD1CC73F2B672F2C815CC6A56B623D",
    ),
    "docs/reference_cases/s4_e2_a_status.json": (
        3076,
        "F0BC99B7AE26BCF1C406EF9E6B56AA72BE6AD57BC6F8FAC7EB871387B6B680DB",
    ),
    "tests/test_s4_e2_a_exact_kinematics.py": (
        10443,
        "64862C2FAA4B96C99358B21ADFCABD16BA26CBEDB06E1E4CDBBD1EE5254D6E6D",
    ),
    "tests/test_s4_e2_a_qualification.py": (
        10984,
        "F8F3676E224D43E507CB0B7189FCC36130BAEA60836E43BA3FA67C1B338549A8",
    ),
}

NEW_PATHS = {
    "docs/agent_plans/S4_E2_A_SOURCE_KINEMATICS_PLAN.md",
    "docs/S4_E2_A_FORMULATION_DERIVATION.md",
    "docs/S4_E2_A_EXTENSION_CLOSURE.md",
    "docs/S4_E2_A_QUALIFICATION_REPORT.md",
    "docs/S4_E2_A_INDEPENDENT_REVIEW.md",
    "docs/reference_cases/s4_e2_a_baseline.json",
    "docs/reference_cases/s4_e2_a_environment.json",
    "docs/reference_cases/s4_e2_a_test_inventory.json",
    "docs/reference_cases/s4_e2_a_source_registry.json",
    "docs/reference_cases/s4_e2_a_identity.json",
    "docs/reference_cases/s4_e2_a_cases.json",
    "docs/reference_cases/s4_e2_a_oracle.py",
    "docs/reference_cases/s4_e2_a_contract.json",
    "docs/reference_cases/s4_e2_a_output.json",
    "docs/reference_cases/s4_e2_a_status.json",
    "tests/test_s4_e2_a_exact_kinematics.py",
    "tests/test_s4_e2_a_qualification.py",
    "tests/test_s4_e2_a_closeout.py",
}

PRODUCTION = {
    "src/anyfesolver/__init__.py": (
        24779,
        "0C782CCE93C1346F8A9B6DB832156A4F2689B33460C58F5966E9DE2169C2B8F0",
    ),
    "src/anyfesolver/anystructure_fem_mode.py": (
        60057,
        "2FEACAAFD2A516B8ACCD919124A18614B0C3096EC0EEAF02DF6BC4280A80616E",
    ),
    "src/anyfesolver/elements.py": (
        190422,
        "5D0AF716CD2E466EB831B1896553DE06236AC1BA80A84BD1B09C7E4CEBBDE670",
    ),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def _strict_json(path: Path, size: int, digest: str) -> dict[str, object]:
    raw = path.read_bytes()
    assert len(raw) == size and _sha(raw) == digest
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert raw == canonical
    return value


def _canonical_lf(path: Path) -> bytes:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    assert b"\r" not in raw
    return raw


def test_s4_e2_a_closeout_is_exact_fail_closed_and_production_safe() -> None:
    documents: dict[str, dict[str, object]] = {}
    for relative, (size, digest) in EXPECTED.items():
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == size and _sha(raw) == digest
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
        if relative.endswith(".json"):
            documents[relative] = _strict_json(ROOT / relative, size, digest)

    contract = documents["docs/reference_cases/s4_e2_a_contract.json"]
    output = documents["docs/reference_cases/s4_e2_a_output.json"]
    status = documents["docs/reference_cases/s4_e2_a_status.json"]
    baseline = documents["docs/reference_cases/s4_e2_a_baseline.json"]
    inventory = documents["docs/reference_cases/s4_e2_a_test_inventory.json"]
    sources = documents["docs/reference_cases/s4_e2_a_source_registry.json"]
    identity = documents["docs/reference_cases/s4_e2_a_identity.json"]

    assert set(contract["allowed_extent"]["new_paths"]) == NEW_PATHS
    assert contract["allowed_extent"]["modified"] == [".gitattributes"]
    assert contract["allowed_extent"]["production_paths"] == []
    for record in contract["input_identities"].values():
        relative = record["path"]
        size, digest = EXPECTED[relative]
        assert record == {"bytes": size, "path": relative, "sha256": digest}

    terminal = "BLOCKED_E2_A_SOURCE_OR_FORMULATION_IDENTITY"
    reason = "RANK_SUFFICIENT_DISPLACEMENT_ENRICHMENT_NONUNIQUE"
    assert contract["scientific_terminal"] == {"reason": reason, "value": terminal}
    assert contract["mechanics_execution"] == "FORBIDDEN_AFTER_SOURCE_IDENTITY_BLOCK"
    assert output["candidate_terminal"] == terminal and output["reason"] == reason
    assert output["contract_sha256"] == EXPECTED[
        "docs/reference_cases/s4_e2_a_contract.json"
    ][1]
    assert set(output["downstream_gates"].values()) == {"NOT_RUN_IDENTITY_AMBIGUOUS"}
    assert output["e1_rh"] == "DEFERRED_NOT_RUN"
    assert output["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert output["production"] == {
        "legacy_shell_default": True,
        "public_api_changed": False,
        "selector_available": False,
        "serialization_changed": False,
    }

    candidate = status["candidate"]
    assert candidate["candidate_id"] == output["candidate_id"]
    assert candidate["terminal"] == terminal and candidate["reason"] == reason
    assert candidate["contract"]["sha256"] == output["contract_sha256"]
    assert candidate["output"]["sha256"] == EXPECTED[
        "docs/reference_cases/s4_e2_a_output.json"
    ][1]
    assert candidate["report"]["sha256"] == EXPECTED[
        "docs/S4_E2_A_QUALIFICATION_REPORT.md"
    ][1]
    assert candidate["downstream_gates"] == output["downstream_gates"]
    assert status["independent_review"] == {
        "bytes": 8610,
        "path": "docs/S4_E2_A_INDEPENDENT_REVIEW.md",
        "sha256": "45EDC88299B8FC1C5618924F4B74A01B32EB7A211646E08BDDE36A0F606CB023",
        "verdict": "ACCEPT_NO_P0_OR_P1",
    }
    assert status["relationship"] == {
        "e1_rh": "DEFERRED_NOT_RUN",
        "mechanics_classified": False,
        "source_identity_block_is_mechanics_no_go": False,
    }
    assert status["production"] == {
        "legacy_shell_default": True,
        "production_activation": False,
        "public_api_changed": False,
        "selector_available": False,
        "serialization_changed": False,
    }
    assert status["historical_results"] == {
        "candidate_a": "NO_GO_CANDIDATE_A_DISCRETE_PAIR",
        "candidate_b": "NO_GO_CANDIDATE_B",
        "candidate_c": "NO_GO_CANDIDATE_C_QUOTIENT_INF_SUP",
        "candidate_e0": "BLOCKED_CANDIDATE_E_SOURCE_OR_IDENTITY",
        "candidate_e1_a": "NO_GO_CANDIDATE_E1_A_RANK_DEFICIENCY",
        "candidate_e1_r": "PROVISIONAL_GO_CANDIDATE_E1_R_PLANAR_REGULARIZER_ONLY",
        "rank_four": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
    }

    certificate = output["certificate"]
    witnesses = certificate["affine_geometry_witnesses"]
    assert set(witnesses) == {"square", "skew_rational"}
    assert witnesses["square"]["difference_strain_energy"] == "128/35"
    assert witnesses["skew_rational"]["cofactor_map"] == [["12", "-4"], ["-5", "3"]]
    assert witnesses["skew_rational"]["cofactor_pairing"] == [["16", "0"], ["0", "16"]]
    assert witnesses["skew_rational"]["difference_strain_energy"] == "305584/175"
    assert certificate["nonuniqueness"]["status"] == (
        "TWO_NON_EQUIVALENT_AFFINE_COVARIANT_DISPLACEMENT_LIFTS"
    )
    assert certificate["hostile_e1_a"]["full_rank_upper_bound"] == 17
    assert certificate["hostile_e1_a"]["immutable_terminal"] == (
        "NO_GO_CANDIDATE_E1_A_RANK_DEFICIENCY"
    )

    assert baseline["authority"]["commit"] == BASE_COMMIT
    assert baseline["authority"]["tree"] == "1ee60da4717055f5cc1b37ff9369877bb1867861"
    assert inventory["total_reference_count"] == 110
    assert inventory["total_reference_is_not_one_live_suite"] is True
    assert inventory["tiers"]["e0_immutable"]["count"] == 94
    assert inventory["tiers"]["e0_immutable"]["node_ids_canonical_lf_sha256"] == (
        "29EF584E9B51E8420934A519B3C1E71BDD3082EFDC89DBADA4FCE0FFE8997B9F"
    )
    assert inventory["tiers"]["e1_committed"]["count"] == 16
    assert inventory["tiers"]["e1_committed"]["historical_report_statement_preserved"] == 15
    assert inventory["tiers"]["e1_committed"]["node_ids_canonical_lf_sha256"] == (
        "9835FB4580C886B52BFF5961A30CD78E921B5CEED92A918312149032748A7F63"
    )
    assert status["baseline_verification"]["one_live_successor_suite"] is False

    for record in baseline["immutable_e1"].values():
        if isinstance(record, dict) and "path" in record:
            raw = (ROOT / record["path"]).read_bytes()
            assert len(raw) == record["bytes"] and _sha(raw) == record["sha256"]
    assert sources["design_input"]["sha256"] == (
        "5BF221C0B75425E292D80EF59CA4B6613445DD0621664F9350043E3B5B9B3C68"
    )
    assert sources["identity_gate"]["complete_H_selected"] is False
    assert {record["class"] for record in sources["statements"]} == {None, "B", "D", "P"}
    sestra = [record for record in sources["statements"] if record["id"] == "sestra_formulation_context"]
    assert len(sestra) == 1 and sestra[0]["class"] == "B"
    assert sources["copyright_boundary"] == {
        "committed_external_pdf_content": False,
        "committed_manual_content": False,
        "committed_page_images": False,
        "committed_quotations": False,
        "permitted_record": "bibliography_raw_identity_equation_map_and_independent_derivation_only",
    }
    assert identity["enrichment_gate"]["selected_H"] is None
    assert identity["enrichment_gate"]["selected_B"] is None

    review = (ROOT / "docs/S4_E2_A_INDEPENDENT_REVIEW.md").read_text(encoding="utf-8")
    report = (ROOT / "docs/S4_E2_A_QUALIFICATION_REPORT.md").read_text(encoding="utf-8")
    assert "ACCEPT" in review and "no P0 or P1" in review
    assert terminal in review and terminal in report
    assert "not a failed rank test" in report
    assert "DEFERRED_NOT_RUN" in review and "DEFERRED_NOT_RUN" in report

    for relative, (size, digest) in PRODUCTION.items():
        raw = _canonical_lf(ROOT / relative)
        assert len(raw) == size and _sha(raw) == digest

    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "docs/reference_cases/s4_e2_a_* text eol=lf" in attrs
    assert "tests/test_s4_e2_a_* text eol=lf" in attrs
    assert not any(path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")) for path in NEW_PATHS)

    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert diff.returncode == untracked.returncode == cached.returncode == 0
    assert cached.stdout == b""
    observed = set(diff.stdout.decode().splitlines()) | set(untracked.stdout.decode().splitlines())
    preserved = (
        ".s4_candidate_a_pinned/",
        ".s4_stage_m_execution/",
        ".s4_stage_m_mpmath/",
        ".s4_stage_m_mpmath_clean/",
        ".s4_stage_m_patch_tools/",
        "tmp/",
    )
    candidate_paths = {path for path in observed if not path.startswith(preserved)}
    assert candidate_paths == NEW_PATHS | {".gitattributes"}
    assert not any(
        path.startswith(("src/", ".github/", "pyproject.toml")) for path in candidate_paths
    )
