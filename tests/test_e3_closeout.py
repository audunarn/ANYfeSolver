from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "2ac678a7f94c250fe433f66378a83508d86ee499"

EXPECTED = {
    ".gitattributes": (1769, "FCE05E44B73DBC45CAF9D964E2FEE7E07492D2EF4320B122FE2152470E4739B2"),
    "docs/agent_plans/S4_E3_HW29_MITC9I_ROUTE_SELECTION_PLAN.md": (6092, "F2572FED30BAA18EE66029207438E1C64F2D5B208B834B261EC585B25E70DFCC"),
    "docs/agent_plans/S4_E3_A_VARIATIONAL_CLOSURE_STUDY_PLAN.md": (4794, "5903DFEC12D7F4331493CDFEFFB04ACBB22F94EA681993366D80957E403B09FA"),
    "docs/E3_HW29_SOURCE_IDENTITY.md": (5586, "50A3A953E31758B301A28906AF677236C9959DB04DA0F99F2CF8C02A8B07550C"),
    "docs/E3_MITC9I_REFERENCE.md": (6539, "F5115FB1EF1E41C6FA101E5C89918E15A1B8D493C5E377D21507BA1BBAF20CAA"),
    "docs/E3_ROUTE_SELECTION_REPORT.md": (4575, "D76876A388BAEDBB2C1DF23C91D1E5E7A99BD365439902904B017D9A7A5B93B9"),
    "docs/E3_ROUTE_SELECTION_INDEPENDENT_REVIEW.md": (12187, "4AFF194BB36ADF2D12A477917A8289C67E9DEAAB48EC29522FE0432CD6813617"),
    "docs/E3_ROUTE_SELECTION_COMPLETION.md": (2292, "6312D75B196146811E3AC731A4641AFC7C9F5BA4981D2C8B1301EF97685DE704"),
    "docs/reference_cases/e3_baseline.json": (1398, "D080BCC53EE4C19BFB49551B46826B66328BC9E8581EB254BBF0D5504FF54A67"),
    "docs/reference_cases/e3_environment.json": (1382, "F8F3461E06AFDC7B6627501B1E85C91095A88BC07EC76F8B0054D8C352770F43"),
    "docs/reference_cases/e3_test_inventory.json": (1264, "6FCBD5D28BCA20A3AF89469E426CF0F003CCA6C6E52B000D6A4A3B79FA95E9A0"),
    "docs/reference_cases/e3_source_registry.json": (2863, "3F28EEF4E2E83EE82BE9487233694547F946154B39114A785D61D341296322C7"),
    "docs/reference_cases/e3_search_log.json": (1559, "6A327A8120995CC52A943890E7C4EC6B7171C1F0F5F3C14300A82B5483C57495"),
    "docs/reference_cases/e3_material_fixtures.json": (908, "A9DAA9960E5B0FDC65653AC7E9BA88CBC39EE9B0A9FD786AB854D5965422266F"),
    "docs/reference_cases/e3_hw29_source_coverage.json": (5509, "5469057E9038ADDC904D4115B7C332B6E7D7488396A9C7C7DEB933D09D4D5AFE"),
    "docs/reference_cases/e3_hw29_cases.json": (1874, "1A6A3960DEC3E9E806B24E4CEF9531EC45DD8858A22F684013D8A7F81097F2DE"),
    "docs/reference_cases/e3_hw29_oracle.py": (21342, "CFDB4B762E641C3958D7B67373AABD745A591AFF5EE79FA27E0BD9EC8B53369F"),
    "docs/reference_cases/e3_hw29_contract.json": (2331, "E07C60EDE72DDD6D19D686F79978C3F0D1826DA91B1D2552534063BD28C394A0"),
    "docs/reference_cases/e3_hw29_output.json": (2441, "3D9E9C858CAD14CB3BDEBFC8866E971658F02E71B16573A320BAF0B08DFE9806"),
    "docs/reference_cases/e3_mitc9i_source_map.json": (3655, "7E3679EE0BD25245C26EF4D4C259CA3F8B838FD9ED98F6D053A1B3E4B35C039E"),
    "docs/reference_cases/e3_mitc9i_cases.json": (2153, "B25F0F7787DC8B56B08E4FAA0B1DE6E7AE6D34B80E9BE68E2B202BD4926D33E5"),
    "docs/reference_cases/e3_mitc9i_oracle.py": (24718, "1DB0E4C9A882E1250C596DA63118EBD835F57AC054104F8196BCE9F90F63ED6B"),
    "docs/reference_cases/e3_mitc9i_contract.json": (2116, "86824E91A460AEAC9F67B213048E471AF968C7AA9FE2C43E6B61B148A5C8FBED"),
    "docs/reference_cases/e3_mitc9i_output.json": (2475, "00A6603A7B163CBC4A25B7FDF74647DDC1BDA300D478598F1403E4582AF5B575"),
    "docs/reference_cases/e3_route_contract.json": (3610, "B39EE05F48EB4D5CF4A1A09C0FF20891886BB388631756FD08328CEE4FB99BF9"),
    "docs/reference_cases/e3_route_output.json": (708, "A2D3283C1F01A26EF01986A4C5396B6C07797C250B7D2BD3BDA21AD1E14C273E"),
    "docs/reference_cases/e3_route_status.json": (5577, "2A13A3C2AA0C86303A7EDC0DAF018133370565E091ADB0E2C76D93E535930790"),
    "tests/test_e3_baseline.py": (5792, "0B8739519F1176C1E2D1AD8AA30835D568C79E580F15284D64108F90D14D0B85"),
    "tests/test_e3_hw29_identity.py": (9882, "A9A6BCA9FC07C618FBC06E2BAF5C5313ECFDAA5B3CF8962E7F556B84614A1D8E"),
    "tests/test_e3_mitc9i_reference.py": (9312, "DA2ACE6005ADA2DE41D94C6BE0F328DD9ECD34A53BEB627FCD5516FABE6D2636"),
    "tests/test_e3_route_selection.py": (7989, "25502F3306EFD3750C2D91C277DC28444C67AA64D573B60A6F80234629DEFAB5"),
}

PRODUCTION = {
    "src/anyfesolver/__init__.py": (24779, "0C782CCE93C1346F8A9B6DB832156A4F2689B33460C58F5966E9DE2169C2B8F0"),
    "src/anyfesolver/anystructure_fem_mode.py": (60057, "2FEACAAFD2A516B8ACCD919124A18614B0C3096EC0EEAF02DF6BC4280A80616E"),
    "src/anyfesolver/elements.py": (190422, "5D0AF716CD2E466EB831B1896553DE06236AC1BA80A84BD1B09C7E4CEBBDE670"),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        assert key not in value
        value[key] = item
    return value


def _strict_json(relative: str) -> dict[str, object]:
    size, digest = EXPECTED[relative]
    raw = (ROOT / relative).read_bytes()
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


def test_e3_closeout_is_content_addressed_fail_closed_and_production_safe() -> None:
    documents: dict[str, dict[str, object]] = {}
    for relative, (size, digest) in EXPECTED.items():
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == size and _sha(raw) == digest
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
        if relative.endswith(".json"):
            documents[relative] = _strict_json(relative)

    hw_contract = documents["docs/reference_cases/e3_hw29_contract.json"]
    hw_output = documents["docs/reference_cases/e3_hw29_output.json"]
    q9_contract = documents["docs/reference_cases/e3_mitc9i_contract.json"]
    q9_output = documents["docs/reference_cases/e3_mitc9i_output.json"]
    route_contract = documents["docs/reference_cases/e3_route_contract.json"]
    route_output = documents["docs/reference_cases/e3_route_output.json"]
    status = documents["docs/reference_cases/e3_route_status.json"]
    baseline = documents["docs/reference_cases/e3_baseline.json"]
    inventory = documents["docs/reference_cases/e3_test_inventory.json"]
    sources = documents["docs/reference_cases/e3_source_registry.json"]

    new_paths = set(route_contract["allowed_extent"]["new_paths"])
    assert len(new_paths) == 31
    assert route_contract["allowed_extent"]["modified"] == [".gitattributes"]
    assert route_contract["allowed_extent"]["production_paths"] == []
    assert new_paths == (set(EXPECTED) - {".gitattributes"}) | {"tests/test_e3_closeout.py"}

    for contract in (hw_contract, q9_contract, route_contract):
        for record in contract["input_identities"].values():
            relative = record["path"]
            size, digest = EXPECTED[relative]
            assert record == {"bytes": size, "path": relative, "sha256": digest}

    assert hw_output["candidate_id"] == "study_e3_p.hw29_linear_isotropic_identity_v1"
    assert hw_output["contract_sha256"] == EXPECTED["docs/reference_cases/e3_hw29_contract.json"][1]
    assert hw_output["component_terminal"] == "BLOCKED_E3_P_HW29_PUBLIC_SOURCE"
    assert hw_output["certificate"]["source_closure"] == {
        "closed_rows": 9,
        "missing_indispensable_ids": [
            "eadg2_map",
            "exact_local_condensation",
            "linear_virtual_work_loads_and_physical_recovery",
            "mixed_shear_4_plus_4_maps",
            "uncondensed_discrete_functional_and_29_field_order",
        ],
        "total_rows": 14,
    }
    assert set(hw_output["certificate"]["unsupported_outcomes"].values()) == {
        "NOT_RUN_MISSING_PRINTED_BLOCKS",
        "NOT_RUN_MISSING_PRINTED_MAPS",
        "NOT_RUN_MISSING_SHELL_TRANSFORMATION",
        "NOT_RUN_PUBLIC_SOURCE_BLOCK",
    }
    for case in hw_output["certificate"]["gamma_stabilization"]["cases"].values():
        assert case["rank_gamma_outer_gamma"] == 1
        assert case["zero_row_sum_ground_coupling"] is True
        assert case["constant_drill_theta"] == "0"

    assert q9_output["reference_id"] == "reference_e3_q9.mitc9i_open_theory_extraction_v1"
    assert q9_output["contract_sha256"] == EXPECTED["docs/reference_cases/e3_mitc9i_contract.json"][1]
    assert q9_output["status"] == "GO_REFERENCE_E3_Q9_MITC9I_PARTIAL_PACKET"
    assert q9_output["hw29_route_gate"] == "NONE"
    assert q9_output["certificate"]["hw29_independence"] == {
        "affects_hw29": False,
        "route_gate": "NONE",
    }
    assert q9_output["certificate"]["covc"]["category"] == (
        "CENTRE_JACOBIAN_APPROXIMATION_NOT_EXACT_COVARIANCE"
    )
    assert len(q9_output["certificate"]["finite_rotation"]["missing_explicit_details"]) == 5

    assert route_output == {
        "component_statuses": {
            "hw29": "BLOCKED_E3_P_HW29_PUBLIC_SOURCE",
            "mitc9i": "GO_REFERENCE_E3_Q9_MITC9I_PARTIAL_PACKET",
        },
        "conditional_successor": {
            "bytes": 4794,
            "path": "docs/agent_plans/S4_E3_A_VARIATIONAL_CLOSURE_STUDY_PLAN.md",
            "sha256": "5903DFEC12D7F4331493CDFEFFB04ACBB22F94EA681993366D80957E403B09FA",
        },
        "contract_sha256": EXPECTED["docs/reference_cases/e3_route_contract.json"][1],
        "overall_release_terminal": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
        "production": {"legacy_shell_default": True, "production_changes": False},
        "route_authorization": "AUTHORIZE_E3_A_VARIATIONAL_CLOSURE_STUDY",
        "route_terminal": "UNCLASSIFIED_E3_Q4_FORMULATION_ROUTE",
        "schema": "anysolver.s4.e3-route-output-v1",
        "status": "complete",
    }
    assert status["components"]["hw29"]["terminal"] == hw_output["component_terminal"]
    assert status["components"]["hw29"]["terminal_is_mechanics_no_go"] is False
    assert status["components"]["hw29"]["unsupported_mechanics"] == "NOT_RUN"
    assert status["components"]["mitc9i"]["terminal"] == q9_output["status"]
    assert status["components"]["mitc9i"]["route_gate"] == "NONE"
    assert status["route"]["terminal"] == route_output["route_terminal"]
    assert status["route"]["authorization"] == route_output["route_authorization"]
    assert status["independent_review"] == {
        "bytes": 12187,
        "path": "docs/E3_ROUTE_SELECTION_INDEPENDENT_REVIEW.md",
        "sha256": "4AFF194BB36ADF2D12A477917A8289C67E9DEAAB48EC29522FE0432CD6813617",
        "verdict": "ACCEPT_NO_P0_OR_P1",
    }
    assert status["production"] == {
        "candidate_registered": False,
        "legacy_shell_default": True,
        "overall_release_terminal": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
        "production_changes": False,
        "public_api_changed": False,
        "selector_available": False,
        "serialization_changed": False,
    }

    assert baseline["authority"] == {
        "branch": "codex/s4-e3-hw29-mitc9i-route-selection",
        "commit": BASE_COMMIT,
        "tree": "f7382e2b88343ac29c9a9e3c424f618a3652cc01",
    }
    assert inventory["total_reference_count"] == 118
    assert inventory["total_reference_is_not_one_live_suite"] is True
    assert [(inventory["tiers"][key]["count"]) for key in ("e0", "e1", "e2")] == [94, 16, 8]
    assert inventory["tiers"]["e0"]["node_ids_canonical_lf_sha256"] == (
        "29EF584E9B51E8420934A519B3C1E71BDD3082EFDC89DBADA4FCE0FFE8997B9F"
    )
    assert inventory["tiers"]["e1"]["node_ids_canonical_lf_sha256"] == (
        "9835FB4580C886B52BFF5961A30CD78E921B5CEED92A918312149032748A7F63"
    )
    assert inventory["tiers"]["e1"]["historical_report_statement_preserved"] == 15

    assert sources["design_input"]["sha256"] == (
        "7D86FE7A6D205BFEDDA4C884A2AFAD5C80EF0F3DE6BA350C48BBB2150BFC5108"
    )
    assert sources["hw29"]["detailed_sources"][1]["sha256"] == (
        "E6AFAADE32B33D710D3C038635FE2AD2729E32FB952C5EC6706E68A93A3B1860"
    )
    assert sources["mitc9i"]["open_primary"]["sha256"] == (
        "5C66A76D39682F71C13208E71AFFA585FD3CD1E284185360B825572DC8BA048B"
    )
    assert sources["mitc9i"]["user_supplied_local_copy"]["byte_identical_to_open_primary"] is True
    assert sources["related_background"]["degenerated_q4_penalty_paper"]["sha256"] == (
        "B67AF5A43CB36FEC9E0D8CDAD745B391F9F5FC1861C842A249E2B982BDACD5E8"
    )
    assert all(value is False for value in sources["copyright_boundary"].values())

    review = (ROOT / "docs/E3_ROUTE_SELECTION_INDEPENDENT_REVIEW.md").read_text(encoding="utf-8")
    completion = (ROOT / "docs/E3_ROUTE_SELECTION_COMPLETION.md").read_text(encoding="utf-8")
    assert "ACCEPT_NO_P0_OR_P1" in review
    assert "BLOCKED_E3_P_HW29_PUBLIC_SOURCE" in review and "source-identity block" in completion
    assert "GO_REFERENCE_E3_Q9_MITC9I_PARTIAL_PACKET" in review and "non-gating" in completion
    assert "UNCLASSIFIED_E3_Q4_FORMULATION_ROUTE" in completion
    assert "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED" in completion

    for relative, (size, digest) in PRODUCTION.items():
        raw = _canonical_lf(ROOT / relative)
        assert len(raw) == size and _sha(raw) == digest

    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for rule in (
        "docs/agent_plans/S4_E3_* text eol=lf",
        "docs/E3_* text eol=lf",
        "docs/reference_cases/e3_* text eol=lf",
        "tests/test_e3_* text eol=lf",
    ):
        assert rule in attrs
    assert not any(path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")) for path in new_paths)

    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    untracked = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert diff.returncode == untracked.returncode == cached.returncode == 0
    assert cached.stdout == b""
    observed = set(diff.stdout.decode().splitlines()) | set(untracked.stdout.decode().splitlines())
    preserved = (
        ".s4_candidate_a_pinned/", ".s4_stage_m_execution/", ".s4_stage_m_mpmath/",
        ".s4_stage_m_mpmath_clean/", ".s4_stage_m_patch_tools/", "tmp/",
    )
    candidate_paths = {path for path in observed if not path.startswith(preserved)}
    assert candidate_paths == new_paths | {".gitattributes"}
    assert not any(path.startswith(("src/", ".github/", "pyproject.toml")) for path in candidate_paths)
