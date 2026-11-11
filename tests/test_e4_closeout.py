from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "c55ad9e5f8e78b1749c4152e4ba66b6f9e20b198"
WS_SUCCESSOR = "docs/agent_plans/S4_E4_WS_STABILITY_QUALIFICATION_PLAN.md"

EXPECTED = {
    ".gitattributes": (1894, "60425638D89A10E0EFFEE7C6CF2D40C7A0A9538453779B3C140FCACBD8C35886"),
    "docs/E4_BASELINE_AND_AUTHORITY.md": (1010, "41088EDEEB9B7A2DB8A13FFD22F476AAD69E7D86E8BD3AB8DB086688807F0D01"),
    "docs/E4_COMPLETION.md": (2943, "64E8592A023634DB1995B0AEAA1A0274C83890C03A7C3939DFF27378BCA87F20"),
    "docs/E4_INDEPENDENT_REVIEW.md": (13311, "E3E9C529C2912CD0983941158AB615C9FE4D0903EEAFDF780E6762EB14B222B7"),
    "docs/E4_OPEN_CORE_IDENTITY.md": (6688, "BAFC21DC85C0CD9101C30ACC5D84F4BC57F3394EA4C0AEFB31CCFA7E43655E5D"),
    "docs/E4_PL_VARIATIONAL_CLOSURE.md": (8302, "14BDA35109FE8C653B85BF890C36CC454CDE938BCDC3820CA39479EC620EFB4D"),
    "docs/E4_ROUTE_REPORT.md": (5696, "0DA0B9ED4F604BD8D476E289102B0D44779EFB73F3F1BDD2BA26AB21CED9FF2D"),
    "docs/E4_WS_FEASIBILITY_THEOREM.md": (3839, "80E02F2564D6DE6D5E1A66857A78D97FCA83C828AEAB20BFE4707408EAD7BF19"),
    "docs/agent_plans/S4_E4_PL_LINEAR_QUALIFICATION_PLAN.md": (4670, "912322A8158255F17DDA44A3BB8FD59EFF1FC3B6B1E9D6BBB22B4E49A72BD193"),
    "docs/agent_plans/S4_E4_VARIATIONAL_DRILL_CLOSURE_PLAN.md": (5570, "BE515556E2019CDE69E4E7489FD9200F16CDE8D82C032131776EA0520DEB59A1"),
    "docs/reference_cases/e4_allowed_extent.json": (2052, "A34D96E442DADA69F7EBD2A9E3888B2885386431BC82D6BF80AD57B851A9842E"),
    "docs/reference_cases/e4_baseline.json": (3309, "7A404185E3F15FA56B589264FA5C816031B3BF25BA8003D081844B517EABB793"),
    "docs/reference_cases/e4_core_cases.json": (5435, "FE08E59D9E01073E04251C52544EDF10C4CC86861EA8B98FC6CB1A645427FEF2"),
    "docs/reference_cases/e4_core_contract.json": (2284, "8768ABDC5D77B5FCC1643AF69920EFEB83F27DDFBDE17525F06EE2B53A5C1678"),
    "docs/reference_cases/e4_core_oracle.py": (30906, "C829DD61CEF0D42369995DE74EF1630C64AF94368300D154933D87B5EE885E9F"),
    "docs/reference_cases/e4_core_output.json": (2832, "F9C39E4E92D690F6FDECB756E114A30B579E4526FBDB84E73A260938E069BC14"),
    "docs/reference_cases/e4_core_source_map.json": (2758, "594C74AD59486AE6A23074079E610ED9E1625DA15B626A97BB98E31ED55F1EC1"),
    "docs/reference_cases/e4_environment.json": (1235, "EC3FE4B95C556F8CF6983083FD29EBA974D9FB953E33017CA3C37CBDC37B3B6F"),
    "docs/reference_cases/e4_pl_cases.json": (4064, "37D0BA2197246A8D752916EDF40BBDF8E946946E93756C1B042FE374DFF53B59"),
    "docs/reference_cases/e4_pl_contract.json": (3165, "9B3B2C151B2A910862F2D61ADBFACC8AEB72E7214E6A096B0C53BF2CF447A547"),
    "docs/reference_cases/e4_pl_oracle.py": (44814, "789B7DB6906ADD454B2C65001EDBA28F5AC9DBC3FB2AB598D3EE2F95D3F0D447"),
    "docs/reference_cases/e4_pl_output.json": (4920, "D6E0DA7E3300BCF691C87875B1FD3F215A6C70F611C52BB74A6579F20772E62D"),
    "docs/reference_cases/e4_pl_source_map.json": (3460, "8919A38DB727D5E863DA70161209C80F2A0F01851392A13A3080354F849B6B66"),
    "docs/reference_cases/e4_route_contract.json": (2374, "BD8F7C0FF49377224E9B2E9BE804B3423D6299898F66B5C117B0A744FC07A453"),
    "docs/reference_cases/e4_route_output.json": (1371, "796A33AC0C01645A72E28A94C43688126B2D14EBAAB2A0C40ABB3CAC05582461"),
    "docs/reference_cases/e4_source_registry.json": (3421, "66C395568FB4BCC90BCD57D9B8167E204C92A4390BBA643F3FA8516470CA4FA3"),
    "docs/reference_cases/e4_status.json": (4427, "4D72F7974FAFD2D3D738AB5B7F8FA962C82BCF9629F6C5A911A49D6CE3BE7EF1"),
    "docs/reference_cases/e4_test_inventory.json": (3581, "9B6F67242586BFC5A661D2790AFA8774254D68116871D8ACA4E3D1F126D220DA"),
    "docs/reference_cases/e4_ws_cases.json": (1224, "07C6ADC51095CF318EAECABECFF193FBB0CE2A2E2B604A9692F111BEBB8892E0"),
    "docs/reference_cases/e4_ws_contract.json": (3033, "AAB8E8ECC846FE652F9FAC1A2F02FC50B0AB8E179D808132AB3AF758BC7F82B8"),
    "docs/reference_cases/e4_ws_oracle.py": (12042, "10FA79C24BEE43E30653D69207E9A6F78C1F4D86238BABAD2B81266EFDC37985"),
    "docs/reference_cases/e4_ws_output.json": (1327, "9BBAF85E85734A36DA22B4780F450A5957CCF4ABE358BB24AEDAAB977C77057E"),
    "docs/reference_cases/e4_ws_source_map.json": (1204, "F421572687A814108C92F756DCCB7483429D0E67AA35268E4D6014F1BA9848E4"),
    "tests/test_e4_baseline.py": (9222, "0B7B4C6359517A966833BB441D57F610D3527F996923290BA9559BBE1EADA864"),
    "tests/test_e4_core_identity.py": (7937, "021A32350DDE166853A8D7BE85F98CF80A822D00C9C4EE774BFB995DED67691E"),
    "tests/test_e4_pl_identity.py": (8572, "667FE0E3676746776B1706FCE7903EB2BFEA6BE19625D4097BC420543672370B"),
    "tests/test_e4_route.py": (8516, "59A3DE6B6684B0808AF18FD1B68C07E9D3696437C2F1238C84B5133ED02003F7"),
    "tests/test_e4_ws_feasibility.py": (6383, "C42A69CDFCD766468485B53E600BE83B4373A399C09164A86168765F5CE0A1C4"),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def _strict_json(relative: str) -> dict[str, object]:
    raw = (ROOT / relative).read_bytes()
    assert (len(raw), _sha(raw)) == EXPECTED[relative]
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert raw == canonical
    return value


def _verify_record(record: dict[str, object]) -> None:
    relative = str(record["path"])
    raw = (ROOT / relative).read_bytes()
    assert len(raw) == record["bytes"] and _sha(raw) == record["sha256"]


def test_e4_closeout_is_content_addressed_canonical_and_fail_closed() -> None:
    json_documents: dict[str, dict[str, object]] = {}
    for relative, identity in EXPECTED.items():
        raw = (ROOT / relative).read_bytes()
        assert (len(raw), _sha(raw)) == identity
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
        if relative.endswith(".json"):
            json_documents[relative] = _strict_json(relative)

    status = json_documents["docs/reference_cases/e4_status.json"]
    assert status["authority"] == {
        "branch": "codex/s4-e4-variational-drill-closure",
        "commit": BASE_COMMIT,
        "tree": "e7e35bb880a88a8f7d736d32652c80442d8b9ec1",
    }
    assert status["components"]["core"]["terminal"] == "GO_E4_OPEN_CORE_IDENTITY"
    assert status["components"]["ws"] == {
        "broader_architectures": "UNCLASSIFIED_NEW_IDENTITY_REQUIRES_SEPARATE_PLAN",
        "contract": {
            "bytes": 3033,
            "path": "docs/reference_cases/e4_ws_contract.json",
            "sha256": "AAB8E8ECC846FE652F9FAC1A2F02FC50B0AB8E179D808132AB3AF758BC7F82B8",
        },
        "output": {
            "bytes": 1327,
            "path": "docs/reference_cases/e4_ws_output.json",
            "sha256": "9BBAF85E85734A36DA22B4780F450A5957CCF4ABE358BB24AEDAAB977C77057E",
        },
        "scope": "DIRECT_ADDITIVE_POST_CORE_IDENTITY_ONLY",
        "study_id": "study_e4_ws.wg2020_local_weak_symmetry_feasibility_v1",
        "terminal": "NO_GO_E4_WS_LOCAL_CONDENSATION_AND_RANK",
    }
    assert status["components"]["pl"]["terminal"] == (
        "PROVISIONAL_GO_E4_PL_LINEAR_QUALIFICATION_PLAN"
    )
    assert status["overall"] == {
        "immediate_successor": "docs/agent_plans/S4_E4_PL_LINEAR_QUALIFICATION_PLAN.md",
        "route_status": "PASSING_BRANCH_PRESENT",
        "route_terminal": "PROVISIONAL_GO_E4_PL_LINEAR_QUALIFICATION_PLAN",
        "run_status": "COMPLETE",
        "ws_new_identity_status": "UNCLASSIFIED_NEW_IDENTITY_REQUIRES_SEPARATE_PLAN",
    }
    assert status["production"] == {
        "candidate_registered": False,
        "legacy_shell_default": True,
        "production_changes": False,
        "public_api_changed": False,
        "selector_available": False,
        "serialization_changed": False,
        "terminal": "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED",
    }
    for component in status["components"].values():
        _verify_record(component["contract"])
        _verify_record(component["output"])
    for key, record in status["evidence"].items():
        if key == "tests":
            for test_record in record.values():
                _verify_record(test_record)
        else:
            _verify_record(record)

    review = (ROOT / "docs/E4_INDEPENDENT_REVIEW.md").read_text(encoding="utf-8")
    completion = (ROOT / "docs/E4_COMPLETION.md").read_text(encoding="utf-8")
    assert "ACCEPT_NO_P0_OR_P1" in review and "17 passed" in review
    assert EXPECTED["docs/reference_cases/e4_status.json"][1] in completion
    assert EXPECTED["docs/E4_INDEPENDENT_REVIEW.md"][1] in completion
    assert "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED" in completion


def test_e4_closeout_preserves_source_scope_and_no_production_boundary() -> None:
    sources = _strict_json("docs/reference_cases/e4_source_registry.json")
    assert set(sources["copyright_boundary"].values()) == {False}
    assert sources["sources"]["wt2011"]["access"] == (
        "PUBLIC_AUTHOR_UPLOAD_AND_USER_SUPPLIED_LAWFUL_LOCAL_COPY"
    )
    assert sources["sources"]["wt2011"]["license"] == (
        "COPYRIGHT_RESTRICTED_OR_UNCLEAR_REUSE"
    )
    assert sources["sources"]["mitc9i"]["role"] == (
        "P_SHELL_SCALAR_PL_NORMALIZATION_ONLY"
    )
    assert sources["sources"]["wg2020"]["license"] == "CC-BY-4.0"

    core = _strict_json("docs/reference_cases/e4_core_output.json")["certificate"]
    assert core["scope"]["core_classification_operator"] == "SOURCE_EXACT_WG_F_G_H_D_S_K5"
    assert core["scope"]["generic_I35_surrogate"] == "FORBIDDEN_NOT_USED"
    pl = _strict_json("docs/reference_cases/e4_pl_output.json")["certificate"]
    assert pl["identity"]["core_operator"] == "SOURCE_EXACT_WG_D_Q_K0"
    assert pl["identity"]["scalar_normalization"] == "MITC9i_18_19_WITH_GAMMA_EQUAL_G"

    permitted = {"argparse", "fractions", "hashlib", "json", "pathlib", "sys"}
    for name in ("core", "ws", "pl"):
        path = ROOT / f"docs/reference_cases/e4_{name}_oracle.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert imports <= permitted | {"__future__"}
        assert "anyfesolver" not in imports

    plan = (ROOT / "docs/agent_plans/S4_E4_PL_LINEAR_QUALIFICATION_PLAN.md").read_text(
        encoding="utf-8"
    )
    for grade in ("S235", "S275", "S355", "S420", "S460"):
        assert grade in plan
    assert "compatible with" in plan and "DNV analysis workflows" in plan
    assert "never DNV-approved" in plan
    assert "Density may be recorded" in plan and "no independent drilling inertia" in plan
    assert "range(T5)" in plan and "Direct nodal drilling/normal moments" in plan

    production = subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", "src", ".github", "pyproject.toml"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert production == ""


def test_e4_closeout_extent_is_exact_and_preserves_historical_roots() -> None:
    allowed = _strict_json("docs/reference_cases/e4_allowed_extent.json")
    assert allowed["modified_paths"] == [".gitattributes"]
    assert allowed["production_paths"] == []
    assert allowed["conditional_paths"][WS_SUCCESSOR] == (
        "ONLY_IF_PROVISIONAL_GO_E4_WS_STABILITY_QUALIFICATION_PLAN"
    )
    expected_new = set(allowed["new_paths"]) - {WS_SUCCESSOR}
    assert "tests/test_e4_closeout.py" in expected_new
    assert not (ROOT / WS_SUCCESSOR).exists()

    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "-c", "core.excludesFile=NUL", "ls-files", "--others", "--exclude-standard"],
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
        ".pytest_tmp_e4_closeout/",
    )
    candidate_paths = {path for path in observed if not path.startswith(preserved)}
    assert candidate_paths == expected_new | {".gitattributes"}
    assert not any(path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")) for path in candidate_paths)
    assert not any(path.startswith(("src/", ".github/", "pyproject.toml")) for path in candidate_paths)

    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for rule in (
        "docs/agent_plans/S4_E4_* text eol=lf",
        "docs/E4_* text eol=lf",
        "docs/reference_cases/e4_* text eol=lf",
        "tests/test_e4_* text eol=lf",
    ):
        assert rule in attrs
