from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "docs/E4_OPEN_CORE_IDENTITY.md": (6688, "BAFC21DC85C0CD9101C30ACC5D84F4BC57F3394EA4C0AEFB31CCFA7E43655E5D"),
    "docs/E4_WS_FEASIBILITY_THEOREM.md": (3839, "80E02F2564D6DE6D5E1A66857A78D97FCA83C828AEAB20BFE4707408EAD7BF19"),
    "docs/E4_PL_VARIATIONAL_CLOSURE.md": (8302, "14BDA35109FE8C653B85BF890C36CC454CDE938BCDC3820CA39479EC620EFB4D"),
    "docs/E4_ROUTE_REPORT.md": (5696, "0DA0B9ED4F604BD8D476E289102B0D44779EFB73F3F1BDD2BA26AB21CED9FF2D"),
    "docs/agent_plans/S4_E4_PL_LINEAR_QUALIFICATION_PLAN.md": (4670, "912322A8158255F17DDA44A3BB8FD59EFF1FC3B6B1E9D6BBB22B4E49A72BD193"),
    "docs/reference_cases/e4_core_contract.json": (2284, "8768ABDC5D77B5FCC1643AF69920EFEB83F27DDFBDE17525F06EE2B53A5C1678"),
    "docs/reference_cases/e4_core_output.json": (2832, "F9C39E4E92D690F6FDECB756E114A30B579E4526FBDB84E73A260938E069BC14"),
    "docs/reference_cases/e4_ws_contract.json": (3033, "AAB8E8ECC846FE652F9FAC1A2F02FC50B0AB8E179D808132AB3AF758BC7F82B8"),
    "docs/reference_cases/e4_ws_output.json": (1327, "9BBAF85E85734A36DA22B4780F450A5957CCF4ABE358BB24AEDAAB977C77057E"),
    "docs/reference_cases/e4_pl_contract.json": (3165, "9B3B2C151B2A910862F2D61ADBFACC8AEB72E7214E6A096B0C53BF2CF447A547"),
    "docs/reference_cases/e4_pl_output.json": (4920, "D6E0DA7E3300BCF691C87875B1FD3F215A6C70F611C52BB74A6579F20772E62D"),
    "docs/reference_cases/e4_route_contract.json": (2374, "BD8F7C0FF49377224E9B2E9BE804B3423D6299898F66B5C117B0A744FC07A453"),
    "docs/reference_cases/e4_route_output.json": (1371, "796A33AC0C01645A72E28A94C43688126B2D14EBAAB2A0C40ABB3CAC05582461"),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def _load(relative: str) -> dict[str, object]:
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


def test_e4_component_contracts_outputs_and_hash_dag_are_exact() -> None:
    documents: dict[str, dict[str, object]] = {}
    for relative, identity in EXPECTED.items():
        raw = (ROOT / relative).read_bytes()
        assert (len(raw), _sha(raw)) == identity
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
        if relative.endswith(".json"):
            documents[relative] = _load(relative)

    pairs = [
        ("core", "GO_E4_OPEN_CORE_IDENTITY"),
        ("ws", "NO_GO_E4_WS_LOCAL_CONDENSATION_AND_RANK"),
        ("pl", "PROVISIONAL_GO_E4_PL_LINEAR_QUALIFICATION_PLAN"),
    ]
    for name, terminal in pairs:
        contract_path = f"docs/reference_cases/e4_{name}_contract.json"
        output_path = f"docs/reference_cases/e4_{name}_output.json"
        contract = documents[contract_path]
        output = documents[output_path]
        assert contract["scientific_terminal"] == terminal
        assert output["terminal"] == terminal
        assert output["contract_sha256"] == EXPECTED[contract_path][1]
        assert output["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
        assert output["production_changed"] is False
        for record in contract["input_identities"].values():
            _verify_record(record)

    core = documents["docs/reference_cases/e4_core_output.json"]["certificate"]
    assert core["embedding"]["embedded_rank"] == 14
    assert core["embedding"]["embedded_nullity"] == 10
    assert core["scope"]["core_classification_operator"] == "SOURCE_EXACT_WG_F_G_H_D_S_K5"
    assert core["scope"]["generic_I35_surrogate"] == "FORBIDDEN_NOT_USED"
    for case in core["geometries"].values():
        assert case["ranks"] == {"D": 35, "F": 14, "Gq": 14, "H": 21, "K5": 14}
        assert case["Gq_B_row_equivalent"] is True
    assert core["coordinate_split"] == {
        "QD_T_QD_I4": True,
        "T5_T_QD_zero": True,
        "T5_T_T5_I20": True,
        "complete_I24": True,
    }
    ws = documents["docs/reference_cases/e4_ws_output.json"]["certificate"]
    assert ws["local_condensation"]["finite_multiplier_schur_exists"] is False
    assert ws["exact_witness"]["K0_rank"] == 14
    assert ws["exact_witness"]["prohibited_CtC_completion_rank"] == 18
    pl = documents["docs/reference_cases/e4_pl_output.json"]["certificate"]
    for case in pl["geometries"].values():
        assert (case["retained_constraint_rank"], case["hourglass_rank"]) == (3, 1)
        assert (case["rank"], case["nullity"]) == (18, 6)
        assert all(case["rigid_images_zero"].values())
        assert set(case["covariance"].values()) == {True, 8}


def test_e4_route_selects_only_the_conditional_pl_plan() -> None:
    contract = _load("docs/reference_cases/e4_route_contract.json")
    output = _load("docs/reference_cases/e4_route_output.json")
    assert output["contract_sha256"] == EXPECTED["docs/reference_cases/e4_route_contract.json"][1]
    for record in contract["input_identities"].values():
        _verify_record(record)

    assert output["components"] == {
        "core": {
            "study_id": "study_e4_core.wg2020_n7_k0_full_integration_reference_v1",
            "terminal": "GO_E4_OPEN_CORE_IDENTITY",
        },
        "pl": {
            "study_id": "study_e4_pl.wg2020_surface_reduced_perturbed_lagrange_v1",
            "terminal": "PROVISIONAL_GO_E4_PL_LINEAR_QUALIFICATION_PLAN",
        },
        "ws": {
            "scope": "DIRECT_ADDITIVE_POST_CORE_IDENTITY_ONLY",
            "study_id": "study_e4_ws.wg2020_local_weak_symmetry_feasibility_v1",
            "terminal": "NO_GO_E4_WS_LOCAL_CONDENSATION_AND_RANK",
        },
    }
    assert output["route"]["terminal"] == "PROVISIONAL_GO_E4_PL_LINEAR_QUALIFICATION_PLAN"
    assert output["route"]["authorization"] == (
        "EXECUTE_ONLY_SEPARATELY_REVIEWED_E4_PL_LINEAR_QUALIFICATION_PLAN"
    )
    _verify_record(output["route"]["conditional_successor"])
    assert output["route"]["ws_followup"] == "NOT_AUTHORIZED_WITHIN_E4_0_IDENTITY"
    assert output["route"]["broader_ws"] == (
        "UNCLASSIFIED_NEW_IDENTITY_REQUIRES_SEPARATE_PLAN"
    )
    assert contract["route_rule"]["ws_scope"] == "DIRECT_ADDITIVE_POST_CORE_IDENTITY_ONLY"
    assert not (ROOT / "docs/agent_plans/S4_E4_WS_STABILITY_QUALIFICATION_PLAN.md").exists()
    assert output["production"] == {
        "candidate_registered": False,
        "legacy_shell_default": True,
        "production_changes": False,
        "public_api_changed": False,
        "selector_available": False,
        "serialization_changed": False,
    }
    assert output["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"


def test_e4_oracles_are_closed_world_standard_library_programs() -> None:
    permitted = {
        "argparse",
        "fractions",
        "hashlib",
        "json",
        "pathlib",
        "sys",
    }
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
        source = path.read_text(encoding="utf-8")
        assert "BLOCKED_E4_CONTRACT_NONDETERMINISM_OR_REVIEW" in source
        assert "allow_nan=False" in source
