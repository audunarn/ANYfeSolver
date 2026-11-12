from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
E0_COMMIT = "87b639499187736c59d87bc4aa8e6bd7f819d28b"

EXPECTED = {
    ".gitattributes": (1502, "1989E8C7412D004ADF6C4F819ED23AB4CDA5E932D7E7D0619D9C2B27122F1DDD"),
    "docs/agent_plans/S4_CANDIDATE_E1_ALLMAN_SESTRA_QUALIFICATION_PLAN.md": (
        5885,
        "16093C1B1E95AAC790E5AC0F4A6D19927782A0D24194108367B77BCDB5CA6BBE",
    ),
    "docs/S4_CANDIDATE_E1_A_DERIVATION.md": (
        2666,
        "BF40E075122C5F53DD7335F3A6FF3649393B5E25B98490DC389CCF2619B747E2",
    ),
    "docs/S4_CANDIDATE_E1_R_DERIVATION.md": (
        3867,
        "37B4C31FE326414339EE1EB9E8052161FF572DB13FB457FFEA71AEBAAF5322B1",
    ),
    "docs/S4_CANDIDATE_E1_QUALIFICATION_REPORT.md": (
        5586,
        "72DCCDDA0374946FB41DD3A47967196E025EAB9157959D1472D2EE488A0A30AA",
    ),
    "docs/S4_CANDIDATE_E1_A_INDEPENDENT_REVIEW.md": (
        11997,
        "9C1A18580763271B3D82E5F31879D9C9065FD672CDA662C0DB956BF8583D3391",
    ),
    "docs/S4_CANDIDATE_E1_R_INDEPENDENT_REVIEW.md": (
        11418,
        "4DAF53C81559B38368D84695DD9D7E96770E86823B5E62D213707F0F3F5DDF4E",
    ),
    "docs/reference_cases/s4_candidate_e1_baseline.json": (
        2622,
        "EA7E81C38912F14CB89CFD98302B6A8478D878939F7CFC1E3A60439667A745C1",
    ),
    "docs/reference_cases/s4_candidate_e1_environment.json": (
        1330,
        "F2DB5FF809FE0ED35ABE398FBFCECD133F2E8C36E96D1AB5C79354784F7216DE",
    ),
    "docs/reference_cases/s4_candidate_e1_source_registry.json": (
        2628,
        "C25197408932746D04C0651D082D5435369CEF94CFAF03BD3A12F8521A24B375",
    ),
    "docs/reference_cases/s4_candidate_e1_material_fixtures.json": (
        737,
        "F29886ED86AC83081E04D4A352D3F25BA304393DB5C0FA64A3BCF4338D4EFA07",
    ),
    "docs/reference_cases/s4_candidate_e1_test_inventory.json": (
        1751,
        "3290ACA0B30CD8C23A2508543DC8889D1F0795F38CF237AF7E826833E230EA16",
    ),
    "docs/reference_cases/s4_candidate_e1_a_identity.json": (
        802,
        "1A5D7A2E174A1BF7903DD4B188F56D7BDF2F1BC53639D3BE14FFFA5C010110FE",
    ),
    "docs/reference_cases/s4_candidate_e1_a_cases.json": (
        571,
        "F654F446ECDCED1F80FE86C092425D1AC95EA2F244FD0D20BEE80D52F95EE11A",
    ),
    "docs/reference_cases/s4_candidate_e1_a_oracle.py": (
        22724,
        "DBD69B6A3128848A100F3F76BC21BF3885D041CA2B17CA9EAEF0949E60A2EBEB",
    ),
    "docs/reference_cases/s4_candidate_e1_a_contract.json": (
        3877,
        "78ACB0EACC002B79C17A1E2C434FB890F64C7C178CA56493A7145F8E0EC5BFFA",
    ),
    "docs/reference_cases/s4_candidate_e1_a_output.json": (
        2397,
        "8022ECC3FB9D78637851EAF751044ABEE3C7E09D428160302450B726BD710788",
    ),
    "docs/reference_cases/s4_candidate_e1_r_identity.json": (
        1382,
        "201E8B7C33F055BF6BCC17CE2EB3FFDB5502C438013EB33419868990FACABA5E",
    ),
    "docs/reference_cases/s4_candidate_e1_r_cases.json": (
        1256,
        "695FBD1A4F07806444B26E3350F436FF9055A0816968ECFE65F20567B3B71EA9",
    ),
    "docs/reference_cases/s4_candidate_e1_r_oracle.py": (
        44176,
        "C45CE53597F5DC5A90B051B7BC336D8BD114A92ACFB20F1BF03A47C2117FA02E",
    ),
    "docs/reference_cases/s4_candidate_e1_r_contract.json": (
        4120,
        "9F3F19DD7BE8868D98E7B487FDD488DB9A77ACA429F12FB9824261551B6F7A4C",
    ),
    "docs/reference_cases/s4_candidate_e1_r_output.json": (
        5041,
        "ED26CF65363AD97BFA57234EA6CC7C708D8E94B4D477AAC64F5C6BAFB44B749B",
    ),
    "docs/reference_cases/s4_candidate_e1_status.json": (
        2230,
        "D9DDF6EFF2BC2A8C261F988BE9A7598867588D7BF78A1D0398EFE041C3CCC22D",
    ),
    "tests/test_s4_candidate_e1_a_exact_rank.py": (
        2715,
        "9D9FE91A1E77747215B9620D9CDFE0C13492BCE0F917696769E69B48847A7E6A",
    ),
    "tests/test_s4_candidate_e1_a_qualification.py": (
        5671,
        "B4E894462E1B3480EFB9A00BE8D92D374B145B36CFDA7AC4C834917084252529",
    ),
    "tests/test_s4_candidate_e1_r_exact_regularizer.py": (
        4005,
        "470F8734FE1E76BAAFD1F82289DE8BC48D9B1B8BD6350EABAB5EC0A0CA7C7318",
    ),
    "tests/test_s4_candidate_e1_r_qualification.py": (
        7718,
        "7253255EE7F28DCFE793ADE814640BDF94D14F8680B2AD55C5CD92393F132619",
    ),
}

NEW_PATHS = {
    "docs/agent_plans/S4_CANDIDATE_E1_ALLMAN_SESTRA_QUALIFICATION_PLAN.md",
    "docs/S4_CANDIDATE_E1_A_DERIVATION.md",
    "docs/S4_CANDIDATE_E1_R_DERIVATION.md",
    "docs/S4_CANDIDATE_E1_QUALIFICATION_REPORT.md",
    "docs/S4_CANDIDATE_E1_A_INDEPENDENT_REVIEW.md",
    "docs/S4_CANDIDATE_E1_R_INDEPENDENT_REVIEW.md",
    "docs/reference_cases/s4_candidate_e1_baseline.json",
    "docs/reference_cases/s4_candidate_e1_environment.json",
    "docs/reference_cases/s4_candidate_e1_source_registry.json",
    "docs/reference_cases/s4_candidate_e1_material_fixtures.json",
    "docs/reference_cases/s4_candidate_e1_test_inventory.json",
    "docs/reference_cases/s4_candidate_e1_a_identity.json",
    "docs/reference_cases/s4_candidate_e1_a_cases.json",
    "docs/reference_cases/s4_candidate_e1_a_oracle.py",
    "docs/reference_cases/s4_candidate_e1_a_contract.json",
    "docs/reference_cases/s4_candidate_e1_a_output.json",
    "docs/reference_cases/s4_candidate_e1_r_identity.json",
    "docs/reference_cases/s4_candidate_e1_r_cases.json",
    "docs/reference_cases/s4_candidate_e1_r_oracle.py",
    "docs/reference_cases/s4_candidate_e1_r_contract.json",
    "docs/reference_cases/s4_candidate_e1_r_output.json",
    "docs/reference_cases/s4_candidate_e1_status.json",
    "tests/test_s4_candidate_e1_a_exact_rank.py",
    "tests/test_s4_candidate_e1_a_qualification.py",
    "tests/test_s4_candidate_e1_r_exact_regularizer.py",
    "tests/test_s4_candidate_e1_r_qualification.py",
    "tests/test_s4_candidate_e1_closeout.py",
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
    "src/anyfesolver/fe_core.py": (
        17364,
        "553CE5A7C6FE86CD10D562A1B7683AF2CC84A7E43E77D201823651F3047B9EFD",
    ),
    "src/anyfesolver/shell_sections.py": (
        11530,
        "C9ECF93AB0E0A9B2A0D57A0252A21C4BF2A885551D5C96C64C1E2D3167919456",
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


def _strict_json(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, object]:
    raw = path.read_bytes()
    assert len(raw) == expected_bytes and _sha(raw) == expected_sha256
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


def _canonical_lf(path: Path, expected_bytes: int, expected_sha256: str) -> str:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    canonical = raw.replace(b"\r\n", b"\n")
    assert b"\r" not in canonical
    assert len(canonical) == expected_bytes and _sha(canonical) == expected_sha256
    return canonical.decode("utf-8")


def _assignment(tree: ast.AST, name: str) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(matches) == 1
    return matches[0]


def test_s4_candidate_e1_closeout_is_exact_separate_and_production_safe() -> None:
    documents: dict[str, dict[str, object]] = {}
    for relative, (size, digest) in EXPECTED.items():
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == size and _sha(raw) == digest
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw and raw.endswith(b"\n")
        if relative.endswith(".json"):
            documents[relative] = _strict_json(ROOT / relative, size, digest)

    a_contract = documents["docs/reference_cases/s4_candidate_e1_a_contract.json"]
    r_contract = documents["docs/reference_cases/s4_candidate_e1_r_contract.json"]
    a_output = documents["docs/reference_cases/s4_candidate_e1_a_output.json"]
    r_output = documents["docs/reference_cases/s4_candidate_e1_r_output.json"]
    status = documents["docs/reference_cases/s4_candidate_e1_status.json"]
    baseline = documents["docs/reference_cases/s4_candidate_e1_baseline.json"]
    inventory = documents["docs/reference_cases/s4_candidate_e1_test_inventory.json"]
    sources = documents["docs/reference_cases/s4_candidate_e1_source_registry.json"]
    materials = documents["docs/reference_cases/s4_candidate_e1_material_fixtures.json"]

    for contract in (a_contract, r_contract):
        assert set(contract["allowed_extent"]["new_paths"]) == NEW_PATHS
        assert contract["allowed_extent"]["modified"] == [".gitattributes"]
        assert contract["allowed_extent"]["production_paths"] == []
        for record in contract["input_identities"].values():
            relative = record["path"]
            size, digest = EXPECTED[relative]
            assert record == {"bytes": size, "path": relative, "sha256": digest}
    a_specific = {"cases", "derivation", "identity", "oracle"}
    for key in a_specific:
        assert a_contract["input_identities"][key] != r_contract["input_identities"][key]

    assert a_output["contract_sha256"] == EXPECTED[
        "docs/reference_cases/s4_candidate_e1_a_contract.json"
    ][1]
    assert a_output["candidate_terminal"] == "NO_GO_CANDIDATE_E1_A_RANK_DEFICIENCY"
    assert a_output["reason"] == "COMMON_DRILL_NULL_RANK_AT_MOST_17"
    assert a_output["certificate"]["rank_theorem"] == {
        "augmented_rigid_common_drill_rank": 7,
        "core_rank_upper_bound": 14,
        "full_rank_upper_bound": 17,
        "required_rank": 18,
    }
    assert a_output["e1_r_combined_or_used"] is False
    assert set(a_output["downstream_stages"].values()) == {
        "NOT_RUN_DUE_TO_EXACT_RANK_SCREEN",
        "NOT_IN_E1_SCOPE",
    }

    assert r_output["contract_sha256"] == EXPECTED[
        "docs/reference_cases/s4_candidate_e1_r_contract.json"
    ][1]
    assert r_output["candidate_terminal"] == (
        "PROVISIONAL_GO_CANDIDATE_E1_R_PLANAR_REGULARIZER_ONLY"
    )
    assert r_output["qualified_scope"] == {
        "fallback_pattern_only": True,
        "modal_or_transient": False,
        "physical_rank_18_element": False,
        "production_activation": False,
    }
    assert r_output["certificate"]["eligibility"]["current_legacy_host"] == {
        "e1_r_stiffness_applied": False,
        "reason": "EXISTING_LEGACY_DRILL_STIFFNESS",
        "status": "INELIGIBLE",
    }
    legacy_mass = r_output["certificate"]["mass"]["legacy_host"]
    assert legacy_mass["status"] == "INELIGIBLE_EXISTING_DRILL_MASS"
    assert legacy_mass["existing_positive_drill_rotary_inertia"] is True
    assert r_output["certificate"]["separation"] == {
        "e1_a_combined_or_used": False,
        "legacy_host_modified_or_used": False,
        "rank_18_claimed": False,
        "sestra_binary_reproduction_claimed": False,
    }

    assert status["candidates"]["e1_a"]["contract"]["sha256"] == a_output[
        "contract_sha256"
    ]
    assert status["candidates"]["e1_a"]["output"]["sha256"] == EXPECTED[
        "docs/reference_cases/s4_candidate_e1_a_output.json"
    ][1]
    assert status["candidates"]["e1_r"]["contract"]["sha256"] == r_output[
        "contract_sha256"
    ]
    assert status["candidates"]["e1_r"]["output"]["sha256"] == EXPECTED[
        "docs/reference_cases/s4_candidate_e1_r_output.json"
    ][1]
    assert status["relationship"] == {
        "combined_candidate": False,
        "e1_r_changes_e1_a_terminal": False,
        "residual_rank_or_mass_combination_authorized": False,
    }
    assert status["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert status["production"] == {
        "legacy_shell_default": True,
        "production_activation": False,
        "public_api_changed": False,
        "selector_available": False,
        "serialization_changed": False,
    }

    assert baseline["authority"]["e0_commit"] == E0_COMMIT
    assert baseline["authority"]["e0_tree"] == "c01fd5cab7b63325e6cb5b70000f4586d4788563"
    assert baseline["authority"]["production_qualification_base"] == (
        "a9b45ca95303bc4b30b893fbb0d7177f9c98db03"
    )
    accepted = inventory["accepted_pre_e1"]
    assert accepted["count"] == 94
    assert accepted["collection_order"] == (
        "inherited_85_then_e0_source_gate_8_then_e0_closeout_1"
    )
    assert accepted["node_ids_canonical_lf_bytes"] == 9842
    assert accepted["node_ids_canonical_lf_sha256"] == (
        "29EF584E9B51E8420934A519B3C1E71BDD3082EFDC89DBADA4FCE0FFE8997B9F"
    )
    assert inventory["composition"]["inherited_85"]["node_count"] == 85
    assert inventory["composition"]["e0_source_gate"]["node_count"] == 8
    assert inventory["composition"]["e0_closeout"]["node_count"] == 1

    assert sources["copyright_boundary"] == {
        "committed_manual_content": False,
        "committed_page_images": False,
        "committed_quotations": False,
        "permitted_record": "identity_page_map_and_independently_derived_equations_only",
    }
    assert sources["sources"]["sestra_86_manual"]["role"].startswith(
        "technical_evidence_"
    )
    assert sources["sources"]["sestra_86_manual"]["sha256"] == (
        "68E904E8E1B6800BC04FAA60299E17DEB29E3AB79B1823E09A7DD2F1C02FB1F3"
    )
    assert sources["sources"]["sestra_11_manual"]["sha256"] == (
        "A7F0D3C4135B9ADC025229F3A91C1FB60E755F3CE3F176EA0EF4B6D7555A6334"
    )
    assert sources["sources"]["attached_candidate_e_plan"]["sha256"] == (
        "4499DA192F97D9BF7D89C3A9A8B5A68E6201CA5E2350E30918583464BF0E98EA"
    )
    assert set(sources["excluded_hypotheses"]) == {
        "k_D_sqrt_det_As0",
        "j_D_rho_A_ell_squared",
        "absolute_drill_to_ground_diagonal",
        "e1_a_plus_e1_r_without_successor_plan",
    }
    assert not any(path.lower().endswith((".pdf", ".png", ".jpg")) for path in NEW_PATHS)

    assert materials["rp_c208"] == {
        "classification": "recommended_practice_not_class_rule",
        "grades": ["S235", "S275", "S355", "S420", "S460"],
        "row_count": 17,
        "source_edition": "September_2019_amended_October_2022",
    }
    assert materials["compatibility"] == {
        "dnv_approval": False,
        "new_public_fields": [],
        "reporting": "compatible_with_DNV_analysis_workflows",
        "ru_ship_project_edition": "July_2025",
        "ru_ship_records_in_anymaterial": False,
    }

    for review in (
        "docs/S4_CANDIDATE_E1_A_INDEPENDENT_REVIEW.md",
        "docs/S4_CANDIDATE_E1_R_INDEPENDENT_REVIEW.md",
    ):
        text = (ROOT / review).read_text(encoding="utf-8")
        assert "ACCEPT" in text and "No P0 or P1" in text
        assert "94" in text and "15" in text

    source_text: dict[str, str] = {
        relative: _canonical_lf(ROOT / relative, size, digest)
        for relative, (size, digest) in PRODUCTION.items()
    }
    elements_text = source_text["src/anyfesolver/elements.py"]
    elements_tree = ast.parse(elements_text)
    shell = [
        node
        for node in elements_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ShellElement"
    ]
    assert len(shell) == 1
    init = [
        node
        for node in shell[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    mass = [
        node
        for node in shell[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_mass_matrix"
    ]
    assert len(init) == len(mass) == 1
    positional = init[0].args.posonlyargs + init[0].args.args
    defaults = [None] * (len(positional) - len(init[0].args.defaults)) + list(
        init[0].args.defaults
    )
    defaults_by_name = {
        argument.arg: ast.literal_eval(default)
        for argument, default in zip(positional, defaults)
        if default is not None
    }
    assert defaults_by_name["drilling_stabilization"] == 1.0e-3
    assert "formulation" not in {argument.arg for argument in positional}
    mass_text = ast.get_source_segment(elements_text, mass[0])
    assert mass_text is not None
    assert "for d in range(3):" in mass_text
    assert "M_local[3 + d::6, 3 + d::6] += rotational" in mass_text
    shell_text = ast.get_source_segment(elements_text, shell[0])
    assert shell_text is not None
    assert 'getattr(self, "drilling_stabilization", 1.0e-3)' in shell_text
    assert "B_d.T @ (drilling_stiffness * np.eye(1)) @ B_d" in shell_text

    package_tree = ast.parse(source_text["src/anyfesolver/__init__.py"])
    exports = set(ast.literal_eval(_assignment(package_tree, "__all__").value))
    assert not any("E1" in name or "candidate_e1" in name.lower() for name in exports)

    diff = subprocess.run(
        ["git", "diff", "--name-only", E0_COMMIT],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    untracked = subprocess.run(
        [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "ls-files",
            "--others",
            "--exclude-standard",
        ],
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
    observed = set(diff.stdout.decode().splitlines()) | set(
        untracked.stdout.decode().splitlines()
    )
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
        path.startswith(("src/", ".github/", "pyproject.toml"))
        for path in candidate_paths
    )
