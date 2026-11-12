from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/reference_cases/s4_improved_qualification_final_status.json"
REPORT = ROOT / "docs/S4_IMPROVED_QUALIFICATION_FINAL_STATUS.md"
RESTRICTED_CONTRACT = ROOT / "docs/reference_cases/s4_restricted_release_contract.json"

MANIFEST_SHA256 = "E3E8F3AA2DD6BA4193358AEDFC7F01889A80544313041531A1840218C09C29C1"
REPORT_SHA256 = "7F6760769CD2B91817B654985C0C0972285CA9AEEDB334853BD2A37F707ED994"
RESTRICTED_CONTRACT_LF_SHA256 = (
    "754AF29788B640E0FBD3806A89733A8E68065070E6366400F6B4D2476299E5B4"
)

UPSTREAM = {
    "candidate_a": (
        ROOT / "docs/reference_cases/s4_candidate_a_open_output.json",
        2644,
        "C42911E11BB1F1FA091F29FD0E3F5A3617310EF5F06C686E57C013171242B63C",
    ),
    "candidate_b": (
        ROOT / "docs/reference_cases/s4_stage_m_mechanics_output.json",
        5824196,
        "3A26052DB79CE914FF8A1FCA7835F3B86C15F1D351754B45CA904753D8EFDA0D",
    ),
    "candidate_c": (
        ROOT / "docs/reference_cases/s4_candidate_c_quotient_output.json",
        3701,
        "A44ED2DD5F11A0BBF9A0CB8D01B869A1D7E12632B3E85E773A804FC2CCC140B6",
    ),
    "rank_four": (
        ROOT / "docs/reference_cases/s4_drill_constraint_oracle_output.json",
        1434454,
        "8005C6D285263E33FF7F6D4B5138D5FBE4EFAB6A95834C401F94AF044ACD9E1B",
    ),
}

SOURCE_IDENTITIES = {
    ROOT / "src/anyfesolver/elements.py": (
        190422,
        "5D0AF716CD2E466EB831B1896553DE06236AC1BA80A84BD1B09C7E4CEBBDE670",
    ),
    ROOT / "src/anyfesolver/__init__.py": (
        24779,
        "0C782CCE93C1346F8A9B6DB832156A4F2689B33460C58F5966E9DE2169C2B8F0",
    ),
    ROOT / "src/anyfesolver/anystructure_fem_mode.py": (
        60057,
        "2FEACAAFD2A516B8ACCD919124A18614B0C3096EC0EEAF02DF6BC4280A80616E",
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


def _decode_json(raw: bytes) -> dict[str, object]:
    assert not raw.startswith(b"\xef\xbb\xbf")
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )


def _canonical_json(path: Path, expected_sha256: str) -> dict[str, object]:
    raw = path.read_bytes()
    assert _sha(raw) == expected_sha256
    assert b"\r" not in raw and raw.endswith(b"\n")
    value = _decode_json(raw)
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


def _canonical_lf(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    lf = raw.replace(b"\r\n", b"\n")
    assert b"\r" not in lf
    assert len(lf) == expected_bytes
    assert _sha(lf) == expected_sha256
    return lf


def _assignment(tree: ast.AST, name: str) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(matches) == 1
    return matches[0]


def test_s4_improved_qualification_closeout_is_canonical_and_fail_closed() -> None:
    manifest = _canonical_json(MANIFEST, MANIFEST_SHA256)
    report_raw = REPORT.read_bytes()
    assert _sha(report_raw) == REPORT_SHA256
    assert b"\r" not in report_raw and report_raw.endswith(b"\n")

    assert manifest["schema"] == "anysolver.s4.improved-qualification-final-status-v1"
    assert manifest["status"] == "complete"
    assert manifest["authority"] == {
        "accepted_base_commit": "2cb8c53cd1097380c872ba2802ec0eacc5198304",
        "accepted_base_tree": "f95d74e3ed1bb760f622e188f75f62a8b7ae43f6",
        "candidate_c_commit": "bdc040adb1e87ccd58443d72cbb224eb8d1fb8d6",
        "candidate_c_tree": "0842801fa32d62cf276f2e0f3cda4d6f3f4bfe2d",
    }
    assert manifest["verification"] == {
        "accepted_pre_closeout_test_count": 84,
        "closeout_test_count": 1,
        "required_combined_test_count": 85,
    }
    assert all(value is False for value in manifest["exclusions"].values())
    assert manifest["qualification_boundary"]["all_possible_formulations_exhausted"] is False
    assert manifest["qualification_boundary"]["new_program_required"] is True

    documents = {
        key: _canonical_json(path, digest)
        for key, (path, expected_bytes, digest) in UPSTREAM.items()
        if path.stat().st_size == expected_bytes
    }
    assert set(documents) == set(UPSTREAM)
    assert documents["candidate_a"]["pair_terminal"] == "NO_GO_CANDIDATE_A_DISCRETE_PAIR"
    assert documents["candidate_a"]["overall_release_terminal"] == (
        "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    )
    assert documents["candidate_b"]["candidate_terminal"] == "NO_GO_CANDIDATE_B"
    assert documents["candidate_b"]["overall_stage_m_status"] == (
        "BLOCKED_PRIMARY_SOURCE_UNAVAILABLE"
    )
    assert documents["candidate_c"]["candidate_terminal"] == (
        "NO_GO_CANDIDATE_C_QUOTIENT_INF_SUP"
    )
    assert documents["candidate_c"]["overall_release_terminal"] == (
        "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    )
    assert documents["rank_four"]["formulation_identity"] == (
        "mitc4_plus_d_published_2025_linear_spin_constrained_research_v1"
    )
    assert documents["rank_four"]["scientific_summary"]["outcome"] == (
        "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    )

    results = manifest["candidate_results"]
    for key, (path, expected_bytes, digest) in UPSTREAM.items():
        assert results[key]["output"] == {
            "bytes": expected_bytes,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest,
        }
    assert results["candidate_a"]["terminal"] == "NO_GO_CANDIDATE_A_DISCRETE_PAIR"
    assert results["candidate_b"]["terminal"] == "NO_GO_CANDIDATE_B"
    assert results["candidate_c"]["terminal"] == "NO_GO_CANDIDATE_C_QUOTIENT_INF_SUP"
    assert results["rank_four"]["terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"

    reviews = manifest["independent_reviews"]
    assert set(reviews) == {"candidate_a", "candidate_c"}
    for record in reviews.values():
        review_path = ROOT / record["path"]
        review_raw = review_path.read_bytes()
        assert not review_raw.startswith(b"\xef\xbb\xbf")
        assert b"\r" not in review_raw and review_raw.endswith(b"\n")
        assert len(review_raw) == record["bytes"]
        assert _sha(review_raw) == record["sha256"]

    production = manifest["production_boundary"]
    assert production["default_formulation_id"] == "anysolver.shell_element.legacy_s4"
    assert production["legacy_shell_default"] is True
    assert production["overall_release_terminal"] == "NO_GO_PRODUCTION_RESTRICTION_UNCHANGED"
    assert production["activation_available"] is False
    assert production["selector_available"] is False
    assert production["serialized_token_available"] is False
    assert production["public_api_changed"] is False

    source_bytes = {
        path: _canonical_lf(path, expected_bytes, digest)
        for path, (expected_bytes, digest) in SOURCE_IDENTITIES.items()
    }
    elements_tree = ast.parse(source_bytes[ROOT / "src/anyfesolver/elements.py"].decode("utf-8"))
    shell_class = [
        node
        for node in elements_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ShellElement"
    ]
    assert len(shell_class) == 1
    init = [
        node
        for node in shell_class[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert len(init) == 1
    positional = init[0].args.posonlyargs + init[0].args.args
    defaults = [None] * (len(positional) - len(init[0].args.defaults)) + list(
        init[0].args.defaults
    )
    default_by_name = {
        argument.arg: ast.literal_eval(default)
        for argument, default in zip(positional, defaults)
        if default is not None
    }
    assert "formulation" not in {argument.arg for argument in positional}
    assert default_by_name["drilling_stabilization"] == 1.0e-3
    assert default_by_name["reduced_integration"] is False
    assert default_by_name["hourglass_stabilization"] == 1.0e-8

    element_types = _assignment(elements_tree, "ELEMENT_TYPES").value
    assert isinstance(element_types, ast.Dict)
    shell_aliases = {
        ast.literal_eval(key)
        for key, value in zip(element_types.keys, element_types.values)
        if isinstance(value, ast.Name) and value.id == "ShellElement"
    }
    assert shell_aliases == {
        "shell",
        "shell3",
        "tri3",
        "tria3",
        "t3",
        "s3",
        "shell6",
        "tri6",
        "tria6",
        "t6",
        "s6",
    }

    package_tree = ast.parse(source_bytes[ROOT / "src/anyfesolver/__init__.py"].decode("utf-8"))
    exports = set(ast.literal_eval(_assignment(package_tree, "__all__").value))
    assert "ShellElement" in exports
    assert exports.isdisjoint(
        {
            "IMPROVED_RESEARCH_ID",
            "RESTRICTED_RELEASE_STATUS",
            "S4RestrictedReleaseStatus",
            "s4_restricted_policy",
        }
    )

    mode_tree = ast.parse(
        source_bytes[ROOT / "src/anyfesolver/anystructure_fem_mode.py"].decode("utf-8")
    )
    production_calls = [
        node
        for node in ast.walk(mode_tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "ShellElement")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "ShellElement")
        )
    ]
    assert production_calls
    assert all(
        keyword.arg != "formulation"
        for call in production_calls
        for keyword in call.keywords
    )

    restricted_lf = _canonical_lf(
        RESTRICTED_CONTRACT,
        4192,
        RESTRICTED_CONTRACT_LF_SHA256,
    )
    restricted = _decode_json(restricted_lf)
    assert restricted["default_formulation"] == {
        "changed": False,
        "id": "anysolver.shell_element.legacy_s4",
        "identity_kind": "descriptive_release_identity",
    }
    assert restricted["improved_formulation"]["production_activation_available"] is False
    assert restricted["improved_formulation"]["selector_available"] is False
    assert restricted["improved_formulation"]["serialized_token_available"] is False
    assert restricted["dormant_artifacts"]["production_dispatch_wired"] is False
    assert restricted["dormant_artifacts"]["root_exported"] is False
