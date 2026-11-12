from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import anyfesolver
from anyfesolver.elements import ShellElement
from anyfesolver.shell_formulations import s4_restricted_policy as policy


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "reference_cases" / "s4_restricted_release_contract.json"
STATUS_PATH = ROOT / "docs" / "S4_RESTRICTED_RELEASE_STATUS.md"

EXPECTED_SQUARE = (
    ("rank_B", 16),
    ("N", 8),
    ("G", 1),
    ("P", 7),
    ("R", 6),
    ("R_N", 6),
    ("R_G", 0),
    ("RQ", 6),
    ("Z", 1),
)

EXPECTED_REASONS = (
    "s4_improved.research_only",
    "s4_improved.positive_mass_zero_stiffness_z",
    "s4_improved.threshold_sensitive_geometry",
    "s4_improved.coupling_unqualified",
    "s4_improved.nonlinear_unqualified",
    "s4_improved.geometric_stiffness_unqualified",
    "s4_improved.buckling_unqualified",
    "s4_improved.recovery_unqualified",
    "s4_improved.optimized_batches_unqualified",
    "s4_improved.provenance_unavailable_or_unqualified",
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_release_contract_matches_passive_python_status() -> None:
    contract = _contract()
    status = policy.RESTRICTED_RELEASE_STATUS

    assert policy.RELEASE_CONTRACT_SCHEMA == "anysolver.s4.restricted-release"
    assert policy.RELEASE_CONTRACT_VERSION == 1
    assert status.default_formulation_id == "anysolver.shell_element.legacy_s4"
    assert status.improved_research_id == (
        "mitc4_plus_d_published_2025_eq21_eq25_reference_v2"
    )
    assert status.production_activation_available is False
    assert status.gauge_constraint_applied is False
    assert status.square_nullspace == EXPECTED_SQUARE
    assert status.restricted_reason_codes == EXPECTED_REASONS

    assert contract["schema"] == policy.RELEASE_CONTRACT_SCHEMA
    assert contract["version"] == policy.RELEASE_CONTRACT_VERSION
    assert contract["default_formulation"]["changed"] is False
    assert contract["improved_formulation"]["production_activation_available"] is False
    assert tuple(contract["improved_formulation"]["restricted_reason_codes"]) == EXPECTED_REASONS
    assert contract["gauge_policy"] == {
        "constraint_applied": False,
        "definition": "G=ker(B_w) intersection ker(H_w)",
        "removal_action": "none",
    }
    assert contract["square_nullspace"] == {
        "rank_B": 16,
        "N": 8,
        "G": 1,
        "P": 7,
        "R": 6,
        "R_N": 6,
        "R_G": 0,
        "RQ": 6,
        "Z": 1,
        "z_classification": "positive_mass_zero_stiffness_not_gauge",
    }


def test_legacy_shell_defaults_and_root_exports_are_unchanged() -> None:
    signature = inspect.signature(ShellElement)
    assert signature.parameters["drilling_stabilization"].default == 1.0e-3
    assert signature.parameters["reduced_integration"].default is False
    assert signature.parameters["hourglass_stabilization"].default == 1.0e-8
    assert "formulation" not in signature.parameters

    forbidden = {
        "IMPROVED_RESEARCH_ID",
        "RESTRICTED_RELEASE_STATUS",
        "S4RestrictedReleaseStatus",
        "s4_restricted_policy",
    }
    assert forbidden.isdisjoint(anyfesolver.__all__)
    assert all(not hasattr(anyfesolver, name) for name in forbidden)


def test_policy_is_stdlib_only_and_not_wired_to_hot_paths() -> None:
    policy_path = Path(policy.__file__).resolve()
    tree = ast.parse(policy_path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports <= {"__future__", "dataclasses"}

    for relative in (
        "src/anyfesolver/elements.py",
        "src/anyfesolver/assembly.py",
        "src/anyfesolver/matrix_assembly.py",
        "src/anyfesolver/fe_core.py",
        "src/anyfesolver/activity.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "s4_restricted_policy" not in source
        assert "IMPROVED_RESEARCH_ID" not in source


def test_dormant_evidence_and_release_warning_are_present() -> None:
    contract = _contract()
    for relative in contract["dormant_artifacts"]["modules"]:
        assert (ROOT / relative).is_file()
    for relative in contract["dormant_artifacts"]["scripts"]:
        assert (ROOT / relative).is_file()
    assert contract["dormant_artifacts"]["root_exported"] is False
    assert contract["dormant_artifacts"]["production_dispatch_wired"] is False

    evidence = contract["accepted_evidence"]
    assert evidence["integration_base"]["commit"] == "4db31b633d0f886fcb4ad82946a982eb6fadde0e"
    assert evidence["nullspace_proof"]["commit"] == "cfaf9c7a6e51e1cc0c3113648f84835e917fca2a"
    assert evidence["geometry_handoff"]["commit"] == "931ed76943dc84fb9d01b26a5d6dd4c46af3d74a"
    assert evidence["batch_qualification"]["commit"] == "89ea46d8c1b1365b1d4a390ce6f34e2609c434f9"

    status_doc = STATUS_PATH.read_text(encoding="utf-8")
    assert "not reported as passed" in status_doc
    assert "superseded" in status_doc
    for reason in EXPECTED_REASONS:
        assert reason in status_doc
