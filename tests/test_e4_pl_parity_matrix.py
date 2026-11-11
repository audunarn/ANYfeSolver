from __future__ import annotations

import ast
import json
from pathlib import Path

from anyfesolver import QualifiedE4PLShellElement
from anyfesolver.elements import LegacyShellElement, create_element


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "reference_cases" / "e4_pl_q1j_parity_matrix.json"


def _public_shell_api() -> set[str]:
    tree = ast.parse((ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"Element", "ShellElement"}:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                ):
                    result.add(child.name)
    return result


def test_parity_matrix_classifies_every_public_shell_capability() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert matrix["schema"] == "anysolver.s4.e4-pl-q1m-parity-matrix-v3"
    assert set(matrix["public_api"]) == _public_shell_api()
    capabilities = matrix["capabilities"]
    assert len({row["id"] for row in capabilities}) == len(capabilities)
    required = {row["id"]: row for row in capabilities if row["required"]}
    for capability in (
        "warped_curved_geometry",
        "generalized_shell_sections",
        "geometric_stiffness_and_buckling",
        "nonlinear_geometry",
        "mass_and_dynamics",
        "contact_and_coupling",
        "physical_recovery",
        "deletion_damage_and_restart_state",
        "batched_assembly_and_performance",
        "performance_and_verification_script_routing",
        "ecosystem_anyfem_default_routing",
        "factory_and_default_activation",
    ):
        assert capability in required


def test_every_required_parity_item_is_closed_at_default_activation() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    open_markers = ("PENDING", "BLOCKED", "LEGACY_FALLBACK")
    open_required = [
        row for row in matrix["capabilities"]
        if row["required"] and any(marker in row["status"] for marker in open_markers)
    ]
    assert open_required == []
    assert matrix["activation"] == {
        "burn_in_contract": "Q1M_TWO_CLEAN_RELEASE_GATES",
        "default_factory": "QUALIFIED_E4_PL_Q4",
        "explicit_legacy_rollback": "DEPRECATED_AVAILABLE_THROUGH_0.4.x",
        "legacy_q4_removal_not_before": "0.5.0",
        "legacy_shell_default": False,
        "public_package_export": True,
        "qualified_element_factory_alias": True,
    }
    default = create_element("shell", 1, [1, 2, 3, 4], "steel")
    rollback = create_element("legacy-shell", 2, [1, 2, 3, 4], "steel")
    assert type(default) is QualifiedE4PLShellElement
    assert type(rollback) is LegacyShellElement
    assert QualifiedE4PLShellElement.__module__ == "anyfesolver.e4_pl_element"


def test_warped_and_performance_rows_are_closed_before_activation() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in matrix["capabilities"]}
    assert by_id["warped_curved_geometry"]["status"] == (
        "QUALIFIED_DIRECT_VARYING_FRAME_Q4"
    )
    assert by_id["batched_assembly_and_performance"]["status"] == (
        "SHARED_GEOMETRY_COLD_WARM_AND_NONLINEAR_BATCH_PARITY_TESTED"
    )
    assert by_id["factory_and_default_activation"]["status"] == (
        "ACTIVATED_QUALIFIED_E4_PL_Q4_WITH_DEPRECATED_LEGACY_ROLLBACK"
    )


def test_runtime_shell_dispatch_accepts_subclasses_without_exact_name_checks() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in matrix["capabilities"]}
    assert by_id["runtime_exact_class_name_dispatch"]["status"] == "CLOSED_ISINSTANCE_DISPATCH"
    runtime_source = (ROOT / "src" / "anyfesolver" / "runtime.py").read_text(encoding="utf-8")
    test_case_source = (ROOT / "src" / "anyfesolver" / "test_cases.py").read_text(encoding="utf-8")
    assert '.__class__.__name__ != "ShellElement"' not in runtime_source
    assert '.__class__.__name__ == "ShellElement"' not in runtime_source
    assert ".__class__.__name__ == 'ShellElement'" not in test_case_source
    assert "isinstance(element, _full_backend.ShellElement)" in runtime_source
    assert "isinstance(element, ShellElement)" in test_case_source
