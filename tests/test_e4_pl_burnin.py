from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import pytest

from scripts import run_portable_ci as portable_ci

from anyfesolver import (
    LEGACY_Q4_AVAILABLE_THROUGH,
    LEGACY_Q4_REMOVAL_TARGET,
    LegacyQ4DeprecationWarning,
    NativeParityE4PLS3V2DShellElement,
    QualifiedE4PLShellElement,
    ShellElement,
    create_element,
    create_shell_element,
    shell_formulation_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "reference_cases" / "e4_pl_q1m_burnin_contract.json"
CONTRACT_CYCLE0 = (
    ROOT / "docs" / "reference_cases" / "e4_pl_q1m_burnin_contract_cycle0.json"
)
CONTRACT_CYCLE1 = (
    ROOT / "docs" / "reference_cases" / "e4_pl_q1m_burnin_contract_cycle1.json"
)
CONTRACT_CYCLE2 = (
    ROOT / "docs" / "reference_cases" / "e4_pl_q1m_burnin_contract_cycle2.json"
)
CONTRACT_CYCLE3 = (
    ROOT / "docs" / "reference_cases" / "e4_pl_q1m_burnin_contract_cycle3.json"
)
CONTRACT_CYCLE4 = (
    ROOT / "docs" / "reference_cases" / "e4_pl_q1m_burnin_contract_cycle4.json"
)


def _gate_module():
    path = ROOT / "scripts" / "run_e4_pl_burnin_gate.py"
    spec = importlib.util.spec_from_file_location("e4_pl_burnin_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prospective_gate_result(gate) -> dict[str, object]:
    contract = gate.strict_json_load(CONTRACT)
    sibling_authority = contract.get("sibling_authority")
    sibling_commits = (
        {
            name: sibling_authority[name]["commit"]
            for name in gate.SIBLING_NAMES
        }
        if sibling_authority is not None
        else {
            "ANYfem": contract["anyfem_commit"],
            "ANYmesh": "2" * 40,
            "ANYgeometry": "3" * 40,
            "ANYmaterial": "4" * 40,
            "ANYfileIO": "5" * 40,
        }
    )
    request_rows = [
        {
            "lane": lane,
            "request_id": request["request_id"],
            "request_sha256": request["request_sha256"],
        }
        for lane, request in sorted(contract["resource_requests"].items())
    ]
    lanes = {}
    for lane, paths in gate.gate_inventories().items():
        request_id = (
            contract["resource_requests"][lane]["request_id"]
            if lane in contract["resource_requests"]
            else None
        )
        lanes[lane] = {
            "command_sha256": (
                contract["non_resource_commands"][lane]["command_sha256"]
                if lane in contract["non_resource_commands"]
                else contract["resource_requests"][lane]["command_sha256"]
            ),
            "exit_code": 0,
            "finished_at_utc": "2026-08-24T10:01:00Z",
            "inventory": paths,
            "log": {
                "bytes": len(lane),
                "sha256": hashlib.sha256(lane.encode("ascii")).hexdigest(),
            },
            "resource_request_id": request_id,
            "started_at_utc": "2026-08-24T10:00:00Z",
            "status": "PASS",
        }
    return {
        "candidate": {
            "clean": True,
            "commit": "1" * 40,
            "repository": "ANYfeSolver",
            "tree": "a" * 40,
        },
        "clean_gate_index": 1,
        "hard_gates": {
            name: {
                "evidence_nodes": nodes,
                "observed": True,
                "status": "PASS",
            }
            for name, nodes in sorted(contract["hard_performance_gates"].items())
        },
        "lanes": lanes,
        "legacy_removal_authorized": False,
        "package_result": {"bytes": 1, "sha256": "6" * 64},
        "performance_baseline": {
            "measurements": {
                name: {
                    "mad_ns": 3,
                    "median_ns": 6,
                    "p95_ns": 11,
                    "samples_ns": list(range(1, 12)),
                }
                for name in contract["performance_baseline"]["measurement_names"]
            },
            "repetitions": 11,
            "schema": "anysolver.s4.e4-pl-q1m-performance-baseline-v1",
            "speed_claim": "GATE_1_BASELINE_ONLY_NO_SPEED_CLAIM",
            "warmups": 1,
        },
        "production_boundary": contract["production_boundary"],
        "resource_requests": request_rows,
        "rollback": {
            "state": "NO_UNRESOLVED_ROLLBACK_INCIDENT",
            "unresolved_incidents": [],
        },
        "schema": contract["gate_result_schema"],
        "siblings": {
            name: {
                "clean": True,
                "commit": commit,
                "tree": (
                    sibling_authority[name]["tree"]
                    if sibling_authority is not None
                    else "b" * 40
                ),
            }
            for name, commit in sibling_commits.items()
        },
        "wheel": {
            "bytes": 123,
            "filename": "anysolver-0.3.0-py3-none-any.whl",
            "sha256": "7" * 64,
        },
    }


def _prospective_package_result(gate, gate_result) -> dict[str, object]:
    contract = gate.strict_json_load(CONTRACT)
    content = b"record\n"
    log = {
        "bytes": len(content),
        "returncode": 0,
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    empty_log = {
        "bytes": 0,
        "returncode": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    names = [name for name, _distribution, _package in gate.LOCAL_DISTRIBUTIONS]
    identities = {
        "ANYfeSolver": gate_result["candidate"],
        **{name: gate_result["siblings"][name] for name in names if name != "ANYfeSolver"},
    }
    return {
        "build_logs": {name: copy.deepcopy(log) for name in names},
        "install_log": copy.deepcopy(log),
        "schema": contract["package_result_schema"],
        "smoke": {
            "diagnostics_schema": contract["diagnostics_schema"],
            "legacy_warning": contract["legacy_q4"]["warning"],
            "non_q4_types": ["ShellElement"] * 4,
            "origins": {
                package: f"{package}/__init__.py"
                for _name, _distribution, package in gate.LOCAL_DISTRIBUTIONS
            },
            "q4_type": "QualifiedE4PLShellElement",
        },
        "smoke_log": {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()},
        "sources": {
            name: {
                "archive": {"bytes": 1, "sha256": "a" * 64},
                "archive_log": copy.deepcopy(empty_log),
                "commit": identities[name]["commit"],
                "content": {"files": 1, "sha256": "b" * 64},
                "tree": identities[name]["tree"],
            }
            for name in names
        },
        "status": "PASS",
        "wheels": {
            name: {
                "bytes": 123,
                "filename": f"{distribution.lower()}-1-py3-none-any.whl",
                "sha256": "c" * 64,
            }
            for name, distribution, _package in gate.LOCAL_DISTRIBUTIONS
        },
    }


def _blocked_at(gate, record, failed_lane: str, incident: str):
    failed_index = gate.GATE_LANES.index(failed_lane)
    for index, lane in enumerate(gate.GATE_LANES):
        lane_record = record["lanes"][lane]
        if index == failed_index:
            lane_record.update({"exit_code": 1, "status": "FAIL"})
        elif index > failed_index:
            record["lanes"][lane] = {
                key: lane_record[key]
                for key in ("command_sha256", "inventory", "resource_request_id", "status")
            }
            record["lanes"][lane]["status"] = "NOT_RUN"
    if record["lanes"]["performance"]["status"] != "PASS":
        record.pop("hard_gates", None)
        record.pop("performance_baseline", None)
    if record["lanes"]["package"]["status"] != "PASS":
        record.pop("package_result", None)
        record.pop("wheel", None)
    record["clean_gate_index"] = 0
    record["rollback"] = {
        "state": "UNRESOLVED_LEGACY_Q4_ROLLBACK_INCIDENT",
        "unresolved_incidents": [incident],
    }
    return record


def test_burn_in_contract_and_runtime_diagnostics_are_aligned() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == "anysolver.s4.e4-pl-q1m-burn-in-contract-v1"
    assert contract["default"] == "QUALIFIED_E4_PL_Q4"
    assert contract["burn_in"] == {
        "clean_release_gates_required": 2,
        "counter_reset": "ANY_UNRESOLVED_LEGACY_Q4_ROLLBACK_INCIDENT",
        "earliest_removal_release": LEGACY_Q4_REMOVAL_TARGET,
        "legacy_q4_available_through": LEGACY_Q4_AVAILABLE_THROUGH,
        "state": "ACTIVE",
    }
    assert contract["evidence_lanes"]["ecosystem"] == (
        "ANYFEM_PUBLIC_SELECTOR_AND_ENGINEERING_REFERENCE"
    )
    assert contract["evidence_lanes"]["package"] == (
        "WHEEL_BUILD_AND_EXTRACTED_INSTALL_SMOKE"
    )
    assert contract["gate_result_schema"] == "anysolver.s4.e4-pl-q1m-gate-result-v2"
    assert contract["package_result_schema"] == "anysolver.s4.e4-pl-q1m-package-lane-v2"
    assert contract["adjudication"] == {
        "accepted_blocked_verdict": "ACCEPT_Q1M_CORRECTION_5_BLOCKED_GATE_NO_P0_P1",
        "accepted_success_verdict": "ACCEPT_Q1M_BURN_IN_GATE_1_NO_P0_P1",
        "blocked_commit_subject": "docs: record E4 PL Q1M correction-5 blocked gate",
        "blocked_terminal": "BLOCKED_E4_PL_Q1M_CORRECTION_5_BURN_IN_GATE",
        "blocked_paths": [
            "docs/reference_cases/e4_pl_q1m_correction5_blocked_gate_result.json",
            "docs/reference_cases/e4_pl_q1m_correction5_blocked_status.json",
            "docs/reference_cases/e4_pl_q1m_correction5_blocked_review.json",
        ],
        "review_independence": {
            "did_not_author_candidate": True,
            "did_not_execute_resource_lanes": True,
            "reviewed_frozen_evidence_only": True,
        },
        "review_required_keys": [
            "findings",
            "reviewed_inputs",
            "reviewer_independence",
            "schema",
            "verdict",
        ],
        "reviewed_input_hashes": [
            "contract_sha256",
            "gate_result_sha256",
            "status_sha256",
        ],
        "status_required_keys": [
            "clean_gate_index",
            "gate_result_sha256",
            "legacy_removal_authorized",
            "schema",
            "terminal",
        ],
        "success_terminal": "Q1M_CLEAN_GATE_1_OF_2_RECORDED",
        "success_paths": [
            "docs/reference_cases/e4_pl_q1m_gate_result.json",
            "docs/reference_cases/e4_pl_q1m_status.json",
            "docs/reference_cases/e4_pl_q1m_review.json",
        ],
        "success_commit_subject": "docs: record E4 PL Q1M clean burn-in gate 1",
    }
    # The recorded repository identities belong to the original burn-in contract.
    assert contract["repository_cleanliness"] == {
        "ANYfem": "FULLY_CLEAN_INCLUDING_UNTRACKED",
        "ANYfileIO": "TRACKED_AND_INDEX_CLEAN_HEAD_ARCHIVE_UNTRACKED_EXCLUDED",
        "ANYgeometry": "TRACKED_AND_INDEX_CLEAN_HEAD_ARCHIVE_UNTRACKED_EXCLUDED",
        "ANYmaterial": "TRACKED_AND_INDEX_CLEAN_HEAD_ARCHIVE_UNTRACKED_EXCLUDED",
        "ANYmesh": "TRACKED_AND_INDEX_CLEAN_HEAD_ARCHIVE_UNTRACKED_EXCLUDED",
        "ANYsolver": "FULLY_CLEAN_INCLUDING_UNTRACKED",
    }
    assert contract["sibling_authority"] == {
        "ANYfem": {
            "commit": "ba8b21b9cf2732168b099cfedc7508789bdcfbb3",
            "tree": "49a150bcccece1fef92dd627ae54689a545e0e61",
        },
        "ANYfileIO": {
            "commit": "9b1e5adea77a20155bbc23866af8c9aad853ddfd",
            "tree": "70b406be2574adceab4a7b688c0e489e0937df5d",
        },
        "ANYgeometry": {
            "commit": "6fb06c8b68b73dd0630aa41ac81ef999ef610457",
            "tree": "a563515df3ab24e7df388009b7412582e840e31e",
        },
        "ANYmaterial": {
            "commit": "74100a95988a633e311f8eb21df3d24cbb6bcc0d",
            "tree": "0d3c57eba577f243a3749b3f1102cbb94a3b51bf",
        },
        "ANYmesh": {
            "commit": "c9dad1d0a37d920e9fb95d1f6d0f12fbb1bf9fbf",
            "tree": "d443f008173003d560fb55673b0d90bf92f65e03",
        },
    }
    for lane, authority in contract["non_resource_commands"].items():
        assert authority["command_sha256"] == hashlib.sha256(
            authority["command"].encode("utf-8")
        ).hexdigest(), lane
    request_root = ROOT.parents[2] / ".resource-manager" / "requests"
    seen_request_ids = set()
    for lane, authority in contract["resource_requests"].items():
        request_path = request_root / f"{authority['request_id']}.json"
        assert authority["request_id"] not in seen_request_ids
        seen_request_ids.add(authority["request_id"])
        assert re.fullmatch(r"[0-9a-f]{32}", authority["request_id"]), lane
        assert re.fullmatch(r"[0-9a-f]{64}", authority["request_sha256"]), lane
        assert re.fullmatch(r"[0-9a-f]{64}", authority["command_sha256"]), lane
        if request_root.is_dir():
            request = json.loads(request_path.read_text(encoding="utf-8"))
            assert request["request_id"] == authority["request_id"], lane
            assert hashlib.sha256(request_path.read_bytes()).hexdigest() == authority[
                "request_sha256"
            ], lane
            assert hashlib.sha256(request["command"].encode("utf-8")).hexdigest() == (
                authority["command_sha256"]
            ), lane
    plan = (
        ROOT / "docs" / "agent_plans" / "S4_E4_PL_Q1M_BURNIN_HARDENING_PLAN.md"
    ).read_text(encoding="utf-8")
    for value in (
        contract["gate_result_schema"],
        contract["package_result_schema"],
        contract["status_schema"],
        contract["review_schema"],
        contract["adjudication"]["success_terminal"],
        contract["adjudication"]["blocked_terminal"],
        contract["adjudication"]["accepted_success_verdict"],
        contract["adjudication"]["accepted_blocked_verdict"],
        contract["adjudication"]["success_commit_subject"],
        contract["adjudication"]["blocked_commit_subject"],
        *contract["adjudication"]["success_paths"],
        *contract["adjudication"]["blocked_paths"],
    ):
        assert value in plan
    assert contract["hard_performance_gates"] == {
        "batch_path_equality": [
            "tests/test_e4_pl_workflow_parity.py::test_global_assembly_uses_candidate_scalar_kernel_and_activity_lifecycle",
            "tests/test_e4_pl_workflow_parity.py::test_structured_mesh_cold_assembly_reuses_translation_equivalent_geometry",
        ],
        "q4_numerical_parity": [
            "tests/test_e4_pl_workflow_parity.py::test_global_assembly_uses_candidate_scalar_kernel_and_activity_lifecycle",
        ],
        "warm_cache_reuse": [
            "tests/test_e4_pl_workflow_parity.py::test_structured_mesh_cold_assembly_reuses_translation_equivalent_geometry",
        ],
    }
    assert contract["performance_baseline"] == {
        "measurement_names": [
            "qualified_q4_cached_tangent",
            "qualified_q4_warm_global_assembly",
        ],
        "repetitions": 11,
        "schema": "anysolver.s4.e4-pl-q1m-performance-baseline-v1",
        "speed_claim": "GATE_1_BASELINE_ONLY_NO_SPEED_CLAIM",
        "warmups": 1,
    }
    assert {
        "ECOSYSTEM_ANYFEM_ENGINEERING_REFERENCE",
        "NO_PRODUCTION_OR_PERFORMANCE_SELECTOR_BYPASS",
        "PACKAGE_BUILD_AND_INSTALLED_WHEEL_SMOKE",
    } <= set(contract["removal_gates"])
    default = shell_formulation_diagnostics(node_count=4)
    assert default == {
        "schema": "anysolver.shell-formulation-diagnostics-v1",
        "node_count": 4,
        "requested_formulation": None,
        "selected_formulation": "e4-pl",
        "production_default": True,
        "topology_policy": "QUALIFIED_E4_PL_Q4",
        "legacy_q4": {
            "state": "DEPRECATED_BURN_IN_ROLLBACK",
            "available_through": "0.4.x",
            "removal_not_before": "0.5.0",
            "selector": "legacy",
            "warning": "LegacyQ4DeprecationWarning",
        },
    }
    rollback = shell_formulation_diagnostics(node_count=4, formulation="legacy_s4")
    assert rollback["selected_formulation"] == "legacy"
    assert rollback["topology_policy"] == "DEPRECATED_LEGACY_Q4_ROLLBACK"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    migration = (ROOT / "docs" / "E4_PL_MIGRATION.md").read_text(encoding="utf-8")
    assert "docs/E4_PL_MIGRATION.md" in readme
    assert "0.4.x" in migration and "0.5.0" in migration


def test_legacy_q4_factory_and_direct_class_emit_dedicated_warning() -> None:
    with pytest.warns(LegacyQ4DeprecationWarning, match="temporary rollback"):
        rollback = create_shell_element(
            1, [1, 2, 3, 4], "steel", formulation="legacy"
        )
    with pytest.warns(LegacyQ4DeprecationWarning, match="temporary rollback"):
        alias = create_element("legacy-shell", 2, [1, 2, 3, 4], "steel")
    with pytest.warns(LegacyQ4DeprecationWarning, match="temporary rollback"):
        direct = ShellElement(3, [1, 2, 3, 4], "steel")
    assert type(rollback) is ShellElement
    assert type(alias) is ShellElement
    assert type(direct) is ShellElement


def test_default_q4_and_s3_and_preserved_higher_order_topologies_do_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default = create_shell_element(1, [1, 2, 3, 4], "steel")
        tri3 = create_shell_element(
            2,
            [1, 2, 3],
            "steel",
            reference_normal=(0.0, 0.0, 1.0),
        )
        tri6 = create_shell_element(3, list(range(1, 7)), "steel")
        q8 = create_shell_element(4, list(range(1, 9)), "steel")
    assert type(default) is QualifiedE4PLShellElement
    assert type(tri3) is NativeParityE4PLS3V2DShellElement
    assert type(tri6) is ShellElement
    assert type(q8) is ShellElement
    assert [item for item in caught if item.category is LegacyQ4DeprecationWarning] == []
    assert shell_formulation_diagnostics(node_count=3)["topology_policy"] == (
        "NATIVE_PARITY_E4_PL_S3_V2D_DEFAULT"
    )
    for count in (6, 8):
        diagnostic = shell_formulation_diagnostics(node_count=count)
        assert diagnostic["topology_policy"] == "PRESERVED_LEGACY_NON_Q4"


def test_diagnostics_fail_closed_for_unknown_topology_and_formulation() -> None:
    with pytest.raises(ValueError, match="node_count"):
        shell_formulation_diagnostics(node_count=5)
    with pytest.raises(ValueError, match="Unknown shell formulation"):
        shell_formulation_diagnostics(node_count=4, formulation="experimental")
    with pytest.raises(ValueError, match="only for four-node"):
        shell_formulation_diagnostics(node_count=8, formulation="e4-pl")


def test_migration_surface_is_mechanics_free_and_production_bypasses_stay_closed() -> None:
    diagnostics_source = ast.parse(
        (ROOT / "src" / "anyfesolver" / "elements.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in diagnostics_source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "shell_formulation_diagnostics"
    )
    called = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called <= {"int", "str", "ValueError", "_normalized_shell_formulation"}

    bypasses: list[str] = []
    legacy_calls: set[str] = set()
    routed_scripts: set[str] = set()
    routed_performance_tests: set[str] = set()
    performance_paths = [
        ROOT / relative for relative in _gate_module().inventory()["performance"]
    ]
    guarded_paths = [
        *(ROOT / "src" / "anyfesolver").rglob("*.py"),
        *(ROOT / "scripts").glob("*.py"),
        *performance_paths,
    ]
    for path in guarded_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        shell_aliases = {"ShellElement"}
        legacy_aliases = {"LegacyShellElement"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for imported in node.names:
                local_name = imported.asname or imported.name
                if imported.name == "ShellElement":
                    shell_aliases.add(local_name)
                elif imported.name == "LegacyShellElement":
                    legacy_aliases.add(local_name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if called_name in shell_aliases or called_name == "ShellElement":
                bypasses.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if called_name in legacy_aliases or called_name == "LegacyShellElement":
                legacy_calls.add(path.relative_to(ROOT).as_posix())
            if (
                path.parent == ROOT / "scripts"
                and called_name == "create_element"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "shell"
            ):
                routed_scripts.add(path.relative_to(ROOT).as_posix())
            if (
                path in performance_paths
                and called_name == "create_element"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "shell"
            ):
                routed_performance_tests.add(path.relative_to(ROOT).as_posix())
    assert bypasses == []
    assert legacy_calls == {
        "src/anyfesolver/elements.py",
        "tests/test_performance_improvements.py",
    }
    assert routed_scripts >= {
        "scripts/benchmark_advanced_s4_batches.py",
        "scripts/benchmark_contact_work_buffer.py",
        "scripts/benchmark_damage_matrix_updates.py",
        "scripts/benchmark_impact_reduced_assembly.py",
        "scripts/benchmark_sol_ultra_performance.py",
        "scripts/verify_sol_ultra_numerics.py",
    }
    assert routed_performance_tests >= {
        "tests/test_advanced_s4_batches.py",
        "tests/test_corotational_performance.py",
        "tests/test_damage_matrix_performance.py",
        "tests/test_impact_tangent_reuse.py",
        "tests/test_nonlinear_performance.py",
        "tests/test_nonlinear_performance_batch_b.py",
        "tests/test_performance_improvements.py",
        "tests/test_vectorized_hill48.py",
    }


def test_every_test_file_has_one_burn_in_lane_and_heavy_files_are_serialized() -> None:
    gate = _gate_module()
    lanes = gate.inventory()
    discovered = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("test_*.py")
    }
    classified = [path for paths in lanes.values() for path in paths]
    assert len(classified) == len(set(classified))
    assert set(classified) == discovered
    assert "tests/test_e4_pl_burnin.py" in lanes["quick"]
    assert "tests/test_fe_solver_shell_verification.py" in lanes["functional"]
    assert "tests/test_nonlinear_performance.py" in lanes["performance"]
    assert "tests/test_e4_pl_s3_mixed_eigen_performance.py" in lanes["extended"]
    assert "tests/test_e4_pl_s3_mixed_eigen_performance.py" not in lanes[
        "performance"
    ]
    assert "tests/test_e4_pl_q1v_contract.py" in lanes["extended"]
    assert "tests/test_e4_pl_s3_opt_in.py" in lanes["additive"]
    assert "tests/test_e4_pl_s3_reference_batch.py" in lanes["additive"]
    assert "tests/test_e4_pl_s3_mixed_eigen_performance.py" not in (
        lanes["quick"] + lanes["functional"] + lanes["additive"]
    )
    assert all("performance" not in Path(path).name for path in lanes["functional"])


def test_reference_batch_additive_override_preserves_optional_performance() -> None:
    gate = _gate_module()
    assert gate.classify_test(Path("tests/test_e4_pl_s3_reference_batch.py")) == (
        "additive"
    )
    assert gate.classify_test(
        Path("tests/test_e4_pl_s3_mixed_eigen_performance.py")
    ) == "extended"
    assert gate.classify_test(Path("tests/test_nonlinear_performance.py")) == (
        "performance"
    )


def test_ci_lane_is_exactly_quick_plus_functional_plus_additive() -> None:
    gate = _gate_module()
    real_lanes = gate.inventory()
    real_ci_selection = {
        *real_lanes["quick"],
        *real_lanes["functional"],
        *real_lanes["additive"],
    }
    assert "tests/test_e4_pl_s3_reference_batch.py" in real_ci_selection
    assert (
        "tests/test_e4_pl_s3_mixed_eigen_performance.py"
        not in real_ci_selection
    )
    assert "tests/test_nonlinear_performance.py" not in real_ci_selection
    policy = gate._validate_ci_policy(
        gate.strict_json_load(gate.S3_Q4_CONTRACT_PATH)
    )
    assert policy["required_lanes"] == ["quick", "functional", "additive"]
    assert policy["extent"] == "COMPLETE_FROZEN_INVENTORIES"
    assert policy["smoke_or_representative_only_forbidden"] is True

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert (
        "python scripts/run_portable_ci.py --workers 2 --timeout-seconds 2400\n"
        "          --matrix-shard-index ${{ matrix.shard-index }} "
        "--matrix-shard-count 8\n"
        "          --matrix-subshard-index ${{ matrix.subshard-index }} --matrix-subshard-count 4"
        in workflow
    )
    rows = re.findall(r'(?m)^\s+- \{os: (windows-latest|ubuntu-latest), python-version: "(3\.\d+)", shard-index: (\d), subshard-index: (\d)\}$', workflow)
    expected = {(('windows-latest' if i % 2 == 0 else 'ubuntu-latest'),
                 f"3.{11+i//2}", str(i), str(j)) for i in range(8) for j in range(4)}
    assert len(rows) == 32 and set(rows) == expected
    assert "push:\n    branches: [main]\n  pull_request:" in workflow
    assert "python scripts/run_e4_pl_burnin_gate.py ci" not in workflow
    # The burn-in contract remains immutable historical evidence. Ordinary CI is
    # allowed to advance its exact sibling pins after coordinated releases; the
    # active pins are validated by the compatibility tests instead.
    for authority in gate.strict_json_load(CONTRACT)["sibling_authority"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", authority["commit"])


def test_portable_ci_inventory_is_unique_and_excludes_long_lanes() -> None:
    lanes = portable_ci.inventory()
    modules = portable_ci.merge_test_modules()
    assert modules
    assert len(modules) == len(set(modules))
    registered = {
        module for lane in portable_ci.MERGE_LANES for module in lanes[lane]
    }
    historical = {
        module
        for module in registered
        if portable_ci._is_portable_historical_module(module)
    }
    assert set(modules) == registered - historical
    assert set(portable_ci.POST_CLOSEOUT_HISTORICAL_MODULES) <= registered
    assert set(portable_ci.PORTABLE_CURRENT_S3_SUCCESSOR_MODULES) <= set(modules)
    assert historical
    assert set(modules).isdisjoint(lanes["performance"])
    assert set(modules).isdisjoint(lanes["extended"])


def test_portable_ci_partition_is_deterministic_disjoint_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = {"a.py": 100, "b.py": 80, "c.py": 30, "d.py": 20, "e.py": 10}
    monkeypatch.setattr(portable_ci, "_module_weight", weights.__getitem__)
    first = portable_ci.partition_modules(tuple(weights), 3)
    second = portable_ci.partition_modules(tuple(reversed(weights)), 3)
    assert first == second
    assigned = [module for bucket in first for module in bucket]
    assert len(assigned) == len(set(assigned)) == len(weights)
    assert set(assigned) == set(weights)


def test_portable_ci_weights_are_line_ending_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = "tests/test_line_endings.py"
    path = tmp_path / module
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(portable_ci, "ROOT", tmp_path)

    path.write_bytes(b"first\nsecond\n")
    lf_weight = portable_ci._module_weight(module)
    path.write_bytes(b"first\r\nsecond\r\n")

    assert portable_ci._module_weight(module) == lf_weight


def test_portable_ci_executes_process_state_sensitive_modules_in_fresh_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = (
        "tests/test_geometry_panel.py",
        "tests/test_generalized_section_api_workflows.py",
        "tests/test_generalized_shell_sections.py",
        "tests/test_a.py",
        "tests/test_b.py",
    )
    monkeypatch.setattr(portable_ci, "_module_weight", lambda _module: 1)
    partitions = portable_ci.execution_partitions(modules, 2)

    isolated = tuple(
        (module,)
        for module in sorted(
            {
                "tests/test_geometry_panel.py",
                "tests/test_generalized_section_api_workflows.py",
                "tests/test_generalized_shell_sections.py",
            }
        )
    )
    assert partitions[-3:] == isolated
    assert all(
        set(portable_ci.PORTABLE_ISOLATED_MODULES).isdisjoint(partition)
        for partition in partitions[:-3]
    )
    assigned = [module for partition in partitions for module in partition]
    assert sorted(assigned) == sorted(modules)


def test_portable_ci_observed_costs_separate_the_four_heaviest_modules() -> None:
    partitions = portable_ci.partition_modules(
        portable_ci.merge_test_modules(), 4
    )
    heaviest = {
        "tests/test_beam_shell_verification.py",
        "tests/test_e4_pl_q4_state_lifecycle.py",
        "tests/test_e4_pl_s3_restart_history.py",
        "tests/test_e4_pl_s3_state_lifecycle.py",
    }
    owners = {
        index
        for index, partition in enumerate(partitions)
        if heaviest.intersection(partition)
    }
    assert owners == set(range(4))


def test_portable_ci_matrix_shards_cover_general_once_and_s3_critical_always() -> None:
    modules = portable_ci.merge_test_modules()
    critical = set(portable_ci.PORTABLE_CURRENT_S3_SUCCESSOR_MODULES)
    shards = [portable_ci.matrix_modules(modules, index, 8) for index in range(8)]
    for shard in shards:
        assert critical <= set(shard)
        assert len(shard) == len(set(shard))
    general_counts = Counter(
        module
        for shard in shards
        for module in shard
        if module not in critical
    )
    assert set(general_counts) == set(modules) - critical
    assert set(general_counts.values()) == {1}


@pytest.mark.parametrize(
    ("index", "count"),
    [(0, None), (None, 8), (-1, 8), (8, 8), (0, 0), (True, 8), (0, True)],
)
def test_portable_ci_matrix_shards_reject_invalid_bounds(index, count) -> None:
    with pytest.raises(ValueError, match="matrix shard"):
        portable_ci.matrix_modules(portable_ci.merge_test_modules(), index, count)


@pytest.mark.parametrize("workers", [0, -1, True])
def test_portable_ci_partition_rejects_invalid_worker_counts(workers: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        portable_ci.partition_modules(("a.py",), workers)


def test_portable_ci_worker_is_headless_isolated_and_single_threaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANYSOLVER_CI_EXPECTED_NODES", "hostile")
    monkeypatch.setenv("ANYSOLVER_FUNCTIONAL_RESULT", "hostile")
    monkeypatch.setenv("ANYSOLVER_BURNIN_ACTIVE_TEST_LANE", "functional")
    environment = portable_ci._worker_environment(tmp_path)
    assert "ANYSOLVER_CI_EXPECTED_NODES" not in environment
    assert "ANYSOLVER_FUNCTIONAL_RESULT" not in environment
    assert "ANYSOLVER_BURNIN_ACTIVE_TEST_LANE" not in environment
    assert environment["ANY3DVIEW_DISABLE_GPU"] == "1"
    for name in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert environment[name] == "1"
    modules = ("tests/test_a.py", "tests/test_b.py")
    command = portable_ci._worker_command(modules, tmp_path)
    assert command[-2:] == list(modules)
    assert f"--basetemp={tmp_path / 'basetemp'}" in command
    assert "--collect-only" not in command
    local_patch_command = portable_ci._worker_command(
        ("tests/test_local_patch_transition.py",), tmp_path
    )
    assert sum(item.startswith("--deselect=") for item in local_patch_command) == 2
    pardiso_command = portable_ci._worker_command(
        ("tests/test_fe_solver_infrastructure.py",), tmp_path
    )
    assert sum(item.startswith("--deselect=") for item in pardiso_command) == 1
    v2d_command = portable_ci._worker_command(
        ("tests/test_e4_pl_s3_v2d_linear_native_parity.py",), tmp_path
    )
    assert sum(item.startswith("--deselect=") for item in v2d_command) == 1
    v6g_command = portable_ci._worker_command(
        ("tests/test_e4_pl_s3_v6g_recovery_current_eigen.py",), tmp_path
    )
    assert sum(item.startswith("--deselect=") for item in v6g_command) == 2
    mixed_command = portable_ci._worker_command(
        ("tests/test_e4_pl_s3_mixed_mesh_qualification_runner.py",), tmp_path
    )
    assert sum(item.startswith("--deselect=") for item in mixed_command) == 2


def test_pytest_lane_uses_and_cleans_workspace_local_basetemp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate_module()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(
        gate,
        "_local_roots",
        lambda: {
            name: tmp_path
            for name, _distribution, _package in gate.LOCAL_DISTRIBUTIONS
        },
    )

    def fake_metadata_overlay(_roots, destination):
        destination.mkdir()
        return {}

    monkeypatch.setattr(gate, "_write_source_metadata_overlay", fake_metadata_overlay)
    monkeypatch.setattr(
        gate,
        "_pytest_environment",
        lambda **_kwargs: {"Q1M_TEST": "1"},
    )
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        observed.update(command=list(command), cwd=cwd, env=env, check=check)
        basetemp_arg = next(item for item in command if item.startswith("--basetemp="))
        basetemp = Path(basetemp_arg.split("=", 1)[1])
        assert basetemp.parent == tmp_path / ".pytest_tmp_q1m_runtime"
        (basetemp / "nested").mkdir(parents=True)
        (basetemp / "nested" / "probe.txt").write_text("probe\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert gate._run_pytest_lane("functional", ["tests/test_probe.py"]) == 0
    assert observed["cwd"] == tmp_path
    assert observed["env"] == {"Q1M_TEST": "1"}
    assert observed["check"] is False
    assert not (tmp_path / ".pytest_tmp_q1m_runtime").exists()


def test_pytest_lane_uses_python_311_cleanup_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate_module()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(
        gate,
        "_local_roots",
        lambda: {
            name: tmp_path
            for name, _distribution, _package in gate.LOCAL_DISTRIBUTIONS
        },
    )
    monkeypatch.setattr(
        gate,
        "_write_source_metadata_overlay",
        lambda _roots, destination: destination.mkdir(),
    )
    monkeypatch.setattr(gate, "_pytest_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )
    callbacks: list[dict[str, object]] = []
    real_rmtree = gate.shutil.rmtree

    def fake_rmtree(path, **kwargs):
        callbacks.append(dict(kwargs))
        real_rmtree(path, **kwargs)

    monkeypatch.setattr(gate.shutil, "rmtree", fake_rmtree)

    assert gate._run_pytest_lane("ci", ["tests/test_probe.py"]) == 0
    assert callbacks
    assert all(set(item) == {"onerror"} for item in callbacks)


def test_pytest_lane_cleans_basetemp_when_metadata_overlay_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate_module()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_local_roots", lambda: {})

    def reject_overlay(_roots, _destination):
        raise gate.EvidenceError("metadata rejected")

    monkeypatch.setattr(gate, "_write_source_metadata_overlay", reject_overlay)

    with pytest.raises(gate.EvidenceError, match="metadata rejected"):
        gate._run_pytest_lane("functional", ["tests/test_probe.py"])
    assert not (tmp_path / ".pytest_tmp_q1m_runtime").exists()


def test_pytest_lane_keeps_source_metadata_visible_during_real_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate_module()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    projects = {
        "ANYgeometry": ("ANYgeometry", "0.2.4"),
        "ANYmaterial": ("ANYmaterial", "0.1.1"),
        "ANYmesh": ("ANYmesher", "0.2.5"),
        "ANYfileIO": ("ANYfileio", "0.2.0"),
        "ANYfeSolver": ("ANYfeSolver", "0.3.0"),
    }
    roots = {}
    for repository, (distribution, version) in projects.items():
        project = tmp_path if repository == "ANYfeSolver" else tmp_path / repository
        project.mkdir(exist_ok=True)
        (project / "pyproject.toml").write_text(
            f'[project]\nname = "{distribution}"\nversion = "{version}"\n',
            encoding="utf-8",
        )
        roots[repository] = project
    monkeypatch.setattr(gate, "_local_roots", lambda: roots)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_metadata_probe.py").write_text(
        "from importlib import metadata\n\n"
        "def test_frozen_source_metadata_is_visible(tmp_path):\n"
        "    assert tmp_path.is_dir()\n"
        "    assert metadata.version('ANYmesher') == '0.2.5'\n",
        encoding="utf-8",
    )

    assert gate._run_pytest_lane(
        "functional", ["tests/test_metadata_probe.py"]
    ) == 0
    assert not (tmp_path / ".pytest_tmp_q1m_runtime").exists()


def test_package_lane_is_isolated_and_covers_the_declared_wheel_smoke() -> None:
    gate = _gate_module()
    assert gate.PACKAGE_CHECKS == (
        "BUILD_LOCAL_WHEELS_WITHOUT_ISOLATION",
        "INSTALL_FRESH_TARGET_WITHOUT_DEPENDENCY_RESOLUTION",
        "REJECT_SOURCE_TREE_IMPORTS",
        "VERIFY_QUALIFIED_Q4_DEFAULT",
        "VERIFY_NON_Q4_PRESERVATION",
        "VERIFY_DIAGNOSTICS_EXPORT",
        "VERIFY_EXPLICIT_LEGACY_WARNING",
    )
    assert [(name, distribution, package) for name, distribution, package in gate.LOCAL_DISTRIBUTIONS] == [
        ("ANYgeometry", "ANYgeometry", "anygeometry"),
        ("ANYmaterial", "ANYmaterial", "anymaterial"),
        ("ANYmesh", "ANYmesher", "anymesher"),
        ("ANYfileIO", "ANYfileio", "anyfileio"),
        ("ANYfeSolver", "ANYfeSolver", "anyfesolver"),
    ]
    source = (ROOT / "scripts" / "run_e4_pl_burnin_gate.py").read_text(encoding="utf-8")
    assert 'environment.pop("PYTHONPATH", None)' in source
    assert 'environment.pop("PYTHONHOME", None)' in source
    assert 'environment["PYTHONNOUSERSITE"] = "1"' in source
    assert '"archive"' in source
    assert '"--format=zip"' in source
    assert '"--untracked-files=no"' in source
    assert "_copy_source_snapshot" not in source
    assert '"--no-build-isolation"' in source
    assert '"--no-deps"' in source
    assert '"-I"' in source
    assert "source repository leaked onto sys.path" in gate.PACKAGE_SMOKE
    assert "QualifiedE4PLShellElement" in gate.PACKAGE_SMOKE
    assert "LegacyQ4DeprecationWarning" in gate.PACKAGE_SMOKE


def test_package_snapshot_uses_only_clean_tracked_head(tmp_path: Path) -> None:
    gate = _gate_module()
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "q1m@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Q1M Test"],
        check=True,
    )
    (repository / "tracked.txt").write_text("authority\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"], check=True
    )
    (repository / "untracked-s3.txt").write_text("must not enter archive\n", encoding="utf-8")

    snapshot = tmp_path / "snapshot"
    source = gate._archive_head_snapshot(
        repository,
        tmp_path / "archives" / "repository.zip",
        snapshot,
        environment=os.environ.copy(),
    )
    assert (snapshot / "tracked.txt").read_text(encoding="utf-8") == "authority\n"
    assert not (snapshot / "untracked-s3.txt").exists()
    assert source["commit"] == subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert source["content"]["files"] == 1

    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked or index changes"):
        gate._archive_head_snapshot(
            repository,
            tmp_path / "archives" / "dirty.zip",
            tmp_path / "dirty-snapshot",
            environment=os.environ.copy(),
        )


def test_package_result_binds_source_archives_and_wheel_identity() -> None:
    gate = _gate_module()
    gate_result = _prospective_gate_result(gate)
    package = _prospective_package_result(gate, gate_result)
    gate.validate_package_result(package)

    mutation = copy.deepcopy(package)
    mutation["sources"]["ANYmesh"]["content"]["sha256"] = "invalid"
    with pytest.raises(gate.EvidenceError, match="invalid format"):
        gate.validate_package_result(mutation)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    gate = _gate_module()
    with pytest.raises(gate.EvidenceError, match="duplicate JSON key"):
        gate.strict_json_loads('{"schema":"one","schema":"two"}')
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(gate.EvidenceError, match="non-finite"):
            gate.strict_json_loads('{"value":' + token + "}")
    with pytest.raises(gate.EvidenceError, match="non-finite"):
        gate.canonical_json_bytes({"value": float("nan")})


def test_gate_result_schema_is_canonical_and_exclusive(
    tmp_path: Path, monkeypatch
) -> None:
    gate = _gate_module()
    record = _prospective_gate_result(gate)
    first = gate.canonical_gate_result_bytes(record)
    second = gate.canonical_gate_result_bytes(copy.deepcopy(record))
    assert first == second
    assert first.endswith(b"\n")
    assert first == gate.canonical_json_bytes(gate.strict_json_loads(first))
    output = tmp_path / "gate.json"
    with pytest.raises(gate.EvidenceError, match="repository/request/log bindings"):
        gate.write_gate_result_exclusive(output, record)
    monkeypatch.setattr(gate, "canonical_gate_result_bytes", lambda *args, **kwargs: first)
    bindings = {
        "repository_paths": {},
        "lane_log_paths": {},
        "package_result_path": tmp_path / "package.json",
        "request_paths": {},
        "wheel_path": tmp_path / "candidate.whl",
    }
    gate.write_gate_result_exclusive(output, record, **bindings)
    assert output.read_bytes() == first
    with pytest.raises(gate.EvidenceError, match="refusing to replace"):
        gate.write_gate_result_exclusive(output, record, **bindings)


def test_gate_result_rejects_missing_mutated_and_reused_authority() -> None:
    gate = _gate_module()
    base = _prospective_gate_result(gate)

    mutations = []

    missing_lane = copy.deepcopy(base)
    del missing_lane["lanes"]["package"]
    mutations.append(missing_lane)

    changed_inventory = copy.deepcopy(base)
    changed_inventory["lanes"]["quick"]["inventory"] = []
    mutations.append(changed_inventory)

    arbitrary_command = copy.deepcopy(base)
    arbitrary_command["lanes"]["quick"]["command_sha256"] = "f" * 64
    mutations.append(arbitrary_command)

    empty_log = copy.deepcopy(base)
    empty_log["lanes"]["package"]["log"] = {
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    mutations.append(empty_log)

    contradictory_exit = copy.deepcopy(base)
    contradictory_exit["lanes"]["anyfem"]["exit_code"] = 1
    mutations.append(contradictory_exit)

    changed_hash = copy.deepcopy(base)
    changed_hash["resource_requests"][0]["request_sha256"] = "8" * 64
    mutations.append(changed_hash)

    reused_id = copy.deepcopy(base)
    reused_id["resource_requests"][1]["request_id"] = reused_id["resource_requests"][0]["request_id"]
    mutations.append(reused_id)

    dirty_candidate = copy.deepcopy(base)
    dirty_candidate["candidate"]["clean"] = False
    mutations.append(dirty_candidate)

    wrong_anyfem = copy.deepcopy(base)
    wrong_anyfem["siblings"]["ANYfem"]["commit"] = "9" * 40
    mutations.append(wrong_anyfem)

    wrong_wheel_hash = copy.deepcopy(base)
    wrong_wheel_hash["wheel"]["sha256"] = "not-a-hash"
    mutations.append(wrong_wheel_hash)

    early_removal = copy.deepcopy(base)
    early_removal["legacy_removal_authorized"] = True
    mutations.append(early_removal)

    for mutation in mutations:
        with pytest.raises(gate.EvidenceError):
            gate.validate_gate_result(mutation)


def test_failed_or_rollback_gate_cannot_increment_clean_counter() -> None:
    gate = _gate_module()
    unrecorded_failure = _blocked_at(
        gate,
        _prospective_gate_result(gate),
        "functional",
        "FUNCTIONAL_GATE_FAILED",
    )
    unrecorded_failure["rollback"] = {
        "state": "NO_UNRESOLVED_ROLLBACK_INCIDENT",
        "unresolved_incidents": [],
    }
    with pytest.raises(gate.EvidenceError, match="rollback incident"):
        gate.validate_gate_result(unrecorded_failure)

    failed = _blocked_at(
        gate,
        _prospective_gate_result(gate),
        "functional",
        "FUNCTIONAL_GATE_FAILED",
    )
    gate.validate_gate_result(failed)

    incident = _prospective_gate_result(gate)
    incident["rollback"] = {
        "state": "UNRESOLVED_LEGACY_Q4_ROLLBACK_INCIDENT",
        "unresolved_incidents": ["INCIDENT-1"],
    }
    incident["clean_gate_index"] = 0
    gate.validate_gate_result(incident)

    invalid = copy.deepcopy(failed)
    invalid["clean_gate_index"] = 1
    with pytest.raises(gate.EvidenceError, match="clean_gate_index"):
        gate.validate_gate_result(invalid)


def test_fail_fast_blocked_result_forbids_fabricated_downstream_evidence() -> None:
    gate = _gate_module()
    blocked = _blocked_at(
        gate, _prospective_gate_result(gate), "quick", "QUICK_GATE_FAILED"
    )
    gate.validate_gate_result(blocked)
    assert "package_result" not in blocked and "wheel" not in blocked
    assert "performance_baseline" not in blocked and "hard_gates" not in blocked

    fabricated = copy.deepcopy(blocked)
    fabricated["wheel"] = {
        "bytes": 1,
        "filename": "fabricated.whl",
        "sha256": "a" * 64,
    }
    with pytest.raises(gate.EvidenceError, match="package artifacts are forbidden"):
        gate.validate_gate_result(fabricated)

    logged_not_run = copy.deepcopy(blocked)
    logged_not_run["lanes"]["package"]["log"] = {
        "bytes": 1,
        "sha256": "b" * 64,
    }
    with pytest.raises(gate.EvidenceError, match="keys mismatch"):
        gate.validate_gate_result(logged_not_run)

    out_of_order = copy.deepcopy(blocked)
    out_of_order["lanes"]["functional"]["status"] = "PASS"
    with pytest.raises(gate.EvidenceError):
        gate.validate_gate_result(out_of_order)


def test_repository_identity_validation_fails_closed_on_dirty_inputs(monkeypatch) -> None:
    gate = _gate_module()
    record = _prospective_gate_result(gate)
    paths = {name: Path(name) for name in ("ANYfeSolver", *gate.SIBLING_NAMES)}

    def dirty_status(_repository: Path) -> dict[str, object]:
        raw = b" M tracked.py\0"
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    monkeypatch.setattr(gate, "_functional_source_status", dirty_status)
    with pytest.raises(gate.EvidenceError, match="not completely clean"):
        gate.validate_gate_result(record, repository_paths=paths)


def test_repository_cleanliness_policy_is_full_for_candidate_and_anyfem(monkeypatch) -> None:
    gate = _gate_module()
    record = _prospective_gate_result(gate)
    paths = {name: Path(name) for name in ("ANYfeSolver", *gate.SIBLING_NAMES)}
    status_order = []

    def clean_status(repository: Path) -> dict[str, object]:
        status_order.append(repository.name)
        return {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}

    def clean_git(repository: Path, *args: str) -> str:
        name = repository.name
        identity = record["candidate"] if name == "ANYfeSolver" else record["siblings"][name]
        return identity["tree"] if args[-1] == "HEAD^{tree}" else identity["commit"]

    monkeypatch.setattr(gate, "_functional_source_status", clean_status)
    monkeypatch.setattr(gate, "_git", clean_git)
    gate.validate_gate_result(record, repository_paths=paths)
    assert status_order == sorted(("ANYfeSolver", *gate.SIBLING_NAMES))


def test_complete_sibling_authority_is_fail_closed() -> None:
    gate = _gate_module()
    record = _prospective_gate_result(gate)
    gate.validate_gate_result(record)

    record["siblings"]["ANYmesh"]["tree"] = "0" * 40
    with pytest.raises(gate.EvidenceError, match="ANYmesh tree"):
        gate.validate_gate_result(record)


def test_external_log_and_wheel_hashes_are_verified(tmp_path: Path) -> None:
    gate = _gate_module()
    record = _prospective_gate_result(gate)
    log_paths = {}
    for lane in gate.GATE_LANES:
        path = tmp_path / f"{lane}.log"
        content = f"{lane}\n".encode("ascii")
        if lane == "performance":
            observation = {
                "hard_gates": record["hard_gates"],
                "performance_baseline": record["performance_baseline"],
                "schema": gate.PERFORMANCE_OBSERVATION_SCHEMA,
            }
            content += gate.PERFORMANCE_BASELINE_MARKER
            content += gate.canonical_json_bytes(observation)
        path.write_bytes(content)
        log_paths[lane] = path
        record["lanes"][lane]["log"] = gate.file_hash_record(path)
    wheel = tmp_path / record["wheel"]["filename"]
    wheel.write_bytes(b"wheel bytes")
    record["wheel"] = gate.wheel_hash_record(wheel)
    gate.validate_gate_result(record, lane_log_paths=log_paths, wheel_path=wheel)

    record["lanes"]["quick"]["log"]["sha256"] = "0" * 64
    with pytest.raises(gate.EvidenceError, match="external log mismatch"):
        gate.validate_gate_result(record, lane_log_paths=log_paths, wheel_path=wheel)


def test_package_source_override_is_explicit_and_resolved(
    tmp_path: Path, monkeypatch
) -> None:
    gate = _gate_module()
    frozen = tmp_path / "frozen-anygeometry"
    frozen.mkdir()
    (frozen / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setenv("Q1M_ANYGEOMETRY_ROOT", str(frozen))

    roots = gate._local_roots()

    assert roots["ANYgeometry"] == frozen.resolve()
    assert roots["ANYfeSolver"] == ROOT.resolve()


def test_pytest_source_metadata_overlay_binds_the_frozen_source_graph(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    roots = gate._local_roots()
    overlay = tmp_path / "source-distributions"

    versions = gate._write_source_metadata_overlay(roots, overlay)

    expected_versions = {}
    for repository, distribution, _module in gate.LOCAL_DISTRIBUTIONS:
        metadata = gate.tomllib.loads(
            (roots[repository] / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        assert metadata["name"] == distribution
        expected_versions[distribution] = metadata["version"]
    assert versions == expected_versions
    environment = gate._pytest_environment(
        roots=roots,
        metadata_overlay=overlay,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from importlib import metadata; "
                "print(metadata.version('ANYmesher')); "
                "print(metadata.version('ANYmaterial')); "
                "from anymesher import Mesh; "
                "print(Mesh.__name__)"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        expected_versions["ANYmesher"],
        expected_versions["ANYmaterial"],
        "Mesh",
    ]


def test_source_metadata_overlay_rejects_a_mismatched_project(
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    roots = gate._local_roots()
    impostor = tmp_path / "impostor"
    impostor.mkdir()
    (impostor / "pyproject.toml").write_text(
        '[project]\nname = "not-anymesher"\nversion = "0.2.5"\n',
        encoding="utf-8",
    )
    roots["ANYmesh"] = impostor

    with pytest.raises(gate.EvidenceError, match="distribution name mismatch"):
        gate._write_source_metadata_overlay(roots, tmp_path / "metadata")


def test_final_validation_binds_requests_logs_package_and_wheel(
    tmp_path: Path, monkeypatch
) -> None:
    gate = _gate_module()
    contract = copy.deepcopy(gate.strict_json_load(CONTRACT))
    record = _prospective_gate_result(gate)

    request_paths = {}
    for lane, authority in contract["resource_requests"].items():
        request = {"command": f"registered {lane}", "request_id": authority["request_id"]}
        path = tmp_path / "requests" / f"{authority['request_id']}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(gate.canonical_json_bytes(request))
        request_paths[lane] = path
        authority["request_sha256"] = gate.file_hash_record(path)["sha256"]
        authority["command_sha256"] = hashlib.sha256(
            request["command"].encode("utf-8")
        ).hexdigest()
        row = next(item for item in record["resource_requests"] if item["lane"] == lane)
        row["request_sha256"] = authority["request_sha256"]
        record["lanes"][lane]["command_sha256"] = authority["command_sha256"]

    log_paths = {}
    for lane in gate.GATE_LANES:
        path = tmp_path / f"{lane}.log"
        payload = f"{lane} passed\n".encode("ascii")
        if lane == "performance":
            observation = {
                "hard_gates": record["hard_gates"],
                "performance_baseline": record["performance_baseline"],
                "schema": gate.PERFORMANCE_OBSERVATION_SCHEMA,
            }
            payload += gate.PERFORMANCE_BASELINE_MARKER + gate.canonical_json_bytes(observation)
        path.write_bytes(payload)
        log_paths[lane] = path
        record["lanes"][lane]["log"] = gate.file_hash_record(path)

    wheel = tmp_path / record["wheel"]["filename"]
    wheel.write_bytes(b"preserved installed-wheel candidate")
    record["wheel"] = gate.wheel_hash_record(wheel)
    package = _prospective_package_result(gate, record)
    package["wheels"]["ANYfeSolver"] = record["wheel"]
    package_path = tmp_path / "package-result.json"
    package_path.write_bytes(gate.canonical_json_bytes(package))
    record["package_result"] = gate.file_hash_record(package_path)

    monkeypatch.setattr(gate, "_validate_repository_identities", lambda *_args: None)
    repositories = {name: tmp_path / name for name in ("ANYfeSolver", *gate.SIBLING_NAMES)}
    gate.validate_final_gate_result(
        record,
        contract=contract,
        repository_paths=repositories,
        lane_log_paths=log_paths,
        package_result_path=package_path,
        request_paths=request_paths,
        wheel_path=wheel,
    )

    package["sources"]["ANYmesh"]["commit"] = "d" * 40
    package_path.write_bytes(gate.canonical_json_bytes(package))
    record["package_result"] = gate.file_hash_record(package_path)
    with pytest.raises(gate.EvidenceError, match="source identity mismatch"):
        gate.validate_final_gate_result(
            record,
            contract=contract,
            repository_paths=repositories,
            lane_log_paths=log_paths,
            package_result_path=package_path,
            request_paths=request_paths,
            wheel_path=wheel,
        )


def test_status_and_independent_review_bind_exact_canonical_inputs(tmp_path: Path) -> None:
    gate = _gate_module()
    contract = gate.strict_json_load(CONTRACT)
    record = _prospective_gate_result(gate)
    gate_path = tmp_path / "gate.json"
    gate_path.write_bytes(gate.canonical_json_bytes(record))
    status = {
        "clean_gate_index": 1,
        "gate_result_sha256": gate.file_hash_record(gate_path)["sha256"],
        "legacy_removal_authorized": False,
        "schema": contract["status_schema"],
        "terminal": contract["adjudication"]["success_terminal"],
    }
    status_path = tmp_path / "status.json"
    status_path.write_bytes(gate.canonical_json_bytes(status))
    review = {
        "findings": [],
        "reviewed_inputs": {
            "contract_sha256": gate.file_hash_record(CONTRACT)["sha256"],
            "gate_result_sha256": gate.file_hash_record(gate_path)["sha256"],
            "status_sha256": gate.file_hash_record(status_path)["sha256"],
        },
        "reviewer_independence": contract["adjudication"]["review_independence"],
        "schema": contract["review_schema"],
        "verdict": contract["adjudication"]["accepted_success_verdict"],
    }
    review_path = tmp_path / "review.json"
    review_path.write_bytes(gate.canonical_json_bytes(review))
    with pytest.raises(gate.EvidenceError, match="outside the repository"):
        gate.validate_adjudication_files(
            gate_path,
            status_path,
            review_path,
            repository_root=ROOT / "docs",
        )
    gate.validate_adjudication_files(
        gate_path, status_path, review_path, repository_root=None
    )

    review["reviewed_inputs"]["status_sha256"] = "0" * 64
    review_path.write_bytes(gate.canonical_json_bytes(review))
    with pytest.raises(gate.EvidenceError, match="reviewed-input hashes"):
        gate.validate_adjudication_files(
            gate_path, status_path, review_path, repository_root=None
        )


def test_blocked_cycles_remain_verifiable_under_immutable_authority() -> None:
    gate = _gate_module()
    evidence_root = ROOT / "docs" / "reference_cases"
    historical = (
        ("e4_pl_q1m_blocked", CONTRACT_CYCLE0),
        ("e4_pl_q1m_correction1_blocked", CONTRACT_CYCLE1),
        ("e4_pl_q1m_correction2_blocked", CONTRACT_CYCLE2),
        ("e4_pl_q1m_correction3_blocked", CONTRACT_CYCLE3),
        ("e4_pl_q1m_correction4_blocked", CONTRACT_CYCLE4),
    )
    blocked_request_ids: set[str] = set()
    for stem, authority_path in historical:
        gate_path = evidence_root / f"{stem}_gate_result.json"
        status_path = evidence_root / f"{stem}_status.json"
        review_path = evidence_root / f"{stem}_review.json"

        review = gate.strict_json_load(review_path)
        assert gate.file_hash_record(authority_path)["sha256"] == review[
            "reviewed_inputs"
        ]["contract_sha256"]
        gate.validate_adjudication_files(
            gate_path,
            status_path,
            review_path,
            contract_path=authority_path,
        )
        blocked_gate = gate.strict_json_load(gate_path)
        blocked_request_ids.update(
            row["request_id"] for row in blocked_gate["resource_requests"]
        )

    # A current repository name cannot authenticate a sealed record, and a
    # modified authority cannot opt into the historical repository identity.
    historical_contract = gate.strict_json_load(authority_path)
    wrong_repository = copy.deepcopy(blocked_gate)
    wrong_repository["candidate"]["repository"] = "ANYfeSolver"
    with pytest.raises(gate.EvidenceError, match="candidate.repository must be ANYsolver"):
        gate.validate_gate_result(wrong_repository, contract=historical_contract)
    mutated_contract = dict(historical_contract, default="changed")
    with pytest.raises(gate.EvidenceError, match="candidate.repository must be ANYfeSolver"):
        gate.validate_gate_result(blocked_gate, contract=mutated_contract)

    live_contract = gate.strict_json_load(CONTRACT)
    with pytest.raises(gate.EvidenceError, match="candidate.repository must be ANYfeSolver"):
        gate.validate_gate_result(blocked_gate, contract=live_contract)
    live_request_ids = {
        authority["request_id"]
        for authority in live_contract["resource_requests"].values()
    }
    assert live_request_ids.isdisjoint(blocked_request_ids)


def test_performance_hard_gates_and_timing_baseline_are_strict(tmp_path: Path) -> None:
    gate = _gate_module()
    record = _prospective_gate_result(gate)
    gate.validate_gate_result(record)

    changed_gate = copy.deepcopy(record)
    changed_gate["hard_gates"]["warm_cache_reuse"]["observed"] = False
    with pytest.raises(gate.EvidenceError, match="status contradicts"):
        gate.validate_gate_result(changed_gate)

    failed_gate = copy.deepcopy(record)
    failed_gate["hard_gates"]["warm_cache_reuse"].update(
        {"observed": False, "status": "FAIL"}
    )
    failed_gate["clean_gate_index"] = 0
    failed_gate["rollback"] = {
        "state": "UNRESOLVED_LEGACY_Q4_ROLLBACK_INCIDENT",
        "unresolved_incidents": ["WARM_CACHE_GATE_FAILED"],
    }
    gate.validate_gate_result(failed_gate)

    changed_statistic = copy.deepcopy(record)
    changed_statistic["performance_baseline"]["measurements"][
        "qualified_q4_cached_tangent"
    ]["p95_ns"] = 10
    with pytest.raises(gate.EvidenceError, match="timing samples"):
        gate.validate_gate_result(changed_statistic)

    observation = {
        "hard_gates": record["hard_gates"],
        "performance_baseline": record["performance_baseline"],
        "schema": gate.PERFORMANCE_OBSERVATION_SCHEMA,
    }
    log = tmp_path / "performance.log"
    log.write_bytes(
        b"pytest output\n"
        + gate.PERFORMANCE_BASELINE_MARKER
        + gate.canonical_json_bytes(observation)
    )
    assert gate.extract_performance_observation(log) == observation

    duplicate = tmp_path / "duplicate.log"
    marker = gate.PERFORMANCE_BASELINE_MARKER + gate.canonical_json_bytes(observation)
    duplicate.write_bytes(marker + marker)
    with pytest.raises(gate.EvidenceError, match="exactly one"):
        gate.extract_performance_observation(duplicate)


def test_performance_baseline_statistic_definition_is_deterministic() -> None:
    path = ROOT / "scripts" / "measure_e4_pl_q1m_baseline.py"
    spec = importlib.util.spec_from_file_location("e4_pl_q1m_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.timing_summary(list(range(1, 12))) == {
        "mad_ns": 3,
        "median_ns": 6,
        "p95_ns": 11,
        "samples_ns": list(range(1, 12)),
    }
