"""Focused tests for the independent Sol Ultra comparison harness."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_sol_ultra_numerics.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location(
        "anyfesolver_sol_ultra_verification_harness",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(harness, metric, *, status: str = "completed", reason=None):
    case = {
        "status": status,
        "description": "unit fixture",
        "metrics": {"value": metric} if status == "completed" else {},
        "observations": {},
        "performance": {
            "wall_seconds": 1.0,
            "peak_rss_bytes": 100,
            "python_peak_allocated_bytes": None,
            "qualification_role": "informational_only",
        },
    }
    if reason is not None:
        case["reason"] = reason
    return {
        "schema_version": harness.SCHEMA_VERSION,
        "artifact_kind": "sol_ultra_numerical_capture",
        "generated_at_utc": "2026-08-11T00:00:00Z",
        "label": "fixture",
        "source": {"commit": "abc", "branch": "fixture", "dirty": False},
        "environment": {},
        "suite": {"name": "fixture", "requested_cases": ["fixture"]},
        "acceptance_criteria": harness.ACCEPTANCE_CRITERIA,
        "cases": {"fixture": case},
    }


def test_full_payload_relative_norm_comparison_passes_and_detects_failure(harness) -> None:
    reference_metric = harness.numeric_metric(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "global_matrix",
    )
    close_metric = harness.numeric_metric(
        np.asarray([[1.0, 2.0], [3.0, 4.0 + 1.0e-13]]),
        "global_matrix",
    )
    passed = harness.compare_documents(
        _artifact(harness, reference_metric),
        _artifact(harness, close_metric),
    )
    assert passed["status"] == "passed"
    metric_result = passed["cases"]["fixture"]["metrics"]["value"]
    assert metric_result["status"] == "passed"
    assert metric_result["relative_l2_error"] > 0.0

    far_metric = harness.numeric_metric(
        np.asarray([[1.0, 2.0], [3.0, 4.01]]),
        "global_matrix",
    )
    failed = harness.compare_documents(
        _artifact(harness, reference_metric),
        _artifact(harness, far_metric),
    )
    assert failed["status"] == "failed"
    assert failed["summary"]["metric_failures"] == 1


def test_baseline_tolerance_is_authoritative_and_checksum_is_verified(harness) -> None:
    baseline_metric = harness.numeric_metric([1.0, 2.0], "internal_force")
    candidate_metric = harness.numeric_metric([1.0, 2.0], "internal_force")
    candidate_metric["comparison"]["rtol"] = 1.0
    report = harness.compare_documents(
        _artifact(harness, baseline_metric),
        _artifact(harness, candidate_metric),
    )
    assert report["status"] == "passed"
    assert report["warnings"][0]["reason"].startswith(
        "candidate_tolerance_metadata_differs"
    )

    corrupted = _artifact(harness, deepcopy(baseline_metric))
    corrupted["cases"]["fixture"]["metrics"]["value"]["values"][0] = 9.0
    with pytest.raises(ValueError, match="checksum mismatch"):
        harness.validate_capture(corrupted)


def test_recovery_stress_gate_has_a_stress_only_roundoff_floor(harness) -> None:
    reference = harness.numeric_metric([6.821210263296962e-9], "recovery_stress")
    roundoff = harness.numeric_metric([0.0], "recovery_stress")
    report = harness.compare_documents(
        _artifact(harness, reference),
        _artifact(harness, roundoff),
    )
    assert report["status"] == "passed"
    assert report["cases"]["fixture"]["metrics"]["value"]["threshold"] == pytest.approx(
        1.0e-7
    )

    material_difference = harness.numeric_metric([1.0e-6], "recovery_stress")
    report = harness.compare_documents(
        _artifact(harness, reference),
        _artifact(harness, material_difference),
    )
    assert report["status"] == "failed"
    assert harness.ACCEPTANCE_CRITERIA["recovery"]["atol"] == 1.0e-12
    assert harness.ACCEPTANCE_CRITERIA["plastic_state"]["atol"] == 1.0e-12


def test_environment_differences_are_visible_warnings(harness) -> None:
    metric = harness.numeric_metric([1.0], "internal_force")
    baseline = _artifact(harness, metric)
    candidate = _artifact(harness, deepcopy(metric))
    baseline["environment"] = {"python": "3.13.0", "numpy": "2.3.0"}
    candidate["environment"] = {"python": "3.14.0", "numpy": "2.3.0"}
    report = harness.compare_documents(baseline, candidate)
    assert report["status"] == "passed"
    assert report["summary"]["environment_differences"] == 1
    assert report["environment_differences"][0]["field"] == "python"
    assert "environment_mismatch" in harness.render_markdown(report)

    parsed = harness.build_parser().parse_args(
        [
            "capture",
            "--label",
            "targeted",
            "--output",
            "artifact.json",
            "--solver-root",
            str(ROOT),
        ]
    )
    assert parsed.solver_root == ROOT


def test_nonincrease_and_upper_bound_gates_are_scientific_not_exact(harness) -> None:
    baseline_count = harness.nonincrease_metric(12)
    candidate_count = harness.nonincrease_metric(9)
    report = harness.compare_documents(
        _artifact(harness, baseline_count),
        _artifact(harness, candidate_count),
    )
    assert report["status"] == "passed"

    baseline_bound = harness.upper_bound_metric(1.0e-9, 1.0e-8, "residual")
    candidate_bound = harness.upper_bound_metric(2.0e-9, 1.0e-8, "residual")
    report = harness.compare_documents(
        _artifact(harness, baseline_bound),
        _artifact(harness, candidate_bound),
    )
    assert report["status"] == "passed"

    candidate_bad = harness.upper_bound_metric(2.0e-7, 1.0e-8, "residual")
    report = harness.compare_documents(
        _artifact(harness, baseline_bound),
        _artifact(harness, candidate_bad),
    )
    assert report["status"] == "failed"

    # Detailed iteration histories remain auditable, while a shorter history
    # does not fail qualification when the separately gated total improved.
    baseline_history = harness.informational_numeric_metric([4, 3, 3], "iterations")
    candidate_history = harness.informational_numeric_metric([3, 2], "iterations")
    report = harness.compare_documents(
        _artifact(harness, baseline_history),
        _artifact(harness, candidate_history),
    )
    assert report["status"] == "passed"
    result = report["cases"]["fixture"]["metrics"]["value"]
    assert result["reason"] == "informational_shape_change"


def test_unavailable_case_is_explicit_and_makes_report_incomplete(harness) -> None:
    baseline = _artifact(harness, harness.numeric_metric([1.0], "internal_force"))
    candidate = _artifact(
        harness,
        None,
        status="unavailable",
        reason="case_timeout_after_1_seconds",
    )
    report = harness.compare_documents(baseline, candidate)
    assert report["status"] == "incomplete"
    assert report["cases"]["fixture"]["status"] == "unavailable"
    assert report["unavailable"][0]["reason"]["candidate"] == (
        "case_timeout_after_1_seconds"
    )
    markdown = harness.render_markdown(report)
    assert "Overall status: **INCOMPLETE**" in markdown
    assert "case_timeout_after_1_seconds" in markdown


def test_hill48_capture_is_deterministic_and_unselected_cases_are_named(harness) -> None:
    first = harness.execute_case("hill48_material")
    second = harness.execute_case("hill48_material")
    assert first["status"] == second["status"] == "completed"
    assert set(first["metrics"]) == set(second["metrics"])
    for name, metric in first["metrics"].items():
        assert metric["signature"]["sha256_float64_le"] == (
            second["metrics"][name]["signature"]["sha256_float64_le"]
        )

    document = harness.capture_document(
        label="focused",
        suite="quick",
        explicit_cases=["hill48_material"],
        isolate_cases=False,
    )
    harness.validate_capture(document)
    assert document["cases"]["hill48_material"]["status"] == "completed"
    assert document["cases"]["nonlinear_impact"]["status"] == "unavailable"
    assert document["cases"]["nonlinear_impact"]["reason"] == (
        "not_selected_by_suite:quick"
    )
    assert document["cases"]["nonlinear_impact_direct_reduced"]["status"] == (
        "unavailable"
    )
    assert document["cases"]["nonlinear_impact_direct_reduced"]["reason"] == (
        "not_selected_by_suite:quick"
    )


def test_plastic_impact_metrics_capture_state_damage_and_deletion_timing(harness) -> None:
    diagnostics = {
        "element_states": {1: {"alpha": np.asarray([0.01, 0.02])}},
        "element_state_history": [
            {"time": 0.0, "max_equivalent_plastic_strain": 0.0},
            {"time": 0.1, "max_equivalent_plastic_strain": 0.02},
        ],
        "state_von_mises_history": ({1: 1000.0}, {1: 1100.0}),
        "plastic_work_proxy": [0.0, 0.02],
        "plastic_impact_damage_summary": {
            "enabled": True,
            "deleted_count": 1,
            "deleted_fraction": 1.0,
            "deleted_element_ids": [1],
            "softened_element_ids": [],
            "max_damage": 1.2,
            "max_utilization": 1.2,
            "max_equivalent_plastic_strain": 0.02,
            "records": [
                {
                    "element_id": 1,
                    "history": [
                        {
                            "time": 0.1,
                            "step_index": 4,
                            "equivalent_plastic_strain": 0.02,
                            "utilization": 1.2,
                            "damage": 1.2,
                            "scale": 1.0e-6,
                            "location": "layer[2]",
                        }
                    ],
                }
            ],
            "deletion_records": [
                {
                    "element_id": 1,
                    "element_type": "ShellElement",
                    "step_index": 4,
                    "load_factor": 0.1,
                    "trigger_name": "max_equivalent_plastic_strain",
                    "trigger_value": 0.02,
                    "threshold": 0.016,
                    "location": "layer[2]",
                    "measure": 0.25,
                }
            ],
        },
        "erosion_summary": {
            "all_eroded_element_ids": [1],
            "damage_triggered_element_ids": [1],
            "active_softened_element_ids": [],
            "residual_stiffness_model": "qualified fixture",
        },
    }
    metrics = {}

    unavailable = harness._append_plastic_impact_metrics(metrics, diagnostics)

    assert unavailable == []
    assert "plastic.element_states.alpha" in metrics
    assert metrics["plastic.element_states.alpha"]["values"] == [0.01, 0.02]
    assert metrics["damage.record_count"]["value"] == 1
    assert metrics["damage.history.event_ids"]["shape"] == [1, 2]
    assert metrics["damage.deletion_count"]["value"] == 1
    assert metrics["damage.deletion.event_ids"]["values"] == [1.0, 4.0]
    assert metrics["damage.deletion.0.trigger_name"]["value"] == (
        "max_equivalent_plastic_strain"
    )


def test_metric_shape_change_and_missing_metric_fail_closed(harness) -> None:
    baseline = _artifact(harness, harness.numeric_metric([1.0, 2.0], "internal_force"))
    shape_changed = _artifact(
        harness,
        harness.numeric_metric([[1.0, 2.0]], "internal_force"),
    )
    report = harness.compare_documents(baseline, shape_changed)
    assert report["status"] == "failed"
    assert report["cases"]["fixture"]["metrics"]["value"]["reason"] == (
        "shape_mismatch"
    )

    missing = deepcopy(baseline)
    missing["cases"]["fixture"]["metrics"] = {}
    report = harness.compare_documents(baseline, missing)
    assert report["status"] == "failed"
    assert report["cases"]["fixture"]["metrics"]["value"]["reason"] == (
        "missing_candidate_metric"
    )
