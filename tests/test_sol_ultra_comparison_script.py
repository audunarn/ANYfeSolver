from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_sol_ultra_performance.py"
BASELINE_SHA = "a" * 40
FINAL_SHA = "b" * 40


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _case(*, warm_seconds: float) -> dict[str, Any]:
    return {
        "case_id": "synthetic_case",
        "status": "completed",
        "measurements": {
            "cold": {"wall_seconds": 2.0},
            "warm": {
                "wall_seconds": {"median": warm_seconds},
                "python_peak_bytes": {"median": 1024.0},
                "process_peak_rss_bytes": {"median": 4096.0},
            },
        },
        "phases": {
            "constitutive_update": {
                "available": False,
                "median_seconds": None,
                "samples_seconds": [],
            },
            "total_wall_time": {
                "available": True,
                "median_seconds": warm_seconds,
                "samples_seconds": [warm_seconds],
            },
        },
        "representative_warm_payload": {
            "results": {"history_storage_mode": "selected"}
        },
    }


def _benchmark_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = {
        "schema_name": "anyfesolver.sol_ultra.benchmark",
        "schema_version": 1,
        "report_kind": "baseline",
        "revision": {
            "head_sha": BASELINE_SHA,
            "origin_main_sha": BASELINE_SHA,
            "merge_base_sha": BASELINE_SHA,
            "branch": "perf2/baseline",
        },
        "campaign_context": {
            "revision": {
                "initial_performance_2_sha": BASELINE_SHA,
                "origin_main_sha": BASELINE_SHA,
                "merge_base_sha": BASELINE_SHA,
            }
        },
        "suite": {"name": "full", "repeats": 1},
        "summary": {"failed_count": 0},
        "phase_schema": ["constitutive_update", "total_wall_time"],
        "environment": {},
        "cases": [_case(warm_seconds=1.0)],
        "known_limitations": ["synthetic fixture"],
    }
    final = copy.deepcopy(baseline)
    final.update(report_kind="final")
    final["revision"] = {
        "head_sha": FINAL_SHA,
        "origin_main_sha": BASELINE_SHA,
        "merge_base_sha": BASELINE_SHA,
        "branch": "performance_2",
    }
    final["suite"]["repeats"] = 3
    final["cases"] = [_case(warm_seconds=0.5)]
    return baseline, final


def _numerical_artifact() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "sol_ultra_numerical_comparison",
        "status": "passed",
        "summary": {
            "case_counts": {"passed": 1, "failed": 0, "unavailable": 0},
            "metric_failures": 0,
            "unavailable_cases": 0,
        },
        "baseline": {
            "source": {"commit": BASELINE_SHA, "dirty": False}
        },
        "candidate": {
            "source": {"commit": FINAL_SHA, "dirty": False}
        },
    }


def _thread_artifact() -> dict[str, Any]:
    return {
        "summary": {
            "status": "completed",
            "failed_sections": [],
            "default_thread_policy_changed": False,
        },
        "environment": {"revision": {"head_sha": FINAL_SHA}},
        "workloads": {},
    }


def _run_comparison(
    tmp_path: Path,
    *,
    include_numerical: bool,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    baseline, final = _benchmark_artifacts()
    baseline_path = tmp_path / "baseline.json"
    final_path = tmp_path / "final.json"
    numerical_path = tmp_path / "numerical.json"
    thread_path = tmp_path / "threads.json"
    decision_path = tmp_path / "decision.md"
    output_path = tmp_path / "comparison.md"
    _write_json(baseline_path, baseline)
    _write_json(final_path, final)
    if include_numerical:
        _write_json(numerical_path, _numerical_artifact())
    _write_json(thread_path, _thread_artifact())
    decision_path.write_text(
        "# Decisions\n\n## Decision table\n\n"
        "| Workstream | Decision |\n"
        "| --- | --- |\n"
        "| Synthetic | promote |\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline_path),
            "--final",
            str(final_path),
            "--numerical",
            str(numerical_path),
            "--thread-scaling",
            str(thread_path),
            "--decision-log",
            str(decision_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, output_path


def test_complete_synthetic_comparison_preserves_unavailable_phase(
    tmp_path: Path,
) -> None:
    completed, output_path = _run_comparison(
        tmp_path,
        include_numerical=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = output_path.read_text(encoding="utf-8")
    assert "Report status: **COMPLETE**" in report
    assert "| `synthetic_case`" in report
    assert "2.000x" in report
    assert "| `constitutive_update` | 0/1 | 0/1 | 1/1 |" in report
    assert "Unavailable means" in report
    assert "constitutive_update speedup" not in report


def test_missing_numerical_gate_returns_two(tmp_path: Path) -> None:
    completed, output_path = _run_comparison(
        tmp_path,
        include_numerical=False,
    )

    assert completed.returncode == 2
    assert '"status": "incomplete"' in completed.stdout
    assert "missing numerical comparison" in completed.stdout
    assert "Report status: **INCOMPLETE**" in output_path.read_text(encoding="utf-8")
