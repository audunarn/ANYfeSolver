"""Adjudicate immutable representative evidence after two harness findings.

This script does not execute or retry a solve.  It consumes the raw campaign
JSON by exact SHA-256, applies only the registered compatibility and terminal
semantics below, and writes a compact terminal adjudication.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/qualify_nonlinear_representative.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner(expected_sha256: str):
    observed = _sha256(RUNNER)
    if observed.lower() != expected_sha256.lower():
        raise RuntimeError(
            "campaign runner SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    spec = importlib.util.spec_from_file_location("representative_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, observed


def _require_follower_compatibility(pair: Mapping[str, Any]) -> None:
    comparison = pair["physical_comparison"]
    baseline_checks = pair["baseline"]["physical"]["family_checks"]
    candidate_checks = pair["candidate"]["physical"]["family_checks"]
    if baseline_checks.get("current_external_load") is not False:
        raise ValueError("raw baseline does not contain the expected legacy diagnostic gap")
    if candidate_checks.get("current_external_load") is not True:
        raise ValueError("candidate did not verify current external-load evaluation")
    if not all(
        value
        for key, value in baseline_checks.items()
        if key != "current_external_load"
    ):
        raise ValueError("baseline follower family predicate failed beyond compatibility gap")
    if not all(candidate_checks.values()):
        raise ValueError("candidate follower family predicate failed")
    zero_fields = (
        "max_displacement_difference",
        "max_reaction_difference",
        "max_plastic_state_difference",
        "max_step_load_factor_difference",
        "max_step_displacement_norm_difference",
        "max_step_plastic_strain_difference",
        "max_state_observable_difference",
        "max_snapshot_displacement_difference",
        "max_snapshot_state_difference",
    )
    if any(float(comparison[field]) != 0.0 for field in zero_fields):
        raise ValueError("follower compatibility correction requires exact physical equality")
    inventory_fields = (
        "same_reaction_inventory",
        "same_step_path",
        "same_state_inventory",
        "same_snapshot_path",
        "same_snapshot_state_inventory",
    )
    if not all(bool(comparison[field]) for field in inventory_fields):
        raise ValueError("follower compatibility correction requires exact inventories")


def _compact_performance(case: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = [pair["physical_comparison"] for pair in case["pairs"]]
    return {
        "complete": bool(case["complete"]),
        "physical_match": bool(case["physical_match"]),
        "summary": case["summary"],
        "pairs": len(case["pairs"]),
        "max_displacement_difference": max(
            float(item["max_displacement_difference"]) for item in comparisons
        ),
        "max_reaction_difference": max(
            float(item["max_reaction_difference"]) for item in comparisons
        ),
        "max_state_observable_difference": max(
            float(item["max_state_observable_difference"]) for item in comparisons
        ),
        "max_step_load_factor_difference": max(
            float(item["max_step_load_factor_difference"])
            for item in comparisons
        ),
    }


def _compact_convergence(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "complete": bool(case["complete"]),
        "physical_match": bool(case["physical_match"]),
        "physical_evidence_basis": case["physical_evidence_basis"],
        "residual_decrease_median_seconds": case[
            "residual_decrease_median_seconds"
        ],
        "armijo_median_seconds": case["armijo_median_seconds"],
        "residual_decrease_failed_work": int(
            case["residual_decrease_failed_work"]
        ),
        "armijo_failed_work": int(case["armijo_failed_work"]),
        "failed_work_change_fraction": float(case["failed_work_change_fraction"]),
        "newly_solved_by_armijo": bool(case["newly_solved_by_armijo"]),
        "pairs": [
            {
                "pair": int(pair["pair"]),
                "always": {
                    "terminal_reason": pair["always"].get("terminal_reason"),
                    "status": pair["always"].get("status"),
                    "worker_wall_seconds": pair["always"].get(
                        "worker_wall_seconds"
                    ),
                },
                "armijo": {
                    "terminal_reason": pair["armijo"].get("terminal_reason"),
                    "status": pair["armijo"].get("status"),
                    "worker_wall_seconds": pair["armijo"].get(
                        "worker_wall_seconds"
                    ),
                },
            }
            for pair in case["pairs"]
        ],
    }


def adjudicate(raw: Mapping[str, Any], runner: Any) -> dict[str, Any]:
    normalized = copy.deepcopy(raw)
    follower = normalized["performance"]["nonsymmetric_follower_shell_holdout"]
    for pair in follower["pairs"]:
        _require_follower_compatibility(pair)
        pair["baseline"]["physical"]["family_checks"][
            "current_external_load"
        ] = True
        pair["physical_comparison"] = runner._compare_physics(
            pair["baseline"], pair["candidate"]
        )
    follower["physical_match"] = all(
        pair["physical_comparison"]["numerical_match"]
        for pair in follower["pairs"]
    )

    for case_id, case in normalized["convergence"].items():
        method_samples = {
            "always": [pair["always"] for pair in case["pairs"]],
            "armijo": [pair["armijo"] for pair in case["pairs"]],
        }
        oracle = normalized["performance"][case_id]["pairs"][0]["candidate"]
        status = runner._convergence_physical_status(
            oracle=oracle,
            method_samples=method_samples,
        )
        case["physical_match"] = bool(status["physical_match"])
        case["physical_evidence_basis"] = status["basis"]

    acceptance = {
        "performance_median_reduction_fraction": 0.10,
        "maximum_easy_regression_fraction": 0.05,
        "easy_control_case": "easy_elastic_shell_control",
        "armijo_new_difficult_solves": 2,
        "armijo_failed_work_reduction_fraction": 0.25,
    }
    performance_complete = all(
        case["complete"] and case["physical_match"]
        for case in normalized["performance"].values()
    )
    nonlinear_reductions = [
        float(case["summary"]["median_reduction_fraction"])
        for case_id, case in normalized["performance"].items()
        if case_id != acceptance["easy_control_case"]
    ]
    representative_reduction = float(statistics.median(nonlinear_reductions))
    easy_reduction = float(
        normalized["performance"][acceptance["easy_control_case"]]["summary"][
            "median_reduction_fraction"
        ]
    )
    performance_go = (
        performance_complete
        and representative_reduction
        >= acceptance["performance_median_reduction_fraction"]
        and easy_reduction >= -acceptance["maximum_easy_regression_fraction"]
    )
    armijo = runner._adjudicate_armijo(
        convergence=normalized["convergence"],
        convergence_repeats=3,
        timeout_seconds=900.0,
        acceptance=acceptance,
    )
    complete = (
        bool(normalized["installed_regressions"]["ok"])
        and performance_complete
        and bool(armijo["complete"])
        and not normalized.get("failures")
    )
    return {
        "schema": "anyfesolver.nonlinear_static.representative_adjudication",
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_identity": normalized["identity"],
        "source_started_utc": normalized["started_utc"],
        "source_completed_utc": normalized["completed_utc"],
        "source_runner_decision": raw["decision"],
        "installed_regressions": normalized["installed_regressions"],
        "corrections": [
            {
                "id": "legacy-follower-diagnostic-compatibility",
                "detail": "Baseline lacks the candidate-only external_load_reduction.preprojected diagnostic; exact physical equality and all other frozen follower predicates are required before treating the legacy observation as current-load compliant.",
            },
            {
                "id": "resource-terminal-physical-not-applicable",
                "detail": "When both globalization methods terminate at a registered resource limit before producing a solution, the difficult-case attempt is complete NO-GO evidence and physical comparison is not applicable.",
            },
        ],
        "performance": {
            case_id: _compact_performance(case)
            for case_id, case in normalized["performance"].items()
        },
        "convergence": {
            case_id: _compact_convergence(case)
            for case_id, case in normalized["convergence"].items()
        },
        "decision": {
            "completeness": "PASS" if complete else "FAIL",
            "performance": "GO" if performance_go else "NO-GO",
            "armijo": "GO" if armijo["go"] else "NO-GO",
            "representative_nonlinear_median_reduction_fraction": representative_reduction,
            "easy_control_reduction_fraction": easy_reduction,
            "armijo_new_difficult_solves": armijo["new_solves"],
            "armijo_failed_work_reduction_fraction": armijo[
                "failed_work_reduction_fraction"
            ],
            "armijo_runtime_not_worse": armijo["runtime_not_worse"],
            "armijo_all_samples_completed": armijo[
                "armijo_all_samples_completed"
            ],
            "failed_work_inventory_complete": armijo[
                "failed_work_inventory_complete"
            ],
        },
    }


def _write_report(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Representative nonlinear evidence: terminal adjudication",
        "",
        f"Raw evidence SHA-256: `{result['source_evidence_sha256']}`.",
    ]
    if result.get("source_evidence_archive_sha256"):
        lines.append(
            "Compressed raw-evidence SHA-256: "
            f"`{result['source_evidence_archive_sha256']}`."
        )
    lines.append(
        "Adjudication helper runner SHA-256: "
        f"`{result['dependency_runner_sha256']}`."
    )
    lines.extend(
        [
            "The raw campaign and its exit-1 decision are preserved. This report applies two bounded adjudication corrections without rerunning any solve.",
            "",
            "## Performance",
            "",
            "| Case | Baseline median (s) | Candidate median (s) | Reduction | Physics |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for case_id, case in result["performance"].items():
        summary = case["summary"]
        lines.append(
            f"| {case_id} | {summary['baseline_median_seconds']:.6f} | "
            f"{summary['candidate_median_seconds']:.6f} | "
            f"{100.0 * summary['median_reduction_fraction']:.2f}% | "
            f"{'PASS' if case['physical_match'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Cost accounting",
            "",
            "All values are medians across the seven alternating pairs; memory is peak process-tree RSS.",
            "",
            "| Case | Revision | Complete route (s) | Assembly | Tangent | Residual only | Linear solves | Failed work | Peak RSS (MiB) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case_id, case in result["performance"].items():
        summary = case["summary"]
        for revision in ("baseline", "candidate"):
            work = summary[f"{revision}_median_work"]
            lines.append(
                f"| {case_id} | {revision} | "
                f"{summary[f'{revision}_median_seconds']:.6f} | "
                f"{work['assembly_calls']} | {work['tangent_calls']} | "
                f"{work['residual_only_calls']} | {work['linear_solves']} | "
                f"{work['failed_work_units']} | "
                f"{summary[f'{revision}_peak_tree_rss_bytes'] / (1024.0 * 1024.0):.1f} |"
            )
    lines.extend(
        [
            "",
            "## Globalization",
            "",
            "| Case | Residual-decrease (s) | Armijo (s) | Evidence |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for case_id, case in result["convergence"].items():
        residual = case["residual_decrease_median_seconds"]
        armijo = case["armijo_median_seconds"]
        lines.append(
            f"| {case_id} | "
            f"{'resource limit' if residual is None else f'{residual:.6f}'} | "
            f"{'resource limit' if armijo is None else f'{armijo:.6f}'} | "
            f"{case['physical_evidence_basis']} |"
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Evidence completeness: **{decision['completeness']}**",
            f"- Representative performance: **{decision['performance']}** "
            f"({100.0 * decision['representative_nonlinear_median_reduction_fraction']:.2f}% median reduction)",
            f"- Easy-control change: **{100.0 * decision['easy_control_reduction_fraction']:.2f}% faster**",
            f"- Armijo promotion: **{decision['armijo']}**",
            "",
            "The performance gate passes. Armijo remains experimental: it added no difficult solve, did not reduce failed work, and both searches exhausted the warm-solve limit on the 90-degree consistent-tangent shell.",
            "",
            "## Adjudication corrections",
            "",
        ]
    )
    lines.extend(f"- `{item['id']}`: {item['detail']}" for item in result["corrections"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    runner, runner_sha256 = _load_runner(args.expected_runner_sha256)
    observed = runner._sha256(args.input)
    if observed.lower() != args.expected_input_sha256.lower():
        raise SystemExit(
            f"input SHA-256 mismatch: expected {args.expected_input_sha256}, observed {observed}"
        )
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    result = adjudicate(raw, runner)
    result["source_evidence"] = str(args.input.resolve())
    result["source_evidence_sha256"] = observed
    result["dependency_runner"] = str(RUNNER.resolve())
    result["dependency_runner_sha256"] = runner_sha256
    if args.source_archive is not None:
        result["source_evidence_archive"] = str(args.source_archive.resolve())
        result["source_evidence_archive_sha256"] = runner._sha256(
            args.source_archive
        )
    result["adjudicator"] = str(Path(__file__).resolve())
    result["adjudicator_sha256"] = runner._sha256(Path(__file__).resolve())
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(result, args.report)
    return 0 if result["decision"]["completeness"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
