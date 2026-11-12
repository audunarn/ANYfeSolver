"""Render the deterministic Sol Ultra baseline/final comparison report.

The benchmark runner owns measurements.  This script only validates and
compares stored artifacts; it never runs a solver or manufactures a missing
phase value.  Missing release evidence is written as ``n/a`` and makes the
command exit with status 2 unless ``--allow-incomplete`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "performance"
BENCHMARK_SCHEMA = "anyfesolver.sol_ultra.benchmark"
BENCHMARK_SCHEMA_VERSION = 1


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _nested(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _markdown(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        value = f"{value:.9g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _seconds(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.6f}s"


def _bytes(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(number) < 1024.0 or candidate == units[-1]:
            break
        number /= 1024.0
    return f"{number:.2f} {unit}"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _number(numerator)
    bottom = _number(denominator)
    if top is None or bottom is None or bottom <= 0.0:
        return None
    return top / bottom


def _ratio_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def _case_index(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for case in report.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in result:
            raise ValueError(f"duplicate or empty benchmark case id: {case_id!r}")
        result[case_id] = case
    return result


def _measurement(case: Mapping[str, Any], temperature: str, name: str) -> Any:
    value = _nested(case, "measurements", temperature, name)
    if temperature == "warm" and isinstance(value, Mapping):
        return value.get("median")
    return value


def _initial_revision(report: Mapping[str, Any], name: str) -> Any:
    return _nested(report, "campaign_context", "revision", name, default=None)


def _validate_benchmark(
    report: Mapping[str, Any], *, expected_kind: str, path: Path
) -> list[str]:
    issues: list[str] = []
    if report.get("schema_name") != BENCHMARK_SCHEMA:
        issues.append(f"{path}: unsupported schema_name {report.get('schema_name')!r}")
    if report.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        issues.append(f"{path}: unsupported schema_version {report.get('schema_version')!r}")
    if report.get("report_kind") != expected_kind:
        issues.append(
            f"{path}: report_kind is {report.get('report_kind')!r}, expected {expected_kind!r}"
        )
    if _nested(report, "summary", "failed_count", default=0) != 0:
        issues.append(f"{path}: benchmark contains failed cases")
    return issues


def _extract_markdown_section(text: str, heading: str) -> str | None:
    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start + 1 : end]).strip()


def _sanitize_path(path: Sequence[str]) -> str:
    return ".".join(
        "<runtime-id>" if re.fullmatch(r"\d{6,}", item) else item for item in path
    )


_BACKEND_KEYS = {
    "backend",
    "history_storage_mode",
    "mapping_kind",
    "recovery_backend",
    "selected_backend",
    "selected_sparse_backend",
}
_FALLBACK_KEYS = {
    "disabled_reason",
    "exclusion_reasons",
    "fallback_reason",
    "fallback_reasons",
}


def _observation_value(value: Any) -> str | None:
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _collect_observations(
    value: Any,
    path: tuple[str, ...] = (),
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    backends: list[tuple[str, str]] = []
    fallbacks: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key in sorted(value, key=str):
            key = str(raw_key)
            item = value[raw_key]
            item_path = (*path, key)
            rendered = _observation_value(item)
            if key in _BACKEND_KEYS and rendered is not None and not isinstance(item, Mapping):
                backends.append((_sanitize_path(item_path), rendered))
            if key in _FALLBACK_KEYS and rendered is not None:
                fallbacks.append((_sanitize_path(item_path), rendered))
            nested_backends, nested_fallbacks = _collect_observations(item, item_path)
            backends.extend(nested_backends)
            fallbacks.extend(nested_fallbacks)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            nested_backends, nested_fallbacks = _collect_observations(
                item, (*path, str(index))
            )
            backends.extend(nested_backends)
            fallbacks.extend(nested_fallbacks)
    return backends, fallbacks


def _deduplicate(items: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(set(items), key=lambda item: (item[0], item[1]))


def _environment_rows() -> tuple[tuple[str, ...], ...]:
    return (
        ("runtime", "python_version"),
        ("runtime", "python_implementation"),
        ("runtime", "platform"),
        ("runtime", "cpu"),
        ("runtime", "physical_cores"),
        ("runtime", "logical_cores"),
        ("packages", "numpy"),
        ("packages", "scipy"),
        ("packages", "numba"),
        ("packages", "llvmlite"),
        ("packages", "pypardiso"),
        ("packages", "mkl"),
        ("packages", "threadpoolctl"),
        ("packages", "ANYfeSolver"),
        ("packages", "ANYmaterial"),
        ("packages", "ANYmesher"),
        ("packages", "ANYgeometry"),
        ("packages", "ANYfileio"),
        ("jit", "enabled"),
        ("jit", "backend"),
    )


def render_report(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    baseline_path: Path,
    final_path: Path,
    numerical: Mapping[str, Any] | None,
    numerical_path: Path,
    thread_scaling: Mapping[str, Any] | None,
    thread_path: Path,
    decision_table: str | None,
    decision_path: Path,
    issues: Sequence[str],
) -> str:
    baseline_cases = _case_index(baseline)
    final_cases = _case_index(final)
    ordered_ids = list(baseline_cases)
    ordered_ids.extend(sorted(set(final_cases) - set(baseline_cases)))

    initial_sha = (
        _initial_revision(baseline, "initial_performance_2_sha")
        or _nested(baseline, "revision", "head_sha")
    )
    origin_main = (
        _initial_revision(baseline, "origin_main_sha")
        or _nested(baseline, "revision", "origin_main_sha")
    )
    merge_base = (
        _initial_revision(baseline, "merge_base_sha")
        or _nested(baseline, "revision", "merge_base_sha")
    )
    final_sha = _nested(final, "revision", "head_sha")

    lines = [
        "# ANYfeSolver Sol Ultra baseline/final comparison",
        "",
        f"- Report status: **{'INCOMPLETE' if issues else 'COMPLETE'}**",
        f"- Immutable initial `performance_2`: `{_markdown(initial_sha)}`",
        f"- Contemporaneous `origin/main`: `{_markdown(origin_main)}`",
        f"- Merge-base: `{_markdown(merge_base)}`",
        f"- Qualified source candidate on `performance_2`: `{_markdown(final_sha)}`",
        f"- Baseline artifact: `{baseline_path.as_posix()}`",
        f"- Final artifact: `{final_path.as_posix()}`",
        "",
    ]
    if issues:
        lines.extend(["## Release-evidence gaps", ""])
        lines.extend(f"- {item}" for item in issues)
        lines.append("")

    lines.extend(
        [
            "## Environment",
            "",
            "| Field | Baseline | Final | Match |",
            "| --- | --- | --- | --- |",
        ]
    )
    for path in _environment_rows():
        baseline_value = _nested(baseline, "environment", *path)
        final_value = _nested(final, "environment", *path)
        lines.append(
            f"| `{'.'.join(path)}` | {_markdown(baseline_value)} | "
            f"{_markdown(final_value)} | {'yes' if baseline_value == final_value else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Cold and warm timing",
            "",
            "Speedup is `baseline / final`; values above 1 are faster. Each row "
            "is independent and is not averaged across unlike workloads.",
            "",
            "| Case | Baseline status | Final status | Baseline cold | Final cold | Cold speedup | Baseline warm median | Final warm median | Warm speedup |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case_id in ordered_ids:
        base_case = baseline_cases.get(case_id, {})
        final_case = final_cases.get(case_id, {})
        base_cold = _measurement(base_case, "cold", "wall_seconds")
        final_cold = _measurement(final_case, "cold", "wall_seconds")
        base_warm = _measurement(base_case, "warm", "wall_seconds")
        final_warm = _measurement(final_case, "warm", "wall_seconds")
        lines.append(
            f"| `{case_id}` | {_markdown(base_case.get('status'))} | "
            f"{_markdown(final_case.get('status'))} | {_seconds(base_cold)} | "
            f"{_seconds(final_cold)} | {_ratio_text(_ratio(base_cold, final_cold))} | "
            f"{_seconds(base_warm)} | {_seconds(final_warm)} | "
            f"{_ratio_text(_ratio(base_warm, final_warm))} |"
        )

    lines.extend(
        [
            "",
            "## Peak-memory evidence",
            "",
            "`Final / baseline` below 1 uses less memory. Python peaks are "
            "per invocation. Process peak RSS can be process-lifetime cumulative "
            "on this platform and is therefore audit evidence, not an isolated "
            "per-case allocation measurement.",
            "",
            "| Case | Baseline warm Python peak | Final warm Python peak | Final / baseline | Baseline process peak RSS | Final process peak RSS | Final / baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case_id in ordered_ids:
        base_case = baseline_cases.get(case_id, {})
        final_case = final_cases.get(case_id, {})
        base_python = _measurement(base_case, "warm", "python_peak_bytes")
        final_python = _measurement(final_case, "warm", "python_peak_bytes")
        base_rss = _measurement(base_case, "warm", "process_peak_rss_bytes")
        final_rss = _measurement(final_case, "warm", "process_peak_rss_bytes")
        lines.append(
            f"| `{case_id}` | {_bytes(base_python)} | {_bytes(final_python)} | "
            f"{_ratio_text(_ratio(final_python, base_python))} | {_bytes(base_rss)} | "
            f"{_bytes(final_rss)} | {_ratio_text(_ratio(final_rss, base_rss))} |"
        )

    phase_names = list(baseline.get("phase_schema", []))
    phase_names.extend(
        name for name in final.get("phase_schema", []) if name not in phase_names
    )
    lines.extend(
        [
            "",
            "## Normalized phase coverage",
            "",
            "Unavailable means that the benchmark case exposed no normalized "
            "timer for that phase. It does not mean zero work and no phase "
            "speedup is inferred from a null value.",
            "",
            "| Phase | Baseline available | Final available | Final unavailable |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for phase in phase_names:
        baseline_available = sum(
            bool(_nested(case, "phases", phase, "available", default=False))
            for case in baseline_cases.values()
        )
        final_available = sum(
            bool(_nested(case, "phases", phase, "available", default=False))
            for case in final_cases.values()
        )
        lines.append(
            f"| `{phase}` | {baseline_available}/{len(baseline_cases)} | "
            f"{final_available}/{len(final_cases)} | "
            f"{len(final_cases) - final_available}/{len(final_cases)} |"
        )

    lines.extend(["", "## Independent numerical qualification", ""])
    if numerical is None:
        lines.append(f"No numerical comparison was available at `{numerical_path.as_posix()}`.")
    else:
        counts = _nested(numerical, "summary", "case_counts", default={})
        lines.extend(
            [
                f"- Overall status: **{str(numerical.get('status', 'unknown')).upper()}**",
                f"- Passed cases: {_markdown(counts.get('passed') if isinstance(counts, Mapping) else None)}",
                f"- Failed cases: {_markdown(counts.get('failed') if isinstance(counts, Mapping) else None)}",
                f"- Unavailable cases: {_markdown(_nested(numerical, 'summary', 'unavailable_cases'))}",
                f"- Metric failures: {_markdown(_nested(numerical, 'summary', 'metric_failures'))}",
                f"- Candidate commit: `{_markdown(_nested(numerical, 'candidate', 'source', 'commit'))}`",
                "- Timing and memory in this artifact are informational; numerical tolerances alone determine its pass/fail result.",
            ]
        )

    lines.extend(["", "## Thread-scaling summary", ""])
    if thread_scaling is None:
        lines.append(f"No thread-scaling evidence was available at `{thread_path.as_posix()}`.")
    else:
        lines.extend(
            [
                f"- Evidence revision: `{_markdown(_nested(thread_scaling, 'environment', 'revision', 'head_sha'))}`",
                f"- Campaign status: **{str(_nested(thread_scaling, 'summary', 'status', default='unknown')).upper()}**",
                f"- Default policy changed: {_markdown(_nested(thread_scaling, 'summary', 'default_thread_policy_changed'))}",
                "",
                "| Workload | Threads | Warm wall median | Warm phase | Phase median | Relative error vs one thread | Thread policy restored |",
                "| --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        workloads = thread_scaling.get("workloads", {})
        if isinstance(workloads, Mapping):
            for workload_name in sorted(workloads):
                workload = workloads[workload_name]
                entries = workload.get("entries", []) if isinstance(workload, Mapping) else []
                for entry in sorted(
                    (item for item in entries if isinstance(item, Mapping)),
                    key=lambda item: int(item.get("threads", 0)),
                ):
                    phases = _nested(entry, "measurements", "warm", "phase_seconds", default={})
                    if isinstance(phases, Mapping) and phases:
                        phase_name = sorted(phases)[0]
                        phase_median = _nested(phases, phase_name, "median")
                    else:
                        phase_name = "n/a"
                        phase_median = None
                    lines.append(
                        f"| `{workload_name}` | {_markdown(entry.get('threads'))} | "
                        f"{_seconds(_nested(entry, 'measurements', 'warm', 'wall_seconds', 'median'))} | "
                        f"`{phase_name}` | {_seconds(phase_median)} | "
                        f"{_markdown(_nested(entry, 'numerical_comparison', 'relative_l2_error'))} | "
                        f"{_markdown(_nested(entry, 'measurements', 'warm', 'all_thread_policies_restored'))} |"
                    )
        lines.extend(
            [
                "",
                "Decision: keep one thread as the qualification and recommended "
                "explicit policy. Do not infer an automatic default from logical-core count.",
            ]
        )

    lines.extend(["", "## Backend and fallback observations", ""])
    observed_any = False
    for case_id in ordered_ids:
        final_case = final_cases.get(case_id)
        if not isinstance(final_case, Mapping):
            continue
        payload = final_case.get("representative_warm_payload", {})
        selected: dict[str, Any] = {}
        if isinstance(payload, Mapping):
            for root_name in ("backend", "diagnostics", "resources", "results"):
                if root_name in payload:
                    selected[root_name] = payload[root_name]
        backends, fallbacks = _collect_observations(selected)
        backends = _deduplicate(backends)
        fallbacks = _deduplicate(fallbacks)
        if not backends and not fallbacks:
            continue
        observed_any = True
        lines.extend([f"### `{case_id}`", ""])
        for path, value in backends:
            lines.append(f"- Backend `{path}`: `{_markdown(value)}`")
        for path, value in fallbacks:
            lines.append(f"- Fallback `{path}`: `{_markdown(value)}`")
        lines.append("")
    if not observed_any:
        lines.extend(
            [
                "No normalized backend/fallback field was present in the selected "
                "final benchmark payloads. Consult result-local diagnostics; absence "
                "from this section is not evidence that every fast path ran.",
                "",
            ]
        )

    lines.extend(["## Promoted, deferred, and rejected workstreams", ""])
    if decision_table is None:
        lines.append(f"No decision table was available at `{decision_path.as_posix()}`.")
    else:
        lines.append(decision_table)

    limitations: list[str] = []
    for report in (baseline, final):
        for limitation in report.get("known_limitations", []):
            if str(limitation) not in limitations:
                limitations.append(str(limitation))
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "- Workstream microbenchmarks establish path-specific gates; this report does not relabel them as full-suite end-to-end measurements.",
            "- The benchmark JSON does not record Git dirty state. Release captures must be made from a separately verified clean checkout; the independent numerical artifact records this provenance.",
            "- Normalized phase coverage is intentionally sparse. Nested diagnostic counters may be more specific but are not substituted for absent phase timers.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "$env:PYPARDISO_MKL_RT = 'C:\\Python\\Python313\\Library\\bin\\mkl_rt.3.dll'",
            "$python = 'C:\\Github\\ANYfeSolver\\.venv\\Scripts\\python.exe'",
            "$baselineRoot = 'C:\\Github\\ANYfeSolver\\.perf2-worktrees\\baseline'",
            "$candidateRoot = (Resolve-Path -LiteralPath .).Path",
            "",
            "& $python -m pytest tests -q --basetemp=.pytest_tmp_sol_ultra_final",
            "$dirty = git status --porcelain",
            "if ($dirty) { throw \"Qualified source checkout is dirty before numerical capture: $dirty\" }",
            "& $python scripts/verify_sol_ultra_numerics.py capture --solver-root $candidateRoot --label candidate --suite full --output .sol_ultra_verify_candidate.json",
            "& $python scripts/verify_sol_ultra_numerics.py capture --solver-root $baselineRoot --label baseline --suite full --output .sol_ultra_verify_baseline.json",
            "& $python scripts/verify_sol_ultra_numerics.py compare --baseline .sol_ultra_verify_baseline.json --candidate .sol_ultra_verify_candidate.json --json-report reports/performance/sol_ultra_numerical_comparison.json --markdown-report reports/performance/sol_ultra_independent_verification.md",
            "",
            "& $python scripts/benchmark_nonlinear_assembly.py --nx 20 --ny 10 --repeats 10",
            "& $python scripts/benchmark_nonlinear_assembly.py --nx 20 --ny 10 --repeats 10 --weighted-mpc-rows 12",
            "& $python scripts/benchmark_sol_ultra_performance.py --suite full --repeats 3 --label final --output reports/performance/sol_ultra_final.json --no-markdown",
            "& $python scripts/benchmark_sol_ultra_thread_scaling.py --repeats 3 --output .sol_ultra_thread_scaling.json",
            "& $python scripts/compare_sol_ultra_performance.py --baseline reports/performance/sol_ultra_baseline.json --final reports/performance/sol_ultra_final.json --numerical reports/performance/sol_ultra_numerical_comparison.json --thread-scaling .sol_ultra_thread_scaling.json --decision-log reports/performance/sol_ultra_decision_log.md --output reports/performance/sol_ultra_comparison.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPORT_ROOT / "sol_ultra_baseline.json",
    )
    parser.add_argument(
        "--final",
        type=Path,
        default=REPORT_ROOT / "sol_ultra_final.json",
    )
    parser.add_argument(
        "--numerical",
        type=Path,
        default=REPORT_ROOT / "sol_ultra_numerical_comparison.json",
    )
    parser.add_argument(
        "--thread-scaling",
        type=Path,
        default=ROOT / ".sol_ultra_thread_scaling.json",
    )
    parser.add_argument(
        "--decision-log",
        type=Path,
        default=REPORT_ROOT / "sol_ultra_decision_log.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "sol_ultra_comparison.md",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write an explicitly incomplete draft and return success",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    baseline = _load_json(args.baseline)
    final = _load_json(args.final)
    issues = _validate_benchmark(
        baseline, expected_kind="baseline", path=args.baseline
    )
    issues.extend(_validate_benchmark(final, expected_kind="final", path=args.final))

    baseline_cases = _case_index(baseline)
    final_cases = _case_index(final)
    if set(baseline_cases) != set(final_cases):
        issues.append(
            "baseline/final case inventories differ: "
            f"missing={sorted(set(baseline_cases) - set(final_cases))}, "
            f"extra={sorted(set(final_cases) - set(baseline_cases))}"
        )
    if _nested(final, "suite", "name") != "full":
        issues.append("final benchmark suite is not 'full'")
    repeats = _nested(final, "suite", "repeats")
    if not isinstance(repeats, int) or repeats < 3:
        issues.append("final benchmark does not contain at least three warm repeats")
    if _nested(final, "revision", "branch") != "performance_2":
        issues.append("final benchmark was not captured on branch 'performance_2'")
    initial_sha = (
        _initial_revision(baseline, "initial_performance_2_sha")
        or _nested(baseline, "revision", "head_sha")
    )
    if _nested(final, "revision", "merge_base_sha") != initial_sha:
        issues.append(
            "final performance revision does not retain the immutable baseline as merge-base"
        )
    for environment_path in _environment_rows():
        baseline_value = _nested(baseline, "environment", *environment_path)
        final_value = _nested(final, "environment", *environment_path)
        if baseline_value != final_value:
            issues.append(
                "baseline/final environment mismatch at "
                f"{'.'.join(environment_path)}: {baseline_value!r} != {final_value!r}"
            )

    numerical: dict[str, Any] | None = None
    if args.numerical.exists():
        numerical = _load_json(args.numerical)
        if numerical.get("schema_version") != 1:
            issues.append("numerical artifact has an unsupported schema version")
        if numerical.get("artifact_kind") != "sol_ultra_numerical_comparison":
            issues.append("numerical artifact kind is not sol_ultra_numerical_comparison")
        if numerical.get("status") != "passed":
            issues.append(f"numerical comparison status is {numerical.get('status')!r}")
        if _nested(numerical, "summary", "metric_failures", default=0) != 0:
            issues.append("numerical comparison reports metric failures")
        if _nested(numerical, "summary", "unavailable_cases", default=0) != 0:
            issues.append("numerical comparison has unavailable cases")
        final_sha = _nested(final, "revision", "head_sha")
        candidate_sha = _nested(numerical, "candidate", "source", "commit")
        if not final_sha or candidate_sha != final_sha:
            issues.append(
                "numerical candidate commit does not match the qualified source revision in the final performance artifact"
            )
        if _nested(numerical, "baseline", "source", "commit") != initial_sha:
            issues.append(
                "numerical baseline commit does not match the immutable performance baseline"
            )
        if _nested(numerical, "candidate", "source", "dirty") is not False:
            issues.append("numerical candidate capture is dirty or lacks clean-state provenance")
        if _nested(numerical, "baseline", "source", "dirty") is not False:
            issues.append("numerical baseline capture is dirty or lacks clean-state provenance")
    else:
        issues.append(f"missing numerical comparison: {args.numerical}")

    thread_scaling: dict[str, Any] | None = None
    if args.thread_scaling.exists():
        thread_scaling = _load_json(args.thread_scaling)
        if _nested(thread_scaling, "summary", "status") != "completed":
            issues.append("thread-scaling campaign did not complete")
        if _nested(thread_scaling, "summary", "failed_sections", default=[]):
            issues.append("thread-scaling campaign contains failed sections")
    else:
        issues.append(f"missing thread-scaling evidence: {args.thread_scaling}")

    decision_table: str | None = None
    if args.decision_log.exists():
        decision_text = args.decision_log.read_text(encoding="utf-8")
        decision_table = _extract_markdown_section(decision_text, "## Decision table")
        if decision_table is None:
            issues.append("decision log has no '## Decision table' section")
    else:
        issues.append(f"missing decision log: {args.decision_log}")

    report = render_report(
        baseline,
        final,
        baseline_path=args.baseline,
        final_path=args.final,
        numerical=numerical,
        numerical_path=args.numerical,
        thread_scaling=thread_scaling,
        thread_path=args.thread_scaling,
        decision_table=decision_table,
        decision_path=args.decision_log,
        issues=issues,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "incomplete" if issues else "completed",
                "issues": issues,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if issues and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
