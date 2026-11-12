"""Measure paired S4-improved kernel samples without hiding unavailable paths.

This runner is intentionally separated from scientific qualification.  It
records every adjacent scalar/compiled sample, cold setup, retained bytes,
backend activation, and fallback reason.  Eleven-sample or full campaigns are
PERF-lease gated; ``--list`` is always safe.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT_ROOT = ROOT / "reports" / "s4_improved"
SCHEMA_NAME = "anyfesolver.s4_improved.benchmark"
SCHEMA_VERSION = 1
PERF_LEASE_ENV = "ANYSOLVER_PERF_LEASE"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    stretch_ratio: float
    hard_ratio: float
    unit: str = "seconds"


BENCHMARK_CASES = (
    BenchmarkCase("linear_K", "Linear K shell-local work", 1.40, 1.70),
    BenchmarkCase("mass", "Consistent mass shell-local work", 1.15, 1.30),
    BenchmarkCase("KG", "Geometric stiffness shell-local work", 1.40, 1.80),
    BenchmarkCase(
        "elastic_residual_tangent",
        "Persistent elastic residual and tangent",
        1.75,
        2.50,
    ),
    BenchmarkCase(
        "direct_reduced_residual_tangent",
        "Direct-reduced elastic residual and tangent",
        2.25,
        3.50,
    ),
    BenchmarkCase("plastic_residual_tangent", "Plastic residual and tangent", 1.50, 2.00),
    BenchmarkCase("recovery", "Compiled large-selection recovery", 1.60, 2.25),
    BenchmarkCase("retained_bytes", "Retained improved numeric bytes per Q4", 1.50, 1.80, "bytes"),
)


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values:
        return {"samples": [], "median": None, "minimum": None, "maximum": None}
    return {
        "samples": values,
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "mean": float(statistics.fmean(values)),
    }


def paired_samples(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    repeats: int,
    warmup: bool = True,
) -> dict[str, Any]:
    """Return adjacent, order-balanced samples and Python allocation peaks."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup:
        baseline()
        candidate()
    baseline_seconds: list[float] = []
    candidate_seconds: list[float] = []
    pair_order: list[str] = []
    peak_python_bytes = 0
    tracemalloc.start()
    try:
        for sample in range(repeats):
            calls = (
                (("baseline", baseline), ("candidate", candidate))
                if sample % 2 == 0
                else (("candidate", candidate), ("baseline", baseline))
            )
            pair_order.append("baseline_candidate" if sample % 2 == 0 else "candidate_baseline")
            for label, operation in calls:
                gc.collect()
                start = time.perf_counter()
                operation()
                elapsed = float(time.perf_counter() - start)
                if label == "baseline":
                    baseline_seconds.append(elapsed)
                else:
                    candidate_seconds.append(elapsed)
                _current, peak = tracemalloc.get_traced_memory()
                peak_python_bytes = max(peak_python_bytes, int(peak))
    finally:
        tracemalloc.stop()
    baseline_summary = _summary(baseline_seconds)
    candidate_summary = _summary(candidate_seconds)
    baseline_median = baseline_summary["median"]
    candidate_median = candidate_summary["median"]
    ratio = (
        None
        if baseline_median is None or candidate_median is None or baseline_median <= 0.0
        else float(candidate_median / baseline_median)
    )
    return {
        "pair_order": pair_order,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_over_baseline": ratio,
        "python_peak_bytes": peak_python_bytes,
    }


def _git(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _load_case_factories() -> tuple[Mapping[str, Callable[[], Any]] | None, str | None]:
    """Load the optional benchmark adapter supplied with the completed batch.

    The adapter is absent during pre-kernel scaffolding.  That is reported as
    unavailable and never converted into a successful timing result.
    """

    try:
        module = importlib.import_module(
            "anyfesolver.shell_formulations.mitc4_plus_d_batch"
        )
    except Exception as exc:
        return None, f"batch_module_unavailable: {exc!r}"
    factory = getattr(module, "qualification_benchmark_factories", None)
    if not callable(factory):
        return None, "qualification_benchmark_factories_not_implemented"
    try:
        factories = factory()
    except Exception as exc:
        return None, f"benchmark_factory_failed: {exc!r}"
    if not isinstance(factories, Mapping):
        return None, "benchmark_factory_returned_non_mapping"
    return factories, None


def _run_case(
    spec: BenchmarkCase,
    factories: Mapping[str, Callable[[], Any]],
    repeats: int,
) -> dict[str, Any]:
    factory = factories.get(spec.case_id)
    if factory is None:
        return {**asdict(spec), "status": "unavailable", "fallback_reason": "case_factory_missing"}
    try:
        fixture = factory()
        baseline = fixture["baseline"]
        candidate = fixture["candidate"]
        setup = fixture.get("setup", {})
        samples = paired_samples(baseline, candidate, repeats=repeats)
        ratio = samples["candidate_over_baseline"]
        gate = (
            "unavailable"
            if ratio is None
            else "hard_fail"
            if ratio > spec.hard_ratio
            else "stretch"
            if ratio > spec.stretch_ratio
            else "pass"
        )
        return {
            **asdict(spec),
            "status": "completed",
            "gate": gate,
            "measurements": samples,
            "setup": setup,
            "activation": fixture.get("activation", {}),
            "fallback_reason": fixture.get("fallback_reason"),
        }
    except Exception as exc:
        return {**asdict(spec), "status": "failed", "error": repr(exc)}


def benchmark_manifest() -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "cases": [asdict(case) for case in BENCHMARK_CASES],
        "protocol": {
            "warm_jit_outside_timing": True,
            "paired_adjacent_samples": True,
            "balanced_pair_order": True,
            "full_minimum_repeats": 11,
            "default_numba_threads": 1,
            "live_geometry_calls_allowed": 0,
            "unavailable_is_pass": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--perf-lease", default=os.environ.get(PERF_LEASE_ENV))
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "performance.json")
    parser.add_argument("--allow-unavailable", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.list:
        print(json.dumps(benchmark_manifest(), indent=2, sort_keys=True))
        return 0
    if (args.full or args.repeats >= 11) and not args.perf_lease:
        raise SystemExit(
            f"full/11-sample benchmarking requires --perf-lease or {PERF_LEASE_ENV}"
        )
    if args.full and args.repeats < 11:
        raise SystemExit("--full requires --repeats >= 11")

    factories, unavailable_reason = _load_case_factories()
    if factories is None:
        cases = [
            {**asdict(spec), "status": "unavailable", "fallback_reason": unavailable_reason}
            for spec in BENCHMARK_CASES
        ]
    else:
        cases = [_run_case(spec, factories, args.repeats) for spec in BENCHMARK_CASES]
    unavailable = sum(case["status"] == "unavailable" for case in cases)
    failed = sum(
        case["status"] == "failed" or case.get("gate") == "hard_fail" for case in cases
    )
    report = {
        **benchmark_manifest(),
        "report_kind": str(args.label),
        "revision": {"head_sha": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current")},
        "suite": {"name": "full" if args.full else "smoke", "repeats": args.repeats},
        "perf_lease": args.perf_lease,
        "cases": cases,
        "summary": {"failed": failed, "unavailable": unavailable, "completed": len(cases) - failed - unavailable},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    if failed:
        return 1
    if unavailable and not args.allow_unavailable:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
