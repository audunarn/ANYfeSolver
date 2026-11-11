"""Run local FE solver infrastructure benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.benchmarks import DEFAULT_BENCHMARK_PATH, write_benchmark_report


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# FE Infrastructure Benchmark Report",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Commit: {report.get('commit') or 'unknown'}",
        f"- Platform: {report.get('environment', {}).get('platform')}",
        f"- Python: {report.get('environment', {}).get('python')}",
        "",
        "## Cases",
        "",
    ]
    for name, case in report.get("cases", {}).items():
        timing = case.get("timing", {})
        memory = case.get("memory", {})
        lines.append(f"### {name}")
        lines.append(f"- Status: {case.get('status')}")
        if case.get("shell_order"):
            lines.append(f"- Shell order: {case.get('shell_order')}")
        if "jit_enabled" in case:
            lines.append(f"- JIT enabled: {case.get('jit_enabled')}")
            if case.get("jit_disabled_reason"):
                lines.append(f"- JIT disabled reason: {case.get('jit_disabled_reason')}")
        if "parallel_threads" in case:
            lines.append(f"- Parallel threads: {case.get('parallel_threads')}")
        lines.append(f"- Wall time: {timing.get('wall_seconds', 0.0):.6f} s")
        lines.append(f"- Peak Python memory: {memory.get('peak_bytes', 0)} bytes")
        backend = case.get("backend")
        if isinstance(backend, dict) and backend.get("backend"):
            lines.append(f"- Solver backend: {backend.get('backend')}")
            if backend.get("auto_backend_policy"):
                lines.append(f"- Backend policy: {backend.get('auto_backend_policy')}")
        cache = case.get("cache") or case.get("factorization_cache")
        if isinstance(cache, dict):
            lines.append(
                f"- Factorization cache: {cache.get('hits', 0)} hits, "
                f"{cache.get('misses', 0)} misses, {cache.get('entries', 0)} entries"
            )
        for key, value in sorted(timing.items()):
            if key != "wall_seconds" and isinstance(value, (int, float)):
                lines.append(f"- {key}: {value:.6g}")
        results = case.get("results", {})
        for key in (
            "relative_force_error",
            "relative_tangent_error",
            "matrix_difference_norm",
            "static_deleted_records",
            "static_max_fracture_utilization",
            "impact_max_damage",
            "impact_deleted_count",
            "sub_softening_rebuilds_skipped",
        ):
            value = results.get(key)
            if isinstance(value, bool):
                lines.append(f"- {key}: {value}")
            elif isinstance(value, (int, float)):
                lines.append(f"- {key}: {value:.6g}")
        lines.append("")
    lines.extend(["## Known Limitations", ""])
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BENCHMARK_PATH, help="JSON benchmark output path.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional Markdown benchmark report path.")
    args = parser.parse_args()

    report = write_benchmark_report(args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, args.markdown)
    print(json.dumps({"status": "completed", "cases": list(report["cases"].keys()), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
