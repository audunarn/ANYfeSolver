"""Compare stored S4-improved benchmark artifacts against frozen hard gates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "s4_improved"
SCHEMA_NAME = "anyfesolver.s4_improved.benchmark"
SCHEMA_VERSION = 1


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _index(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for case in report.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in result:
            raise ValueError(f"duplicate or empty case id {case_id!r}")
        result[case_id] = case
    return result


def _median(case: Mapping[str, Any], side: str = "candidate") -> float | None:
    measurements = case.get("measurements")
    if not isinstance(measurements, Mapping):
        return None
    values = measurements.get(side)
    if not isinstance(values, Mapping):
        return None
    return _number(values.get("median"))


def compare_reports(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    issues: list[str] = []
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("schema_name") != SCHEMA_NAME:
            issues.append(f"{label}: unsupported schema_name")
        if report.get("schema_version") != SCHEMA_VERSION:
            issues.append(f"{label}: unsupported schema_version")
    baseline_cases = _index(baseline)
    candidate_cases = _index(candidate)
    if set(baseline_cases) != set(candidate_cases):
        issues.append("baseline/candidate case inventories differ")
    rows = []
    for case_id in sorted(set(baseline_cases) | set(candidate_cases)):
        old = baseline_cases.get(case_id, {})
        new = candidate_cases.get(case_id, {})
        old_value = _median(old)
        new_value = _median(new)
        ratio = None if old_value is None or new_value is None or old_value <= 0.0 else new_value / old_value
        stretch = _number(new.get("stretch_ratio"))
        hard = _number(new.get("hard_ratio"))
        if ratio is None or hard is None or stretch is None:
            status = "unavailable"
        elif ratio > hard:
            status = "hard_fail"
        elif ratio > stretch:
            status = "stretch"
        else:
            status = "pass"
        rows.append(
            {
                "case_id": case_id,
                "baseline_median": old_value,
                "candidate_median": new_value,
                "candidate_over_baseline": ratio,
                "stretch_ratio": stretch,
                "hard_ratio": hard,
                "status": status,
                "activation": new.get("activation"),
                "fallback_reason": new.get("fallback_reason"),
            }
        )
    hard_failures = [row["case_id"] for row in rows if row["status"] == "hard_fail"]
    unavailable = [row["case_id"] for row in rows if row["status"] == "unavailable"]
    return {
        "schema_name": "anyfesolver.s4_improved.comparison",
        "schema_version": 1,
        "issues": issues,
        "cases": rows,
        "summary": {
            "status": "failed" if issues or hard_failures else "incomplete" if unavailable else "passed",
            "hard_failures": hard_failures,
            "unavailable": unavailable,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# S4 improved performance comparison",
        "",
        f"Status: `{report['summary']['status']}`",
        "",
        "| Case | Baseline | Candidate | Ratio | Stretch | Hard | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["cases"]:
        def shown(value: Any, suffix: str = "") -> str:
            number = _number(value)
            return "n/a" if number is None else f"{number:.6g}{suffix}"

        lines.append(
            f"| `{row['case_id']}` | {shown(row['baseline_median'])} | "
            f"{shown(row['candidate_median'])} | {shown(row['candidate_over_baseline'], 'x')} | "
            f"{shown(row['stretch_ratio'], 'x')} | {shown(row['hard_ratio'], 'x')} | "
            f"{row['status']} |"
        )
    if report["issues"]:
        lines.extend(["", "## Validation issues", ""])
        lines.extend(f"- {issue}" for issue in report["issues"])
    lines.extend(
        [
            "",
            "Missing or unavailable measurements remain `n/a`; this report never "
            "manufactures a pass from absent evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=REPORT_ROOT / "baseline_performance.json")
    parser.add_argument("--candidate", type=Path, default=REPORT_ROOT / "performance.json")
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "performance.md")
    parser.add_argument("--json-output", type=Path, default=REPORT_ROOT / "performance_comparison.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_reports(_load(args.baseline), _load(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report), encoding="utf-8")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = report["summary"]["status"]
    if status == "passed" or (status == "incomplete" and args.allow_incomplete):
        return 0
    return 2 if status == "incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
