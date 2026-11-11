"""Write sparse buckling validity report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.buckling_validity import DEFAULT_BUCKLING_VALIDITY_PATH, write_buckling_validity_report  # noqa: E402


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Buckling Validity Report",
        "",
        f"- Status: {report['status']}",
        f"- Euler column relative error: {report['euler_column']['relative_error']:.3e}",
        f"- Sparse solver: {report['euler_column']['diagnostics']['solver']}",
        f"- Max buckling residual: {report['euler_column']['diagnostics']['max_residual_norm']:.3e}",
        f"- Higher-mode ratio to first: {report['higher_mode_range']['ratio_to_first']:.6g}",
        f"- Repeated-mode groups: {len(report['repeated_modes']['groups'])}",
        "",
        "## Known Limitations",
        "",
    ]
    for item in report["known_limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUCKLING_VALIDITY_PATH)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    report = write_buckling_validity_report(args.output)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.markdown is not None:
        print(f"Wrote {args.markdown}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
