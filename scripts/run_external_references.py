"""Generate or execute external FE reference cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.external_references import (
    DEFAULT_CALCULIX_RUN_PATH,
    DEFAULT_EXTERNAL_REFERENCE_PATH,
    write_external_reference_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXTERNAL_REFERENCE_PATH)
    parser.add_argument("--deck-dir", type=Path, default=Path("reports/external_references/decks"))
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run CalculiX and enforce parsed numerical comparisons; without this flag only decks are generated.",
    )
    parser.add_argument("--calculix", type=Path, default=None, help="Explicit ccx executable path (otherwise env/PATH discovery)")
    parser.add_argument(
        "--calculix-arg",
        action="append",
        default=[],
        help="Argument inserted before '-i JOB'; repeat for wrapper commands.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_CALCULIX_RUN_PATH)
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-case execution timeout in seconds")
    args = parser.parse_args()
    if not args.execute and (args.calculix is not None or args.calculix_arg):
        parser.error("--calculix and --calculix-arg require --execute")

    report = write_external_reference_report(
        args.output,
        deck_dir=args.deck_dir,
        markdown=args.markdown,
        execute=args.execute,
        calculix_executable=args.calculix,
        calculix_args=args.calculix_arg,
        run_dir=args.run_dir,
        timeout_seconds=args.timeout,
    )
    print(f"Wrote {args.output}")
    if args.markdown is not None:
        print(f"Wrote {args.markdown}")
    print(f"Generated {len(report.get('cases', []))} external reference decks in {args.deck_dir}")
    print(f"External reference status: {report.get('status')}")
    return 0 if report.get("status") in {"passed", "not_executed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
