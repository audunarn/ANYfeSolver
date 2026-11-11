"""Run the manifest-driven beam/shell solver verification report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.beam_shell_verification import DEFAULT_BEAM_SHELL_VERIFICATION_PATH, write_beam_shell_verification_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BEAM_SHELL_VERIFICATION_PATH)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--case-id", action="append", default=None, help="Run only selected case id; may be repeated.")
    parser.add_argument(
        "--external-reference-report",
        type=Path,
        default=None,
        help="Existing executed report to preserve/consume, or destination for a deck-only handoff report.",
    )
    args = parser.parse_args()

    report = write_beam_shell_verification_report(
        args.output,
        markdown=args.markdown,
        selected_ids=args.case_id,
        external_reference_report=args.external_reference_report,
    )
    print(f"Wrote {args.output}")
    if args.markdown is not None:
        print(f"Wrote {args.markdown}")
    print(f"Status: {report['status']} counts={report['counts']}")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
