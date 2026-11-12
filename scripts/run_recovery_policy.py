"""Generate the FE selective-recovery/resource-policy smoke report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.recovery_policy import DEFAULT_RECOVERY_POLICY_PATH, write_recovery_policy_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECOVERY_POLICY_PATH)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    report = write_recovery_policy_report(args.output, markdown=args.markdown)
    print(f"Wrote {args.output}")
    if args.markdown is not None:
        print(f"Wrote {args.markdown}")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
