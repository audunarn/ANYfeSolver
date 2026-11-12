"""Generate deterministic FE solver baseline JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.baselines import DEFAULT_BASELINE_PATH, write_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE_PATH, help="Baseline JSON output path.")
    parser.add_argument("--no-timing", action="store_true", help="Write null timing fields for deterministic diffs.")
    args = parser.parse_args()

    document = write_baseline(args.output, include_timing=not args.no_timing)
    print(f"Wrote {args.output} with {len(document['cases'])} FE baseline cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
