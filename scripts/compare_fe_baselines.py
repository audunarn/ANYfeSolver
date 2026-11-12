"""Compare FE solver baseline JSON against a freshly generated candidate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.baselines import DEFAULT_BASELINE_PATH, compare_baseline_documents, generate_baseline_document, load_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_BASELINE_PATH, help="Reference baseline JSON path.")
    parser.add_argument("--candidate", type=Path, default=None, help="Optional candidate baseline JSON path. If omitted, generate one in memory.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON comparison report path.")
    args = parser.parse_args()

    reference = load_baseline(args.reference)
    if args.candidate is None:
        candidate = generate_baseline_document(include_timing=False)
    else:
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))

    report = compare_baseline_documents(reference, candidate)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        with tempfile.NamedTemporaryFile("w", delete=True, encoding="utf-8") as tmp:
            json.dump(report, tmp)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
