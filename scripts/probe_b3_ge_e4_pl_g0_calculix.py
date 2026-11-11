"""G0 external-runner smoke: execute and parse one existing analytical plate case.

This is deliberately not a mixed B3-GE/E4-PL external comparison. It verifies
the executable, deck, parser and comparison path before new reference cases
are registered and derived.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anyfesolver.external_references import (
    _pressure_plate_case,
    calculix_solver_provenance,
    run_calculix_reference_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    executable = args.executable.resolve(strict=True)
    provenance = calculix_solver_provenance(executable)
    case = _pressure_plate_case(args.root / "decks")
    result = run_calculix_reference_case(
        case,
        executable=executable,
        run_root=args.root / "runs",
        timeout_seconds=120.0,
        solver_provenance=provenance,
    )
    record = {"purpose": "G0A external execution/parser smoke, not mixed qualification",
              "solver": provenance, "case": case.to_dict(), "result": result}
    (args.root / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "comparisons": result["comparisons"],
                      "result_path": str(args.root / "result.json")}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
