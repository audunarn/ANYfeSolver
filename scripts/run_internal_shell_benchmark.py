"""Run internal shell convergence benchmarks and write text tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver import run_simple_supported_shell_convergence, write_internal_shell_convergence_table


def _parse_divisions(value: str) -> Sequence[int]:
    divisions = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        division = int(stripped)
        if division <= 0:
            raise argparse.ArgumentTypeError("divisions must be positive integers")
        divisions.append(division)
    if not divisions:
        raise argparse.ArgumentTypeError("at least one division is required")
    return tuple(divisions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "shell_benchmarks")
    parser.add_argument("--divisions", type=_parse_divisions, default=(2, 4, 8))
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--pressure", type=float, default=1000.0)
    parser.add_argument("--stress-reference", type=float, default=None)
    parser.add_argument("--displacement-reference", type=float, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for use_8node, filename in ((False, "S4_internal.txt"), (True, "S8_internal.txt")):
        results = run_simple_supported_shell_convergence(
            divisions=args.divisions,
            length=args.length,
            width=args.width,
            thickness=args.thickness,
            pressure=args.pressure,
            use_8node_elements=use_8node,
            stress_reference=args.stress_reference,
            displacement_reference=args.displacement_reference,
        )
        path = write_internal_shell_convergence_table(results, args.output_dir / filename)
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
