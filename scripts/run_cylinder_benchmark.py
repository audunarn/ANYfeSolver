"""Run the internal cylindrical shell benchmark and optionally write JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver import CylinderBenchmarkConfig, run_cylindrical_shell_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports") / "cylinder_benchmark.json")
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--pressure", type=float, default=100_000.0)
    parser.add_argument("--circumferential", type=int, default=16)
    parser.add_argument("--height-divisions", type=int, default=8)
    parser.add_argument("--use-8node-elements", action="store_true")
    parser.add_argument("--open-ends", action="store_true")
    args = parser.parse_args()

    config = CylinderBenchmarkConfig(
        radius=args.radius,
        height=args.height,
        thickness=args.thickness,
        pressure=args.pressure,
        num_circumferential=args.circumferential,
        num_height=args.height_divisions,
        use_8node_elements=args.use_8node_elements,
        closed_end_axial_load=not args.open_ends,
    )
    result = run_cylindrical_shell_benchmark(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"solver: {result.solver_status}")
    print(f"nominal hoop: {result.nominal.hoop_stress / 1.0e6:.6g} MPa")
    print(f"nominal axial: {result.nominal.axial_stress / 1.0e6:.6g} MPa")
    print(f"FE von Mises max: {result.fe_max_von_mises / 1.0e6:.6g} MPa")
    print(f"FE von Mises p95: {result.fe_p95_von_mises / 1.0e6:.6g} MPa")
    print(f"FE mid-height von Mises p95: {result.fe_mid_height_p95_von_mises / 1.0e6:.6g} MPa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
