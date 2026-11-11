"""Bounded contact-free structural mesh comparison for the central D1 patch."""

import argparse
import json
from pathlib import Path
import time

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_finite_area_load import build_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", nargs=2, type=int, action="append",
                        metavar=("X_DIVISIONS", "Y_DIVISIONS"))
    args = parser.parse_args()

    rows = []
    cases = args.case or ((8, 4), (8, 8), (16, 4))
    for x_divisions, y_divisions in cases:
        model, load, _definition, _settings = build_case(
            x_divisions, rows=y_divisions, total_load_n=2.0,
            normal_stiffness=100_000.0, geometry_profile="central",
        )
        center_id = 1 + (y_divisions // 2) * (x_divisions + 1)
        center_id += x_divisions // 2
        started = time.perf_counter()
        result = solve_static_nonlinear(
            model, load, control="force", kinematics="von_karman",
            num_steps=1, max_iterations=12, min_step_fraction=0.5,
        )
        rows.append({
            "x_divisions": x_divisions,
            "y_divisions": y_divisions,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "center_uz_m": float(result.displacements[
                model.mesh.get_node(center_id).dofs[2]
            ]),
            "elapsed_seconds": time.perf_counter() - started,
        })
    args.output.write_text(json.dumps({
        "profile": "contact_free_central_2n_von_karman_dev",
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
