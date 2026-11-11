"""Planar Q4/S3 press, slide, open and recontact under distributed force.

Run from the ANYfeSolver checkout with ``python -m
examples.general_contact_distributed_q4_s3_cycle``. The default eight-Q4 mesh
is a short demonstration. The independently reviewed response uses 24 Q4s;
general C1 acceptance remains open.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import anyfesolver
from anyfesolver import solve_static_nonlinear, validate_general_contact

from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def run(columns: int) -> dict:
    if columns not in (8, 16, 24):
        raise ValueError("choose 8, 16 or 24 Q4 source faces")
    model, definition, settings, program = build_case(
        columns, four_phase=True, steps_per_segment=3,
        load_profile="distributed", total_force_n=0.3,
        target_right_x_m=1.3,
    )
    settings = replace(
        settings,
        augmentation_schur_maximum_pressure_pa=250.0,
        augmentation_schur_maximum_step_pa=200.0,
    )
    capability = validate_general_contact(model, definition, settings)
    if not capability.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(capability.issues))
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed" or len(result.steps) != 12:
        raise RuntimeError(
            f"contact stopped after {len(result.steps)} steps: "
            f"{result.status}: {result.failure_reason}"
        )
    history = result.info["force_displacement_history"]
    records = [{
        "step": int(row["step_index"]),
        "normal_force_n": float(row["contact"]["integration_force"][2]),
        "support_vertical_reaction_n": sum(
            float(values[2]) for values in result.steps[index].support_reactions.values()
        ),
        "minimum_gap_m": float(row["contact"]["minimum_gap"]),
        "active_pressure_modes": int(row["contact"]["active_pressure_dofs"]),
        "pressure_change_pa": float(row["normal_multiplier_change"]),
    } for index, row in enumerate(history)]
    if (not all(row["contact"]["gap_tolerance_met"] for row in history)
            or max(item["pressure_change_pa"] for item in records)
            > settings.pressure_tolerance):
        raise RuntimeError("a contact acceptance gate was not met")
    return {
        "capability_profile": capability.profile,
        "columns": columns,
        "status": result.status,
        "gap_tolerance_m": definition.interactions[0].gap_tolerance,
        "has_restart_checkpoint": result.restart_checkpoint is not None,
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "steps": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, choices=(8, 16, 24), default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.columns)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
