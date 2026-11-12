"""Bounded four-Q4/two-S3 selected-pressure coupled press development probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver import solve_static_nonlinear, validate_general_contact
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    model, definition, settings, program = build_case(
        4, steps_per_segment=2, load_profile="tip",
        total_force_n=0.1, target_right_x_m=1.3,
    )
    settings = replace(
        settings, q4_source_pressure_basis="selected_q1_knots_dev",
        q4_source_pressure_knot_columns=(0, 1, 3, 4),
    )
    admission = validate_general_contact(model, definition, settings)
    if not admission.valid:
        raise ValueError(f"selected-basis contact admission failed: {admission.issues}")
    started = time.perf_counter()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    elapsed = time.perf_counter()-started
    terminal_replay_equal = None
    split_path_restart_equal = None
    changed_knot_rejected = None
    if result.status == "completed" and result.restart_checkpoint is not None:
        replay_model, replay_definition, _replay_settings, replay_program = build_case(
            4, steps_per_segment=2, load_profile="tip",
            total_force_n=0.1, target_right_x_m=1.3,
        )
        replay = solve_static_nonlinear(
            replay_model, contact=replay_definition,
            contact_settings=settings, contact_path_program=replay_program,
            max_iterations=18, emit_restart_checkpoint=True,
            restart_checkpoint=result.to_restart_checkpoint(),
        )
        terminal_replay_equal = bool(
            replay.status == "completed"
            and np.array_equal(replay.displacements, result.displacements)
            and replay.to_restart_checkpoint() == result.to_restart_checkpoint()
        )
        half_model, half_definition, _, half_program = build_case(
            4, steps_per_segment=2, load_profile="tip",
            total_force_n=0.1, target_right_x_m=1.3,
        )
        half = solve_static_nonlinear(
            half_model, contact=half_definition, contact_settings=settings,
            contact_path_program=half_program, max_iterations=18,
            max_load_factor=0.5, emit_restart_checkpoint=True,
        )
        if half.status == "completed" and half.restart_checkpoint is not None:
            resumed_model, resumed_definition, _, resumed_program = build_case(
                4, steps_per_segment=2, load_profile="tip",
                total_force_n=0.1, target_right_x_m=1.3,
            )
            resumed = solve_static_nonlinear(
                resumed_model, contact=resumed_definition,
                contact_settings=settings, contact_path_program=resumed_program,
                max_iterations=18, emit_restart_checkpoint=True,
                restart_checkpoint=half.to_restart_checkpoint(),
            )
            split_path_restart_equal = bool(
                resumed.status == "completed"
                and np.allclose(resumed.displacements, result.displacements,
                                rtol=1e-7, atol=1e-9)
                and resumed.to_restart_checkpoint() == result.to_restart_checkpoint()
            )
        changed_model, changed_definition, _, changed_program = build_case(
            4, steps_per_segment=2, load_profile="tip",
            total_force_n=0.1, target_right_x_m=1.3,
        )
        try:
            solve_static_nonlinear(
                changed_model, contact=changed_definition,
                contact_settings=replace(
                    settings, q4_source_pressure_knot_columns=(0, 2, 4),
                ),
                contact_path_program=changed_program, max_iterations=18,
                restart_checkpoint=result.to_restart_checkpoint(),
            )
        except ValueError as exc:
            changed_knot_rejected = "checkpoint" in str(exc).lower()
        else:
            changed_knot_rejected = False
    final_contact = result.info.get("general_contact")
    report = {
        "candidate": "selected_q1_knots_4q4_two_s3_press_dev",
        "source_faces": 4,
        "target_faces": 2,
        "selected_pressure_knot_columns": [0, 1, 3, 4],
        "total_downward_force_n": 0.1,
        "steps_per_segment": 2,
        "elapsed_seconds": elapsed,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "profile": admission.profile,
        "contact": final_contact,
        "history": result.info.get("force_displacement_history"),
        "augmentation_attempts": result.info.get("contact_augmentation_attempts"),
        "contact_failure": result.info.get("contact_failure"),
        "has_restart_checkpoint": result.restart_checkpoint is not None,
        "terminal_checkpoint_replay_exact": terminal_replay_equal,
        "half_path_steps": len(half.steps) if result.status == "completed" else None,
        "split_path_restart_equal": split_path_restart_equal,
        "changed_knot_restart_rejected": changed_knot_rejected,
        "final_displacement_norm_m": float(np.linalg.norm(result.displacements)),
    }
    output.write_text(json.dumps(report, indent=2, default=str)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("contact", "history", "augmentation_attempts",
                                     "contact_failure")}, indent=2))
    if (result.status != "completed" or not terminal_replay_equal
            or not split_path_restart_equal or not changed_knot_rejected):
        raise RuntimeError("selected-knot coupled press or restart check did not complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
