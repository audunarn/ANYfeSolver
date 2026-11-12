"""Check a fresh-model restart of a mixed-shell four-phase path."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def main(reference_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"development output already exists: {output_path}")
    full = json.loads(reference_path.read_text(encoding="utf-8"))
    if (full.get("status") != "completed"
            or full.get("columns") not in (8, 16, 24)
            or full.get("four_phase") is not True
            or full.get("steps_per_segment") != 3
            or full.get("accepted_steps") != 12
            or len(full.get("history", ())) != 12):
        raise ValueError("reference is not a completed twelve-step mixed path")

    def candidate():
        model, definition, settings, program = build_case(
            full["columns"], four_phase=True, steps_per_segment=3,
            load_profile=full["load_profile"],
            total_force_n=full["total_downward_force_n"],
            target_right_x_m=full.get("target_right_x_m", 1.0),
        )
        settings = replace(
            settings,
            augmentation_schur_maximum_pressure_pa=(
                full["schur_maximum_pressure_pa"]),
            augmentation_schur_maximum_step_pa=(
                full["schur_maximum_step_pa"]),
            augmentation_schur_maximum_relinearizations=(
                full["schur_maximum_relinearizations"]),
            augmentation_schur_maximum_attempts=(
                full.get("schur_maximum_attempts", 2)),
        )
        return model, definition, settings, program

    started = time.perf_counter()
    model, definition, settings, program = candidate()
    prefix = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    if prefix.status != "completed" or len(prefix.steps) != 6:
        raise RuntimeError(f"split prefix did not complete: {prefix.failure_reason}")
    checkpoint = prefix.to_restart_checkpoint()
    checkpoint_bytes = json.dumps(
        checkpoint, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    model, definition, settings, program = candidate()
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        restart_checkpoint=checkpoint, emit_restart_checkpoint=True,
    )
    if resumed.status != "completed" or len(resumed.steps) != 12:
        raise RuntimeError(
            f"split continuation did not complete: {resumed.failure_reason}"
        )
    history = resumed.info["force_displacement_history"]
    differences = []
    for saved, actual in zip(full["history"], history, strict=True):
        source = saved["contact"]
        contact = actual["contact"]
        differences.append({
            "step": int(actual["step_index"]),
            "active_nodes_match": (
                int(contact["active_pressure_dofs"])
                == int(source["active_pressure_dofs"])
            ),
            "force_difference_n": float(np.max(np.abs(
                np.asarray(contact["integration_force"])
                - np.asarray(source["integration_force"])
            ))),
            "gap_difference_m": abs(
                float(contact["minimum_gap"])-float(source["minimum_gap"])
            ),
            "multiplier_change_difference_pa": abs(
                float(actual["normal_multiplier_change"])
                - float(saved["normal_multiplier_change"])
            ),
            "gap_tolerance_met": bool(contact["gap_tolerance_met"]),
            "pressure_gate_met": (
                float(actual["normal_multiplier_change"])
                <= settings.pressure_tolerance
            ),
            "force_wrench_imbalance_n": float(np.linalg.norm(
                np.asarray(contact["integration_force"])
                + np.asarray(contact["target_force"])
            )),
            "moment_wrench_imbalance_nm": float(np.linalg.norm(
                np.asarray(contact["integration_moment"])
                + np.asarray(contact["target_moment"])
            )),
        })
    final_displacement_difference_m = float(np.max(np.abs(
        np.asarray(resumed.displacements, dtype=float)
        - np.asarray(full["final_displacements"], dtype=float)
    )))
    final_pressure_difference_pa = float(np.max(np.abs(
        np.asarray(history[-1]["contact"]["normal_multiplier_estimate_pa"], dtype=float)
        - np.asarray(full["history"][-1]["contact"]["normal_multiplier_estimate_pa"], dtype=float)
    )))
    record = {
        "candidate": "mixed_q4_s3_split_restart_dev",
        "reference": str(reference_path),
        "columns": full["columns"],
        "target_right_x_m": full.get("target_right_x_m", 1.0),
        "prefix_steps": len(prefix.steps),
        "resumed_steps": len(resumed.steps),
        "prefix_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "elapsed_seconds": time.perf_counter()-started,
        "maximum_force_difference_n": max(x["force_difference_n"] for x in differences),
        "maximum_gap_difference_m": max(x["gap_difference_m"] for x in differences),
        "maximum_multiplier_change_difference_pa": max(
            x["multiplier_change_difference_pa"] for x in differences
        ),
        "final_displacement_difference_m": final_displacement_difference_m,
        "final_pressure_difference_pa": final_pressure_difference_pa,
        "all_active_nodes_match": all(x["active_nodes_match"] for x in differences),
        "all_gap_and_pressure_gates_met": all(
            x["gap_tolerance_met"] and x["pressure_gate_met"]
            for x in differences
        ),
        "differences": differences,
    }
    output_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key.startswith(("prefix_", "resumed_", "maximum_", "all_", "elapsed_"))},
                     indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python restart_probe.py FULL_PATH_JSON OUTPUT")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
