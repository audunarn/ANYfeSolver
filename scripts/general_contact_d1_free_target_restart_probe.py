"""Development fresh-model split restart for the deformable-target cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_free_target_press_probe import (
    build_free_target_case,
)


def main(reference_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("status") != "completed"
            or reference.get("accepted_steps") != 12
            or not reference.get("four_phase")
            or reference.get("columns") != 8
            or reference.get("equilibrium_tolerance") != 1.0e-8):
        raise ValueError("reference is not the completed free-target cycle")
    started = time.perf_counter()
    report = {
        "candidate": "free_target_8q4_fresh_model_split_restart_dev",
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "split_after_step": 6,
        "equilibrium_tolerance": 1.0e-8,
    }
    try:
        model, definition, settings, program = build_free_target_case(
            8, four_phase=True)
        prefix = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            tolerance=1.0e-8, max_load_factor=0.5,
            emit_restart_checkpoint=True,
        )
        report["prefix_status"] = prefix.status
        report["prefix_failure_reason"] = prefix.failure_reason
        report["prefix_steps"] = len(prefix.steps)
        if prefix.status != "completed" or len(prefix.steps) != 6:
            raise RuntimeError("six-step prefix did not complete")
        checkpoint = prefix.to_restart_checkpoint()
        report["prefix_checkpoint_sha256"] = hashlib.sha256(
            json.dumps(checkpoint, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")
        ).hexdigest()
        fresh_model, fresh_definition, fresh_settings, fresh_program = (
            build_free_target_case(8, four_phase=True)
        )
        resumed = solve_static_nonlinear(
            fresh_model, contact=fresh_definition,
            contact_settings=fresh_settings,
            contact_path_program=fresh_program, max_iterations=18,
            tolerance=1.0e-8, restart_checkpoint=checkpoint,
            emit_restart_checkpoint=True,
        )
        report["resumed_status"] = resumed.status
        report["resumed_failure_reason"] = resumed.failure_reason
        report["resumed_steps"] = len(resumed.steps)
        if resumed.status != "completed" or len(resumed.steps) != 12:
            raise RuntimeError("fresh-model continuation did not complete")
        history = resumed.info["force_displacement_history"]
        differences = []
        for saved, actual in zip(reference["history"], history, strict=True):
            contact = actual["contact"]
            force = np.asarray(contact["integration_force"], dtype=float)
            target = np.asarray(contact["target_force"], dtype=float)
            differences.append({
                "step": int(actual["step_index"]),
                "step_matches": int(actual["step_index"]) == saved["step"],
                "active_matches": int(contact["active_pressure_dofs"])
                == saved["active_pressure_dofs"],
                "normal_force_absolute_difference_n": abs(
                    float(force[2])-saved["normal_force_n"]),
                "gap_absolute_difference_m": abs(
                    float(contact["minimum_gap"])-saved["minimum_gap_m"]),
                "pressure_change_absolute_difference_pa": abs(
                    float(actual["normal_multiplier_change"])
                    -saved["pressure_change_pa"]),
                "pressure_rank_matches": int(contact["pressure_gram_scaled_rank"])
                == saved["pressure_rank"],
                "wrench_force_imbalance_n": float(np.linalg.norm(force+target)),
                "gap_gate_met": bool(contact["gap_tolerance_met"]),
                "pressure_gate_met": (
                    actual["normal_multiplier_change"]
                    <= fresh_settings.pressure_tolerance),
                "source_pressure_node_ids": contact["source_pressure_node_ids"],
                "normal_multiplier_estimate_pa": contact[
                    "normal_multiplier_estimate_pa"],
            })
        report["differences"] = differences
        # Development metadata parity only: the full-cycle reference did
        # not retain the pressure vectors needed for a state-identity gate.
        report["all_step_identities_match"] = all(
            row["step_matches"] and row["active_matches"]
            and row["pressure_rank_matches"] for row in differences)
        report["all_gap_and_pressure_gates_met"] = all(
            row["gap_gate_met"] and row["pressure_gate_met"]
            for row in differences)
        for field in ("normal_force_absolute_difference_n",
                      "gap_absolute_difference_m",
                      "pressure_change_absolute_difference_pa",
                      "wrench_force_imbalance_n"):
            report[f"maximum_{field}"] = max(row[field] for row in differences)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.perf_counter()-started
    output_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "failure_reason", "prefix_status", "prefix_steps",
        "resumed_status", "resumed_steps", "all_step_identities_match",
        "all_gap_and_pressure_gates_met",
        "maximum_normal_force_absolute_difference_n",
        "maximum_gap_absolute_difference_m",
        "maximum_pressure_change_absolute_difference_pa",
        "elapsed_seconds",
    ) if key in report}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.reference, args.output)
