"""Compare the force-driven four-phase split restart with saved full-path data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_mixed_jet2_schur_probe import make_candidate


def main(reference_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"development output already exists: {output_path}")
    full = json.loads(reference_path.read_text(encoding="utf-8"))
    if (full.get("candidate") != "force_driven_q4_s3_jet2_schur_four_phase_dev"
            or full.get("status") != "completed"
            or full.get("accepted_steps") != 8
            or len(full.get("history", ())) != 8):
        raise ValueError("full-path reference is not the completed eight-step candidate")
    model, definition, settings, program, _ = make_candidate()
    started = time.perf_counter()
    prefix = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        max_load_factor=0.5, emit_restart_checkpoint=True,
    )
    if prefix.status != "completed" or len(prefix.steps) != 4:
        raise RuntimeError(f"split prefix did not complete: {prefix.failure_reason}")
    checkpoint = prefix.to_restart_checkpoint()
    checkpoint_bytes = json.dumps(
        checkpoint, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    fresh_model, fresh_definition, fresh_settings, fresh_program, _ = make_candidate()
    resumed = solve_static_nonlinear(
        fresh_model, contact=fresh_definition, contact_settings=fresh_settings,
        contact_path_program=fresh_program, max_iterations=18,
        restart_checkpoint=checkpoint, emit_restart_checkpoint=True,
    )
    if resumed.status != "completed" or len(resumed.steps) != 8:
        raise RuntimeError(f"split continuation did not complete: {resumed.failure_reason}")
    history = resumed.info["force_displacement_history"]
    if len(history) != 8:
        raise RuntimeError("split continuation did not retain all eight phases")
    differences = []
    for saved, actual in zip(full["history"], history, strict=True):
        contact = actual["contact"]
        differences.append({
            "step": actual["step_index"],
            "active_nodes_match": (
                int(contact["active_pressure_dofs"])
                == int(saved["active_pressure_dofs"])
            ),
            "normal_force_absolute_difference_n": abs(
                float(contact["integration_force"][2])
                - float(saved["integration_force_n"][2])
            ),
            "minimum_gap_absolute_difference_m": abs(
                float(contact["minimum_gap"])-float(saved["minimum_gap_m"])
            ),
            "multiplier_change_absolute_difference_pa": abs(
                float(actual["normal_multiplier_change"])
                - float(saved["normal_multiplier_change_pa"])
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
    record = {
        "candidate": "force_driven_q4_s3_jet2_schur_split_restart_dev",
        "reference": str(reference_path),
        "prefix_steps": len(prefix.steps),
        "resumed_steps": len(resumed.steps),
        "prefix_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "elapsed_seconds": time.perf_counter()-started,
        "maximum_force_difference_n": max(
            row["normal_force_absolute_difference_n"] for row in differences
        ),
        "maximum_gap_difference_m": max(
            row["minimum_gap_absolute_difference_m"] for row in differences
        ),
        "maximum_multiplier_change_difference_pa": max(
            row["multiplier_change_absolute_difference_pa"] for row in differences
        ),
        "all_active_nodes_match": all(
            row["active_nodes_match"] for row in differences
        ),
        "all_gap_and_pressure_gates_met": all(
            row["gap_tolerance_met"] and row["pressure_gate_met"]
            for row in differences
        ),
        "differences": differences,
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "prefix_steps", "resumed_steps", "maximum_force_difference_n",
        "maximum_gap_difference_m", "maximum_multiplier_change_difference_pa",
        "all_active_nodes_match", "all_gap_and_pressure_gates_met",
        "elapsed_seconds",
    )}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python restart_probe.py FULL_PATH_JSON OUTPUT")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
