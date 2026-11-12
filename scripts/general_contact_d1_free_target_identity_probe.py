"""Compare a fresh uninterrupted free-target cycle with a retained split run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_free_target_press_probe import build_free_target_case


def main(split_path: Path, full_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    full_bytes = full_path.read_bytes()
    full = json.loads(full_bytes)
    if (split.get("status") != "completed"
            or split.get("resumed_steps") != 12
            or not split.get("all_gap_and_pressure_gates_met")
            or len(split.get("differences", ())) != 12):
        raise ValueError("split reference is not a completed gated cycle")
    if (full.get("status") != "completed"
            or full.get("accepted_steps") != 12
            or len(full.get("history", ())) != 12):
        raise ValueError("full reference is not a completed cycle")
    started = time.perf_counter()
    report = {
        "candidate": "free_target_8q4_uninterrupted_vs_split_state_dev",
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "full_sha256": hashlib.sha256(full_bytes).hexdigest(),
        "equilibrium_tolerance": 1.0e-8,
    }
    try:
        model, definition, settings, program = build_free_target_case(
            8, four_phase=True)
        result = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            tolerance=1.0e-8, emit_restart_checkpoint=True,
        )
        report["solver_status"] = result.status
        report["failure_reason"] = result.failure_reason
        history = result.info.get("force_displacement_history", ())
        report["accepted_steps"] = len(result.steps)
        if result.status != "completed" or len(history) != 12:
            raise RuntimeError("uninterrupted cycle did not complete twelve steps")
        rows = []
        for expected, original, actual in zip(
                split["differences"], full["history"], history, strict=True):
            contact = actual["contact"]
            ids = tuple(contact["source_pressure_node_ids"])
            reference_ids = tuple(expected["source_pressure_node_ids"])
            pressure = np.asarray(contact["normal_multiplier_estimate_pa"],
                                  dtype=float)
            reference_pressure = np.asarray(
                expected["normal_multiplier_estimate_pa"], dtype=float)
            if pressure.shape != reference_pressure.shape:
                raise ValueError("pressure-space shape changed")
            force = np.asarray(contact["integration_force"], dtype=float)
            target = np.asarray(contact["target_force"], dtype=float)
            rows.append({
                "step": int(actual["step_index"]),
                "step_matches": (int(actual["step_index"]) == expected["step"]
                                 == original["step"]),
                "pressure_ids_match": ids == reference_ids,
                "pressure_vector_max_absolute_difference_pa": float(
                    np.max(np.abs(pressure-reference_pressure))),
                "normal_force_absolute_difference_n": abs(
                    float(force[2])-float(original["normal_force_n"])),
                "gap_absolute_difference_m": abs(
                    float(contact["minimum_gap"])
                    -float(original["minimum_gap_m"])),
                "pressure_change_absolute_difference_pa": abs(
                    float(actual["normal_multiplier_change"])
                    -float(original["pressure_change_pa"])),
                "active_count_matches": int(contact["active_pressure_dofs"])
                == int(original["active_pressure_dofs"]),
                "wrench_force_imbalance_n": float(np.linalg.norm(force+target)),
                "gap_gate_met": bool(contact["gap_tolerance_met"]),
                "pressure_gate_met": float(actual["normal_multiplier_change"])
                <= settings.pressure_tolerance,
                "active_pressure_dofs": int(contact["active_pressure_dofs"]),
            })
        report["rows"] = rows
        report["all_steps_and_pressure_ids_match"] = all(
            row["step_matches"] and row["pressure_ids_match"]
            and row["active_count_matches"] for row in rows)
        report["maximum_pressure_vector_difference_pa"] = max(
            row["pressure_vector_max_absolute_difference_pa"] for row in rows)
        for field in ("normal_force_absolute_difference_n",
                      "gap_absolute_difference_m",
                      "pressure_change_absolute_difference_pa"):
            report[f"maximum_{field}"] = max(row[field] for row in rows)
        report["all_contact_gates_met"] = all(
            row["gap_gate_met"] and row["pressure_gate_met"] for row in rows)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.perf_counter()-started
    output_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "failure_reason", "accepted_steps",
        "all_steps_and_pressure_ids_match",
        "maximum_pressure_vector_difference_pa", "all_contact_gates_met",
        "elapsed_seconds") if key in report}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.split, args.full, args.output)
