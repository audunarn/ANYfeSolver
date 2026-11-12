"""Exercise opening-to-recontact restart of the D1 displacement path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver import solve_static_nonlinear

from scripts.general_contact_d1_open_ring_displacement_dev import build_case


def main(reference: Path, full_result: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(full_result.read_text(encoding="utf-8"))
    if saved.get("status") != "completed" or saved.get("accepted_steps") != 8:
        raise ValueError("restart comparison requires the complete eight-step path")
    model, definition, settings, program, *_ = build_case(reference, full_cycle=True)
    started = time.perf_counter()
    first = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        max_load_factor=0.75, emit_restart_checkpoint=True,
    )
    if first.status != "completed" or len(first.steps) != 6:
        raise AssertionError((first.status, first.failure_reason, len(first.steps)))
    if first.info["force_displacement_history"][-1]["contact"][
            "active_pressure_dofs"] != 0:
        raise AssertionError("checkpoint must be at the opening step")
    checkpoint = first.to_restart_checkpoint()
    resumed = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        restart_checkpoint=checkpoint, emit_restart_checkpoint=True,
    )
    if resumed.status != "completed" or len(resumed.steps) != 8:
        raise AssertionError((resumed.status, resumed.failure_reason,
                              len(resumed.steps)))
    reference_history = saved["history"]
    resumed_history = resumed.info["force_displacement_history"][6:]
    if [row["step_index"] for row in resumed_history] != [7, 8]:
        raise AssertionError("resumed steps are not the recontact segment")
    keys = ("integration_force", "target_force", "integration_moment",
            "target_moment")
    force_difference = max(
        float(np.max(np.abs(np.asarray(actual["contact"][key])
                            - expected["contact"][key])))
        for actual, expected in zip(resumed_history, reference_history[6:])
        for key in keys
    )
    reaction_difference = max(
        float(np.max(np.abs(np.asarray(actual["reaction_generalized"])
                            - expected["reaction_generalized"])))
        for actual, expected in zip(resumed_history, reference_history[6:])
    )
    gap_difference = max(
        abs(actual["contact"]["minimum_gap"]
            - expected["contact"]["minimum_gap"])
        for actual, expected in zip(resumed_history, reference_history[6:])
    )
    expected_modes = [row["contact"]["active_pressure_dofs"]
                      for row in reference_history[6:]]
    actual_modes = [row["contact"]["active_pressure_dofs"]
                    for row in resumed_history]
    if actual_modes != expected_modes or force_difference > 1e-10 or (
            reaction_difference > 1e-10 or gap_difference > 1e-10):
        raise AssertionError((actual_modes, expected_modes, force_difference,
                              reaction_difference, gap_difference))
    payload = {
        "status": "completed", "checkpoint_step": 6,
        "checkpoint_active_pressure_dofs": 0,
        "resumed_steps": [row["step_index"] for row in resumed_history],
        "resumed_active_pressure_dofs": actual_modes,
        "maximum_wrench_difference": force_difference,
        "maximum_reaction_difference": reaction_difference,
        "maximum_gap_difference_m": gap_difference,
        "elapsed_seconds": time.perf_counter() - started,
        "reference_result": str(full_result),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("full_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.reference, args.full_result, args.output)
