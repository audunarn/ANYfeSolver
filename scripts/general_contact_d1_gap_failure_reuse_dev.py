"""Check model reuse after a genuine gap-limited D1 press failure."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_interior_press_dev import build_interior_press


def case(force_n: float):
    return build_interior_press(
        force_n=force_n, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=4, source_rows=4,
        target_columns=1, target_rows=1, load_shape="interior_uniform",
        slide_body="target",
    )


def solve(model, definition, settings, program):
    return solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    model, definition, settings, high_program = case(0.12)
    failed = solve(model, definition, settings, high_program)
    if (failed.status == "completed" or len(failed.steps) != 1
            or failed.info.get("contact_failure") is None):
        raise AssertionError((failed.status, len(failed.steps),
                              failed.failure_reason))
    low_program = replace(high_program, nodal_forces=tuple(
        replace(item, values_n=tuple(value/3.0 for value in item.values_n))
        for item in high_program.nodal_forces
    ))
    low_settings = replace(settings, complementarity_force_scale_n=0.04)
    reused = solve(model, definition, low_settings, low_program)
    fresh_model, fresh_definition, fresh_settings, fresh_program = case(0.04)
    fresh = solve(fresh_model, fresh_definition, fresh_settings, fresh_program)
    if reused.status != "completed" or fresh.status != "completed":
        raise AssertionError((reused.status, reused.failure_reason,
                              fresh.status, fresh.failure_reason))
    displacement_difference = float(np.max(np.abs(
        reused.displacements-fresh.displacements,
    )))
    reuse_history = reused.info["force_displacement_history"]
    fresh_history = fresh.info["force_displacement_history"]
    wrench_difference = max(float(np.max(np.abs(
        np.asarray(a["contact"][key])-b["contact"][key],
    ))) for a, b in zip(reuse_history, fresh_history)
        for key in ("integration_force", "target_force",
                    "integration_moment", "target_moment"))
    reaction_difference = max(float(np.max(np.abs(
        np.asarray(a["reaction_generalized"])-b["reaction_generalized"],
    ))) for a, b in zip(reuse_history, fresh_history))
    checkpoints_equal = (
        reused.to_restart_checkpoint() == fresh.to_restart_checkpoint()
    )
    if (displacement_difference > 1.0e-10
            or wrench_difference > 1.0e-10
            or reaction_difference > 1.0e-10
            or not checkpoints_equal):
        raise AssertionError((displacement_difference, wrench_difference,
                              reaction_difference, checkpoints_equal))
    payload = {
        "status": "completed", "failed_path_status": failed.status,
        "failed_path_reason": failed.failure_reason,
        "failed_path_accepted_steps": len(failed.steps),
        "reused_path_status": reused.status,
        "fresh_path_status": fresh.status,
        "maximum_displacement_difference_m": displacement_difference,
        "maximum_contact_wrench_difference": wrench_difference,
        "maximum_generalized_reaction_difference": reaction_difference,
        "serialized_restart_checkpoints_equal": checkpoints_equal,
        "elapsed_seconds": time.perf_counter()-started,
    }
    output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
