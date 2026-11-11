"""Development split-restart check for the exact D1 four-phase endpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver import solve_static_nonlinear
from scripts.general_contact_d1_coupled_refinement import build_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    full = json.loads(args.reference.read_text(encoding="utf-8"))
    if (full.get("status") != "completed" or full.get("accepted_steps") != 4
            or full.get("shared_edge_tie_step_m") != 3.0e-8
            or full.get("pressure_tolerance_pa") != 1.0e-5):
        raise ValueError("reference is not the completed exact D1 four-phase profile")
    model, definition, settings, program = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
        four_phase=True, slide_distance=0.05,
        pressure_tolerance=1.0e-5,
        schur_pressure_cap=100.0, schur_step_cap=50.0,
    )
    settings = replace(
        settings, q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
    )
    common = dict(contact=definition, contact_settings=settings,
                  contact_path_program=program, max_iterations=12,
                  emit_restart_checkpoint=True)
    start = time.perf_counter()
    prefix = solve_static_nonlinear(model, max_load_factor=0.75, **common)
    if (prefix.status != "completed" or len(prefix.steps) != 3
            or prefix.restart_checkpoint is None):
        raise AssertionError("exact D1 prefix did not accept opening and checkpoint")
    resumed = solve_static_nonlinear(
        model, restart_checkpoint=prefix.to_restart_checkpoint(), **common,
    )
    if (resumed.status != "completed" or len(resumed.steps) != 4
            or resumed.restart_checkpoint is None):
        raise AssertionError("exact D1 checkpoint did not resume through recontact")
    reference_displacement = full["final_nodal_displacements"]
    errors = []
    for node_id, expected in reference_displacement.items():
        actual = resumed.displacements[model.mesh.get_node(int(node_id)).dofs]
        errors.append(float(np.max(np.abs(actual-np.asarray(expected)))))
    final = resumed.info["force_displacement_history"][-1]
    reference_final = full["force_displacement_history"][-1]
    pressure_error = float(np.max(np.abs(
        np.asarray(final["contact"]["normal_multiplier_estimate_pa"])
        - np.asarray(reference_final["contact"]["normal_multiplier_estimate_pa"])
    )))
    reaction_error = float(np.max(np.abs(
        np.asarray(final["contact"]["integration_force"])
        - np.asarray(reference_final["contact"]["integration_force"])
    )))
    result = {
        "reference": str(args.reference),
        "source": "development split restart",
        "prefix_steps": len(prefix.steps),
        "resumed_steps": len(resumed.steps),
        "prefix_checkpoint_version": prefix.restart_checkpoint["version"],
        "resumed_checkpoint_version": resumed.restart_checkpoint["version"],
        "maximum_nodal_displacement_error_m": max(errors),
        "maximum_multiplier_error_pa": pressure_error,
        "maximum_contact_force_error_n": reaction_error,
        "resumed_gap_tolerance_met": bool(final["contact"]["gap_tolerance_met"]),
        "resumed_pressure_change_pa": final["normal_multiplier_change"],
        "elapsed_seconds": time.perf_counter()-start,
    }
    result["development_check_passed"] = bool(
        result["maximum_nodal_displacement_error_m"] <= 1.0e-9
        and pressure_error <= 1.0e-7
        and reaction_error <= 1.0e-6
        and result["resumed_gap_tolerance_met"]
        and result["resumed_pressure_change_pa"] <= settings.pressure_tolerance
    )
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["development_check_passed"]:
        raise AssertionError("exact D1 split restart differed from full path")


if __name__ == "__main__":
    main()
