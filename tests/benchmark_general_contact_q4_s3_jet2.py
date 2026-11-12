"""Exploratory whole-route comparison for the opt-in Q4/S3 derivative."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    solve_static_nonlinear,
)
from test_general_contact_q4_s3_refined_patch import _two_q4_two_s3


def run(mode):
    model, definition, settings = _two_q4_two_s3()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], gap_tolerance=0.005,
    ),))
    settings = replace(
        settings,
        maximum_augmentations=8,
        maximum_q4_source_integration_triangles=128,
        general_s3_differentiation=mode,
        **({
            "q4_source_support_gradient_tolerance": 1.0e-5,
            "q4_source_support_hessian_tolerance": 1.0e-5,
            "q4_source_weighted_gap_gradient_tolerance": 1.0e-5,
            "q4_source_weighted_gap_hessian_tolerance": 1.0e-5,
        } if mode != "existing" else {}),
    )
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (7, 8, 9, 10),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4, 5, 6),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.01505)),
        steps_per_segment=1,
    )
    before = perf_counter()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
    )
    elapsed = perf_counter()-before
    contact = result.info.get("general_contact", {})
    return result, {
        "mode": mode,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "elapsed_s": elapsed,
        "accepted_steps": len(result.steps),
        "total_newton_iterations": result.info.get("total_newton_iterations"),
        "active_pressure_dofs": contact.get("active_pressure_dofs"),
        "gap_tolerance_met": contact.get("gap_tolerance_met"),
        "energy_evaluations_last": contact.get("energy_evaluations"),
        "integration_triangles_last": contact.get("integration_triangles"),
        "integration_force": contact.get("integration_force"),
        "target_force": contact.get("target_force"),
    }


if __name__ == "__main__":
    baseline, old = run("existing")
    print(json.dumps(old, default=lambda value: value.tolist()), flush=True)
    candidate, new = run("jet2_q4_s3_dev")
    print(json.dumps(new, default=lambda value: value.tolist()), flush=True)
    result = {
        "baseline": old,
        "candidate": new,
        "maximum_displacement_difference": float(np.max(np.abs(
            baseline.displacements-candidate.displacements))),
        "note": "One exploratory pair; no statistical or acceptance claim.",
    }
    print(json.dumps(result, indent=2, sort_keys=True,
                     default=lambda value: value.tolist()), flush=True)
    assert baseline.status == candidate.status == "completed"
    assert old["gap_tolerance_met"] and new["gap_tolerance_met"]
