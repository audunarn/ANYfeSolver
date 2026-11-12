"""Opt-in eight-Q4/two-S3 projected contact press with a guarded pivot.

Run from a source checkout with ``python examples/general_contact_refined_q4_s3_eight_q4_press.py``.
This development example is deliberately resource-intensive and reports the
actual solver status; it does not imply engineering qualification.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathNodalForce, ContactPathProgram,
    ContactSolverSettings, ContactSurface, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


count = 8
model = FEModel("mixed_eight_q4_x_strict_press")
model.add_material("elastic", 1.0e5, 0.3, density=1.0)
for row, y in enumerate((0.1, 1.1)):
    for index in range(count+1):
        model.add_node(1 + (count+1)*row + index,
                       0.1 + index/count, y, 0.04004)
for node_id, xyz in enumerate((
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
), 19):
    model.add_node(node_id, *xyz)
for index in range(count):
    element_id = index+1
    model.add_element(element_id, create_shell_element(
        element_id, (1+index, 2+index, count+3+index, count+2+index),
        "elastic", thickness=0.05,
        reference_normal=(0.0, 0.0, 1.0),
    ))
for element_id, nodes in ((9, (19, 20, 22)), (10, (20, 21, 22))):
    model.add_element(element_id, create_shell_element(
        element_id, nodes, "elastic", thickness=0.05,
        formulation="e4-pl-s3-v2d",
        reference_normal=(0.0, 0.0, 1.0),
        reference_surface_offset=0.01,
    ))
model.add_boundary_condition(BoundaryCondition(
    "target_fixed", (19, 20, 21, 22),
    {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
))
model.add_boundary_condition(BoundaryCondition(
    "source_guide", tuple(range(1, 19)),
    {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
))
model.add_boundary_condition(BoundaryCondition(
    "source_left_support", (1, 10), {"uz": 0.0},
))
definition = ContactDefinition(
    surfaces=(ContactSurface("q4_source", tuple(range(1, 9)), "bottom"),
              ContactSurface("s3_target", (9, 10), "top")),
    rigid_bodies=(),
    interactions=(ContactInteraction(
        "pair", "q4_source", "s3_target", 10000.0, 5.0e-5,
        normal_law="augmented",
    ),),
)
settings = ContactSolverSettings(
    pair_motion="general", normal_discretization="primal_mortar",
    general_s3_difference_step=2.0e-5,
    maximum_general_s3_energy_evaluations=14144,
    maximum_candidate_pairs=32,
    q4_source_area_tolerance=1.0e-9,
    q4_source_weighted_gap_tolerance=1.0e-10,
    maximum_q4_source_integration_triangles=300,
    pressure_tolerance=0.0002, maximum_augmentations=16,
    augmentation_acceleration="active_secant_pivot",
    augmentation_pivot_max_pressure_pa=20.0,
    augmentation_pivot_difference_step_m=1.0e-7,
)
program = ContactPathProgram(
    controls=(ContactPathControl.node(1, "ux"),
              ContactPathControl.node(10, "ux")),
    waypoints=((0.0, 0.0), (0.0, 0.0),
               (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)),
    steps_per_segment=2,
    nodal_forces=tuple(
        ContactPathNodalForce(node_id, "uz", (0.0, -0.05, -0.05, 0.0, -0.05))
        for node_id in (9, 18)
    ),
)
parser = argparse.ArgumentParser(description=__doc__)
path = parser.add_mutually_exclusive_group()
path.add_argument("--first-slide", action="store_true",
                  help="continue through the first sliding increment")
path.add_argument("--full-cycle", action="store_true",
                  help="press, slide 50 mm, open and recontact")
parser.add_argument("--roundoff-support", action="store_true",
                    help="opt into force-bounded multiplier transfer")
parser.add_argument("--equilibrated-pressure-map", action="store_true",
                    help="opt into a three-equilibrium pressure-map pivot")
parser.add_argument("--checkpoint-output", type=Path,
                    help="write a completed version-2 restart checkpoint")
args = parser.parse_args()
if args.equilibrated_pressure_map and not args.roundoff_support:
    parser.error("the demonstrated pressure-map profile requires --roundoff-support")
if args.roundoff_support:
    settings = replace(
        settings, augmentation_support_resolution="roundoff",
        augmentation_support_force_budget_n=1.0e-8,
    )
if args.equilibrated_pressure_map:
    settings = replace(
        settings, augmentation_equilibrated_pressure_step_pa=0.01,
    )
report = validate_general_contact(model, definition, settings)
assert report.valid, report.issues
start = time.perf_counter()
result = solve_static_nonlinear(
    model, contact=definition, contact_settings=settings,
    contact_path_program=program, max_iterations=18,
    max_load_factor=(1.0 if args.full_cycle else
                     0.375 if args.first_slide else 0.25),
    emit_restart_checkpoint=True,
)
if args.checkpoint_output is not None and result.status == "completed":
    args.checkpoint_output.write_text(
        json.dumps(result.to_restart_checkpoint(), sort_keys=True,
                   default=lambda value: (value.tolist() if hasattr(value, "tolist")
                                          else str(value))),
        encoding="utf-8",
    )
contact = result.info["general_contact"]
failure = result.info.get("contact_failure")
print(json.dumps({
    "elapsed_s": time.perf_counter() - start,
    "status": result.status,
    "failure_reason": result.failure_reason,
    "failure_details": (failure.get("augmentation_trace", [])[-1:]
                        if failure else None),
    "accepted_steps": len(result.steps),
    "active_pressure_dofs": contact["active_pressure_dofs"],
    "lower_gap_m": contact["minimum_gap"],
    "gap_tolerance_met": contact["gap_tolerance_met"],
    "source_force_n": contact["integration_force"],
    "history": [{
        "step": row["step_index"],
        "gap_m": row["contact"]["minimum_gap"],
        "active_pressure_dofs": row["contact"]["active_pressure_dofs"],
        "normal_source_force_n": row["contact"]["integration_force"][2],
        "normal_multiplier_change_pa": row["normal_multiplier_change"],
        "augmentations": row["augmentation_iterations"],
        "acceleration_applied": row.get("augmentation_acceleration_applied"),
        "pivot_attempts": row.get("pivot_attempts"),
        "pivot_applied": row.get("pivot_applied"),
        "pivot_rejected": row.get("pivot_rejected"),
        "pressure_map_attempts": row.get("pressure_map_attempts"),
        "support_resets": row.get("support_resets"),
} for row in result.info["force_displacement_history"]],
}, sort_keys=True, default=lambda value: (
    value.item() if hasattr(value, "item") else str(value)
)))
