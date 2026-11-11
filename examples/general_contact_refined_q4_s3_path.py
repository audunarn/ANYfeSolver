"""Development example: two Q4 pressure faces contacting two S3 shell faces."""

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
)
from anyfesolver.fe_core import FEModel


model = FEModel("refined_q4_s3_contact_example")
model.add_material("elastic", 1.0e5, 0.3, density=1.0)
source = (
    (0.1, 0.1, 0.055), (0.6, 0.1, 0.055), (1.1, 0.1, 0.055),
    (0.1, 1.1, 0.055), (0.6, 1.1, 0.055), (1.1, 1.1, 0.055),
)
target = (
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
)
for node, position in enumerate((*source, *target), 1):
    model.add_node(node, *position)
for element_id, nodes in ((1, (1, 2, 5, 4)), (2, (2, 3, 6, 5))):
    model.add_element(element_id, create_shell_element(
        element_id, nodes, "elastic", thickness=0.05,
        reference_normal=(0.0, 0.0, 1.0),
    ))
for element_id, nodes in ((3, (7, 8, 10)), (4, (8, 9, 10))):
    model.add_element(element_id, create_shell_element(
        element_id, nodes, "elastic", thickness=0.05,
        formulation="e4-pl-s3-v2d", reference_normal=(0.0, 0.0, 1.0),
        reference_surface_offset=0.01,
    ))
model.add_boundary_condition(BoundaryCondition(
    "target_fixed", (7, 8, 9, 10),
    {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
))
model.add_boundary_condition(BoundaryCondition(
    "source_guide", (1, 2, 3, 4, 5, 6),
    {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
))
contact = ContactDefinition(
    surfaces=(ContactSurface("q4_source", (1, 2), "bottom"),
              ContactSurface("s3_target", (3, 4), "top")),
    rigid_bodies=(),
    interactions=(ContactInteraction(
        "pair", "q4_source", "s3_target", 1000.0, 0.005,
        normal_law="augmented",
    ),),
)
settings = ContactSolverSettings(
    pair_motion="general", normal_discretization="primal_mortar",
    general_s3_difference_step=2.0e-5,
    maximum_general_s3_energy_evaluations=5000,
    q4_source_area_tolerance=1.0e-9,
    q4_source_weighted_gap_tolerance=1.0e-10,
    maximum_q4_source_integration_triangles=300,
    pressure_tolerance=0.02, maximum_augmentations=8,
)
program = ContactPathProgram(
    controls=(ContactPathControl.node(1, "ux"),
              ContactPathControl.node(1, "uz")),
    waypoints=((0.0, 0.0), (0.0, -0.01505)),
    steps_per_segment=1,
)
result = solve_static_nonlinear(
    model, contact=contact, contact_settings=settings,
    contact_path_program=program, max_iterations=12,
    emit_restart_checkpoint=True,
)
assert result.status == "completed", result.failure_reason
summary = result.info["general_contact"]
print(json.dumps({
    "profile": summary["profile"],
    "conservative_lower_gap_m": float(summary["minimum_gap"]),
    "gap_budget_m": 0.005,
    "active_pressure_nodes": int(summary["active_pressure_dofs"]),
    "source_force_N": summary["integration_force"],
    "target_force_N": summary["target_force"],
    "source_moment_Nm": summary["integration_moment"],
    "target_moment_Nm": summary["target_moment"],
    "pair_geometry_evaluations": int(summary["pair_geometry_evaluations"]),
    "checkpoint_version": int(result.restart_checkpoint["version"]),
}, sort_keys=True))
