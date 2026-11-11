"""Development two-S3/one-Q4 projected normal path and restart."""

from __future__ import annotations

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("general_s3_q4_patch_augmented_path")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    source = ((0.1, 0.1, 0.065), (1.1, 0.1, 0.065),
              (1.1, 1.1, 0.065), (0.1, 1.1, 0.065))
    target = ((0., 0., -0.01), (1., 0., -0.01),
              (1., 1., -0.01), (0., 1., -0.01))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element_id, nodes in ((1, (1, 2, 4)), (2, (2, 3, 4))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    model.add_element(3, create_shell_element(
        3, (5, 6, 7, 8), "elastic", thickness=0.05,
        reference_normal=(0, 0, 1),
    ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (5, 6, 7, 8),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source_bottom", (1, 2), "bottom"),
                  ContactSurface("target_top", (3,), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "patch", "source_bottom", "target_top", 1000.0, 0.02,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
        q4_target_weighted_gap_tolerance=1e-11,
        maximum_q4_target_integration_triangles=500,
        pressure_tolerance=0.001,
        maximum_augmentations=8,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    path = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.0155),
                   (0.025, -0.0155), (0.025, 0.0), (0.025, -0.0155)),
        steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=path, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(f"contact stopped: {result.status}: {result.failure_reason}")
    phases = result.info["force_displacement_history"]
    if phases[0]["augmentation_iterations"] <= 1:
        raise RuntimeError("normal multiplier update was not exercised")
    if phases[2]["contact"]["normal_multiplier_estimate_pa"] != [0.0]*4:
        raise RuntimeError("opening did not release normal multipliers")
    if not all(row["contact"]["gap_tolerance_met"] for row in phases):
        raise RuntimeError("independent physical gap exceeded its declared tolerance")
    print(json.dumps({
        "status": result.status,
        "profile": result.info["general_contact"]["profile"],
        "pressure_node_ids": result.info["general_contact"]["source_pressure_node_ids"],
        "phases": [
            {"target_m": row["target"],
             "active_pressure_dofs": row["contact"]["active_pressure_dofs"],
             "augmentation_iterations": row["augmentation_iterations"],
             "normal_multiplier_pa": row["contact"]["normal_multiplier_estimate_pa"],
             "lower_gap_m": row["contact"]["minimum_gap"],
             "weighted_gap_error_m3": row["contact"]["weighted_gap_integration_error_m3"],
             "normal_force_N": row["contact"]["integration_force"][2]}
            for row in phases
        ],
        "normal_multiplier_pa": result.info["general_contact"]["normal_multiplier_estimate_pa"],
        "restart_schema_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
