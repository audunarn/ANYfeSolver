"""Development Q4 source against two S3 targets: press, slide and recontact."""

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
    model = FEModel("general_q4_s3_patch_path")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    source = ((0.1, 0.1, 0.055), (1.1, 0.1, 0.055),
              (1.1, 1.1, 0.055), (0.1, 1.1, 0.055))
    target = ((0., 0., 0.), (1., 0., 0.),
              (1., 1., 0.), (0., 1., 0.))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.05,
        reference_normal=(0, 0, 1),
    ))
    for element_id, nodes in ((2, (5, 6, 8)), (3, (6, 7, 8))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
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
        surfaces=(ContactSurface("source_bottom", (1,), "bottom"),
                  ContactSurface("target_top", (2, 3), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "patch", "source_bottom", "target_top", 1000.0, 0.02,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    path = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.02),
                   (0.05, -0.02), (0.05, 0.0), (0.05, -0.02)),
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
    if not all(row["contact"]["gap_tolerance_met"] for row in phases):
        raise RuntimeError("independent physical gap exceeded its declared tolerance")
    print(json.dumps({
        "status": result.status,
        "profile": result.info["general_contact"]["profile"],
        "pressure_node_ids": result.info["general_contact"]["source_pressure_node_ids"],
        "phases": [
            {"target_m": row["target"],
             "active_pressure_dofs": row["contact"]["active_pressure_dofs"],
             "lower_gap_m": row["contact"]["minimum_gap"],
             "area_error_m2": row["contact"]["area_integration_error_m2"],
             "weighted_gap_error_m3": row["contact"]["weighted_gap_integration_error_m3"],
             "normal_force_N": row["contact"]["integration_force"][2]}
            for row in phases
        ],
        "restart_schema_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
