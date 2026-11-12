"""Development S3/S3 physical-skin contact through the nonlinear static solver."""

from __future__ import annotations

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main():
    model = FEModel("general_s3_nodal_pair")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    source = ((0.4, 0.2, 0.065), (0.9, 0.2, 0.065),
              (0.4, 0.7, 0.065))
    target = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (0.0, 1.0, 0.0))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element_id, nodes in ((1, (1, 2, 3)), (2, (4, 5, 6))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (4, 5, 6),
        {dof: 0.0 for dof in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", (1, 2, 3),
        {dof: 0.0 for dof in ("ux", "uy", "rx", "ry", "rz")},
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", (1,), "bottom"),
                  ContactSurface("target", (2,), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 1.0e3, 2.0e-2,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2.0e-5,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    drive = LoadCase("drive")
    drive.add_nodal_load(1, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, drive, contact=definition, contact_settings=settings,
        kinematics="von_karman", control="displacement",
        displacement_control=DisplacementControl(
            node_id=1, dof="uz", target_displacement=-0.020,
        ),
        num_steps=2, max_iterations=15, emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(f"contact stopped: {result.status}: {result.failure_reason}")
    summary = result.info["general_contact"]
    print(json.dumps({
        "status": result.status,
        "profile": summary["profile"],
        "source_force_N": summary["integration_force"],
        "target_force_N": summary["target_force"],
        "conservative_gap_m": summary["minimum_gap"],
        "gap_tolerance_met": summary["gap_tolerance_met"],
        "normalized_weighted_gap_m": summary["normalized_weighted_gap_m"],
        "numerical_pressure_pa": summary["numerical_pressure_pa"],
        "restart_schema_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
