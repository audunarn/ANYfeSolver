"""Development refined S3 mortar patch with explicit contact budgets."""

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
    model = FEModel("refined_s3_patch_development")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    small = 0.1
    source = (
        (0.0, 0.0), (small, 0.0), (small, small),
        (0.0, small), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
    )
    target = ((-0.2, -0.3), (1.3, -0.2),
              (1.2, 1.3), (-0.3, 1.2))
    # The formulation's 50 mm thickness and +10 mm reference offset put the
    # declared source bottom at z=20 mm and target top at z=0.
    for node, (x, y) in enumerate(source, 1):
        model.add_node(node, x, y, 0.055)
    for node, (x, y) in enumerate(target, 101):
        model.add_node(node, x, y, -0.015)
    source_faces = ((1, 2, 3), (1, 3, 4), (2, 5, 6),
                    (2, 6, 3), (4, 3, 6), (4, 6, 7))
    target_faces = ((101, 102, 103), (101, 103, 104))
    for element_id, nodes in enumerate((*source_faces, *target_faces), 1):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", (101, 102, 103, 104),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", tuple(range(1, 8)),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source_bottom", tuple(range(1, 7)), "bottom"),
                  ContactSurface("target_top", (7, 8), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "refined", "source_bottom", "target_top", 1000.0, 0.005,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
        maximum_general_s3_energy_evaluations=7800,
        maximum_candidate_pairs=12,
        pressure_tolerance=1.0,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported refined contact: " + "; ".join(report.issues))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, -0.0205)), steps_per_segment=1,
    )
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(f"refined contact stopped: {result.failure_reason}")
    contact = result.info["general_contact"]
    if not contact["gap_tolerance_met"] or contact["active_pressure_dofs"] == 0:
        raise RuntimeError("refined development contact did not meet its declared checks")
    print(json.dumps({
        "status": result.status,
        "profile": contact["profile"],
        "path_profile": result.info["contact_path"]["profile"],
        "source_pressure_node_ids": contact["source_pressure_node_ids"],
        "active_pressure_dofs": contact["active_pressure_dofs"],
        "minimum_gap_m": contact["minimum_gap"],
        "gap_tolerance_m": 0.005,
        "integration_force_N": contact["integration_force"],
        "target_force_N": contact["target_force"],
        "recovery_pair_geometry_evaluations": contact["pair_geometry_evaluations"],
        "restart_schema_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
