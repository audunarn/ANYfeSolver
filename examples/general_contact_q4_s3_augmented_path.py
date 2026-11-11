"""Development Q4 Q1 projected contact: press, slide, open and recontact."""

from __future__ import annotations

import argparse
import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    projected_modes = parser.add_mutually_exclusive_group()
    projected_modes.add_argument(
        "--projected", action="store_true",
        help="use the opt-in Q4/S3 V1 matching-projection gap certificate",
    )
    projected_modes.add_argument(
        "--projected-tight", action="store_true",
        help="use the verified V1 50-micrometre gap-budget lifecycle settings",
    )
    args = parser.parse_args()
    projected = args.projected or args.projected_tight
    closure = -0.01504 if args.projected_tight else -0.01505
    model = FEModel("general_q4_s3_augmented_path")
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
            "patch", "source_bottom", "target_top",
            10000.0 if args.projected_tight else 1000.0,
            0.00005 if args.projected_tight else
            0.0002 if args.projected else 0.005,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
        pressure_tolerance=0.02, maximum_augmentations=16,
        q4_source_gap_certificate=("projected_matching" if projected
                                   else "whole_face"),
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    path = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, closure),
                   (0.05, closure), (0.05, 0.0),
                   (0.05, closure)),
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
    if [row["contact"]["active_pressure_dofs"] > 0 for row in phases] != [
        True, True, False, True,
    ]:
        raise RuntimeError("unexpected contact activation history")
    if phases[2]["contact"]["normal_multiplier_estimate_pa"] != [0.0]*4:
        raise RuntimeError("normal multipliers did not release on opening")
    payload = {
        "status": result.status,
        "profile": result.info["general_contact"]["profile"],
        **({"gap_tolerance_m": definition.interactions[0].gap_tolerance,
            "q4_source_gap_certificate": settings.q4_source_gap_certificate}
           if projected else {}),
        **({"normal_stiffness_pa_per_m": definition.interactions[0].normal_stiffness,
            "corner_closure_m": -closure}
           if args.projected_tight else {}),
        "pressure_node_ids": result.info["general_contact"]["source_pressure_node_ids"],
        "phases": [
            {"target_m": row["target"],
             "active_pressure_dofs": row["contact"]["active_pressure_dofs"],
             "lower_gap_m": row["contact"]["minimum_gap"],
             "area_error_m2": row["contact"]["area_integration_error_m2"],
             "weighted_gap_error_m3": row["contact"]["weighted_gap_integration_error_m3"],
             "normal_force_N": row["contact"]["integration_force"][2],
             "augmentation_iterations": row["augmentation_iterations"],
             "normal_multipliers_pa": row["contact"]["normal_multiplier_estimate_pa"]}
            for row in phases
        ],
        "restart_schema_version": result.restart_checkpoint["version"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
