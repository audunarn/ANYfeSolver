"""Development Q4/S3 patch path at a 50 µm independent gap budget.

One Q4 source and two nonmatching S3 target faces press, slide, open and
recontact through the ordinary nonlinear static route. This is a focused
development case, not C1 engineering acceptance.
"""

from __future__ import annotations

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel


THICKNESS_M = 0.05
GAP_BUDGET_M = 5e-5
PRESS_M = -0.01502


def make_case():
    model = FEModel("q4_s3_tight_patch_path")
    model.add_material("elastic", 1e5, 0.3, density=1.0)
    source = ((0.1, 0.1, 0.055), (1.1, 0.1, 0.055),
              (1.1, 1.1, 0.055), (0.1, 1.1, 0.055))
    target = ((0., 0., 0.), (1., 0., 0.),
              (1., 1., 0.), (0., 1., 0.))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=THICKNESS_M,
        reference_normal=(0, 0, 1),
    ))
    for element_id, nodes in ((2, (5, 6, 8)), (3, (6, 7, 8))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=THICKNESS_M,
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
            "patch", "source_bottom", "target_top", 1000.0,
            GAP_BUDGET_M,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-6,
        q4_source_area_tolerance=1e-9,
        q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=300,
    )
    path = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, PRESS_M),
                   (0.05, PRESS_M), (0.05, 0.0), (0.05, PRESS_M)),
        steps_per_segment=1,
    )
    return model, definition, settings, path


def main() -> None:
    model, definition, settings, path = make_case()
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=path, max_iterations=15,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(f"contact stopped: {result.status}: {result.failure_reason}")
    phases = result.info["force_displacement_history"]
    active = [row["contact"]["active_pressure_dofs"] for row in phases]
    if (len(phases) != 4 or not all(row["contact"]["gap_tolerance_met"]
                                   for row in phases)
            or not (active[0] > 0 and active[1] > 0
                    and active[2] == 0 and active[3] > 0)):
        raise RuntimeError("tight Q4/S3 phase or physical-gap gate failed")
    print(json.dumps({
        "status": result.status,
        "profile": result.info["general_contact"]["profile"],
        "thickness_m": THICKNESS_M,
        "gap_budget_m": GAP_BUDGET_M,
        "pressure_node_ids": result.info["general_contact"]["source_pressure_node_ids"],
        "phases": [
            {"target_m": row["target"],
             "active_pressure_dofs": row["contact"]["active_pressure_dofs"],
             "lower_gap_m": row["contact"]["minimum_gap"],
             "normal_force_n": row["contact"]["integration_force"][2]}
            for row in phases
        ],
        "restart_schema_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
