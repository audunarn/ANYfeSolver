"""Two Q4 and two S3 shell faces: press, slide, open and recontact.

Development demonstration through ordinary nonlinear static analysis. The
50 µm independent gap budget is 10^-3 of the 50 mm shell thickness. This
case is not C1 engineering acceptance.
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
GAP_BUDGET_M = 5.0e-5
PRESS_M = -0.01501
SLIDE_M = 0.05


def make_case():
    model = FEModel("refined_q4_s3_tight_path")
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
            element_id, nodes, "elastic", thickness=THICKNESS_M,
            reference_normal=(0.0, 0.0, 1.0),
        ))
    for element_id, nodes in ((3, (7, 8, 10)), (4, (8, 9, 10))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=THICKNESS_M,
            formulation="e4-pl-s3-v2d",
            reference_normal=(0.0, 0.0, 1.0),
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
    definition = ContactDefinition(
        surfaces=(ContactSurface("q4_source", (1, 2), "bottom"),
                  ContactSurface("s3_target", (3, 4), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", 1000.0, GAP_BUDGET_M,
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
        pressure_tolerance=0.02, maximum_augmentations=16,
    )
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(1, "uz")),
        waypoints=((0.0, 0.0), (0.0, PRESS_M),
                   (SLIDE_M, PRESS_M), (SLIDE_M, 0.0),
                   (SLIDE_M, PRESS_M)),
        steps_per_segment=2,
    )
    return model, definition, settings, program


def solve_case():
    model, definition, settings, program = make_case()
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(f"contact stopped: {result.status}: {result.failure_reason}")
    history = result.info["force_displacement_history"]
    if len(history) != program.total_steps or any(
        not row["contact"]["gap_tolerance_met"] for row in history
    ):
        raise RuntimeError("contact path did not meet the physical gap budget")
    for row in history:
        contact = row["contact"]
        if any(abs(a+b) > 1.0e-6 for a, b in zip(
            contact["integration_force"], contact["target_force"],
        )) or any(abs(a+b) > 1.0e-6 for a, b in zip(
            contact["integration_moment"], contact["target_moment"],
        )):
            raise RuntimeError("paired contact wrench is unbalanced")
    if ([history[i]["contact"]["active_pressure_dofs"] > 0
         for i in (1, 3, 5, 7)] != [True, True, False, True]):
        raise RuntimeError("press/slide/open/recontact activity is inconsistent")
    return result


def main():
    result = solve_case()
    history = result.info["force_displacement_history"]
    print(json.dumps({
        "status": result.status,
        "profile": result.info["general_contact"]["profile"],
        "thickness_m": THICKNESS_M,
        "gap_budget_m": GAP_BUDGET_M,
        "source_pressure_node_ids": result.info["general_contact"]["source_pressure_node_ids"],
        "phases": [{
            "target_m": history[i]["target"],
            "active_pressure_dofs": history[i]["contact"]["active_pressure_dofs"],
            "lower_gap_m": history[i]["contact"]["minimum_gap"],
            "source_force_N": history[i]["contact"]["integration_force"],
        } for i in (1, 3, 5, 7)],
        "restart_schema_version": result.restart_checkpoint["version"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
