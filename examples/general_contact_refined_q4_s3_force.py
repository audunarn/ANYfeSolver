"""Development example: feasible force-driven contact of two Q4/two S3 patches."""

import json

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase,
    NonlinearConvergenceSettings, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


GAP_BUDGET_M = 5.0e-5


def make_case():
    model = FEModel("refined_q4_s3_feasible_force")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    source = (
        (0.1, 0.1, 0.04), (0.6, 0.1, 0.04), (1.1, 0.1, 0.04),
        (0.1, 1.1, 0.04), (0.6, 1.1, 0.04), (1.1, 1.1, 0.04),
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
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_left_clamp", (1, 4), {"uz": 0.0},
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("q4_source", (1, 2), "bottom"),
                  ContactSurface("s3_target", (3, 4), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", 1000.0,
            GAP_BUDGET_M, normal_law="augmented",
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
    load = LoadCase("press")
    for node in (3, 6):
        load.add_nodal_load(node, forces=(0.0, 0.0, -0.05))
    return model, definition, settings, load


def solve_case():
    model, definition, settings, load = make_case()
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=4,
        max_iterations=12, min_step_fraction=0.1,
        convergence_settings=NonlinearConvergenceSettings(
            growth_factor=1.0, max_step_factor=1.0,
        ), emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(f"contact stopped: {result.status}: {result.failure_reason}")
    return result


def main():
    result = solve_case()
    contact = result.info["general_contact"]
    if not contact["gap_tolerance_met"] or contact["active_pressure_dofs"] == 0:
        raise RuntimeError("force-driven contact did not meet the physical gap gate")
    for first, second in (("integration_force", "target_force"),
                          ("integration_moment", "target_moment")):
        if not np.allclose(np.asarray(contact[first])+contact[second],
                           0.0, rtol=0.0, atol=1.0e-6):
            raise RuntimeError("force-driven contact wrench is unbalanced")
    print(json.dumps({
        "profile": contact["profile"],
        "gap_budget_m": GAP_BUDGET_M,
        "conservative_lower_gap_m": contact["minimum_gap"],
        "active_pressure_nodes": contact["active_pressure_dofs"],
        "accepted_load_factors": [row["load_factor"] for row in
                                  result.info["force_displacement_history"]],
        "source_force_N": contact["integration_force"],
        "target_force_N": contact["target_force"],
        "source_moment_Nm": contact["integration_moment"],
        "target_moment_Nm": contact["target_moment"],
        "checkpoint_version": result.restart_checkpoint["version"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
