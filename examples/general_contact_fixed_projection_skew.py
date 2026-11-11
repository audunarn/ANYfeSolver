"""Development example: a restricted skew Q4/S3 partial-contact approach.

The affine MPC field leaves nine local normal DOFs free. It is a controlled
contact discriminator, not a general finite-sliding shell example.
"""

import json

import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathProgram, ContactSolverSettings,
    ContactSurface, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel


GAP_BUDGET_M = 5.e-5
X_GRID_M = (.1, .7, .9, .91, .915, .92, 1.1)
Y_GRID_M = (.1, .125, .15, .2, 1.1)
TARGET_XY_M = ((.2, -.1), (.92, 0.), (.86, 1.15), (.15, .95))


def make_case():
    model = FEModel("fixed_projection_skew_q4_s3")
    model.add_material("elastic", 1.e5, .3, density=1.)
    width, rows = len(X_GRID_M), len(Y_GRID_M)
    for row, y in enumerate(Y_GRID_M):
        for col, x in enumerate(X_GRID_M):
            model.add_node(1+row*width+col, x, y, .055)
    source_faces = []
    for row in range(rows-1):
        for col in range(width-1):
            lower = 1+row*width+col
            element_id = len(source_faces)+1
            model.add_element(element_id, create_shell_element(
                element_id, (lower, lower+1, lower+width+1, lower+width),
                "elastic", thickness=.05, reference_normal=(0., 0., 1.),
            ))
            source_faces.append(element_id)
    first_target = width*rows+1
    for index, (x, y) in enumerate(TARGET_XY_M):
        model.add_node(first_target+index, x, y, 0.)
    for nodes in ((first_target, first_target+1, first_target+3),
                  (first_target+1, first_target+2, first_target+3)):
        element_id = len(model.mesh.elements)+1
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0., 0., 1.),
            reference_surface_offset=.01,
        ))

    source_nodes = tuple(range(1, width*rows+1))
    target_nodes = tuple(range(first_target, first_target+4))
    left_nodes = tuple(1+row*width for row in range(rows))
    right_nodes = (width, rows*width)
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", target_nodes,
        {name: 0. for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", source_nodes,
        {name: 0. for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_left", left_nodes, {"uz": 0.},
    ))
    first_master = model.mesh.dof_manager.get_node_dofs(right_nodes[0])[2]
    last_master = model.mesh.dof_manager.get_node_dofs(right_nodes[1])[2]
    for row, y in enumerate(Y_GRID_M):
        for col, x in enumerate(X_GRID_M):
            node_id = 1+row*width+col
            if node_id in left_nodes or node_id in right_nodes:
                continue
            if row in (0, 1, 2) and x in (.91, .915, .92):
                continue
            dependent = model.mesh.dof_manager.get_node_dofs(node_id)[2]
            wx = (x-X_GRID_M[0])/(X_GRID_M[-1]-X_GRID_M[0])
            wy = (y-Y_GRID_M[0])/(Y_GRID_M[-1]-Y_GRID_M[0])
            model.add_constraint_equation(
                terms=((dependent, 1.),
                       (first_master, -wx*(1.-wy)),
                       (last_master, -wx*wy)),
                dependent_dof=dependent, source_id=f"affine_normal_{node_id}",
            )

    definition = ContactDefinition(
        surfaces=(ContactSurface("source", tuple(source_faces), "bottom"),
                  ContactSurface("target", (len(source_faces)+1,
                                            len(source_faces)+2), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 1.e8, GAP_BUDGET_M,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        q4_source_gap_certificate="fixed_projection_graph",
        general_s3_difference_step=2.e-6,
        maximum_general_s3_energy_evaluations=45000,
        maximum_candidate_pairs=64,
        q4_source_area_tolerance=1.e-10,
        q4_source_weighted_gap_tolerance=1.e-11,
        maximum_q4_source_integration_triangles=500,
        pressure_tolerance=.2, maximum_augmentations=17,
    )
    program = ContactPathProgram(
        controls=tuple(ContactPathControl.node(node, "uz")
                       for node in right_nodes),
        waypoints=((0., 0.), (-.018453, -.018453)),
        steps_per_segment=2,
    )
    return model, definition, settings, program


def solve_case():
    model, definition, settings, program = make_case()
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    return solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=25, tolerance=1.e-8,
        emit_restart_checkpoint=True,
    )


def main():
    result = solve_case()
    if result.status != "completed":
        raise RuntimeError(f"contact stopped: {result.status}: {result.failure_reason}")
    contact = result.info["general_contact"]
    force = np.asarray(contact["integration_force"])
    target_force = np.asarray(contact["target_force"])
    moment = np.asarray(contact["integration_moment"])
    target_moment = np.asarray(contact["target_moment"])
    force_balance = float(np.linalg.norm(force+target_force)/np.linalg.norm(force))
    moment_balance = float(np.linalg.norm(moment+target_moment)/np.linalg.norm(moment))
    if (contact["active_pressure_dofs"] != 1 or not contact["gap_tolerance_met"]
            or force_balance > 1.e-6 or moment_balance > 1.e-6):
        raise RuntimeError("restricted skew contact did not meet its development gates")
    print(json.dumps({
        "profile": contact["profile"],
        "active_pressure_dofs": contact["active_pressure_dofs"],
        "gap_budget_m": GAP_BUDGET_M,
        "certified_lower_gap_m": contact["minimum_gap"],
        "sampled_overlap_gap_m": contact["sampled_overlap_gap"],
        "normal_force_n": float(force[2]),
        "force_balance_relative": force_balance,
        "moment_balance_relative": moment_balance,
        "augmentation_iterations": result.info["force_displacement_history"][-1][
            "augmentation_iterations"],
        "normal_multiplier_change_pa": result.info[
            "force_displacement_history"][-1]["normal_multiplier_change"],
        "checkpoint_version": result.restart_checkpoint["version"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
