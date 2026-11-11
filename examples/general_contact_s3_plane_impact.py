"""Guided S3 V2D impact with declared massless rotational coordinates."""

import argparse
import numpy as np

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactNodalLoadHistory,
    ContactSurface,
    ContactTransientSolverControls,
    LoadCase,
    RigidContactBody,
    TransientConfig,
    create_shell_element,
    solve_transient_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant-downward-load-n", type=float, default=0.0)
    parser.add_argument("--ramp-downward-load-n", type=float, default=0.0,
                        help="dead nodal force ramped from zero to this value at .012s")
    parser.add_argument("--one-sided-rates", action="store_true",
                        help="opt into right directional massless rates at touch (development)")
    parser.add_argument("--algebraic-path", action="store_true",
                        help="lift the physical quadratic interval through algebraic equilibrium (development)")
    parser.add_argument("--event-subdivision", action="store_true",
                        help="integrate at localized represented contact events (development)")
    args = parser.parse_args()
    if args.algebraic_path and not args.one_sided_rates:
        parser.error("--algebraic-path requires --one-sided-rates")
    if args.event_subdivision and not args.algebraic_path:
        parser.error("--event-subdivision requires --algebraic-path")
    if (not np.isfinite(args.constant_downward_load_n)
            or args.constant_downward_load_n < 0.0):
        parser.error("constant downward load must be finite and nonnegative")
    if not np.isfinite(args.ramp_downward_load_n) or args.ramp_downward_load_n < 0.0:
        parser.error("ramp downward load must be finite and nonnegative")
    if args.ramp_downward_load_n and args.constant_downward_load_n:
        parser.error("select either constant or ramped load")
    director = np.asarray((0.002, 0.0, 1.0), dtype=float)
    director /= np.linalg.norm(director)
    span = 0.1
    tangent_x = span*np.asarray((director[2], 0.0, -director[0]))
    tangent_y = np.asarray((0.0, span, 0.0))
    model = FEModel("s3_descriptor_plane_impact_development")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, corner in enumerate((np.zeros(3), tangent_x, tangent_y), 1):
        model.add_node(node_id, *(corner + .035*director))
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", formulation="e4-pl-s3-v2d",
        thickness=.05, reference_normal=tuple(director),
        reference_surface_offset=.01,
    ))
    model.add_boundary_condition(BoundaryCondition(
        "in_plane_guide", (1, 2, 3), {"ux": 0.0, "uy": 0.0},
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("bottom_skin", (1,), side="bottom"),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0.0, 0.0, -.01-span*director[0]),
            (0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "s3_floor", "bottom_skin", "floor", 1.0e5, .001,
        ),),
    )
    initial_velocity = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id in (1, 2, 3):
        initial_velocity[model.mesh.dof_manager.get_node_dofs(node_id)[2]] = -1.0
    load = None
    downward_force = args.constant_downward_load_n or args.ramp_downward_load_n
    if downward_force:
        load = LoadCase("uniform_s3_normal_dead_load")
        for node_id in (1, 2, 3):
            load.add_nodal_load(
                node_id,
                forces=(0.0, 0.0, -downward_force/3.0),
            )
    result = solve_transient_contact(
        model, definition,
        TransientConfig(.0002, .012, initial_velocity=initial_velocity),
        base_load_case=load,
        nodal_load_history=(ContactNodalLoadHistory(((0., 0.), (.012, 1.)))
                            if args.ramp_downward_load_n else None),
        solver_controls=ContactTransientSolverControls(
            allow_s3_descriptor_dev=True, s3_one_sided_rates_dev=args.one_sided_rates,
            s3_algebraic_path_dev=args.algebraic_path,
            s3_event_subdivision_dev=args.event_subdivision,
        ),
    )
    ry = [model.mesh.dof_manager.get_node_dofs(node)[4] for node in (1, 2, 3)]
    summary = {
        "profile": result.profile,
        "accepted_steps": len(result.newton_iterations),
        "cutbacks": result.cutbacks,
        "minimum_swept_gap_m": float(np.min(result.swept_minimum_gaps)),
        "final_active_pressure_nodes": result.contact[-1].active_pressure_dofs,
        "maximum_final_ry_rad": float(np.max(np.abs(result.displacements[-1, ry]))),
        "maximum_residual_n": float(np.max(result.equilibrium_residuals)),
        "external_nodal_work_j": float(result.external_nodal_work_j[-1]),
    }
    if args.algebraic_path:
        summary["represented_path_transition_times_s"] = result.represented_path_transition_times
    if args.event_subdivision:
        summary["localized_contact_event_times_s"] = result.localized_contact_event_times
        summary["event_endpoint_evaluations"] = result.event_endpoint_evaluations
    print(summary)


if __name__ == "__main__":
    main()
