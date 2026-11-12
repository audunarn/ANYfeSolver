"""Force-controlled shell contact with a dimensional gap budget.

Run with ANYfeSolver installed: python examples/general_contact_sphere_tight_gap.py
The large sphere radius makes this a well-resolved, nearly flat benchmark.
"""

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, RigidContactBody,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("q4_sphere_tight_gap")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, (x, y) in enumerate((
        (-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1),
    ), 1):
        model.add_node(node_id, x, y, -0.01)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3, 4), "elastic", thickness=0.02,
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guide", (1, 2, 3, 4),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "normal_anchor", (1,), {"uz": 0.0},
    ))
    load = LoadCase("press")
    for node_id in (2, 3, 4):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, 0.5))
    definition = ContactDefinition(
        surfaces=(ContactSurface("top_skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody(
            "sphere", "sphere", (0.0, 0.0, 5000.0), radius=5000.0,
        ),),
        interactions=(ContactInteraction(
            "skin_sphere", "top_skin", "sphere", 1.0e7, 2.0e-5,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", maximum_shell_tilt=0.05,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=4, max_iterations=18, emit_restart_checkpoint=True,
    )
    contact = result.info.get("general_contact", {})
    if (result.status != "completed" or
            not contact.get("gap_tolerance_met", False) or
            contact.get("active_pressure_dofs", 0) == 0):
        raise RuntimeError(f"contact stopped: {result.failure_reason}; {contact}")
    print(json.dumps({
        "profile": contact["profile"],
        "status": result.status,
        "gap_tolerance_m": 2.0e-5,
        "minimum_certified_gap_m": contact["minimum_gap"],
        "minimum_sampled_gap_m": contact["minimum_sampled_gap"],
        "active_pressure_dofs": contact["active_pressure_dofs"],
        "shell_force_N": contact["shell_force"],
        "sphere_force_N": contact["rigid_force"],
        "restart_checkpoint_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
