"""A prescribed large-radius sphere presses a Q4 shell through static steps.

Run with ANYfeSolver installed: python examples/general_contact_moving_sphere.py
The 20 µm gap budget is 10^-3 of shell thickness; this is development smoke.
"""

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase,
    NonlinearConvergenceSettings, RigidContactBody,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("q4_moving_sphere")
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
    contact = ContactDefinition(
        surfaces=(ContactSurface("top_skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody(
            "tool", "sphere", (0.0, 0.0, 5000.000002), radius=5000.0,
            path_translation=(0.0, 0.0, -0.00001),
        ),),
        interactions=(ContactInteraction(
            "tool_skin", "top_skin", "tool", 1.0e7, 2.0e-5,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general",
                                     maximum_shell_tilt=0.05)
    report = validate_general_contact(model, contact, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    result = solve_static_nonlinear(
        model, LoadCase("prescribed_tool"),
        contact=contact, contact_settings=settings,
        convergence_settings=NonlinearConvergenceSettings(
            growth_factor=1.0, max_step_factor=1.0,
        ),
        num_steps=8, max_iterations=18, emit_restart_checkpoint=True,
    )
    summary = result.info.get("general_contact", {})
    if (result.status != "completed" or
            not summary.get("gap_tolerance_met", False) or
            summary.get("active_pressure_dofs", 0) == 0):
        raise RuntimeError(f"moving sphere stopped: {result.failure_reason}")
    print(json.dumps({
        "profile": summary["profile"],
        "status": result.status,
        "accepted_steps": len(result.steps),
        "gap_tolerance_m": 2.0e-5,
        "minimum_certified_gap_m": summary["minimum_gap"],
        "active_pressure_dofs": summary["active_pressure_dofs"],
        "tool_position_m": summary["rigid_tool_position"],
        "shell_force_N": summary["shell_force"],
        "sphere_force_N": summary["rigid_force"],
        "rigid_contact_work_trapezoid_J": summary["rigid_contact_work_trapezoid"],
        "prescribed_actuator_work_trapezoid_J": summary[
            "prescribed_actuator_work_trapezoid"
        ],
        "swept_trial_cutbacks": summary["swept_trial_cutbacks"],
        "restart_checkpoint_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
