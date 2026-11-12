"""Press a Q4 shell's physical top skin against a fixed analytic sphere.

Run from the repository root with ANYfeSolver installed:
    python examples/general_contact_sphere.py
"""

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    RigidContactBody, create_shell_element, solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("q4_physical_skin_sphere")
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
    load = LoadCase("approach")
    for node_id in range(1, 5):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, 0.1))
    definition = ContactDefinition(
        surfaces=(ContactSurface("top_skin", (1,), side="top"),),
        rigid_bodies=(RigidContactBody(
            "sphere", "sphere", (0.0, 0.0, 5.0), radius=5.0,
        ),),
        interactions=(ContactInteraction(
            "skin_sphere", "top_skin", "sphere", 1.0e5, 4.0e-3,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", maximum_shell_tilt=0.05,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported sphere contact: " + "; ".join(report.issues))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            weighted_dofs={(node, "uz"): 0.25 for node in range(1, 5)},
            target_displacement=0.003,
        ),
        num_steps=2, max_iterations=18, emit_restart_checkpoint=True,
    )
    contact = result.info.get("general_contact", {})
    if result.status != "completed" or not contact.get("gap_tolerance_met", False):
        raise RuntimeError(f"sphere contact stopped: {result.failure_reason}")
    print(json.dumps({
        "profile": contact["profile"],
        "status": result.status,
        "accepted_steps": len(result.steps),
        "minimum_certified_gap_m": contact["minimum_gap"],
        "minimum_sampled_gap_m": contact["minimum_sampled_gap"],
        "minimum_weighted_gap_m": contact["minimum_weighted_gap"],
        "pressure_node_count": contact["pressure_node_count"],
        "candidate_faces": contact["candidate_faces"],
        "gap_cells_checked": contact["gap_cells_checked"],
        "active_pressure_dofs": contact["active_pressure_dofs"],
        "shell_force_N": contact["shell_force"],
        "sphere_force_N": contact["rigid_force"],
        "restart_checkpoint_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
