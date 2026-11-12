"""Fixed sphere against a declared round classical B3 centreline.

Run with ANYfeSolver installed: python examples/general_contact_round_beam_sphere.py
This is a static development example, not C2 engineering acceptance.
"""

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction, ContactSolverSettings,
    ContactSurface, DisplacementControl, LoadCase, RigidContactBody,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("round_b3_fixed_sphere")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, position in enumerate((
        (-0.5, 0.0, 0.03), (0.0, 0.0, 0.03), (0.5, 0.0, 0.03),
    ), 1):
        model.add_node(node, *position)
    model.add_element(1, QuadraticBeamElement(
        1, [1, 2, 3], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "ends_fixed", (1, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),),
        rigid_bodies=(RigidContactBody(
            "sphere", "sphere", (0.0, 0.0, 0.0), radius=0.01,
        ),),
        interactions=(ContactInteraction(
            "tube_sphere", "tube", "sphere", normal_stiffness=None,
            line_stiffness=1.0e4, gap_tolerance=0.004,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-7,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError(report.issues)
    load = LoadCase("middle_drive")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2, dof="uz", target_displacement=-0.012,
        ),
        num_steps=8, max_iterations=20, emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(result.failure_reason)
    summary = result.info["general_contact"]
    print(json.dumps({
        "profile": summary["profile"],
        "status": result.status,
        "middle_displacement_m": result.displacements[
            model.mesh.dof_manager.get_node_dofs(2)[2]
        ],
        "minimum_gap_m": summary["minimum_gap"],
        "declared_gap_tolerance_m": definition.interactions[0].gap_tolerance,
        "line_contact_energy_J": summary["line_contact_energy"],
        "beam_force_N": summary["beam_force"],
        "rigid_force_N": summary["rigid_force"],
        "active_line_length_m": summary["active_line_length"],
        "integration_points": summary["integration_points"],
        "maximum_integration_relative_error": summary["maximum_integration_relative_error"],
        "restart_checkpoint_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
