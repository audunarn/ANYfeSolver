"""A prescribed infinite cylinder presses a declared round B2 centreline.

Run after installing ANYfeSolver:
python examples/general_contact_round_beam_moving_cylinder.py
All values use SI units. This is a development example, not qualification.
"""

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, RigidContactBody,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("moving_cylinder_on_round_beam")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    model.add_node(1, -0.5, 0.0, 0.03)
    model.add_node(2, 0.5, 0.0, 0.03)
    model.add_element(1, BeamElement(
        1, (1, 2), "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guided_tip", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("small_tip_force")
    load.add_nodal_load(2, forces=(0.0, 0.0, 1.0e-5))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),),
        rigid_bodies=(RigidContactBody(
            "cylinder", "cylinder", (0.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0), radius=0.01,
            path_translation=(0.0, 0.0, 0.012),
        ),),
        interactions=(ContactInteraction(
            "tube_cylinder", "tube", "cylinder", None, 0.004,
            line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-7,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("; ".join(report.issues))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        num_steps=8, max_iterations=20, emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(result.failure_reason)
    contact = result.info["general_contact"]
    print("profile:", contact["profile"])
    print("minimum gap [m]:", contact["minimum_gap"])
    print("beam force [N]:", contact["beam_force"])
    print("rigid force [N]:", contact["rigid_force"])
    print("rigid tool position [m]:", contact["rigid_tool_position"])
    print("tool work, trapezoid [J]:", contact["rigid_contact_work_trapezoid"])
    print("checkpoint version:", result.to_restart_checkpoint()["version"])


if __name__ == "__main__":
    main()
