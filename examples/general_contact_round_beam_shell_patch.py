"""B2 or B3 round beam pressing across an internal planar Q4 seam.

Run after installing ANYfeSolver: python examples/general_contact_round_beam_shell_patch.py
Pass --b3 for the quadratic beam route.
All dimensions use SI units. This is a development example, not qualification.
"""

import argparse

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def main(*, b3: bool = False) -> None:
    model = FEModel("beam_across_internal_shell_seam")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    beam_points = (((-0.5, 0.0, 0.025), (0.0, 0.0, 0.025), (0.5, 0.0, 0.025))
                   if b3 else ((-0.5, 0.0, 0.025), (0.5, 0.0, 0.025)))
    shell_start = len(beam_points) + 1
    left = tuple(range(shell_start, shell_start + 4))
    right = (shell_start + 1, shell_start + 4,
             shell_start + 5, shell_start + 2)
    shell_nodes = tuple(range(shell_start, shell_start + 6))
    for node, point in enumerate((
        *beam_points,
        (-1.0, -1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0), (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0), (1.0, 1.0, 0.0),
    ), 1):
        model.add_node(node, *point)
    model.add_element(1, (QuadraticBeamElement if b3 else BeamElement)(
        1, list(range(1, shell_start)), "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    for eid, nodes in ((2, left), (3, right)):
        model.add_element(eid, create_shell_element(
            eid, nodes, "elastic", thickness=0.02, formulation="e4-pl",
            reference_normal=(0, 0, 1),
        ))
    model.add_boundary_condition(BoundaryCondition(
        "fixed_skin", shell_nodes,
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "fixed_beam_root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guided_beam_tip", tuple(range(2, shell_start)),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_across_seam")
    load.add_nodal_load(shell_start - 1, forces=(0.0, 0.0, -0.1 if b3 else -1.0))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=0.01),
                  ContactSurface("skin", (2, 3), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "line_skin", "tube", "skin", None, 0.002,
            line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", beam_shell_rim_clearance=0.02,
        beam_shell_difference_step=1.0e-8,
        line_integration_relative_tolerance=1.0e-8,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("; ".join(report.issues))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        tolerance=1.0e-8 if b3 else 1.0e-6,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(result.failure_reason)
    contact = result.info["general_contact"]
    print("profile:", contact["profile"])
    print("minimum gap [m]:", contact["minimum_gap"])
    print("beam force [N]:", contact["beam_force"])
    print("shell force [N]:", contact["shell_force"])
    print("minimum exterior rim clearance [m]:", contact["minimum_rim_clearance"])
    print("checkpoint version:", result.to_restart_checkpoint()["version"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3", action="store_true", help="use a quadratic B3 cell")
    main(b3=parser.parse_args().b3)
