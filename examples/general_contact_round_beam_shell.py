"""One round B2 centreline contacting a physical S3 or planar Q4 shell skin.

Run with ``python examples/general_contact_round_beam_shell.py`` after installing
ANYfeSolver and its normal dependencies. Pass ``--q4`` for the planar Q4 route.
Lengths are metres and forces newtons.
"""

import argparse

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def main(*, line_stiffness: float = 1.0e4,
         gap_tolerance: float = 0.002,
         shell_kind: str = "s3") -> None:
    if shell_kind not in {"s3", "q4"}:
        raise ValueError("shell_kind must be s3 or q4")
    model = FEModel(f"round_beam_on_{shell_kind}_skin")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    shell_points = (
        ((-1.0, -1.0, 0.0), (2.0, -1.0, 0.0), (-1.0, 2.0, 0.0))
        if shell_kind == "s3" else
        ((-1.0, -1.0, 0.0), (2.0, -1.0, 0.0),
         (2.0, 2.0, 0.0), (-1.0, 2.0, 0.0))
    )
    shell_nodes = tuple(range(3, 3 + len(shell_points)))
    for identifier, coordinates in enumerate((
        (0.0, 0.0, 0.025), (0.5, 0.0, 0.025),
        *shell_points,
    ), 1):
        model.add_node(identifier, *coordinates)
    model.add_element(1, BeamElement(
        1, [1, 2], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    model.add_element(2, create_shell_element(
        2, shell_nodes, "elastic", thickness=0.02,
        formulation=("e4-pl-s3-v2d" if shell_kind == "s3" else "e4-pl"),
        reference_normal=(0, 0, 1),
        **({"reference_surface_offset": 0.0} if shell_kind == "s3" else {}),
    ))
    model.add_boundary_condition(BoundaryCondition(
        "fixed_skin", shell_nodes,
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "fixed_root", (1,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guided_tip", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_tube")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))

    definition = ContactDefinition(
        surfaces=(
            ContactSurface("tube", (1,), beam_radius=0.01),
            ContactSurface("skin", (2,), side="top"),
        ),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "tube_on_skin", "tube", "skin", None, gap_tolerance,
            line_stiffness=line_stiffness,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", beam_shell_rim_clearance=0.02,
        beam_shell_difference_step=1.0e-8,
        line_integration_relative_tolerance=1.0e-8,
    )
    admission = validate_general_contact(model, definition, settings)
    if not admission.valid:
        raise RuntimeError("; ".join(admission.issues))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(
            f"{result.failure_reason}; load_factor={result.load_factor}; "
            f"contact={result.info.get('general_contact')}"
        )
    contact = result.info["general_contact"]
    print("profile:", contact["profile"])
    print("minimum gap [m]:", contact["minimum_gap"])
    print("beam force [N]:", contact["beam_force"])
    print("shell force [N]:", contact["shell_force"])
    print("minimum rim clearance [m]:", contact["minimum_rim_clearance"])
    print("checkpoint version:", result.to_restart_checkpoint()["version"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4", action="store_true", help="use a planar Q4 skin")
    arguments = parser.parse_args()
    main(shell_kind="q4" if arguments.q4 else "s3")
