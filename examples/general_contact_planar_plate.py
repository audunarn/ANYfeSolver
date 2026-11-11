"""Run the opt-in C0 shell/plane contact profile through nonlinear statics.

From the repository root:
    python examples/general_contact_planar_plate.py

The initial plate touches the plane without penetration.  Its guided normal
translation gives an independent scalar reaction check: four nodal loads of
0.1 N require a 0.4 N plane reaction and a 0.4 micrometre penalty gap.
"""

from __future__ import annotations

import json

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactSurface,
    LoadCase,
    RigidContactBody,
    create_shell_element,
    solve_static_nonlinear,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("general_contact_guided_plate")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, position in enumerate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
         (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)), 1
    ):
        model.add_node(node_id, *position)
    model.add_element(
        1, create_shell_element(1, [1, 2, 3, 4], "elastic", thickness=0.05)
    )
    model.add_boundary_condition(BoundaryCondition(
        "normal_guides", [1, 2, 3, 4],
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))

    load = LoadCase("push_into_plane")
    for node_id in (1, 2, 3, 4):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.1))
    contact = ContactDefinition(
        surfaces=(ContactSurface("plate_midsurface", (1,)),),
        rigid_bodies=(RigidContactBody(
            "fixed_plane", "plane", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        ),),
        interactions=(ContactInteraction(
            "plate_plane", "plate_midsurface", "fixed_plane",
            normal_stiffness=1.0e6, gap_tolerance=1.0e-6,
        ),),
    )

    result = solve_static_nonlinear(
        model, load, contact=contact, kinematics="von_karman",
        num_steps=1, max_iterations=12,
    )
    summary = result.info["general_contact"]
    if (result.status != "completed" or result.load_factor < 1.0
            or not summary["gap_tolerance_met"]):
        raise RuntimeError(
            f"contact solve stopped: {result.status}: {result.failure_reason}"
        )
    print(json.dumps({
        "profile": summary["profile"],
        "status": result.status,
        "minimum_gap_m": summary["minimum_gap"],
        "plate_normal_force_N": summary["shell_force"][2],
        "plane_normal_force_N": summary["rigid_force"][2],
        "gap_tolerance_met": summary["gap_tolerance_met"],
    }, indent=2))


if __name__ == "__main__":
    main()
