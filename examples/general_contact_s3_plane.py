"""Run the S3 V2D physical-bottom-skin plane-contact profile.

From an environment containing this checkout:
    python examples/general_contact_s3_plane.py
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
    model = FEModel("s3_physical_skin_plane")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for node_id, (x, y) in enumerate(((0.0, 0.0), (1.0, 0.0),
                                       (0.0, 1.0)), 1):
        model.add_node(node_id, x, y, 0.035)
    model.add_element(1, create_shell_element(
        1, (1, 2, 3), "elastic", formulation="e4-pl-s3-v2d",
        thickness=0.05, reference_normal=(0.0, 0.0, 1.0),
        reference_surface_offset=0.01,
    ))
    model.add_boundary_condition(BoundaryCondition(
        "normal_guides", (1, 2, 3),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_skin")
    for node_id in (1, 2, 3):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.1))
    contact = ContactDefinition(
        surfaces=(ContactSurface("physical_bottom", (1,), side="bottom"),),
        rigid_bodies=(RigidContactBody(
            "floor", "plane", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "skin_floor", "physical_bottom", "floor", 1.0e6, 1.0e-6,
        ),),
    )
    result = solve_static_nonlinear(
        model, load, contact=contact, kinematics="von_karman",
        num_steps=1, max_iterations=12,
    )
    summary = result.info.get("general_contact", {})
    if (result.status != "completed" or result.load_factor < 1.0
            or not summary.get("gap_tolerance_met", False)):
        raise RuntimeError(
            f"contact solve stopped: {result.status}: {result.failure_reason}"
        )
    print(json.dumps({
        "profile": summary["profile"],
        "status": result.status,
        "minimum_gap_m": summary["minimum_gap"],
        "shell_force_N": summary["shell_force"],
        "plane_force_N": summary["rigid_force"],
        "gap_tolerance_met": summary["gap_tolerance_met"],
    }, indent=2))


if __name__ == "__main__":
    main()
