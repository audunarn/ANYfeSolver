"""Move a Q4 skin across two nonmatching Q4 target faces.

Run from the repository root with the installed package on Python's path:
    python examples/general_contact_patch_sliding.py
"""

from __future__ import annotations

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("restricted_patch_sliding_example")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    lower = ((0.2, 0.1), (0.7, 0.1), (1.2, 0.1),
             (0.2, 1.1), (0.7, 1.1), (1.2, 1.1))
    upper = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for node, (x, y) in enumerate(lower, 1):
        model.add_node(node, x, y, 0.0)
    for node, (x, y) in enumerate(upper, 7):
        model.add_node(node, x, y, 0.05)
    for element, nodes in ((1, (1, 2, 5, 4)),
                           (2, (2, 3, 6, 5)),
                           (3, (7, 8, 9, 10))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", tuple(range(1, 7)),
        {dof: 0.0 for dof in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guide", (7, 8, 9, 10),
        {dof: 0.0 for dof in ("uy", "rx", "ry", "rz")},
    ))
    master = int(model.mesh.dof_manager.get_node_dofs(7)[0])
    for node in (8, 9, 10):
        slave = int(model.mesh.dof_manager.get_node_dofs(node)[0])
        model.add_constraint_equation(
            terms=((slave, 1.0), (master, -1.0)),
            source_id=f"uniform_upper_x_{node}",
        )
    press = LoadCase("press")
    for node in (7, 8, 9, 10):
        press.add_nodal_load(node, forces=(0.0, 0.0, -0.1))
    drive = LoadCase("drive")
    drive.add_nodal_load(7, forces=(1.0, 0.0, 0.0))
    definition = ContactDefinition(
        surfaces=(ContactSurface("upper_bottom", (3,), side="bottom"),
                  ContactSurface("lower_top", (1, 2), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "patch_pair", "upper_bottom", "lower_top", 1.0e6, 6.0e-6,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(report.issues))
    result = solve_static_nonlinear(
        model, drive, constant_load_case=press,
        contact=definition, contact_settings=settings,
        kinematics="von_karman", control="displacement",
        displacement_control=DisplacementControl(
            node_id=7, dof="ux", target_displacement=0.1,
        ),
        num_steps=2, max_iterations=12,
    )
    summary = result.info.get("general_contact", {})
    if result.status != "completed" or not summary.get("gap_tolerance_met", False):
        raise RuntimeError(f"contact stopped: {result.status}: {result.failure_reason}")
    print(json.dumps({
        "profile": summary["profile"],
        "status": result.status,
        "lateral_displacement_m": float(
            result.displacements[model.mesh.dof_manager.get_node_dofs(7)[0]]
        ),
        "overlapping_face_pairs": summary["overlapping_pairs"],
        "current_overlap_area_m2": summary["current_overlap_area"],
        "upper_contact_force_N": summary["integration_force"],
        "lower_contact_force_N": summary["target_force"],
        "conservative_lower_gap_m": summary["conservative_lower_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
