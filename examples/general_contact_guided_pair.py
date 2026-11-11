"""Run the exploratory guided Q4/Q4 pair through nonlinear statics.

From the repository root:
    python examples/general_contact_guided_pair.py

Two lower Q4 faces oppose one upper Q4 face. Their physical skins initially
touch. The guides permit only normal translation, so the 0.4 N applied load
has an independent scalar contact-force and penalty-gap reference.
"""

from __future__ import annotations

import json

from anyfesolver import (
    BoundaryCondition,
    ContactDefinition,
    ContactInteraction,
    ContactSurface,
    LoadCase,
    create_shell_element,
    solve_static_nonlinear,
    validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("general_contact_guided_pair")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    lower = (
        (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.0, 1.0, 0.0),
    )
    upper = (
        (0.0, 0.0, 0.05), (1.0, 0.0, 0.05),
        (1.0, 1.0, 0.05), (0.0, 1.0, 0.05),
    )
    for node_id, position in enumerate((*lower, *upper), 1):
        model.add_node(node_id, *position)
    for element_id, nodes in (
        (1, (1, 2, 5, 4)), (2, (2, 3, 6, 5)), (3, (7, 8, 9, 10)),
    ):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "lower_fixed", (1, 2, 3, 4, 5, 6),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "upper_guided", (7, 8, 9, 10),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    load = LoadCase("press_upper")
    for node_id in (7, 8, 9, 10):
        load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.1))
    contact = ContactDefinition(
        surfaces=(
            ContactSurface("upper_bottom", (3,), side="bottom"),
            ContactSurface("lower_top", (1, 2), side="top"),
        ),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "shell_pair", "upper_bottom", "lower_top",
            normal_stiffness=1.0e6, gap_tolerance=1.0e-6,
        ),),
    )
    report = validate_general_contact(model, contact)
    if not report.valid:
        raise RuntimeError("unsupported contact definition: " + "; ".join(report.issues))
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
        "reference_overlap_area_m2": summary["reference_overlap_area"],
        "minimum_gap_m": summary["minimum_gap"],
        "upper_force_N": summary["integration_force"],
        "lower_force_N": summary["target_force"],
        "lower_support_reaction_N": list(
            result.steps[-1].support_reactions["lower_fixed"][:3]
        ),
        "gap_tolerance_met": summary["gap_tolerance_met"],
    }, indent=2))


if __name__ == "__main__":
    main()
