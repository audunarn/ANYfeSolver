"""A declared round B3 centreline contacts a fixed rigid plane.

Run with ANYfeSolver installed: python examples/general_contact_round_beam_plane.py
This demonstrates the classical line-penalty development profile, not C2 acceptance.
"""

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    RigidContactBody, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("round_b3_plane")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, x in ((1, -0.5), (2, 0.0), (3, 0.5)):
        model.add_node(node, x, 0.0, 0.015)
    model.add_element(1, QuadraticBeamElement(
        1, [1, 2, 3], "elastic",
        {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "ends", (1, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    contact = ContactDefinition(
        surfaces=(ContactSurface(
            "beam_tube", (1,), beam_radius=0.01, beam_end_caps="none",
        ),),
        rigid_bodies=(RigidContactBody(
            "support_plane", "plane", (0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        ),),
        interactions=(ContactInteraction(
            "tube_plane", "beam_tube", "support_plane",
            normal_stiffness=None, gap_tolerance=0.002,
            line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(pair_motion="general")
    admission = validate_general_contact(model, contact, settings)
    if not admission.valid:
        raise RuntimeError("unsupported beam contact: " + "; ".join(admission.issues))
    load = LoadCase("midpoint_drive")
    load.add_nodal_load(2, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=contact, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=2, dof="uz", target_displacement=-0.006,
        ),
        num_steps=4, max_iterations=18, emit_restart_checkpoint=True,
    )
    summary = result.info.get("general_contact", {})
    if result.status != "completed" or not summary.get("gap_tolerance_met", False):
        raise RuntimeError(f"beam-plane contact stopped: {result.failure_reason}")
    print(json.dumps({
        "profile": summary["profile"],
        "status": result.status,
        "accepted_steps": len(result.steps),
        "midpoint_displacement_m": result.steps[-1].control_value,
        "beam_radius_m": 0.01,
        "minimum_gap_m": summary["minimum_gap"],
        "declared_gap_tolerance_m": 0.002,
        "active_line_length_m": summary["active_line_length"],
        "active_intervals": summary["active_intervals"],
        "integration_points": summary["integration_points"],
        "line_stiffness_N_per_m2": summary["line_stiffness_N_per_m2"],
        "beam_force_N": summary["beam_force"],
        "plane_force_N": summary["rigid_force"],
        "restart_checkpoint_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
