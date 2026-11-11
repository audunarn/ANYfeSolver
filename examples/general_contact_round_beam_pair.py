"""Two declared round B2 beams meet under prescribed displacement.

Run with ANYfeSolver installed: python examples/general_contact_round_beam_pair.py
This is a distributed line-contact development example, not C2 acceptance.
"""

import json

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def main() -> None:
    model = FEModel("crossing_round_b2_pair")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, position in enumerate((
        (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0),
        (0.0, -0.5, 0.03), (0.0, 0.5, 0.03),
    ), 1):
        model.add_node(node, *position)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4,
               "J": 1.0e-4}
    model.add_element(1, BeamElement(1, [1, 2], "elastic", section))
    model.add_element(2, BeamElement(2, [3, 4], "elastic", section))
    model.add_boundary_condition(BoundaryCondition(
        "beam_a_fixed", (1, 2),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_b_root", (3,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "beam_b_tip_guide", (4,),
        {name: 0.0 for name in ("ux", "uy", "rx", "ry", "rz")},
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("a", (1,), beam_radius=0.01),
                  ContactSurface("b", (2,), beam_radius=0.01)),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "a_b", "a", "b", normal_stiffness=None,
            line_stiffness=1.0e4, gap_tolerance=0.004,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-7,
    )
    admission = validate_general_contact(model, definition, settings)
    if not admission.valid:
        raise RuntimeError("unsupported pair: " + "; ".join(admission.issues))
    load = LoadCase("tip_drive")
    load.add_nodal_load(4, forces=(0.0, 0.0, -1.0))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=4, dof="uz", target_displacement=-0.025,
        ),
        num_steps=8, max_iterations=20, emit_restart_checkpoint=True,
    )
    contact = result.info.get("general_contact", {})
    if result.status != "completed" or not contact.get("gap_tolerance_met", False):
        raise RuntimeError(f"beam pair stopped: {result.failure_reason}")
    print(json.dumps({
        "profile": contact["profile"],
        "status": result.status,
        "accepted_steps": len(result.steps),
        "tip_displacement_m": result.steps[-1].control_value,
        "minimum_gap_m": contact["minimum_gap"],
        "declared_gap_tolerance_m": 0.004,
        "contact_energy_J": contact["contact_energy"],
        "beam_a_force_N": contact["beam_a_force"],
        "beam_b_force_N": contact["beam_b_force"],
        "active_source_intervals": contact["active_source_intervals"],
        "integration_points": contact["integration_points"],
        "maximum_integration_relative_error": contact[
            "maximum_integration_relative_error"
        ],
        "swept_trial_cutbacks": contact["swept_trial_cutbacks"],
        "restart_checkpoint_version": result.restart_checkpoint["version"],
    }, indent=2))


if __name__ == "__main__":
    main()
