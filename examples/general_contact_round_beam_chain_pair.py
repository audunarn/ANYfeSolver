"""Two connected B2 chains in distributed round-centreline contact.

Run after installing ANYfeSolver: python examples/general_contact_round_beam_chain_pair.py
Use --consistent for the opt-in development tangent with moving-owner terms.
SI units are used throughout. This is a development example, not qualification.
"""

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl,
    LoadCase, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def main(*, beam_chain_tangent: str = "finite_difference") -> None:
    model = FEModel("round_b2_chain_pair")
    model.add_material("elastic", 1.0e3, 0.3, density=1.0)
    for node, point in enumerate((
        (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (-1.0, 0.025, 0.0), (0.0, 0.025, 0.0), (1.0, 0.025, 0.0),
    ), 1):
        model.add_node(node, *point)
    section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}
    for eid, nodes in ((1, (1, 2)), (2, (2, 3)),
                       (3, (4, 5)), (4, (5, 6))):
        model.add_element(eid, BeamElement(eid, nodes, "elastic", section))
    model.add_boundary_condition(BoundaryCondition(
        "fixed_a", (1, 2, 3),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "root_b", (4,),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "guided_b", (5, 6),
        {name: 0.0 for name in ("ux", "uz", "rx", "ry", "rz")},
    ))
    load = LoadCase("drive_chain")
    load.add_nodal_load(6, forces=(0.0, -1.0, 0.0))
    definition = ContactDefinition(
        surfaces=(ContactSurface("a", (1, 2), beam_radius=0.01),
                  ContactSurface("b", (3, 4), beam_radius=0.01)),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "a_b", "a", "b", None, 0.004, line_stiffness=1.0e4,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", line_integration_relative_tolerance=1.0e-6,
        maximum_line_subdivisions=12, sliding_difference_step=1.0e-7,
        beam_chain_tangent=beam_chain_tangent,
    )
    admission = validate_general_contact(model, definition, settings)
    if not admission.valid:
        raise RuntimeError("; ".join(admission.issues))
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="displacement",
        displacement_control=DisplacementControl(
            node_id=6, dof="uy", target_displacement=-0.008,
        ),
        num_steps=4, max_iterations=15, tolerance=1.0e-8,
        emit_restart_checkpoint=True,
    )
    if result.status != "completed":
        raise RuntimeError(result.failure_reason)
    contact = result.info["general_contact"]
    print("profile:", contact["profile"])
    print("minimum gap [m]:", contact["minimum_gap"])
    print("chain A force [N]:", contact["beam_a_force"])
    print("chain B force [N]:", contact["beam_b_force"])
    print("integration relative error:", contact["maximum_integration_relative_error"])
    print("checkpoint version:", result.to_restart_checkpoint()["version"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consistent", action="store_true")
    args = parser.parse_args()
    main(beam_chain_tangent="consistent_dev" if args.consistent else "finite_difference")
