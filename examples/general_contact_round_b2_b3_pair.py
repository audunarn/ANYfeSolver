"""Opt-in static contact between one round B2 and one round B3 cell."""

import argparse

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear,
)
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from anyfesolver.fe_core import FEModel


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--radius-scaled-gap", action="store_true",
    help="use a 50 micrometre gap budget for the 50 mm beam radius",
)
arguments = parser.parse_args()

model = FEModel("round_b2_b3_pair_example")
model.add_material("elastic", 1.0e3, 0.3, density=1.0)
for node_id, xyz in enumerate((
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 0.12, 0.0), (0.5, 0.12, 0.0), (1.0, 0.12, 0.0),
), 1):
    model.add_node(node_id, *xyz)
section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4, "J": 1.0e-4}
model.add_element(1, BeamElement(1, (1, 2), "elastic", section))
model.add_element(2, QuadraticBeamElement(
    2, (3, 4, 5), "elastic", section,
))
model.add_boundary_condition(BoundaryCondition(
    "b2_fixed", (1, 2),
    {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
))
model.add_boundary_condition(BoundaryCondition(
    "b3_guide", (3, 4, 5),
    {name: 0.0 for name in ("ux", "uz", "rx", "ry", "rz")},
))
load = LoadCase("guided_drive")
load.add_nodal_load(5, forces=(0.0, -1.0, 0.0))
contact = ContactDefinition(
    surfaces=(ContactSurface("b2", (1,), beam_radius=0.05),
              ContactSurface("b3", (2,), beam_radius=0.05)),
    rigid_bodies=(),
    interactions=(ContactInteraction(
        "pair", "b2", "b3", normal_stiffness=None,
        gap_tolerance=5.0e-5 if arguments.radius_scaled_gap else 0.015,
        line_stiffness=1.0e4,
    ),),
)
settings = ContactSolverSettings(
    pair_motion="general", line_integration_relative_tolerance=1.0e-6,
    maximum_line_subdivisions=16, beam_pair_gap_accuracy=1.0e-5,
    maximum_beam_pair_gap_cells=10000, maximum_beam_pair_gap_depth=25,
)
result = solve_static_nonlinear(
    model, load, contact=contact, contact_settings=settings,
    control="displacement",
    displacement_control=DisplacementControl(
        node_id=5, dof="uy",
        target_displacement=-0.02004 if arguments.radius_scaled_gap else -0.03,
    ),
    num_steps=4, max_iterations=20, kinematics="von_karman",
    emit_restart_checkpoint=True,
)
assert result.status == "completed", result.failure_reason
summary = result.info["general_contact"]
print("profile:", summary["profile"])
print("conservative lower gap (m):", summary["independent_gap_lower_bound"])
print("B2 contact force (N):", summary["beam_a_force"])
print("B3 contact force (N):", summary["beam_b_force"])
print("line quadrature points:", summary["integration_points"])
print("checkpoint version:", result.restart_checkpoint["version"])
