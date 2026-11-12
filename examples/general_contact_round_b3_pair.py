"""Opt-in distributed static contact between two classical round B3 beams."""

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, DisplacementControl, LoadCase,
    solve_static_nonlinear,
)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel


model = FEModel("round_b3_pair_example")
model.add_material("elastic", 1.0e3, 0.3, density=1.0)
for node_id, xyz in enumerate((
    (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 0.12, 0.0), (0.5, 0.12, 0.0), (1.0, 0.12, 0.0),
), 1):
    model.add_node(node_id, *xyz)
section = {"area": 1.0e-3, "Iy": 1.0e-4, "Iz": 1.0e-4,
           "J": 1.0e-4}
model.add_element(1, QuadraticBeamElement(
    1, (1, 2, 3), "elastic", section,
))
model.add_element(2, QuadraticBeamElement(
    2, (4, 5, 6), "elastic", section,
))
model.add_boundary_condition(BoundaryCondition(
    "beam_a_fixed", (1, 2, 3),
    {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
))
model.add_boundary_condition(BoundaryCondition(
    "beam_b_guide", (4, 5, 6),
    {name: 0.0 for name in ("ux", "uz", "rx", "ry", "rz")},
))
load = LoadCase("guided_drive")
load.add_nodal_load(6, forces=(0.0, -1.0, 0.0))
contact = ContactDefinition(
    surfaces=(ContactSurface("a", (1,), beam_radius=0.05),
              ContactSurface("b", (2,), beam_radius=0.05)),
    rigid_bodies=(),
    interactions=(ContactInteraction(
        "a_b", "a", "b", normal_stiffness=None,
        gap_tolerance=0.015, line_stiffness=1.0e4,
    ),),
)
settings = ContactSolverSettings(
    pair_motion="general", line_integration_relative_tolerance=1.0e-6,
    maximum_line_subdivisions=16,
    beam_pair_gap_accuracy=1.0e-5,
    maximum_beam_pair_gap_cells=10000,
    maximum_beam_pair_gap_depth=25,
)
result = solve_static_nonlinear(
    model, load, contact=contact, contact_settings=settings,
    control="displacement",
    displacement_control=DisplacementControl(
        node_id=6, dof="uy", target_displacement=-0.03,
    ),
    num_steps=4, max_iterations=20, kinematics="von_karman",
    emit_restart_checkpoint=True,
)
assert result.status == "completed", result.failure_reason
summary = result.info["general_contact"]
print("profile:", summary["profile"])
print("conservative lower gap (m):", summary["independent_gap_lower_bound"])
print("beam A contact force (N):", summary["beam_a_force"])
print("beam B contact force (N):", summary["beam_b_force"])
print("line quadrature points:", summary["integration_points"])
print("projection evaluations:", summary["projection_evaluations"])
print("checkpoint version:", result.restart_checkpoint["version"])
