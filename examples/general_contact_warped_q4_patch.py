"""Two edge-sharing warped Q4 faces against an analytic rigid plane."""

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, RigidContactBody,
    create_shell_element, solve_static_nonlinear,
)
from anyfesolver.fe_core import FEModel


model = FEModel("two_warped_q4_plane_example")
model.add_material("elastic", 1.0e5, 0.3, density=1.0)
for node_id, xyz in enumerate((
    (0.0, 0.0, 0.0), (0.5, 0.0, 0.0004), (1.0, 0.0, 0.0008),
    (0.0, 1.0, 0.0003), (0.5, 1.0, -0.0001), (1.0, 1.0, -0.0004),
), 1):
    model.add_node(node_id, *xyz)
for element_id, nodes in enumerate(((1, 2, 5, 4), (2, 3, 6, 5)), 1):
    model.add_element(element_id, create_shell_element(
        element_id, nodes, "elastic", thickness=0.08,
        reference_normal=(0.0, 0.0, 1.0),
    ))
model.add_boundary_condition(BoundaryCondition(
    "root", (1, 4),
    {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
))
load = LoadCase("tip_push")
for node_id in (3, 6):
    load.add_nodal_load(node_id, forces=(0.0, 0.0, -0.5))

contact = ContactDefinition(
    surfaces=(ContactSurface("skin", (1, 2)),),
    rigid_bodies=(RigidContactBody(
        "floor", "plane", (0.0, 0.0, -0.005), (0.0, 0.0, 1.0),
    ),),
    interactions=(ContactInteraction(
        "support", "skin", "floor", 1.0e6, 0.003,
    ),),
)
settings = ContactSolverSettings(
    warped_q4_plane=True, maximum_shell_tilt=0.05,
    warped_q4_quadrature_order=6, warped_q4_control_order=12,
    warped_q4_integration_tolerance=1.0e-6,
    warped_q4_gap_accuracy=1.0e-4,
    warped_q4_maximum_gap_cells=10000,
    warped_q4_maximum_gap_depth=16,
)
result = solve_static_nonlinear(
    model, load, contact=contact, contact_settings=settings,
    num_steps=2, max_iterations=25, kinematics="von_karman",
    emit_restart_checkpoint=True,
)
assert result.status == "completed", result.failure_reason
report = result.info["general_contact"]
print("profile:", report["profile"])
print("independent lower gap (m):", report["independent_gap_lower_bound"])
print("shell contact force (N):", report["shell_force"])
print("active pressure supports:", report["active_pressure_dofs"])
print("checkpoint version:", result.restart_checkpoint["version"])
