"""One round B3 beam presses against two fixed tools in an elastic static solve.

Development profile only. Run with a package containing the documented profile:
python examples/general_contact_multiple_round_beam_tools.py
"""
import json

from anyfesolver import (BoundaryCondition, ContactDefinition, ContactInteraction,
                       ContactSolverSettings, ContactSurface, LoadCase,
                       RigidContactBody, solve_static_nonlinear, validate_general_contact)
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel


def main():
    model = FEModel("round_beam_two_fixed_tools")
    model.add_material("elastic", 1.e3, .3, density=1.)
    for node, x in enumerate((-.5, 0., .5), 1):
        model.add_node(node, x, .015, .015)
    model.add_element(1, QuadraticBeamElement(
        1, [1, 2, 3], "elastic", {"area": 1.e-3, "Iy": 1.e-4, "Iz": 1.e-4, "J": 1.e-4}))
    model.add_boundary_condition(BoundaryCondition(
        "ends", (1, 3), {dof: 0. for dof in ("ux", "uy", "uz", "rx", "ry", "rz")}))
    model.add_boundary_condition(BoundaryCondition(
        "middle_guide", (2,), {dof: 0. for dof in ("ux", "rx", "ry", "rz")}))
    definition = ContactDefinition(
        surfaces=(ContactSurface("tube", (1,), beam_radius=.01, beam_end_caps="none"),),
        rigid_bodies=(RigidContactBody("y_tool", "plane", (0., 0., 0.), normal=(0., 1., 0.)),
                      RigidContactBody("z_tool", "plane", (0., 0., 0.), normal=(0., 0., 1.))),
        interactions=(ContactInteraction("y_contact", "tube", "y_tool", None, .001,
                                         line_stiffness=1.e5),
                      ContactInteraction("z_contact", "tube", "z_tool", None, .0008,
                                         line_stiffness=2.e5)))
    settings = ContactSolverSettings(pair_motion="general")
    admission = validate_general_contact(model, definition, settings)
    if not admission.valid:
        raise RuntimeError("unsupported contact: "+"; ".join(admission.issues))
    load = LoadCase("two_component_dead_force")
    load.add_nodal_load(2, forces=(0., -10., -12.))
    result = solve_static_nonlinear(model, load, contact=definition, contact_settings=settings,
                                    num_steps=4, max_iterations=22, tolerance=1.e-8,
                                    emit_restart_checkpoint=True)
    summary = result.info["general_contact"]
    if result.status != "completed" or not summary["gap_tolerance_met"]:
        raise RuntimeError(result.failure_reason or "contact gate failed")
    print(json.dumps({"status": result.status, "profile": summary["profile"],
                      "accepted_steps": len(result.steps), "contact": summary,
                      "checkpoint_version": result.restart_checkpoint["version"]}, indent=2))


if __name__ == "__main__":
    main()
