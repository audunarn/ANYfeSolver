"""Bounded planar Q4/S3 finite-penalty development example (SI units).

Press a guided shell, translate a fixed-shape target, open and recontact.
Partial target coverage and active prescribed source nodes are intentional.
No friction, finite-rotation, warped-shell or general rim admission is implied.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction, ContactPathControl,
    ContactPathProgram, ContactSolverSettings, ContactSurface, FEModel,
    create_shell_element, solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.nonlinear_restart import (
    canonical_checkpoint_json_bytes, load_nonlinear_checkpoint,
)


def build():
    model = FEModel("partial_shell_cycle")
    model.add_material("elastic", 1e5, .3, density=1.)
    xy = ((.1,.1),(.6,.1),(1.1,.1),(.1,1.1),(.6,1.1),(1.1,1.1),
          (0.,0.),(1.,0.),(1.,1.),(0.,1.))
    for node, (x,y) in enumerate(xy, 1):
        model.add_node(node, x, y, .0401 if node <= 6 else 0.)
    for eid, nodes in ((1,(1,2,5,4)),(2,(2,3,6,5)),
                       (3,(7,8,10)),(4,(8,9,10))):
        extra = ({} if eid <= 2 else
                 dict(formulation="e4-pl-s3-v2d", reference_surface_offset=.01))
        model.add_element(eid, create_shell_element(
            eid, nodes, "elastic", thickness=.05,
            reference_normal=(0.,0.,1.), **extra))
    model.add_boundary_condition(BoundaryCondition("source_guides", tuple(range(1,7)),
        {name:0. for name in ("ux","uy","rx","ry","rz")}))
    model.add_boundary_condition(BoundaryCondition("target_guides", tuple(range(7,11)),
        {name:0. for name in ("uy","uz","rx","ry","rz")}))
    for node in (8,9,10):
        model.add_constraint_equation(terms=(
            (int(model.mesh.dof_manager.get_node_dofs(node)[0]),1.),
            (int(model.mesh.dof_manager.get_node_dofs(7)[0]),-1.)),
            source_id=f"target_translation_{node}")
    definition = ContactDefinition(surfaces=(
        ContactSurface("source", (1,2), "bottom"),
        ContactSurface("target", (3,4), "top")), rigid_bodies=(),
        interactions=(ContactInteraction("pair", "source", "target", 1e7, 5e-5,
                                        normal_law="projected_penalty"),))
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_difference_step=2e-5, maximum_general_s3_energy_evaluations=5000,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_area_tolerance=1e-9, q4_source_weighted_gap_tolerance=1e-10,
        maximum_q4_source_integration_triangles=128,
        q4_source_support_gradient_tolerance=1e-5,
        q4_source_support_hessian_tolerance=1e-5,
        q4_source_weighted_gap_gradient_tolerance=1e-5,
        q4_source_weighted_gap_hessian_tolerance=1e-5,
        q4_source_gap_certificate="overlap_local_dev",
        augmentation_acceleration="none", augmentation_support_resolution="exact")
    program = ContactPathProgram(controls=(ContactPathControl.node(1,"uz"),
        ContactPathControl.node(4,"uz"), ContactPathControl.node(7,"ux")),
        waypoints=((0.,0.,0.),(-130e-6,-130e-6,0.),
                   (-130e-6,-130e-6,.08),(-130e-6,-130e-6,.16),
                   (0.,0.,.16),(-130e-6,-130e-6,.16)), steps_per_segment=2)
    return model, definition, settings, program


def solve(fraction=1., checkpoint=None):
    model, definition, settings, program = build()
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(report.issues)
    result = solve_static_nonlinear(model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_load_factor=fraction, max_iterations=18,
        tolerance=1e-8, emit_restart_checkpoint=True, restart_checkpoint=checkpoint)
    if result.status != "completed":
        raise RuntimeError(result.failure_reason)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", action="store_true", help="persist active/open prefixes and resume")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = None
    fractions = (.6,.8,1.) if args.restart else (1.,)
    for fraction in fractions:
        result = solve(fraction, checkpoint)
        if fraction < 1.:
            path = args.output.with_name(args.output.stem +
                (".active" if fraction == .6 else ".open") + ".checkpoint.json")
            with path.open("xb") as stream:
                stream.write(canonical_checkpoint_json_bytes(result.restart_checkpoint))
            checkpoint = load_nonlinear_checkpoint(path.read_bytes())
    rows = []
    for history in result.info["force_displacement_history"]:
        contact = history["contact"]
        rows.append({"step":history["step_index"], "active":contact["active_pressure_dofs"],
            "source_force_n":contact["integration_force"],
            "target_force_n":contact["target_force"], "gap_lower_m":contact["minimum_gap"],
            "control_reaction":history["reaction_generalized"]})
    payload = dict(status=result.status, profile=result.info["force_displacement_history"][-1]["contact"]["profile"],
        lifecycle=rows, final_displacements=result.displacements.tolist(),
        checkpoint=result.restart_checkpoint, accepted=False, released=False)
    args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=result.status, poses=len(rows), output=str(args.output))))


if __name__ == "__main__":
    main()
