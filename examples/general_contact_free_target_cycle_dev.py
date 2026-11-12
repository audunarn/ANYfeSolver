"""Development example: deformable Q4/S3 press, slide, open and recontact.

Run with ``python general_contact_free_target_cycle_dev.py`` against an
installed ANYfeSolver package. The contact route is implemented and locally
tested; the complete D1 capability is not independently accepted or released.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import anyfesolver
from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathNodalForce, ContactPathProgram,
    ContactSolverSettings, ContactSurface, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def build_case():
    """Build the bounded two-body example without a source-checkout helper."""
    model = FEModel("mixed_4_q4_x_jet2_schur_press")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    for row, y in enumerate((0.1, 1.1)):
        for column in range(5):
            model.add_node(1+5*row+column, 0.1+column/4, y, 0.04004)
    for node, (x, y) in zip((11, 12, 13, 14),
                             ((0.0, 0.0), (1.3, 0.0),
                              (1.3, 1.0), (0.0, 1.0)), strict=True):
        model.add_node(node, x, y, 0.0)
    for element_id in range(1, 5):
        model.add_element(element_id, create_shell_element(
            element_id,
            (element_id, element_id+1, element_id+6, element_id+5),
            "elastic", thickness=0.05,
            reference_normal=(0.0, 0.0, 1.0),
        ))
    for element_id, nodes in ((5, (11, 12, 14)), (6, (12, 13, 14))):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d",
            reference_normal=(0.0, 0.0, 1.0),
            reference_surface_offset=0.01,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", tuple(range(1, 11)),
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_left_support", (1, 6), {"uz": 0.0},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "target_left_clamp", (11, 14),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    definition = ContactDefinition(
        surfaces=(
            ContactSurface("q4_source", (1, 2, 3, 4), "bottom"),
            ContactSurface("s3_target", (5, 6), "top"),
        ),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", 10000.0, 5.0e-5,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        q4_source_gap_certificate="projected_matching",
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=14144,
        maximum_candidate_pairs=32,
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=300,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        pressure_tolerance=1.0e-5, maximum_augmentations=7,
        enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=0.3,
        augmentation_acceleration="none",
        augmentation_schur_maximum_modes=None,
        augmentation_schur_maximum_pressure_pa=None,
        augmentation_schur_maximum_step_pa=None,
        augmentation_schur_difference_step_m=None,
        augmentation_schur_maximum_attempts=1,
        augmentation_schur_maximum_relinearizations=0,
    )
    force_factors = (0.0, 1.0, 1.0, 0.0, 1.0)
    nodal_weights = {
        1+5*row+column:
        (0.5 if column in (0, 4) else 1.0)/4*0.5
        for row in range(2) for column in range(5)
    }
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(6, "ux")),
        waypoints=((0.0, 0.0), (0.0, 0.0), (0.05, 0.05),
                   (0.05, 0.05), (0.05, 0.05)),
        steps_per_segment=2,
        nodal_forces=tuple(ContactPathNodalForce(
            node, "uz", tuple(-0.3*weight*factor
                              for factor in force_factors),
        ) for node, weight in nodal_weights.items()),
    )
    return model, definition, settings, program


def _physical_skin_vertices(model, element, displacement, side):
    """Reconstruct this example's small-rotation skin without contact code."""
    reference = np.asarray([
        (model.mesh.nodes[node].x, model.mesh.nodes[node].y,
         model.mesh.nodes[node].z) for node in element.node_ids
    ], dtype=float)
    if len(element.node_ids) == 4:
        director = np.cross(reference[1]-reference[0],
                            reference[3]-reference[0])
        director /= np.linalg.norm(director)
        offset = (-0.5 if side == "bottom" else 0.5)*element.thickness
    else:
        director = (np.asarray(element.reference_normal, dtype=float)
                    * element.director_polarity)
        offset = 0.5*element.thickness-element.reference_surface_offset
    vertices = []
    for node_id, point in zip(element.node_ids, reference, strict=True):
        dofs = model.mesh.dof_manager.get_node_dofs(node_id)
        rotation = displacement[dofs[3:6]]
        vertices.append(point+displacement[dofs[:3]]
                        +offset*(director+np.cross(rotation, director)))
    return np.asarray(vertices)


def _independent_projected_gap_bound(model, definition, displacement,
                                     gap_budget):
    """Bound physical target-normal gap over this planar XY-overlap example.

    A Q4 point and its signed distance from one planar S3 face are bilinear in
    each Q4 parameter box, so the box's four corners bound its gap. Boxes
    crossing an S3 footprint edge are subdivided; unresolved geometry fails.
    This bound does not claim closest-point or edge-only feature contact.
    """
    source_surface, target_surface = definition.surfaces
    sources = [_physical_skin_vertices(
        model, model.mesh.elements[element_id], displacement,
        source_surface.side) for element_id in source_surface.element_ids]
    targets = [_physical_skin_vertices(
        model, model.mesh.elements[element_id], displacement,
        target_surface.side) for element_id in target_surface.element_ids]
    lower = np.inf
    retained = 0
    visited = 0
    scale = max(1.0, *(float(np.max(np.abs(face)))
                       for face in (*sources, *targets)))
    for source in sources:
        for target in targets:
            xy = np.column_stack((target[1, :2]-target[0, :2],
                                  target[2, :2]-target[0, :2]))
            inverse = np.linalg.inv(xy)
            condition = (np.linalg.norm(xy, ord=np.inf)
                         * np.linalg.norm(inverse, ord=np.inf))
            if not np.isfinite(condition) or condition > 1.0e6:
                raise RuntimeError("independent gap XY projection is ill-conditioned")
            roundoff = 256*np.finfo(float).eps*condition*scale
            bary_roundoff = (128*np.finfo(float).eps
                             *(1.0+np.linalg.norm(inverse, ord=np.inf)*scale))
            normal = np.cross(target[1]-target[0], target[2]-target[0])
            normal /= np.linalg.norm(normal)
            if normal[2] <= 0.5:
                raise RuntimeError("independent target-normal direction is invalid")
            boxes = [(0.0, 1.0, 0.0, 1.0, 0)]
            while boxes:
                a0, a1, b0, b1, depth = boxes.pop()
                visited += 1
                if visited > 100_000:
                    raise RuntimeError("independent gap box budget exhausted")
                parameters = ((a0, b0), (a1, b0), (a1, b1), (a0, b1))
                corners = np.asarray([
                    (1-a)*(1-b)*source[0]+a*(1-b)*source[1]
                    +a*b*source[2]+(1-a)*b*source[3]
                    for a, b in parameters
                ])
                transverse = (corners[:, :2]-target[0, :2]) @ inverse.T
                barycentric = np.column_stack((
                    1.0-transverse.sum(axis=1), transverse,
                ))
                if np.any(np.max(barycentric, axis=0) < -bary_roundoff):
                    continue
                corner_gap = (corners-target[0]) @ normal
                local_lower = float(np.min(corner_gap)-roundoff)
                interior = np.all(barycentric >= bary_roundoff, axis=1)
                if np.any(interior & (corner_gap < -gap_budget-roundoff)):
                    raise RuntimeError("independent projected gap violates budget")
                if local_lower >= -gap_budget:
                    lower = min(lower, local_lower)
                    retained += 1
                    continue
                if np.all(np.min(barycentric, axis=0) >= bary_roundoff):
                    raise RuntimeError(
                        f"independent projected gap violates budget: {local_lower} m"
                    )
                if depth >= 14:
                    raise RuntimeError("independent gap unresolved at overlap edge")
                amid = 0.5*(a0+a1)
                bmid = 0.5*(b0+b1)
                boxes.extend((
                    (a0, amid, b0, bmid, depth+1),
                    (amid, a1, b0, bmid, depth+1),
                    (a0, amid, bmid, b1, depth+1),
                    (amid, a1, bmid, b1, depth+1),
                ))
    if retained == 0:
        raise RuntimeError("independent projected gap has no represented overlap")
    return float(lower)


def run() -> dict:
    model, definition, settings, program = build_case()
    capability = validate_general_contact(model, definition, settings)
    if not capability.valid:
        raise RuntimeError("unsupported contact: " + "; ".join(capability.issues))
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
        record_increment_snapshots=True,
    )
    if result.status != "completed" or len(result.steps) != 8:
        raise RuntimeError(
            f"contact stopped after {len(result.steps)} steps: "
            f"{result.status}: {result.failure_reason}"
        )
    history = result.info["force_displacement_history"]
    records = []
    if len(result.snapshots) != len(history):
        raise RuntimeError("accepted contact pose snapshots are incomplete")
    for index, (row, snapshot) in enumerate(zip(
            history, result.snapshots, strict=True)):
        contact = row["contact"]
        source = np.asarray(contact["integration_force"], dtype=float)
        target = np.asarray(contact["target_force"], dtype=float)
        source_moment = np.asarray(contact["integration_moment"], dtype=float)
        target_moment = np.asarray(contact["target_moment"], dtype=float)
        force_balance = float(np.linalg.norm(source+target))
        moment_balance = float(np.linalg.norm(source_moment+target_moment))
        complementarity = float(row["complementarity_relative_force"])
        if (not contact["gap_tolerance_met"]
                or complementarity > 1e-8
                or force_balance > 1e-9
                or moment_balance > 1e-9):
            raise RuntimeError(f"contact gate failed at step {index+1}")
        independent_gap = _independent_projected_gap_bound(
            model, definition, snapshot.displacements,
            definition.interactions[0].gap_tolerance,
        )
        if independent_gap < -definition.interactions[0].gap_tolerance:
            raise RuntimeError(f"independent gap failed at step {index+1}")
        records.append({
            "step": int(row["step_index"]),
            "normal_force_n": float(source[2]),
            "minimum_gap_m": float(contact["minimum_gap"]),
            "independent_projected_minimum_gap_m": independent_gap,
            "active_pressure_modes": int(contact["active_pressure_dofs"]),
            "complementarity_relative_force": complementarity,
            "pair_force_balance_n": force_balance,
            "pair_moment_balance_nm": moment_balance,
        })
    if not (records[1]["normal_force_n"] > 0.0
            and records[5]["active_pressure_modes"] == 0
            and records[5]["normal_force_n"] == 0.0
            and records[6]["normal_force_n"] > 0.0):
        raise RuntimeError("expected press, opening and recontact sequence missing")
    if result.restart_checkpoint is None:
        raise RuntimeError("contact checkpoint missing")
    open_model, open_definition, open_settings, open_program = build_case()
    opened = solve_static_nonlinear(
        open_model, contact=open_definition, contact_settings=open_settings,
        contact_path_program=open_program, max_iterations=18,
        max_load_factor=0.75, emit_restart_checkpoint=True,
    )
    if (opened.status != "completed" or len(opened.steps) != 6
            or opened.restart_checkpoint is None
            or opened.info["force_displacement_history"][-1]["contact"][
                "active_pressure_dofs"] != 0):
        raise RuntimeError("open-state checkpoint was not accepted")
    resumed_model, resumed_definition, resumed_settings, resumed_program = build_case()
    resumed = solve_static_nonlinear(
        resumed_model, contact=resumed_definition,
        contact_settings=resumed_settings,
        contact_path_program=resumed_program, max_iterations=18,
        restart_checkpoint=opened.to_restart_checkpoint(),
    )
    # Restart results include the six checkpointed steps and two new steps.
    if resumed.status != "completed" or len(resumed.steps) != 8:
        raise RuntimeError(
            f"split restart did not complete recontact: status={resumed.status}, "
            f"steps={len(resumed.steps)}, reason={resumed.failure_reason}"
        )
    displacement_error = float(np.max(np.abs(
        resumed.displacements-result.displacements)))
    full_force = np.asarray(result.info["general_contact"]["integration_force"])
    resumed_force = np.asarray(resumed.info["general_contact"]["integration_force"])
    force_error = float(np.max(np.abs(resumed_force-full_force)))
    if displacement_error > 1.0e-10 or force_error > 1.0e-10:
        raise RuntimeError("split restart differs from uninterrupted contact path")
    return {
        "status": result.status,
        "capability_profile": capability.profile,
        "qualification_status": "development_only",
        "source_q4_faces": 4,
        "deformable_target_s3_faces": 2,
        "gap_tolerance_m": definition.interactions[0].gap_tolerance,
        "has_restart_checkpoint": True,
        "split_restart_verified": True,
        "restart_maximum_displacement_difference": displacement_error,
        "restart_maximum_contact_force_difference_n": force_error,
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "steps": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
