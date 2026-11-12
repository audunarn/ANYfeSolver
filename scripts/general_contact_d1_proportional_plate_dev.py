"""Parked development control: equal clamped plates share a load equally.

In the matching linear continuum problem, both plates have the same operator
and zero initial gap. The solution w_source = w_target has p = q / 2 for any
compact transverse load q. Discrete Q4/S3 differences are measured here;
the identity is not presumed to hold on a coarse nonmatching mesh. The present
exactly coincident-boundary fixture stops before mechanics recovery with a
shared-edge derivative tie, including with the explicit tie setting. Retain
this reproducer without treating it as an accepted analytical benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction, ContactSurface,
    ContactSolverSettings, LoadCase, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from scripts.general_contact_d1_interior_press_dev import (
    _physical_bump_axis_weights,
)


def build_case(source_cells: int, target_cells: int,
               total_force_n: float):
    if source_cells not in (4, 8) or target_cells not in (3, 4, 6, 8):
        raise ValueError("bounded plate control uses 4/8 Q4 and 3/4/6/8 S3")
    model = FEModel("d1_equal_plate_partition_development")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    source_ids = {}
    target_ids = {}
    next_node = 1
    for ids, cells, z in ((source_ids, source_cells, 0.05),
                          (target_ids, target_cells, 0.0)):
        for row in range(cells+1):
            for column in range(cells+1):
                ids[row, column] = next_node
                model.add_node(next_node, column/cells, row/cells, z)
                next_node += 1
    source_faces = []
    target_faces = []
    next_element = 1
    for row in range(source_cells):
        for column in range(source_cells):
            nodes = (source_ids[row, column],
                     source_ids[row, column+1],
                     source_ids[row+1, column+1],
                     source_ids[row+1, column])
            model.add_element(next_element, create_shell_element(
                next_element, nodes, "elastic", thickness=0.05,
                reference_normal=(0.0, 0.0, 1.0)))
            source_faces.append(next_element)
            next_element += 1
    for row in range(target_cells):
        for column in range(target_cells):
            for nodes in ((target_ids[row, column],
                           target_ids[row, column+1],
                           target_ids[row+1, column]),
                          (target_ids[row, column+1],
                           target_ids[row+1, column+1],
                           target_ids[row+1, column])):
                model.add_element(next_element, create_shell_element(
                    next_element, nodes, "elastic", thickness=0.05,
                    formulation="e4-pl-s3-v2d",
                    reference_normal=(0.0, 0.0, 1.0),
                    reference_surface_offset=0.0))
                target_faces.append(next_element)
                next_element += 1
    fixed = {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")}
    for name, ids, cells in (("source_rim_clamp", source_ids, source_cells),
                             ("target_rim_clamp", target_ids, target_cells)):
        rim = tuple(node for (row, column), node in ids.items()
                    if row in (0, cells) or column in (0, cells))
        model.add_boundary_condition(BoundaryCondition(name, rim, fixed))
    x_weights = _physical_bump_axis_weights(source_cells)
    y_weights = _physical_bump_axis_weights(source_cells)
    load = LoadCase("compact_equal_plate_load")
    for (row, column), node in source_ids.items():
        weight = float(x_weights[column]*y_weights[row])
        if weight > 1.0e-15:
            load.add_nodal_load(node, forces=(0.0, 0.0,
                                              -total_force_n*weight))
    definition = ContactDefinition(
        surfaces=(ContactSurface("q4_bottom", tuple(source_faces), "bottom"),
                  ContactSurface("s3_top", tuple(target_faces), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "equal_plate_pair", "q4_bottom", "s3_top", 1.0e4, 5.0e-5,
            normal_law="augmented"),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        q4_source_gap_certificate="projected_matching",
        q4_s3_shared_edge_tie_step_m=3.0e-8,
        q4_s3_shared_edge_tie_direction=(1.0, 0.0, 0.0),
        q4_s3_shared_edge_tie_relative_tolerance=1.0e-6,
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=14144,
        general_s3_differentiation="jet2_q4_s3_dev",
        maximum_candidate_pairs=max(32, len(source_faces)*len(target_faces)),
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=300,
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        enforcement_solver="monolithic_active_set",
        complementarity_force_scale_n=total_force_n,
        pressure_tolerance=1.0e-5, maximum_augmentations=7,
        augmentation_acceleration="none",
        d1_interior_profile=True,
        d1_interior_coverage_tolerance_m2=1.0e-10,
        d1_interior_boundary_force_tolerance_n=1.0e-10,
    )
    return model, load, definition, settings, source_ids


def main(output: Path, *, source_cells: int, target_cells: int,
         total_force_n: float):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    model, load, definition, settings, source_ids = build_case(
        source_cells, target_cells, total_force_n)
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"fixture capability rejected: {report.issues}")
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        kinematics="von_karman", num_steps=2, max_iterations=18,
    )
    contact = result.info.get("general_contact", {})
    pressure = dict(zip(contact.get("source_pressure_node_ids", ()),
                        contact.get("numerical_pressure_pa", ())))
    rim = {node for (row, column), node in source_ids.items()
           if row in (0, source_cells) or column in (0, source_cells)}
    normal_force = (float(contact["integration_force"][2])
                    if "integration_force" in contact else None)
    payload = {
        "candidate": "equal_clamped_plate_partition_development",
        "source_cells": source_cells, "target_cells": target_cells,
        "total_applied_force_n": total_force_n,
        "exact_continuum_contact_force_n": total_force_n/2,
        "status": result.status, "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "normal_contact_force_n": normal_force,
        "relative_analytical_force_error": (
            abs(normal_force-total_force_n/2)/(total_force_n/2)
            if normal_force is not None else None),
        "minimum_gap_m": contact.get("minimum_gap"),
        "sampled_overlap_gap_m": contact.get("sampled_overlap_gap"),
        "active_pressure_dofs": contact.get("active_pressure_dofs"),
        "rim_pressure_pa": {str(node): value for node, value in pressure.items()
                             if node in rim and value > 0.0},
        "contact": contact,
        "elapsed_seconds": time.perf_counter()-started,
    }
    output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "failure_reason", "accepted_steps",
        "normal_contact_force_n", "relative_analytical_force_error",
        "minimum_gap_m", "active_pressure_dofs", "rim_pressure_pa",
        "elapsed_seconds",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-cells", type=int, choices=(4, 8), default=4)
    parser.add_argument("--target-cells", type=int, choices=(3, 4, 6, 8),
                        default=3)
    parser.add_argument("--force-n", type=float, default=0.04)
    args = parser.parse_args()
    main(args.output, source_cells=args.source_cells,
         target_cells=args.target_cells, total_force_n=args.force_n)
