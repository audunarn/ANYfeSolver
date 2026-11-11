"""Development probe for mesh-consistent finite-area Q4/S3 contact loading.

This case separates response refinement from the left-edge displacement fixture.
Its source patch, target, distributed load and supported edge occupy the same
physical region at every mesh size. Results are exploratory, not qualification.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import anymaterial
import anyfesolver

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactSolverSettings, ContactSurface, LoadCase, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def build_case(divisions: int, *, total_load_n: float = 0.4,
               normal_stiffness: float = 1000.0,
               supported_schur: bool = False,
               geometry_profile: str = "overhang",
               schur_mode_cap: int = 16,
               rows: int | None = None):
    if divisions < 2 or divisions % 2:
        raise ValueError("divisions must be an even integer of at least two")
    rows = divisions if rows is None else rows
    if type(rows) is not int or rows < 2 or rows % 2:
        raise ValueError("rows must be an even integer of at least two")
    if geometry_profile not in {"overhang", "central"}:
        raise ValueError("geometry_profile must be overhang or central")
    model = FEModel(
        f"general_contact_finite_area_{divisions}"
        if rows == divisions else f"general_contact_finite_area_{divisions}x{rows}"
    )
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    stride = divisions + 1
    source_nodes = tuple(range(1, (rows+1)*stride + 1))
    for row in range(rows+1):
        for column in range(stride):
            model.add_node(1 + row*stride + column,
                           (0.1 if geometry_profile == "overhang" else 0.0)
                           + column/divisions,
                           (0.1 if geometry_profile == "overhang" else 0.0)
                           + row/rows,
                           0.04005)
    target_nodes = tuple(range((rows+1)*stride + 1,
                               (rows+1)*stride + 5))
    target_corners = (
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        if geometry_profile == "overhang" else
        ((0.24, 0.21), (0.77, 0.21), (0.77, 0.73), (0.24, 0.73))
    )
    for node, (x, y) in zip(target_nodes, target_corners):
        model.add_node(node, x, y, 0.0)

    source_elements = []
    load_weights = {node: 0.0 for node in source_nodes}
    for row in range(rows):
        for column in range(divisions):
            element_id = 1 + row*divisions + column
            first = 1 + row*stride + column
            nodes = (first, first+1, first+stride+1, first+stride)
            model.add_element(element_id, create_shell_element(
                element_id, nodes, "elastic", thickness=0.05,
                reference_normal=(0.0, 0.0, 1.0),
            ))
            source_elements.append(element_id)
            for node in nodes:
                load_weights[node] += 0.25/(divisions*rows)
    for element_id, nodes in (
        (divisions*rows+1, (target_nodes[0], target_nodes[1], target_nodes[3])),
        (divisions*rows+2, (target_nodes[1], target_nodes[2], target_nodes[3])),
    ):
        model.add_element(element_id, create_shell_element(
            element_id, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0.0, 0.0, 1.0),
            reference_surface_offset=0.01,
        ))

    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", target_nodes,
        {axis: 0.0 for axis in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", source_nodes,
        {axis: 0.0 for axis in ("ux", "uy", "rx", "ry", "rz")},
    ))
    supported_edge = (
        tuple(1 + row*stride for row in range(rows+1))
        if geometry_profile == "overhang" else
        tuple(1 + row*stride + column
              for row in range(rows+1) for column in range(stride)
              if row in {0, rows} or column in {0, divisions})
    )
    model.add_boundary_condition(BoundaryCondition(
        "source_supported_edge", supported_edge, {"uz": 0.0},
    ))
    load = LoadCase("uniform_source_pressure")
    for node, weight in load_weights.items():
        load.add_nodal_load(node, forces=(0.0, 0.0, -total_load_n*weight))
    definition = ContactDefinition(
        surfaces=(ContactSurface("q4_source", tuple(source_elements), "bottom"),
                  ContactSurface("s3_target", (divisions*rows+1, divisions*rows+2),
                                 "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "q4_source", "s3_target", normal_stiffness, 5.0e-5,
            normal_law="augmented",
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        q4_source_gap_certificate=(
            "whole_face" if geometry_profile == "overhang"
            else "projected_matching"
        ),
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=5000*divisions*rows,
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=128,
        pressure_tolerance=1.0e-5,
        maximum_augmentations=7,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
    )
    if supported_schur:
        settings = replace(
            settings, augmentation_acceleration="supported_schur_dev",
            augmentation_schur_maximum_modes=schur_mode_cap,
            augmentation_schur_maximum_pressure_pa=200.0,
            augmentation_schur_maximum_step_pa=100.0,
            augmentation_schur_difference_step_m=2.0e-6,
            augmentation_schur_maximum_attempts=2,
            augmentation_schur_maximum_relinearizations=0,
        )
    return model, load, definition, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("divisions", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--normal-stiffness", type=float, default=1000.0)
    parser.add_argument("--supported-schur", action="store_true")
    parser.add_argument("--min-step-fraction", type=float, default=0.5)
    parser.add_argument("--geometry-profile", choices=("overhang", "central"),
                        default="overhang")
    parser.add_argument("--total-load", type=float, default=0.4)
    parser.add_argument("--schur-mode-cap", type=int, default=16)
    parser.add_argument("--max-load-factor", type=float, default=1.0)
    args = parser.parse_args()
    model, load, definition, settings = build_case(
        args.divisions, total_load_n=args.total_load,
        normal_stiffness=args.normal_stiffness,
        supported_schur=args.supported_schur,
        geometry_profile=args.geometry_profile,
        schur_mode_cap=args.schur_mode_cap,
        rows=args.rows,
    )
    capability = validate_general_contact(model, definition, settings)
    if not capability.valid:
        raise ValueError(f"invalid contact model: {capability.issues}")
    start = time.perf_counter()
    result = solve_static_nonlinear(
        model, load, contact=definition, contact_settings=settings,
        control="force", kinematics="von_karman", num_steps=1,
        max_iterations=12, min_step_fraction=args.min_step_fraction,
        max_load_factor=args.max_load_factor,
        emit_restart_checkpoint=True,
    )
    record = {
        "divisions": args.divisions,
        "rows": args.divisions if args.rows is None else args.rows,
        "total_load_n": args.total_load,
        "geometry_profile": args.geometry_profile,
        "normal_law": "augmented",
        "normal_stiffness": args.normal_stiffness,
        "supported_schur": args.supported_schur,
        "min_step_fraction": args.min_step_fraction,
        "schur_mode_cap": args.schur_mode_cap,
        "max_load_factor": args.max_load_factor,
        "elapsed_seconds": time.perf_counter() - start,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "contact_failure": result.info.get("contact_failure"),
        "force_displacement_history": result.info.get("force_displacement_history"),
        "general_contact": result.info.get("general_contact"),
        "solver_info": result.info,
        "has_restart_checkpoint": result.restart_checkpoint is not None,
        "final_displacements": result.displacements.tolist(),
        "runtime_identity": {
            "python": sys.version.split()[0],
            "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
            "anymaterial_module": str(Path(anymaterial.__file__).resolve()),
        },
    }
    args.output.write_text(json.dumps(record, indent=2, default=str) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("contact_failure", "force_displacement_history",
                                     "general_contact", "solver_info")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
