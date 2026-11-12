"""Bounded Q4 bending-direction mixed-contact press refinement probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import time

import anymaterial
import anyfesolver
import numpy as np

from anyfesolver import (
    BoundaryCondition, ContactDefinition, ContactInteraction,
    ContactPathControl, ContactPathNodalForce, ContactPathProgram,
    ContactSolverSettings, ContactSurface, create_shell_element,
    solve_static_nonlinear, validate_general_contact,
)
from anyfesolver.fe_core import FEModel


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _peak_process_memory_mb():
    """Optional development measurement; solver dependencies do not change."""
    try:
        import psutil
    except ImportError:
        return None
    usage = psutil.Process().memory_info()
    return float(getattr(usage, "peak_wset", usage.rss))/1.0e6


def build_case(columns: int, *, four_phase: bool = False,
               steps_per_segment: int = 2,
               load_profile: str = "tip",
               total_force_n: float = 0.1,
               target_right_x_m: float = 1.0,
               target_reference_slope: float = 0.0,
               common_reference_slope: float = 0.0,
               target_columns: int = 1,
               target_rows: int = 1,
               target_young_multiplier: float = 1.0,
               full_width_target: bool = False,
               full_width_target_y0_m: float = -0.1,
               source_rows: int = 1,
               source_y_profile: str | None = None,
               offline_structural_reference_64: bool = False,
               offline_structural_reference_y16: bool = False,
               offline_structural_reference_y24: bool = False,
               offline_structural_reference_y32: bool = False,
               guided_radial_target_refinement: bool = False):
    if columns not in (4, 6, 8, 12, 16, 24, 32) and not (
            columns == 64 and offline_structural_reference_64):
        raise ValueError(
            "only bounded 4/6/8/12/16/24/32-column Q4 refinements are admitted; "
            "64 columns require the offline structural-reference opt-in")
    if target_right_x_m not in (1.0, 1.3, 1.297):
        raise ValueError("unsupported bounded target right edge")
    if full_width_target_y0_m not in (-0.1, -0.113) or (
            not full_width_target and full_width_target_y0_m != -0.1):
        raise ValueError("unsupported full-width target lower edge")
    if not 0.0 <= target_reference_slope <= 1.0e-4:
        raise ValueError("target reference slope must be in [0, 1e-4]")
    if not 0.0 <= common_reference_slope <= 0.1 or (
            common_reference_slope and target_reference_slope):
        raise ValueError("common slope must be in [0, 0.1] without target-only slope")
    radial_target = (guided_radial_target_refinement and columns == source_rows == 12
                     and target_columns == target_rows == 12 and full_width_target
                     and common_reference_slope == target_reference_slope == 0.0)
    if guided_radial_target_refinement and not radial_target:
        raise ValueError("radial target opt-in is limited to the flat 12/12 fixture")
    if target_right_x_m == 1.297 and not radial_target:
        raise ValueError("1.297 m right edge is limited to the radial 12/12 fixture")
    if not radial_target and (target_columns not in (1, 2, 3, 4, 6)
                              or target_rows not in (1, 2, 3, 4, 6)):
        raise ValueError("this development case admits 1/2/3/4/6 target rows and columns")
    if target_young_multiplier not in (1.0, 5.0):
        raise ValueError("development target contrast must be one or five")
    if source_rows not in (1, 2, 4, 6, 8, 12) and not (
            source_rows == 16 and offline_structural_reference_y16) and not (
            source_rows == columns == 24 and offline_structural_reference_y24) and not (
            source_rows == columns == 32 and offline_structural_reference_y32):
        raise ValueError("this development case admits at most twelve source rows; "
                         "larger grids require the offline structural-reference opt-in")
    if source_y_profile not in (None, "rim8", "near_rim8", "near_rim8_ref16",
                                "uniform_ref16", "uniform_ref24", "uniform_ref32"):
        raise ValueError("unknown bounded source y profile")
    if source_y_profile == "uniform_ref16":
        if (not offline_structural_reference_y16 or source_rows != 16
                or columns not in (8, 16)):
            raise ValueError("uniform sixteen-row reference is offline-only at eight or sixteen columns")
    elif source_y_profile == "uniform_ref24":
        if (not offline_structural_reference_y24 or source_rows != 24
                or columns != 24):
            raise ValueError("uniform twenty-four-row reference is offline-only")
    elif source_y_profile == "uniform_ref32":
        if (not offline_structural_reference_y32 or source_rows != 32
                or columns != 32):
            raise ValueError("uniform thirty-two-row reference is offline-only")
    elif source_y_profile == "near_rim8_ref16":
        if (not offline_structural_reference_y16 or source_rows != 16
                or columns != 16):
            raise ValueError("near-rim sixteen-row reference is offline-only")
    elif source_rows in (16, 24, 32):
        raise ValueError("large source rows require an explicit offline profile")
    elif source_y_profile is not None and source_rows != 8:
        raise ValueError("rim profiles require eight source rows")
    if source_y_profile in (None, "uniform_ref16", "uniform_ref24", "uniform_ref32"):
        source_y_knots = np.linspace(0.0, 1.0, source_rows+1)
    else:
        coarse_y = np.asarray((0.0, 0.2, 0.4, 0.6, 0.75,
                               0.825, 0.875,
                               0.9 if source_y_profile == "rim8" else 0.895,
                               1.0))
        if source_y_profile == "near_rim8_ref16":
            source_y_knots = np.empty(17)
            source_y_knots[::2] = coarse_y
            source_y_knots[1::2] = 0.5*(coarse_y[:-1]+coarse_y[1:])
        else:
            source_y_knots = coarse_y
    source_y_weights = np.empty(source_rows+1)
    heights = np.diff(source_y_knots)
    source_y_weights[0] = heights[0]/2
    source_y_weights[-1] = heights[-1]/2
    source_y_weights[1:-1] = (heights[:-1]+heights[1:])/2
    target_y0, target_y1 = ((full_width_target_y0_m, 1.2) if full_width_target
                            else (0.0, 1.0))
    stride = columns+1
    source_element_count = columns*source_rows
    source_nodes = tuple(range(1, (source_rows+1)*stride+1))
    target_nodes = tuple(range((source_rows+1)*stride+1,
                               (source_rows+1)*stride+1
                               +(target_rows+1)*(target_columns+1)))
    model = FEModel(f"mixed_{columns}_q4_x_jet2_schur_press")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    if target_young_multiplier != 1.0:
        model.add_material("elastic_target", 1.0e5*target_young_multiplier,
                           0.3, density=1.0)
    common_normal = np.asarray((-common_reference_slope, 0.0, 1.0))
    common_normal /= np.linalg.norm(common_normal)
    source_reference_height = 0.04004/common_normal[2]
    for row in range(source_rows+1):
        y = 0.1+source_y_knots[row]
        for column in range(stride):
            x = 0.1+column/columns
            model.add_node(1+row*stride+column,
                           x, y, source_reference_height
                           +common_reference_slope*(x-0.5*target_right_x_m))
    target_positions = (
        ((0.0, target_y0, 0.0), (target_right_x_m, target_y0, 0.0),
         (target_right_x_m, target_y1, 0.0), (0.0, target_y1, 0.0))
        if target_columns == 1 and target_rows == 1 else tuple(
            (target_right_x_m*column/target_columns,
             target_y0+(target_y1-target_y0)*row/target_rows, 0.0)
            for row in range(target_rows+1)
            for column in range(target_columns+1)))
    for node, position in zip(target_nodes, target_positions):
        x, y, z = position
        model.add_node(node, x, y,
                       z+(target_reference_slope+common_reference_slope)
                       *(x-0.5*target_right_x_m))
    target_normal = np.asarray((-(target_reference_slope+common_reference_slope),
                                0.0, 1.0))
    target_normal /= np.linalg.norm(target_normal)
    for row in range(source_rows):
        for column in range(columns):
            element_id = 1+row*columns+column
            bottom = 1+row*stride+column
            model.add_element(element_id, create_shell_element(
                element_id,
                (bottom, bottom+1, bottom+stride+1, bottom+stride),
                "elastic", thickness=0.05,
                reference_normal=tuple(common_normal),
            ))
    if target_columns == 1 and target_rows == 1:
        target_faces = ((target_nodes[0], target_nodes[1], target_nodes[3]),
                        (target_nodes[1], target_nodes[2], target_nodes[3]))
    else:
        target_faces = tuple(
            nodes for row in range(target_rows)
            for column in range(target_columns)
            for base in (row*(target_columns+1)+column,)
            for nodes in (
                (target_nodes[base], target_nodes[base+1],
                 target_nodes[base+target_columns+1]),
                (target_nodes[base+1],
                 target_nodes[base+target_columns+2],
                 target_nodes[base+target_columns+1]),
            ))
    for element_id, nodes in enumerate(target_faces,
                                       start=source_element_count+1):
        model.add_element(element_id, create_shell_element(
            element_id, nodes,
            "elastic" if target_young_multiplier == 1.0 else "elastic_target",
            thickness=0.05,
            formulation="e4-pl-s3-v2d",
            reference_normal=tuple(target_normal),
            reference_surface_offset=0.01,
        ))
    model.add_boundary_condition(BoundaryCondition(
        "target_fixed", target_nodes,
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_guide", source_nodes,
        {name: 0.0 for name in ("uy", "rx", "ry", "rz")},
    ))
    model.add_boundary_condition(BoundaryCondition(
        "source_left_support",
        tuple(1+row*stride for row in range(source_rows+1)),
        {"uz": 0.0},
    ))
    definition = ContactDefinition(
        surfaces=(
            ContactSurface("q4_source",
                           tuple(range(1, source_element_count+1)), "bottom"),
            ContactSurface("s3_target",
                           tuple(range(source_element_count+1,
                                       source_element_count+1+len(target_faces))), "top"),
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
        maximum_candidate_pairs=max(32, source_element_count*len(target_faces)),
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=300,
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        pressure_tolerance=1.0e-5, maximum_augmentations=7,
        augmentation_acceleration="supported_schur_dev",
        augmentation_schur_maximum_modes=(source_rows+1)*stride,
        augmentation_schur_maximum_pressure_pa=10.0,
        augmentation_schur_maximum_step_pa=5.0,
        augmentation_schur_difference_step_m=2.0e-6,
        augmentation_schur_maximum_attempts=2,
        augmentation_schur_maximum_relinearizations=0,
    )
    waypoints = ((0.0, 0.0), (0.0, 0.0))
    force_factors = (0.0, 1.0)
    if four_phase:
        waypoints += ((0.05, 0.05), (0.05, 0.05), (0.05, 0.05))
        force_factors += (1.0, 0.0, 1.0)
    if load_profile == "tip":
        nodal_weights = {
            (row+1)*stride:
            source_y_weights[row]
            for row in range(source_rows+1)
        }
    elif load_profile == "distributed":
        nodal_weights = {
            1+row*stride+column:
            ((0.5 if column in (0, columns) else 1.0)/columns)
            *source_y_weights[row]
            for row in range(source_rows+1) for column in range(stride)
        }
    else:
        raise ValueError("load profile must be tip or distributed")
    if total_force_n <= 0.0:
        raise ValueError("total downward force magnitude must be positive")
    program = ContactPathProgram(
        controls=(ContactPathControl.node(1, "ux"),
                  ContactPathControl.node(stride+1, "ux")),
        waypoints=waypoints, steps_per_segment=steps_per_segment,
        nodal_forces=tuple(ContactPathNodalForce(
            node, "uz", tuple(-total_force_n*weight*factor
                              for factor in force_factors),
        ) for node, weight in nodal_weights.items()),
    )
    return model, definition, settings, program


def main(columns: int, output: Path, pressure_cap_pa: float = 10.0,
         four_phase: bool = False, steps_per_segment: int = 2,
         schur_step_cap_pa: float = 5.0,
         load_profile: str = "tip", total_force_n: float = 0.1,
         schur_relinearizations: int = 0,
         target_right_x_m: float = 1.0,
         schur_maximum_attempts: int = 2,
         pressure_basis: str = "fine_nodal_q1",
         target_reference_slope: float = 0.0,
         target_columns: int = 1,
         full_width_target: bool = False,
         source_rows: int = 1,
         source_y_profile: str | None = None,
         enforcement_solver: str = "nested_augmented",
         monolithic_maximum_pivots: int = 12,
         monolithic_linear_solver: str = "direct",
         emit_restart_checkpoint: bool = True) -> None:
    if output.exists():
        raise FileExistsError(f"development output already exists: {output}")
    model, definition, settings, program = build_case(
        columns, four_phase=four_phase,
        steps_per_segment=steps_per_segment, load_profile=load_profile,
        total_force_n=total_force_n,
        target_right_x_m=target_right_x_m,
        target_reference_slope=target_reference_slope,
        target_columns=target_columns,
        full_width_target=full_width_target,
        source_rows=source_rows,
        source_y_profile=source_y_profile,
    )
    if pressure_cap_pa <= 0.0 or schur_step_cap_pa <= 0.0:
        raise ValueError("Schur pressure and step caps must be positive")
    if schur_relinearizations not in (0, 1):
        raise ValueError("this probe admits zero or one Schur relinearization")
    if schur_maximum_attempts not in (2, 4):
        raise ValueError("this probe admits two or four Schur proposals")
    settings = replace(settings,
                       augmentation_schur_maximum_pressure_pa=pressure_cap_pa,
                       augmentation_schur_maximum_step_pa=schur_step_cap_pa,
                       augmentation_schur_maximum_relinearizations=(
                           schur_relinearizations),
                       augmentation_schur_maximum_attempts=(
                           schur_maximum_attempts),
                       q4_source_pressure_basis=pressure_basis)
    if enforcement_solver == "monolithic_active_set":
        settings = replace(
            settings, enforcement_solver=enforcement_solver,
            complementarity_force_scale_n=total_force_n,
            monolithic_maximum_pivots=monolithic_maximum_pivots,
            monolithic_linear_solver=monolithic_linear_solver,
            augmentation_acceleration="none",
            augmentation_schur_maximum_modes=None,
            augmentation_schur_maximum_pressure_pa=None,
            augmentation_schur_maximum_step_pa=None,
            augmentation_schur_difference_step_m=None,
            augmentation_schur_maximum_attempts=1,
            augmentation_schur_maximum_relinearizations=0,
        )
    capability = validate_general_contact(model, definition, settings)
    if not capability.valid:
        raise ValueError(f"unsupported refinement: {capability.issues}")
    started = time.perf_counter()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=emit_restart_checkpoint,
    )
    derivative_module = importlib.import_module(
        "anyfesolver.general_contact.q4_s3_overlap_derivatives"
    )
    record = {
        "candidate": ("mixed_bending_refinement_jet2_active_set_dev"
                      if enforcement_solver == "monolithic_active_set" else
                      "mixed_bending_refinement_jet2_schur_four_phase_dev"
                      if four_phase else
                      "mixed_bending_refinement_jet2_schur_press_dev"),
        "enforcement_solver": enforcement_solver,
        "monolithic_maximum_pivots": monolithic_maximum_pivots,
        "monolithic_linear_solver": monolithic_linear_solver,
        "checkpoint_requested": emit_restart_checkpoint,
        "four_phase": four_phase,
        "steps_per_segment": steps_per_segment,
        "load_profile": load_profile,
        "total_downward_force_n": total_force_n,
        "target_right_x_m": target_right_x_m,
        "target_reference_slope": target_reference_slope,
        "target_columns": target_columns,
        "full_width_target": full_width_target,
        "columns": columns,
        "source_rows": source_rows,
        "source_y_profile": source_y_profile,
        "pressure_basis": pressure_basis,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter()-started,
        "peak_process_memory_mb": _peak_process_memory_mb(),
        "profile": capability.profile,
        "gap_tolerance_m": definition.interactions[0].gap_tolerance,
        "normal_regularization_pa_per_m": definition.interactions[0].normal_stiffness,
        "pressure_tolerance_pa": settings.pressure_tolerance,
        "maximum_augmentations": settings.maximum_augmentations,
        "maximum_candidate_pairs": settings.maximum_candidate_pairs,
        "supported_pressure_mode_cap": settings.augmentation_schur_maximum_modes,
        "schur_maximum_pressure_pa": pressure_cap_pa,
        "schur_maximum_step_pa": schur_step_cap_pa,
        "schur_maximum_relinearizations": schur_relinearizations,
        "schur_maximum_attempts": schur_maximum_attempts,
        "has_checkpoint": result.restart_checkpoint is not None,
        "contact_failure": result.info.get("contact_failure"),
        "contact": result.info.get("general_contact"),
        "history": result.info.get("force_displacement_history"),
        "support_vertical_reactions_n": [
            sum(float(values[2]) for values in step.support_reactions.values())
            for step in result.steps
        ],
        "support_reactions": [step.support_reactions for step in result.steps],
        "augmentation_attempts": result.info.get("contact_augmentation_attempts"),
        "final_displacements": result.displacements.tolist(),
        "runtime_identity": {
            "python": sys.version.split()[0],
            "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
            "anymaterial_module": str(Path(anymaterial.__file__).resolve()),
            "compiled_bjet_mul_enabled": bool(
                derivative_module._COMPILED_BJET_MUL_ENABLED
            ),
        },
    }
    output.write_text(json.dumps(record, indent=2, default=_json_default) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "columns", "status", "failure_reason", "accepted_steps",
        "elapsed_seconds", "has_checkpoint",
    )}, indent=2, default=_json_default))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("columns", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pressure-cap-pa", type=float, default=10.0)
    parser.add_argument("--four-phase", action="store_true")
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--schur-step-cap-pa", type=float, default=5.0)
    parser.add_argument("--load-profile", choices=("tip", "distributed"),
                        default="tip")
    parser.add_argument("--total-force-n", type=float, default=0.1)
    parser.add_argument("--schur-relinearizations", type=int, default=0)
    parser.add_argument("--target-right-x-m", type=float, default=1.0)
    parser.add_argument("--schur-maximum-attempts", type=int, default=2)
    parser.add_argument("--pressure-basis", choices=("fine_nodal_q1", "macro_q1_pair_dev"),
                        default="fine_nodal_q1")
    parser.add_argument("--target-reference-slope", type=float, default=0.0)
    parser.add_argument("--target-columns", type=int, choices=(1, 2), default=1)
    parser.add_argument("--full-width-target", action="store_true")
    parser.add_argument("--source-rows", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--source-y-profile", choices=("rim8", "near_rim8"))
    parser.add_argument("--enforcement-solver", choices=(
        "nested_augmented", "monolithic_active_set"),
        default="nested_augmented")
    parser.add_argument("--monolithic-maximum-pivots", type=int, default=12)
    parser.add_argument("--monolithic-linear-solver", choices=(
        "direct", "block_response_dev"), default="direct")
    parser.add_argument("--no-checkpoint", dest="emit_restart_checkpoint",
                        action="store_false", default=True)
    args = parser.parse_args()
    main(args.columns, args.output, args.pressure_cap_pa, args.four_phase,
         args.steps_per_segment, args.schur_step_cap_pa,
         args.load_profile, args.total_force_n,
         args.schur_relinearizations, args.target_right_x_m,
         args.schur_maximum_attempts, args.pressure_basis,
         args.target_reference_slope, args.target_columns,
         args.full_width_target, args.source_rows, args.source_y_profile,
         args.enforcement_solver, args.monolithic_maximum_pivots,
         args.monolithic_linear_solver, args.emit_restart_checkpoint)
