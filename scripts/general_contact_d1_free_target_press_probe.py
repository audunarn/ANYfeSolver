"""Development press with a deformable two-S3 target clamped at one edge."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from anyfesolver import BoundaryCondition, solve_static_nonlinear
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import (
    _json_default, build_case,
)


def build_free_target_case(columns: int, *, relinearizations: int = 0,
                           steps_per_segment: int = 3,
                           four_phase: bool = False,
                           enforcement_solver: str = "nested_augmented",
                           common_reference_slope: float = 0.0):
    model, definition, settings, program = build_case(
        columns, four_phase=four_phase,
        steps_per_segment=steps_per_segment,
        load_profile="distributed", total_force_n=0.3,
        target_right_x_m=1.3,
        common_reference_slope=common_reference_slope)
    settings = replace(settings,
                       augmentation_schur_maximum_pressure_pa=250,
                       augmentation_schur_maximum_step_pa=200,
                       augmentation_schur_maximum_relinearizations=(
                           relinearizations))
    if enforcement_solver == "monolithic_active_set":
        settings = replace(
            settings, enforcement_solver=enforcement_solver,
            complementarity_force_scale_n=0.3,
            augmentation_acceleration="none",
            augmentation_schur_maximum_modes=None,
            augmentation_schur_maximum_pressure_pa=None,
            augmentation_schur_maximum_step_pa=None,
            augmentation_schur_difference_step_m=None,
            augmentation_schur_maximum_attempts=1,
            augmentation_schur_maximum_relinearizations=0,
        )
    elif enforcement_solver != "nested_augmented":
        raise ValueError("unknown enforcement solver")
    target_ids = tuple(range(2*(columns+1)+1, 2*(columns+1)+5))
    model.boundary_conditions = [
        bc for bc in model.boundary_conditions if bc.name != "target_fixed"]
    model.add_boundary_condition(BoundaryCondition(
        "target_left_clamp", (target_ids[0], target_ids[3]),
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    return model, definition, settings, program


def main(output_path: Path, *, columns: int,
         relinearizations: int = 0,
         steps_per_segment: int = 3,
         equilibrium_tolerance: float = 1.0e-6,
         four_phase: bool = False,
         nonlinear_backtrack: bool = False,
         integration_side: str = "q4",
         pressure_basis: str = "fine_nodal_q1",
         enforcement_solver: str = "nested_augmented") -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    report = {"candidate": "free_target_left_clamped_press_dev",
              "columns": columns, "target_support": "left_edge_clamped",
              "maximum_schur_relinearizations": relinearizations,
              "steps_per_segment": steps_per_segment,
              "equilibrium_tolerance": equilibrium_tolerance,
              "four_phase": four_phase,
              "nonlinear_backtrack": nonlinear_backtrack,
              "integration_side": integration_side,
              "pressure_basis": pressure_basis}
    report["enforcement_solver"] = enforcement_solver
    try:
        model, definition, settings, program = build_free_target_case(
            columns, relinearizations=relinearizations,
            steps_per_segment=steps_per_segment, four_phase=four_phase,
            enforcement_solver=enforcement_solver)
        if integration_side == "s3":
            definition = replace(definition, interactions=(replace(
                definition.interactions[0], surface="s3_target",
                target="q4_source"),))
        elif integration_side != "q4":
            raise ValueError("integration_side must be q4 or s3")
        if nonlinear_backtrack:
            settings = replace(
                settings, augmentation_schur_nonlinear_backtrack_dev=True)
        if pressure_basis != "fine_nodal_q1":
            settings = replace(settings, q4_source_pressure_basis=pressure_basis)
        result = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            tolerance=equilibrium_tolerance,
            emit_restart_checkpoint=True,
        )
        history = result.info.get("force_displacement_history", ())
        report.update({
            "status": result.status,
            "failure_reason": result.failure_reason,
            "accepted_steps": len(result.steps),
            "has_checkpoint": result.restart_checkpoint is not None,
            "final_displacements": result.displacements.tolist(),
            "final_contact": result.info.get("general_contact"),
            "history": [{
                "step": int(row["step_index"]),
                "normal_force_n": float(row["contact"][
                    "integration_force"][2]),
                "target_force_n": row["contact"].get("target_force"),
                "minimum_gap_m": float(row["contact"]["minimum_gap"]),
                "active_pressure_dofs": int(row["contact"][
                    "active_pressure_dofs"]),
                "pressure_change_pa": (float(row["normal_multiplier_change"])
                                       if "normal_multiplier_change" in row else None),
                "pressure_rank": row["contact"].get(
                    "pressure_gram_scaled_rank"),
                "residual_norm": float(row["residual_norm"]),
                "augmentation_iterations": int(row[
                    "augmentation_iterations"]),
                "semismooth_iterations": row.get("semismooth_iterations"),
                "complementarity_relative_force": row.get(
                    "complementarity_relative_force"),
                "schur_attempts": row.get("schur_attempts"),
                "schur_applied": row.get("schur_applied"),
                "schur_rejected": row.get("schur_rejected"),
                **({"augmentation_trace": row.get("augmentation_trace"),
                    "source_pressure_node_ids": row["contact"].get(
                        "source_pressure_node_ids"),
                    "normal_multiplier_estimate_pa": row["contact"].get(
                        "normal_multiplier_estimate_pa")}
                   if nonlinear_backtrack else {}),
            } for row in history],
            "contact_failure": result.info.get("contact_failure"),
        })
    except Exception as exc:
        report.update({"status": "exception",
                       "failure_reason": f"{type(exc).__name__}: {exc}"})
    report["elapsed_seconds"] = time.perf_counter()-started
    output_path.write_text(json.dumps(report, indent=2,
                                      default=_json_default)+"\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "failure_reason", "accepted_steps",
                       "elapsed_seconds", "history") if key in report},
                     indent=2, default=_json_default))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, choices=(4, 8, 16, 24), default=8)
    parser.add_argument("--pressure-basis", choices=("fine_nodal_q1", "macro_q1_pair_dev"),
                        default="fine_nodal_q1")
    parser.add_argument("--relinearizations", type=int, choices=(0, 1),
                        default=0)
    parser.add_argument("--steps-per-segment", type=int, choices=(2, 3, 6),
                        default=3)
    parser.add_argument("--equilibrium-tolerance", type=float,
                        default=1.0e-6)
    parser.add_argument("--four-phase", action="store_true")
    parser.add_argument("--nonlinear-backtrack", action="store_true")
    parser.add_argument("--integration-side", choices=("q4", "s3"),
                        default="q4")
    parser.add_argument("--enforcement-solver",
                        choices=("nested_augmented", "monolithic_active_set"),
                        default="nested_augmented")
    args = parser.parse_args()
    main(args.output, columns=args.columns,
         relinearizations=args.relinearizations,
         steps_per_segment=args.steps_per_segment,
         equilibrium_tolerance=args.equilibrium_tolerance,
         four_phase=args.four_phase,
         nonlinear_backtrack=args.nonlinear_backtrack,
         integration_side=args.integration_side,
         pressure_basis=args.pressure_basis,
         enforcement_solver=args.enforcement_solver)
