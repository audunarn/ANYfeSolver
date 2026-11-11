"""Bounded, source-only timing of the existing eight-Q4 coupled D1 fixture.

This temporarily wraps call sites in this process only. It does not change
the solver, its candidate decisions, or the registered historical cases.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from functools import wraps
import json
from pathlib import Path
import time

import numpy as np

import anyfesolver.nonlinear_static as nonlinear_static
import anyfesolver.general_contact.s3_patch_augmented as augmented
import anyfesolver.general_contact.s3_patch_mortar_energy as mortar
import anyfesolver.general_contact.s3_patch_static as patch_static
import anyfesolver.general_contact.static_path as static_path
import anyfesolver.general_contact.semismooth_path as semismooth_path
import anyfesolver.nonlinear_restart as nonlinear_restart
from scripts.general_contact_d1_selected_vs_fine_coupled_dev import _run
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case
from scripts.general_contact_d1_pressure_bins import integrate_q1_bins


def main(output: Path, *, candidate_rows: int | None = None,
         candidate_columns: int = 4,
         candidate_y_profile: str | None = None,
         compare_residual_only: bool = False,
         monolithic_route: bool = False,
         checkpoint_guard_counts: bool = False,
         omit_checkpoint: bool = False,
         monolithic_maximum_pivots: int = 12,
         monolithic_linear_solver: str = "direct") -> None:
    if output.exists():
        raise FileExistsError(output)
    records = defaultdict(lambda: {"calls": 0, "inclusive_seconds": 0.0,
                                  "exclusive_seconds": 0.0, "exceptions": 0})
    checkpoint_guard_contexts = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    stack: list[list[float]] = []

    def watch(owner, attribute: str, label: str) -> None:
        original = getattr(owner, attribute)

        @wraps(original)
        def measured(*args, **kwargs):
            frame = [time.perf_counter(), 0.0]
            stack.append(frame)
            try:
                return original(*args, **kwargs)
            except BaseException:
                records[label]["exceptions"] += 1
                raise
            finally:
                elapsed = time.perf_counter() - frame[0]
                stack.pop()
                item = records[label]
                item["calls"] += 1
                item["inclusive_seconds"] += elapsed
                item["exclusive_seconds"] += elapsed - frame[1]
                if stack:
                    stack[-1][1] += elapsed

        setattr(owner, attribute, measured)

    # Each binding is process-local. The call-tree sums exclusive time once.
    watch(static_path, "_assemble_nonlinear_system", "coupled_assembly")
    watch(nonlinear_static, "_assemble_nonlinear_system_unchecked",
          "structural_assembly")
    watch(patch_static.GeneralS3PatchManager, "evaluate", "contact_evaluate")
    watch(patch_static.GeneralS3PatchManager, "admissible_step_fraction",
          "swept_step_check")
    watch(augmented.MovingS3PatchProjectedNormal, "prepare_work_state",
          "prepare_projected_state")
    watch(augmented.MovingS3PatchProjectedNormal, "_jet2_s3_work",
          "jet2_contact_work")
    watch(mortar.MovingS3PatchPrimalMortarPenalty, "evaluate",
          "aggregate_pair_geometry")
    watch(augmented, "q4_s3_overlap_derivatives", "q4_s3_jet2_pair")
    watch(augmented, "q4_s3_overlap_first_derivatives", "q4_s3_first_pair")
    watch(static_path, "refined_patch_supported_schur", "supported_schur")
    watch(static_path, "factorize", "structural_factorization")
    watch(static_path, "assemble_stiffness_matrix", "initial_linear_stiffness")
    watch(patch_static, "supported_pressure_gram", "pressure_rank_diagnostics")

    if monolithic_route:
        if candidate_rows is None or compare_residual_only:
            raise ValueError("monolithic route profiling requires rows and no residual comparison")
        for attribute, label in (
            ("_prepare_qualified_q4_states_for_nonlinear_solve", "q4_state_prepare"),
            ("_activate_nonlinear_state_storage", "state_storage_activate"),
            ("_commit_nonlinear_state_candidate", "state_commit"),
            ("_materialize_final_nonlinear_states", "state_materialize"),
            ("_finalize_nonlinear_element_states", "state_finalize"),
            ("_seal_final_qualified_q4_states", "q4_state_seal"),
            ("create_nonlinear_checkpoint", "checkpoint_create"),
            ("solve_semismooth_increment", "ssn_increment"),
        ):
            watch(static_path, attribute, label)
        if checkpoint_guard_counts:
            exact_guard = nonlinear_restart._EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD

            def count_guard(model, *, context):
                group = context.split(" at ", 1)[0]
                started_guard = time.perf_counter()
                try:
                    return exact_guard(model, context=context)
                finally:
                    item = checkpoint_guard_contexts[group]
                    item["calls"] += 1
                    item["seconds"] += time.perf_counter()-started_guard

            nonlinear_restart._EXACT_QUALIFIED_COMPONENT_LIFECYCLE_GUARD = count_guard
        watch(nonlinear_restart, "validate_nonlinear_checkpoint",
              "checkpoint_self_validation")
        watch(semismooth_path, "_assemble_nonlinear_system", "ssn_structural_assembly")
        watch(semismooth_path, "linearize", "ssn_contact_linearization")
        watch(semismooth_path, "factorize", "ssn_factorization")
        watch(augmented.MovingS3PatchProjectedNormal,
              "nodal_force_linearization", "raw_nodal_force_work")
        watch(patch_static.GeneralS3PatchManager,
              "_gap_lower", "independent_gap_check")
        model, definition, settings, program = build_case(
            candidate_columns, source_rows=candidate_rows,
            source_y_profile=candidate_y_profile,
            load_profile="distributed", total_force_n=0.3,
            target_right_x_m=1.3,
        )
        settings = replace(
            settings, enforcement_solver="monolithic_active_set",
            complementarity_force_scale_n=0.3,
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
        started = time.perf_counter()
        result = nonlinear_static.solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            emit_restart_checkpoint=not omit_checkpoint,
        )
        record = {
            "method": "development whole-route coarse timing; unchanged mechanics",
            "source_grid": [candidate_columns, candidate_rows],
            "source_y_profile": candidate_y_profile,
            "checkpoint_requested": not omit_checkpoint,
            "monolithic_maximum_pivots": monolithic_maximum_pivots,
            "monolithic_linear_solver": monolithic_linear_solver,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "elapsed_seconds": time.perf_counter()-started,
            "accepted_steps": len(result.steps),
            "final_displacements": result.displacements.tolist(),
            "contact": result.info.get("general_contact"),
            "checkpoint_sha256": (None if result.restart_checkpoint is None
                                  else result.restart_checkpoint["checkpoint_sha256"]),
            "step_counters": [{
                key: item.get(key) for key in (
                    "step_index", "semismooth_iterations", "candidate_evaluations",
                    "backtracks", "complementarity_relative_force",
                )
            } for item in result.info.get("force_displacement_history", ())],
            "timings": dict(sorted(records.items())),
            "checkpoint_guard_contexts": dict(sorted(
                checkpoint_guard_contexts.items(),
                key=lambda pair: pair[1]["seconds"], reverse=True,
            )),
        }
        output.write_text(json.dumps(record, indent=2, default=str)+"\n",
                          encoding="utf-8")
        print(json.dumps({key: record[key] for key in (
            "status", "failure_reason", "elapsed_seconds", "accepted_steps",
            "step_counters",
        )}, indent=2))
        if result.status != "completed":
            raise RuntimeError("timed monolithic fixture did not complete")
        return

    started = time.perf_counter()
    if candidate_rows is not None:
        paired = {"first": [], "second": []}
        if compare_residual_only:
            for name, label in (("q4_s3_overlap_first_derivatives", "first"),
                                ("q4_s3_overlap_derivatives", "second")):
                original = getattr(augmented, name)

                def capture(*args, _original=original, _label=label, **kwargs):
                    value = _original(*args, **kwargs)
                    paired[_label].append(value)
                    return value

                setattr(augmented, name, capture)
        model, definition, settings, _path = build_case(
            candidate_columns, source_rows=candidate_rows,
            load_profile="distributed", total_force_n=0.3,
            target_right_x_m=1.3,
            source_y_profile=candidate_y_profile,
        )
        manager = patch_static.GeneralS3PatchManager(
            model, definition, settings,
        )
        displacement = np.zeros(manager.n_dof)
        for node in range(1, (candidate_rows+1)*(candidate_columns+1)+1):
            displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -50e-6
        residual_only = None
        if compare_residual_only:
            residual_started = time.perf_counter()
            residual_force, _, residual_state = manager.evaluate(
                displacement, tangent=False,
            )
            residual_only = {
                "elapsed_seconds": time.perf_counter()-residual_started,
                "force": residual_force,
                "integration_force": residual_state.integration_force,
                "pressure": np.asarray(residual_state.numerical_pressure_pa),
            }
        tangent_started = time.perf_counter()
        force, tangent, contact = manager.evaluate(displacement, tangent=True)
        tangent_elapsed = time.perf_counter()-tangent_started
        elapsed = time.perf_counter()-started
        if stack:
            raise AssertionError("timing stack did not close")
        record = {
            "method": "process-local single active uniform-close candidate; not a coupled solve or acceptance",
            "source_grid": [candidate_columns, candidate_rows],
            "source_y_profile": candidate_y_profile,
            "target_s3_faces": len(definition.surfaces[1].element_ids),
            "closing_shift_m": -50e-6,
            "elapsed_seconds": elapsed,
            "tangent_elapsed_seconds": tangent_elapsed,
            "unwrapped_seconds": elapsed-sum(
                item["exclusive_seconds"] for item in records.values()),
            "timings": dict(sorted(records.items())),
            "overlapping_pairs": contact.overlapping_pairs,
            "active_pressure_dofs": contact.active_pressure_dofs,
            "contact_force_z_n": float(contact.integration_force[2]),
            "minimum_gap_m": float(contact.minimum_gap),
            "force_norm_n": float(np.linalg.norm(force)),
            "tangent_nonzeros": int(tangent.nnz),
        }
        if residual_only is not None:
            if len(paired["first"]) != len(paired["second"]):
                raise AssertionError("first/full derivative pair counts differ")
            pair_differences = list(zip(paired["first"], paired["second"]))
            record["residual_only_comparison"] = {
                "elapsed_seconds": residual_only["elapsed_seconds"],
                "force_max_difference_n": float(np.max(np.abs(
                    residual_only["force"]-force))),
                "force_l2_difference_n": float(np.linalg.norm(
                    residual_only["force"]-force)),
                "integration_force_max_difference_n": float(np.max(np.abs(
                    np.asarray(residual_only["integration_force"])
                    -np.asarray(contact.integration_force)))),
                "pressure_max_difference_pa": float(np.max(np.abs(
                    residual_only["pressure"]
                    -np.asarray(contact.numerical_pressure_pa)))),
                "paired_derivative_calls": len(pair_differences),
                "different_branch_signatures": sum(
                    first.branch_signature != second.branch_signature
                    for first, second in pair_differences),
                "different_triangle_counts": sum(
                    first.integration_triangles != second.integration_triangles
                    for first, second in pair_differences),
                "maximum_support_gradient_difference": max((
                    float(np.max(np.abs(first.support_gradient
                                        -second.support_gradient)))
                    for first, second in pair_differences), default=0.0),
                "maximum_gap_gradient_difference": max((
                    float(np.max(np.abs(first.weighted_gap_gradient
                                        -second.weighted_gap_gradient)))
                    for first, second in pair_differences), default=0.0),
            }
        if candidate_y_profile == "near_rim8":
            # Independent exact P1 support for this constant-gap, flat pose.
            # It uses declared physical knots and a scalar constant pressure,
            # not the production overlap or pressure-support kernel.
            y_knots = np.asarray((0.0, 0.2, 0.4, 0.6, 0.75,
                                  0.825, 0.875, 0.895, 1.0))
            y_support = np.zeros(candidate_rows+1)
            for row in range(candidate_rows):
                left, right = y_knots[row:row+2]
                clipped = max(0.0, min(right, 0.9)-left)
                fraction = clipped/(right-left)
                y_support[row] += clipped*(1.0-fraction/2.0)
                y_support[row+1] += clipped*fraction/2.0
            x_support = np.full(candidate_columns+1, 1.0/candidate_columns)
            x_support[[0, -1]] /= 2.0
            scalar_pressure_pa = 10000.0*10.0e-6
            expected_source_residual = (-scalar_pressure_pa
                                        *np.outer(y_support, x_support).reshape(-1))
            source_residual = np.asarray([
                force[model.mesh.dof_manager.get_node_dofs(node)[2]]
                for node in range(1, (candidate_rows+1)*(candidate_columns+1)+1)
            ])
            target_start = len(source_residual)+1
            target_residual = np.asarray([
                force[model.mesh.dof_manager.get_node_dofs(node)[2]]
                for node in range(target_start,
                                  target_start+2*(2))
            ])
            pressure_grid = np.asarray(contact.numerical_pressure_pa).reshape(
                candidate_rows+1, candidate_columns+1,
            )
            bins = integrate_q1_bins(
                pressure_grid, 16, 16, source_y_knots=tuple(y_knots),
            )
            expected_bins = np.asarray([
                [scalar_pressure_pa/16.0
                 *max(0.0, min((row+1)/16.0, 0.9)-row/16.0)
                 for _ in range(16)]
                for row in range(16)
            ])
            record["independent_constant_pressure_oracle"] = {
                "pressure_pa": scalar_pressure_pa,
                "source_nodal_uz_residual_max_error_n": float(np.max(
                    np.abs(source_residual-expected_source_residual))),
                "source_integrated_force_error_n": float(abs(
                    -np.sum(source_residual)-0.09)),
                "target_integrated_force_error_n": float(abs(
                    np.sum(target_residual)-0.09)),
                "bin_force_l2_relative_error": float(
                    np.linalg.norm(bins-expected_bins)/np.linalg.norm(expected_bins)),
            }
        output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in record.items()
                          if key != "timings"}, indent=2))
        return
    case = _run("fine_nodal_q1", 8)
    elapsed = time.perf_counter() - started
    if stack:
        raise AssertionError("timing stack did not close")
    contacts = case.get("contact") or {}
    record = {
        "method": "process-local nested inclusive/exclusive perf_counter wrappers",
        "fixture": "eight-Q4/two-S3, 0.3 N distributed, target x=1.3 m, two force steps",
        "basis": "fine_nodal_q1",
        "status": case["status"],
        "failure_reason": case["failure_reason"],
        "accepted_steps": case["accepted_steps"],
        "elapsed_seconds": elapsed,
        "solver_elapsed_seconds": case["elapsed_seconds"],
        "unwrapped_seconds": elapsed - sum(
            item["exclusive_seconds"] for item in records.values()),
        "timings": dict(sorted(records.items())),
        "contact_force_z_n": (contacts.get("integration_force") or [None]*3)[2],
        "minimum_gap_m": contacts.get("minimum_gap"),
        "pressure_modes": len(contacts.get("source_pressure_node_ids") or ()),
        "final_displacements_m": case["final_displacements_m"],
        "step_augmentations": [step["augmentation_iterations"]
                               for step in case["history"]],
        "step_contact_energy_evaluations": [step["contact"]["energy_evaluations"]
                                             for step in case["history"]],
        "step_pair_geometry_evaluations": [step["contact"]["pair_geometry_evaluations"]
                                           for step in case["history"]],
    }
    output.write_text(json.dumps(record, indent=2, default=str) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("final_displacements_m", "timings")}, indent=2))
    if case["status"] != "completed":
        raise RuntimeError("timed D1 fixture did not complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-rows", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--candidate-columns", type=int, choices=(4, 8, 16),
                        default=4)
    parser.add_argument("--candidate-y-profile", choices=("near_rim8",))
    parser.add_argument("--compare-residual-only", action="store_true")
    parser.add_argument("--monolithic-route", action="store_true")
    parser.add_argument("--checkpoint-guard-counts", action="store_true")
    parser.add_argument("--omit-checkpoint", action="store_true")
    parser.add_argument("--monolithic-maximum-pivots", type=int, default=12)
    parser.add_argument("--monolithic-linear-solver",
                        choices=("direct", "block_response_dev"), default="direct")
    args = parser.parse_args()
    main(args.output, candidate_rows=args.candidate_rows,
         candidate_columns=args.candidate_columns,
         candidate_y_profile=args.candidate_y_profile,
         compare_residual_only=args.compare_residual_only,
         monolithic_route=args.monolithic_route,
         checkpoint_guard_counts=args.checkpoint_guard_counts,
         omit_checkpoint=args.omit_checkpoint,
         monolithic_maximum_pivots=args.monolithic_maximum_pivots,
         monolithic_linear_solver=args.monolithic_linear_solver)
