"""Development check of the assembled static tangent at a saved Q4/S3 pose."""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.core import contact_scope, suspend_contact
from anyfesolver.nonlinear_static import (
    _assemble_nonlinear_system, _discard_nonlinear_state_candidate,
    _prepare_initial_states,
)
from scripts.general_contact_d1_distributed_tangent_probe import _branches
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _assemble(model, displacement, states, *, tangent):
    try:
        force, stiffness, _trial = _assemble_nonlinear_system(
            model, displacement, states, 5, tangent=tangent,
            kinematics="von_karman")
        return force, stiffness
    finally:
        _discard_nonlinear_state_candidate(states)


def main(reference_path: Path, output_path: Path,
         nodes: tuple[int, int] = (20, 45),
         axes: tuple[str, ...] = ("ux", "uz"),
         columns: int = 24,
         target_reference_slope: float = 0.0) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    report = {"reference": str(reference_path),
              "candidate": "distributed_coupled_tangent_dev",
              "axes": list(axes), "columns": columns,
              "target_reference_slope": target_reference_slope}
    try:
        saved = json.loads(reference_path.read_text(encoding="utf-8"))
        expected_steps = ((4 if saved["four_phase"] else 1)
                          * saved["steps_per_segment"])
        if (saved["status"], saved["columns"], saved["load_profile"],
                saved["accepted_steps"],
                saved.get("target_reference_slope", 0.0)) != (
                "completed", columns, "distributed", expected_steps,
                target_reference_slope):
            raise ValueError("expected completed matching distributed reference")
        model, definition, settings, _program = build_case(
            columns, four_phase=saved["four_phase"],
            steps_per_segment=saved["steps_per_segment"],
            load_profile="distributed",
            total_force_n=saved["total_downward_force_n"],
            target_right_x_m=saved["target_right_x_m"],
            target_reference_slope=target_reference_slope)
        settings = replace(settings, augmentation_schur_maximum_pressure_pa=250,
                           augmentation_schur_maximum_step_pa=200)
        pose = np.asarray(saved["final_displacements"], dtype=float)
        multipliers = np.asarray(
            saved["contact"]["normal_multiplier_estimate_pa"], dtype=float)
        with contact_scope(model, definition, settings,
                           allow_augmented=True) as manager:
            model.apply_boundary_conditions()
            initial_stiffness, _ = assemble_stiffness_matrix(model)
            zeros = np.zeros(model.mesh.dof_manager.total_dofs)
            _, _, transform, prescribed, independent, _ = (
                build_constraint_transformation(initial_stiffness, zeros, model))
            manager.restrict_tangent_to_reduced_variations(transform)
            manager.set_normal_multipliers(multipliers)
            if pose.size != zeros.size or transform.shape[1] != 4*columns+2:
                raise ValueError("unexpected structural coordinate size")
            pose_error = float(np.max(np.abs(
                pose-(prescribed+transform@pose[independent]))))
            if pose_error > 1e-12:
                raise ValueError("saved pose violates structural supports")
            states, _provenance = _prepare_initial_states(model, None, None, 5)
            index = {int(dof): column for column, dof in enumerate(independent)}
            direction_reduced = np.zeros(transform.shape[1])
            for node in nodes:
                dofs = model.mesh.dof_manager.get_node_dofs(node)
                for axis in axes:
                    direction_reduced[index[int(dofs[0 if axis == "ux" else 2])]] = 0.5
            direction = np.asarray(transform@direction_reduced).reshape(-1)
            source, target = manager._positions(pose)
            state = manager.law.evaluate(source, target, multipliers)
            pressure_error = float(np.max(np.abs(
                state.pressure
                - np.asarray(saved["contact"]["numerical_pressure_pa"],
                             dtype=float))))
            moved = {tuple(int(node) for node in manager.law.source_node_ids).index(
                node) for node in nodes}
            base_branches = _branches(manager.law, source, target, moved)
            base_force, base_tangent = _assemble(model, pose, states, tangent=True)
            with suspend_contact():
                structural_force, structural_tangent = _assemble(
                    model, pose, states, tangent=True)
            reduced_action = np.asarray(
                transform.T@(base_tangent@direction)).reshape(-1)
            contact_action = np.asarray(transform.T@(
                (base_tangent-structural_tangent)@direction)).reshape(-1)
            report.update({
                "free_dofs": int(transform.shape[1]),
                "source_nodes": list(nodes),
                "pose_constraint_error_m": pose_error,
                "pressure_replay_error_pa": pressure_error,
                "base_active_modes": state.active_pressure_dofs,
                "base_internal_force_norm_n": float(np.linalg.norm(base_force)),
                "mapped_contact_force_norm_n": float(np.linalg.norm(
                    transform.T@(base_force-structural_force))),
                "branch_pairs": len(base_branches),
            })
            if (pressure_error > 1e-6 or state.active_pressure_dofs
                    != saved["contact"]["active_pressure_dofs"]):
                raise ValueError("saved contact state did not replay")
            results = []
            for step in (1e-6, 5e-7):
                plus_pose, minus_pose = pose+step*direction, pose-step*direction
                plus_force, _ = _assemble(model, plus_pose, states, tangent=False)
                minus_force, _ = _assemble(model, minus_pose, states, tangent=False)
                with suspend_contact():
                    plus_structural, _ = _assemble(
                        model, plus_pose, states, tangent=False)
                    minus_structural, _ = _assemble(
                        model, minus_pose, states, tangent=False)
                finite = np.asarray(transform.T@(
                    (plus_force-minus_force)/(2*step))).reshape(-1)
                norm = max(float(np.linalg.norm(reduced_action)),
                           float(np.linalg.norm(finite)))
                contact_finite = np.asarray(transform.T@(
                    ((plus_force-plus_structural)
                     -(minus_force-minus_structural))/(2*step))).reshape(-1)
                contact_norm = max(float(np.linalg.norm(contact_action)),
                                   float(np.linalg.norm(contact_finite)))
                plus_source, plus_target = manager._positions(plus_pose)
                minus_source, minus_target = manager._positions(minus_pose)
                plus_state = manager.law.evaluate(
                    plus_source, plus_target, multipliers)
                minus_state = manager.law.evaluate(
                    minus_source, minus_target, multipliers)
                smooth = (
                    np.array_equal(plus_state.pressure > 0, state.pressure > 0)
                    and np.array_equal(minus_state.pressure > 0,
                                       state.pressure > 0)
                    and np.array_equal(plus_state.reference_support > 0,
                                       state.reference_support > 0)
                    and np.array_equal(minus_state.reference_support > 0,
                                       state.reference_support > 0)
                    and plus_state.overlap_signature == state.overlap_signature
                    and minus_state.overlap_signature == state.overlap_signature
                    and _branches(manager.law, plus_source, plus_target, moved)
                    == base_branches
                    and _branches(manager.law, minus_source, minus_target, moved)
                    == base_branches)
                results.append({
                    "step_m": step, "smooth_branch": bool(smooth),
                    "relative_reduced_tangent_error": float(np.linalg.norm(
                        reduced_action-finite))/max(norm, 1e-12),
                    "action_scale_n_per_m": norm,
                    "relative_mapped_contact_tangent_error": float(np.linalg.norm(
                        contact_action-contact_finite))/max(contact_norm, 1e-12),
                    "mapped_contact_action_scale_n_per_m": contact_norm,
                })
            report["runs"] = results
            report["status"] = ("passed_development" if all(
                item["smooth_branch"] and item["action_scale_n_per_m"] > 1e-6
                and item["relative_reduced_tangent_error"] <= 1e-5
                and item["mapped_contact_action_scale_n_per_m"] > 1e-6
                and item["relative_mapped_contact_tangent_error"] <= 1e-5
                for item in results) else "inconclusive_or_failed_development")
    except Exception as exc:
        report.update({"status": "failed_development",
                       "error": f"{type(exc).__name__}: {exc}"})
    report["elapsed_seconds"] = time.perf_counter()-started
    output_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "elapsed_seconds", "error", "runs") if key in report},
                     indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nodes", type=int, nargs=2, default=(20, 45))
    parser.add_argument("--axes", choices=("ux", "uz", "both"), default="both")
    parser.add_argument("--columns", type=int, default=24)
    parser.add_argument("--target-reference-slope", type=float, default=0.0)
    args = parser.parse_args()
    selected_axes = ("ux", "uz") if args.axes == "both" else (args.axes,)
    main(args.reference, args.output, nodes=tuple(args.nodes), axes=selected_axes,
         columns=args.columns,
         target_reference_slope=args.target_reference_slope)
