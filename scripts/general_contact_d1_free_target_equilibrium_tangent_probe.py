"""Development coupled tangent check at an equilibrated free-target pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.core import contact_scope, suspend_contact
from anyfesolver.nonlinear_static import _prepare_initial_states
from scripts.general_contact_d1_distributed_coupled_tangent_probe import _assemble
from scripts.general_contact_d1_distributed_tangent_probe import _branches
from scripts.general_contact_d1_free_target_press_probe import (
    build_free_target_case,
)


def main(reference_path: Path, output_path: Path,
         target_z_weights: tuple[float, float] = (0.5, 0.5),
         source_weight: float = 0.1,
         target_x_weights: tuple[float, float] = (0.0, 0.0),
         source_x_weight: float | None = None,
         source_z_weight: float | None = None,
         difference_order: int = 2,
         steps: tuple[float, ...] = (8e-7, 4e-7)) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    report = {"reference": str(reference_path),
              "candidate": "equilibrated_8q4_free_target_tangent_dev"}
    try:
        saved = json.loads(reference_path.read_text(encoding="utf-8"))
        if ((saved["status"], saved["columns"], saved["accepted_steps"],
             saved["target_support"], saved["four_phase"])
                != ("completed", 8, 3, "left_edge_clamped", False)):
            raise ValueError("expected completed equilibrated 8-Q4 press pose")
        model, definition, settings, _program = build_free_target_case(8)
        pose = np.asarray(saved["final_displacements"], dtype=float)
        multipliers = np.asarray(
            saved["final_contact"]["normal_multiplier_estimate_pa"],
            dtype=float)
        with contact_scope(model, definition, settings,
                           allow_augmented=True) as manager:
            model.apply_boundary_conditions()
            initial_stiffness, _ = assemble_stiffness_matrix(model)
            zeros = np.zeros(model.mesh.dof_manager.total_dofs)
            _, _, transform, prescribed, independent, _ = (
                build_constraint_transformation(initial_stiffness, zeros, model))
            manager.restrict_tangent_to_reduced_variations(transform)
            manager.set_normal_multipliers(multipliers)
            pose_error = float(np.max(np.abs(
                pose-(prescribed+transform@pose[independent]))))
            if pose_error > 1e-12:
                raise ValueError("saved pose violates new supports")
            index = {int(dof): column for column, dof in enumerate(independent)}
            direction_reduced = np.zeros(transform.shape[1])
            for node, weight_x, weight_z in zip(
                    (20, 21), target_x_weights, target_z_weights):
                dofs = model.mesh.dof_manager.get_node_dofs(node)
                direction_reduced[index[int(dofs[0])]] = weight_x
                direction_reduced[index[int(dofs[2])]] = weight_z
            source_x = source_weight if source_x_weight is None else source_x_weight
            source_z = source_weight if source_z_weight is None else source_z_weight
            for node in (8, 17):
                dofs = model.mesh.dof_manager.get_node_dofs(node)
                direction_reduced[index[int(dofs[0])]] = source_x
                direction_reduced[index[int(dofs[2])]] = source_z
            direction = np.asarray(transform@direction_reduced).reshape(-1)
            translational_dofs = {
                int(dof) for node in model.mesh.nodes
                for dof in model.mesh.dof_manager.get_node_dofs(node)[:3]
            }
            force_rows = np.asarray([
                row for row, dof in enumerate(independent)
                if int(dof) in translational_dofs], dtype=np.intp)
            moment_rows = np.asarray([
                row for row, dof in enumerate(independent)
                if int(dof) not in translational_dofs], dtype=np.intp)
            states, _ = _prepare_initial_states(model, None, None, 5)
            source, target = manager._positions(pose)
            base_state = manager.law.evaluate(source, target, multipliers)
            pressure_error = float(np.max(np.abs(
                base_state.pressure
                - np.asarray(saved["final_contact"]["numerical_pressure_pa"],
                             dtype=float))))
            if pressure_error > 1e-6 or base_state.active_pressure_dofs < 1:
                raise ValueError("saved contact state did not replay")
            moved_source = set(range(len(source)))
            branches = _branches(manager.law, source, target, moved_source)
            base_force, base_tangent = _assemble(
                model, pose, states, tangent=True)
            with suspend_contact():
                structural_force, structural_tangent = _assemble(
                    model, pose, states, tangent=True)
            coupled_action = np.asarray(transform.T@(
                base_tangent@direction)).reshape(-1)
            contact_action = np.asarray(transform.T@(
                (base_tangent-structural_tangent)@direction)).reshape(-1)
            runs = []
            for step in steps:
                minus_pose = pose-step*direction
                minus_force, _ = _assemble(
                    model, minus_pose, states, tangent=False)
                with suspend_contact():
                    minus_structural, _ = _assemble(
                        model, minus_pose, states, tangent=False)
                if difference_order == 2:
                    minus2_pose = pose-2*step*direction
                    minus2_force, _ = _assemble(
                        model, minus2_pose, states, tangent=False)
                    with suspend_contact():
                        minus2_structural, _ = _assemble(
                            model, minus2_pose, states, tangent=False)
                    coupled_difference = (
                        3*base_force-4*minus_force+minus2_force)/(2*step)
                    contact_difference = (
                        3*(base_force-structural_force)
                        -4*(minus_force-minus_structural)
                        +(minus2_force-minus2_structural))/(2*step)
                else:
                    coupled_difference = (base_force-minus_force)/step
                    contact_difference = (
                        (base_force-structural_force)
                        -(minus_force-minus_structural))/step
                coupled_finite = np.asarray(
                    transform.T@coupled_difference).reshape(-1)
                contact_finite = np.asarray(
                    transform.T@contact_difference).reshape(-1)
                minus_source, minus_target = manager._positions(minus_pose)
                minus_state = manager.law.evaluate(
                    minus_source, minus_target, multipliers)
                if difference_order == 2:
                    minus2_source, minus2_target = manager._positions(
                        minus2_pose)
                    minus2_state = manager.law.evaluate(
                        minus2_source, minus2_target, multipliers)
                else:
                    minus2_source = minus2_target = minus2_state = None
                branch_checks = {
                    "pressure_active": bool(np.array_equal(
                        minus_state.pressure > 0,
                        base_state.pressure > 0)
                        and (minus2_state is None or np.array_equal(
                            minus2_state.pressure > 0,
                            base_state.pressure > 0))),
                    "reference_support": bool(np.array_equal(
                        minus_state.reference_support > 0,
                        base_state.reference_support > 0)
                        and (minus2_state is None or np.array_equal(
                            minus2_state.reference_support > 0,
                            base_state.reference_support > 0))),
                    "overlap_signature": (
                        minus_state.overlap_signature
                        == base_state.overlap_signature
                        and (minus2_state is None or
                             minus2_state.overlap_signature
                             == base_state.overlap_signature)),
                    "pair_derivatives": (
                        _branches(manager.law, minus_source, minus_target,
                                  moved_source) == branches
                        and (minus2_state is None or
                             _branches(manager.law, minus2_source,
                                       minus2_target, moved_source)
                             == branches)),
                }
                total_norm = max(float(np.linalg.norm(coupled_action)),
                                 float(np.linalg.norm(coupled_finite)))
                contact_norm = max(float(np.linalg.norm(contact_action)),
                                   float(np.linalg.norm(contact_finite)))
                component_errors = {}
                for label, indices in (("force", force_rows),
                                       ("moment", moment_rows)):
                    for kind, analytic, finite in (
                            ("coupled", coupled_action, coupled_finite),
                            ("mapped_contact", contact_action,
                             contact_finite)):
                        scale = max(float(np.linalg.norm(analytic[indices])),
                                    float(np.linalg.norm(finite[indices])))
                        component_errors[f"{kind}_{label}_error"] = (
                            float(np.linalg.norm(
                                analytic[indices]-finite[indices]))
                            / max(scale, 1e-12))
                        component_errors[f"{kind}_{label}_action"] = scale
                runs.append({
                    "step_m": step,
                    "branch_checks": branch_checks,
                    "smooth_branch": bool(all(branch_checks.values())),
                    "relative_coupled_error": float(np.linalg.norm(
                        coupled_action-coupled_finite))/max(total_norm, 1e-12),
                    "relative_mapped_contact_error": float(np.linalg.norm(
                        contact_action-contact_finite))/max(contact_norm,
                                                             1e-12),
                    "coupled_action_n_per_m": total_norm,
                    "mapped_contact_action_n_per_m": contact_norm,
                    **component_errors,
                })
            report.update({
                "free_dofs": int(transform.shape[1]),
                "free_force_rows": len(force_rows),
                "free_moment_rows": len(moment_rows),
                "target_z_weights": list(target_z_weights),
                "target_x_weights": list(target_x_weights),
                "source_weight": source_weight,
                "source_x_weight": source_x,
                "source_z_weight": source_z,
                "difference_order": difference_order,
                "steps_m": list(steps),
                "pose_constraint_error_m": pose_error,
                "pressure_replay_error_pa": pressure_error,
                "base_active_modes": base_state.active_pressure_dofs,
                "overlap_pair_branches": len(branches),
                "mapped_contact_force_norm_n": float(np.linalg.norm(
                    transform.T@(base_force-structural_force))),
                "runs": runs,
                "status": "passed_development" if all(
                    row["smooth_branch"]
                    and row["relative_coupled_error"] <= 1e-5
                    and row["relative_mapped_contact_error"] <= 1e-5
                    and row["mapped_contact_action_n_per_m"] > 1e-6
                    and all(row[f"{kind}_{label}_error"] <= 1e-5
                            and row[f"{kind}_{label}_action"] > 1e-8
                            for kind in ("coupled", "mapped_contact")
                            for label in ("force", "moment"))
                    for row in runs) else "inconclusive_or_failed_development",
            })
    except Exception as exc:
        report.update({"status": "failed_development",
                       "error": f"{type(exc).__name__}: {exc}"})
    report["elapsed_seconds"] = time.perf_counter()-started
    output_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "elapsed_seconds", "error", "free_dofs",
                       "runs") if key in report}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-z-weights", nargs=2, type=float,
                        default=(0.5, 0.5))
    parser.add_argument("--target-x-weights", nargs=2, type=float,
                        default=(0.0, 0.0))
    parser.add_argument("--source-weight", type=float, default=0.1)
    parser.add_argument("--source-x-weight", type=float)
    parser.add_argument("--source-z-weight", type=float)
    parser.add_argument("--difference-order", type=int, choices=(1, 2),
                        default=2)
    parser.add_argument("--steps", nargs="+", type=float,
                        default=(8e-7, 4e-7))
    args = parser.parse_args()
    main(args.reference, args.output,
         target_z_weights=tuple(args.target_z_weights),
         source_weight=args.source_weight,
         target_x_weights=tuple(args.target_x_weights),
         source_x_weight=args.source_x_weight,
         source_z_weight=args.source_z_weight,
         difference_order=args.difference_order,
         steps=tuple(args.steps))
