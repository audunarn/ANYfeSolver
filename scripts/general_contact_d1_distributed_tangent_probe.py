"""Bounded development audit of a saved distributed Q4/S3 contact tangent."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

from anyfesolver.general_contact.q4_s3_overlap_derivatives import q4_s3_overlap_derivatives
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def _branches(law, source, target, moved):
    result = {}
    target_count = len(law.geometry.target_faces)
    tolerances = dict(zip((
        "support_gradient_tolerance", "support_hessian_tolerance",
        "weighted_gap_gradient_tolerance", "weighted_gap_hessian_tolerance",
    ), law.geometry.q4_source_derivative_tolerances or (None,)*4))
    for si, source_nodes in enumerate(law.source_face_indices):
        if not moved.intersection(int(node) for node in source_nodes):
            continue
        for ti, target_nodes in enumerate(law.target_face_indices):
            local_source, local_target = source[source_nodes], target[target_nodes]
            local = law.geometry.pair_laws[si*target_count+ti].evaluate(
                local_source, local_target)
            if local.reference_overlap_area <= 0.0:
                continue
            derivative = q4_s3_overlap_derivatives(
                law.geometry.source_faces[si].reference_positions,
                local_source, local_target,
                area_tolerance=law.geometry.q4_source_area_tolerance,
                weighted_gap_tolerance=law.geometry.q4_source_weighted_gap_tolerance,
                maximum_triangles=law.geometry.maximum_q4_source_integration_triangles,
                allow_warped_source=True, **tolerances)
            result[f"{si}:{ti}"] = derivative.branch_signature
    return result


def main(reference_path: Path, output_path: Path, *, wide_source: bool = False) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    report = {"reference": str(reference_path),
              "candidate": "distributed_24q4_selected_tangent_dev"}
    try:
        saved = json.loads(reference_path.read_text(encoding="utf-8"))
        if (saved["status"], saved["columns"], saved["load_profile"],
                saved["accepted_steps"]) != ("completed", 24, "distributed", 12):
            raise ValueError("expected the completed distributed 24-Q4 cycle")
        model, definition, settings, _ = build_case(
            24, four_phase=True, steps_per_segment=3, load_profile="distributed",
            total_force_n=0.3, target_right_x_m=1.3)
        settings = replace(settings, augmentation_schur_maximum_pressure_pa=250,
                           augmentation_schur_maximum_step_pa=200)
        manager = GeneralS3PatchManager(model, definition, settings)
        source, target = manager._positions(
            np.asarray(saved["final_displacements"], dtype=float))
        law = manager.law
        multipliers = np.asarray(
            saved["contact"]["normal_multiplier_estimate_pa"], dtype=float)
        source_ids = tuple(int(node) for node in law.source_node_ids)
        moved = {source_ids.index(node) for node in (20, 45)}
        source_direction = np.zeros_like(source)
        for node in moved:
            source_direction[node, (0, 2)] = 0.5
        direction = np.concatenate((source_direction, np.zeros_like(target))).ravel()
        axes = (np.asarray(sorted(
            [3*node for node in range(len(source_ids))]
            + [3*node+2 for node, material_id in enumerate(source_ids)
               if material_id not in (1, 26)]), dtype=np.intp)
            if wide_source else np.flatnonzero(direction))
        law.set_tangent_active_coordinates(axes)
        base = law.work(source, target, multipliers, tangent=True)
        pressure_error = float(np.max(np.abs(
            base.state.pressure
            - np.asarray(saved["contact"]["numerical_pressure_pa"], dtype=float))))
        force_z = float(base.residual[:3*len(source)].reshape(-1, 3)[:, 2].sum())
        force_error = abs(force_z+float(saved["contact"]["integration_force"][2]))
        report.update({"source_nodes": [20, 45],
                       "axis_profile": ("all_free_source_ux_uz" if wide_source
                                        else "four_local_axes"),
                       "selected_axes": axes.tolist(),
                       "pressure_replay_error_pa": pressure_error,
                       "source_residual_z_n": force_z,
                       "force_replay_error_n": force_error,
                       "base_active_modes": base.state.active_pressure_dofs})
        if pressure_error > 1e-6 or force_error > 1e-8:
            raise ValueError("saved pose and multipliers do not replay")
        base_branches = _branches(law, source, target, moved)
        action = (base.tangent @ direction)[axes]
        results = []
        for step in (1e-6, 5e-7):
            plus_source, minus_source = (source+step*source_direction,
                                         source-step*source_direction)
            plus = law.work(plus_source, target, multipliers, tangent=False)
            minus = law.work(minus_source, target, multipliers, tangent=False)
            finite = ((plus.residual-minus.residual)/(2*step))[axes]
            norm = max(float(np.linalg.norm(action)), float(np.linalg.norm(finite)))
            smooth = (
                np.array_equal(plus.state.pressure > 0, base.state.pressure > 0)
                and np.array_equal(minus.state.pressure > 0, base.state.pressure > 0)
                and np.array_equal(plus.state.reference_support > 0,
                                   base.state.reference_support > 0)
                and np.array_equal(minus.state.reference_support > 0,
                                   base.state.reference_support > 0)
                and plus.state.overlap_signature == base.state.overlap_signature
                and minus.state.overlap_signature == base.state.overlap_signature
                and _branches(law, plus_source, target, moved) == base_branches
                and _branches(law, minus_source, target, moved) == base_branches)
            results.append({"step_m": step, "smooth_branch": bool(smooth),
                            "relative_selected_error": float(np.linalg.norm(
                                action-finite))/max(norm, 1e-12),
                            "action_scale_n_per_m": norm,
                            "branch_pairs": len(base_branches)})
        report["runs"] = results
        report["status"] = ("passed_development" if all(
            run["smooth_branch"] and run["action_scale_n_per_m"] > 1e-6
            and run["relative_selected_error"] <= 1e-5 for run in results)
            else "inconclusive_or_failed_development")
    except Exception as exc:
        report.update({"status": "failed_development",
                       "error": f"{type(exc).__name__}: {exc}"})
    report["elapsed_seconds"] = time.perf_counter()-started
    output_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "elapsed_seconds", "error", "runs") if key in report},
                     indent=2))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4
                                    and sys.argv[3] != "--wide-source"):
        raise SystemExit(
            "usage: tangent_probe.py REFERENCE_JSON OUTPUT_JSON [--wide-source]")
    main(Path(sys.argv[1]), Path(sys.argv[2]), wide_source=len(sys.argv) == 4)
