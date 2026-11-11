"""Development check of Q4/S3 contact work when the fixed target is moved."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_distributed_tangent_probe import _branches
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def main(reference_path: Path, output_path: Path, *, rigid_z: bool = False,
         negative_one_sided: bool = False) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    report = {"reference": str(reference_path),
              "candidate": "distributed_24q4_target_tangent_dev"}
    try:
        saved = json.loads(reference_path.read_text(encoding="utf-8"))
        if (saved["status"], saved["columns"], saved["load_profile"],
                saved["accepted_steps"]) != ("completed", 24, "distributed", 12):
            raise ValueError("expected completed distributed 24-Q4 reference")
        model, definition, settings, _program = build_case(
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
        target_ids = tuple(int(node) for node in law.target_node_ids)
        if target_ids != (51, 52, 53, 54):
            raise ValueError("unexpected target material identities")
        source_coordinates = 3*len(source)
        axes = np.arange(law.n_coordinates, dtype=np.intp)
        law.set_tangent_active_coordinates(axes)
        direction = np.zeros(law.n_coordinates)
        moved_targets = target_ids if rigid_z else (52, 53)
        for node in moved_targets:
            direction[source_coordinates+3*target_ids.index(node)+2] = 0.5
        target_direction = direction[source_coordinates:].reshape(target.shape)
        base = law.work(source, target, multipliers, tangent=True)
        pressure_error = float(np.max(np.abs(
            base.state.pressure
            - np.asarray(saved["contact"]["numerical_pressure_pa"],
                         dtype=float))))
        if pressure_error > 1e-6 or base.state.active_pressure_dofs != 18:
            raise ValueError("saved pressure did not replay")
        moved_source = set(range(len(source)))
        base_branches = _branches(law, source, target, moved_source)
        action = base.tangent@direction
        report.update({"target_nodes": list(moved_targets),
                       "target_direction": "0.5 uz each",
                       "difference_scheme": ("negative_one_sided" if negative_one_sided
                                             else "central"),
                       "physical_rows": law.n_coordinates,
                       "pressure_replay_error_pa": pressure_error,
                       "base_active_modes": base.state.active_pressure_dofs,
                       "overlap_pair_branches": len(base_branches),
                       "action_scale_n_per_m": float(np.linalg.norm(action))})
        runs = []
        for step in (1e-6, 5e-7):
            minus_target = target-step*target_direction
            minus = law.work(source, minus_target, multipliers, tangent=False)
            plus_target = target+step*target_direction
            plus = (None if negative_one_sided else law.work(
                source, plus_target, multipliers, tangent=False))
            finite = ((base.residual-minus.residual)/step
                      if negative_one_sided else
                      (plus.residual-minus.residual)/(2*step))
            scale = max(float(np.linalg.norm(action)),
                        float(np.linalg.norm(finite)))
            branch_checks = {
                "pressure_active": bool(np.array_equal(
                    minus.state.pressure > 0, base.state.pressure > 0)
                    and (negative_one_sided or np.array_equal(
                        plus.state.pressure > 0, base.state.pressure > 0))),
                "reference_support": bool(np.array_equal(
                    minus.state.reference_support > 0,
                    base.state.reference_support > 0)
                    and (negative_one_sided or np.array_equal(
                        plus.state.reference_support > 0,
                        base.state.reference_support > 0))),
                "overlap_signature": bool(
                    minus.state.overlap_signature == base.state.overlap_signature
                    and (negative_one_sided or plus.state.overlap_signature
                         == base.state.overlap_signature)),
                "pair_derivatives": bool(
                    _branches(law, source, minus_target, moved_source)
                    == base_branches
                    and (negative_one_sided or
                         _branches(law, source, plus_target, moved_source)
                         == base_branches)),
            }
            smooth = all(branch_checks.values())
            runs.append({"step_m": step, "smooth_branch": bool(smooth),
                         "branch_checks": branch_checks,
                         "relative_tangent_error": float(np.linalg.norm(
                             action-finite))/max(scale, 1e-12),
                         "action_scale_n_per_m": scale,
                         "source_action_scale_n_per_m": float(np.linalg.norm(
                             action[:source_coordinates])),
                         "target_action_scale_n_per_m": float(np.linalg.norm(
                             action[source_coordinates:]))})
        report["runs"] = runs
        report["status"] = ("passed_development" if all(
            run["smooth_branch"] and run["action_scale_n_per_m"] > 1e-6
            and run["relative_tangent_error"] <= 1e-5
            for run in runs) else "inconclusive_or_failed_development")
    except Exception as exc:
        report.update({"status": "failed_development",
                       "error": f"{type(exc).__name__}: {exc}"})
    report["elapsed_seconds"] = time.perf_counter()-started
    output_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "elapsed_seconds", "error", "runs") if key in report},
                     indent=2))


if __name__ == "__main__":
    flags = set(sys.argv[3:])
    if len(sys.argv) < 3 or len(sys.argv) > 5 or len(flags) != len(sys.argv[3:]) or not flags.issubset({"--rigid-z", "--negative-one-sided"}):
        raise SystemExit(
            "usage: target_tangent_probe.py REF OUTPUT [--rigid-z] [--negative-one-sided]")
    main(Path(sys.argv[1]), Path(sys.argv[2]),
         rigid_z="--rigid-z" in flags,
         negative_one_sided="--negative-one-sided" in flags)
