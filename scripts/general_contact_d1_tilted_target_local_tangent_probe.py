"""Screen local Q4/S3 contact derivatives at a finite-area tilted pose.

The target tilt is a disposable physical-position perturbation of a saved
full-press pose. It is not an equilibrated structural step or a capability gate.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def main(reference: Path, output: Path, slope: float) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not 0.0 < slope <= 1.0e-4:
        raise ValueError("tilt slope must be in (0, 1e-4]")
    started = time.perf_counter()
    report = {"candidate": "tilted_target_local_tangent_dev",
              "reference": str(reference), "target_dz_dx": slope}
    try:
        saved = json.loads(reference.read_text(encoding="utf-8"))
        if ((saved["status"], saved["columns"], saved["accepted_steps"],
             saved["load_profile"], saved["total_downward_force_n"],
             saved["target_right_x_m"])
                != ("completed", 8, 2, "distributed", 0.3, 1.3)):
            raise ValueError("expected completed eight-Q4 distributed press")
        model, definition, settings, _ = build_case(
            8, steps_per_segment=2, load_profile="distributed",
            total_force_n=0.3, target_right_x_m=1.3)
        settings = replace(settings, augmentation_schur_maximum_pressure_pa=250,
                           augmentation_schur_maximum_step_pa=200)
        manager = GeneralS3PatchManager(model, definition, settings)
        pose = np.asarray(saved["final_displacements"], dtype=float)
        source, target = manager._positions(pose)
        multipliers = np.asarray(
            saved["contact"]["normal_multiplier_estimate_pa"], dtype=float)
        replay = manager.law.evaluate(source, target, multipliers)
        replay_error = float(np.max(np.abs(
            replay.pressure - saved["contact"]["numerical_pressure_pa"])))
        if replay_error > 1.0e-6:
            raise ValueError("saved full-press pressure did not replay")
        target = target.copy()
        target[:, 2] += slope*(target[:, 0]-float(np.mean(target[:, 0])))
        count = len(source)
        local_direction = np.zeros(3*(count+len(target)))
        target_x_axes = 3*count+3*np.arange(len(target))
        local_direction[target_x_axes] = 1.0
        # The selection limits *both* tangent rows and columns. Keep every
        # physical output axis when comparing the complete mapped wrench.
        manager.law.set_tangent_active_coordinates(
            np.arange(local_direction.size, dtype=np.intp))
        base = manager.law.work(source, target, multipliers, tangent=True)
        mapped_action = np.asarray(
            manager.map.T@(base.tangent@local_direction)).reshape(-1)
        force_rows = np.asarray([
            int(dof) for node in model.mesh.nodes
            for dof in model.mesh.dof_manager.get_node_dofs(node)[:3]],
            dtype=np.intp)
        moment_rows = np.asarray([
            int(dof) for node in model.mesh.nodes
            for dof in model.mesh.dof_manager.get_node_dofs(node)[3:]],
            dtype=np.intp)
        report.update({
            "pressure_replay_error_pa": replay_error,
            "base_active_pressure_modes": base.state.active_pressure_dofs,
            "base_minimum_gap_m": base.state.minimum_physical_gap,
            "base_overlap_area_m2": base.state.overlap_area,
        })
        runs = []
        for step in (1.0e-6, 5.0e-7):
            plus = target.copy()
            minus = target.copy()
            plus[:, 0] += step
            minus[:, 0] -= step
            plus_work = manager.law.work(
                source, plus, multipliers, tangent=False)
            minus_work = manager.law.work(
                source, minus, multipliers, tangent=False)
            local_finite = (plus_work.residual-minus_work.residual)/(2*step)
            mapped_finite = np.asarray(manager.map.T@local_finite).reshape(-1)
            smooth = (
                np.array_equal(plus_work.state.pressure > 0,
                               base.state.pressure > 0)
                and np.array_equal(minus_work.state.pressure > 0,
                                   base.state.pressure > 0)
                and np.array_equal(plus_work.state.reference_support > 0,
                                   base.state.reference_support > 0)
                and np.array_equal(minus_work.state.reference_support > 0,
                                   base.state.reference_support > 0)
                and plus_work.state.overlap_signature
                    == base.state.overlap_signature
                and minus_work.state.overlap_signature
                    == base.state.overlap_signature)
            component = {}
            for label, rows in (("force", force_rows),
                                ("moment", moment_rows)):
                a = mapped_action[rows]
                b = mapped_finite[rows]
                scale = max(float(np.linalg.norm(a)),
                            float(np.linalg.norm(b)))
                component[f"mapped_{label}_action"] = scale
                component[f"mapped_{label}_relative_error"] = (
                    float(np.linalg.norm(a-b))/max(scale, 1.0e-12))
            runs.append({"step_m": step, "smooth_observed_branch": bool(smooth),
                         **component})
        report["runs"] = runs
        report["status"] = ("passed_development" if all(
            row["smooth_observed_branch"]
            and row["mapped_force_action"] > 1.0e-6
            and row["mapped_moment_action"] > 1.0e-8
            and row["mapped_force_relative_error"] <= 1.0e-5
            and row["mapped_moment_relative_error"] <= 1.0e-5
            for row in runs) else "inconclusive_or_failed_development")
    except Exception as exc:
        report.update({"status": "failed_development",
                       "error": f"{type(exc).__name__}: {exc}"})
    report["elapsed_seconds"] = time.perf_counter()-started
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--slope", type=float, default=1.0e-5)
    args = parser.parse_args()
    main(args.reference, args.output, args.slope)
