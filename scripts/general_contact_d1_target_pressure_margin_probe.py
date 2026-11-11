"""Explain target-motion active-set changes at the saved 24-Q4 pose."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case


def main(reference_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    saved = json.loads(reference_path.read_text(encoding="utf-8"))
    model, definition, settings, _ = build_case(
        24, four_phase=True, steps_per_segment=3, load_profile="distributed",
        total_force_n=0.3, target_right_x_m=1.3)
    settings = replace(settings, augmentation_schur_maximum_pressure_pa=250,
                       augmentation_schur_maximum_step_pa=200)
    manager = GeneralS3PatchManager(model, definition, settings)
    source, target = manager._positions(np.asarray(
        saved["final_displacements"], dtype=float))
    law = manager.law
    multipliers = np.asarray(saved["contact"]["normal_multiplier_estimate_pa"],
                             dtype=float)
    base = law.evaluate(source, target, multipliers)
    directions = {
        "tilt": np.array([[0, 0, 0], [0, 0, .5],
                          [0, 0, .5], [0, 0, 0]], dtype=float),
        "rigid_z": np.array([[0, 0, .5]]*4, dtype=float),
    }
    rows = []
    base_active = base.pressure > 0
    for name, direction in directions.items():
        for h in (1e-6, 5e-7):
            for sign in (-1, 1):
                state = law.evaluate(source, target+sign*h*direction,
                                     multipliers)
                active = state.pressure > 0
                rows.append({
                    "direction": name, "step_m": sign*h,
                    "changed_pressure_nodes": np.flatnonzero(
                        active != base_active).tolist(),
                    "active_count": int(np.sum(active)),
                    "support_count": int(np.sum(
                        state.reference_support > 0)),
                    "max_normalized_gap_change_m": float(np.max(np.abs(
                        state.normalized_gap-base.normalized_gap))),
                })
    report = {
        "reference": str(reference_path),
        "base_active_count": int(np.sum(base_active)),
        "base_active_indices": np.flatnonzero(base_active).tolist(),
        "rows": rows,
    }
    output_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"base_active_count": report["base_active_count"],
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: target_pressure_margin_probe.py REF OUTPUT")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
