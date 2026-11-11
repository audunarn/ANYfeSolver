"""Measure experimental instantaneous and swept bounds on a saved Q4/S3 pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.q4_source_overlap import q4_source_projected_gap_lower
from anyfesolver.general_contact.q4_overlap_local_bound import (
    overlap_local_lower,
    overlap_local_swept_lower,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_2d_target_press_dev import build_case


def evaluate(saved: Path) -> dict:
    data = json.loads(saved.read_text(encoding="utf-8"))
    source = np.asarray(data["current_source_positions_m"], dtype=float)
    target = np.asarray(data["current_target_positions_m"], dtype=float)
    model, definition, settings, _ = build_case()
    manager = GeneralS3PatchManager(model, definition, settings)
    if (len(source) != len(manager.law.source_node_ids)
            or len(target) != len(manager.law.target_node_ids)):
        raise ValueError("saved pose and current fixture topology differ")
    results = []
    zero_path_results = []
    for depth in range(6):
        bounds, visits = [], 0
        swept_bounds, swept_visits = [], 0
        limiting = None
        for si, sids in enumerate(manager.law.source_face_indices):
            first = source[np.asarray(sids, dtype=int)]
            for ti, tids in enumerate(manager.law.target_face_indices):
                second = target[np.asarray(tids, dtype=int)]
                bound, count = overlap_local_lower(first, second, depth)
                visits += count
                if np.isfinite(bound):
                    bounds.append(bound)
                    if limiting is None or bound < limiting[2]:
                        limiting = (si, ti, bound)
                swept, swept_count = overlap_local_swept_lower(
                    first, first, second, second, depth,
                )
                swept_visits += swept_count
                if np.isfinite(swept):
                    swept_bounds.append(swept)
        results.append({
            "depth": depth,
            "minimum_lower_gap_m": min(bounds) if bounds else float("inf"),
            "limiting_source_target_faces": list(limiting[:2]) if limiting else None,
            "pairs_with_possible_projected_overlap": len(bounds),
            "subrectangles_visited": visits,
        })
        zero_path_results.append({
            "depth": depth,
            "minimum_lower_gap_m": (min(swept_bounds)
                                    if swept_bounds else float("inf")),
            "pairs_with_possible_projected_overlap": len(swept_bounds),
            "subrectangles_visited": swept_visits,
        })
    worst = data["worst_pairs"][0]
    first = source[np.asarray(manager.law.source_face_indices[worst["source_face"]], dtype=int)]
    second = target[np.asarray(manager.law.target_face_indices[worst["target_face"]], dtype=int)]
    return {
        "candidate": "d1_saved_pose_experimental_instantaneous_overlap_local_gap",
        "saved_pose": str(saved),
        "production_reported_gap_lower_m": data["reported_lower_gap_m"],
        "registered_gap_tolerance_m": data["registered_gap_tolerance_m"],
        "production_worst_pair_lower_m": q4_source_projected_gap_lower(first, second),
        "depth_results": results,
        "zero_path_swept_depth_results": zero_path_results,
        "scope": "experimental Q4/S3 linear physical nodal paths; full solver path and production admission unverified",
    }


def main(saved: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    data = evaluate(saved)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["depth_results"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.saved, args.output)
