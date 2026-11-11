"""Locate the conservative gap bottleneck at the rejected 2D target step."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

import anyfesolver
from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact.q4_source_overlap import (
    q4_source_projected_gap_lower,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager

from scripts.general_contact_d1_2d_target_press_dev import build_case


def _box_separation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.maximum(np.min(first[:, :2], axis=0)
                      - np.max(second[:, :2], axis=0), 0.0)
    right = np.maximum(np.min(second[:, :2], axis=0)
                       - np.max(first[:, :2], axis=0), 0.0)
    return float(np.linalg.norm(left + right))


def main(output: Path, *, overlap_local: bool = False) -> None:
    if output.exists():
        raise FileExistsError(output)
    model, definition, settings, program = build_case()
    if overlap_local:
        settings = replace(settings, q4_source_gap_certificate="overlap_local_dev")
    original = GeneralS3PatchManager.admissible_step_fraction
    last = {}

    def capture(self, start, end):
        result = original(self, start, end)
        last.update(manager=self, start=np.asarray(start).copy(),
                    end=np.asarray(end).copy(), fraction=float(result))
        return result

    GeneralS3PatchManager.admissible_step_fraction = capture
    try:
        result = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
        )
    finally:
        GeneralS3PatchManager.admissible_step_fraction = original
    if not last:
        raise ValueError("contact solver did not request a swept fraction")
    manager = last["manager"]
    source, target = manager._positions(last["start"])
    proposed_source, proposed_target = manager._positions(last["end"])
    pair_data = []
    for source_index, source_face in enumerate(manager.law.source_face_indices):
        for target_index, target_face in enumerate(manager.law.target_face_indices):
            first = source[np.asarray(source_face, dtype=int)]
            second = target[np.asarray(target_face, dtype=int)]
            pair_data.append({
                "source_face": source_index,
                "target_face": target_index,
                "bound_m": q4_source_projected_gap_lower(first, second),
                "reference_xy_box_separation_m": _box_separation(first, second),
                "source_height_span_m": float(np.ptp(first[:, 2])),
                "target_height_span_m": float(np.ptp(second[:, 2])),
            })
    pair_data.sort(key=lambda item: item["bound_m"])
    payload = {
        "candidate": "d1_2d_target_conservative_gap_pair_discriminator",
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "status": result.status,
        "certificate": settings.q4_source_gap_certificate,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "swept_fraction": last["fraction"],
        "reported_lower_gap_m": manager._gap_lower((source, target)),
        "registered_gap_tolerance_m": manager.gap_tolerance,
        "source_reference_node_ids": [int(node) for node in manager.law.source_node_ids],
        "target_reference_node_ids": [int(node) for node in manager.law.target_node_ids],
        "current_source_positions_m": source.tolist(),
        "current_target_positions_m": target.tolist(),
        "proposed_source_positions_m": proposed_source.tolist(),
        "proposed_target_positions_m": proposed_target.tolist(),
        "worst_pairs": pair_data[:16],
        "pairs_total": len(pair_data),
        "pairs_with_negative_bound": int(sum(item["bound_m"] < 0.0 for item in pair_data)),
        "pairs_with_disjoint_xy_boxes_and_negative_bound": int(sum(
            item["bound_m"] < 0.0
            and item["reference_xy_box_separation_m"] > 0.0
            for item in pair_data
        )),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "failure_reason", "swept_fraction", "reported_lower_gap_m",
        "pairs_total", "pairs_with_negative_bound",
        "pairs_with_disjoint_xy_boxes_and_negative_bound",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overlap-local", action="store_true")
    args = parser.parse_args()
    main(args.output, overlap_local=args.overlap_local)
