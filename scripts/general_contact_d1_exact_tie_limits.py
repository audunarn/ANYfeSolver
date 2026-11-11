"""Local two-sided Q4/S3 derivative limits at the saved 50 mm slide tie."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver.general_contact.overlap import PlanarFace, planar_projected_overlap
from anyfesolver.general_contact.s3_patch_augmented import (
    _touching_projected_boundaries, _touching_shared_target_edge,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_coupled_refinement import build_case


def _ratio(difference, reference):
    return float(np.linalg.norm(difference) /
                 max(1.0, float(np.linalg.norm(reference))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    geometry = owner.law.geometry
    source = geometry.reference_source.copy()
    target = geometry.reference_target
    source[:, 0] += 0.05
    source[:, 2] -= 0.01504
    multipliers = np.zeros(len(geometry.source_node_ids))
    exact = owner.law.evaluate(source, target, multipliers)
    touching = []
    for source_face, source_ids in enumerate(geometry.source_face_indices):
        first = PlanarFace((1, 1), source[source_ids])
        for target_face, target_ids in enumerate(geometry.target_face_indices):
            second = PlanarFace((2, 1), target[target_ids])
            overlap = planar_projected_overlap(first, second)
            if not _touching_projected_boundaries(first, second):
                continue
            shared = set(map(int, target_ids))
            for other_ids in geometry.target_face_indices:
                other = shared.intersection(map(int, other_ids))
                if len(other) == 2:
                    shared = other
                    break
            touching.append({
                "source_face": source_face,
                "target_face": target_face,
                "overlap_area_m2": 0.0 if overlap is None else overlap.area,
                "shared_edge_owner": _touching_shared_target_edge(
                    first, target[target_ids], target_ids, shared,
                ),
            })
    source_count = len(source)
    coordinate_count = 3*(source_count+len(target))
    slide = np.zeros(coordinate_count)
    slide[:3*source_count:3] = 1.0
    slide /= np.linalg.norm(slide)
    normal = np.zeros(coordinate_count)
    normal[2:3*source_count:3] = 1.0
    normal /= np.linalg.norm(normal)

    rows = []
    for offset in (1.2e-7, 6.0e-8, 3.0e-8, 1.5e-8, 3.0e-9, 3.0e-10):
        candidates = {}
        records = {}
        for sign in (-1, 1):
            smooth = source.copy()
            smooth[:, 0] += sign*offset
            started = time.perf_counter()
            try:
                work = owner.law.work(smooth, target, multipliers,
                                      tangent=True)
            except (ValueError, MemoryError) as exc:
                records[str(sign)] = {
                    "status": "unresolved", "error": str(exc),
                    "elapsed_seconds": time.perf_counter()-started,
                }
                continue
            candidates[sign] = work
            records[str(sign)] = {
                "status": "smooth",
                "active_pressure_dofs": work.state.active_pressure_dofs,
                "minimum_sampled_gap_m": work.state.minimum_physical_gap,
                "force_norm": float(np.linalg.norm(work.residual)),
                "tangent_norm": float(np.linalg.norm(work.tangent)),
                "elapsed_seconds": time.perf_counter()-started,
            }
        row = {"offset_m": offset, "sides": records}
        if len(candidates) == 2:
            left, right = candidates[-1], candidates[1]
            row["two_sided_residual_difference"] = _ratio(
                right.residual-left.residual, right.residual,
            )
            row["two_sided_tangent_difference"] = _ratio(
                right.tangent-left.tangent, right.tangent,
            )
            row["two_sided_slide_action_difference"] = _ratio(
                (right.tangent-left.tangent) @ slide,
                right.tangent @ slide,
            )
            row["two_sided_normal_action_difference"] = _ratio(
                (right.tangent-left.tangent) @ normal,
                right.tangent @ normal,
            )
            row["residual_midpoint_norm"] = float(np.linalg.norm(
                0.5*(right.residual+left.residual)))
        rows.append(row)
    args.output.write_text(json.dumps({
        "profile": "local_exact_50mm_q4_s3_tie_limits_dev",
        "exact_active_pressure_dofs": exact.active_pressure_dofs,
        "exact_minimum_sampled_gap_m": exact.minimum_physical_gap,
        "positive_overlap_touches": touching,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print({key: value for key, value in row.items()
               if key != "sides"})
        print({key: value["status"] for key, value in row["sides"].items()})


if __name__ == "__main__":
    main()
