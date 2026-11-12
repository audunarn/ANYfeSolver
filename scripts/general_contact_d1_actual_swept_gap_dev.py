"""Compare experimental swept bounds on a saved rejected Newton proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_2d_target_press_dev import build_case
from scripts.general_contact_d1_overlap_gap_bound_dev import (
    overlap_local_swept_lower,
)


def _sample_minimum(source_faces, target_faces, source0, source1,
                    target0, target1, fraction):
    minimum = float("inf")
    hits = 0
    for time in np.linspace(0.0, fraction, 9):
        source = (1-time)*source0+time*source1
        target = (1-time)*target0+time*target1
        for source_face in source_faces:
            corners = source[np.asarray(source_face, dtype=int)]
            projection = np.cross(corners[1]-corners[0],
                                  corners[3]-corners[0])
            projection /= np.linalg.norm(projection)
            for target_face in target_faces:
                triangle = target[np.asarray(target_face, dtype=int)]
                axis = np.cross(triangle[1]-triangle[0],
                                triangle[2]-triangle[0])
                denominator = float(axis @ projection)
                if denominator <= 0.0:
                    raise ValueError("sampled target projection inverted")
                edge_matrix = np.stack((triangle[1]-triangle[0],
                                        triangle[2]-triangle[0]), axis=1)
                for u in np.linspace(0.0, 1.0, 7):
                    for v in np.linspace(0.0, 1.0, 7):
                        weights = np.asarray(((1-u)*(1-v), u*(1-v),
                                              u*v, (1-u)*v))
                        point = weights @ corners
                        distance = float(axis @ (point-triangle[0]))/denominator
                        target_point = point-distance*projection
                        bary = np.linalg.lstsq(
                            edge_matrix, target_point-triangle[0], rcond=None,
                        )[0]
                        if (min(bary) < -1.0e-10
                                or sum(bary) > 1.0+1.0e-10):
                            continue
                        du = ((1-v)*(corners[1]-corners[0])
                              + v*(corners[2]-corners[3]))
                        dv = ((1-u)*(corners[3]-corners[0])
                              + u*(corners[2]-corners[1]))
                        normal = np.cross(du, dv)
                        normal /= np.linalg.norm(normal)
                        minimum = min(minimum, float(normal @
                                                     (point-target_point)))
                        hits += 1
    return minimum, hits


def evaluate(saved: Path) -> dict:
    data = json.loads(saved.read_text(encoding="utf-8"))
    required = ("current_source_positions_m", "current_target_positions_m",
                "proposed_source_positions_m", "proposed_target_positions_m")
    if any(name not in data for name in required):
        raise ValueError("saved proposal lacks physical endpoints")
    s0, t0, s1, t1 = (np.asarray(data[name], dtype=float)
                      for name in required)
    model, definition, settings, _ = build_case()
    manager = GeneralS3PatchManager(model, definition, settings)
    source_faces = manager.law.source_face_indices
    target_faces = manager.law.target_face_indices
    results = []
    for fraction in (1.0, 0.5, 0.25, 0.125):
        se = s0 + fraction*(s1-s0)
        te = t0 + fraction*(t1-t0)
        for depth in (3, 4):
            bounds = []
            visited = 0
            limiting = None
            for si, sids in enumerate(source_faces):
                first = s0[np.asarray(sids, dtype=int)]
                second = se[np.asarray(sids, dtype=int)]
                for ti, tids in enumerate(target_faces):
                    third = t0[np.asarray(tids, dtype=int)]
                    fourth = te[np.asarray(tids, dtype=int)]
                    lower, count = overlap_local_swept_lower(
                        first, second, third, fourth, depth,
                    )
                    visited += count
                    if np.isfinite(lower):
                        bounds.append(lower)
                        if limiting is None or lower < limiting[2]:
                            limiting = (si, ti, lower)
            sampled, hits = _sample_minimum(
                source_faces, target_faces, s0, s1, t0, t1, fraction,
            ) if depth == 4 else (None, None)
            results.append({
                "fraction": fraction,
                "depth": depth,
                "swept_lower_gap_m": min(bounds) if bounds else float("inf"),
                "limiting_faces": list(limiting[:2]) if limiting else None,
                "subrectangles_visited": visited,
                "pairs_with_possible_overlap": len(bounds),
                "independent_sampled_gap_m": sampled,
                "independent_sampled_hits": hits,
                "admissible_under_unchanged_gap_budget": bool(
                    min(bounds) >= -manager.gap_tolerance if bounds else True
                ),
            })
    return {
        "candidate": "d1_actual_rejected_newton_proposal_swept_geometry_screen",
        "saved_proposal": str(saved),
        "production_swept_fraction": data["swept_fraction"],
        "registered_gap_tolerance_m": manager.gap_tolerance,
        "results": results,
        "scope": "development linear physical nodal path; samples are not a conservative certificate",
    }


def main(saved: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    payload = evaluate(saved)
    def numpy_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported result type: {type(value).__name__}")
    output.write_text(json.dumps(payload, indent=2, default=numpy_scalar) + "\n",
                      encoding="utf-8")
    print(json.dumps(payload["results"], default=numpy_scalar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.saved, args.output)
