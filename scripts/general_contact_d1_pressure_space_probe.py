"""Exploratory fixed-pose Q4/S3 pressure-space refinement.

The affine physical gap is identical at every source mesh resolution.  The
continuous positive-part pressure integral is computed without a contact
kernel, so differences expose discrete pressure/integration resolution.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver.general_contact.core import contact_scope
from scripts.general_contact_d1_finite_area_load import build_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("divisions", type=int, nargs="+")
    parser.add_argument("--rows", type=int)
    args = parser.parse_args()
    started_total = time.perf_counter()

    target_x0, target_x1 = 0.24, 0.77
    target_y0, target_y1 = 0.21, 0.73
    midpoint = 0.5 * (target_x0 + target_x1)
    left_penetration = 25.0e-6
    slope = 2.0 * left_penetration / (target_x1 - target_x0)
    stiffness = 100_000.0
    # Integral of epsilon * [-slope*(x-midpoint)]_+ over the rectangle.
    reference_force_z = (
        stiffness * slope * (midpoint - target_x0)**2
        * (target_y1 - target_y0) / 2.0
    )
    row_divisions = args.rows
    rows = []
    for divisions in args.divisions:
        model, _load, definition, settings = build_case(
            divisions, total_load_n=2.0, normal_stiffness=stiffness,
            geometry_profile="central", rows=row_divisions,
        )
        with contact_scope(model, definition, settings,
                           allow_augmented=True) as manager:
            displacement = np.zeros(manager.n_dof)
            source_nodes = (divisions + 1) * (
                (divisions if row_divisions is None else row_divisions) + 1
            )
            for node_id in range(1, source_nodes + 1):
                node = model.mesh.get_node(node_id)
                displacement[node.dofs[2]] = (
                    -50.0e-6 + slope * (node.x - midpoint)
                )
            started = time.perf_counter()
            _force, _tangent, summary = manager.evaluate(
                displacement, tangent=False,
            )
            elapsed = time.perf_counter() - started
        reaction = float(summary.integration_force[2])
        source_force = np.asarray(summary.integration_force, dtype=float)
        target_force = np.asarray(summary.target_force, dtype=float)
        source_moment = np.asarray(summary.integration_moment, dtype=float)
        target_moment = np.asarray(summary.target_moment, dtype=float)
        rows.append({
            "divisions": divisions,
            "rows": divisions if row_divisions is None else row_divisions,
            "elapsed_seconds": elapsed,
            "reaction_z_n": reaction,
            "source_force_n": source_force.tolist(),
            "target_force_n": target_force.tolist(),
            "force_balance_norm_n": float(np.linalg.norm(
                source_force + target_force
            )),
            "source_moment_about_origin_nm": source_moment.tolist(),
            "target_moment_about_origin_nm": target_moment.tolist(),
            "moment_balance_norm_nm": float(np.linalg.norm(
                source_moment + target_moment
            )),
            "continuous_local_plane_reference_z_n": reference_force_z,
            "relative_reference_error": abs(reaction - reference_force_z)
                                        / reference_force_z,
            "minimum_gap_m": float(summary.minimum_gap),
            "sampled_overlap_gap_m": (
                None if summary.sampled_overlap_gap is None
                else float(summary.sampled_overlap_gap)
            ),
            "gap_tolerance_met": bool(summary.gap_tolerance_met),
            "active_pressure_dofs": int(summary.active_pressure_dofs),
            "supported_pressure_modes": int(summary.pressure_gram_supported_modes),
            "scaled_pressure_rank": int(summary.pressure_gram_scaled_rank),
            "source_pressure_node_ids": list(summary.source_pressure_node_ids),
            "numerical_pressure_pa": list(summary.numerical_pressure_pa),
            "reference_support_m2": list(summary.reference_support_m2),
            "overlapping_pairs": int(summary.overlapping_pairs),
            "integration_triangles": int(summary.integration_triangles),
            "area_integration_error_m2": float(summary.area_integration_error_m2),
            "weighted_gap_integration_error_m3": float(
                summary.weighted_gap_integration_error_m3
            ),
            "integration_force_error_indicator_n": stiffness * (
                float(summary.weighted_gap_integration_error_m3)
                + left_penetration * float(summary.area_integration_error_m2)
            ),
        })
    args.output.write_text(json.dumps({
        "profile": "fixed_affine_partial_pressure_central_q4_s3_dev",
        "target_x_range_m": [target_x0, target_x1],
        "target_y_range_m": [target_y0, target_y1],
        "left_penetration_m": left_penetration,
        "gap_slope": slope,
        "normal_stiffness_pa_per_m": stiffness,
        "reference_assumption": (
            "continuous local-plane positive-part pressure over the fixed "
            "target rectangle; finite-slope projection differences remain"
        ),
        "elapsed_seconds_total": time.perf_counter() - started_total,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
