"""Development-only pressure/geometry comparison at one accepted central pose.

This does not re-equilibrate the structure or qualify a response reference.
The accepted 8x8 multipliers are interpolated as a pressure field only; each
variant reports its own projected residual and one frozen-pose update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver.general_contact.core import contact_scope
from anyfesolver.general_contact.pressure_overlay import subdivide_q4_pressure_faces
from anyfesolver.general_contact.pressure_space import supported_pressure_gram
from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty,
)
from scripts.general_contact_d1_finite_area_load import build_case


def _sample_grid(values: np.ndarray, x: float, y: float) -> np.ndarray:
    intervals = values.shape[0] - 1
    u = np.clip(x * intervals, 0.0, float(intervals))
    v = np.clip(y * intervals, 0.0, float(intervals))
    i = min(int(np.floor(u)), intervals - 1)
    j = min(int(np.floor(v)), intervals - 1)
    a, b = float(u - i), float(v - j)
    return (
        (1 - a) * (1 - b) * values[j, i]
        + a * (1 - b) * values[j, i + 1]
        + a * b * values[j + 1, i + 1]
        + (1 - a) * b * values[j + 1, i]
    )


def _state_row(label, law, source, target, multipliers, *, gap_lower):
    start = time.perf_counter()
    state = law.evaluate(source, target, multipliers)
    gram = supported_pressure_gram(state.pressure_gram,
                                   state.reference_support)
    projected = law.projected_multiplier(state, multipliers)
    residual = projected - multipliers
    supported = state.reference_support > 0.0
    after = law.evaluate(source, target, projected)
    row = {
        "label": label,
        "pressure_nodes": len(law.source_node_ids),
        "supported_modes": gram.supported_modes,
        "scaled_pressure_rank": gram.scaled_rank,
        "scaled_pressure_condition": gram.scaled_condition,
        "active_pressure_dofs": state.active_pressure_dofs,
        "after_one_update_active_pressure_dofs": after.active_pressure_dofs,
        "independent_gap_lower_m": float(gap_lower),
        "minimum_physical_gap_m": float(state.minimum_physical_gap),
        "overlap_area_m2": float(state.overlap_area),
        "reference_support_sum_m2": float(np.sum(state.reference_support)),
        "reference_measure_pressure_resultant_n": float(
            state.reference_support @ state.pressure
        ),
        "after_one_update_pressure_resultant_n": float(
            after.reference_support @ after.pressure
        ),
        "projected_residual_sup_pa": (
            float(np.max(np.abs(residual[supported])))
            if np.any(supported) else 0.0
        ),
        "projected_residual_support_l2_pa_m": float(np.sqrt(
            np.dot(state.reference_support, residual * residual)
        )),
        "pressure_gram_smallest_scaled_eigenvalue": (
            gram.smallest_scaled_eigenvalue
        ),
        "geometry_evaluation_seconds": time.perf_counter() - start,
    }
    try:
        work_start = time.perf_counter()
        work = law.work(source, target, multipliers, tangent=False)
        physical = -np.asarray(work.residual).reshape(-1, 3)
        source_count = len(law.source_node_ids)
        source_force = np.sum(physical[:source_count], axis=0)
        target_force = np.sum(physical[source_count:], axis=0)
        source_moment = np.sum(np.cross(
            source, physical[:source_count]
        ), axis=0)
        target_moment = np.sum(np.cross(
            target, physical[source_count:]
        ), axis=0)
        row.update({
            "physical_source_force_n": source_force.tolist(),
            "physical_target_force_n": target_force.tolist(),
            "physical_source_moment_nm": source_moment.tolist(),
            "physical_target_moment_nm": target_moment.tolist(),
            "physical_force_balance_n": float(np.linalg.norm(
                source_force + target_force
            )),
            "physical_moment_balance_nm": float(np.linalg.norm(
                source_moment + target_moment
            )),
            "work_evaluation_seconds": time.perf_counter() - work_start,
        })
    except (RuntimeError, ValueError) as exc:
        row["work_failure"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_record", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    prior = json.loads(arguments.pose_record.read_text(encoding="utf-8"))
    if (prior["status"] != "completed" or prior["divisions"] != 8
            or prior["geometry_profile"] != "central"
            or prior["total_load_n"] != 2.0
            or prior["normal_stiffness"] != 100_000.0
            or prior["accepted_steps"] != 1):
        raise ValueError("pose record is not the bounded 8x8 central case")
    displacement8 = np.asarray(prior["final_displacements"], dtype=float)
    multipliers8 = np.asarray(
        prior["general_contact"]["normal_multiplier_estimate_pa"],
        dtype=float,
    )
    model8, _load8, definition8, settings8 = build_case(
        8, total_load_n=2.0, normal_stiffness=100_000.0,
        geometry_profile="central", supported_schur=True,
        schur_mode_cap=64,
    )
    rows = []
    with contact_scope(model8, definition8, settings8,
                       allow_augmented=True) as manager8:
        if displacement8.shape != (manager8.n_dof,):
            raise ValueError("captured structural pose has wrong size")
        if multipliers8.shape != (len(manager8.law.source_node_ids),):
            raise ValueError("captured multiplier field has wrong size")
        source8, target8 = manager8._positions(displacement8)
        gap8 = manager8._gap_lower((source8, target8))
        rows.append(_state_row(
            "original_8x8", manager8.law, source8, target8,
            multipliers8, gap_lower=gap8,
        ))

        original = manager8.law.geometry
        overlay = subdivide_q4_pressure_faces(
            original.source_faces, u_divisions=2, v_divisions=2,
            first_pressure_node_id=max((*original.source_node_ids,
                                        *original.target_node_ids)) + 1,
            reserved_node_ids=original.target_node_ids,
            maximum_pressure_nodes=289,
            maximum_pressure_faces=256,
        )
        if overlay.structural_node_ids != manager8.law.source_node_ids:
            raise ValueError("pressure interpolation source order changed")
        geometry = MovingS3PatchPrimalMortarPenalty(
            overlay.faces, original.target_faces, original.stiffness,
            difference_step=original.difference_step,
            maximum_energy_evaluations=1_000_000,
            q4_source_area_tolerance=original.q4_source_area_tolerance,
            q4_source_weighted_gap_tolerance=(
                original.q4_source_weighted_gap_tolerance
            ),
            maximum_q4_source_integration_triangles=(
                original.maximum_q4_source_integration_triangles
            ),
            q4_source_derivative_tolerances=(
                original.q4_source_derivative_tolerances
            ),
            differentiation="jet2_q4_s3",
            maximum_candidate_pairs=512,
        )
        overlay_law = MovingS3PatchProjectedNormal(geometry)
        overlay_source = overlay.interpolation @ source8
        overlay_multipliers = np.asarray(
            overlay.interpolation @ multipliers8
        ).reshape(-1)
        rows.append(_state_row(
            "same_geometry_2x_pressure_overlay", overlay_law,
            overlay_source, target8, overlay_multipliers, gap_lower=gap8,
        ))

    model10, _load10, definition10, settings10 = build_case(
        10, total_load_n=2.0, normal_stiffness=100_000.0,
        geometry_profile="central", supported_schur=True,
        schur_mode_cap=64,
    )
    with contact_scope(model10, definition10, settings10,
                       allow_augmented=True) as manager10:
        pose8_grid = np.empty((9, 9, 6))
        lambda8_grid = np.empty((9, 9))
        for j in range(9):
            for i in range(9):
                node_id = 1 + j * 9 + i
                pose8_grid[j, i] = displacement8[
                    model8.mesh.get_node(node_id).dofs
                ]
                lambda8_grid[j, i] = multipliers8[node_id - 1]
        displacement10 = np.zeros(manager10.n_dof)
        multipliers10 = np.zeros(len(manager10.law.source_node_ids))
        for j in range(11):
            for i in range(11):
                node_id = 1 + j * 11 + i
                x, y = i / 10.0, j / 10.0
                displacement10[model10.mesh.get_node(node_id).dofs] = (
                    _sample_grid(pose8_grid, x, y)
                )
                multipliers10[node_id - 1] = _sample_grid(
                    lambda8_grid, x, y
                )
        source10, target10 = manager10._positions(displacement10)
        gap10 = manager10._gap_lower((source10, target10))
        rows.append(_state_row(
            "same_pose_sampled_on_10x10", manager10.law,
            source10, target10, multipliers10, gap_lower=gap10,
        ))

    record = {
        "profile": "development_frozen_8x8_pose_pressure_geometry_comparison",
        "pose_record": str(arguments.pose_record),
        "source_8x8_contact_force_n": float(
            prior["general_contact"]["integration_force"][2]
        ),
        "historical_10x10_minus_8x8_force_n": -0.004518326,
        "rows": rows,
        "interpretation_limit": (
            "The pressure multipliers are bilinearly prolonged, not "
            "re-equilibrated; pressure resultants use reference support. "
            "One projected update is evaluated only at the frozen pose."
        ),
    }
    arguments.output.write_text(json.dumps(record, indent=2) + "\n",
                                encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
