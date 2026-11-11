"""Exploratory fixed-pose comparison of Q4 pressure overlay and full mesh."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver.general_contact.core import contact_scope
from anyfesolver.general_contact.pressure_overlay import subdivide_q4_pressure_faces
from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty,
)
from scripts.general_contact_d1_finite_area_load import build_case


def _pose(model, manager, divisions):
    source_nodes = (divisions + 1)**2
    displacement = np.zeros(manager.n_dof)
    slope = 50.0e-6 / 0.53
    for node_id in range(1, source_nodes + 1):
        node = model.mesh.get_node(node_id)
        displacement[node.dofs[2]] = -50.0e-6 + slope*(node.x - 0.505)
    return manager._positions(displacement)


def _force(state):
    return float(np.dot(state.reference_support, state.pressure))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for divisions in (2, 4):
        model, _load, definition, settings = build_case(
            divisions, total_load_n=2.0, normal_stiffness=100_000.0,
            geometry_profile="central",
        )
        with contact_scope(model, definition, settings,
                           allow_augmented=True) as manager:
            source, target = _pose(model, manager, divisions)
            original = manager.law.geometry
            if divisions == 2:
                first_id = max((*original.source_node_ids,
                                *original.target_node_ids)) + 1
                overlay = subdivide_q4_pressure_faces(
                    original.source_faces, u_divisions=2, v_divisions=2,
                    first_pressure_node_id=first_id,
                    reserved_node_ids=original.target_node_ids,
                    maximum_pressure_nodes=25,
                    maximum_pressure_faces=16,
                )
                geometry = MovingS3PatchPrimalMortarPenalty(
                    overlay.faces, original.target_faces, original.stiffness,
                    difference_step=original.difference_step,
                    maximum_energy_evaluations=100_000,
                    q4_source_area_tolerance=original.q4_source_area_tolerance,
                    q4_source_weighted_gap_tolerance=(
                        original.q4_source_weighted_gap_tolerance),
                    maximum_q4_source_integration_triangles=(
                        original.maximum_q4_source_integration_triangles),
                    q4_source_derivative_tolerances=(
                        original.q4_source_derivative_tolerances),
                    differentiation="jet2_q4_s3",
                    maximum_candidate_pairs=32,
                )
                law = MovingS3PatchProjectedNormal(geometry)
                source = overlay.interpolation @ source
                pressure_nodes = len(overlay.pressure_node_ids)
            else:
                law = manager.law
                pressure_nodes = len(original.source_node_ids)
            started = time.perf_counter()
            state = law.evaluate(
                source, target, np.zeros(len(law.source_node_ids)),
            )
            elapsed = time.perf_counter() - started
            rows.append({
                "structural_divisions": divisions,
                "pressure_nodes": pressure_nodes,
                "force_n": _force(state),
                "supported_modes": int(np.count_nonzero(
                    state.reference_support > 0.0)),
                "active_modes": state.active_pressure_dofs,
                "elapsed_seconds": elapsed,
            })
    args.output.write_text(json.dumps({
        "profile": "fixed_pose_q4_pressure_overlay_development",
        "rows": rows,
        "relative_force_difference": abs(rows[0]["force_n"]-rows[1]["force_n"])
                                     / rows[1]["force_n"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
