"""Development probe of pressure-overlaid gap response to real shell DOFs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from anyfesolver.general_contact.core import contact_scope
from anyfesolver.general_contact.pressure_overlay import subdivide_q4_pressure_faces
from anyfesolver.general_contact.s3_patch_augmented import MovingS3PatchProjectedNormal
from anyfesolver.general_contact.s3_patch_mortar_energy import (
    MovingS3PatchPrimalMortarPenalty,
)
from scripts.general_contact_d1_finite_area_load import build_case
from scripts.general_contact_d1_overlay_probe import _pose


def _rank(singular, relative):
    return int(np.count_nonzero(singular > relative*singular[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    divisions = 2
    model, _load, definition, settings = build_case(
        divisions, total_load_n=2.0, normal_stiffness=100_000.0,
        geometry_profile="central",
    )
    with contact_scope(model, definition, settings,
                       allow_augmented=True) as manager:
        original_source, target = _pose(model, manager, divisions)
        original = manager.law.geometry
        overlay = subdivide_q4_pressure_faces(
            original.source_faces, u_divisions=2, v_divisions=2,
            first_pressure_node_id=max((*original.source_node_ids,
                                        *original.target_node_ids)) + 1,
            reserved_node_ids=original.target_node_ids,
            maximum_pressure_nodes=25, maximum_pressure_faces=16,
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
            differentiation="jet2_q4_s3", maximum_candidate_pairs=32,
        )
        law = MovingS3PatchProjectedNormal(geometry)
        multiplier = np.zeros(len(law.source_node_ids))
        baseline = law.evaluate(
            overlay.interpolation @ original_source, target, multiplier,
        )
        supported = np.flatnonzero(baseline.reference_support > 0.0)
        if len(supported) == 0:
            raise ValueError("gap-rank probe has no supported pressure modes")

        def gaps(source):
            state = law.evaluate(
                overlay.interpolation @ source, target, multiplier,
            )
            if not np.array_equal(state.reference_support > 0.0,
                                  baseline.reference_support > 0.0):
                raise ValueError("pressure support changed during gap derivative")
            return state.normalized_gap[supported]

        rows = []
        virtual_map = sparse.kron(overlay.interpolation, sparse.eye(3),
                                  format="csr")
        real_map = manager.map[:3*len(original.source_node_ids), :]
        if overlay.structural_node_ids != original.source_node_ids:
            raise ValueError("overlay and material node order differ")
        free = []
        for node_id in original.source_node_ids:
            node = model.mesh.get_node(node_id)
            column = (node_id-1) % (divisions+1)
            row = (node_id-1) // (divisions+1)
            for axis, dof in enumerate(node.dofs):
                if axis != 2 or (row not in {0, divisions}
                                 and column not in {0, divisions}):
                    free.append(int(dof))
        structural_map = virtual_map @ real_map[:, free]
        for step in (1e-6, 5e-7):
            derivatives = np.empty((len(supported), original_source.size))
            flat = original_source.reshape(-1)
            for axis in range(flat.size):
                plus, minus = flat.copy(), flat.copy()
                plus[axis] += step
                minus[axis] -= step
                derivatives[:, axis] = (
                    gaps(plus.reshape(-1, 3))
                    - gaps(minus.reshape(-1, 3))
                ) / (2.0*step)
            # J_virtual is implicit in the source perturbation; real_map is
            # already in original-source coordinates, so do not apply T twice.
            jacobian = derivatives @ real_map[:, free]
            singular = np.linalg.svd(jacobian, compute_uv=False)
            rows.append({
                "difference_step_m": step,
                "gap_jacobian_shape": list(jacobian.shape),
                "singular_values": singular.tolist(),
                "rank_relative_1e_minus_5": _rank(singular, 1e-5),
                "rank_relative_1e_minus_7": _rank(singular, 1e-7),
                "rank_relative_1e_minus_9": _rank(singular, 1e-9),
            })
        record = {
            "profile": "fixed_pose_overlay_actual_structural_gap_rank_dev",
            "pressure_nodes": len(overlay.pressure_node_ids),
            "supported_pressure_modes": len(supported),
            "real_source_nodes": len(original.source_node_ids),
            "free_source_structural_dofs": len(free),
            "structural_map_shape": list(structural_map.shape),
            "elapsed_seconds": time.perf_counter() - started,
            "rows": rows,
        }
    args.output.write_text(json.dumps(record, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key != "rows"}, indent=2))
    for row in rows:
        print({key: value for key, value in row.items()
               if key != "singular_values"})


if __name__ == "__main__":
    main()
