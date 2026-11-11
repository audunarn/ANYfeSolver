"""Inspect adjacent S3 triangle ownership at a Q4 sliding transition."""

import json

import numpy as np

from general_contact_d1_coupled_refinement import build_case
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def main():
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
        four_phase=True,
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    source_count = 15
    rows = []
    for shift in (0.05-1e-10, 0.05-1e-12, 0.05,
                  0.05+1e-12, 0.05+1e-10):
        u = np.zeros(manager.n_dof)
        for node_id in range(1, source_count+1):
            dofs = model.mesh.get_node(node_id).dofs
            u[dofs[0]] = shift
            u[dofs[2]] = -0.01504
        source, target = manager._positions(u)
        pair_areas = []
        for index in (2, 3):
            si = index // len(manager.law.geometry.target_faces)
            ti = index % len(manager.law.geometry.target_faces)
            source_face = source[manager.law.source_face_indices[si]]
            target_face = target[manager.law.target_face_indices[ti]]
            value = manager.law.geometry.pair_laws[index].evaluate(
                source_face, target_face,
            )
            pair_areas.append({
                "pair": index,
                "overlap_area_m2": value.overlap_area,
                "reference_overlap_area_m2": value.reference_overlap_area,
                "support_sum_m2": float(np.sum(value.reference_support)),
            })
        rows.append({"shift_m": shift, "pairs": pair_areas})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
