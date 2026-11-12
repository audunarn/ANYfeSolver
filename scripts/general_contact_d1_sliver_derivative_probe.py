"""Development probe of a positive Q4/S3 diagonal wedge near zero area."""

import json

import numpy as np

from general_contact_d1_coupled_refinement import build_case
from anyfesolver.general_contact.q4_s3_overlap_derivatives import (
    UnsupportedQ4S3Branch, q4_s3_overlap_derivatives,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def main():
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
        four_phase=True,
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    u = np.zeros(manager.n_dof)
    for node_id in range(1, 16):
        dofs = model.mesh.get_node(node_id).dofs
        u[dofs[0]] = 0.05-3.0e-8
        u[dofs[2]] = -0.01504
    source, target = manager._positions(u)
    index = 2
    source_index = index // len(manager.law.geometry.target_faces)
    target_index = index % len(manager.law.geometry.target_faces)
    source_face = source[manager.law.source_face_indices[source_index]]
    target_face = target[manager.law.target_face_indices[target_index]]
    law = manager.law.geometry.pair_laws[index]
    reference = manager.law.geometry.source_faces[source_index].reference_positions
    local = law.evaluate(source_face, target_face)
    cases = []
    for tolerance in (1e-10, 1e-14, 1e-16, 1e-17, 1e-18):
        try:
            derivative = q4_s3_overlap_derivatives(
                reference, source_face, target_face,
                area_tolerance=settings.q4_source_area_tolerance,
                weighted_gap_tolerance=settings.q4_source_weighted_gap_tolerance,
                maximum_triangles=settings.maximum_q4_source_integration_triangles,
                relative_branch_tolerance=tolerance,
                allow_warped_source=True,
                support_gradient_tolerance=settings.q4_source_support_gradient_tolerance,
                support_hessian_tolerance=settings.q4_source_support_hessian_tolerance,
                weighted_gap_gradient_tolerance=settings.q4_source_weighted_gap_gradient_tolerance,
                weighted_gap_hessian_tolerance=settings.q4_source_weighted_gap_hessian_tolerance,
            )
            cases.append({
                "relative_branch_tolerance": tolerance,
                "status": "evaluated",
                "support_error_m2": float(np.max(np.abs(
                    derivative.reference_support-local.reference_support))),
                "weighted_gap_error_m3": float(np.max(np.abs(
                    derivative.reference_weighted_gap-local.reference_weighted_gap))),
                "maximum_support_gradient": float(np.max(np.abs(
                    derivative.support_gradient))),
                "maximum_support_hessian": float(np.max(np.abs(
                    derivative.support_hessian))),
            })
        except UnsupportedQ4S3Branch as exc:
            cases.append({"relative_branch_tolerance": tolerance,
                          "status": "unsupported", "reason": str(exc)})
    print(json.dumps({"pair": index, "overlap_area_m2": local.overlap_area,
                      "support_sum_m2": float(np.sum(local.reference_support)),
                      "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
