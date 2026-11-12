"""Localize a saved D1 slide tangent discrepancy to Q4/S3 pair geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.q4_s3_overlap_derivatives import (
    UnsupportedQ4S3Branch, q4_s3_overlap_derivatives,
)
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from scripts.general_contact_d1_interior_press_dev import build_interior_press


def main(pose_file: Path, output_file: Path):
    if output_file.exists():
        raise FileExistsError(output_file)
    fixture = json.loads(pose_file.read_text())
    model, definition, settings, _program = build_interior_press(
        source_support="perimeter", source_rows=4,
        load_shape="interior_distributed", force_n=0.04,
        reference_slope=1.0e-4, four_phase=True, slide_body="target",
    )
    manager = GeneralS3PatchManager(model, definition, settings)
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    zeros = np.zeros(model.mesh.dof_manager.total_dofs)
    transform = build_constraint_transformation(stiffness, zeros, model)[2]
    manager.restrict_tangent_to_reduced_variations(transform)
    displacement = np.asarray(fixture["displacements"])

    desired = np.zeros_like(displacement)
    source_dofs = model.mesh.dof_manager.get_node_dofs(13)
    target_lower = model.mesh.dof_manager.get_node_dofs(27)
    target_upper = model.mesh.dof_manager.get_node_dofs(28)
    desired[source_dofs[0]] = 0.25
    desired[source_dofs[2]] = 1.0
    desired[source_dofs[4]] = 0.2
    desired[target_lower[2]] = -0.7
    desired[target_lower[4]] = 0.15
    desired[target_upper[2]] = 0.3
    direction = np.asarray(transform @ (transform.T @ desired)).reshape(-1)
    physical_direction = np.asarray(manager.map @ direction).reshape(-1, 3)
    source_count = len(manager.law.source_node_ids)
    active_source_index = int(np.argmax(fixture["nodal_force_n"]))
    step_m = 5.0e-7
    options = dict(
        area_tolerance=settings.q4_source_area_tolerance,
        weighted_gap_tolerance=settings.q4_source_weighted_gap_tolerance,
        maximum_triangles=settings.maximum_q4_source_integration_triangles,
        allow_warped_source=True,
        support_gradient_tolerance=settings.q4_source_support_gradient_tolerance,
        support_hessian_tolerance=settings.q4_source_support_hessian_tolerance,
        weighted_gap_gradient_tolerance=settings.q4_source_weighted_gap_gradient_tolerance,
        weighted_gap_hessian_tolerance=settings.q4_source_weighted_gap_hessian_tolerance,
    )
    rows = []
    target_count = len(manager.law.target_face_indices)
    for pair_index, _law in enumerate(manager.law.geometry.pair_laws):
        source_face, target_face = divmod(pair_index, target_count)
        source_ids = manager.law.source_face_indices[source_face]
        if active_source_index not in source_ids:
            continue
        target_ids = manager.law.target_face_indices[target_face]
        reference = manager.law.geometry.source_faces[source_face].reference_positions
        local_direction = np.concatenate((
            physical_direction[source_ids],
            physical_direction[source_count:][target_ids],
        )).reshape(-1)
        derivatives = []
        for sign in (0.0, 1.0, -1.0):
            source, target = manager._positions(displacement+sign*step_m*direction)
            try:
                derivatives.append(q4_s3_overlap_derivatives(
                    reference, source[source_ids], target[target_ids], **options,
                ))
            except UnsupportedQ4S3Branch as error:
                if str(error) != "zero projected overlap":
                    raise
                derivatives.append(None)
        base, plus, minus = derivatives
        if base is None or plus is None or minus is None:
            rows.append({"pair_index": pair_index, "overlap_state": [
                item is not None for item in derivatives]})
            continue

        def directional_error(gradient, hessian):
            observed = (getattr(plus, gradient)-getattr(minus, gradient))/(2*step_m)
            predicted = np.einsum("ijk,k->ij", getattr(base, hessian),
                                  local_direction)
            row_errors = np.linalg.norm(observed-predicted, axis=1)
            scale = max(np.linalg.norm(observed),
                        np.linalg.norm(predicted), 1.0e-14)
            return {
                "relative": float(np.linalg.norm(observed-predicted)/scale),
                "observed_norm": float(np.linalg.norm(observed)),
                "predicted_norm": float(np.linalg.norm(predicted)),
                "row_errors": row_errors.tolist(),
                "worst_row_observed": observed[int(np.argmax(row_errors))].tolist(),
                "worst_row_predicted": predicted[int(np.argmax(row_errors))].tolist(),
            }

        rows.append({
            "pair_index": pair_index,
            "branch_same": (base.branch_signature == plus.branch_signature
                            == minus.branch_signature),
            "integration_triangles": [item.integration_triangles
                                      for item in derivatives],
            "support_hessian_directional_error": directional_error(
                "support_gradient", "support_hessian"),
            "weighted_gap_hessian_directional_error": directional_error(
                "weighted_gap_gradient", "weighted_gap_hessian"),
            "overlap_areas_m2": [item.overlap_area for item in derivatives],
        })
    payload = {"pose_file": str(pose_file), "step_m": step_m,
               "active_source_node": manager.law.source_node_ids[active_source_index],
               "pairs": rows}
    output_file.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"pairs": len(rows),
                      "branch_changes": sum(not row.get("branch_same", True) for row in rows),
                      "maximum_gap_hessian_error": max(
                          (row["weighted_gap_hessian_directional_error"]["relative"]
                           for row in rows if "weighted_gap_hessian_directional_error" in row),
                          default=0.0)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.pose, args.output)
