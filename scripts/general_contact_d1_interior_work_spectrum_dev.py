"""Screen the full interior Q1 pressure work at an accepted D1 press pose.

This is a finite-mesh development diagnostic, not an inf-sup certificate.
The production gap gradient is compared with a separately clipped,
frozen-footprint vertical-work oracle from the retained reference programme.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from docs.d1_review_support.d1_two_body_reference_oracle import independent_gap_work
from scripts.general_contact_d1_interior_press_dev import build_interior_press


_Q1_MASS = np.array(((4, 2, 1, 2), (2, 4, 2, 1),
                     (1, 2, 4, 2), (2, 1, 2, 4)), dtype=float)/36.0
_DOF = {"ux": 0, "uy": 1, "uz": 2}


def _reference_mass(manager):
    """Exact Q1 pressure mass on the planar reference Q4 faces."""
    count = len(manager.law.source_node_ids)
    mass = np.zeros((count, count))
    for local_ids in manager.law.source_face_indices:
        points = manager.reference[0][local_ids]
        v1 = points[1]-points[0]
        v2 = points[3]-points[0]
        area = float(np.linalg.norm(np.cross(v1, v2)))
        plane_error = abs(float(np.dot(points[2]-points[0],
                                        np.cross(v1, v2))))/area
        if area <= 0.0 or plane_error > 1.0e-10:
            raise ValueError("pressure mass requires positive planar Q4 faces")
        mass[np.ix_(local_ids, local_ids)] += area*_Q1_MASS
    return mass


def evaluate(path: Path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = {"status": "completed", "accepted_steps": 2,
                "source_support": "perimeter",
                "load_shape": "interior_distributed", "four_phase": False,
                "slide_body": "target", "force_n": 0.04,
                "reference_slope": 1.0e-4}
    if (any(saved.get(key) != value for key, value in expected.items())
            or saved.get("source_rows") not in (4, 8, 16)
            or saved.get("target_s3_faces") not in (2, 4)):
        raise ValueError(f"incompatible accepted press pose: {path}")
    if saved["source_rows"] == 16 and (
            saved.get("source_columns") != 8
            or saved.get("source_y_profile") != "uniform_ref16"
            or saved.get("offline_structural_reference_y16") is not True):
        raise ValueError("sixteen-row spectrum requires saved offline identity")
    model, definition, settings, program = build_interior_press(
        source_support="perimeter", source_rows=saved["source_rows"],
        source_columns=saved["source_columns"],
        target_columns=saved["target_s3_faces"]//2,
        source_y_profile=saved.get("source_y_profile"),
        offline_structural_reference_y16=saved.get(
            "offline_structural_reference_y16", False),
        load_shape="interior_distributed", force_n=0.04,
        reference_slope=1.0e-4, slide_body="target",
    )
    displacement = np.asarray(saved["displacements"], dtype=float)
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    zeros = np.zeros(model.mesh.dof_manager.total_dofs)
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, zeros, model)
    if displacement.shape != zeros.shape or np.linalg.norm(offset) > 1.0e-12:
        raise ValueError("saved press pose or structural constraints changed")
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.restrict_tangent_to_reduced_variations(transform)
    state = linearize(manager, displacement,
                      np.zeros(len(manager.law.source_node_ids)), tangent=False)
    controls = np.zeros((len(program.controls), len(zeros)))
    for index, control in enumerate(program.controls):
        for node, name, weight in control.terms:
            controls[index, model.mesh.dof_manager.get_node_dofs(node)[
                _DOF[name]]] += weight
    reduced_controls = np.asarray(controls @ transform)
    null = linalg.null_space(reduced_controls)
    variations = np.asarray(transform @ null)
    reduced_stiffness = np.asarray(variations.T @ stiffness @ variations)
    symmetry_error = float(np.linalg.norm(
        reduced_stiffness-reduced_stiffness.T)
        / max(np.linalg.norm(reduced_stiffness), 1.0))
    if symmetry_error > 1.0e-9:
        raise ValueError("reference structural stiffness is nonsymmetric")
    reduced_stiffness = 0.5*(reduced_stiffness+reduced_stiffness.T)
    stiffness_min = float(linalg.eigvalsh(reduced_stiffness,
                                         subset_by_index=(0, 0))[0])
    if stiffness_min <= 0.0:
        raise ValueError("constrained structural reference has a mechanism")

    source_ids = manager.law.source_node_ids
    selected = np.array([source_ids.index(node)
                         for node in saved["interior_pressure_node_ids"]],
                        dtype=int)
    supports = state.state.reference_support[selected]
    if np.any(supports <= 0.0):
        raise ValueError("interior pressure mode lacks support")
    mass = _reference_mass(manager)[np.ix_(selected, selected)]
    work = (supports[:, None]
            * np.asarray(state.gap_gradient @ variations)[selected])
    response = linalg.solve(reduced_stiffness, work.T, assume_a="pos")
    dual = 0.5*(work @ response + response.T @ work.T)
    values, vectors = linalg.eigh(dual, mass)

    # This oracle is intentionally frozen to reference XY. Physical skin
    # directors shift current Q4 rows, outside its admitted horizontal-row
    # construction. Compare operator trends, never call this moving-work parity.
    oracle_ids, oracle_support, oracle_work, oracle_area = independent_gap_work(
        model, definition, 0.0)
    if tuple(source_ids) != oracle_ids:
        raise ValueError("independent pressure node order changed")
    independent_work = np.asarray(oracle_work @ variations)[selected]
    independent_response = linalg.solve(reduced_stiffness,
                                        independent_work.T, assume_a="pos")
    independent_dual = 0.5*(
        independent_work @ independent_response
        + independent_response.T @ independent_work.T)
    independent_values = linalg.eigh(independent_dual, mass,
                                     eigvals_only=True)

    # A pressure functional's dual norm over all admissible structural
    # variations is at least its dual norm over this explicit source-motion
    # subspace. Select columns through the actual constraint transform so
    # support equations and any linked coordinates remain intact.
    constrained = transform.tocsr()
    source_vertical_columns = []
    for node in saved["interior_pressure_node_ids"]:
        dof = model.mesh.dof_manager.get_node_dofs(node)[2]
        row = constrained[dof]
        if row.nnz != 1 or abs(row.data[0]-1.0) > 1.0e-12:
            raise ValueError("interior source vertical test motion is constrained")
        source_vertical_columns.append(int(row.indices[0]))
    if len(set(source_vertical_columns)) != len(source_vertical_columns):
        raise ValueError("source vertical test motions are not independent")
    test_space = np.asarray(constrained[:, source_vertical_columns].toarray())
    if np.max(np.abs(controls @ test_space)) > 1.0e-12:
        raise ValueError("source vertical test motions violate path controls")
    test_stiffness = np.asarray(test_space.T @ stiffness @ test_space)
    test_stiffness = 0.5*(test_stiffness+test_stiffness.T)
    linalg.cholesky(test_stiffness, lower=True)
    restricted_work = (supports[:, None]
                       * np.asarray(state.gap_gradient @ test_space)[selected])
    restricted_response = linalg.solve(test_stiffness,
                                       restricted_work.T, assume_a="pos")
    restricted_dual = 0.5*(restricted_work @ restricted_response
                           + restricted_response.T @ restricted_work.T)
    restricted_min = float(linalg.eigh(restricted_dual, mass,
                                       eigvals_only=True)[0])
    mass_values, mass_vectors = linalg.eigh(mass)
    if mass_values[0] <= 0.0:
        raise ValueError("interior pressure mass is not positive")
    inverse_mass_root = ((mass_vectors / np.sqrt(mass_values))
                         @ mass_vectors.T)
    normalized_work = (inverse_mass_root @ restricted_work
                       @ inverse_mass_root)
    work_detection_minimum = float(linalg.svdvals(normalized_work)[-1])
    normalized_stiffness = (inverse_mass_root @ test_stiffness
                            @ inverse_mass_root)
    stiffness_upper = float(linalg.eigvalsh(
        normalized_stiffness, subset_by_index=(len(selected)-1,
                                               len(selected)-1))[0])
    flexural_rigidity = 1.0e5*0.05**3/(12*(1-0.3**2))
    hmin = min(1/saved["source_columns"], 1/saved["source_rows"])
    scaled_stiffness_upper = stiffness_upper*hmin**4/flexural_rigidity
    constructive_scaled_lower = (
        work_detection_minimum**2/scaled_stiffness_upper)
    actual_scaled_lower = restricted_min*flexural_rigidity/hmin**4
    if constructive_scaled_lower > actual_scaled_lower*(1.0+1.0e-8):
        raise ValueError("constructive pressure bound exceeds actual response")
    independent_restricted_work = np.asarray(
        oracle_work @ test_space)[selected]
    independent_restricted_response = linalg.solve(
        test_stiffness, independent_restricted_work.T, assume_a="pos")
    independent_restricted_dual = 0.5*(
        independent_restricted_work @ independent_restricted_response
        + independent_restricted_response.T @ independent_restricted_work.T)
    independent_restricted_min = float(linalg.eigh(
        independent_restricted_dual, mass, eigvals_only=True)[0])
    if (restricted_min <= 0.0
            or restricted_min > values[0]*(1.0+1.0e-7)
            or independent_restricted_min <= 0.0
            or independent_restricted_min > independent_values[0]*(1.0+1.0e-7)):
        raise ValueError("source-motion lower-bound inequality failed")
    # Rotate the explicit source-motion subspace into the common physical
    # reference normal. This removes a global-coordinate tilt from the
    # source energy screen without changing the production work map.
    slope = float(saved["reference_slope"])
    normal = np.asarray((-slope, 0.0, 1.0))
    normal /= np.linalg.norm(normal)
    normal_columns = []
    for node in saved["interior_pressure_node_ids"]:
        rows = [constrained[model.mesh.dof_manager.get_node_dofs(node)[axis]]
                for axis in (0, 2)]
        if any(row.nnz != 1 or abs(row.data[0]-1.0) > 1e-12
               for row in rows):
            raise ValueError("interior source normal test motion is constrained")
        if rows[0].indices[0] == rows[1].indices[0]:
            raise ValueError("source normal components share one structural coordinate")
        normal_columns.append(
            normal[0]*constrained[:, rows[0].indices[0]].toarray().ravel()
            + normal[2]*constrained[:, rows[1].indices[0]].toarray().ravel()
        )
    normal_space = np.column_stack(normal_columns)
    if np.max(np.abs(controls @ normal_space)) > 1e-12:
        raise ValueError("source normal test motions violate path controls")
    normal_stiffness = np.asarray(normal_space.T @ stiffness @ normal_space)
    normal_stiffness = 0.5*(normal_stiffness+normal_stiffness.T)
    linalg.cholesky(normal_stiffness, lower=True)
    normal_work = (supports[:, None]
                   * np.asarray(state.gap_gradient @ normal_space)[selected])
    normal_eta = float(linalg.svdvals(
        inverse_mass_root @ normal_work @ inverse_mass_root)[-1])
    normal_kappa = float(linalg.eigvalsh(
        inverse_mass_root @ normal_stiffness @ inverse_mass_root,
        subset_by_index=(len(selected)-1, len(selected)-1))[0]
        * hmin**4/flexural_rigidity)
    normal_response = linalg.solve(normal_stiffness, normal_work.T,
                                   assume_a="pos")
    normal_dual = 0.5*(normal_work @ normal_response
                       + normal_response.T @ normal_work.T)
    normal_restricted_min = float(linalg.eigh(normal_dual, mass,
                                              eigvals_only=True)[0]
                                  * flexural_rigidity/hmin**4)
    normal_constructive = normal_eta**2/normal_kappa
    if normal_constructive > normal_restricted_min*(1.0+1e-8):
        raise ValueError("normal-motion constructive inequality failed")
    active_ids = [node for node, pressure in zip(
        saved["history"][-1]["contact"]["source_pressure_node_ids"],
        saved["history"][-1]["contact"]["numerical_pressure_pa"])
        if pressure > 0.0]
    return {
        "input": str(path),
        "source_q4_faces": saved["source_q4_faces"],
        "target_s3_faces": saved["target_s3_faces"],
        "active_pressure_node_ids": active_ids,
        "signed_interior_mode_ids": list(saved["interior_pressure_node_ids"]),
        "constrained_structural_coordinates": len(null.T),
        "structural_stiffness_minimum": stiffness_min,
        "structural_symmetry_error": symmetry_error,
        "reference_mass_minimum": float(np.min(linalg.eigvalsh(mass))),
        "production_reference_support_m2": supports.tolist(),
        "independent_frozen_support_m2": oracle_support[selected].tolist(),
        "independent_frozen_overlap_area_m2": oracle_area,
        "production_work_singular_values_squared": values.tolist(),
        "admissible_source_vertical_test_columns": len(source_vertical_columns),
        "source_vertical_restricted_lower_bound": restricted_min,
        "normalized_source_vertical_work_minimum": work_detection_minimum,
        "scaled_source_vertical_stiffness_upper": scaled_stiffness_upper,
        "constructive_scaled_source_vertical_lower_bound": (
            constructive_scaled_lower),
        "actual_scaled_source_vertical_lower_bound": actual_scaled_lower,
        "source_reference_normal": normal.tolist(),
        "normalized_source_normal_work_minimum": normal_eta,
        "scaled_source_normal_stiffness_upper": normal_kappa,
        "constructive_scaled_source_normal_lower_bound": normal_constructive,
        "actual_scaled_source_normal_lower_bound": normal_restricted_min,
        "independent_source_vertical_restricted_lower_bound": (
            independent_restricted_min),
        "independent_vertical_work_singular_values_squared": (
            independent_values.tolist()),
        "smallest_signed_mode_mass_normalized": vectors[:, 0].tolist(),
    }


def main(inputs: list[Path], output: Path):
    if output.exists():
        raise FileExistsError(output)
    cases = [evaluate(path) for path in inputs]
    output.write_text(json.dumps({"cases": cases}, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps({"cases": [{
        "source_q4_faces": case["source_q4_faces"],
        "interior_modes": len(case["signed_interior_mode_ids"]),
        "minimum_work": case["production_work_singular_values_squared"][0],
        "oracle_minimum_work": (
            case["independent_vertical_work_singular_values_squared"][0]),
    } for case in cases]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    main(args.inputs, args.output)
