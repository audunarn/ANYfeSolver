"""Screen the proposed interior pressure-work bound at a saved D1 pose.

This development runner evaluates the full signed interior Q1 space. It does
not freeze an acceptance candidate or establish nonlinear inf-sup stability.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import lsqr

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import linearize
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from scripts.general_contact_d1_interior_press_dev import build_interior_press
from scripts.general_contact_d1_interior_work_spectrum_dev import _reference_mass
from scripts.general_contact_d1_pressure_transport_preflight_dev import _smooth_pose


def screen(saved_path: Path, output: Path,
           analytic_cells: int | None = None,
           independent_check: bool = False,
           lifecycle: bool = False) -> None:
    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    expected = {
        "status": "completed",
        "source_support": "perimeter", "load_shape": "physical_bump",
        "four_phase": lifecycle, "slide_body": "target",
        "force_n": 0.04, "reference_slope": 1.0e-4,
        "target_young_multiplier": 5.0,
        "target_reference_full_left_clamp": True,
        "q4_gap_certificate": "overlap_local_dev",
    }
    if analytic_cells is None and any(
            saved.get(key) != value for key, value in expected.items()):
        raise ValueError("saved pose is outside the deformable contrast profile")
    if analytic_cells is None and (
            (not lifecycle and saved["accepted_steps"] != 2)
            or (lifecycle and (saved["accepted_steps"] < 4
                               or saved["max_path_fraction"] != 1.0))):
        raise ValueError("saved pressure stage is incomplete")
    if analytic_cells is not None and (
            analytic_cells not in (4, 8, 12)
            or saved.get("source_columns") != 12
            or saved.get("target_columns") != 4):
        raise ValueError("analytic pressure family or seed identity changed")
    columns = saved["source_columns"] if analytic_cells is None else analytic_cells
    rows = saved["source_rows"] if analytic_cells is None else analytic_cells
    target_columns, target_rows = saved["target_columns"], saved["target_rows"]
    if (columns != rows or columns not in (4, 6, 8, 12)
            or target_columns != target_rows or target_columns != 4
            or saved.get("source_y_profile") is not None):
        raise ValueError("saved pose is outside the pressure family")
    model, definition, settings, program = build_interior_press(
        force_n=0.04, reference_slope=1.0e-4,
        source_support="perimeter", source_columns=columns, source_rows=rows,
        target_columns=target_columns, target_rows=target_rows,
        target_young_multiplier=5.0,
        target_y0_m=(saved.get("target_y0_m", -0.1)
                     if analytic_cells is None else -0.113),
        target_reference_full_left_clamp=True,
        target_left_edge_motion=saved.get("target_left_edge_motion", False),
        load_shape="physical_bump", slide_body="target",
    )
    settings = replace(settings, q4_source_gap_certificate="overlap_local_dev")
    displacement = (np.asarray(saved["displacements"], dtype=float)
                    if analytic_cells is None else _smooth_pose(model, columns))
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    zeros = np.zeros(model.mesh.dof_manager.total_dofs)
    _, _, transformation, offset, _, _ = build_constraint_transformation(
        stiffness, zeros, model,
    )
    if displacement.shape != zeros.shape or np.linalg.norm(offset) > 1e-12:
        raise ValueError("saved pose or structural constraints changed")
    reduced_pose = lsqr(transformation, displacement,
                        atol=1e-14, btol=1e-14)[0]
    constraint_error = float(np.max(np.abs(
        displacement-transformation @ reduced_pose)))
    if constraint_error > 1e-12:
        raise ValueError(
            f"pressure-screen pose violates structural constraints: "
            f"{constraint_error:.6g}")
    manager = GeneralS3PatchManager(model, definition, settings)
    manager.restrict_tangent_to_reduced_variations(transformation)
    source_ids = manager.law.source_node_ids
    interior_ids = tuple(1+row*(columns+1)+column
                         for row in range(1, rows)
                         for column in range(1, columns))
    if (analytic_cells is None
            and tuple(saved["interior_pressure_node_ids"]) != interior_ids):
        raise ValueError("saved interior pressure identity changed")
    selected = np.array([source_ids.index(node) for node in interior_ids])
    contact = (saved["history"][-1]["contact"]
               if analytic_cells is None else None)
    if contact is not None:
        if tuple(contact["source_pressure_node_ids"]) != tuple(source_ids):
            raise ValueError("saved pressure order differs from the contact model")
        nodal_forces = (np.asarray(contact["numerical_pressure_pa"], dtype=float)
                        * np.asarray(contact["reference_support_m2"], dtype=float))
    else:
        nodal_forces = np.zeros(len(source_ids))
    state = linearize(manager, displacement, nodal_forces, tangent=False)
    if contact is not None:
        if state.state.active_pressure_dofs != contact["active_pressure_dofs"]:
            raise ValueError("saved and rebuilt active pressure counts differ")
        np.testing.assert_allclose(
            state.state.reference_support,
            np.asarray(contact["reference_support_m2"], dtype=float),
            rtol=1e-9, atol=1e-12,
        )
    support = np.asarray(state.state.reference_support)[selected]
    if np.any(support <= 0.0):
        raise ValueError("interior pressure support is nonpositive")
    mass = _reference_mass(manager)[np.ix_(selected, selected)]
    axis = np.zeros((columns+1, columns+1))
    for element in range(columns):
        axis[np.ix_((element, element+1), (element, element+1))] += (
            np.asarray(((2.0, 1.0), (1.0, 2.0)))/(6.0*columns))
    exact_mass = (np.kron(axis, axis)
                  * np.sqrt(1.0+1.0e-8))[np.ix_(
                      np.asarray(interior_ids)-1,
                      np.asarray(interior_ids)-1)]
    independent_mass_error = float(np.linalg.norm(mass-exact_mass)
                                   / np.linalg.norm(exact_mass))
    if independent_mass_error > 1e-11:
        raise ValueError("independent Q1 mass disagrees")
    mass_values, mass_vectors = linalg.eigh(mass)
    if mass_values[0] <= 0.0:
        raise ValueError("interior Q1 mass is singular")
    inverse_mass_root = ((mass_vectors / np.sqrt(mass_values))
                         @ mass_vectors.T)

    face = model.mesh.elements[definition.surfaces[0].element_ids[0]]
    points = np.asarray([(model.mesh.nodes[node].x,
                          model.mesh.nodes[node].y,
                          model.mesh.nodes[node].z)
                         for node in face.node_ids])
    normal = np.cross(points[1]-points[0], points[3]-points[0])
    normal /= np.linalg.norm(normal)
    if abs(normal[1]) > 1e-12 or abs(normal[0]+1e-4*normal[2]) > 1e-12:
        raise ValueError("source physical reference normal changed")
    constrained = transformation.tocsr()
    motions = []
    for node in interior_ids:
        columns_for_node = []
        for component in (0, 2):
            row = constrained[model.mesh.dof_manager.get_node_dofs(node)[component]]
            if row.nnz != 1 or abs(row.data[0]-1.0) > 1e-12:
                raise ValueError("source normal motion is constrained")
            columns_for_node.append(int(row.indices[0]))
        if columns_for_node[0] == columns_for_node[1]:
            raise ValueError("source normal coordinates are linked")
        motions.append(normal[0]*constrained[:, columns_for_node[0]].toarray().ravel()
                       + normal[2]*constrained[:, columns_for_node[1]].toarray().ravel())
    q = np.column_stack(motions)
    q_gram_error = float(np.linalg.norm(
        q.T @ q-np.eye(len(interior_ids)), ord=np.inf))
    if q_gram_error > 1e-12:
        raise ValueError("source-normal test space is not orthonormal")
    controls = np.zeros((len(program.controls), len(zeros)))
    for row, control in enumerate(program.controls):
        for node, component, weight in control.terms:
            controls[row, model.mesh.dof_manager.get_node_dofs(node)[
                {"ux": 0, "uy": 1, "uz": 2}[component]]] += weight
    control_error = float(np.max(np.abs(controls @ q), initial=0.0))
    if control_error > 1e-12:
        raise ValueError("source normal motion violates a path control")

    # The constructive dual bound uses the energy norm on the entire
    # structurally constrained and path-controlled space, not only on Q.
    reduced_controls = np.asarray(controls @ transformation)
    controlled_columns = []
    for row in reduced_controls:
        nonzero = np.flatnonzero(np.abs(row) > 1e-12)
        if len(nonzero) != 1 or abs(row[nonzero[0]]-1.0) > 1e-12:
            raise ValueError("path control is not one retained coordinate")
        controlled_columns.append(int(nonzero[0]))
    if len(set(controlled_columns)) != len(controlled_columns):
        raise ValueError("path controls are not independent")
    free_columns = np.setdiff1d(np.arange(transformation.shape[1]),
                                controlled_columns)
    full_transform = transformation[:, free_columns]
    full_stiffness = np.asarray(
        (full_transform.T @ stiffness @ full_transform).toarray())
    full_symmetry_error = float(np.linalg.norm(
        full_stiffness-full_stiffness.T, ord=np.inf)
        / max(np.linalg.norm(full_stiffness, ord=np.inf), 1.0))
    if full_symmetry_error > 1e-10:
        raise ValueError("full constrained reference stiffness is nonsymmetric")
    full_energy = 0.5*(full_stiffness+full_stiffness.T)
    full_cholesky = linalg.cholesky(full_energy, lower=True)

    b = support[:, None]*np.asarray(state.gap_gradient @ q)[selected]
    k = np.asarray(q.T @ stiffness @ q)
    k = 0.5*(k+k.T)
    linalg.cholesky(k, lower=True)
    a = inverse_mass_root @ b @ inverse_mass_root
    singular_values = linalg.svdvals(a)
    energy = inverse_mass_root @ k @ inverse_mass_root
    actual_dual = a @ linalg.solve(energy, a.T, assume_a="pos")
    actual_dual = 0.5*(actual_dual+actual_dual.T)
    actual_minimum = float(linalg.eigvalsh(
        actual_dual, subset_by_index=(0, 0))[0])
    maximum_energy = float(linalg.eigvalsh(
        energy, subset_by_index=(len(selected)-1, len(selected)-1))[0])
    if face.thickness != 0.05 or face.material_name != "elastic":
        raise ValueError("source section differs from registered fixture")
    rigidity = 1.0e5*face.thickness**3/(12*(1-0.3**2))
    hmin = min(1.0/columns, 1.0/rows)
    eta = float(singular_values[-1])
    kappa = maximum_energy*hmin**4/rigidity
    record = {
        "status": ("exploratory_saved_pose_pressure_screen"
                   if analytic_cells is None else
                   "exploratory_transport_pose_pressure_screen"),
        "saved_pose": str(saved_path),
        "pose_rule": ("saved_accepted_development_pose"
                      if analytic_cells is None else "smooth_analytic_press"),
        "lifecycle_recontact": lifecycle,
        "source_cells": [columns, rows],
        "target_cells": [target_columns, target_rows],
        "target_y0_m": (saved.get("target_y0_m", -0.1)
                        if analytic_cells is None else -0.113),
        "source_reference_normal": normal.tolist(),
        "signed_interior_modes": len(selected),
        "active_modes": state.state.active_pressure_dofs,
        "negative_gap_interior_modes": int(np.count_nonzero(
            np.asarray(state.state.normalized_gap)[selected] < 0.0)),
        "minimum_interior_gap_m": float(np.min(
            np.asarray(state.state.normalized_gap)[selected])),
        "reference_support_min_m2": float(np.min(support)),
        "reference_support_max_m2": float(np.max(support)),
        "mass_minimum_eigenvalue_m2": float(mass_values[0]),
        "independent_mass_relative_error": independent_mass_error,
        "q_orthonormal_gram_error": q_gram_error,
        "path_control_error": control_error,
        "structural_constraint_error": constraint_error,
        "full_constrained_energy_dimension": len(free_columns),
        "full_constrained_stiffness_symmetry_error": full_symmetry_error,
        "full_constrained_minimum_cholesky_pivot": float(np.min(
            np.diag(full_cholesky))),
        "eta": eta,
        "kappa": float(kappa),
        "gamma_constructive": float(eta**2/kappa),
        "gamma_restricted_actual": float(actual_minimum
                                         * rigidity/hmin**4),
        "provisional_eta_level": 0.9,
        "provisional_kappa_level": 2110.0,
        "provisional_levels_met": eta >= 0.9 and kappa <= 2110.0,
        "scope_note": "Reference-energy saved-pose screen only; no nonlinear norm equivalence or acceptance claim.",
    }
    if independent_check:
        from docs.d1_review_support.d1_two_body_current_skin_quadrature import (
            integrate_current_skin,
        )

        oracle_support, oracle_weighted, _area, count, signature = (
            integrate_current_skin(model, definition, displacement))
        oracle_gap = oracle_weighted/oracle_support
        production_gap = np.asarray(state.state.normalized_gap)
        if tuple(sorted(source_ids)) != tuple(source_ids):
            raise ValueError("independent oracle pressure ordering changed")
        _, _, right = linalg.svd(a)
        worst = q @ (inverse_mass_root @ right[-1])
        worst /= np.max(np.abs(worst))
        directions = (
            ("center_normal", q[:, len(selected)//2]),
            ("signed_checker", q @ np.asarray([
                (-1.0)**(i//(columns-1)+i%(columns-1))
                for i in range(len(selected))])),
            ("smallest_singular", worst),
        )
        comparisons = []
        for label, direction in directions:
            for step in (1.0e-7, 3.0e-7):
                plus = integrate_current_skin(
                    model, definition, displacement+step*direction)
                minus = integrate_current_skin(
                    model, definition, displacement-step*direction)
                if plus[4] != signature or minus[4] != signature:
                    raise ValueError(f"independent clip branch changed: {label}")
                independent = (oracle_support[selected]
                               * ((plus[1]/plus[0]-minus[1]/minus[0])
                                  / (2*step))[selected])
                production = (support
                              * np.asarray(state.gap_gradient @ direction)[selected])
                physical_work_derivative = ((plus[1]-minus[1])/(2*step))[selected]
                comparisons.append({
                    "direction": label, "step_m": step,
                    "algorithmic_map_relative_l2": float(np.linalg.norm(
                        independent-production)
                        / max(np.linalg.norm(independent),
                              np.linalg.norm(production), 1.0e-12)),
                    "fixed_pressure_work_extra_l2": float(np.linalg.norm(
                        physical_work_derivative-independent)),
                    "independent_map_l2_m2": float(np.linalg.norm(independent)),
                })
        record["independent_current_skin"] = {
            "quadrature_points": count,
            "branch_signature_sha256": hashlib.sha256(
                repr(signature).encode()).hexdigest(),
            "support_max_relative": float(np.max(np.abs(
                oracle_support-np.asarray(state.state.reference_support))
                / oracle_support)),
            "gap_max_absolute_m": float(np.max(np.abs(
                oracle_gap-production_gap))),
            "directions": comparisons,
        }
        record["independent_map_target"] = 1.0e-5
        record["independent_map_target_met"] = all(
            item["algorithmic_map_relative_l2"] <= 1.0e-5
            for item in comparisons)
        record["development_checks_met"] = (
            record["provisional_levels_met"]
            and record["independent_map_target_met"]
            and record["independent_current_skin"]["support_max_relative"] <= 1e-11)
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--analytic-cells", type=int, choices=(4, 8, 12))
    parser.add_argument("--independent-check", action="store_true")
    parser.add_argument("--lifecycle", action="store_true")
    args = parser.parse_args()
    if args.lifecycle and args.analytic_cells is not None:
        parser.error("lifecycle requires an actual accepted state")
    screen(args.saved, args.output, args.analytic_cells,
           args.independent_check, args.lifecycle)
