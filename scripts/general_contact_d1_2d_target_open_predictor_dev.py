"""Contact-free first-step discriminator for the 2D S3 target press fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import anyfesolver
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.matrix_assembly import assemble_stiffness_matrix

from scripts.general_contact_d1_2d_target_press_dev import build_case


_DOF = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}


def evaluate() -> dict:
    model, definition, settings, program = build_case()
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    ndof = model.mesh.dof_manager.total_dofs
    external = np.zeros(ndof)
    for item, value in zip(program.nodal_forces, program.force_values_at(1)):
        dof = model.mesh.dof_manager.get_node_dofs(item.node_id)[_DOF[item.dof]]
        external[dof] += value
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, external, model,
    )
    if np.linalg.norm(offset) > 1.0e-12:
        raise ValueError("contact-free predictor expects homogeneous fixed supports")
    rows = np.zeros((len(program.controls), ndof))
    for index, control in enumerate(program.controls):
        for node, name, weight in control.terms:
            dof = model.mesh.dof_manager.get_node_dofs(node)[_DOF[name]]
            rows[index, dof] += weight
    controlled = sparse.csr_matrix(rows @ transform)
    reduced = (transform.T @ stiffness @ transform).tocsr()
    rhs = np.asarray(transform.T @ external).reshape(-1)
    targets = program.target_at(1) - rows @ offset
    saddle = sparse.bmat((
        (reduced, -controlled.T),
        (controlled, sparse.csr_matrix((len(targets), len(targets)))),
    ), format="csr")
    solution = np.asarray(spsolve(saddle, np.r_[rhs, targets]))
    if not np.all(np.isfinite(solution)):
        raise ValueError("contact-free structural predictor has a mechanism")
    coordinates = solution[:reduced.shape[0]]
    reactions = solution[reduced.shape[0]:]
    displacement = np.asarray(transform @ coordinates + offset).reshape(-1)
    equilibrium_error = np.linalg.norm(
        reduced @ coordinates - controlled.T @ reactions - rhs,
    ) / max(np.linalg.norm(rhs), 1.0e-12)
    manager = GeneralS3PatchManager(model, definition, settings)
    initial = manager._positions(np.zeros(ndof))
    current = manager._positions(displacement)
    return {
        "candidate": "d1_2d_target_contact_free_linear_first_step_discriminator",
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "structural_model": "linear_reference_stiffness_only",
        "path_step": 1,
        "applied_force_n": float(np.sum(program.force_values_at(1))),
        "control_targets_m": targets.tolist(),
        "control_reactions_n": reactions.tolist(),
        "linear_reduced_equilibrium_relative_error": float(equilibrium_error),
        "reference_conservative_gap_lower_m": float(manager._gap_lower(initial)),
        "predicted_conservative_gap_lower_m": float(manager._gap_lower(current)),
        "registered_gap_tolerance_m": float(manager.gap_tolerance),
        "maximum_absolute_displacement_m": float(np.max(np.abs(displacement))),
        "target_interior_displacement_m": displacement[
            list(model.mesh.dof_manager.get_node_dofs(30))[:3]
        ].tolist(),
    }


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    payload = evaluate()
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "linear_reduced_equilibrium_relative_error",
        "reference_conservative_gap_lower_m",
        "predicted_conservative_gap_lower_m",
        "registered_gap_tolerance_m",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
