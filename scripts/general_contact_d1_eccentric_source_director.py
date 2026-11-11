"""Local Q4/S3 source-director contact work with a resolved eccentric moment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver import (
    ContactDefinition, ContactInteraction, ContactSolverSettings,
    ContactSurface, create_shell_element, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_distributed_tangent_probe import _branches


def build_fixture():
    model = FEModel("d1_eccentric_source_director")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    source = ((0.1, 0.1, 0.055), (0.6, 0.1, 0.055),
              (1.1, 0.1, 0.055), (0.1, 1.1, 0.055),
              (0.6, 1.1, 0.055), (1.1, 1.1, 0.055))
    target = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    for node, position in enumerate((*source, *target), 1):
        model.add_node(node, *position)
    for element, nodes in ((1, (1, 2, 5, 4)), (2, (2, 3, 6, 5))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
            reference_normal=(0.0, 0.0, 1.0)))
    for element, nodes in ((3, (7, 8, 10)), (4, (8, 9, 10))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d",
            reference_normal=(0.0, 0.0, 1.0),
            reference_surface_offset=0.01))
    definition = ContactDefinition(
        surfaces=(ContactSurface("source", (1, 2), "bottom"),
                  ContactSurface("target", (3, 4), "top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "pair", "source", "target", 1.0e6, 0.005,
            normal_law="augmented"),),
    )
    settings = ContactSolverSettings(
        pair_motion="general", normal_discretization="primal_mortar",
        general_s3_differentiation="jet2_q4_s3_dev",
        general_s3_difference_step=2.0e-5,
        maximum_general_s3_energy_evaluations=14144,
        maximum_candidate_pairs=8,
        q4_source_area_tolerance=1.0e-9,
        q4_source_weighted_gap_tolerance=1.0e-10,
        maximum_q4_source_integration_triangles=300,
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        pressure_tolerance=1.0e-5,
    )
    return model, definition, settings


def _branch(manager, pose, multipliers):
    source, target = manager._positions(pose)
    state = manager.law.evaluate(source, target, multipliers)
    signatures = _branches(manager.law, source, target,
                           set(range(len(source))))
    return state, signatures


def _error(actual, finite):
    scale = max(float(np.linalg.norm(actual)),
                float(np.linalg.norm(finite)), 1.0e-12)
    return {"relative_error": float(np.linalg.norm(actual-finite))/scale,
            "action_scale": scale,
            "absolute_error": float(np.linalg.norm(actual-finite))}


def run():
    model, definition, settings = build_fixture()
    capability = validate_general_contact(model, definition, settings)
    if not capability.valid:
        raise RuntimeError("fixture invalid: " + "; ".join(capability.issues))
    manager = GeneralS3PatchManager(model, definition, settings)
    multipliers = np.full(6, 1000.0)
    manager.set_normal_multipliers(multipliers)
    pose = np.zeros(manager.n_dof)
    initial_source, initial_target = manager._positions(pose)
    initial = manager.law.evaluate(initial_source, initial_target,
                                   multipliers)
    direction = np.zeros_like(pose)
    for node in (1, 2, 3, 4, 5, 6):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        pose[dofs[2]] = -0.016
        direction[dofs[4]] = 0.5
    for node in (8, 9):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        pose[dofs[2]] = 0.001
    base, base_signatures = _branch(manager, pose, multipliers)
    force, tangent, result = manager.evaluate(pose, tangent=True)
    if base.active_pressure_dofs < 1:
        raise RuntimeError("eccentric fixture did not activate pressure")
    prediction = np.asarray(tangent@direction).reshape(-1)
    source_dofs = {int(dof) for node in (1, 2, 3, 4, 5, 6)
                   for dof in model.mesh.dof_manager.get_node_dofs(node)}
    target_dofs = {int(dof) for node in (7, 8, 9, 10)
                   for dof in model.mesh.dof_manager.get_node_dofs(node)}
    translations = {int(dof) for node in range(1, 11)
                    for dof in model.mesh.dof_manager.get_node_dofs(node)[:3]}
    groups = {
        "source_force": sorted(source_dofs & translations),
        "source_moment": sorted(source_dofs-translations),
        "target_force": sorted(target_dofs & translations),
        "target_moment": sorted(target_dofs-translations),
    }
    trials = []
    for step in (1.0e-5, 5.0e-6):
        candidates = []
        checks = []
        for sign in (-1.0, 1.0):
            trial_pose = pose+sign*step*direction
            trial, signatures = _branch(manager, trial_pose, multipliers)
            checks.append({
                "pressure": bool(np.array_equal(
                    trial.pressure > 0, base.pressure > 0)),
                "support": bool(np.array_equal(
                    trial.reference_support > 0,
                    base.reference_support > 0)),
                "overlap": trial.overlap_signature == base.overlap_signature,
                "pair": signatures == base_signatures,
            })
            candidates.append(manager.evaluate(trial_pose, tangent=False)[0])
        finite = np.asarray(
            (candidates[1]-candidates[0])/(2*step)).reshape(-1)
        trials.append({
            "step_rad": step,
            "branch_checks": checks,
            "smooth": all(all(check.values()) for check in checks),
            "groups": {name: _error(prediction[rows], finite[rows])
                       for name, rows in groups.items()},
        })
    return {
        "status": "development_discriminator",
        "profile": capability.profile,
        "normal_stiffness_pa_per_m": 1.0e6,
        "fixed_multipliers_pa": multipliers.tolist(),
        "target_tip_rise_m": 0.001,
        "source_downward_motion_m": 0.016,
        "initial_active_pressure_modes": initial.active_pressure_dofs,
        "initial_minimum_gap_m": initial.minimum_physical_gap,
        "active_pressure_modes": base.active_pressure_dofs,
        "integration_force_n": list(result.integration_force),
        "integration_moment_nm": list(result.integration_moment),
        "pair_force_balance_n": float(np.linalg.norm(
            np.asarray(result.integration_force)+result.target_force)),
        "pair_moment_balance_nm": float(np.linalg.norm(
            np.asarray(result.integration_moment)+result.target_moment)),
        "minimum_gap_m": result.minimum_gap,
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(run(), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
