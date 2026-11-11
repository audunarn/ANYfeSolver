"""Elastic sphere stiffened-panel: gravity-preloaded probe continuation.

Declared benchmark continuation for the elastic sphere-panel case (see
``docs/ELASTIC_SPHERE_STIFFENED_PANEL.md``): the actual accepted panel-only
gravity equilibrium state of the clamped eccentric-beam-stiffened panel is
continued through a 1000 N downward global-Z point load on the exact centre
structural node and back, using only public nonlinear solver APIs:

- the single existing case preparation (``prepare_case_models``: canonical
  panel build, fixed sphere skin build, one qualified panel-only gravity
  solve, corotational/work-conjugate, tolerance 1e-8) supplies the accepted
  displacement and element states; the sphere is never solved and gravity is
  not duplicated;
- the load leg continues from a deep copy of that accepted state with
  ``equilibrate_initial_state=False``: the gravity load case is the constant
  load (already carried by the accepted state) and the probe is the
  proportional load ramped over two increments to its full 1000 N;
- the unload leg continues from a deep copy of the accepted peak state the
  same way: gravity+probe is the constant load and the negated probe is the
  proportional load ramped over two increments, ending at gravity only;
- the peak state is compared against a fresh uninterrupted
  ``NonlinearLoadProgram`` gravity-then-probe solve and the unloaded state
  against the original accepted gravity state (rtol 1e-8, atol 1e-10);
- unchanged translation masses, support-resultant agreement (1e-9) and the
  global force/moment balance are verified at the peak and unloaded states
  with the actual loads accounted for; the supplied state, model coordinates,
  supports, materials and sphere are verified exactly unmutated;
- the panel's curved top skin is recovered at the accepted gravity, peak and
  unloaded states and independently reconstructed per corner within the
  existing 1e-13 director tolerance; the sphere skin stays at its
  zero-pose reference (never solved).

The output status is ``gravity_probe_completed``.  Contact response and the
gravity impact remain explicitly unsupported, and this report is not a
checkpoint: no restart, transient or impact history is produced or implied.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from anyfesolver import LoadCase, solve_static_nonlinear
from anyfesolver.corotational import rotation_matrix_from_vector
from anyfesolver.nonlinear_static import NonlinearLoadProgram, NonlinearLoadStage
from anyfesolver.nonlinear_restart import model_configuration_fingerprint

from examples.elastic_sphere_stiffened_panel import (
    FORCE_BALANCE_TOLERANCE,
    GRAVITY,
    MOMENT_BALANCE_TOLERANCE,
    PLATE_TOP_Z,
    REACTION_AGREEMENT_TOLERANCE,
    RELATIVE_MASS_TOLERANCE,
    SOLVE_SETTINGS,
    TARGET_PANEL_MASS,
    _global_balance,
    _runtime_identity,
    build_panel_model,
    prepare_case_models,
    support_reaction_recovery,
    translation_masses,
)
from examples.elastic_sphere_panel_contact_skins import build_skin_maps

SCHEMA = "anyfesolver.elastic-sphere-panel-continuation-v1"
STATUS = "gravity_probe_completed"
CONTACT_RESPONSE_STATUS = "unsupported"
IMPACT_STATUS = "unsupported"

# Declared probe (fixed; do not weaken): 1000 N downward global Z on the
# exact centre structural node, two increments per leg.
PROBE_LOAD_N = 1000.0
PROBE_NUM_STEPS = 2

# Fixed prospective criteria (existing values; never weakened).
POSTCHECK_TOLERANCE = 1.0e-12
COMPARISON_RTOL = 1.0e-8
COMPARISON_ATOL = 1.0e-10
SKIN_RECONSTRUCTION_TOLERANCE = 1.0e-13

UNSUPPORTED_NOTE = (
    "This document reports the completed gravity-preloaded probe continuation "
    "of the panel only. No contact response, gap history, contact force or "
    "impact execution is claimed or implied: the current solver has no "
    "admitted contact solve for this case and no admitted transient route "
    "for the free deformable sphere."
)
NOT_A_CHECKPOINT_NOTE = (
    "This report is a result document, not a checkpoint: it carries no "
    "restart, transient or impact history and cannot be used to resume a "
    "solve. The public initial-state route used here is documented in the "
    "example source."
)

PROBE_LEG_SETTINGS = {
    "kinematics": SOLVE_SETTINGS["kinematics"],
    "corotational_tangent": SOLVE_SETTINGS["corotational_tangent"],
    "num_layers": SOLVE_SETTINGS["num_layers"],
    "num_steps": PROBE_NUM_STEPS,
    "max_iterations": SOLVE_SETTINGS["max_iterations"],
    "tolerance": SOLVE_SETTINGS["tolerance"],
    "max_load_factor": SOLVE_SETTINGS["max_load_factor"],
    "equilibrate_initial_state": False,
    "control": "force",
}


def centre_structural_node(model):
    """The exact centre plate node (x=y=0), fail-closed on topology."""
    node = min(
        model.mesh.nodes.values(),
        key=lambda candidate: abs(candidate.x) + abs(candidate.y),
    )
    if abs(node.x) > 0.0 or abs(node.y) > 0.0:
        raise RuntimeError(
            f"centre structural node {node.id} is not at x=y=0: "
            f"({node.x!r}, {node.y!r})"
        )
    if int(node.id) >= 10000:
        raise RuntimeError(
            f"centre structural node {node.id} is not a plate shell node"
        )
    if len(node.dofs) != 6:
        raise RuntimeError(
            f"centre structural node {node.id} does not carry six DOFs"
        )
    return node


def probe_load_case(node_id: int, name: str, force_z: float) -> LoadCase:
    """One nodal probe load [0, 0, force_z] on the centre node."""
    load_case = LoadCase(name=name)
    load_case.add_nodal_load(
        int(node_id), load_vector=np.array([0.0, 0.0, float(force_z)])
    )
    return load_case


def gravity_probe_load_case(gravity_load_case: LoadCase, node_id: int) -> LoadCase:
    """The constant unload-leg load: full gravity plus the full probe."""
    if gravity_load_case.gravity is None:
        raise RuntimeError("the accepted gravity load case carries no gravity")
    combined = LoadCase(
        name="panel_gravity_probe",
        gravity=np.array(gravity_load_case.gravity, dtype=float, copy=True),
    )
    combined.add_nodal_load(
        int(node_id), load_vector=np.array([0.0, 0.0, -PROBE_LOAD_N])
    )
    return combined


def continue_probe_leg(
    model,
    initial_displacements,
    initial_element_states,
    constant_load_case: LoadCase,
    proportional_load_case: LoadCase,
):
    """Continue an accepted state through one proportional probe leg.

    The supplied displacement and element states are copied before the call
    and passed as owned objects so the caller can verify the solver left them
    exactly unmutated (the public initial-state route must not consume caller
    state).  ``equilibrate_initial_state=False`` keeps the supplied state as
    the committed equilibrium under the constant load.
    """
    return solve_static_nonlinear(
        model,
        proportional_load_case,
        constant_load_case=constant_load_case,
        num_steps=PROBE_NUM_STEPS,
        max_iterations=SOLVE_SETTINGS["max_iterations"],
        tolerance=SOLVE_SETTINGS["tolerance"],
        max_load_factor=SOLVE_SETTINGS["max_load_factor"],
        kinematics=SOLVE_SETTINGS["kinematics"],
        corotational_tangent=SOLVE_SETTINGS["corotational_tangent"],
        num_layers=SOLVE_SETTINGS["num_layers"],
        initial_displacements=initial_displacements,
        initial_element_states=initial_element_states,
        equilibrate_initial_state=False,
    )


def fresh_uninterrupted_program(node_id: int):
    """Fresh gravity-then-probe ``NonlinearLoadProgram`` oracle solve.

    A fresh panel model (same canonical build) is solved from zero through
    one ordered program: the gravity stage to its full factor, then the probe
    stage to its full factor.  The material is purely elastic, so the final
    equilibrium is path-independent and must agree with the continued peak
    state within the existing split-restart comparison bound.
    """
    fresh_model = build_panel_model()
    gravity_case = LoadCase(
        name="panel_gravity", gravity=np.array([0.0, 0.0, -GRAVITY])
    )
    probe_case = probe_load_case(node_id, "centre_probe", -PROBE_LOAD_N)
    program = NonlinearLoadProgram(
        (
            NonlinearLoadStage("gravity", gravity_case, 1.0),
            NonlinearLoadStage("probe", probe_case, 1.0),
        )
    )
    result = solve_static_nonlinear(
        fresh_model,
        None,
        load_program=program,
        num_steps=SOLVE_SETTINGS["num_steps"],
        max_iterations=SOLVE_SETTINGS["max_iterations"],
        tolerance=SOLVE_SETTINGS["tolerance"],
        max_load_factor=program.total_factor,
        kinematics=SOLVE_SETTINGS["kinematics"],
        corotational_tangent=SOLVE_SETTINGS["corotational_tangent"],
        num_layers=SOLVE_SETTINGS["num_layers"],
    )
    return fresh_model, program, result


def _step_history(result) -> list:
    return [
        {
            "step_index": int(step.step_index),
            "load_factor": float(step.load_factor),
            "iterations": int(step.iterations),
            "residual_norm": float(step.residual_norm),
        }
        for step in result.steps
    ]


def _postcheck_summary(result) -> dict:
    postcheck = dict(result.info.get("constraint_postcheck", {}))
    return {
        "status": str(postcheck.get("status")),
        "max_absolute_residual": float(postcheck.get("max_absolute_residual", np.inf)),
        "max_relative_residual": float(postcheck.get("max_relative_residual", np.inf)),
        "equation_count": int(postcheck.get("equation_count", 0)),
    }


def _require_leg_result(result, label: str) -> None:
    """Fail closed on the fixed completion/postcheck criteria."""
    if result.status != "completed":
        raise RuntimeError(
            f"{label} did not complete: status={result.status!r}; "
            f"failure_reason={result.info.get('failure_reason')!r}"
        )
    if str(result.info.get("corotational_tangent")) != "work_conjugate":
        raise RuntimeError(
            f"{label} did not use the work-conjugate corotational tangent: "
            f"{result.info.get('corotational_tangent')!r}"
        )
    postcheck = _postcheck_summary(result)
    if not np.isfinite([
        postcheck["max_absolute_residual"], postcheck["max_relative_residual"]
    ]).all():
        raise RuntimeError(f"{label} constraint postcheck is nonfinite")
    if postcheck["status"] != "passed":
        raise RuntimeError(f"{label} constraint postcheck failed: {postcheck!r}")
    if postcheck["max_absolute_residual"] > POSTCHECK_TOLERANCE:
        raise RuntimeError(
            f"{label} constraint postcheck max absolute residual "
            f"{postcheck['max_absolute_residual']!r} exceeds {POSTCHECK_TOLERANCE!r}"
        )
    if postcheck["max_relative_residual"] > POSTCHECK_TOLERANCE:
        raise RuntimeError(
            f"{label} constraint postcheck max relative residual "
            f"{postcheck['max_relative_residual']!r} exceeds {POSTCHECK_TOLERANCE!r}"
        )


def displacement_agreement(observed, reference, label: str) -> dict:
    """Existing split-restart comparison bound (rtol 1e-8, atol 1e-10)."""
    left = np.asarray(observed, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape:
        raise RuntimeError(
            f"{label}: displacement shapes {left.shape} and {right.shape} differ"
        )
    if left.ndim != 1 or not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise RuntimeError(f"{label}: expected finite displacement vectors")
    difference = np.abs(left - right)
    bound = COMPARISON_ATOL + COMPARISON_RTOL * np.abs(right)
    passed = bool(np.all(difference <= bound))
    summary = {
        "label": label,
        "rtol": COMPARISON_RTOL,
        "atol": COMPARISON_ATOL,
        "max_absolute_difference": float(np.max(difference)) if left.size else 0.0,
        "passed": passed,
    }
    if not passed:
        worst = int(np.argmax(difference / bound))
        summary["worst_dof"] = int(worst)
        summary["worst_observed"] = float(left[worst]) if left.size else None
        summary["worst_reference"] = float(right[worst]) if right.size else None
        raise RuntimeError(f"{label} disagrees beyond the comparison bound: {summary!r}")
    return summary


def _states_equal(left, right) -> bool:
    """Exact recursive equality of deep-copied element-state trees."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _states_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _states_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and left.dtype == right.dtype and bool(
            np.array_equal(left, right)
        )
    return type(left) is type(right) and left == right


def _model_snapshot(model) -> dict:
    """Exact reference snapshot: coordinates, connectivity, materials, BCs."""
    return {
        "configuration_fingerprint": model_configuration_fingerprint(model),
        "nodes": {
            int(node_id): (float(node.x), float(node.y), float(node.z))
            for node_id, node in model.mesh.nodes.items()
        },
        "elements": {
            int(element_id): (
                type(element).__name__,
                tuple(int(n) for n in element.node_ids),
                str(element.material_name),
            )
            for element_id, element in model.mesh.elements.items()
        },
        "materials": {
            str(name): (
                float(material.elastic_modulus),
                float(material.poisson_ratio),
                float(material.density),
                float(material.yield_stress),
                material.hardening_curve is None,
            )
            for name, material in model.materials.items()
        },
        "boundary_conditions": [
            (
                str(boundary.name),
                tuple(int(n) for n in boundary.node_ids),
                tuple(sorted((str(key), float(value))
                             for key, value in boundary.dof_constraints.items())),
            )
            for boundary in model.boundary_conditions
        ],
    }


def _loaded_state_acceptance(model, load_case: LoadCase, displacements, result,
                             label: str) -> dict:
    """Masses, support-resultant agreement and global balance at one state.

    The external load is the actual load carried at that state (the same
    public assembled load vector law the solver uses), so the global force
    and moment balance account for the probe load, and the recovered
    support resultants are compared with the solver-recorded resultants of
    the accepted step under the existing 1e-9 agreement tolerance.
    """
    masses = translation_masses(model)
    for axis, mass in enumerate(masses):
        if abs(mass - TARGET_PANEL_MASS) > RELATIVE_MASS_TOLERANCE * TARGET_PANEL_MASS:
            raise RuntimeError(
                f"{label}: panel translation mass axis {axis}: {mass!r} "
                f"disagrees with target {TARGET_PANEL_MASS!r}"
            )
    recovery = support_reaction_recovery(model, load_case, displacements)
    balance = _global_balance(model, recovery)
    recorded = result.steps[-1].support_reactions if result.steps else {}
    solver_resultants = {
        str(name): [float(value) for value in values]
        for name, values in recorded.items()
    }
    recovered_resultants = {
        str(name): [float(value) for value in values]
        for name, values in recovery["named_resultants"].items()
    }
    if set(solver_resultants) != set(recovered_resultants):
        raise RuntimeError(
            f"{label}: recovered support names do not match the "
            f"solver-recorded supports: {sorted(recovered_resultants)} vs "
            f"{sorted(solver_resultants)}"
        )
    max_resultant_deviation = 0.0
    for name, recorded_values in solver_resultants.items():
        for recorded_value, recovered_value in zip(
            recorded_values, recovered_resultants[name]
        ):
            deviation = abs(recovered_value - recorded_value)
            max_resultant_deviation = max(
                max_resultant_deviation, deviation / max(1.0, abs(recorded_value))
            )
    if max_resultant_deviation > REACTION_AGREEMENT_TOLERANCE:
        raise RuntimeError(
            f"{label}: recovered support resultants disagree with the "
            f"solver-recorded resultants (max relative deviation "
            f"{max_resultant_deviation!r} exceeds "
            f"{REACTION_AGREEMENT_TOLERANCE!r})"
        )
    if balance["force_balance_norm_N"] > FORCE_BALANCE_TOLERANCE:
        raise RuntimeError(
            f"{label}: global force balance {balance['force_balance_norm_N']!r} N "
            f"exceeds {FORCE_BALANCE_TOLERANCE!r}"
        )
    if balance["moment_balance_norm_Nm"] > MOMENT_BALANCE_TOLERANCE:
        raise RuntimeError(
            f"{label}: global moment balance {balance['moment_balance_norm_Nm']!r} Nm "
            f"exceeds {MOMENT_BALANCE_TOLERANCE!r}"
        )
    return {
        "panel_masses_kg": [float(value) for value in masses],
        "panel_target_mass_kg": float(TARGET_PANEL_MASS),
        "relative_mass_tolerance": RELATIVE_MASS_TOLERANCE,
        "recovered_named_resultants": recovered_resultants,
        "solver_recorded_named_resultants": solver_resultants,
        "max_support_resultant_relative_deviation": float(max_resultant_deviation),
        "reaction_agreement_tolerance": REACTION_AGREEMENT_TOLERANCE,
        "applied_force_N": balance["applied_force_N"],
        "applied_moment_Nm": balance["applied_moment_Nm"],
        "reaction_force_N": balance["reaction_force_N"],
        "reaction_moment_Nm": balance["reaction_moment_Nm"],
        "force_balance_norm_N": balance["force_balance_norm_N"],
        "force_balance_tolerance_N": FORCE_BALANCE_TOLERANCE,
        "moment_balance_norm_Nm": balance["moment_balance_norm_Nm"],
        "moment_balance_tolerance_Nm": MOMENT_BALANCE_TOLERANCE,
    }


def _independent_skin_positions(model, skin_map, displacements) -> np.ndarray:
    """Independent per-corner reconstruction X + u + h*Exp([theta]x) d.

    Rebuilds every face-local skin corner from the model's own node
    coordinates and DOFs and the declared per-face director/offset, using
    the public ``rotation_matrix_from_vector`` rotation law directly instead
    of the skin map's ``pose`` path.
    """
    displacements = np.asarray(displacements, dtype=float)
    rows = []
    for face in skin_map.faces:
        for node_id in face.node_ids:
            node = model.mesh.get_node(int(node_id))
            reference = np.asarray(node.coords(), dtype=float)
            local = np.asarray(
                [displacements[dof] for dof in node.dofs[:6]], dtype=float
            )
            rotation = rotation_matrix_from_vector(local[3:])
            director = rotation @ np.asarray(face.director, dtype=float)
            rows.append(
                reference + local[:3] + float(face.physical_offset) * director
            )
    return np.asarray(rows, dtype=float)


def _skin_state_report(model, skin_map, displacements, label: str) -> dict:
    """Recovered skin geometry plus its independent reconstruction check."""
    pose = skin_map.pose(np.asarray(displacements, dtype=float))
    independent = _independent_skin_positions(model, skin_map, displacements)
    if independent.shape != pose.positions.shape:
        raise RuntimeError(
            f"{label}: independent skin reconstruction shape "
            f"{independent.shape} differs from the recovered skin "
            f"{pose.positions.shape}"
        )
    deviation = float(np.max(np.abs(independent - pose.positions)))
    if deviation > SKIN_RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            f"{label}: independent skin reconstruction deviates by "
            f"{deviation!r} m, exceeding {SKIN_RECONSTRUCTION_TOLERANCE!r}"
        )
    return {
        "label": label,
        "corner_count": int(pose.positions.shape[0]),
        "recovered_positions_m": [
            [float(value) for value in row] for row in pose.positions
        ],
        "independent_reconstruction_max_deviation_m": deviation,
        "independent_reconstruction_tolerance_m": SKIN_RECONSTRUCTION_TOLERANCE,
    }


def _centre_fields(model, node, displacements) -> dict:
    displacements = np.asarray(displacements, dtype=float)
    return {
        "node_id": int(node.id),
        "reference_position_m": [float(node.x), float(node.y), float(node.z)],
        "displacement": [
            float(displacements[dof]) for dof in node.dofs[:6]
        ],
        "plate_top_z_m": float(
            PLATE_TOP_Z + displacements[node.dofs[2]]
        ),
    }


def load_leg_discriminator() -> dict:
    """First discriminator: actual preparation plus the load leg, once.

    Runs the single existing case preparation and continues the accepted
    gravity state through the 1000 N probe load leg.  Returns the verdict
    record; ``discriminator_passed`` is true only when the leg completed with
    the work-conjugate corotational tangent and a passing constraint
    postcheck whose max absolute and relative residuals are within 1e-12.
    """
    (
        panel_model,
        sphere_model,
        sphere_geometry,
        gravity_load_case,
        gravity_result,
        gravity_displacements,
    ) = prepare_case_models()
    centre = centre_structural_node(panel_model)
    probe_case = probe_load_case(centre.id, "centre_probe", -PROBE_LOAD_N)
    supplied_displacements = np.array(
        gravity_displacements, dtype=float, copy=True
    )
    supplied_states = copy.deepcopy(dict(gravity_result.element_states))
    displacement_snapshot = np.array(supplied_displacements, copy=True)
    states_snapshot = copy.deepcopy(supplied_states)
    load_result = continue_probe_leg(
        panel_model,
        supplied_displacements,
        supplied_states,
        gravity_load_case,
        probe_case,
    )
    postcheck = _postcheck_summary(load_result)
    verdict = {
        "schema": SCHEMA,
        "gravity_status": str(gravity_result.status),
        "load_leg_status": str(load_result.status),
        "corotational_tangent": str(load_result.info.get("corotational_tangent")),
        "constraint_postcheck": postcheck,
        "postcheck_tolerance": POSTCHECK_TOLERANCE,
        "load_leg_steps": _step_history(load_result),
        "centre_fields": _centre_fields(
            panel_model, centre, load_result.displacements
        ),
        "supplied_state_preserved": bool(
            np.array_equal(supplied_displacements, displacement_snapshot)
            and _states_equal(supplied_states, states_snapshot)
        ),
    }
    verdict["discriminator_passed"] = bool(
        verdict["gravity_status"] == "completed"
        and verdict["load_leg_status"] == "completed"
        and verdict["corotational_tangent"] == "work_conjugate"
        and postcheck["status"] == "passed"
        and postcheck["max_absolute_residual"] <= POSTCHECK_TOLERANCE
        and postcheck["max_relative_residual"] <= POSTCHECK_TOLERANCE
        and verdict["supplied_state_preserved"]
    )
    return verdict


def prepare_continuation() -> dict:
    """Run the full gravity -> probe -> unload continuation and verify it."""
    (
        panel_model,
        sphere_model,
        sphere_geometry,
        gravity_load_case,
        gravity_result,
        gravity_displacements,
    ) = prepare_case_models()

    # Exact nonmutation snapshots of everything the continuation touches.
    panel_snapshot = _model_snapshot(panel_model)
    sphere_snapshot = _model_snapshot(sphere_model)
    gravity_displacements = np.asarray(gravity_displacements, dtype=float)
    gravity_states = copy.deepcopy(dict(gravity_result.element_states))
    gravity_displacement_snapshot = np.array(gravity_displacements, copy=True)
    gravity_states_snapshot = copy.deepcopy(gravity_states)

    centre = centre_structural_node(panel_model)
    probe_case = probe_load_case(centre.id, "centre_probe", -PROBE_LOAD_N)
    release_case = probe_load_case(
        centre.id, "centre_probe_release", PROBE_LOAD_N
    )
    combined_case = gravity_probe_load_case(gravity_load_case, centre.id)

    # Load leg: accepted gravity state -> gravity (constant) + probe.
    load_supplied_displacements = np.array(gravity_displacements, copy=True)
    load_supplied_states = copy.deepcopy(gravity_states)
    load_displacement_snapshot = np.array(load_supplied_displacements, copy=True)
    load_states_snapshot = copy.deepcopy(load_supplied_states)
    load_result = continue_probe_leg(
        panel_model,
        load_supplied_displacements,
        load_supplied_states,
        gravity_load_case,
        probe_case,
    )
    _require_leg_result(load_result, "load leg")
    peak_displacements = np.asarray(load_result.displacements, dtype=float)
    peak_states = copy.deepcopy(dict(load_result.element_states))

    # Unload leg: accepted peak state -> gravity+probe (constant) + release.
    unload_supplied_displacements = np.array(peak_displacements, copy=True)
    unload_supplied_states = copy.deepcopy(peak_states)
    unload_displacement_snapshot = np.array(unload_supplied_displacements, copy=True)
    unload_states_snapshot = copy.deepcopy(unload_supplied_states)
    unload_result = continue_probe_leg(
        panel_model,
        unload_supplied_displacements,
        unload_supplied_states,
        combined_case,
        release_case,
    )
    _require_leg_result(unload_result, "unload leg")
    unload_displacements = np.asarray(unload_result.displacements, dtype=float)

    # Fresh uninterrupted gravity-then-probe oracle and the two fixed
    # comparisons (peak vs fresh, unloaded vs original gravity).
    fresh_model, program, fresh_result = fresh_uninterrupted_program(centre.id)
    _require_leg_result(fresh_result, "fresh uninterrupted program")
    peak_comparison = displacement_agreement(
        peak_displacements, fresh_result.displacements,
        "peak state vs fresh uninterrupted gravity-then-probe program",
    )
    unload_comparison = displacement_agreement(
        unload_displacements, gravity_displacements,
        "unloaded state vs original accepted gravity state",
    )

    # Actual-load acceptance at the peak and unloaded states.
    peak_acceptance = _loaded_state_acceptance(
        panel_model, combined_case, peak_displacements, load_result, "peak state"
    )
    unload_acceptance = _loaded_state_acceptance(
        panel_model, gravity_load_case, unload_displacements, unload_result,
        "unloaded state",
    )

    # Exact nonmutation of every supplied input and of the case models.
    supplied_state_preserved = bool(
        np.array_equal(load_supplied_displacements, load_displacement_snapshot)
        and _states_equal(load_supplied_states, load_states_snapshot)
        and np.array_equal(unload_supplied_displacements, unload_displacement_snapshot)
        and _states_equal(unload_supplied_states, unload_states_snapshot)
    )
    accepted_state_preserved = bool(
        np.array_equal(gravity_displacements, gravity_displacement_snapshot)
        and _states_equal(gravity_states, gravity_states_snapshot)
        and np.array_equal(
            np.asarray(gravity_result.displacements, dtype=float),
            gravity_displacement_snapshot,
        )
        and _states_equal(
            dict(gravity_result.element_states), gravity_states_snapshot
        )
    )
    if not supplied_state_preserved:
        raise RuntimeError(
            "the solver mutated a supplied continuation input state"
        )
    if not accepted_state_preserved:
        raise RuntimeError(
            "the accepted gravity displacement or element states were mutated"
        )
    if _model_snapshot(panel_model) != panel_snapshot:
        raise RuntimeError("the panel model was mutated by the continuation")
    if _model_snapshot(sphere_model) != sphere_snapshot:
        raise RuntimeError("the sphere model was mutated by the continuation")

    # Skins: panel recovered at all three accepted states, sphere at rest.
    sphere_map, panel_map = build_skin_maps(panel_model, sphere_model)
    panel_skin_gravity = _skin_state_report(
        panel_model, panel_map, gravity_displacements,
        "accepted gravity-preloaded panel displacement",
    )
    panel_skin_peak = _skin_state_report(
        panel_model, panel_map, peak_displacements,
        "accepted probe peak panel displacement",
    )
    panel_skin_unload = _skin_state_report(
        panel_model, panel_map, unload_displacements,
        "accepted unloaded panel displacement",
    )
    sphere_skin_reference = _skin_state_report(
        sphere_model, sphere_map, np.zeros(sphere_map.n_dof, dtype=float),
        "sphere zero-pose reference (never solved)",
    )

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "contact_response": CONTACT_RESPONSE_STATUS,
        "impact_status": IMPACT_STATUS,
        "unsupported_note": UNSUPPORTED_NOTE,
        "not_a_checkpoint": NOT_A_CHECKPOINT_NOTE,
        "case": {
            "gravity_m_s2": GRAVITY,
            "probe": {
                "load_N": [0.0, 0.0, -PROBE_LOAD_N],
                "magnitude_N": PROBE_LOAD_N,
                "node_id": int(centre.id),
                "node_reference_position_m": [
                    float(centre.x), float(centre.y), float(centre.z),
                ],
                "dof": "uz",
                "increments_per_leg": PROBE_NUM_STEPS,
            },
            "legs": {
                "load": {
                    "constant_load": "panel gravity (already carried by the "
                    "accepted initial state)",
                    "proportional_load": "centre probe 0 -> 1000 N",
                },
                "unload": {
                    "constant_load": "panel gravity + full centre probe",
                    "proportional_load": "negated centre probe 0 -> 1000 N",
                },
            },
            "settings": {
                "gravity_solve": dict(SOLVE_SETTINGS),
                "probe_legs": dict(PROBE_LEG_SETTINGS),
                "fresh_program": {
                    "kinematics": SOLVE_SETTINGS["kinematics"],
                    "corotational_tangent": SOLVE_SETTINGS["corotational_tangent"],
                    "num_layers": SOLVE_SETTINGS["num_layers"],
                    "num_steps": SOLVE_SETTINGS["num_steps"],
                    "max_iterations": SOLVE_SETTINGS["max_iterations"],
                    "tolerance": SOLVE_SETTINGS["tolerance"],
                    "stages": [
                        {"name": stage.name, "target_factor": float(stage.target_factor)}
                        for stage in program.stages
                    ],
                },
            },
        },
        "gravity_solve": {
            "status": str(gravity_result.status),
            "steps": _step_history(gravity_result),
            "final_load_factor": float(gravity_result.load_factor),
        },
        "load_leg": {
            "status": str(load_result.status),
            "corotational_tangent": str(
                load_result.info.get("corotational_tangent")
            ),
            "constraint_postcheck": _postcheck_summary(load_result),
            "steps": _step_history(load_result),
            "final_load_factor": float(load_result.load_factor),
        },
        "unload_leg": {
            "status": str(unload_result.status),
            "corotational_tangent": str(
                unload_result.info.get("corotational_tangent")
            ),
            "constraint_postcheck": _postcheck_summary(unload_result),
            "steps": _step_history(unload_result),
            "final_load_factor": float(unload_result.load_factor),
        },
        "fresh_program": {
            "status": str(fresh_result.status),
            "corotational_tangent": str(
                fresh_result.info.get("corotational_tangent")
            ),
            "constraint_postcheck": _postcheck_summary(fresh_result),
            "steps": _step_history(fresh_result),
            "final_load_factor": float(fresh_result.load_factor),
            "stage_factors": fresh_result.info.get(
                "load_program_stage_factors", {}
            ),
        },
        "centre_displacements": {
            "gravity": _centre_fields(
                panel_model, centre, gravity_displacements
            ),
            "peak": _centre_fields(panel_model, centre, peak_displacements),
            "unloaded": _centre_fields(
                panel_model, centre, unload_displacements
            ),
        },
        "comparisons": {
            "peak_vs_fresh_program": peak_comparison,
            "unloaded_vs_original_gravity": unload_comparison,
        },
        "acceptance": {
            "peak_state": peak_acceptance,
            "unloaded_state": unload_acceptance,
        },
        "nonmutation": {
            "supplied_state_preserved": supplied_state_preserved,
            "accepted_gravity_state_preserved": accepted_state_preserved,
            "panel_model_coordinates_supports_materials_preserved": True,
            "sphere_model_preserved": True,
            "sphere_solved": False,
            "gravity_duplicated": False,
        },
        "skins": {
            "panel": {
                "face_count": len(panel_map.faces),
                "gravity": panel_skin_gravity,
                "peak": panel_skin_peak,
                "unloaded": panel_skin_unload,
            },
            "sphere": {
                "face_count": len(sphere_map.faces),
                "reference": sphere_skin_reference,
                "note": "the sphere is never solved; its skin stays at the "
                "zero-pose reference",
            },
        },
        "identity": _runtime_identity(),
    }
    _verify_continuation(payload)
    return payload


def _verify_continuation(payload: dict) -> None:
    """Fail closed on every fixed prospective criterion (never weaken)."""
    # Refuse invalid numerical report fields before any inequality comparison;
    # comparisons such as NaN > tolerance are false.
    def require_finite(value):
        if isinstance(value, dict):
            for item in value.values():
                require_finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                require_finite(item)
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise RuntimeError("continuation report contains a nonfinite value")

    require_finite(payload)
    if payload["status"] != STATUS:
        raise RuntimeError("continuation payload has the wrong status")
    if payload["contact_response"] != "unsupported":
        raise RuntimeError("contact response must stay unsupported")
    if payload["impact_status"] != "unsupported":
        raise RuntimeError("impact must stay unsupported")
    for key in ("gravity_solve", "load_leg", "unload_leg", "fresh_program"):
        if payload[key]["status"] != "completed":
            raise RuntimeError(f"{key} did not complete")
    for key in ("load_leg", "unload_leg", "fresh_program"):
        if payload[key]["corotational_tangent"] != "work_conjugate":
            raise RuntimeError(f"{key} did not use the work-conjugate tangent")
        postcheck = payload[key]["constraint_postcheck"]
        if postcheck["status"] != "passed":
            raise RuntimeError(f"{key} constraint postcheck failed")
        if postcheck["max_absolute_residual"] > POSTCHECK_TOLERANCE:
            raise RuntimeError(f"{key} constraint postcheck absolute residual")
        if postcheck["max_relative_residual"] > POSTCHECK_TOLERANCE:
            raise RuntimeError(f"{key} constraint postcheck relative residual")
    if not payload["comparisons"]["peak_vs_fresh_program"]["passed"]:
        raise RuntimeError("peak state comparison failed")
    if not payload["comparisons"]["unloaded_vs_original_gravity"]["passed"]:
        raise RuntimeError("unloaded state comparison failed")
    for label, acceptance in payload["acceptance"].items():
        if (
            acceptance["max_support_resultant_relative_deviation"]
            > acceptance["reaction_agreement_tolerance"]
        ):
            raise RuntimeError(f"{label} support resultant agreement failed")
        if acceptance["force_balance_norm_N"] > acceptance["force_balance_tolerance_N"]:
            raise RuntimeError(f"{label} global force balance failed")
        if (
            acceptance["moment_balance_norm_Nm"]
            > acceptance["moment_balance_tolerance_Nm"]
        ):
            raise RuntimeError(f"{label} global moment balance failed")
        for mass in acceptance["panel_masses_kg"]:
            if abs(mass - TARGET_PANEL_MASS) > (
                RELATIVE_MASS_TOLERANCE * TARGET_PANEL_MASS
            ):
                raise RuntimeError(f"{label} panel mass changed")
    nonmutation = payload["nonmutation"]
    for key in (
        "supplied_state_preserved", "accepted_gravity_state_preserved",
        "panel_model_coordinates_supports_materials_preserved", "sphere_model_preserved",
    ):
        if nonmutation[key] is not True:
            raise RuntimeError(f"nonmutation check {key!r} failed")
    for key in ("sphere_solved", "gravity_duplicated"):
        if nonmutation[key] is not False:
            raise RuntimeError(f"unsupported action {key!r} was performed")
    for group in (payload["skins"]["panel"],):
        for state in ("gravity", "peak", "unloaded"):
            if (
                group[state]["independent_reconstruction_max_deviation_m"]
                > SKIN_RECONSTRUCTION_TOLERANCE
            ):
                raise RuntimeError(
                    f"panel skin {state} independent reconstruction failed"
                )
    if (
        payload["skins"]["sphere"]["reference"][
            "independent_reconstruction_max_deviation_m"
        ] > SKIN_RECONSTRUCTION_TOLERANCE
    ):
        raise RuntimeError("sphere skin independent reconstruction failed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the continuation JSON document to this path")
    args = parser.parse_args(argv)
    payload = prepare_continuation()
    document = json.dumps(payload, indent=2, allow_nan=False)
    if args.output is None:
        print(document)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "contact_response": payload["contact_response"],
        "impact_status": payload["impact_status"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
