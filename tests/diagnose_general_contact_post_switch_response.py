"""One disposable re-equilibrated pressure-direction check after a Schur switch.

This is a development oracle for the original node-1-controlled Q4/S3 fixture.
It perturbs algorithmic pressure during the second Schur call, solves the
constrained structural equilibrium from the same committed material origin,
and restores pressure before the ordinary path continues.
"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import null_space

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact import linearized_pivot, static_path
from test_general_contact_supported_schur_path import _case


model, definition, settings, program = _case()
settings = replace(
    settings,
    general_s3_differentiation="jet2_q4_s3_dev",
    q4_source_support_gradient_tolerance=1e-5,
    q4_source_support_hessian_tolerance=1e-5,
    q4_source_weighted_gap_gradient_tolerance=1e-5,
    q4_source_weighted_gap_hessian_tolerance=1e-5,
    maximum_q4_source_integration_triangles=128,
    augmentation_schur_maximum_attempts=2,
)
original_assemble = static_path._assemble_nonlinear_system
original_schur = static_path.refined_patch_supported_schur
latest_states = {}
schur_calls = 0


def record_assemble(*args, **kwargs):
    latest_states["committed"] = args[2]
    latest_states["num_layers"] = args[3]
    return original_assemble(*args, **kwargs)


def equilibrium_gap(manager, base_u, transform, control_rows,
                    trial_pressure, selected):
    base_q = np.linalg.lstsq(transform.toarray(), base_u, rcond=None)[0]
    reconstruction = np.max(np.abs(transform @ base_q-base_u))
    if reconstruction > 1e-10:
        raise RuntimeError(f"constrained pose reconstruction {reconstruction:g} m")
    null = null_space(np.asarray(control_rows, dtype=float))
    q = base_q.copy()
    origin_states = latest_states["committed"]
    num_layers = latest_states["num_layers"]
    manager.set_normal_multipliers(trial_pressure)
    for iteration in range(8):
        u = np.asarray(transform @ q).reshape(-1)
        internal, tangent, _trial = original_assemble(
            model, u, origin_states, num_layers, kinematics="von_karman",
        )
        residual = np.asarray(null.T @ transform.T @ internal).reshape(-1)
        if np.linalg.norm(residual, ord=np.inf) < 1e-10:
            break
        reduced = np.asarray(null.T @ (transform.T @ tangent @ transform) @ null)
        delta = np.linalg.solve(reduced, residual)
        q -= null @ delta
    else:
        raise RuntimeError("independent pressure perturbation did not equilibrate")
    u = np.asarray(transform @ q).reshape(-1)
    state = manager.law.geometry.evaluate(*manager._positions(u))
    return state.normalized_gap[selected], iteration+1, float(
        np.linalg.norm(residual, ord=np.inf)), state.overlap_signature


def inspect_schur(manager, displacement, tangent, transform, control_rows,
                  **kwargs):
    global schur_calls
    schur_calls += 1
    proposal = original_schur(
        manager, displacement, tangent, transform, control_rows, **kwargs,
    )
    if schur_calls != 2 or proposal is None:
        return proposal
    base_pressure = manager.normal_multipliers.copy()
    base_state = manager.law.evaluate(*manager._positions(displacement),
                                      base_pressure)
    selected = np.flatnonzero(base_state.reference_support > 0.0)
    active = np.flatnonzero(base_state.pressure > 0.0)
    active_response = linearized_pivot._constrained_gap_response(
        manager, displacement, tangent, transform, control_rows,
        selected, active,
        difference_step=kwargs["difference_step"],
    )
    if active_response is None:
        raise RuntimeError("post-switch response unavailable")
    response = np.zeros((len(selected), len(selected)))
    response[:, np.searchsorted(selected, active)] = active_response
    proposal_direction = proposal.multipliers[selected]-base_pressure[selected]
    base_gap = base_state.normalized_gap[selected]
    base_signature = manager.law.geometry.evaluate(
        *manager._positions(displacement),
    ).overlap_signature
    print(f"supported={selected.tolist()} active={active.tolist()} "
          f"base_pressure_pa={base_pressure[selected].tolist()} "
          f"proposal_direction_pa={proposal_direction.tolist()} "
          f"base_signature={base_signature}")
    try:
        for fraction in (0.001, 0.01, 0.25, 0.5, 1.0):
            delta = fraction*proposal_direction
            trial_pressure = base_pressure.copy()
            trial_pressure[selected] += delta
            actual_gap, iterations, residual, signature = equilibrium_gap(
                manager, np.asarray(displacement), transform, control_rows,
                trial_pressure, selected,
            )
            predicted = response @ delta
            actual = actual_gap-base_gap
            predicted_argument = (trial_pressure[selected]
                                  - manager.stiffness*(base_gap+predicted))
            actual_argument = (trial_pressure[selected]
                               - manager.stiffness*actual_gap)
            predicted_residual = np.maximum(0.0, predicted_argument)-trial_pressure[selected]
            actual_residual = np.maximum(0.0, actual_argument)-trial_pressure[selected]
            error = np.max(np.abs(actual-predicted))
            scale = max(np.max(np.abs(actual)), np.max(np.abs(predicted)), 1e-20)
            print(f"fraction={fraction} delta_pa={delta.tolist()} "
                  f"predicted_gap_change_m={predicted.tolist()} "
                  f"actual_gap_change_m={actual.tolist()} "
                  f"relative_directional_error={error/scale:.9g} "
                  f"predicted_projected_residual_pa={np.max(np.abs(predicted_residual)):.9g} "
                  f"actual_projected_residual_pa={np.max(np.abs(actual_residual)):.9g} "
                  f"predicted_active={selected[predicted_argument > 0.0].tolist()} "
                  f"actual_active={selected[actual_argument > 0.0].tolist()} "
                  f"newton_iterations={iterations} equilibrium_residual_n={residual:.9g} "
                  f"signature_unchanged={signature == base_signature}")
    finally:
        manager.set_normal_multipliers(base_pressure)
    return proposal


static_path._assemble_nonlinear_system = record_assemble
static_path.refined_patch_supported_schur = inspect_schur
try:
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=12,
        emit_restart_checkpoint=True,
    )
    print(f"path_status={result.status} reason={result.failure_reason} "
          f"accepted_steps={len(result.steps)} schur_calls={schur_calls}")
finally:
    static_path._assemble_nonlinear_system = original_assemble
    static_path.refined_patch_supported_schur = original_schur
