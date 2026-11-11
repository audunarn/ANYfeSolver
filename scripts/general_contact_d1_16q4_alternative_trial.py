"""Disposable 16-Q4 branch candidate through unchanged static verification.

This development experiment substitutes only the first pressure proposal.
The static owner still re-equilibrates and applies its usual active-set,
physical-gap, merit, rollback and increment-acceptance checks.
"""

import json
from pathlib import Path
import sys

import numpy as np

import anyfesolver.general_contact.linearized_pivot as pivot
from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact.supported_schur import SupportedSchurProposal

from general_contact_d1_coupled_refinement import build_case


def main(captured_path, output_path):
    captured = json.loads(Path(captured_path).read_text(encoding="utf-8"))
    alternatives = captured["bounded_least_squares_solutions"]
    selected = min(alternatives, key=lambda item: (
        item["residual_inf_pa"],
        np.linalg.norm(np.asarray(item["pressure"])
                       - np.asarray(captured["initial_pressure_pa"])),
    ))
    original = pivot.propose_supported_schur_step
    proposals = 0

    def substitute(multipliers, normalized_gap, supported,
                   equilibrated_gap_response, **kwargs):
        nonlocal proposals
        proposals += 1
        if proposals != 1:
            return original(multipliers, normalized_gap, supported,
                            equilibrated_gap_response, **kwargs)
        modes = np.flatnonzero(supported)
        if modes.tolist() != captured["supported_indices"]:
            raise ValueError("captured supported modes do not match")
        lam = np.asarray(multipliers, dtype=float)
        gap = np.asarray(normalized_gap, dtype=float)[modes]
        pressure = lam[modes]
        response = np.asarray(equilibrated_gap_response, dtype=float)
        np.testing.assert_allclose(pressure, captured["initial_pressure_pa"],
                                   rtol=1.0e-9, atol=1.0e-12)
        np.testing.assert_allclose(gap, captured["initial_normalized_gap_m"],
                                   rtol=1.0e-9, atol=1.0e-12)
        np.testing.assert_allclose(response,
                                   captured["equilibrated_gap_response_m_per_pa"],
                                   rtol=1.0e-7, atol=1.0e-10)
        candidate = np.asarray(selected["pressure"], dtype=float)
        if (np.any(candidate < 0.0)
                or np.max(candidate) > kwargs["maximum_pressure"]
                or np.max(np.abs(candidate-pressure))
                > kwargs["maximum_pressure_step"]):
            raise ValueError("captured pressure proposal exceeds bounds")
        stiffness = kwargs["normal_stiffness"]
        current_arg = pressure-stiffness*gap
        new_gap = gap+response @ (candidate-pressure)
        predicted_arg = candidate-stiffness*new_gap
        before = float(np.linalg.norm(
            np.maximum(0.0, current_arg)-pressure, ord=np.inf))
        after = float(np.linalg.norm(
            np.maximum(0.0, predicted_arg)-candidate, ord=np.inf))
        if after >= 0.75*before:
            raise ValueError("captured proposal misses predicted merit")
        full = lam.copy()
        full[modes] = candidate
        active = tuple(int(modes[index]) for index in
                       np.flatnonzero(predicted_arg > 0.0))
        diagnostics = kwargs.get("diagnostics")
        if diagnostics is not None:
            diagnostics["schur_proposal_kind"] = "independent_bounded_trial"
            diagnostics["schur_independent_predicted_residual_pa"] = after
        return SupportedSchurProposal(full, active, None, 1.0, before, after)

    pivot.propose_supported_schur_step = substitute
    try:
        model, definition, settings, program = build_case(16)
        result = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=12,
            emit_restart_checkpoint=True,
        )
    finally:
        pivot.propose_supported_schur_step = original
    record = {
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "has_restart_checkpoint": result.restart_checkpoint is not None,
        "proposal_calls": proposals,
        "contact_failure": result.info.get("contact_failure"),
        "force_displacement_history": result.info.get(
            "force_displacement_history"),
    }
    Path(output_path).write_text(json.dumps(record, indent=2, default=str)
                                 + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("contact_failure",
                                     "force_displacement_history")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
