"""Capture one coupled pressure response and test bounded alternatives.

Development-only readout. No candidate is applied to the mechanical solve.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import anyfesolver.general_contact.linearized_pivot as pivot
from anyfesolver import solve_static_nonlinear

from general_contact_d1_coupled_refinement import build_case


def main(output, columns, proposal_index):
    captured = []
    original = pivot.propose_supported_schur_step

    def capture(multipliers, normalized_gap, supported,
                equilibrated_gap_response, **kwargs):
        selected = np.flatnonzero(supported)
        captured.append({
            "supported_indices": selected.copy(),
            "multipliers": np.asarray(multipliers)[selected].copy(),
            "gap": np.asarray(normalized_gap)[selected].copy(),
            "response": np.asarray(equilibrated_gap_response).copy(),
            "stiffness": kwargs["normal_stiffness"],
            "max_step": kwargs["maximum_pressure_step"],
            "max_pressure": kwargs["maximum_pressure"],
        })
        return original(multipliers, normalized_gap, supported,
                        equilibrated_gap_response, **kwargs)

    pivot.propose_supported_schur_step = capture
    try:
        model, definition, settings, program = build_case(columns)
        result = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=12,
            emit_restart_checkpoint=True,
        )
    finally:
        pivot.propose_supported_schur_step = original
    if len(captured) <= proposal_index:
        raise RuntimeError("the requested pressure response was not reached")
    state = captured[proposal_index]
    lam = state["multipliers"]
    gap = state["gap"]
    response = state["response"]
    epsilon = state["stiffness"]
    lower = np.maximum(0.0, lam-state["max_step"])
    upper = np.minimum(state["max_pressure"], lam+state["max_step"])

    def projected_residual(candidate):
        predicted_gap = gap + response @ (candidate-lam)
        return np.maximum(0.0, candidate-epsilon*predicted_gap)-candidate

    before = float(np.linalg.norm(projected_residual(lam), ord=np.inf))
    solutions = []
    for seed in (lam, lower, upper, np.clip(lam+0.01, lower, upper)):
        optimum = least_squares(projected_residual, seed,
                                bounds=(lower, upper),
                                xtol=1.0e-13, ftol=1.0e-13,
                                gtol=1.0e-13, max_nfev=500)
        candidate = optimum.x
        solutions.append({
            "pressure": candidate.tolist(),
            "predicted_active_indices": state["supported_indices"][
                candidate-epsilon*(gap+response @ (candidate-lam)) > 0.0
            ].tolist(),
            "residual_inf_pa": float(np.linalg.norm(
                projected_residual(candidate), ord=np.inf)),
            "residual_l2_pa": float(np.linalg.norm(
                projected_residual(candidate))),
            "success": bool(optimum.success),
            "evaluations": int(optimum.nfev),
        })
    evidence = {
        "columns": columns,
        "proposal_index": proposal_index,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "candidate_count": len(captured),
        "supported_indices": state["supported_indices"].tolist(),
        "initial_pressure_pa": lam.tolist(),
        "initial_normalized_gap_m": gap.tolist(),
        "equilibrated_gap_response_m_per_pa": response.tolist(),
        "normal_stiffness_pa_per_m": epsilon,
        "maximum_pressure_step_pa": state["max_step"],
        "maximum_pressure_pa": state["max_pressure"],
        "initial_projected_residual_pa": before,
        "required_residual_pa": 0.75*before,
        "bounded_least_squares_solutions": solutions,
    }
    Path(output).write_text(json.dumps(evidence, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({
        "status": result.status,
        "initial_residual_pa": before,
        "required_residual_pa": 0.75*before,
        "best_bounded_residual_pa": min(x["residual_inf_pa"]
                                        for x in solutions),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--proposal-index", type=int, default=1)
    args = parser.parse_args()
    main(args.output, args.columns, args.proposal_index)
