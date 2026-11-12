"""Bounded local-compliance scale screen for the D1 two-hat cut family.

This reads production shell/contact work at touching, but makes no contact
law or acceptance claim. It distinguishes the cancelling global pressure
response from a diagonal structural-compliance proxy for individual hats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.semismooth import (
    linearize, reference_nodal_support,
)
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from docs.d1_review_support.d1_cut_front_shell_work import build_patch
from docs.d1_review_support.d1_cut_front_two_hat_limit import (
    limit_matrix, null_coefficients,
)
from docs.d1_review_support.d1_jump_stabilization_screen import (
    X, Y, edge_jump_matrix,
)


def _pressure(theta: float, node_ids: tuple[int, ...]) -> np.ndarray:
    factors = null_coefficients(sum(limit_matrix()))
    b = theta**2/2.0-theta**3/3.0
    d = 0.25+theta-theta**2+theta**3/3.0
    beta = -b/d
    coefficients = np.zeros(15)
    for column, factor in ((2, factors[0]), (3, factors[1])):
        for row, y_factor in enumerate((-beta/2.0, beta, 1.0)):
            coefficients[5*row+column] = factor*y_factor
    return coefficients[np.asarray(node_ids)-1]


def screen(theta: float) -> dict:
    model, definition, settings = build_patch(theta, touching=True)
    manager = GeneralS3PatchManager(model, definition, settings)
    zero = np.zeros(model.mesh.dof_manager.total_dofs)
    node_ids = tuple(manager.law.pressure_mode_ids)
    raw = linearize(manager, zero, np.zeros(len(node_ids)), tangent=False)
    support = np.asarray(raw.state.reference_support)
    full_support = reference_nodal_support(manager.law)
    pressure = _pressure(theta, node_ids)

    # Independent tensor-Q1 support on the horizontal-cut source grid.
    x_support = np.array((0.25, 0.5, 0.5, 0.5, 0.25))
    y_support = np.array((0.5, 0.5+theta-theta**2/2.0, theta**2/2.0))
    expected = np.outer(y_support, x_support).ravel()[
        np.asarray(node_ids)-1
    ]
    expected_full = np.outer((0.5, 1.0, 0.5), x_support).ravel()[
        np.asarray(node_ids)-1
    ]
    if not np.allclose(support, expected, rtol=1.0e-12, atol=1.0e-14):
        raise AssertionError("production pressure support differs from exact Q1 integral")
    if not np.allclose(full_support, expected_full, rtol=1.0e-12, atol=1.0e-14):
        raise AssertionError("production full support differs from exact Q1 integral")

    stiffness = assemble_stiffness_matrix(model)[0]
    _, _, transform, offset, _, _ = build_constraint_transformation(
        stiffness, zero, model,
    )
    if np.linalg.norm(offset) != 0.0:
        raise AssertionError("unexpected structural constraint offset")
    reduced = np.asarray((transform.T @ stiffness @ transform).toarray())
    if np.linalg.eigvalsh(0.5*(reduced+reduced.T))[0] <= 0.0:
        raise AssertionError("contact-free reduced shell energy is not positive")
    if np.linalg.norm(reduced-reduced.T) > 1.0e-10*np.linalg.norm(reduced):
        raise AssertionError("contact-free reduced shell energy is not symmetric")
    diagonal = np.diag(reduced)
    if np.any(diagonal <= 0.0):
        raise AssertionError("local structural compliance has nonpositive diagonal")

    # Each row maps a 1-Pa pressure coefficient into reduced nodal work.
    # Report quadratic values divided by (1 Pa)^2 so units are m^3/Pa.
    work = np.asarray((support[:, None]*raw.gap_gradient) @ transform)
    structural_response = np.linalg.solve(reduced, work.T)
    schur = work @ structural_response
    if np.linalg.norm(schur-schur.T) > 1.0e-10*max(np.linalg.norm(schur), 1.0e-30):
        raise AssertionError("structural pressure-response block is not symmetric")
    signed = pressure @ work
    global_response = float(pressure @ schur @ pressure)
    direct_global = float(signed @ np.linalg.solve(reduced, signed))
    if not np.isclose(global_response, direct_global, rtol=1.0e-9, atol=1.0e-20):
        raise AssertionError("pressure-response and direct structural solves differ")
    exact_individual_hat = float(np.dot(pressure**2, np.diag(schur)))
    if exact_individual_hat <= 0.0:
        raise AssertionError("exact individual-hat compliance is not positive")
    individual_hat_response = np.sum(work**2/diagonal[None, :], axis=1)
    local_diagonal = float(np.dot(pressure**2, individual_hat_response))

    x, y = np.meshgrid(X, Y)
    order = np.asarray(node_ids)-1
    basis = np.column_stack((np.ones(15), x.ravel(), y.ravel(),
                             (x*y).ravel()))[order]
    projection = basis @ np.linalg.solve(
        basis.T @ (support[:, None]*basis), basis.T*support[None, :],
    )
    fluctuation = np.eye(len(node_ids))-projection
    weight = np.diag(support/full_support)
    jump = edge_jump_matrix()[np.ix_(order, order)]
    stabilized = fluctuation.T @ weight @ jump @ weight @ fluctuation
    stabilized = 0.5*(stabilized+stabilized.T)
    jump_work = float(pressure @ stabilized @ pressure)
    if not jump_work > 0.0:
        raise AssertionError("cut mode has no projected-jump work")
    return {
        "theta": theta,
        "pressure_node_ids": node_ids,
        "nonzero_pressure_coefficients_pa": [
            {"node_id": node_id, "pressure_pa": float(value)}
            for node_id, value in zip(node_ids, pressure) if value != 0.0
        ],
        "global_reduced_compliance_m3_per_pa": global_response,
        "exact_individual_hat_compliance_m3_per_pa": exact_individual_hat,
        "individual_hat_diagonal_proxy_m3_per_pa": local_diagonal,
        "projected_jump_m2": jump_work,
        "exact_hat_to_jump_scale_m_per_pa": exact_individual_hat/jump_work,
        "diagonal_proxy_to_exact_hat_ratio": local_diagonal/exact_individual_hat,
        "proxy_to_jump_scale_m_per_pa": local_diagonal/jump_work,
        "global_over_theta6": global_response/theta**6,
        "exact_hat_over_theta4": exact_individual_hat/theta**4,
        "proxy_over_theta4": local_diagonal/theta**4,
        "jump_over_theta4": jump_work/theta**4,
        "pressure_support_max_abs_error_m2": float(np.max(np.abs(support-expected))),
        "full_support_max_abs_error_m2": float(np.max(np.abs(full_support-expected_full))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if any(not 0.0 < theta <= 0.1 for theta in args.theta):
        parser.error("each theta must lie in (0, .1]")
    if not 1 <= len(args.theta) <= 5:
        parser.error("one to five cut values are allowed in this bounded screen")
    result = {
        "status": "development_scale_screen_only",
        "source": "production structural/contact work, independent exact support/jump",
        "rows": [screen(theta) for theta in args.theta],
    }
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps([{
        "theta": row["theta"],
        "global_over_theta6": row["global_over_theta6"],
        "exact_hat_over_theta4": row["exact_hat_over_theta4"],
        "proxy_over_theta4": row["proxy_over_theta4"],
        "jump_over_theta4": row["jump_over_theta4"],
        "exact_hat_to_jump_scale_m_per_pa": row["exact_hat_to_jump_scale_m_per_pa"],
        "proxy_to_jump_scale_m_per_pa": row["proxy_to_jump_scale_m_per_pa"],
    } for row in result["rows"]]))


if __name__ == "__main__":
    main()
