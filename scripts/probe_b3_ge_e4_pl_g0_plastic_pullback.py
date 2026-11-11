"""Bounded G0B F05 S3 plastic pullback diagnostic; no product admission."""

from __future__ import annotations

import json

import numpy as np

from anyfesolver._ge_beam3_variational_shell import deformation
from anyfesolver.e4_pl_s3_v2d_state import canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve


def main() -> int:
    reference = np.asarray(
        ((0.0, 0.0, 0.0), (1.2, 0.05, 0.0), (0.2, 0.95, 0.0)),
        dtype=np.float64,
    )
    model = FEModel("g0-plastic-pullback-diagnostic")
    curve = DNVC208MaterialCurve(
        sigma_prop=80.0e6, sigma_yield=85.0e6,
        sigma_yield_2=100.0e6, eps_p_y1=0.002, eps_p_y2=0.02,
        K=280.0e6, n=0.18,
    )
    material = model.add_material(
        "steel", 210.0e9, 0.3, density=7850.0, hardening_curve=curve,
    )
    for node_id, coordinate in enumerate(reference, start=1):
        model.add_node(node_id, *coordinate)
    element = create_shell_element(
        1, [1, 2, 3], "steel", formulation="e4-pl-s3-v2d",
        thickness=0.018, reference_normal=(0.0, 0.0, 1.0),
    )
    model.add_element(1, element)
    virgin = element.init_model_bound_nonlinear_state(model.mesh, material, 5)
    committed_u = np.zeros(18)
    committed_u[6], committed_u[12] = 0.008, -0.002
    _, _, predecessor = element.compute_nonlinear_response(
        model.mesh, material, committed_u, virgin, 5, True,
    )
    before = canonical_json_bytes(predecessor)

    candidate_u = committed_u.copy()
    candidate_u[6] += 0.0002
    candidate_u[12] -= 0.00005
    candidate_u[3:6] = (0.003, -0.002, 0.001)
    candidate_u[9:12] = (-0.001, 0.002, -0.003)
    candidate_u[15:18] = (0.001, 0.001, 0.002)

    def evaluate(global_u: np.ndarray, tangent: bool):
        local_u, mapping, second = deformation(reference, global_u)
        force, local_tangent, trial = element.compute_nonlinear_response(
            model.mesh, material, local_u, predecessor, 5, tangent,
        )
        pulled_force = mapping.T @ force
        if not tangent:
            return pulled_force, None, trial
        pulled_tangent = (
            mapping.T @ local_tangent @ mapping
            + np.einsum("i,ijk->jk", force, second)
        )
        return pulled_force, pulled_tangent, trial

    force, tangent, candidate = evaluate(candidate_u, True)
    direction = np.sin(np.arange(18, dtype=np.float64) + 0.3)
    direction /= np.linalg.norm(direction)
    expected = tangent @ direction
    errors = []
    for step in (2.0e-6, 2.0e-7, 2.0e-8):
        plus, _, _ = evaluate(candidate_u + step * direction, False)
        minus, _, _ = evaluate(candidate_u - step * direction, False)
        numerical = (plus - minus) / (2.0 * step)
        errors.append(float(np.linalg.norm(expected - numerical)
                            / max(1.0, np.linalg.norm(numerical))))

    active = bool(np.any(candidate["alpha"] > predecessor["alpha"]))
    parent_unchanged = canonical_json_bytes(predecessor) == before
    finite = bool(np.isfinite(force).all() and np.isfinite(tangent).all())
    plateau = any(a <= 1.0e-5 and b <= 1.0e-5
                  for a, b in zip(errors, errors[1:]))
    success = active and parent_unchanged and finite and plateau
    print(json.dumps({
        "case": "F05_S3_V2D_PLASTIC_PULLBACK_DIAGNOSTIC_V1",
        "status": "diagnostic_pass" if success else "diagnostic_fail",
        "active_plasticity": active,
        "predecessor_unchanged": parent_unchanged,
        "finite": finite,
        "relative_skew": float(np.linalg.norm(tangent - tangent.T)
                               / max(1.0, np.linalg.norm(tangent))),
        "steps": (2.0e-6, 2.0e-7, 2.0e-8),
        "directional_errors": errors,
    }, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
