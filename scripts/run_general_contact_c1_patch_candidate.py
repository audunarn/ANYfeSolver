"""Run the prospective C1 mixed-patch analytic work/geometry candidate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from anyfesolver import (
    ContactDefinition, ContactInteraction, ContactSolverSettings,
    ContactSurface, create_shell_element, validate_general_contact,
)
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.patch_sliding import _RestrictedSlidingPatchFEManager


CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "reference_cases" / "general_contact_c1_patch_candidate_v0.json"


def _model() -> tuple[FEModel, ContactDefinition, ContactSolverSettings]:
    model = FEModel("c1_mixed_patch_candidate_v0")
    model.add_material("elastic", 1.0e5, 0.3, density=1.0)
    lower = ((0.2, 0.1), (1.2, 0.1), (1.2, 1.1), (0.2, 1.1))
    upper = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for node, (x, y) in enumerate(lower, 1):
        model.add_node(node, x, y, 0.0)
    for node, (x, y) in enumerate(upper, 5):
        model.add_node(node, x, y, 0.04)
    for element, nodes in ((1, (1, 2, 3)), (2, (1, 3, 4))):
        model.add_element(element, create_shell_element(
            element, nodes, "elastic", thickness=0.05,
            formulation="e4-pl-s3-v2d", reference_normal=(0, 0, 1),
            reference_surface_offset=0.01,
        ))
    model.add_element(3, create_shell_element(
        3, (5, 6, 7, 8), "elastic", thickness=0.05,
    ))
    definition = ContactDefinition(
        surfaces=(ContactSurface("upper", (3,), side="bottom"),
                  ContactSurface("lower", (1, 2), side="top")),
        rigid_bodies=(),
        interactions=(ContactInteraction(
            "mixed", "upper", "lower", 1.0e6, 6.0e-6,
        ),),
    )
    settings = ContactSolverSettings(
        pair_motion="restricted_sliding", sliding_difference_step=1.0e-8,
    )
    return model, definition, settings


def _relative(actual, reference, scale: float) -> float:
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(reference)) / scale)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["id"] != "GENERAL_CONTACT_C1_PATCH_CANDIDATE_V0":
        raise ValueError("wrong candidate contract")
    model, definition, settings = _model()
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError("candidate admission failed: " + "; ".join(report.issues))
    manager = _RestrictedSlidingPatchFEManager(model, definition, settings)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs)
    for node in (5, 6, 7, 8):
        displacement[model.mesh.dof_manager.get_node_dofs(node)[2]] = -1.0e-6
    force, tangent, result = manager.evaluate(displacement)
    expected = contract["independent_reference"]
    criteria = contract["criteria"]
    measurements = {
        "area_m2": float(result.current_overlap_area),
        "minimum_sampled_gap_m": float(result.minimum_sampled_gap),
        "conservative_lower_gap_m": float(result.conservative_lower_gap),
        "normal_force_N": float(result.integration_force[2]),
        "upper_moment_Nm": list(map(float, result.integration_moment)),
        "target_force_N": list(map(float, result.target_force)),
        "target_moment_Nm": list(map(float, result.target_moment)),
        "penalty_energy_J": float(result.penalty_energy),
        "pressure_rank": int(result.pressure_rank),
        "active_pressure_dofs": int(result.active_pressure_dofs),
        "gap_tolerance_met": bool(result.gap_tolerance_met),
    }
    checks: dict[str, bool] = {}
    checks["overlap_area"] = abs(measurements["area_m2"] - contract["geometry"]["overlap_area_m2"]) <= criteria["overlap_area_absolute_m2"]
    checks["force"] = abs(measurements["normal_force_N"] - expected["normal_force_N"]) / contract["force_scale_N"] <= criteria["force_relative"]
    checks["moment"] = _relative(measurements["upper_moment_Nm"], expected["upper_normal_moment_Nm"], contract["moment_scale_Nm"]) <= criteria["moment_relative"]
    checks["energy"] = abs(measurements["penalty_energy_J"] - expected["penalty_energy_J"]) / contract["energy_scale_J"] <= criteria["energy_relative"]
    checks["physical_gap"] = (abs(measurements["minimum_sampled_gap_m"] - expected["pointwise_gap_m"]) <= criteria["gap_absolute_m"]
                              and measurements["conservative_lower_gap_m"] >= -definition.interactions[0].gap_tolerance
                              and measurements["gap_tolerance_met"])
    checks["pressure_space"] = (measurements["pressure_rank"] == expected["pressure_rank"]
                                and measurements["active_pressure_dofs"] == expected["active_pressure_dofs"])
    checks["paired_wrench"] = (
        _relative(np.asarray(result.integration_force) + np.asarray(result.target_force),
                  np.zeros(3), contract["force_scale_N"]) <= criteria["paired_wrench_relative"]
        and _relative(np.asarray(result.integration_moment) + np.asarray(result.target_moment),
                      np.zeros(3), contract["moment_scale_Nm"]) <= criteria["paired_wrench_relative"]
    )
    exchanged = replace(definition,
        surfaces=definition.surfaces[::-1],
        interactions=(replace(definition.interactions[0], surface="lower", target="upper"),))
    exchanged_report = validate_general_contact(model, exchanged, settings)
    if not exchanged_report.valid:
        raise ValueError("role exchange admission failed: " + "; ".join(exchanged_report.issues))
    exchanged_result = _RestrictedSlidingPatchFEManager(
        model, exchanged, settings,
    ).evaluate(displacement, tangent=False)[2]
    measurements["role_exchange_normal_force_N"] = float(exchanged_result.target_force[2])
    measurements["role_exchange_energy_J"] = float(exchanged_result.penalty_energy)
    checks["role_exchange"] = (
        abs(measurements["role_exchange_normal_force_N"] - measurements["normal_force_N"])
        / contract["force_scale_N"] <= criteria["role_exchange_integrated_relative"]
        and abs(measurements["role_exchange_energy_J"] - measurements["penalty_energy_J"])
        / contract["energy_scale_J"] <= criteria["role_exchange_integrated_relative"]
    )
    direction = np.zeros_like(displacement)
    for node in (5, 6, 7, 8):
        dofs = model.mesh.dof_manager.get_node_dofs(node)
        direction[dofs[0]] = 0.02
        direction[dofs[2]] = -1.0e-6
    derivative = tangent @ direction
    tangent_errors = []
    for step in criteria["tangent_steps"]:
        plus = manager.evaluate(displacement + step * direction, tangent=False)[0]
        minus = manager.evaluate(displacement - step * direction, tangent=False)[0]
        finite = (plus - minus) / (2.0 * step)
        tangent_errors.append(_relative(finite, derivative,
                                        max(float(np.linalg.norm(finite)), 1.0e-12)))
    measurements["directional_tangent_relative_errors"] = tangent_errors
    checks["smooth_tangent"] = all(
        error <= criteria["directional_tangent_relative"] for error in tangent_errors
    )
    output = {
        "candidate": contract["id"],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "contract_path": str(CONTRACT),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "measurements": measurements,
        "checks": checks,
        "mechanical_candidate_pass": all(checks.values()),
        "engineering_acceptance": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": output["candidate"],
                      "mechanical_candidate_pass": output["mechanical_candidate_pass"],
                      "checks": checks, "result_path": str(args.output)}, indent=2))
    return 0 if output["mechanical_candidate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
