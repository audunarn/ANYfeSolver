"""One bounded development run of the force-driven Q4/S3 four-phase path."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import anymaterial
import anyfesolver

from anyfesolver import solve_static_nonlinear, validate_general_contact
from examples.general_contact_refined_q4_s3_mixed_path import make_case


def make_candidate():
    model, definition, settings, program = make_case()
    definition = replace(definition, interactions=(replace(
        definition.interactions[0], normal_stiffness=10000.0,
    ),))
    settings = replace(
        settings,
        q4_source_gap_certificate="projected_matching",
        general_s3_differentiation="jet2_q4_s3_dev",
        q4_source_support_gradient_tolerance=1.0e-5,
        q4_source_support_hessian_tolerance=1.0e-5,
        q4_source_weighted_gap_gradient_tolerance=1.0e-5,
        q4_source_weighted_gap_hessian_tolerance=1.0e-5,
        pressure_tolerance=1.0e-5,
        maximum_augmentations=7,
        augmentation_acceleration="supported_schur_dev",
        augmentation_schur_maximum_modes=6,
        augmentation_schur_maximum_pressure_pa=10.0,
        augmentation_schur_maximum_step_pa=5.0,
        augmentation_schur_difference_step_m=2.0e-6,
        augmentation_schur_maximum_attempts=2,
        augmentation_schur_maximum_relinearizations=0,
    )
    capability = validate_general_contact(model, definition, settings)
    if not capability.valid:
        raise ValueError(f"mixed-path candidate is unsupported: {capability.issues}")
    return model, definition, settings, program, capability


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"development output already exists: {output}")
    model, definition, settings, program, capability = make_candidate()
    started = time.perf_counter()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    history = result.info.get("force_displacement_history", ())
    record = {
        "candidate": "force_driven_q4_s3_jet2_schur_four_phase_dev",
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter() - started,
        "profile": capability.profile,
        "normal_stiffness_pa_per_m": 10000.0,
        "gap_tolerance_m": definition.interactions[0].gap_tolerance,
        "pressure_tolerance_pa": settings.pressure_tolerance,
        "maximum_augmentations": settings.maximum_augmentations,
        "has_checkpoint": result.restart_checkpoint is not None,
        "contact_failure": result.info.get("contact_failure"),
        "history": [{
            "step": row.get("step_index"),
            "target_m": row.get("target"),
            "nodal_forces_n": row.get("nodal_forces_n"),
            "active_pressure_dofs": row["contact"]["active_pressure_dofs"],
            "minimum_gap_m": row["contact"]["minimum_gap"],
            "gap_tolerance_met": bool(row["contact"]["gap_tolerance_met"]),
            "integration_force_n": row["contact"]["integration_force"],
            "target_force_n": row["contact"]["target_force"],
            "area_integration_error_m2": row["contact"].get(
                "area_integration_error_m2"),
            "weighted_gap_integration_error_m3": row["contact"].get(
                "weighted_gap_integration_error_m3"),
            "normal_multiplier_change_pa": row.get("normal_multiplier_change"),
            "augmentation_iterations": row.get("augmentation_iterations"),
            "energy_evaluations": row["contact"].get("energy_evaluations"),
            "pair_geometry_evaluations": row["contact"].get(
                "pair_geometry_evaluations"),
        } for row in history],
        "runtime_identity": {
            "python": sys.version.split()[0],
            "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
            "anymaterial_module": str(Path(anymaterial.__file__).resolve()),
        },
    }
    output.write_text(json.dumps(record, indent=2, default=str) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "status", "failure_reason", "accepted_steps", "elapsed_seconds",
        "has_checkpoint", "history",
    )}, indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python general_contact_d1_mixed_jet2_schur_probe.py OUTPUT")
    main(Path(sys.argv[1]))
