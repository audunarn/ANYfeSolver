"""One bounded D1-interior target-normal-displacement development pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import anyfesolver
from anyfesolver import (
    BoundaryCondition, ContactPathControl, ContactPathProgram,
    solve_static_nonlinear, validate_general_contact,
)

from scripts.general_contact_d1_interior_press_dev import build_interior_press
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import _json_default


def build_case(normal_motion_m: float = 6.0e-5,
               force_reference_n: float = 5.0e-5):
    model, definition, settings, _ = build_interior_press(
        source_support="perimeter", source_rows=4, source_columns=4,
        target_columns=2, load_shape="interior_distributed", force_n=0.04,
        reference_slope=1.0e-4, slide_body="target",
    )
    # The contact-free elastic two-control predictor is 1.15667915e-4 N
    # per 10 µm of equal target motion in this fixture. Keep the explicit
    # development force scale below that structural reaction bound.
    settings = replace(settings,
                       complementarity_force_scale_n=force_reference_n)
    target_start = 26
    target_left_top = target_start + 3
    target_middle_bottom = target_start + 1
    target_middle_top = target_start + 4
    model.add_boundary_condition(BoundaryCondition(
        "target_left_x_anchor", (target_start, target_left_top),
        {"ux": 0.0},
    ))
    program = ContactPathProgram(
        controls=(ContactPathControl.node(target_middle_bottom, "uz"),
                  ContactPathControl.node(target_middle_top, "uz")),
        waypoints=((0.0, 0.0), (normal_motion_m, normal_motion_m)),
        steps_per_segment=2,
    )
    return model, definition, settings, program


def main(output: Path, normal_motion_m: float, force_reference_n: float):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    model, definition, settings, program = build_case(
        normal_motion_m, force_reference_n)
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"fixture capability rejected: {report.issues}")
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    payload = {
        "candidate": "d1_interior_target_normal_displacement_development",
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "source_q4_faces": 16,
        "target_s3_faces": 4,
        "normal_motion_m": normal_motion_m,
        "force_reference_n": force_reference_n,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter()-started,
        "history": result.info.get("force_displacement_history"),
        "contact_failure": result.info.get("contact_failure"),
        "displacements": result.displacements.tolist(),
    }
    output.write_text(json.dumps(payload, indent=2, default=_json_default)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "failure_reason", "accepted_steps", "elapsed_seconds",
        "contact_failure",
    )}, default=_json_default))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--normal-motion-m", type=float, default=6.0e-5)
    parser.add_argument("--force-reference-n", type=float, default=5.0e-5)
    args = parser.parse_args()
    main(args.output, args.normal_motion_m, args.force_reference_n)
