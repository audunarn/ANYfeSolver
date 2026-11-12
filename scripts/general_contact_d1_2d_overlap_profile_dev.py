"""Bounded integrated 2D target press with explicit overlap-local gap profile."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import anyfesolver
from anyfesolver import solve_static_nonlinear, validate_general_contact

from scripts.general_contact_d1_2d_target_press_dev import build_case
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import _json_default


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    model, definition, settings, program = build_case()
    settings = replace(settings, q4_source_gap_certificate="overlap_local_dev")
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"overlap-local development profile rejected: {report.issues}")
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    payload = {
        "candidate": "d1_2d_target_overlap_local_profile_development",
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "profile": report.profile,
        "certificate": settings.q4_source_gap_certificate,
        "force_n": 0.04,
        "reference_slope": 1.0e-4,
        "registered_gap_tolerance_m": definition.interactions[0].gap_tolerance,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter()-started,
        "displacements": result.displacements.tolist(),
        "history": result.info.get("force_displacement_history"),
        "contact_failure": result.info.get("contact_failure"),
        "restart_checkpoint": (result.to_restart_checkpoint()
                               if result.status == "completed" else None),
    }
    output.write_text(json.dumps(payload, indent=2, default=_json_default)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "profile", "status", "failure_reason", "accepted_steps",
        "elapsed_seconds",
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
