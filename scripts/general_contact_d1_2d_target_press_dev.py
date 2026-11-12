"""Bounded D1 press with a two-dimensional, deformable S3 target patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import anyfesolver
from anyfesolver import BoundaryCondition, solve_static_nonlinear, validate_general_contact

from scripts.general_contact_d1_interior_press_dev import build_interior_press
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import _json_default


def build_case():
    model, definition, settings, program = build_interior_press(
        force_n=0.04, reference_slope=1.0e-4,
        target_columns=2, target_rows=2,
        source_support="perimeter", source_columns=4, source_rows=4,
        load_shape="interior_distributed", slide_body="target",
    )
    target_start = 26
    target_width = 3
    other_boundary = tuple(
        target_start + row*target_width + column
        for row in range(3) for column in range(3)
        if (row in (0, 2) or column in (0, 2))
        and (row, column) not in ((0, 0), (2, 0))
    )
    model.add_boundary_condition(BoundaryCondition(
        "target_other_perimeter_clamp", other_boundary,
        {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
    ))
    return model, definition, settings, program


def main(output: Path):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    model, definition, settings, program = build_case()
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"two-dimensional target rejected: {report.issues}")
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    payload = {
        "candidate": "d1_interior_two_dimensional_target_press_development",
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "source_q4_faces": 16,
        "target_s3_faces": 8,
        "force_n": 0.04,
        "reference_slope": 1.0e-4,
        "target_interior_node_id": 30,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "elapsed_seconds": time.perf_counter()-started,
        "displacements": result.displacements.tolist(),
        "history": result.info.get("force_displacement_history"),
        "contact_failure": result.info.get("contact_failure"),
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
    main(parser.parse_args().output)
