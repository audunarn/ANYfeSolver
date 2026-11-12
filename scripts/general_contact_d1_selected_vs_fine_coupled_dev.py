"""Compare selected and fine pressure bases on one bounded coupled Q4/S3 press."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from anyfesolver import solve_static_nonlinear, validate_general_contact
from anyfesolver.general_contact.pressure_basis import (
    build_q4_source_selected_q1_basis,
)
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import build_case
from scripts.general_contact_d1_selected_field_compare import _mass


def _run(basis: str, columns: int,
         custom_knots: tuple[int, ...] | None = None) -> dict:
    model, definition, settings, program = build_case(
        columns, steps_per_segment=2, load_profile="distributed",
        total_force_n=0.3, target_right_x_m=1.3,
    )
    selected_knots = (custom_knots if custom_knots is not None
                      else (0, 1, 2, 4) if columns == 4
                      else (0, 1, 2, 4, 6, 8))
    if basis == "selected_q1_knots_dev":
        settings = replace(
            settings, q4_source_pressure_basis=basis,
            q4_source_pressure_knot_columns=selected_knots,
        )
    report = validate_general_contact(model, definition, settings)
    if not report.valid:
        raise ValueError(f"{basis} admission: {report.issues}")
    start = time.perf_counter()
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=settings,
        contact_path_program=program, max_iterations=18,
        emit_restart_checkpoint=True,
    )
    contact = result.info.get("general_contact") or {}
    record = {
        "basis": basis,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "elapsed_seconds": time.perf_counter()-start,
        "accepted_steps": len(result.steps),
        "final_displacements_m": result.displacements.tolist(),
        "contact": contact,
        "history": result.info.get("force_displacement_history"),
        "augmentation_attempts": result.info.get("contact_augmentation_attempts"),
        "contact_failure": result.info.get("contact_failure"),
        "profile": report.profile,
    }
    if result.status == "completed":
        mode_ids = tuple(contact["source_pressure_node_ids"])
        mode_pressures = np.asarray(contact["numerical_pressure_pa"], dtype=float)
        faces = tuple(tuple(model.mesh.elements[face].node_ids)
                      for face in definition.surfaces[0].element_ids)
        if basis == "selected_q1_knots_dev":
            transform = build_q4_source_selected_q1_basis(
                faces, settings.q4_source_pressure_knot_columns,
            )
            if mode_ids != transform.mode_ids:
                raise AssertionError("selected pressure mode IDs changed")
            fine_ids = transform.fine_source_node_ids
            fine_pressure = transform.as_array() @ mode_pressures
        else:
            fine_ids = mode_ids
            fine_pressure = mode_pressures
        record["fine_node_ids"] = fine_ids
        record["fine_pressure_pa"] = fine_pressure.tolist()
    return record


def main(output: Path, columns: int = 4, basis_only: str | None = None,
         custom_knots: tuple[int, ...] | None = None) -> None:
    if output.exists():
        raise FileExistsError(output)
    if columns not in (4, 8):
        raise ValueError("this comparison admits four or eight source Q4 faces")
    if custom_knots is not None and (basis_only != "selected_q1_knots_dev"
            or columns != 8 or len(custom_knots) != 6
            or custom_knots[0] != 0 or custom_knots[-1] != 8
            or any(right <= left for left, right in
                   zip(custom_knots, custom_knots[1:]))):
        raise ValueError("custom knots need one selected eight-Q4 six-knot case")
    if basis_only is not None:
        output.write_text(json.dumps({
            "candidate": f"selected_vs_fine_{columns}q4_{basis_only}_dev",
            "source_q4_faces": columns,
            "target_s3_faces": 2,
            "distributed_downward_force_n": 0.3,
            "target_right_x_m": 1.3,
            "steps_per_segment": 2,
            "selected_knot_columns": list(custom_knots or
                ((0, 1, 2, 4) if columns == 4 else (0, 1, 2, 4, 6, 8))),
            "case": _run(basis_only, columns, custom_knots),
        }, indent=2, default=str) + "\n", encoding="utf-8")
        return
    fine = _run("fine_nodal_q1", columns)
    selected = _run("selected_q1_knots_dev", columns)
    comparison = None
    if fine["status"] == selected["status"] == "completed":
        if fine["fine_node_ids"] != selected["fine_node_ids"]:
            raise AssertionError("fine pressure node identities differ")
        fine_p = np.asarray(fine["fine_pressure_pa"])
        selected_p = np.asarray(selected["fine_pressure_pa"])
        fine_u = np.asarray(fine["final_displacements_m"])
        selected_u = np.asarray(selected["final_displacements_m"])
        fine_c, selected_c = fine["contact"], selected["contact"]
        comparison = {
            "pressure_nodal_relative_l2": float(
                np.linalg.norm(selected_p-fine_p)/np.linalg.norm(fine_p)
            ),
            "displacement_relative_l2": float(
                np.linalg.norm(selected_u-fine_u)/np.linalg.norm(fine_u)
            ),
            "normal_force_difference_n": float(
                np.asarray(selected_c["integration_force"])[2]
                - np.asarray(fine_c["integration_force"])[2]
            ),
            "minimum_gap_difference_m": float(
                selected_c["minimum_gap"]-fine_c["minimum_gap"]
            ),
        }
    record = {
        "candidate": f"selected_vs_fine_{columns}q4_two_s3_distributed_press_dev",
        "question": "Does selected pressure space preserve the fine-basis coupled field on this partial-contact fixture?",
        "same_mechanics": {
            "source_q4_faces": columns,
            "target_s3_faces": 2,
            "distributed_downward_force_n": 0.3,
            "target_right_x_m": 1.3,
            "steps_per_segment": 2,
            "selected_knot_columns": ([0, 1, 2, 4] if columns == 4
                                      else [0, 1, 2, 4, 6, 8]),
        },
        "fine": fine,
        "selected": selected,
        "comparison": comparison,
    }
    output.write_text(json.dumps(record, indent=2, default=str)+"\n",
                      encoding="utf-8")
    print(json.dumps({
        "fine_status": fine["status"],
        "selected_status": selected["status"],
        "fine_seconds": fine["elapsed_seconds"],
        "selected_seconds": selected["elapsed_seconds"],
        "comparison": comparison,
    }, indent=2))
    if comparison is None:
        raise RuntimeError("coupled selected/fine press did not complete")


def compare_saved(fine_path: Path, selected_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fine_data = fine_path.read_bytes()
    selected_data = selected_path.read_bytes()
    fine_record = json.loads(fine_data)
    selected_record = json.loads(selected_data)
    identity_keys = ("source_q4_faces", "target_s3_faces",
                     "distributed_downward_force_n", "target_right_x_m",
                     "steps_per_segment")
    if any(fine_record.get(key) != selected_record.get(key)
           for key in identity_keys):
        raise ValueError("saved cases do not have matching mechanics")
    fine = fine_record["case"]
    selected = selected_record["case"]
    if (fine["status"] != "completed" or selected["status"] != "completed"
            or fine["basis"] != "fine_nodal_q1"
            or selected["basis"] != "selected_q1_knots_dev"
            or fine["fine_node_ids"] != selected["fine_node_ids"]):
        raise ValueError("saved cases need completed matching physical source nodes")
    count = len(fine["fine_node_ids"]) // 2
    if count != 9 or len(fine["fine_node_ids"]) != 2 * count:
        raise ValueError("this comparison needs the eight-Q4 strip")
    fine_p = np.asarray(fine["fine_pressure_pa"], dtype=float).reshape(2, count)
    selected_p = np.asarray(selected["fine_pressure_pa"], dtype=float).reshape(2, count)
    if not np.all(np.isfinite(fine_p)) or not np.all(np.isfinite(selected_p)):
        raise ValueError("saved pressures must be finite")
    mx = _mass(np.linspace(0.0, 1.0, count), 1.0)
    my = _mass(np.array((0.0, 1.0)), 0.9)
    def norm_sq(field: np.ndarray) -> float:
        return float(np.sum(field * (my @ field @ mx)))
    pressure_difference = selected_p - fine_p
    fine_u = np.asarray(fine["final_displacements_m"], dtype=float)
    selected_u = np.asarray(selected["final_displacements_m"], dtype=float)
    fine_force = float(fine["contact"]["integration_force"][2])
    selected_force = float(selected["contact"]["integration_force"][2])
    report = {
        "method": "same-mechanics eight-Q4/two-S3 coupled discrete comparison",
        "fine_sha256": hashlib.sha256(fine_data).hexdigest(),
        "selected_sha256": hashlib.sha256(selected_data).hexdigest(),
        "full_pressure_relative_l2_over_overlap": float(
            np.sqrt(norm_sq(pressure_difference) / norm_sq(fine_p))),
        "pressure_maximum_node_difference_pa": float(np.max(np.abs(pressure_difference))),
        "displacement_relative_l2": float(np.linalg.norm(selected_u-fine_u)
                                           /np.linalg.norm(fine_u)),
        "fine_normal_force_n": fine_force,
        "selected_normal_force_n": selected_force,
        "normal_force_relative_difference": (selected_force-fine_force)/fine_force,
        "fine_minimum_gap_m": fine["contact"]["minimum_gap"],
        "selected_minimum_gap_m": selected["contact"]["minimum_gap"],
        "fine_pressure_modes": fine["contact"]["pressure_gram_supported_modes"],
        "selected_pressure_modes": selected["contact"]["pressure_gram_supported_modes"],
        "fine_elapsed_seconds": fine["elapsed_seconds"],
        "selected_elapsed_seconds": selected["elapsed_seconds"],
    }
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--basis-only", choices=("fine_nodal_q1",
                                                  "selected_q1_knots_dev"))
    parser.add_argument("--compare-saved", nargs=2, type=Path,
                        metavar=("FINE", "SELECTED"))
    parser.add_argument("--selected-knots", nargs="+", type=int)
    args = parser.parse_args()
    if args.compare_saved is None:
        main(args.output, args.columns, args.basis_only,
             tuple(args.selected_knots) if args.selected_knots is not None else None)
    else:
        compare_saved(*args.compare_saved, args.output)
