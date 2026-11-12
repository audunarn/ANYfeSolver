"""Capture rejected local Q4/S3 geometry from one bounded D1 development run.

This instruments only the current process. The captured coordinates are inputs
to an independent high-precision area oracle, not acceptance evidence alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anyfesolver
from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact import s3_patch_augmented as patch
from anyfesolver.general_contact.q4_s3_overlap_derivatives import UnsupportedQ4S3Branch

from scripts.general_contact_d1_interior_press_dev import build_interior_press


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    captures = []
    last = {}
    original_derivative = patch.q4_s3_overlap_derivatives
    original_gate = patch._q4_s3_pair_value_rtol

    def derivative(reference, source, target, **kwargs):
        result = original_derivative(reference, source, target, **kwargs)
        last.clear()
        last.update({
            "reference_q4_m": reference.tolist(),
            "source_q4_m": source.tolist(),
            "target_s3_m": target.tolist(),
            "jet2_overlap_area_m2": float(result.overlap_area),
            "jet2_area_resolution_m2": float(result.overlap_area_resolution),
            "jet2_reference_support_m2": result.reference_support.tolist(),
        })
        return result

    def gate(float_area, jet2_area, resolution):
        try:
            return original_gate(float_area, jet2_area, resolution)
        except UnsupportedQ4S3Branch as exc:
            if len(captures) < 5:
                captures.append({
                    **last,
                    "float_overlap_area_m2": float(float_area),
                    "jet2_overlap_area_m2": float(jet2_area),
                    "jet2_area_resolution_m2": float(resolution),
                    "rejection": str(exc),
                })
                output.write_text(json.dumps({
                    "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
                    "source": "corrected_left_edge_first_press_development",
                    "captures": captures,
                }, indent=2) + "\n", encoding="utf-8")
            raise

    patch.q4_s3_overlap_derivatives = derivative
    patch._q4_s3_pair_value_rtol = gate
    try:
        model, definition, settings, program = build_interior_press(
            source_support="perimeter", source_rows=4,
            load_shape="interior_distributed", force_n=0.04,
            reference_slope=1.0e-4, four_phase=True,
            slide_body="target",
        )
        result = solve_static_nonlinear(
            model, contact=definition, contact_settings=settings,
            contact_path_program=program, max_iterations=18,
            max_load_factor=0.125, emit_restart_checkpoint=False,
        )
    finally:
        patch.q4_s3_overlap_derivatives = original_derivative
        patch._q4_s3_pair_value_rtol = original_gate
    payload = {
        "anyfesolver_module": str(Path(anyfesolver.__file__).resolve()),
        "source": "corrected_left_edge_first_press_development",
        "status": result.status,
        "failure_reason": result.failure_reason,
        "accepted_steps": len(result.steps),
        "captures": captures,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.status,
                      "failure_reason": result.failure_reason,
                      "capture_count": len(captures)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
