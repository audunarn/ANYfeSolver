"""Compare a captured smooth wedge's Jet2 work derivatives with float geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.q4_s3_overlap_derivatives import (
    q4_s3_overlap_derivatives,
)
from anyfesolver.general_contact.q4_source_overlap import q4_source_overlap_integrals


INTEGRATION = dict(area_tolerance=1.0e-9,
                   weighted_gap_tolerance=1.0e-10,
                   maximum_triangles=300)
DERIVATIVES = dict(allow_warped_source=True,
                   support_gradient_tolerance=1.0e-5,
                   support_hessian_tolerance=1.0e-5,
                   weighted_gap_gradient_tolerance=1.0e-5,
                   weighted_gap_hessian_tolerance=1.0e-5)


def main(capture_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    with np.load(capture_path) as capture:
        reference = capture["source_reference"].copy()
        source = capture["local_source"].copy()
        target = capture["local_target"].copy()
        pressure = capture["pressure"][capture["source_nodes"]].copy()
        pair_index = int(capture["pair_index"])

    direction = np.zeros((7, 3))
    direction[:4, 0] = 1.0
    direction[:4, 2] = 0.2
    flat = direction.ravel()

    def float_value(step: float):
        trial_source = source + step*direction[:4]
        trial_target = target + step*direction[4:]
        result = q4_source_overlap_integrals(
            reference, trial_source, trial_target, **INTEGRATION,
        )
        return result, float(pressure @ result.weighted_gap)

    base_float, base_work = float_value(0.0)
    jet = q4_s3_overlap_derivatives(
        reference, source, target, **INTEGRATION, **DERIVATIVES,
    )
    jet_gradient = float(pressure @ (jet.weighted_gap_gradient @ flat))
    jet_hessian = float(sum(
        pressure[i] * flat @ jet.weighted_gap_hessian[i] @ flat
        for i in range(4)
    ))
    rows = []
    for step in (4.0e-10, 2.0e-10, 1.0e-10, 5.0e-11, 2.5e-11):
        plus_float, plus_work = float_value(step)
        minus_float, minus_work = float_value(-step)
        plus_jet = q4_s3_overlap_derivatives(
            reference, source+step*direction[:4],
            target+step*direction[4:], **INTEGRATION, **DERIVATIVES,
        )
        minus_jet = q4_s3_overlap_derivatives(
            reference, source-step*direction[:4],
            target-step*direction[4:], **INTEGRATION, **DERIVATIVES,
        )
        quotient_gradient = (plus_work-minus_work)/(2*step)
        quotient_hessian = (plus_work-2*base_work+minus_work)/(step*step)
        rows.append({
            "step_m": step,
            "same_jet_branch": bool(
                plus_jet.branch_signature == jet.branch_signature
                == minus_jet.branch_signature
            ),
            "positive_float_areas": bool(
                plus_float.overlap_area > 0 and minus_float.overlap_area > 0
            ),
            "float_area_plus_m2": float(plus_float.overlap_area),
            "float_area_minus_m2": float(minus_float.overlap_area),
            "quotient_gradient_j_per_m": quotient_gradient,
            "jet_gradient_j_per_m": jet_gradient,
            "gradient_relative_error": abs(quotient_gradient-jet_gradient)
                / max(abs(jet_gradient), 1.0e-30),
            "quotient_hessian_j_per_m2": quotient_hessian,
            "jet_hessian_j_per_m2": jet_hessian,
            "hessian_relative_error": abs(quotient_hessian-jet_hessian)
                / max(abs(jet_hessian), 1.0e-30),
        })
    record = {
        "candidate": "captured_q4_s3_physical_edge_wedge_derivative_dev",
        "capture": str(capture_path),
        "pair_index": pair_index,
        "base_float_area_m2": float(base_float.overlap_area),
        "base_jet_area_m2": float(jet.overlap_area),
        "base_work_j": base_work,
        "pressure_pa": pressure.tolist(),
        "direction": flat.tolist(),
        "rows": rows,
    }
    output_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"base_float_area_m2": record["base_float_area_m2"],
                      "base_jet_area_m2": record["base_jet_area_m2"],
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.capture, args.output)
