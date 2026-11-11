"""Capture a rejected Q4/S3 value comparison without changing its decision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact import s3_patch_augmented as patch
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import main


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "reports/general_contact_d1_distributed_pair28_capture_20260927.json"
REPORT = ROOT / "reports/general_contact_d1_mixed_16q4_distributed_capture_press_20260927.json"
if CAPTURE.exists() or REPORT.exists():
    raise FileExistsError("development capture or report already exists")

original_derivative = patch.q4_s3_overlap_derivatives
original_compare = patch._q4_s3_pair_values_agree
latest: dict[str, np.ndarray] = {}


def record_derivative(reference, source, target, **kwargs):
    latest.clear()
    latest.update(source_reference=np.asarray(reference).copy(),
                  local_source=np.asarray(source).copy(),
                  local_target=np.asarray(target).copy())
    return original_derivative(reference, source, target, **kwargs)


def record_comparison(local_values, derivative_values, **kwargs):
    agrees = original_compare(local_values, derivative_values, **kwargs)
    if (not agrees and kwargs.get("normwise")
            and abs(kwargs["local_area"]-0.00787259623212) < 1e-8
            and not CAPTURE.exists()):
        record = {key: value.tolist() for key, value in latest.items()}
        record.update({
            "local_values_m3": np.asarray(local_values).tolist(),
            "derivative_values_m3": np.asarray(derivative_values).tolist(),
            "comparison": kwargs,
        })
        CAPTURE.write_text(json.dumps(record, indent=2)+"\n",
                           encoding="utf-8")
    return agrees


patch.q4_s3_overlap_derivatives = record_derivative
patch._q4_s3_pair_values_agree = record_comparison
main(16, REPORT, pressure_cap_pa=250.0, four_phase=False,
     steps_per_segment=2, schur_step_cap_pa=200.0,
     load_profile="distributed", total_force_n=0.3,
     schur_relinearizations=0, target_right_x_m=1.3,
     schur_maximum_attempts=2)
if not CAPTURE.exists():
    raise RuntimeError("the rejected pair-28 value comparison was not captured")
