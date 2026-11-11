"""Bounded F05 S3 branch-path check around the existing yielded pullback trial.

Decision: do the sampled derivative endpoints hide an active-set or hardening
segment change?  This finite path screen cannot prove continuous invariance.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from probe_b3_ge_e4_pl_g0_f05_s3_pose_pullback_dev_v1 import _fixture, _material
from anyfesolver.e4_pl_s3_v2d_state import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/probe_b3_ge_e4_pl_g0_f05_s3_branch_path_dev_v1.py",
    "scripts/probe_b3_ge_e4_pl_g0_f05_s3_pose_pullback_dev_v1.py",
    "src/anyfesolver/e4_pl_s3_v2d_element.py",
    "src/anyfesolver/e4_pl_s3_v2d_state.py",
    "src/anyfesolver/plasticity.py",
    "src/anyfesolver/material_curves.py",
    "src/anyfesolver/_ge_beam3_variational_shell.py",
)
HALF_WIDTH = 2e-6
STEPS = (2e-6, 2e-7)
GRID = np.unique(np.r_[np.linspace(-HALF_WIDTH, HALF_WIDTH, 33),
                       np.asarray(STEPS), -np.asarray(STEPS)])


def _segment(alpha: np.ndarray, first: float, second: float) -> np.ndarray:
    return np.where(alpha <= first, 0, np.where(alpha <= second, 1, 2))


def main() -> None:
    model, material, element, predecessor, q, _joint, _frames = _fixture()
    before = canonical_json_bytes(predecessor)
    parent_alpha = np.asarray(predecessor["alpha"], dtype=float)
    central_force, central_tangent, central_state = _material(
        q, model, material, element, predecessor, True)
    first = float(material.hardening_curve.eps_p_y1)
    second = float(material.hardening_curve.eps_p_y2)
    central_alpha = np.asarray(central_state["alpha"], dtype=float)
    central_active = central_alpha > parent_alpha
    central_segment = _segment(central_alpha, first, second)
    directions = []
    for phase, scale in ((.3, 1.), (.7, .73)):
        direction = np.zeros(30)
        direction[:24] = np.sin(np.arange(24)*scale + phase)
        direction[:24] /= np.linalg.norm(direction[:24])
        direction[24:] = 1e5*np.cos(np.arange(6)*.53 + phase)
        alpha_samples = []
        active_changes = 0
        segment_changes = 0
        forces = {}
        for offset in GRID:
            force, _tangent, state = _material(
                q + offset*direction, model, material, element, predecessor, False)
            alpha = np.asarray(state["alpha"], dtype=float)
            alpha_samples.append(alpha)
            active_changes += int(np.count_nonzero((alpha > parent_alpha) != central_active))
            segment_changes += int(np.count_nonzero(
                _segment(alpha, first, second) != central_segment))
            if any(offset == sign*h for h in STEPS for sign in (-1., 1.)):
                forces[float(offset)] = force
        alphas = np.stack(alpha_samples)
        measured = []
        for h in STEPS:
            difference = (forces[float(h)] - forces[float(-h)])/(2*h)
            predicted = central_tangent @ direction
            measured.append({"step": h, "relative_derivative_error": float(
                np.linalg.norm(predicted-difference)/max(1., np.linalg.norm(difference)))})
        active_alpha = alphas[:, central_active]
        directions.append({
            "phase": phase,
            "sample_count": len(GRID),
            "active_classification_changes": active_changes,
            "hardening_segment_changes": segment_changes,
            "minimum_active_alpha_increment": float(np.min(
                active_alpha - parent_alpha[central_active])),
            "minimum_active_distance_to_hardening_break": float(np.min(
                np.minimum(np.abs(active_alpha-first), np.abs(active_alpha-second)))),
            "derivative": measured,
        })
    immutable = canonical_json_bytes(predecessor) == before
    passed = (bool(np.any(central_active)) and immutable and
              all(row["active_classification_changes"] == 0 and
                  row["hardening_segment_changes"] == 0 and
                  row["minimum_active_alpha_increment"] > 0 and
                  all(item["relative_derivative_error"] <= 1e-5
                      for item in row["derivative"])
                  for row in directions))
    result = {
        "case": "B3_E4_G0_F05_S3_BRANCH_PATH_DEV_V1",
        "command": "python -B scripts/probe_b3_ge_e4_pl_g0_f05_s3_branch_path_dev_v1.py",
        "status": "passed_bounded_development_screen" if passed else "failed_bounded_development_screen",
        "question": "Does the existing S3 derivative plateau hide a sampled active or hardening-segment change?",
        "decision": "A pass strengthens the local smooth-branch interpretation; a failure returns to S3 local tangent diagnosis.",
        "source_sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                          for path in SOURCE_PATHS},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "numba_disable_jit": True,
                        "PYTHONPATH": "src;scripts;.", "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "fixture": {"accepted_alpha_max": float(np.max(parent_alpha)),
                    "trial_alpha_max": float(np.max(central_alpha)),
                    "active_station_count": int(np.count_nonzero(central_active)),
                    "hardening_breaks": [first, second],
                    "grid_half_width": HALF_WIDTH,
                    "grid_points_per_direction": len(GRID),
                    "predecessor_byte_identical": immutable,
                    "central_material_force_norm": float(np.linalg.norm(central_force))},
        "directions": directions,
        "boundary": "Finite path samples only; no continuous branch proof, mixed-owner integration, production adapter, F05 or G0B acceptance.",
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
