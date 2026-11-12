"""Pure physical torque on the existing matched E01 native 8x4 edge rib."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.b3_e4_e01_native_point_refinement_screen import definition
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from anyfesolver.stiffened_shell import create_analysis


TORQUE_NM = 100.
INTERVALS = 8
TRANSVERSE = 4


def solve(sign: int):
    model, tip = definition(INTERVALS, sign, TRANSVERSE)
    owner = create_analysis(definition=model)
    result = owner.solve_linear((NodalAction(
        "beam", tip, (0., 0., 0., TORQUE_NM, 0., 0.)),))
    if result.status != "completed":
        raise RuntimeError(f"native E01 pure torque: {result.status}")
    base = 6*owner._beam_index[tip]
    tip_pose = result.beam_coordinates[base:base+6]
    q = np.r_[result.shell_displacement, result.beam_coordinates]
    internal_work = float(q@(owner._K@q))
    applied_work = float(TORQUE_NM*tip_pose[3])
    support_force = np.zeros(3)
    support_moment = np.zeros(3)
    for i, position in enumerate(owner._shell_xyz):
        row = result.reactions[6*i:6*i+6]
        support_force += row[:3]
        support_moment += row[3:]+np.cross(position, row[:3])
    force_error = float(np.max(np.abs(support_force)))
    moment_error = float(np.max(np.abs(support_moment+
                                         (TORQUE_NM, 0., 0.))))
    work_error = abs(internal_work-applied_work)
    if (result.free_equilibrium_ratio > 1e-8 or
            result.constraint_max > 1e-9 or force_error > 1e-5 or
            moment_error > 1e-5 or
            work_error > 1e-8*max(abs(applied_work), 1e-12)):
        raise ValueError("native torque equilibrium/work check failed")
    return {
        "offset_sign": sign,
        "shell_intervals": INTERVALS,
        "transverse_intervals": TRANSVERSE,
        "tip_translation_m": tip_pose[:3].tolist(),
        "tip_rotation_rad": tip_pose[3:].tolist(),
        "shell_node_displacements": {
            str(node): result.shell_displacement[6*i:6*i+6].tolist()
            for i, node in enumerate(owner._shell_ids)},
        "support_force_error_n": force_error,
        "support_moment_error_nm": moment_error,
        "internal_work_j": internal_work,
        "applied_work_j": applied_work,
        "work_error_j": work_error,
        "free_equilibrium_ratio": result.free_equilibrium_ratio,
        "constraint_max": result.constraint_max,
    }


def main(output: Path):
    if output.exists():
        raise FileExistsError("preserve previous E01 torque evidence")
    cases = [solve(sign) for sign in (-1, 1)]
    record = {
        "classification": "E01 native pure torque; development only",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "load": "+100 N m about beam x at tip",
        "cases": cases,
        "limit": "Torque influence, not E01 point-law acceptance.",
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = main(args.output)
    print(json.dumps({case["offset_sign"]: case["tip_rotation_rad"][0]
                      for case in record["cases"]}))
