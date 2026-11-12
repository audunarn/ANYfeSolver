"""Separate E01 physical reciprocity from sparse factorization roundoff."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.b3_e4_e01_native_point_refinement_screen import definition
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from anyfesolver.stiffened_shell import create_analysis


def main(output: Path):
    if output.exists():
        raise FileExistsError("preserve earlier E01 reciprocity evidence")
    model, tip = definition(8, +1, 4)
    owner = create_analysis(definition=model)
    root = owner._shell_count+6*owner._beam_index[tip]
    systems = []
    for fx, mx in ((1000., 0.), (0., 100.)):
        result = owner.solve_linear((NodalAction(
            "beam", tip, (fx, 0., 0., mx, 0., 0.)),))
        if result.status != "completed":
            raise RuntimeError("completed E01 reciprocity case required")
        q = np.r_[result.shell_displacement, result.beam_coordinates]
        solution = np.r_[q[owner._free], result.multipliers]
        load = np.zeros(owner._K.shape[0])
        load[root] = fx
        load[root+3] = mx
        rhs = np.r_[load[owner._free], np.zeros(owner._C.shape[0])]
        systems.append((solution, rhs))
    def cross_pair(items):
        q = []
        for solution, _rhs in items:
            row = np.zeros(owner._K.shape[0])
            row[owner._free] = solution[:len(owner._free)]
            q.append(row)
        a = q[0][root+3]/1000.
        b = q[1][root]/100.
        return {"force_to_rotation_rad_per_n": float(a),
                "torque_to_translation_m_per_nm": float(b),
                "torque_to_rotation_rad_per_nm": float(q[1][root+3]/100.),
                "force_to_translation_m_per_n": float(q[0][root]/1000.),
                "relative_difference": float(abs(a-b)/max(abs(a), abs(b)))}
    initial = cross_pair(systems)
    skew = (owner._A-owner._A.T).tocoo()
    max_skew = float(np.max(np.abs(skew.data), initial=0.))
    max_entry = float(np.max(np.abs(owner._A.data), initial=0.))
    before_residual = max(float(np.max(np.abs(rhs-owner._A@solution)))
                          for solution, rhs in systems)
    corrected = [(solution+owner._linear_factor.solve(
        rhs-owner._A@solution), rhs) for solution, rhs in systems]
    after_residual = max(float(np.max(np.abs(rhs-owner._A@solution)))
                         for solution, rhs in corrected)
    refined = cross_pair(corrected)
    record = {
        "classification": "E01 torque reciprocity; development only",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "shell_intervals": 8,
        "transverse_intervals": 4,
        "offset_sign": +1,
        "linear_factorizations": owner._linear_factorizations,
        "automatic_linear_refinements": owner._linear_refinements,
        "maximum_relative_assembled_matrix_skew": max_skew/max_entry,
        "maximum_kkt_residual_before_manual_refinement": before_residual,
        "maximum_kkt_residual_after_manual_refinement": after_residual,
        "as_returned": initial,
        "after_one_manual_refinement": refined,
        "limit": ("A bounded numerical conditioning check. Manual refinement "
                  "is not a product policy change or E01 acceptance."),
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = main(args.output)
    print(json.dumps({"as_returned": record["as_returned"]["relative_difference"],
                      "after_refinement": record[
                          "after_one_manual_refinement"]["relative_difference"],
                      "matrix_skew": record[
                          "maximum_relative_assembled_matrix_skew"]}))
