"""Recover the native E01 shell covector under +100 N m physical torque."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.b3_e4_e01_native_point_refinement_screen import definition
from anyfesolver import _stiffened_shell_ge_experimental as ge
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from anyfesolver.stiffened_shell import create_analysis


def main(output: Path):
    if output.exists():
        raise FileExistsError("preserve earlier E01 action evidence")
    model, tip = definition(8, +1, 4)
    owner = create_analysis(definition=model)
    result = owner.solve_linear((NodalAction(
        "beam", tip, (0., 0., 0., 100., 0., 0.)),))
    if result.status != "completed":
        raise RuntimeError("completed native torque solve required")
    shell_model, ids, _xyz, index, pieces = ge._shell_reference(model.shell)
    count = 6*len(ids)
    reference, tangent, reference_error = ge._assemble_shell(
        shell_model, pieces, count, np.ones(count))
    internal = np.asarray(tangent@result.shell_displacement+reference)
    constrained = np.asarray(owner._C.T@result.multipliers)[:count]
    fixed = np.array([6*index[1+row*9]+d
                      for row in range(5) for d in range(6)])
    free = np.setdiff1d(np.arange(count), fixed)
    free_error = float(np.max(np.abs((internal+constrained)[free])))
    if (reference_error > 1e-8 or free_error > 1e-5 or
            not np.isfinite(internal).all()):
        raise ValueError("native torque shell action invalid")
    internal[fixed] = 0.
    record = {
        "classification": "E01 same-action shell covector; development only",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "ge_adapter_sha256": sha256(Path(ge.__file__).read_bytes()).hexdigest(),
        "load_source": "native coupled GE +100 N m beam-tip physical x",
        "shell_intervals": 8,
        "transverse_intervals": 4,
        "loads": {str(node): internal[6*i:6*i+6].tolist()
                  for i, node in enumerate(ids)},
        "native_coupled_shell_displacements": {
            str(node): result.shell_displacement[6*i:6*i+6].tolist()
            for i, node in enumerate(ids)},
        "reference_error": reference_error,
        "coupled_shell_free_error": free_error,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True,
                                 allow_nan=False)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = main(args.output)
    print(json.dumps({"free_error": record["coupled_shell_free_error"]}))
