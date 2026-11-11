"""Native E4-PL shell-only response for the E01 rotation discriminator.

Uses the already compiled Q4 local adapter and a bounded sparse solve. It is
a development diagnostic, not a replacement public analysis route.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from anyfesolver import _stiffened_shell_ge_experimental as ge
from anyfesolver.stiffened_shell import create_analysis
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from b3_e4_e01_native_point_refinement_screen import definition


INTERVALS = 8
LOAD_N = 1000.


def _cross(a, b):
    return np.cross(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def solve(axis: int | None, loads: dict[int, tuple[float, ...]] | None = None,
          transverse_intervals: int = 1):
    panel = definition(INTERVALS, 1, transverse_intervals)[0].shell
    model, ids, xyz, index, pieces = ge._shell_reference(panel)
    count = 6 * len(ids)
    reference, tangent, reference_error = ge._assemble_shell(
        model, pieces, count, np.ones(count))
    fixed = np.array([6 * index[node] + d
                      for row in range(transverse_intervals + 1)
                      for node in (1 + row * (INTERVALS + 1),)
                      for d in range(6)])
    free = np.setdiff1d(np.arange(count), fixed)
    applied = np.zeros(count)
    if loads is None:
        assert axis is not None
        applied[6 * index[INTERVALS + 1] + axis] = LOAD_N
    else:
        for node, row in loads.items():
            if node not in index or len(row) != 6 or not np.isfinite(row).all():
                raise ValueError("finite known six-coordinate shell loads required")
            applied[6 * index[node]:6 * index[node] + 6] = row
        if np.any(applied[fixed]):
            raise ValueError("same-action input loads a fixed shell coordinate")
    displacement = np.zeros(count)
    displacement[free] = spsolve(tangent[free][:, free],
                                 (applied - reference)[free])
    if not np.isfinite(displacement).all():
        raise ValueError("finite native shell response required")
    residual = np.asarray(tangent @ displacement + reference - applied)
    free_error = float(np.max(np.abs(residual[free])))
    reactions = residual.reshape(-1, 6)
    supports = {node: tuple(float(v) for v in reactions[index[node]])
                for node in (1 + row * (INTERVALS + 1)
                             for row in range(transverse_intervals + 1))}
    force = sum((np.asarray(row[:3]) for row in supports.values()),
                np.zeros(3))
    moment = sum((np.asarray(row[3:]) + _cross(xyz[index[node]], row[:3])
                  for node, row in supports.items()), np.zeros(3))
    applied_rows = applied.reshape(-1, 6)
    applied_force = applied_rows[:, :3].sum(axis=0)
    applied_moment = sum((row[3:] + _cross(xyz[i], row[:3])
                          for i, row in enumerate(applied_rows)), np.zeros(3))
    force_error = float(np.max(np.abs(force + applied_force)))
    moment_error = float(np.max(np.abs(moment + applied_moment)))
    work_error = abs(float(displacement @ (tangent @ displacement + reference)
                           - displacement @ applied))
    applied_work = float(displacement @ applied)
    if (reference_error > 1e-8 or free_error > 1e-5
            or force_error > 1e-5 or moment_error > 1e-5
            or work_error > 1e-8 * abs(applied_work)):
        raise ValueError(f"native shell balance/work failed: "
                         f"{reference_error=:.6g}, {free_error=:.6g}, "
                         f"{force_error=:.6g}, {moment_error=:.6g}, "
                         f"{work_error=:.6g}")
    return {"load_axis": axis,
            "node_displacements": {
                node: tuple(float(v) for v in displacement[6 * i:6 * i + 6])
                for i, node in enumerate(ids)},
            "support_reactions": supports,
            "reference_error": reference_error,
            "free_residual_n": free_error,
            "force_error_n": force_error,
            "moment_error_nm": moment_error,
            "work_error_j": work_error,
            "applied_work_j": applied_work}


def main(output: Path, load_file: Path | None = None,
         transverse_intervals: int = 1):
    if output.exists():
        raise FileExistsError("preserve earlier shell-only evidence")
    record = {"classification": "E01 native shell-only discriminator; not acceptance",
              "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "native_panel_source_sha256": sha256(Path(
                  __file__).with_name(
                      "b3_e4_e01_native_point_refinement_screen.py").read_bytes()).hexdigest(),
              "ge_adapter_sha256": sha256(Path(ge.__file__).read_bytes()).hexdigest(),
              "shell_intervals": INTERVALS,
              "transverse_intervals": transverse_intervals,
              "cases": [], "failure": None}
    loads = None
    if load_file is not None:
        raw = load_file.read_bytes()
        saved = json.loads(raw)
        if (saved["shell_intervals"] != INTERVALS
                or saved.get("transverse_intervals", 1) != transverse_intervals
                or saved["classification"] !=
                "E01 same-action shell covector; development only"):
            raise ValueError("incompatible same-action input")
        loads = {int(node): tuple(float(v) for v in row)
                 for node, row in saved["loads"].items()}
        record["load_sha256"] = sha256(raw).hexdigest()
    try:
        for axis in ((None,) if loads is not None else (1, 2)):
            record["cases"].append(solve(axis, loads, transverse_intervals))
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
    output.write_text(json.dumps(record, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return record


def coupled_shell_action(output: Path, transverse_intervals: int = 1):
    """Save the native coupled solution's free shell covector for one replay."""
    if output.exists():
        raise FileExistsError("preserve earlier same-action evidence")
    definition_, tip = definition(INTERVALS, 1, transverse_intervals)
    owner = create_analysis(definition=definition_)
    result = owner.solve_linear((NodalAction(
        "beam", tip, (1000., 0., 0., 0., 0., 0.)),))
    if result.status != "completed":
        raise RuntimeError(f"native coupled solve: {result.status}")
    model, ids, _xyz, index, pieces = ge._shell_reference(definition_.shell)
    count = 6 * len(ids)
    reference, tangent, reference_error = ge._assemble_shell(
        model, pieces, count, np.ones(count))
    internal = np.asarray(tangent @ result.shell_displacement + reference)
    constrained = np.asarray(owner._C.T @ result.multipliers)[:count]
    fixed = np.array([6 * index[node] + d
                      for row in range(transverse_intervals + 1)
                      for node in (1 + row * (INTERVALS + 1),)
                      for d in range(6)])
    free = np.setdiff1d(np.arange(count), fixed)
    coupled_shell_error = float(np.max(np.abs(
        (internal + constrained)[free])))
    if (reference_error > 1e-8 or coupled_shell_error > 1e-5
            or not np.isfinite(internal).all()):
        raise ValueError(f"native shell action invalid: "
                         f"{reference_error=:.6g}, {coupled_shell_error=:.6g}")
    internal[fixed] = 0.
    loads = {str(node): [float(v) for v in internal[6 * i:6 * i + 6]]
             for i, node in enumerate(ids)}
    record = {
        "classification": "E01 same-action shell covector; development only",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "ge_adapter_sha256": sha256(Path(ge.__file__).read_bytes()).hexdigest(),
        "load_source": "native coupled GE +1000 N beam-tip x, eight shell intervals",
        "shell_intervals": INTERVALS,
        "transverse_intervals": transverse_intervals,
        "loads": loads,
        "native_coupled_shell_displacements": {
            str(node): [float(v) for v in result.shell_displacement[6 * i:6 * i + 6]]
            for i, node in enumerate(ids)},
        "reference_error": reference_error,
        "coupled_shell_free_error": coupled_shell_error,
    }
    output.write_text(json.dumps(record, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--coupled-action-output", type=Path)
    parser.add_argument("--load-file", type=Path)
    parser.add_argument("--transverse-intervals", type=int, default=1)
    args = parser.parse_args()
    if args.transverse_intervals not in (1, 4):
        parser.error("bounded shell screen admits one or four width cells")
    if args.coupled_action_output is not None:
        result = coupled_shell_action(args.coupled_action_output,
                                      args.transverse_intervals)
        print(json.dumps({"coupled_shell_free_error":
                          result["coupled_shell_free_error"],
                          "loaded_nodes": len(result["loads"])}))
    else:
        result = main(args.output, args.load_file,
                      args.transverse_intervals)
        print(json.dumps({"failure": result["failure"],
                          "completed_cases": len(result["cases"])}))
        if result["failure"]:
            raise SystemExit(1)
