"""Bounded supported native Fy influence on the saved 1:1 and 2:1 C1 grids.

This checks loaded transfer under the existing selective-fz experimental rows.
It does not select that multiplier space or consume formal F03 authority.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

import anyfesolver
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_weak_line_candidate import LineMultiplierSpace
from scripts.b3_e4_f03_ccx_solid_bond_screen import digest
from scripts.b3_e4_f03_material_line_loaded import _scales, _solve
from scripts.b3_e4_f03_material_line_position import MaterialLinePosition
from scripts.b3_e4_f03_material_line_space_screen import _rows
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


LOAD_N = 100.
MAX_SECONDS_PER_CASE = 120.


def _wrench(wrenches: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.r_[wrenches[:, :3].sum(axis=0),
                 (wrenches[:, 3:]+np.cross(positions, wrenches[:, :3])).sum(axis=0)]


def one_case(xs, ys, shell_k, beam_halves: int) -> dict:
    started = monotonic()
    beam = build(beam_halves, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    line = MaterialLinePosition(xs, ys, beam)
    p0, _ = _rows(line, LineMultiplierSpace(
        "piecewise_constant", tuple(np.linspace(0., 1., 9))))
    fz, _ = _rows(line, LineMultiplierSpace(
        "piecewise_constant", tuple(np.linspace(0., 1., 5))))
    rows = np.vstack([p0[axis::6] if axis != 2 else fz[2::6]
                      for axis in range(6)])
    shell_count = 6*line.shell_count
    primal = sparse.block_diag((sparse.csr_matrix(shell_k), beam.tangent),
                               format="csr")
    residual = np.r_[np.zeros(shell_count), beam.residual]
    if rows.shape != (44, primal.shape[0]):
        raise ArithmeticError("saved selective-fz native row layout changed")
    shell_fixed = {6*(j*len(xs)+i)+axis for j in range(len(ys))
                   for i, x in enumerate(xs) if abs(x+.5) < 1e-12
                   for axis in range(6)}
    beam_fixed = {shell_count+int(index) for index in beam.context.fixed}
    free = np.asarray(sorted(set(range(primal.shape[0]))-shell_fixed-beam_fixed))
    scales, energy_scale = _scales(shell_count, beam, free, row_count=44)
    terminal = shell_count+6*beam.n_half_cells
    load = np.zeros(primal.shape[0])
    load[terminal+1] = LOAD_N
    state, equation, diagnostic = _solve(
        primal, residual, rows, free, scales, energy_scale, load)
    reduced = rows[:, free].T
    right = (load-residual-primal@state)[free]
    dual, _, rank, _ = np.linalg.lstsq(reduced, right, rcond=1e-10)
    if rank != 44 or np.linalg.norm(reduced@dual-right) > 1e-8*LOAD_N:
        raise ArithmeticError("native Fy multiplier recovery failed")
    interface = -(rows.T@dual)
    station_count = len(line.stations)
    position = line.reference_jacobian().reshape(station_count, 3, -1)
    orientation = line.reference_orientation_jacobian().reshape(station_count, 3, -1)
    continuous_gap = np.c_[position@state, orientation@state]
    bonded_sites = np.array([site.x >= .9-1e-12 for site in line.stations])
    if not bonded_sites.any():
        raise ArithmeticError("bonded C1 trace stations missing")
    shell_positions = line.reference_nodes
    beam_positions = beam.context.initial.mechanical.positions
    shell_support = equation[:shell_count].reshape(-1, 6)
    beam_support = equation[shell_count:shell_count+beam.context.nodal_count].reshape(-1, 6)
    shell_support_wrench = _wrench(shell_support, shell_positions)
    beam_support_wrench = _wrench(beam_support, beam_positions)
    applied_wrench = np.array((0., LOAD_N, 0., -.0675*LOAD_N, 0., LOAD_N))
    balance = shell_support_wrench+beam_support_wrench+applied_wrench
    if np.max(np.abs(balance)) > 1e-5:
        raise ArithmeticError("native supported Fy wrench does not balance")
    if monotonic()-started > MAX_SECONDS_PER_CASE:
        raise TimeoutError("bounded native Fy case exceeded 120 seconds")
    return {
        "beam_halves": beam_halves,
        "bonded_halves": int(sum(np.diff(beam.abscissae)[
            beam.abscissae[:-1] >= .9-1e-12] > 0.)),
        "matrix": diagnostic,
        "applied_wrench_N_Nm": applied_wrench.tolist(),
        "shell_support_wrench_N_Nm": shell_support_wrench.tolist(),
        "beam_root_support_wrench_N_Nm": beam_support_wrench.tolist(),
        "wrench_imbalance_N_Nm": balance.tolist(),
        "shell_interface_wrench_N_Nm": _wrench(
            interface[:shell_count].reshape(-1, 6), shell_positions).tolist(),
        "tip_pose_increment": state[terminal:terminal+6].tolist(),
        "applied_load_work_J": float(load@state),
        "interface_virtual_work_J": float(interface@state),
        "max_weak_gap": float(np.max(np.abs(rows@state))),
        "max_bonded_continuous_gap_by_channel": np.max(
            np.abs(continuous_gap[bonded_sites]), axis=0).tolist(),
        "bonded_station_x_m": [float(site.x) for site, keep in
                               zip(line.stations, bonded_sites) if keep],
        "bonded_continuous_gap": continuous_gap[bonded_sites].tolist(),
        "elapsed_s": monotonic()-started,
    }


def evaluate() -> dict:
    source = Path(__file__).resolve().parents[1]
    if Path(anyfesolver.__file__).resolve().parent != (source / "src" / "anyfesolver").resolve():
        raise RuntimeError("bounded Fy runner imported another ANYfeSolver checkout")
    xs, ys = local_grid(8, transverse_cells=4)
    shell_k, _ = raw_e4_tangent(xs, ys, thickness=.035)
    return {
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "classification": "loaded native selective-fz line, not physical law or multiplier qualification",
        "source_git": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "source_sha256": digest(Path(__file__)),
        "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
        "physical_model": "x8/y4 35 mm Q4 plate, square generalized B3 with C1 2 um reference, left shell and B3 root fixed, x=.9-1 m line",
        "load": "+100 N Fy at native beam tip reference axis",
        "cases": [one_case(xs, ys, shell_k, halves) for halves in (18, 14)],
        "effective_threadpools": threadpool_info(),
        "env": {key: os.environ.get(key) for key in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_DISABLE_JIT")},
        "limitations": ["single x8 shell grid", "selective-fz row space is experimental", "native and solid transverse kinematics and bond enforcement differ", "not an F03 or G0B qualification"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"cases": [(case["beam_halves"], case["tip_pose_increment"][1],
                                 case["elapsed_s"]) for case in result["cases"]]}))


if __name__ == "__main__":
    main()
