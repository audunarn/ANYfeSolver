"""Private bounded 2:1 least-action audit; no F03 qualification claim."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

import anyfesolver
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_info, threadpool_limits

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_weak_line_candidate import LineMultiplierSpace
from scripts.b3_e4_f03_c1_native_mixed_metric import _beam_metric, _line_metric, MAX_ROWS
from scripts.b3_e4_f03_material_line_position import MaterialLinePosition
from scripts.b3_e4_f03_material_line_space_screen import _rows
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate() -> dict:
    started = monotonic()
    deadline = started + 120.
    source = Path(__file__).resolve().parents[1]
    if Path(anyfesolver.__file__).resolve().parent != (source / "src" / "anyfesolver").resolve():
        raise RuntimeError("wrong ANYfeSolver checkout")
    xs, ys = local_grid(8, transverse_cells=4)
    shell, _ = raw_e4_tangent(xs, ys, thickness=.035)
    beam = build(14, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    bonded = int(np.sum(np.diff(beam.abscissae)[beam.abscissae[:-1] >= .9-1e-12] > 0.))
    if bonded != 4 or beam.tangent is None:
        raise ArithmeticError("x8/B3-14 bonded layout changed")
    ns = len(shell)
    sf = np.array([i for i in range(ns) if (i//6) % len(xs) != 0])
    drill = sf[sf % 6 == 5]
    sa = sf[sf % 6 != 5]
    bf = np.asarray(beam.context.free, dtype=int)
    dd = shell[np.ix_(drill, drill)]
    dp = shell[np.ix_(drill, sa)]
    quotient = shell[np.ix_(sa, sa)] - shell[np.ix_(sa, drill)] @ linalg.solve(dd, dp, assume_a="pos")
    linalg.cholesky(quotient, lower=True)
    beam_metric, _, beam_layout = _beam_metric(beam)
    beam_metric = beam_metric[np.ix_(bf, bf)]
    primal_metric = linalg.block_diag(quotient, beam_metric)
    line = MaterialLinePosition(xs, ys, beam)
    p0, p0_mass = _rows(line, LineMultiplierSpace("piecewise_constant", tuple(np.linspace(0., 1., 9))))
    fz, fz_mass = _rows(line, LineMultiplierSpace("piecewise_constant", tuple(np.linspace(0., 1., 5))))
    p0_metric, p0_info = _line_metric("piecewise_constant", 8, p0_mass)
    fz_metric, _ = _line_metric("piecewise_constant", 4, fz_mass)
    channels = [p0[axis::6] if axis != 2 else fz[2::6] for axis in range(6)]
    def block(metric, axis, count):
        ids = 6*np.arange(count)+axis
        return metric[np.ix_(ids, ids)]
    dual_metric = linalg.block_diag(*[
        block(fz_metric, axis, 4) if axis == 2 else block(p0_metric, axis, 8)
        for axis in range(6)])
    rows = np.vstack(channels)
    if np.max(np.abs(rows[:, drill])) > 1e-12:
        raise ArithmeticError("physical line action couples eliminated drill")
    free = np.r_[sa, ns+bf]
    C = rows[:, free]
    total_rows = len(primal_metric)+len(C)
    if total_rows > MAX_ROWS or monotonic() > deadline:
        raise TimeoutError("bounded resource limit")
    Rp = linalg.cholesky(primal_metric, lower=True)
    Rd = linalg.cholesky(dual_metric, lower=True)
    normalized = linalg.solve_triangular(Rd, C, lower=True)
    normalized = linalg.solve_triangular(Rp, normalized.T, lower=True).T
    U, singular, Vh = linalg.svd(normalized, full_matrices=False)
    lam = linalg.solve_triangular(Rd.T, U[:, -1], lower=False)
    action = C.T @ lam
    response = linalg.cho_solve((Rp, True), action)
    shell_action, beam_action = action[:len(sa)], action[len(sa):]
    shell_response, beam_response = response[:len(sa)], response[len(sa):]
    shell_work = float(shell_action @ shell_response)
    beam_work = float(beam_action @ beam_response)
    dual_norm = float(lam @ dual_metric @ lam)
    action_work = float(action @ response)
    pairing = float(lam @ (C @ response))
    counts = [8, 8, 4, 8, 8, 8]
    offsets = np.r_[0, np.cumsum(counts)]
    names = ("fx", "fy", "fz", "mx", "my", "mz")
    channel_report = {}
    for axis, name in enumerate(names):
        ids = slice(offsets[axis], offsets[axis+1])
        component = np.zeros_like(lam)
        component[ids] = lam[ids]
        component_action = C.T @ component
        cs, cb = component_action[:len(sa)], component_action[len(sa):]
        channel_report[name] = {
            "dual_density": lam[ids].tolist(),
            "dual_norm_sq": float(component @ dual_metric @ component),
            "own_shell_work_sq": float(cs @ linalg.cho_solve((linalg.cholesky(quotient, lower=True), True), cs)),
            "own_beam_work_sq": float(cb @ linalg.cho_solve((linalg.cholesky(beam_metric, lower=True), True), cb)),
            "cross_shell_work_with_full": float(cs @ shell_response),
            "cross_beam_work_with_full": float(cb @ beam_response),
        }
    full_response = np.zeros(ns + beam.context.count)
    full_response[sa] = shell_response
    full_response[drill] = -linalg.solve(dd, dp @ shell_response, assume_a="pos")
    full_response[ns+bf] = beam_response
    station_count = len(line.stations)
    jac_position = line.reference_jacobian().reshape(station_count, 3, -1)
    jac_rotation = line.reference_orientation_jacobian().reshape(station_count, 3, -1)
    station_gap = np.concatenate((jac_position @ full_response, jac_rotation @ full_response), axis=1)
    station_shell = np.concatenate((jac_position[:, :, :ns] @ full_response[:ns],
                                    jac_rotation[:, :, :ns] @ full_response[:ns]), axis=1)
    station_beam = station_gap - station_shell
    if monotonic() > deadline:
        raise TimeoutError("bounded wall deadline")
    return {
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE", "source_git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": _digest(Path(__file__)), "base_runner_sha256": _digest(source / "scripts" / "b3_e4_f03_c1_material_line_mixed_metric.py"),
        "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
        "rows": total_rows, "constraint_rows": len(C), "beam_bond_halves": bonded,
        "beam_layout": beam_layout, "beta": float(singular[-1]),
        "dual_norm_sq": dual_norm, "shell_work_sq": shell_work, "beam_work_sq": beam_work,
        "action_work_sq": action_work, "pairing_work_sq": pairing,
        "beta_sq": float(singular[-1]**2), "svd_defect": float(np.linalg.norm(normalized @ Vh[-1] - singular[-1]*U[:, -1])),
        "channel": channel_report,
        "station_x_m": [float(s.x) for s in line.stations],
        "station_weights_m": [float(s.weight) for s in line.stations],
        "station_gap": station_gap.tolist(), "station_shell": station_shell.tolist(), "station_beam": station_beam.tolist(),
        "p0_line_norm": p0_info, "elapsed_s": monotonic()-started,
        "effective_threadpools": threadpool_info(),
        "env": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_DISABLE_JIT")},
    }


if __name__ == "__main__":
    output = Path(os.environ["TEMP"]) / "B3_E4_F03_C1_X8_B14_FY_ROTATION_AUDIT_DEV_V1.json"
    if output.exists():
        raise FileExistsError(output)
    with threadpool_limits(limits=1):
        result = evaluate()
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("rows", "beta", "shell_work_sq", "beam_work_sq", "elapsed_s")}))
