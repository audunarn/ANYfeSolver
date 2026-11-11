"""Bounded x16/y4 material-line loaded solves with exact E4 drill elimination.

Numerical drilling coordinates are reconstructed and checked in the
original full equations. This is a private development diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
import warnings

import anyfesolver
import numpy as np
import scipy
from scipy import linalg, sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from threadpoolctl import threadpool_info, threadpool_limits

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_weak_line_candidate import LineMultiplierSpace
from scripts.b3_e4_f03_material_line_loaded import _scales
from scripts.b3_e4_f03_material_line_position import MaterialLinePosition
from scripts.b3_e4_f03_material_line_space_screen import (
    _p0_gap_modes, _rows, _scaled_row_diagnostics,
)
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


MAX_ROWS = 1900
MAX_SECONDS = 120.


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left-right)/max(1., np.linalg.norm(right)))


def evaluate(*, shell_cells: int = 16,
             beam_half_cells: int = 12,
             p0_intervals: int = 4,
             mode_resolved_mx_only: bool = False,
             space_kind: str = "piecewise_constant",
             selective_yaw: bool = False,
             selective_fz: bool = False,
             yaw_intervals: int | None = None) -> dict:
    if selective_fz:
        if ((shell_cells, beam_half_cells, p0_intervals) != (8, 18, 8) or
                selective_yaw or mode_resolved_mx_only or
                space_kind != "piecewise_constant" or yaw_intervals is not None):
            raise ValueError("bounded x8/B3-18 fz m4 versus five P0 m8 fixture required")
    if selective_yaw:
        if (shell_cells, beam_half_cells, p0_intervals) not in (
                (2, 12, 2), (4, 14, 4), (8, 18, 8), (16, 14, 8)):
            raise ValueError("bounded selective-yaw refinement fixture required")
    elif not selective_fz and (shell_cells != 16 or
            (beam_half_cells, p0_intervals) not in (
            (12, 4), (14, 4), (14, 8))):
        raise ValueError("bounded x16 P0 ratio fixture required")
    if space_kind not in ("piecewise_constant", "continuous_linear",
                          "dual_linear"):
        raise ValueError("bounded material-line multiplier kind required")
    if selective_yaw and mode_resolved_mx_only:
        raise ValueError("bounded selective-yaw Mx/Fz comparison required")
    yaw_fixture = (shell_cells, beam_half_cells, p0_intervals, yaw_intervals)
    if selective_yaw and space_kind == "piecewise_constant" and (
            yaw_fixture != (16, 14, 8, 16)):
        raise ValueError("only independent x16/B3-14 P0 yaw m16 is admitted")
    if yaw_intervals is not None and (
            not selective_yaw or
            yaw_fixture not in ((8, 18, 8, 7), (16, 14, 8, 16))):
        raise ValueError("only fixed selective yaw m7/m16 diagnostics admitted")
    if space_kind != "piecewise_constant" and not selective_yaw and (
            (beam_half_cells, p0_intervals) != (14, 8) or
            not mode_resolved_mx_only):
        raise ValueError("bounded x16/B3-14/m8 Mx space comparison required")
    if mode_resolved_mx_only and (beam_half_cells, p0_intervals) not in ((14, 4), (14, 8)):
        raise ValueError("bounded fixed-4:1 x16/B3-14 P0 mode required")
    expected = (Path(__file__).resolve().parents[1]/"src"/"anyfesolver").resolve()
    if Path(anyfesolver.__file__).resolve().parent != expected:
        raise RuntimeError("bounded quotient imported another ANYfeSolver checkout")
    started = monotonic()
    xs, ys = local_grid(shell_cells, transverse_cells=4)
    shell, shell_physical_operator = raw_e4_tangent(xs, ys, thickness=.035)
    beam = build(beam_half_cells, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    if beam.tangent is None or beam.residual is None:
        raise ArithmeticError("assembled native fixture required")
    ns = len(shell)
    shell_free = np.array([i for i in range(ns) if (i//6) % len(xs) != 0])
    drill = shell_free[shell_free % 6 == 5]
    physical = shell_free[shell_free % 6 != 5]
    beam_free = np.asarray(beam.context.free, dtype=int)
    full_free = np.r_[shell_free, ns+beam_free]
    reduced_free = np.r_[physical, ns+beam_free]
    line = MaterialLinePosition(xs, ys, beam)
    knots = tuple(np.linspace(0., 1., p0_intervals+1))
    p0_rows, p0_measures = _rows(line, LineMultiplierSpace(
        "piecewise_constant", knots))
    if selective_yaw:
        yaw_count = shell_cells//2 if yaw_intervals is None else yaw_intervals
        yaw_knots = tuple(np.linspace(0., 1., yaw_count+1))
        yaw_rows, yaw_measures = _rows(line, LineMultiplierSpace(
            space_kind, yaw_knots))
        rows = np.vstack([p0_rows[axis::6] for axis in range(5)] +
                         [yaw_rows[5::6]])
        valid_measure = (np.min(p0_measures) > 1e-12 and
                         np.min(yaw_measures) > 1e-12 and
                         abs(np.sum(p0_measures)-.1) <= 1e-12 and
                         abs(np.sum(yaw_measures)-.1) <= 1e-12)
    elif selective_fz:
        fz_count = 4
        fz_rows, fz_measures = _rows(line, LineMultiplierSpace(
            "piecewise_constant", tuple(np.linspace(0., 1., fz_count+1))))
        rows = np.vstack([fz_rows[2::6] if axis == 2 else
                          p0_rows[axis::6] for axis in range(6)])
        valid_measure = (np.min(p0_measures) > 1e-12 and
                         np.min(fz_measures) > 1e-12 and
                         abs(np.sum(p0_measures)-.1) <= 1e-12 and
                         abs(np.sum(fz_measures)-.1) <= 1e-12)
    else:
        rows, measures = _rows(line, LineMultiplierSpace(space_kind, knots))
        valid_measure = (np.min(measures) > 1e-12 and
                         abs(np.sum(measures)-.1) <= 1e-12 and
                         (space_kind != "piecewise_constant" or np.allclose(
                             measures, .1/p0_intervals, rtol=0., atol=1e-12)))
    row_count = len(rows)
    if (rows.shape != (row_count, ns+beam.context.count) or
            not valid_measure or
            np.max(np.abs(rows[:, drill])) > 1e-12):
        raise ArithmeticError("reviewed line rows act on changed drilling coordinates")
    full_scales, energy_scale = _scales(ns, beam, full_free,
                                       row_count=row_count)
    row_check = _scaled_row_diagnostics(rows, full_free,
                                         full_scales[:len(full_free)])
    if row_check["scaled_row_rank"] != row_count:
        raise ArithmeticError("bounded P0 supported row rank changed")

    dd = shell[np.ix_(drill, drill)]
    dp = shell[np.ix_(drill, physical)]
    pd = shell[np.ix_(physical, drill)]
    pp = shell[np.ix_(physical, physical)]
    if max(_relative(dd, dd.T), _relative(pd, dp.T)) > 1e-10:
        raise ArithmeticError("drilling partition is not conservative")
    linalg.cholesky(dd, lower=True, check_finite=True)
    lift = linalg.solve(dd, dp, assume_a="pos", check_finite=True)
    quotient = pp-pd@lift
    test = np.sin(np.arange(len(physical))*.173)
    test_drill = -lift@test
    shell_test = np.zeros(ns)
    shell_test[physical], shell_test[drill] = test, test_drill
    drill_check = float(np.linalg.norm(dd@test_drill+dp@test)/
                        max(1., np.linalg.norm(dp@test)))
    schur_check = _relative((shell@shell_test)[physical], quotient@test)
    if max(drill_check, schur_check) > 1e-8:
        raise ArithmeticError(
            f"drilling quotient action reconstruction failed: "
            f"drill={drill_check:.6g}, schur={schur_check:.6g}")
    if monotonic()-started > MAX_SECONDS:
        raise TimeoutError("bounded quotient preparation deadline")

    beam_block = beam.tangent[beam_free, :][:, beam_free].tocsr()
    k_reduced = sparse.block_diag((sparse.csr_matrix(quotient), beam_block),
                                   format="csr")
    c = sparse.csr_matrix(rows[:, reduced_free])
    dimension = k_reduced.shape[0]+c.shape[0]
    if dimension > MAX_ROWS:
        raise ValueError("bounded exact-drill KKT row cap exceeded")
    kkt = sparse.bmat(((k_reduced, c.T), (c, None)), format="csr")
    scales, scale_energy = _scales(ns, beam, reduced_free,
                                   row_count=row_count)
    if energy_scale != scale_energy:
        raise ArithmeticError("full/quotient physical scales changed")
    scaled = (sparse.diags(scales)@kkt@sparse.diags(scales)/energy_scale).tocsc()
    accepted = np.r_[np.zeros(ns), beam.residual]
    if np.max(np.abs(accepted[drill])) != 0.:
        raise ArithmeticError("nonzero drill predecessor would require RHS Schur shift")
    shell_reference = line.reference_nodes
    beam_reference = (beam.context.initial.mechanical.positions+
                      beam.context.initial.mechanical.position_low)
    station_position = line.reference_jacobian().reshape(len(line.stations), 3, -1)
    station_orientation = line.reference_orientation_jacobian().reshape(
        len(line.stations), 3, -1)
    weights = np.asarray([station.weight for station in line.stations])
    cases = {}
    load_specs = (("tip_mx_100nm", 3, 100.),) if mode_resolved_mx_only else (
        ("tip_mx_100nm", 3, 100.), ("tip_fz_1000n", 2, 1000.))
    for label, axis, magnitude in load_specs:
        if monotonic()-started > MAX_SECONDS:
            raise TimeoutError("bounded exact-drill loaded deadline")
        load = np.zeros(ns+beam.context.count)
        load[ns+6*beam.n_half_cells+axis] = magnitude
        rhs_full = load-accepted
        if np.max(np.abs(rhs_full[drill])) != 0.:
            raise ArithmeticError("external drill load would require RHS Schur shift")
        rhs = np.r_[rhs_full[reduced_free]*scales[:len(reduced_free)]/
                    energy_scale, np.zeros(row_count)]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            answer = spsolve(scaled, rhs)
        if not np.isfinite(answer).all():
            raise ArithmeticError("nonfinite bounded quotient solve")
        state = np.zeros(ns+beam.context.count)
        state[reduced_free] = scales[:len(reduced_free)]*answer[:len(reduced_free)]
        state[drill] = -lift@state[physical]
        dual = scales[len(reduced_free):]*answer[len(reduced_free):]
        equation = accepted+np.r_[shell@state[:ns],
                                  beam.tangent@state[ns:]]+rows.T@dual-load
        gap = rows@state
        full_free_residual = float(np.linalg.norm(equation[full_free])/
                                   max(1., np.linalg.norm(load[full_free])))
        drill_residual = float(np.max(np.abs(equation[drill])))
        weak_gap = float(np.max(np.abs(gap)))
        work = float(state@np.r_[shell@state[:ns],
                                 beam.tangent@state[ns:]])
        external_work = float(rhs_full@state)
        work_relative = abs(work-external_work)/max(1e-12, abs(external_work))
        shell_work = float(state[:ns]@shell_physical_operator@state[:ns])
        shell_support = equation[:ns].reshape(-1, 6)
        beam_support = equation[ns:ns+beam.context.nodal_count].reshape(-1, 6)
        support_force = shell_support[:, :3].sum(axis=0)+beam_support[:, :3].sum(axis=0)
        support_moment = (np.sum(np.cross(shell_reference,
                                          shell_support[:, :3])+
                                 shell_support[:, 3:], axis=0)+
                          np.sum(np.cross(beam_reference,
                                          beam_support[:, :3])+
                                 beam_support[:, 3:], axis=0))
        applied_force = np.zeros(3)
        applied_moment = np.zeros(3)
        if axis < 3:
            applied_force[axis] = magnitude
            applied_moment = np.cross(beam_reference[-1], applied_force)
        else:
            applied_moment[axis-3] = magnitude
        position_gap = np.einsum("kij,j->ki", station_position, state)
        rotation_gap = np.einsum("kij,j->ki", station_orientation, state)
        terminal = ns+6*beam.n_half_cells
        case = {
            "status": "BOUNDED_SOLVE_PASSED",
            "full_free_equilibrium_ratio": full_free_residual,
            "drilling_equilibrium_max": drill_residual,
            "weak_constraint_max": weak_gap,
            "work_relative_error": work_relative,
            "support_force_balance_n": float(np.linalg.norm(support_force+applied_force)),
            "support_moment_balance_nm": float(np.linalg.norm(support_moment+applied_moment)),
            "shell_physical_work_j": shell_work,
            "tip_translation_m": state[terminal:terminal+3].tolist(),
            "tip_rotation_rad": state[terminal+3:terminal+6].tolist(),
            "position_rms_m": float(np.sqrt(np.sum(weights*np.sum(position_gap**2,
                                                            axis=1))/.1)),
            "orientation_rms_rad": float(np.sqrt(np.sum(weights*np.sum(rotation_gap**2,
                                                               axis=1))/.1)),
            "position_max_m": float(np.max(np.linalg.norm(position_gap, axis=1))),
            "orientation_max_rad": float(np.max(np.linalg.norm(rotation_gap, axis=1))),
        }
        if mode_resolved_mx_only or selective_yaw or selective_fz:
            case["continuous_gap_modes"] = _p0_gap_modes(
                line, position_gap, rotation_gap, p0_intervals,
                p0_rows@state)
        if (max(full_free_residual, weak_gap, work_relative) > 1e-7 or
                case["support_force_balance_n"] > 1e-5 or
                case["support_moment_balance_nm"] > 1e-5 or
                shell_work <= 0.):
            raise ArithmeticError(f"reconstructed full loaded system failed: {case}")
        cases[label] = case
    yaw_model = f" / yaw m{yaw_count}" if selective_yaw else ""
    fz_model = " / fz m4" if selective_fz else ""
    return {
        "classification": "bounded exact-drill loaded material-line space diagnostic",
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "source_git": subprocess.check_output(("git", "rev-parse", "HEAD"),
                                              text=True).strip(),
        "model": (f"C1 Q4 x{shell_cells}/y4 35 mm shell, B3-{beam_half_cells}, "
                  f"material-line {'P0-five plus '+space_kind+' yaw' if selective_yaw else space_kind} "
                  f"P0 m{p0_intervals}{yaw_model}{fz_model}, "
                  "original full equations reconstructed"),
        "shell_cells": shell_cells,
        "beam_half_cells": beam_half_cells,
        "p0_intervals": p0_intervals,
        "space_kind": space_kind,
        "selective_yaw": selective_yaw,
        "selective_fz": selective_fz,
        "fz_intervals": fz_count if selective_fz else p0_intervals,
        "yaw_intervals": (yaw_count if selective_yaw else None),
        "mode_resolved_mx_only": mode_resolved_mx_only,
        "full_kkt_rows": len(full_free)+row_count,
        "quotient_kkt_rows": dimension,
        "quotient_nnz": int(scaled.nnz),
        "max_rows": MAX_ROWS,
        "max_seconds": MAX_SECONDS,
        "elapsed_s": monotonic()-started,
        "row_diagnostic": {key: row_check[key] for key in (
            "scaled_row_rank", "active_primal_columns", "rank_tolerance")},
        "drill_partition_rows": len(drill),
        "drill_test_relative_residual": drill_check,
        "schur_test_relative_residual": schur_check,
        "cases": cases,
        "limits": ["private exact algebraic elimination, not production condensation",
                   "one level and terminal nodal loads per run, not mesh-uniform stability",
                   "no physical connection accuracy or multiplier-space selection"],
        "provenance": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "anyfesolver_import": str(Path(anyfesolver.__file__).resolve()),
            "threadpools": threadpool_info(),
            "env": {key: os.environ.get(key) for key in
                    ("NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shell-cells", type=int, choices=(2, 4, 8, 16), default=16)
    parser.add_argument("--beam-half-cells", type=int, choices=(12, 14, 18), default=12)
    parser.add_argument("--p0-intervals", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--mode-resolved-mx-only", action="store_true")
    parser.add_argument("--selective-yaw", action="store_true")
    parser.add_argument("--selective-fz", action="store_true")
    parser.add_argument("--yaw-intervals", type=int)
    parser.add_argument("--space-kind", choices=("piecewise_constant",
                                                  "continuous_linear",
                                                  "dual_linear"),
                        default="piecewise_constant")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        result = evaluate(shell_cells=args.shell_cells,
                          beam_half_cells=args.beam_half_cells,
                          p0_intervals=args.p0_intervals,
                          mode_resolved_mx_only=args.mode_resolved_mx_only,
                          space_kind=args.space_kind,
                          selective_yaw=args.selective_yaw,
                          selective_fz=args.selective_fz,
                          yaw_intervals=args.yaw_intervals)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"elapsed_s": result["elapsed_s"],
                      "rows": result["quotient_kkt_rows"],
                      "cases": {name: value["status"]
                                for name, value in result["cases"].items()}}))


if __name__ == "__main__":
    main()
