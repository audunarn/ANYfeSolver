"""Bounded native retained-KKT metric for the current C1 material-line rows.

This is a private G0 diagnostic. The auxiliary Riesz metric is not assembled
into the mechanics tangent. These bounded levels and one off-ratio comparison
are not a stability certificate or a multiplier-space selection.
"""

from __future__ import annotations

import argparse
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
from scripts.b3_e4_f03_c1_native_mixed_metric import (
    _beam_metric, _case, _drilling_metric, _line_metric, E, NU, L_LINE,
    MAX_ROWS, T,
)
from scripts.b3_e4_f03_material_line_position import MaterialLinePosition
from scripts.b3_e4_f03_material_line_space_screen import _rows
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


BEAM_HALVES = {2: 12, 4: 14, 8: 18}
KINDS = ("piecewise_constant", "continuous_linear", "dual_linear",
         "fz_half_p0")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _channel(metric: np.ndarray, axis: int, count: int) -> np.ndarray:
    ids = 6*np.arange(count)+axis
    return metric[np.ix_(ids, ids)]


def _scalar_q4_transverse_metric(xs: np.ndarray,
                                 ys: np.ndarray) -> np.ndarray:
    """Independent scalar H1/Q4 comparator, in joules for metre-valued w.

    The bending rigidity divided by the fixed physical line length squared
    sets the gradient scale. This is an auxiliary trace norm, not a shell law.
    """
    n = len(xs)*len(ys)
    result = np.zeros((n, n))
    coefficient = E*T**3/(12*(1-NU**2)*L_LINE**2)
    gauss = (-1/np.sqrt(3), 1/np.sqrt(3))
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            hx, hy = xs[i+1]-xs[i], ys[j+1]-ys[j]
            nodes = np.array((j*len(xs)+i, j*len(xs)+i+1,
                              (j+1)*len(xs)+i+1, (j+1)*len(xs)+i))
            local = np.zeros((4, 4))
            for xi in gauss:
                for eta in gauss:
                    shape = np.array(((1-xi)*(1-eta), (1+xi)*(1-eta),
                                      (1+xi)*(1+eta), (1-xi)*(1+eta)))/4
                    dndx = np.array((-(1-eta), 1-eta, 1+eta,
                                      -(1+eta)))/(2*hx)
                    dndy = np.array((-(1-xi), -(1+xi), 1+xi,
                                      1-xi))/(2*hy)
                    local += (coefficient*hx*hy/4)*(
                        np.outer(dndx, dndx)+np.outer(dndy, dndy)+
                        np.outer(shape, shape)/L_LINE**2)
            result[np.ix_(nodes, nodes)] += local
    return result


def _riesz_work_sq(metric: np.ndarray, action: np.ndarray) -> float:
    factor = linalg.cho_factor(metric, lower=True, check_finite=True)
    solution = linalg.cho_solve(factor, action, check_finite=True)
    return float(action@solution)


def evaluate(shell_cells: int, kind: str, *, exact_drill_quotient: bool = False,
             trace_only: bool = False, fz_riesz_audit: bool = False,
             beam_half_cells: int | None = None) -> dict:
    if shell_cells not in BEAM_HALVES or kind not in KINDS:
        raise ValueError("registered 1:1 material-line fixture required")
    beam_halves = (BEAM_HALVES[shell_cells] if beam_half_cells is None
                   else beam_half_cells)
    if (shell_cells, beam_halves) not in ((2, 12), (4, 14), (8, 18),
                                         (8, 14)):
        raise ValueError("material-line metric is bounded to registered pairs")
    if kind == "fz_half_p0" and (shell_cells, beam_halves) not in (
            (4, 14), (8, 18), (8, 14)):
        raise ValueError("fz half-resolution screen is bounded to x4/14, x8/18 or x8/14")
    if fz_riesz_audit and not (trace_only and exact_drill_quotient and
                               kind == "piecewise_constant"):
        raise ValueError("fz audit requires exact-quotient P0 trace metric")
    expected = (Path(__file__).resolve().parents[1]/"src"/"anyfesolver").resolve()
    if Path(anyfesolver.__file__).resolve().parent != expected:
        raise RuntimeError("mixed metric imported another ANYfeSolver checkout")
    start = monotonic()
    deadline = start+120.
    xs, ys = local_grid(shell_cells, transverse_cells=4)
    shell, shell_physical = raw_e4_tangent(xs, ys, thickness=T)
    beam = build(beam_halves, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    if beam.tangent is None:
        raise ArithmeticError("native retained tangent required")
    beam_bond_halves = int(np.sum(np.diff(beam.abscissae)[
        beam.abscissae[:-1] >= .9-1e-12] > 0.))
    if (shell_cells, beam_bond_halves) not in ((2, 2), (4, 4), (8, 8),
                                              (8, 4)):
        raise ArithmeticError("registered beam bond inventory changed")
    ns = len(shell)
    shell_free = np.array([i for i in range(ns)
                           if (i//6) % len(xs) != 0])
    beam_free = np.asarray(beam.context.free, dtype=int)
    drill_check = None
    if exact_drill_quotient:
        drill = shell_free[shell_free % 6 == 5]
        shell_active = shell_free[shell_free % 6 != 5]
        dd = shell[np.ix_(drill, drill)]
        dp = shell[np.ix_(drill, shell_active)]
        pd = shell[np.ix_(shell_active, drill)]
        pp = shell[np.ix_(shell_active, shell_active)]
        linalg.cholesky(dd, lower=True, check_finite=True)
        lift = linalg.solve(dd, dp, assume_a="pos", check_finite=True)
        shell_quotient = pp-pd@lift
        quotient_skew = float(np.linalg.norm(shell_quotient-shell_quotient.T)/
                              max(1., np.linalg.norm(shell_quotient)))
        if quotient_skew > 1e-10:
            raise ArithmeticError("exact shell drilling quotient is asymmetric")
        linalg.cholesky(shell_quotient, lower=True, check_finite=True)
        probe = np.sin(np.arange(len(shell_active))*.173)
        full_probe = np.zeros(ns)
        full_probe[shell_active] = probe
        full_probe[drill] = -lift@probe
        drill_check = {
            "quotient_relative_skew": quotient_skew,
            "drill_equation_relative": float(np.linalg.norm(
                (shell@full_probe)[drill])/max(1., np.linalg.norm(dp@probe))),
            "physical_action_relative": float(np.linalg.norm(
                (shell@full_probe)[shell_active]-shell_quotient@probe)/
                max(1., np.linalg.norm(shell_quotient@probe))),
            "drill_rows": len(drill),
        }
        if max(drill_check["drill_equation_relative"],
               drill_check["physical_action_relative"]) > 1e-8:
            raise ArithmeticError("exact shell drilling reconstruction failed")
        shell_metric_active = shell_quotient
        shell_tangent_active = shell_quotient
        shell_physical_active = shell_quotient
        shell_chart_active = np.zeros_like(shell_quotient)
    else:
        shell_active = shell_free
        R_shell = shell_physical.copy()
        drill_ids = 6*np.arange(len(xs)*len(ys))+5
        R_shell[np.ix_(drill_ids, drill_ids)] += _drilling_metric(xs, ys)
        shell_metric_active = R_shell[np.ix_(shell_free, shell_free)]
        shell_tangent_active = shell[np.ix_(shell_free, shell_free)]
        shell_physical_active = shell_physical[np.ix_(shell_free, shell_free)]
        shell_chart_active = shell_metric_active-shell_physical_active
    free = np.r_[shell_active, ns+beam_free]
    R_beam, R_resultant, layout = _beam_metric(beam)
    R_primal = linalg.block_diag(
        shell_metric_active,
        R_beam[np.ix_(beam_free, beam_free)])
    K = linalg.block_diag(
        shell_tangent_active,
        beam.tangent.toarray()[np.ix_(beam_free, beam_free)])
    line = MaterialLinePosition(xs, ys, beam)
    p0_rows, p0_masses = _rows(line, LineMultiplierSpace(
        "piecewise_constant", tuple(np.linspace(0., 1., shell_cells+1))))
    if abs(np.sum(p0_masses)-L_LINE) > 1e-12:
        raise ArithmeticError("P0 line measure changed")
    p0_metric, p0_info = _line_metric(
        "piecewise_constant", shell_cells, p0_masses)
    fz_info = None
    if kind == "piecewise_constant":
        rows, R_line = p0_rows, p0_metric
        yaw_info = None
    elif kind == "fz_half_p0":
        fz_intervals = shell_cells//2
        fz_rows, fz_masses = _rows(line, LineMultiplierSpace(
            "piecewise_constant", tuple(np.linspace(0., 1.,
                                                    fz_intervals+1))))
        if abs(np.sum(fz_masses)-L_LINE) > 1e-12:
            raise ArithmeticError("coarse fz line measure changed")
        fz_metric, fz_info = _line_metric(
            "piecewise_constant", fz_intervals, fz_masses)
        rows = np.vstack([fz_rows[2::6] if axis == 2 else
                          p0_rows[axis::6] for axis in range(6)])
        R_line = linalg.block_diag(*[
            _channel(fz_metric, 2, len(fz_masses)) if axis == 2 else
            _channel(p0_metric, axis, len(p0_masses))
            for axis in range(6)])
        yaw_info = None
    else:
        yaw_intervals = shell_cells//2
        yaw_rows, yaw_masses = _rows(line, LineMultiplierSpace(
            kind, tuple(np.linspace(0., 1., yaw_intervals+1))))
        if abs(np.sum(yaw_masses)-L_LINE) > 1e-12:
            raise ArithmeticError("yaw line measure changed")
        yaw_metric, yaw_info = _line_metric(kind, yaw_intervals, yaw_masses)
        rows = np.vstack([p0_rows[axis::6] for axis in range(5)] +
                         [yaw_rows[5::6]])
        R_line = linalg.block_diag(*[
            _channel(p0_metric, axis, len(p0_masses)) for axis in range(5)
        ], _channel(yaw_metric, 5, len(yaw_masses)))
    if exact_drill_quotient and np.max(np.abs(rows[:, drill])) > 1e-12:
        raise ArithmeticError("line rows unexpectedly act on shell drilling")
    C = rows[:, free]
    R = linalg.block_diag(R_primal, R_line)
    if len(K)+len(C) > MAX_ROWS or monotonic() > deadline:
        raise TimeoutError("bounded native KKT resource limit reached")
    primal_chol = linalg.cholesky(R_primal, lower=True, check_finite=True)
    line_chol = linalg.cholesky(R_line, lower=True, check_finite=True)
    normed_C = linalg.solve_triangular(line_chol, C, lower=True)
    normed_C = linalg.solve_triangular(
        primal_chol, normed_C.T, lower=True).T
    row_rank = int(np.linalg.matrix_rank(normed_C))
    if row_rank != len(C):
        raise ArithmeticError("normed material-line rows lost rank")
    if trace_only:
        u, singular, vh = linalg.svd(normed_C, full_matrices=False,
                                    check_finite=True)
        weak_dual = linalg.solve_triangular(
            line_chol.T, u[:, -1], lower=False, check_finite=True)
        if kind == "piecewise_constant":
            channels = [np.arange(axis, len(C), 6) for axis in range(6)]
        elif kind == "fz_half_p0":
            counts = [len(p0_masses)]*6
            counts[2] = len(fz_masses)
            offsets = np.r_[0, np.cumsum(counts)]
            channels = [np.arange(offsets[i], offsets[i+1])
                        for i in range(6)]
        else:
            n_p0 = len(p0_masses)
            channels = [np.arange(axis*n_p0, (axis+1)*n_p0)
                        for axis in range(5)] + [np.arange(5*n_p0, len(C))]
        shares = {name: float(weak_dual[ids] @
                              R_line[np.ix_(ids, ids)] @ weak_dual[ids])
                  for name, ids in zip(("fx", "fy", "fz", "mx", "my", "mz"),
                                       channels)}
        if abs(sum(shares.values())-1.) > 1e-8:
            raise ArithmeticError("trace channel metric share changed")
        report = {
            "rows": len(K)+len(C), "constraint_rows": len(C),
            "beta_min": float(singular[-1]),
            "operator_max": float(singular[0]),
            "normed_condition": float(singular[0]/singular[-1]),
            "trace_channel_metric_share": shares,
            "left_right_svd_defect": float(np.linalg.norm(
                normed_C@vh[-1]-singular[-1]*u[:, -1])),
            "dual_metric_norm_sq": float(weak_dual@R_line@weak_dual),
        }
        if fz_riesz_audit:
            fz_rows = channels[2]
            fz_dual = np.zeros_like(weak_dual)
            fz_dual[fz_rows] = weak_dual[fz_rows]
            fz_dual /= np.sqrt(fz_dual@R_line@fz_dual)
            fz_action = C.T@fz_dual
            shell_n = len(shell_active)
            shell_action = fz_action[:shell_n]
            beam_action = fz_action[shell_n:]
            shell_w = shell_active % 6 == 2
            other_shell_action = np.linalg.norm(shell_action[~shell_w])
            shell_w_action = shell_action[shell_w]
            scalar_metric = _scalar_q4_transverse_metric(xs, ys)
            scalar_ids = shell_active[shell_w]//6
            if len(np.unique(scalar_ids)) != len(scalar_ids):
                raise ArithmeticError("scalar shell trace has duplicate nodes")
            scalar_metric = scalar_metric[np.ix_(scalar_ids, scalar_ids)]
            shell_work = _riesz_work_sq(shell_metric_active, shell_action)
            beam_work = _riesz_work_sq(
                R_beam[np.ix_(beam_free, beam_free)], beam_action)
            scalar_work = _riesz_work_sq(scalar_metric, shell_w_action)
            direct_work = float(np.linalg.norm(
                normed_C.T@(line_chol.T@fz_dual))**2)
            split_defect = abs(shell_work+beam_work-direct_work)/max(
                1., abs(direct_work))
            if (other_shell_action > 1e-9*max(1., np.linalg.norm(shell_action))
                    or split_defect > 1e-9):
                raise ArithmeticError("physical fz action or Riesz split changed")
            report["fz_riesz_audit"] = {
                "least_mode_fz_metric_share_before_isolation": shares["fz"],
                "isolated_fz_dual_metric_norm_sq": float(
                    fz_dual@R_line@fz_dual),
                "isolated_fz_dual_action_N": fz_dual[fz_rows].tolist(),
                "shell_other_channel_action_norm_N": float(
                    other_shell_action),
                "shell_exact_drill_supremal_work_sq": shell_work,
                "beam_native_supremal_work_sq": beam_work,
                "scalar_q4_h1_shell_supremal_work_sq": scalar_work,
                "total_native_supremal_work_sq": direct_work,
                "native_trace_beta_for_isolated_fz": float(np.sqrt(
                    direct_work)),
                "scalar_shell_to_e4_shell_work_ratio": scalar_work/shell_work,
                "split_relative_defect": split_defect,
                "scalar_norm": "independent Q4 H1; E*T^3/[12(1-nu^2)*L_line^2]; plus zeroth/L_line^2",
                "status": "finite-level diagnostic; no interface or shell-law qualification",
            }
    else:
        A = np.block([[K, C.T], [C, np.zeros((len(C), len(C)))]])
        shell_n = len(shell_active)
        beam_n = len(beam_free)
        beam_resultant = R_resultant[np.ix_(beam_free, beam_free)]
        pieces = {"C": C, "blocks": [
            (("shell_reduced_full_tangent" if exact_drill_quotient
              else "shell_physical"), slice(0, shell_n), shell_physical_active),
            ("shell_drilling_chart", slice(0, shell_n), shell_chart_active),
            ("beam_kinematic", slice(shell_n, shell_n+beam_n),
             R_beam[np.ix_(beam_free, beam_free)]-beam_resultant),
            ("beam_complementary", slice(shell_n, shell_n+beam_n),
             beam_resultant),
            ("line_dual", slice(shell_n+beam_n, None), R_line),
        ]}
        report = _case(A, R, pieces, len(C), deadline)
    if monotonic() > deadline:
        raise TimeoutError("bounded native mixed metric deadline")
    return {
        "classification": "private material-line native retained KKT metric",
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "decision": "finite-level trace or mixed behavior only; no uniform stability or multiplier selection",
        "geometry": (f"C1 Q4 x{shell_cells}/y4; B3-{beam_halves}; "
                     f"{shell_cells}:{beam_bond_halves} shell:beam bond"),
        "beam_bond_half_cells": beam_bond_halves,
        "space_kind": kind,
        "trace_only": trace_only,
        "fz_riesz_audit": fz_riesz_audit,
        "representation": ("exact_drilling_quotient" if exact_drill_quotient
                           else "full_shell_chart"),
        "drilling_quotient_check": drill_check,
        "p0_intervals": shell_cells,
        "yaw_intervals": shell_cells//2 if kind in
                         ("continuous_linear", "dual_linear") else None,
        "fz_intervals": shell_cells//2 if kind == "fz_half_p0" else shell_cells,
        "row_rank": row_rank,
        "primal_rows": len(K),
        "beam_layout": layout,
        "p0_line_norm": p0_info,
        "yaw_line_norm": yaw_info,
        "fz_line_norm": fz_info,
        "report": report,
        "elapsed_s": monotonic()-start,
        "runner_sha256": _hash(Path(__file__)),
        "source_git": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
        "effective_threadpools": threadpool_info(),
        "env": {name: os.environ.get(name) for name in (
            "NUMBA_DISABLE_JIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "limits": ["one level per run", "1900 dense KKT rows", "120 s wall deadline",
                   "auxiliary Riesz norm not added to tangent",
                   "no physical transfer or mesh-uniform claim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell-cells", type=int, required=True)
    parser.add_argument("--beam-half-cells", type=int)
    parser.add_argument("--space-kind", choices=KINDS, required=True)
    parser.add_argument("--exact-drill-quotient", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--fz-riesz-audit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        result = evaluate(args.shell_cells, args.space_kind,
                          exact_drill_quotient=args.exact_drill_quotient,
                          trace_only=args.trace_only,
                          fz_riesz_audit=args.fz_riesz_audit,
                          beam_half_cells=args.beam_half_cells)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"rows": result["report"]["rows"],
                      "row_rank": result["row_rank"],
                      "beta_min": result["report"]["beta_min"],
                      "elapsed_s": result["elapsed_s"]}))


if __name__ == "__main__":
    main()
