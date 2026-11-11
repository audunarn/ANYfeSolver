"""Bounded bridge from a saved 2:1 mode to the actual virgin L1 work map.

Reconstructs the old auxiliary least-action field from its saved multiplier;
does not rerun the SVD or claim an accepted F03 stability metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import anyfesolver
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_info, threadpool_limits

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts._b3_e4_f03_weak_line_candidate import LineMultiplierSpace
from scripts.b3_e4_f03_c1_native_mixed_metric import _beam_metric, _line_metric
from scripts.b3_e4_f03_l1_native_connection_probe import (
    L1LogConnection, shell_trace,
)
from scripts.b3_e4_f03_c1_fy_saved_mode_projection import _bases
from scripts.b3_e4_f03_material_line_position import MaterialLinePosition
from scripts.b3_e4_f03_material_line_space_screen import _rows
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


def _recover(saved: dict):
    xs, ys = local_grid(8, transverse_cells=4)
    shell, physical_shell = raw_e4_tangent(xs, ys, thickness=.035)
    beam = build(14, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    ns = len(shell)
    sf = np.array([i for i in range(ns) if (i // 6) % len(xs) != 0])
    drill = sf[sf % 6 == 5]
    sa = sf[sf % 6 != 5]
    dd = shell[np.ix_(drill, drill)]
    dp = shell[np.ix_(drill, sa)]
    quotient = shell[np.ix_(sa, sa)] - shell[np.ix_(sa, drill)] @ linalg.solve(
        dd, dp, assume_a="pos")
    bf = np.asarray(beam.context.free, dtype=int)
    beam_metric, _, _ = _beam_metric(beam)
    beam_metric = beam_metric[np.ix_(bf, bf)]
    primal = linalg.block_diag(quotient, beam_metric)
    line = MaterialLinePosition(xs, ys, beam)
    p0, p0_mass = _rows(
        line, LineMultiplierSpace("piecewise_constant", tuple(np.linspace(0., 1., 9))))
    fz, _ = _rows(
        line, LineMultiplierSpace("piecewise_constant", tuple(np.linspace(0., 1., 5))))
    _line_metric("piecewise_constant", 8, p0_mass)
    names = ("fx", "fy", "fz", "mx", "my", "mz")
    rows = np.vstack([p0[axis::6] if axis != 2 else fz[2::6]
                      for axis in range(6)])
    lam = np.concatenate([np.asarray(saved["channel"][name]["dual_density"])
                          for name in names])
    free = np.r_[sa, ns + bf]
    action = rows[:, free].T @ lam
    response = linalg.cho_solve(linalg.cho_factor(primal, lower=True), action)
    qs = np.zeros(ns)
    qb = np.zeros(beam.context.count)
    qs[sa] = response[:len(sa)]
    qs[drill] = -linalg.solve(dd, dp @ qs[sa], assume_a="pos")
    qb[bf] = response[len(sa):]
    old_gap = np.concatenate((
        line.reference_jacobian().reshape(len(line.stations), 3, -1) @
        np.r_[qs, qb],
        line.reference_orientation_jacobian().reshape(
            len(line.stations), 3, -1) @ np.r_[qs, qb]), axis=1)
    saved_gap = np.asarray(saved["station_gap"])
    if np.max(np.abs(old_gap - saved_gap)) > 1e-13:
        raise AssertionError("saved historical gap did not reconstruct")
    shell_work = float(action[:len(sa)] @ response[:len(sa)])
    beam_work = float(action[len(sa):] @ response[len(sa):])
    if (abs(shell_work - saved["shell_work_sq"]) > 1e-9 or
            abs(beam_work - saved["beam_work_sq"]) > 1e-9):
        raise AssertionError("saved least-action work did not reconstruct")
    return xs, ys, shell, physical_shell, beam, line, qs, qb


def _l1_station(xs, ys, beam, qs, qb, station: float):
    edge = min(int(np.searchsorted(xs, station, side="right")) - 1, len(xs) - 2)
    alpha = (station - xs[edge]) / (xs[edge + 1] - xs[edge])
    yzero = int(np.flatnonzero(np.isclose(ys, 0., rtol=0., atol=1e-14))[0])
    nodes = (yzero * len(xs) + edge, yzero * len(xs) + edge + 1)
    reference = np.array(((xs[edge], 0., 0.), (xs[edge + 1], 0., 0.)))
    local_q = qs.reshape(-1, 6)[list(nodes)]
    element, cell, weight, _ = beam._locate(station)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    state = beam.context.initial.mechanical
    operator = L1LogConnection.at_reference(
        reference, np.zeros((2, 6)), .035, port, state,
        beam_offset=(0., 0., -.05), station=alpha)
    gap, direction = operator.evaluate(
        np.zeros((2, 6)), state,
        shell_direction=local_q, beam_direction=qb)
    if np.max(np.abs(gap)) > 2e-13:
        raise AssertionError("L1 virgin gap is nonzero")
    _, frame, shell_step, _ = shell_trace(
        reference, np.zeros((2, 6)), .035, local_q, station=alpha)
    beam_spatial = port.tangent(state) @ qb
    _, _, beam_frame = port.pose_pair(state)
    arm = beam_frame @ np.array((0., 0., -.05))
    bottom_step = beam_spatial[:3] + np.cross(beam_spatial[3:], arm)
    slip = frame.T @ (bottom_step - shell_step)
    return operator, local_q, direction, slip


def analyze(saved: dict) -> dict:
    xs, ys, shell, physical_shell, beam, line, qs, qb = _recover(saved)
    compared = []
    for old, point in zip(saved["station_gap"], line.stations):
        _, _, l1, slip = _l1_station(xs, ys, beam, qs, qb, point.x)
        old = np.asarray(old)
        compared.append((np.max(np.abs(l1[:3] + old[:3])),
                         np.max(np.abs(l1[3:] - old[3:])),
                         np.max(np.abs(l1[:3] - slip))))
    bridge = np.max(compared, axis=0)
    midpoints = []
    cuts = np.linspace(.9, 1., 9)
    for a, b in zip(cuts[:-1], cuts[1:]):
        x = float((a + b) / 2.)
        _, _, gap, slip = _l1_station(xs, ys, beam, qs, qb, x)
        midpoints.append({"x_m": x, "l1_gap": gap.tolist(),
                          "physical_bottom_minus_top_slip_m": slip.tolist()})
    gauss_x, gauss_w, gauss_gap = [], [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        for alpha in (.5 - .5 / np.sqrt(3.), .5 + .5 / np.sqrt(3.)):
            station = float(a + alpha * (b - a))
            _, _, gap, _ = _l1_station(xs, ys, beam, qs, qb, station)
            gauss_x.append(station)
            gauss_w.append(float((b - a) / 2.))
            gauss_gap.append(gap)
    gauss_x = np.asarray(gauss_x)
    gauss_w = np.asarray(gauss_w)
    gauss_gap = np.asarray(gauss_gap)
    projected = {}
    for name, channel_bases in _bases(gauss_x, gauss_w).items():
        channels = []
        for j, basis in enumerate(channel_bases):
            gram = basis.T @ (gauss_w[:, None] * basis)
            rows = basis.T @ (gauss_w * gauss_gap[:, j])
            fitted = basis @ np.linalg.solve(gram, rows)
            norm = float(np.sqrt(np.dot(gauss_w, gauss_gap[:, j] ** 2)))
            missing = float(np.sqrt(np.dot(
                gauss_w, (gauss_gap[:, j] - fitted) ** 2)))
            channels.append({
                "channel": ("fx", "fy", "fz", "mx", "my", "mz")[j],
                "weighted_gap_norm": norm,
                "unresolved_fraction": missing / norm if norm else 0.,
                "max_abs_weak_row": float(np.max(np.abs(rows))),
            })
        projected[name] = channels
    unit = np.array((2., -3., 5., 7., -11., 13.))
    parity = []
    for x in (float(cuts[0] + .2113248654 * (cuts[1] - cuts[0])),
              float((cuts[3] + cuts[4]) / 2.),
              float(cuts[-2] + .7886751346 * (cuts[-1] - cuts[-2]))):
        operator, local_q, direction, _ = _l1_station(xs, ys, beam, qs, qb, x)
        value, gap, residual, _ = operator.work_tangent(
            np.zeros((2, 6)), beam.context.initial.mechanical, unit)
        independent = float(unit @ direction)
        transpose = float(residual @ np.r_[local_q.ravel(), qb])
        parity.append({"x_m": x, "work": value, "gap_max": float(np.max(np.abs(gap))),
                       "directional_work": independent, "transpose_work": transpose,
                       "absolute_difference": abs(independent - transpose)})
    p = np.concatenate([
        beam.context.nodal_count + 24 * element + np.arange(6, 24)
        for element in range(len(beam.context.elements))])
    free = np.asarray(beam.context.free, dtype=int)
    x = np.array([j for j in free if j not in set(p)], dtype=int)
    native = beam.tangent.toarray()
    hxx = native[np.ix_(x, x)]
    hxp = native[np.ix_(x, p)]
    hpp = native[np.ix_(p, p)]
    stationary = linalg.cho_solve(linalg.cho_factor(-hpp, lower=True),
                                  hxp.T @ qb[x])
    beam_energy = float(qb[x] @ hxx @ qb[x] + 2 * qb[x] @ hxp @ stationary +
                        stationary @ hpp @ stationary)
    shell_energy = float(qs @ physical_shell @ qs)
    if min(shell_energy, beam_energy) <= 0:
        raise AssertionError("nonpositive physical-core mode energy")
    return {
        "status": "VIRGIN_L1_SAVED_MODE_BRIDGE_ONLY",
        "source_git_of_mode": saved["source_git"],
        "current_source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
        "mode_action_work_shell": saved["shell_work_sq"],
        "mode_action_work_beam": saved["beam_work_sq"],
        "physical_core_shell_quadratic_j": shell_energy,
        "stationary_reduced_native_quadratic_j": beam_energy,
        "historical_to_l1_max_translation_sign_error": float(bridge[0]),
        "historical_to_l1_max_rotation_error": float(bridge[1]),
        "l1_to_physical_slip_max_error_m": float(bridge[2]),
        "off_quadrature_midpoints": midpoints,
        "actual_l1_two_point_projected": projected,
        "independent_work_transpose": parity,
        "effective_threadpools": threadpool_info(),
        "limit": "Virgin zero-width law and one saved mode; no formal F03 scoring.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        report = analyze(json.loads(args.saved.read_text()))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "historical_to_l1_max_translation_sign_error",
        "historical_to_l1_max_rotation_error", "l1_to_physical_slip_max_error_m",
        "physical_core_shell_quadratic_j",
        "stationary_reduced_native_quadratic_j")}))
