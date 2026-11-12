"""Evaluate the preserved supported 2:1 mode through the virgin BL5 trace.

This reuses the saved auxiliary coordinates and does not rerun the historical
eigensolve, select a multiplier basis, or execute formal F03 assessment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import anyfesolver
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from anyfesolver._stiffened_shell_bl5_local_experimental import (
    BL5LocalOperator, LAW_ID,
)
from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
from scripts.b3_e4_f03_c1_fy_l1_mode_bridge import _recover, _l1_station


def _station(xs, ys, beam, qs, qb, x):
    shell_edge = min(int(np.searchsorted(xs, x, side="right"))-1, len(xs)-2)
    alpha = (x-xs[shell_edge])/(xs[shell_edge+1]-xs[shell_edge])
    yzero = int(np.flatnonzero(np.isclose(ys, 0., rtol=0., atol=1e-14))[0])
    nodes = (yzero*len(xs)+shell_edge, yzero*len(xs)+shell_edge+1)
    shell_reference = np.array(((xs[shell_edge], 0., 0.),
                                (xs[shell_edge+1], 0., 0.)))
    shell_q = qs.reshape(-1, 6)[list(nodes)]
    element, cell, weight, _ = beam._locate(x)
    state = beam.context.initial.mechanical
    reference = beam.context.elements[element][1].operator.reference
    port = NativeHalfCellPort.from_reference(
        reference, beam.context.nodes[element], element, cell, weight,
        beam.context.nodal_count, beam.context.count)
    operator = BL5LocalOperator.at_reference(
        shell_reference, np.zeros((2, 6)), .035, port, state,
        station=alpha)
    local = operator.evaluate(np.zeros((2, 6)), state, np.zeros(5))
    if np.max(np.abs(local.gap)) > 2e-13:
        raise AssertionError("virgin BL5 reference gap is not zero")
    direction = np.r_[shell_q.ravel(), qb[local.beam_slots]]
    return local.gap_jacobian@direction


def _saved_mean_row_guard(saved, xs, ys, beam, qs, qb):
    """Check physical mean-row work and each native port in the saved field."""
    x = np.asarray(saved["station_x_m"], dtype=float)
    weights = np.asarray(saved["station_weights_m"], dtype=float)
    historical = np.asarray(saved["station_gap"], dtype=float)
    shell = np.asarray(saved["station_shell"], dtype=float)
    native = np.asarray(saved["station_beam"], dtype=float)
    if (x.shape != (24,) or weights.shape != (24,) or
            any(a.shape != (24, 6) for a in (historical, shell, native)) or
            not all(np.isfinite(a).all() for a in
                    (x, weights, historical, shell, native)) or
            np.min(weights) <= 0. or abs(float(weights.sum())-.1) > 1e-14 or
            np.max(np.abs(shell+native-historical)) > 1e-12):
        raise AssertionError("complete saved mean-row station work required")
    cuts = np.linspace(.9, 1., 9)
    names = ("fx", "fy", "fz", "mx", "my", "mz")
    sign = np.array((-1., -1., -1., 1., 1., 1.))
    state = beam.context.initial.mechanical
    historical_work = np.zeros(2)
    reconstructed_work = np.zeros(2)
    max_native_gap_error = 0.
    max_native_station_work_error = 0.
    max_total_gap_error = 0.
    for k, station in enumerate(x):
        segment = int(np.searchsorted(cuts, station, side="right"))-1
        if not 0 <= segment < 8:
            raise AssertionError("saved station left the physical bond")
        # Historical P0 rows are mean gaps. The stored duals are conjugate to
        # those means; physical action densities divide by the row measure.
        density = np.array([
            saved["channel"][name]["dual_density"][
                segment//2 if name == "fz" else segment]
            /(.1/(4 if name == "fz" else 8)) for name in names])
        l1_density = sign*density
        operator, local_q, l1_total, _ = _l1_station(
            xs, ys, beam, qs, qb, float(station))
        _, l1_native = operator.evaluate(
            np.zeros((2, 6)), state,
            shell_direction=np.zeros_like(local_q), beam_direction=qb)
        total_error = float(np.max(np.abs(l1_total-sign*historical[k])))
        native_error = float(np.max(np.abs(l1_native-sign*native[k])))
        old_station_work = weights[k]*float(density@native[k])
        new_station_work = weights[k]*float(l1_density@l1_native)
        max_total_gap_error = max(max_total_gap_error, total_error)
        max_native_gap_error = max(max_native_gap_error, native_error)
        max_native_station_work_error = max(
            max_native_station_work_error,
            abs(old_station_work-new_station_work))
        historical_work += weights[k]*np.array((density@shell[k],
                                                 density@native[k]))
        reconstructed_work += weights[k]*np.array((
            l1_density@(l1_total-l1_native),
            l1_density@l1_native))
    expected = np.array((saved["shell_work_sq"], saved["beam_work_sq"]))
    work_error = max(float(np.max(np.abs(historical_work-expected))),
                     float(np.max(np.abs(reconstructed_work-expected))),
                     abs(float(reconstructed_work.sum())-saved["pairing_work_sq"]))
    if (max_total_gap_error > 1e-12 or max_native_gap_error > 1e-12 or
            max_native_station_work_error > 1e-12 or work_error > 1e-10):
        raise AssertionError("saved mean-row or station native work changed")
    return {
        "historical_shell_beam_work": historical_work.tolist(),
        "reconstructed_l1_shell_beam_work": reconstructed_work.tolist(),
        "max_total_gap_error": max_total_gap_error,
        "max_native_gap_error": max_native_gap_error,
        "max_native_station_work_error": max_native_station_work_error,
        "max_mean_row_work_error": work_error,
    }


def analyze(saved):
    xs, ys, _shell, _physical_shell, beam, _line, qs, qb = _recover(saved)
    work_guard = _saved_mean_row_guard(saved, xs, ys, beam, qs, qb)
    cuts = np.linspace(.9, 1., 9)
    beam_cuts = np.asarray(beam.abscissae)
    beam_cuts = beam_cuts[(beam_cuts >= .9-1e-14) &
                          (beam_cuts <= 1.+1e-14)]
    if (beam_cuts.shape != (5,) or
            abs(beam_cuts[0]-.9) > 1e-14 or
            abs(beam_cuts[-1]-1.) > 1e-14 or
            np.min(np.diff(beam_cuts)) <= 0.):
        raise AssertionError("four native bonded half-cells required")
    gauss = (.5-.5/np.sqrt(3.), .5+.5/np.sqrt(3.))
    stations, weights, gap, intervals, translation_parity = [], [], [], [], []
    for edge, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
        for alpha in gauss:
            x = float(left+alpha*(right-left))
            stations.append(x)
            weights.append(float(.5*(right-left)))
            bl5_gap = _station(xs, ys, beam, qs, qb, x)
            _, _, l1_gap, _ = _l1_station(xs, ys, beam, qs, qb, x)
            translation_parity.append(float(np.max(np.abs(
                bl5_gap[:3]+l1_gap[:3]))))
            gap.append(bl5_gap)
            intervals.append(min(int(np.searchsorted(
                beam_cuts, x, side="right"))-1, 3))
    stations = np.asarray(stations)
    weights = np.asarray(weights)
    gap = np.asarray(gap)
    intervals = np.asarray(intervals)
    if (gap.shape != (16, 5) or
            not np.isfinite(gap).all() or
            abs(float(weights.sum())-.1) > 1e-14):
        raise AssertionError("complete finite BL5 bond sampling required")
    if max(translation_parity) > 1e-12:
        raise AssertionError("BL5/L1 physical translation sign map changed")
    projected = {}
    for span in (1, 2):
        nrows = 4//span
        rows = np.zeros((5, nrows))
        measures = np.zeros(nrows)
        for index, edge in enumerate(intervals):
            column = edge//span
            rows[:, column] += weights[index]*gap[index]
            measures[column] += weights[index]
        fitted = np.array([rows[:, edge//span]/measures[edge//span]
                           for edge in intervals])
        projected[f"P0_span_{span}"] = {
            "row_measures_m": measures.tolist(),
            "weak_rows": rows.tolist(),
            "channels": [
                {"channel": name,
                 "continuous_l2": float(np.sqrt(np.dot(weights, gap[:, j]**2))),
                 "projected_l2": float(np.sqrt(np.dot(weights, fitted[:, j]**2))),
                 "unresolved_l2": float(np.sqrt(np.dot(
                     weights, (gap[:, j]-fitted[:, j])**2))),
                 "max_abs_weak_row": float(np.max(np.abs(rows[j])))}
                for j, name in enumerate(("tx", "ty", "tz", "director_1",
                                          "director_2"))],
        }
    return {
        "status": "BL5_SAVED_MODE_DEVELOPMENT_SCREEN_ONLY",
        "law": LAW_ID,
        "saved_source_git": saved["source_git"],
        "current_source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "imported_anyfesolver": str(Path(anyfesolver.__file__).resolve()),
        "station_x_m": stations.tolist(),
        "station_weights_m": weights.tolist(),
        "native_bond_halfcell_cuts_m": beam_cuts.tolist(),
        "continuous_gap": gap.tolist(),
        "max_abs_gap_by_channel": np.max(np.abs(gap), axis=0).tolist(),
        "max_bl5_plus_l1_translation_difference": max(translation_parity),
        "saved_mean_row_work_guard": work_guard,
        "projection": projected,
        "effective_threadpools": threadpool_info(),
        "limit": ("One reconstructed saved supported 2:1 field. No BL5 loaded "
                  "response, physical-fidelity ranking, stability assessment, "
                  "or formal F03 execution."),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        result = analyze(json.loads(args.saved.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"],
                      "max_abs_gap_by_channel": result["max_abs_gap_by_channel"],
                      "output": str(args.output)}))
