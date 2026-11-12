"""Compare two virgin F03 traces on the frozen 2:1 Fy/rotation mode.

This is a same-native-model first-variation screen. It does not establish a
physical reference, select a bonded line, or consume formal F03 authority.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

import anyfesolver
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_c1_fy_l1_mode_bridge import _l1_station, _recover
from scripts.b3_e4_f03_finite_patch_line_work import (
    FinitePatchLineFirstVariation,
)


def _density(saved: dict, segment: int) -> np.ndarray:
    """Map the historical material-line action to the L1 shell frame."""
    names = ("fx", "fy", "fz", "mx", "my", "mz")
    values = []
    for name in names:
        dual = saved["channel"][name]["dual_density"]
        count = 4 if name == "fz" else 8
        if len(dual) != count:
            raise ValueError("frozen P0 multiplier row count changed")
        # _rows divides each integrated P0 row by its physical measure.
        # Its dual is conjugate to a mean gap, not a density per metre.
        values.append(dual[segment // 2 if name == "fz" else segment] /
                      (.1/count))
    values = np.asarray(values, dtype=float)
    values[:3] *= -1.  # Historical and L1 translation gaps have opposite signs.
    return values


def evaluate(saved: dict, source_path: Path) -> dict:
    expected = (Path(__file__).resolve().parents[1] / "src" / "anyfesolver").resolve()
    actual = Path(anyfesolver.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError("saved-mode screen imported another checkout")
    xs, ys, _shell, _physical, beam, _line, qs, qb = _recover(saved)
    if (saved["source_git"] != "f50583dace8499aaf0f3e43affb11fa0f8d435d9" or
            len(xs) < 12 or len(qs) != len(xs)*len(ys)*6 or
            len(qb) != beam.context.count):
        raise ValueError("frozen 2:1 saved mode required")
    selected = np.flatnonzero((xs >= .9-1e-12) & (xs <= 1.+1e-12))
    cuts = xs[selected]
    if len(cuts) != 9 or not np.allclose(cuts, np.linspace(.9, 1., 9)):
        raise ValueError("eight-segment bond interval changed")
    shell_direction = qs.reshape(len(ys), len(xs), 6).transpose(1, 0, 2)[selected]
    reference = np.array([[(x, y, 0.) for y in ys] for x in cuts])
    state = beam.context.initial.mechanical

    def port_at(station: float) -> NativeHalfCellSpatialPort:
        element, cell, weight, _ = beam._locate(station)
        return NativeHalfCellSpatialPort(beam.context, element, cell, weight)

    patch = FinitePatchLineFirstVariation.at_reference(
        reference, cuts, ys, port_at, state,
        longitudinal_width=.1, transverse_bounds=(-.05, .05),
        thickness=.035)
    density = np.array([_density(saved, i) for i in range(8)])
    zero = np.zeros(patch.shell_shape)
    _gaps, checked_work = patch.evaluate(
        zero, state, shell_direction, qb, density)
    zero_beam = np.zeros_like(qb)
    split = {name: np.zeros(2) for name in ("l1", "finite_patch")}
    segment_work = []
    station_gap = []
    station_beam_errors = []
    for item in patch.stations:
        station = item.patch.station
        lam = density[item.segment]
        operator, local_q, l1_full, slip = _l1_station(
            xs, ys, beam, qs, qb, station)
        _gap, l1_shell = operator.evaluate(
            np.zeros((2, 6)), state, shell_direction=local_q,
            beam_direction=zero_beam)
        _gap, l1_beam = operator.evaluate(
            np.zeros((2, 6)), state, shell_direction=np.zeros((2, 6)),
            beam_direction=qb)
        if np.max(np.abs(l1_full-l1_shell-l1_beam)) > 1e-12:
            raise AssertionError("L1 first-variation split changed")

        shell, shell_frame, shell_step, frame_step = item.patch.directional(
            zero, shell_direction)
        high, low, beam_frame = item.port.pose_pair(state)
        arm = beam_frame @ item.beam_offset
        beam_point = high+low+arm
        jac = item.joint.evaluate(
            np.array((shell, beam_point)),
            np.array((shell_frame, beam_frame)), np.zeros(6)
        ).constraint_jacobian
        skew = frame_step @ shell_frame.T
        spin = np.array((skew[2, 1], skew[0, 2], skew[1, 0]))
        patch_shell = jac[:, :6] @ np.r_[shell_step, spin]
        spatial = item.port.tangent(state) @ qb
        beam_step = spatial[:3] + np.cross(spatial[3:], arm)
        patch_beam = jac[:, 6:] @ np.r_[beam_step, spatial[3:]]
        l1_beam_work = float(item.measure * lam @ l1_beam)
        patch_beam_work = float(item.measure * lam @ patch_beam)
        station_error = abs(l1_beam_work-patch_beam_work)
        station_scale = max(abs(l1_beam_work), abs(patch_beam_work), 1e-12)
        if station_error > 1e-10*station_scale:
            raise AssertionError("same-action native beam work changed at a station")
        station_beam_errors.append(station_error)
        for name, shell_gap, beam_gap in (
                ("l1", l1_shell, l1_beam),
                ("finite_patch", patch_shell, patch_beam)):
            split[name] += item.measure * np.array((
                lam @ shell_gap, lam @ beam_gap))
        segment_work.append({
            "station_x_m": station,
            "segment": item.segment,
            "l1_shell_work": float(item.measure * lam @ l1_shell),
            "patch_shell_work": float(item.measure * lam @ patch_shell),
            "common_beam_work": l1_beam_work,
            "patch_beam_work": patch_beam_work,
            "station_beam_work_error": station_error,
        })
        station_gap.append({
            "station_x_m": station,
            "l1": l1_full.tolist(),
            "finite_patch": (patch_shell+patch_beam).tolist(),
            "bottom_minus_top_slip_m": slip.tolist(),
        })
    beam_error = abs(split["l1"][1]-split["finite_patch"][1])
    patch_error = abs(checked_work-float(sum(split["finite_patch"])))
    historical_work = float(fsum((saved["shell_work_sq"], saved["beam_work_sq"])))
    l1_total = float(fsum(split["l1"]))
    historical_error = abs(l1_total-historical_work)
    scale = max(abs(checked_work), abs(float(sum(split["l1"]))), 1.)
    if max(beam_error, patch_error) > 1e-10*scale:
        raise AssertionError("same-action beam or patch work did not match")
    if historical_error > 1e-10*max(abs(l1_total), abs(historical_work), 1e-12):
        raise AssertionError("L1 total work differs from frozen mean-row action")
    return {
        "classification": "development same-action trace screen; no F03 acceptance",
        "saved_mode_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "saved_mode_source_git": saved["source_git"],
        "historical_mean_row_action_work": historical_work,
        "imported_anyfesolver": str(actual),
        "shell_grid_nodes": len(xs)*len(ys),
        "native_beam_halfcells": len(beam.context.elements)*2,
        "bond_segments": patch.segment_count,
        "stations": len(patch.stations),
        "fixed_patch_longitudinal_width_m": .1,
        "fixed_patch_transverse_bounds_m": [-.05, .05],
        "l1_shell_beam_work": split["l1"].tolist(),
        "finite_patch_shell_beam_work": split["finite_patch"].tolist(),
        "shell_work_difference": float(split["finite_patch"][0]-split["l1"][0]),
        "same_beam_work_error": float(beam_error),
        "max_station_beam_work_error": max(station_beam_errors),
        "l1_historical_total_work_error": historical_error,
        "patch_work_internal_error": float(patch_error),
        "station_work": segment_work,
        "station_directional_gaps": station_gap,
        "effective_threadpools": threadpool_info(),
        "limitation": "Virgin auxiliary mode; no matched physical reference, stability or law selection.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing development evidence")
    saved = json.loads(args.saved.read_text(encoding="utf-8"))
    with threadpool_limits(limits=1):
        result = evaluate(saved, args.saved)
    args.output.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "l1_shell_beam_work", "finite_patch_shell_beam_work",
        "shell_work_difference", "same_beam_work_error")}))
