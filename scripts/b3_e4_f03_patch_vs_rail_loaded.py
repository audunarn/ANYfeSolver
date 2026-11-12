"""Bounded same-model loaded comparison of two unselected shell-side maps.

The finite patch and physical rail share the flat E4/native B3 fixture,
supports, retained variables and applied tip actions. This is a development
decision screen, not an external reference or a selected bonded-line law.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anyfesolver
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_finite_patch_line_work import (
    FinitePatchLineFirstVariation,
)
from scripts.b3_e4_f03_material_line_loaded import _scales, _solve
from scripts.b3_e4_f03_physical_rail_loaded_screen import (
    evaluate as evaluate_rail,
    fixture_identity,
)
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent


def _patch_rows(line: FinitePatchLineFirstVariation) -> np.ndarray:
    """Dense independent oracle only for the fixed, small development model."""
    shell_size = int(np.prod(line.shell_shape))
    if (line.segment_count != 4 or shell_size != 90 or
            line.beam_size > 1000):
        raise ValueError("bounded four-segment patch oracle required")
    rows = np.zeros((6*line.segment_count, shell_size+line.beam_size))
    zero = np.zeros(line.shell_shape)
    direction = np.zeros(line.shell_shape)
    state = line.stations[0].port.context.initial.mechanical
    for item in line.stations:
        shell, shell_frame = item.patch.pose(zero)
        high, low, beam_frame = item.port.pose_pair(state)
        arm = beam_frame@item.beam_offset
        beam_point = high+low+arm
        jacobian = item.joint.evaluate(
            np.array((shell, beam_point)),
            np.array((shell_frame, beam_frame)), np.zeros(6)
        ).constraint_jacobian
        row = slice(6*item.segment, 6*(item.segment+1))
        for column in range(shell_size):
            direction.flat[column] = 1.
            _, frame, point_step, frame_step = item.patch.directional(
                zero, direction)
            direction.flat[column] = 0.
            skew = frame_step@frame.T
            spin = np.array((skew[2, 1], skew[0, 2], skew[1, 0]))
            rows[row, column] += item.measure * (
                jacobian[:, :6]@np.r_[point_step, spin])
        native = item.port.tangent(state)
        native_point = native[:3]+np.cross(native[3:].T, arm).T
        native_pose = np.vstack((native_point, native[3:]))
        rows[row, shell_size:] += item.measure*jacobian[:, 6:]@native_pose
    return rows


def evaluate() -> dict:
    expected_package = (Path(__file__).resolve().parents[1] /
                        "src" / "anyfesolver").resolve()
    actual_package = Path(anyfesolver.__file__).resolve().parent
    if actual_package != expected_package:
        raise RuntimeError("loaded comparison imported another checkout")

    xs = np.linspace(0., 1., 9)
    ys = np.array((-.1, 0., .1))
    bond_start = 4
    bond_x = xs[bond_start:]
    thickness = .035
    shell_total, shell_physical = raw_e4_tangent(
        xs, ys, thickness=thickness)
    beam = build(8, z_offset=.0675)
    if beam.tangent is None or beam.residual is None:
        raise ArithmeticError("assembled native retained beam required")
    state = beam.context.initial.mechanical

    def port_at(station):
        element, cell, weight, _ = beam._locate(station)
        return NativeHalfCellSpatialPort(
            beam.context, element, cell, weight)

    reference = np.array(
        [[(x, y, 0.) for y in ys] for x in bond_x])
    line = FinitePatchLineFirstVariation.at_reference(
        reference, bond_x, ys, port_at, state,
        longitudinal_width=.1, transverse_bounds=(-.05, .05),
        thickness=thickness)
    local_rows = _patch_rows(line)
    shell_size = shell_total.shape[0]
    beam_size = beam.context.count
    constraint = np.zeros((24, shell_size+beam_size))
    for i in range(len(bond_x)):
        for j in range(len(ys)):
            source = 6*(i*len(ys)+j)
            target = 6*(j*len(xs)+bond_start+i)
            constraint[:, target:target+6] += (
                local_rows[:, source:source+6])
    constraint[:, shell_size:] = local_rows[:, 90:]

    rng = np.random.default_rng(572)
    dq = rng.normal(scale=.01, size=line.shell_shape)
    db = rng.normal(scale=.01, size=beam_size)
    density = rng.normal(size=(line.segment_count, 6))
    _, actual_work = line.evaluate(
        np.zeros(line.shell_shape), state, dq, db, density)
    global_direction = np.zeros(shell_size+beam_size)
    for i in range(len(bond_x)):
        for j in range(len(ys)):
            target = 6*(j*len(xs)+bond_start+i)
            global_direction[target:target+6] = dq[i, j]
    global_direction[shell_size:] = db
    predicted_work = float(
        density.ravel()@constraint@global_direction)
    work_map_error = abs(actual_work-predicted_work)
    if work_map_error > 3e-11:
        raise ArithmeticError("finite patch row map lost virtual work")

    shell_nodes = np.array(
        [(x, y, 0.) for y in ys for x in xs])
    base_fixed = {
        6*(j*len(xs))+axis
        for j in range(len(ys)) for axis in range(6)
    }
    primal_k = {
        "total": shell_total,
        "physical_core": shell_physical,
    }
    results = {}
    terminal = shell_size+6*beam.n_half_cells
    for shell_law, shell_k in primal_k.items():
        primal = sparse.block_diag(
            (sparse.csr_matrix(shell_k), beam.tangent),
            format="csr")
        fixed = set(base_fixed)
        if shell_law == "physical_core":
            fixed.update(range(5, shell_size, 6))
        free = np.array(sorted(
            set(range(primal.shape[0]))-fixed), dtype=int)
        scales, energy_scale = _scales(
            shell_size, beam, free, row_count=24)
        residual = np.r_[np.zeros(shell_size), beam.residual]
        for label, axis, magnitude in (
                ("tip_mx_100nm", 3, 100.),
                ("tip_fz_1000n", 2, 1000.)):
            load = np.zeros(primal.shape[0])
            load[terminal+axis] = magnitude
            solution, equation, diagnostics = _solve(
                primal, residual, constraint, free, scales,
                energy_scale, load)
            shell_step = solution[:shell_size]
            support = equation[:shell_size].reshape(-1, 6)
            force = support[:, :3].sum(axis=0)
            moment = np.sum(
                np.cross(shell_nodes, support[:, :3])+support[:, 3:],
                axis=0)
            applied_force = np.zeros(3)
            applied_moment = np.zeros(3)
            if axis < 3:
                applied_force[axis] = magnitude
                applied_moment = np.cross(
                    (1., 0., .0675), applied_force)
            else:
                applied_moment[axis-3] = magnitude
            balance = np.r_[
                force+applied_force, moment+applied_moment]
            physical_quadratic = float(
                shell_step@shell_physical@shell_step)
            numerical_quadratic = float(
                shell_step@(shell_total-shell_physical)@shell_step)
            if (physical_quadratic <= 0. or
                    max(np.max(np.abs(balance[:3]))/1000.,
                        np.max(np.abs(balance[3:]))/100.) > 1e-7):
                raise ArithmeticError(
                    "finite patch lost physical support balance")
            results[f"{shell_law}/{label}"] = {
                "diagnostics": diagnostics,
                "tip_uz_m": float(solution[terminal+2]),
                "tip_rx_rad": float(solution[terminal+3]),
                "shell_physical_quadratic_j": physical_quadratic,
                "shell_numerical_quadratic_j": numerical_quadratic,
                "support_force_n": force.tolist(),
                "support_moment_nm": moment.tolist(),
                "global_balance": balance.tolist(),
            }

    rail = evaluate_rail()
    own_fixture = fixture_identity(
        xs, ys, bond_start, thickness, (0., 0., -.05),
        beam, base_fixed)
    if (rail["fixture_identity"] != own_fixture or
            rail["shell_elements"] != 16 or
            rail["native_b3_halfcells"] != 8 or
            rail["bond_segments"] != 4 or
            rail["multiplier_rows"] != 24):
        raise AssertionError("physical rail comparison fixture changed")
    comparison = {}
    for name, patch in results.items():
        old = rail["cases"][name]
        primary = "tip_rx_rad" if name.endswith("tip_mx_100nm") else "tip_uz_m"
        if abs(old[primary]) <= 1e-15:
            raise ArithmeticError("rail primary response is too small for a ratio")
        comparison[name] = {
            "primary_response": primary,
            "primary_ratio_patch_to_rail":
                patch[primary]/old[primary],
            "shell_physical_quadratic_ratio_patch_to_rail":
                patch["shell_physical_quadratic_j"]/
                old["shell_physical_quadratic_j"],
        }
    return {
        "classification": "bounded same-model development comparison; no F03 acceptance",
        "imported_anyfesolver": str(actual_package),
        "numerical_thread_environment": {
            name: os.getenv(name) for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "effective_threadpools": threadpool_info(),
        "shell_elements": 16,
        "native_b3_halfcells": 8,
        "bond_segments": 4,
        "multiplier_rows": 24,
        "fixture_identity": own_fixture,
        "patch_longitudinal_width_m": .1,
        "patch_transverse_bounds_m": [-.05, .05],
        "work_map_absolute_error": work_map_error,
        "patch_cases": results,
        "rail_cases": rail["cases"],
        "comparison": comparison,
        "limitations": [
            "virgin flat linearization; no multiplier curvature",
            "one fixed interface space and mesh ratio",
            "not a matched external physical reference",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing comparison")
    result = evaluate()
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "development_comparison_complete",
        "work_map_absolute_error": result["work_map_absolute_error"],
        "comparison": result["comparison"],
    }))
