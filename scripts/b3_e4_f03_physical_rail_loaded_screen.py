"""Bounded loaded KKT screen for the unselected physical-width rail.

This is a flat elastic development fixture, not a bonded-line qualification
or an external reference comparison. It keeps the native B3 retained blocks.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import anyfesolver

from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_flat_rail_six_channel_map import (
    flat_rail_p0_shell_pose_action,
)
from scripts.b3_e4_f03_l1_physical_rail import SegmentedRailWeakWork
from scripts.b3_e4_f03_material_line_loaded import _scales, _solve
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent


def fixture_identity(xs, ys, bond_start, thickness, beam_offset, beam,
                     shell_fixed):
    """Bind a same-model comparison to geometry, native section and supports."""
    tangent = beam.tangent.tocsr()
    state = beam.context.initial.mechanical
    arrays = (
        tangent.data, tangent.indices, tangent.indptr, beam.residual,
        state.positions, state.position_low, state.nodal_frames,
        state.cell_rotations,
    )
    digest = sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return {
        "shell_x_m": np.asarray(xs).tolist(),
        "shell_y_m": np.asarray(ys).tolist(),
        "bond_x_m": np.asarray(xs)[bond_start:].tolist(),
        "shell_thickness_m": float(thickness),
        "beam_offset_local_m": np.asarray(beam_offset).tolist(),
        "beam_axis_reference_z_m": .0675,
        "native_operator_and_reference_sha256": digest.hexdigest(),
        "fixed_shell_dofs": sorted(int(index) for index in shell_fixed),
        "native_beam_root": "free",
        "tip_actions": [["tip_mx_100nm", 3, 100.],
                        ["tip_fz_1000n", 2, 1000.]],
    }


def evaluate() -> dict:
    expected_package = (Path(__file__).resolve().parents[1]
                        / "src" / "anyfesolver").resolve()
    actual_package = Path(anyfesolver.__file__).resolve().parent
    if actual_package != expected_package:
        raise RuntimeError("loaded screen imported another ANYfeSolver checkout")
    xs = np.linspace(0., 1., 9)
    ys = np.array((-.1, 0., .1))
    bond_start = 4
    bond_x = xs[bond_start:]
    bounds = (-.05, .05)
    thickness = .035
    beam_offset = np.array((0., 0., -.05))
    shell_total, shell_physical = raw_e4_tangent(
        xs, ys, thickness=thickness)
    beam = build(8, z_offset=.0675)
    if beam.tangent is None or beam.residual is None:
        raise ArithmeticError("assembled native retained beam required")
    state = beam.context.initial.mechanical

    def port_at(station):
        element, cell, weight, _ = beam._locate(station)
        return NativeHalfCellSpatialPort(beam.context, element, cell, weight)

    reference = np.array([[(x, y, 0.) for y in ys] for x in bond_x])
    line = SegmentedRailWeakWork.at_reference(
        reference, ys, bounds, thickness, port_at, state,
        beam_offset=tuple(beam_offset),
        expected_reference_offset_shell=(0., 0., 0.))
    shell_size = shell_total.shape[0]
    beam_size = beam.context.count
    local_shell = flat_rail_p0_shell_pose_action(
        bond_x, ys, bounds, thickness).reshape(24, -1)
    shell_pose = np.zeros((24, shell_size))
    for j in range(len(ys)):
        for i in range(len(bond_x)):
            source = 6*(j*len(bond_x)+i)
            target = 6*(j*len(xs)+bond_start+i)
            shell_pose[:, target:target+6] += local_shell[:, source:source+6]
    native_pose = np.zeros((24, beam_size))
    for sample, connection in enumerate(line.connections):
        segment = sample//2
        jacobian = connection.port.tangent(state)
        _, _, frame = connection.port.pose_pair(state)
        arm = frame@beam_offset
        point = jacobian[:3]+np.cross(jacobian[3:].T, arm).T
        native_pose[6*segment:6*segment+6] += (
            line.weights[sample]*np.vstack((point, jacobian[3:])))
    constraint = np.hstack((-shell_pose, native_pose))

    # Check the assembled rows against the actual local relative-pose law.
    rng = np.random.default_rng(419)
    shell_direction = rng.normal(scale=.01, size=(len(ys), len(xs), 6))
    beam_direction = rng.normal(scale=.01, size=beam_size)
    integrated = np.zeros((len(bond_x)-1, 6))
    for sample, connection in enumerate(line.connections):
        segment = sample//2
        local = shell_direction[:, bond_start+segment:
                                bond_start+segment+2, :].transpose(1, 0, 2)
        _, directional_gap = connection.evaluate(
            np.zeros_like(local), state, shell_direction=local,
            beam_direction=beam_direction)
        integrated[segment] += line.weights[sample]*directional_gap
    map_error = float(np.max(np.abs(
        constraint@np.r_[shell_direction.ravel(), beam_direction]-
        integrated.ravel())))
    if map_error > 2e-12:
        raise ArithmeticError("physical-width rail assembly work map changed")

    shell_fixed = np.array([6*(j*len(xs))+axis
                            for j in range(len(ys)) for axis in range(6)])
    # The native beam root stays free; all applied work must reach the plate.
    base_fixed = set(shell_fixed.tolist())
    fixture = fixture_identity(
        xs, ys, bond_start, thickness, beam_offset, beam, base_fixed)
    terminal = shell_size+6*beam.n_half_cells
    shell_nodes = np.array([(x, y, 0.) for y in ys for x in xs])
    cases = {}
    for shell_law, shell_k, remove_drilling in (
            ("total", shell_total, False),
            ("physical_core", shell_physical, True)):
        primal = sparse.block_diag((sparse.csr_matrix(shell_k),
                                    beam.tangent), format="csr")
        fixed = set(base_fixed)
        if remove_drilling:
            fixed.update(range(5, shell_size, 6))
        free = np.array(sorted(set(range(primal.shape[0]))-fixed), dtype=int)
        scales, energy_scale = _scales(
            shell_size, beam, free, row_count=24)
        residual = np.r_[np.zeros(shell_size), beam.residual]
        for label, axis, magnitude in (("tip_mx_100nm", 3, 100.),
                                       ("tip_fz_1000n", 2, 1000.)):
            load = np.zeros(primal.shape[0])
            load[terminal+axis] = magnitude
            solution, equation, diagnostics = _solve(
                primal, residual, constraint, free, scales,
                energy_scale, load)
            shell_step = solution[:shell_size]
            support = equation[:shell_size].reshape(-1, 6)
            support_force = support[:, :3].sum(axis=0)
            support_moment = np.sum(
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
            balance = np.r_[support_force+applied_force,
                            support_moment+applied_moment]
            # q^T K q is twice the conventional linear elastic strain energy.
            physical_quadratic = float(
                shell_step@shell_physical@shell_step)
            numerical_quadratic = float(
                shell_step@(shell_total-shell_physical)@shell_step)
            if (physical_quadratic <= 0. or
                    max(np.max(np.abs(balance[:3]))/1000.,
                        np.max(np.abs(balance[3:]))/100.) > 1e-7):
                raise ArithmeticError("loaded rail lacks physical support balance")
            cases[f"{shell_law}/{label}"] = {
                "diagnostics": diagnostics,
                "tip_uz_m": float(solution[terminal+2]),
                "tip_rx_rad": float(solution[terminal+3]),
                "shell_physical_quadratic_j": physical_quadratic,
                "shell_numerical_quadratic_j": numerical_quadratic,
                "support_force_n": support_force.tolist(),
                "support_moment_nm": support_moment.tolist(),
                "global_balance": balance.tolist(),
            }
    return {
        "classification": "bounded physical-width rail loaded development screen; no G0B acceptance",
        "shell_elements": 8*2,
        "native_b3_halfcells": 8,
        "bond_segments": 4,
        "multiplier_rows": 24,
        "fixture_identity": fixture,
        "physical_strip_m": bounds,
        "shell_thickness_m": thickness,
        "beam_axis_z_m": .0675,
        "native_beam_root": "free",
        "shell_x0_edge": "fixed",
        "imported_anyfesolver": str(actual_package),
        "integrated_reference_map_error": map_error,
        "cases": cases,
        "limitations": [
            "flat virgin linearization and one fixed P0 interface space",
            "no same-law external reference or mesh convergence",
            "no nonlinear, plastic, curved or finite-motion acceptance",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "status": "development_screen_complete",
        "cases": {name: {
            "tip_uz_m": case["tip_uz_m"],
            "tip_rx_rad": case["tip_rx_rad"],
            "shell_physical_quadratic_j": case["shell_physical_quadratic_j"],
            "shell_numerical_quadratic_j": case["shell_numerical_quadratic_j"],
            "global_balance": case["global_balance"],
        } for name, case in result["cases"].items()},
    }))
