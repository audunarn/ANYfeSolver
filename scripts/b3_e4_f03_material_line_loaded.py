"""Bounded same-primal loaded screen for two distinct C1 connection laws.

This is a private G0 development comparison, not an F03 qualification run.
The constant six-row multiplier is provisional. Both laws use the same
Q4 E4 and native retained B3 tangent, supports, loads and scaling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import warnings

import anyfesolver
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from scripts._b3_e4_f03_native_fixture import SECTION_DIAGONAL, build
from scripts.b3_e4_f03_distributed_surface_finite import FiniteSurfaceLineHybrid
from scripts.b3_e4_f03_material_line_position import MaterialLinePosition
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


def material_line_mean_rows(line: MaterialLinePosition) -> np.ndarray:
    """Integrate its six continuous gap derivatives into provisional P0 rows."""
    position = line.reference_jacobian().reshape(len(line.stations), 3, -1)
    orientation = line.reference_orientation_jacobian().reshape(
        len(line.stations), 3, -1)
    weights = np.array([station.weight for station in line.stations])
    if abs(sum(weights)-.1) > 1e-12:
        raise ArithmeticError("material-line reference measure changed")
    rows = np.vstack((np.einsum("k,kij->ij", weights, position)/.1,
                      np.einsum("k,kij->ij", weights, orientation)/.1))
    if rows.shape != (6, position.shape[2]) or not np.isfinite(rows).all():
        raise ArithmeticError("finite provisional material-line rows required")
    return rows


def _scales(shell_size: int, beam, free: np.ndarray,
            row_count: int = 6) -> tuple[np.ndarray, float]:
    if type(row_count) is not int or not 1 <= row_count <= 60:
        raise ValueError("bounded loaded line row count required")
    force_scale = 1e6
    displacement_scale = force_scale/(210e9*.035)
    nodal_scale = np.full(shell_size+beam.context.nodal_count,
                          displacement_scale)
    resultants = np.r_[np.tile(SECTION_DIAGONAL[:3]*displacement_scale, 2),
                       np.tile(SECTION_DIAGONAL[3:]*displacement_scale, 4)]
    retained = np.tile(np.r_[np.full(6, displacement_scale), resultants],
                       len(beam.context.elements))
    primal = np.r_[nodal_scale, retained]
    if len(primal) != shell_size+beam.context.count:
        raise ArithmeticError("native retained scaling layout changed")
    return np.r_[primal[free], np.full(row_count, force_scale)], (
        force_scale*displacement_scale)


def _solve(primal: sparse.csr_matrix, residual: np.ndarray,
           constraint: np.ndarray, free: np.ndarray,
           scales: np.ndarray, energy_scale: float,
           load: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    row_count = constraint.shape[0]
    # This bounded diagnostic now also covers the 45-row physical shell
    # trace image. The complete system remains below the registered 1,900-row
    # development cap; this does not enlarge that cap or production scope.
    if (row_count < 1 or row_count > 60 or
            len(free)+row_count > 1900 or
            scales.shape != (len(free)+row_count,)):
        raise ValueError("bounded constraint and scaling rows required")
    rows = sparse.csr_matrix(constraint[:, free])
    system = sparse.bmat(((primal[free, :][:, free], rows.T),
                          (rows, None)), format="csr")
    scaled = (sparse.diags(scales)@system@sparse.diags(scales)
              /energy_scale).tocsc()
    rhs = np.r_[(load-residual)[free]*scales[:len(free)]/energy_scale,
                np.zeros(row_count)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        answer = spsolve(scaled, rhs)
    if not np.isfinite(answer).all():
        raise ArithmeticError("bounded material-line KKT factorization failed")
    state = np.zeros(primal.shape[0])
    state[free] = scales[:len(free)]*answer[:len(free)]
    multipliers = scales[len(free):]*answer[len(free):]
    equation = residual+primal@state+constraint.T@multipliers-load
    gap = constraint@state
    effective_load = load-residual
    work_error = abs(state@(primal@state)-effective_load@state)
    diagnostics = {
        "dimension": int(scaled.shape[0]),
        "nnz": int(scaled.nnz),
        "row_rank": int(np.linalg.matrix_rank(constraint[:, free])),
        "free_residual_ratio": float(np.linalg.norm(equation[free]) /
                                     max(1., np.linalg.norm(load[free]))),
        "constraint_max": float(np.max(np.abs(gap))),
        "work_absolute_error_j": float(work_error),
        "work_relative_error": float(work_error/max(
            1e-12, abs(effective_load@state))),
    }
    if (diagnostics["row_rank"] != row_count or
            max(diagnostics["free_residual_ratio"],
                diagnostics["constraint_max"],
                diagnostics["work_relative_error"]) > 1e-7):
        raise ArithmeticError(
            f"bounded loaded material-line KKT failed: {diagnostics}")
    return state, equation, diagnostics


def run() -> dict:
    expected_package = (Path(__file__).resolve().parents[1]
                        /"src"/"anyfesolver").resolve()
    actual_package = Path(anyfesolver.__file__).resolve().parent
    if actual_package != expected_package:
        raise RuntimeError("bounded loaded screen imported another ANYfeSolver checkout")
    xs, ys = local_grid(2, transverse_cells=4)
    beam = build(12, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    shell_size = 6*len(xs)*len(ys)
    shell_tangent, shell_physical = raw_e4_tangent(xs, ys, thickness=.035)
    if beam.tangent is None or beam.residual is None:
        raise ArithmeticError("native assembled fixture required")
    primal = sparse.block_diag((sparse.csr_matrix(shell_tangent),
                                beam.tangent), format="csr")
    residual = np.r_[np.zeros(shell_size), beam.residual]
    material = MaterialLinePosition(xs, ys, beam, mixed_s3=False)
    strip = FiniteSurfaceLineHybrid(xs, ys, beam, thickness=.035)
    constraints = {"material_line": material_line_mean_rows(material),
                   "reduced_strip_pose": strip.reference_trace()}
    if any(rows.shape != (6, primal.shape[0]) for rows in
           constraints.values()):
        raise ArithmeticError("same-primal six-row comparison required")
    fixed_shell = {6*(j*len(xs)+i)+axis
                   for j in range(len(ys)) for i, x in enumerate(xs)
                   if abs(x+.5) < 1e-12 for axis in range(6)}
    fixed_beam = {shell_size+int(index) for index in beam.context.fixed}
    fixed = np.array(sorted(fixed_shell | fixed_beam), dtype=int)
    free = np.setdiff1d(np.arange(primal.shape[0]), fixed)
    scales, energy_scale = _scales(shell_size, beam, free)
    station_position = material.reference_jacobian().reshape(
        len(material.stations), 3, -1)
    station_orientation = material.reference_orientation_jacobian().reshape(
        len(material.stations), 3, -1)
    station_weights = np.array([station.weight for station in material.stations])
    shell_reference = material.reference_nodes
    beam_reference = (beam.context.initial.mechanical.positions
                      +beam.context.initial.mechanical.position_low)
    terminal = shell_size+6*beam.n_half_cells
    results = {}
    for law, constraint in constraints.items():
        outputs = {}
        for label, axis, magnitude in (("tip_mx_100nm", 3, 100.),
                                       ("tip_fz_1000n", 2, 1000.)):
            load = np.zeros(primal.shape[0])
            load[terminal+axis] = magnitude
            state, equation, diagnostics = _solve(
                primal, residual, constraint, free, scales,
                energy_scale, load)
            shell_step = state[:shell_size].reshape(-1, 6)
            beam_step = state[shell_size:]
            position_gap = np.einsum("kij,j->ki", station_position, state)
            orientation_gap = np.einsum("kij,j->ki", station_orientation, state)
            shell_support = equation[:shell_size].reshape(-1, 6)
            beam_support = equation[shell_size:shell_size+
                                    beam.context.nodal_count].reshape(-1, 6)
            force = (shell_support[:, :3].sum(axis=0)
                     +beam_support[:, :3].sum(axis=0))
            moment = (np.sum(np.cross(shell_reference,
                                      shell_support[:, :3])
                             +shell_support[:, 3:], axis=0)
                      +np.sum(np.cross(beam_reference,
                                      beam_support[:, :3])
                             +beam_support[:, 3:], axis=0))
            applied_force = np.zeros(3)
            applied_moment = np.zeros(3)
            if axis < 3:
                applied_force[axis] = magnitude
                applied_moment = np.cross(beam_reference[-1], applied_force)
            else:
                applied_moment[axis-3] = magnitude
            diagnostics.update({
                "support_force_balance_n": float(np.linalg.norm(
                    force+applied_force)),
                "support_moment_balance_nm": float(np.linalg.norm(
                    moment+applied_moment)),
                "tip_translation_m": beam_step[6*beam.n_half_cells:
                                                6*beam.n_half_cells+3].tolist(),
                "tip_rotation_rad": beam_step[6*beam.n_half_cells+3:
                                              6*beam.n_half_cells+6].tolist(),
                "shell_physical_work_j": float(
                    state[:shell_size]@shell_physical@state[:shell_size]),
                "material_line_position_rms_m": float(np.sqrt(
                    np.sum(station_weights*np.sum(position_gap**2, axis=1))/.1)),
                "material_line_orientation_rms_rad": float(np.sqrt(
                    np.sum(station_weights*np.sum(orientation_gap**2,
                                                   axis=1))/.1)),
            })
            if (diagnostics["support_force_balance_n"] > 1e-5 or
                    diagnostics["support_moment_balance_nm"] > 1e-5 or
                    diagnostics["shell_physical_work_j"] <= 0.):
                raise ArithmeticError("bounded loaded balance or physical work failed")
            outputs[label] = diagnostics
        results[law] = outputs
    return {
        "classification": "private same-primal provisional-P0 loaded discriminator",
        "scope": "Q4 35 mm C1 E4/native retained generalized B3, two loads",
        "qualification": "development only; F03 and G0B remain open",
        "anyfesolver_import_path": str(Path(anyfesolver.__file__).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS", "NUMBA_DISABLE_JIT")},
        "shell_nodes": int(len(shell_reference)),
        "beam_half_cells": int(beam.n_half_cells),
        "material_line_station_count": len(material.stations),
        "result": results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
