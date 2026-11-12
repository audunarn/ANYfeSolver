"""Bounded same-body five-channel C1 off-ratio BL5 development comparison.

The x8/B3-14 generalized body, supports and end wrenches remain fixed while
the multiplier space changes from four P0 cells to five continuous P1 knots.
This is neither a multiplier selection nor a formal F03 assessment.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import anyfesolver
import numpy as np
from scipy import sparse

from anyfesolver._stiffened_shell_bl5_local_experimental import BL5LocalOperator, LAW_ID
from anyfesolver._stiffened_shell_l1_local_experimental import NativeHalfCellPort
from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_c1_fy_supported_native import _wrench
from scripts.b3_e4_f03_material_line_loaded import _scales, _solve
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant


ROOT = Path(__file__).resolve().parents[1]
START, END = .9, 1.


def _samples(xs, ys, beam, shell_size):
    cuts = np.asarray(beam.abscissae)
    cuts = cuts[(cuts >= START - 1e-14) & (cuts <= END + 1e-14)]
    if (cuts.shape != (5,) or abs(cuts[0] - START) > 1e-14 or
            abs(cuts[-1] - END) > 1e-14 or np.min(np.diff(cuts)) <= 0.):
        raise ValueError("four native C1 terminal half-cells required")
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(3)
    samples = []
    for half, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
        for xi, wi in zip(gauss_x, gauss_w):
            x = float(.5 * (left + right) + .5 * (right - left) * xi)
            weight = float(.5 * (right - left) * wi)
            edge = min(int(np.searchsorted(xs, x, side="right")) - 1, len(xs) - 2)
            alpha = (x - xs[edge]) / (xs[edge + 1] - xs[edge])
            yzero = int(np.flatnonzero(np.isclose(ys, 0., atol=1e-14, rtol=0.))[0])
            nodes = (yzero * len(xs) + edge, yzero * len(xs) + edge + 1)
            shell_reference = np.array(((xs[edge], 0., 0.), (xs[edge + 1], 0., 0.)))
            element, cell, fraction, _ = beam._locate(x)
            reference = beam.context.elements[element][1].operator.reference
            port = NativeHalfCellPort.from_reference(
                reference, beam.context.nodes[element], element, cell, fraction,
                beam.context.nodal_count, beam.context.count)
            operator = BL5LocalOperator.at_reference(
                shell_reference, np.zeros((2, 6)), .035, port,
                beam.context.initial.mechanical, station=alpha)
            local = operator.evaluate(np.zeros((2, 6)),
                                      beam.context.initial.mechanical, np.zeros(5))
            if np.max(np.abs(local.gap)) > 2e-13:
                raise ArithmeticError("nonzero virgin five-channel gap")
            slots = np.r_[[6 * node + axis for node in nodes for axis in range(6)],
                          shell_size + local.beam_slots]
            if local.gap_jacobian.shape != (5, 21) or slots.shape != (21,):
                raise ArithmeticError("BL5 local coordinate contract changed")
            samples.append((half, x, weight, slots, local.gap_jacobian))
    if len(samples) != 12 or abs(sum(sample[2] for sample in samples) - .1) > 1e-14:
        raise ArithmeticError("C1 bonded quadrature inventory changed")
    return cuts, samples


def _rows(samples, size, basis):
    knots = 4 if basis == "p0" else 5
    rows = np.zeros((5 * knots, size))
    for half, x, weight, slots, jac in samples:
        if basis == "p0":
            terms = ((half, 1.),)
        else:
            alpha = (x - (START + .025 * half)) / .025
            terms = ((half, 1. - alpha), (half + 1, alpha))
        for knot, shape in terms:
            for channel in range(5):
                rows[5 * knot + channel, slots] += weight * shape * jac[channel]
    return rows


def evaluate():
    if Path(anyfesolver.__file__).resolve().parent != (ROOT / "src" / "anyfesolver").resolve():
        raise RuntimeError("wrong ANYfeSolver source checkout")
    xs, ys = local_grid(8, transverse_cells=4)
    shell, _ = raw_e4_tangent(xs, ys, thickness=.035)
    beam = build(14, z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile="c1_tangent_2um")
    if beam.tangent is None or beam.residual is None:
        raise ArithmeticError("assembled native C1 body required")
    shell_size = shell.shape[0]
    primal = sparse.block_diag((sparse.csr_matrix(shell), beam.tangent), format="csr")
    residual = np.r_[np.zeros(shell_size), beam.residual]
    cuts, samples = _samples(xs, ys, beam, shell_size)
    shell_fixed = {6 * (j * len(xs) + i) + axis
                   for j in range(len(ys)) for i, x in enumerate(xs)
                   if abs(x + .5) < 1e-12 for axis in range(6)}
    beam_fixed = {shell_size + int(index) for index in beam.context.fixed}
    fixed = np.array(sorted(shell_fixed | beam_fixed), dtype=int)
    free = np.setdiff1d(np.arange(primal.shape[0]), fixed)
    shell_positions = np.array([(x, y, 0.) for y in ys for x in xs])
    beam_positions = beam.context.initial.mechanical.positions
    terminal = shell_size + 6 * beam.n_half_cells
    results = {}
    for basis in ("p0", "continuous_p1"):
        constraint = _rows(samples, primal.shape[0], basis)
        scales, energy_scale = _scales(shell_size, beam, free, row_count=len(constraint))
        cases = {}
        for name, axis in (("fy", 1), ("mx", 3)):
            load = np.zeros(primal.shape[0])
            load[terminal + axis] = 100.
            state, equation, diagnostics = _solve(
                primal, residual, constraint, free, scales, energy_scale, load)
            right = (load - residual - primal @ state)[free]
            multipliers, _, rank, _ = np.linalg.lstsq(
                constraint[:, free].T, right, rcond=1e-10)
            if rank != len(constraint) or np.linalg.norm(
                    constraint[:, free].T @ multipliers - right) > 1e-6:
                raise ArithmeticError("physical BL5 multiplier recovery failed")
            local_gaps = np.array([jac @ state[slots] for _, _, _, slots, jac in samples])
            weights = np.array([sample[2] for sample in samples])
            interface = -(constraint.T @ multipliers)
            shell_support = _wrench(equation[:shell_size].reshape(-1, 6), shell_positions)
            beam_support = _wrench(
                equation[shell_size:shell_size + beam.context.nodal_count].reshape(-1, 6),
                beam_positions)
            applied = np.array((0., 100., 0., -6.75, 0., 100.)) if name == "fy" else np.array((0., 0., 0., 100., 0., 0.))
            balance = shell_support + beam_support + applied
            if np.max(np.abs(balance)) > 1e-5:
                raise ArithmeticError("supported physical wrench does not balance")
            cases[name] = {
                "diagnostics": diagnostics,
                "applied_work_J": float(load @ state),
                "tip_pose_increment": state[terminal:terminal + 6].tolist(),
                "shell_support_wrench": shell_support.tolist(),
                "beam_root_support_wrench": beam_support.tolist(),
                "shell_interface_wrench": _wrench(
                    interface[:shell_size].reshape(-1, 6), shell_positions).tolist(),
                "wrench_imbalance": balance.tolist(),
                "gap_rms_by_channel": np.sqrt(
                    np.sum(weights[:, None] * local_gaps**2, axis=0) / .1).tolist(),
                "gap_max_by_channel": np.max(np.abs(local_gaps), axis=0).tolist(),
                "sample_x_m": [sample[1] for sample in samples],
                "sample_weight_m": weights.tolist(),
                "sample_gap": local_gaps.tolist(),
            }
        results[basis] = {"row_count": len(constraint), "cases": cases}
    return {
        "identity": "R8_BL5_C1_OFFRATIO_NATIVE_FIVE_CHANNEL_DEV_V1",
        "status": "DEVELOPMENT_SAME_BODY_ONLY_NOT_QUALIFIED",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "law": LAW_ID, "shell_cells_x": 8, "beam_halfcells": 14,
        "bond_cuts_m": cuts.tolist(), "physical_dofs": primal.shape[0],
        "fixed_dofs": len(fixed), "body_nnz": primal.nnz,
        "cases": results,
        "limitation": "same native x8/B3-14 body and five-channel law; one grid, two multiplier spaces; no solid-body accuracy, stability, convergence or F03 claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous native diagnostic")
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "fy_work": {key: value["cases"]["fy"]["applied_work_J"]
                                  for key, value in result["cases"].items()},
                      "mx_work": {key: value["cases"]["mx"]["applied_work_J"]
                                  for key, value in result["cases"].items()}}))


if __name__ == "__main__":
    main()
