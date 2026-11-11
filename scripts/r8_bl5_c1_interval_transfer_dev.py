"""Compare saved C1 native/solid P0 interface transfer on unchanged bodies.

This is a one-grid development diagnostic. It re-solves the exact bounded
source KKT systems because their prior JSON results did not retain q or lambda.
It does not assess F03 stability or qualify a continuous connection law.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import anyfesolver
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from scripts._b3_e4_f03_native_fixture import build
from scripts.b3_e4_f03_c1_fy_supported_native import _wrench as native_wrench
from scripts.b3_e4_f03_material_line_loaded import _scales, _solve
from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
from scripts.b3_e4_f03_square_section_control import square_torsion_constant
from scripts.r8_bl5_c1_line_kkt_dev import _assemble, _read, _wrench as solid_wrench
from scripts.r8_bl5_c1_line_reference_dev import _five_channel_rows
from scripts.r8_bl5_c1_offratio_native_dev import _rows as native_rows, _samples
from scripts.r8_bl5_c1_solid_p0_counterfactual_dev import _matrix, _p0_rows


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / ".development/bl5-c1-line-reference-20261008-v7/prepared"
NATIVE_SAVED = ROOT / ".development/bl5-c1-native-fivechannel-20261008-v1/result.json"
SOLID_SAVED = ROOT / ".development/bl5-c1-solid-p0-20261008-v2/result.json"
INTERVAL_M = .025


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _near(actual: float, expected: float, label: str) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > 2e-9 * max(abs(expected), 1e-12):
        raise ArithmeticError(f"saved {label} work identity changed: {actual} versus {expected}")


def _native() -> dict:
    saved = json.loads(NATIVE_SAVED.read_text(encoding="utf-8"))
    if saved["status"] != "DEVELOPMENT_SAME_BODY_ONLY_NOT_QUALIFIED":
        raise ValueError("saved native C1 identity changed")
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
    rows = native_rows(samples, primal.shape[0], "p0")
    if rows.shape != (20, primal.shape[0]) or not np.allclose(np.diff(cuts), INTERVAL_M):
        raise ValueError("native C1 half-cell inventory changed")
    shell_fixed = {6 * (j * len(xs) + i) + axis
                   for j in range(len(ys)) for i, x in enumerate(xs)
                   if abs(x + .5) < 1e-12 for axis in range(6)}
    beam_fixed = {shell_size + int(index) for index in beam.context.fixed}
    fixed = np.array(sorted(shell_fixed | beam_fixed), dtype=int)
    free = np.setdiff1d(np.arange(primal.shape[0]), fixed)
    scales, energy_scale = _scales(shell_size, beam, free, row_count=20)
    shell_positions = np.array([(x, y, 0.) for y in ys for x in xs])
    terminal = shell_size + 6 * beam.n_half_cells
    cases = {}
    for name, axis in (("fy", 1), ("mx", 3)):
        load = np.zeros(primal.shape[0])
        load[terminal + axis] = 100.
        q, equation, checks = _solve(primal, residual, rows, free, scales,
                                     energy_scale, load)
        right = (load - residual - primal @ q)[free]
        lam, _, rank, _ = np.linalg.lstsq(rows[:, free].T, right, rcond=1e-10)
        if rank != 20 or np.linalg.norm(rows[:, free].T @ lam - right) > 1e-6:
            raise ArithmeticError("native physical multiplier recovery changed")
        _near(float(load @ q), saved["cases"]["p0"]["cases"][name]["applied_work_J"],
              f"native {name}")
        intervals = []
        for half in range(4):
            force = -(rows[5 * half:5 * (half + 1)].T @ lam[5 * half:5 * (half + 1)])
            shell_force = force[:shell_size]
            shell_work = float(q[:shell_size] @ shell_force)
            beam_work = float(q[shell_size:] @ force[shell_size:])
            if abs(shell_work + beam_work) > 2e-8 * max(abs(float(load @ q)), 1e-12):
                raise ArithmeticError("native interval dual work does not cancel")
            intervals.append({
                "x_m": [float(cuts[half]), float(cuts[half + 1])],
                "shell_exerted_wrench": native_wrench(
                    shell_force.reshape(-1, 6), shell_positions).tolist(),
                "shell_work_J": shell_work,
                "beam_work_J": beam_work,
                # Native rows integrate the gap; these are physical interval
                # resultants only in the declared flat-reference row frame.
                "integrated_dual": (INTERVAL_M * lam[5 * half:5 * (half + 1)]).tolist(),
            })
        summed = np.sum([r["shell_exerted_wrench"] for r in intervals], axis=0)
        expected = np.asarray(saved["cases"]["p0"]["cases"][name]["shell_interface_wrench"])
        if np.max(np.abs(summed - expected)) > 1e-6:
            raise ArithmeticError("native interval wrench does not reconstruct saved result")
        cases[name] = {"applied_work_J": float(load @ q), "intervals": intervals,
                       "total_shell_wrench": summed.tolist(),
                       "free_equilibrium": checks["free_residual_ratio"],
                       "weak_gap": checks["constraint_max"],
                       "root_support_norm": float(np.linalg.norm(equation[fixed]))}
    return {"source_sha256": _sha(Path(__file__)), "saved_sha256": _sha(NATIVE_SAVED),
            "physical_shell_dofs": shell_size, "retained_and_beam_dofs": beam.context.count,
            "cases": cases}


def _solid() -> dict:
    saved = json.loads(SOLID_SAVED.read_text(encoding="utf-8"))
    metadata_path = PREPARED / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (saved["status"] != "SAME_SOLID_BODY_ENFORCEMENT_DIAGNOSTIC_NOT_QUALIFIED"
            or saved["prepared_metadata_sha256"] != _sha(metadata_path)):
        raise ValueError("saved solid C1 identity changed")
    models = {name: _read(PREPARED / f"{name}.inp") for name in ("fy", "mx")}
    for name, model in models.items():
        if _sha(PREPARED / f"{name}.inp") != metadata["cases"][name]["deck_sha256"]:
            raise ValueError("saved solid C1 deck changed")
    first, second = models["fy"], models["mx"]
    if (first["nodes"].keys() != second["nodes"].keys()
            or first["elements"] != second["elements"]
            or first["domains"] != second["domains"]
            or first["supports"] != second["supports"]
            or any(np.max(np.abs(first["nodes"][n] - second["nodes"][n])) > 0.
                   for n in first["nodes"])):
        raise ValueError("solid C1 bodies or roots differ")
    k, order, index, assembly = _assemble(first)
    size = k.shape[0]
    strong = _five_channel_rows(first["nodes"], first["domains"]["Plate"],
                                first["domains"]["Stiffener"])
    c = _matrix(_p0_rows(strong), index, size)
    if c.shape != (20, size):
        raise ValueError("solid P0 row inventory changed")
    fixed = np.array([index[n] + j for n in sorted(first["supports"]) for j in range(3)])
    free = np.setdiff1d(np.arange(size), fixed)
    if len(fixed) != 726:
        raise ValueError("saved solid support inventory changed")
    positions = np.array([first["nodes"][n] for n in order])
    plate_nodes = first["domains"]["Plate"]
    plate_mask = np.array([n in plate_nodes for n in order])
    plate_dofs = np.repeat(plate_mask, 3)
    loads = np.zeros((size, 2))
    for j, name in enumerate(("fy", "mx")):
        for (node, axis), value in models[name]["loads"].items():
            loads[index[node] + axis - 1, j] += value
    cf = c[:, free]
    diagonal = k[free, :][:, free].diagonal()
    if np.min(diagonal) <= 0. or not np.isfinite(diagonal).all():
        raise ArithmeticError("invalid solid physical diagonal")
    dq = 1. / np.sqrt(diagonal)
    dl = 1. / np.sqrt(np.asarray((cf @ sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    if not np.isfinite(dl).all():
        raise ArithmeticError("invalid solid dual scaling")
    dp, dm = sparse.diags(dq), sparse.diags(dl)
    system = sparse.bmat([[dp @ k[free, :][:, free] @ dp, dp @ cf.T @ dm],
                          [dm @ cf @ dp, None]], format="csc")
    rhs = np.vstack((dq[:, None] * loads[free, :], np.zeros((20, 2))))
    factor = splu(system)
    solution = factor.solve(rhs)
    for _ in range(3):
        solution += factor.solve(rhs - system @ solution)
    q_all = np.zeros_like(loads)
    q_all[free, :] = dq[:, None] * solution[:len(free), :]
    lam_all = dl[:, None] * solution[len(free):, :]
    cases = {}
    for j, name in enumerate(("fy", "mx")):
        q, load, lam = q_all[:, j], loads[:, j], lam_all[:, j]
        _near(float(load @ q), saved["cases"][name]["solid_p0_work_J"], f"solid {name}")
        residual = np.asarray(k @ q + c.T @ lam - load).ravel()
        if np.max(np.abs(residual[free])) / 100. > 1e-8 or np.max(np.abs(c @ q)) > 1e-9:
            raise ArithmeticError("solid source KKT controls changed")
        intervals = []
        for half in range(4):
            force = -np.asarray(c[5 * half:5 * (half + 1)].T
                                @ lam[5 * half:5 * (half + 1)]).ravel()
            plate_force = force.copy()
            plate_force[~plate_dofs] = 0.
            bar_force = force - plate_force
            shell_work = float(q @ plate_force)
            beam_work = float(q @ bar_force)
            if abs(shell_work + beam_work) > 2e-8 * max(abs(float(load @ q)), 1e-12):
                raise ArithmeticError("solid interval dual work does not cancel")
            w_plate = solid_wrench(positions, plate_force)
            w_bar = solid_wrench(positions, bar_force)
            if np.max(np.abs(w_plate + w_bar)) > 1e-8 * 100.:
                raise ArithmeticError("solid interval physical action-reaction failed")
            intervals.append({
                "x_m": [.9 + half * INTERVAL_M, .9 + (half + 1) * INTERVAL_M],
                "shell_exerted_wrench": w_plate.tolist(),
                "beam_exerted_wrench": w_bar.tolist(),
                "shell_work_J": shell_work,
                "beam_work_J": beam_work,
                # Solid rows are mean gaps; lambda is already an interval dual.
                "integrated_dual": lam[5 * half:5 * (half + 1)].tolist(),
            })
        cases[name] = {"applied_work_J": float(load @ q), "intervals": intervals,
                       "total_shell_wrench": np.sum(
                           [r["shell_exerted_wrench"] for r in intervals], axis=0).tolist(),
                       "free_equilibrium": float(np.max(np.abs(residual[free])) / 100.),
                       "weak_gap": float(np.max(np.abs(c @ q)))}
    return {"saved_sha256": _sha(SOLID_SAVED),
            "prepared_metadata_sha256": _sha(metadata_path),
            "physical_dofs": size, "assembly": assembly, "cases": cases}


def evaluate() -> dict:
    if Path(anyfesolver.__file__).resolve().parent != (ROOT / "src" / "anyfesolver").resolve():
        raise RuntimeError("wrong ANYfeSolver source checkout")
    native, solid = _native(), _solid()
    comparison = {}
    for name in ("fy", "mx"):
        rows = []
        for n, s in zip(native["cases"][name]["intervals"],
                        solid["cases"][name]["intervals"], strict=True):
            if not np.allclose(n["x_m"], s["x_m"], rtol=0., atol=2e-14):
                raise ArithmeticError("native/solid material intervals differ")
            rows.append({"x_m": n["x_m"],
                         "native_shell_wrench": n["shell_exerted_wrench"],
                         "solid_shell_wrench": s["shell_exerted_wrench"],
                         "native_shell_work_J": n["shell_work_J"],
                         "solid_shell_work_J": s["shell_work_J"]})
        comparison[name] = rows
    return {"identity": "R8_BL5_C1_DUAL_SCALED_INTERVAL_TRANSFER_DEV_V1",
            "status": "DEVELOPMENT_LOCAL_TRANSFER_NOT_QUALIFIED",
            "script_sha256": _sha(Path(__file__)),
            "native": native, "solid": solid, "comparison": comparison,
            "limitation": "one C1 grid; distinct native/solid body, root, end-face warping and trace; no BL5 accuracy, multiplier-space selection or formal F03 assessment"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous C1 transfer result")
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(args.output)}))


if __name__ == "__main__":
    main()
