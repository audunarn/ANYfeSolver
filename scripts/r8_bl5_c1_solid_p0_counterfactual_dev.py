"""Compare strong and exact half-cell P0 terminal rows on one saved C1 solid body.

The same C3D20R stiffness, root and Fy/Mx face loads are used for both row
spaces. This isolates finite-grid enforcement sensitivity, not physical truth.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info

from scripts.r8_bl5_c1_line_kkt_dev import _assemble, _read, _wrench
from scripts.r8_bl5_c1_line_reference_dev import _five_channel_rows


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / ".development/bl5-c1-line-reference-20261008-v7/prepared"
STRONG = ROOT / ".development/bl5-c1-kkt-20261008-v3/result.json"
NATIVE = ROOT / ".development/bl5-c1-native-fivechannel-20261008-v1/result.json"
HALF_WEIGHTS = ((5./12., 8./12., -1./12.),
                (-1./12., 8./12., 5./12.))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _matrix(rows: list[dict[tuple[int, int], float]], index: dict[int, int],
            size: int) -> sparse.csc_matrix:
    values, row_indices, column_indices = [], [], []
    for i, row in enumerate(rows):
        for (node, axis), value in row.items():
            values.append(value)
            row_indices.append(i)
            column_indices.append(index[node] + axis - 1)
    return sparse.coo_matrix((values, (row_indices, column_indices)),
                             shape=(len(rows), size)).tocsc()


def _p0_rows(strong: list[dict[tuple[int, int], float]]) -> list[dict[tuple[int, int], float]]:
    if len(strong) != 25:
        raise ValueError("five stations with five physical rows each required")
    rows = []
    for interval in range(4):
        parent = 2 * (interval // 2)
        weights = HALF_WEIGHTS[interval % 2]
        for channel in range(5):
            row: dict[tuple[int, int], float] = {}
            for station, weight in enumerate(weights):
                for slot, value in strong[5 * (parent + station) + channel].items():
                    row[slot] = row.get(slot, 0.) + weight * value
            rows.append({slot: value for slot, value in row.items() if abs(value) > 1e-15})
    return rows


def _quadrature_check() -> float:
    # C3D20 terminal geometry is straight; its sampled five-channel trace is
    # quadratic along each .05 m parent cell. These are exact half-cell means.
    gauss, weights = np.polynomial.legendre.leggauss(3)
    errors = []
    for half in range(2):
        computed = np.zeros(3)
        for point, weight in zip(gauss, weights):
            t = half + (1. + point) / 2.
            shape = ((t - 1.) * (t - 2.) / 2., t * (2. - t), t * (t - 1.) / 2.)
            computed += weight * np.asarray(shape) / 2.
        errors.append(float(np.max(np.abs(computed - HALF_WEIGHTS[half]))))
    return max(errors)


def _interior_max(station: np.ndarray) -> list[float]:
    gauss, _ = np.polynomial.legendre.leggauss(3)
    values = []
    for interval in range(4):
        parent = 2 * (interval // 2)
        half = interval % 2
        for point in gauss:
            t = half + (1. + point) / 2.
            shape = np.array(((t - 1.) * (t - 2.) / 2.,
                              t * (2. - t), t * (t - 1.) / 2.))
            values.append(shape @ station[parent:parent + 3])
    return np.max(np.abs(values), axis=0).tolist()


def evaluate() -> dict:
    if Path(anyfesolver.__file__).resolve().parent != (ROOT / "src" / "anyfesolver").resolve():
        raise RuntimeError("wrong ANYfeSolver checkout")
    meta_path = PREPARED / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    strong_saved = json.loads(STRONG.read_text(encoding="utf-8"))
    native_saved = json.loads(NATIVE.read_text(encoding="utf-8"))
    if (metadata["identity"] != "R8_BL5_C1_DETACHED_F2_FIVE_CHANNEL_DEV_V7"
            or strong_saved["status"] != "INDEPENDENT_KKT_ONLY_NOT_QUALIFIED"
            or native_saved["status"] != "DEVELOPMENT_SAME_BODY_ONLY_NOT_QUALIFIED"
            or strong_saved["prepared_metadata_sha256"] != _sha(meta_path)):
        raise ValueError("exact saved C1 development identities required")
    if _quadrature_check() > 1e-14:
        raise ArithmeticError("half-cell integration weights are not exact")

    models = {name: _read(PREPARED / f"{name}.inp") for name in ("fy", "mx")}
    for name in models:
        if _sha(PREPARED / f"{name}.inp") != metadata["cases"][name]["deck_sha256"]:
            raise ValueError("saved C1 deck changed")
    first, second = models["fy"], models["mx"]
    if (first["nodes"].keys() != second["nodes"].keys()
            or first["elements"] != second["elements"]
            or first["supports"] != second["supports"]
            or any(np.max(np.abs(first["nodes"][n] - second["nodes"][n])) > 0.
                   for n in first["nodes"])):
        raise ValueError("Fy/Mx solid bodies or roots differ")
    k, order, index, assembly = _assemble(first)
    size = k.shape[0]
    strong_rows = _five_channel_rows(first["nodes"], first["domains"]["Plate"],
                                     first["domains"]["Stiffener"])
    c_strong = _matrix(strong_rows, index, size)
    c = _matrix(_p0_rows(strong_rows), index, size)
    if c.shape[0] != 20 or c_strong.shape[0] != 25:
        raise ArithmeticError("C1 row inventories changed")
    active = np.unique(c.tocoo().col)
    if np.linalg.matrix_rank(c[:, active].toarray(), tol=1e-11) != 20:
        raise ArithmeticError("half-cell P0 row rank deficient")
    fixed = np.array([index[n] + j for n in sorted(first["supports"]) for j in range(3)])
    if len(fixed) != 726:
        raise ValueError("saved solid root changed")
    free = np.setdiff1d(np.arange(size), fixed)
    positions = np.array([first["nodes"][n] for n in order])
    rigid = []
    for axis in range(6):
        field = np.tile(np.eye(3)[axis], (len(order), 1)) if axis < 3 else np.cross(
            np.eye(3)[axis - 3], positions)
        rigid.append(float(np.max(np.abs(c @ field.ravel()))))
    if max(rigid) > 1e-11:
        raise ArithmeticError("P0 rows violate common rigid motion")

    loads = np.zeros((size, 2))
    for j, name in enumerate(("fy", "mx")):
        for (node, axis), value in models[name]["loads"].items():
            loads[index[node] + axis - 1, j] += value
    kff, cf = k[free, :][:, free], c[:, free]
    diagonal = kff.diagonal()
    if np.min(diagonal) <= 0. or not np.isfinite(diagonal).all():
        raise ArithmeticError("invalid C1 physical diagonal")
    dq = 1. / np.sqrt(diagonal)
    dl = 1. / np.sqrt(np.asarray((cf @ sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    if not np.isfinite(dl).all():
        raise ArithmeticError("invalid P0 dual scaling")
    dp, dm = sparse.diags(dq), sparse.diags(dl)
    system = sparse.bmat([[dp @ kff @ dp, dp @ cf.T @ dm],
                          [dm @ cf @ dp, None]], format="csc")
    rhs = np.vstack((dq[:, None] * loads[free, :], np.zeros((20, 2))))
    factor = splu(system)
    solution = factor.solve(rhs)
    for _ in range(3):
        solution += factor.solve(rhs - system @ solution)
    displacement = np.zeros_like(loads)
    displacement[free, :] = dq[:, None] * solution[:len(free), :]
    multiplier = dl[:, None] * solution[len(free):, :]

    cases = {}
    for j, name in enumerate(("fy", "mx")):
        q, load, lam = displacement[:, j], loads[:, j], multiplier[:, j]
        internal = np.asarray(k @ q).ravel()
        interface = np.asarray(c.T @ lam).ravel()
        residual = internal + interface - load
        support = np.zeros(size)
        support[fixed] = residual[fixed]
        work = float(load @ q)
        weak_gap = np.asarray(c @ q).ravel()
        station_gap = np.asarray(c_strong @ q).reshape(5, 5)
        balance = _wrench(positions, load + support)
        checks = {
            "free_equilibrium": float(np.max(np.abs(residual[free])) / 100.),
            "weak_gap_m": float(np.max(np.abs(weak_gap))),
            "energy_work": abs(float(q @ internal) - work) / max(abs(work), 1e-12),
            "constraint_work": abs(float(q @ interface)) / max(abs(work), 1e-12),
            "force_balance": float(np.max(np.abs(balance[:3])) / 100.),
            "moment_balance": float(np.max(np.abs(balance[3:])) / 100.),
        }
        if any(not np.isfinite(value) or value > (1e-9 if key == "weak_gap_m" else 1e-8)
               for key, value in checks.items()):
            raise ArithmeticError(f"{name} P0 KKT control failed: {checks}")
        strong_work = float(strong_saved["cases"][name]["applied_work_J"])
        native_work = float(native_saved["cases"]["p0"]["cases"][name]["applied_work_J"])
        cases[name] = {
            "solid_p0_work_J": work,
            "solid_strong_work_J": strong_work,
            "native_p0_work_J": native_work,
            "solid_p0_over_strong": work / strong_work,
            "native_p0_over_solid_p0": native_work / work,
            "strong_station_max_by_channel": np.max(np.abs(station_gap), axis=0).tolist(),
            "interior_gauss_max_by_channel": _interior_max(station_gap),
            "weak_gap_max_m": checks["weak_gap_m"],
            "checks": checks,
        }
    cross = loads.T @ displacement
    reciprocity = abs(float(cross[0, 1] - cross[1, 0])) / max(float(np.linalg.norm(cross)), 1e-12)
    if reciprocity > 1e-8:
        raise ArithmeticError("same P0 operator Fy/Mx reciprocity failed")
    return {
        "identity": "R8_BL5_C1_SOLID_HALF_CELL_P0_COUNTERFACTUAL_DEV_V1",
        "status": "SAME_SOLID_BODY_ENFORCEMENT_DIAGNOSTIC_NOT_QUALIFIED",
        "source_sha256": _sha(Path(__file__)),
        "prepared_metadata_sha256": _sha(meta_path),
        "strong_kkt_result_sha256": _sha(STRONG),
        "native_saved_result_sha256": _sha(NATIVE),
        "physical_dofs": size,
        "supported_dofs": len(fixed),
        "p0_rows": 20,
        "strong_rows": 25,
        "kkt_nnz": system.nnz,
        "assembly": assembly,
        "quadrature_weight_error": _quadrature_check(),
        "rigid_gap_by_mode": rigid,
        "reciprocity": reciprocity,
        "cases": cases,
        "effective_runtime": {
            "affinity": psutil.Process().cpu_affinity(),
            "numba_disable_jit": int(numba.config.DISABLE_JIT),
            "numba_threads": numba.get_num_threads(),
            "threadpools": threadpool_info(),
            "anyfesolver_import": str(Path(anyfesolver.__file__).resolve()),
        },
        "limitation": "one saved C1 solid body, exact half-cell P0 source KKT; not external execution, native body equivalence, P0 selection, mesh convergence or formal F03",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior counterfactual result")
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "cases": {name: {key: item[key] for key in
                                        ("solid_p0_over_strong", "native_p0_over_solid_p0")}
                                for name, item in result["cases"].items()}}))


if __name__ == "__main__":
    main()
