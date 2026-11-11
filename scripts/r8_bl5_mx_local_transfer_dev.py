"""Same-body BL5 Mx local gap and wrench diagnosis for 80/160 P0 rows.

Development-only: neither lower gaps nor smoother wrenches select a multiplier
space. The comparison holds the C3D20 shell, beam, load and supports fixed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info

from scripts import r8_bl5_mx_matched_refinement_dev as matched
from scripts import r8_bl5_mx_coupled_shell_refinement_dev as shell_ref
from scripts import r8_bl5_mx_torsion_refinement_dev as beam_ref
from scripts import r8_bl5_weak_continuum_reference as continuum
from scripts import r8_bl5_mx_section_port_dev as section_port


ROOT = Path(__file__).resolve().parents[1]
FINE_SHA256 = "65bab6152c4e2c2885eda352ef525b3243ca426902f88a0adfaacbf951bb6bac"
COARSE_SHA256 = "6555b7270e7f84c6d9459eab17b87cfc5314e24cebee74e90c2a7092a3ec7784"
SHELL_LEVEL = (16, 8, 4)
BEAM_LEVEL = (32, 8, 4)
SEGMENTS = 32
TORQUE_NM = 10.


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _check_runtime() -> dict:
    pools = threadpool_info()
    result = {"cpu_affinity": psutil.Process().cpu_affinity(),
              "numba_disable_jit": int(numba.config.DISABLE_JIT),
              "numba_threads": int(numba.get_num_threads()),
              "threadpools": pools,
              "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())}
    if (result["cpu_affinity"] != [0] or result["numba_disable_jit"] != 1 or
            result["numba_threads"] != 1 or not pools or
            any(item["num_threads"] != 1 for item in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT / "src" / "anyfesolver"):
        raise ValueError("effective source/runtime changed")
    return result


def _solve(model: dict, k: sparse.csr_matrix, c: sparse.csr_matrix,
           scale: np.ndarray, load: np.ndarray, expected_work: float) -> tuple[np.ndarray, np.ndarray, dict]:
    _, _, fixed, free, _ = continuum.layout(model)
    physical = k[free, :][:, free].tocsc()
    constraint = c[:, free].tocsc()
    diagonal = physical.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
        raise ValueError("positive finite KKT coordinate diagonals required")
    dq = 1. / np.sqrt(diagonal)
    row_norm = np.sqrt(np.asarray((constraint @ sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    if not np.isfinite(row_norm).all() or np.any(row_norm <= 0.):
        raise ValueError("finite nonzero physical constraint rows required")
    dl = 1. / row_norm
    dp, dm = sparse.diags(dq), sparse.diags(dl)
    system = sparse.bmat([[dp @ physical @ dp, dp @ constraint.T @ dm],
                          [dm @ constraint @ dp, None]], format="csc")
    rhs = np.r_[dq * load[free], np.zeros(c.shape[0])]
    factor = splu(system)
    state = factor.solve(rhs)
    for _ in range(6):
        state += factor.solve(rhs - system @ state)
    q = np.zeros(len(load))
    q[free] = dq * state[:len(free)]
    lam = dl * state[len(free):]
    internal = np.asarray(k @ q).ravel()
    connection = np.asarray(-c.T @ lam).ravel()
    residual = internal - connection - load
    support = np.zeros_like(q)
    support[fixed] = residual[fixed]
    xyz = np.array([model["physical_nodes"][node]
                    for node in sorted(model["physical_nodes"])])
    external_wrench = section_port._wrench(xyz, load + support)
    work = float(load @ q)
    checks = {
        "finite_state": bool(np.isfinite(q).all() and np.isfinite(lam).all()),
        "free_equilibrium": float(np.linalg.norm(residual[free]) / TORQUE_NM),
        "weak_gap": float(np.max(np.abs(c @ q / scale))),
        "work": abs(float(q @ internal)-work) / max(abs(work), 1e-12),
        "interface_work": abs(float(q @ connection)) / max(abs(work), 1e-12),
        "global_wrench": float(np.linalg.norm(external_wrench) / TORQUE_NM),
        "saved_work": abs(work-expected_work) / max(abs(expected_work), 1e-12),
    }
    limits = {name: (1e-9 if name == "weak_gap" else 1e-8)
              for name in checks if name != "finite_state"}
    if not checks["finite_state"] or any(checks[name] > limit
                                         for name, limit in limits.items()):
        raise ValueError(f"same-body physical control failed: {checks}")
    return q, lam, {"checks": checks, "limits": limits,
                    "applied_work_J": work, "kkt_nnz": system.nnz}


def _wrench(model: dict, order: tuple[int, ...], nodes: tuple[int, ...],
            vector: np.ndarray) -> np.ndarray:
    index = {node: j for j, node in enumerate(order)}
    result = np.zeros(6)
    for node in nodes:
        j = index[node]
        force = vector[3*j:3*j+3]
        result[:3] += force
        result[3:] += np.cross(model["physical_nodes"][node], force)
    return result


def _trace(model: dict, q: np.ndarray, c_fine: sparse.csr_matrix,
           lam_fine: np.ndarray, order: tuple[int, ...]) -> dict:
    _, index, _, _, _ = continuum.layout(model)
    faces = matched._face_index(model)
    points, weights = np.polynomial.legendre.leggauss(3)
    integral_sq = np.zeros(5)
    maximum = np.zeros(5)
    means = np.asarray(c_fine @ q).reshape(5, SEGMENTS) * SEGMENTS
    gap_samples = []
    for edge in range(SEGMENTS):
        sample = []
        for point, weight in zip(points, weights):
            x = (edge + .5 + .5*point) / SEGMENTS
            gap = matched._five_rows(model, index, faces, x) @ q
            integral_sq += .5 / SEGMENTS * weight * gap**2
            maximum = np.maximum(maximum, np.abs(gap))
            sample.append(gap.tolist())
        gap_samples.append(sample)
    beam_wrenches = []
    shell_wrenches = []
    for edge in range(SEGMENTS):
        rows = c_fine[[channel*SEGMENTS+edge for channel in range(5)], :]
        multipliers = lam_fine[[channel*SEGMENTS+edge for channel in range(5)]]
        action = -np.asarray(rows.T @ multipliers).ravel()
        beam_wrenches.append(_wrench(model, order, model["beam_nodes"], action).tolist())
        shell_wrenches.append(_wrench(model, order, model["shell_nodes"], action).tolist())
    beam_wrenches = np.asarray(beam_wrenches)
    shell_wrenches = np.asarray(shell_wrenches)
    balance = beam_wrenches + shell_wrenches
    return {"rms_pointwise_gap_channels": np.sqrt(integral_sq).tolist(),
            "max_pointwise_gap_channels": maximum.tolist(),
            "max_abs_fine_segment_mean_channels": np.max(np.abs(means), axis=1).tolist(),
            "fine_segment_mean_gaps": means.tolist(),
            "three_gauss_gap_samples": gap_samples,
            "beam_segment_wrenches_about_origin": beam_wrenches.tolist(),
            "shell_segment_wrenches_about_origin": shell_wrenches.tolist(),
            "max_segment_wrench_balance": float(np.max(np.abs(balance))),
            "beam_total_wrench_about_origin": beam_wrenches.sum(axis=0).tolist(),
            "shell_total_wrench_about_origin": shell_wrenches.sum(axis=0).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous local-transfer output")
    fine = json.loads(args.fine.read_text(encoding="utf-8"))
    coarse = json.loads(args.coarse.read_text(encoding="utf-8"))
    if (digest(args.fine) != FINE_SHA256 or digest(args.coarse) != COARSE_SHA256 or
            fine["status"] != "DEVELOPMENT_DISCRIMINATOR_NOT_QUALIFIED" or
            coarse["status"] != "DEVELOPMENT_DISCRIMINATOR_NOT_QUALIFIED" or
            fine["new_interface_rows"] != 160 or
            coarse["aggregated_interface_rows"] != 80 or
            tuple(fine["fixed_shell_level"]) != SHELL_LEVEL or
            tuple(coarse["fixed_beam_level"]) != BEAM_LEVEL):
        raise ValueError("exact saved 80/160 row pair required")
    runtime = _check_runtime()
    started = time.perf_counter()
    shell_nodes, shell_elements = shell_ref._shell_mesh(*SHELL_LEVEL)
    beam_nodes, beam_elements = beam_ref._mesh(*BEAM_LEVEL)
    shell = shell_ref._shell_body(shell_nodes, shell_elements, SHELL_LEVEL)
    beam = beam_ref._body(beam_nodes, beam_elements, BEAM_LEVEL)
    model = shell_ref._combined(shell_nodes, shell_elements, beam_nodes, beam_elements)
    k = sparse.block_diag((shell, beam), format="csr")
    c_fine, scale_fine = matched._weak_rows(model, SEGMENTS)
    coarse_index = np.arange(80).repeat(2)
    fine_index = np.array([channel*SEGMENTS+2*edge+offset
                           for channel in range(5) for edge in range(16)
                           for offset in range(2)])
    aggregate = sparse.coo_matrix((np.ones(160), (coarse_index, fine_index)),
                                  shape=(80, 160)).tocsr()
    c_coarse = (aggregate @ c_fine).tocsr()
    scale_coarse = np.full(80, 1./16.)
    tip, _ = beam_ref._port(beam_nodes, beam_elements, 1., BEAM_LEVEL)
    load = np.zeros(k.shape[0])
    load[-beam.shape[0]:] = tip.T @ np.array((0., 0., 0., TORQUE_NM, 0., 0.))
    order, _, _, _, _ = continuum.layout(model)
    cases = {}
    for name, c, scale, prior in (
            ("fine160", c_fine, scale_fine, fine),
            ("aggregated80", c_coarse, scale_coarse, coarse)):
        expected = prior["new_resolved_work_J"] if name == "fine160" else prior["aggregated_resolved_work_J"]
        q, lam, control = _solve(model, k, c, scale, load, float(expected))
        # Coarse work uses one multiplier per two integrated child segments.
        lam_fine = lam if name == "fine160" else np.asarray(aggregate.T @ lam).ravel()
        cases[name] = {"control": control, "trace": _trace(model, q, c_fine, lam_fine, order),
                       "multiplier_density": lam_fine.reshape(5, SEGMENTS).tolist()}
    for item in cases.values():
        if item["trace"]["max_segment_wrench_balance"] > 1e-8:
            raise ValueError("physical segment wrench balance changed")
    record = {"identity": "R8_BL5_MX_SAME_BODY_LOCAL_TRANSFER_DEV_V1",
              "status": "DEVELOPMENT_LOCAL_TRANSFER_NOT_QUALIFIED",
              "source_sha256": digest(Path(__file__)),
              "fine_input_sha256": digest(args.fine),
              "coarse_input_sha256": digest(args.coarse),
              "shell_level": SHELL_LEVEL, "beam_level": BEAM_LEVEL,
              "physical_dofs": k.shape[0], "physical_segments": SEGMENTS,
              "cases": cases, "runtime": runtime,
              "wall_seconds": time.perf_counter()-started,
              "limitation": "Finite-grid supported response and physical transfer only; no line-law or inf-sup qualification"}
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": record["status"],
                      "fine_rms": cases["fine160"]["trace"]["rms_pointwise_gap_channels"],
                      "coarse_rms": cases["aggregated80"]["trace"]["rms_pointwise_gap_channels"],
                      "wall_seconds": record["wall_seconds"]}))


if __name__ == "__main__":
    main()
