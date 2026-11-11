"""Bounded strong-line versus weak-P0 Fy/Mx C3D20 reference design.

The strong reference samples the same five material thickness-secant channels
at shell stations. It is a small-strain solid proxy, not an accepted native
director law, a finite-area weld, or F03 evidence.
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
LEVEL = (8, 4, 2)
WEAK_SHA256 = "69ac74e73dcc63e306aecdfb3d21b32088350905dcefddb4f9f0122e78a4feb3"
FORCE_Y_N = 100.
TORQUE_X_NM = 10.


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict:
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


def _strong_rows(model: dict, stations: int) -> sparse.csr_matrix:
    _, index, _, _, _ = continuum.layout(model)
    faces = matched._face_index(model)
    rows = np.stack([matched._five_rows(model, index, faces, edge/(stations-1))
                     for edge in range(stations)], axis=0)
    # Channel-major: translation x/y/z, then two physical material-line tilts.
    return sparse.csr_matrix(rows.transpose(1, 0, 2).reshape(5*stations, -1))


def _solve(model: dict, k: sparse.csr_matrix, c: sparse.csr_matrix,
           scale: np.ndarray, loads: np.ndarray, tip: np.ndarray) -> dict:
    _, _, fixed, free, _ = continuum.layout(model)
    physical = k[free, :][:, free].tocsc()
    constraint = c[:, free].tocsc()
    diagonal = physical.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
        raise ValueError("positive finite physical diagonal required")
    dq = 1./np.sqrt(diagonal)
    row_norm = np.sqrt(np.asarray((constraint @ sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    if not np.isfinite(row_norm).all() or np.any(row_norm <= 0.):
        raise ValueError("finite nonzero physical constraint rows required")
    dl = 1./row_norm
    dp, dm = sparse.diags(dq), sparse.diags(dl)
    system = sparse.bmat([[dp @ physical @ dp, dp @ constraint.T @ dm],
                          [dm @ constraint @ dp, None]], format="csc")
    factor = splu(system)
    q = np.zeros_like(loads)
    multipliers = np.zeros((c.shape[0], loads.shape[1]))
    cases = {}
    order = tuple(sorted(model["physical_nodes"]))
    xyz = np.array([model["physical_nodes"][node] for node in order])
    for j, name in enumerate(("fy", "mx")):
        load = loads[:, j]
        rhs = np.r_[dq*load[free], np.zeros(c.shape[0])]
        state = factor.solve(rhs)
        for _ in range(6):
            state += factor.solve(rhs-system @ state)
        q[free, j] = dq*state[:len(free)]
        multipliers[:, j] = dl*state[len(free):]
        internal = np.asarray(k @ q[:, j]).ravel()
        interface = np.asarray(-c.T @ multipliers[:, j]).ravel()
        residual = internal-interface-load
        support = np.zeros(len(load))
        support[fixed] = residual[fixed]
        applied_wrench = section_port._wrench(xyz, load)
        balanced = section_port._wrench(xyz, load+support)
        port = tip @ q[-tip.shape[1]:, j]
        work = float(load @ q[:, j])
        checks = {
            "finite_state": bool(np.isfinite(q[:, j]).all() and np.isfinite(multipliers[:, j]).all()),
            "free_equilibrium": float(np.linalg.norm(residual[free])/max(FORCE_Y_N, TORQUE_X_NM)),
            "weak_or_station_gap": float(np.max(np.abs(c @ q[:, j]/scale))),
            "strain_work": abs(float(q[:, j] @ internal)-work)/max(abs(work), 1e-12),
            "interface_work": abs(float(q[:, j] @ interface))/max(abs(work), 1e-12),
            "force_balance": float(np.max(np.abs(balanced[:3]))/FORCE_Y_N),
            "moment_balance": float(np.max(np.abs(balanced[3:]))/TORQUE_X_NM),
            "port_duality": abs(work-(FORCE_Y_N*port[1] if name == "fy"
                                     else TORQUE_X_NM*port[3]))/max(abs(work), 1e-12),
        }
        limits = {key: (1e-9 if key == "weak_or_station_gap" else 1e-8)
                  for key in checks if key != "finite_state"}
        if not checks["finite_state"] or any(checks[key] > limit for key, limit in limits.items()):
            raise ValueError(f"{name} physical KKT/work controls failed: {checks}")
        cases[name] = {"applied_work_J": work, "port_pose": port.tolist(),
                       "applied_wrench_about_origin": applied_wrench.tolist(),
                       "support_wrench_about_origin": (balanced-applied_wrench).tolist(),
                       "checks": checks, "limits": limits}
    work_matrix = loads.T @ q
    reciprocity = abs(float(work_matrix[0, 1]-work_matrix[1, 0])) / max(
        float(np.max(np.abs(work_matrix))), 1e-12)
    if reciprocity > 1e-8:
        raise ValueError("conservative cross-work reciprocity failed")
    return {"cases": cases, "cross_work_matrix_J": work_matrix.tolist(),
            "cross_work_reciprocity": reciprocity,
            "physical_dofs": k.shape[0], "constraint_rows": c.shape[0],
            "matrix_nnz": k.nnz, "kkt_nnz": system.nnz}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-weak", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior strong-hinge design attempt")
    if digest(args.saved_weak) != WEAK_SHA256:
        raise ValueError("exact independent 8x4x2 weak result required")
    prior = json.loads(args.saved_weak.read_text(encoding="utf-8"))
    if (prior["status"] != "DEVELOPMENT_MATCHED_REFINEMENT_NOT_QUALIFIED" or
            tuple(prior["level"]) != LEVEL or prior["multiplier_rows"] != 40):
        raise ValueError("saved weak case identity or completion changed")
    runtime = _runtime()
    started = time.perf_counter()
    shell_nodes, shell_elements = shell_ref._shell_mesh(*LEVEL)
    beam_nodes, beam_elements = beam_ref._mesh(*LEVEL)
    model = shell_ref._combined(shell_nodes, shell_elements, beam_nodes, beam_elements)
    shell = shell_ref._shell_body(shell_nodes, shell_elements, LEVEL)
    beam = beam_ref._body(beam_nodes, beam_elements, LEVEL)
    k = sparse.block_diag((shell, beam), format="csr")
    c_weak, weak_scale = matched._weak_rows(model, LEVEL[0])
    c_strong = _strong_rows(model, LEVEL[0]+1)
    strong_scale = np.ones(c_strong.shape[0])
    tip, tip_meta = beam_ref._port(beam_nodes, beam_elements, 1., LEVEL)
    loads = np.zeros((k.shape[0], 2))
    loads[-beam.shape[0]:, 0] = tip.T @ np.array((0., FORCE_Y_N, 0., 0., 0., 0.))
    loads[-beam.shape[0]:, 1] = tip.T @ np.array((0., 0., 0., TORQUE_X_NM, 0., 0.))
    weak = _solve(model, k, c_weak, weak_scale, loads, tip)
    expected = float(prior["resolved"]["applied_work_J"])
    if abs(weak["cases"]["mx"]["applied_work_J"]-expected)/expected > 1e-8:
        raise ValueError("same-body weak Mx result does not reproduce frozen reference")
    strong = _solve(model, k, c_strong, strong_scale, loads, tip)
    record = {"identity": "R8_BL5_C3D20_STRONG_HINGE_FY_MX_DESIGN_DEV_V1",
              "status": "DEVELOPMENT_REFERENCE_DESIGN_NOT_QUALIFIED",
              "source_sha256": digest(Path(__file__)),
              "saved_weak_sha256": digest(args.saved_weak),
              "level": LEVEL, "shell_elements": len(shell_elements),
              "beam_elements": len(beam_elements), "tip_port": tip_meta,
              "weak": weak, "strong": strong,
              "runtime": runtime, "wall_seconds": time.perf_counter()-started,
              "limitation": "single-grid five-channel thickness-secant proxy; strong station sampling is not an accepted native BL5 law or converged engineering reference"}
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": record["status"],
                      "weak_mx_work": weak["cases"]["mx"]["applied_work_J"],
                      "strong_mx_work": strong["cases"]["mx"]["applied_work_J"],
                      "weak_fy_work": weak["cases"]["fy"]["applied_work_J"],
                      "strong_fy_work": strong["cases"]["fy"]["applied_work_J"],
                      "wall_seconds": record["wall_seconds"]}))


if __name__ == "__main__":
    main()
