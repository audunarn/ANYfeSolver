"""One fixed-beam/weak-action C3D20 shell-thickness discriminator at 16x8.

Development diagnosis only. The independently saved 16x8x4 result is the
unchanged-geometry control; no CalculiX or F03 assessment occurs here.
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
from scripts import r8_bl5_mx_coupled_body_refinement_dev as coupled
from scripts import r8_bl5_mx_torsion_refinement_dev as beam_ref
from scripts import r8_bl5_weak_continuum_reference as continuum
from scripts import r8_bl5_mx_section_port_dev as section_port

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA256 = "c81720f503d972dff685ec114446eece5abeef4c1d7d9a1d9df56cf43ff79729"
BEAM_LEVEL = (16, 8, 4)
SHELL_LEVEL = (16, 8, 8)
TORQUE_NM = 10.


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def diagnostic_solve(model, body, c, scale, load, tip):
    """Repeat the established scaled KKT and expose every unchanged guard."""
    _, _, fixed, free, _ = continuum.layout(model)
    physical = body[free, :][:, free].tocsc()
    constraint = c[:, free].tocsc()
    dq = 1. / np.sqrt(physical.diagonal())
    row_norm = np.sqrt(np.asarray((constraint @ sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    dl = 1. / row_norm
    dp, dm = sparse.diags(dq), sparse.diags(dl)
    system = sparse.bmat([[dp @ physical @ dp, dp @ constraint.T @ dm],
                          [dm @ constraint @ dp, None]], format="csc")
    rhs = np.r_[dq * load[free], np.zeros(c.shape[0])]
    factor = splu(system)
    state = factor.solve(rhs)
    # The larger thickness-refined KKT needs the same physical tolerances;
    # additional residual correction is a numerical solve repair, not a gate relaxation.
    for _ in range(6):
        state += factor.solve(rhs-system @ state)
    q = np.zeros(len(load))
    q[free] = dq * state[:len(free)]
    lam = dl * state[len(free):]
    internal = np.asarray(body @ q).ravel()
    interface = np.asarray(-c.T @ lam).ravel()
    residual = internal-interface-load
    support = np.zeros_like(q)
    support[fixed] = residual[fixed]
    work = float(load @ q)
    xyz = np.array([model["physical_nodes"][node]
                    for node in sorted(model["physical_nodes"])])
    total_wrench = section_port._wrench(xyz, load+support)
    order = tuple(sorted(model["physical_nodes"]))
    index = {node: i for i, node in enumerate(order)}
    side = {}
    for domain in ("shell", "beam"):
        ids = tuple(sorted(model[domain+"_nodes"]))
        slots = np.array([3*index[node]+axis for node in ids for axis in range(3)])
        side[domain] = section_port._wrench(xyz[[index[node] for node in ids]],
                                            interface[slots]).tolist()
    spin = float((tip @ q[-3*len(model["beam_nodes"]):])[3])
    checks = {
        "finite_state": bool(np.isfinite(q).all() and np.isfinite(lam).all()),
        "equation": float(np.linalg.norm(residual[free])/TORQUE_NM),
        "gap": float(np.max(np.abs(c @ q / scale))),
        "relative_work": abs(float(q @ internal)-work)/max(abs(work), 1e-12),
        "interface_work": abs(float(q @ interface))/max(abs(work), 1e-12),
        "global_wrench": float(np.linalg.norm(total_wrench)/TORQUE_NM),
        "interface_wrench": float(np.linalg.norm(np.asarray(side["shell"])+np.asarray(side["beam"]))/TORQUE_NM),
        "port_work": abs(work-TORQUE_NM*spin)/max(abs(work), 1e-12),
    }
    limits = {name: (1e-9 if name == "gap" else 1e-8) for name in checks if name != "finite_state"}
    passed = checks["finite_state"] and all(checks[name] <= limit for name, limit in limits.items())
    return {"status": "PASSED" if passed else "FAILED_CONTROL",
            "checks": checks, "limits": limits,
            "applied_work_J": work, "tip_section_spin_x_rad": spin,
            "total_external_support_wrench": total_wrench.tolist(),
            "interface_wrench": side, "physical_dofs": len(load),
            "matrix_nnz": body.nnz, "kkt_nnz": system.nnz}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous discriminator output")
    prior = json.loads(args.independent.read_text(encoding="utf-8"))
    if (digest(args.independent) != PARENT_SHA256 or
            prior["status"] != "DEVELOPMENT_MATCHED_REFINEMENT_NOT_QUALIFIED" or
            tuple(prior["level"]) != BEAM_LEVEL or prior["multiplier_rows"] != 80):
        raise ValueError("exact saved 16x8x4 matched case required")
    pools = threadpool_info()
    runtime = {"cpu_affinity": psutil.Process().cpu_affinity(),
               "numba_disable_jit": int(numba.config.DISABLE_JIT),
               "numba_threads": int(numba.get_num_threads()),
               "threadpools": pools,
               "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())}
    if (runtime["cpu_affinity"] != [0] or runtime["numba_disable_jit"] != 1 or
            runtime["numba_threads"] != 1 or not pools or
            any(item["num_threads"] != 1 for item in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT / "src" / "anyfesolver"):
        raise ValueError("effective source/runtime does not match bounded contract")
    started = time.perf_counter()
    beam_nodes, beam_elements = beam_ref._mesh(*BEAM_LEVEL)
    beam = beam_ref._body(beam_nodes, beam_elements, BEAM_LEVEL)
    tip, meta = beam_ref._port(beam_nodes, beam_elements, 1., BEAM_LEVEL)
    shell_nodes, shell_elements = shell_ref._shell_mesh(*SHELL_LEVEL)
    shell = shell_ref._shell_body(shell_nodes, shell_elements, SHELL_LEVEL)
    model = shell_ref._combined(shell_nodes, shell_elements, beam_nodes, beam_elements)
    body = sparse.block_diag((shell, beam), format="csr")
    c, scale = matched._weak_rows(model, BEAM_LEVEL[0])
    load = np.zeros(body.shape[0])
    load[-beam.shape[0]:] = tip.T @ np.array((0., 0., 0., TORQUE_NM, 0., 0.))
    result = diagnostic_solve(model, body, c, scale, load, tip)
    old = float(prior["resolved"]["applied_work_J"])
    new = float(result["applied_work_J"])
    record = {"identity": "R8_BL5_MX_FIXED_BEAM_SHELL_THICKNESS_DEV_V2",
              "status": ("DEVELOPMENT_DISCRIMINATOR_NOT_QUALIFIED" if result["status"] == "PASSED"
                         else "DEVELOPMENT_CONTROL_FAILED"),
              "source_sha256": digest(Path(__file__)),
              "independent_sha256": digest(args.independent),
              "fixed_beam_level": BEAM_LEVEL,
              "fixed_interface_rows": int(c.shape[0]),
              "old_shell_level": BEAM_LEVEL,
              "new_shell_level": SHELL_LEVEL,
              "old_resolved_work_J": old,
              "new_resolved_work_J": new,
              "relative_shell_thickness_change": (new-old)/old,
              "native_work_J": prior["native"]["applied_work_J"],
              "new_native_over_resolved": prior["native"]["applied_work_J"]/new,
              "new_shell_elements": len(shell_elements),
              "fixed_beam_elements": len(beam_elements),
              "tip_port": meta, "result": result,
              "runtime": runtime, "solve_wall_seconds": time.perf_counter()-started,
              "limitation": "One shell-thickness refinement only; does not establish convergence or select BL5"}
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "status", "relative_shell_thickness_change", "new_native_over_resolved",
        "solve_wall_seconds")}))
    if record["status"] != "DEVELOPMENT_DISCRIMINATOR_NOT_QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
