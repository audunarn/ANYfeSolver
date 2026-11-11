"""Same-law weak-P0 Mx coupled contrast varying only the resolved beam mesh.

The shell body, physical line, twenty P0 rows, support and torque are fixed.
This is a development diagnosis, not external or F03 qualification.
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

from scripts import r8_bl5_mx_section_port_dev as section_port
from scripts import r8_bl5_mx_torsion_refinement_dev as torsion
from scripts import r8_bl5_released_director_reference as solid
from scripts import r8_bl5_weak_continuum_reference as continuum


ROOT = Path(__file__).resolve().parents[1]
PRIOR_SHA256 = "84a0280817174df9946f455dd6939842c21d6648bd23796d149906300914fe1b"
TORQUE_NM = 10.


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _combined(old: dict, beam_nodes: dict, beam_elements: dict) -> dict:
    shell_nodes = tuple(sorted(old["shell_nodes"]))
    shell_elements = tuple(sorted(old["shell_elements"]))
    node_offset = max(shell_nodes)
    element_offset = max(shell_elements)
    nodes = {node: old["physical_nodes"][node] for node in shell_nodes}
    nodes.update({node_offset+node: position for node, position in beam_nodes.items()})
    elements = {eid: old["elements"][eid] for eid in shell_elements}
    elements.update({element_offset+eid: tuple(node_offset+node for node in ids)
                     for eid, ids in beam_elements.items()})
    tip_node = min(node_offset+node for node, position in beam_nodes.items()
                   if abs(position[0]-1.) < 1e-14)
    return {"physical_nodes": nodes, "elements": elements,
            "root_nodes": old["root_nodes"], "shell_nodes": shell_nodes,
            "beam_nodes": tuple(node_offset+node for node in beam_nodes),
            "shell_elements": shell_elements,
            "beam_elements": tuple(element_offset+eid for eid in beam_elements),
            "load_node": tip_node, "load_N": (0., 0., 0.)}


def _shell_body(old: dict) -> sparse.csr_matrix:
    k, _ = continuum.assemble_body(old)
    _, index, _, _, _ = continuum.layout(old)
    slots = np.array([index[node]+axis for node in sorted(old["shell_nodes"])
                      for axis in range(3)])
    return k[slots, :][:, slots].tocsr()


def _solve(model: dict, body: sparse.csr_matrix, c: sparse.csr_matrix,
           scale: np.ndarray, load: np.ndarray, tip: np.ndarray) -> dict:
    _, _, fixed, free, _ = continuum.layout(model)
    physical = body[free, :][:, free].tocsc()
    constraint = c[:, free].tocsc()
    diagonal = physical.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
        raise ValueError("positive finite coordinate diagonals required for scaling")
    dq = 1./np.sqrt(diagonal)
    row_norm = np.sqrt(np.asarray((constraint@sparse.diags(dq)).power(2).sum(axis=1)).ravel())
    if np.any(row_norm <= 0.) or not np.isfinite(row_norm).all():
        raise ValueError("finite nonzero physical constraint rows required")
    dl = 1./row_norm
    dp = sparse.diags(dq)
    dm = sparse.diags(dl)
    system = sparse.bmat([[dp@physical@dp, dp@constraint.T@dm],
                          [dm@constraint@dp, None]], format="csc")
    rhs = np.r_[dq*load[free], np.zeros(c.shape[0])]
    factor = splu(system)
    state = factor.solve(rhs)
    for _ in range(3):
        state += factor.solve(rhs-system@state)
    q = np.zeros(len(load))
    q[free] = dq*state[:len(free)]
    lam = dl*state[len(free):]
    internal = np.asarray(body@q).ravel()
    interface = np.asarray(-c.T@lam).ravel()
    residual = internal-interface-load
    support = np.zeros_like(q)
    support[fixed] = residual[fixed]
    work = float(load@q)
    relative_work = abs(float(q@internal)-work)/max(abs(work), 1e-12)
    interface_work = abs(float(q@interface))/max(abs(work), 1e-12)
    equation = float(np.linalg.norm(residual[free])/TORQUE_NM)
    gap = float(np.max(np.abs(c@q/scale)))
    xyz = np.array([model["physical_nodes"][node]
                    for node in sorted(model["physical_nodes"])])
    total_wrench = section_port._wrench(xyz, load+support)
    side = {}
    order = tuple(sorted(model["physical_nodes"]))
    index = {node: i for i, node in enumerate(order)}
    for domain in ("shell", "beam"):
        ids = tuple(sorted(model[domain+"_nodes"]))
        slots = np.array([3*index[node]+axis for node in ids for axis in range(3)])
        side[domain] = section_port._wrench(xyz[[index[node] for node in ids]],
                                            interface[slots]).tolist()
    beam_count = 3*len(model["beam_nodes"])
    spin = float((tip@q[-beam_count:])[3])
    if (not np.isfinite(q).all() or not np.isfinite(lam).all() or equation > 1e-8 or
            gap > 1e-9 or relative_work > 1e-8 or interface_work > 1e-8 or
            np.linalg.norm(total_wrench)/TORQUE_NM > 1e-8 or
            np.linalg.norm(np.asarray(side["shell"])+np.asarray(side["beam"]))/TORQUE_NM > 1e-8 or
            abs(work-TORQUE_NM*spin)/max(abs(work), 1e-12) > 1e-8):
        raise ValueError("coupled equilibrium, gap, physical work, or wrench failed")
    return {"applied_work_J": work, "tip_section_spin_x_rad": spin,
            "equation_ratio": equation, "scaled_gap_max": gap,
            "relative_internal_work_error": relative_work,
            "relative_interface_work_error": interface_work,
            "total_external_support_wrench": total_wrench.tolist(),
            "interface_wrench": side,
            "multiplier_rows": c.shape[0], "physical_dofs": len(load),
            "matrix_nnz": body.nnz, "kkt_nnz": system.nnz}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--torsion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("preserve prior coupled refinement")
    if _sha(args.prior) != PRIOR_SHA256:
        raise ValueError("exact saved Fy/Mx local-transfer comparison required")
    prior = json.loads(args.prior.read_text(encoding="utf-8"))
    receipt = json.loads((args.torsion_dir/"receipt.json").read_text(encoding="utf-8"))
    if (prior["status"] != "DEVELOPMENT_LOCAL_TRANSFER_NOT_QUALIFIED" or
            receipt["status"] != "COMPLETED" or receipt["exit_code"] != 0 or
            set(receipt["level_output_sha256"]) != {"2x2x1.json", "4x4x2.json", "8x8x4.json"} or
            any(_sha(args.torsion_dir/"levels"/name) != digest
                for name, digest in receipt["level_output_sha256"].items())):
        raise ValueError("exact complete beam-only refinement input required")
    pools = threadpool_info()
    if (psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(p["num_threads"] != 1 for p in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT/"src"/"anyfesolver"):
        raise ValueError("effective source/runtime changed")
    _, old = solid.build_reference()
    shell = _shell_body(old)
    args.output_dir.mkdir(parents=True)
    for level in torsion.LEVELS:
        started = time.perf_counter()
        nodes, elements = torsion._mesh(*level)
        model = _combined(old, nodes, elements)
        beam = torsion._body(nodes, elements, level)
        body = sparse.block_diag((shell, beam), format="csr")
        c, scale = continuum.constraints(model, "weak")
        tip, meta = torsion._port(nodes, elements, 1., level)
        load = np.zeros(body.shape[0])
        load[-beam.shape[0]:] = tip.T @ np.array((0., 0., 0., TORQUE_NM, 0., 0.))
        result = _solve(model, body, c, scale, load, tip)
        external_work = prior["cases"]["mx"]["resolved"]["summary"]["applied_work_J"]
        if level == torsion.LEVELS[0] and abs(result["applied_work_J"]-external_work)/external_work > 1e-7:
            raise ValueError("coarse coupled internal/external same-body work differs")
        native_work = prior["cases"]["mx"]["native"]["summary"]["applied_work_J"]
        record = {"status": "DEVELOPMENT_COUPLED_BEAM_REFINEMENT_NOT_QUALIFIED",
                  "identity": "R8_BL5_MX_FIXED_SHELL_BEAM_REFINEMENT_DEV_V1",
                  "source_sha256": _sha(Path(__file__)), "prior_sha256": PRIOR_SHA256,
                  "torsion_receipt_sha256": _sha(args.torsion_dir/"receipt.json"),
                  "level": level, "beam_nodes": len(nodes), "beam_elements": len(elements),
                  "fixed_shell_nodes": len(old["shell_nodes"]), "tip_port": meta,
                  "native_work_J": native_work, "saved_external_coarse_work_J": external_work,
                  "native_over_resolved": native_work/result["applied_work_J"],
                  "result": result,
                  "effective_runtime": {"cpu_affinity": psutil.Process().cpu_affinity(),
                                        "numba_disable_jit": int(numba.config.DISABLE_JIT),
                                        "numba_threads": int(numba.get_num_threads()),
                                        "threadpools": pools,
                                        "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())},
                  "level_wall_seconds": time.perf_counter()-started}
        name = "x".join(map(str, level))+".json"
        (args.output_dir/name).write_text(json.dumps(record, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")
        print(json.dumps({"level": level,
                          "native_over_resolved": record["native_over_resolved"],
                          "wall_seconds": record["level_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
