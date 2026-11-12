"""Same-law Mx coupled contrast varying only the resolved shell mesh.

The fine C3D20 beam, five-channel weak-P0 interface, loads and physical root
are held fixed. This is a development diagnosis, not F03 or external acceptance.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import time

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from threadpoolctl import threadpool_info

from scripts import r8_bl5_mx_coupled_body_refinement_dev as coupled
from scripts import r8_bl5_mx_torsion_refinement_dev as torsion
from scripts import r8_bl5_released_director_reference as solid
from scripts import r8_bl5_weak_continuum_reference as continuum


ROOT = Path(__file__).resolve().parents[1]
BEAM_LEVEL = (8, 8, 4)
SHELL_LEVELS = ((2, 2, 1), (4, 4, 2), (8, 8, 4))
TORQUE_NM = 10.


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _shell_mesh(nx: int, ny: int, nz: int) -> tuple[dict, dict]:
    nodes: dict[int, tuple] = {}
    elements: dict[int, tuple] = {}
    seen: dict[tuple, int] = {}
    for ix, iy, iz in itertools.product(range(nx), range(ny), range(nz)):
        ids = []
        for ref in solid._REFERENCE:
            key = (2*ix+ref[0]+1, 2*iy+ref[1]+1, 2*iz+ref[2]+1)
            if key not in seen:
                node = len(nodes)+1
                seen[key] = node
                nodes[node] = (key[0]/(2*nx), -.1+.2*key[1]/(2*ny),
                               -.01+.02*key[2]/(2*nz))
            ids.append(seen[key])
        elements[len(elements)+1] = tuple(ids)
    if len(elements) != nx*ny*nz:
        raise ValueError("incomplete conforming C3D20 shell mesh")
    return nodes, elements


def _shell_body(nodes: dict, elements: dict, level: tuple) -> sparse.csr_matrix:
    spans = (1./level[0], .2/level[1], .02/level[2])
    ke = torsion._local_stiffness(spans)
    rows, cols, values = [], [], []
    for ids in elements.values():
        _, _, _, _, actual = solid._brick(ids, nodes)
        if any(abs(a-b) > 1e-14 for a, b in zip(actual, spans)):
            raise ValueError("nonuniform or distorted shell brick")
        slots = np.array([3*(node-1)+axis for node in ids for axis in range(3)])
        rows.append(np.repeat(slots, 60))
        cols.append(np.tile(slots, 60))
        values.append(ke.ravel())
    count = 3*len(nodes)
    matrix = sparse.coo_matrix((np.concatenate(values),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(count, count)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.nnz == 0 or not np.isfinite(matrix.data).all():
        raise ValueError("finite sparse shell body required")
    return matrix


def _combined(shell_nodes: dict, shell_elements: dict,
              beam_nodes: dict, beam_elements: dict) -> dict:
    node_offset = max(shell_nodes)
    element_offset = max(shell_elements)
    nodes = dict(shell_nodes)
    nodes.update({node_offset+node: position for node, position in beam_nodes.items()})
    elements = dict(shell_elements)
    elements.update({element_offset+eid: tuple(node_offset+node for node in ids)
                     for eid, ids in beam_elements.items()})
    root = tuple(node for node, position in shell_nodes.items()
                 if abs(position[0]) < 1e-14)
    tip = min(node_offset+node for node, position in beam_nodes.items()
              if abs(position[0]-1.) < 1e-14)
    return {"physical_nodes": nodes, "elements": elements,
            "root_nodes": root, "shell_nodes": tuple(shell_nodes),
            "beam_nodes": tuple(node_offset+node for node in beam_nodes),
            "shell_elements": tuple(shell_elements),
            "beam_elements": tuple(element_offset+eid for eid in beam_elements),
            "load_node": tip, "load_N": (0., 0., 0.)}


def _coarse_control(old: dict, nodes: dict, matrix: sparse.csr_matrix) -> dict:
    old_matrix = coupled._shell_body(old)
    xyz = {tuple(round(v, 12) for v in old["physical_nodes"][node]): i
           for i, node in enumerate(sorted(old["shell_nodes"]))}
    if len(xyz) != len(nodes):
        raise ValueError("coarse shell-node inventory differs")
    slots = np.array([3*xyz[tuple(round(v, 12) for v in nodes[node])]+axis
                      for node in nodes for axis in range(3)])
    old_reordered = old_matrix[slots, :][:, slots]
    delta = (matrix-old_reordered).tocsr()
    relative = float(np.max(np.abs(delta.data))/max(np.max(np.abs(old_reordered.data)), 1.)) if delta.nnz else 0.
    if relative > 1e-10:
        raise ValueError("new shell body fails independent coarse assembly")
    return {"shell_stiffness_entry_relative_max": relative}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("preserve prior shell refinement output")
    receipt = json.loads((args.prior_dir/"receipt.json").read_text(encoding="utf-8"))
    if (receipt["status"] != "COMPLETED" or receipt["exit_code"] != 0 or
            set(receipt["level_output_sha256"]) != {"2x2x1.json", "4x4x2.json", "8x8x4.json"} or
            any(_sha(args.prior_dir/"levels"/name) != digest
                for name, digest in receipt["level_output_sha256"].items())):
        raise ValueError("exact complete fixed-shell beam-refinement input required")
    prior = json.loads((args.prior_dir/"levels/8x8x4.json").read_text(encoding="utf-8"))
    if prior["status"] != "DEVELOPMENT_COUPLED_BEAM_REFINEMENT_NOT_QUALIFIED":
        raise ValueError("completed fine-beam parent required")
    pools = threadpool_info()
    if (psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(p["num_threads"] != 1 for p in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT/"src"/"anyfesolver"):
        raise ValueError("effective source/runtime changed")
    _, old = solid.build_reference()
    beam_nodes, beam_elements = torsion._mesh(*BEAM_LEVEL)
    beam = torsion._body(beam_nodes, beam_elements, BEAM_LEVEL)
    tip, meta = torsion._port(beam_nodes, beam_elements, 1., BEAM_LEVEL)
    args.output_dir.mkdir(parents=True)
    for level in SHELL_LEVELS:
        started = time.perf_counter()
        shell_nodes, shell_elements = _shell_mesh(*level)
        shell = _shell_body(shell_nodes, shell_elements, level)
        control = _coarse_control(old, shell_nodes, shell) if level == SHELL_LEVELS[0] else None
        model = _combined(shell_nodes, shell_elements, beam_nodes, beam_elements)
        body = sparse.block_diag((shell, beam), format="csr")
        c, scale = continuum.constraints(model, "weak")
        load = np.zeros(body.shape[0])
        load[-beam.shape[0]:] = tip.T @ np.array((0., 0., 0., TORQUE_NM, 0., 0.))
        result = coupled._solve(model, body, c, scale, load, tip)
        if level == SHELL_LEVELS[0]:
            previous_work = prior["result"]["applied_work_J"]
            if abs(result["applied_work_J"]-previous_work)/previous_work > 1e-10:
                raise ValueError("coarse shell/fine beam solved control changed")
        record = {"status": "DEVELOPMENT_FIXED_BEAM_SHELL_REFINEMENT_NOT_QUALIFIED",
                  "identity": "R8_BL5_MX_FIXED_BEAM_SHELL_REFINEMENT_DEV_V1",
                  "source_sha256": _sha(Path(__file__)),
                  "prior_receipt_sha256": _sha(args.prior_dir/"receipt.json"),
                  "level": level, "shell_nodes": len(shell_nodes),
                  "shell_elements": len(shell_elements),
                  "fixed_beam_nodes": len(beam_nodes),
                  "fixed_beam_elements": len(beam_elements),
                  "coarse_independent_control": control,
                  "native_work_J": prior["native_work_J"],
                  "native_over_resolved": prior["native_work_J"]/result["applied_work_J"],
                  "result": result, "tip_port": meta,
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
