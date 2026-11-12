"""Three-level C3D20 bar torsion screen with the same area-L2 section ports.

This tests whether the coarse resolved beam-only Mx discrepancy is a solid
discretization effect. It is not a coupled BL5 or formal F03 assessment.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import time
import warnings

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from threadpoolctl import threadpool_info

from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_mx_section_port_dev as parent
from scripts import r8_bl5_released_director_reference as solid
from scripts import r8_bl5_weak_continuum_reference as continuum


ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA256 = "2ecdfe791189d8c22ff2bd2ecae3835fd264e0773aee40670c29247df133fe89"
LEVELS = ((2, 2, 1), (4, 4, 2), (8, 8, 4))
TORQUE_NM = 10.


def _mesh(nx: int, ny: int, nz: int) -> tuple[dict, dict]:
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
                nodes[node] = (key[0]/(2*nx), -.02+.04*key[1]/(2*ny),
                               .01+.06*key[2]/(2*nz))
            ids.append(seen[key])
        elements[len(elements)+1] = tuple(ids)
    if len(nodes) != len(set(nodes)) or len(elements) != nx*ny*nz:
        raise ValueError("conforming C3D20 refinement topology failed")
    return nodes, elements


def _local_stiffness(spans: tuple[float, float, float]) -> np.ndarray:
    """Uniform Cartesian C3D20 B^T D B with 3x3x3 quadrature."""
    points, weights = np.polynomial.legendre.leggauss(3)
    d = solid._elastic_matrix()
    ke = np.zeros((60, 60))
    volume_scale = np.prod(spans)/8.
    for ii in itertools.product(range(3), repeat=3):
        _, grad = solid._shape(tuple(points[i] for i in ii))
        physical = grad * (2./np.asarray(spans))[None, :]
        b = np.zeros((6, 60))
        for j, (gx, gy, gz) in enumerate(physical):
            b[:, 3*j:3*j+3] = ((gx, 0, 0), (0, gy, 0), (0, 0, gz),
                                (gy, gx, 0), (0, gz, gy), (gz, 0, gx))
        ke += b.T @ d @ b * volume_scale * np.prod([weights[i] for i in ii])
    if not np.isfinite(ke).all():
        raise ValueError("nonfinite C3D20 local stiffness")
    return ke


def _body(nodes: dict, elements: dict, level: tuple) -> sparse.csr_matrix:
    spans = (1./level[0], .04/level[1], .06/level[2])
    ke = _local_stiffness(spans)
    rows, cols, values = [], [], []
    for ids in elements.values():
        _, low, high, _, actual = solid._brick(ids, nodes)
        if any(abs(a-b) > 1e-14 for a, b in zip(actual, spans)):
            raise ValueError("nonuniform or distorted refinement element")
        slots = np.array([3*(node-1)+axis for node in ids for axis in range(3)])
        rows.append(np.repeat(slots, 60))
        cols.append(np.tile(slots, 60))
        values.append(ke.ravel())
    n = 3*len(nodes)
    matrix = sparse.coo_matrix((np.concatenate(values),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or matrix.nnz == 0:
        raise ValueError("finite nonempty sparse body required")
    return matrix


def _port(nodes: dict, elements: dict, x: float,
          level: tuple) -> tuple[np.ndarray, dict]:
    face_ids = tuple(node for node, p in nodes.items() if abs(p[0]-x) < 1e-14)
    lookup = {node: i for i, node in enumerate(face_ids)}
    metric_nodes = np.zeros((len(face_ids), len(face_ids)))
    points, weights = np.polynomial.legendre.leggauss(3)
    area = 0.
    faces = 0
    for ids in elements.values():
        _, low, high, _, span = solid._brick(ids, nodes)
        side = -1. if abs(low[0]-x) < 1e-14 else 1. if abs(high[0]-x) < 1e-14 else None
        if side is None:
            continue
        faces += 1
        for i, eta in enumerate(points):
            for j, zeta in enumerate(points):
                shape, _ = solid._shape((side, eta, zeta))
                trace = np.zeros(len(face_ids))
                for node, value in zip(ids, shape):
                    if node in lookup:
                        trace[lookup[node]] += value
                    elif abs(value) > 1e-13:
                        raise ValueError("off-face C3D20 interpolation is nonzero")
                weight = span[1]*span[2]/4.*weights[i]*weights[j]
                metric_nodes += weight*np.outer(trace, trace)
                area += weight
    if faces != level[1]*level[2] or abs(area-.04*.06) > 1e-14:
        raise ValueError("complete physical section area required")
    metric = np.kron(metric_nodes, np.eye(3))
    rigid_face = np.zeros((3*len(face_ids), 6))
    for i, node in enumerate(face_ids):
        radius = np.asarray(nodes[node])-np.array((x, 0., .04))
        rigid_face[3*i:3*i+3, :3] = np.eye(3)
        for axis in range(3):
            rigid_face[3*i:3*i+3, 3+axis] = np.cross(np.eye(3)[axis], radius)
    gram = rigid_face.T @ metric @ rigid_face
    if np.min(np.linalg.eigvalsh(gram)) <= 0.:
        raise ValueError("section rigid modes are unrepresentable")
    local = np.linalg.solve(gram, rigid_face.T @ metric)
    full = np.zeros((6, 3*len(nodes)))
    for i, node in enumerate(face_ids):
        full[:, 3*(node-1):3*node] = local[:, 3*i:3*i+3]
    error = float(np.max(np.abs(local @ rigid_face - np.eye(6))))
    if error > 1e-12:
        raise ValueError("refined section projection fails rigid identity")
    return full, {"area_m2": area, "faces": faces, "face_nodes": len(face_ids),
                  "rigid_projection_error": error}


def _control(nodes: dict, matrix: sparse.csr_matrix, root: np.ndarray,
             tip: np.ndarray, prior: dict) -> dict:
    """The coarse mesh must reproduce both independent old assembly and port."""
    _, old = solid.build_reference()
    old_k, _ = continuum.assemble_body(old)
    _, index, _, _, _ = continuum.layout(old)
    old_by_xyz = {tuple(round(v, 12) for v in old["physical_nodes"][node]): node
                  for node in old["beam_nodes"]}
    if len(old_by_xyz) != len(nodes):
        raise ValueError("coarse reference node inventory differs")
    slots = np.array([index[old_by_xyz[tuple(round(v, 12) for v in nodes[node])]]+axis
                      for node in nodes for axis in range(3)])
    old_body = old_k[slots, :][:, slots].tocsr()
    difference = (matrix-old_body).tocsr()
    stiffness_relative = float(np.max(np.abs(difference.data)) /
                               max(np.max(np.abs(old_body.data)), 1.)) if difference.nnz else 0.
    old_root, _ = parent._port(old, 0.)
    old_tip, _ = parent._port(old, 1.)
    old_beam_ids = tuple(sorted(old["beam_nodes"]))
    position = {node: i for i, node in enumerate(old_beam_ids)}
    port_slots = np.array([3*position[old_by_xyz[tuple(round(v, 12) for v in nodes[node])]]+axis
                           for node in nodes for axis in range(3)])
    root_error = float(np.max(np.abs(root-old_root[:, port_slots])))
    tip_error = float(np.max(np.abs(tip-old_tip[:, port_slots])))
    if max(stiffness_relative, root_error, tip_error) > 1e-10:
        raise ValueError("new refinement machinery fails exact coarse oracle")
    return {"stiffness_entry_relative_max": stiffness_relative,
            "root_port_max_difference": root_error,
            "tip_port_max_difference": tip_error}


def _solve(nodes: dict, matrix: sparse.csr_matrix, root: np.ndarray,
           tip: np.ndarray) -> dict:
    torque = np.array((0., 0., 0., TORQUE_NM, 0., 0.))
    load = tip.T @ torque
    xyz = np.array([nodes[node] for node in nodes])
    if np.max(np.abs(parent._wrench(xyz, load)-torque)) > 1e-9:
        raise ValueError("refined section torque wrench changed")
    system = sparse.bmat([[matrix, sparse.csr_matrix(root.T)],
                          [sparse.csr_matrix(root), None]], format="csc")
    rhs = np.r_[load, np.zeros(6)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        state = spsolve(system, rhs)
    n = len(load)
    q = state[:n]
    residual = np.asarray(system @ state-rhs).ravel()
    reaction = np.asarray(matrix @ q-load).ravel()
    work = float(load @ q)
    internal = float(q @ (matrix @ q))
    root_wrench = parent._wrench(xyz, reaction)
    if (not np.isfinite(state).all() or np.linalg.norm(residual)/TORQUE_NM > 1e-8 or
            np.max(np.abs(root @ q)) > 1e-9 or
            abs(internal-work)/max(abs(work), 1e-12) > 1e-8 or
            np.linalg.norm(root_wrench+torque)/TORQUE_NM > 1e-8):
        raise ValueError("refined torsion equilibrium, work, or wrench failed")
    return {"applied_work_J": work, "tip_section_spin_x_rad": float((tip @ q)[3]),
            "equation_ratio": float(np.linalg.norm(residual)/TORQUE_NM),
            "root_gap_max": float(np.max(np.abs(root @ q))),
            "relative_work_error": abs(internal-work)/max(abs(work), 1e-12),
            "support_wrench_about_origin": root_wrench.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("preserve prior refinement output directory")
    if sha256(args.parent.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("exact saved section-port contrast required")
    prior = json.loads(args.parent.read_text(encoding="utf-8"))
    pools = threadpool_info()
    if (prior["status"] != "DEVELOPMENT_SECTION_PORT_CONTRAST_NOT_QUALIFIED" or
            psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(p["num_threads"] != 1 for p in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT / "src" / "anyfesolver"):
        raise ValueError("saved parent or effective source/runtime changed")
    args.output_dir.mkdir(parents=True)
    analytical = native.rectangular_section()
    exact_work = TORQUE_NM**2 / analytical["stiffness"][3]
    for level in LEVELS:
        started = time.perf_counter()
        nodes, elements = _mesh(*level)
        matrix = _body(nodes, elements, level)
        root, root_meta = _port(nodes, elements, 0., level)
        tip, tip_meta = _port(nodes, elements, 1., level)
        control = _control(nodes, matrix, root, tip, prior) if level == LEVELS[0] else None
        result = _solve(nodes, matrix, root, tip)
        if level == LEVELS[0] and abs(result["applied_work_J"]-
                                  prior["cases"]["port/section_dual"]["applied_work_J"])/exact_work > 1e-10:
            raise ValueError("coarse solved section-port control differs")
        record = {"status": "DEVELOPMENT_TORSION_REFINEMENT_NOT_QUALIFIED",
                  "identity": "R8_BL5_MX_C3D20_TORSION_REFINEMENT_DEV_V1",
                  "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                  "parent_sha256": PARENT_SHA256, "level": level,
                  "nodes": len(nodes), "elements": len(elements), "matrix_nnz": matrix.nnz,
                  "root_port": root_meta, "tip_port": tip_meta,
                  "coarse_independent_control": control,
                  "analytical_Saint_Venant_work_J": exact_work,
                  "relative_to_analytical": result["applied_work_J"]/exact_work-1.,
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
        print(json.dumps({"level": level, "relative_to_analytical": record["relative_to_analytical"],
                          "wall_seconds": record["level_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
