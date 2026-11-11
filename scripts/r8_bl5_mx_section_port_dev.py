"""Saved C3D20 Mx beam-only root/traction contrast; development evidence only.

The face port is the area-L2 projection of displacement onto six rigid section
fields. Its transpose distributes a work-conjugate section wrench. It is a
declared diagnostic idealization, not a replacement B3-GE section law.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import warnings

import anyfesolver
import numba
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from threadpoolctl import threadpool_info

from scripts import r8_bl5_ccx_weak_p0_dev as deck
from scripts import r8_bl5_mx_beam_body_dev as prior_code
from scripts import r8_bl5_matched_native_reference as native
from scripts import r8_bl5_released_director_reference as solid
from scripts import r8_bl5_weak_continuum_reference as continuum


ROOT = Path(__file__).resolve().parents[1]
PRIOR_SHA256 = "03731cba008cce0d50a23c516ee4fb90f99ced4d37d0916bdF427A8FA73EFAE4".lower()
TORQUE_NM = 10.


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _port(model: dict, x: float) -> tuple[np.ndarray, dict]:
    """Area-L2 pose projection from exact quadratic end-face interpolation."""
    beam_ids = tuple(sorted(model["beam_nodes"]))
    beam_index = {node: i for i, node in enumerate(beam_ids)}
    xyz = model["physical_nodes"]
    mass = np.zeros((len(beam_ids), len(beam_ids)))
    points, weights = np.polynomial.legendre.leggauss(3)
    area = 0.
    faces = 0
    for eid in model["beam_elements"]:
        ids, low, high, center, span = solid._brick(model["elements"][eid], xyz)
        side = -1. if abs(low[0] - x) < 1e-14 else 1. if abs(high[0] - x) < 1e-14 else None
        if side is None:
            continue
        faces += 1
        local = np.array([beam_index[node] for node in ids])
        coords = np.array([xyz[node] for node in ids])
        for i, eta in enumerate(points):
            for j, zeta in enumerate(points):
                shape, grad = solid._shape((side, eta, zeta))
                jac = coords.T @ grad
                measure = float(np.linalg.norm(np.cross(jac[:, 1], jac[:, 2])))
                if not np.isfinite(measure) or measure <= 0.:
                    raise ValueError("invalid C3D20 physical end-face measure")
                weight = measure * weights[i] * weights[j]
                mass[np.ix_(local, local)] += weight * np.outer(shape, shape)
                area += weight
    if faces != 2 or abs(area - .04 * .06) > 1e-14:
        raise ValueError("exact two-face beam section and area required")
    metric = np.kron(mass, np.eye(3))
    rigid = np.zeros((3 * len(beam_ids), 6))
    for i, node in enumerate(beam_ids):
        position = np.asarray(xyz[node]) - np.array((x, 0., .04))
        rigid[3*i:3*i+3, :3] = np.eye(3)
        for axis in range(3):
            rigid[3*i:3*i+3, 3+axis] = np.cross(np.eye(3)[axis], position)
    gram = rigid.T @ metric @ rigid
    if np.min(np.linalg.eigvalsh(gram)) <= 0.:
        raise ValueError("physical face cannot represent six rigid section modes")
    projection = np.linalg.solve(gram, rigid.T @ metric)
    error = float(np.max(np.abs(projection @ rigid - np.eye(6))))
    if error > 1e-12:
        raise ValueError("section-port rigid-motion identity failed")
    return projection, {"area_m2": area, "faces": faces,
                        "rigid_projection_error": error,
                        "section_center_m": [x, 0., .04]}


def _wrench(beam_xyz: np.ndarray, force: np.ndarray) -> np.ndarray:
    forces = force.reshape(-1, 3)
    return np.r_[forces.sum(axis=0), np.cross(beam_xyz, forces).sum(axis=0)]


def _solve(body: sparse.csr_matrix, load: np.ndarray, root: np.ndarray | None,
           root_slots: np.ndarray, beam_xyz: np.ndarray, tip_port: np.ndarray) -> dict:
    n = len(load)
    if root is None:
        free = np.setdiff1d(np.arange(n), root_slots)
        q = np.zeros(n)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            q[free] = spsolve(body[free, :][:, free], load[free])
        reaction = np.asarray(body @ q - load).ravel()
        root_gap = float(np.max(np.abs(q[root_slots])))
        equation = reaction[free]
        root_kind = "all_face_nodes_fixed"
    else:
        system = sparse.bmat([[body, sparse.csr_matrix(root.T)],
                              [sparse.csr_matrix(root), None]], format="csc")
        rhs = np.r_[load, np.zeros(6)]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            state = spsolve(system, rhs)
        q = state[:n]
        reaction = np.asarray(body @ q - load).ravel()
        root_gap = float(np.max(np.abs(root @ q)))
        equation = np.asarray(system @ state - rhs).ravel()
        root_kind = "area_L2_section_pose_fixed"
    if not np.isfinite(q).all() or not np.isfinite(reaction).all():
        raise ValueError("nonfinite section-port response")
    work = float(load @ q)
    strain = float(q @ (body @ q))
    tip_wrench = _wrench(beam_xyz, load)
    root_wrench = _wrench(beam_xyz, reaction)
    if (np.linalg.norm(equation) / TORQUE_NM > 1e-8 or root_gap > 1e-9 or
            abs(strain-work)/max(abs(work), 1e-12) > 1e-8 or
            np.linalg.norm(tip_wrench + root_wrench)/TORQUE_NM > 1e-8):
        raise ValueError("equilibrium, root, wrench, or work check failed")
    return {"root": root_kind, "applied_work_J": work,
            "section_tip_spin_x_rad": float((tip_port @ q)[3]),
            "free_equation_ratio": float(np.linalg.norm(equation)/TORQUE_NM),
            "root_gap_max": root_gap,
            "relative_internal_work_error": abs(strain-work)/max(abs(work), 1e-12),
            "tip_wrench_about_origin": tip_wrench.tolist(),
            "root_wrench_about_origin": root_wrench.tolist()}


def evaluate(saved: Path, earlier: Path) -> dict:
    if _sha(earlier) != PRIOR_SHA256:
        raise ValueError("exact prior beam-only result required")
    prior = json.loads(earlier.read_text(encoding="utf-8"))
    if prior["status"] != "DEVELOPMENT_BODY_DISCRIMINATOR_NOT_QUALIFIED":
        raise ValueError("completed prior body discriminator required")
    pools = threadpool_info()
    if (psutil.Process().cpu_affinity() != [0] or numba.config.DISABLE_JIT != 1 or
            numba.get_num_threads() != 1 or not pools or
            any(p["num_threads"] != 1 for p in pools) or
            Path(anyfesolver.__file__).resolve().parent != ROOT / "src" / "anyfesolver"):
        raise ValueError("effective CPU/thread/JIT/source route changed")
    metadata = json.loads((saved / "metadata.json").read_text(encoding="utf-8"))
    deck_bytes, expected = deck.deck("mx")
    _, model = deck.solid.build_reference()
    if (metadata != json.loads(json.dumps(expected, allow_nan=False)) or
            (saved / "weak.inp").read_bytes() != deck_bytes or
            metadata["model_sha256"] != model["model_sha256"] or
            _sha(saved / "weak.inp") != prior["input_sha256"]["saved_deck"] or
            _sha(saved / "metadata.json") != prior["input_sha256"]["saved_metadata"]):
        raise ValueError("saved Mx geometry/load/deck identity changed")
    k, _ = continuum.assemble_body(model)
    _, index, _, _, _ = continuum.layout(model)
    beam_ids = tuple(sorted(model["beam_nodes"]))
    slots = np.array([index[node]+axis for node in beam_ids for axis in range(3)])
    beam_xyz = np.array([model["physical_nodes"][node] for node in beam_ids])
    body = k[slots, :][:, slots].tocsr()
    old_load = np.asarray(metadata["physical_tip_load_N"], dtype=float)[slots]
    root_port, root_meta = _port(model, 0.)
    tip_port, tip_meta = _port(model, 1.)
    root_slots = np.array([3*i+axis for i, node in enumerate(beam_ids)
                           if abs(model["physical_nodes"][node][0]) < 1e-14
                           for axis in range(3)])
    section_wrench = np.array((0., 0., 0., TORQUE_NM, 0., 0.))
    port_load = tip_port.T @ section_wrench
    old_wrench = _wrench(beam_xyz, old_load)
    port_wrench = _wrench(beam_xyz, port_load)
    if (np.max(np.abs(old_wrench-section_wrench)) > 1e-9 or
            np.max(np.abs(port_wrench-section_wrench)) > 1e-9 or
            np.max(np.abs(root_port[:, root_slots])) == 0.):
        raise ValueError("old and port loads must have the same physical wrench")
    cases = {}
    for root_name, root in (("face", None), ("port", root_port)):
        for load_name, load in (("saved_traction", old_load), ("section_dual", port_load)):
            cases[f"{root_name}/{load_name}"] = _solve(
                body, load, root, root_slots, beam_xyz, tip_port)
    old = cases["face/saved_traction"]["applied_work_J"]
    original = prior["resolved_beam_only"]["applied_work_J"]
    if abs(old-original)/max(abs(original), 1e-12) > 1e-10:
        raise ValueError("exact prior resolved beam-only control changed")
    section = native.rectangular_section()
    analytical_twist = TORQUE_NM / (section["stiffness"][3])  # L = 1 m
    analytical_work = TORQUE_NM * analytical_twist
    native_work = float(prior["native_beam_only"]["applied_work_J"])
    port_work = cases["port/section_dual"]["applied_work_J"]
    return {"status": "DEVELOPMENT_SECTION_PORT_CONTRAST_NOT_QUALIFIED",
            "identity": "R8_BL5_MX_SECTION_PORT_CONTRAST_DEV_V3",
            "source_sha256": _sha(Path(__file__)),
            "input_sha256": {"saved_deck": _sha(saved / "weak.inp"),
                             "saved_metadata": _sha(saved / "metadata.json"),
                             "earlier_beam_only": _sha(earlier)},
            "effective_runtime": {"cpu_affinity": psutil.Process().cpu_affinity(),
                                  "numba_disable_jit": int(numba.config.DISABLE_JIT),
                                  "numba_threads": int(numba.get_num_threads()),
                                  "threadpools": pools,
                                  "anyfesolver_import": str(Path(anyfesolver.__file__).resolve())},
            "root_port": root_meta, "tip_port": tip_meta,
            "saved_traction_wrench": old_wrench.tolist(),
            "section_dual_wrench": port_wrench.tolist(),
            "load_difference_zero_rigid_wrench": float(np.max(np.abs(old_wrench-port_wrench))),
            "load_vector_l2_difference_N": float(np.linalg.norm(old_load-port_load)),
            "load_vector_relative_difference": float(np.linalg.norm(old_load-port_load)/
                                                     max(np.linalg.norm(old_load), 1e-12)),
            "load_vector_max_difference_N": float(np.max(np.abs(old_load-port_load))),
            "cases": cases,
            "prior_native_beam_only_work_J": native_work,
            "prismatic_Saint_Venant_check": {
                "declared_scope": "1 m prismatic elastic bar, torque 10 N m, free end warping",
                "J_m4": section["J"], "J_tail_absolute_bound_m4": section["J_tail_bound"],
                "G_Pa": section["stiffness"][3] / section["J"],
                "analytical_tip_spin_rad": analytical_twist,
                "analytical_applied_work_J": analytical_work,
                "native_relative_work_difference": (native_work-analytical_work)/analytical_work,
                "resolved_port_relative_work_difference": (port_work-analytical_work)/analytical_work},
            "limitations": ["One saved linear C3D20 beam-only geometry and one torque resultant.",
                            "The area-L2 section port is a declared diagnostic idealization, not the native retained-coordinate section law.",
                            "No coupled transfer, external execution, full mesh ratio, or F03 acceptance follows."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior development output")
    result = evaluate(args.saved, args.earlier)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "work_J": {name: row["applied_work_J"]
                                 for name, row in result["cases"].items()}}))


if __name__ == "__main__":
    main()
