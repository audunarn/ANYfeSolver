"""Bounded off-grid BL5 P0/P1 transfer screen; not formal F03 assessment.

The two physical grids compare 1:1, 1:2 and 2:1 shell:beam trace counts
under the same supports, section, eccentricity and physical Fy/Mx loads.
Completion, balance, work and sampled gaps are recorded without treating a
smaller gap or one ratio as a multiplier-space qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import anyfesolver
from anyfesolver import stiffened_shell
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits
from anyfesolver._stiffened_shell_l1_assembly_experimental import compile_lines
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical

from b3_e4_f03_bl5_joint_growth_dev import _case
from test_stiffened_shell_l1_assembly_experimental import (
    _bl5_joint_mesh_definition,
)


ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ((4, 4), (4, 8), (8, 4),
           (8, 8), (8, 16), (16, 8))
LOADS = {"fy": (0., 100., 0., 0., 0., 0.),
         "mx": (0., 0., 0., 10., 0., 0.)}


def analyze(*, ratio_probe: bool = False):
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("off-grid screen imported another ANYfeSolver")
    report = {
        "status": ("BL5_P1_SEVEN_RATIO_FEASIBILITY_ONLY" if ratio_probe else
                   "BL5_P1_OFFGRID_DEVELOPMENT_ONLY"),
        "source_git": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "product_sha256": sha256((ROOT / "src" / "anyfesolver" /
            "_stiffened_shell_l1_assembly_experimental.py").read_bytes()).hexdigest(),
        "imported_anyfesolver": str(expected),
        "geometry_m": {"length": 1., "width": .2,
                       "shell_thickness": .035,
                       "beam_axis_height": .0675},
        "loads": LOADS, "cases": [],
        "limitations": ("Flat Q4, native B3, reference-linear development "
                        "screen only. Ratios are one small resolution each. "
                        "No physical external comparator, inf-sup/mixed-system "
                        "stability, locking, three-level convergence, or F03 "
                        "acceptance."),
    }
    layouts = (((2, 16), (4, 16), (8, 16), (16, 16),
                (16, 8), (16, 4), (16, 2)) if ratio_probe else LAYOUTS)
    bases = (("p0", "continuous_p1_coarse") if ratio_probe else
             ("p0", "continuous_p1"))
    with threadpool_limits(limits=1):
        report["threadpools"] = threadpool_info()
        if any(pool.get("num_threads", 1) != 1
               for pool in report["threadpools"]):
            raise RuntimeError("numerical thread limit not effective")
        for shell_edges, beam_halfcells in layouts:
            for basis in bases:
                definition = _bl5_joint_mesh_definition(
                    shell_edges, beam_halfcells, 1)
                if basis != "p0":
                    definition = replace(definition, lines=(replace(
                        definition.lines[0], basis=basis),))
                try:
                    owner = stiffened_shell.create_analysis(definition=definition)
                except ValueError as exc:
                    if "independent supported attachment rows required" not in str(exc):
                        raise
                    report["cases"].append({
                        "shell_edges": shell_edges,
                        "beam_halfcells": beam_halfcells,
                        "shell_to_beam_ratio": shell_edges / beam_halfcells,
                        "basis": basis,
                        "status": "REJECTED_DEPENDENT_SUPPORTED_ROWS",
                        "failure_reason": str(exc),
                    })
                    continue
                try:
                    scaled = (owner._C[:, owner._free].toarray()*
                              owner._coordinate_scale[owner._free][None, :] /
                              owner._constraint_scale[:, None])
                    terminal = 101 + beam_halfcells
                    entry = {
                        "shell_edges": shell_edges,
                        "beam_halfcells": beam_halfcells,
                        "shell_to_beam_ratio": shell_edges / beam_halfcells,
                        "basis": basis,
                        "model_identity": owner.preflight()["identity"],
                        "matrix_order": int(owner._A.shape[0]),
                        "matrix_nnz": int(owner._A.nnz),
                        "multiplier_rows": int(owner._C.shape[0]),
                        "bounded_scaled_row_rank": int(
                            np.linalg.matrix_rank(scaled)),
                        "cases": {name: _case(owner, terminal, name, load)
                                  for name, load in LOADS.items()},
                        "factorizations": owner.preflight()[
                            "reference_linear_factorizations"],
                        "status": "COMPILED",
                    }
                    report["cases"].append(entry)
                finally:
                    owner.close()
    return report


def probe_rank():
    """Bounded auditor-scale rank diagnosis, not an interface norm."""
    report = {"status": "BL5_P1_OFFGRID_RANK_DIAGNOSTIC_ONLY",
              "source_git": subprocess.check_output(
                  ("git", "rev-parse", "HEAD"), cwd=ROOT,
                  text=True).strip(),
              "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": [],
              "limitations": "Small dense constraint-row diagnostic only; not inf-sup or stability."}
    with threadpool_limits(limits=1):
        report["threadpools"] = threadpool_info()
        if any(pool.get("num_threads", 1) != 1
               for pool in report["threadpools"]):
            raise RuntimeError("numerical thread limit not effective")
        for shell_edges, beam_halfcells in ((4, 4), (4, 8), (8, 16)):
            definition = _bl5_joint_mesh_definition(
                shell_edges, beam_halfcells, 1)
            owner = stiffened_shell.create_analysis(definition=definition)
            try:
                p1 = replace(definition.lines[0], basis="continuous_p1")
                candidate, _ = compile_lines(
                    (p1,), shell_ids=owner._shell_ids,
                    shell_xyz=owner._shell_xyz,
                    shell_cells=(definition.shell.q4_elements+
                                 definition.shell.s3_elements),
                    thickness=definition.shell.thickness_m,
                    beam_ids=owner._beam_ids,
                    beam_nodes=owner._beam_nodes,
                    beam_parts=owner._beam_parts,
                    shell_count=owner._shell_count,
                    beam_count=owner._beam_count)
                row_scale = np.asarray(
                    abs(candidate)@owner._coordinate_scale).reshape(-1)
                if np.any(row_scale <= 0.):
                    raise ValueError("nonzero P1 row scaling required")
                scaled = (sparse.diags(1./row_scale) @
                          candidate[:, owner._free] @
                          sparse.diags(owner._coordinate_scale[owner._free])
                          ).toarray()
                u, values, _ = linalg.svd(scaled, full_matrices=False)
                tolerance = (values[0]*max(scaled.shape)*
                             np.finfo(values.dtype).eps)
                rank = int(np.count_nonzero(values > tolerance))
                left = u[:, -1]
                leading = np.argsort(-np.abs(left))[:8]
                report["cases"].append({
                    "shell_edges": shell_edges,
                    "beam_halfcells": beam_halfcells,
                    "rows": int(scaled.shape[0]),
                    "columns": int(scaled.shape[1]),
                    "rank": rank,
                    "rank_tolerance": float(tolerance),
                    "largest_singular": float(values[0]),
                    "smallest_singular": float(values[-1]),
                    "left_residual": float(np.linalg.norm(scaled.T@left)),
                    "dominant_left_rows": [
                        {"row": int(row), "coefficient": float(left[row])}
                        for row in leading],
                })
            finally:
                owner.close()
    return report


def probe_coarse_trace():
    """One reference-abscissa P1 rule on the coarser of two uniform traces."""
    expected = (ROOT / "src" / "anyfesolver" / "__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected:
        raise RuntimeError("coarse-trace probe imported another ANYfeSolver")
    report = {"status": "BL5_P1_COARSE_TRACE_DEVELOPMENT_ONLY",
              "source_git": subprocess.check_output(
                  ("git", "rev-parse", "HEAD"), cwd=ROOT,
                  text=True).strip(),
              "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "imported_anyfesolver": str(expected), "cases": [],
              "limitations": ("Uniform straight flat-Q4 traces only; the "
                              "coarse reference knots are a diagnostic, not "
                              "a derived general attachment, stability norm, "
                              "physical comparison or F03 assessment.")}
    with threadpool_limits(limits=1):
        report["threadpools"] = threadpool_info()
        if any(pool.get("num_threads", 1) != 1
               for pool in report["threadpools"]):
            raise RuntimeError("numerical thread limit not effective")
        for shell_edges, beam_halfcells in ((4, 4), (4, 8),
                                            (8, 8), (8, 16)):
            definition = _bl5_joint_mesh_definition(
                shell_edges, beam_halfcells, 1)
            owner = stiffened_shell.create_analysis(definition=definition)
            try:
                knots = min(shell_edges, beam_halfcells)
                positions = np.array([owner._beam_nodes[node][0]
                                      for node in owner._beam_ids])
                frames = np.array([owner._beam_nodes[node][1]
                                   for node in owner._beam_ids])
                state = Mechanical(
                    positions, np.zeros_like(positions), frames,
                    np.tile(np.eye(3), (len(owner._beam_parts), 2, 1, 1)),
                    np.zeros((len(owner._beam_parts), 18)))
                rows, columns, values = [], [], []
                local_samples = []
                for local, slots, weight, edge, alpha in owner._line_samples:
                    response = local.evaluate(
                        np.zeros((2, 6)), state, np.zeros(5))
                    if np.max(np.abs(response.gap)) > 2e-13:
                        raise ValueError("stress-free coarse-trace samples required")
                    position = knots*(edge+alpha)/beam_halfcells
                    index = min(int(np.floor(position)), knots-1)
                    eta = position-index
                    for channel in range(5):
                        for node, coefficient in ((index, 1.-eta),
                                                  (index+1, eta)):
                            row = channel*(knots+1)+node
                            rows.extend([row]*len(slots))
                            columns.extend(slots.tolist())
                            values.extend((weight*coefficient*
                                response.gap_jacobian[channel]).tolist())
                    local_samples.append((response.gap_jacobian, slots))
                constraints = sparse.coo_matrix(
                    (values, (rows, columns)),
                    shape=(5*(knots+1), owner._K.shape[0])).tocsr()
                constraints.sum_duplicates()
                constraints.eliminate_zeros()
                free = owner._free
                row_scale = np.asarray(
                    abs(constraints)@owner._coordinate_scale).reshape(-1)
                scaled = (sparse.diags(1./row_scale) @
                          constraints[:, free] @
                          sparse.diags(owner._coordinate_scale[free])).toarray()
                numerical_rank = int(np.linalg.matrix_rank(scaled))
                reduced = constraints[:, free]
                matrix = sparse.bmat([
                    [owner._K[free, :][:, free], reduced.T],
                    [reduced, None]], format="csc")
                factor = splu(matrix)
                answers = {}
                terminal = 101+beam_halfcells
                tip_position = owner._beam_nodes[terminal][0]
                for name, wrench in LOADS.items():
                    load = np.zeros(owner._K.shape[0])
                    start = owner._shell_count+6*owner._beam_index[terminal]
                    load[start:start+6] = wrench
                    rhs = np.r_[load[free], np.zeros(constraints.shape[0])]
                    solution = factor.solve(rhs)
                    solution += factor.solve(rhs-matrix@solution)
                    q = np.zeros(owner._K.shape[0])
                    q[free] = solution[:len(free)]
                    lam = solution[len(free):]
                    residual = np.asarray(
                        owner._K@q+constraints.T@lam-load).reshape(-1)
                    sampled = np.array([jac@q[slots]
                                        for jac, slots in local_samples])
                    physical = residual[:owner._shell_count+
                                        6*len(owner._beam_ids)].reshape(-1, 6)
                    points = np.vstack((owner._shell_xyz, positions))
                    reaction_force = physical[:, :3].sum(axis=0)
                    reaction_moment = (physical[:, 3:]+np.cross(
                        points, physical[:, :3])).sum(axis=0)
                    expected_force = np.array(wrench[:3])
                    expected_moment = (np.array(wrench[3:])+
                                       np.cross(tip_position, expected_force))
                    applied_work = float(load@q)
                    internal_work = float(q@(owner._K@q))
                    answers[name] = {
                        "applied_work_j": applied_work,
                        "work_difference_j": applied_work-internal_work,
                        "free_equilibrium_max": float(np.max(
                            np.abs(residual[free]))),
                        "weak_gap_max": float(np.max(np.abs(
                            constraints@q))),
                        "sampled_translation_gap_max_m": float(np.max(
                            np.abs(sampled[:, :3]))),
                        "sampled_director_gap_max": float(np.max(
                            np.abs(sampled[:, 3:]))),
                        "force_balance_max_n": float(np.max(np.abs(
                            reaction_force+expected_force))),
                        "moment_balance_max_nm": float(np.max(np.abs(
                            reaction_moment+expected_moment))),
                    }
                report["cases"].append({
                    "shell_edges": shell_edges,
                    "beam_halfcells": beam_halfcells,
                    "knot_intervals": knots,
                    "rows": int(constraints.shape[0]),
                    "numerical_rank": numerical_rank,
                    "matrix_order": int(matrix.shape[0]),
                    "matrix_nnz": int(matrix.nnz),
                    "cases": answers,
                })
            finally:
                owner.close()
    return report


def main(output: Path, *, rank_only: bool = False,
         coarse_probe: bool = False, ratio_probe: bool = False):
    if output.exists():
        raise FileExistsError("preserve prior off-grid evidence")
    if sum((rank_only, coarse_probe, ratio_probe)) > 1:
        raise ValueError("one diagnostic mode per execution")
    report = (probe_rank() if rank_only else
              probe_coarse_trace() if coarse_probe else
              analyze(ratio_probe=ratio_probe))
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "models": len(report["cases"]),
                      "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--coarse-probe", action="store_true")
    parser.add_argument("--ratio-probe", action="store_true")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.output, rank_only=args.rank_only,
         coarse_probe=args.coarse_probe,
         ratio_probe=args.ratio_probe)
