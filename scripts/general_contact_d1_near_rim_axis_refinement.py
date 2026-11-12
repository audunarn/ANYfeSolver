"""Separate x/front and y/rim refinement of the independent D1 Q1 oracle.

This is a source-only development discriminator. It does not run ANYfeSolver's
coupled shell model or establish pressure-space qualification by itself.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from docs.d1_review_support import d1_rim_oracle as oracle


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _errors(bins, reference):
    difference = bins[:9]-reference[:9]
    return {
        "bin_l1_relative": float(np.abs(difference).sum()
                                 / np.abs(reference[:9]).sum()),
        "bin_l2_relative": float(np.linalg.norm(difference)
                                 / np.linalg.norm(reference[:9])),
        "rim_force_relative": float(abs(bins[8].sum()-reference[8].sum())
                                    / abs(reference[8].sum())),
        "empty_row_force_n": float(bins[9].sum()),
    }


def _active_compliance(nx, ynodes, pressure):
    """Diagnose the source-only active pressure-to-gap response."""
    ny = len(ynodes)-1
    kx = oracle.stiff1d(nx)
    mx = oracle.mass1d(nx)
    ky = oracle.stiff1d(ny, ynodes)
    my = oracle.mass1d(ny, nodes=ynodes)
    stiffness = oracle.D*(
        sparse.kron(sparse.csr_matrix(my), sparse.csr_matrix(kx))
        +sparse.kron(sparse.csr_matrix(ky), sparse.csr_matrix(mx)))
    bx = oracle.mass1d(nx)
    by = oracle.mass1d(ny, oracle.YR, ynodes)
    work = sparse.kron(sparse.csr_matrix(by), sparse.csr_matrix(bx)).tocsr()
    support = np.asarray(work.sum(axis=1)).ravel()
    active = np.flatnonzero(np.asarray(pressure).reshape(-1) > 1e-9)
    if len(active) == 0 or np.any(support[active] <= 0):
        raise AssertionError("active pressure lacks positive support")
    free = np.ones((nx+1)*(ny+1), dtype=bool)
    free[np.arange(0, len(free), nx+1)] = False
    factor = sparse_linalg.splu(stiffness.tocsr()[free][:, free].tocsc())
    active_work = work[active][:, free]
    response = np.asarray(active_work @ factor.solve(active_work.T.toarray()))
    response = 0.5*(response+response.T)
    diagonal = np.diag(response)
    if np.any(diagonal <= 0):
        raise AssertionError("active pressure compliance has nonpositive diagonal")
    scaled = response/np.sqrt(np.outer(diagonal, diagonal))
    eigenvalues = np.linalg.eigvalsh(scaled)
    return {
        "active_modes": len(active),
        "minimum_support_m2": float(np.min(support[active])),
        "maximum_support_m2": float(np.max(support[active])),
        "unit_diagonal_minimum_eigenvalue": float(eigenvalues[0]),
        "unit_diagonal_maximum_eigenvalue": float(eigenvalues[-1]),
        "unit_diagonal_eigenvalue_ratio": float(eigenvalues[0]/eigenvalues[-1]),
    }


def _physical_gap(solution):
    """Minimum Q1 obstacle gap at grid corners and the clipped rim."""
    y = np.asarray(solution["yn"])
    displacement = np.asarray(solution["w"])
    interior = displacement[y <= oracle.YR]
    lower = int(np.searchsorted(y, oracle.YR, side="right")-1)
    if y[lower] == oracle.YR:
        rim = displacement[lower]
    else:
        fraction = (oracle.YR-y[lower])/(y[lower+1]-y[lower])
        rim = (1-fraction)*displacement[lower]+fraction*displacement[lower+1]
    # A bilinear Q1 field has its minimum over a clipped rectangular cell at
    # one of these vertices. The rim values cover the cut cells.
    return float(oracle.G0+min(np.min(interior), np.min(rim)))


def main(output):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    root = Path("reports")
    base_path = root / (
        "general_contact_d1_16x8_near_rim_guarded_no_checkpoint_"
        "comparison_identity_dev_20260928.json")
    both_path = root / (
        "general_contact_d1_16x8_near_rim_guarded_no_checkpoint_"
        "source_refinement_dev_20260928.json")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    both = json.loads(both_path.read_text(encoding="utf-8"))
    oracle_sha = _hash(Path(oracle.__file__))
    if (base["status"] != "development_comparison_only"
            or base["mesh"] != [16, 8]
            or both["coarse_grid"] != [16, 8]
            or both["refined_grid"] != [32, 16]
            or base["oracle_sha256"] != oracle_sha
            or both["oracle_sha256"] != oracle_sha
            or both["saved_comparison_sha256"] != _hash(base_path)):
        raise ValueError("saved near-rim reference identities differ")
    y = np.asarray(base["source_y_knots_m"], dtype=float)
    refined_y = np.sort(np.r_[y, 0.5*(y[:-1]+y[1:])])
    np.testing.assert_allclose(refined_y, both["refined_y_knots_m"],
                               rtol=0.0, atol=0.0)
    edges_x = np.linspace(0.0, 1.0, 9)
    edges_y = np.r_[np.linspace(0.0, 0.9, 10), 1.0]
    reference = np.asarray(base["fv_bins_n"], dtype=float)
    base_bins = np.asarray(base["source_only_bins_n"], dtype=float)
    both_bins = np.asarray(both["refined_bins_n"], dtype=float)
    if any(field.shape != (10, 8) for field in
           (reference, base_bins, both_bins)):
        raise ValueError("saved physical-bin shapes differ")
    cases = {}
    bins = {(16, 8): base_bins, (32, 16): both_bins}
    for nx, ny, ynodes in ((16, 8, y), (32, 8, y),
                            (16, 16, refined_y), (32, 16, refined_y)):
        solution = oracle.solve_q1(nx, ny, ynodes=ynodes)
        field = oracle.q1_bins(solution, edges_x, edges_y)
        if (nx, ny) in bins:
            np.testing.assert_allclose(field, bins[nx, ny], rtol=1e-10,
                                       atol=1e-12)
        if (solution["min_wgap"] < -1e-9
                or solution["eq"] > 1e-7
                or solution["comp"] > 1e-7
                or np.min(solution["p"]) < -1e-10):
            raise AssertionError("independent Q1 solution failed KKT checks")
        if abs(field.sum()-solution["reaction"]) > 1e-10:
            raise AssertionError("physical bins lose the integrated reaction")
        bins[nx, ny] = field
        cases[f"{nx}x{ny}"] = {
            "reaction_n": solution["reaction"],
            "equilibrium_n": solution["eq"],
            "complementarity_pa_m": solution["comp"],
            "minimum_normalized_gap_m": solution["min_wgap"],
            "minimum_physical_q1_gap_m": _physical_gap(solution),
            "active_pressure_modes": int(np.count_nonzero(solution["p"] > 1e-9)),
            "active_compliance": _active_compliance(nx, ynodes, solution["p"]),
            "fv_error": _errors(field, reference),
            "bins_n": field.tolist(),
        }
    difference_x = bins[32, 8]-bins[16, 8]
    difference_y = bins[16, 16]-bins[16, 8]
    difference_both = bins[32, 16]-bins[16, 8]
    scale = float(np.linalg.norm(bins[32, 16][:9]))
    record = {
        "status": "development_independent_q1_axis_refinement",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "oracle_sha256": oracle_sha,
        "base_comparison_sha256": _hash(base_path),
        "both_axis_refinement_sha256": _hash(both_path),
        "common_bins": [8, 9],
        "obstacle_rim_y_m": oracle.YR,
        "base_y_knots_m": y.tolist(),
        "refined_y_knots_m": refined_y.tolist(),
        "base_16x8_fv_error": _errors(base_bins, reference),
        "both_32x16_fv_error": _errors(both_bins, reference),
        "all_four_source_only_cases": cases,
        "x_change_bin_l2_relative": float(np.linalg.norm(difference_x[:9])/scale),
        "y_change_bin_l2_relative": float(np.linalg.norm(difference_y[:9])/scale),
        "both_change_bin_l2_relative": float(np.linalg.norm(difference_both[:9])/scale),
        "axis_interaction_bin_l2_relative": float(np.linalg.norm(
            (difference_both-difference_x-difference_y)[:9])/scale),
        "elapsed_seconds": time.perf_counter()-started,
    }
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "four_corner_errors": {key: value["fv_error"]
                                for key, value in cases.items()},
        "x_change": record["x_change_bin_l2_relative"],
        "y_change": record["y_change_bin_l2_relative"],
        "interaction": record["axis_interaction_bin_l2_relative"],
        "elapsed_seconds": record["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python general_contact_d1_near_rim_axis_refinement.py OUTPUT")
    main(Path(sys.argv[1]))
