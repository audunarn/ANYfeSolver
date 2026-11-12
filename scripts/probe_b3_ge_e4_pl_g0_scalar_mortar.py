"""G0 F03-S scalar trace inventory; not vector/rotation mortar qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from anyfesolver.ge_beam3_curved_reference import p2_shape_functions


def trace(s: float, beam_elements: int, shell_elements: int) -> tuple[np.ndarray, np.ndarray]:
    beam = np.zeros(2 * beam_elements + 1)
    cell = min(int(s * beam_elements), beam_elements - 1)
    xi = 2 * (s * beam_elements - cell) - 1
    beam[2 * cell:2 * cell + 3] = p2_shape_functions(float(xi))
    shell = np.zeros(shell_elements + 1)
    cell = min(int(s * shell_elements), shell_elements - 1)
    local = s * shell_elements - cell
    shell[cell:cell + 2] = (1. - local, local)
    return beam, shell


def multiplier(s: float, knots: np.ndarray, family: str) -> np.ndarray:
    cell = min(int(np.searchsorted(knots, s, side="right") - 1), len(knots) - 2)
    cell = max(0, cell)
    local = (s - knots[cell]) / (knots[cell + 1] - knots[cell])
    if family == "P0":
        result = np.zeros(len(knots) - 1)
        result[cell] = 1.
    else:
        result = np.zeros(len(knots))
        if family == "P1":
            result[cell:cell + 2] = (1. - local, local)
        elif family == "DUAL":
            result[cell:cell + 2] = (2. - 3. * local, 3. * local - 1.)
        else:
            raise ValueError(family)
    return result


def rank(values: np.ndarray) -> int:
    singular = np.linalg.svd(values, compute_uv=False)
    return int(np.sum(singular > 1e-10 * singular[0])) if singular.size else 0


def row(beam_elements: int, shell_elements: int, family: str, grid: str,
        samples: list[tuple[float, float]], union: np.ndarray) -> dict:
    if grid == "beam":
        knots = np.linspace(0., 1., beam_elements + 1)
    elif grid == "shell":
        knots = np.linspace(0., 1., shell_elements + 1)
    else:
        knots = union
    n_b, n_s = 2 * beam_elements + 1, shell_elements + 1
    count = len(knots) - 1 if family == "P0" else len(knots)
    b = np.zeros((count, n_b + n_s))
    mass = np.zeros((count, count))
    h = np.zeros((n_b + n_s, n_b + n_s))
    g = np.zeros_like(h)
    means = np.zeros(count)
    for s, weight in samples:
        beam, shell = trace(s, beam_elements, shell_elements)
        difference = np.r_[beam, -shell]
        mu = multiplier(s, knots, family)
        b += weight * np.outer(mu, difference)
        mass += weight * np.outer(mu, mu)
        h[:n_b, :n_b] += weight * np.outer(beam, beam)
        h[n_b:, n_b:] += weight * np.outer(shell, shell)
        g += weight * np.outer(difference, difference)
        means += weight * mu
    # Exact correspondence after the beam axis is offset by 0.2.
    xb = np.linspace(0., 1., n_b)
    xs = np.linspace(0., 1., n_s)
    patch_error = max(float(np.linalg.norm(b @ np.r_[np.full(n_b, .2), np.zeros(n_s)] - .2 * means)),
                      float(np.linalg.norm(b @ np.r_[np.full(n_b, 1.2), np.ones(n_s)] - .2 * means)),
                      float(np.linalg.norm(b @ np.r_[xb + .2, xs] - .2 * means)))
    # Cholesky whitening gives the interface operator in L2 trace/multiplier norms.
    lh = linalg.cholesky(h, lower=True)
    lm = linalg.cholesky(mass, lower=True)
    whitened = linalg.solve_triangular(lm, b, lower=True) @ linalg.solve_triangular(
        lh.T, np.eye(n_b + n_s), lower=False)
    singular = np.linalg.svd(whitened, compute_uv=False)
    row_rank = rank(whitened)
    physical_rank = rank(g)
    return {"beam_elements": beam_elements, "shell_elements": shell_elements,
            "family": family, "grid": grid, "multiplier_dimension": count,
            "trace_dimension": n_b + n_s, "row_rank": row_rank,
            "row_rank_loss": count - row_rank, "difference_rank": physical_rank,
            "unresolved_difference_modes": physical_rank - row_rank,
            "beta_nonzero": float(singular[row_rank - 1]) if row_rank else 0.,
            "patch_error": patch_error}


def run() -> dict:
    roots, weights = np.polynomial.legendre.leggauss(4)
    records = []
    cases = []
    for n in (4, 8, 16):
        for ratio in (1, 2, 4, 8):
            cases.append((n, n * ratio))
            if ratio > 1:
                cases.append((n * ratio, n))
    for shell_elements, beam_elements in cases:
        union = np.unique(np.r_[np.linspace(0., 1., beam_elements + 1),
                                np.linspace(0., 1., shell_elements + 1)])
        samples = []
        for start, end in zip(union[:-1], union[1:]):
            samples.extend((float((start + end) / 2 + (end - start) * root / 2),
                            float((end - start) * weight / 2))
                           for root, weight in zip(roots, weights))
        for family, grid in (("P0", "union"), ("P1", "shell"), ("P1", "beam"),
                             ("P1", "union"), ("DUAL", "union")):
            records.append(row(beam_elements, shell_elements, family, grid, samples, union))
    passed = all(r["patch_error"] <= 1e-12 and r["row_rank"] <= r["difference_rank"]
                 and np.isfinite(r["beta_nonzero"]) for r in records)
    summary = {"case": "G0_F03_SCALAR_TRACE_V1", "status": "diagnostic_pass" if passed else "failed",
               "geometry": "unit line, beam normal offset 0.2", "quadrature": "4-point Gauss per union interval",
               "rank_relative_threshold": 1e-10, "rows": records,
               "counts": {"meshes": len(cases), "candidate_rows": len(records),
                          "row_rank_deficient": sum(r["row_rank_loss"] > 0 for r in records),
                          "unresolved_modes": sum(r["unresolved_difference_modes"] > 0 for r in records),
                          "largest_patch_error": max(r["patch_error"] for r in records)}}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["counts"],
                      "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "diagnostic_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
