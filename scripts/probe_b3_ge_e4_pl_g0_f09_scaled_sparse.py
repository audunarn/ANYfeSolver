"""Bounded sparse-solve screen on the preserved F09 mixed elbow, not admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from scripts import probe_b3_ge_e4_pl_g0_f09_mixed_elbow_conditioning as elbow


def _case(area: float, oracle: dict) -> dict:
    section = LinearSection.isotropic(
        E=1., nu=0., area=area, Iy=1., Iz=1., J=1., ky=1., kz=1.)
    terminal = np.array((area / 2, area / 3, area / 5,
                         .25, -.125, .375))
    load = np.zeros(18)
    load[12:] = terminal
    old_u, matrix, reduced = elbow._float_graph(elbow._members(section), load)
    old_tip = np.asarray(oracle["binary64_sparse_tip"])
    if not np.array_equal(old_u[12:], old_tip):
        raise ArithmeticError("preserved default sparse response changed")
    exact_tip = np.asarray(oracle["exact_rounded_matrix_tip"])
    physical_tip = np.asarray(oracle["physical_force_method_tip"])
    diagonal = reduced.diagonal()
    if not (np.isfinite(diagonal).all() and np.all(diagonal > 0.)):
        raise ArithmeticError("positive finite diagonal required")
    variants = []
    for scaled in (False, True):
        coordinate_scale = (1. / np.sqrt(diagonal)
                            if scaled else np.ones(len(diagonal)))
        d = sparse.diags(coordinate_scale, format="csc")
        system = (d @ reduced @ d).tocsc()
        rhs = coordinate_scale * load[6:]
        for ordering in ("NATURAL", "COLAMD", "MMD_AT_PLUS_A"):
            for threshold in (0., 1.):
                variant = {"scaled": scaled, "ordering": ordering,
                           "diag_pivot_thresh": threshold}
                try:
                    free = coordinate_scale * splu(
                        system, permc_spec=ordering,
                        diag_pivot_thresh=threshold).solve(rhs)
                except RuntimeError as exc:
                    variant.update(status="FACTORIZATION_FAILED",
                                   detail=str(exc))
                    variants.append(variant)
                    continue
                u = np.zeros(18)
                u[6:] = free
                residual = matrix @ u - load
                forces = residual[6:][[0, 1, 2, 6, 7, 8]]
                moments = residual[6:][[3, 4, 5, 9, 10, 11]]
                tip = u[12:]
                variant.update(
                    status="FINITE" if np.isfinite(free).all()
                    else "NONFINITE",
                    tip=tip.tolist(),
                    tip_vs_exact_component_scaled_max=float(np.max(
                        np.abs(tip - exact_tip) /
                        np.maximum(1., np.abs(exact_tip)))),
                    tip_vs_physical_component_scaled_max=float(np.max(
                        np.abs(tip - physical_tip) /
                        np.maximum(1., np.abs(physical_tip)))),
                    free_force_residual_to_tip_force=float(
                        np.max(np.abs(forces)) /
                        np.max(np.abs(terminal[:3]))),
                    free_moment_residual_to_tip_moment=float(
                        np.max(np.abs(moments)) /
                        np.max(np.abs(terminal[3:]))),
                )
                variants.append(variant)
    return {
        "area_m2": area,
        "matrix_order": int(reduced.shape[0]),
        "matrix_nnz": int(reduced.nnz),
        "rounded_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(reduced.toarray()).tobytes()).hexdigest(),
        "baseline_sparse_tip_exact_match": True,
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    reference_bytes = args.reference.read_bytes()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    if reference_hash != "cb9e88a0124cf74e7f22179e6fb82239e0edde232055981e47fffd9f13813271":
        raise ValueError("F09 historical reference identity changed")
    reference = json.loads(reference_bytes)
    started = time.perf_counter()
    cases = {name: _case(float(reference[name]["area_m2"]), reference[name])
             for name in ("moderate", "extreme")}
    result = {
        "classification": "UNREGISTERED_F09_SPARSE_DIAGNOSTIC_NOT_ADMISSION",
        "reference_sha256": reference_hash,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elbow_source_sha256": hashlib.sha256(
            Path(elbow.__file__).read_bytes()).hexdigest(),
        "elapsed_s_not_benchmark": time.perf_counter() - started,
        "cases": cases,
        "limitations": [
            "two areas do not define a supported physical conditioning range",
            "sparse factorization recovery alone does not qualify global B2 admission",
            "one-metre/one-radian tip floors are diagnostic, not physical tolerances",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
