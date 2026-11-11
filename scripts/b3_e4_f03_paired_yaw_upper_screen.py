"""Bounded full-P0 yaw work screen against the supported E4 physical core.

This is a development diagnostic. It does not establish a continuum bound,
native mixed-system stability, or formal F03 acceptance.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_info
import anyfesolver

from scripts.b3_e4_f03_l1_physical_rail import transverse_shape_integrals
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent


def flat_yaw_p0_action(xs: np.ndarray, ys: np.ndarray,
                       weights: np.ndarray, thickness: float) -> np.ndarray:
    """Integrated virgin rail yaw rows, including top-surface director work."""
    full_x_cells = len(xs)-1
    rail_cells = full_x_cells//2
    action = np.zeros((rail_cells, 6*len(xs)*len(ys)))
    for i in range(rail_cells):
        left = full_x_cells//2+i
        for j, weight in enumerate(weights):
            for node, sign in ((left, -1.), (left+1, 1.)):
                base = 6*(j*len(xs)+node)
                action[i, base+1] += sign*weight
                # Top_y = u_y - (thickness/2)*theta_x at virgin state.
                action[i, base+3] -= sign*(thickness/2.)*weight
    return action


def screen(full_x_cells: int, transverse_cells: int) -> dict:
    if (full_x_cells, transverse_cells) not in ((8, 2), (16, 4)):
        raise ValueError("bounded first-two-grid yaw screen only")
    start = time.perf_counter()
    xs = np.linspace(0., 1., full_x_cells+1)
    ys = np.linspace(-.1, .1, transverse_cells+1)
    weights = transverse_shape_integrals(ys, (-.047, .046))
    _, physical = raw_e4_tangent(xs, ys, thickness=.035)
    # The x=0 edge is fixed. Numerical drilling has no physical-core work.
    free = np.array([6*(j*len(xs)+i)+axis
                     for j in range(len(ys))
                     for i in range(1, len(xs))
                     for axis in range(5)], dtype=int)
    k = physical[np.ix_(free, free)]
    skew = float(np.linalg.norm(k-k.T) / np.linalg.norm(k))
    rail_cells = full_x_cells//2
    h = .5/rail_cells
    action = flat_yaw_p0_action(xs, ys, weights, .035)
    action = action[:, free] / np.sqrt(h)  # P0 L2-orthonormal densities
    eigenvalues, vectors = linalg.eigh(k, check_finite=True)
    max_eigenvalue = float(eigenvalues[-1])
    numerical_floor = (32.*np.finfo(float).eps*len(free)*max_eigenvalue)
    null = eigenvalues <= numerical_floor
    transformed = vectors.T @ action.T
    action_scale = float(linalg.svdvals(action)[0])
    null_action = (float(np.linalg.norm(transformed[null], ord=2))
                   if np.any(null) else 0.)
    # This positive-subspace number is not a finite full-space quotient
    # when the action couples to a true null or an unresolved near-null.
    positive = ~null
    positive_compliance = ((transformed[positive].T / eigenvalues[positive])
                           @ transformed[positive])
    upper = float(linalg.eigvalsh(positive_compliance)[-1])
    return {
        "full_x_cells": full_x_cells,
        "transverse_cells": transverse_cells,
        "degrees_of_freedom": physical.shape[0],
        "supported_physical_coordinates": len(free),
        "yaw_p0_rows": rail_cells,
        "rail_hx_over_hy": h/(.2/transverse_cells),
        "relative_skew": skew,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": max_eigenvalue,
        "numerical_floor": float(numerical_floor),
        "near_null_count": int(null.sum()),
        "near_null_action_2norm": null_action,
        "near_null_action_relative": null_action/action_scale,
        "positive_subspace_upper_quotient": upper,
        "elapsed_seconds": time.perf_counter()-start,
    }


if __name__ == "__main__":
    expected = (Path(__file__).resolve().parents[1]/"src"/"anyfesolver").resolve()
    actual = Path(anyfesolver.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError("yaw screen imported another ANYfeSolver checkout")
    print(json.dumps({"kind": "environment", "threadpools": threadpool_info(),
                      "imported_anyfesolver": str(actual)}),
          flush=True)
    for nx, ny in ((8, 2), (16, 4)):
        print(json.dumps({"kind": "yaw_upper_screen", **screen(nx, ny)}),
              flush=True)
