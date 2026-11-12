"""Bounded paired-grid six-channel shell pose-work diagnostic.

This is a source-level development screen of the shell side only. It neither
proves a mesh-uniform bound nor qualifies a multiplier space or the mixed KKT.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anyfesolver
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_info

from scripts.b3_e4_f03_flat_rail_six_channel_map import (
    flat_rail_p0_shell_pose_action,
)
from scripts.b3_e4_f03_l1_physical_rail import transverse_shape_integrals
from scripts.b3_e4_f03_paired_yaw_upper_screen import flat_yaw_p0_action
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent


def screen(full_x_cells: int, transverse_cells: int) -> dict:
    if (full_x_cells, transverse_cells) not in ((8, 2), (16, 4)):
        raise ValueError("bounded first-two-grid six-channel screen only")
    start = time.perf_counter()
    xs = np.linspace(0., 1., full_x_cells + 1)
    ys = np.linspace(-.1, .1, transverse_cells + 1)
    rail_start = full_x_cells // 2
    rail_xs = xs[rail_start:]
    rail_cells = len(rail_xs) - 1
    thickness = .035
    _, physical = raw_e4_tangent(xs, ys, thickness=thickness)

    # The clamped x=0 edge removes rigid modes. Drilling has no physical work.
    free = np.array([6 * (j * len(xs) + i) + axis
                     for j in range(len(ys))
                     for i in range(1, len(xs))
                     for axis in range(5)], dtype=int)
    stiffness = physical[np.ix_(free, free)]
    relative_skew = float(np.linalg.norm(stiffness - stiffness.T) /
                          np.linalg.norm(stiffness))
    local = flat_rail_p0_shell_pose_action(
        rail_xs, ys, (-.047, .046), thickness).reshape(6 * rail_cells, -1)
    action = np.zeros((6 * rail_cells, physical.shape[0]))
    for j in range(len(ys)):
        for i in range(len(rail_xs)):
            local_base = 6 * (j * len(rail_xs) + i)
            global_base = 6 * (j * len(xs) + rail_start + i)
            action[:, global_base:global_base + 6] += (
                local[:, local_base:local_base + 6])
    yaw_weights = transverse_shape_integrals(ys, (-.047, .046))
    yaw_oracle = flat_yaw_p0_action(xs, ys, yaw_weights, thickness)
    yaw_map_error = float(np.max(np.abs(action[5::6] - yaw_oracle)))
    if yaw_map_error > 1e-12:
        raise ArithmeticError("six-channel scatter disagrees with yaw work oracle")

    # P0 L2-orthonormal density coordinates. Moment density is represented
    # by an equivalent force density using the frozen 0.1 m length.
    h = .5 / rail_cells
    characteristic_length = .1
    channel_scale = np.array((1., 1., 1., characteristic_length,
                              characteristic_length, characteristic_length))
    work = action[:, free].copy()
    work *= np.tile(channel_scale, rail_cells)[:, None] / np.sqrt(h)

    eigenvalues, vectors = linalg.eigh(stiffness, check_finite=True)
    largest = float(eigenvalues[-1])
    numerical_floor = float(32. * np.finfo(float).eps * len(free) * largest)
    null = eigenvalues <= numerical_floor
    transformed = vectors.T @ work.T
    action_norm = float(linalg.svdvals(work)[0])
    null_action = (float(linalg.svdvals(transformed[null])[0])
                   if np.any(null) else 0.)
    if np.all(null):
        raise ArithmeticError("no positive physical subspace on supported grid")
    positive = ~null
    compliance = ((transformed[positive].T / eigenvalues[positive])
                  @ transformed[positive])
    compliance = .5 * (compliance + compliance.T)
    values, modes = linalg.eigh(compliance, check_finite=True)
    leading = modes[:, -1].reshape(rail_cells, 6)
    channel_fraction = np.sum(leading * leading, axis=0)
    per_channel = {}
    for channel, name in enumerate(("fx", "fy", "fz", "mx", "my", "mz")):
        rows = np.arange(channel, 6 * rail_cells, 6)
        per_channel[name] = float(linalg.eigvalsh(
            compliance[np.ix_(rows, rows)])[-1])
    return {
        "full_x_cells": full_x_cells,
        "transverse_cells": transverse_cells,
        "rail_cells": rail_cells,
        "degrees_of_freedom": physical.shape[0],
        "supported_physical_coordinates": len(free),
        "work_rows": 6 * rail_cells,
        "rail_hx_over_hy": h / (.2 / transverse_cells),
        "physical_length_scale_m": characteristic_length,
        "relative_skew": relative_skew,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": largest,
        "numerical_floor": numerical_floor,
        "near_null_count": int(null.sum()),
        "near_null_action_relative": null_action / action_norm,
        "independent_yaw_row_error": yaw_map_error,
        "positive_subspace_upper_quotient": float(values[-1]),
        "leading_mode_density_fraction_by_channel": dict(zip(
            ("fx", "fy", "fz", "mx", "my", "mz"),
            channel_fraction.tolist())),
        "single_channel_upper_quotient": per_channel,
        "elapsed_seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    expected = (Path(__file__).resolve().parents[1] / "src" / "anyfesolver").resolve()
    actual = Path(anyfesolver.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError("six-channel screen imported another ANYfeSolver checkout")
    print(json.dumps({"kind": "environment", "threadpools": threadpool_info(),
                      "imported_anyfesolver": str(actual)}), flush=True)
    for nx, ny in ((8, 2), (16, 4)):
        print(json.dumps({"kind": "six_channel_work_screen", **screen(nx, ny)}),
              flush=True)
