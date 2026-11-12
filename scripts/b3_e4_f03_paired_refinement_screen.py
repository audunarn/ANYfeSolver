"""Bounded development screen for the physical-core rail yaw witness.

This compares the earlier longitudinal-only counterexample with paired Q4
refinement. It tests two constructed fields, not the whole multiplier space.
"""

import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info
import anyfesolver

from scripts.b3_e4_f03_l1_physical_rail import transverse_shape_integrals
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent


def screen(full_x_cells: int, transverse_cells: int) -> dict:
    if (full_x_cells, transverse_cells) not in ((8, 2), (16, 4), (32, 8)):
        raise ValueError("only the bounded paired grids are admitted")
    start = time.perf_counter()
    xs = np.linspace(0., 1., full_x_cells + 1)
    ys = np.linspace(-.1, .1, transverse_cells + 1)
    # A fixed material strip deliberately avoids alignment with all grids.
    strip = (-.047, .046)
    weights = transverse_shape_integrals(ys, strip)
    transverse = (-1.) ** np.arange(len(ys))
    mean = float(weights @ transverse)
    _, physical = raw_e4_tangent(xs, ys, thickness=.035)

    rail_cells = full_x_cells // 2
    h = .5 / rail_cells
    density = (-1.) ** (np.arange(rail_cells) // 2)
    cases = {}
    for name in ("macro_p0", "constant_endpoint"):
        amplitudes = np.zeros(rail_cells + 1)
        if name == "macro_p0":
            amplitudes[2::4] = h
            active_density = density
        else:
            amplitudes[-1] = 1.
            active_density = np.ones(rail_cells)
        q = np.zeros(physical.shape[0])
        for i, amplitude in enumerate(amplitudes):
            for j, phi in enumerate(transverse):
                q[6 * (j * len(xs) + full_x_cells // 2 + i) + 1] = amplitude * phi
        # Exact flat-reference rail yaw variation: transverse average of
        # d(u_y)/dx. Integrating a constant density on each cell cancels h.
        work = float(mean * (active_density @ np.diff(amplitudes)))
        energy = float(q @ physical @ q)
        density_norm = np.sqrt(.5)
        cases[name] = {
            "absolute_work": abs(work),
            "physical_energy": energy,
            "normalized_work": abs(work) / (np.sqrt(energy) * density_norm),
            "density_l2_norm": density_norm,
        }
    return {
        "full_x_cells": full_x_cells,
        "rail_x_cells": rail_cells,
        "transverse_cells": transverse_cells,
        "rail_hx_over_hy": h / (.2 / transverse_cells),
        "degrees_of_freedom": physical.shape[0],
        "fixed_strip": strip,
        "transverse_strip_average": mean,
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    expected = (Path(__file__).resolve().parents[1]/"src"/"anyfesolver").resolve()
    actual = Path(anyfesolver.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError("paired screen imported another ANYfeSolver checkout")
    print(json.dumps({"kind": "environment", "threadpools": threadpool_info(),
                      "imported_anyfesolver": str(actual)}),
          flush=True)
    for nx, ny in ((8, 2), (16, 4), (32, 8)):
        print(json.dumps({"kind": "screen", **screen(nx, ny)}), flush=True)
