"""Independent finite-volume continuum obstacle discriminator for D1.

This development tool does not import ANYfeSolver. Fixed shell rotations reduce
the guided strip's transverse continuum work to shear energy. Cell-centred
finite volumes and a primal active set solve that scalar variational
inequality. The full-width obstacle has an exact one-dimensional solution;
the partial-width obstacle is the actual linearized development fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


YOUNG_PA = 1.0e5
POISSON = 0.3
THICKNESS_M = 0.05
SHEAR_FACTOR = 5.0 / 6.0
SHEAR_RIGIDITY_N_PER_M = (
    SHEAR_FACTOR * YOUNG_PA * THICKNESS_M / (2.0 * (1.0 + POISSON))
)
DOWNWARD_LOAD_N_PER_M2 = 0.3
INITIAL_GAP_M = 4.0e-5
PARTIAL_CONTACT_HEIGHT_M = 0.9


@dataclass(frozen=True)
class Solution:
    nx: int
    ny: int
    width_y: float
    displacement: np.ndarray
    pressure_force: np.ndarray
    active: np.ndarray
    iterations: int
    minimum_gap_m: float
    minimum_active_reaction_n: float
    left_support_n: float
    contact_reaction_n: float
    total_load_n: float
    equilibrium_defect_n: float


def _operator(nx: int, ny: int) -> sparse.csr_matrix:
    dx = 1.0 / nx
    dy = 1.0 / ny
    size = nx * ny
    diagonal = np.zeros(size)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def edge(first: int, second: int, conductance: float) -> None:
        diagonal[first] += conductance
        diagonal[second] += conductance
        rows.extend((first, second))
        columns.extend((second, first))
        values.extend((-conductance, -conductance))

    for j in range(ny):
        for i in range(nx):
            cell = j * nx + i
            if i == 0:
                diagonal[cell] += 2.0 * SHEAR_RIGIDITY_N_PER_M * dy / dx
            if i + 1 < nx:
                edge(cell, cell + 1, SHEAR_RIGIDITY_N_PER_M * dy / dx)
            if j + 1 < ny:
                edge(cell, cell + nx, SHEAR_RIGIDITY_N_PER_M * dx / dy)
    rows.extend(range(size))
    columns.extend(range(size))
    values.extend(diagonal.tolist())
    return sparse.coo_matrix((values, (rows, columns)),
                             shape=(size, size)).tocsr()


def solve(nx: int, ny: int, contact_height: float) -> Solution:
    if nx < 8 or ny < 8 or not (0.0 < contact_height <= 1.0):
        raise ValueError("unsupported grid or obstacle height")
    dx = 1.0 / nx
    dy = 1.0 / ny
    area = dx * dy
    stiffness = _operator(nx, ny)
    force = np.full(nx * ny, -DOWNWARD_LOAD_N_PER_M2 * area)
    contact_rows = ((np.arange(ny) + 0.5) * dy < contact_height)
    obstacle = np.repeat(contact_rows, nx)
    if not np.any(obstacle):
        raise ValueError("obstacle has no represented area")

    unconstrained = np.asarray(spsolve(stiffness, force)).reshape(-1)
    active = obstacle & (unconstrained < -INITIAL_GAP_M)
    displacement = unconstrained.copy()
    gap_tolerance = 1.0e-12
    reaction_tolerance = 1.0e-12
    for iteration in range(1, 2 * nx * ny + 1):
        free = ~active
        displacement[active] = -INITIAL_GAP_M
        right = force[free] - stiffness[free][:, active] @ displacement[active]
        displacement[free] = spsolve(stiffness[free][:, free], right)
        reaction = np.asarray(stiffness @ displacement - force).reshape(-1)
        violations = obstacle & free & (
            displacement < -INITIAL_GAP_M - gap_tolerance
        )
        negative_reaction = active & (reaction < -reaction_tolerance)
        if not np.any(violations) and not np.any(negative_reaction):
            break
        active = (active | violations) & ~negative_reaction
    else:
        raise RuntimeError("continuum obstacle active set did not settle")

    reaction = np.asarray(stiffness @ displacement - force).reshape(-1)
    # Only obstacle cells may carry reaction; free-cell residuals are roundoff.
    pressure_force = np.where(active, reaction, 0.0)
    contact_reaction = float(np.sum(pressure_force))
    left_support = float(-np.sum(
        2.0 * SHEAR_RIGIDITY_N_PER_M * dy / dx
        * displacement.reshape(ny, nx)[:, 0]
    ))
    total_load = float(DOWNWARD_LOAD_N_PER_M2)
    equilibrium_defect = abs(contact_reaction + left_support - total_load)
    minimum_gap = float(np.min(displacement[obstacle] + INITIAL_GAP_M))
    minimum_reaction = float(np.min(reaction[active])) if np.any(active) else 0.0
    if (minimum_gap < -gap_tolerance or minimum_reaction < -reaction_tolerance
            or equilibrium_defect > 1.0e-9
            or np.max(np.abs(reaction[~active])) > 1.0e-9):
        raise RuntimeError("continuum obstacle KKT or balance check failed")
    return Solution(
        nx, ny, contact_height, displacement.reshape(ny, nx),
        pressure_force.reshape(ny, nx), active.reshape(ny, nx), iteration,
        minimum_gap, minimum_reaction, left_support, contact_reaction,
        total_load, equilibrium_defect,
    )


def summarize(solution: Solution) -> dict:
    nx, ny = solution.nx, solution.ny
    x = (np.arange(nx) + 0.5) / nx
    x_reaction = np.sum(solution.pressure_force, axis=0)
    cumulative = np.cumsum(x_reaction) / solution.contact_reaction_n
    quantiles = {
        str(level): float(x[np.searchsorted(cumulative, level)])
        for level in (0.05, 0.5, 0.95)
    }
    return {
        "nx": nx,
        "ny": ny,
        "contact_height_m": solution.width_y,
        "iterations": solution.iterations,
        "active_cells": int(np.sum(solution.active)),
        "contact_reaction_n": solution.contact_reaction_n,
        "left_support_n": solution.left_support_n,
        "equilibrium_defect_n": solution.equilibrium_defect_n,
        "minimum_cell_gap_m": solution.minimum_gap_m,
        "minimum_active_cell_reaction_n": solution.minimum_active_reaction_n,
        "reaction_x_quantile_m": quantiles,
        "minimum_displacement_m": float(np.min(solution.displacement)),
    }


def _full_width_exact_check(solution: Solution) -> dict:
    front = np.sqrt(2.0 * SHEAR_RIGIDITY_N_PER_M * INITIAL_GAP_M
                    / DOWNWARD_LOAD_N_PER_M2)
    x = (np.arange(solution.nx) + 0.5) / solution.nx
    exact = np.where(
        x < front,
        -DOWNWARD_LOAD_N_PER_M2 / SHEAR_RIGIDITY_N_PER_M
        * (front * x - 0.5 * x * x),
        -INITIAL_GAP_M,
    )
    exact_reaction = DOWNWARD_LOAD_N_PER_M2 * (1.0 - front)
    return {
        "exact_front_m": float(front),
        "exact_contact_reaction_n": float(exact_reaction),
        "reaction_error_n": float(solution.contact_reaction_n-exact_reaction),
        "maximum_cell_displacement_error_m": float(np.max(
            np.abs(solution.displacement - exact[None, :])
        )),
    }


def main() -> None:
    full = solve(64, 80, 1.0)
    report = {
        "method": "independent_scalar_shear_finite_volume_obstacle_dev",
        "shear_rigidity_n_per_m": SHEAR_RIGIDITY_N_PER_M,
        "full_width_control": {
            **summarize(full), **_full_width_exact_check(full),
        },
        "partial_width": [
            summarize(solve(nx, ny, PARTIAL_CONTACT_HEIGHT_M))
            for nx, ny in ((32, 40), (64, 80), (128, 160))
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
