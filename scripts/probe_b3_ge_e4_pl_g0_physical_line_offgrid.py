"""G0 F03-P-v4: off-grid shell rail work and exact S3 integration breaks.

This is a bounded shell-only diagnostic, not selection of a line multiplier.
The independent midpoint load checks the segmented Gauss integration; the
normal-moment gradient also has an analytic infinitesimal pose-fit oracle.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.sparse.linalg import spsolve

import probe_b3_ge_e4_pl_g0_physical_line as prior
from anyfesolver._ge_beam3_mixed_ad import so3_log, transpose
from anyfesolver._ge_beam3_variational_shell import rotation_jets


START, END = -0.4, 0.4
RAIL = 0.23
SAMPLES = ((-0.2, -RAIL), (0.2, -RAIL),
           (0.2, RAIL), (-0.2, RAIL))


def sample_gradient() -> tuple[np.ndarray, float, float]:
    reference = np.array([(dx, y, 0.0) for dx, y in SAMPLES])
    potential = so3_log(transpose(rotation_jets(reference, reference)))[2]
    derivative = potential.gradient.reshape(4, 6)
    denominator = sum(dx * dx + y * y for dx, y in SAMPLES)
    analytic = np.array([(y / denominator, -dx / denominator) for dx, y in SAMPLES])
    error = float(np.linalg.norm(analytic - derivative[:, :2]))
    force = np.sum(analytic, axis=0)
    moment = sum(dx * row[1] - y * row[0]
                 for (dx, y), row in zip(SAMPLES, analytic))
    assert np.linalg.norm(derivative[:, 2:]) <= 1e-12
    assert error <= 1e-10
    assert np.linalg.norm(force) <= 1e-12
    assert abs(moment + 1.0) <= 1e-12
    return analytic, error, float(moment)


def segments(
    xs: np.ndarray, ys: np.ndarray, topology: str
) -> tuple[list[float], int]:
    breaks = {START, END}
    diagonals = set()
    for shift, y in SAMPLES:
        for edge in xs:
            station = float(edge - shift)
            if START < station < END:
                breaks.add(round(station, 14))
        if topology == "S3-V2D":
            j = int(np.searchsorted(ys, y, side="right") - 1)
            j = min(max(j, 0), len(ys) - 2)
            eta = float((y - ys[j]) / (ys[j + 1] - ys[j]))
            assert 0.0 < eta < 1.0
            for i in range(len(xs) - 1):
                diagonal = float(xs[i] + eta * (xs[i + 1] - xs[i]) - shift)
                if START < diagonal < END:
                    diagonals.add(round(diagonal, 14))
    breaks.update(diagonals)
    return sorted(breaks), len(diagonals)


def load_at(
    station: float,
    measure: float,
    force: np.ndarray,
    gradient: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    topology: str,
    hosts: set[tuple[int, ...]],
) -> None:
    for (dx, y), row in zip(SAMPLES, gradient):
        weights = prior.host_weights(station + dx, y, xs, ys, topology)
        hosts.add(tuple(sorted(node for node, _ in weights)))
        for node, shape in weights:
            force[2 * node:2 * node + 2] += measure * shape * row


def case(n: int, topology: str, gradient: np.ndarray) -> dict:
    xs, ys, positions, stiffness, elements = prior.mesh_stiffness(n, topology)
    breaks, diagonals = segments(xs, ys, topology)
    assert topology != "S3-V2D" or diagonals > 0
    exact = np.zeros(2 * len(positions))
    gauss = np.zeros_like(exact)
    roots, weights = np.polynomial.legendre.leggauss(4)
    hosts: set[tuple[int, ...]] = set()
    for left, right in zip(breaks[:-1], breaks[1:]):
        length = right - left
        midpoint = 0.5 * (left + right)
        load_at(midpoint, length, exact, gradient, xs, ys, topology, hosts)
        for root, weight in zip(roots, weights):
            station = midpoint + 0.5 * length * root
            load_at(station, 0.5 * length * weight, gauss,
                    gradient, xs, ys, topology, hosts)
    load_error = float(np.linalg.norm(exact - gauss) /
                       max(np.linalg.norm(exact), 1e-30))
    assert load_error <= 1e-12

    fixed = np.array([2 * (j * (n + 1) + i) + axis
                      for j in range(n + 1) for i in range(n + 1)
                      if i in (0, n) or j in (0, n) for axis in (0, 1)], dtype=int)
    free = np.setdiff1d(np.arange(len(exact)), fixed)
    displacement = np.zeros_like(exact)
    displacement[free] = spsolve(stiffness[free, :][:, free].tocsc(), exact[free])
    residual = np.asarray(stiffness @ displacement - exact)
    support = np.zeros_like(exact)
    support[fixed] = residual[fixed]
    total_applied = exact.reshape(-1, 2).sum(axis=0)
    force_balance = np.linalg.norm((support + exact).reshape(-1, 2).sum(axis=0))
    applied_moment = float(np.sum(positions[:, 0] * exact[1::2] -
                                  positions[:, 1] * exact[0::2]))
    support_moment = float(np.sum(positions[:, 0] * support[1::2] -
                                  positions[:, 1] * support[0::2]))
    work = float(exact @ displacement)
    twice_energy = float(displacement @ (stiffness @ displacement))
    relative_work = abs(work - twice_energy) / max(
        abs(work), abs(twice_energy), 1e-30
    )
    free_error = float(np.linalg.norm(residual[free]))
    result = dict(
        n=n, topology=topology, shell_elements=elements,
        nnz=int(stiffness.nnz), segments=len(breaks) - 1,
        diagonal_splits=diagonals, unique_hosts=len(hosts),
        load_vector_error=load_error, applied_force_norm=float(np.linalg.norm(total_applied)),
        force_balance=float(force_balance), applied_moment=applied_moment,
        support_moment=support_moment, free_residual=free_error,
        compliance=work, twice_energy=twice_energy,
        relative_work_error=relative_work,
        applied_load=exact.tolist(),
        displacement=displacement.tolist(),
        support_reactions=support.tolist(),
    )
    assert np.isfinite(displacement).all() and work > 0.0
    assert result["applied_force_norm"] <= 1e-8
    assert abs(applied_moment + 0.8) <= 1e-8
    assert force_balance <= 1e-8 and abs(support_moment - 0.8) <= 1e-8
    assert free_error <= 1e-8 and relative_work <= 1e-8
    return result


def main() -> int:
    gradient, gradient_error, reference_moment = sample_gradient()
    rows = [case(n, topology, gradient)
            for topology in ("Q4", "S3-V2D") for n in (4, 8, 16)]
    print(json.dumps({
        "profile": "G0-F03-P-v4", "status": "diagnostic_pass",
        "gradient_error": gradient_error, "reference_moment": reference_moment,
        "results": rows,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
