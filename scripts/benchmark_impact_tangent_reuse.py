"""Benchmark and qualify nonlinear-impact modified-Newton tangent reuse."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, Iterable

import numpy as np

import anyfesolver as fs
from anyfesolver.contact import _verification_contact_panel


def _float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("expected one or more positive comma-separated numbers")
    return values


def _run(dt: float, penalty: float, reuse_iterations: int) -> tuple[Any, float]:
    model = _verification_contact_panel()
    start = time.perf_counter()
    result = fs.solve_transient_sphere_impact(
        model,
        fs.TransientConfig(dt=float(dt), t_end=0.04),
        fs.RigidSphereImpact(
            "tangent_reuse_benchmark",
            radius=0.1,
            mass=1.0,
            start_point=(0.5, 0.5, 0.12),
            travel_direction=(0.0, 0.0, -1.0),
            speed=2.0,
        ),
        fs.SphereContactConfig(
            penalty_stiffness=float(penalty),
            max_contact_iterations=40,
        ),
        nonlinear_config=fs.NonlinearTransientConfig(
            enabled=True,
            max_iterations=15,
            max_cutbacks=4,
            tangent_reuse_iterations=int(reuse_iterations),
        ),
    )
    return result, time.perf_counter() - start


def _relative_error(candidate: np.ndarray, oracle: np.ndarray) -> float:
    candidate_array = np.asarray(candidate, dtype=float)
    oracle_array = np.asarray(oracle, dtype=float)
    difference = float(np.max(np.abs(candidate_array - oracle_array), initial=0.0))
    scale = max(float(np.max(np.abs(oracle_array), initial=0.0)), 1.0)
    return difference / scale


def _case(dt: float, penalty: float, reuse_iterations: int, repeats: int) -> Dict[str, Any]:
    legacy_times = []
    reuse_times = []
    legacy = reused = None
    for repeat in range(int(repeats)):
        order: Iterable[int] = (0, reuse_iterations) if repeat % 2 == 0 else (reuse_iterations, 0)
        for budget in order:
            result, elapsed = _run(dt, penalty, budget)
            if budget == 0:
                legacy = result
                legacy_times.append(elapsed)
            else:
                reused = result
                reuse_times.append(elapsed)
    assert legacy is not None and reused is not None
    legacy_diag = legacy.diagnostics
    reuse_diag = reused.diagnostics
    legacy_median = float(statistics.median(legacy_times))
    reuse_median = float(statistics.median(reuse_times))
    return {
        "dt": float(dt),
        "penalty_stiffness": float(penalty),
        "reuse_iterations": int(reuse_iterations),
        "status": {"legacy": legacy.status, "reuse": reused.status},
        "elapsed_median_s": {"legacy": legacy_median, "reuse": reuse_median},
        "speedup": legacy_median / max(reuse_median, 1.0e-30),
        "factorization_count": {
            "legacy": int(legacy_diag["factorization_count"]),
            "reuse": int(reuse_diag["factorization_count"]),
        },
        "factorization_reuse_count": int(reuse_diag["factorization_reuse_count"]),
        "factorization_reduction_fraction": 1.0
        - float(reuse_diag["factorization_count"]) / max(float(legacy_diag["factorization_count"]), 1.0),
        "cutback_count": {
            "legacy": int(legacy_diag["cutback_count"]),
            "reuse": int(reuse_diag["cutback_count"]),
        },
        "iteration_count": {
            "legacy": int(sum(legacy_diag["iteration_counts"])),
            "reuse": int(sum(reuse_diag["iteration_counts"])),
        },
        "relative_errors": {
            "displacement_history": _relative_error(reused.displacements, legacy.displacements),
            "contact_force_history": _relative_error(reused.contact_force_history, legacy.contact_force_history),
            "sphere_position_history": _relative_error(reused.sphere_positions, legacy.sphere_positions),
            "sphere_velocity_history": _relative_error(reused.sphere_velocities, legacy.sphere_velocities),
            "peak_contact_force": abs(float(reused.peak_contact_force) - float(legacy.peak_contact_force))
            / max(abs(float(legacy.peak_contact_force)), 1.0),
            "maximum_penetration": abs(float(reused.max_penetration) - float(legacy.max_penetration))
            / max(abs(float(legacy.max_penetration)), 1.0),
            "momentum_balance": abs(
                float(reused.sphere_momentum_balance_error) - float(legacy.sphere_momentum_balance_error)
            ),
        },
        "refresh_reason_counts": reuse_diag["refresh_reason_counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=_float_list, default=(0.002, 0.0025, 0.004))
    parser.add_argument("--penalty", type=_float_list, default=(2000.0, 4000.0, 8000.0))
    parser.add_argument("--reuse-iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.reuse_iterations <= 0:
        parser.error("--reuse-iterations must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    # Compile and initialize solver backends before recording wall time.
    _run(float(args.dt[0]), float(args.penalty[0]), 0)
    cases = [
        _case(dt, penalty, args.reuse_iterations, args.repeats)
        for dt in args.dt
        for penalty in args.penalty
    ]
    payload = {
        "benchmark": "nonlinear_impact_tangent_reuse",
        "cases": cases,
        "minimum_speedup": min(case["speedup"] for case in cases),
        "median_speedup": float(statistics.median(case["speedup"] for case in cases)),
        "minimum_factorization_reduction_fraction": min(
            case["factorization_reduction_fraction"] for case in cases
        ),
        "maximum_history_relative_error": max(
            max(case["relative_errors"].values()) for case in cases
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
