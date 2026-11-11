"""Benchmark nonlinear impact with full versus direct reduced assembly."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from typing import Any, Dict

import numpy as np

import anyfesolver as fs
from anyfesolver.assembly import build_constraint_transformation
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.elements import create_element
from anyfesolver.fe_core import FEModel
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.nonlinear_performance_bootstrap import get_nonlinear_assembly_plan
from anyfesolver.nonlinear_reduced_assembly import (
    assemble_reduced_system,
    build_reduced_assembly_plan,
)


def _panel(divisions: int) -> FEModel:
    model = FEModel(f"impact_direct_reduced_{divisions}x{divisions}")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    node_ids = {}
    node_id = 1
    for row in range(divisions + 1):
        for column in range(divisions + 1):
            model.add_node(
                node_id,
                column / divisions,
                row / divisions,
                0.0,
            )
            node_ids[(column, row)] = node_id
            node_id += 1
    element_id = 1
    for row in range(divisions):
        for column in range(divisions):
            model.add_element(
                element_id,
                create_element(
                    "shell",
                    element_id,
                    [
                        node_ids[(column, row)],
                        node_ids[(column + 1, row)],
                        node_ids[(column + 1, row + 1)],
                        node_ids[(column, row + 1)],
                    ],
                    "soft",
                    thickness=0.05,
                ),
            )
            element_id += 1
    model.add_boundary_condition(
        BoundaryCondition(
            "retain-only-transverse-motion",
            list(model.mesh.nodes),
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    return model


def _with_activation_threshold(value: str, callback):
    name = "FE_SOLVER_BATCH_C_MIN_ESTIMATED_ASSEMBLIES"
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        return callback()
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _impact(divisions: int, dt: float, t_end: float, direct: bool):
    def solve():
        return fs.solve_transient_sphere_impact(
            _panel(divisions),
            fs.TransientConfig(dt=dt, t_end=t_end, hht_alpha=-0.05),
            fs.RigidSphereImpact(
                "assembly_benchmark",
                radius=0.1,
                mass=1.0,
                start_point=(0.5, 0.5, 0.11),
                travel_direction=(0.0, 0.0, -1.0),
                speed=1.0,
            ),
            fs.SphereContactConfig(
                penalty_stiffness=4000.0,
                max_contact_iterations=30,
            ),
            nonlinear_config=fs.NonlinearTransientConfig(
                enabled=True,
                max_iterations=12,
                max_cutbacks=3,
                tangent_reuse_iterations=2,
            ),
        )

    return _with_activation_threshold("0" if direct else "1000000000", solve)


def _assembly_microbenchmark(divisions: int, iterations: int) -> Dict[str, Any]:
    model = _panel(divisions)
    model.apply_boundary_conditions()
    stiffness, _ = assemble_stiffness_matrix(model)
    zero = np.zeros(stiffness.shape[0], dtype=float)
    _K_red, _F_red, transformation, u0, _independent, _info = (
        build_constraint_transformation(stiffness, zero, model)
    )
    nonlinear_plan = get_nonlinear_assembly_plan(model, 5)
    reduced_plan = build_reduced_assembly_plan(nonlinear_plan, transformation)
    rng = np.random.default_rng(20260811)
    q = rng.normal(scale=1.0e-6, size=transformation.shape[1])
    displacement = np.asarray(transformation @ q + u0, dtype=float).reshape(-1)

    full_force, full_tangent, _ = nonlinear_plan.assemble(
        displacement,
        {},
        tangent=True,
    )
    expected_force = np.asarray(transformation.T @ full_force, dtype=float).reshape(-1)
    expected_tangent = (transformation.T @ full_tangent @ transformation).tocsr()
    direct_force, direct_tangent, _ = assemble_reduced_system(
        nonlinear_plan,
        reduced_plan,
        displacement,
        {},
        tangent=True,
    )
    np.testing.assert_allclose(direct_force, expected_force, rtol=2.0e-11, atol=1.0e-8)
    np.testing.assert_allclose(
        direct_tangent.toarray(),
        expected_tangent.toarray(),
        rtol=2.0e-11,
        atol=1.0e-5,
    )

    def full_batch() -> None:
        for _ in range(iterations):
            force, tangent, _states = nonlinear_plan.assemble(
                displacement,
                {},
                tangent=True,
            )
            np.asarray(transformation.T @ force, dtype=float).reshape(-1)
            (transformation.T @ tangent @ transformation).tocsr()

    def direct_batch() -> None:
        for _ in range(iterations):
            assemble_reduced_system(
                nonlinear_plan,
                reduced_plan,
                displacement,
                {},
                tangent=True,
            )

    full_times = []
    direct_times = []
    for order in ((full_batch, direct_batch), (direct_batch, full_batch)) * 2:
        for callback in order:
            start = time.perf_counter()
            callback()
            elapsed = time.perf_counter() - start
            (direct_times if callback is direct_batch else full_times).append(elapsed)
    full_median = float(statistics.median(full_times))
    direct_median = float(statistics.median(direct_times))
    return {
        "iterations_per_batch": int(iterations),
        "full_assemble_then_project_median_s": full_median,
        "direct_reduced_median_s": direct_median,
        "speedup": full_median / max(direct_median, 1.0e-30),
        "plan": reduced_plan.diagnostics(),
        "parity": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divisions", type=int, default=6)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--assembly-iterations", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--t-end", type=float, default=0.04)
    args = parser.parse_args()
    if args.divisions <= 0 or args.batches <= 0 or args.assembly_iterations <= 0:
        parser.error("divisions, batches, and assembly-iterations must be positive")

    # Compile kernels and populate retained geometry before timed batches.
    _impact(min(args.divisions, 2), args.dt, min(args.t_end, 0.01), direct=True)

    full_times = []
    direct_times = []
    last_full = None
    last_direct = None
    gc.disable()
    try:
        for batch in range(args.batches):
            paths = (False, True) if batch % 2 == 0 else (True, False)
            for direct in paths:
                start = time.perf_counter()
                result = _impact(args.divisions, args.dt, args.t_end, direct)
                elapsed = time.perf_counter() - start
                if direct:
                    direct_times.append(elapsed)
                    last_direct = result
                else:
                    full_times.append(elapsed)
                    last_full = result
    finally:
        gc.enable()

    assert last_full is not None and last_direct is not None
    if last_full.status != last_direct.status:
        raise AssertionError("full and direct impact statuses differ")
    np.testing.assert_allclose(
        last_direct.displacements,
        last_full.displacements,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        last_direct.contact_force_history,
        last_full.contact_force_history,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    full_median = float(statistics.median(full_times))
    direct_median = float(statistics.median(direct_times))
    full_count = int(last_full.diagnostics["full_coordinate_assembly_count"])
    direct_full_count = int(last_direct.diagnostics["full_coordinate_assembly_count"])
    output = {
        "model": {
            "divisions": int(args.divisions),
            "elements": int(args.divisions**2),
            "nodes": int((args.divisions + 1) ** 2),
        },
        "impact": {
            "batches": int(args.batches),
            "full_median_s": full_median,
            "direct_median_s": direct_median,
            "speedup": full_median / max(direct_median, 1.0e-30),
            "full_coordinate_assembly_count": full_count,
            "direct_path_full_coordinate_assembly_count": direct_full_count,
            "direct_reduced_assembly_count": int(
                last_direct.diagnostics["direct_reduced_assembly_count"]
            ),
            "full_assembly_avoided_fraction": 1.0
            - direct_full_count / max(full_count, 1),
            "direct_diagnostics": last_direct.diagnostics[
                "impact_reduced_assembly"
            ],
            "parity": True,
        },
        "assembly_kernel": _assembly_microbenchmark(
            args.divisions,
            args.assembly_iterations,
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
