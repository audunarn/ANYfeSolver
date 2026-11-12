from __future__ import annotations

import argparse
import sys
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy import sparse

from anyfesolver import nonlinear_performance
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.nonlinear_performance_batch_c import (
    assemble_reduced_system,
    build_reduced_assembly_plan,
)
from anyfesolver.nonlinear_performance_bootstrap import (
    clear_nonlinear_assembly_cache,
    get_nonlinear_assembly_plan,
    install_nonlinear_performance_optimizations,
)


def _time_call(function, repeats: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples


def _ensure_performance_layer():
    """Activate the existing fast layer and return its retained scalar oracle."""

    if not install_nonlinear_performance_optimizations():
        raise RuntimeError(
            "Performance layer is disabled; unset FE_SOLVER_DISABLE_FAST_NL "
            "before running this benchmark"
        )
    original = nonlinear_performance._ORIGINAL_ASSEMBLER
    if original is None:
        raise RuntimeError(
            "Performance layer installation did not retain the original assembler"
        )
    return original


def _constraint_transformation(model, weighted_mpc_rows: int) -> sparse.csr_matrix:
    mesh = model.mesh
    total_dofs = int(mesh.dof_manager.total_dofs)
    minimum_x = min(float(node.x) for node in mesh.nodes.values())
    fixed = {
        int(dof)
        for node in mesh.nodes.values()
        if np.isclose(float(node.x), minimum_x)
        for dof in node.dofs
    }
    independent = np.asarray(
        [dof for dof in range(total_dofs) if dof not in fixed],
        dtype=np.intp,
    )
    if independent.size < 2:
        raise RuntimeError("Benchmark model does not contain enough independent DOFs")
    independent_index = {
        int(dof): index for index, dof in enumerate(independent)
    }
    rows = independent.tolist()
    columns = [independent_index[int(dof)] for dof in independent]
    values = [1.0] * independent.size

    slave_candidates = sorted(fixed)[: max(int(weighted_mpc_rows), 0)]
    for offset, slave in enumerate(slave_candidates):
        first = offset % independent.size
        second = (offset + 1) % independent.size
        rows.extend((slave, slave))
        columns.extend((first, second))
        values.extend((0.75, 0.25))

    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(total_dofs, independent.size),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy, persistent full-CSR and direct reduced nonlinear "
            "assembly"
        )
    )
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument(
        "--weighted-mpc-rows",
        type=int,
        default=0,
        help="Map this many otherwise-fixed DOFs to two reduced masters",
    )
    args = parser.parse_args()

    original = _ensure_performance_layer()

    model = generate_simple_panel_mesh(
        4.0,
        2.0,
        0.012,
        num_divisions_x=args.nx,
        num_divisions_y=args.ny,
    )
    transformation = _constraint_transformation(model, args.weighted_mpc_rows)
    rng = np.random.default_rng(20260618)
    reduced_displacement = rng.normal(
        scale=2.0e-5,
        size=transformation.shape[1],
    )
    displacement = np.asarray(
        transformation @ reduced_displacement,
        dtype=float,
    ).reshape(-1)
    committed = {}

    original(model, displacement, committed, args.layers, tangent=True)
    clear_nonlinear_assembly_cache(model)
    nonlinear_plan = get_nonlinear_assembly_plan(model, args.layers)
    nonlinear_plan.assemble(displacement, committed, tangent=True)
    reduced_plan = build_reduced_assembly_plan(
        nonlinear_plan,
        transformation,
    )
    assemble_reduced_system(
        nonlinear_plan,
        reduced_plan,
        displacement,
        committed,
        tangent=True,
    )

    def legacy_reduced():
        force, tangent, states = original(
            model,
            displacement,
            committed,
            args.layers,
            tangent=True,
        )
        return (
            np.asarray(transformation.T @ force, dtype=float).reshape(-1),
            (transformation.T @ tangent @ transformation).tocsr(),
            states,
        )

    def persistent_full_reduced():
        force, tangent, states = nonlinear_plan.assemble(
            displacement,
            committed,
            tangent=True,
        )
        return (
            np.asarray(transformation.T @ force, dtype=float).reshape(-1),
            (transformation.T @ tangent @ transformation).tocsr(),
            states,
        )

    def direct_reduced():
        return assemble_reduced_system(
            nonlinear_plan,
            reduced_plan,
            displacement,
            committed,
            tangent=True,
        )

    legacy_median, legacy_samples = _time_call(
        legacy_reduced,
        args.repeats,
    )
    full_median, full_samples = _time_call(
        persistent_full_reduced,
        args.repeats,
    )
    direct_median, direct_samples = _time_call(
        direct_reduced,
        args.repeats,
    )

    full_speedup = (
        legacy_median / full_median if full_median > 0.0 else float("inf")
    )
    direct_speedup = (
        legacy_median / direct_median if direct_median > 0.0 else float("inf")
    )
    batch_c_speedup = (
        full_median / direct_median if direct_median > 0.0 else float("inf")
    )
    print(f"elements: {model.mesh.num_elements}")
    print(f"full DOFs: {model.mesh.dof_manager.total_dofs}")
    print(f"reduced DOFs: {transformation.shape[1]}")
    print(f"weighted MPC rows: {args.weighted_mpc_rows}")
    print(f"legacy + projection median: {legacy_median:.6f} s")
    print(f"persistent full + projection median: {full_median:.6f} s")
    print(f"direct reduced median: {direct_median:.6f} s")
    print(f"persistent speedup vs legacy: {full_speedup:.3f}x")
    print(f"direct speedup vs legacy: {direct_speedup:.3f}x")
    print(f"Batch C speedup vs full projection: {batch_c_speedup:.3f}x")
    print(f"legacy samples: {[round(value, 6) for value in legacy_samples]}")
    print(f"full samples: {[round(value, 6) for value in full_samples]}")
    print(f"direct samples: {[round(value, 6) for value in direct_samples]}")
    print(f"nonlinear plan diagnostics: {nonlinear_plan.diagnostics()}")
    print(f"reduced plan diagnostics: {reduced_plan.diagnostics()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
