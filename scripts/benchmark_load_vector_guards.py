"""Benchmark qualified-lifecycle guard cost in load-vector/load-matrix assembly.

Builds a synthetic mixed model of barge-example size (default: about 4.3k
nodes, 3.2k qualified Q4/S3 shells and 1.0k generic beams) and reports the
per-call cost of ``assemble_load_vector`` and the per-case cost of
``assemble_load_matrix`` for several load-case shapes. It also prints a
SHA-256 of the load vectors so before/after runs can be compared bit-for-bit.

    python scripts/benchmark_load_vector_guards.py [--cases 10] [--profile]
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anyfesolver.boundary import LoadCase  # noqa: E402
from anyfesolver.e4_pl_element import QualifiedE4PLShellElement  # noqa: E402
from anyfesolver.e4_pl_s3_element import QualifiedE4PLS3ShellElement  # noqa: E402
from anyfesolver.elements import BeamElement  # noqa: E402
from anyfesolver.fe_core import FEModel  # noqa: E402
from anyfesolver.matrix_assembly import (  # noqa: E402
    assemble_load_matrix,
    assemble_load_vector,
)


def build_model(
    grid: int = 66, shells: int = 3185, beams: int = 1060
) -> tuple[FEModel, list[int]]:
    model = FEModel("load-vector-guard-benchmark")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    node = lambda i, j: i * grid + j + 1  # noqa: E731
    for i in range(grid):
        for j in range(grid):
            model.add_node(node(i, j), 0.7 * i, 0.7 * j, 0.0)
    shell_ids: list[int] = []
    element_id = 1
    cells = [(i, j) for i in range(grid - 1) for j in range(grid - 1)]
    for i, j in cells:
        if len(shell_ids) >= shells:
            break
        n1, n2, n3, n4 = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
        if (i + j) % 3 == 0 and len(shell_ids) + 2 <= shells:
            for ids in ([n1, n2, n3], [n1, n3, n4]):
                model.add_element(
                    element_id,
                    QualifiedE4PLS3ShellElement(
                        element_id, ids, "steel", thickness=0.01,
                        reference_normal=(0.0, 0.0, 1.0),
                    ),
                )
                shell_ids.append(element_id)
                element_id += 1
        else:
            model.add_element(
                element_id,
                QualifiedE4PLShellElement(
                    element_id, [n1, n2, n3, n4], "steel", thickness=0.01
                ),
            )
            shell_ids.append(element_id)
            element_id += 1
    placed = 0
    for i in range(grid - 1):
        for j in range(grid):
            if placed >= beams:
                break
            model.add_element(
                element_id,
                BeamElement(
                    element_id, [node(i, j), node(i + 1, j)], "steel",
                    cross_section={"area": 0.004, "Iy": 1e-5, "Iz": 1e-5, "J": 1e-5},
                ),
            )
            element_id += 1
            placed += 1
    return model, shell_ids


def make_cases(model: FEModel, shell_ids: list[int], count: int, kind: str):
    cases = []
    rng = np.random.default_rng(7)
    for k in range(count):
        lc = LoadCase(f"{kind}-{k}")
        if kind == "nodal":
            lc.add_nodal_load(1 + k, forces=[1.0e3 * (k + 1), 0.0, -2.0e3])
        elif kind in ("acceleration", "gravity"):
            if kind == "gravity":
                lc.set_gravity(gz=-9.81 * (1.0 + k / max(count, 1)))
            else:
                lc.set_acceleration(ax=0.7 * (k + 1), ay=-0.3, az=-9.81)
            lc.add_node_mass(1, 12.0 + k)
        elif kind == "pressure1":
            lc.add_pressure_load(shell_ids[k % len(shell_ids)], 1.0e4 * (k + 1))
        elif kind == "pressure_all":
            for sid in shell_ids:
                lc.add_pressure_load(sid, 1.0e4 * (1.0 + rng.random()))
        else:
            raise ValueError(kind)
        cases.append(lc)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--kinds", default="nodal,pressure1,pressure_all")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--grid", type=int, default=66)
    parser.add_argument("--shells", type=int, default=3185)
    parser.add_argument("--beams", type=int, default=1060)
    args = parser.parse_args()

    t0 = time.perf_counter()
    model, shell_ids = build_model(args.grid, args.shells, args.beams)
    mesh = model.mesh
    print(
        f"model: nodes={mesh.num_nodes} elements={len(mesh.elements)} "
        f"shells={len(shell_ids)} beams={len(mesh.elements) - len(shell_ids)} "
        f"build={time.perf_counter() - t0:.1f}s"
    )
    digest = hashlib.sha256()
    for kind in args.kinds.split(","):
        cases = make_cases(model, shell_ids, args.cases, kind)
        samples = []
        for lc in cases[: min(3, len(cases))]:
            start = time.perf_counter()
            vector, _info = assemble_load_vector(model, lc)
            samples.append(time.perf_counter() - start)
            digest.update(np.ascontiguousarray(vector).tobytes())
        print(
            f"assemble_load_vector[{kind}]: median {statistics.median(samples):.6f}s "
            f"(n={len(samples)})"
        )
        if args.profile:
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()
            matrix, _ = assemble_load_matrix(model, cases)
            profiler.disable()
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(18)
        else:
            start = time.perf_counter()
            matrix, _ = assemble_load_matrix(model, cases)
            elapsed = time.perf_counter() - start
            print(
                f"assemble_load_matrix[{kind}]: {elapsed:.6f}s for {len(cases)} cases "
                f"= {elapsed / len(cases):.6f}s/case"
            )
        digest.update(np.ascontiguousarray(matrix).tobytes())
    print(f"load-vector sha256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
