"""Paired screen for nonlinear-static reaction reuse and Armijo work.

The timing comparison runs in one warmed process. Pair order alternates and
the same-revision oracle forces the guarded reaction-recovery fallback, so the
measured difference isolates accepted-force reuse without repeatedly charging
Python or optional-backend startup. It is not a revision-to-revision baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from anyfesolver import nonlinear_performance_batch_c
from anyfesolver.boundary import BoundaryCondition, FixedSupport, LoadCase
from anyfesolver.elements import Element
from anyfesolver.fe_core import FEModel
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.nonlinear_performance_bootstrap import nonlinear_performance_status
from anyfesolver.nonlinear_static import solve_static_nonlinear


class _AxialSpring(Element):
    """Small exact element used to hold benchmark physics constant."""

    def __init__(
        self,
        element_id: int,
        node_ids: list[int],
        *,
        stiffness: float = 1.0,
        cubic: float = 0.0,
    ) -> None:
        super().__init__(element_id, node_ids, "default")
        self.stiffness = float(stiffness)
        self.cubic = float(cubic)

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh):
        return np.asarray(
            [mesh.get_node(node_id).coords() for node_id in self.node_ids],
            dtype=float,
        )

    @staticmethod
    def _matrix(value: float) -> np.ndarray:
        matrix = np.zeros((12, 12), dtype=float)
        matrix[0, 0] = value
        matrix[0, 6] = -value
        matrix[6, 0] = -value
        matrix[6, 6] = value
        return matrix

    def compute_stiffness_matrix(self, mesh, material):
        return self._matrix(self.stiffness)

    def compute_nonlinear_response(
        self,
        mesh,
        material,
        u_elem,
        state=None,
        num_layers: int = 5,
        tangent: bool = True,
    ):
        del state, num_layers
        displacement = np.asarray(u_elem, dtype=float)
        extension = float(displacement[6] - displacement[0])
        force_value = self.stiffness * extension + self.cubic * extension**3
        force = np.zeros(12, dtype=float)
        force[0], force[6] = -force_value, force_value
        tangent_value = self.stiffness + 3.0 * self.cubic * extension**2
        matrix = self._matrix(tangent_value) if tangent else None
        return force, matrix, {"extension": extension}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser


def _shell_case() -> tuple[FEModel, LoadCase, dict[str, Any]]:
    model = generate_simple_panel_mesh(
        1.0,
        1.0,
        0.01,
        num_divisions_x=2,
        num_divisions_y=2,
    )
    model.name = "reaction-reuse-clamped-shell"
    model.clear_boundary_conditions()
    edge_nodes = []
    for node_id, node in model.mesh.nodes.items():
        x, y, _z = node.coords()
        if min(x, y, 1.0 - x, 1.0 - y) <= 1.0e-12:
            edge_nodes.append(int(node_id))
    model.add_boundary_condition(
        BoundaryCondition(
            "clamped-translations",
            edge_nodes,
            {"ux": 0.0, "uy": 0.0, "uz": 0.0},
        )
    )
    load = LoadCase("fixed-pressure")
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), 2.0e4)
    return model, load, {
        "num_steps": 40,
        "max_iterations": 20,
        "tolerance": 1.0e-9,
        "convergence_settings": "legacy",
    }


def _hardening_case() -> tuple[FEModel, LoadCase, dict[str, Any]]:
    model = FEModel("armijo-hardening-spring")
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    model.add_element(1, _AxialSpring(1, [1, 2], cubic=10.0))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(
        BoundaryCondition(
            "guide",
            [2],
            {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    load = LoadCase("pull")
    load.add_nodal_load(2, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, load, {
        "num_steps": 1,
        "max_iterations": 25,
        "tolerance": 1.0e-11,
    }


def _physical_digest(result) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(result.displacements, dtype="<f8").tobytes())
    for step in result.steps:
        digest.update(
            np.asarray(
                [
                    step.load_factor,
                    step.residual_norm,
                    step.displacement_norm,
                    step.max_equivalent_plastic_strain,
                ],
                dtype="<f8",
            ).tobytes()
        )
        for name, values in sorted(step.support_reactions.items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _solve_shell(*, reuse: bool) -> dict[str, Any]:
    original = nonlinear_performance_batch_c.materialize_full_internal_force
    if not reuse:
        nonlinear_performance_batch_c.materialize_full_internal_force = (
            lambda payload, total_dofs: None
        )
    started = time.perf_counter()
    try:
        model, load, options = _shell_case()
        result = solve_static_nonlinear(model, load, **options)
    finally:
        nonlinear_performance_batch_c.materialize_full_internal_force = original
    elapsed = time.perf_counter() - started
    performance = result.info["nonlinear_performance"]
    solver = performance["solver"]
    recovery = result.info["reaction_force_recovery"]
    return {
        "complete_route_wall_seconds": elapsed,
        "solver_seconds": float(result.info.get("solve_time", 0.0)),
        "status": result.status,
        "load_factor": float(result.load_factor),
        "steps": len(result.steps),
        "iterations": int(result.info.get("total_newton_iterations", 0)),
        "physical_sha256": _physical_digest(result),
        "reaction_force_reuse_count": int(
            recovery["accepted_force_reuse_count"]
        ),
        "reaction_force_reassembly_count": int(recovery["full_reassembly_count"]),
        "assembly_calls": int(performance["assembly"]["calls"]),
        "tangent_calls": int(performance["assembly"]["tangent_calls"]),
        "residual_only_calls": int(
            performance["assembly"]["residual_only_calls"]
        ),
        "direct_reduced_assembly": bool(
            performance["direct_reduced_assembly"]["activated"]
        ),
        "linear_solves": int(solver["linear_solves"]),
        "linear_factorizations": int(solver["linear_factorizations"]),
        "rejected_full_steps": int(solver["rejected_full_steps"]),
        "backtracks": int(solver["backtracks"]),
        "promotion_evaluations": int(solver["promotion_evaluations"]),
        "failed_work": dict(solver["failed_work"]),
    }


def _globalization_case(line_search: str) -> dict[str, Any]:
    model, load, options = _hardening_case()
    started = time.perf_counter()
    result = solve_static_nonlinear(
        model,
        load,
        convergence_settings={"profile": "legacy", "line_search": line_search},
        **options,
    )
    elapsed = time.perf_counter() - started
    solver = result.info["nonlinear_performance"]["solver"]
    return {
        "complete_route_wall_seconds": elapsed,
        "status": result.status,
        "load_factor": float(result.load_factor),
        "iterations": int(result.info.get("total_newton_iterations", 0)),
        "physical_sha256": _physical_digest(result),
        "linear_solves": int(solver["linear_solves"]),
        "rejected_full_steps": int(solver["rejected_full_steps"]),
        "backtracks": int(solver["backtracks"]),
        "failed_work": dict(solver["failed_work"]),
    }


def _median(samples: list[dict[str, Any]], field: str) -> float:
    return float(statistics.median(float(sample[field]) for sample in samples))


def _memory_probe(*, reuse: bool) -> dict[str, Any]:
    tracemalloc.start()
    try:
        sample = _solve_shell(reuse=reuse)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "python_traced_peak_bytes": int(peak),
        "status": sample["status"],
        "physical_sha256": sample["physical_sha256"],
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("anyfesolver", "numpy", "scipy", "numba"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _source_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _main(args: argparse.Namespace) -> int:
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    # Charge optional backend/JIT setup to neither side of the paired screen.
    _solve_shell(reuse=False)
    _solve_shell(reuse=True)

    oracle_name = "forced_reassembly_oracle"
    candidate_name = "accepted_force_reuse"
    samples: dict[str, list[dict[str, Any]]] = {
        oracle_name: [],
        candidate_name: [],
    }
    pairs = []
    implementations: dict[str, Callable[[], dict[str, Any]]] = {
        oracle_name: lambda: _solve_shell(reuse=False),
        candidate_name: lambda: _solve_shell(reuse=True),
    }
    for pair_index in range(args.repeats):
        order = (
            [oracle_name, candidate_name]
            if pair_index % 2 == 0
            else [candidate_name, oracle_name]
        )
        pair: dict[str, Any] = {"pair": pair_index + 1, "order": order}
        for name in order:
            sample = implementations[name]()
            samples[name].append(sample)
            pair[name] = sample
        pairs.append(pair)

    oracle_time = _median(samples[oracle_name], "complete_route_wall_seconds")
    candidate_time = _median(samples[candidate_name], "complete_route_wall_seconds")
    physical_match = all(
        oracle["status"] == candidate["status"] == "completed"
        and oracle["physical_sha256"] == candidate["physical_sha256"]
        for oracle, candidate in zip(samples[oracle_name], samples[candidate_name])
    )
    work_reduction = all(
        oracle["reaction_force_reassembly_count"] > 0
        and candidate["reaction_force_reassembly_count"] == 0
        and candidate["reaction_force_reuse_count"] > 0
        for oracle, candidate in zip(samples[oracle_name], samples[candidate_name])
    )
    convergence = {
        "residual_decrease": _globalization_case("always"),
        "armijo": _globalization_case("armijo"),
    }
    memory = {
        oracle_name: _memory_probe(reuse=False),
        candidate_name: _memory_probe(reuse=True),
        "scope": "one warmed solve; Python allocations observed by tracemalloc",
    }
    payload = {
        "schema": "anyfesolver.nonlinear_static.performance_convergence",
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
            "nonlinear_performance": nonlinear_performance_status(),
            "source_revision": _source_revision(),
        },
        "protocol": {
            "case": "2x2 clamped Q4 shell under fixed pressure",
            "repeats": int(args.repeats),
            "serial": True,
            "alternating_pair_order": True,
            "warmup_runs_per_variant": 1,
            "resource_config": "solver default; no explicit memory or thread limit",
            "comparison_scope": (
                "same-revision in-process oracle; no revision baseline claim"
            ),
            "forced_reassembly_oracle": (
                "accepted-force materialization disabled to force guarded "
                "reaction reassembly"
            ),
            "accepted_force_reuse": (
                "generation-checked accepted force used for reaction recovery"
            ),
        },
        "pairs": pairs,
        "memory_probe": memory,
        "summary": {
            "physical_match": physical_match,
            "reaction_reassembly_eliminated": work_reduction,
            "oracle_median_seconds": oracle_time,
            "reuse_median_seconds": candidate_time,
            "reuse_over_oracle": candidate_time / oracle_time,
            "median_time_change_percent": 100.0
            * (candidate_time / oracle_time - 1.0),
        },
        "convergence_screen": convergence,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    return 0 if physical_match and work_reduction else 2


def main() -> int:
    return _main(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
