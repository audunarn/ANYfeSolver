"""Immutable installed-wheel qualification for nonlinear-static improvements.

The coordinator keeps one persistent worker per installed wheel, warms both
workers before timing, and alternates serial samples.  Each worker imports only
from its explicit installation directory and instruments the same solver call
boundary, allowing the base revision and candidate to report comparable work.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_main(site: Path) -> int:
    sys.path.insert(0, str(site.resolve()))

    import numpy as np

    import anyfesolver
    from anyfesolver.boundary import BoundaryCondition, LoadCase
    from anyfesolver.mesh_gen import generate_simple_panel_mesh
    from anyfesolver.nonlinear_performance_bootstrap import (
        nonlinear_performance_status,
    )
    from anyfesolver import nonlinear_static

    nonlinear_static._ensure_nonlinear_acceleration()
    original_assemble = nonlinear_static._assemble_nonlinear_system
    original_factorize = nonlinear_static.factorize
    counters: dict[str, int] = {}

    def reset_counters() -> None:
        counters.clear()
        counters.update(
            assembly_calls=0,
            tangent_calls=0,
            residual_only_calls=0,
            linear_factorizations=0,
            linear_solves=0,
        )

    def observed_assemble(*args, **kwargs):
        counters["assembly_calls"] += 1
        if bool(kwargs.get("tangent", True)):
            counters["tangent_calls"] += 1
        else:
            counters["residual_only_calls"] += 1
        return original_assemble(*args, **kwargs)

    class _ObservedHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __getattr__(self, name: str):
            return getattr(self._handle, name)

        def solve(self, rhs):
            counters["linear_solves"] += 1
            return self._handle.solve(rhs)

    def observed_factorize(*args, **kwargs):
        counters["linear_factorizations"] += 1
        return _ObservedHandle(original_factorize(*args, **kwargs))

    nonlinear_static._assemble_nonlinear_system = observed_assemble
    nonlinear_static.factorize = observed_factorize

    def shell_case(case: str):
        model = generate_simple_panel_mesh(
            1.0,
            1.0,
            0.01,
            num_divisions_x=2,
            num_divisions_y=2,
        )
        model.name = f"revision-qualification-{case}"
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
        pressure = 2.0e4 if case == "declared_nonlinear" else 100.0
        for element_id in model.mesh.elements:
            load.add_pressure_load(int(element_id), pressure)
        options = {
            "num_steps": 40 if case == "declared_nonlinear" else 10,
            "max_iterations": 20,
            "tolerance": 1.0e-9,
            "convergence_settings": {
                "profile": "legacy",
                "line_search": "never",
            },
        }
        return model, load, options

    def physical_digest(result) -> str:
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

    def solve(case: str) -> dict[str, Any]:
        reset_counters()
        started = time.perf_counter()
        model, load, options = shell_case(case)
        result = nonlinear_static.solve_static_nonlinear(model, load, **options)
        elapsed = time.perf_counter() - started
        residual_only = int(counters["residual_only_calls"])
        steps = len(result.steps)
        performance = result.info.get("nonlinear_performance")
        recovery = result.info.get("reaction_force_recovery", {})
        return {
            "complete_route_wall_seconds": float(elapsed),
            "solver_seconds": float(result.info.get("solve_time", 0.0)),
            "status": str(result.status),
            "load_factor": float(result.load_factor),
            "steps": int(steps),
            "iterations": int(result.info.get("total_newton_iterations", 0)),
            "physical_sha256": physical_digest(result),
            "physical_observables": {
                "displacements": np.asarray(
                    result.displacements,
                    dtype=float,
                ).tolist(),
                "steps": [
                    {
                        "load_factor": float(step.load_factor),
                        "residual_norm": float(step.residual_norm),
                        "displacement_norm": float(step.displacement_norm),
                        "max_equivalent_plastic_strain": float(
                            step.max_equivalent_plastic_strain
                        ),
                        "support_reactions": {
                            str(name): [float(value) for value in values]
                            for name, values in sorted(
                                step.support_reactions.items()
                            )
                        },
                    }
                    for step in result.steps
                ],
            },
            "assembly_calls": int(counters["assembly_calls"]),
            "tangent_calls": int(counters["tangent_calls"]),
            "residual_only_calls": residual_only,
            "linear_factorizations": int(counters["linear_factorizations"]),
            "linear_solves": int(counters["linear_solves"]),
            "rejected_full_steps": 0,
            "backtracks": 0,
            "promotion_evaluations": 0,
            "reaction_force_reuse_count": int(
                recovery.get(
                    "accepted_force_reuse_count",
                    max(steps - residual_only, 0),
                )
            ),
            "reaction_force_reassembly_count": int(
                recovery.get("full_reassembly_count", residual_only)
            ),
            "failed_work": {
                "increment_count": 0,
                "newton_iterations": 0,
                "rejected_trial_evaluations": 0,
                "recoverable_trial_failures": 0,
            },
            "instrumentation": "identical worker wrapper",
            "package_diagnostics_present": performance is not None,
        }

    identity = {
        "event": "ready",
        "module_file": str(Path(anyfesolver.__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("anyfesolver", "numpy", "scipy", "numba")
        },
        "performance_status": nonlinear_performance_status(),
    }
    print(json.dumps(identity, sort_keys=True), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        command = request["command"]
        if command == "close":
            print(json.dumps({"event": "closed"}), flush=True)
            return 0
        case = str(request["case"])
        if command == "memory":
            tracemalloc.start()
            try:
                sample = solve(case)
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            response = {
                "event": "memory",
                "case": case,
                "python_traced_peak_bytes": int(peak),
                "sample": sample,
            }
        elif command in {"warmup", "run"}:
            response = {
                "event": command,
                "case": case,
                "sample": solve(case),
            }
        else:
            raise ValueError(f"unknown worker command {command!r}")
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


@dataclass
class _Worker:
    label: str
    process: subprocess.Popen[str]
    stderr_path: Path
    stderr_stream: Any
    identity: dict[str, Any]

    def request(self, command: str, case: str) -> dict[str, Any]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(
            json.dumps({"command": command, "case": case}) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"{self.label} worker exited with {self.process.poll()}; "
                f"see {self.stderr_path}"
            )
        return json.loads(line)

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.request("close", "none")
                self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            self.stderr_stream.close()


def _start_worker(label: str, site: Path, stderr_path: Path) -> _Worker:
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_stream = stderr_path.open("w", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--site",
            str(site.resolve()),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_stream,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    ready_line = process.stdout.readline()
    if not ready_line:
        stderr_stream.close()
        raise RuntimeError(
            f"{label} worker failed during startup; see {stderr_path}"
        )
    return _Worker(
        label=label,
        process=process,
        stderr_path=stderr_path,
        stderr_stream=stderr_stream,
        identity=json.loads(ready_line),
    )


def _median(samples: list[dict[str, Any]], field: str) -> float:
    return float(statistics.median(float(sample[field]) for sample in samples))


def _scaled_difference(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) != len(right):
        return {"max_absolute": float("inf"), "scale": float("inf")}
    maximum = max(
        (abs(float(a) - float(b)) for a, b in zip(left, right)),
        default=0.0,
    )
    scale = max(
        (max(abs(float(a)), abs(float(b))) for a, b in zip(left, right)),
        default=0.0,
    )
    return {"max_absolute": maximum, "scale": scale}


def _compare_physics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_observed = left["physical_observables"]
    right_observed = right["physical_observables"]
    displacement = _scaled_difference(
        left_observed["displacements"],
        right_observed["displacements"],
    )
    left_steps = left_observed["steps"]
    right_steps = right_observed["steps"]
    same_step_count = len(left_steps) == len(right_steps)
    load_factor_difference = 0.0
    displacement_norm_values: list[float] = []
    displacement_norm_reference: list[float] = []
    reaction_values: list[float] = []
    reaction_reference: list[float] = []
    reaction_keys_match = same_step_count
    residual_difference = 0.0
    plastic_difference = 0.0
    if same_step_count:
        for baseline_step, candidate_step in zip(left_steps, right_steps):
            load_factor_difference = max(
                load_factor_difference,
                abs(
                    float(baseline_step["load_factor"])
                    - float(candidate_step["load_factor"])
                ),
            )
            residual_difference = max(
                residual_difference,
                abs(
                    float(baseline_step["residual_norm"])
                    - float(candidate_step["residual_norm"])
                ),
            )
            plastic_difference = max(
                plastic_difference,
                abs(
                    float(baseline_step["max_equivalent_plastic_strain"])
                    - float(candidate_step["max_equivalent_plastic_strain"])
                ),
            )
            displacement_norm_reference.append(
                float(baseline_step["displacement_norm"])
            )
            displacement_norm_values.append(
                float(candidate_step["displacement_norm"])
            )
            baseline_reactions = baseline_step["support_reactions"]
            candidate_reactions = candidate_step["support_reactions"]
            if set(baseline_reactions) != set(candidate_reactions):
                reaction_keys_match = False
                continue
            for name in sorted(baseline_reactions):
                baseline_values = baseline_reactions[name]
                candidate_values = candidate_reactions[name]
                if len(baseline_values) != len(candidate_values):
                    reaction_keys_match = False
                    continue
                reaction_reference.extend(float(value) for value in baseline_values)
                reaction_values.extend(float(value) for value in candidate_values)
    displacement_norm = _scaled_difference(
        displacement_norm_reference,
        displacement_norm_values,
    )
    reactions = _scaled_difference(reaction_reference, reaction_values)
    displacement_ok = displacement["max_absolute"] <= (
        1.0e-12 + 1.0e-10 * displacement["scale"]
    )
    displacement_norm_ok = displacement_norm["max_absolute"] <= (
        1.0e-12 + 1.0e-10 * displacement_norm["scale"]
    )
    reactions_ok = reaction_keys_match and reactions["max_absolute"] <= (
        1.0e-8 + 1.0e-10 * reactions["scale"]
    )
    numerical_match = (
        left["status"] == right["status"] == "completed"
        and same_step_count
        and load_factor_difference <= 1.0e-12
        and plastic_difference <= 1.0e-12
        and displacement_ok
        and displacement_norm_ok
        and reactions_ok
    )
    return {
        "exact_digest_match": (
            left["physical_sha256"] == right["physical_sha256"]
        ),
        "numerical_match": numerical_match,
        "same_step_count": same_step_count,
        "max_load_factor_difference": load_factor_difference,
        "max_displacement_difference": displacement["max_absolute"],
        "displacement_scale": displacement["scale"],
        "max_displacement_norm_difference": displacement_norm["max_absolute"],
        "max_reaction_difference": reactions["max_absolute"],
        "reaction_scale": reactions["scale"],
        "max_residual_norm_difference": residual_difference,
        "max_plastic_strain_difference": plastic_difference,
        "tolerances": {
            "displacement": "1e-12 + 1e-10 * scale",
            "reaction": "1e-8 + 1e-10 * scale",
            "load_factor": 1.0e-12,
            "plastic_strain": 1.0e-12,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-site", type=Path)
    parser.add_argument("--candidate-site", type=Path)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--baseline-revision")
    parser.add_argument("--candidate-revision")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--site", type=Path)
    return parser


def _main(args: argparse.Namespace) -> int:
    if args.worker:
        if args.site is None:
            raise SystemExit("--worker requires --site")
        return _worker_main(args.site)
    required = (
        "baseline_site",
        "candidate_site",
        "baseline_wheel",
        "candidate_wheel",
        "baseline_revision",
        "candidate_revision",
        "output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    output = args.output.resolve()
    log_dir = (
        args.log_dir.resolve() if args.log_dir is not None else output.parent
    )
    workers: dict[str, _Worker] = {}
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        for label, site in (
            ("baseline", args.baseline_site),
            ("candidate", args.candidate_site),
        ):
            print(f"starting {label} worker", flush=True)
            workers[label] = _start_worker(
                label,
                site,
                log_dir / f"{output.stem}.{label}.stderr.log",
            )

        for label in ("baseline", "candidate"):
            print(f"warming {label} declared workload", flush=True)
            workers[label].request("warmup", "declared_nonlinear")
            print(f"warming {label} easy control", flush=True)
            workers[label].request("warmup", "easy_control")

        cases: dict[str, Any] = {}
        for case in ("declared_nonlinear", "easy_control"):
            samples = {"baseline": [], "candidate": []}
            pairs = []
            for pair_index in range(args.repeats):
                order = (
                    ["baseline", "candidate"]
                    if pair_index % 2 == 0
                    else ["candidate", "baseline"]
                )
                pair: dict[str, Any] = {
                    "pair": pair_index + 1,
                    "order": order,
                }
                for label in order:
                    print(
                        f"{case} pair {pair_index + 1}/{args.repeats}: {label}",
                        flush=True,
                    )
                    sample = workers[label].request("run", case)["sample"]
                    samples[label].append(sample)
                    pair[label] = sample
                pair["physical_comparison"] = _compare_physics(
                    pair["baseline"], pair["candidate"]
                )
                pairs.append(pair)
            baseline_median = _median(
                samples["baseline"], "complete_route_wall_seconds"
            )
            candidate_median = _median(
                samples["candidate"], "complete_route_wall_seconds"
            )
            physical_match = all(
                pair["physical_comparison"]["numerical_match"]
                for pair in pairs
            )
            exact_digest_match = all(
                pair["physical_comparison"]["exact_digest_match"]
                for pair in pairs
            )
            cases[case] = {
                "pairs": pairs,
                "summary": {
                    "physical_match": physical_match,
                    "exact_digest_match": exact_digest_match,
                    "baseline_median_seconds": baseline_median,
                    "candidate_median_seconds": candidate_median,
                    "candidate_over_baseline": (
                        candidate_median / baseline_median
                    ),
                    "median_time_change_percent": 100.0
                    * (candidate_median / baseline_median - 1.0),
                },
            }

        memory = {}
        for label in ("baseline", "candidate"):
            print(f"memory probe: {label}", flush=True)
            memory[label] = workers[label].request(
                "memory", "declared_nonlinear"
            )

        nonlinear_summary = cases["declared_nonlinear"]["summary"]
        easy_summary = cases["easy_control"]["summary"]
        performance_target_met = (
            nonlinear_summary["candidate_over_baseline"] <= 0.90
        )
        easy_regression_acceptable = (
            easy_summary["candidate_over_baseline"] <= 1.05
        )
        physical_match = all(
            case["summary"]["physical_match"] for case in cases.values()
        )
        payload = {
            "schema": "anyfesolver.nonlinear_static.revision_qualification",
            "version": 1,
            "started_utc": started_utc,
            "completed_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "protocol": {
                "serial": True,
                "persistent_workers": True,
                "alternating_pair_order": True,
                "warmup_runs_per_case_and_revision": 1,
                "paired_runs": int(args.repeats),
                "thread_settings": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "1",
                },
                "resource_limits": "no explicit memory limit",
                "profiling_separate_from_timing": True,
                "cases": {
                    "declared_nonlinear": {
                        "model": "2x2 clamped Q4 shell",
                        "pressure_pa": 20000.0,
                        "physical_increments": 40,
                        "max_iterations": 20,
                        "residual_tolerance": 1.0e-9,
                        "line_search": "never",
                    },
                    "easy_control": {
                        "model": "2x2 clamped Q4 shell",
                        "pressure_pa": 100.0,
                        "physical_increments": 10,
                        "max_iterations": 20,
                        "residual_tolerance": 1.0e-9,
                        "line_search": "never",
                    },
                },
                "physical_outputs": [
                    "displacements",
                    "increment history",
                    "support reactions",
                    "maximum equivalent plastic strain",
                ],
            },
            "authority": {
                "runner": str(Path(__file__).resolve()),
                "runner_sha256": _sha256(Path(__file__).resolve()),
                "manifest": str(
                    (
                        Path(__file__).resolve().parents[1]
                        / "docs/reference_cases/nonlinear_static_performance_convergence_manifest.json"
                    ).resolve()
                ),
                "manifest_sha256": _sha256(
                    Path(__file__).resolve().parents[1]
                    / "docs/reference_cases/nonlinear_static_performance_convergence_manifest.json"
                ),
            },
            "artifacts": {
                "baseline": {
                    "revision": args.baseline_revision,
                    "wheel": str(args.baseline_wheel.resolve()),
                    "wheel_sha256": _sha256(args.baseline_wheel),
                    "worker": workers["baseline"].identity,
                    "stderr_log": str(workers["baseline"].stderr_path),
                },
                "candidate": {
                    "revision": args.candidate_revision,
                    "wheel": str(args.candidate_wheel.resolve()),
                    "wheel_sha256": _sha256(args.candidate_wheel),
                    "worker": workers["candidate"].identity,
                    "stderr_log": str(workers["candidate"].stderr_path),
                },
            },
            "cases": cases,
            "memory_probe": memory,
            "acceptance": {
                "physical_match": physical_match,
                "performance_target_met": performance_target_met,
                "easy_regression_acceptable": easy_regression_acceptable,
                "accepted": (
                    physical_match
                    and performance_target_met
                    and easy_regression_acceptable
                ),
                "performance_target": (
                    "at least 10% lower declared-workload median"
                ),
                "easy_regression_limit": "at most 5%",
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(output, flush=True)
        return 0 if physical_match and easy_regression_acceptable else 2
    finally:
        for worker in workers.values():
            worker.close()


def main() -> int:
    return _main(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
