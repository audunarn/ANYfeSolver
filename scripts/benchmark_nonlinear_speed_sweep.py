"""Paired whole-route configuration-guard benchmark; no timing acceptance gate.

Run serially with a parent process timeout and explicit thread environment.
The same-revision oracle restores the former guard after every conversion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_nonlinear_static_improvements import _physical_digest
import anyfesolver
import numpy as np
import scipy
from threadpoolctl import threadpool_info
from anyfesolver import nonlinear_static as ns
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.mesh_gen import generate_simple_panel_mesh
from anyfesolver.jit_compiler import JIT_ENABLED


def runtime():
    numba_version = None
    numba_threads = None
    if JIT_ENABLED:
        import numba
        numba_version = numba.__version__
        numba_threads = numba.get_num_threads()
    return dict(threadpools=threadpool_info(), numba_threads=numba_threads,
                dependencies=dict(numpy=np.__version__, scipy=scipy.__version__,
                                  numba=numba_version),
                anyfesolver_module=str(Path(anyfesolver.__file__).resolve()))


def case(divisions, steps):
    model = generate_simple_panel_mesh(1., 1., .01,
                                      num_divisions_x=divisions,
                                      num_divisions_y=divisions)
    model.clear_boundary_conditions()
    edge = []
    for key, node in model.mesh.nodes.items():
        x, y, _z = node.coords()
        if min(x, y, 1. - x, 1. - y) <= 1e-12:
            edge.append(int(key))
    model.add_boundary_condition(BoundaryCondition(
        "edge-translations", edge, {"ux": 0., "uy": 0., "uz": 0.}))
    load = LoadCase("pressure")
    for element_id in model.mesh.elements:
        load.add_pressure_load(int(element_id), 2e4)
    return model, load, dict(num_steps=steps, max_iterations=20,
                            tolerance=1e-9, convergence_settings="legacy")


def run(divisions, steps, variant):
    original = ns._inert_nonlinear_config_scalar
    if variant == "every_conversion_guard":
        ns._inert_nonlinear_config_scalar = lambda value: False
    start = time.perf_counter()
    try:
        model, load, options = case(divisions, steps)
        result = ns.solve_static_nonlinear(model, load, **options)
    finally:
        ns._inert_nonlinear_config_scalar = original
    elapsed = time.perf_counter() - start
    performance = result.info["nonlinear_performance"]
    return dict(variant=variant, wall_seconds=elapsed, status=result.status,
                load_factor=float(result.load_factor), steps=len(result.steps),
                iterations=int(result.info["total_newton_iterations"]),
                step_iterations=[int(step.iterations) for step in result.steps],
                physical_sha256=_physical_digest(result),
                assembly=performance["assembly"], solver=performance["solver"],
                solver_thread_policy=result.info["thread_policy"],
                runtime_after_solve=runtime())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = dict(python=platform.python_version(), jit_enabled=bool(JIT_ENABLED),
                  runtime_at_start=runtime(),
                  environment={key: os.environ.get(key) for key in (
                      "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "NUMBA_NUM_THREADS", "FE_SOLVER_DISABLE_NONLINEAR_ACCELERATION")},
                  source_sha256=hashlib.sha256(
                      (ROOT / "src/anyfesolver/nonlinear_static.py").read_bytes()).hexdigest(),
                  command=sys.argv, cases=[])
    variants = ("every_conversion_guard", "inert_builtin_guard")
    for divisions, steps in ((2, 5), (2, 40), (8, 5)):
        samples = []
        warmups = [run(divisions, steps, variant) for variant in variants]
        for repeat in range(args.repeats):
            order = variants if repeat % 2 == 0 else variants[::-1]
            samples.extend(run(divisions, steps, variant) for variant in order)
        all_results = warmups + samples
        parity = (all(item["status"] == "completed" for item in all_results)
                  and len({item["physical_sha256"] for item in all_results}) == 1
                  and len({item["iterations"] for item in all_results}) == 1
                  and len({tuple(item["step_iterations"])
                           for item in all_results}) == 1)
        medians = {variant: statistics.median(
            item["wall_seconds"] for item in samples if item["variant"] == variant)
                   for variant in variants}
        output["cases"].append(dict(divisions=divisions, steps=steps,
                                    physical_parity=parity, warmups=warmups,
                                    samples=samples, median_seconds=medians,
                                    improvement_percent=100. * (
                                        1. - medians[variants[1]] / medians[variants[0]])))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        if not parity:
            raise RuntimeError("paired complete-route physical parity failed")
    print(json.dumps([dict(divisions=item["divisions"], steps=item["steps"],
                           physical_parity=item["physical_parity"],
                           median_seconds=item["median_seconds"],
                           improvement_percent=item["improvement_percent"])
                      for item in output["cases"]], indent=2))


if __name__ == "__main__":
    main()
