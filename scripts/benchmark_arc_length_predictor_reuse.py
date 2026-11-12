"""Matched whole-route development timing; no timing acceptance threshold.

Parent must enforce a process timeout and shared execution budget. Each case
has paired warmups and at least three alternating candidate/oracle pairs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anyfesolver
import numpy as np
import pytest
import scipy
from threadpoolctl import threadpool_info
from anyfesolver.jit_compiler import JIT_ENABLED


def runtime():
    numba_version = numba_threads = None
    if JIT_ENABLED:
        import numba
        numba_version, numba_threads = numba.__version__, numba.get_num_threads()
    return dict(threadpools=threadpool_info(), numba_threads=numba_threads,
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, numba=numba_version),
                anyfesolver_module=str(Path(anyfesolver.__file__).resolve()))


def digest(result):
    h = hashlib.sha256()
    def feed(value):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                h.update(str(key).encode()); feed(value[key])
        elif isinstance(value, np.ndarray):
            h.update(str(value.shape).encode()); h.update(str(value.dtype).encode())
            h.update(value.tobytes())
        elif isinstance(value, (list, tuple)):
            for item in value: feed(item)
        else:
            h.update(repr(value).encode())
    feed(dict(displacements=result.displacements, states=result.element_states,
              steps=[step.to_dict() for step in result.steps], load_factor=result.load_factor,
              status=result.status, failure_reason=result.info["failure_reason"],
              adaptation=result.info["adaptation_history"]))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three alternating pairs are required")
    fixture = ROOT / "tests/test_arc_length_predictor_reuse.py"
    helpers = runpy.run_path(str(fixture))
    output = dict(command=sys.argv, python=platform.python_version(),
        runtime_at_start=runtime(), jit_enabled=bool(JIT_ENABLED),
        environment={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
            "FE_SOLVER_DISABLE_NONLINEAR_ACCELERATION")},
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "src/anyfesolver/arc_length.py", fixture, Path(__file__))}, cases=[])
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        for retry in (False, True):
            case = dict(forced_retry=retry, warmups=[], samples=[], physical_parity=False)
            output["cases"].append(case)
            reference = None
            for repeat in range(-1, args.repeats):
                order = (False, True) if repeat % 2 == 0 else (True, False)
                for cached in order:
                    start = time.perf_counter()
                    result, rejected = helpers["run_case"](patch, cached=cached, reject=retry)
                    elapsed = time.perf_counter() - start
                    row = dict(cached=cached, wall_seconds=elapsed, status=result.status,
                        retries=result.info["total_retries"], predictor=result.info["predictor_reuse"],
                        forced_failures=len(rejected), physical_sha256=digest(result),
                        solver_thread_policy=result.info.get("thread_policy"), runtime_after_solve=runtime())
                    case["warmups" if repeat == -1 else "samples"].append(row)
                    save()
                    if retry:
                        assert rejected, "fixture did not exercise a retry"
                    assert len(result.steps) == 2, "fixture did not complete its two steps"
                    if reference is None:
                        reference = result
                    else:
                        helpers["assert_same"](reference, result)
            case["physical_parity"] = True
            case["median_seconds"] = {
                "cached" if cached else "oracle": statistics.median(
                    row["wall_seconds"] for row in case["samples"] if row["cached"] == cached)
                for cached in (False, True)}
            save()
    print(json.dumps([dict(forced_retry=case["forced_retry"], physical_parity=case["physical_parity"],
                           median_seconds=case["median_seconds"]) for case in output["cases"]], indent=2))


if __name__ == "__main__":
    main()
