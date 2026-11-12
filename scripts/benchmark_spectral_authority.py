"""Bounded whole-route benchmark for spectral authority optimizations.

Run this script with ANYfeSolver and ANYstructure selected through ``PYTHONPATH``.
It exercises the real ANYstructure runtime adapter, warms one retained analysis
context, and records only subsequent complete static-plus-buckling runs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from anyfesolver.runtime import RuntimeAnalysisContext
from anystruct import fem_integration


def _rss_sampler(stop: threading.Event, samples: list[int]) -> None:
    process = None
    try:
        import psutil

        process = psutil.Process()
    except ImportError:
        pass

    def current_rss() -> int | None:
        if process is not None:
            try:
                return int(process.memory_info().rss)
            except OSError:
                pass
        if os.name != "nt":
            return None

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_memory_info = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        ok = get_memory_info(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.WorkingSetSize) if ok else None

    while not stop.wait(0.01):
        value = current_rss()
        if value is not None:
            samples.append(value)
    value = current_rss()
    if value is not None:
        samples.append(value)


def _run_once(
    snapshot: Any,
    options: Any,
    context: RuntimeAnalysisContext,
) -> dict[str, Any]:
    rss_samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_rss_sampler,
        args=(stop, rss_samples),
        daemon=True,
    )
    sampler.start()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    result = fem_integration.run_runtime_fem(
        snapshot,
        options,
        analysis_context=context,
    )
    wall_seconds = float(time.perf_counter() - wall_started)
    cpu_seconds = float(time.process_time() - cpu_started)
    stop.set()
    sampler.join(timeout=1.0)

    prestress = result.summary.get("prestress_summary", {})
    performance = prestress.get("buckling_performance", {})
    return {
        "status": str(result.status),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": max(rss_samples) if rss_samples else None,
        "buckling_factors": [float(value) for value in result.buckling_factors],
        "phase_timings_seconds": performance.get("phase_timings_seconds", {}),
        "matrix_diagnostics": performance.get("matrix_diagnostics", {}),
        "solver": performance.get("solver"),
        "runtime_context": prestress.get("runtime_analysis_context", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--modes", type=int, default=5)
    parser.add_argument("--fidelity", default="coarse")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    app = fem_integration.example_runtime_app()
    snapshot = fem_integration.active_line_snapshot(app)
    options = fem_integration.RuntimeFEMOptions(
        mesh_fidelity=str(args.fidelity),
        pressure_pa=snapshot.pressure_pa,
        load_scale=1.0,
        include_stiffeners=True,
        include_girders=True,
        include_end_lids=True,
        num_buckling_modes=int(args.modes),
        mesh_size_m=0.0,
        top_bottom_moment_nm=app._fem_default_top_bottom_moment_nm,
    )
    with RuntimeAnalysisContext() as context:
        warmup = _run_once(snapshot, options, context)
        measurements = [
            _run_once(snapshot, options, context)
            for _ in range(args.repetitions)
        ]

    expected = np.asarray(measurements[0]["buckling_factors"], dtype=float)
    if warmup["status"] != "ok" or any(
        record["status"] != "ok"
        or np.asarray(record["buckling_factors"], dtype=float).shape
        != expected.shape
        or not np.allclose(
            np.asarray(record["buckling_factors"], dtype=float),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        for record in measurements
    ):
        raise RuntimeError("benchmark route was not deterministic and successful")

    payload = {
        "schema": "ANYSOLVER_SPECTRAL_AUTHORITY_BENCHMARK_V1",
        "fixture": {
            "name": "anystructure_example_girder_panel",
            "fidelity": str(args.fidelity),
            "modes": int(args.modes),
            "retained_context": True,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "warmup": warmup,
        "measurements": measurements,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
