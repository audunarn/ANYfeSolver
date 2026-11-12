"""Measure bounded ANYfeSolver thread scaling through public solver APIs.

The measurements in this script are evidence, not test thresholds.  Each
workload uses :class:`anyfesolver.recovery.ResourceConfig` so that assembly,
native solver, and recovery thread controls are exercised exactly as an
application exercises them.  A one-thread result is retained as a numerical
reference, and every invocation audits restoration of Numba/native limits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from anyfesolver import (
    AnalysisSession,
    LoadCase,
    RecoveryConfig,
    ResourceConfig,
    recover_element_stresses_with_report,
    solve_linear_many,
    solve_static_nonlinear,
)
from anyfesolver.jit_compiler import jit_diagnostics
from anyfesolver.mesh_gen import generate_beam_mesh, generate_simple_panel_mesh
from anyfesolver.threading_policy import current_solver_threads


SCHEMA_NAME = "anyfesolver.sol_ultra.thread_scaling"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT = ROOT / "reports" / "performance" / "sol_ultra_thread_scaling.json"
REQUIRED_ASSEMBLY_THREADS = (1, 2, 4, 8, 16)
DEFAULT_SOLVER_THREADS = (1, 2, 4, 8)
DEFAULT_RECOVERY_THREADS = (1, 2, 4, 8)


@dataclass(frozen=True)
class CampaignConfig:
    """Explicit, serializable dimensions and thread sweeps."""

    assembly_threads: Tuple[int, ...] = REQUIRED_ASSEMBLY_THREADS
    solver_threads: Tuple[int, ...] = DEFAULT_SOLVER_THREADS
    recovery_threads: Tuple[int, ...] = DEFAULT_RECOVERY_THREADS
    repeats: int = 3
    assembly_nx: int = 20
    assembly_ny: int = 10
    linear_nx: int = 60
    linear_ny: int = 30
    recovery_nx: int = 20
    recovery_ny: int = 10
    nonlinear_steps: int = 2

    def __post_init__(self) -> None:
        for name in ("assembly_threads", "solver_threads", "recovery_threads"):
            values = tuple(int(value) for value in getattr(self, name))
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive thread counts")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicate thread counts")
        for name in (
            "repeats",
            "assembly_nx",
            "assembly_ny",
            "linear_nx",
            "linear_ny",
            "recovery_nx",
            "recovery_ny",
            "nonlinear_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for name in ("assembly_threads", "solver_threads", "recovery_threads"):
            result[name] = list(result[name])
        return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str) -> Optional[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={ROOT}",
        "-C",
        str(ROOT),
        *args,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _cpu_and_memory() -> Dict[str, Optional[int]]:
    values: Dict[str, Optional[int]] = {
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "process_cpu_affinity_count": None,
        "physical_memory_bytes": None,
        "available_memory_bytes": None,
    }
    try:
        import psutil

        process = psutil.Process()
        values.update(
            {
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "process_cpu_affinity_count": (
                    len(process.cpu_affinity())
                    if hasattr(process, "cpu_affinity")
                    else None
                ),
                "physical_memory_bytes": int(psutil.virtual_memory().total),
                "available_memory_bytes": int(psutil.virtual_memory().available),
            }
        )
    except Exception:
        pass
    return values


def _native_pools() -> list[Dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
    except Exception:
        return []
    return [
        {
            "user_api": pool.get("user_api"),
            "internal_api": pool.get("internal_api"),
            "prefix": pool.get("prefix"),
            "filepath": pool.get("filepath"),
            "version": pool.get("version"),
            "num_threads": pool.get("num_threads"),
        }
        for pool in pools
    ]


def collect_environment() -> Dict[str, Any]:
    """Collect enough provenance to reproduce or reject a comparison."""

    return {
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            **_cpu_and_memory(),
        },
        "revision": {
            "head_sha": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "origin_main_sha": _git("rev-parse", "origin/main"),
            "source_root": str(ROOT),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "ANYfeSolver",
                "numpy",
                "scipy",
                "numba",
                "llvmlite",
                "pypardiso",
                "threadpoolctl",
                "psutil",
            )
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYPARDISO_MKL_RT",
            )
        },
        "native_threadpools": _native_pools(),
        "jit": _jsonable(jit_diagnostics()),
    }


def _process_memory() -> Dict[str, Optional[int]]:
    result = {"rss_bytes": None, "process_peak_rss_bytes": None}
    try:
        import psutil

        info = psutil.Process().memory_info()
        result["rss_bytes"] = int(info.rss)
        peak = getattr(info, "peak_wset", None)
        result["process_peak_rss_bytes"] = None if peak is None else int(peak)
    except Exception:
        pass
    return result


def _thread_state() -> Dict[str, Any]:
    jit = jit_diagnostics()
    return {
        "solver_context_threads": current_solver_threads(),
        "numba_threads": jit.get("num_threads"),
        "native_pools": _native_pools(),
    }


def _pool_key(pool: Mapping[str, Any]) -> str:
    return "|".join(
        str(pool.get(name) or "")
        for name in ("user_api", "internal_api", "prefix", "filepath")
    )


def _restoration_report(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Dict[str, Any]:
    before_pools = {_pool_key(pool): pool.get("num_threads") for pool in before["native_pools"]}
    after_pools = {_pool_key(pool): pool.get("num_threads") for pool in after["native_pools"]}
    changed = {
        key: {"before": before_pools[key], "after": after_pools[key]}
        for key in before_pools.keys() & after_pools.keys()
        if before_pools[key] != after_pools[key]
    }
    missing = sorted(before_pools.keys() - after_pools.keys())
    newly_loaded = sorted(after_pools.keys() - before_pools.keys())
    solver_restored = (
        before.get("solver_context_threads") == after.get("solver_context_threads")
    )
    numba_restored = before.get("numba_threads") == after.get("numba_threads")
    restored = solver_restored and numba_restored and not changed and not missing
    return {
        "restored": bool(restored),
        "solver_context_restored": bool(solver_restored),
        "numba_threads_restored": bool(numba_restored),
        "native_pool_threads_restored": not changed and not missing,
        "changed_native_pools": changed,
        "missing_native_pools": missing,
        "newly_loaded_native_pools": newly_loaded,
        "note": (
            "Pools loaded for the first time during a cold invocation are reported "
            "but do not constitute a restoration failure."
        ),
    }


def _vector_signature(vector: np.ndarray) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(np.asarray(vector, dtype=np.float64).reshape(-1))
    return {
        "size": int(contiguous.size),
        "l2_norm": float(np.linalg.norm(contiguous)),
        "maximum_absolute": float(np.max(np.abs(contiguous))) if contiguous.size else 0.0,
        "sha256_float64_bytes": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _compact_backend_diagnostics(value: Any) -> Any:
    """Keep backend evidence while bounding content-derived cache signatures."""

    if not isinstance(value, Mapping):
        return value
    result = dict(value)
    signature = result.pop("signature", None)
    if signature is not None:
        encoded = str(signature).encode("utf-8")
        result["signature_sha256"] = hashlib.sha256(encoded).hexdigest()
        result["signature_utf8_bytes"] = len(encoded)
    return result


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> Optional[float]:
    candidate = np.asarray(candidate, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if candidate.shape != reference.shape:
        return None
    denominator = max(float(np.linalg.norm(reference)), 1.0)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _invoke(function: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    before_threads = _thread_state()
    before_memory = _process_memory()
    tracemalloc.start()
    start = time.perf_counter()
    comparison_vector: Optional[np.ndarray] = None
    try:
        raw_payload = dict(function())
        comparison = raw_payload.pop("_comparison_vector", None)
        if comparison is not None:
            comparison_vector = np.asarray(comparison, dtype=float).reshape(-1).copy()
            raw_payload["output_signature"] = _vector_signature(comparison_vector)
        status = str(raw_payload.get("status", "completed"))
        payload = _jsonable(raw_payload)
    except Exception as exc:
        status = "error"
        payload = {
            "status": "error",
            "exception": {"type": type(exc).__name__, "message": str(exc)},
        }
    wall_seconds = time.perf_counter() - start
    _current_bytes, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_memory = _process_memory()
    after_threads = _thread_state()
    return {
        "status": status,
        "wall_seconds": float(wall_seconds),
        "python_peak_bytes": int(python_peak_bytes),
        "process_rss_before_bytes": before_memory["rss_bytes"],
        "process_rss_after_bytes": after_memory["rss_bytes"],
        "process_peak_rss_bytes": after_memory["process_peak_rss_bytes"],
        "thread_state_before": before_threads,
        "thread_state_after": after_threads,
        "thread_restoration": _restoration_report(before_threads, after_threads),
        "payload": payload,
        "_comparison_vector": comparison_vector,
    }


def _numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": [],
            "median": None,
            "minimum": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "samples": samples,
        "median": float(statistics.median(samples)),
        "minimum": float(min(samples)),
        "maximum": float(max(samples)),
        "mean": float(statistics.fmean(samples)),
    }


def _public_measurement(invocation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in invocation.items()
        if not key.startswith("_")
    }


def _warm_summary(invocations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    representative = next(
        (
            invocation
            for invocation in reversed(invocations)
            if invocation.get("status") == "completed"
        ),
        invocations[-1] if invocations else None,
    )
    numeric_payload_keys = sorted(
        {
            str(key)
            for invocation in invocations
            for key, value in invocation.get("payload", {}).items()
            if str(key).endswith("_seconds") and isinstance(value, (int, float))
        }
    )
    return {
        "repeats": len(invocations),
        "observed_statuses": [str(item.get("status")) for item in invocations],
        "wall_seconds": _numeric_summary([float(item["wall_seconds"]) for item in invocations]),
        "python_peak_bytes": _numeric_summary(
            [float(item["python_peak_bytes"]) for item in invocations]
        ),
        "process_peak_rss_bytes": _numeric_summary(
            [
                float(item["process_peak_rss_bytes"])
                for item in invocations
                if item.get("process_peak_rss_bytes") is not None
            ]
        ),
        "phase_seconds": {
            key: _numeric_summary(
                [
                    float(invocation["payload"][key])
                    for invocation in invocations
                    if isinstance(invocation.get("payload", {}).get(key), (int, float))
                ]
            )
            for key in numeric_payload_keys
        },
        "all_thread_policies_restored": all(
            bool(item["thread_restoration"]["restored"]) for item in invocations
        ),
        "representative_payload": None if representative is None else representative["payload"],
        "representative_thread_restoration": (
            None if representative is None else representative["thread_restoration"]
        ),
    }


WorkloadFactory = Callable[
    [int],
    Tuple[
        Callable[[], Dict[str, Any]],
        Dict[str, int],
        Optional[Callable[[], None]],
    ],
]


def _run_sweep(
    *,
    control: str,
    thread_counts: Sequence[int],
    repeats: int,
    factory: WorkloadFactory,
) -> Dict[str, Any]:
    entries = []
    private_vectors: list[Optional[np.ndarray]] = []
    for thread_count in thread_counts:
        cleanup: Optional[Callable[[], None]] = None
        try:
            invocation, topology, cleanup = factory(int(thread_count))
        except Exception as exc:
            def invocation(exc: Exception = exc) -> Dict[str, Any]:
                raise exc

            topology = {}
        cold = _invoke(invocation)
        warm = [_invoke(invocation) for _ in range(int(repeats))]
        representative = next(
            (
                item
                for item in reversed(warm)
                if item.get("_comparison_vector") is not None
            ),
            cold if cold.get("_comparison_vector") is not None else None,
        )
        private_vectors.append(
            None if representative is None else representative["_comparison_vector"]
        )
        statuses = [cold["status"], *(item["status"] for item in warm)]
        cleanup_report: Dict[str, Any] = {"status": "not_required"}
        if cleanup is not None:
            try:
                cleanup()
                cleanup_report = {"status": "completed"}
            except Exception as exc:
                cleanup_report = {
                    "status": "failed",
                    "exception": {"type": type(exc).__name__, "message": str(exc)},
                }
                statuses.append("cleanup_error")
        entries.append(
            {
                "threads": int(thread_count),
                "status": "completed" if all(status == "completed" for status in statuses) else "failed",
                "topology": topology,
                "measurements": {
                    "cold": _public_measurement(cold),
                    "warm": _warm_summary(warm),
                },
                "cleanup": cleanup_report,
                "numerical_comparison": {},
            }
        )

    reference = next((vector for vector in private_vectors if vector is not None), None)
    reference_threads = next(
        (
            int(entry["threads"])
            for entry, vector in zip(entries, private_vectors)
            if vector is not None
        ),
        None,
    )
    for entry, vector in zip(entries, private_vectors):
        entry["numerical_comparison"] = {
            "reference_threads": reference_threads,
            "relative_l2_error": (
                None if reference is None or vector is None else _relative_error(vector, reference)
            ),
            "comparison_available": reference is not None and vector is not None,
        }
    return {
        "controlled_resource": control,
        "thread_counts": [int(value) for value in thread_counts],
        "reference_policy": "first completed count (configured with one thread by default)",
        "entries": entries,
        "status": "completed" if all(entry["status"] == "completed" for entry in entries) else "failed",
    }


def _topology(model: Any) -> Dict[str, int]:
    return {
        "nodes": int(model.mesh.num_nodes),
        "elements": int(model.mesh.num_elements),
        "dofs": int(model.mesh.dof_manager.total_dofs),
    }


def _pressure_load(model: Any, name: str, pressure: float) -> LoadCase:
    load = LoadCase(name)
    for element_id in sorted(model.mesh.elements):
        load.add_pressure_load(int(element_id), float(pressure))
    return load


def _nonlinear_factory(config: CampaignConfig) -> WorkloadFactory:
    def factory(
        thread_count: int,
    ) -> Tuple[Callable[[], Dict[str, Any]], Dict[str, int], None]:
        model = generate_simple_panel_mesh(
            4.0,
            2.0,
            0.012,
            config.assembly_nx,
            config.assembly_ny,
        )
        load = _pressure_load(model, "thread_scaling_nonlinear", 250.0)
        resources = ResourceConfig(
            solver_threads=1,
            assembly_threads=thread_count,
            recovery_threads=1,
            deterministic=True,
            metadata={"campaign": "sol_ultra_thread_scaling", "phase": "assembly"},
        )

        def invoke() -> Dict[str, Any]:
            active_numba_threads: list[Optional[int]] = []

            def observe_threads(_message: str) -> None:
                if not active_numba_threads:
                    active_numba_threads.append(jit_diagnostics().get("num_threads"))

            result = solve_static_nonlinear(
                model,
                load,
                max_load_factor=1.0,
                num_steps=config.nonlinear_steps,
                max_iterations=12,
                tolerance=1.0e-7,
                convergence_settings="legacy",
                resource_config=resources,
                status_callback=observe_threads,
            )
            return {
                "status": "completed" if result.converged else "failed",
                "resource_config": resources.to_dict(),
                "solver_status": result.status,
                "load_factor": float(result.load_factor),
                "step_count": len(result.steps),
                "total_newton_iterations": int(
                    result.info.get("total_newton_iterations", 0)
                ),
                "solve_seconds": float(result.info.get("solve_time", 0.0)),
                "backend": {
                    "api": "solve_static_nonlinear",
                    "thread_policy": result.info.get("thread_policy"),
                    "active_numba_threads_observed": (
                        active_numba_threads[0] if active_numba_threads else None
                    ),
                    "jit_after_call": jit_diagnostics(),
                },
                "_comparison_vector": result.displacements,
            }

        return invoke, _topology(model), None

    return factory


def _linear_factory(config: CampaignConfig) -> WorkloadFactory:
    def factory(
        thread_count: int,
    ) -> Tuple[Callable[[], Dict[str, Any]], Dict[str, int], Callable[[], None]]:
        model = generate_simple_panel_mesh(
            4.0,
            2.0,
            0.012,
            config.linear_nx,
            config.linear_ny,
        )
        loads = [
            _pressure_load(
                model,
                f"thread_scaling_linear_{index}",
                250.0 * float(index),
            )
            for index in range(1, 5)
        ]
        session = AnalysisSession(model, max_factorizations=1)
        resources = ResourceConfig(
            solver_threads=thread_count,
            assembly_threads=1,
            recovery_threads=1,
            deterministic=True,
            metadata={"campaign": "sol_ultra_thread_scaling", "phase": "solver"},
        )

        def invoke() -> Dict[str, Any]:
            displacement, info = solve_linear_many(
                model,
                loads,
                resource_config=resources,
                session=session,
            )
            return {
                "status": (
                    "completed" if info.get("status") == "converged" else "failed"
                ),
                "resource_config": resources.to_dict(),
                "solver_status": info.get("status"),
                "solve_seconds": float(info.get("solve_time", 0.0)),
                "backend": {
                    "api": "solve_linear_many",
                    "thread_policy": info.get("thread_policy"),
                    "linear_solver": _compact_backend_diagnostics(info.get("backend")),
                    "factorization_cache": info.get("factorization_cache"),
                    "analysis_session": info.get("analysis_session"),
                },
                "_comparison_vector": displacement,
            }

        return invoke, _topology(model), session.close

    return factory


def _flatten_recovery(stresses: Mapping[int, Mapping[str, Any]]) -> np.ndarray:
    blocks = [
        np.asarray(stresses[element_id].get("von_mises", ()), dtype=float).reshape(-1)
        for element_id in sorted(stresses)
    ]
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=float)


def _recovery_factory(config: CampaignConfig) -> WorkloadFactory:
    def factory(
        thread_count: int,
    ) -> Tuple[Callable[[], Dict[str, Any]], Dict[str, int], None]:
        model = generate_simple_panel_mesh(
            4.0,
            2.0,
            0.012,
            config.recovery_nx,
            config.recovery_ny,
        )
        displacement = np.random.default_rng(20260811).normal(
            scale=2.0e-6,
            size=model.mesh.dof_manager.total_dofs,
        )
        recovery = RecoveryConfig(components=["von_mises"])
        resources = ResourceConfig(
            solver_threads=1,
            assembly_threads=1,
            recovery_threads=thread_count,
            deterministic=True,
            metadata={"campaign": "sol_ultra_thread_scaling", "phase": "recovery"},
        )

        def invoke() -> Dict[str, Any]:
            stresses, report = recover_element_stresses_with_report(
                model,
                displacement,
                recovery,
                resource_config=resources,
            )
            values = _flatten_recovery(stresses)
            finite = bool(np.all(np.isfinite(values)))
            return {
                "status": (
                    "completed"
                    if finite and len(stresses) == model.mesh.num_elements
                    else "failed"
                ),
                "resource_config": resources.to_dict(),
                "recovered_element_count": len(stresses),
                "all_values_finite": finite,
                "recovery_seconds": float(report.elapsed_seconds),
                "backend": {
                    "api": "recover_element_stresses_with_report",
                    "execution_report": report.to_dict(),
                },
                "_comparison_vector": values,
            }

        return invoke, _topology(model), None

    return factory


def _exception_restoration_audit() -> Dict[str, Any]:
    """Raise inside a public nonlinear call after both scopes are entered."""

    model = generate_beam_mesh(1.0, num_divisions=1)
    load = LoadCase("thread_restoration_exception")
    load.add_nodal_load(2, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    resources = ResourceConfig(
        solver_threads=2,
        assembly_threads=2,
        recovery_threads=1,
        deterministic=True,
        metadata={"campaign": "sol_ultra_thread_scaling", "phase": "exception_audit"},
    )
    callback_entered = False
    before = _thread_state()

    def intentional_failure(_message: str) -> None:
        nonlocal callback_entered
        callback_entered = True
        raise RuntimeError("intentional thread-restoration audit")

    observed: Optional[Exception] = None
    try:
        solve_static_nonlinear(
            model,
            load,
            num_steps=1,
            max_iterations=2,
            convergence_settings="legacy",
            resource_config=resources,
            status_callback=intentional_failure,
        )
    except Exception as exc:  # preserve evidence for ordinary backend failures
        observed = exc
    after = _thread_state()
    restoration = _restoration_report(before, after)
    expected = (
        callback_entered
        and isinstance(observed, RuntimeError)
        and "intentional thread-restoration audit" in str(observed)
    )
    return {
        "status": "completed" if expected and restoration["restored"] else "failed",
        "resource_config": resources.to_dict(),
        "callback_entered_inside_nonlinear_solve": bool(callback_entered),
        "expected_exception_observed": bool(expected),
        "exception": (
            None
            if observed is None
            else {"type": type(observed).__name__, "message": str(observed)}
        ),
        "thread_state_before": before,
        "thread_state_after": after,
        "thread_restoration": restoration,
    }


def run_campaign(
    config: CampaignConfig,
    *,
    run_exception_audit: bool = True,
) -> Dict[str, Any]:
    """Run all independent sweeps and return one JSON-ready report."""

    start = time.perf_counter()
    workloads = {
        "nonlinear_assembly": _run_sweep(
            control="ResourceConfig.assembly_threads",
            thread_counts=config.assembly_threads,
            repeats=config.repeats,
            factory=_nonlinear_factory(config),
        ),
        "linear_solver": _run_sweep(
            control="ResourceConfig.solver_threads",
            thread_counts=config.solver_threads,
            repeats=config.repeats,
            factory=_linear_factory(config),
        ),
        "stress_recovery": _run_sweep(
            control="ResourceConfig.recovery_threads",
            thread_counts=config.recovery_threads,
            repeats=config.repeats,
            factory=_recovery_factory(config),
        ),
    }
    audit = _exception_restoration_audit() if run_exception_audit else {
        "status": "not_run",
        "reason": "disabled by caller",
    }
    failed = [name for name, result in workloads.items() if result["status"] != "completed"]
    if audit["status"] == "failed":
        failed.append("exception_thread_restoration")
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "report_kind": "thread_scaling_evidence",
        "environment": collect_environment(),
        "campaign": {
            "config": config.to_dict(),
            "total_wall_seconds": float(time.perf_counter() - start),
            "measurement_policy": {
                "cold": (
                    "First invocation for a fresh model and thread-count configuration; "
                    "it is not a fresh-process startup measurement."
                ),
                "warm": "Subsequent invocations on the same model in the same interpreter.",
                "warm_statistic": "median",
                "memory": (
                    "Python allocation peak is reset per invocation with tracemalloc; "
                    "process peak RSS is process-lifetime peak where the OS exposes it. "
                    "Models are prepared before timed invocations, so invocation peaks "
                    "exclude initial model construction."
                ),
                "promotion": (
                    "Timing values are evidence only. No wall-clock threshold changes "
                    "test status and no default thread count is selected automatically."
                ),
                "isolation": (
                    "Only one resource dimension varies per workload; other explicit "
                    "ResourceConfig thread limits remain one."
                ),
            },
        },
        "workloads": workloads,
        "thread_policy_restoration": {
            "normal_calls": {
                name: all(
                    entry["measurements"]["cold"]["thread_restoration"]["restored"]
                    and entry["measurements"]["warm"]["all_thread_policies_restored"]
                    for entry in result["entries"]
                )
                for name, result in workloads.items()
            },
            "exception_call": audit,
        },
        "summary": {
            "status": "completed" if not failed else "failed",
            "failed_sections": failed,
            "timings_are_acceptance_thresholds": False,
            "default_thread_policy_changed": False,
        },
        "known_limitations": [
            "Compare timings only on the same machine, revision, native libraries, and power policy.",
            "Logical-core count alone is not a basis for choosing a production default.",
            "Configuration-cold measurements share already-imported native libraries and JIT code compiled by earlier sweeps.",
            "Python threads can lose to serial recovery for small or inexpensive element groups; coarse alternatives are reported without promotion claims.",
        ],
    }


def _parse_threads(value: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thread counts must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("thread counts must be unique positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure assembly, sparse-solver, and recovery thread scaling"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--assembly-threads", type=_parse_threads, default=None)
    parser.add_argument("--solver-threads", type=_parse_threads, default=None)
    parser.add_argument("--recovery-threads", type=_parse_threads, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny models and 1/2-thread sweeps for invocation checks",
    )
    parser.add_argument(
        "--skip-exception-audit",
        action="store_true",
        help="Skip the deliberate in-scope exception restoration audit",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> CampaignConfig:
    quick = bool(args.quick)
    return CampaignConfig(
        assembly_threads=(
            args.assembly_threads
            if args.assembly_threads is not None
            else ((1, 2) if quick else REQUIRED_ASSEMBLY_THREADS)
        ),
        solver_threads=(
            args.solver_threads
            if args.solver_threads is not None
            else ((1, 2) if quick else DEFAULT_SOLVER_THREADS)
        ),
        recovery_threads=(
            args.recovery_threads
            if args.recovery_threads is not None
            else ((1, 2) if quick else DEFAULT_RECOVERY_THREADS)
        ),
        repeats=(args.repeats if args.repeats is not None else (1 if quick else 3)),
        assembly_nx=4 if quick else 20,
        assembly_ny=2 if quick else 10,
        linear_nx=6 if quick else 60,
        linear_ny=3 if quick else 30,
        recovery_nx=4 if quick else 20,
        recovery_ny=2 if quick else 10,
        nonlinear_steps=1 if quick else 2,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = _config_from_args(args)
    report = run_campaign(config, run_exception_audit=not args.skip_exception_audit)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"status: {report['summary']['status']}")
    return 0 if report["summary"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
