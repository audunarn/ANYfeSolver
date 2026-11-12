"""Bounded, reproducible medium/fine spectral profiling gate.

The coordinator never solves in-process: every mechanics attempt is an owned
child process so that a timeout, memory limit, or Ctrl-C can cleanly kill its
whole process tree.  ``--_child`` is intentionally private and is also useful
for the small process-control tests in ``tests/``.
"""

from __future__ import annotations

import argparse
import cProfile
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pstats
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


def _disable_windows_error_dialogs() -> None:
    """Keep a native crash from leaving an unattended WER dialog behind."""
    if os.name != "nt":
        return
    import ctypes
    # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX.
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


# This must precede imports which can load native extensions in the worker.
if "--_child" in sys.argv:
    _disable_windows_error_dialogs()

import psutil  # noqa: E402  (child crash-dialog setup must happen first)


PROCESS_TIMEOUT_SECONDS = 600
WAVE_TIMEOUT_SECONDS = 1800
TREE_RSS_LIMIT_BYTES = 24 * 1024**3
POLL_SECONDS = 0.10


def _json_safe(value: Any) -> Any:
    """Make diagnostics from numpy/dataclasses suitable for durable JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _checkpoint(stage: str) -> None:
    print(json.dumps({"checkpoint": stage, "unix": time.time()}), flush=True)


def _tracked_tree(root: Path) -> dict[str, Any]:
    root = root.resolve()
    def git(*args: str) -> str:
        completed = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args], text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"cannot bind git tree {root}: {completed.stderr.strip()}")
        return completed.stdout.strip()
    commit = git("rev-parse", "HEAD")
    tracked_dirty = bool(git("diff", "--name-only") or git("diff", "--cached", "--name-only"))
    if tracked_dirty:
        raise RuntimeError(f"refusing non-clean tracked tree: {root}")
    return {"root": str(root), "commit": commit, "tree": git("rev-parse", "HEAD^{tree}"), "tracked_clean": True, "porcelain": git("status", "--porcelain")}


def _module_provenance(module: Any, distribution: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"module": module.__name__, "origin": str(Path(module.__file__).resolve()), "version": version}


def _complete_example_snapshot_inputs(app: Any) -> None:
    # The standalone example lacks these optional GUI fields. Its __getattr__
    # loads the entire Application for missing names. Explicit None has exactly
    # the reader's absent-value meaning and keeps the example's own loads.
    for name in ("_new_shell_psd", "_new_shell_Nsd", "_new_shell_Msd"):
        if name not in vars(app):
            setattr(app, name, None)


def _runtime_load_options(app: Any, snapshot: Any) -> dict[str, float]:
    """Mirror RuntimeFEMWindow's effective example loads without creating Tk."""
    moment = float(snapshot.top_bottom_moment_nm or 0.0)
    if moment == 0.0:
        moment = float(vars(app).get("_fem_default_top_bottom_moment_nm", 0.0))
    return {
        "pressure_pa": float(snapshot.pressure_pa),
        "top_bottom_moment_nm": moment,
        "torsional_moment_nm": float(snapshot.torsional_moment_nm or 0.0),
        "shear_force_n": float(snapshot.shear_force_n or 0.0),
        "axial_force_n": float(snapshot.axial_force_n or 0.0),
    }


def _fixture_inputs(adapter_root: Path, solver_root: Path, fixture: str, fidelity: str, modes: int):
    """Build only the real adapter fixture; no synthetic mechanics fixture."""
    sys.path.insert(0, str(solver_root / "src"))
    sys.path.insert(0, str(adapter_root))
    from anystruct import fem_integration  # type: ignore

    app = fem_integration.example_runtime_app(fixture)
    _complete_example_snapshot_inputs(app)
    snapshot = fem_integration.active_line_snapshot(app)
    options = fem_integration.RuntimeFEMOptions(
        mesh_fidelity=fidelity,
        **_runtime_load_options(app, snapshot),
        include_stiffeners=True,
        include_girders=True,
        include_end_lids=True,
        analysis_type="linear static + eigenvalue",
        runtime_solver="stepwise",
        num_buckling_modes=modes,
    )
    geometry = fem_integration.runtime_geometry_summary(snapshot, options)
    config = fem_integration._solver_config_from_options(options)
    _checkpoint("geometry.begin")
    generated = fem_integration.build_runtime_generated_geometry(
        fem_integration.runtime_geometry_projection(geometry, config), config
    )
    _checkpoint("geometry.end")
    digest = hashlib.sha256(json.dumps(_json_safe({"geometry": geometry, "options": vars(options)}), sort_keys=True).encode("utf-8")).hexdigest()
    return fem_integration, snapshot, options, config, generated, geometry, digest


def _geometry_counts(generated: Any, model: Any | None = None) -> dict[str, int]:
    data = generated if isinstance(generated, dict) else {}
    result = {
        "nodes": len(data.get("nodes", ()) or ()),
        "shell_elements": len(data.get("shells", ()) or ()),
        "beam_elements": len(data.get("beams", ()) or ()),
    }
    if model is not None:
        mesh = getattr(model, "mesh", None)
        result["model_nodes"] = len(getattr(mesh, "nodes", ()) or ())
        result["model_elements"] = len(getattr(mesh, "elements", ()) or ())
        manager = getattr(mesh, "dof_manager", None)
        result["dofs"] = int(getattr(manager, "total_dofs", 0) or 0)
    return result


def _original_pencil_residuals(diagnostics: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                current = f"{path}.{key}" if path else str(key)
                if "residual" in str(key).lower() and ("pencil" in str(key).lower() or "original" in str(key).lower()):
                    found[current] = _json_safe(item)
                walk(item, current)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
    walk(diagnostics)
    return found


def _result_values(route: str, result: Any, modes: int) -> tuple[list[float], dict[str, Any], Any]:
    status = getattr(result, "solver_status", None) if route == "modal" else getattr(result, "status", None)
    if status != "ok":
        raise RuntimeError(f"{route} failed with status {status!r}")
    raw = getattr(result, "frequencies_hz", ()) if route == "modal" else getattr(result, "buckling_factors", ())
    values = [float(value) for value in raw]
    if len(values) != modes:
        raise RuntimeError(f"{route} requested {modes} modes but returned {len(values)}")
    diagnostics = _json_safe(getattr(result, "diagnostics", {}))
    if route == "buckling":
        diagnostics = {"diagnostics": diagnostics, "summary": _json_safe(getattr(result, "summary", {}))}
    if route == "modal":
        vectors = [getattr(mode, "mode_shape", None) for mode in result.modes]
        residuals = [float(getattr(mode, "residual_norm", float("nan"))) for mode in result.modes]
    else:
        mode_rows = getattr(result, "_benchmark_raw_modes", ())
        vectors = [mode.mode_shape for mode in mode_rows]
        residuals = [float(mode.residual_norm) for mode in mode_rows]
    if len(vectors) != modes or any(vector is None for vector in vectors):
        raise RuntimeError(f"{route} did not expose all {modes} mode vectors for later clustered-subspace verification")
    return values, {"diagnostics": diagnostics, "original_pencil_residuals": _original_pencil_residuals(diagnostics), "mode_residual_norms": residuals}, vectors


def _timed(callable_: Any, route: str, modes: int) -> dict[str, Any]:
    _checkpoint(f"{route}.solve.begin")
    started = time.perf_counter()
    cpu_started = time.process_time()
    result = callable_()
    duration = time.perf_counter() - started
    cpu_duration = time.process_time() - cpu_started
    _checkpoint(f"{route}.solve.end")
    values, details, vectors = _result_values(route, result, modes)
    return {"seconds": duration, "cpu_seconds": cpu_duration, "returned_modes": len(values), "eigenvalues": values, "_mode_vectors": vectors, **details}


def _modal_model(config: Any, generated: Any) -> Any:
    from anyfesolver import anystructure_fem_mode as backend  # type: ignore
    backend_config = backend.AnyStructureFEMConfig(
        pressure_pa=abs(float(config.pressure_pa)), pressure_sign=-1.0, load_scale=1.0,
        num_buckling_modes=int(config.num_buckling_modes), solver_type="direct", stress_percentile=95.0,
        add_inplane_edge_loads=False, auto_idealize_member_plates_as_beams=True,
        exclude_idealized_member_plates=True, require_idealized_member_beams=False,
        elastic_modulus=float(config.elastic_modulus_pa), poisson_ratio=float(config.poisson_ratio),
        yield_stress=float(config.yield_stress_pa),
    )
    return backend.build_fe_model_from_generated_geometry(generated, backend_config)


def _run_actual(args: argparse.Namespace, mode_artifact: Path | None = None, profiler: Any = None) -> dict[str, Any]:
    adapter_root, solver_root = Path(args.adapter_root), Path(args.solver_root)
    prepared_started = time.perf_counter()
    fem, snapshot, options, config, generated, geometry, digest = _fixture_inputs(adapter_root, solver_root, args.fixture, args.fidelity, args.modes)
    import anyfesolver  # type: ignore
    if not Path(anyfesolver.__file__).resolve().is_relative_to(solver_root.resolve() / "src"):
        raise RuntimeError("solver import escaped the bound checkout")
    if not Path(fem.__file__).resolve().is_relative_to(adapter_root.resolve()):
        raise RuntimeError("adapter import escaped the bound checkout")
    records: dict[str, list[dict[str, Any]]] = {"cold_warmup": [], "cold_samples": [], "retained_warmup": [], "retained_samples": []}
    _checkpoint("model.begin")
    model = _modal_model(config, generated) if args.route == "modal" else None
    _checkpoint("model.end")
    preparation_seconds = time.perf_counter() - prepared_started
    if args.route == "modal":
        from anyfesolver import AnalysisSession, solve_free_vibration  # type: ignore
        cold = lambda: solve_free_vibration(model, num_modes=args.modes)
        retained_session = AnalysisSession(model)
        retained = lambda: solve_free_vibration(model, num_modes=args.modes, session=retained_session)
    else:
        from anyfesolver.runtime import RuntimeAnalysisContext  # type: ignore
        original_run = fem.fe_solver.run_production_fem
        captured = []
        def observed_run(*positional, **keywords):
            value = original_run(*positional, **keywords)
            captured[:] = list(value.buckling_modes)
            return value
        def adapter_run(context=None):
            captured.clear()
            value = fem.run_runtime_fem(snapshot, options, precomputed_generated_geometry=generated, precomputed_geometry_is_imperfection=False, analysis_context=context)
            object.__setattr__(value, "_benchmark_raw_modes", tuple(captured))
            return value
        fem.fe_solver.run_production_fem = observed_run
        cold = adapter_run
        retained_context = RuntimeAnalysisContext()
        retained = lambda: adapter_run(retained_context)
    try:
        records["cold_warmup"].append(_timed(cold, args.route, args.modes))
        for _ in range(args.repetitions): records["cold_samples"].append(_timed(cold, args.route, args.modes))
        records["retained_warmup"].append(_timed(retained, args.route, args.modes))
        for index in range(args.repetitions):
            if profiler is not None and index == args.repetitions - 1:
                records["retained_samples"].append(profiler.runcall(_timed, retained, args.route, args.modes))
            else:
                records["retained_samples"].append(_timed(retained, args.route, args.modes))
    finally:
        if args.route == "modal": retained_session.close()
        else:
            fem.fe_solver.run_production_fem = original_run
            retained_context.close()
    # Keep the retained sample's vectors out of timing JSON; these are required
    # for later cluster/subspace checks, not for benchmark comparison display.
    selected = records["retained_samples"][0]
    import numpy as np
    expected = np.asarray(selected["eigenvalues"], dtype=float)
    for category in records.values():
        for item in category:
            observed = np.asarray(item["eigenvalues"], dtype=float)
            if not np.isfinite(observed).all() or not np.allclose(observed, expected, rtol=1e-12, atol=1e-12):
                raise RuntimeError("spectral values changed across repeated analyses")
            if not np.isfinite(item["mode_residual_norms"]).all():
                raise RuntimeError("nonfinite mode residual")
    vectors = selected.pop("_mode_vectors")
    for category in records.values():
        for item in category:
            item.pop("_mode_vectors", None)
    vector_artifact = None
    if mode_artifact is not None:
        import numpy as np
        matrix = np.column_stack([np.asarray(vector, dtype=float).reshape(-1) for vector in vectors])
        np.savez_compressed(mode_artifact, mode_vectors=matrix, eigenvalues=np.asarray(selected["eigenvalues"], dtype=float), residual_norms=np.asarray(selected["mode_residual_norms"], dtype=float))
        vector_artifact = mode_artifact.name
    context_note = []
    if args.fixture == "cylinder" and options.include_end_lids:
        context_note.append("Cylinder fixture intentionally uses include_end_lids=True; retained-context eligibility is deliberately not inferred from this profile.")
    return {
        "schema": "spectral-medium-fine-gate-v1", "fixture": args.fixture, "fidelity": args.fidelity,
        "route": args.route, "requested_modes": args.modes, "repetitions": args.repetitions,
        "fixture_digest_sha256": digest, "geometry": _json_safe(geometry), "counts": _geometry_counts(generated, model),
        "records": records, "context_diagnostics": context_note,
        "preparation": {"seconds": preparation_seconds, "scope": "real adapter fixture + generated geometry" + (" + modal model build" if args.route == "modal" else "")},
        "timed_scope": "solve_free_vibration only; model preparation is reported separately" if args.route == "modal" else "adapter run_runtime_fem with precomputed generated geometry",
        "mode_vectors_artifact": vector_artifact,
        "modules": {"adapter": _module_provenance(fem, "anystructure"), "solver": _module_provenance(anyfesolver, "anyfesolver")},
        "numerical_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "numba", "psutil")},
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }


def _profile_call(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    profiler = cProfile.Profile()
    payload = _run_actual(args, None, profiler)
    raw = output.parent / "profile.pstats"
    profiler.dump_stats(str(raw))
    stats = pstats.Stats(profiler)
    calls = [
        {"function": f"{key[0]}:{key[1]}:{key[2]}", "primitive_calls": value[0], "total_calls": value[1], "total_seconds": value[2], "cumulative_seconds": value[3]}
        for key, value in stats.stats.items()
    ]
    calls.sort(key=lambda item: item["cumulative_seconds"], reverse=True)
    _write_json(output.parent / "profile_calls.json", {"scope": "last retained analysis only; warmup excluded", "raw_profile": raw.name, "calls": calls})
    return payload


def _child(args: argparse.Namespace) -> int:
    _disable_windows_error_dialogs()
    faulthandler.enable()
    output = Path(args.child_output)
    try:
        payload = _profile_call(args, output) if args.profile_child else _run_actual(args, output.parent / "mode_vectors.npz")
        _write_json(output, payload)
        return 0
    except BaseException as error:
        _write_json(output, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        return 1


def _remember_tree(process: subprocess.Popen[Any], known: dict[int, psutil.Process]) -> None:
    """Record descendants while the root still exists for later cleanup."""
    try:
        root = psutil.Process(process.pid)
        known[root.pid] = root
        for item in root.children(recursive=True):
            known[item.pid] = item
    except psutil.NoSuchProcess:
        return
    except psutil.AccessDenied as error:
        raise RuntimeError(f"cannot inspect child process tree: {error}") from error


def _live_processes(known: dict[int, psutil.Process]) -> list[psutil.Process]:
    live: list[psutil.Process] = []
    for item in known.values():
        try:
            if item.is_running():
                live.append(item)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError(f"cannot inspect child process tree: {error}") from error
    return live


def _terminate_tree(process: subprocess.Popen[Any], known: dict[int, psutil.Process] | None = None) -> list[str]:
    """Best-effort cleanup, returning processes that could not be confirmed gone."""
    tracked = known if known is not None else {}
    problems: list[str] = []
    try:
        _remember_tree(process, tracked)
    except RuntimeError as error:
        problems.append(str(error))
    if os.name == "nt" and process.poll() is None:
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
        except (subprocess.SubprocessError, OSError) as error:
            problems.append(f"taskkill failed: {error}")
    processes = list(tracked.values())
    for child in reversed(processes):
        try: child.terminate()
        except psutil.NoSuchProcess: pass
        except psutil.AccessDenied as error: problems.append(f"cannot terminate {child.pid}: {error}")
    try:
        _, alive = psutil.wait_procs(processes, timeout=3)
    except psutil.Error as error:
        problems.append(f"cannot wait for child process tree: {error}")
        alive = processes
    for child in alive:
        try: child.kill()
        except psutil.NoSuchProcess: pass
        except psutil.AccessDenied as error: problems.append(f"cannot kill {child.pid}: {error}")
    try: process.wait(timeout=3)
    except subprocess.TimeoutExpired: pass
    for child in processes:
        try:
            if child.is_running():
                problems.append(f"surviving process {child.pid}")
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            problems.append(f"cannot confirm process {child.pid} exited: {error}")
    return problems


def _tree_rss(process: subprocess.Popen[Any], known: dict[int, psutil.Process]) -> int:
    _remember_tree(process, known)
    total = 0
    for item in _live_processes(known):
        try:
            total += item.memory_info().rss
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError(f"cannot inspect child process tree memory: {error}") from error
    return total


def _fatal_log_marker(log: Any) -> str | None:
    log.flush()
    log.seek(0)
    tail = b""
    while block := log.read(65536):
        text = (tail + block).lower()
        for marker in (b"fatal python error", b"windows fatal exception", b"fatal exception"):
            if marker in text:
                return marker.decode("ascii")
        tail = text[-32:]
    return None


def _run_bounded_child(command: Sequence[str], *, process_timeout: float = PROCESS_TIMEOUT_SECONDS, rss_limit: int = TREE_RSS_LIMIT_BYTES, log_path: Path | None = None) -> dict[str, Any]:
    """Run one owned child, retaining its peak tree RSS and always cleaning it."""
    log = log_path.open("xb+") if log_path else tempfile.TemporaryFile(mode="w+b")
    process = subprocess.Popen(list(command), stdout=log, stderr=log, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    started, peak = time.monotonic(), 0
    reason: str | None = None
    known: dict[int, psutil.Process] = {}
    try:
        while process.poll() is None:
            peak = max(peak, _tree_rss(process, known))
            if peak > rss_limit: reason = f"tree RSS exceeded {rss_limit} bytes"
            elif time.monotonic() - started > process_timeout: reason = f"child exceeded {process_timeout} seconds"
            if reason:
                cleanup = _terminate_tree(process, known)
                if cleanup:
                    reason = f"{reason}; cleanup incomplete: {'; '.join(cleanup)}"
                raise RuntimeError(reason)
            time.sleep(POLL_SECONDS)
        peak = max(peak, _tree_rss(process, known))
        if process.returncode:
            raise RuntimeError(f"child exited {process.returncode}; see {log_path}")
        descendants = [item for item in _live_processes(known) if item.pid != process.pid]
        if descendants:
            cleanup = _terminate_tree(process, known)
            detail = f"; cleanup incomplete: {'; '.join(cleanup)}" if cleanup else ""
            raise RuntimeError(f"child exited but left descendant processes running{detail}")
        if marker := _fatal_log_marker(log):
            raise RuntimeError(f"child emitted {marker!r}; see {log_path}")
        return {"wall_seconds": time.monotonic() - started, "peak_tree_rss_bytes": peak}
    finally:
        cleanup = _terminate_tree(process, known)
        try:
            log.close()
        finally:
            if cleanup:
                raise RuntimeError(f"child process cleanup incomplete: {'; '.join(cleanup)}")


def _coordinator(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory must be new and exclusive: {output}")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint.json"
    manifest = {"status": "running", "started_unix": time.time(), "interpreter": sys.executable, "python": sys.version, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "solver_tree": _tracked_tree(Path(args.solver_root)), "adapter_tree": _tracked_tree(Path(args.adapter_root)), "limits": {"process_seconds": args.timeout_seconds, "wave_seconds": WAVE_TIMEOUT_SECONDS, "tree_rss_bytes": TREE_RSS_LIMIT_BYTES}}
    _write_json(checkpoint, manifest)
    started = time.monotonic()
    child_output = output / "worker.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--_child", "--solver-root", args.solver_root, "--adapter-root", args.adapter_root, "--child-output", str(child_output), "--fixture", args.fixture, "--fidelity", args.fidelity, "--route", args.route, "--modes", str(args.modes), "--repetitions", str(args.repetitions)]
    try:
        child_metrics = _run_bounded_child(command, process_timeout=args.timeout_seconds, log_path=output / "worker.log")
        if time.monotonic() - started > WAVE_TIMEOUT_SECONDS: raise RuntimeError("wave exceeded 1800 seconds")
        payload = json.loads(child_output.read_text(encoding="utf-8"))
        if payload.get("status") == "failed": raise RuntimeError(payload["error"])
        payload["process"] = child_metrics
        if args.profile:
            profile_output = output / "profile_worker.json"
            remaining = WAVE_TIMEOUT_SECONDS - (time.monotonic() - started)
            if remaining <= 0: raise RuntimeError("wave exceeded 1800 seconds")
            _run_bounded_child([*command, "--child-output", str(profile_output), "--profile-child"], process_timeout=min(args.timeout_seconds, remaining), log_path=output / "profile.log")
            profile_payload = json.loads(profile_output.read_text(encoding="utf-8"))
            if profile_payload.get("status") == "failed": raise RuntimeError(profile_payload["error"])
            payload["profile"] = {"raw": "profile.pstats", "calls": "profile_calls.json", "instrumented_separately": True}
        for key, root in (("solver_tree", args.solver_root), ("adapter_tree", args.adapter_root)):
            final_identity = _tracked_tree(Path(root))
            if final_identity != manifest[key]: raise RuntimeError(f"input changed during run: {root}")
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest["harness_sha256"]: raise RuntimeError("harness changed during run")
        if time.monotonic() - started > WAVE_TIMEOUT_SECONDS: raise RuntimeError("wave exceeded 1800 seconds")
        payload["status"] = "ok"
        _write_json(output / "result.json", payload)
        manifest.update({"status": "ok", "finished_unix": time.time()})
        _write_json(checkpoint, manifest)
        return 0
    except BaseException as error:
        manifest.update({"status": "failed", "finished_unix": time.time(), "error": f"{type(error).__name__}: {error}"})
        _write_json(checkpoint, manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-root", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--fixture", choices=("girder_panel", "cylinder"), required=True)
    parser.add_argument("--fidelity", choices=("coarse", "medium", "fine"), required=True)
    parser.add_argument("--route", choices=("modal", "buckling"), required=True)
    parser.add_argument("--modes", type=int, choices=(5, 10), default=5)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=PROCESS_TIMEOUT_SECONDS)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", help=argparse.SUPPRESS)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repetitions < 1: _parser().error("--repetitions must be positive")
    if not 0 < args.timeout_seconds <= PROCESS_TIMEOUT_SECONDS: _parser().error("--timeout-seconds must be in (0, 600]")
    if args._child:
        if not args.child_output: _parser().error("child output is required")
        return _child(args)
    if not args.output_dir: _parser().error("--output-dir is required")
    return _coordinator(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
