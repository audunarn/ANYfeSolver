"""One-use external resource/identity attestation for the F06 v4 S3 case.

This supervisor does not decide mechanics acceptance. It consumes an exact
registration once, runs its fixed command, and preserves raw outputs even on
failure. A separate adjudicator applies the preregistered scientific gates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic, sleep

import psutil
if sys.platform == "win32":
    import msvcrt
    import win32api
    import win32event
    import win32job
    import win32process
else:
    msvcrt = win32api = win32event = win32job = win32process = None


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "G0_F06_V4_SCIENTIFIC_REGISTRATION_V1"
ATTESTATION_SCHEMA = "G0_F06_V4_SUPERVISOR_ATTESTATION_V1"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SIDECARS = ("base", "rebase", "successor", "control")
MEASUREMENT_SOURCES = frozenset((
    "scripts/probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py",
    "tests/test_b3_e4_f06_owned_rebase_v4.py",
    "docs/agent_plans/B3_GE_E4_PL_G0_F06_V4_SCIENTIFIC_CONTRACT_DRAFT.md",
    "scripts/probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3.py",
    "src/anyfesolver/_ge_beam3_shell_joint_state.py",
    "src/anyfesolver/_ge_beam3_shell_joint_trial.py",
    "src/anyfesolver/_ge_beam3_g3c_local_shell.py",
    "src/anyfesolver/_ge_beam3_coupled_beam_subdomain.py",
    "src/anyfesolver/_ge_beam3_pose_joint.py",
    "src/anyfesolver/_ge_beam3_retained_generalized.py",
    "src/anyfesolver/_ge_beam3_retained_generalized_state.py",
    "src/anyfesolver/_ge_beam3_retained_generalized_program.py",
    "src/anyfesolver/_ge_beam3_native_generalized_element.py",
    "src/anyfesolver/_ge_beam3_generalized_ellipsoid_section.py",
    "src/anyfesolver/_ge_beam3_generalized_cell.py",
    "src/anyfesolver/_ge_beam3_generalized_static_boundary.py",
    "src/anyfesolver/_ge_beam3_p5/algebra.py",
    "src/anyfesolver/_ge_beam3_p5_seeded/core.py",
    "src/anyfesolver/e4_pl_s3_v2d_element.py",
    "src/anyfesolver/control.py",
    "src/anyfesolver/_native_reference_modal.py",
    "src/anyfesolver/_ge_beam3_native_generalized_loading.py",
    "src/anyfesolver/_ge_beam3_native_line_loading.py",
    "src/anyfesolver/_ge_beam3_centered_reference.py",
    "src/anyfesolver/fe_core.py",
))
REQUIRED_SOURCES = MEASUREMENT_SOURCES | frozenset((
    "docs/agent_plans/B3_GE_E4_PL_G0_F06_OWNED_REBASE_V4_PROTOCOL.md",
    "scripts/adjudicate_b3_ge_e4_pl_g0_f06_v4.py",
    "tests/test_b3_e4_f06_v4_adjudicator.py",
    "scripts/supervise_b3_ge_e4_pl_g0_f06_v4.py",
    "tests/test_b3_e4_f06_v4_supervisor.py",
))


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _measurement_canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside_root(relative: str) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError("relative source path required")
    path = (ROOT / relative).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("source path escapes repository")
    if not path.is_file():
        raise ValueError("source path is not a file")
    return path


def _observed_sources(sources: dict[str, str]) -> dict[str, str]:
    if type(sources) is not dict or set(sources) != REQUIRED_SOURCES:
        raise ValueError("exact F06 source identity inventory required")
    made = {}
    for name, expected in sources.items():
        if type(expected) is not str or len(expected) != 64:
            raise ValueError("source SHA-256 identity malformed")
        made[name] = _sha(_inside_root(name))
    return made


def _environment() -> dict:
    if not os.environ.get("NUMBA_CACHE_DIR"):
        raise ValueError("explicit NUMBA_CACHE_DIR required before registration claim")
    local_src = str((ROOT / "src").resolve())
    if local_src not in sys.path:
        sys.path.insert(0, local_src)
    import numpy
    import scipy
    import anyfesolver
    import threadpoolctl

    expected_anyfesolver = (ROOT / "src/anyfesolver/__init__.py").resolve()
    if Path(anyfesolver.__file__).resolve() != expected_anyfesolver:
        raise ValueError("supervisor must import the frozen local ANYfeSolver")
    return dict(python_executable=str(Path(sys.executable).resolve()),
                python_version=sys.version.split()[0],
                numpy_version=numpy.__version__, scipy_version=scipy.__version__,
                anyfesolver_import=str(Path(anyfesolver.__file__).resolve()),
                psutil_version=psutil.__version__,
                threadpoolctl_version=threadpoolctl.__version__,
                pywin32_version=package_version("pywin32"),
                win32job_import=str(Path(win32job.__file__).resolve()),
                numba_version=package_version("numba"),
                llvmlite_version=package_version("llvmlite"),
                platform=platform.platform(),
                numba_cache_dir=os.environ.get("NUMBA_CACHE_DIR"),
                python_environment={key: os.environ.get(key) for key in
                                    ("PYTHONPATH", "PYTHONNOUSERSITE")},
                thread_environment={key: os.environ.get(key) for key in THREAD_KEYS},
                blas=threadpoolctl.threadpool_info())


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"),
                                   cwd=ROOT, text=True).strip()


def _require_clean_tracked_tree() -> None:
    result = subprocess.run(("git", "diff", "--quiet", "HEAD", "--"),
                            cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode == 1:
        raise ValueError("tracked worktree differs from registered HEAD")
    if result.returncode != 0:
        raise ValueError("tracked worktree cleanliness check failed: " +
                         result.stderr.decode("utf-8", errors="replace").strip())


def _registration(path: Path, evidence: Path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical(value):
        raise ValueError("canonical registration JSON required")
    if value.get("schema") != SCHEMA or value.get("source_commit") != _git_head():
        raise ValueError("registration schema/source commit mismatch")
    _require_clean_tracked_tree()
    if Path(value.get("evidence_dir", "")).resolve() != evidence:
        raise ValueError("registration evidence directory mismatch")
    observed_sources = _observed_sources(value.get("sources"))
    if observed_sources != value["sources"]:
        raise ValueError("registration source identities mismatch")
    observed_environment = _environment()
    if value.get("environment") != observed_environment:
        raise ValueError("registration environment identities mismatch")
    test_log = evidence / "test.log"
    if (not test_log.is_file() or value.get("test_log_sha256") != _sha(test_log)):
        raise ValueError("registered focused test log identity mismatch")
    expected = [str(Path(sys.executable).resolve()),
                str((ROOT / "scripts/probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py").resolve()),
                "--output-dir", str((evidence / "measurement").resolve())]
    if value.get("command") != dict(argv=expected, cwd=str(ROOT.resolve())):
        raise ValueError("registration exact command mismatch")
    limits = value.get("limits")
    if (type(limits) is not dict or set(limits) !=
            {"wall_seconds", "sampled_tree_rss_bytes"} or
            limits["wall_seconds"] != 180 or
            limits["sampled_tree_rss_bytes"] != 2 * 1024**3):
        raise ValueError("registration resource limits mismatch")
    return value, sha256(raw).hexdigest(), observed_sources, observed_environment


def _launch_owned(argv: list[str], stdout, stderr):
    """Assign a suspended child before any of its code can create descendants."""
    if sys.platform != "win32":
        raise RuntimeError("F06 scientific supervisor requires Windows Job Objects")
    job = win32job.CreateJobObject(None, "")
    process_handle = None
    try:
        limits = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, limits)
        with open(os.devnull, "rb") as stdin:
            handles = tuple(msvcrt.get_osfhandle(stream.fileno())
                            for stream in (stdin, stdout, stderr))
            inherited = tuple(os.get_handle_inheritable(handle) for handle in handles)
            try:
                for handle in handles:
                    os.set_handle_inheritable(handle, True)
                startup = win32process.STARTUPINFO()
                startup.dwFlags |= win32process.STARTF_USESTDHANDLES
                startup.hStdInput, startup.hStdOutput, startup.hStdError = handles
                process_handle, thread_handle, pid, _ = win32process.CreateProcess(
                    None, subprocess.list2cmdline(argv), None, None, True,
                    win32process.CREATE_SUSPENDED | win32process.CREATE_NO_WINDOW,
                    os.environ.copy(), str(ROOT), startup)
                try:
                    win32job.AssignProcessToJobObject(job, process_handle)
                    win32process.ResumeThread(thread_handle)
                except BaseException:
                    win32process.TerminateProcess(process_handle, 1)
                    raise
                finally:
                    win32api.CloseHandle(thread_handle)
            finally:
                for handle, was_inherited in zip(handles, inherited):
                    os.set_handle_inheritable(handle, was_inherited)
        return job, process_handle, pid
    except BaseException:
        if process_handle is not None:
            win32api.CloseHandle(process_handle)
        win32api.CloseHandle(job)
        raise


def _job_sample(job) -> tuple[int, int, tuple[int, ...]]:
    """Sample all active job members, including children of an exited root."""
    active = win32job.QueryInformationJobObject(
        job, win32job.JobObjectBasicAccountingInformation)["ActiveProcesses"]
    pids = tuple(win32job.QueryInformationJobObject(
        job, win32job.JobObjectBasicProcessIdList))
    total = 0
    for pid in pids:
        try:
            total += psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError(f"job RSS access denied for PID {pid}") from error
    return active, total, pids


def _wait_job_empty(job, seconds: float) -> None:
    deadline = monotonic() + seconds
    while win32job.QueryInformationJobObject(
            job, win32job.JobObjectBasicAccountingInformation)["ActiveProcesses"]:
        if monotonic() >= deadline:
            raise RuntimeError("owned Windows job did not terminate within bound")
        sleep(0.05)


def _artifact(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return dict(file=path.name, byte_count=path.stat().st_size, sha256=_sha(path))


def supervise(registration_path: Path, evidence: Path) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("F06 scientific supervisor requires Windows Job Objects")
    registration_path = registration_path.resolve(strict=True)
    evidence = evidence.resolve(strict=True)
    if registration_path.parent != evidence:
        raise ValueError("registration must be inside its evidence directory")
    registration, registration_sha256, sources, environment = _registration(
        registration_path, evidence)
    claim = evidence / "execution.claim"
    stdout_path, stderr_path = evidence / "stdout.log", evidence / "stderr.log"
    attestation_path = evidence / "attestation.json"
    for path in (claim, stdout_path, stderr_path, attestation_path,
                 evidence / "measurement"):
        if path.exists():
            raise FileExistsError(f"F06 registration already consumed: {path.name}")
    claim_error = None
    try:
        with claim.open("xb") as stream:
            stream.write(_canonical(dict(registration_sha256=registration_sha256,
                                         claimed_at_utc=_utc())))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise  # A concurrent supervisor owns this registration.
    except BaseException as error:
        if not claim.exists():
            raise
        claim_error = f"{type(error).__name__}: {error}"
    started_utc, started = _utc(), monotonic()
    peak, samples, violation, launch_error, exit_code = 0, 0, None, claim_error, None
    root_pid, last_job_pids = None, ()
    job = process_handle = stdout = stderr = None
    try:
        if claim_error is None:
            stdout = stdout_path.open("xb")
            stderr = stderr_path.open("xb")
            job, process_handle, root_pid = _launch_owned(
                registration["command"]["argv"], stdout, stderr)
            while True:
                active, rss, last_job_pids = _job_sample(job)
                samples += 1
                peak = max(peak, rss)
                if monotonic() - started > registration["limits"]["wall_seconds"]:
                    violation = "EXTERNAL_WALL_TIMEOUT"
                    break
                if peak > registration["limits"]["sampled_tree_rss_bytes"]:
                    violation = "SAMPLED_TREE_RSS_LIMIT"
                    break
                if active == 0:
                    break
                sleep(0.1)
    except BaseException as error:
        launch_error = f"{type(error).__name__}: {error}"
    finally:
        if job is not None:
            try:
                if win32job.QueryInformationJobObject(
                        job, win32job.JobObjectBasicAccountingInformation)["ActiveProcesses"]:
                    win32job.TerminateJobObject(job, 1)
                _wait_job_empty(job, 5.)
            except BaseException as error:
                message = f"{type(error).__name__}: {error}"
                launch_error = message if launch_error is None else launch_error + "; " + message
            finally:
                # Closing a KILL_ON_JOB_CLOSE job is the final containment backstop.
                try:
                    win32api.CloseHandle(job)
                except BaseException as error:
                    message = f"{type(error).__name__}: {error}"
                    launch_error = message if launch_error is None else launch_error + "; " + message
        if process_handle is not None:
            try:
                if win32event.WaitForSingleObject(process_handle, 5000) != win32event.WAIT_OBJECT_0:
                    raise RuntimeError("owned root process did not terminate within bound")
                exit_code = win32process.GetExitCodeProcess(process_handle)
            except BaseException as error:
                message = f"{type(error).__name__}: {error}"
                launch_error = message if launch_error is None else launch_error + "; " + message
            finally:
                try:
                    win32api.CloseHandle(process_handle)
                except BaseException as error:
                    message = f"{type(error).__name__}: {error}"
                    launch_error = message if launch_error is None else launch_error + "; " + message
        for stream in (stdout, stderr):
            if stream is not None:
                try:
                    stream.flush(); os.fsync(stream.fileno())
                except BaseException as error:
                    message = f"{type(error).__name__}: {error}"
                    launch_error = message if launch_error is None else launch_error + "; " + message
                finally:
                    try:
                        stream.close()
                    except BaseException as error:
                        message = f"{type(error).__name__}: {error}"
                        launch_error = message if launch_error is None else launch_error + "; " + message
    measurement = evidence / "measurement"
    artifact_errors = []
    def safe_artifact(path):
        try:
            return _artifact(path)
        except BaseException as error:
            artifact_errors.append(f"{path.name}: {type(error).__name__}: {error}")
            return None
    artifacts = dict(metrics=safe_artifact(measurement / "metrics.json"),
                     checkpoints={name: safe_artifact(measurement / f"{name}.checkpoint")
                                  for name in SIDECARS})
    observed_child_environment, child_environment_match = None, None
    if exit_code == 0 and violation is None and launch_error is None:
        try:
            if artifacts["metrics"] is None:
                raise ValueError("successful child omitted metrics.json")
            if any(row is None for row in artifacts["checkpoints"].values()):
                raise ValueError("successful child omitted required checkpoint sidecars")
            raw_metrics = (measurement / "metrics.json").read_bytes()
            metrics = json.loads(raw_metrics)
            if type(metrics) is not dict or raw_metrics != _measurement_canonical(metrics):
                raise ValueError("child metrics are not canonical JSON")
            observed_child_environment = metrics["metrics"]["effective_environment"]
            child_environment_match = observed_child_environment == environment
            if not child_environment_match:
                raise ValueError("child effective environment differs from registration")
        except BaseException as error:
            launch_error = f"{type(error).__name__}: {error}"
    def safe_hash(path):
        try:
            return _sha(path) if path.is_file() else None
        except BaseException as error:
            artifact_errors.append(f"{path.name}: {type(error).__name__}: {error}")
            return None
    stdout_hash, stderr_hash = safe_hash(stdout_path), safe_hash(stderr_path)
    test_log_hash = safe_hash(evidence / "test.log")
    if test_log_hash != registration["test_log_sha256"]:
        message = "registered focused test log changed after claim"
        launch_error = message if launch_error is None else launch_error + "; " + message
    postrun_sources_match = None
    try:
        postrun_sources_match = _observed_sources(registration["sources"]) == sources
        if not postrun_sources_match:
            raise ValueError("registered sources changed during execution")
        if _sha(registration_path) != registration_sha256:
            raise ValueError("registration changed during execution")
        if _git_head() != registration["source_commit"]:
            raise ValueError("registered HEAD changed during execution")
        _require_clean_tracked_tree()
    except BaseException as error:
        message = f"{type(error).__name__}: {error}"
        launch_error = message if launch_error is None else launch_error + "; " + message
    if artifact_errors:
        message = "; ".join(artifact_errors)
        launch_error = message if launch_error is None else launch_error + "; " + message
    attestation = dict(schema=ATTESTATION_SCHEMA,
                       registration_sha256=registration_sha256,
                       command=registration["command"],
                       observed_sources=sources,
                       observed_environment=environment,
                       start_utc=started_utc, end_utc=_utc(),
                       wall_seconds=monotonic() - started,
                       peak_tree_rss_bytes=peak,
                       rss_sample_count=samples, root_pid=root_pid,
                       last_job_pids=list(last_job_pids),
                       exit_code=exit_code, resource_violation=violation,
                       launch_error=launch_error,
                       observed_child_environment=observed_child_environment,
                       child_environment_match=child_environment_match,
                       stdout_sha256=stdout_hash, stderr_sha256=stderr_hash,
                       test_log_sha256=test_log_hash,
                       metrics_sha256=None if artifacts["metrics"] is None
                           else artifacts["metrics"]["sha256"],
                       artifacts=artifacts)
    with attestation_path.open("xb") as stream:
        stream.write(_canonical(attestation))
        stream.flush()
        os.fsync(stream.fileno())
    return attestation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = supervise(args.registration, args.evidence_dir)
    print(json.dumps(dict(exit_code=result["exit_code"],
                          resource_violation=result["resource_violation"],
                          attestation=str((args.evidence_dir / "attestation.json").resolve())),
                     sort_keys=True))
    return 0 if result["exit_code"] == 0 and result["resource_violation"] is None \
        and result["launch_error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
