"""Verify one F03 child against its registration after its start marker.

This verifier performs bounded metadata and file checks only.  It neither
assembles a model nor creates terminal receipts or scientific evidence.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tomllib

import psutil
import threadpoolctl

from scripts import _b3_e4_f03_local_scientific_archive as archive
from scripts import _b3_e4_f03_local_scientific_cases as cases
from scripts import _b3_e4_f03_local_scientific_lifecycle as lifecycle
from scripts import _b3_e4_f03_local_scientific_registration as registration


def observed_hardware() -> registration.HardwareIdentity:
    machine = platform.machine()
    return registration.HardwareIdentity(
        socket.gethostname(), platform.platform(), machine,
        platform.processor() or machine, os.cpu_count(), psutil.virtual_memory().total,
    )


def observed_package(name: str) -> tuple[str, str]:
    if name not in registration.PACKAGE_NAMES:
        raise ValueError("unregistered F03 package name")
    import_name = "anyfesolver" if name == "ANYfeSolver" else name
    spec = importlib.util.find_spec(import_name)
    if spec is None or not spec.origin:
        raise ValueError(f"F03 package {name} has no import origin")
    return importlib.metadata.version(name), str(Path(spec.origin).resolve())


def candidate_anyfesolver_origin(repository_root: str) -> Path:
    """The source-backed F03 route must import the registered candidate."""
    return Path(repository_root).resolve() / "src" / "anyfesolver" / "__init__.py"


def candidate_anyfesolver_version(repository_root: str) -> str:
    with (Path(repository_root).resolve() / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    version = project["version"]
    if type(version) is not str or not version:
        raise ValueError("F03 candidate ANYfeSolver version is invalid")
    return version


def verify_loaded_sources(repository_root: str) -> None:
    expected = Path(repository_root).resolve() / "scripts"
    if Path(__file__).resolve() != expected / Path(__file__).name:
        raise ValueError("F03 child identity code is outside registered source")
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("scripts."):
            continue
        location = getattr(module, "__file__", None)
        if location is not None and not Path(location).resolve().is_relative_to(expected):
            raise ValueError(f"F03 child loaded module is outside registered source: {name}")


def observed_git(repository_root: str) -> tuple[str, bool]:
    def command(*args: str) -> str:
        completed = subprocess.run(
            ("git", "-C", repository_root, *args),
            check=True, capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip()
    return command("rev-parse", "HEAD"), not bool(
        command("status", "--porcelain", "--untracked-files=all")
    )


def verify_child_context(
    directory: Path,
    envelope: archive.CaseArchivePlan,
    source_commit: str,
    registration_sha256: str,
) -> tuple[cases.CasePreflight, ...]:
    """Reopen the immutable registration and reject any changed child context."""
    directory = Path(directory)
    if type(envelope) is not archive.CaseArchivePlan:
        raise ValueError("validated F03 child plan envelope required")
    run_root = directory.parent.parent
    value, observed_sha256 = registration.read_registration(
        run_root, registration_sha256,
    )
    verify_loaded_sources(value.repository_root)
    expected_dir = registration.case_directory(value.run_root, envelope.case_id)
    if (
        observed_sha256 != registration_sha256
        or directory != expected_dir
        or source_commit != value.source_commit
        or envelope.source_commit != value.source_commit
    ):
        raise ValueError("F03 child directory, source, or registration mismatch")
    start, _ = lifecycle.read_start(directory, envelope, registration_sha256)
    if start["pid"] != os.getpid() or start["parent_pid"] != os.getppid():
        raise ValueError("F03 child does not own its start marker")
    records = cases.preflight_inventory()
    bound = value.cases[tuple(item.case_id for item in value.cases).index(envelope.case_id)]
    record = next(item for item in records if item.case.case_id == envelope.case_id)
    if (
        cases.inventory_sha256(records) != value.inventory_sha256
        or record.topology_sha256 != bound.topology_sha256
        or lifecycle.plan_sha256(envelope) != bound.plan_sha256
    ):
        raise ValueError("F03 child plan or preflight identity mismatch")
    runtime = value.runtime
    if (
        Path(sys.executable).resolve() != Path(runtime.executable).resolve()
        or platform.python_implementation() != runtime.implementation
        or platform.python_version() != runtime.version
    ):
        raise ValueError("F03 child Python runtime identity mismatch")
    for package in value.packages:
        version, origin = observed_package(package.name)
        if version != package.version or Path(origin).resolve() != Path(package.origin).resolve():
            raise ValueError(f"F03 child package identity mismatch: {package.name}")
        if (package.name == "ANYfeSolver"
                and (Path(origin).resolve() != candidate_anyfesolver_origin(value.repository_root)
                     or version != candidate_anyfesolver_version(value.repository_root))):
            raise ValueError("F03 child ANYfeSolver origin/version disagrees with candidate")
    if observed_hardware() != value.hardware:
        raise ValueError("F03 child hardware identity mismatch")
    commit, clean = observed_git(value.repository_root)
    if commit != value.source_commit or not clean:
        raise ValueError("F03 child tracked source identity mismatch")
    if any(os.environ.get(name) != setting for name, setting in value.threads):
        raise ValueError("F03 child thread environment mismatch")
    pools = threadpoolctl.threadpool_info()
    if not pools or any(type(pool.get("num_threads")) is not int or pool["num_threads"] != 1
                        for pool in pools):
        raise ValueError("F03 child native thread pool mismatch")
    return records
