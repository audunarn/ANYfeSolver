"""Synthetic post-start child identity checks; no numerical work is launched."""
import os
import platform
import subprocess
from types import SimpleNamespace
import sys

import pytest

from scripts import _b3_e4_f03_local_scientific_archive as archive
from scripts import _b3_e4_f03_local_scientific_cases as cases
from scripts import _b3_e4_f03_local_scientific_child_identity as identity
from scripts import _b3_e4_f03_local_scientific_lifecycle as lifecycle
from scripts import _b3_e4_f03_local_scientific_registration as registration


SOURCE = "a" * 40
DIGEST = "b" * 64


def _context(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    run_root = tmp_path / "run"
    case_dir = run_root / "cases" / "Q1_L0"
    source_root.mkdir()
    version = identity.observed_package("ANYfeSolver")[0]
    (source_root / "pyproject.toml").write_text(
        f"[project]\nname = 'ANYfeSolver'\nversion = '{version}'\n", encoding="ascii",
    )
    solver_origin = source_root / "src" / "anyfesolver" / "__init__.py"
    solver_origin.parent.mkdir(parents=True)
    solver_origin.write_text("__version__ = '0.4.7'\n", encoding="ascii")
    case_dir.mkdir(parents=True)
    records = cases.preflight_inventory()
    plan = archive.make_case_plan(records[0], records, SOURCE)
    original_observed_package = identity.observed_package

    def observed_package(name):
        version, origin = original_observed_package(name)
        return version, str(solver_origin) if name == "ANYfeSolver" else origin

    monkeypatch.setattr(identity, "observed_package", observed_package)
    packages = tuple(
        SimpleNamespace(name=name, version=version, origin=origin)
        for name in registration.PACKAGE_NAMES
        for version, origin in (identity.observed_package(name),)
    )
    value = SimpleNamespace(
        run_root=str(run_root), repository_root=str(source_root),
        source_commit=SOURCE, inventory_sha256=cases.inventory_sha256(records),
        cases=tuple(SimpleNamespace(
            case_id=record.case.case_id, topology_sha256=record.topology_sha256,
            plan_sha256=lifecycle.plan_sha256(
                archive.make_case_plan(record, records, SOURCE)
            ),
        ) for record in records),
        runtime=SimpleNamespace(
            executable=sys.executable,
            implementation=platform.python_implementation(),
            version=platform.python_version(),
        ),
        packages=packages,
        hardware=identity.observed_hardware(),
        threads=registration.thread_contract(),
    )
    monkeypatch.setattr(identity.registration, "read_registration", lambda *args: (value, DIGEST))
    monkeypatch.setattr(identity, "verify_loaded_sources", lambda *_: None)
    monkeypatch.setattr(identity.lifecycle, "read_start", lambda *args: (
        {"pid": os.getpid(), "parent_pid": os.getppid()}, DIGEST,
    ))
    monkeypatch.setattr(identity, "observed_git", lambda *args: (SOURCE, True))
    monkeypatch.setattr(identity.threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    for name, setting in value.threads:
        monkeypatch.setenv(name, setting)
    return case_dir, plan, records, value


def test_child_identity_reopens_registration_after_owned_start(tmp_path, monkeypatch):
    directory, plan, records, _ = _context(tmp_path, monkeypatch)
    assert identity.verify_child_context(directory, plan, SOURCE, DIGEST) == records


def test_child_identity_rejects_stale_registration_and_start(tmp_path, monkeypatch):
    directory, plan, _, _ = _context(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="registration mismatch"):
        identity.verify_child_context(directory, plan, SOURCE, "c" * 64)
    monkeypatch.setattr(identity.lifecycle, "read_start", lambda *args: (
        {"pid": os.getpid() + 1, "parent_pid": os.getppid()}, DIGEST,
    ))
    with pytest.raises(ValueError, match="start marker"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)
    monkeypatch.setattr(identity.lifecycle, "read_start", lambda *args: (
        {"pid": os.getpid(), "parent_pid": os.getppid() + 1}, DIGEST,
    ))
    with pytest.raises(ValueError, match="start marker"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)


def test_child_identity_rejects_runtime_package_and_hardware_changes(tmp_path, monkeypatch):
    directory, plan, _, value = _context(tmp_path, monkeypatch)
    original = value.runtime.version
    value.runtime.version = "0.0.0"
    with pytest.raises(ValueError, match="runtime identity"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)
    value.runtime.version = original
    package = value.packages[0]
    mutated = list(value.packages)
    mutated[0] = SimpleNamespace(name=package.name, version="0.0.0", origin=package.origin)
    value.packages = tuple(mutated)
    with pytest.raises(ValueError, match="package identity"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)
    value.packages = (package, *value.packages[1:])
    hardware = {name: getattr(value.hardware, name) for name in (
        "hostname", "platform", "machine", "processor", "logical_cpus",
        "physical_memory_bytes",
    )}
    hardware["logical_cpus"] += 1
    value.hardware = SimpleNamespace(**hardware)
    with pytest.raises(ValueError, match="hardware identity"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)


def test_child_rejects_external_anyfesolver_even_if_registration_matches(tmp_path, monkeypatch):
    directory, plan, _, value = _context(tmp_path, monkeypatch)
    external = tmp_path / "external_anyfesolver.py"
    external.write_text("external\n", encoding="ascii")
    package = value.packages[0]
    value.packages = (SimpleNamespace(name="ANYfeSolver", version=package.version,
                                      origin=str(external)), *value.packages[1:])
    observed = identity.observed_package
    monkeypatch.setattr(identity, "observed_package",
                        lambda name: (package.version, str(external))
                        if name == "ANYfeSolver" else observed(name))
    with pytest.raises(ValueError, match="origin/version disagrees"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)


def test_child_identity_rejects_git_thread_and_pool_changes(tmp_path, monkeypatch):
    directory, plan, _, _ = _context(tmp_path, monkeypatch)
    monkeypatch.setattr(identity, "observed_git", lambda *args: (SOURCE, False))
    with pytest.raises(ValueError, match="tracked source"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)
    monkeypatch.setattr(identity, "observed_git", lambda *args: (SOURCE, True))
    monkeypatch.setenv(registration.thread_contract()[0][0], "2")
    with pytest.raises(ValueError, match="thread environment"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)
    monkeypatch.setenv(registration.thread_contract()[0][0], "1")
    monkeypatch.setattr(identity.threadpoolctl, "threadpool_info", lambda: [{"num_threads": 2}])
    with pytest.raises(ValueError, match="native thread pool"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)


def test_child_identity_rejects_plan_and_directory_changes(tmp_path, monkeypatch):
    directory, plan, _, value = _context(tmp_path, monkeypatch)
    bound = value.cases[0]
    value.cases = (SimpleNamespace(
        case_id=bound.case_id, topology_sha256=bound.topology_sha256,
        plan_sha256="c" * 64,
    ), *value.cases[1:])
    with pytest.raises(ValueError, match="plan or preflight"):
        identity.verify_child_context(directory, plan, SOURCE, DIGEST)
    value.cases = (bound, *value.cases[1:])
    with pytest.raises(ValueError, match="directory"):
        identity.verify_child_context(directory.parent, plan, SOURCE, DIGEST)


def test_observed_git_rejects_untracked_importable_source(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    def git(*args):
        return subprocess.run(
            ("git", "-C", str(repository), *args),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    git("init", "-q")
    (repository / "tracked.py").write_text("VALUE = 1\n", encoding="ascii")
    git("add", "tracked.py")
    git("-c", "user.name=F03 Test", "-c", "user.email=f03@example.invalid",
        "commit", "-qm", "synthetic baseline")
    commit, clean = identity.observed_git(str(repository))
    assert commit == git("rev-parse", "HEAD")
    assert clean
    (repository / "sitecustomize.py").write_text("VALUE = 2\n", encoding="ascii")
    assert identity.observed_git(str(repository)) == (commit, False)
