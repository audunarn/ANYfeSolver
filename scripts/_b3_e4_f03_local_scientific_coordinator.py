"""Serial, one-use execution of an already registered F03 scientific run.

This is the sole launcher for the local scientific protocol.  It never creates
or replaces a registration.  A claim is written before the first child, so a
crash, cancellation, timeout, or invalid evidence cannot cause an implicit
retry.  Only the registered child command may produce mechanical evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import os
import json
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Callable, Sequence

from scripts import _b3_e4_f03_local_scientific_aggregate as aggregate
from scripts import _b3_e4_f03_local_scientific_archive as archive
from scripts import _b3_e4_f03_local_scientific_budgets as budget_contract
from scripts import _b3_e4_f03_local_scientific_case_adjudicator as adjudicator
from scripts import _b3_e4_f03_local_scientific_cases as cases
from scripts import _b3_e4_f03_local_scientific_child_identity as child_identity
from scripts import _b3_e4_f03_local_scientific_evidence_index as evidence_index
from scripts import _b3_e4_f03_local_scientific_lifecycle as lifecycle
from scripts import _b3_e4_f03_local_scientific_registration as registration
from scripts import _b3_e4_f03_local_scientific_supervisor as supervisor


CLAIM_FILE = "execution_claim.json"
CLAIM_SCHEMA = "f03_local_scientific_execution_claim_v1"


LaunchCutoff = supervisor.LaunchCutoff


def _verify_loaded_sources(repository_root: Path) -> None:
    """Prevent attesting one checkout while executing modules from another."""
    child_identity.verify_loaded_sources(str(repository_root))


def _owned_file_identity(path: Path) -> tuple[int, str]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or not 0 < info.st_size <= registration.MAX_IDENTITY_FILE_BYTES
            or (hasattr(info, "st_file_attributes") and info.st_file_attributes & 0x400)):
        raise ValueError("F03 registration requires a bounded owned identity file")
    digest = hashlib.sha256()
    count = 0
    with evidence_index._open_no_follow(path) as stream:
        opened = os.fstat(stream.fileno())
        if ((opened.st_dev, opened.st_ino, opened.st_size)
                != (info.st_dev, info.st_ino, info.st_size)):
            raise ValueError("F03 identity file changed before registration read")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            if count > registration.MAX_IDENTITY_FILE_BYTES:
                raise ValueError("F03 identity file exceeds registration bound")
            digest.update(block)
    after = path.lstat()
    if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            or count != info.st_size):
        raise ValueError("F03 identity file changed during registration read")
    return count, digest.hexdigest()


def prepare_registration(
    repository_root: Path, run_root: Path,
) -> tuple[registration.Registration, str]:
    """Freeze one clean source/environment and create an exclusive registration.

    This prepares plan archives but launches no child.  A partial directory is
    intentionally retained if preparation fails and cannot be reused.
    """
    source = Path(repository_root).resolve(strict=True)
    run_root = Path(run_root).absolute()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("F03 scientific run root must be new and one-use")
    if not run_root.parent.is_dir() or run_root.parent.is_symlink():
        raise ValueError("owned F03 run parent required")
    source_resolved = source.resolve()
    run_resolved = run_root.resolve()
    if (source_resolved == run_resolved
            or run_resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(run_resolved)):
        raise ValueError("F03 source and run roots must be separate")
    registration._owned_directory(source)
    _verify_loaded_sources(source)
    commit, clean = child_identity.observed_git(str(source))
    if not clean or not registration._commit(commit):
        raise ValueError("clean committed F03 source required before registration")
    local_main = _main_commit(str(source), "refs/heads/main")
    remote_main = _main_commit(str(source), "refs/remotes/origin/main")
    if local_main != remote_main or not registration._commit(local_main):
        raise ValueError("exact accepted local/remote main required")
    source_identities = []
    for role, relative in registration.SOURCE_ROLES:
        count, digest = _owned_file_identity(source.joinpath(*relative.split("/")))
        source_identities.append(registration.SourceIdentity(role, relative, count, digest))
    runtime_path = Path(sys.executable).resolve(strict=True)
    runtime_count, runtime_digest = _owned_file_identity(runtime_path)
    runtime = registration.RuntimeIdentity(
        str(runtime_path), runtime_count, runtime_digest,
        platform.python_implementation(), platform.python_version(),
    )
    packages = []
    for name in registration.PACKAGE_NAMES:
        version, origin = child_identity.observed_package(name)
        path = Path(origin).resolve(strict=True)
        count, digest = _owned_file_identity(path)
        packages.append(registration.PackageIdentity(
            name, version, str(path), count, digest,
        ))
    solver_package = next(item for item in packages if item.name == "ANYfeSolver")
    if (solver_package.origin != str(child_identity.candidate_anyfesolver_origin(str(source)))
            or solver_package.version != child_identity.candidate_anyfesolver_version(str(source))):
        raise ValueError("F03 registration requires candidate ANYfeSolver origin/version")
    hardware = child_identity.observed_hardware()
    records = cases.preflight_inventory()
    run_root.mkdir()
    bindings = []
    for record in records:
        directory = registration.case_directory(str(run_root), record.case.case_id)
        directory.mkdir(parents=True)
        plan = archive.make_case_plan(record, records, commit)
        archive.write_plan(directory / "plan.zip", plan)
        _, plan_digest = _owned_file_identity(directory / "plan.zip")
        bindings.append(registration.CaseBinding(
            record.case.case_id, record.topology_sha256,
            record.case.kkt_order,
            f"cases/{record.case.case_id}/plan.zip",
            lifecycle.plan_sha256(plan), plan_digest,
        ))
    commands = tuple(registration.CommandBinding(
        bound.case_id, str(source),
        registration.child_argv(
            runtime.executable, str(run_root), bound.case_id, commit,
        ),
        f"cases/{bound.case_id}/stdout.log",
        f"cases/{bound.case_id}/stderr.log",
    ) for bound in bindings)
    registered = registration.Registration(
        registration.SCHEMA,
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        str(source), str(run_root), commit, local_main, True,
        cases.inventory_sha256(records), tuple(source_identities), runtime,
        tuple(packages), hardware, registration.thread_contract(),
        registration.process_contract(), registration.resource_contract(),
        registration.tolerance_contract(), tuple(bindings), commands,
        registration.TERMINAL_PRECEDENCE,
    )
    digest = registration.write_registration(run_root, registered)
    retained, observed = registration.read_registration(run_root, digest)
    if retained != registered or observed != digest:
        raise ValueError("F03 registration changed immediately after write")
    final_commit, final_clean = child_identity.observed_git(str(source))
    if final_commit != commit or not final_clean:
        raise ValueError("F03 source changed during registration preparation")
    return registered, digest


def _main_commit(repository_root: str, ref: str) -> str:
    completed = subprocess.run(
        ("git", "-C", repository_root, "rev-parse", "--verify", ref),
        check=True, capture_output=True, text=True, timeout=10,
    )
    return completed.stdout.strip()


def _preflight(
    run_root: Path, expected_sha256: str,
) -> registration.Registration:
    """Rebind all registration, source, runtime, and environment identities."""
    registered, digest = registration.read_registration(run_root, expected_sha256)
    if digest != expected_sha256:
        raise ValueError("F03 coordinator registration digest mismatch")
    _verify_loaded_sources(Path(registered.repository_root))
    commit, clean = child_identity.observed_git(registered.repository_root)
    if commit != registered.source_commit or not clean:
        raise ValueError("F03 coordinator requires exact clean source commit")
    if (
        _main_commit(registered.repository_root, "refs/heads/main")
        != registered.accepted_main_commit
        or _main_commit(registered.repository_root, "refs/remotes/origin/main")
        != registered.accepted_main_commit
    ):
        raise ValueError("F03 accepted local/remote main identity changed")
    runtime = registered.runtime
    if (
        Path(sys.executable).resolve() != Path(runtime.executable).resolve()
        or platform.python_implementation() != runtime.implementation
        or platform.python_version() != runtime.version
    ):
        raise ValueError("F03 coordinator Python runtime identity changed")
    for package in registered.packages:
        version, origin = child_identity.observed_package(package.name)
        if version != package.version or Path(origin).resolve() != Path(package.origin).resolve():
            raise ValueError(f"F03 coordinator package identity changed: {package.name}")
        if (package.name == "ANYfeSolver"
                and (Path(origin).resolve()
                     != child_identity.candidate_anyfesolver_origin(registered.repository_root)
                     or version != child_identity.candidate_anyfesolver_version(
                         registered.repository_root))):
            raise ValueError("F03 coordinator ANYfeSolver origin/version disagrees with candidate")
    if child_identity.observed_hardware() != registered.hardware:
        raise ValueError("F03 coordinator hardware identity changed")
    if (
        registered.threads != registration.thread_contract()
        or registered.process_limits != registration.process_contract()
        or registered.resource_limits != registration.resource_contract()
        or registered.tolerances != registration.tolerance_contract()
    ):
        raise ValueError("F03 coordinator execution contract changed")
    return registered


def _unused(run_root: Path, value: registration.Registration) -> None:
    if any(_occupied(path) for path in (
        run_root / CLAIM_FILE, run_root / aggregate.aggregate_receipt.FILENAME,
    )):
        raise FileExistsError("F03 scientific run was already claimed or closed")
    for bound in value.cases:
        directory = registration.case_directory(value.run_root, bound.case_id)
        if any(_occupied(directory / name) for name in (
            "start.json", "terminal.json", "case_receipt.json", "stdout.log", "stderr.log",
        )):
            raise FileExistsError(f"F03 registered case was already consumed: {bound.case_id}")


def _occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _claim(run_root: Path, value: registration.Registration, digest: str) -> None:
    document = {
        "schema": CLAIM_SCHEMA,
        "registration_sha256": digest,
        "source_commit": value.source_commit,
        "case_ids": [item.case_id for item in value.cases],
        "claimed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    with (run_root / CLAIM_FILE).open("xb") as stream:
        stream.write(payload)


def _run_one(
    run_root: Path,
    value: registration.Registration,
    digest: str,
    record: cases.CasePreflight,
    records: tuple[cases.CasePreflight, ...],
    budgets: budget_contract.ScientificAdjudicationBudgets,
    *, launch_deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    case_id = record.case.case_id
    directory = registration.case_directory(value.run_root, case_id)
    command = next(item for item in value.commands if item.case_id == case_id)
    bound = next(item for item in value.cases if item.case_id == case_id)
    adjudicator._verify_registered_plan_bytes(directory / "plan.zip", bound.plan_archive_sha256)
    plan = archive.read_plan(directory / "plan.zip", record, records, value.source_commit)
    if lifecycle.plan_sha256(plan) != bound.plan_sha256:
        raise ValueError("F03 coordinator case plan identity changed")
    if command.argv != registration.child_argv(
        value.runtime.executable, value.run_root, case_id, value.source_commit,
    ) or command.working_directory != value.repository_root:
        raise ValueError("F03 registered child command changed")
    if command.argv.count(registration.REGISTRATION_TOKEN) != 1:
        raise ValueError("F03 registered command requires one digest token")
    launched_argv = tuple(
        digest if item == registration.REGISTRATION_TOKEN else item
        for item in command.argv
    )
    if any(_occupied(directory / name) for name in (
        "start.json", "terminal.json", "case_receipt.json", "stdout.log", "stderr.log",
    )):
        raise FileExistsError(f"F03 case has already been launched: {case_id}")
    environment = supervisor.frozen_thread_environment()
    if any(environment.get(name) != setting for name, setting in value.threads):
        raise ValueError("F03 child thread settings changed")
    stdout_path = run_root.joinpath(*command.stdout_path.split("/"))
    stderr_path = run_root.joinpath(*command.stderr_path.split("/"))
    # The child inventories its case directory before the supervisor finishes.
    # Keep the bounded streams private until then: an open Windows log writer
    # cannot be reopened by the child's no-follow evidence scanner.
    stdout = io.BytesIO()
    stderr = io.BytesIO()

    def write_logs() -> None:
        first_error = None
        for path, stream in ((stdout_path, stdout), (stderr_path, stderr)):
            try:
                payload = stream.getvalue()
                if len(payload) > supervisor.LOG_BYTES_PER_STREAM:
                    raise ValueError("F03 captured log exceeds byte cap")
                with path.open("xb") as target:
                    target.write(payload)
            except (OSError, ValueError) as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    try:
        if launch_deadline is not None and monotonic() >= launch_deadline:
            raise LaunchCutoff("F03 serial launch cutoff reached before child start")
        terminal = supervisor.supervise(
            launched_argv,
            cwd=Path(command.working_directory),
            environment=environment,
            stdout=stdout, stderr=stderr,
            limits=supervisor.SCIENTIFIC_LIMITS,
            launch_deadline=launch_deadline,
            log_byte_limit=supervisor.LOG_BYTES_PER_STREAM,
        )
    except BaseException as exc:
        try:
            write_logs()
        except (OSError, ValueError) as log_error:
            raise exc from log_error
        if isinstance(exc, supervisor.SupervisionCancelled):
            lifecycle.write_terminal(directory, plan, digest, exc.receipt)
            adjudicator.close_failed_case(directory, plan, digest)
        raise
    write_logs()
    lifecycle.write_terminal(directory, plan, digest, terminal)
    # A changed source cannot be used to interpret the just-written evidence.
    if _preflight(run_root, digest) != value:
        raise ValueError("F03 registration changed after child execution")
    if terminal.reason != "completed" or terminal.exit_code != 0:
        adjudicator.close_failed_case(directory, plan, digest)
        return
    try:
        adjudicator.adjudicate_case(
            directory, record, records, value.source_commit, digest, budgets,
        )
    except (OSError, ValueError) as exc:
        adjudicator.close_failed_case(
            directory, plan, digest,
            evidence_failure=f"{type(exc).__name__}: {exc}",
        )


def run_registered(
    run_root: Path,
    *,
    registration_sha256: str,
    budgets: budget_contract.ScientificAdjudicationBudgets,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[aggregate.aggregate_receipt.AggregateReceipt, str]:
    """Consume one registration serially, then write its aggregate decision."""
    run_root = Path(run_root)
    budget_contract.require_scientific_budgets(budgets)
    if not registration._digest(registration_sha256):
        raise ValueError("exact F03 registration digest required")
    value = _preflight(run_root, registration_sha256)
    records = cases.preflight_inventory()
    if tuple(item.case.case_id for item in records) != tuple(item.case_id for item in value.cases):
        raise ValueError("F03 registered case order changed")
    _unused(run_root, value)
    _claim(run_root, value, registration_sha256)
    begun = monotonic()
    cutoff = float(dict(value.process_limits)["serial_launch_cutoff_seconds"])
    deadline = begun + cutoff
    try:
        for record in records:
            if monotonic() >= deadline:
                break
            if _preflight(run_root, registration_sha256) != value:
                raise ValueError("F03 registration changed before child launch")
            try:
                _run_one(
                    run_root, value, registration_sha256, record, records, budgets,
                    launch_deadline=deadline, monotonic=monotonic,
                )
            except LaunchCutoff:
                break
    except (KeyboardInterrupt, OSError, ValueError):
        # The supervised child has already been closed by _run_one.  Seal the
        # consumed prefix and explicit missing suffix before propagating the
        # original cancellation or prelaunch/closure failure.  Never resume or
        # launch a replacement case.
        if _preflight(run_root, registration_sha256) != value:
            raise ValueError("F03 registration changed during cancellation closure")
        aggregate.adjudicate_and_write(
            run_root, registration_sha256=registration_sha256, budgets=budgets,
        )
        raise
    if _preflight(run_root, registration_sha256) != value:
        raise ValueError("F03 registration changed before aggregate adjudication")
    return aggregate.adjudicate_and_write(
        run_root, registration_sha256=registration_sha256, budgets=budgets,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-use F03 local scientific coordinator")
    operations = parser.add_subparsers(dest="operation", required=True)
    prepare = operations.add_parser("prepare", help="freeze one new registration; launch nothing")
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    execute = operations.add_parser("run", help="consume one existing registration")
    execute.add_argument("--run-root", type=Path, required=True)
    execute.add_argument("--registration-sha256", required=True)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        _, digest = prepare_registration(args.repository_root, args.run_root)
        print(f"F03 registration SHA-256: {digest}")
    else:
        value, digest = run_registered(
            args.run_root, registration_sha256=args.registration_sha256,
            budgets=budget_contract.make_scientific_budgets(),
        )
        print(f"F03 aggregate: {value.status}; SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
