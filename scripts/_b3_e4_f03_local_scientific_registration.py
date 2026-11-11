"""Immutable registration schema for the bounded F03 scientific run.

This module records identities and contracts only.  It cannot launch a child,
assemble an operator, factor a matrix, solve, adjudicate, or authorize a run.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from scripts import _b3_e4_f03_local_scientific_archive as archive
from scripts import _b3_e4_f03_local_scientific_cases as cases
from scripts import _b3_e4_f03_local_scientific_cross_math as cross_math
from scripts import _b3_e4_f03_local_scientific_evidence_index as evidence_index
from scripts import _b3_e4_f03_local_scientific_lifecycle as lifecycle
from scripts import _b3_e4_f03_local_scientific_supervisor as supervisor


SCHEMA = "f03_local_scientific_registration_v1"
MAX_BYTES = 256 * 1024
MAX_IDENTITY_FILE_BYTES = 256 * 1024 * 1024
DIGEST = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
REGISTRATION_TOKEN = "{registration_sha256}"
ENTRY_MODULE = "scripts._b3_e4_f03_local_scientific_case_entry"

SOURCE_ROLES = (
    ("registration", "scripts/_b3_e4_f03_local_scientific_registration.py"),
    ("coordinator", "scripts/_b3_e4_f03_local_scientific_coordinator.py"),
    ("aggregate", "scripts/_b3_e4_f03_local_scientific_aggregate.py"),
    ("aggregate_receipt", "scripts/_b3_e4_f03_local_scientific_aggregate_receipt.py"),
    ("cross_snapshot", "scripts/_b3_e4_f03_local_scientific_cross_snapshot.py"),
    ("cross_math", "scripts/_b3_e4_f03_local_scientific_cross_math.py"),
    ("cross_bundle", "scripts/_b3_e4_f03_local_scientific_cross_bundle.py"),
    ("case_entry", "scripts/_b3_e4_f03_local_scientific_case_entry.py"),
    ("child_identity", "scripts/_b3_e4_f03_local_scientific_child_identity.py"),
    ("case_producer", "scripts/_b3_e4_f03_local_scientific_case_producer.py"),
    ("case_adjudicator", "scripts/_b3_e4_f03_local_scientific_case_adjudicator.py"),
    ("case_receipt", "scripts/_b3_e4_f03_local_scientific_case_receipt.py"),
    ("supervisor", "scripts/_b3_e4_f03_local_scientific_supervisor.py"),
    ("lifecycle", "scripts/_b3_e4_f03_local_scientific_lifecycle.py"),
    ("archive", "scripts/_b3_e4_f03_local_scientific_archive.py"),
    ("cases", "scripts/_b3_e4_f03_local_scientific_cases.py"),
    ("budgets", "scripts/_b3_e4_f03_local_scientific_budgets.py"),
)
PACKAGE_NAMES = ("ANYfeSolver", "numpy", "psutil", "scipy", "threadpoolctl")
TERMINAL_PRECEDENCE = (
    "FAILED_EVIDENCE",
    "FAILED_EXECUTION",
    "FAILED_NUMERICAL_SCREEN",
    "UNRESOLVED_NUMERICAL_SCREEN",
    "UNRESOLVED_LOCAL_REFERENCE",
    "PASSED_BOUNDED_LOCAL_REFERENCE_SCREEN",
)


def process_contract() -> tuple[tuple[str, str], ...]:
    return (
        ("child_wall_seconds", "1800"),
        ("child_tree_rss_bytes", str(6 * 1024**3)),
        ("supervisor_poll_seconds", "0.1"),
        ("serial_launch_cutoff_seconds", "7200"),
        ("stdout_log_bytes_per_case", str(supervisor.LOG_BYTES_PER_STREAM)),
        ("stderr_log_bytes_per_case", str(supervisor.LOG_BYTES_PER_STREAM)),
        ("log_drain_shutdown_seconds", str(supervisor.LOG_DRAIN_SECONDS)),
        ("raw_archive_bytes_per_case", str(cases.ARCHIVE_BYTES)),
    )


def resource_contract() -> tuple[tuple[str, int], ...]:
    """Case evidence and cross-comparator resource ceilings for one run."""
    return archive.resource_contract() + (
        ("cross_union_intervals", cross_math.MAX_UNION_INTERVALS),
        ("cross_functional_integer_bits", cross_math.MAX_FUNCTIONAL_BITS),
        ("cross_functional_operations", cross_math.MAX_FUNCTIONAL_OPERATIONS),
        ("cross_functional_workspace_bytes", cross_math.MAX_FUNCTIONAL_WORK_BYTES),
        ("cross_matrix_nnz", cross_math.MAX_MATRIX_NNZ),
        ("cross_case_nnz", cross_math.MAX_CASE_NNZ),
        ("cross_product_paths", cross_math.MAX_PRODUCT_PATHS),
        ("cross_matrix_workspace_bytes", cross_math.MAX_MATRIX_WORK_BYTES),
    )


def tolerance_contract() -> tuple[tuple[str, str], ...]:
    """Frozen named numerical and cross-case acceptance limits."""
    return (
        ("spatial_identity_relative", "1e-10"),
        ("integration_block_relative", "1e-11"),
        ("integration_sensitivity_relative", "1e-6"),
        ("saved_state_alternate_relative", "1e-6"),
        ("condition_2_upper", "1e10"),
        ("amplified_backward_upper", "1e-8"),
        ("positivity_scaled_skew", "1e-12"),
        ("state_diagonal_relative", "1e-12"),
        ("state_coordinate_relative", "1e-11"),
        ("saved_action_absolute", "1e-11"),
        ("trace_relative", "1e-11"),
        ("root_rank_relative", "1e-10"),
        ("equilibrium_integration_relative", "1e-11"),
        ("equilibrium_scaled_skew", "1e-12"),
        ("stationarity_scaled", "1e-10"),
        ("root_compatibility_scaled", "1e-10"),
        ("energy_identity_scaled", "1e-10"),
        ("functional_resolution_relative", "1e-10"),
        ("settling_scalar_relative", "1e-2"),
        ("settling_scalar_floor_scale", "1e-3"),
        ("settling_last_change_slack_scale", "1e-5"),
        ("nested_ritz_energy_scale", "1e-10"),
        ("nested_energy_identity_relative", "1e-10"),
        ("settling_functional_relative", "1e-2"),
        ("settling_functional_floor_newtons", "1000"),
        ("settling_functional_last_change_slack_newtons", "10"),
    )


def thread_contract() -> tuple[tuple[str, str], ...]:
    return tuple((name, "1") for name in supervisor.THREADS)


def _digest(value: object) -> bool:
    return type(value) is str and DIGEST.fullmatch(value) is not None


def _commit(value: object) -> bool:
    return type(value) is str and COMMIT.fullmatch(value) is not None


def _relative_path(value: object) -> bool:
    if type(value) is not str or not value or "\\" in value or ":" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute() and path.as_posix() == value
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def _absolute_path(value: object) -> bool:
    return (
        type(value) is str and bool(value) and Path(value).is_absolute()
        and str(Path(value)) == value and ".." not in Path(value).parts
    )


def case_directory(run_root: str, case_id: str) -> Path:
    cases.case_by_id(case_id)
    if not _absolute_path(run_root):
        raise ValueError("absolute F03 run root required")
    return Path(run_root) / "cases" / case_id


def child_argv(runtime: str, run_root: str, case_id: str,
               source_commit: str) -> tuple[str, ...]:
    if not _absolute_path(runtime) or not _commit(source_commit):
        raise ValueError("validated F03 child command identities required")
    return (
        runtime, "-m", ENTRY_MODULE, "--directory",
        str(case_directory(run_root, case_id)), "--source-commit",
        source_commit, "--registration-sha256", REGISTRATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    role: str
    path: str
    byte_count: int
    sha256: str

    def __post_init__(self):
        if (
            type(self.role) is not str
            or not self.role
            or not _relative_path(self.path)
            or type(self.byte_count) is not int
            or not 0 < self.byte_count <= MAX_IDENTITY_FILE_BYTES
            or not _digest(self.sha256)
        ):
            raise ValueError("invalid F03 registered source identity")


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    executable: str
    executable_bytes: int
    executable_sha256: str
    implementation: str
    version: str

    def __post_init__(self):
        if (
            not _absolute_path(self.executable)
            or type(self.executable_bytes) is not int
            or not 0 < self.executable_bytes <= MAX_IDENTITY_FILE_BYTES
            or not _digest(self.executable_sha256)
            or any(type(value) is not str or not value for value in (self.implementation, self.version))
        ):
            raise ValueError("invalid F03 registered runtime identity")


@dataclass(frozen=True, slots=True)
class PackageIdentity:
    name: str
    version: str
    origin: str
    origin_bytes: int
    origin_sha256: str

    def __post_init__(self):
        if (
            type(self.name) is not str
            or not self.name
            or type(self.version) is not str
            or not self.version
            or not _absolute_path(self.origin)
            or type(self.origin_bytes) is not int
            or not 0 < self.origin_bytes <= MAX_IDENTITY_FILE_BYTES
            or not _digest(self.origin_sha256)
        ):
            raise ValueError("invalid F03 registered package identity")


@dataclass(frozen=True, slots=True)
class HardwareIdentity:
    hostname: str
    platform: str
    machine: str
    processor: str
    logical_cpus: int
    physical_memory_bytes: int

    def __post_init__(self):
        if (
            any(type(value) is not str or not value for value in (
                self.hostname, self.platform, self.machine, self.processor,
            ))
            or type(self.logical_cpus) is not int
            or self.logical_cpus <= 0
            or type(self.physical_memory_bytes) is not int
            or self.physical_memory_bytes <= 0
        ):
            raise ValueError("invalid F03 registered hardware identity")


@dataclass(frozen=True, slots=True)
class CaseBinding:
    case_id: str
    topology_sha256: str
    kkt_order: int
    plan_path: str
    plan_sha256: str
    plan_archive_sha256: str

    def __post_init__(self):
        try:
            expected = cases.case_by_id(self.case_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid F03 registered case identity") from exc
        if (
            not _digest(self.topology_sha256)
            or type(self.kkt_order) is not int
            or self.kkt_order != expected.kkt_order
            or not _relative_path(self.plan_path)
            or not _digest(self.plan_sha256)
            or not _digest(self.plan_archive_sha256)
        ):
            raise ValueError("invalid F03 registered case binding")


@dataclass(frozen=True, slots=True)
class CommandBinding:
    case_id: str
    working_directory: str
    argv: tuple[str, ...]
    stdout_path: str
    stderr_path: str

    def __post_init__(self):
        try:
            cases.case_by_id(self.case_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid F03 registered command case") from exc
        if (
            not _absolute_path(self.working_directory)
            or type(self.argv) is not tuple
            or not 1 <= len(self.argv) <= 32
            or any(type(value) is not str or not value or len(value) > 4096 for value in self.argv)
            or not _relative_path(self.stdout_path)
            or not _relative_path(self.stderr_path)
            or self.stdout_path == self.stderr_path
        ):
            raise ValueError("invalid F03 registered child command")


@dataclass(frozen=True, slots=True)
class Registration:
    schema: str
    created_utc: str
    repository_root: str
    run_root: str
    source_commit: str
    accepted_main_commit: str
    clean_source: bool
    inventory_sha256: str
    sources: tuple[SourceIdentity, ...]
    runtime: RuntimeIdentity
    packages: tuple[PackageIdentity, ...]
    hardware: HardwareIdentity
    threads: tuple[tuple[str, str], ...]
    process_limits: tuple[tuple[str, str], ...]
    resource_limits: tuple[tuple[str, int], ...]
    tolerances: tuple[tuple[str, str], ...]
    cases: tuple[CaseBinding, ...]
    commands: tuple[CommandBinding, ...]
    terminal_precedence: tuple[str, ...]

    def __post_init__(self):
        try:
            if type(self.created_utc) is not str or UTC_TIMESTAMP.fullmatch(self.created_utc) is None:
                raise ValueError("noncanonical F03 UTC registration timestamp")
            timestamp = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("invalid F03 registration timestamp") from exc
        expected_cases = tuple(case.case_id for case in cases.CASES)
        if (
            self.schema != SCHEMA
            or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
            or not _absolute_path(self.repository_root)
            or not _absolute_path(self.run_root)
            or self.run_root == self.repository_root
            or not _commit(self.source_commit)
            or not _commit(self.accepted_main_commit)
            or type(self.clean_source) is not bool
            or not self.clean_source
            or not _digest(self.inventory_sha256)
            or type(self.sources) is not tuple
            or any(type(item) is not SourceIdentity for item in self.sources)
            or tuple((item.role, item.path) for item in self.sources) != SOURCE_ROLES
            or type(self.runtime) is not RuntimeIdentity
            or type(self.packages) is not tuple
            or any(type(item) is not PackageIdentity for item in self.packages)
            or tuple(item.name for item in self.packages) != PACKAGE_NAMES
            or type(self.hardware) is not HardwareIdentity
            or self.threads != thread_contract()
            or self.process_limits != process_contract()
            or self.resource_limits != resource_contract()
            or self.tolerances != tolerance_contract()
            or type(self.cases) is not tuple
            or any(type(item) is not CaseBinding for item in self.cases)
            or tuple(item.case_id for item in self.cases) != expected_cases
            or type(self.commands) is not tuple
            or any(type(item) is not CommandBinding for item in self.commands)
            or tuple(item.case_id for item in self.commands) != expected_cases
            or self.terminal_precedence != TERMINAL_PRECEDENCE
            or len({item.plan_path for item in self.cases}) != len(self.cases)
            or len({item.stdout_path for item in self.commands}) != len(self.commands)
            or len({item.stderr_path for item in self.commands}) != len(self.commands)
            or any(
                bound.plan_path != f"cases/{bound.case_id}/plan.zip"
                for bound in self.cases
            )
            or any(
                command.working_directory != self.repository_root
                or command.argv != child_argv(
                    self.runtime.executable, self.run_root,
                    command.case_id, self.source_commit,
                )
                or command.stdout_path != f"cases/{command.case_id}/stdout.log"
                or command.stderr_path != f"cases/{command.case_id}/stderr.log"
                for command in self.commands
            )
        ):
            raise ValueError("invalid complete F03 scientific registration")
        records = cases.preflight_inventory()
        if (
            self.inventory_sha256 != cases.inventory_sha256(records)
            or any(
                bound.topology_sha256 != record.topology_sha256
                or bound.plan_sha256 != lifecycle.plan_sha256(
                    archive.make_case_plan(record, records, self.source_commit)
                )
                for bound, record in zip(self.cases, records)
            )
        ):
            raise ValueError("stale or fabricated F03 registered case inventory")


def _source_payload(value: SourceIdentity) -> dict:
    return {
        "role": value.role, "path": value.path,
        "byte_count": value.byte_count, "sha256": value.sha256,
    }


def _package_payload(value: PackageIdentity) -> dict:
    return {
        "name": value.name, "version": value.version, "origin": value.origin,
        "origin_bytes": value.origin_bytes, "origin_sha256": value.origin_sha256,
    }


def _case_payload(value: CaseBinding) -> dict:
    return {
        "case_id": value.case_id, "topology_sha256": value.topology_sha256,
        "kkt_order": value.kkt_order, "plan_path": value.plan_path,
        "plan_sha256": value.plan_sha256,
        "plan_archive_sha256": value.plan_archive_sha256,
    }


def _command_payload(value: CommandBinding) -> dict:
    return {
        "case_id": value.case_id,
        "working_directory": value.working_directory,
        "argv": list(value.argv),
        "stdout_path": value.stdout_path,
        "stderr_path": value.stderr_path,
    }


def payload(value: Registration) -> bytes:
    if type(value) is not Registration:
        raise ValueError("validated F03 registration required")
    document = {
        "schema": value.schema,
        "created_utc": value.created_utc,
        "repository_root": value.repository_root,
        "run_root": value.run_root,
        "source_commit": value.source_commit,
        "accepted_main_commit": value.accepted_main_commit,
        "clean_source": value.clean_source,
        "inventory_sha256": value.inventory_sha256,
        "sources": [_source_payload(item) for item in value.sources],
        "runtime": {
            "executable": value.runtime.executable,
            "executable_bytes": value.runtime.executable_bytes,
            "executable_sha256": value.runtime.executable_sha256,
            "implementation": value.runtime.implementation,
            "version": value.runtime.version,
        },
        "packages": [_package_payload(item) for item in value.packages],
        "hardware": {
            "hostname": value.hardware.hostname,
            "platform": value.hardware.platform,
            "machine": value.hardware.machine,
            "processor": value.hardware.processor,
            "logical_cpus": value.hardware.logical_cpus,
            "physical_memory_bytes": value.hardware.physical_memory_bytes,
        },
        "threads": {name: setting for name, setting in value.threads},
        "process_limits": {name: setting for name, setting in value.process_limits},
        "resource_limits": {name: limit for name, limit in value.resource_limits},
        "tolerances": {name: limit for name, limit in value.tolerances},
        "cases": [_case_payload(item) for item in value.cases],
        "commands": [_command_payload(item) for item in value.commands],
        "terminal_precedence": list(value.terminal_precedence),
    }
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if len(encoded) > MAX_BYTES:
        raise ValueError("F03 registration exceeds declared bound")
    return encoded


def _exact_dict(value: object, keys: tuple[str, ...], label: str) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"invalid F03 registration {label} inventory")
    return value


def _pairs(value: object, expected: tuple, label: str, value_type: type) -> tuple:
    if type(value) is not dict or set(value) != {name for name, _ in expected}:
        raise ValueError(f"invalid F03 registration {label} contract")
    if any(type(value[name]) is not value_type for name, _ in expected):
        raise ValueError(f"invalid F03 registration {label} value type")
    return tuple((name, value[name]) for name, _ in expected)


def decode(encoded: bytes) -> Registration:
    if type(encoded) is not bytes or len(encoded) > MAX_BYTES:
        raise ValueError("invalid bounded F03 registration payload")
    try:
        value = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid F03 registration JSON") from exc
    keys = (
        "schema", "created_utc", "repository_root", "run_root", "source_commit",
        "accepted_main_commit", "clean_source", "inventory_sha256", "sources",
        "runtime", "packages", "hardware", "threads", "process_limits",
        "resource_limits", "tolerances", "cases", "commands", "terminal_precedence",
    )
    value = _exact_dict(value, keys, "root")
    if any(type(value[name]) is not list for name in ("sources", "packages", "cases", "commands")):
        raise ValueError("invalid F03 registration sequence inventory")
    source_keys = ("role", "path", "byte_count", "sha256")
    sources = tuple(SourceIdentity(**_exact_dict(item, source_keys, "source")) for item in value["sources"])
    runtime_keys = ("executable", "executable_bytes", "executable_sha256", "implementation", "version")
    runtime = RuntimeIdentity(**_exact_dict(value["runtime"], runtime_keys, "runtime"))
    package_keys = ("name", "version", "origin", "origin_bytes", "origin_sha256")
    packages = tuple(PackageIdentity(**_exact_dict(item, package_keys, "package")) for item in value["packages"])
    hardware_keys = (
        "hostname", "platform", "machine", "processor", "logical_cpus", "physical_memory_bytes",
    )
    hardware = HardwareIdentity(**_exact_dict(value["hardware"], hardware_keys, "hardware"))
    case_keys = (
        "case_id", "topology_sha256", "kkt_order", "plan_path", "plan_sha256",
        "plan_archive_sha256",
    )
    bound_cases = tuple(CaseBinding(**_exact_dict(item, case_keys, "case")) for item in value["cases"])
    command_keys = ("case_id", "working_directory", "argv", "stdout_path", "stderr_path")
    commands = tuple(
        CommandBinding(
            **{
                **_exact_dict(item, command_keys, "command"),
                "argv": tuple(item["argv"]) if type(item.get("argv")) is list else (),
            },
        )
        for item in value["commands"]
    )
    if type(value["terminal_precedence"]) is not list:
        raise ValueError("invalid F03 registration terminal precedence inventory")
    result = Registration(
        value["schema"], value["created_utc"], value["repository_root"],
        value["run_root"],
        value["source_commit"], value["accepted_main_commit"], value["clean_source"],
        value["inventory_sha256"], sources, runtime, packages, hardware,
        _pairs(value["threads"], thread_contract(), "thread", str),
        _pairs(value["process_limits"], process_contract(), "process", str),
        _pairs(value["resource_limits"], resource_contract(), "resource", int),
        _pairs(value["tolerances"], tolerance_contract(), "tolerance", str),
        bound_cases, commands, tuple(value["terminal_precedence"]),
    )
    if payload(result) != encoded:
        raise ValueError("noncanonical F03 registration encoding")
    return result


def _owned_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("F03 registration requires an owned directory")
    if hasattr(info, "st_file_attributes") and info.st_file_attributes & 0x400:
        raise ValueError("F03 registration directory reparse point rejected")


def _verify_run_roots(directory: Path, value: Registration) -> None:
    _owned_directory(directory)
    source = Path(value.repository_root)
    _owned_directory(source)
    source_resolved = source.resolve()
    run_resolved = directory.resolve()
    if (
        source_resolved == run_resolved
        or run_resolved.is_relative_to(source_resolved)
        or source_resolved.is_relative_to(run_resolved)
    ):
        raise ValueError("F03 source and run roots must be separate")


def _verify_plan_archives(directory: Path, value: Registration) -> None:
    records = cases.preflight_inventory()
    if cases.inventory_sha256(records) != value.inventory_sha256:
        raise ValueError("stale F03 registration preflight inventory")
    _owned_directory(directory / "cases")
    for bound, record in zip(value.cases, records):
        case_dir = directory / "cases" / bound.case_id
        _owned_directory(case_dir)
        path = case_dir / "plan.zip"
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("F03 registered plan requires an owned regular file")
        if hasattr(info, "st_file_attributes") and info.st_file_attributes & 0x400:
            raise ValueError("F03 registered plan reparse point rejected")
        with evidence_index._open_no_follow(path) as stream:
            encoded = stream.read(archive.MAX_PLAN_BYTES + 1)
        if len(encoded) > archive.MAX_PLAN_BYTES:
            raise ValueError("F03 registered plan exceeds declared bound")
        if hashlib.sha256(encoded).hexdigest() != bound.plan_archive_sha256:
            raise ValueError("F03 registered plan archive digest mismatch")
        plan = archive.read_plan(path, record, records, value.source_commit)
        if lifecycle.plan_sha256(plan) != bound.plan_sha256:
            raise ValueError("F03 registered plan payload digest mismatch")


def _verify_file(path: Path, byte_count: int, digest: str, label: str) -> None:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != byte_count
        or (hasattr(info, "st_file_attributes") and info.st_file_attributes & 0x400)
    ):
        raise ValueError(f"F03 registered {label} is not the owned regular file")
    observed = hashlib.sha256()
    count = 0
    with evidence_index._open_no_follow(path) as stream:
        opened = stream.fileno()
        opened_info = os.fstat(opened)
        if (
            opened_info.st_dev != info.st_dev
            or opened_info.st_ino != info.st_ino
            or opened_info.st_size != info.st_size
        ):
            raise ValueError(f"F03 registered {label} changed during open")
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > byte_count or count > MAX_IDENTITY_FILE_BYTES:
                raise ValueError(f"F03 registered {label} exceeded its byte bound")
            observed.update(block)
    if count != byte_count or not hmac.compare_digest(observed.hexdigest(), digest):
        raise ValueError(f"F03 registered {label} byte identity mismatch")


def _verify_identity_files(value: Registration) -> None:
    root = Path(value.repository_root)
    for item in value.sources:
        path = root.joinpath(*PurePosixPath(item.path).parts)
        parent = path.parent
        while parent != root:
            _owned_directory(parent)
            parent = parent.parent
        _verify_file(path, item.byte_count, item.sha256, f"source {item.role}")
    _verify_file(
        Path(value.runtime.executable), value.runtime.executable_bytes,
        value.runtime.executable_sha256, "runtime",
    )
    for item in value.packages:
        _verify_file(
            Path(item.origin), item.origin_bytes, item.origin_sha256,
            f"package {item.name}",
        )


def write_registration(directory: Path, value: Registration) -> str:
    directory = Path(directory)
    if not directory.is_dir() or str(directory) != value.run_root:
        raise ValueError("registered F03 run directory required")
    _verify_run_roots(directory, value)
    _verify_identity_files(value)
    _verify_plan_archives(directory, value)
    encoded = payload(value)
    digest = hashlib.sha256(encoded).hexdigest()
    if (directory / "registration.json").exists() or (directory / "registration.sha256").exists():
        raise FileExistsError("F03 registration is single-use")
    with (directory / "registration.json").open("xb") as stream:
        stream.write(encoded)
    with (directory / "registration.sha256").open("xb") as stream:
        stream.write((digest + "\n").encode("ascii"))
    return digest


def read_registration(directory: Path, expected_sha256: str | None = None) -> tuple[Registration, str]:
    directory = Path(directory)
    _owned_directory(directory)
    json_path = directory / "registration.json"
    digest_path = directory / "registration.sha256"
    for path in (json_path, digest_path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("F03 registration requires owned regular files")
        if hasattr(info, "st_file_attributes") and info.st_file_attributes & 0x400:
            raise ValueError("F03 registration reparse point rejected")
    with evidence_index._open_no_follow(json_path) as stream:
        encoded = stream.read(MAX_BYTES + 1)
    if len(encoded) > MAX_BYTES:
        raise ValueError("F03 registration exceeds declared bound")
    digest = hashlib.sha256(encoded).hexdigest()
    with evidence_index._open_no_follow(digest_path) as stream:
        claimed = stream.read(66)
    if (
        claimed != (digest + "\n").encode("ascii")
        or (expected_sha256 is not None and (
            not _digest(expected_sha256) or not hmac.compare_digest(digest, expected_sha256)
        ))
    ):
        raise ValueError("F03 registration digest mismatch")
    value = decode(encoded)
    if str(directory) != value.run_root:
        raise ValueError("F03 registration run directory mismatch")
    _verify_run_roots(directory, value)
    _verify_identity_files(value)
    _verify_plan_archives(directory, value)
    return value, digest
