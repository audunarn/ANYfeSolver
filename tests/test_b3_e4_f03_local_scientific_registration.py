"""Bounded registration-contract tests; no scientific case is run."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import _b3_e4_f03_local_scientific_archive as archive
from scripts import _b3_e4_f03_local_scientific_cases as cases
from scripts import _b3_e4_f03_local_scientific_lifecycle as lifecycle
from scripts import _b3_e4_f03_local_scientific_registration as registration
from scripts import _b3_e4_f03_local_scientific_supervisor as supervisor
from scripts import _b3_e4_f03_local_functional as functional
from scripts import _b3_e4_f03_sparse_equilibrium_checks as equilibrium
from scripts import _b3_e4_f03_sparse_integration_checks as integration
from scripts import _b3_e4_f03_sparse_positivity as positivity
from scripts import _b3_e4_f03_sparse_root_rank as root_rank
from scripts import _b3_e4_f03_sparse_state_binding as state_binding
from scripts import _b3_e4_f03_sparse_trace_binding as trace
from scripts import _b3_e4_f03_spatial_contract as spatial


SOURCE = "a" * 40
DIGEST = "b" * 64


def _registration(root: Path):
    repository = str(root / "repository")
    run_root = str(root / "registration")
    Path(repository).mkdir()
    Path(run_root).mkdir()
    runtime_path = str(root / "python.exe")
    source_identities = []
    for role, source_path in registration.SOURCE_ROLES:
        path = Path(repository).joinpath(*source_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = f"synthetic source {role}\n".encode("ascii")
        path.write_bytes(encoded)
        source_identities.append(registration.SourceIdentity(
            role, source_path, len(encoded), hashlib.sha256(encoded).hexdigest(),
        ))
    runtime_bytes = b"synthetic python executable\n"
    Path(runtime_path).write_bytes(runtime_bytes)
    package_identities = []
    for name in registration.PACKAGE_NAMES:
        path = root / f"{name}.py"
        encoded = f"synthetic package {name}\n".encode("ascii")
        path.write_bytes(encoded)
        package_identities.append(registration.PackageIdentity(
            name, "1.0", str(path), len(encoded), hashlib.sha256(encoded).hexdigest(),
        ))
    records = cases.preflight_inventory()
    for record in records:
        case_dir = Path(run_root) / "cases" / record.case.case_id
        case_dir.mkdir(parents=True)
        archive.write_plan(
            case_dir / "plan.zip",
            archive.make_case_plan(record, records, SOURCE),
        )
    bindings = tuple(
        registration.CaseBinding(
            record.case.case_id,
            record.topology_sha256,
            record.case.kkt_order,
            f"cases/{record.case.case_id}/plan.zip",
            lifecycle.plan_sha256(archive.make_case_plan(record, records, SOURCE)),
            hashlib.sha256((Path(run_root) / "cases" / record.case.case_id / "plan.zip").read_bytes()).hexdigest(),
        )
        for record in records
    )
    commands = tuple(
        registration.CommandBinding(
            bound.case_id, repository,
            registration.child_argv(runtime_path, run_root, bound.case_id, SOURCE),
            f"cases/{bound.case_id}/stdout.log",
            f"cases/{bound.case_id}/stderr.log",
        )
        for bound in bindings
    )
    return registration.Registration(
        registration.SCHEMA, "2026-09-23T10:00:00Z", repository, run_root,
        SOURCE, "c" * 40, True, cases.inventory_sha256(records),
        tuple(source_identities),
        registration.RuntimeIdentity(
            runtime_path, len(runtime_bytes), hashlib.sha256(runtime_bytes).hexdigest(),
            "CPython", "3.14.0",
        ),
        tuple(package_identities),
        registration.HardwareIdentity("host", "Windows", "AMD64", "CPU", 8, 16 * 1024**3),
        registration.thread_contract(), registration.process_contract(),
        registration.resource_contract(), registration.tolerance_contract(),
        bindings, commands, registration.TERMINAL_PRECEDENCE,
    )


def _mutate(encoded: bytes, field: str, value):
    raw = json.loads(encoded)
    raw[field] = value
    return (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def test_registration_roundtrip_is_canonical_and_exclusive(tmp_path):
    value = _registration(tmp_path)
    directory = tmp_path / "registration"
    digest = registration.write_registration(directory, value)
    assert digest == hashlib.sha256(registration.payload(value)).hexdigest()
    assert registration.read_registration(directory, digest) == (value, digest)
    with pytest.raises(FileExistsError, match="single-use"):
        registration.write_registration(directory, value)
    assert tuple(item.case_id for item in value.cases) == tuple(
        item.case_id for item in cases.CASES
    )


def test_registration_reader_rejects_shared_cases_junction(tmp_path):
    value = _registration(tmp_path)
    run_root = tmp_path / "registration"
    digest = registration.write_registration(run_root, value)
    cases_dir = run_root / "cases"
    relocated = tmp_path / "relocated-cases"
    cases_dir.rename(relocated)
    try:
        os.symlink(relocated, cases_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="owned directory"):
        registration.read_registration(run_root, digest)


def test_registered_limits_match_current_executable_checks():
    process = dict(registration.process_contract())
    tolerances = dict(registration.tolerance_contract())
    assert float(process["child_wall_seconds"]) == supervisor.SCIENTIFIC_LIMITS.seconds
    assert int(process["child_tree_rss_bytes"]) == supervisor.SCIENTIFIC_LIMITS.rss_bytes
    assert float(process["supervisor_poll_seconds"]) == supervisor.SCIENTIFIC_LIMITS.poll_seconds
    assert float(tolerances["spatial_identity_relative"]) == spatial.IDENTITY_LIMIT
    assert float(tolerances["integration_block_relative"]) == float(integration.BLOCK_LIMIT)
    assert float(tolerances["integration_sensitivity_relative"]) == float(integration.PERTURBATION_LIMIT)
    assert float(tolerances["positivity_scaled_skew"]) == float(positivity.SKEW_LIMIT)
    assert float(tolerances["state_diagonal_relative"]) == float(state_binding.D_LIMIT)
    assert float(tolerances["state_coordinate_relative"]) == float(state_binding.STATE_LIMIT)
    assert float(tolerances["saved_action_absolute"]) == float(state_binding.ACTION_LIMIT)
    assert float(tolerances["trace_relative"]) == float(trace.LIMIT)
    assert float(tolerances["root_rank_relative"]) == float(root_rank.RATIO)
    assert float(tolerances["equilibrium_integration_relative"]) == float(equilibrium.INTEGRATION_LIMIT)
    assert float(tolerances["equilibrium_scaled_skew"]) == float(equilibrium.SKEW_LIMIT)
    assert float(tolerances["stationarity_scaled"]) == float(equilibrium.STATIONARITY_LIMIT)
    assert float(tolerances["root_compatibility_scaled"]) == float(equilibrium.ROOT_LIMIT)
    assert float(tolerances["energy_identity_scaled"]) == float(equilibrium.ENERGY_LIMIT)
    assert float(tolerances["functional_resolution_relative"]) == functional.RESOLUTION_LIMIT


@pytest.mark.parametrize("stamp", (
    "2026-09-23T10:00:00+00:00", "2026-09-23T10:00:00.000Z",
    "2026-09-23T10:00:00ZZ", "2026-09-31T10:00:00Z",
))
def test_registration_rejects_noncanonical_or_invalid_utc_stamp(tmp_path, stamp):
    value = _registration(tmp_path)
    with pytest.raises(ValueError, match="timestamp"):
        replace(value, created_utc=stamp)


def test_registration_rejects_redirected_child_command_and_case_path(tmp_path):
    value = _registration(tmp_path)
    first = value.commands[0]
    altered = replace(first, argv=(*first.argv[:-1], "0" * 64))
    with pytest.raises(ValueError, match="registration"):
        replace(value, commands=(altered, *value.commands[1:]))
    altered = replace(first, working_directory=str(tmp_path))
    with pytest.raises(ValueError, match="registration"):
        replace(value, commands=(altered, *value.commands[1:]))
    bound = replace(value.cases[0], plan_path="outside/plan.zip")
    with pytest.raises(ValueError, match="registration"):
        replace(value, cases=(bound, *value.cases[1:]))


def test_registration_rejects_fabricated_inventory_topology_and_plan_identity(tmp_path):
    value = _registration(tmp_path)
    with pytest.raises(ValueError, match="stale or fabricated"):
        replace(value, inventory_sha256=DIGEST)
    bound = replace(value.cases[0], topology_sha256=DIGEST)
    with pytest.raises(ValueError, match="stale or fabricated"):
        replace(value, cases=(bound, *value.cases[1:]))
    bound = replace(value.cases[0], plan_sha256=DIGEST)
    with pytest.raises(ValueError, match="stale or fabricated"):
        replace(value, cases=(bound, *value.cases[1:]))


def test_registration_requires_identical_owned_plan_archives_at_write_and_read(tmp_path):
    value = _registration(tmp_path)
    directory = tmp_path / "registration"
    plan_path = directory / value.cases[0].plan_path
    original = plan_path.read_bytes()
    plan_path.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="archive digest"):
        registration.write_registration(directory, value)
    assert not (directory / "registration.json").exists()
    plan_path.write_bytes(original)
    registration.write_registration(directory, value)
    plan_path.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="archive digest"):
        registration.read_registration(directory)


def test_registration_verifies_source_runtime_and_package_bytes(tmp_path):
    value = _registration(tmp_path)
    directory = tmp_path / "registration"
    fabricated = replace(value.sources[0], sha256=DIGEST)
    changed = replace(value, sources=(fabricated, *value.sources[1:]))
    with pytest.raises(ValueError, match="source registration byte identity"):
        registration.write_registration(directory, changed)
    assert not (directory / "registration.json").exists()
    registration.write_registration(directory, value)
    runtime_path = Path(value.runtime.executable)
    runtime_path.write_bytes(b"replaced runtime with same length"[:value.runtime.executable_bytes].ljust(
        value.runtime.executable_bytes, b"x"
    ))
    with pytest.raises(ValueError, match="runtime byte identity"):
        registration.read_registration(directory)
    runtime_path.write_bytes(b"synthetic python executable\n")
    package_path = Path(value.packages[0].origin)
    package_path.write_bytes(b"x" * value.packages[0].origin_bytes)
    with pytest.raises(ValueError, match="package ANYfeSolver byte identity"):
        registration.read_registration(directory)


def test_registration_rejects_hardlinked_source_file(tmp_path):
    value = _registration(tmp_path)
    source = Path(value.repository_root) / value.sources[0].path
    twin = tmp_path / "source-hardlink"
    try:
        twin.hardlink_to(source)
    except OSError as exc:
        pytest.skip(f"hardlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="owned regular file"):
        registration.write_registration(tmp_path / "registration", value)


def test_registration_rejects_filesystem_alias_of_source_as_run_root(tmp_path):
    value = _registration(tmp_path)
    alias = value.run_root.upper()
    try:
        aliased = alias != value.run_root and os.path.samefile(alias, value.run_root)
    except OSError:
        aliased = False
    if not aliased:
        pytest.skip("no case-insensitive filesystem alias of the run root available")
    changed = replace(
        value, repository_root=alias,
        commands=tuple(replace(command, working_directory=alias)
                       for command in value.commands),
    )
    with pytest.raises(ValueError, match="roots must be separate"):
        registration.write_registration(tmp_path / "registration", changed)


def test_registration_requires_run_archive_outside_source_checkout(tmp_path):
    value = _registration(tmp_path)
    nested = Path(value.repository_root) / "run"
    nested.mkdir()
    changed = replace(
        value, run_root=str(nested),
        commands=tuple(replace(command, argv=registration.child_argv(
            value.runtime.executable, str(nested), command.case_id,
            value.source_commit,
        )) for command in value.commands),
    )
    with pytest.raises(ValueError, match="roots must be separate"):
        registration.write_registration(nested, changed)


@pytest.mark.parametrize("path", (
    "cases//Q1_L0/plan.zip", "cases/../Q1_L0/plan.zip",
    "C:cases/Q1_L0/plan.zip", "cases\\Q1_L0\\plan.zip",
))
def test_registration_rejects_noncanonical_relative_paths(path):
    assert not registration._relative_path(path)


def test_registration_rejects_type_coercion_and_noncanonical_json(tmp_path):
    value = _registration(tmp_path)
    encoded = registration.payload(value)
    raw = json.loads(encoded)
    raw["resource_limits"][next(iter(raw["resource_limits"]))] = True
    with pytest.raises(ValueError, match="value type"):
        registration.decode(_mutate(encoded, "resource_limits", raw["resource_limits"]))
    raw = json.loads(encoded)
    raw["threads"][next(iter(raw["threads"]))] = 1
    with pytest.raises(ValueError, match="value type"):
        registration.decode(_mutate(encoded, "threads", raw["threads"]))
    with pytest.raises(ValueError, match="noncanonical"):
        registration.decode(json.dumps(json.loads(encoded)).encode("ascii"))
    with pytest.raises(ValueError, match="bounded"):
        registration.decode(b"x" * (registration.MAX_BYTES + 1))


def test_registration_rejects_changed_digest_and_partial_sidecar(tmp_path):
    value = _registration(tmp_path)
    directory = tmp_path / "registration"
    digest = registration.write_registration(directory, value)
    with pytest.raises(ValueError, match="digest"):
        registration.read_registration(directory, "0" * 64)
    (directory / "registration.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="digest"):
        registration.read_registration(directory, digest)
    (directory / "registration.sha256").unlink()
    with pytest.raises(FileNotFoundError):
        registration.read_registration(directory)
    with pytest.raises(FileExistsError, match="single-use"):
        registration.write_registration(directory, value)


def test_registration_cannot_be_relocated_or_point_child_at_source_checkout(tmp_path):
    value = _registration(tmp_path)
    wrong = tmp_path / "other"
    wrong.mkdir()
    with pytest.raises(ValueError, match="run directory"):
        registration.write_registration(wrong, value)
    directory = tmp_path / "registration"
    registration.write_registration(directory, value)
    for name in ("registration.json", "registration.sha256"):
        (wrong / name).write_bytes((directory / name).read_bytes())
    with pytest.raises(ValueError, match="run directory"):
        registration.read_registration(wrong)
    wrong_argv = registration.child_argv(
        value.runtime.executable, value.repository_root, value.cases[0].case_id,
        value.source_commit,
    )
    changed = replace(value.commands[0], argv=wrong_argv)
    with pytest.raises(ValueError, match="registration"):
        replace(value, commands=(changed, *value.commands[1:]))


def test_registration_reader_rejects_symlinked_identity(tmp_path):
    value = _registration(tmp_path)
    directory = tmp_path / "registration"
    registration.write_registration(directory, value)
    other = tmp_path / "other.json"
    other.write_bytes((directory / "registration.json").read_bytes())
    (directory / "registration.json").unlink()
    try:
        (directory / "registration.json").symlink_to(other)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="regular files|reparse"):
        registration.read_registration(directory)
