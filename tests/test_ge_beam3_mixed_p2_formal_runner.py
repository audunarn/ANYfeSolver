"""Tests for the sealed, one-time GE Beam3 P2 formal runner."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/reference_cases/ge_beam3_mixed_p2_formal_runner.py"


def _load_runner():
    name = f"ge_beam3_mixed_p2_formal_runner_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


runner = _load_runner()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-c", f"safe.directory={repository.resolve()}", *arguments),
        cwd=repository,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("GIT_")
            },
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _init_repository(root: Path, files: dict[str, bytes]) -> tuple[Path, str, str]:
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.name", "P2 Runner Test")
    _git(root, "config", "user.email", "p2-runner@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "test: freeze materialization inputs")
    return root, _git(root, "rev-parse", "HEAD"), _git(
        root, "show", "-s", "--format=%T", "HEAD"
    )


def _repository(tmp_path: Path, name: str = "repository") -> tuple[Path, str, str]:
    return _init_repository(
        tmp_path / name,
        {
            ".gitattributes": b"registered.txt text eol=lf\n",
            "registered.txt": b"alpha\nbeta\n",
        },
    )


def _dependency_repository(tmp_path: Path, name: str = "ANYfileIO") -> Path:
    repository, _commit, _tree = _init_repository(
        tmp_path / name,
        {
            ".gitattributes": b"*.py text eol=lf\n",
            "src/anyfileio/__init__.py": b"VALUE = 17\n",
            "src/anyfileio/codec.py": b"CODEC = 'bound'\n",
        },
    )
    return repository


def _registry(tmp_path: Path, name: str = "registry") -> tuple[Path, object]:
    root = tmp_path / name
    root.mkdir()
    marker = {
        "registry_id": "A" * 64,
        "schema": runner.REGISTRY_IDENTITY_SCHEMA,
        "study_id": runner.STUDY_ID,
        "terminal": "REGISTERED_GE_BEAM3_P2_EXECUTION_REGISTRY",
    }
    (root / runner.REGISTRY_IDENTITY_NAME).write_bytes(runner.canonical_bytes(marker))
    return root, runner.validate_registry_root(root)


def _file_row(path: str, index: int = 1) -> dict[str, object]:
    return {
        "bytes": index,
        "git_blob_oid": format(index, "040x"),
        "mode": "100644",
        "path": path,
        "sha256": format(index, "064X"),
    }


def _fake_runtime() -> object:
    return runner.RuntimeIdentity(
        bootstrap_sha256="0" * 64,
        distribution_seeds=("numpy", "scipy", "pytest"),
        distributions=(
            {
                "distribution": "NumPy",
                "distribution_root_sha256": "1" * 64,
                "file_count": 1,
                "file_manifest_sha256": "2" * 64,
                "module": "numpy",
                "origin_bytes": 1,
                "origin_location_sha256": "3" * 64,
                "origin_sha256": "4" * 64,
                "record_sha256": "5" * 64,
                "total_bytes": 1,
                "version": "test",
            },
        ),
        interpreter={
            "abi": "test-abi",
            "bytes": 1,
            "cache_tag": "test-tag",
            "executable_location_sha256": "6" * 64,
            "executable_name": "python.exe",
            "implementation": "CPython",
            "sha256": "7" * 64,
            "version": "3.test",
        },
        pytest_transitive_closure=("Pytest",),
    )


def _snapshot(
    *,
    anyfileio: object | None = None,
    registry: object | None = None,
    runtime: object | None = None,
    harness_commit: str = "A" * 40,
    harness_tree: str = "B" * 40,
    solver_files: tuple[dict[str, object], ...] | None = None,
) -> object:
    dependency_rows = (_file_row("src/anyfileio/__init__.py"),)
    dependency = anyfileio or runner.DependencyIdentity(
        commit="D" * 40,
        file_count=1,
        files=dependency_rows,
        manifest_sha256=runner._sha256(runner.canonical_bytes(list(dependency_rows))),
        total_bytes=1,
        tree="F" * 40,
        tree_files=dependency_rows,
    )
    registry_identity = registry or runner.RegistryIdentity(
        identity_sha256="8" * 64,
        registry_id="9" * 64,
        root_sha256="A" * 64,
    )
    return runner.RepositorySnapshot(
        anyfileio=dependency,
        harness_commit=harness_commit,
        harness_tree=harness_tree,
        manifest_sha256="C" * 64,
        producer_input_bindings={
            Path(path).name: {
                "bytes": index + 1,
                "sha256": format(index + 101, "064X")[-64:],
            }
            for index, path in enumerate(runner.PRODUCER_BOUND_RELATIVES)
        },
        registry=registry_identity,
        runtime=runtime or _fake_runtime(),
        solver_files=solver_files or (_file_row("registered.txt", 2),),
        test_bindings={
            spec.test_id: {
                "git_blob_oid": format(index + 1, "040x"),
                "sha256": format(index + 1, "064X")[-64:],
            }
            for index, spec in enumerate(runner.FOCUSED_TESTS)
        },
    )


def _passing_cycle() -> dict[str, object]:
    rows = [
        {
            "collected_count": index + 1,
            "finding_group": "NONE",
            "node_ids_sha256": format(index + 17, "064X")[-64:],
            "outcome": "PASS",
            "passed": True,
            "record_sha256": format(index + 31, "064X")[-64:],
            "test_id": spec.test_id,
        }
        for index, spec in enumerate(runner.FOCUSED_TESTS)
    ]
    return {
        "collected_count": sum(row["collected_count"] for row in rows),
        "failed_test_file_count": 0,
        "passed_test_file_count": len(rows),
        "scientific": {
            "checker_replica_count": 2,
            "checker_replicas_byte_identical": True,
            "checker_sha256": "8" * 64,
            "finding_groups": [],
            "outcome": "PASS",
            "passed": True,
            "proof": {
                "bytes": 1000,
                "content_sha256": "9" * 64,
                "proof_sha256": "A" * 64,
                "raw_record_count": 21,
            },
            "raw_record_count": 21,
        },
        "test_file_count": len(rows),
        "tests": rows,
    }


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_bytes(value))


def _authority_checks(
    tmp_path: Path, snapshot: object
) -> tuple[tuple[Path, Path], str]:
    paths = (tmp_path / "authority-check-1.json", tmp_path / "authority-check-2.json")
    payload = runner.canonical_bytes(runner.expected_authority_check(snapshot))
    for path in paths:
        path.write_bytes(payload)
    return paths, runner._sha256(payload)


def _validated_authorization() -> object:
    return runner.ValidatedAuthorization(
        attempt_id="b" * 32,
        authority_check_sha256="A" * 64,
        binding={"bytes": 100, "git_blob_oid": "3" * 40, "sha256": "4" * 64},
        commit="5" * 40,
        command={"argv": ["sealed"], "sha256": "6" * 64},
        request_id="a" * 32,
        review={
            "bytes": 100,
            "git_blob_oid": "7" * 40,
            "sha256": "8" * 64,
            "verdict": runner.EXECUTION_REVIEW_VERDICT,
        },
        tree="9" * 40,
    )


def _make_rehearsal(path: Path, snapshot: object, check_sha256: str) -> dict[str, object]:
    cycle = runner._decorate_cycle(_passing_cycle(), snapshot)
    record = runner._aggregate(
        mode=runner.REHEARSAL_MODE,
        snapshot=snapshot,
        cycle=cycle,
        cycle_hashes=[runner._sha256(runner.canonical_bytes(cycle))],
        cycles_byte_identical=True,
        authority_check_sha256=check_sha256,
        authorization=None,
        materialization_sha256="B" * 64,
    )
    _write_canonical(path, record)
    return record


def test_frozen_scope_bounds_terminals_and_formal_api_are_exact() -> None:
    assert runner.FROZEN_BOUNDS.authority_record() == {
        "child_wall_seconds": 600,
        "complete_wave_wall_seconds": 1800,
        "inactivity_seconds": 300,
        "maximum_concurrent_workers": 3,
        "memory_limit_gib_per_process_tree": 24,
        "no_automatic_retry": True,
        "numerical_library_threads_per_worker": 1,
        "required_formal_cycle_count": 2,
    }
    assert [spec.test_id for spec in runner.FOCUSED_TESTS] == [
        "PREREGISTRATION", "RECOVERY_CORRECTION", "SOLVER_CHART", "STATE_RESTART",
        "LOAD_MASS_RECOVERY", "MODAL_BUCKLING", "ACCEPTED_STATIC_CORE",
    ]
    assert runner.TERMINAL_PRECEDENCE[-1] == "PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY"
    parameters = inspect.signature(runner.run).parameters
    assert "cycle_runner" not in parameters
    assert "bounds" not in parameters
    assert "authorization_path" not in parameters
    assert "rehearsal_aggregate_path" in parameters
    assert runner.AUTHORIZATION_PATHS == tuple(
        sorted((runner.AUTHORITY_RELATIVE_PATH, runner.EXECUTION_REVIEW_RELATIVE_PATH))
    )


def test_nonclassifying_test_double_cannot_emit_formal_schema_or_terminal() -> None:
    first = runner.run_nonclassifying_test_double(_passing_cycle)
    assert first == runner.run_nonclassifying_test_double(_passing_cycle)
    assert first["schema"] == runner.TEST_DOUBLE_SCHEMA
    assert first["terminal"] == "NONCLASSIFYING_GE_BEAM3_P2_TEST_DOUBLE_ONLY"
    assert first["schema"] != runner.AGGREGATE_SCHEMA
    assert first["terminal"] not in runner.TERMINAL_PRECEDENCE


def test_formal_cli_attestation_rejects_programmatic_or_changed_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = {"argv": ["python", "runner", "--formal"], "sha256": "A" * 64}
    monkeypatch.setattr(runner, "_ACTIVE_CLI_ARGV", None)
    with pytest.raises(runner.AuthorizationError, match="exact authorized CLI"):
        runner._attest_formal_cli(command)
    monkeypatch.setattr(runner, "_ACTIVE_CLI_ARGV", tuple(command["argv"]))
    runner._attest_formal_cli(command)
    command["argv"].append("--changed")
    with pytest.raises(runner.AuthorizationError, match="exact authorized CLI"):
        runner._attest_formal_cli(command)

    absolute_runner = str(RUNNER.resolve())
    monkeypatch.setattr(sys, "argv", ["relative-runner.py", "--formal"])
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [str(Path(sys.executable).resolve()), "relative-runner.py", "--formal"],
    )
    captured = runner._actual_cli_argv(("--formal",))
    assert captured[1] == "relative-runner.py"
    command = {
        "argv": [str(Path(sys.executable).resolve()), absolute_runner, "--formal"],
        "sha256": "B" * 64,
    }
    monkeypatch.setattr(runner, "_ACTIVE_CLI_ARGV", captured)
    with pytest.raises(runner.AuthorizationError, match="exact authorized CLI"):
        runner._attest_formal_cli(command)
    monkeypatch.setattr(sys, "argv", [absolute_runner, "--formal"])
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [str(Path(sys.executable).resolve()), absolute_runner, "--formal"],
    )
    monkeypatch.setattr(
        runner, "_ACTIVE_CLI_ARGV", runner._actual_cli_argv(("--formal",))
    )
    runner._attest_formal_cli(command)
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [str(Path(sys.executable).resolve()), "-O", absolute_runner, "--formal"],
    )
    with pytest.raises(runner.AuthorizationError, match="interpreter switches"):
        runner._actual_cli_argv(("--formal",))


def test_strict_canonical_json_rejects_duplicate_nonfinite_and_noncanonical(tmp_path: Path) -> None:
    for name, payload in (
        ("duplicate.json", b'{"a":1,"a":2}\n'),
        ("nonfinite.json", b'{"a":NaN}\n'),
        ("spaced.json", b'{"a": 1}\n'),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        with pytest.raises(runner.AuthorizationError):
            runner.strict_canonical_json(path)


def test_git_blob_hashing_uses_committed_bytes_and_dirty_tree_fails_closed(tmp_path: Path) -> None:
    repository, commit, tree = _repository(tmp_path)
    binding = runner.git_blob_binding(repository, commit, "registered.txt")
    assert binding["sha256"] == runner._sha256(b"alpha\nbeta\n")
    (repository / "registered.txt").write_bytes(b"alpha\r\nbeta\r\n")
    assert runner.git_blob_binding(repository, commit, "registered.txt") == binding
    with pytest.raises(runner.BaselineError, match="not clean"):
        runner._require_clean(repository)
    runner._verify_commit(repository, commit, tree)
    with pytest.raises(runner.BaselineError, match="tree mismatch"):
        runner._verify_commit(repository, commit, "0" * 40)


def test_anyfileio_binding_is_full_tree_exact_path_free_and_clean(tmp_path: Path) -> None:
    repository = _dependency_repository(tmp_path)
    identity = runner.validate_anyfileio_repository(repository)
    assert identity.commit == _git(repository, "rev-parse", "HEAD")
    assert identity.file_count == 2
    assert len(identity.tree_files) == 3
    record = identity.record()
    assert set(record) == {
        "commit", "file_count", "manifest_sha256", "total_bytes", "tree",
        "tree_file_count", "tree_manifest_sha256", "tree_total_bytes",
    }
    assert "\\" not in json.dumps(record) and "/" not in json.dumps(record)


def test_anyfileio_missing_dirty_and_assume_unchanged_mutation_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(runner.AuthorizationError, match="explicit ANYfileIO"):
        runner.validate_anyfileio_repository(None)
    dirty = _dependency_repository(tmp_path, "dirty")
    (dirty / "src/anyfileio/codec.py").write_bytes(b"CODEC = 'dirty'\n")
    with pytest.raises(runner.BaselineError, match="not clean"):
        runner.validate_anyfileio_repository(dirty)
    hidden = _dependency_repository(tmp_path, "hidden")
    relative = "src/anyfileio/codec.py"
    _git(hidden, "update-index", "--assume-unchanged", relative)
    (hidden / relative).write_bytes(b"CODEC = 'hidden'\n")
    assert _git(hidden, "status", "--porcelain=v1", "--untracked-files=all") == ""
    with pytest.raises(runner.BaselineError, match="differs from its Git blob"):
        runner.validate_anyfileio_repository(hidden)


def test_registry_is_explicit_external_canonical_and_identity_bound(tmp_path: Path) -> None:
    with pytest.raises(runner.AuthorizationError, match="explicit external registry"):
        runner.validate_registry_root(None)
    root, identity = _registry(tmp_path)
    assert identity.registry_id == "A" * 64
    assert identity.identity_sha256 == runner._sha256((root / runner.REGISTRY_IDENTITY_NAME).read_bytes())
    (root / runner.REGISTRY_IDENTITY_NAME).write_bytes(b'{"schema":"wrong"}\n')
    with pytest.raises(runner.AuthorizationError, match="malformed"):
        runner.validate_registry_root(root)


def test_runtime_identity_binds_interpreter_numpy_scipy_pytest_and_records() -> None:
    identity = runner.runtime_identity()
    record = identity.record()
    assert record["interpreter"]["sha256"] == runner._sha256(Path(sys.executable).resolve().read_bytes())
    distributions = {row["distribution"]: row for row in record["distributions"]}
    assert {
        "ANYgeometry", "ANYmaterial", "ANYmesher", "NumPy", "PyYAML",
        "Pytest", "SciPy", "charset-normalizer", "llvmlite", "numba",
        "threadpoolctl",
    } <= set(distributions)
    if sys.platform == "win32":
        assert "pywin32" in distributions
    closure = {
        runner.canonicalize_name(value)
        for value in record["pytest_transitive_closure"]
    }
    assert {"pytest", "iniconfig", "packaging", "pluggy", "pygments"} <= closure
    assert ("colorama" in closure) is (sys.platform == "win32")
    assert record["bootstrap_sha256"] == runner._sha256(
        runner.ISOLATED_BOOTSTRAP.encode("utf-8")
    )
    for row in distributions.values():
        assert row["file_count"] > 0 and row["total_bytes"] > 0
        assert len(row["file_manifest_sha256"]) == 64
        assert row["import_file_count"] > 0 and row["import_total_bytes"] > 0
        assert len(row["import_manifest_sha256"]) == 64
        assert len(row["origin_relative_sha256"]) == 64
        assert len(row["record_sha256"]) == 64
        assert len(row["origin_sha256"]) == 64


def test_missing_dependency_or_registry_refuses_before_side_effects(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output, work = tmp_path / "aggregate.json", tmp_path / "work"
    with pytest.raises(runner.AuthorizationError, match="explicit ANYfileIO"):
        runner.run(mode=runner.REHEARSAL_MODE, output_path=output, work_root=work, repository=repository)
    with pytest.raises(runner.AuthorizationError, match="external registry"):
        runner.run(
            mode=runner.REHEARSAL_MODE, output_path=output, work_root=work,
            repository=repository, anyfileio_repository=tmp_path / "dependency",
        )
    assert not output.exists() and not work.exists()


@pytest.mark.parametrize(
    ("groups", "terminal"),
    [
        (["PROCESS_OR_EVIDENCE", "BASELINE_OR_AUTHORITY"], runner.TERMINAL_PRECEDENCE[0]),
        (["PROCESS_OR_EVIDENCE", "SOLVER_CHART_OR_STATE"], runner.TERMINAL_PRECEDENCE[1]),
        (["SOLVER_CHART_OR_STATE", "LOAD_MASS_OR_RECOVERY"], runner.TERMINAL_PRECEDENCE[2]),
        (["LOAD_MASS_OR_RECOVERY", "MODAL_OR_BUCKLING"], runner.TERMINAL_PRECEDENCE[3]),
        (["MODAL_OR_BUCKLING"], runner.TERMINAL_PRECEDENCE[4]),
        ([], runner.TERMINAL_PRECEDENCE[5]),
    ],
)
def test_terminal_precedence(groups: list[str], terminal: str) -> None:
    records = [{"finding_group": group, "passed": False} for group in groups]
    assert runner.adjudicate(records) == terminal


def test_postclaim_authority_failures_keep_terminal_precedence() -> None:
    for exception in (
        runner.BaselineError("changed"),
        runner.AuthorizationError("invalid"),
    ):
        group = runner._postclaim_failure_group(exception)
        assert group == "BASELINE_OR_AUTHORITY"
        assert runner.adjudicate_cycle(
            runner._blocked_cycle(type(exception).__name__.upper(), group)
        ) == runner.TERMINAL_PRECEDENCE[0]
    assert runner._postclaim_failure_group(RuntimeError("crash")) == "PROCESS_OR_EVIDENCE"


def test_classifying_cycle_requires_two_identical_strict_records() -> None:
    snapshot = _snapshot()
    assert [
        runner._required_cycle_count(runner.FORMAL_MODE, terminal)
        for terminal in runner.TERMINAL_PRECEDENCE
    ] == [1, 1, 2, 2, 2, 2]
    assert runner._required_cycle_count(
        runner.REHEARSAL_MODE, runner.TERMINAL_PRECEDENCE[-1]
    ) == 1
    passing = runner._decorate_cycle(_passing_cycle(), snapshot)
    assert runner.validate_cycle_record(passing, snapshot) == runner.TERMINAL_PRECEDENCE[-1]
    proof_hash = runner._sha256(runner.canonical_bytes(passing))
    accepted = runner._aggregate(
        mode=runner.FORMAL_MODE, snapshot=snapshot, cycle=passing,
        cycle_hashes=[proof_hash, proof_hash], cycles_byte_identical=True,
        authority_check_sha256="A" * 64,
        authorization=_validated_authorization(), materialization_sha256="B" * 64,
    )
    assert accepted["terminal"] == runner.TERMINAL_PRECEDENCE[-1]
    one_cycle = runner._aggregate(
        mode=runner.FORMAL_MODE, snapshot=snapshot, cycle=passing,
        cycle_hashes=[proof_hash], cycles_byte_identical=False,
        authority_check_sha256="A" * 64,
        authorization=_validated_authorization(), materialization_sha256="B" * 64,
    )
    assert one_cycle["terminal"] == runner.TERMINAL_PRECEDENCE[1]

    no_go = json.loads(runner.canonical_bytes(passing))
    no_go["scientific"]["outcome"] = "SCIENTIFIC_FINDING"
    no_go["scientific"]["passed"] = False
    no_go["scientific"]["finding_groups"] = ["LOAD_MASS_OR_RECOVERY"]
    assert runner.validate_cycle_record(no_go, snapshot) == runner.TERMINAL_PRECEDENCE[3]
    no_go_hash = runner._sha256(runner.canonical_bytes(no_go))
    accepted_no_go = runner._aggregate(
        mode=runner.FORMAL_MODE, snapshot=snapshot, cycle=no_go,
        cycle_hashes=[no_go_hash, no_go_hash], cycles_byte_identical=True,
        authority_check_sha256="A" * 64,
        authorization=_validated_authorization(), materialization_sha256="B" * 64,
    )
    assert accepted_no_go["terminal"] == runner.TERMINAL_PRECEDENCE[3]
    mutated = json.loads(runner.canonical_bytes(no_go))
    mutated["passed_test_file_count"] -= 1
    with pytest.raises(runner.EvidenceError, match="counts"):
        runner.validate_cycle_record(mutated, snapshot)


def test_process_cycle_shapes_reject_impossible_replica_correlations() -> None:
    snapshot = _snapshot()
    for scientific in (
        {
            **runner._program_process_failure("PRODUCER_PROCESS_OR_OUTPUT_FAILURE"),
            "checker_replica_count": 2,
        },
        {
            **runner._program_process_failure("CHECKER_REPLICA_DISAGREEMENT"),
            "proof": _passing_cycle()["scientific"]["proof"],
        },
    ):
        cycle = runner._decorate_cycle(_passing_cycle(), snapshot)
        cycle["scientific"] = scientific
        with pytest.raises(runner.EvidenceError, match="scientific"):
            runner.validate_cycle_record(cycle, snapshot)
    blocked = runner._blocked_cycle("FAILURE")
    blocked["collected_count"] = False
    assert not runner._is_embedded_blocked_cycle(blocked)


def test_guard_and_scientific_dispositions_are_cross_correlated() -> None:
    snapshot = _snapshot()
    guard_failure_with_science = runner._decorate_cycle(_passing_cycle(), snapshot)
    first = guard_failure_with_science["tests"][0]
    first["outcome"] = "TEST_FAILURE"
    first["passed"] = False
    first["finding_group"] = runner.FOCUSED_TESTS[0].finding_group
    guard_failure_with_science["passed_test_file_count"] -= 1
    guard_failure_with_science["failed_test_file_count"] += 1
    with pytest.raises(runner.EvidenceError, match="scientific"):
        runner.validate_cycle_record(guard_failure_with_science, snapshot)

    all_guards_with_skip = runner._decorate_cycle(_passing_cycle(), snapshot)
    all_guards_with_skip["scientific"] = runner._program_process_failure(
        "SCIENTIFIC_LANE_NOT_LAUNCHED_AFTER_FOCUSED_GUARD"
    )
    with pytest.raises(runner.EvidenceError, match="scientific"):
        runner.validate_cycle_record(all_guards_with_skip, snapshot)

    baseline = runner._decorate_cycle(_passing_cycle(), snapshot)
    removed = baseline["tests"][0]
    binding = snapshot.test_bindings[runner.FOCUSED_TESTS[0].test_id]
    baseline["tests"][0] = {
        "finding_group": "BASELINE_OR_AUTHORITY",
        "git_blob_oid": binding["git_blob_oid"],
        "outcome": "PROCESS_FAILURE",
        "passed": False,
        "reason": "PRELAUNCH_IDENTITY_FAILURE",
        "sha256": binding["sha256"],
        "test_id": runner.FOCUSED_TESTS[0].test_id,
    }
    baseline["collected_count"] -= removed["collected_count"]
    baseline["passed_test_file_count"] -= 1
    baseline["failed_test_file_count"] += 1
    baseline["scientific"] = runner._program_process_failure(
        "SCIENTIFIC_LANE_NOT_LAUNCHED_AFTER_FOCUSED_GUARD"
    )
    assert runner.validate_cycle_record(baseline, snapshot) == runner.TERMINAL_PRECEDENCE[0]

    baseline["tests"][0]["reason"] = "ARBITRARY_REASON"
    with pytest.raises(runner.EvidenceError, match="process record"):
        runner.validate_cycle_record(baseline, snapshot)


def test_authority_check_only_is_path_free_deterministic_and_replica_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, dependency = tmp_path / "repository", tmp_path / "ANYfileIO"
    repository.mkdir(); dependency.mkdir()
    registry_root, registry_identity = _registry(tmp_path)
    snapshot = _snapshot(registry=registry_identity)
    monkeypatch.setattr(runner, "validate_repository", lambda *args, **kwargs: snapshot)
    outputs = (tmp_path / "check-a.json", tmp_path / "check-b.json")
    for output in outputs:
        runner.write_authority_check(
            output_path=output, repository=repository,
            anyfileio_repository=dependency, registry_root=registry_root,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    raw, record = runner.validate_authority_checks(outputs, snapshot)
    assert record == runner.expected_authority_check(snapshot)
    assert "\\" not in raw.decode() and "/" not in raw.decode()
    mutated = tmp_path / "mutated.json"
    changed = dict(record); changed["test_file_count"] += 1
    _write_canonical(mutated, changed)
    with pytest.raises(runner.AuthorizationError, match="not byte-identical"):
        runner.validate_authority_checks((outputs[0], mutated), snapshot)


def test_passing_rehearsal_is_mandatory_exact_and_hash_bound(tmp_path: Path) -> None:
    snapshot, check_sha = _snapshot(), "E" * 64
    path = tmp_path / "rehearsal.json"
    record = _make_rehearsal(path, snapshot, check_sha)
    assert runner.validate_rehearsal_aggregate(path, snapshot, check_sha)["sha256"] == runner._sha256(path.read_bytes())
    with pytest.raises(runner.AuthorizationError, match="requires a passing rehearsal"):
        runner.validate_rehearsal_aggregate(None, snapshot, check_sha)
    record["terminal"] = "NONCLASSIFYING_GE_BEAM3_P2_REHEARSAL_FINDING"
    _write_canonical(path, record)
    with pytest.raises(runner.AuthorizationError, match="does not bind"):
        runner.validate_rehearsal_aggregate(path, snapshot, check_sha)
    fabricated = {
        "candidate_id": runner.CANDIDATE_ID,
        "mode": runner.REHEARSAL_MODE,
        "schema": runner.AGGREGATE_SCHEMA,
        "study_id": runner.STUDY_ID,
        "terminal": "NONCLASSIFYING_GE_BEAM3_P2_REHEARSAL_PASS",
    }
    _write_canonical(path, fabricated)
    with pytest.raises(runner.AuthorizationError, match="does not bind"):
        runner.validate_rehearsal_aggregate(path, snapshot, check_sha)


def test_execution_review_and_authority_bind_runtime_registry_command_and_one_time(tmp_path: Path) -> None:
    registry_root, registry_identity = _registry(tmp_path)
    snapshot = _snapshot(registry=registry_identity)
    command = {"argv": ["exact", "command"], "sha256": "1" * 64}
    executor = {"bytes": 1, "git_blob_oid": "2" * 40, "path": runner.RUNNER_RELATIVE, "sha256": "3" * 64}
    rehearsal = {"bytes": 10, "sha256": "4" * 64, "terminal": "NONCLASSIFYING_GE_BEAM3_P2_REHEARSAL_PASS"}
    review = runner.expected_execution_review(
        snapshot, authority_check_sha256="5" * 64, command=command,
        executor=executor, rehearsal_aggregate=rehearsal,
    )
    assert set(review) == {"findings", "reviewed_inputs", "reviewer_independence", "schema", "verdict"}
    review_binding = {"bytes": 10, "git_blob_oid": "6" * 40, "sha256": "7" * 64, "verdict": runner.EXECUTION_REVIEW_VERDICT}
    authority = runner.expected_authorization(
        snapshot, "5" * 64, request_id="deadbeef" * 4,
        attempt_id="facefeed" * 4, registry_root=registry_root, command=command,
        executor=executor, review=review_binding, rehearsal_aggregate=rehearsal,
    )
    assert authority["request"]["one_time"] is True
    assert authority["request"]["no_retry"] is True
    assert authority["request"]["registry"] == registry_identity.record()
    assert authority["runtime"] == snapshot.runtime.record()


def test_exact_one_parent_two_path_git_authorization_overlay(tmp_path: Path) -> None:
    repository, harness_commit, harness_tree = _init_repository(
        tmp_path / "ANYfeSolver", {runner.RUNNER_RELATIVE: RUNNER.read_bytes()}
    )
    registry_root, registry_identity = _registry(tmp_path)
    snapshot = _snapshot(
        registry=registry_identity, harness_commit=harness_commit,
        harness_tree=harness_tree,
        solver_files=runner._git_tree_manifest(repository, harness_commit),
    )
    command = runner.normalized_formal_command(
        repository=repository, anyfileio_repository=tmp_path / "ANYfileIO",
        registry_root=registry_root,
        authority_check_paths=(tmp_path / "check-1.json", tmp_path / "check-2.json"),
        rehearsal_aggregate_path=tmp_path / "rehearsal.json",
        work_root=tmp_path / "work", output_path=tmp_path / "aggregate.json",
    )
    executor = runner._executor_binding(repository, harness_commit)
    rehearsal = {"bytes": 10, "sha256": "B" * 64, "terminal": "NONCLASSIFYING_GE_BEAM3_P2_REHEARSAL_PASS"}
    review_record = runner.expected_execution_review(
        snapshot, authority_check_sha256="A" * 64, command=command,
        executor=executor, rehearsal_aggregate=rehearsal,
    )
    review_path = repository / runner.EXECUTION_REVIEW_RELATIVE_PATH
    _write_canonical(review_path, review_record)
    review_raw = review_path.read_bytes()
    review_binding = {
        "bytes": len(review_raw),
        "git_blob_oid": _git(repository, "hash-object", f"--path={runner.EXECUTION_REVIEW_RELATIVE_PATH}", "--", runner.EXECUTION_REVIEW_RELATIVE_PATH),
        "sha256": runner._sha256(review_raw),
        "verdict": runner.EXECUTION_REVIEW_VERDICT,
    }
    authority_record = runner.expected_authorization(
        snapshot, "A" * 64, request_id="deadbeef" * 4,
        attempt_id="facefeed" * 4, registry_root=registry_root, command=command,
        executor=executor, review=review_binding, rehearsal_aggregate=rehearsal,
    )
    _write_canonical(repository / runner.AUTHORITY_RELATIVE_PATH, authority_record)
    _git(repository, "add", runner.EXECUTION_REVIEW_RELATIVE_PATH, runner.AUTHORITY_RELATIVE_PATH)
    _git(repository, "commit", "-m", runner.AUTHORIZATION_SUBJECT)
    validated = runner.validate_formal_overlay(
        repository=repository, snapshot=snapshot, authority_check_sha256="A" * 64,
        command=command, registry_root=registry_root, rehearsal_aggregate=rehearsal,
    )
    assert validated.request_id == "deadbeef" * 4
    assert validated.attempt_id == "facefeed" * 4
    with pytest.raises(runner.AuthorizationError, match="review"):
        runner.validate_formal_overlay(
            repository=repository, snapshot=snapshot, authority_check_sha256="A" * 64,
            command={**command, "sha256": "0" * 64}, registry_root=registry_root,
            rehearsal_aggregate=rehearsal,
        )


def test_one_time_claim_precedes_children_and_second_use_is_refused(tmp_path: Path) -> None:
    registry_root, registry_identity = _registry(tmp_path)
    snapshot, validated = _snapshot(registry=registry_identity), _validated_authorization()
    output, work = tmp_path / "aggregate.json", tmp_path / "work"
    claim = runner.acquire_one_time_claim(
        validated, snapshot, registry_root=registry_root, output_path=output, work_root=work,
    )
    assert claim.claim_path.is_file() and not work.exists() and not output.exists()
    runner._release_slot(claim)
    with pytest.raises(runner.AuthorizationError, match="already consumed"):
        runner.acquire_one_time_claim(
            validated, snapshot, registry_root=registry_root,
            output_path=output, work_root=work,
        )


def test_terminal_receipt_precedes_promotion_and_recovers_without_children(tmp_path: Path) -> None:
    registry_root, registry_identity = _registry(tmp_path)
    snapshot, validated = _snapshot(registry=registry_identity), _validated_authorization()
    output, work = tmp_path / "aggregate.json", tmp_path / "work"
    claim = runner.acquire_one_time_claim(
        validated, snapshot, registry_root=registry_root, output_path=output, work_root=work,
    )
    blocked = runner._blocked_cycle("MATERIALIZATION_FAILURE")
    blocked_hash = runner._persist_blocked_cycle(work, blocked)
    core = runner._aggregate(
        mode=runner.FORMAL_MODE, snapshot=snapshot, cycle=blocked,
        cycle_hashes=[blocked_hash], cycles_byte_identical=False,
        authority_check_sha256="A" * 64, authorization=validated,
        materialization_sha256=None,
    )
    runner.write_terminal_receipt(
        claim, validated, snapshot, aggregate_core=core,
        output_path=output, work_root=work,
    )
    assert claim.receipt_path.is_file() and not output.exists()
    runner._release_slot(claim)
    receipt_raw, receipt_record = runner.strict_canonical_json(claim.receipt_path)
    receipt_binding = {
        "bytes": len(receipt_raw),
        "sha256": runner._sha256(receipt_raw),
        "terminal": core["terminal"],
    }
    expected = runner._with_consumption(
        receipt_record["aggregate_core"],
        claim=claim.binding,
        receipt=receipt_binding,
        request_id=validated.request_id,
        attempt_id=validated.attempt_id,
    )
    pending = output.with_name(f".{output.name}.{validated.attempt_id}.pending")
    pending.write_bytes(runner.canonical_bytes(expected))
    recovered = runner.recover_publication(
        validated=validated, snapshot=snapshot, registry_root=registry_root,
        output_path=output, work_root=work,
    )
    assert recovered is not None
    assert recovered["terminal"] == runner.TERMINAL_PRECEDENCE[1]
    assert output.read_bytes() == runner.canonical_bytes(recovered)
    assert not pending.exists()
    assert runner.recover_publication(
        validated=validated, snapshot=snapshot, registry_root=registry_root,
        output_path=output, work_root=work,
    ) == recovered
    output.write_bytes(b"mismatched publication")
    with pytest.raises(runner.AuthorizationError, match="does not match"):
        runner.recover_publication(
            validated=validated, snapshot=snapshot, registry_root=registry_root,
            output_path=output, work_root=work,
        )


def test_terminal_receipt_revalidates_immutable_claim(tmp_path: Path) -> None:
    registry_root, registry_identity = _registry(tmp_path)
    snapshot, validated = _snapshot(registry=registry_identity), _validated_authorization()
    output, work = tmp_path / "aggregate.json", tmp_path / "work"
    claim = runner.acquire_one_time_claim(
        validated, snapshot, registry_root=registry_root,
        output_path=output, work_root=work,
    )
    blocked = runner._blocked_cycle("PROCESS_FAILURE")
    core = runner._aggregate(
        mode=runner.FORMAL_MODE, snapshot=snapshot, cycle=blocked,
        cycle_hashes=[runner._sha256(runner.canonical_bytes(blocked))],
        cycles_byte_identical=False,
        authority_check_sha256=validated.authority_check_sha256,
        authorization=validated, materialization_sha256=None,
    )
    _raw, mutated = runner.strict_canonical_json(claim.claim_path)
    mutated["runtime_sha256"] = "D" * 64
    _write_canonical(claim.claim_path, mutated)
    try:
        with pytest.raises(runner.AuthorizationError, match="claim changed"):
            runner.write_terminal_receipt(
                claim, validated, snapshot, aggregate_core=core,
                output_path=output, work_root=work,
            )
        assert not claim.receipt_path.exists()
    finally:
        runner._release_slot(claim)


def test_terminal_receipt_rejects_mutated_classifying_cycle(tmp_path: Path) -> None:
    snapshot, validated = _snapshot(), _validated_authorization()
    cycle = runner._decorate_cycle(_passing_cycle(), snapshot)
    cycle["passed_test_file_count"] -= 1
    raw = runner.canonical_bytes(cycle)
    work = tmp_path / "work"
    for index in (1, 2):
        path = work / f"cycle-{index}" / "cycle-summary.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
    core = runner._aggregate(
        mode=runner.FORMAL_MODE, snapshot=snapshot, cycle=cycle,
        cycle_hashes=[runner._sha256(raw), runner._sha256(raw)],
        cycles_byte_identical=True,
        authority_check_sha256=validated.authority_check_sha256,
        authorization=validated, materialization_sha256="B" * 64,
    )
    claim = runner.OneTimeClaim(
        binding={"bytes": 1, "sha256": "C" * 64},
        claim_path=tmp_path / "claim.json", lock_path=tmp_path / "lock",
        owner_path=tmp_path / "owner.json", receipt_path=tmp_path / "receipt.json",
    )
    with pytest.raises(runner.AuthorizationError, match="cycle is malformed"):
        runner.write_terminal_receipt(
            claim, validated, snapshot, aggregate_core=core,
            output_path=tmp_path / "aggregate.json", work_root=work,
        )
    assert not claim.receipt_path.exists()


@pytest.fixture(scope="module")
def exact_runtime() -> object:
    return runner.runtime_identity()


def test_closed_world_materialization_and_isolated_bound_dependency_import(
    tmp_path: Path, exact_runtime: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = b"import json,os,pathlib,sys\nimport anyfileio\nout=pathlib.Path(sys.argv[1])\nr={'origin':str(pathlib.Path(anyfileio.__file__).resolve()),'plugin':os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'),'value':anyfileio.VALUE}\nout.write_text(json.dumps(r,sort_keys=True,separators=(',',':'))+'\\n',encoding='utf-8')\n"
    solver, commit, tree = _init_repository(
        tmp_path / "ANYfeSolver",
        {"tool.py": script, "src/anyfesolver/__init__.py": b"VALUE = 'solver'\n"},
    )
    dependency = _dependency_repository(tmp_path)
    dep_identity = runner.validate_anyfileio_repository(dependency)
    registry_root, registry_identity = _registry(tmp_path)
    snapshot = _snapshot(
        anyfileio=dep_identity, registry=registry_identity, runtime=exact_runtime,
        harness_commit=commit, harness_tree=tree,
        solver_files=runner._git_tree_manifest(solver, commit),
    )
    work = tmp_path / "work"; work.mkdir()
    made = runner.materialize_execution(
        repository=solver, anyfileio_repository=dependency,
        snapshot=snapshot, work_root=work,
    )
    try:
        runner.validate_materialization(made, snapshot)
        child, result_path = tmp_path / "isolated-child", tmp_path / "isolated-child/result.json"
        monkeypatch.setenv("PYTHONPATH", "UNBOUND_SENTINEL")
        process = runner.run_bounded_process(
            runner._isolated_command(made, snapshot, script_relative="tool.py", arguments=(str(result_path),)),
            directory=child, result_path=result_path,
            environment=runner._child_environment(made.solver, made.anyfileio, child),
            bounds=runner.ExecutionBounds(child_wall_seconds=60, complete_wave_wall_seconds=60, inactivity_seconds=30, memory_limit_bytes=4 * 1024**3, maximum_concurrent_workers=1, numerical_library_threads=1),
            absolute_deadline=time.monotonic() + 60,
        )
        assert process == runner.ProcessResult(None, 0, True)
        record = json.loads(result_path.read_bytes())
        assert record["value"] == 17 and record["plugin"] == "1"
        assert Path(record["origin"]).is_relative_to(made.anyfileio)
        extra = made.solver / "ignored-transient.txt"; extra.write_bytes(b"not registered")
        with pytest.raises(runner.BaselineError, match="exact and clean|unregistered"):
            runner.validate_materialization(made, snapshot)
        extra.unlink()
        target = made.solver / "tool.py"; target.chmod(stat.S_IWRITE | stat.S_IREAD)
        original = target.read_bytes(); target.write_bytes(original + b"# mutation\n")
        with pytest.raises(runner.BaselineError, match="exact and clean|bytes changed"):
            runner.validate_materialization(made, snapshot)
        target.write_bytes(original)
    finally:
        for root in (made.solver, made.anyfileio):
            for path in root.rglob("*"):
                if path.is_file():
                    path.chmod(stat.S_IWRITE | stat.S_IREAD)


def test_actual_lane_import_closure_is_bound_and_runtime_mutation_refuses(
    tmp_path: Path, exact_runtime: object
) -> None:
    dependency = ROOT.parents[2] / "ANYfileIO"
    if not (dependency / "src/anyfileio/__init__.py").is_file():
        pytest.skip("local bound ANYfileIO checkout is unavailable")
    snapshot = _snapshot(runtime=exact_runtime)
    made = runner.MaterializedExecution(
        anyfileio=dependency,
        manifest_path=tmp_path / "unused-materialization.json",
        manifest_sha256="A" * 64,
        root=ROOT,
        solver=ROOT,
    )
    bounds = runner.ExecutionBounds(
        child_wall_seconds=120,
        complete_wave_wall_seconds=120,
        inactivity_seconds=60,
        memory_limit_bytes=4 * 1024**3,
        maximum_concurrent_workers=1,
        numerical_library_threads=1,
    )
    output = tmp_path / "runtime-audit/result.json"
    result = runner.run_bounded_process(
        runner._isolated_command(
            made,
            snapshot,
            script_relative=runner.RUNNER_RELATIVE,
            arguments=("--runtime-import-audit-worker", "--output", str(output)),
        ),
        directory=tmp_path / "runtime-audit",
        result_path=output,
        environment=runner._child_environment(ROOT, dependency, tmp_path / "runtime-audit"),
        bounds=bounds,
        absolute_deadline=time.monotonic() + 120,
    )
    assert result == runner.ProcessResult(None, 0, True)
    _raw, record = runner.strict_canonical_json(output)
    assert record["terminal"] == "NONCLASSIFYING_GE_BEAM3_P2_RUNTIME_IMPORT_AUDIT_PASS"
    assert record["loaded_lane_count"] == len(runner.FOCUSED_TESTS) + 2

    changed_rows = [dict(row) for row in exact_runtime.distributions]
    changed_rows[0]["origin_sha256"] = "F" * 64
    mutated_runtime = replace(
        exact_runtime, distributions=tuple(changed_rows)
    )
    mutated_snapshot = replace(snapshot, runtime=mutated_runtime)
    refused_output = tmp_path / "runtime-mutation/result.json"
    refused = runner.run_bounded_process(
        runner._isolated_command(
            made,
            mutated_snapshot,
            script_relative=runner.RUNNER_RELATIVE,
            arguments=(
                "--runtime-import-audit-worker",
                "--output",
                str(refused_output),
            ),
        ),
        directory=tmp_path / "runtime-mutation",
        result_path=refused_output,
        environment=runner._child_environment(
            ROOT, dependency, tmp_path / "runtime-mutation"
        ),
        bounds=bounds,
        absolute_deadline=time.monotonic() + 120,
    )
    assert refused.returncode != 0
    assert not refused_output.exists()


def test_prelaunch_materialization_mutation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made = runner.MaterializedExecution(
        anyfileio=tmp_path / "ANYfileIO", manifest_path=tmp_path / "materialization.json",
        manifest_sha256="A" * 64, root=tmp_path / "materialized", solver=tmp_path / "ANYfeSolver",
    )
    monkeypatch.setattr(
        runner, "validate_materialization",
        lambda *_args: (_ for _ in ()).throw(runner.BaselineError("changed")),
    )
    with pytest.raises(runner.BaselineError, match="changed"):
        runner._run_one_test(
            made, _snapshot(), tmp_path / "child", runner.FOCUSED_TESTS[0],
            runner.FROZEN_BOUNDS, time.monotonic() + 10,
        )
    assert not (tmp_path / "child").exists()


def test_bounded_process_terminates_timeout_and_reports_failure(tmp_path: Path) -> None:
    environment = dict(os.environ)
    bounds = runner.ExecutionBounds(child_wall_seconds=0.08, complete_wave_wall_seconds=2, inactivity_seconds=1, memory_limit_bytes=1024**3, maximum_concurrent_workers=1, numerical_library_threads=1)
    timeout = runner.run_bounded_process(
        (sys.executable, "-c", "import time;time.sleep(3)"),
        directory=tmp_path / "timeout", result_path=tmp_path / "timeout/result.json",
        environment=environment, bounds=bounds,
        absolute_deadline=time.monotonic() + 2, poll_seconds=0.01,
    )
    assert timeout.forced_reason == "CHILD_WALL_LIMIT" and timeout.tree_drained is True
    failed = runner.run_bounded_process(
        (sys.executable, "-c", "raise SystemExit(7)"),
        directory=tmp_path / "failure", result_path=tmp_path / "failure/result.json",
        environment=environment, bounds=bounds,
        absolute_deadline=time.monotonic() + 2, poll_seconds=0.01,
    )
    assert failed == runner.ProcessResult(None, 7, True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_posix_descendant_is_killed_even_when_group_leader_exits(tmp_path: Path) -> None:
    child, pid_path = tmp_path / "leader", tmp_path / "leader/descendant.pid"
    command = (
        sys.executable, "-c",
        "import pathlib,subprocess,sys;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid))",
        str(pid_path),
    )
    result = runner.run_bounded_process(
        command, directory=child, result_path=child / "none.json",
        environment=dict(os.environ),
        bounds=runner.ExecutionBounds(child_wall_seconds=5, complete_wave_wall_seconds=10, inactivity_seconds=5, memory_limit_bytes=1024**3, maximum_concurrent_workers=1, numerical_library_threads=1),
        absolute_deadline=time.monotonic() + 10,
    )
    assert result.tree_drained is True
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), signal.SIGCONT)


def test_worker_evidence_missing_malformed_and_undrained_is_process_failure(tmp_path: Path) -> None:
    spec = runner.FOCUSED_TESTS[0]
    missing = runner._validate_worker_record(tmp_path / "missing.json", spec, runner.ProcessResult(None, 0, True))
    assert missing["reason"] == "WORKER_PROCESS_OR_OUTPUT_FAILURE"
    malformed = tmp_path / "malformed.json"; _write_canonical(malformed, {"schema": runner.WORKER_SCHEMA})
    assert runner._validate_worker_record(malformed, spec, runner.ProcessResult(None, 0, True))["reason"] == "MALFORMED_WORKER_RECORD"
    undrained = runner._validate_worker_record(malformed, spec, runner.ProcessResult("PROCESS_TREE_DRAIN_FAILURE", 0, False))
    assert undrained["reason"] == "PROCESS_TREE_DRAIN_FAILURE"


def test_aggregate_is_path_timing_and_measurement_free() -> None:
    snapshot = _snapshot(); cycle = runner._decorate_cycle(_passing_cycle(), snapshot)
    aggregate = runner._aggregate(
        mode=runner.FORMAL_MODE, snapshot=snapshot, cycle=cycle,
        cycle_hashes=[runner._sha256(runner.canonical_bytes(cycle))] * 2,
        cycles_byte_identical=True, authority_check_sha256="E" * 64,
        authorization=_validated_authorization(), materialization_sha256="F" * 64,
    )
    forbidden = {"path", "cwd", "command", "duration", "elapsed", "timing", "rss", "pid", "stdout", "stderr"}
    def inspect_value(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert key.lower() not in forbidden; inspect_value(item)
        elif isinstance(value, list):
            for item in value: inspect_value(item)
        elif isinstance(value, float):
            pytest.fail("aggregate contains a floating measurement")
        elif isinstance(value, str):
            assert "\\" not in value and "/" not in value
    inspect_value(aggregate)
    assert aggregate["result"]["scientific"]["proof"]["raw_record_count"] == 21
    assert json.loads(runner.canonical_bytes(aggregate)) == aggregate


def test_child_environment_is_closed_and_forces_one_numerical_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository, dependency = tmp_path / "ANYfeSolver", tmp_path / "ANYfileIO"
    repository.mkdir(); dependency.mkdir()
    monkeypatch.setenv("PYTHONPATH", "UNBOUND_SENTINEL")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--unbound")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "AVX2")
    monkeypatch.setenv("SCIPY_ARRAY_API", "1")
    monkeypatch.setenv("ANY_UNREGISTERED_BEHAVIOR", "1")
    monkeypatch.setenv("VIRTUAL_ENV", "C:/unbound")
    environment = runner._child_environment(repository, dependency, tmp_path)
    assert {environment[name] for name in runner.THREAD_VARIABLES} == {"1"}
    assert environment["PYTHONPATH"] == os.pathsep.join((str((repository / "src").resolve()), str((dependency / "src").resolve())))
    assert "UNBOUND_SENTINEL" not in environment["PYTHONPATH"]
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "PYTEST_ADDOPTS" not in environment and "VIRTUAL_ENV" not in environment
    for name in (
        "ANY_UNREGISTERED_BEHAVIOR", "NUMBA_DISABLE_JIT",
        "NPY_DISABLE_CPU_FEATURES", "PYTHONOPTIMIZE", "SCIPY_ARRAY_API",
    ):
        assert name not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["Q1M_ANYFILEIO_ROOT"] == str(dependency.resolve())


def test_runner_does_not_author_or_expose_public_integration() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "--emit-authority" not in source and "--create-authority" not in source
    assert "cycle_runner=" not in source
    assert "src/anyfesolver/__init__.py" not in runner.HARNESS_PATHS
    assert "PROVISIONAL_GO_GE_BEAM3_P2_PRIVATE_PARITY" in source
    assert "PROVISIONAL_GO_GE_BEAM3_STRAIGHT_STANDALONE_OPT_IN" not in source
