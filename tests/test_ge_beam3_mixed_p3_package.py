"""Harness tests for the P3 installed-wheel gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT
    / "docs"
    / "reference_cases"
    / "ge_beam3_mixed_p3_package_gate.py"
)
SPEC = importlib.util.spec_from_file_location("ge_beam3_mixed_p3_package_gate", GATE_PATH)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _valid_installed_record(*, replacement_hash: str | None = None) -> dict[str, Any]:
    digest = replacement_hash or ("A" * 64)
    return {
        "array_hashes": {
            name: digest for name in GATE.INSTALLED_ARRAY_HASH_KEYS
        },
        "check_count": len(GATE.INSTALLED_CHECK_KEYS),
        "checks": {name: True for name in GATE.INSTALLED_CHECK_KEYS},
        "element_record_sha256": digest,
        "formulation_id": GATE.QUALIFIED_ID,
        "loaded_anyfesolver_module_count": 7,
        "schema": GATE.CHECK_SCHEMA,
        "selector": GATE.SELECTOR,
        "state_layout_id": GATE.STATE_LAYOUT_ID,
        "state_record_sha256": digest,
        "state_schema": GATE.STATE_SCHEMA,
        "state_version": GATE.STATE_VERSION,
    }


def _valid_package_record(wheel: Path) -> dict[str, Any]:
    return {
        "candidate_commit": "a" * 40,
        "candidate_tree": "b" * 40,
        "canonical_run_count": 2,
        "correctness_record_sha256": "C" * 64,
        "correctness_records_byte_identical": True,
        "formulation_id": GATE.QUALIFIED_ID,
        "installed_runner_sha256": "D" * 64,
        "package_gate_passed": True,
        "process_containment": GATE.PROCESS_CONTAINMENT_ID,
        "schema": GATE.SCHEMA,
        "selector": GATE.SELECTOR,
        "wheel": GATE.assert_regular_wheel(wheel),
    }


def test_canonical_json_rejects_duplicates_nonfinite_and_noncanonical() -> None:
    value = {"a": [1, True, None], "z": "value"}
    raw = GATE.canonical_json_bytes(value)
    assert raw == b'{"a":[1,true,null],"z":"value"}\n'
    assert GATE.strict_canonical_json_loads(raw) == value
    with pytest.raises(GATE.PackageGateError, match="duplicate"):
        GATE.strict_canonical_json_loads(b'{"a":1,"a":2}\n')
    with pytest.raises(GATE.PackageGateError, match="nonfinite"):
        GATE.strict_canonical_json_loads(b'{"a":NaN}\n')
    with pytest.raises(GATE.PackageGateError, match="not canonical"):
        GATE.strict_canonical_json_loads(b'{"z": "value", "a": [1]}\n')


def test_installed_record_schema_is_closed_and_fail_closed() -> None:
    record = _valid_installed_record()
    assert GATE._validate_installed_correctness_record(record) == record

    mutations: list[dict[str, Any]] = []
    missing = dict(record)
    missing.pop("selector")
    mutations.append(missing)
    unknown = dict(record)
    unknown["unexpected"] = True
    mutations.append(unknown)
    failed = dict(record)
    failed["checks"] = dict(record["checks"])
    failed["checks"][next(iter(GATE.INSTALLED_CHECK_KEYS))] = False
    mutations.append(failed)
    non_boolean = dict(record)
    non_boolean["checks"] = dict(record["checks"])
    non_boolean["checks"][next(iter(GATE.INSTALLED_CHECK_KEYS))] = 1
    mutations.append(non_boolean)
    malformed_hash = dict(record)
    malformed_hash["state_record_sha256"] = "a" * 64
    mutations.append(malformed_hash)
    wrong_count = dict(record)
    wrong_count["check_count"] = len(GATE.INSTALLED_CHECK_KEYS) - 1
    mutations.append(wrong_count)
    for mutation in mutations:
        with pytest.raises(GATE.PackageGateError):
            GATE._validate_installed_correctness_record(mutation)


def test_package_aggregate_schema_is_closed_and_binds_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"candidate")
    record = _valid_package_record(wheel)
    assert GATE._validate_package_aggregate(record) == record

    mutations = []
    failed = dict(record)
    failed["package_gate_passed"] = False
    mutations.append(failed)
    reused_shape = dict(record)
    reused_shape["canonical_run_count"] = True
    mutations.append(reused_shape)
    wrong_commit = dict(record)
    wrong_commit["candidate_commit"] = "A" * 40
    mutations.append(wrong_commit)
    bad_wheel = dict(record)
    bad_wheel["wheel"] = dict(record["wheel"])
    bad_wheel["wheel"]["sha256"] = "d" * 64
    mutations.append(bad_wheel)
    unknown = dict(record)
    unknown["unregistered"] = True
    mutations.append(unknown)
    for mutation in mutations:
        with pytest.raises(GATE.PackageGateError):
            GATE._validate_package_aggregate(mutation)


def test_exclusive_output_and_exact_regular_wheel_identity(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    GATE.write_exclusive(output, {"passed": True})
    with pytest.raises(FileExistsError):
        GATE.write_exclusive(output, {"passed": True})

    wheel = tmp_path / "ANYsolver-0.4.2-py3-none-any.whl"
    wheel.write_bytes(b"frozen-wheel")
    identity = GATE.assert_regular_wheel(wheel)
    assert identity == {
        "bytes": len(b"frozen-wheel"),
        "filename": wheel.name,
        "sha256": GATE.sha256_file(wheel),
    }
    empty = tmp_path / "empty.whl"
    empty.touch()
    with pytest.raises(GATE.PackageGateError, match="nonempty"):
        GATE.assert_regular_wheel(empty)


def test_bounded_child_records_output_and_terminates_timeout(tmp_path: Path) -> None:
    success = tmp_path / "success.log"
    GATE.run_bounded(
        [sys.executable, "-c", "print('bounded-ok')"],
        cwd=tmp_path,
        log_path=success,
        timeout_seconds=10,
    )
    assert success.read_text(encoding="utf-8").strip() == "bounded-ok"

    timeout = tmp_path / "timeout.log"
    with pytest.raises(GATE.PackageGateError, match="exceeded"):
        GATE.run_bounded(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            log_path=timeout,
            timeout_seconds=1,
        )

    inactive = tmp_path / "inactive.log"
    with pytest.raises(GATE.PackageGateError, match="no log or CPU progress"):
        GATE.run_bounded(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            log_path=inactive,
            timeout_seconds=10,
            inactivity_seconds=1,
        )


def test_isolated_probe_flags_honor_the_frozen_hash_seed(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-s",
        "-P",
        "-B",
        "-c",
        "print(hash('GE_BEAM3_P3'))",
    ]
    outputs = [
        subprocess.check_output(
            command,
            cwd=tmp_path,
            env=GATE._bounded_environment(),
            text=True,
            encoding="utf-8",
        )
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]


def test_package_gate_builds_once_and_requires_identical_fresh_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    runner_bytes = GATE_PATH.read_bytes()

    def fake_git(_repository: Path, *arguments: str) -> str:
        if arguments[0] == "status":
            return ""
        if arguments[-1] == "HEAD":
            return "a" * 40
        if arguments[-1] == "HEAD^{tree}":
            return "b" * 40
        raise AssertionError(arguments)

    def fake_git_bytes(_repository: Path, *arguments: str) -> bytes:
        assert arguments[:2] == ("cat-file", "blob")
        return runner_bytes

    class FakeTar:
        def __enter__(self) -> "FakeTar":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def extractall(self, path: Path, *, filter: str) -> None:
            assert filter == "data"
            runner = path / GATE.RUNNER_RELATIVE_PATH
            runner.parent.mkdir(parents=True)
            runner.write_bytes(runner_bytes)
            (path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        timeout_seconds: int = GATE.CHILD_TIMEOUT_SECONDS,
        deadline_monotonic: float | None = None,
    ) -> None:
        del cwd, timeout_seconds, deadline_monotonic
        calls.append(tuple(map(str, command)))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"diagnostic\n")
        if "archive" in command:
            target = next(
                Path(item.split("=", 1)[1])
                for item in command
                if item.startswith("--output=")
            )
            target.write_bytes(b"candidate-archive")
        if "wheel" in command:
            destination = Path(command[command.index("--wheel-dir") + 1])
            (destination / "ANYsolver-0.4.2-py3-none-any.whl").write_bytes(
                b"one-frozen-wheel"
            )
        if "venv" in command:
            environment = Path(command[-1])
            (environment / "Scripts").mkdir(parents=True, exist_ok=True)
            (environment / "Scripts" / "python.exe").write_bytes(b"python")
        if "--installed-check" in command:
            output = Path(command[command.index("--output") + 1])
            GATE.write_exclusive(output, _valid_installed_record())

    monkeypatch.setattr(GATE, "_git", fake_git)
    monkeypatch.setattr(GATE, "_git_bytes", fake_git_bytes)
    monkeypatch.setattr(GATE.tarfile, "open", lambda *_args, **_kwargs: FakeTar())
    monkeypatch.setattr(GATE, "run_bounded", fake_run)
    monkeypatch.setattr(GATE, "_require_formal_process_containment", lambda: None)
    output_root = tmp_path / "gate"
    aggregate = GATE.run_package_gate(ROOT, output_root)
    assert aggregate["package_gate_passed"] is True
    assert aggregate["canonical_run_count"] == 2
    assert aggregate["correctness_records_byte_identical"] is True
    assert sum("wheel" in call for call in calls) == 1
    assert sum("venv" in call for call in calls) == 2
    assert sum("--installed-check" in call for call in calls) == 2
    assert (output_root / "installed-check-runner.py").is_file()
    assert aggregate["installed_runner_sha256"] == GATE.sha256_file(
        output_root / "installed-check-runner.py"
    )
    for call in (item for item in calls if "--installed-check" in item):
        assert str(output_root / "installed-check-runner.py") in call
        assert "--forbidden-repository-root" in call
    stored = GATE.strict_canonical_json_loads(
        (output_root / "package-aggregate.json").read_bytes()
    )
    assert stored == aggregate


def test_package_gate_disagreement_creates_no_canonical_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_index = 0
    runner_bytes = GATE_PATH.read_bytes()

    def fake_git(_repository: Path, *arguments: str) -> str:
        if arguments[0] == "status":
            return ""
        return "a" * 40

    def fake_git_bytes(_repository: Path, *arguments: str) -> bytes:
        assert arguments[:2] == ("cat-file", "blob")
        return runner_bytes

    class FakeTar:
        def __enter__(self) -> "FakeTar":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def extractall(self, path: Path, *, filter: str) -> None:
            assert filter == "data"
            runner = path / GATE.RUNNER_RELATIVE_PATH
            runner.parent.mkdir(parents=True)
            runner.write_bytes(runner_bytes)
            (path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        timeout_seconds: int = GATE.CHILD_TIMEOUT_SECONDS,
        deadline_monotonic: float | None = None,
    ) -> None:
        nonlocal check_index
        del cwd, timeout_seconds, deadline_monotonic
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"diagnostic\n")
        if "archive" in command:
            target = next(
                Path(item.split("=", 1)[1])
                for item in command
                if item.startswith("--output=")
            )
            target.write_bytes(b"candidate-archive")
        if "wheel" in command:
            destination = Path(command[command.index("--wheel-dir") + 1])
            (destination / "ANYsolver-0.4.2-py3-none-any.whl").write_bytes(b"wheel")
        if "venv" in command:
            environment = Path(command[-1])
            (environment / "Scripts").mkdir(parents=True, exist_ok=True)
            (environment / "Scripts" / "python.exe").write_bytes(b"python")
        if "--installed-check" in command:
            check_index += 1
            output = Path(command[command.index("--output") + 1])
            digest = ("A" if check_index == 1 else "B") * 64
            GATE.write_exclusive(
                output, _valid_installed_record(replacement_hash=digest)
            )

    monkeypatch.setattr(GATE, "_git", fake_git)
    monkeypatch.setattr(GATE, "_git_bytes", fake_git_bytes)
    monkeypatch.setattr(GATE.tarfile, "open", lambda *_args, **_kwargs: FakeTar())
    monkeypatch.setattr(GATE, "run_bounded", fake_run)
    monkeypatch.setattr(GATE, "_require_formal_process_containment", lambda: None)
    output_root = tmp_path / "disagreement"
    with pytest.raises(GATE.PackageGateError, match="disagree"):
        GATE.run_package_gate(ROOT, output_root)
    assert not (output_root / "package-aggregate.json").exists()


def test_gate_source_freezes_isolation_and_resource_controls() -> None:
    source = GATE_PATH.read_text(encoding="utf-8")
    assert '"--no-build-isolation"' in source
    assert 'environment["PYTHONNOUSERSITE"] = "1"' in source
    assert 'environment.pop("PYTHONPATH", None)' in source
    assert "CHILD_TIMEOUT_SECONDS = 600" in source
    assert "WAVE_TIMEOUT_SECONDS = 1800" in source
    assert "INACTIVITY_SECONDS = 300" in source
    assert '"taskkill", "/PID"' in source
    assert 'str(python),\n                "-s",\n                "-P"' in source
    assert 'str(python_by_role[role]),\n                "-s",\n                "-P"' in source
    assert 'external_runner = output_root / "installed-check-runner.py"' in source
    assert 'candidate_source = output_root / "candidate-source"' in source
    assert '"--forbidden-repository-root"' in source
    assert (
        'PROCESS_CONTAINMENT_ID = "WINDOWS_JOB_OBJECT_SUSPENDED_ASSIGN_V1"'
        in source
    )
    assert "_require_formal_process_containment()" in source
