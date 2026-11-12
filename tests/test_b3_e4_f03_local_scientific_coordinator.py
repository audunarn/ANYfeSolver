"""Synthetic lifecycle checks.  No registered scientific child is launched."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest

from scripts import _b3_e4_f03_local_scientific_budgets as budgets
from scripts import _b3_e4_f03_local_scientific_cases as cases
from scripts import _b3_e4_f03_local_scientific_coordinator as coordinator
from scripts import _b3_e4_f03_local_scientific_registration as registration
from scripts import _b3_e4_f03_local_scientific_supervisor as supervisor


SOURCE = "a" * 40
DIGEST = "b" * 64


def test_loaded_module_guard_binds_invoking_checkout(tmp_path):
    source = Path(coordinator.__file__).resolve().parents[1]
    coordinator._verify_loaded_sources(source)
    with pytest.raises(ValueError, match="outside registered source"):
        coordinator._verify_loaded_sources(tmp_path)


def _run_value(root: Path):
    return SimpleNamespace(
        run_root=str(root), source_commit=SOURCE, repository_root=str(root.parent),
        accepted_main_commit="c" * 40,
        runtime=SimpleNamespace(executable="python"),
        packages=(), hardware="hardware",
        threads=registration.thread_contract(),
        process_limits=registration.process_contract(),
        resource_limits=registration.resource_contract(),
        tolerances=registration.tolerance_contract(),
        cases=tuple(SimpleNamespace(case_id=case.case_id)
                    for case in cases.CASES),
    )


def test_coordinator_claims_once_and_runs_exact_case_order(monkeypatch, tmp_path):
    value = _run_value(tmp_path)
    launched = []
    monkeypatch.setattr(coordinator, "_preflight", lambda *_: value)
    monkeypatch.setattr(coordinator, "_run_one",
                        lambda _, __, ___, record, *args, **kwargs: launched.append(record.case.case_id))
    monkeypatch.setattr(coordinator.aggregate, "adjudicate_and_write",
                        lambda *args, **kwargs: ("aggregate", DIGEST))
    expected = coordinator.run_registered(
        tmp_path, registration_sha256=DIGEST,
        budgets=budgets.make_scientific_budgets(),
    )
    assert expected == ("aggregate", DIGEST)
    assert launched == [case.case_id for case in cases.CASES]
    assert (tmp_path / coordinator.CLAIM_FILE).exists()
    with pytest.raises(FileExistsError, match="already claimed"):
        coordinator.run_registered(
            tmp_path, registration_sha256=DIGEST,
            budgets=budgets.make_scientific_budgets(),
        )
    assert launched == [case.case_id for case in cases.CASES]


def test_cutoff_stops_new_children_but_closes_available_aggregate(monkeypatch, tmp_path):
    value = _run_value(tmp_path)
    launched = []
    times = iter((0.0, 0.0, 7200.0))
    monkeypatch.setattr(coordinator, "_preflight", lambda *_: value)
    monkeypatch.setattr(coordinator, "_run_one",
                        lambda _, __, ___, record, *args, **kwargs: launched.append(record.case.case_id))
    monkeypatch.setattr(coordinator.aggregate, "adjudicate_and_write",
                        lambda *args, **kwargs: ("incomplete", DIGEST))
    assert coordinator.run_registered(
        tmp_path, registration_sha256=DIGEST,
        budgets=budgets.make_scientific_budgets(), monotonic=lambda: next(times),
    ) == ("incomplete", DIGEST)
    assert launched == [cases.CASES[0].case_id]


def test_cancelled_run_seals_prefix_and_never_relaunches(monkeypatch, tmp_path):
    value = _run_value(tmp_path)
    launched = []
    sealed = []
    cancellation = supervisor.SupervisionCancelled(supervisor.SupervisionReceipt(
        "cancelled", None, 123, 1.0, 1024, (123,), (123,), "KeyboardInterrupt",
    ))
    monkeypatch.setattr(coordinator, "_preflight", lambda *_: value)

    def run_one(_, __, ___, record, *args, **kwargs):
        launched.append(record.case.case_id)
        raise cancellation

    monkeypatch.setattr(coordinator, "_run_one", run_one)
    monkeypatch.setattr(coordinator.aggregate, "adjudicate_and_write",
                        lambda *args, **kwargs: sealed.append("aggregate"))
    with pytest.raises(supervisor.SupervisionCancelled):
        coordinator.run_registered(
            tmp_path, registration_sha256=DIGEST,
            budgets=budgets.make_scientific_budgets(),
        )
    assert launched == [cases.CASES[0].case_id]
    assert sealed == ["aggregate"]
    with pytest.raises(FileExistsError, match="already claimed"):
        coordinator.run_registered(
            tmp_path, registration_sha256=DIGEST,
            budgets=budgets.make_scientific_budgets(),
        )


def test_prelaunch_io_failure_seals_missing_case_inventory(monkeypatch, tmp_path):
    value = _run_value(tmp_path)
    sealed = []
    monkeypatch.setattr(coordinator, "_preflight", lambda *_: value)
    monkeypatch.setattr(coordinator, "_run_one",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("log open denied")))
    monkeypatch.setattr(coordinator.aggregate, "adjudicate_and_write",
                        lambda *args, **kwargs: sealed.append("aggregate"))
    with pytest.raises(OSError, match="log open denied"):
        coordinator.run_registered(
            tmp_path, registration_sha256=DIGEST,
            budgets=budgets.make_scientific_budgets(),
        )
    assert sealed == ["aggregate"]
    assert (tmp_path / coordinator.CLAIM_FILE).exists()


def test_dangling_case_output_symlink_rejected_before_claim(monkeypatch, tmp_path):
    value = _run_value(tmp_path)
    directory = tmp_path / "cases" / cases.CASES[0].case_id
    directory.mkdir(parents=True)
    try:
        os.symlink(directory / "missing.log", directory / "stdout.log")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    monkeypatch.setattr(coordinator, "_preflight", lambda *_: value)
    with pytest.raises(FileExistsError, match="already consumed"):
        coordinator.run_registered(
            tmp_path, registration_sha256=DIGEST,
            budgets=budgets.make_scientific_budgets(),
        )
    assert not (tmp_path / coordinator.CLAIM_FILE).exists()


def test_dirty_source_or_changed_accepted_main_fails_preflight(monkeypatch, tmp_path):
    value = _run_value(tmp_path)
    monkeypatch.setattr(coordinator, "_verify_loaded_sources", lambda *_: None)
    monkeypatch.setattr(coordinator.registration, "read_registration",
                        lambda *_: (value, DIGEST))
    monkeypatch.setattr(coordinator.child_identity, "observed_git",
                        lambda *_: (SOURCE, False))
    with pytest.raises(ValueError, match="clean source"):
        coordinator._preflight(tmp_path, DIGEST)
    monkeypatch.setattr(coordinator.child_identity, "observed_git",
                        lambda *_: (SOURCE, True))
    monkeypatch.setattr(coordinator, "_main_commit", lambda *args: "d" * 40)
    with pytest.raises(ValueError, match="accepted local/remote main"):
        coordinator._preflight(tmp_path, DIGEST)


def test_registration_preparation_is_exclusive_and_launch_free(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        "[project]\nname = 'ANYfeSolver'\nversion = '1.0'\n", encoding="ascii",
    )
    for role, relative in registration.SOURCE_ROLES:
        path = source.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {role}\n", encoding="ascii")
    solver_origin = source / "src" / "anyfesolver" / "__init__.py"
    solver_origin.parent.mkdir(parents=True)
    solver_origin.write_text("__version__ = '0.4.7'\n", encoding="ascii")
    package_paths = {}
    for name in registration.PACKAGE_NAMES:
        path = solver_origin if name == "ANYfeSolver" else tmp_path / f"{name}.py"
        if name != "ANYfeSolver":
            path.write_text(f"package {name}\n", encoding="ascii")
        package_paths[name] = path
    hardware = registration.HardwareIdentity(
        "host", "Windows", "AMD64", "CPU", 8, 16 * 1024**3,
    )
    monkeypatch.setattr(coordinator.child_identity, "observed_git",
                        lambda *_: (SOURCE, True))
    monkeypatch.setattr(coordinator, "_main_commit", lambda *_: "c" * 40)
    monkeypatch.setattr(coordinator.child_identity, "observed_package",
                        lambda name: ("1.0", str(package_paths[name])))
    monkeypatch.setattr(coordinator.child_identity, "observed_hardware",
                        lambda: hardware)
    monkeypatch.setattr(coordinator, "_verify_loaded_sources", lambda *_: None)
    run_root = tmp_path / "run"
    registered, digest = coordinator.prepare_registration(source, run_root)
    assert registration.read_registration(run_root, digest) == (registered, digest)
    assert tuple(item.case_id for item in registered.cases) == tuple(
        item.case_id for item in cases.CASES
    )
    assert all((run_root / item.plan_path).is_file() for item in registered.cases)
    assert not (run_root / coordinator.CLAIM_FILE).exists()
    assert all(not (run_root / "cases" / item.case_id / "start.json").exists()
               for item in registered.cases)
    with pytest.raises(FileExistsError, match="must be new"):
        coordinator.prepare_registration(source, run_root)
    (tmp_path / "ANYfeSolver.py").write_text("external solver\n", encoding="ascii")
    monkeypatch.setattr(coordinator.child_identity, "observed_package",
                        lambda name: ("1.0", str(tmp_path / f"{name}.py")))
    with pytest.raises(ValueError, match="candidate ANYfeSolver origin/version"):
        coordinator.prepare_registration(source, tmp_path / "run-external-origin")
    assert not (tmp_path / "run-external-origin").exists()


def _one_case_value(root: Path):
    record = cases.preflight_inventory()[0]
    directory = root / "cases" / record.case.case_id
    directory.mkdir(parents=True)
    command = SimpleNamespace(
        case_id=record.case.case_id,
        argv=registration.child_argv(sys.executable, str(root), record.case.case_id, SOURCE),
        working_directory=str(root.parent),
        stdout_path=f"cases/{record.case.case_id}/stdout.log",
        stderr_path=f"cases/{record.case.case_id}/stderr.log",
    )
    value = SimpleNamespace(
        run_root=str(root), source_commit=SOURCE, repository_root=str(root.parent),
        runtime=SimpleNamespace(executable=sys.executable),
        threads=registration.thread_contract(),
        commands=(command,),
        cases=(SimpleNamespace(case_id=record.case.case_id,
                               plan_archive_sha256=DIGEST, plan_sha256=DIGEST),),
    )
    return value, record, directory


def test_one_case_writes_terminal_before_failure_closure(monkeypatch, tmp_path):
    value, record, directory = _one_case_value(tmp_path)
    events = []
    monkeypatch.setattr(coordinator.archive, "read_plan", lambda *args: SimpleNamespace())
    monkeypatch.setattr(coordinator.adjudicator, "_verify_registered_plan_bytes",
                        lambda *args: None)
    monkeypatch.setattr(coordinator.lifecycle, "plan_sha256", lambda *args: DIGEST)
    terminal = supervisor.SupervisionReceipt(
        "time_limit", None, 123, 1801.0, 1024, (123,), (123,), None,
    )
    def supervised(argv, **kwargs):
        assert argv[-1] == DIGEST
        assert registration.REGISTRATION_TOKEN not in argv
        assert not (directory / "stdout.log").exists()
        assert not (directory / "stderr.log").exists()
        kwargs["stdout"].write(b"captured stdout")
        kwargs["stderr"].write(b"captured stderr")
        events.append("launch")
        return terminal

    monkeypatch.setattr(coordinator.supervisor, "supervise", supervised)
    monkeypatch.setattr(coordinator.lifecycle, "write_terminal",
                        lambda *args: events.append("terminal"))
    monkeypatch.setattr(coordinator, "_preflight", lambda *args: value)
    monkeypatch.setattr(coordinator.adjudicator, "close_failed_case",
                        lambda *args, **kwargs: events.append("closure"))
    monkeypatch.setattr(coordinator.adjudicator, "adjudicate_case",
                        lambda *args: pytest.fail("failed child cannot be numerically adjudicated"))
    coordinator._run_one(
        tmp_path, value, DIGEST, record, (record,), budgets.make_scientific_budgets(),
    )
    assert events == ["launch", "terminal", "closure"]
    assert (directory / "stdout.log").exists()
    assert (directory / "stderr.log").exists()
    assert (directory / "stdout.log").read_bytes() == b"captured stdout"
    assert (directory / "stderr.log").read_bytes() == b"captured stderr"


def test_typed_cancellation_closes_case_and_propagates(monkeypatch, tmp_path):
    value, record, directory = _one_case_value(tmp_path)
    events = []
    monkeypatch.setattr(coordinator.archive, "read_plan", lambda *args: SimpleNamespace())
    monkeypatch.setattr(coordinator.adjudicator, "_verify_registered_plan_bytes",
                        lambda *args: None)
    monkeypatch.setattr(coordinator.lifecycle, "plan_sha256", lambda *args: DIGEST)
    receipt = supervisor.SupervisionReceipt(
        "cancelled", None, 123, 1.0, 1024, (123,), (123,), "KeyboardInterrupt",
    )

    def cancelled(*args, **kwargs):
        raise supervisor.SupervisionCancelled(receipt)

    monkeypatch.setattr(coordinator.supervisor, "supervise", cancelled)
    monkeypatch.setattr(coordinator.lifecycle, "write_terminal",
                        lambda *args: events.append("terminal"))
    monkeypatch.setattr(coordinator.adjudicator, "close_failed_case",
                        lambda *args, **kwargs: events.append("closure"))
    with pytest.raises(supervisor.SupervisionCancelled):
        coordinator._run_one(
            tmp_path, value, DIGEST, record, (record,), budgets.make_scientific_budgets(),
        )
    assert events == ["terminal", "closure"]
    assert (directory / "stdout.log").exists()


def test_log_write_failure_preserves_typed_cancellation(monkeypatch, tmp_path):
    value, record, directory = _one_case_value(tmp_path)
    monkeypatch.setattr(coordinator.archive, "read_plan", lambda *args: SimpleNamespace())
    monkeypatch.setattr(coordinator.adjudicator, "_verify_registered_plan_bytes",
                        lambda *args: None)
    monkeypatch.setattr(coordinator.lifecycle, "plan_sha256", lambda *args: DIGEST)
    receipt = supervisor.SupervisionReceipt(
        "cancelled", None, 123, 1.0, 1024, (123,), (123,), "KeyboardInterrupt",
    )

    def cancelled(*args, **kwargs):
        (directory / "stdout.log").write_bytes(b"unexpected competing file")
        kwargs["stderr"].write(b"cancelled after useful diagnostic")
        raise supervisor.SupervisionCancelled(receipt)

    monkeypatch.setattr(coordinator.supervisor, "supervise", cancelled)
    with pytest.raises(supervisor.SupervisionCancelled) as captured:
        coordinator._run_one(
            tmp_path, value, DIGEST, record, (record,), budgets.make_scientific_budgets(),
        )
    assert isinstance(captured.value.__cause__, FileExistsError)
    assert (directory / "stderr.log").read_bytes() == b"cancelled after useful diagnostic"
