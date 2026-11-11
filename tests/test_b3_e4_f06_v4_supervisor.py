"""Disposable one-use and identity checks for the F06 external supervisor."""

import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

import pytest
import psutil

from scripts import supervise_b3_ge_e4_pl_g0_f06_v4 as supervisor

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="F06 scientific supervisor requires Windows Job Objects")


def _registration(tmp_path, monkeypatch):
    root = tmp_path / "source"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py"
    metrics_text = supervisor._canonical(dict(
        metrics=dict(effective_environment={"disposable": True}))).decode("utf-8")
    runner.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "p=Path(sys.argv[2]);p.mkdir()\n"
        f"(p/'metrics.json').write_bytes({metrics_text.encode('utf-8')!r})\n"
        "for n in ('base','rebase','successor','control'):\n"
        " (p/(n+'.checkpoint')).write_bytes(n.encode())\n"
        "print('completed')\n", encoding="utf-8")
    monkeypatch.setattr(supervisor, "ROOT", root)
    monkeypatch.setattr(supervisor, "REQUIRED_SOURCES", frozenset((
        "scripts/" + runner.name,)))
    monkeypatch.setattr(supervisor, "_git_head", lambda: "test-head")
    monkeypatch.setattr(supervisor, "_require_clean_tracked_tree", lambda: None)
    monkeypatch.setattr(supervisor, "_environment", lambda: {"disposable": True})
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    test_log = evidence / "test.log"
    test_log.write_text("2 passed\n", encoding="utf-8")
    value = dict(schema=supervisor.SCHEMA, source_commit="test-head",
                 sources={"scripts/" + runner.name: supervisor._sha(runner)},
                 environment={"disposable": True}, evidence_dir=str(evidence.resolve()),
                 command=dict(argv=[str(Path(sys.executable).resolve()),
                                    str(runner.resolve()), "--output-dir",
                                    str((evidence / "measurement").resolve())],
                              cwd=str(root.resolve())),
                 limits=dict(wall_seconds=180, sampled_tree_rss_bytes=2 * 1024**3),
                 test_log_sha256=supervisor._sha(test_log))
    registration = evidence / "registration.json"
    registration.write_bytes(supervisor._canonical(value))
    return runner, evidence, registration


def test_supervisor_runs_once_and_attests_raw_artifacts(tmp_path, monkeypatch):
    _, evidence, registration = _registration(tmp_path, monkeypatch)
    result = supervisor.supervise(registration, evidence)
    assert result["exit_code"] == 0 and result["resource_violation"] is None
    assert result["registration_sha256"] == supervisor._sha(registration)
    assert result["observed_sources"] == json.loads(registration.read_text())["sources"]
    assert result["metrics_sha256"] == supervisor._sha(evidence / "measurement/metrics.json")
    assert result["test_log_sha256"] == supervisor._sha(evidence / "test.log")
    assert result["artifacts"]["checkpoints"]["successor"]["sha256"] == \
        supervisor._sha(evidence / "measurement/successor.checkpoint")
    assert (evidence / "execution.claim").is_file()
    with pytest.raises(FileExistsError, match="already consumed"):
        supervisor.supervise(registration, evidence)


def test_supervisor_rejects_source_mutation_before_claim(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path, monkeypatch)
    runner.write_text(runner.read_text(encoding="utf-8") + "# changed\n",
                      encoding="utf-8")
    with pytest.raises(ValueError, match="source identities mismatch"):
        supervisor.supervise(registration, evidence)
    assert not (evidence / "execution.claim").exists()


def _replace_runner(runner, registration, code):
    runner.write_text(code, encoding="utf-8")
    value = json.loads(registration.read_bytes())
    value["sources"]["scripts/" + runner.name] = supervisor._sha(runner)
    registration.write_bytes(supervisor._canonical(value))


def _disposable_limits(monkeypatch, *, wall=180, rss=2 * 1024**3):
    original = supervisor._registration
    def limited(path, evidence):
        value, digest, sources, environment = original(path, evidence)
        value["limits"] = dict(wall_seconds=wall, sampled_tree_rss_bytes=rss)
        return value, digest, sources, environment
    monkeypatch.setattr(supervisor, "_registration", limited)


def test_supervisor_waits_for_owned_descendant_after_root_exits(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path, monkeypatch)
    original = runner.read_text(encoding="utf-8")
    spawn = ("import subprocess\n"
             "subprocess.Popen([sys.executable,'-c',"
             "\"import pathlib,sys,time;time.sleep(.5);"
             "pathlib.Path(sys.argv[1]).write_text('child')\","
             "str(p/'descendant.marker')])\n")
    _replace_runner(runner, registration, original.replace("print('completed')\n", spawn))
    started = monotonic()
    result = supervisor.supervise(registration, evidence)
    assert result["exit_code"] == 0 and result["resource_violation"] is None
    assert result["launch_error"] is None
    assert monotonic() - started >= .4
    assert (evidence / "measurement/descendant.marker").read_text() == "child"
    assert result["child_environment_match"] is True
    assert result["rss_sample_count"] >= 2


def test_supervisor_times_out_and_preserves_partial_evidence(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path, monkeypatch)
    _replace_runner(runner, registration,
        "from pathlib import Path\nimport sys,time\n"
        "p=Path(sys.argv[2]);p.mkdir();(p/'base.checkpoint').write_bytes(b'partial')\n"
        "print('started',flush=True);time.sleep(30)\n")
    _disposable_limits(monkeypatch, wall=.35)
    result = supervisor.supervise(registration, evidence)
    assert result["resource_violation"] == "EXTERNAL_WALL_TIMEOUT"
    assert result["exit_code"] != 0 and result["launch_error"] is None
    assert result["artifacts"]["checkpoints"]["base"]["sha256"] == \
        supervisor._sha(evidence / "measurement/base.checkpoint")
    assert result["artifacts"]["metrics"] is None
    assert result["stdout_sha256"] == supervisor._sha(evidence / "stdout.log")
    assert json.loads((evidence / "attestation.json").read_bytes()) == result


def test_supervisor_detects_sampled_job_memory(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path, monkeypatch)
    _replace_runner(runner, registration,
        "import time\nheld=bytearray(128*1024*1024)\ntime.sleep(30)\n")
    _disposable_limits(monkeypatch, rss=64 * 1024**2)
    result = supervisor.supervise(registration, evidence)
    assert result["resource_violation"] == "SAMPLED_TREE_RSS_LIMIT"
    assert result["peak_tree_rss_bytes"] > 64 * 1024**2
    assert result["exit_code"] != 0 and result["launch_error"] is None
    assert (evidence / "attestation.json").is_file()


def test_supervisor_attests_child_failure_and_postclaim_launch_error(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path / "child", monkeypatch)
    _replace_runner(runner, registration,
        "from pathlib import Path\nimport sys\n"
        "p=Path(sys.argv[2]);p.mkdir();(p/'base.checkpoint').write_bytes(b'partial')\n"
        "print('failed',file=sys.stderr);raise SystemExit(7)\n")
    result = supervisor.supervise(registration, evidence)
    assert result["exit_code"] == 7 and result["launch_error"] is None
    assert result["artifacts"]["checkpoints"]["base"] is not None
    assert (evidence / "stderr.log").read_text().strip() == "failed"
    assert (evidence / "attestation.json").is_file()

    _, evidence2, registration2 = _registration(tmp_path / "launch", monkeypatch)
    def fail_launch(*_args):
        raise RuntimeError("disposable launch failure")
    monkeypatch.setattr(supervisor, "_launch_owned", fail_launch)
    failed = supervisor.supervise(registration2, evidence2)
    assert failed["exit_code"] is None
    assert "disposable launch failure" in failed["launch_error"]
    assert (evidence2 / "execution.claim").is_file()
    assert (evidence2 / "attestation.json").is_file()


def test_supervisor_terminates_orphaned_descendant_on_timeout(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path, monkeypatch)
    _replace_runner(runner, registration,
        "from pathlib import Path\nimport subprocess,sys\n"
        "p=Path(sys.argv[2]);p.mkdir()\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
        "(p/'descendant.pid').write_text(str(child.pid))\n")
    _disposable_limits(monkeypatch, wall=.35)
    result = supervisor.supervise(registration, evidence)
    assert result["resource_violation"] == "EXTERNAL_WALL_TIMEOUT"
    assert result["exit_code"] == 0  # Root exited before its owned descendant.
    assert result["launch_error"] is None
    pid = int((evidence / "measurement/descendant.pid").read_text())
    assert not psutil.pid_exists(pid)


def test_supervisor_attests_postclaim_log_setup_failure(tmp_path, monkeypatch):
    _, evidence, registration = _registration(tmp_path, monkeypatch)
    original = Path.open
    def denied(self, *args, **kwargs):
        if self == evidence / "stderr.log":
            raise PermissionError("disposable log setup failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", denied)
    result = supervisor.supervise(registration, evidence)
    assert result["exit_code"] is None
    assert "disposable log setup failure" in result["launch_error"]
    assert (evidence / "execution.claim").is_file()
    assert (evidence / "attestation.json").is_file()
    assert result["stdout_sha256"] == supervisor._sha(evidence / "stdout.log")
    assert result["stderr_sha256"] is None


def test_supervisor_requires_exact_source_inventory_before_claim(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "REQUIRED_SOURCES", frozenset((
        "scripts/" + runner.name, "scripts/missing_required.py")))
    with pytest.raises(ValueError, match="exact F06 source identity inventory"):
        supervisor.supervise(registration, evidence)
    assert not (evidence / "execution.claim").exists()


def test_supervisor_freezes_direct_runner_imports():
    assert {
        "src/anyfesolver/control.py",
        "src/anyfesolver/_native_reference_modal.py",
        "src/anyfesolver/_ge_beam3_native_generalized_loading.py",
        "src/anyfesolver/_ge_beam3_native_line_loading.py",
        "src/anyfesolver/_ge_beam3_centered_reference.py",
        "src/anyfesolver/fe_core.py",
    } <= supervisor.MEASUREMENT_SOURCES


def test_supervisor_rejects_staged_and_unstaged_tracked_changes(tmp_path, monkeypatch):
    root = tmp_path / "tracked"
    root.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    tracked = root / "operator.py"
    tracked.write_text("reference\n", encoding="utf-8")
    subprocess.run(("git", "add", "operator.py"), cwd=root, check=True)
    subprocess.run(("git", "-c", "user.name=F06 Test", "-c",
        "user.email=f06-test@example.invalid", "commit", "-qm", "fixture"),
        cwd=root, check=True)
    monkeypatch.setattr(supervisor, "ROOT", root)
    supervisor._require_clean_tracked_tree()
    tracked.write_text("unstaged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked worktree differs"):
        supervisor._require_clean_tracked_tree()
    subprocess.run(("git", "add", "operator.py"), cwd=root, check=True)
    with pytest.raises(ValueError, match="tracked worktree differs"):
        supervisor._require_clean_tracked_tree()


def test_supervisor_rejects_dirty_tree_before_claim(tmp_path, monkeypatch):
    _, evidence, registration = _registration(tmp_path, monkeypatch)
    def dirty():
        raise ValueError("tracked worktree differs from registered HEAD")
    monkeypatch.setattr(supervisor, "_require_clean_tracked_tree", dirty)
    with pytest.raises(ValueError, match="tracked worktree differs"):
        supervisor.supervise(registration, evidence)
    assert not (evidence / "execution.claim").exists()


def test_supervisor_attests_dirty_tree_after_claim(tmp_path, monkeypatch):
    _, evidence, registration = _registration(tmp_path, monkeypatch)
    calls = 0
    def clean_once():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("tracked worktree differs from registered HEAD")
    monkeypatch.setattr(supervisor, "_require_clean_tracked_tree", clean_once)
    result = supervisor.supervise(registration, evidence)
    assert calls == 2
    assert result["exit_code"] == 0
    assert "tracked worktree differs" in result["launch_error"]
    assert (evidence / "attestation.json").is_file()


def test_supervisor_rejects_child_effective_environment_mismatch(tmp_path, monkeypatch):
    runner, evidence, registration = _registration(tmp_path, monkeypatch)
    other = supervisor._measurement_canonical(dict(
        metrics=dict(effective_environment={"disposable": False})))
    original = runner.read_text(encoding="utf-8")
    line = next(row for row in original.splitlines() if "metrics.json" in row)
    _replace_runner(runner, registration, original.replace(line,
        f"(p/'metrics.json').write_bytes({other!r})"))
    result = supervisor.supervise(registration, evidence)
    assert result["exit_code"] == 0
    assert result["child_environment_match"] is False
    assert "child effective environment differs" in result["launch_error"]
    assert result["artifacts"]["metrics"] is not None
    assert (evidence / "attestation.json").is_file()


def test_supervisor_attests_claim_sync_failure_without_launch(tmp_path, monkeypatch):
    _, evidence, registration = _registration(tmp_path, monkeypatch)
    original = supervisor.os.fsync
    calls = 0
    def fail_once(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disposable claim sync failure")
        return original(fd)
    monkeypatch.setattr(supervisor.os, "fsync", fail_once)
    result = supervisor.supervise(registration, evidence)
    assert "disposable claim sync failure" in result["launch_error"]
    assert result["exit_code"] is None and result["rss_sample_count"] == 0
    assert (evidence / "execution.claim").is_file()
    assert not (evidence / "measurement").exists()
    assert (evidence / "attestation.json").is_file()
