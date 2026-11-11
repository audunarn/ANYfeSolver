"""Stubbed CLI coverage for the friction physical-path example.

No solver execution: ``solve_case`` is replaced with stubbed results, so these
tests cover the command-line contract only (process/import time, not
numerical seconds, is their cost). Completed runs keep the exact legacy JSON
keys; noncompleted runs report the actual failure instead of losing it to a
secondary missing-checkpoint error.
"""
import copy
import json
import os
import sys

import pytest

import examples.general_contact_s3_friction_physical_path as cli
from anyfesolver.control import SolveCancelled

CHECKPOINT = {"schema": "ANYSOLVER_NONLINEAR_RESTART_V2", "version": 1,
              "displacements": [0.0, 1.0, 2.0],
              "path_state": {"mode": "contact_path"}}
PHASES = [{"step_index": 1, "reaction_generalized": [0.5, -0.25]}]


class StubResult:
    def __init__(self, status="completed", failure_reason=None,
                 checkpoint=CHECKPOINT):
        self.status = status
        self.info = {"force_displacement_history": copy.deepcopy(PHASES),
                     "failure_reason": failure_reason}
        self._checkpoint = checkpoint

    @property
    def failure_reason(self):
        return self.info.get("failure_reason")

    def to_restart_checkpoint(self):
        if self._checkpoint is None:
            raise ValueError(
                "this result does not contain a nonlinear restart checkpoint")
        return copy.deepcopy(self._checkpoint)

    def restart_checkpoint_bytes(self):
        return json.dumps(self.to_restart_checkpoint(), sort_keys=True,
                          separators=(",", ":")).encode()


def install_stub(monkeypatch, result=None, calls=None, error=None):
    def solve_case(case, *, through_step=7, restart=None):
        if calls is not None:
            calls.append((through_step, restart))
        if error is not None:
            raise error
        return result
    monkeypatch.setattr(cli, "solve_case", solve_case)


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv",
                        ["general_contact_s3_friction_physical_path"]
                        + [str(argument) for argument in argv])
    cli.main()


def test_completed_direct_checkpoint_and_legacy_invocation_match(monkeypatch, tmp_path):
    output = tmp_path / "completed.json"
    checkpoint_output = tmp_path / "checkpoint.json"
    calls = []
    install_stub(monkeypatch, result=StubResult(), calls=calls)
    run_main(monkeypatch, ["--through-step", "7", "--output", output,
                           "--checkpoint-output", checkpoint_output])
    assert calls == [(7, None)]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert list(document) == ["status", "physical_scope", "phases", "checkpoint"]
    assert document["status"] == "completed"
    assert document["physical_scope"] == \
        "translating_source_normal_constrained_elastic_target"
    assert document["phases"] == PHASES
    assert document["checkpoint"] == CHECKPOINT
    assert checkpoint_output.read_bytes() == \
        StubResult().restart_checkpoint_bytes()
    assert json.loads(checkpoint_output.read_text(encoding="utf-8")) == \
        document["checkpoint"]
    legacy = tmp_path / "legacy.json"
    run_main(monkeypatch, ["--output", legacy])
    assert legacy.read_text(encoding="utf-8") == output.read_text(encoding="utf-8")
    assert sorted(path.name for path in tmp_path.iterdir()) == \
        ["checkpoint.json", "completed.json", "legacy.json"]


def test_noncompleted_writes_truthful_json_and_exits_nonzero(monkeypatch, tmp_path,
                                                             capsys):
    output = tmp_path / "stopped.json"
    checkpoint_output = tmp_path / "checkpoint.json"
    install_stub(monkeypatch, result=StubResult(
        status="stopped_at_limit", failure_reason="contact_gap_exceeded",
        checkpoint=None))
    with pytest.raises(SystemExit) as raised:
        run_main(monkeypatch, ["--through-step", "3", "--output", output,
                               "--checkpoint-output", checkpoint_output])
    assert raised.value.code == 1
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == "stopped_at_limit"
    assert document["checkpoint"] is None
    assert document["failure_reason"] == "contact_gap_exceeded"
    assert document["phases"] == PHASES
    assert not checkpoint_output.exists()
    summary = capsys.readouterr().err
    assert "stopped_at_limit" in summary
    assert "contact_gap_exceeded" in summary
    assert str(output) in summary


def test_path_collisions_refused_before_solving(monkeypatch, tmp_path):
    restart = tmp_path / "prefix.json"
    restart.write_text(json.dumps(CHECKPOINT), encoding="utf-8")
    output = tmp_path / "out.json"
    calls = []
    install_stub(monkeypatch, result=StubResult(), calls=calls)
    collisions = (
        ["--restart", restart, "--output", restart],
        ["--restart", restart, "--output", output, "--checkpoint-output", restart],
        ["--output", output, "--checkpoint-output", output],
        ["--restart", restart, "--output", tmp_path / "." / "prefix.json"],
    )
    for argv in collisions:
        with pytest.raises(SystemExit, match="different files"):
            run_main(monkeypatch, argv)
    assert calls == []
    assert json.loads(restart.read_text(encoding="utf-8")) == CHECKPOINT
    assert not output.exists()


def test_cancelled_and_refused_input_write_no_output(monkeypatch, tmp_path):
    output = tmp_path / "out.json"
    calls = []
    install_stub(monkeypatch, calls=calls,
                 error=SolveCancelled("injected cancellation"))
    with pytest.raises(SolveCancelled, match="injected cancellation"):
        run_main(monkeypatch, ["--output", output])
    assert not output.exists() and calls == [(7, None)]
    bad_restart = tmp_path / "bad.json"
    bad_restart.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        run_main(monkeypatch, ["--restart", bad_restart, "--output", output])
    assert not output.exists() and calls == [(7, None)]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bad.json"]


def test_hardlinked_existing_paths_refused_before_solving(monkeypatch, tmp_path):
    restart = tmp_path / "prefix.json"
    original_bytes = json.dumps(CHECKPOINT).encode()
    restart.write_bytes(original_bytes)
    output = tmp_path / "out.json"
    checkpoint_output = tmp_path / "checkpoint.json"
    os.link(restart, output)
    os.link(restart, checkpoint_output)
    calls = []
    install_stub(monkeypatch, result=StubResult(), calls=calls)
    with pytest.raises(SystemExit, match="different files"):
        run_main(monkeypatch, ["--restart", restart, "--output", output,
                               "--checkpoint-output", checkpoint_output])
    assert calls == []
    assert restart.read_bytes() == original_bytes
    assert output.read_bytes() == original_bytes


def test_replacement_failure_after_temp_preserves_old_bytes(monkeypatch, tmp_path):
    output = tmp_path / "completed.json"
    old_bytes = b'{"status": "previous run"}\n'
    output.write_bytes(old_bytes)
    checkpoint_output = tmp_path / "checkpoint.json"
    install_stub(monkeypatch, result=StubResult())

    def failing_replace(source, destination):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(cli.os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        run_main(monkeypatch, ["--output", output,
                               "--checkpoint-output", checkpoint_output])
    assert output.read_bytes() == old_bytes
    assert not list(tmp_path.glob("*.tmp"))
    assert not checkpoint_output.exists()


def test_checkpoint_write_failure_reports_published_output(monkeypatch, tmp_path):
    output = tmp_path / "completed.json"
    missing_directory = tmp_path / "missing"
    checkpoint_output = missing_directory / "checkpoint.json"
    install_stub(monkeypatch, result=StubResult())
    with pytest.raises(SystemExit) as raised:
        run_main(monkeypatch, ["--output", output,
                               "--checkpoint-output", checkpoint_output])
    assert "published" in str(raised.value.code)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == "completed" and document["checkpoint"] == CHECKPOINT
    assert not missing_directory.exists()
    assert not list(tmp_path.glob("*.tmp"))
