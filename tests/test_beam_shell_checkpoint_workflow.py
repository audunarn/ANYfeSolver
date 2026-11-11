"""Public completed-prefix workflow; existing scientific tolerances retained."""
import copy
import json
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver import CancellationToken
from anyfesolver.nonlinear_restart import load_nonlinear_checkpoint
from anyfesolver.constraint_audit import constraint_residual_summary
from examples import beam_shell_coupling as example
from test_nonlinear_beam_shell_slice3_continuation import (
    _assert_states_allclose, _independent_equilibrium,
)

ROOT = Path(__file__).resolve().parents[1]


def _cli(folder, name, coupling, *flags):
    command = [sys.executable, str(ROOT / "examples/beam_shell_coupling.py"),
               "--coupling", coupling, "--case", "plastic", *map(str, flags)]
    start = time.perf_counter()
    with (folder / (name + ".stdout.txt")).open("wb") as out, (
            folder / (name + ".stderr.txt")).open("wb") as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
        try:
            code = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=False)
            else:
                process.kill()
            process.wait()
            raise
        finally:
            (folder / (name + ".command.json")).write_text(json.dumps(dict(
                command=command, exit_code=process.returncode,
                timeout_seconds=240, wall_seconds=time.perf_counter()-start)))
    assert code == 0, (folder / (name + ".stderr.txt")).read_text()
    payload = json.loads((folder / (name + ".stdout.txt")).read_text())
    assert payload["completed"] and payload["status"] == "completed"
    assert payload["constraint_postcheck"]["status"] == "passed"
    assert payload["control_error"] <= 1e-8
    return payload


@pytest.fixture(scope="module", params=("direct", "interpolated"))
def completed_path(request, tmp_path_factory):
    coupling = request.param
    folder = tmp_path_factory.mktemp("checkpoint_" + coupling)
    paths = {name: folder / (name + ".json") for name in ("full", "prefix", "resumed")}
    prior = os.environ.get("BEAM_SHELL_VERIFIED_PREFIX_EVIDENCE")
    if prior:
        # Impact-based retry only: the repair changes bound-state admission,
        # which fresh full/prefix runs do not enter. Preserve their measured
        # successful processes and rerun affected serialized resume paths.
        source = Path(prior) / ("checkpoint_" + coupling + "0")
        reused = {}
        for name in ("full", "prefix"):
            receipt = json.loads((source / (name + ".command.json")).read_text())
            assert receipt["exit_code"] == 0
            for suffix in (".json", ".stdout.txt", ".stderr.txt", ".command.json"):
                original = source / (name + suffix)
                shutil.copyfile(original, folder / original.name)
                reused[str(original)] = hashlib.sha256(original.read_bytes()).hexdigest()
        (folder / "reused-evidence.json").write_text(json.dumps(reused, indent=2))
        full, prefix = (json.loads((folder / (name + ".stdout.txt")).read_text())
                        for name in ("full", "prefix"))
        assert full["completed"] and prefix["completed"]
    else:
        full = _cli(folder, "full", coupling, "--checkpoint-out", paths["full"])
        prefix = _cli(folder, "prefix", coupling, "--stage", "prefix",
                      "--checkpoint-out", paths["prefix"])
    before = paths["prefix"].read_bytes()
    resumed = _cli(folder, "resumed", coupling, "--restart", paths["prefix"],
                   "--checkpoint-out", paths["resumed"])
    assert paths["prefix"].read_bytes() == before
    return dict(coupling=coupling, folder=folder, paths=paths,
                summaries=dict(full=full, prefix=prefix, resumed=resumed))


def _read(path):
    return load_nonlinear_checkpoint(path.read_bytes())


def _states(payload):
    return {record["element_id"]: record["state"] for record in payload["element_states"]}


def test_separate_process_checkpoint_matches_uninterrupted_history(completed_path):
    run = completed_path
    full, resumed = (_read(run["paths"][name]) for name in ("full", "resumed"))
    np.testing.assert_allclose(full["displacements"], resumed["displacements"],
                               rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(run["summaries"]["full"]["achieved_load_factor"],
                               run["summaries"]["resumed"]["achieved_load_factor"],
                               rtol=1e-6, atol=1e-9)
    _assert_states_allclose(_states(full), _states(resumed))
    for name, target in (("full", -11/60), ("prefix", -.10), ("resumed", -11/60)):
        summary = run["summaries"][name]
        assert abs(summary["achieved_control"] - target) <= 1e-8
        states = _states(_read(run["paths"][name]))
        assert max(np.max(np.asarray(state["alpha"])) for state in states.values()
                   if isinstance(state, dict) and "alpha" in state) > 0


def test_saved_states_independently_reconstruct_equilibrium(completed_path):
    run = completed_path
    case = example.build_model(run["coupling"])
    grid = {(ix, iy): 1 + ix*2 + iy for ix in range(3) for iy in range(2)}
    for name in ("full", "prefix", "resumed"):
        payload = _read(run["paths"][name])
        result = SimpleNamespace(displacements=np.asarray(payload["displacements"]),
                                 element_states=_states(payload),
                                 load_factor=run["summaries"][name]["achieved_load_factor"])
        record = dict(model=case.model, load=case.load, grid=grid, result=result,
                      coupling=run["coupling"])
        before = copy.deepcopy(payload)
        oracle = _independent_equilibrium(record)
        assert oracle["relative_residual"] <= 1e-7
        assert np.linalg.norm(oracle["balance"]) <= 1e-6 * max(
            np.linalg.norm(oracle["applied_wrench"]), 1.)
        # Stored global translations contain the finite offset rotation; the
        # solver's constraint certificate uses the matching kinematic pullback.
        assert run["summaries"][name]["constraint_postcheck"]["status"] == "passed"
        assert payload == before


def test_cancelled_resume_preserves_completed_prefix(completed_path):
    run = completed_path
    path = run["paths"]["prefix"]
    before = path.read_bytes()
    checkpoint = load_nonlinear_checkpoint(before)
    original = copy.deepcopy(checkpoint)
    token = CancellationToken()

    def cancel(event):
        if event.get("type") == "nonlinear_static_step":
            token.cancel("cancel resumed path after accepted increment")

    result = example.run_plastic(run["coupling"], restart_checkpoint=checkpoint,
        checkpoint_out=path, cancellation_token=token, progress_callback=cancel)
    assert result["status"] == "cancelled" and not result["completed"]
    assert result["accepted_increments"] == 1
    assert not result["checkpoint_written"]
    assert path.read_bytes() == before and checkpoint == original


def test_rejected_trials_and_failed_terminal_restart(completed_path, monkeypatch):
    run = completed_path
    prefix_path = run["paths"]["prefix"]
    checkpoint = _read(prefix_path)
    original = copy.deepcopy(checkpoint)
    before = prefix_path.read_bytes()
    actual_solver = example.solve_static_nonlinear
    observed = []

    def observe(*args, **kwargs):
        result = actual_solver(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(example, "solve_static_nonlinear", observe)
    # A public initial-state run permits intentionally different iteration
    # controls; a serialized restart correctly refuses that contract change.
    case = example.build_model(run["coupling"])
    u_input = np.asarray(checkpoint["displacements"])
    state_input = _states(checkpoint)
    states_before = copy.deepcopy(state_input)
    result = actual_solver(case.model, case.load, control="displacement",
        displacement_control=example.DisplacementControl(node_id=case.probe_node_id,
            dof="uz", target_displacement=-11/60), kinematics="corotational",
        num_steps=1, max_iterations=1, tolerance=1e-8,
        initial_displacements=u_input, initial_element_states=state_input,
        equilibrate_initial_state=False, record_increment_snapshots=True,
        emit_restart_checkpoint=True)
    assert result.status != "completed"
    assert len(result.snapshots) == len(result.steps)
    np.testing.assert_allclose(result.displacements, np.asarray(checkpoint["displacements"]),
                               rtol=1e-14, atol=1e-15)
    assert state_input == states_before
    assert checkpoint == original and prefix_path.read_bytes() == before
    failed = result.restart_checkpoint_bytes()
    monkeypatch.setattr(example, "DC_MAX_ITERATIONS", 1)
    refused = example.run_plastic(run["coupling"], restart_checkpoint=failed,
                                 checkpoint_out=prefix_path)
    assert refused["status"] == "refused" and not refused["completed"]
    assert "continuable completed static target" in refused["stop_reason"]
    assert prefix_path.read_bytes() == before


@pytest.mark.parametrize("flags", (("--restart", "x"), ("--stage", "prefix"),
    ("--checkpoint-out", "x"), ("--case", "plastic", "--stage", "prefix", "--restart", "x")))
def test_incompatible_cli_flags_refused_before_solve(flags):
    with pytest.raises(SystemExit) as error:
        example.main(list(flags))
    assert error.value.code == 2
