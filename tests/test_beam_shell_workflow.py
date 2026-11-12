"""Focused public workflow tests for the standalone beam-shell example.

These are integration tests of the registered pre-yield public workflow in
``examples/beam_shell_coupling.py``: actual public solves for both coupling
laws, independent true-elastic controls for both coupling laws, cooperative
cancellation with and without accepted increments, and truthful CLI
JSON/exit-status behaviour.
They are routine contract checks, not scientific qualification.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from anyfesolver import (
    CancellationToken,
    CoupledBeamShellElement,
    InterpolatedBeamShellMPCElement,
    solve_static_nonlinear,
)
from anyfesolver.elements import BeamElement

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "beam_shell_coupling.py"

_CACHE: dict = {}


def _example_module():
    return runpy.run_path(str(EXAMPLE))


def _cached_payload(coupling: str) -> dict:
    if coupling not in _CACHE:
        _CACHE[coupling] = _example_module()["run"](coupling)
    return _CACHE[coupling]


@pytest.mark.parametrize("coupling", ("direct", "interpolated"))
def test_build_model_registered_structure(coupling):
    example = _example_module()
    case = example["build_model"](coupling)
    model = case.model
    elements = list(model.mesh.elements.values())
    shells = [
        e for e in elements
        if not isinstance(e, (BeamElement, CoupledBeamShellElement,
                             InterpolatedBeamShellMPCElement))
    ]
    assert len(shells) == 2
    beams = [e for e in elements if isinstance(e, BeamElement)]
    direct = [e for e in elements if isinstance(e, CoupledBeamShellElement)]
    interpolated = [
        e for e in elements if isinstance(e, InterpolatedBeamShellMPCElement)
    ]
    n_beam = 2 if coupling == "direct" else 3
    assert len(beams) == n_beam
    assert len(direct) + len(interpolated) == n_beam + 1
    if coupling == "direct":
        assert len(direct) == 3 and not interpolated
        beam_x = sorted(float(model.mesh.nodes[n].x) for n in (100, 101, 102))
        assert beam_x == [0.0, 0.5, 1.0]
    else:
        assert len(direct) == 2 and len(interpolated) == 2
        beam_x = sorted(float(model.mesh.nodes[n].x) for n in (100, 101, 102, 103))
        assert beam_x == [0.0, pytest.approx(1.0 / 3.0), pytest.approx(2.0 / 3.0), 1.0]
        for element in interpolated:
            weights = np.asarray(element.shape_weights, dtype=float)
            assert len(element.shell_node_ids) == 2
            assert float(weights.sum()) == pytest.approx(1.0)
            assert np.all(weights > 0.0)
    # Registered tip load: -2000 N uz on every shell-tip node, nothing else.
    assert case.probe_node_id == 5
    assert set(case.load.nodal_loads) == {5, 6}
    for value in case.load.nodal_loads.values():
        np.testing.assert_array_equal(value, [0.0, 0.0, -2000.0, 0.0, 0.0, 0.0])
    # Registered material: canonical S355 curve with a hardening law.
    material = model.get_material("steel")
    assert material.hardening_curve is not None
    assert material.elastic_modulus == pytest.approx(2.1e11)
    assert material.density == pytest.approx(7850.0)


@pytest.mark.parametrize("coupling", ("direct", "interpolated"))
def test_run_actual_public_solve_completed_payload(coupling):
    payload = _cached_payload(coupling)
    assert payload["coupling"] == coupling
    assert payload["status"] == "completed"
    assert payload["stop_reason"] == "target_load_factor_reached"
    assert payload["requested_load_factor"] == 1.0
    assert abs(payload["achieved_load_factor"] - 1.0) <= 1.0e-8
    assert payload["accepted_increments"] == 6
    postcheck = payload["constraint_postcheck"]
    assert postcheck["status"] == "passed"
    assert (
        float(postcheck["max_relative_residual"])
        <= float(postcheck["relative_tolerance"])
    )
    probe = payload["probe_displacement"]
    assert np.isfinite(probe) and probe < 0.0
    assert payload["completed"] is True
    # The JSON document is serializable and keeps the same truth values.
    round_trip = json.loads(json.dumps(payload))
    assert round_trip["completed"] is True
    assert round_trip["status"] == "completed"


@pytest.mark.parametrize("coupling", ("direct", "interpolated"))
def test_probe_matches_true_elastic_control(coupling):
    """Independent control per coupling law: a distinct linear-elastic
    material (no hardening curve, same geometry and controls) must
    reproduce that coupling's example probe displacement, which is only
    possible while the response is genuinely pre-yield."""
    example = _example_module()
    payload = _cached_payload(coupling)
    case = example["build_model"](coupling)
    case.model.add_material("steel", 2.1e11, 0.3, density=7850.0)
    assert case.model.get_material("steel").hardening_curve is None
    result = solve_static_nonlinear(
        case.model,
        case.load,
        kinematics="corotational",
        num_steps=example["NUM_STEPS"],
        max_iterations=example["MAX_ITERATIONS"],
        tolerance=example["TOLERANCE"],
    )
    assert result.status == "completed"
    dofs = case.model.mesh.dof_manager.get_node_dofs(case.probe_node_id)
    elastic_probe = float(np.asarray(result.displacements)[dofs[2]])
    relative_error = abs(payload["probe_displacement"] - elastic_probe) / abs(
        elastic_probe
    )
    assert relative_error <= 1.0e-7


def test_run_cancelled_before_any_increment_reports_empty_history():
    example = _example_module()
    token = CancellationToken()
    token.cancel("cancelled before the solve starts")
    payload = example["run"]("direct", cancellation_token=token)
    assert payload["status"] == "cancelled"
    assert payload["stop_reason"] == "cancelled before the solve starts"
    assert payload["accepted_increments"] == 0
    assert payload["achieved_load_factor"] is None
    assert payload["probe_displacement"] is None
    assert payload["constraint_postcheck"] is None
    assert payload["completed"] is False
    assert json.loads(json.dumps(payload))["completed"] is False


def test_run_solve_cancelled_after_first_accepted_event_maps_to_noncompletion():
    """run() must map an actual public SolveCancelled raised after the first
    accepted increment to non-completion JSON with the accepted history and
    no success claim. The registered force-controlled route checks the
    cancellation token at every Newton iteration, so cancelling from the
    public progress callback stops the real solve after one accepted
    increment; no solver behaviour is substituted here."""
    example = _example_module()
    token = CancellationToken()

    def cancel_after_first_accepted(event):
        if (
            event.get("type") == "nonlinear_static_step"
            and not token.is_cancelled
        ):
            token.cancel("cancelled after one accepted increment")

    payload = example["run"](
        "direct",
        cancellation_token=token,
        progress_callback=cancel_after_first_accepted,
    )
    assert payload["status"] == "cancelled"
    assert payload["stop_reason"] == "cancelled after one accepted increment"
    assert payload["accepted_increments"] == 1
    assert payload["achieved_load_factor"] == pytest.approx(1.0 / 6.0)
    assert payload["probe_displacement"] is None
    assert payload["constraint_postcheck"] is None
    assert payload["completed"] is False
    assert json.loads(json.dumps(payload))["completed"] is False


def test_cli_stdout_json_exit_status_and_output_file(capsys, tmp_path):
    scope = runpy.run_path(str(EXAMPLE))
    main = scope["main"]
    completed_payload = {
        "coupling": "interpolated",
        "status": "completed",
        "stop_reason": "target_load_factor_reached",
        "requested_load_factor": 1.0,
        "achieved_load_factor": 1.0,
        "probe_displacement": -1.0e-3,
        "accepted_increments": 6,
        "constraint_postcheck": {"status": "passed"},
        "completed": True,
    }
    stopped_payload = dict(completed_payload, status="stopped_at_limit",
                           completed=False, achieved_load_factor=0.5)
    recorded = []

    def fake_run(coupling=None, **kwargs):
        recorded.append(coupling)
        return completed_payload if len(recorded) == 1 else stopped_payload

    # runpy.run_path returns a copy of the module namespace; main.__globals__
    # is the live namespace the CLI actually reads.
    main.__globals__["run"] = fake_run
    output_path = tmp_path / "summary.json"
    assert main(["--output", str(output_path)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first == completed_payload
    assert json.loads(output_path.read_text(encoding="utf-8")) == completed_payload
    assert main(["--coupling", "direct"]) == 1
    second = json.loads(capsys.readouterr().out)
    assert second == stopped_payload
    assert recorded == ["interpolated", "direct"]


def test_cli_rejects_unknown_coupling():
    scope = runpy.run_path(str(EXAMPLE))
    with pytest.raises(SystemExit) as excinfo:
        scope["main"](["--coupling", "glued"])
    assert excinfo.value.code == 2
