"""Actual-model bridge discriminator and bounded state/refusal regression set."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from anyfesolver.general_contact.structural_state_bridge import TwoBodyStateBridge
from examples.elastic_sphere_panel_state_bridge import prepare_bridge_case, report_state_bridge
from examples.elastic_sphere_panel_continuation import _model_snapshot, _states_equal


@pytest.fixture(scope="module")
def prepared():
    return prepare_bridge_case()


def test_first_actual_model_discriminator(prepared):
    case, bridge = prepared
    report = report_state_bridge(case, bridge)
    assert report["status"] == "structural_state_bridge_prepared"
    assert report["sphere_unrestrained"] and report["rollback_exact"]
    assert len(case.sphere_map.faces) == 320 and len(case.panel_map.faces) == 100
    assert report["panel_reduced_dofs"] < report["panel_full_dofs"]
    assert any(b.rotating is not None for b in bridge._bodies)
    destination = os.environ.get("BRIDGE_REPORT")
    if destination:
        Path(destination).write_text(json.dumps(report, indent=2, allow_nan=False))


def test_exact_inventory_roundtrip_and_independent_body_changes(prepared):
    case, bridge = prepared
    state = bridge.accepted()
    a, b = bridge.unpack(state.displacement)
    assert np.array_equal(bridge.pack(case.sphere, a, case.panel, b), state.displacement)
    inventory = bridge.inventory()
    assert [row[-1] for row in inventory] == list(range(bridge.total_dofs))
    for label, node_id, local, dof, combined in inventory:
        model = case.sphere if label == "sphere" else case.panel
        assert model.mesh.nodes[node_id].dofs[local] == dof
        assert combined == dof + (0 if label == "sphere" else bridge.sizes[0])
    a[0] += .001
    packed = bridge.pack(case.sphere, a, case.panel, b)
    assert np.array_equal(packed[bridge.sizes[0]:], state.displacement[bridge.sizes[0]:])
    assert np.array_equal(bridge.accepted().displacement, state.displacement)


def test_existing_rotating_map_directional_derivative(prepared):
    _, bridge = prepared
    state = bridge.accepted()
    q = bridge.reduced_coordinates(state)
    direction = np.sin(np.arange(bridge.reduced_dofs) + .3)
    direction /= np.linalg.norm(direction)
    analytic = bridge.jacobian(state) @ direction
    for step in (1e-6, 3e-7):  # existing finite-skin fixed FD steps/relative criterion
        finite = (bridge.expand(q + step * direction) - bridge.expand(q - step * direction)) / (2*step)
        assert np.linalg.norm(finite-analytic) / np.linalg.norm(analytic) <= 1e-5


@pytest.mark.parametrize("ending", ["normal", "failure", "cancel"])
def test_disposable_trial_exact_restoration(prepared, ending):
    case, bridge = prepared
    before = bridge.accepted()
    models = (_model_snapshot(case.sphere), _model_snapshot(case.panel))
    try:
        with bridge.trial(np.zeros(bridge.reduced_dofs)) as trial:
            trial.displacement[:] += .002
            trial.sphere_history["probe"] = np.array([2.])
            trial.panel_history.clear()
            if ending == "failure":
                raise RuntimeError("rejected")
            if ending == "cancel":
                raise KeyboardInterrupt()
    except (RuntimeError, KeyboardInterrupt):
        pass
    after = bridge.accepted()
    assert np.array_equal(before.displacement, after.displacement)
    assert _states_equal(before.panel_history, after.panel_history)
    assert _states_equal(before.sphere_history, after.sphere_history)
    assert _states_equal(models, (_model_snapshot(case.sphere), _model_snapshot(case.panel)))


def test_foreign_states_models_and_vector_refusal(prepared):
    case, bridge = prepared
    state = bridge.accepted()
    foreign = copy.copy(state)
    foreign.owner = object()
    with pytest.raises(ValueError, match="foreign"):
        bridge.jacobian(foreign)
    with pytest.raises(ValueError, match="identity"):
        bridge.pack(case.panel, case.poses["near"], case.sphere, case.panel_q)
    for vector in (np.zeros((bridge.total_dofs, 1)), np.zeros(1), np.full(bridge.total_dofs, np.nan)):
        with pytest.raises(ValueError):
            bridge.unpack(vector)
    state.panel_history["bad"] = np.array([np.inf])
    with pytest.raises(ValueError, match="nonfinite"):
        bridge.pullback(state, np.zeros(bridge.total_dofs))


def test_stale_model_and_dof_refusal(prepared):
    case, bridge = prepared
    node = next(iter(case.sphere.mesh.nodes.values()))
    old_x = node.x
    try:
        node.x += .001
        with pytest.raises(ValueError, match="stale"):
            bridge.accepted()
    finally:
        node.x = old_x
    old_dofs = list(node.dofs)
    try:
        node.dofs[0], node.dofs[1] = node.dofs[1], node.dofs[0]
        with pytest.raises(ValueError, match="stale"):
            bridge.accepted()
    finally:
        node.dofs[:] = old_dofs
    bridge.accepted()


def test_invalid_accepted_constraints_refused(prepared):
    case, bridge = prepared
    panel = case.panel_q.copy()
    panel[bridge._bodies[1].info["fixed_dofs"][0]] += .001
    with pytest.raises(ValueError, match="violates"):
        TwoBodyStateBridge(case.sphere, case.panel, case.poses["near"], panel)
