"""Two-body coordinate/state bridge with the existing rotating panel ties.

Run with --output bridge.json. Gravity is prepared once. Applied vectors are
kinematic/virtual-work probes, not a pressure law or contact equilibrium.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.structural_state_bridge import TwoBodyStateBridge
from examples.elastic_sphere_panel_contact_work import prepare_work_case, evaluate_case
from examples.elastic_sphere_panel_continuation import _model_snapshot, _states_equal
from examples.elastic_sphere_stiffened_panel import _runtime_identity


def prepare_bridge_case():
    case = prepare_work_case()
    bridge = TwoBodyStateBridge(
        case.sphere, case.panel, case.poses["near"], case.panel_q,
        panel_history=dict(case.result.element_states))
    return case, bridge


def report_state_bridge(case=None, bridge=None):
    if case is None:
        case, bridge = prepare_bridge_case()
    before = (_model_snapshot(case.sphere), _model_snapshot(case.panel))
    state = bridge.accepted()
    sphere, panel = bridge.unpack(state.displacement)
    assert np.array_equal(sphere, case.poses["near"])
    assert np.array_equal(panel, case.panel_q)
    assert _states_equal(state.panel_history, dict(case.result.element_states))
    q = bridge.reduced_coordinates(state)
    reconstruction_error = float(np.max(np.abs(bridge.expand(q) - state.displacement)))
    assert reconstruction_error <= 1e-12  # existing s8 constraint postcheck
    work = evaluate_case(case, "near")
    force = bridge.pack(case.sphere, work.source_coordinate_work,
                        case.panel, work.target_coordinate_work)
    J = bridge.jacobian(state)
    direction = np.cos(np.arange(bridge.reduced_dofs, dtype=float))
    direction /= np.linalg.norm(direction)
    pulled = bridge.pullback(state, force)
    error = abs(float(force @ (J @ direction) - pulled @ direction))
    assert error <= 1e-13  # existing finite-skin virtual-work tolerance
    # Contact work on panel shells does not by itself exercise slave beams.
    # This separately labelled unit-norm generalized probe loads every DOF.
    slave_probe = np.sin(np.arange(bridge.total_dofs, dtype=float) + 1.)
    slave_probe /= np.linalg.norm(slave_probe)
    slave_pulled = bridge.pullback(state, slave_probe)
    slave_error = abs(float(slave_probe @ (J @ direction) - slave_pulled @ direction))
    assert slave_error <= 1e-13
    assert bridge.reduced_sizes[0] == bridge.sizes[0]  # no sphere restraints
    try:
        with bridge.trial() as trial:
            trial.displacement[:] = 0.
            trial.panel_history.clear()
            raise RuntimeError("declared rejected trial")
    except RuntimeError as error_message:
        if str(error_message) != "declared rejected trial":
            raise
    after = bridge.accepted()
    assert np.array_equal(after.displacement, state.displacement)
    assert _states_equal(after.panel_history, state.panel_history)
    assert _states_equal(before, (_model_snapshot(case.sphere), _model_snapshot(case.panel)))
    assert np.array_equal(case.panel_q, panel)
    return dict(
        status="structural_state_bridge_prepared", identity=_runtime_identity(),
        contact_response="unsupported", impact="unsupported", restart_checkpoint=False,
        interpretation="coordinate/state bridge and virtual work only; no contact law",
        target_drop_m=10., sphere_full_dofs=bridge.sizes[0], panel_full_dofs=bridge.sizes[1],
        sphere_reduced_dofs=bridge.reduced_sizes[0], panel_reduced_dofs=bridge.reduced_sizes[1],
        sphere_unrestrained=True, face_inventory=dict(sphere=320, panel=100),
        dof_inventory=bridge.inventory(),
        displacement=state.displacement.tolist(),
        constraint_reconstruction_maxabs=reconstruction_error,
        integrated_gap_work_error=error, generalized_slave_probe_work_error=slave_error,
        full_gap_work=force.tolist(), reduced_gap_work=pulled.tolist(),
        gravity_acceptance=case.acceptance, rollback_exact=True, model_preload_unchanged=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = report_state_bridge()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
