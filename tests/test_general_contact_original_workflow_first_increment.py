"""Actual first ORIGINAL increment through the unchanged production caller.

The adopted v2 initial-contact seed selection
(``linearized_kkt_merit_globalization_dev``, revision
``origin_linearized_kkt_seed_merit_globalization_v2``) is captured as
ADMITTED on the exact original fixture by
``test_general_contact_original_workflow_seed_decision.py`` (full
proposal, admissible fraction 1.0, QP terminal ``optimal`` in 5
iterations, seed merit 0.0781887807266064 < origin merit 0.125). This
module runs the actual ordinary first increment of the SAME original
8-macro programme - registered step 1, ``max_load_factor=0.125`` ->
``requested_step=1`` - through the unchanged production caller with no
observer hook, no cancellation and no gate change, using the existing
owned helpers ``_original_case``/``_seeded`` from the seed-decision
module. The actual returned result is captured into the guard TEMP
BEFORE any assertion: status/accepted step, displacements, contact
summary, checkpoint presence, and the normalized production gate
quantities (force balance relative to the unchanged 0.018 N
complementarity scale, moment balance relative to the separate
dimensional moment scale Fref*1m = 0.018 N m under the same numerical
criterion, the independent physical gap bound against the unchanged
50 um budget, complete positive finite 25 pressure rows, finite
300-coordinate accepted state, the production gap tolerance read as
plain truthiness so an actual ``numpy.bool_(True)`` is not a test type
failure, and programme identity). The same call requests
``emit_restart_checkpoint=True`` so the accepted state carries a real
public producer checkpoint, which is captured in full and required to
exist. The production all-gates predicate is preserved:
every recorded gate must hold; none is relaxed, replaced or averaged.
This is the requested FIRST INCREMENT ONLY: it is not a complete 8-step
trajectory, not a runtime improvement claim, and not installed or
engineering acceptance.
"""
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact import semismooth_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEED_DECISION_MODULE = (Path(__file__).resolve().parent
                        / "test_general_contact_original_workflow_seed_decision.py")
INDEPENDENT_GAP_MODULE = (REPOSITORY_ROOT
                          / "examples"
                          / "general_contact_d1_interior_controls_dev.py")
CAPTURE_NAME = "original-workflow-first-increment-v1.json"
INPUT_FREE_CAPTURE_NAME = "original-workflow-first-increment-input-free-v1.json"
V2_SELECTION = "linearized_kkt_merit_globalization_dev"


def _missing_original_inputs():
    owned = _load_module("original_workflow_seed_decision_owned_helpers",
                         SEED_DECISION_MODULE)
    return owned._missing_original_inputs()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_array(summary, key):
    value = summary.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if np.all(np.isfinite(array)) else None


@pytest.mark.skipif(bool(_missing_original_inputs()),
                    reason='exact original fixture inputs absent: '
                           + ', '.join(_missing_original_inputs()))
def test_actual_first_original_increment_completes_under_production_gates():
    owned = _load_module("original_workflow_seed_decision_owned_helpers",
                         SEED_DECISION_MODULE)
    independent_gap_bound = _load_module(
        "original_workflow_independent_gap_reference",
        INDEPENDENT_GAP_MODULE)._independent_projected_gap_bound
    model, definition, settings, program = owned._original_case()
    seeded = owned._seeded(settings, V2_SELECTION)
    program_contract_before = program.contract()
    boundary_before = repr(model.boundary_conditions)
    constraints_before = repr(model.constraint_equations)
    # Actual ordinary solve: no observer hook, no cancellation, no gate
    # change; the unchanged original 8-macro programme, first registered
    # step only.
    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    result = solve_static_nonlinear(
        model, contact=definition, contact_settings=seeded,
        contact_path_program=program, max_iterations=18,
        max_load_factor=0.125, emit_restart_checkpoint=True)
    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    # Actual returned fields, read only as needed.
    u = np.asarray(result.displacements, dtype=float)
    info = result.info
    contact_path = dict(info.get("contact_path") or {})
    accepted_step = contact_path.get("accepted_step")
    history = list(info.get("force_displacement_history") or [])
    contact_summary = dict(info.get("general_contact") or {})
    budget = float(definition.interactions[0].gap_tolerance)
    force_scale = float(settings.complementarity_force_scale_n)
    # Separate dimensional moment scale Fref*1m = 0.018 N m; the moment
    # balance gate keeps the same numerical 1e-6 criterion.
    moment_scale = force_scale * 1.0
    support = _finite_array(contact_summary, "reference_support_m2")
    gaps = _finite_array(contact_summary, "normalized_weighted_gap_m")
    pressure = _finite_array(contact_summary, "numerical_pressure_pa")
    rows_complete = (support is not None and gaps is not None
                     and pressure is not None
                     and support.shape == gaps.shape == pressure.shape == (25,)
                     and np.all(support > 0.0))
    integration_force = _finite_array(contact_summary, "integration_force")
    target_force = _finite_array(contact_summary, "target_force")
    integration_moment = _finite_array(contact_summary, "integration_moment")
    target_moment = _finite_array(contact_summary, "target_moment")
    force_balance_relative = (
        float(np.linalg.norm(integration_force + target_force))/force_scale
        if integration_force is not None and target_force is not None
        else None)
    moment_balance_relative = (
        float(np.linalg.norm(integration_moment + target_moment))/moment_scale
        if integration_moment is not None and target_moment is not None
        else None)
    independent_gap_lower_m = independent_gap_bound(
        model, definition, u, budget)
    identity_unchanged = (program.contract() == program_contract_before
                          and repr(model.boundary_conditions) == boundary_before
                          and repr(model.constraint_equations)
                          == constraints_before)
    gates = {
        "requested_first_increment_completed": bool(
            result.status == "completed"
            and accepted_step == 1
            and len(result.steps) == 1
            and result.steps[0].step_index == 1
            and len(history) == 1
            and history[0]["step_index"] == 1),
        "finite_300_coordinate_accepted_state": bool(
            u.shape == (300,) and np.all(np.isfinite(u))),
        "production_gap_tolerance_met": bool(
            contact_summary.get("gap_tolerance_met")),
        "complete_positive_finite_25_pressure_rows": bool(rows_complete),
        "force_balance_original_gate": bool(
            force_balance_relative is not None
            and force_balance_relative <= 1.0e-6),
        "moment_balance_original_gate": bool(
            moment_balance_relative is not None
            and moment_balance_relative <= 1.0e-6),
        "independent_physical_gap_original_gate": bool(
            independent_gap_lower_m >= -budget),
        "restart_checkpoint_produced": bool(
            result.restart_checkpoint is not None),
        "programme_identity_unchanged": bool(identity_unchanged),
    }
    capture = {
        "schema": "ORIGINAL_WORKFLOW_FIRST_INCREMENT_V1",
        "selection": V2_SELECTION,
        "requested_increment": {
            "max_load_factor": 0.125,
            "requested_registered_step": 1,
            "total_registered_steps": 8,
            "max_iterations": 18,
        },
        "status": result.status,
        "status_category": result.status_category,
        "failure_reason": result.failure_reason,
        "stop_reason": result.stop_reason,
        "load_factor": float(result.load_factor),
        "peak_load_factor": float(result.peak_load_factor),
        "last_converged_load_factor": float(result.last_converged_load_factor),
        "steps": [step.to_dict() for step in result.steps],
        "accepted_step": accepted_step,
        "contact_path": contact_path,
        "displacements": u.tolist(),
        "displacement_norm_m": float(np.linalg.norm(u)),
        "history": history,
        "contact_summary": contact_summary,
        "restart_checkpoint": owned._plain(result.restart_checkpoint),
        "emit_restart_checkpoint_requested": True,
        "normalized_gate_metrics": {
            "force_scale_n": force_scale,
            "moment_scale_nm": moment_scale,
            "gap_budget_m": budget,
            "force_balance_relative": force_balance_relative,
            "moment_balance_relative": moment_balance_relative,
            "independent_gap_lower_m": (
                float(independent_gap_lower_m)
                if np.isfinite(independent_gap_lower_m)
                else str(float(independent_gap_lower_m))),
        },
        "gates": gates,
        "all_gates_predicate": bool(all(gates.values())),
        "provenance": owned._provenance(seeded, program),
        "scope": ("actual first ORIGINAL increment only (registered step 1 "
                  "of the unchanged 8-macro programme); not a complete "
                  "8-step trajectory, not a runtime improvement claim, "
                  "not installed/engineering acceptance"),
    }
    # Capture the actual returned evidence BEFORE any assertion.
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(owned._plain(capture), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    # Assertions: actual requested first increment completion and finite
    # state, then the preserved production all-gates predicate.
    assert result.status == "completed", result.status
    assert accepted_step == 1, accepted_step
    assert contact_path.get("total_steps") == 8
    assert len(result.steps) == 1 and result.steps[0].step_index == 1
    assert result.steps[0].load_factor == 0.125
    assert result.load_factor == 0.125
    assert len(history) == 1 and history[0]["step_index"] == 1
    assert result.restart_checkpoint is not None, "no producer checkpoint"
    assert identity_unchanged
    failed = {name: value for name, value in gates.items() if not value}
    assert not failed, f"production gates failed: {failed}"


def test_input_free_first_increment_completes_under_production_gates():
    """Input-free first increment on the committed small-patch mixed
    active/split case: no external fixture input, no consumed registered
    authority script, and no original-fixture identity claim. The
    production gates that apply to this case are checked unchanged; the
    fixture-specific 300-coordinate/25-row/0.018 N gates remain
    exclusively in the guarded original node above."""
    owned = _load_module("original_workflow_seed_decision_owned_helpers",
                         SEED_DECISION_MODULE)
    from test_general_contact_admission_globalization_driver_small_patch import (
        _case as small_patch_case, _enabled as small_patch_enabled, _solve)
    model, definition, settings, program = small_patch_case()
    seeded = owned._seeded(small_patch_enabled(settings), V2_SELECTION)
    program_contract_before = program.contract()
    boundary_before = repr(model.boundary_conditions)
    constraints_before = repr(model.constraint_equations)
    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    result = _solve(model, definition, seeded, program,
                    max_load_factor=0.5, emit_restart_checkpoint=True)
    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    u = np.asarray(result.displacements, dtype=float)
    contact_summary = dict(result.info.get("general_contact") or {})
    seed_record = dict(result.info.get("contact_initial_seed") or {})
    identity_unchanged = (program.contract() == program_contract_before
                          and repr(model.boundary_conditions) == boundary_before
                          and repr(model.constraint_equations)
                          == constraints_before)
    gates = {
        "requested_first_increment_completed": bool(
            result.status == "completed" and len(result.steps) == 1),
        "finite_accepted_state": bool(np.all(np.isfinite(u))),
        "production_gap_tolerance_met": bool(
            contact_summary.get("gap_tolerance_met")),
        "seed_selection_recorded": bool(
            seed_record.get("selection") == V2_SELECTION
            and seed_record.get("outcome") in {"admitted", "refused"}),
        "restart_checkpoint_produced": bool(
            result.restart_checkpoint is not None),
        "programme_identity_unchanged": bool(identity_unchanged),
    }
    capture = {
        "schema": "ORIGINAL_WORKFLOW_FIRST_INCREMENT_INPUT_FREE_V1",
        "selection": V2_SELECTION,
        "status": result.status,
        "status_category": result.status_category,
        "failure_reason": result.failure_reason,
        "steps": [step.to_dict() for step in result.steps],
        "displacement_norm_m": float(np.linalg.norm(u)),
        "contact_summary": contact_summary,
        "seed_record": seed_record,
        "gates": gates,
        "all_gates_predicate": bool(all(gates.values())),
        "provenance": owned._provenance(seeded, program),
        "scope": ("input-free first increment on the committed small-patch "
                  "mixed active/split case; not the original fixture, "
                  "not a complete trajectory claim"),
    }
    # Capture the actual returned evidence BEFORE any assertion.
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), INPUT_FREE_CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(owned._plain(capture), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    assert result.status == "completed", result.failure_reason
    assert len(result.steps) == 1
    assert result.restart_checkpoint is not None, "no producer checkpoint"
    assert identity_unchanged
    failed = {name: value for name, value in gates.items() if not value}
    assert not failed, f"production gates failed: {failed}"
