"""First actual initializer decisions on the unchanged original 300-coordinate fixture.

This module captures the production first decisions of the opt-in
initial-contact seed on the exact original fixture (16 Q4 source / 32 S3
target / 300 coordinates / fine 25 Q1 pressure rows / 0.018 N / 8 macro
steps / 50 um gap budget), through the private
``semismooth_path._INITIAL_SEED_DECISION_HOOK`` observer. The hook receives
the unchanged production decision payload after the actual admission
predicates ran (actual-mechanics candidate reevaluation, conservative swept
admissibility, full scaled merit comparison) and before any Newton
iteration; each capture then raises ``SolveCancelled`` through the
proven-propagating channel so the solve stops with the origin untouched.

Two selections are captured. The ``linearized_kkt_dev`` selection records
the unchanged v1 behavior (on this fixture: the internal bounded QP
refusal ``qp_iteration_cap``). The
``linearized_kkt_merit_globalization_dev`` selection records the adopted
v2 candidate: the anti-cycling QP release alternatives plus the bounded
backtracking of the complete algorithmic correction as a disposable
initial Newton iterate, with every trial reevaluated through the same
actual gates. The captures record truthfully: the exact origin and
proposed bound q/reactions/z, the live origin/candidate normalized gaps,
the candidate residual/control/NCP values, the actual swept-certificate
fractions, the actual scaled merits, the production outcomes, and the
numerical source/effective runtime provenance, all written to the guard
TEMP. Coherent rollback is checked by an immediate identical rerun: the
same decision must recur bitwise (modulo timing), proving no partial
state survived the cancellation, and the model/program identity must be
unchanged. No acceptance gate is evaluated beyond the production ones and
no trajectory claim is made.
"""
import importlib.util
import json
import os
import tempfile
import platform
from pathlib import Path

import numpy as np
import pytest
import scipy

from anyfesolver import solve_static_nonlinear
from anyfesolver.control import SolveCancelled
from anyfesolver.general_contact import semismooth_path
from anyfesolver.general_contact.core import _contact_settings_contract

FIXTURE_DIR = Path("C:/ANYworkspaceWork/contact_joint_dual_full8_20261010")
DESCRIPTOR_PATH = FIXTURE_DIR / "force-full.json"
V1_CAPTURE_NAME = "original-workflow-seed-decision-v1.json"
V2_CAPTURE_NAME = "original-workflow-seed-decision-v2.json"


def _missing_original_inputs():
    """Exact real external inputs the original fixture nodes require."""
    return [str(path) for path in (FIXTURE_DIR / "candidate_fixture.py",
                                   DESCRIPTOR_PATH)
            if not path.is_file()]


def _original_case():
    spec = importlib.util.spec_from_file_location(
        "candidate_fixture", FIXTURE_DIR / "candidate_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptor = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    model, definition, settings, program = module.build_candidate(descriptor)
    # Exact original fixture identity; nothing may change these.
    assert model.mesh.dof_manager.total_dofs == 300
    assert len(model.mesh.elements) == 48
    assert len(definition.surfaces[0].element_ids) == 16
    assert len(definition.surfaces[1].element_ids) == 32
    assert program.total_steps == 8
    assert np.array_equal(program.target_at(8), (0.04, 0.00008, 0.0026))
    assert np.array_equal(program.target_at(1),
                          np.asarray((0.04, 0.00008, 0.0026))/8.0)
    assert settings.d1_interior_profile
    assert settings.complementarity_force_scale_n == 0.018
    assert settings.shell_skin_kinematics == "reference_linear"
    assert definition.interactions[0].gap_tolerance == 5.0e-5
    names = [bc.name for bc in model.boundary_conditions]
    assert names.count("target_outer_physical_clamp") == 1
    assert "source_left_bending_clamp" in names and "source_lateral_guide" in names
    return model, definition, settings, program


def _seeded(settings, selection):
    from dataclasses import replace
    return replace(settings, initial_contact_seed=selection)


def _provenance(seeded, program):
    runtime = os.environ.get("AUTOMATIC_RUNTIME_RESULT")
    record = {"python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__}
    if runtime:
        root = Path(runtime).parent
        install = root / "install.json"
        if install.is_file():
            with open(install, encoding="utf-8") as handle:
                manifest = json.load(handle)
            record["source_manifest"] = {
                "mode": manifest.get("mode"),
                "payload_files": len(manifest.get("payload_sha256", {})),
                "payload_sha256": manifest.get("payload_sha256")}
        start = root / "runtime-start.json"
        if start.is_file():
            with open(start, encoding="utf-8") as handle:
                observed = json.load(handle)
            record["effective_runtime"] = {
                "jit_disabled": observed.get("jit_disabled"),
                "numba_threads": observed.get("numba_threads"),
                "pools": [{k: p.get(k) for k in ("user_api", "num_threads")}
                          for p in observed.get("pools", [])]}
    record["settings_contract"] = _contact_settings_contract(seeded)
    record["program_contract"] = program.contract()
    return record


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _decision_capture(payload):
    decision = payload["record"]
    capture = {"schema": "ORIGINAL_WORKFLOW_SEED_DECISION_V1",
               "selection": decision["selection"],
               "outcome": decision["outcome"],
               "reason": decision.get("reason"),
               "decision_record": {key: value for key, value
                                   in decision.items() if key != "cost_s"},
               "origin_q": np.asarray(payload["origin_q"]).tolist(),
               "origin_reactions":
                   np.asarray(payload["origin_reactions"]).tolist(),
               "origin_nodal_forces":
                   np.asarray(payload["origin_nodal_forces"]).tolist()}
    for key in ("origin_normalized_gap", "origin_residual_n",
                "origin_constraint", "proposed_q", "proposed_reactions",
                "proposed_nodal_forces", "candidate_normalized_gap",
                "candidate_residual_n", "candidate_constraint",
                "candidate_ncp_n", "admitted_q", "admitted_reactions",
                "admitted_nodal_forces"):
        if key in payload:
            capture[key] = np.asarray(payload[key]).tolist()
    for key in ("origin_merit", "admissible_fraction", "seed_merit",
                "candidate_merit"):
        if key in decision or key in payload:
            capture[key] = float(decision.get(key, payload.get(key)))
    return capture


def _cancelled_seed_decision(model, definition, seeded, program,
                               *, max_load_factor=1.0/8.0):
    """One cancelled solve; returns the captured decision payload."""
    captured = {}

    def hook(payload):
        captured["payload"] = payload
        raise SolveCancelled(
            "original-workflow first seed decision captured",
            stage="contact_path.step:1.semismooth.initial_seed.decision")

    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    semismooth_path._INITIAL_SEED_DECISION_HOOK = hook
    try:
        with pytest.raises(SolveCancelled,
                            match="original-workflow first seed decision "
                                  "captured"):
            solve_static_nonlinear(
                model, contact=definition, contact_settings=seeded,
                contact_path_program=program, max_iterations=18,
                max_load_factor=max_load_factor)
    finally:
        semismooth_path._INITIAL_SEED_DECISION_HOOK = None
    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    return captured["payload"]


def _assert_identical_rerun(model, definition, seeded, program, first,
                              *, max_load_factor=1.0/8.0):
    second = _cancelled_seed_decision(model, definition, seeded, program,
                                      max_load_factor=max_load_factor)
    for key in ("origin_q", "origin_reactions", "origin_nodal_forces",
                "origin_normalized_gap", "origin_residual_n",
                "origin_constraint", "proposed_q", "proposed_reactions",
                "proposed_nodal_forces", "candidate_normalized_gap",
                "candidate_residual_n", "candidate_constraint",
                "candidate_ncp_n", "admitted_q", "admitted_reactions",
                "admitted_nodal_forces"):
        if key in first or key in second:
            assert np.array_equal(np.asarray(first[key]),
                                  np.asarray(second[key])), key
    first_record = {k: v for k, v in first["record"].items()
                    if k != "cost_s"}
    second_record = {k: v for k, v in second["record"].items()
                     if k != "cost_s"}
    assert first_record == second_record
    for key in ("origin_merit", "seed_merit"):
        if key in first["record"]:
            assert float(first["record"][key]) == float(
                second["record"][key])
    if "candidate_merit" in first:
        assert float(first["candidate_merit"]) == float(
            second["candidate_merit"])
    return second


@pytest.mark.skipif(bool(_missing_original_inputs()),
                    reason='exact original fixture inputs absent: '
                           + ', '.join(_missing_original_inputs()))
def test_first_actual_v1_seed_decision_is_captured_before_newton():
    model, definition, settings, program = _original_case()
    seeded = _seeded(settings, "linearized_kkt_dev")
    program_contract_before = program.contract()
    boundary_before = repr(model.boundary_conditions)
    constraints_before = repr(model.constraint_equations)
    payload = _cancelled_seed_decision(model, definition, seeded, program)
    # Coherent rollback: the cancelled solve left the declared model and
    # programme identity exactly unchanged.
    assert program.contract() == program_contract_before
    assert repr(model.boundary_conditions) == boundary_before
    assert repr(model.constraint_equations) == constraints_before
    decision = payload["record"]
    assert decision["selection"] == "linearized_kkt_dev"
    assert decision["outcome"] in {"admitted", "refused"}
    origin_q = np.asarray(payload["origin_q"], dtype=float)
    origin_mu = np.asarray(payload["origin_reactions"], dtype=float)
    origin_z = np.asarray(payload["origin_nodal_forces"], dtype=float)
    assert np.all(np.isfinite(origin_q)) and origin_q.size > 0
    assert np.all(np.isfinite(origin_mu))
    assert np.all(np.isfinite(origin_z)) and np.all(origin_z >= 0.0)
    origin_merit = (float(decision["origin_merit"])
                    if "origin_merit" in decision else None)
    if origin_merit is not None:
        assert np.isfinite(origin_merit) and origin_merit >= 0.0
    if decision["outcome"] == "admitted":
        # Re-verify the unchanged production predicates locally so the
        # capture cannot misreport an admission.
        assert float(decision["admissible_fraction"]) == 1.0
        assert np.all(np.asarray(payload["proposed_nodal_forces"]) >= 0.0)
        assert float(payload["candidate_merit"]) < origin_merit*(1.0-1.0e-4)
        assert "seed_merit" in decision
    else:
        assert isinstance(decision.get("reason"), str) and decision["reason"]
        # The explicit legacy refusal on this fixture is proposal-level
        # (the bounded QP iteration cap): the record carries no origin
        # merit and the payload no proposal arrays, and neither may be
        # fabricated for a refusal that never evaluated them.
        assert decision["reason"] == "qp_iteration_cap"
        assert "origin_merit" not in decision
        assert "proposed_q" not in payload
    capture = _decision_capture(payload)
    capture["provenance"] = _provenance(seeded, program)
    capture["rollback_check"] = "identical rerun pending"
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), V1_CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(_plain(capture), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    # Coherent rollback, functional part: an immediate rerun on the same
    # unchanged model must reproduce the same decision bitwise (modulo
    # timing), proving no partial state survived the cancellation.
    _assert_identical_rerun(model, definition, seeded, program, payload)
    capture["rollback_check"] = "identical_rerun_confirmed"
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), V1_CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(_plain(capture), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")


@pytest.mark.skipif(bool(_missing_original_inputs()),
                    reason='exact original fixture inputs absent: '
                           + ', '.join(_missing_original_inputs()))
def test_first_actual_v2_seed_decision_changes_the_obstruction():
    model, definition, settings, program = _original_case()
    seeded = _seeded(settings, "linearized_kkt_merit_globalization_dev")
    payload = _cancelled_seed_decision(model, definition, seeded, program)
    decision = payload["record"]
    assert decision["selection"] == "linearized_kkt_merit_globalization_dev"
    assert decision["revision"] == "origin_linearized_kkt_seed_merit_globalization_v2"
    assert decision["qp_release_alternatives"] is True
    assert decision["outcome"] in {"admitted", "refused"}
    # The changed decision: the internal QP no longer refuses through its
    # iteration cap; any remaining refusal comes from the actual gates.
    if "qp_terminal" in decision:
        assert decision["qp_terminal"] == "optimal"
        # An admitted decision carries no refusal reason at all; a refused
        # one must not blame the QP iteration cap anymore.
        assert decision.get("reason") != "qp_iteration_cap"
    origin_merit = (float(decision["origin_merit"])
                    if "origin_merit" in decision else None)
    if origin_merit is not None:
        assert np.isfinite(origin_merit) and origin_merit >= 0.0
    if decision["outcome"] == "admitted":
        # Re-verify the unchanged production predicates locally, including
        # the bounded-globalization record when a trial was admitted.
        assert float(decision["seed_merit"]) < origin_merit*(1.0-1.0e-4)
        assert float(payload["candidate_merit"]) == float(
            decision["seed_merit"])
        admitted_z = np.asarray(payload["admitted_nodal_forces"]
                                if "admitted_nodal_forces" in payload
                                else payload["proposed_nodal_forces"])
        assert np.all(admitted_z >= 0.0)
        bounded = decision.get("bounded_globalization")
        if bounded is not None:
            assert 0.0 < float(bounded["admitted_fraction"]) < 1.0
            assert bounded["full_proposal_reason"]
            assert bounded["trials"][-1]["outcome"] == "admitted"
            assert np.array_equal(
                np.asarray(payload["admitted_q"]),
                np.asarray(payload["proposed_q"])
                * float(bounded["admitted_fraction"]))
    else:
        assert isinstance(decision.get("reason"), str) and decision["reason"]
        bounded = decision.get("bounded_globalization")
        if bounded is not None:
            assert bounded["outcome"] == "no_safe_bounded_trial"
            assert isinstance(bounded["trials"], list)
    capture = _decision_capture(payload)
    capture["provenance"] = _provenance(seeded, program)
    capture["rollback_check"] = "identical rerun pending"
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), V2_CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(_plain(capture), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    _assert_identical_rerun(model, definition, seeded, program, payload)
    capture["rollback_check"] = "identical_rerun_confirmed"
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), V2_CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(_plain(capture), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")


def _refusal_regression(model, definition, seeded, program, *, patch,
                         max_load_factor=1.0/8.0):
    """One cancelled v2 solve with a patched refusal entry; returns the
    captured decision payload. Both refusal entries must keep an explicit
    refusal record, never reach the bounded-globalization trials, and
    leave the model/programme identity unchanged."""
    captured = {}

    def hook(payload):
        captured["payload"] = payload
        raise SolveCancelled(
            "original-workflow refusal regression captured",
            stage="contact_path.step:1.semismooth.initial_seed.decision")

    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    semismooth_path._INITIAL_SEED_DECISION_HOOK = hook
    with patch:
        try:
            with pytest.raises(SolveCancelled,
                                match="original-workflow refusal regression "
                                      "captured"):
                solve_static_nonlinear(
                    model, contact=definition, contact_settings=seeded,
                    contact_path_program=program, max_iterations=18,
                    max_load_factor=max_load_factor)
        finally:
            semismooth_path._INITIAL_SEED_DECISION_HOOK = None
    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    return captured["payload"]


@pytest.mark.skipif(bool(_missing_original_inputs()),
                    reason='exact original fixture inputs absent: '
                           + ', '.join(_missing_original_inputs()))
def test_first_actual_v2_origin_evaluation_failure_refusal_is_explicit():
    # R1 regression: an origin-evaluation failure under v2 must keep its
    # explicit refusal record; the bounded-globalization block must not
    # run (no finite proposal exists) and must not crash.
    model, definition, settings, program = _original_case()
    seeded = _seeded(settings, "linearized_kkt_merit_globalization_dev")
    program_contract_before = program.contract()
    boundary_before = repr(model.boundary_conditions)
    constraints_before = repr(model.constraint_equations)
    original_assemble = semismooth_path._assemble_nonlinear_system
    calls = {"n": 0}

    def failing_assemble(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # The seed origin evaluate is the first assembly of a fresh
            # step-1 solve; fail exactly that one.
            raise ValueError("synthetic origin evaluation failure")
        return original_assemble(*args, **kwargs)

    class patch:
        def __enter__(self):
            semismooth_path._assemble_nonlinear_system = failing_assemble
        def __exit__(self, *exc):
            semismooth_path._assemble_nonlinear_system = original_assemble

    payload = _refusal_regression(model, definition, seeded, program,
                                   patch=patch())
    decision = payload["record"]
    assert decision["selection"] == "linearized_kkt_merit_globalization_dev"
    assert decision["outcome"] == "refused"
    assert decision["reason"].startswith("origin_evaluation_failed")
    assert "bounded_globalization" not in decision
    assert "origin_merit" not in decision
    assert np.all(np.asarray(payload["origin_q"]) == 0.0)
    assert np.all(np.asarray(payload["origin_nodal_forces"]) == 0.0)
    assert program.contract() == program_contract_before
    assert repr(model.boundary_conditions) == boundary_before
    assert repr(model.constraint_equations) == constraints_before


@pytest.mark.skipif(bool(_missing_original_inputs()),
                    reason='exact original fixture inputs absent: '
                           + ', '.join(_missing_original_inputs()))
def test_first_actual_v2_proposal_refusal_is_explicit():
    # R1 regression: a refused proposal under v2 must keep its explicit
    # refusal record; the bounded-globalization block must not run (no
    # finite proposal exists) and must not crash.
    model, definition, settings, program = _original_case()
    seeded = _seeded(settings, "linearized_kkt_merit_globalization_dev")
    program_contract_before = program.contract()
    boundary_before = repr(model.boundary_conditions)
    constraints_before = repr(model.constraint_equations)
    from anyfesolver.general_contact.initial_contact_seed import (
        InitialContactSeedRefusal)
    original_propose = semismooth_path.propose_linearized_kkt_seed

    def refusing_propose(**kwargs):
        return InitialContactSeedRefusal(
            "synthetic_proposal_refusal",
            {"revision": ("origin_linearized_kkt_seed_merit_"
                          "globalization_v2"),
             "selection": "linearized_kkt_merit_globalization_dev"})

    class patch:
        def __enter__(self):
            semismooth_path.propose_linearized_kkt_seed = refusing_propose
        def __exit__(self, *exc):
            semismooth_path.propose_linearized_kkt_seed = original_propose

    payload = _refusal_regression(model, definition, seeded, program,
                                   patch=patch())
    decision = payload["record"]
    assert decision["selection"] == "linearized_kkt_merit_globalization_dev"
    assert decision["outcome"] == "refused"
    assert decision["reason"] == "synthetic_proposal_refusal"
    assert "bounded_globalization" not in decision
    assert "origin_merit" not in decision
    assert np.all(np.asarray(payload["origin_q"]) == 0.0)
    assert np.all(np.asarray(payload["origin_nodal_forces"]) == 0.0)
    assert program.contract() == program_contract_before
    assert repr(model.boundary_conditions) == boundary_before
    assert repr(model.constraint_equations) == constraints_before


# ----------------------------------------------------------------------
# Input-free current math/refusal checks on mixed active/split nodes.
#
# The exact original fixture inputs above are external host artifacts;
# where they are absent the guarded nodes skip with the exact missing
# paths. The checks below are always active and need no external input:
# they exercise the same production seed-decision math and the same
# explicit refusal records on the committed small-patch case, whose seed
# proposal is genuinely mixed (one active pressure mode against
# twenty-four split modes). No consumed registered authority script is
# run and no original-fixture identity claim is made here.


def _input_free_case():
    from test_general_contact_admission_globalization_driver_small_patch import (
        _case as small_patch_case, _enabled as small_patch_enabled)
    model, definition, settings, program = small_patch_case()
    return model, definition, small_patch_enabled(settings), program


def test_input_free_seed_decision_current_math_on_mixed_active_split_nodes():
    model, definition, settings, program = _input_free_case()
    program_contract_before = program.contract()
    boundary_before = repr(model.boundary_conditions)
    constraints_before = repr(model.constraint_equations)
    for selection in ("linearized_kkt_dev",
                      "linearized_kkt_merit_globalization_dev"):
        seeded = _seeded(settings, selection)
        payload = _cancelled_seed_decision(
            model, definition, seeded, program, max_load_factor=0.5)
        decision = payload["record"]
        assert decision["selection"] == selection
        assert decision["outcome"] in {"admitted", "refused"}
        origin_z = np.asarray(payload["origin_nodal_forces"], dtype=float)
        assert np.all(np.isfinite(origin_z)) and np.all(origin_z >= 0.0)
        if selection == "linearized_kkt_merit_globalization_dev":
            assert decision["revision"] == (
                "origin_linearized_kkt_seed_merit_globalization_v2")
            assert decision["qp_release_alternatives"] is True
            if "qp_terminal" in decision:
                assert decision["qp_terminal"] == "optimal"
                assert decision.get("reason") != "qp_iteration_cap"
        if decision["outcome"] == "admitted":
            # Current math on the mixed active/split nodes: the proposal
            # carries both active and split pressure modes and passes the
            # unchanged production admission predicates locally.
            assert float(decision["admissible_fraction"]) == 1.0
            proposed_z = np.asarray(payload["proposed_nodal_forces"],
                                    dtype=float)
            assert np.all(np.isfinite(proposed_z))
            assert np.all(proposed_z >= 0.0)
            assert np.any(proposed_z > 0.0) and np.any(proposed_z == 0.0)
            assert float(decision["seed_merit"]) < float(
                decision["origin_merit"])*(1.0-1.0e-4)
        else:
            assert isinstance(decision.get("reason"), str) and decision["reason"]
        # Coherent rollback: an immediate rerun on the same unchanged
        # model reproduces the same decision bitwise (modulo timing).
        _assert_identical_rerun(model, definition, seeded, program, payload,
                                max_load_factor=0.5)
    assert program.contract() == program_contract_before
    assert repr(model.boundary_conditions) == boundary_before
    assert repr(model.constraint_equations) == constraints_before


def test_input_free_seed_refusals_remain_explicit_on_mixed_active_split_nodes():
    model, definition, settings, program = _input_free_case()
    seeded = _seeded(settings, "linearized_kkt_merit_globalization_dev")
    program_contract_before = program.contract()
    boundary_before = repr(model.boundary_conditions)
    constraints_before = repr(model.constraint_equations)
    original_assemble = semismooth_path._assemble_nonlinear_system
    calls = {"n": 0}

    def failing_assemble(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # The seed origin evaluate is the first assembly of a fresh
            # step-1 solve; fail exactly that one.
            raise ValueError("synthetic input-free origin evaluation failure")
        return original_assemble(*args, **kwargs)

    class assemble_patch:
        def __enter__(self):
            semismooth_path._assemble_nonlinear_system = failing_assemble
        def __exit__(self, *exc):
            semismooth_path._assemble_nonlinear_system = original_assemble

    payload = _refusal_regression(model, definition, seeded, program,
                                  patch=assemble_patch(),
                                  max_load_factor=0.5)
    decision = payload["record"]
    assert decision["outcome"] == "refused"
    assert decision["reason"].startswith("origin_evaluation_failed")
    assert "bounded_globalization" not in decision
    assert "origin_merit" not in decision
    assert np.all(np.asarray(payload["origin_q"]) == 0.0)
    assert np.all(np.asarray(payload["origin_nodal_forces"]) == 0.0)

    from anyfesolver.general_contact.initial_contact_seed import (
        InitialContactSeedRefusal)
    original_propose = semismooth_path.propose_linearized_kkt_seed

    def refusing_propose(**kwargs):
        return InitialContactSeedRefusal(
            "synthetic_input_free_proposal_refusal",
            {"revision": ("origin_linearized_kkt_seed_merit_"
                          "globalization_v2"),
             "selection": "linearized_kkt_merit_globalization_dev"})

    class propose_patch:
        def __enter__(self):
            semismooth_path.propose_linearized_kkt_seed = refusing_propose
        def __exit__(self, *exc):
            semismooth_path.propose_linearized_kkt_seed = original_propose

    payload = _refusal_regression(model, definition, seeded, program,
                                  patch=propose_patch(),
                                  max_load_factor=0.5)
    decision = payload["record"]
    assert decision["outcome"] == "refused"
    assert decision["reason"] == "synthetic_input_free_proposal_refusal"
    assert "bounded_globalization" not in decision
    assert "origin_merit" not in decision
    assert np.all(np.asarray(payload["origin_q"]) == 0.0)
    assert np.all(np.asarray(payload["origin_nodal_forces"]) == 0.0)
    assert program.contract() == program_contract_before
    assert repr(model.boundary_conditions) == boundary_before
    assert repr(model.constraint_equations) == constraints_before
