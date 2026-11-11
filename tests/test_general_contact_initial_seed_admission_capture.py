"""Real seed-admission decision capture on the coupled 2x2 fixture.

The full seeded half prefix cannot fit any guard cap: one real half solve
of this fixture measures 167 s (default-none, attempt-05 of this scope,
terminated 'diverged') and the seeded run was still executing at the
180 s cap (attempt-06). The origin capture (attempt-07) already recorded
that the fixture's origin-linearized KKT QP is cheap and returns a
proposal; the open discriminator is whether the actual path/merit admits
that proposal. This module captures exactly that next stage: the private
``_INITIAL_SEED_DECISION_HOOK`` receives the production decision payload
after the unchanged admission predicates ran (actual-mechanics candidate
reevaluation, swept admissible fraction, full scaled merit comparison)
and raises ``SolveCancelled`` — the proven-propagating channel — so the
solve stops before any Newton iteration with the origin untouched.

The capture records truthfully: the exact origin and proposed poses, the
live origin/candidate normalized gaps, the candidate force/constraint/
NCP values, the actual physical swept-certificate fraction, the actual
scaled merits, and whether production's unchanged decision admits or
refuses the seed. Sampled gaps are recorded as data, never as safety
proof; the swept fraction and the production decision are the proof
channel. No acceptance gate is evaluated beyond the production ones and
no trajectory claim is made.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from anyfesolver.control import SolveCancelled
from anyfesolver.general_contact import semismooth_path
from anyfesolver.general_contact.core import _contact_settings_contract

from test_general_contact_admission_globalization_driver_small_patch import (
    _case,
    _enabled,
    _solve,
)

_CAPTURE_NAME = "initial-seed-admission-capture-v1.json"


def _seeded(settings):
    from dataclasses import replace
    return replace(_enabled(settings),
                   initial_contact_seed="linearized_kkt_dev")


def _source_manifest():
    runtime = os.environ.get("AUTOMATIC_RUNTIME_RESULT")
    if not runtime:
        return None
    install = os.path.join(os.path.dirname(runtime), "install.json")
    if not os.path.isfile(install):
        return None
    with open(install, encoding="utf-8") as handle:
        record = json.load(handle)
    return {"payload_sha256": record.get("payload_sha256"),
            "mode": record.get("mode")}


def test_real_seed_admission_decision_is_captured_before_newton():
    model, definition, settings, program = _case()
    seeded = _seeded(settings)
    captured = {}

    def hook(payload):
        captured["payload"] = payload
        raise SolveCancelled("initial-seed admission decision captured",
                             stage="contact_path.step:1.semismooth."
                                   "initial_seed.decision")

    assert semismooth_path._INITIAL_SEED_DECISION_HOOK is None
    semismooth_path._INITIAL_SEED_DECISION_HOOK = hook
    try:
        with pytest.raises(SolveCancelled,
                            match="initial-seed admission decision captured"):
            _solve(model, definition, seeded, program, max_load_factor=0.5)
    finally:
        semismooth_path._INITIAL_SEED_DECISION_HOOK = None
    payload = captured["payload"]
    decision = payload["record"]
    assert decision["selection"] == "linearized_kkt_dev"
    assert decision["outcome"] in {"admitted", "refused"}

    def array(name):
        return np.asarray(payload[name], dtype=float).reshape(-1)

    origin_q = array("origin_q")
    origin_mu = array("origin_reactions")
    origin_z = array("origin_nodal_forces")
    assert np.all(np.isfinite(origin_q)) and origin_q.size > 0
    assert np.all(np.isfinite(origin_mu))
    assert np.all(np.isfinite(origin_z)) and np.all(origin_z >= 0.0)
    origin_merit = (float(decision["origin_merit"])
                    if "origin_merit" in decision else None)
    if origin_merit is not None:
        assert np.isfinite(origin_merit) and origin_merit >= 0.0
    record = {"schema": "INITIAL_SEED_ADMISSION_CAPTURE_V1",
              "selection": "linearized_kkt_dev",
              "source_manifest": _source_manifest(),
              "settings_contract": _contact_settings_contract(seeded),
              "program": {"steps_per_segment": program.steps_per_segment,
                          "controls": len(program.controls),
                          "nodal_forces": len(program.nodal_forces)},
              "complementarity_force_scale_n":
                  float(seeded.complementarity_force_scale_n),
              "outcome": decision["outcome"],
              "reason": decision.get("reason"),
              "origin_merit": origin_merit,
              "origin_q": origin_q.tolist(),
              "origin_reactions": origin_mu.tolist(),
              "origin_nodal_forces": origin_z.tolist()}
    if "admissible_fraction" in decision:
        record["admissible_fraction"] = float(
            decision["admissible_fraction"])
    for key in ("revision", "total_pressure_modes",
                "supported_pressure_modes", "qp_iterations", "qp_terminal",
                "free_gap_min_m", "gap_response_eigenvalue_min",
                "gap_response_eigenvalue_max", "seed_pressure_force_n",
                "cost_s"):
        if key in decision:
            record[f"proposal_{key}"] = decision[key]
    if "proposed_q" in payload:
        proposed_q = array("proposed_q")
        proposed_mu = array("proposed_reactions")
        proposed_z = array("proposed_nodal_forces")
        assert proposed_q.shape == origin_q.shape
        assert proposed_mu.shape == origin_mu.shape
        assert proposed_z.shape == origin_z.shape
        assert np.all(np.isfinite(proposed_q))
        assert np.all(np.isfinite(proposed_mu))
        assert np.all(np.isfinite(proposed_z))
        record["proposed_q"] = proposed_q.tolist()
        record["proposed_reactions"] = proposed_mu.tolist()
        record["proposed_nodal_forces"] = proposed_z.tolist()
        record["origin_normalized_gap"] = (
            array("origin_normalized_gap").tolist())
        record["origin_residual_norm_n"] = float(
            np.linalg.norm(array("origin_residual_n")))
        record["origin_constraint_max"] = float(np.max(
            np.abs(array("origin_constraint")), initial=0.0))
    if "candidate_normalized_gap" in payload:
        candidate_gap = array("candidate_normalized_gap")
        candidate_residual = array("candidate_residual_n")
        candidate_constraint = array("candidate_constraint")
        candidate_ncp = array("candidate_ncp_n")
        assert candidate_gap.shape == origin_z.shape
        assert np.all(np.isfinite(candidate_gap))
        assert np.all(np.isfinite(candidate_residual))
        assert np.all(np.isfinite(candidate_constraint))
        assert np.all(np.isfinite(candidate_ncp))
        record["candidate_normalized_gap"] = candidate_gap.tolist()
        record["candidate_residual_norm_n"] = float(
            np.linalg.norm(candidate_residual))
        record["candidate_constraint_max"] = float(
            np.max(np.abs(candidate_constraint), initial=0.0))
        record["candidate_ncp_sum_n"] = float(np.sum(np.abs(candidate_ncp)))
        record["candidate_merit"] = float(payload["candidate_merit"])
        scale = float(seeded.complementarity_force_scale_n)
        record["candidate_residual_scaled"] = (
            record["candidate_residual_norm_n"]/scale)
        record["candidate_ncp_scaled"] = record["candidate_ncp_sum_n"]/scale
    # The unchanged production predicates decide; re-verify their exact
    # decision locally so the capture cannot misreport an outcome.
    if decision["outcome"] == "admitted":
        assert record["admissible_fraction"] == 1.0
        assert np.all(array("proposed_nodal_forces") >= 0.0)
        assert record["candidate_merit"] < origin_merit*(1.0 - 1.0e-4)
        assert "seed_merit" in decision
        record["seed_merit"] = float(decision["seed_merit"])
    else:
        assert isinstance(decision.get("reason"), str) and decision["reason"]
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), _CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
