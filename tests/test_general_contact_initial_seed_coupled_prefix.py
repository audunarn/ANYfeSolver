"""Coupled 2x2 initial-contact-seed prefix: enabled outcome on real mechanics.

The existing small-patch driver fixture is used unchanged: the 2x2 Q4
source against the two-S3 nonmatching target, 20um initial normal gap,
0.04 N two-step interior nodal force ramp, unchanged 50um gap budget,
joint overlap certificate explicitly effective, D1/Jet2/fine-Q1/
monolithic/primal-mortar route. The only difference from the banked
default-none witness is the initial_contact_seed selector; mechanics,
program, tolerances and every acceptance gate are identical.

Measured fact that shapes this module (attempt-05 of this scope, same
immutable source snapshot): ONE real half solve of this fixture costs
~167 s under the guard's serial NUMBA_DISABLE_JIT=1 environment and the
default-none reference half terminates with status 'diverged' — the
first-ever completed execution of this prefix (every earlier campaign
job was killed by its cap mid-solve). Two half solves therefore cannot
fit one 180 s job, so the seedNone-vs-enabled comparison is banked
across attempts: the default-none outcome is attempt-05's evidence, and
this module records the enabled outcome truthfully. No acceleration or
nonlinear feasibility is assumed: the seed outcome (admitted or
refused, with counts, reason and cost) and the trajectory outcome are
both recorded as they occur. Only if the prefix becomes available is
the COMPLETE accepted public checkpoint exported under the attempt TEMP
directory for later connected macro rollback/restart checks; no
internal Newton checkpoint is ever exported.
"""
import json
import os
import tempfile
from dataclasses import asdict, replace

import numpy as np
import pytest

from anyfesolver.control import SolveCancelled

from test_general_contact_admission_globalization_driver_small_patch import (
    _CHECKPOINT_HASH_KEY,
    _assert_active_contact,
    _case,
    _enabled,
    _solve,
)

_PREFIX_CHECKPOINT_NAME = "initial-seed-coupled-prefix-checkpoint-v1.json"
_PREFIX_IDENTIFIERS_NAME = "initial-seed-coupled-prefix-identifiers-v1.json"
_PREFIX_OUTCOME_NAME = "initial-seed-coupled-prefix-outcome-v1.json"


def _seeded(settings):
    """Fixture settings with only the seed selector enabled."""
    enabled = replace(_enabled(settings),
                      initial_contact_seed="linearized_kkt_dev")
    # The joint overlap certificate stays explicitly effective.
    assert enabled.q4_source_overlap_dual_dev is True
    assert enabled.q4_source_gap_certificate == "overlap_local_dev"
    assert enabled.initial_contact_seed == "linearized_kkt_dev"
    return enabled


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


class _CancelAt:
    def __init__(self, stage):
        self.stage = stage

    def raise_if_cancelled(self, stage=""):
        if stage == self.stage:
            raise SolveCancelled("injected initial-seed cancellation",
                                 stage=stage)


def test_cancellation_during_initial_seed_propagates():
    model, definition, settings, program = _case()
    token = _CancelAt("contact_path.step:1.semismooth.initial_seed")
    with pytest.raises(SolveCancelled,
                       match="injected initial-seed cancellation"):
        _solve(model, definition, _seeded(settings), program,
               cancellation_token=token)


def test_coupled_2x2_prefix_seeded_outcome_recorded_and_exported():
    model, definition, settings, program = _case()
    seeded = _solve(model, definition, _seeded(settings), program,
                    max_load_factor=0.5, emit_restart_checkpoint=True)
    # The first macro attempt always runs the seed on a fresh seeded
    # solve, so the effective selection/outcome/cost record exists in
    # the public info regardless of the trajectory outcome.
    seed = seeded.info["contact_initial_seed"]
    assert seed["selection"] == "linearized_kkt_dev"
    assert seed["outcome"] in {"admitted", "refused"}
    assert seed["constrained_factorizations"] == 1
    assert seed["constrained_solves"] >= 1
    assert type(seed["qp_iterations"]) is int
    assert seed["cost_s"] >= 0.0
    if seed["outcome"] == "admitted":
        assert seed["admissible_fraction"] == 1.0
        assert seed["seed_merit"] < seed["origin_merit"]
    else:
        assert isinstance(seed["reason"], str) and seed["reason"]
    temp = (os.environ.get("TEMP") or tempfile.gettempdir())
    outcome = {
        "schema": "INITIAL_SEED_COUPLED_PREFIX_OUTCOME_V1",
        "selection": "linearized_kkt_dev",
        "status": seeded.status,
        "failure_reason": seeded.failure_reason,
        "steps": len(seeded.steps),
        "seed_record": seed,
        "source_manifest": _source_manifest(),
    }
    with open(os.path.join(temp, _PREFIX_OUTCOME_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(outcome, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if seeded.status != "completed":
        # Truthful refused trajectory: the origin-only default
        # enforcement outcome is recorded; nothing is exported as an
        # accepted checkpoint because none exists.
        assert seeded.status in {"diverged", "stopped_at_limit"}
        assert isinstance(seeded.failure_reason, str)
        assert seeded.failure_reason
        return
    # The prefix became available: the COMPLETE accepted public
    # checkpoint is exported under TEMP as an immutable bound
    # preparation for connected macro rollback/restart checks.
    assert len(seeded.steps) == 1
    assert np.all(np.isfinite(seeded.displacements))
    checkpoint = seeded.to_restart_checkpoint()
    record = checkpoint["path_state"]["history"][0]
    _assert_active_contact(record)
    assert seeded.info["contact_initial_seed"] == seed
    if record.get("initial_seed") is not None:
        # Present when the seeded macro attempt itself was accepted;
        # a continuation-substep acceptance keeps the seed observation
        # at the info level.
        assert record["initial_seed"] == seed
    with open(os.path.join(temp, _PREFIX_CHECKPOINT_NAME), "wb") as handle:
        handle.write(seeded.restart_checkpoint_bytes())
    identifiers = {
        "schema": "INITIAL_SEED_COUPLED_PREFIX_ACCEPTED_IDENTIFIERS_V1",
        "fixture_module":
            "test_general_contact_admission_globalization_driver_small_patch",
        "prefix_module": "test_general_contact_initial_seed_coupled_prefix",
        "checkpoint_sha256": checkpoint[_CHECKPOINT_HASH_KEY],
        "model_fingerprint": checkpoint["model_fingerprint"],
        "analysis_fingerprint": checkpoint["analysis_fingerprint"],
        "program": program.contract(),
        "settings": asdict(_seeded(settings)),
        "seed_record": seed,
        "seeded_iterations": record["semismooth_iterations"],
        "source_manifest": _source_manifest(),
    }
    with open(os.path.join(temp, _PREFIX_IDENTIFIERS_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(identifiers, handle, indent=2, sort_keys=True)
        handle.write("\n")
    # Restart binds settings and method: a seedNone-settings restart of
    # the seeded checkpoint is refused before any Newton work.
    fresh = _case()
    with pytest.raises(ValueError, match="incompatible"):
        _solve(fresh[0], fresh[1], _enabled(fresh[2]), fresh[3],
               restart_checkpoint=checkpoint)
