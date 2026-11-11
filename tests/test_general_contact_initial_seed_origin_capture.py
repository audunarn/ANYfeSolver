"""Real-origin seed proposal capture on the coupled 2x2 fixture.

The full seeded half prefix cannot fit any guard cap: one real half
solve of this fixture measures 167 s (default-none, attempt-05 of this
scope, terminated 'diverged') and the seeded run was still executing at
the 180 s cap (attempt-06). This module instead captures the production
seed proposal at the REAL committed origin of the SAME fixture and
mechanics: the wrapper calls the real ``propose_linearized_kkt_seed``
with the production blocks (constrained structural tangent, full
supported normalized-gap gradients, live residual and constraint) and
then raises ``SolveCancelled`` — the proven-propagating channel — so
the solve stops after one origin tangent evaluation, before any Newton
work, with the origin untouched.

The capture records truthfully whether the reference-linearized KKT
operator admits a nonnegative supported pressure proposal at the real
cold-start origin, with the refusal reason, operator/evaluation counts
and cost, and the predicted versus actual gap/support structure. No
acceptance gate is evaluated here and no trajectory claim is made.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from anyfesolver.control import SolveCancelled
from anyfesolver.general_contact import semismooth_path
from anyfesolver.general_contact.initial_contact_seed import (
    propose_linearized_kkt_seed,
)

from test_general_contact_admission_globalization_driver_small_patch import (
    _case,
    _enabled,
    _solve,
)

_CAPTURE_NAME = "initial-seed-origin-capture-v1.json"


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


def test_real_origin_seed_proposal_is_captured_with_counts_and_reason():
    model, definition, settings, program = _case()
    captured = {}

    def capturing(**kwargs):
        captured["inputs"] = kwargs
        captured["proposal"] = propose_linearized_kkt_seed(**kwargs)
        raise SolveCancelled("initial-seed proposal captured",
                             stage="contact_path.step:1.semismooth."
                                   "initial_seed")

    semismooth_path.propose_linearized_kkt_seed = capturing
    try:
        with pytest.raises(SolveCancelled,
                            match="initial-seed proposal captured"):
            _solve(model, definition, _seeded(settings), program,
                   max_load_factor=0.5)
    finally:
        semismooth_path.propose_linearized_kkt_seed = (
            propose_linearized_kkt_seed)
    # The production call reached the real origin with live blocks.
    inputs = captured["inputs"]

    def dense(block):
        return np.asarray(block.toarray() if hasattr(block, "toarray")
                          else block, dtype=float)

    q = np.asarray(inputs["q"], dtype=float).reshape(-1)
    residual = np.asarray(inputs["residual"], dtype=float).reshape(-1)
    constraint = np.asarray(inputs["constraint"], dtype=float).reshape(-1)
    kqq = dense(inputs["kqq"])
    kqz = dense(inputs["kqz"])
    gap_q = dense(inputs["gap_q"])
    normalized_gap = np.asarray(inputs["normalized_gap"], dtype=float)
    support = np.asarray(inputs["support"], dtype=float)
    assert kqq.shape == (q.size, q.size)
    assert kqz.shape == (q.size, normalized_gap.size)
    assert gap_q.shape == (normalized_gap.size, q.size)
    assert residual.size == q.size
    assert support.shape == normalized_gap.shape
    assert np.all(np.isfinite(kqq)) and np.all(np.isfinite(kqz))
    # Pressure force z has N modes, normalized gap has m, H has m/N.
    proposal = captured["proposal"]
    record = {"schema": "INITIAL_SEED_ORIGIN_CAPTURE_V1",
              "selection": "linearized_kkt_dev",
              "source_manifest": _source_manifest(),
              "pressure_modes": int(normalized_gap.size),
              "supported_modes": int(np.count_nonzero(support > 0.0)),
              "reduced_coordinates": int(q.size),
              "control_constraints": int(constraint.size)}
    if proposal.refused:
        record.update(outcome="refused", reason=proposal.reason,
                      **proposal.diagnostics)
        assert isinstance(proposal.reason, str) and proposal.reason
    else:
        z = np.asarray(proposal.nodal_forces, dtype=float).reshape(-1)
        assert z.shape == normalized_gap.shape
        assert np.all(np.isfinite(z))
        record.update(outcome="proposed",
                      min_pressure_force_n=float(z.min()),
                      max_pressure_force_n=float(z.max()),
                      total_pressure_force_n=float(z.sum()),
                      **proposal.diagnostics)
    # Operator/evaluation counts and cost appear once the constrained
    # factorization stage was reached; early block refusals predate it.
    if "constrained_factorizations" in record:
        assert record["constrained_factorizations"] == 1
    if "constrained_solves" in record:
        assert record["constrained_solves"] >= 1
    if "qp_iterations" in record:
        assert type(record["qp_iterations"]) is int
    if "cost_s" in record:
        assert record["cost_s"] >= 0.0
    with open(os.path.join((os.environ.get("TEMP") or tempfile.gettempdir()), _CAPTURE_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
