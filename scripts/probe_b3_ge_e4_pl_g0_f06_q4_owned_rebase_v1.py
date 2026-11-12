"""Preparatory curved-Q4 owner-issued loaded rebase and continuation candidate.

This G0-only prototype has its own schema and owner identity. It reuses the
S3 prototype transaction mechanics but applies the Q4 virgin elastic history
policy; no scientific registration or product authority is implied.

Restoration replays and validates the embedded two-step legacy prefix through
its ordinary owner. Only the rebased and successor snapshots are restored
directly. A replay-free overall restore claim requires separately trusted
accepted-base provenance and is outside this candidate.
"""

from __future__ import annotations

import numpy as np

from anyfesolver.control import cancellation_safe_point
from anyfesolver._ge_beam3_g3c_local_shell import deformation
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_shell_joint_state import CoupledShellBeamAnalysis, decode
from anyfesolver._native_reference_modal import _owned

from scripts.probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3 import (
    fixture, numeric_state, shell_spatial,
)
from scripts.probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4 import (
    TARGETS, OwnedRebasePrototype, PrototypeState,
    _check_tangent_directions, _hash, _near,
)


SCHEMA = "G0_F06_CURVED_Q4_OWNED_REBASE_CANDIDATE_V1"


class Q4OwnedRebaseCandidate(OwnedRebasePrototype):
    """One curved-Q4 rebase at accepted cursor two and one changed successor."""

    def __init__(self):
        seed = fixture("Q4", True)
        self.owner = CoupledShellBeamAnalysis(seed.assembly, targets=TARGETS,
            shell_fixed=seed.shell_fixed, nodal_forces=seed.forces,
            max_iterations=seed.max_iterations, max_backtracks=seed.max_backtracks)
        from threading import Lock
        self._lock = Lock()
        self._consumed_base = set()
        self._head_sha256 = None
        self._generation = None

    def _rebased_state(self, base):
        a = self.owner.assembly
        n = len(a.coordinates)
        anchor = np.array([_exp_terms(row[3:])[0]
            for row in base.shell_u.reshape(n, 6)])
        u = np.array(base.shell_u, copy=True)
        u.reshape(n, 6)[:, 3:] = 0.
        state = PrototypeState(base.mechanical, _owned(u), base.multipliers,
            base.beam_origins, base.beam_histories, base.shell_origin,
            base.shell_history, _owned(anchor), base.cursor)
        old = deformation(np.asarray(a.coordinates), base.shell_u,
            np.tile(np.eye(3), (n, 1, 1)))
        new = deformation(np.asarray(a.coordinates), state.shell_u, state.anchor)
        for label in ("deformation", "rotations", "current_positions"):
            _near(getattr(old, label), getattr(new, label), label)
        old_response, old_residual, _, _ = self.owner._evaluate(base,
            base.mechanical, base.shell_u, base.multipliers, base.cursor)
        new_response, new_residual = self._anchored_evaluate(state,
            state.mechanical, state.shell_u, state.multipliers, state.cursor)
        _near(old_response.constraints, new_response.constraints, "joint gap")
        _near(old_residual, new_residual, "loaded residual")
        _check_tangent_directions(old_response, base.shell_u,
            new_response, state.shell_u, a.count)
        for axis, name in ((slice(0, 3), "force"), (slice(3, 6), "moment")):
            _near(old_residual[:a.shell_count].reshape(-1, 6)[:, axis],
                new_residual[:a.shell_count].reshape(-1, 6)[:, axis], name)
        origin = decode(base.shell_history)
        _, _, old_work, old_candidate = shell_spatial(a, origin, old, base.shell_u)
        _, _, new_work, new_candidate = shell_spatial(a, origin, new, state.shell_u)
        _near(old_work, new_work, "shell virtual work")
        old_paths, old_values = numeric_state(old_candidate)
        new_paths, new_values = numeric_state(new_candidate)
        if old_paths != new_paths:
            raise ValueError("F06 Q4 shell material state paths changed")
        if max(np.linalg.norm(old_values), np.linalg.norm(new_values)) > 1.e-12:
            raise ValueError("F06 Q4 virgin elastic history policy changed")
        _near(old_values, new_values, "Q4 shell material history")
        if (state.beam_origins != base.beam_origins or
                canonical(state.beam_histories) != canonical(base.beam_histories) or
                state.shell_origin != base.shell_origin or
                state.shell_history != base.shell_history):
            raise ValueError("F06 accepted material history changed")
        return state

    def _event(self, base_checkpoint, base, state):
        event = super()._event(base_checkpoint, base, state)
        event["schema"] = SCHEMA
        return event

    def _restore(self, checkpoint, expected_sha256, cancellation_token=None):
        if type(checkpoint) is not bytes or _hash(checkpoint) != expected_sha256:
            raise ValueError("external F06 prototype checkpoint hash required")
        value = decode(checkpoint)
        if (type(value) is not dict or set(value) != {
                "schema", "definition_sha256", "base_checkpoint", "base_sha256",
                "cursor", "base_state", "base_state_sha256", "anchor",
                "rebased_state", "rebased_state_sha256", "successor",
                "production_qualified"} or value["schema"] != SCHEMA or
                value["definition_sha256"] != self.owner.identity or
                value["production_qualified"] is not False or value["cursor"] != 2 or
                type(value["base_checkpoint"]) is not str):
            raise ValueError("F06 Q4 checkpoint schema/identity mismatch")
        base_raw = value["base_checkpoint"].encode("ascii")
        if value["base_state_sha256"] != _hash(canonical(value["base_state"])):
            raise ValueError("F06 captured base state hash mismatch")
        # The embedded base hash can itself be consistently rewritten. Full
        # legacy acceptance checks bind policy, definition, genesis, and both
        # issued records before the Q4 snapshots are decoded directly.
        accepted, records = self.owner._restore(base_raw, value["base_sha256"],
            cancellation_token)
        if (accepted.cursor != 2 or len(records) != 2 or
                canonical(accepted.descriptor()) != canonical(value["base_state"])):
            raise ValueError("F06 Q4 embedded legacy base acceptance mismatch")
        base = self._captured_base(base_raw, value["base_sha256"], value["base_state"])
        state = self._snapshot_state(value["rebased_state"])
        if (canonical(value["anchor"]) != canonical(state.anchor) or
                value["rebased_state_sha256"] != _hash(canonical(state.descriptor()))):
            raise ValueError("F06 Q4 rebase event mismatch")
        oracle = self._rebased_state(base)
        if canonical(state.descriptor()) != canonical(oracle.descriptor()):
            raise ValueError("F06 Q4 rebase event mismatch")
        if value["successor"] is not None:
            predecessor_sha256 = _hash(canonical({**value, "successor": None}))
            state = self._direct_successor(state, value["successor"], predecessor_sha256)
        cancellation_safe_point(cancellation_token, "f06.restore.complete")
        return state, value
