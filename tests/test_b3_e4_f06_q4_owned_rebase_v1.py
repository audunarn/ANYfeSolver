"""Bounded synthetic checks for the separate curved-Q4 F06 candidate.

Q4 restoration replays the embedded legacy two-step prefix, then directly
decodes the rebased and successor snapshots without replaying those events.
"""

from hashlib import sha256
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest

os.environ.setdefault("NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "f06_q4_numba"))

from anyfesolver._ge_beam3_g3c_local_shell import deformation
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_shell_joint_state import decode
from anyfesolver.control import CancellationToken, SolveCancelled
from scripts.probe_b3_ge_e4_pl_g0_f06_q4_owned_rebase_v1 import (
    Q4OwnedRebaseCandidate, SCHEMA,
)
from scripts.probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3 import shell_spatial


def _hash(raw):
    return sha256(raw).hexdigest()


class _CancelAt(CancellationToken):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def raise_if_cancelled(self, stage=""):
        if stage == self.stage:
            self.cancel("synthetic Q4 cancellation")
        super().raise_if_cancelled(stage)


@pytest.fixture(scope="module")
def loaded():
    source = Q4OwnedRebaseCandidate()
    prefix = source.legacy_prefix()
    assert prefix.status == "paused" and prefix.state.cursor == 2
    return source, prefix


def test_q4_owner_rebase_direct_restore_and_changed_successor(loaded, monkeypatch):
    source, prefix = loaded
    frozen = canonical(prefix.state.descriptor())
    original = prefix.checkpoint
    assert source.head_sha256 == _hash(original) and source.generation == 0
    assert np.linalg.norm(prefix.state.shell_u.reshape(-1, 6)[:, 3:]) > 1.e-8
    assert np.linalg.norm(prefix.state.multipliers) > 1.e-8

    rebase = source.rebase(original, expected_sha256=_hash(original))
    assert source.head_sha256 == rebase.checkpoint_sha256 and source.generation == 1
    assert decode(rebase.checkpoint)["schema"] == SCHEMA
    assert canonical(prefix.state.descriptor()) == frozen and prefix.checkpoint == original
    assert rebase.state.shell_history == prefix.state.shell_history
    assert canonical(rebase.state.beam_histories) == canonical(prefix.state.beam_histories)
    assert np.count_nonzero(rebase.state.shell_u.reshape(-1, 6)[:, 3:]) == 0

    fresh = Q4OwnedRebaseCandidate()
    legacy_restore = fresh.owner._restore
    replayed_bases = []
    def validate_legacy_base(raw, expected_sha256, cancellation_token=None):
        assert raw == original and expected_sha256 == _hash(original)
        replayed_bases.append(raw)
        return legacy_restore(raw, expected_sha256, cancellation_token)
    monkeypatch.setattr(fresh.owner, "_restore", validate_legacy_base)
    restored = fresh.restore(rebase.checkpoint,
        expected_sha256=rebase.checkpoint_sha256)
    assert len(replayed_bases) == 1
    assert canonical(restored.descriptor()) == canonical(rebase.state.descriptor())
    assert fresh.head_sha256 == rebase.checkpoint_sha256 and fresh.generation == 1

    changed = fresh.continue_loaded(rebase.checkpoint,
        expected_sha256=rebase.checkpoint_sha256)
    assert len(replayed_bases) == 2
    assert changed.status == "ACCEPTED_LOADED_CURSOR_3"
    assert changed.state.cursor == 3 and fresh.generation == 2
    assert fresh.head_sha256 == changed.checkpoint_sha256
    assert canonical(changed.state.descriptor()) != canonical(restored.descriptor())
    assert np.linalg.norm(changed.state.shell_u-restored.shell_u) > 1.e-8
    assert changed.state.shell_origin == restored.shell_history
    assert canonical(changed.state.beam_origins) == canonical(restored.beam_histories)
    assert fresh.owner.targets[2] == .75

    resumed = Q4OwnedRebaseCandidate()
    monkeypatch.setattr(resumed, "_accepted_successor", lambda *args, **kwargs:
        pytest.fail("normal Q4 restore replayed the successor"))
    direct = resumed.restore(changed.checkpoint,
        expected_sha256=changed.checkpoint_sha256)
    assert canonical(direct.descriptor()) == canonical(changed.state.descriptor())
    assert resumed.head_sha256 == changed.checkpoint_sha256 and resumed.generation == 2

    control = source.owner.solve(checkpoint=original,
        expected_sha256=_hash(original), stop_after=3)
    assert control.status == "completed" and control.state.cursor == 3
    a = source.owner.assembly
    identity = np.tile(np.eye(3), (len(a.coordinates), 1, 1))
    old = deformation(np.asarray(a.coordinates), control.state.shell_u, identity)
    new = deformation(np.asarray(a.coordinates), direct.shell_u, direct.anchor)
    assert np.linalg.norm(old.current_positions-new.current_positions) < 1.e-8
    assert np.linalg.norm(old.rotations-new.rotations) < 1.e-8
    assert np.linalg.norm(control.state.mechanical.positions-direct.mechanical.positions) < 1.e-8
    assert np.linalg.norm(control.state.mechanical.resultants-direct.mechanical.resultants) < 1.e-8
    control_response, control_residual, _, _ = source.owner._evaluate(control.state,
        control.state.mechanical, control.state.shell_u,
        control.state.multipliers, 3)
    rebased_response, rebased_residual = resumed._anchored_evaluate(restored,
        direct.mechanical, direct.shell_u, direct.multipliers, 3)
    assert np.linalg.norm(control_residual-rebased_residual) < 1.e-8
    assert np.linalg.norm(control_response.constraints-rebased_response.constraints) < 1.e-8
    _, _, old_work, _ = shell_spatial(a, decode(control.state.shell_origin),
        old, control.state.shell_u)
    _, _, new_work, _ = shell_spatial(a, decode(direct.shell_origin),
        new, direct.shell_u)
    assert abs(old_work-new_work) < 1.e-8


@pytest.mark.parametrize("mutation", ("policy", "definition", "initial", "record"))
def test_q4_rejects_consistently_rehashed_legacy_base(loaded, mutation):
    _, prefix = loaded
    source = Q4OwnedRebaseCandidate()
    rebase = source.rebase(prefix.checkpoint,
        expected_sha256=_hash(prefix.checkpoint))
    event = decode(rebase.checkpoint)
    base = decode(prefix.checkpoint)
    if mutation == "policy":
        base["policy"] = "tampered-policy"
    elif mutation == "definition":
        base["definition"]["targets"][0] += .01
    elif mutation == "initial":
        base["initial"]["cursor"] = 1
    else:
        base["records"][0]["metric"] += .01
        body = {key: value for key, value in base["records"][0].items()
            if key != "record_sha256"}
        base["records"][0]["record_sha256"] = _hash(canonical(body))
    base_raw = canonical(base)
    event["base_checkpoint"] = base_raw.decode("ascii")
    event["base_sha256"] = _hash(base_raw)
    event_raw = canonical(event)
    candidate = Q4OwnedRebaseCandidate()
    with pytest.raises(ValueError, match="checkpoint|binding|replay|disagreement"):
        candidate.restore(event_raw, expected_sha256=_hash(event_raw))
    assert candidate.head_sha256 is None and candidate.generation is None


def test_q4_owner_rejects_stale_and_foreign_checkpoints(loaded):
    _, prefix = loaded
    source = Q4OwnedRebaseCandidate()
    rebase = source.rebase(prefix.checkpoint,
        expected_sha256=_hash(prefix.checkpoint))
    assert source.generation == 1
    with pytest.raises(ValueError, match="stale base"):
        source.rebase(prefix.checkpoint, expected_sha256=_hash(prefix.checkpoint))
    with pytest.raises(ValueError, match="external F06 prototype checkpoint hash"):
        Q4OwnedRebaseCandidate().restore(b"{}", expected_sha256="0"*64)
    foreign = decode(rebase.checkpoint)
    foreign["schema"] = "G0_F06_CURVED_S3_OWNED_REBASE_PROTOTYPE_V1"
    foreign_raw = canonical(foreign)
    with pytest.raises(ValueError, match="definition|schema|identity"):
        Q4OwnedRebaseCandidate().restore(foreign_raw,
            expected_sha256=_hash(foreign_raw))


def test_q4_cancellation_does_not_publish_rebase(loaded):
    _, prefix = loaded
    candidate = Q4OwnedRebaseCandidate()
    with pytest.raises(SolveCancelled):
        candidate.rebase(prefix.checkpoint,
            expected_sha256=_hash(prefix.checkpoint),
            cancellation_token=_CancelAt("f06.rebase.before_commit"))
    assert candidate.head_sha256 is None and candidate.generation is None


def test_q4_rehashed_anchor_and_successor_are_rejected(loaded):
    _, prefix = loaded
    source = Q4OwnedRebaseCandidate()
    rebase = source.rebase(prefix.checkpoint,
        expected_sha256=_hash(prefix.checkpoint))
    changed = source.continue_loaded(rebase.checkpoint,
        expected_sha256=rebase.checkpoint_sha256)
    assert source.head_sha256 == changed.checkpoint_sha256 and source.generation == 2
    with pytest.raises(ValueError, match="stale successor predecessor"):
        source.continue_loaded(rebase.checkpoint,
            expected_sha256=rebase.checkpoint_sha256)
    assert source.head_sha256 == changed.checkpoint_sha256 and source.generation == 2

    bad_anchor = decode(rebase.checkpoint)
    bad_anchor["rebased_state"]["anchor"][1][0][0] += .01
    bad_anchor["anchor"][1][0][0] += .01
    bad_anchor["rebased_state_sha256"] = _hash(canonical(bad_anchor["rebased_state"]))
    bad_anchor_raw = canonical(bad_anchor)
    with pytest.raises(ValueError, match="rebase event"):
        Q4OwnedRebaseCandidate().restore(bad_anchor_raw,
            expected_sha256=_hash(bad_anchor_raw))

    bad_successor = decode(changed.checkpoint)
    bad_successor["successor"]["state"]["shell_u"][8] += .01
    bad_successor_raw = canonical(bad_successor)
    with pytest.raises(ValueError, match="successor|trial|history|gap"):
        Q4OwnedRebaseCandidate().restore(bad_successor_raw,
            expected_sha256=_hash(bad_successor_raw))


def test_q4_cancellation_does_not_publish_successor(loaded):
    _, prefix = loaded
    source = Q4OwnedRebaseCandidate()
    rebase = source.rebase(prefix.checkpoint,
        expected_sha256=_hash(prefix.checkpoint))
    with pytest.raises(SolveCancelled):
        source.continue_loaded(rebase.checkpoint,
            expected_sha256=rebase.checkpoint_sha256,
            cancellation_token=_CancelAt("f06.successor.before_commit"))
    assert source.head_sha256 == rebase.checkpoint_sha256 and source.generation == 1
