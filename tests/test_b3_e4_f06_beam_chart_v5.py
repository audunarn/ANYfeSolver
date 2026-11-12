"""Bounded loaded S3 accepted beam-chart development check."""

import os
from pathlib import Path
import tempfile

import numpy as np
import pytest

from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_shell_joint_state import decode

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "f06_v5_numba"))

from scripts.probe_b3_ge_e4_pl_g0_f06_beam_chart_v5 import (
    BeamChartV5Owner, digest,
)


def test_loaded_beam_chart_rebase_restore_and_successor():
    prototype = BeamChartV5Owner()
    prefix = prototype.prefix()
    assert prefix.status == "paused" and prefix.state.cursor == 2
    frozen = canonical(prefix.state.descriptor())
    rebased = prototype.rebase(prefix.checkpoint,
        expected_sha256=digest(prefix.checkpoint))
    assert rebased.state.cursor == 2
    assert prototype.head_sha256 == rebased.checkpoint_sha256
    assert canonical(prefix.state.descriptor()) == frozen
    assert np.linalg.norm(prefix.state.mechanical.nodal_frames -
        rebased.state.nodal_anchor) == 0
    assert np.linalg.norm(prefix.state.mechanical.cell_rotations -
        rebased.state.cell_anchor) == 0
    assert not np.any(rebased.state.nodal_theta)
    assert not np.any(rebased.state.cell_theta)
    assert rebased.state.inner.beam_histories == prefix.state.beam_histories
    assert rebased.state.inner.shell_history == prefix.state.shell_history

    fresh = BeamChartV5Owner()
    restored = fresh.restore(rebased.checkpoint,
        expected_sha256=rebased.checkpoint_sha256)
    assert canonical(restored.descriptor()) == canonical(rebased.state.descriptor())

    # Rehashed semantic corruption must fail before the accepted head moves.
    event = decode(rebased.checkpoint)
    event["accepted"]["nodal_anchor"][0][0][0] += 1.e-3
    bad = canonical(event)
    with pytest.raises(ValueError):
        fresh.restore(bad, expected_sha256=digest(bad))
    assert fresh.head_sha256 == rebased.checkpoint_sha256

    successor = fresh.continue_loaded(rebased.checkpoint,
        expected_sha256=rebased.checkpoint_sha256)
    assert successor.state.cursor == 3
    assert fresh.head_sha256 == successor.checkpoint_sha256
    assert np.max(np.linalg.norm(successor.state.nodal_theta, axis=1)) > 1.e-5
    assert np.max(np.linalg.norm(successor.state.cell_theta.reshape(-1, 3),
        axis=1)) > 1.e-5
    assert canonical(successor.state.nodal_anchor) == canonical(rebased.state.nodal_anchor)
    assert canonical(successor.state.cell_anchor) == canonical(rebased.state.cell_anchor)
    assert successor.state.inner.beam_origins == rebased.state.inner.beam_histories
    assert successor.state.inner.shell_origin == rebased.state.inner.shell_history

    restarted = BeamChartV5Owner()
    recovered = restarted.restore(successor.checkpoint,
        expected_sha256=successor.checkpoint_sha256)
    assert canonical(recovered.descriptor()) == canonical(successor.state.descriptor())
    wrong_lineage = decode(successor.checkpoint)
    wrong_lineage["rebase_sha256"] = "0" * 64
    bad_successor = canonical(wrong_lineage)
    with pytest.raises(ValueError, match="lineage"):
        restarted.restore(bad_successor, expected_sha256=digest(bad_successor))
    assert restarted.head_sha256 == successor.checkpoint_sha256
    with pytest.raises(ValueError, match="stale"):
        fresh.continue_loaded(rebased.checkpoint,
            expected_sha256=rebased.checkpoint_sha256)
