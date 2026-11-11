"""Development-only accepted beam-chart envelope around the curved S3 v4 owner.

The v4 mechanical solver still advances physical B3 matrices. This owner holds
the equivalent nodal and retained-cell chart coordinates in its accepted state
and restart bytes; it does not claim a production chart-coordinate beam solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_shell_joint_state import decode, decoded_array
from scripts.probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4 import (
    OwnedRebasePrototype, PrototypeRun,
)


SCHEMA = "G0_F06_CURVED_S3_BEAM_ACCEPTED_CHART_DEV_V5"


def digest(raw):
    return sha256(raw).hexdigest()


def _owned(value):
    result = np.array(value, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite beam chart")
    result.setflags(write=False)
    return result


def _chart(physical, anchor):
    frames = np.asarray(physical, dtype=float).reshape(-1, 3, 3)
    bases = np.asarray(anchor, dtype=float).reshape(-1, 3, 3)
    if frames.shape != bases.shape:
        raise ValueError("beam chart frame shape mismatch")
    theta = Rotation.from_matrix(frames @ bases.transpose(0, 2, 1)).as_rotvec()
    return _owned(theta.reshape(np.asarray(physical).shape[:-1]))


def _verify_frames(physical, anchor, theta):
    frames = np.asarray(physical, dtype=float).reshape(-1, 3, 3)
    bases = np.asarray(anchor, dtype=float).reshape(-1, 3, 3)
    coords = np.asarray(theta, dtype=float).reshape(-1, 3)
    if frames.shape != bases.shape or len(coords) != len(frames):
        raise ValueError("beam accepted chart shape mismatch")
    for name, value in (("physical", frames), ("anchor", bases), ("coordinate", coords)):
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite beam {name}")
    identity = np.eye(3)
    if (np.max(np.abs(bases @ bases.transpose(0, 2, 1) - identity)) > 1e-9 or
            np.min(np.linalg.det(bases)) < 1-1e-9):
        raise ValueError("beam accepted anchor is not a rotation")
    made = Rotation.from_rotvec(np.array(coords, copy=True)).as_matrix() @ bases
    if np.max(np.abs(made - frames)) > 1e-9:
        raise ValueError("beam accepted chart does not reconstruct physical frames")


@dataclass(frozen=True)
class AcceptedBeamChart:
    inner: object
    nodal_anchor: np.ndarray
    nodal_theta: np.ndarray
    cell_anchor: np.ndarray
    cell_theta: np.ndarray

    @classmethod
    def from_inner(cls, inner, nodal_anchor=None, cell_anchor=None):
        mechanical = inner.mechanical
        if nodal_anchor is None:
            nodal_anchor = mechanical.nodal_frames
        if cell_anchor is None:
            cell_anchor = mechanical.cell_rotations
        result = cls(inner, _owned(nodal_anchor),
            _chart(mechanical.nodal_frames, nodal_anchor), _owned(cell_anchor),
            _chart(mechanical.cell_rotations, cell_anchor))
        result.validate()
        return result

    @property
    def cursor(self):
        return self.inner.cursor

    def validate(self):
        mechanical = self.inner.mechanical
        if self.nodal_anchor.shape != mechanical.nodal_frames.shape or \
                self.nodal_theta.shape != (len(mechanical.nodal_frames), 3) or \
                self.cell_anchor.shape != mechanical.cell_rotations.shape or \
                self.cell_theta.shape != mechanical.cell_rotations.shape[:-1]:
            raise ValueError("beam accepted chart inventory mismatch")
        _verify_frames(mechanical.nodal_frames, self.nodal_anchor, self.nodal_theta)
        _verify_frames(mechanical.cell_rotations, self.cell_anchor, self.cell_theta)

    def descriptor(self):
        return dict(inner=self.inner.descriptor(), nodal_anchor=self.nodal_anchor,
            nodal_theta=self.nodal_theta, cell_anchor=self.cell_anchor,
            cell_theta=self.cell_theta)


@dataclass(frozen=True)
class ChartRun:
    state: AcceptedBeamChart
    checkpoint: bytes
    status: str

    @property
    def checkpoint_sha256(self):
        return digest(self.checkpoint)


class BeamChartV5Owner:
    """Stages v4 transactions, then commits one charted accepted checkpoint."""

    def __init__(self):
        self._lock = Lock()
        self._head = None

    @property
    def head_sha256(self):
        return self._head

    @staticmethod
    def prefix():
        return OwnedRebasePrototype().legacy_prefix()

    @staticmethod
    def _issue(inner_run, state, rebase_sha256):
        event = dict(schema=SCHEMA, inner_checkpoint=inner_run.checkpoint.decode("ascii"),
            inner_sha256=digest(inner_run.checkpoint), rebase_sha256=rebase_sha256,
            accepted=state.descriptor(), production_qualified=False)
        return ChartRun(state, canonical(event), inner_run.status)

    @staticmethod
    def _decode(raw, expected_sha256):
        if type(raw) is not bytes or digest(raw) != expected_sha256:
            raise ValueError("external v5 checkpoint hash required")
        event = decode(raw)
        if (type(event) is not dict or set(event) != {"schema", "inner_checkpoint",
                "inner_sha256", "rebase_sha256", "accepted", "production_qualified"} or
                event["schema"] != SCHEMA or event["production_qualified"] is not False or
                type(event["inner_checkpoint"]) is not str or canonical(event) != raw):
            raise ValueError("v5 accepted checkpoint schema mismatch")
        inner_raw = event["inner_checkpoint"].encode("ascii")
        if digest(inner_raw) != event["inner_sha256"]:
            raise ValueError("v5 inner checkpoint hash mismatch")
        return event, inner_raw

    @classmethod
    def _validated(cls, raw, expected_sha256):
        event, inner_raw = cls._decode(raw, expected_sha256)
        inner_owner = OwnedRebasePrototype()
        inner = inner_owner.restore(inner_raw, expected_sha256=event["inner_sha256"])
        inner_event = decode(inner_raw)
        rebased = inner_event["rebased_state"]["mechanical"]
        beam = inner_owner.owner.assembly.beam
        rebased_mechanical = beam.make(rebased, decoded=True)
        data = event["accepted"]
        if type(data) is not dict or set(data) != {"inner", "nodal_anchor",
                "nodal_theta", "cell_anchor", "cell_theta"}:
            raise ValueError("v5 accepted chart fields mismatch")
        state = AcceptedBeamChart(inner,
            decoded_array(data["nodal_anchor"], rebased_mechanical.nodal_frames.shape),
            decoded_array(data["nodal_theta"], (len(rebased_mechanical.nodal_frames), 3)),
            decoded_array(data["cell_anchor"], rebased_mechanical.cell_rotations.shape),
            decoded_array(data["cell_theta"], rebased_mechanical.cell_rotations.shape[:-1]))
        state.validate()
        if canonical(state.descriptor()) != canonical(data):
            raise ValueError("v5 accepted chart encoding mismatch")
        if (canonical(state.nodal_anchor) != canonical(rebased_mechanical.nodal_frames) or
                canonical(state.cell_anchor) != canonical(rebased_mechanical.cell_rotations)):
            raise ValueError("v5 beam anchors disagree with rebase event")
        if state.cursor == 2:
            if event["rebase_sha256"] != event["inner_sha256"]:
                raise ValueError("v5 rebase lineage mismatch")
            if np.any(state.nodal_theta) or np.any(state.cell_theta):
                raise ValueError("v5 rebased chart coordinates must be zero")
        elif state.cursor == 3:
            rebased_inner = inner_owner._snapshot_state(inner_event["rebased_state"])
            rebased_state = AcceptedBeamChart.from_inner(rebased_inner)
            rebased_raw = canonical({**inner_event, "successor": None})
            prior = cls._issue(PrototypeRun(rebased_inner, rebased_raw,
                "REBASED_AT_LOADED_CURSOR_2"), rebased_state, digest(rebased_raw))
            if event["rebase_sha256"] != prior.checkpoint_sha256:
                raise ValueError("v5 successor rebase lineage mismatch")
        else:
            raise ValueError("v5 cursor mismatch")
        return state, event, inner_owner

    def rebase(self, base, *, expected_sha256):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent v5 operation")
        try:
            if self._head is not None:
                raise ValueError("v5 base generation already consumed")
            staged = OwnedRebasePrototype()
            run = staged.rebase(base, expected_sha256=expected_sha256)
            state = AcceptedBeamChart.from_inner(run.state)
            issued = self._issue(run, state, digest(run.checkpoint))
            self._validated(issued.checkpoint, issued.checkpoint_sha256)
            self._head = issued.checkpoint_sha256
            return issued
        finally:
            self._lock.release()

    def restore(self, checkpoint, *, expected_sha256):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent v5 operation")
        try:
            state, _, _ = self._validated(checkpoint, expected_sha256)
            if self._head is not None and self._head != expected_sha256:
                raise ValueError("v5 stale restore disagrees with accepted head")
            self._head = expected_sha256
            return state
        finally:
            self._lock.release()

    def continue_loaded(self, checkpoint, *, expected_sha256):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent v5 operation")
        try:
            if self._head is not None and self._head != expected_sha256:
                raise ValueError("v5 stale continuation predecessor")
            state, event, staged = self._validated(checkpoint, expected_sha256)
            if state.cursor != 2:
                raise ValueError("one loaded v5 successor required")
            inner_run = staged.continue_loaded(
                event["inner_checkpoint"].encode("ascii"),
                expected_sha256=event["inner_sha256"])
            successor = AcceptedBeamChart.from_inner(inner_run.state,
                state.nodal_anchor, state.cell_anchor)
            issued = self._issue(inner_run, successor, expected_sha256)
            self._validated(issued.checkpoint, issued.checkpoint_sha256)
            self._head = issued.checkpoint_sha256
            return issued
        finally:
            self._lock.release()
