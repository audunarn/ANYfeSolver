"""G0-only owner-issued loaded chart rebase and restart prototype.

This is deliberately separate from the V1 coupled owner and its checkpoint.
It admits only the curved S3 fixture below and one rebase at cursor two.
Every fresh F06 control, prototype owner and private checkpoint replay in this
path takes its Newton step from the private deterministic fixed-order binary64
reference solve below; the public CoupledShellBeamAnalysis default remains
numpy.linalg.solve and legacy checkpoints keep that backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import perf_counter
import argparse
import os
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from anyfesolver.control import CancellationToken, SolveCancelled, cancellation_safe_point
from anyfesolver._ge_beam3_g3c_local_shell import deformation
from anyfesolver._ge_beam3_pose_joint import _exp_terms
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver._ge_beam3_shell_joint_state import (
    CoupledShellBeamAnalysis, SCHEMA as LEGACY_SCHEMA, decode, decoded_array,
)
from anyfesolver._ge_beam3_shell_joint_trial import ShellBeamTrial
from anyfesolver._native_reference_modal import _owned

from scripts.probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3 import fixture, shell_spatial


SCHEMA = "G0_F06_CURVED_S3_OWNED_REBASE_PROTOTYPE_V1"
TARGETS = (.5, 1., .75)
PILOT_FLOORS = {
    "translation": (1.e-8, "length"),
    "rotation": (1.e-9, "radian"),
    "force": (1.e-10, "force"),
    "moment": (1.e-10, "moment"),
    "work": (1.e-12, "force_length"),
    "strain": (1.e-12, "dimensionless_strain"),
    "curvature": (1.e-12, "inverse_length"),
    "stress": (1.e-10, "force_per_area"),
    "gap_translation": (1.e-10, "length"),
    "gap_rotation": (1.e-10, "radian"),
    "solver_scaled_residual": (1.e-12, "mixed_solver_scale"),
    "tangent_fd": (1.e-10, "scaled_directional_residual"),
}

# Private backend identifier.  The unchanged V1 measurement payload carries no
# backend key (the strict adjudicator allows only effective_environment as an
# extra field); the selected implementation stays bound by the full probe
# source_sha256 and the root records this identifier in ordinary receipt
# metadata outside the frozen metric schema.
REFERENCE_LINEAR_SOLVE_ID = (
    "F06_OWNED_REBASE_V4_PRIVATE_DETERMINISTIC_FIXED_ORDER_PARTIAL_PIVOT_LU_BINARY64_V1")


def _reference_linear_solve(matrix, rhs):
    """Private deterministic general LU solve for the F06 Newton step.

    Fixed-order binary64 partial-pivot LU on owned copies of both operands:
    the first row of maximal magnitude wins pivot ties, elimination and
    substitution use vectorized elementwise row updates in one fixed order
    (no BLAS dot reductions, no FMA reassociation), an exact zero pivot is
    refused with its column, complex operands are refused before any real
    cast so imaginary parts are never discarded, nonfinite operands and
    nonfinite intermediate elimination or substitution updates are refused
    and no diagonal regularization is applied.  This backend is private to
    the F06 owned-rebase prototype path; every other CoupledShellBeamAnalysis
    caller keeps the numpy.linalg.solve default.
    """
    if np.iscomplexobj(matrix) or np.iscomplexobj(rhs):
        raise ValueError("F06 reference solve refuses complex operands")
    a = np.array(matrix, dtype=float, copy=True)
    b = np.array(rhs, dtype=float, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0]:
        raise ValueError("F06 reference solve requires a nonempty square matrix")
    n = a.shape[0]
    if b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError("F06 reference solve right-hand side shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("F06 reference solve nonfinite operand")
    x = b.reshape(n, 1) if b.ndim == 1 else b
    for k in range(n-1):
        pivot = k+int(np.argmax(np.abs(a[k:, k])))
        if a[pivot, k] == 0.:
            raise ValueError(
                f"F06 reference solve exactly singular: zero pivot column {k}")
        if pivot != k:
            a[[k, pivot]] = a[[pivot, k]]
            x[[k, pivot]] = x[[pivot, k]]
        factors = a[k+1:, k]/a[k, k]
        if not np.isfinite(factors).all():
            raise ValueError("F06 reference solve nonfinite elimination multiplier")
        a[k+1:, k] = factors
        a[k+1:, k+1:] -= factors[:, None]*a[k, k+1:]
        x[k+1:] -= factors[:, None]*x[k]
        if not np.isfinite(a[k+1:, k+1:]).all() or not np.isfinite(x[k+1:]).all():
            raise ValueError(
                "F06 reference solve nonfinite intermediate elimination update")
    for j in range(n-1, -1, -1):
        if a[j, j] == 0.:
            raise ValueError(
                f"F06 reference solve exactly singular: zero pivot column {j}")
        x[j] = x[j]/a[j, j]
        if not np.isfinite(x[j]).all():
            raise ValueError(
                "F06 reference solve nonfinite intermediate substitution update")
        if j:
            x[:j] -= a[:j, j][:, None]*x[j]
            if not np.isfinite(x[:j]).all():
                raise ValueError(
                    "F06 reference solve nonfinite intermediate substitution update")
    if not np.isfinite(x).all():
        raise ValueError("F06 reference solve produced nonfinite values")
    return x.reshape(-1) if b.ndim == 1 else x


def _hash(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _near(a, b, label, tolerance=1.e-9):
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"F06 {label} shape/nonfinite disagreement")
    if np.linalg.norm(x-y) > tolerance*max(1., np.linalg.norm(x), np.linalg.norm(y)):
        raise ValueError(f"F06 {label} physical disagreement")


def _metric(left, right, floor_name):
    """Retain raw norms and the declared same-unit pilot normalization."""
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("F06 pilot metric shape/nonfinite disagreement")
    floor, unit = PILOT_FLOORS[floor_name]
    left_norm, right_norm = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    absolute = float(np.linalg.norm(x-y))
    scale = max(floor, left_norm, right_norm)
    return dict(unit=unit, floor=floor, left_values=x.tolist(), right_values=y.tolist(),
        left_norm=left_norm,
        right_norm=right_norm, absolute_difference=absolute,
        normalization_scale=scale, normalized_difference=absolute/scale)


def _indexed_metric(left, right, floor_names, index_names):
    """Keep every physical component and the worst same-unit comparison."""
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.shape != y.shape or x.ndim != len(index_names) or x.shape[-1] != len(floor_names):
        raise ValueError("F06 indexed metric layout disagreement")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("F06 indexed metric nonfinite value")
    floors = np.array([PILOT_FLOORS[name][0] for name in floor_names])
    scales = np.maximum(np.maximum(np.abs(x), np.abs(y)), floors)
    raw, normalized = np.abs(x-y), np.abs(x-y)/scales
    worst = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    return dict(index_names=index_names, component_roles=list(floor_names),
        component_units=[PILOT_FLOORS[name][1] for name in floor_names],
        component_floors=floors.tolist(), left_values=x.tolist(), right_values=y.tolist(),
        absolute_differences=raw.tolist(), normalization_scales=scales.tolist(),
        normalized_differences=normalized.tolist(),
        maximum=dict(indices=dict(zip(index_names, map(int, worst))),
            absolute_difference=float(raw[worst]),
            normalized_difference=float(normalized[worst]),
            left_value=float(x[worst]), right_value=float(y[worst]),
            scale=float(scales[worst])))


def _rotation_angles(left, right):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError("F06 pilot rotation frame shape mismatch")
    relative = a@np.swapaxes(b, -1, -2)
    sine = np.linalg.norm(np.stack((relative[:, 2, 1]-relative[:, 1, 2],
        relative[:, 0, 2]-relative[:, 2, 0],
        relative[:, 1, 0]-relative[:, 0, 1]), axis=1), axis=1)/2.
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1.)/2., -1., 1.)
    return np.arctan2(sine, cosine)


def _rotation_metric(left, right, reference):
    floor, unit = PILOT_FLOORS["rotation"]
    raw = float(np.linalg.norm(_rotation_angles(left, right)))
    left_norm = float(np.linalg.norm(_rotation_angles(left, reference)))
    right_norm = float(np.linalg.norm(_rotation_angles(right, reference)))
    scale = max(floor, left_norm, right_norm)
    return dict(unit=unit, floor=floor,
        left_frames=np.asarray(left, dtype=float).tolist(),
        right_frames=np.asarray(right, dtype=float).tolist(),
        reference_frames=np.asarray(reference, dtype=float).tolist(),
        left_norm=left_norm,
        right_norm=right_norm, absolute_difference=raw,
        normalization_scale=scale, normalized_difference=raw/scale)


def _beam_section_metrics(left_owner, left, right_owner, right):
    a = left_owner.assembly
    b = right_owner.assembly
    if a.beam.identity != b.beam.identity:
        raise ValueError("F06 paired beam definition changed")
    old, new = left_owner._beam_recovery(left), right_owner._beam_recovery(right)
    if len(old) != len(new) or len(old) != len(a.beam.elements):
        raise ValueError("F06 beam member inventory changed")
    station_ids, old_strain, new_strain, old_resultant, new_resultant = [], [], [], [], []
    split = {name: [[], []] for name in (
        "strain_high", "strain_low", "resultant_high", "resultant_low")}
    for member, (x, y) in enumerate(zip(old, new)):
        if x["element_id"] != y["element_id"] or len(x["stations"]) != len(y["stations"]):
            raise ValueError("F06 beam station inventory changed")
        ids, xs, ys, xr, yr = [], [], [], [], []
        member_split = {name: [[], []] for name in split}
        for p, q in zip(x["stations"], y["stations"]):
            if (p["cell"], p["station"], p["xi"]) != (q["cell"], q["station"], q["xi"]):
                raise ValueError("F06 beam recovery station identity changed")
            ids.append(dict(cell=int(p["cell"]), station=int(p["station"]), xi=float(p["xi"])))
            for key, field in (("strain_high", "strain"),
                    ("strain_low", "strain_low"),
                    ("resultant_high", "resultants"),
                    ("resultant_low", "resultants_low")):
                member_split[key][0].append(np.asarray(p[field]))
                member_split[key][1].append(np.asarray(q[field]))
            xs.append(np.asarray(p["strain"])+np.asarray(p["strain_low"]))
            ys.append(np.asarray(q["strain"])+np.asarray(q["strain_low"]))
            xr.append(np.asarray(p["resultants"])+np.asarray(p["resultants_low"]))
            yr.append(np.asarray(q["resultants"])+np.asarray(q["resultants_low"]))
        station_ids.append(dict(member_index=member, element_id=int(x["element_id"]), stations=ids))
        for key in split:
            split[key][0].append(member_split[key][0])
            split[key][1].append(member_split[key][1])
        old_strain.append(xs); new_strain.append(ys)
        old_resultant.append(xr); new_resultant.append(yr)
    retained = _indexed_metric(left.mechanical.resultants, right.mechanical.resultants,
        ["force"]*6+["moment"]*12, ("member", "component"))
    retained["component_labels"] = [
        f"cell_{cell}_translation_conjugate_{axis}_force"
        for cell in range(2) for axis in ("x", "y", "z")
    ] + [
        f"cell_{cell}_{end}_rotation_conjugate_{axis}_moment"
        for cell in range(2) for end in ("left", "right")
        for axis in ("x", "y", "z")]
    strain = _indexed_metric(old_strain, new_strain,
        ["strain"]*3+["curvature"]*3, ("member", "station", "component"))
    resultant = _indexed_metric(old_resultant, new_resultant,
        ["force"]*3+["moment"]*3, ("member", "station", "component"))
    split_metrics = {key: _indexed_metric(values[0], values[1],
        (["strain"]*3+["curvature"]*3) if key.startswith("strain") else
        (["force"]*3+["moment"]*3), ("member", "station", "component"))
        for key, values in split.items()}
    return dict(station_identity=station_ids, retained_resultants=retained,
        section_strain=strain, section_resultants=resultant,
        section_split=split_metrics)


def physical_tangent(response, u, count):
    """Columns in left-spatial shell increments; other columns unchanged."""
    p = np.eye(count)
    for i, row in enumerate(u.reshape(-1, 6)):
        _, jac, _ = _exp_terms(row[3:])
        p[6*i+3:6*i+6, 6*i+3:6*i+6] = jac
    return np.linalg.solve(p.T, response.tangent.T).T


def _pilot_direction(assembly, shell_u):
    """Fixed nonzero shell, beam/retained and multiplier physical direction."""
    spatial = np.zeros(assembly.count)
    rows = spatial[:assembly.shell_count].reshape(-1, 6)
    rows[1] = (.13, -.09, .07, .31, -.23, .17)
    rows[2] = (-.08, .12, -.04, -.19, .29, .11)
    beam = spatial[assembly.shell_count:-6]
    for index, dof in enumerate(assembly.beam.free[:9]):
        beam[dof] = .11*np.sin(index+1)
    for index, dof in enumerate(assembly.beam.free[-8:]):
        beam[dof] = .07*np.cos(index+1)
    spatial[-6:] = (.09, -.06, .04, -.03, .05, -.02)
    spatial /= np.linalg.norm(spatial)
    chart = spatial.copy()
    for index, row in enumerate(shell_u.reshape(-1, 6)):
        sl = slice(6*index+3, 6*index+6)
        chart[sl] = np.linalg.solve(_exp_terms(row[3:])[1], spatial[sl])
    return spatial, chart


def _dead_external(owner, parameter, mechanical, shell_u, multipliers):
    """Shared frozen dead-load evaluator; state arguments permit FD of this call path."""
    return parameter*owner.forces


def _check_tangent_directions(old, old_u, new, new_u, count):
    left, right = physical_tangent(old, old_u, count), physical_tangent(new, new_u, count)
    rng = np.random.default_rng(606)
    for index in range(3):
        direction = rng.standard_normal(count)
        direction /= np.linalg.norm(direction)
        _near(left@direction, right@direction, f"chart tangent direction {index}")


@dataclass(frozen=True)
class PrototypeState:
    mechanical: object
    shell_u: np.ndarray
    multipliers: np.ndarray
    beam_origins: tuple
    beam_histories: tuple
    shell_origin: bytes
    shell_history: bytes
    anchor: np.ndarray
    cursor: int

    def descriptor(self):
        return dict(mechanical=self.mechanical, shell_u=self.shell_u,
            multipliers=self.multipliers, beam_origins=self.beam_origins,
            beam_histories=self.beam_histories, shell_origin=decode(self.shell_origin),
            shell_history=decode(self.shell_history), anchor=self.anchor, cursor=self.cursor)


@dataclass(frozen=True)
class PrototypeRun:
    state: PrototypeState
    checkpoint: bytes
    status: str

    @property
    def checkpoint_sha256(self):
        return _hash(self.checkpoint)


class OwnedRebasePrototype:
    """Owns an event checkpoint and the one anchored successor transaction."""

    def __init__(self):
        seed = fixture("S3-V2D", True)
        self.owner = CoupledShellBeamAnalysis(seed.assembly, targets=TARGETS,
            shell_fixed=seed.shell_fixed, nodal_forces=seed.forces,
            max_iterations=seed.max_iterations, max_backtracks=seed.max_backtracks,
            _linear_solve=_reference_linear_solve)
        self._lock = Lock()
        self._consumed_base = set()
        self._head_sha256 = None
        self._generation = None

    @property
    def head_sha256(self):
        return self._head_sha256

    @property
    def generation(self):
        return self._generation

    def legacy_prefix(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent F06 prototype operation")
        try:
            run = self.owner.solve(stop_after=2)
            if run.status == "paused" and run.state.cursor == 2 and self._head_sha256 is None:
                self._head_sha256, self._generation = _hash(run.checkpoint), 0
            return run
        finally:
            self._lock.release()

    def _anchored_evaluate(self, predecessor, mechanical, u, multipliers, cursor):
        owner = self.owner; a = owner.assembly
        owner.guard()
        mechanical, u, multipliers = owner._arrays(mechanical, u, multipliers)
        parameter = owner.targets[cursor-1]
        origins, shell_origin = predecessor.beam_histories, predecessor.shell_history
        br, bj, _, responses, _ = a.beam.assemble(mechanical, parameter, origins)
        kin = deformation(np.asarray(a.coordinates), u, predecessor.anchor)
        sr, spatial_jac, _, candidate = shell_spatial(a, decode(shell_origin), kin, u)
        p = np.eye(a.shell_count)
        for i, row in enumerate(u.reshape(-1, 6)):
            _, jac, _ = _exp_terms(row[3:])
            p[6*i+3:6*i+6, 6*i+3:6*i+6] = jac
        sj = spatial_jac@p
        bi = a.beam.index[a.beam_node]
        shell = u.reshape(-1, 6)[a.shell_node]
        ports = a.joint.evaluate_ports(
            np.array([a.coordinates[a.shell_node]+shell[:3], mechanical.positions[bi]]),
            np.array([predecessor.anchor[a.shell_node]@a.frame,
                      mechanical.nodal_frames[bi]]), multipliers,
            rotation_coordinates=np.array([shell[3:], np.zeros(3)]),
            charted_ports=(True, False),
            position_low=np.array([np.zeros(3), mechanical.position_low[bi]]))
        jr = ports.pose.residual
        jj = ports.pose.spatial_jacobian@ports.coordinate_map
        residual = np.r_[sr, br, np.zeros(6)]
        tangent = np.zeros((a.count, a.count))
        tangent[:a.shell_count, :a.shell_count] = sj
        tangent[a.shell_count:-6, a.shell_count:-6] = bj
        residual[list(a.slots)] += jr
        tangent[np.ix_(a.slots, a.slots)] += jj
        if not np.isfinite(residual).all() or not np.isfinite(tangent).all():
            raise ValueError("nonfinite anchored F06 trial")
        fingerprint = _hash(canonical(dict(mechanical=mechanical, shell_u=u,
            multipliers=multipliers, anchor=predecessor.anchor, parameter=parameter)))
        response = ShellBeamTrial(a.identity, _hash(canonical(predecessor.descriptor())),
            fingerprint, _owned(residual), _owned(tangent), ports.pose.constraints,
            _owned(jr), _owned(jj), canonical(candidate),
            canonical(tuple(r.history for r in responses)))
        loaded = residual.copy()
        loaded[:len(owner.forces)] -= _dead_external(owner, parameter,
            mechanical, u, multipliers)
        owner.guard()
        return response, loaded

    def _rebased_state(self, base):
        a = self.owner.assembly; n = len(a.coordinates)
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
        old_response, old_residual, _, _ = self.owner._evaluate(base, base.mechanical,
            base.shell_u, base.multipliers, base.cursor)
        new_response, new_residual = self._anchored_evaluate(
            state, state.mechanical, state.shell_u, state.multipliers, state.cursor)
        _near(old_response.constraints, new_response.constraints, "joint gap")
        _near(old_residual, new_residual, "loaded residual")
        _check_tangent_directions(old_response, base.shell_u,
            new_response, state.shell_u, a.count)
        # Resolve translational forces and rotational moments separately.
        for axis, name in ((slice(0, 3), "force"), (slice(3, 6), "moment")):
            _near(old_residual[:a.shell_count].reshape(-1, 6)[:, axis],
                new_residual[:a.shell_count].reshape(-1, 6)[:, axis], name)
        origin = decode(base.shell_history)
        _, _, old_work, old_candidate = shell_spatial(a, origin, old, base.shell_u)
        _, _, new_work, new_candidate = shell_spatial(a, origin, new, state.shell_u)
        _near(old_work, new_work, "shell virtual work")
        for name in ("layer_strain", "layer_stress"):
            x, y = np.asarray(old_candidate[name]), np.asarray(new_candidate[name])
            if np.linalg.norm(x) <= 1.e-8 or np.linalg.norm(y) <= 1.e-8:
                raise ValueError(f"nontrivial S3 {name} response required")
            _near(x, y, name)
        if canonical(old_candidate) != canonical(new_candidate):
            # Numerical fields may differ in their final bits across charts.
            from scripts.probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3 import numeric_state
            p0, v0 = numeric_state(old_candidate)
            p1, v1 = numeric_state(new_candidate)
            if p0 != p1:
                raise ValueError("F06 shell material state paths changed")
            _near(v0, v1, "shell material history")
        if (state.beam_origins != base.beam_origins or
                canonical(state.beam_histories) != canonical(base.beam_histories) or
                state.shell_origin != base.shell_origin or
                state.shell_history != base.shell_history):
            raise ValueError("F06 accepted material history changed")
        return state

    def _event(self, base_checkpoint, base, state):
        return dict(schema=SCHEMA, definition_sha256=self.owner.identity,
            base_checkpoint=base_checkpoint.decode("ascii"),
            base_sha256=_hash(base_checkpoint), cursor=base.cursor,
            base_state=base.descriptor(),
            base_state_sha256=_hash(canonical(base.descriptor())),
            anchor=state.anchor, rebased_state=state.descriptor(),
            rebased_state_sha256=_hash(canonical(state.descriptor())),
            successor=None, production_qualified=False)

    def _snapshot_state(self, data, cursor=2):
        if type(data) is not dict or set(data) != {
                "mechanical", "shell_u", "multipliers", "beam_origins",
                "beam_histories", "shell_origin", "shell_history", "anchor", "cursor"}:
            raise ValueError("F06 direct rebased snapshot shape mismatch")
        if type(data["cursor"]) is not int or data["cursor"] != cursor:
            raise ValueError("F06 direct accepted snapshot cursor mismatch")
        beam = self.owner.assembly.beam
        n = len(self.owner.assembly.coordinates)
        state = PrototypeState(
            beam.make(data["mechanical"], decoded=True),
            decoded_array(data["shell_u"], (6*n,)),
            decoded_array(data["multipliers"], (6,)),
            beam.histories(data["beam_origins"], decoded=True),
            beam.histories(data["beam_histories"], decoded=True),
            canonical(data["shell_origin"]), canonical(data["shell_history"]),
            decoded_array(data["anchor"], (n, 3, 3)), cursor)
        if canonical(state.descriptor()) != canonical(data):
            raise ValueError("F06 direct rebased snapshot encoding mismatch")
        return state

    def _captured_base(self, raw, expected_sha256, captured):
        """Bind the directly captured cursor-two state to the V1 bytes."""
        if _hash(raw) != expected_sha256:
            raise ValueError("F06 prototype base hash mismatch")
        outer = decode(raw)
        if (type(outer) is not dict or set(outer) != {
                "schema", "policy", "definition_sha256", "definition", "initial",
                "records", "production_qualified"} or
                outer["schema"] != LEGACY_SCHEMA or
                outer["definition_sha256"] != self.owner.identity or
                outer["production_qualified"] is not False or
                type(outer["records"]) is not list or len(outer["records"]) != 2 or
                canonical(outer) != raw or
                canonical(outer["records"][-1]["state"]) != canonical(captured)):
            raise ValueError("F06 captured base/checkpoint state mismatch")
        n = len(self.owner.assembly.coordinates)
        fields = {key: captured[key] for key in (
            "mechanical", "shell_u", "multipliers", "beam_origins",
            "beam_histories", "shell_origin", "shell_history", "cursor")}
        fields["anchor"] = np.tile(np.eye(3), (n, 1, 1))
        return self._snapshot_state(fields)

    def rebase(self, checkpoint, *, expected_sha256, cancellation_token=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent F06 prototype operation")
        try:
            cancellation_safe_point(cancellation_token, "f06.rebase.start")
            if type(checkpoint) is not bytes or _hash(checkpoint) != expected_sha256:
                raise ValueError("external F06 base checkpoint hash required")
            if self._head_sha256 is not None and self._head_sha256 != expected_sha256:
                raise ValueError("F06 stale base is not this owner's accepted head")
            if expected_sha256 in self._consumed_base:
                raise ValueError("F06 stale base generation already rebased by this owner")
            base, records = self.owner._restore(checkpoint, expected_sha256, cancellation_token)
            if base.cursor != 2 or len(records) != 2:
                raise ValueError("loaded cursor-two F06 prefix required")
            before = canonical(base.descriptor())
            state = self._rebased_state(base)
            if canonical(base.descriptor()) != before:
                raise ValueError("F06 base state mutated")
            event = self._event(checkpoint, base, state)
            raw = canonical(event)
            cancellation_safe_point(cancellation_token, "f06.rebase.before_commit")
            self._consumed_base.add(expected_sha256)
            self._head_sha256, self._generation = _hash(raw), 1
            return PrototypeRun(state, raw, "REBASED_AT_LOADED_CURSOR_2")
        finally:
            self._lock.release()

    def _accepted_successor(self, predecessor, mechanical, u, multipliers, iterations,
                            predecessor_sha256):
        owner = self.owner
        if type(iterations) is not int or not 0 <= iterations <= owner.max_iterations:
            raise ValueError("bounded F06 successor iteration count required")
        if type(predecessor_sha256) is not str or len(predecessor_sha256) != 64:
            raise ValueError("F06 successor predecessor generation hash required")
        mechanical, u, multipliers = owner._arrays(mechanical, u, multipliers)
        response, residual = self._anchored_evaluate(predecessor, mechanical, u,
            multipliers, 3)
        _, metric, correction = owner._step(response, residual, mechanical, u, multipliers)
        if max(metric, correction, float(np.linalg.norm(response.constraints))) > 1.e-11:
            raise ValueError("F06 anchored successor not equilibrated")
        histories = owner.assembly.beam.histories(decode(response.beam_histories), decoded=True)
        state = PrototypeState(mechanical, _owned(u), _owned(multipliers),
            predecessor.beam_histories, histories, predecessor.shell_history,
            response.shell_candidate, predecessor.anchor, 3)
        record = dict(generation=2, predecessor_sha256=predecessor_sha256,
            iterations=iterations, state=state.descriptor(),
            residual=residual, constraints=response.constraints, metric=metric,
            correction=correction, tangent_sha256=_hash(canonical(response.tangent)))
        return state, record

    def _direct_successor(self, predecessor, row, predecessor_sha256):
        """Decode the issued state and check one current trial, without replay."""
        required = {"generation", "predecessor_sha256", "iterations", "state",
            "residual", "constraints", "metric", "correction", "tangent_sha256"}
        if (type(row) is not dict or set(row) != required or row["generation"] != 2 or
                row["predecessor_sha256"] != predecessor_sha256 or
                type(row["iterations"]) is not int or
                not 0 <= row["iterations"] <= self.owner.max_iterations or
                type(row["tangent_sha256"]) is not str or
                len(row["tangent_sha256"]) != 64):
            raise ValueError("F06 successor generation/record mismatch")
        state = self._snapshot_state(row["state"], cursor=3)
        if (canonical(state.anchor) != canonical(predecessor.anchor) or
                canonical(state.beam_origins) != canonical(predecessor.beam_histories) or
                state.shell_origin != predecessor.shell_history):
            raise ValueError("F06 direct successor material/chart lineage mismatch")
        owner = self.owner; a = owner.assembly
        owner._arrays(state.mechanical, state.shell_u, state.multipliers)
        kin = deformation(np.asarray(a.coordinates), state.shell_u, state.anchor)
        bi = a.beam.index[a.beam_node]
        shell = state.shell_u.reshape(-1, 6)[a.shell_node]
        joint = a.joint.evaluate(
            np.array([kin.current_positions[a.shell_node],
                      state.mechanical.positions[bi]]),
            np.array([kin.rotations[a.shell_node]@a.frame,
                      state.mechanical.nodal_frames[bi]]), state.multipliers,
            position_low=np.array([np.zeros(3), state.mechanical.position_low[bi]]))
        constraints = decoded_array(row["constraints"], (6,))
        _near(joint.constraints, constraints, "direct successor joint gap")
        residual = decoded_array(row["residual"], (a.count,))
        metric = float(np.linalg.norm((residual/owner.scale)[list(owner.free)]))
        if (type(row["metric"]) is not float or type(row["correction"]) is not float or
                not np.isfinite((row["metric"], row["correction"])).all() or
                abs(metric-row["metric"]) > 1.e-13 or
                max(metric, row["correction"], float(np.linalg.norm(constraints))) > 1.e-11):
            raise ValueError("F06 direct successor equilibrium record mismatch")
        response, actual_residual = self._anchored_evaluate(predecessor,
            state.mechanical, state.shell_u, state.multipliers, 3)
        if state.shell_history != response.shell_candidate:
            raise ValueError("F06 direct successor shell history disagreement")
        beam_candidate = a.beam.histories(decode(response.beam_histories), decoded=True)
        if canonical(state.beam_histories) != canonical(beam_candidate):
            raise ValueError("F06 direct successor beam history disagreement")
        if canonical(actual_residual) != canonical(residual):
            raise ValueError("F06 direct successor residual disagreement")
        if _hash(canonical(response.tangent)) != row["tangent_sha256"]:
            raise ValueError("F06 direct successor tangent disagreement")
        if canonical(response.constraints) != canonical(constraints):
            raise ValueError("F06 direct successor trial gap disagreement")
        _, actual_metric, actual_correction = owner._step(response,
            actual_residual, state.mechanical, state.shell_u, state.multipliers)
        if (abs(actual_metric-row["metric"]) > 1.e-13 or
                abs(actual_correction-row["correction"]) > 1.e-13 or
                max(actual_metric, actual_correction,
                    float(np.linalg.norm(response.constraints))) > 1.e-11):
            raise ValueError("F06 direct successor correction disagreement")
        return state

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
            raise ValueError("F06 prototype checkpoint schema/identity mismatch")
        base_raw = value["base_checkpoint"].encode("ascii")
        if value["base_state_sha256"] != _hash(canonical(value["base_state"])):
            raise ValueError("F06 captured base state hash mismatch")
        base = self._captured_base(base_raw, value["base_sha256"], value["base_state"])
        # The continuation consumes the rebased snapshot directly. It validates
        # captured geometry and constitutive response, without V1 history replay.
        state = self._snapshot_state(value["rebased_state"])
        if (canonical(value["anchor"]) != canonical(state.anchor) or
                value["rebased_state_sha256"] != _hash(canonical(state.descriptor()))):
            raise ValueError("F06 prototype rebase event mismatch")
        oracle = self._rebased_state(base)
        if canonical(state.descriptor()) != canonical(oracle.descriptor()):
            raise ValueError("F06 prototype rebase event mismatch")
        if value["successor"] is not None:
            predecessor_sha256 = _hash(canonical({**value, "successor": None}))
            state = self._direct_successor(state, value["successor"], predecessor_sha256)
        cancellation_safe_point(cancellation_token, "f06.restore.complete")
        return state, value

    def audit_legacy_replay(self, checkpoint, *, expected_sha256):
        """Optional full V1 replay oracle, never called by normal restoration."""
        state, event = self._restore(checkpoint, expected_sha256)
        base_raw = event["base_checkpoint"].encode("ascii")
        legacy, records = self.owner._restore(base_raw, event["base_sha256"])
        if legacy.cursor != 2 or len(records) != 2:
            raise ValueError("F06 diagnostic V1 replay prefix mismatch")
        if canonical(legacy.descriptor()) != canonical(event["base_state"]):
            raise ValueError("F06 diagnostic V1 replay disagrees with captured base")
        if canonical(self._rebased_state(legacy).descriptor()) != canonical(event["rebased_state"]):
            raise ValueError("F06 diagnostic V1 replay disagrees with rebase")
        if event["successor"] is not None:
            row = event["successor"]
            predecessor = self._snapshot_state(event["rebased_state"])
            data = row["state"]
            mechanical = self.owner.assembly.beam.make(data["mechanical"], decoded=True)
            u = decoded_array(data["shell_u"], (self.owner.assembly.shell_count,))
            multipliers = decoded_array(data["multipliers"], (6,))
            replayed, expected = self._accepted_successor(predecessor, mechanical,
                u, multipliers, row["iterations"], row["predecessor_sha256"])
            if canonical(row) != canonical(expected) or canonical(state.descriptor()) != canonical(replayed.descriptor()):
                raise ValueError("F06 diagnostic successor replay disagreement")
        return state

    def restore(self, checkpoint, *, expected_sha256, cancellation_token=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent F06 prototype operation")
        try:
            state, event = self._restore(checkpoint, expected_sha256, cancellation_token)
            predecessor_hash = _hash(canonical({**event, "successor": None}))
            admissible = {expected_sha256}
            if state.cursor == 2:
                admissible.add(event["base_sha256"])
            else:
                admissible.add(predecessor_hash)
            if self._head_sha256 is not None and self._head_sha256 not in admissible:
                raise ValueError("F06 stale restore disagrees with accepted head")
            cancellation_safe_point(cancellation_token, "f06.restore.before_commit")
            self._head_sha256 = expected_sha256
            self._generation = 1 if state.cursor == 2 else 2
            return state
        finally:
            self._lock.release()

    def continue_loaded(self, checkpoint, *, expected_sha256, cancellation_token=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent F06 prototype operation")
        try:
            if self._head_sha256 is not None and self._head_sha256 != expected_sha256:
                raise ValueError("F06 stale successor predecessor is not accepted head")
            accepted, event = self._restore(checkpoint, expected_sha256, cancellation_token)
            if accepted.cursor != 2 or event["successor"] is not None:
                raise ValueError("one F06 anchored successor from rebase event required")
            self._head_sha256, self._generation = expected_sha256, 1
            owner = self.owner; a = owner.assembly
            mechanical, u, lm = accepted.mechanical, accepted.shell_u, accepted.multipliers
            for iteration in range(owner.max_iterations+1):
                cancellation_safe_point(cancellation_token, "f06.successor.before_assembly")
                response, residual = self._anchored_evaluate(accepted, mechanical, u, lm, 3)
                step, metric, correction = owner._step(response, residual, mechanical, u, lm)
                if max(metric, correction, float(np.linalg.norm(response.constraints))) <= 1.e-11:
                    proposed, record = self._accepted_successor(accepted, mechanical, u,
                        lm, iteration, expected_sha256)
                    output = canonical({**event, "successor": record})
                    cancellation_safe_point(cancellation_token, "f06.successor.before_commit")
                    self._head_sha256, self._generation = _hash(output), 2
                    return PrototypeRun(proposed, output, "ACCEPTED_LOADED_CURSOR_3")
                if iteration == owner.max_iterations:
                    raise RuntimeError("F06 anchored Newton limit")
                for cut in range(owner.max_backtracks+1):
                    cancellation_safe_point(cancellation_token, "f06.successor.before_trial")
                    delta = step*(.5**cut)
                    try:
                        candidate = a.beam.advance(mechanical, delta[a.shell_count:-6])
                        cu, cm = u+delta[:a.shell_count], lm+delta[-6:]
                        _, trial = self._anchored_evaluate(accepted, candidate, cu, cm, 3)
                    except ValueError as error:
                        if "requires cutback" in str(error):
                            continue
                        raise
                    changed = float(np.linalg.norm((trial/owner.scale)[list(owner.free)]))
                    if changed < metric:
                        mechanical, u, lm = candidate, cu, cm
                        break
                else:
                    raise RuntimeError("F06 anchored line search limit")
        finally:
            self._lock.release()


def _paired_metrics(control_owner, control, prototype, rebased, cursor):
    a = control_owner.assembly
    n = len(a.coordinates)
    identity = np.tile(np.eye(3), (n, 1, 1))
    old_kin = deformation(np.asarray(a.coordinates), control.shell_u, identity)
    new_kin = deformation(np.asarray(a.coordinates), rebased.shell_u, rebased.anchor)
    old_trial, old_residual, _, _ = control_owner._evaluate(control,
        control.mechanical, control.shell_u, control.multipliers, cursor)
    new_trial, new_residual = prototype._anchored_evaluate(rebased,
        rebased.mechanical, rebased.shell_u, rebased.multipliers, cursor)
    old_shell = shell_spatial(a, decode(control.shell_origin), old_kin,
        control.shell_u)
    new_shell = shell_spatial(a, decode(rebased.shell_origin), new_kin,
        rebased.shell_u)
    old_beam = a.beam.assemble(control.mechanical, TARGETS[cursor-1],
        control.beam_origins)[0]
    new_beam = a.beam.assemble(rebased.mechanical, TARGETS[cursor-1],
        rebased.beam_origins)[0]
    old_history, new_history = decode(control.shell_history), decode(rebased.shell_history)
    layer_strain = _indexed_metric(
        np.asarray(old_history["layer_strain"]).reshape(-1, a.layers, 3),
        np.asarray(new_history["layer_strain"]).reshape(-1, a.layers, 3),
        ["strain"]*3, ("integration_station", "layer", "component"))
    layer_stress = _indexed_metric(
        np.asarray(old_history["layer_stress"]).reshape(-1, a.layers, 3),
        np.asarray(new_history["layer_stress"]).reshape(-1, a.layers, 3),
        ["stress"]*3, ("integration_station", "layer", "component"))
    beam_sections = _beam_section_metrics(control_owner, control,
        prototype.owner, rebased)
    virtual = np.linspace(-.03, .05, a.beam.nodal_count)
    spatial, _ = _pilot_direction(a, rebased.shell_u)
    old_tangent = physical_tangent(old_trial, control.shell_u, a.count)@spatial
    new_tangent = physical_tangent(new_trial, rebased.shell_u, a.count)@spatial
    fixed = np.setdiff1d(np.arange(a.count), np.asarray(control_owner.free))
    if not np.array_equal(fixed, np.arange(6)):
        raise ValueError("F06 pilot requires exact six-DOF shell support")
    old_shell_rows = old_residual[:a.shell_count].reshape(-1, 6)
    new_shell_rows = new_residual[:a.shell_count].reshape(-1, 6)
    beam_slice = slice(a.shell_count, a.shell_count+a.beam.nodal_count)
    old_beam_rows = old_residual[beam_slice].reshape(-1, 6)
    new_beam_rows = new_residual[beam_slice].reshape(-1, 6)
    return dict(
        shell_translation=_metric(old_kin.current_positions, new_kin.current_positions,
            "translation"),
        beam_translation=_metric(control.mechanical.positions, rebased.mechanical.positions,
            "translation"),
        shell_rotation=_rotation_metric(old_kin.rotations, new_kin.rotations, identity),
        beam_rotation=_rotation_metric(control.mechanical.nodal_frames,
            rebased.mechanical.nodal_frames, a.beam.reference_frames),
        shell_force=_metric(old_shell_rows[:, :3], new_shell_rows[:, :3], "force"),
        shell_moment=_metric(old_shell_rows[:, 3:], new_shell_rows[:, 3:], "moment"),
        beam_nodal_force=_metric(old_beam_rows[:, :3], new_beam_rows[:, :3], "force"),
        beam_nodal_moment=_metric(old_beam_rows[:, 3:], new_beam_rows[:, 3:], "moment"),
        native_beam_mixed_residual=_metric(old_beam/a.beam.scale,
            new_beam/a.beam.scale, "solver_scaled_residual"),
        support_force=_metric(old_residual[fixed[:3]], new_residual[fixed[:3]], "force"),
        support_moment=_metric(old_residual[fixed[3:]], new_residual[fixed[3:]], "moment"),
        shell_work={**_metric([old_shell[2]], [new_shell[2]], "work"),
            "left_value": float(old_shell[2]), "right_value": float(new_shell[2])},
        beam_nodal_virtual_work={**_metric(
            [old_beam[:a.beam.nodal_count]@virtual],
            [new_beam[:a.beam.nodal_count]@virtual], "work"),
            "virtual_direction_norm": float(np.linalg.norm(virtual))},
        shell_layer_strain=_metric(old_history["layer_strain"],
            new_history["layer_strain"], "strain"),
        shell_layer_stress=_metric(old_history["layer_stress"],
            new_history["layer_stress"], "stress"),
        shell_layer_maxima=dict(strain=layer_strain, stress=layer_stress),
        beam_section_response=beam_sections,
        gap_translation=_metric(old_trial.constraints[:3], new_trial.constraints[:3],
            "gap_translation"),
        gap_rotation=_metric(old_trial.constraints[3:], new_trial.constraints[3:],
            "gap_rotation"),
        free_solver_residual=_metric(
            (old_residual/control_owner.scale)[list(control_owner.free)],
            (new_residual/prototype.owner.scale)[list(prototype.owner.free)],
            "solver_scaled_residual"),
        full_solver_residual=_metric(old_residual/control_owner.scale,
            new_residual/prototype.owner.scale, "solver_scaled_residual"),
        physical_tangent_direction=_metric(old_tangent/control_owner.scale,
            new_tangent/prototype.owner.scale, "tangent_fd"),
        accepted_shell_history_equal=bool(control.shell_history == rebased.shell_history),
        accepted_beam_history_equal=bool(canonical(control.beam_histories) ==
            canonical(rebased.beam_histories)),
    )


def _fd_metrics(prototype, predecessor, state, cursor):
    """Raw FD plateau for the actual mixed anchored assembly."""
    a = prototype.owner.assembly
    spatial, chart = _pilot_direction(a, state.shell_u)
    response, _ = prototype._anchored_evaluate(predecessor, state.mechanical,
        state.shell_u, state.multipliers, cursor)
    expected = physical_tangent(response, state.shell_u, a.count)@spatial
    rows = []
    beam_start = a.shell_count
    beam_end = beam_start+a.beam.nodal_count
    for step in (2.e-3, 1.e-3, 5.e-4, 2.5e-4):
        forward = a.beam.advance(state.mechanical, step*chart[a.shell_count:-6])
        backward = a.beam.advance(state.mechanical, -step*chart[a.shell_count:-6])
        _, plus = prototype._anchored_evaluate(predecessor, forward,
            state.shell_u+step*chart[:a.shell_count],
            state.multipliers+step*chart[-6:], cursor)
        _, minus = prototype._anchored_evaluate(predecessor, backward,
            state.shell_u-step*chart[:a.shell_count],
            state.multipliers-step*chart[-6:], cursor)
        observed = (plus-minus)/(2*step)
        es = expected[:a.shell_count].reshape(-1, 6)
        os_ = observed[:a.shell_count].reshape(-1, 6)
        eb = expected[beam_start:beam_end].reshape(-1, 6)
        ob = observed[beam_start:beam_end].reshape(-1, 6)
        rows.append(dict(step_size=step,
            assembled_scaled=_metric(expected/prototype.owner.scale,
                observed/prototype.owner.scale, "tangent_fd"),
            shell_force=_metric(es[:, :3], os_[:, :3], "force"),
            shell_moment=_metric(es[:, 3:], os_[:, 3:], "moment"),
            beam_nodal_force=_metric(eb[:, :3], ob[:, :3], "force"),
            beam_nodal_moment=_metric(eb[:, 3:], ob[:, 3:], "moment"),
            retained_mixed=_metric(expected[beam_end:-6]/prototype.owner.scale[beam_end:-6],
                observed[beam_end:-6]/prototype.owner.scale[beam_end:-6],
                "tangent_fd"),
            gap_translation=_metric(expected[-6:-3], observed[-6:-3],
                "gap_translation"),
            gap_rotation=_metric(expected[-3:], observed[-3:], "gap_rotation")))
    return dict(direction_spatial=spatial.tolist(), direction_chart=chart.tolist(),
        direction_shell_norm=float(np.linalg.norm(spatial[:a.shell_count])),
        direction_beam_retained_norm=float(np.linalg.norm(spatial[a.shell_count:-6])),
        direction_multiplier_norm=float(np.linalg.norm(spatial[-6:])),
        samples=rows)


def _physical_ports(prototype, state):
    a = prototype.owner.assembly
    bi = a.beam.index[a.beam_node]
    kin = deformation(np.asarray(a.coordinates), state.shell_u, state.anchor)
    shell = state.shell_u.reshape(-1, 6)[a.shell_node]
    positions = np.array([a.coordinates[a.shell_node]+shell[:3],
        state.mechanical.positions[bi]])
    frames = np.array([kin.rotations[a.shell_node]@a.frame,
        state.mechanical.nodal_frames[bi]])
    low = np.array([np.zeros(3), state.mechanical.position_low[bi]])
    return positions, frames, low


def _shift_ports(positions, frames, direction, step):
    q = np.array([_exp_terms(step*direction[i, 3:])[0]@frames[i]
        for i in range(2)])
    return positions+step*direction[:, :3], q


def _component_fd(prototype, predecessor, state, cursor):
    """Differentiate each actual operator with the fixed mixed physical direction."""
    a = prototype.owner.assembly
    spatial, chart = _pilot_direction(a, state.shell_u)
    shell_spatial_direction = spatial[:a.shell_count]
    shell_chart_direction = chart[:a.shell_count]
    beam_direction = spatial[a.shell_count:-6]
    beam_node_dofs = np.asarray(a.beam.model.mesh.dof_manager.get_node_dofs(a.beam_node))
    port_direction = np.vstack((shell_spatial_direction.reshape(-1, 6)[a.shell_node],
        beam_direction[beam_node_dofs]))
    joint_direction = np.r_[port_direction.ravel(), spatial[-6:]]
    kin = deformation(np.asarray(a.coordinates), state.shell_u, state.anchor)
    origin = decode(predecessor.shell_history)
    _, shell_jac, _, _ = shell_spatial(a, origin, kin, state.shell_u)
    shell_expected = shell_jac@shell_spatial_direction
    _, beam_jac, _, _, _ = a.beam.assemble(state.mechanical,
        TARGETS[cursor-1], predecessor.beam_histories)
    beam_expected = beam_jac@beam_direction
    positions, frames, low = _physical_ports(prototype, state)
    joint = a.joint.evaluate(positions, frames, state.multipliers, position_low=low)
    joint_expected = joint.spatial_jacobian@joint_direction
    rows = []
    for step in (2.e-3, 1.e-3, 5.e-4, 2.5e-4):
        shell_rows = []
        beam_rows = []
        joint_rows = []
        external_rows = []
        for sign in (1., -1.):
            changed_u = state.shell_u+sign*step*shell_chart_direction
            changed_kin = deformation(np.asarray(a.coordinates), changed_u, state.anchor)
            shell_rows.append(shell_spatial(a, origin, changed_kin, changed_u)[0])
            changed_beam = a.beam.advance(state.mechanical, sign*step*beam_direction)
            beam_rows.append(a.beam.assemble(changed_beam, TARGETS[cursor-1],
                predecessor.beam_histories)[0])
            x, q = _shift_ports(positions, frames, port_direction, sign*step)
            joint_rows.append(a.joint.evaluate(x, q,
                state.multipliers+sign*step*spatial[-6:], position_low=low).residual)
            external_rows.append(_dead_external(prototype.owner, TARGETS[cursor-1],
                changed_beam, changed_u, state.multipliers+sign*step*spatial[-6:]))
        shell_fd = (shell_rows[0]-shell_rows[1])/(2*step)
        beam_fd = (beam_rows[0]-beam_rows[1])/(2*step)
        joint_fd = (joint_rows[0]-joint_rows[1])/(2*step)
        external_fd = ((external_rows[0]-external_rows[1])/(2*step)).reshape(-1, 6)
        se, so = shell_expected.reshape(-1, 6), shell_fd.reshape(-1, 6)
        be = beam_expected[:a.beam.nodal_count].reshape(-1, 6)
        bo = beam_fd[:a.beam.nodal_count].reshape(-1, 6)
        je, jo = joint_expected[:12].reshape(2, 6), joint_fd[:12].reshape(2, 6)
        retained = slice(a.beam.nodal_count, a.beam.count)
        rows.append(dict(step_size=step,
            shell_force=_metric(se[:, :3], so[:, :3], "force"),
            shell_moment=_metric(se[:, 3:], so[:, 3:], "moment"),
            native_beam_nodal_force=_metric(be[:, :3], bo[:, :3], "force"),
            native_beam_nodal_moment=_metric(be[:, 3:], bo[:, 3:], "moment"),
            native_beam_retained=_metric(beam_expected[retained]/a.beam.scale[retained],
                beam_fd[retained]/a.beam.scale[retained], "tangent_fd"),
            joint_port_force=_metric(je[:, :3], jo[:, :3], "force"),
            joint_port_moment=_metric(je[:, 3:], jo[:, 3:], "moment"),
            joint_translation_gap=_metric(joint_expected[12:15],
                joint_fd[12:15], "gap_translation"),
            joint_rotation_gap=_metric(joint_expected[15:18],
                joint_fd[15:18], "gap_rotation"),
            dead_external_force=_metric(np.zeros_like(external_fd[:, :3]),
                external_fd[:, :3], "force"),
            dead_external_couple=_metric(np.zeros_like(external_fd[:, 3:]),
                external_fd[:, 3:], "moment")))
    return dict(direction_shell=shell_spatial_direction.tolist(),
        direction_shell_chart=shell_chart_direction.tolist(),
        direction_beam_retained=beam_direction.tolist(),
        direction_joint_port=port_direction.tolist(),
        direction_multiplier=spatial[-6:].tolist(),
        external_operator_scope="SHARED_CONSTANT_DEAD_LOAD_CALL_PATH", samples=rows)


def _port_conjugacy(prototype, state):
    """Complete 12-port lambda·gap work and six compatible rigid zero-work modes."""
    a = prototype.owner.assembly
    positions, frames, low = _physical_ports(prototype, state)
    pose = a.joint.evaluate(positions, frames, state.multipliers, position_low=low)
    reference_length = 1.
    reference_force = float(np.linalg.norm(prototype.owner.forces.reshape(-1, 6)[-1, :3]))
    work_scale = reference_force*reference_length
    variations = dict(
        shell_only=np.array([[.17, -.11, .08, .09, -.07, .05], [0.]*6]),
        beam_only=np.array([[0.]*6, [-.13, .16, -.06, -.04, .08, .11]]),
        mixed=np.array([[.12, .03, -.09, .07, .04, -.05],
            [-.08, -.14, .11, .06, -.03, .09]]))
    cases = {}
    for name, direction in variations.items():
        analytic = float(pose.residual[:12]@direction.ravel())
        samples = []
        for step in (2.e-3, 1.e-3, 5.e-4, 2.5e-4):
            values = []
            for sign in (1., -1.):
                x, q = _shift_ports(positions, frames, direction, sign*step)
                values.append(a.joint.evaluate(x, q, state.multipliers,
                    position_low=low).potential)
            finite_difference = float((values[0]-values[1])/(2*step))
            samples.append(dict(step_size=step, plus_potential=float(values[0]),
                minus_potential=float(values[1]), derivative=finite_difference,
                absolute_error=abs(analytic-finite_difference),
                reference_scaled_error=abs(analytic-finite_difference)/work_scale))
        cases[name] = dict(direction=direction.tolist(),
            spatial_residual=pose.residual[:12].tolist(),
            analytic_work=analytic, reference_scaled_work=analytic/work_scale,
            samples=samples)
    rigid = {}
    physical_positions = positions+low
    for kind in ("translation", "rotation"):
        for axis in range(3):
            v, omega = np.zeros(3), np.zeros(3)
            if kind == "translation":
                v[axis] = reference_length
            else:
                omega[axis] = 1.
            direction = np.zeros((2, 6))
            for port in range(2):
                direction[port, :3] = v+np.cross(omega, physical_positions[port])
                direction[port, 3:] = omega
            work = float(pose.residual[:12]@direction.ravel())
            rigid[f"{kind}_{axis}"] = dict(direction=direction.tolist(),
                work=work, absolute_work=abs(work),
                reference_scaled_work=abs(work)/work_scale)
    return dict(reference_length=reference_length, reference_force=reference_force,
        reference_work=work_scale, positions=physical_positions.tolist(),
        frames=frames.tolist(), multipliers=state.multipliers.tolist(),
        constraints=pose.constraints.tolist(), spatial_residual=pose.residual[:12].tolist(),
        variations=cases, compatible_rigid_zero_work=rigid)


def _fixed_reference_balance(owner, state, loaded, cursor):
    """Frozen one-support/one-tip-force closure about the common origin."""
    a = owner.assembly
    expected_tip = np.array([0., 1.e-4, 2.e-4])
    if (tuple(owner.shell_fixed) != tuple(range(6)) or
            not np.array_equal(owner.forces[:-6], np.zeros(len(owner.forces)-6)) or
            not np.array_equal(owner.forces[-6:-3], expected_tip) or
            not np.array_equal(owner.forces[-3:], np.zeros(3))):
        raise ValueError("F06 fixed one-support dead-tip-force inventory changed")
    reference_length = 1.
    reference_force = float(np.sqrt(5.)*1.e-4)
    reference_moment = reference_force*reference_length
    if not np.isclose(np.linalg.norm(owner.forces[-6:-3]), reference_force,
            rtol=1.e-15, atol=0.):
        raise ValueError("F06 fixed tip force reference changed")
    kin = deformation(np.asarray(a.coordinates), state.shell_u, state.anchor)
    support_position = kin.current_positions[0]
    tip_position = state.mechanical.positions[-1]+state.mechanical.position_low[-1]
    support_force, support_moment = loaded[:3], loaded[3:6]
    applied_force = TARGETS[cursor-1]*expected_tip
    force = support_force+applied_force
    moment = support_moment+np.cross(support_position, support_force)+np.cross(
        tip_position, applied_force)
    return dict(reference_origin=[0., 0., 0.], reference_length=reference_length,
        reference_force=reference_force, reference_moment=reference_moment,
        load_factor=TARGETS[cursor-1], support_position=support_position.tolist(),
        tip_position=tip_position.tolist(), support_force=support_force.tolist(),
        support_moment=support_moment.tolist(), applied_force=applied_force.tolist(),
        applied_couple=[0., 0., 0.], force_closure=force.tolist(),
        moment_closure=moment.tolist(),
        force_raw_norm=float(np.linalg.norm(force)),
        moment_raw_norm=float(np.linalg.norm(moment)),
        force_reference_scaled=float(np.linalg.norm(force))/reference_force,
        moment_reference_scaled=float(np.linalg.norm(moment))/reference_moment)


def _effective_environment():
    """Observe the actual child process using the supervisor's field layout."""
    import importlib.metadata
    import anyfesolver
    import psutil
    import scipy
    import threadpoolctl
    if sys.platform == "win32":
        import win32job
        pywin32_version = importlib.metadata.version("pywin32")
        win32job_import = str(Path(win32job.__file__).resolve())
    else:
        pywin32_version = None
        win32job_import = None

    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    return dict(python_executable=str(Path(sys.executable).resolve()),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        anyfesolver_import=str(Path(anyfesolver.__file__).resolve()),
        psutil_version=psutil.__version__,
        threadpoolctl_version=threadpoolctl.__version__,
        pywin32_version=pywin32_version,
        win32job_import=win32job_import,
        numba_version=importlib.metadata.version("numba"),
        llvmlite_version=importlib.metadata.version("llvmlite"),
        platform=platform.platform(),
        numba_cache_dir=os.environ.get("NUMBA_CACHE_DIR"),
        python_environment={key: os.environ.get(key)
            for key in ("PYTHONPATH", "PYTHONNOUSERSITE")},
        thread_environment={key: os.environ.get(key) for key in keys},
        blas=threadpoolctl.threadpool_info())


def _pilot_sources():
    names = ("scripts/probe_b3_ge_e4_pl_g0_f06_owned_rebase_v4.py",
        "tests/test_b3_e4_f06_owned_rebase_v4.py",
        "docs/agent_plans/B3_GE_E4_PL_G0_F06_V4_SCIENTIFIC_CONTRACT_DRAFT.md",
        "scripts/probe_b3_ge_e4_pl_g0_f06_loaded_rebase_v3.py",
        "src/anyfesolver/_ge_beam3_shell_joint_state.py",
        "src/anyfesolver/_ge_beam3_shell_joint_trial.py",
        "src/anyfesolver/_ge_beam3_g3c_local_shell.py",
        "src/anyfesolver/_ge_beam3_coupled_beam_subdomain.py",
        "src/anyfesolver/_ge_beam3_pose_joint.py",
        "src/anyfesolver/_ge_beam3_retained_generalized.py",
        "src/anyfesolver/_ge_beam3_retained_generalized_state.py",
        "src/anyfesolver/_ge_beam3_retained_generalized_program.py",
        "src/anyfesolver/_ge_beam3_native_generalized_element.py",
        "src/anyfesolver/_ge_beam3_generalized_ellipsoid_section.py",
        "src/anyfesolver/_ge_beam3_generalized_cell.py",
        "src/anyfesolver/_ge_beam3_generalized_static_boundary.py",
        "src/anyfesolver/_ge_beam3_p5/algebra.py",
        "src/anyfesolver/_ge_beam3_p5_seeded/core.py",
        "src/anyfesolver/e4_pl_s3_v2d_element.py",
        "src/anyfesolver/control.py",
        "src/anyfesolver/_native_reference_modal.py",
        "src/anyfesolver/_ge_beam3_native_generalized_loading.py",
        "src/anyfesolver/_ge_beam3_native_line_loading.py",
        "src/anyfesolver/_ge_beam3_centered_reference.py",
        "src/anyfesolver/fe_core.py")
    return {name: _hash((ROOT/name).read_bytes()) for name in names}


class _PilotCancelAt(CancellationToken):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def raise_if_cancelled(self, stage=""):
        if stage == self.stage:
            self.cancel("disposable F06 pilot cancellation")
        super().raise_if_cancelled(stage)


def _negative_checks(prefix, rebased, successor, completed_owner):
    """Disposable hash, semantic and stale-head rejection observations."""
    results = {}
    def rejected(name, expected, owner, action):
        before = (owner.head_sha256, owner.generation)
        try:
            action()
        except ValueError as error:
            if expected not in str(error):
                raise RuntimeError(f"F06 pilot {name} rejected for a different reason") from error
            after = (owner.head_sha256, owner.generation)
            if after != before:
                raise RuntimeError(f"F06 pilot {name} advanced accepted head") from error
            results[name] = dict(status="REJECTED", type=type(error).__name__,
                reason=str(error), before_head_sha256=before[0],
                before_generation=before[1], after_head_sha256=after[0],
                after_generation=after[1])
        else:
            raise RuntimeError(f"F06 pilot {name} unexpectedly accepted")
    bad_hash_owner = OwnedRebasePrototype()
    rejected("bad_external_hash", "external F06 prototype checkpoint hash",
        bad_hash_owner, lambda: bad_hash_owner.restore(rebased.checkpoint,
            expected_sha256="0"*64))
    anchor = decode(rebased.checkpoint)
    anchor["anchor"][1][0][0] += .01
    anchor["rebased_state"]["anchor"][1][0][0] += .01
    anchor["rebased_state_sha256"] = _hash(canonical(anchor["rebased_state"]))
    raw_anchor = canonical(anchor)
    anchor_owner = OwnedRebasePrototype()
    rejected("rehashed_anchor", "rebase event mismatch",
        anchor_owner, lambda: anchor_owner.restore(raw_anchor,
            expected_sha256=_hash(raw_anchor)))
    history = decode(successor.checkpoint)
    history["successor"]["state"]["shell_history"]["layer_stress"][0][0] += 1.e-4
    raw_history = canonical(history)
    history_owner = OwnedRebasePrototype()
    rejected("rehashed_successor_history", "shell history disagreement",
        history_owner, lambda: history_owner.restore(raw_history,
            expected_sha256=_hash(raw_history)))
    zero = decode(successor.checkpoint)
    zero["successor"]["residual"] = [0.]*len(zero["successor"]["residual"])
    zero["successor"]["metric"] = 0.
    raw_zero = canonical(zero)
    zero_owner = OwnedRebasePrototype()
    rejected("rehashed_zero_residual", "residual disagreement",
        zero_owner, lambda: zero_owner.restore(raw_zero,
            expected_sha256=_hash(raw_zero)))
    rejected("stale_successor_commit", "stale successor predecessor",
        completed_owner, lambda: completed_owner.continue_loaded(rebased.checkpoint,
            expected_sha256=rebased.checkpoint_sha256))
    cancel_rebase = OwnedRebasePrototype()
    before_cancel_rebase = (cancel_rebase.head_sha256, cancel_rebase.generation)
    try:
        cancel_rebase.rebase(prefix.checkpoint,
            expected_sha256=_hash(prefix.checkpoint),
            cancellation_token=_PilotCancelAt("f06.rebase.before_commit"))
    except SolveCancelled as error:
        if cancel_rebase.head_sha256 is not None or cancel_rebase.generation is not None:
            raise RuntimeError("F06 cancelled rebase advanced accepted head") from error
        results["cancel_rebase_before_commit"] = dict(status="REJECTED",
            type=type(error).__name__, reason=str(error),
            before_head_sha256=before_cancel_rebase[0],
            before_generation=before_cancel_rebase[1],
            after_head_sha256=cancel_rebase.head_sha256,
            after_generation=cancel_rebase.generation)
    else:
        raise RuntimeError("F06 pilot cancelled rebase unexpectedly accepted")
    cancel_successor = OwnedRebasePrototype()
    cancel_successor.restore(rebased.checkpoint,
        expected_sha256=rebased.checkpoint_sha256)
    before_cancel_successor = (cancel_successor.head_sha256,
        cancel_successor.generation)
    try:
        cancel_successor.continue_loaded(rebased.checkpoint,
            expected_sha256=rebased.checkpoint_sha256,
            cancellation_token=_PilotCancelAt("f06.successor.before_commit"))
    except SolveCancelled as error:
        if (cancel_successor.head_sha256 != rebased.checkpoint_sha256 or
                cancel_successor.generation != 1):
            raise RuntimeError("F06 cancelled successor changed predecessor head") from error
        results["cancel_successor_before_commit"] = dict(status="REJECTED",
            type=type(error).__name__, reason=str(error),
            before_head_sha256=before_cancel_successor[0],
            before_generation=before_cancel_successor[1],
            after_head_sha256=cancel_successor.head_sha256,
            after_generation=cancel_successor.generation)
    else:
        raise RuntimeError("F06 pilot cancelled successor unexpectedly accepted")
    return results


def _control_state(control, assembly):
    anchor = np.tile(np.eye(3), (len(assembly.coordinates), 1, 1))
    return PrototypeState(control.mechanical, control.shell_u, control.multipliers,
        control.beam_origins, control.beam_histories, control.shell_origin,
        control.shell_history, _owned(anchor), control.cursor)


def measure_pilot(*, checkpoint_dir=None):
    """Collect raw preparatory metrics; no registration or gate receipt."""
    started = perf_counter()
    effective_environment = _effective_environment()
    source = OwnedRebasePrototype()
    if source.owner._linear_solve is not _reference_linear_solve:
        raise RuntimeError("F06 private reference linear solve is not configured")
    prefix = source.legacy_prefix()
    if prefix.status != "paused" or prefix.state.cursor != 2:
        raise RuntimeError(f"loaded F06 seed failed: {prefix.status}: {prefix.failure}")
    prefix_seconds = perf_counter()-started
    rebased = source.rebase(prefix.checkpoint, expected_sha256=_hash(prefix.checkpoint))
    fresh = OwnedRebasePrototype()
    restored = fresh.restore(rebased.checkpoint,
        expected_sha256=rebased.checkpoint_sha256)
    rebase_seconds = perf_counter()-started-prefix_seconds
    successor = fresh.continue_loaded(rebased.checkpoint,
        expected_sha256=rebased.checkpoint_sha256)
    replayed = OwnedRebasePrototype().restore(successor.checkpoint,
        expected_sha256=successor.checkpoint_sha256)
    if restored.cursor != 2 or replayed.cursor != 3:
        raise RuntimeError("F06 prototype replay cursor disagreement")
    successor_seconds = perf_counter()-started-prefix_seconds-rebase_seconds
    control = source.owner.solve(checkpoint=prefix.checkpoint,
        expected_sha256=_hash(prefix.checkpoint), stop_after=3)
    if control.status != "completed" or control.state.cursor != 3:
        raise RuntimeError(f"F06 pilot control failed: {control.status}: {control.failure}")
    a = fresh.owner.assembly
    paired = dict(cursor_2=_paired_metrics(source.owner, prefix.state,
        fresh, restored, 2), cursor_3=_paired_metrics(source.owner, control.state,
        fresh, successor.state, 3))
    control_2 = _control_state(prefix.state, source.owner.assembly)
    control_3 = _control_state(control.state, source.owner.assembly)
    tangent_fd = dict(cursor_2=_fd_metrics(fresh, restored, restored, 2),
        cursor_3=_fd_metrics(fresh, restored, successor.state, 3))
    component_fd = dict(cursor_2=dict(
        control=_component_fd(source, control_2, control_2, 2),
        rebased=_component_fd(fresh, restored, restored, 2)),
        cursor_3=dict(
        control=_component_fd(source, control_2, control_3, 3),
        rebased=_component_fd(fresh, restored, successor.state, 3)))
    interface_work = {}
    for name, control_state, rebased_state in (
            ("cursor_2", control_2, restored),
            ("cursor_3", control_3, successor.state)):
        old_work = _port_conjugacy(source, control_state)
        new_work = _port_conjugacy(fresh, rebased_state)
        scale = old_work["reference_work"]
        pair = {key: dict(control=old_work["variations"][key]["analytic_work"],
            rebased=new_work["variations"][key]["analytic_work"],
            absolute_difference=abs(old_work["variations"][key]["analytic_work"]-
                new_work["variations"][key]["analytic_work"]),
            reference_scaled_difference=abs(old_work["variations"][key]["analytic_work"]-
                new_work["variations"][key]["analytic_work"])/scale)
            for key in old_work["variations"]}
        interface_work[name] = dict(control=old_work, rebased=new_work,
            paired_virtual_work=pair)
    _, control_loaded_2, _, _ = source.owner._evaluate(prefix.state,
        prefix.state.mechanical, prefix.state.shell_u, prefix.state.multipliers, 2)
    _, control_loaded_3, _, _ = source.owner._evaluate(control.state,
        control.state.mechanical, control.state.shell_u, control.state.multipliers, 3)
    _, loaded_2 = fresh._anchored_evaluate(restored, restored.mechanical,
        restored.shell_u, restored.multipliers, 2)
    _, loaded_3 = fresh._anchored_evaluate(successor.state, successor.state.mechanical,
        successor.state.shell_u, successor.state.multipliers, 3)
    kin2 = deformation(np.asarray(a.coordinates), restored.shell_u, restored.anchor)
    kin3 = deformation(np.asarray(a.coordinates), successor.state.shell_u,
        successor.state.anchor)
    work2 = shell_spatial(a, decode(restored.shell_origin), kin2, restored.shell_u)[2]
    work3 = shell_spatial(a, decode(successor.state.shell_origin), kin3,
        successor.state.shell_u)[2]
    fixed_balance = dict(cursor_2=dict(
        control=_fixed_reference_balance(source.owner, control_2, control_loaded_2, 2),
        rebased=_fixed_reference_balance(fresh.owner, restored, loaded_2, 2)),
        cursor_3=dict(
        control=_fixed_reference_balance(source.owner, control_3, control_loaded_3, 3),
        rebased=_fixed_reference_balance(fresh.owner, successor.state, loaded_3, 3)))
    changes = dict(
        beam_rotation=_rotation_metric(restored.mechanical.nodal_frames,
            successor.state.mechanical.nodal_frames, a.beam.reference_frames),
        support_force=_metric(loaded_2[:3], loaded_3[:3], "force"),
        support_moment=_metric(loaded_2[3:6], loaded_3[3:6], "moment"),
        shell_work={**_metric([work2], [work3], "work"),
            "cursor_2_value": float(work2), "cursor_3_value": float(work3)},
        multiplier_force=_metric(restored.multipliers[:3],
            successor.state.multipliers[:3], "force"),
        multiplier_moment=_metric(restored.multipliers[3:],
            successor.state.multipliers[3:], "moment"))
    terminal_balance = fixed_balance["cursor_3"]["rebased"]
    balance = dict(force=dict(raw_norm=terminal_balance["force_raw_norm"],
        physical_scale=terminal_balance["reference_force"],
        normalized=terminal_balance["force_reference_scaled"]),
        moment=dict(raw_norm=terminal_balance["moment_raw_norm"],
        physical_scale=terminal_balance["reference_moment"],
        normalized=terminal_balance["moment_reference_scaled"]))
    negatives = _negative_checks(prefix, rebased, successor, fresh)
    replay_audit = {}
    for name, run in (("rebase", rebased), ("successor", successor)):
        audited = OwnedRebasePrototype().audit_legacy_replay(run.checkpoint,
            expected_sha256=run.checkpoint_sha256)
        direct_hash = _hash(canonical(run.state.descriptor()))
        audited_hash = _hash(canonical(audited.descriptor()))
        if direct_hash != audited_hash:
            raise RuntimeError(f"F06 {name} optional replay audit disagreement")
        replay_audit[name] = dict(status="MATCHED", direct_state_sha256=direct_hash,
            audited_state_sha256=audited_hash)
    measurement_seconds = perf_counter()-started-prefix_seconds-rebase_seconds-successor_seconds
    checkpoints = dict(base=prefix.checkpoint, rebase=rebased.checkpoint,
        successor=successor.checkpoint, control=control.checkpoint)
    sidecars = {name: dict(file=f"{name}.checkpoint", sha256=_hash(raw),
        byte_count=len(raw)) for name, raw in checkpoints.items()}
    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir)
        if not directory.is_dir():
            raise ValueError("F06 pilot checkpoint directory must exist")
        for name, raw in checkpoints.items():
            with (directory/sidecars[name]["file"]).open("xb") as stream:
                stream.write(raw)
    if _effective_environment() != effective_environment:
        raise ValueError("F06 child process environment changed during measurement")
    payload = dict(schema="G0_F06_CURVED_S3_V4_SCIENTIFIC_MEASUREMENT_V1",
        scope="G0_S3_OWNED_REBASE_MEASUREMENT_NOT_ADJUDICATED",
        effective_environment=effective_environment,
        source_commit=subprocess.check_output(("git", "rev-parse", "HEAD"),
            cwd=ROOT, text=True).strip(), source_sha256=_pilot_sources(),
        targets=TARGETS,
        provisional_floors={key: dict(value=value, unit=unit)
            for key, (value, unit) in PILOT_FLOORS.items()},
        checkpoint_sha256=dict(base=_hash(prefix.checkpoint),
            rebase=rebased.checkpoint_sha256, successor=successor.checkpoint_sha256,
            control=_hash(control.checkpoint)),
        statuses=dict(base=prefix.status, rebase=rebased.status,
            successor=successor.status, control=control.status),
        paired=paired, tangent_fd=tangent_fd, component_fd=component_fd,
        interface_work=interface_work, checkpoint_sidecars=sidecars,
        checkpoint_sidecars_written=checkpoint_dir is not None,
        cursor_3_changes=changes, cursor_3_balance=balance,
        fixed_reference_balance=fixed_balance,
        negative_checks=negatives, optional_replay_audit=replay_audit,
        runtime_seconds=dict(base=prefix_seconds, rebase_and_restore=rebase_seconds,
            successor_and_restore=successor_seconds, measurement=measurement_seconds,
            total=perf_counter()-started),
        production_qualified=False)
    return dict(metrics_sha256=_hash(canonical(payload)), metrics=payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unregistered F06 v4 G0 measurement pilot")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path,
        help="optional new file for the canonical pilot JSON; existing files are never replaced")
    outputs.add_argument("--output-dir", type=Path,
        help="new exclusive directory with metrics.json and four raw checkpoint sidecars")
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=False, exist_ok=False)
    raw = canonical(measure_pilot(checkpoint_dir=args.output_dir))
    if args.output is None:
        if args.output_dir is None:
            sys.stdout.buffer.write(raw)
        else:
            target = args.output_dir/"metrics.json"
            with target.open("xb") as stream:
                stream.write(raw)
            print(str(target.resolve()))
    else:
        with args.output.open("xb") as stream:
            stream.write(raw)
        print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
